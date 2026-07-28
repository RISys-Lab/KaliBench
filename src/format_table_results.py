#!/usr/bin/env python3
import argparse
import csv
import math
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

MODE_MAP = {
    'unrestricted': 'U',
    'restricted': 'R',
    'hinted': 'H',
}
MODE_ORDER = ['U', 'R', 'H']
METRIC_MAP = {
    'summary_exact_match': 'EM',
    'summary_optional_f1': 'OptF1',
    'summary_positional_f1': 'PosF1',
    'summary_tool_score': 'Tool',
    'summary_total_score': 'Total',
}
METRIC_ORDER = ['Tool', 'OptF1', 'PosF1', 'Total', 'EM']
AVG_TS_COL = 'Avg.TS'
REQUIRED_COLUMNS = [
    'config_inference_api',
    'config_mode',
    'config_model',
    'config_output_policy',
    *METRIC_MAP.keys(),
]
ACRONYM_TOKENS = {
    'api': 'API',
    'cpu': 'CPU',
    'dpo': 'DPO',
    'f1': 'F1',
    'gppo': 'GPPO',
    'grpo': 'GRPO',
    'gpu': 'GPU',
    'gpt': 'GPT',
    'html': 'HTML',
    'json': 'JSON',
    'ocr': 'OCR',
    'oss': 'OSS',
    'pdf': 'PDF',
    'ppo': 'PPO',
    'rl': 'RL',
    'rlaif': 'RLAIF',
    'rlhf': 'RLHF',
    'sft': 'SFT',
    'sql': 'SQL',
    'xml': 'XML',
}


def extract_model_name(raw: str) -> str:
    raw = str(raw).strip()
    if '/' in raw:
        return raw.split('/')[-1]
    return raw


def parse_size_to_float(size_text: str) -> float:
    cleaned = str(size_text).strip()
    match = re.fullmatch(r'(\d+(?:\.\d+)?)\s*[bB]', cleaned)
    if not match:
        raise ValueError(f"Invalid size override '{size_text}'. Expected format like 7B or 70B.")
    return float(match.group(1))


def extract_size_b(raw_model: str) -> Optional[float]:
    name = extract_model_name(raw_model)
    matches = re.findall(r'(\d+(?:\.\d+)?)\s*[bB]\b', name)
    if matches:
        return float(matches[-1])
    return None


def format_size_label(size_b: float) -> str:
    if float(size_b).is_integer():
        return f'{int(size_b)}B'
    return f'{float(size_b):g}B'


def size_group_label(size_b: Optional[float], grouping: str) -> Tuple[str, float]:
    if size_b is None or pd.isna(size_b):
        return 'Unknown', 999.0
    if grouping == 'exact':
        return format_size_label(float(size_b)), float(size_b)
    if size_b <= 10:
        return '8B', 8.0
    if size_b < 50:
        return '20B-32B', 20.0
    return '70B+', 70.0


def smart_capitalize_model_name(name: str) -> str:
    def capitalize_alpha_prefix(token: str) -> str:
        lower = token.lower()
        if lower in ACRONYM_TOKENS:
            return ACRONYM_TOKENS[lower]

        size_match = re.fullmatch(r'(\d+(?:\.\d+)?)([bB])', token)
        if size_match:
            return f"{size_match.group(1)}B"

        m = re.match(r'([A-Za-z]+)(.*)', token)
        if not m:
            return token

        prefix, suffix = m.groups()
        prefix_lower = prefix.lower()
        if prefix_lower in ACRONYM_TOKENS:
            cap_prefix = ACRONYM_TOKENS[prefix_lower]
        else:
            cap_prefix = prefix_lower[0].upper() + prefix_lower[1:]

        suffix = re.sub(r'(?<=\d)b\b', 'B', suffix)
        return cap_prefix + suffix

    parts = re.split(r'([-_])', str(name))
    return ''.join(
        part if part in {'-', '_'} else capitalize_alpha_prefix(part)
        for part in parts
    )


def variant_suffix(api: str, policy: str) -> str:
    api = str(api).strip()
    policy = str(policy).strip()
    parts = [api]
    if policy != 'strict':
        parts.append(policy)
    return ' [' + '; '.join(parts) + ']'


def build_display_model(row: pd.Series) -> str:
    return smart_capitalize_model_name(extract_model_name(row['config_model'])) + variant_suffix(
        row['config_inference_api'], row['config_output_policy']
    )


def parse_override_items(items: Optional[Iterable[str]]) -> Dict[str, float]:
    overrides: Dict[str, float] = {}
    for item in items or []:
        if ':' not in item:
            raise ValueError(
                f"Invalid override '{item}'. Expected format ModelName:7B"
            )
        model_text, size_text = item.rsplit(':', 1)
        model_text = model_text.strip()
        if not model_text:
            raise ValueError(f"Invalid override '{item}'. Model name cannot be empty.")
        size_b = parse_size_to_float(size_text)
        raw_key = str(model_text).strip()
        norm_key = extract_model_name(model_text)
        overrides[raw_key] = size_b
        overrides[norm_key] = size_b
    return overrides


def apply_size_override(raw_model: str, overrides: Dict[str, float]) -> Optional[float]:
    raw_key = str(raw_model).strip()
    norm_key = extract_model_name(raw_model)
    if raw_key in overrides:
        return overrides[raw_key]
    if norm_key in overrides:
        return overrides[norm_key]
    return extract_size_b(raw_model)


def find_missing_size_models(df: pd.DataFrame, size_overrides: Dict[str, float]) -> List[str]:
    missing: Set[str] = set()
    for raw_model in df['config_model'].dropna().astype(str):
        if apply_size_override(raw_model, size_overrides) is None:
            missing.add(extract_model_name(raw_model))
    return sorted(missing, key=lambda x: x.lower())



def print_missing_size_models(df: pd.DataFrame, size_overrides: Dict[str, float]) -> None:
    missing_models = find_missing_size_models(df, size_overrides)
    if not missing_models:
        return

    print('Models with missing model size after overrides:', file=sys.stderr)
    for model in missing_models:
        pretty = smart_capitalize_model_name(model)
        print(f'  - {pretty}   example: --override-size {model}:7B', file=sys.stderr)


def build_long_frame(df: pd.DataFrame, grouping: str, size_overrides: Dict[str, float]) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')

    out = df.copy()
    out['model_name'] = out['config_model'].map(extract_model_name)
    out['size_b'] = out['config_model'].map(lambda x: apply_size_override(x, size_overrides))
    size_pairs = out['size_b'].map(lambda x: size_group_label(x, grouping))
    out['size_group'] = size_pairs.map(lambda x: x[0])
    out['size_group_sort'] = size_pairs.map(lambda x: x[1])
    out['mode_short'] = out['config_mode'].map(MODE_MAP)
    out['display_model'] = out.apply(build_display_model, axis=1)

    long_df = out.rename(columns=METRIC_MAP)
    for metric in METRIC_ORDER:
        long_df[metric] = pd.to_numeric(long_df[metric], errors='coerce')

    ordered_cols = [
        'size_group',
        'size_group_sort',
        'size_b',
        'display_model',
        'model_name',
        'config_inference_api',
        'config_output_policy',
        'config_mode',
        'mode_short',
        *METRIC_ORDER,
    ]
    return long_df[ordered_cols].copy()


def validate_sort_key(sort_by: str) -> Tuple[str, str]:
    if sort_by == AVG_TS_COL:
        return sort_by, ''
    parts = str(sort_by).split('_')
    if len(parts) != 2:
        raise ValueError(
            f"Invalid --sort-by '{sort_by}'. Expected Metric_Mode or {AVG_TS_COL}."
        )
    metric, mode = parts
    if metric not in METRIC_ORDER:
        raise ValueError(
            f"Invalid metric '{metric}' in --sort-by. Choose from: {', '.join(METRIC_ORDER)}"
        )
    if mode not in MODE_ORDER:
        raise ValueError(
            f"Invalid mode '{mode}' in --sort-by. Choose from: {', '.join(MODE_ORDER)}"
        )
    return metric, mode


def sort_within_groups(
    wide_df: pd.DataFrame,
    sort_by: str,
    ascending: bool,
) -> pd.DataFrame:
    sort_col = sort_by
    if sort_col not in wide_df.columns:
        raise ValueError(f"Sort column '{sort_col}' not found in wide table.")

    groups: List[pd.DataFrame] = []
    size_order = (
        wide_df[['size_group', 'size_group_sort']]
        .drop_duplicates()
        .sort_values(['size_group_sort', 'size_group'], ascending=[True, True], kind='stable')
    )

    for _, group_row in size_order.iterrows():
        size_group = group_row['size_group']
        sub = wide_df[wide_df['size_group'] == size_group].copy()
        sub = sub.sort_values(
            by=[sort_col, 'display_model'],
            ascending=[ascending, True],
            na_position='last',
            kind='stable',
        )
        groups.append(sub)

    return pd.concat(groups, ignore_index=True) if groups else wide_df.copy()


def build_wide_frame(long_df: pd.DataFrame, sort_by: str, ascending: bool) -> pd.DataFrame:
    base_cols = [
        'size_group',
        'size_group_sort',
        'size_b',
        'display_model',
        'model_name',
        'config_inference_api',
        'config_output_policy',
    ]

    pivot = long_df.pivot_table(
        index=base_cols,
        columns='mode_short',
        values=METRIC_ORDER,
        aggfunc='first',
    )

    desired_columns = pd.MultiIndex.from_product([METRIC_ORDER, MODE_ORDER])
    pivot = pivot.reindex(columns=desired_columns)
    pivot.columns = [f'{metric}_{mode}' for metric, mode in pivot.columns]
    wide = pivot.reset_index()

    total_cols = [f'Total_{mode}' for mode in MODE_ORDER]
    wide[AVG_TS_COL] = wide[total_cols].mean(axis=1, skipna=True)
    wide = sort_within_groups(wide, sort_by=sort_by, ascending=ascending)
    return wide.reset_index(drop=True)


def format_value(value: float, bold: bool, underline: bool, precision: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return '--'
    s = f'{(value * 100):.{precision}f}'
    if bold and underline:
        return f'\\textbf{{\\underline{{{s}}}}}'
    if bold:
        return f'\\textbf{{{s}}}'
    if underline:
        return f'\\underline{{{s}}}'
    return s


def escape_latex(text: str) -> str:
    repl = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
        '\\': r'\textbackslash{}',
    }
    return ''.join(repl.get(ch, ch) for ch in str(text))


def compute_bold_lookup(wide_df: pd.DataFrame) -> Dict[Tuple[str, str], float]:
    value_cols = [f'{metric}_{mode}' for metric in METRIC_ORDER for mode in MODE_ORDER]
    value_cols.append(AVG_TS_COL)
    lookup: Dict[Tuple[str, str], float] = {}
    for size_group, sub in wide_df.groupby('size_group', sort=False):
        for col in value_cols:
            max_val = sub[col].max(skipna=True)
            if pd.notna(max_val):
                lookup[(size_group, col)] = float(max_val)
    return lookup


def compute_underline_lookup(wide_df: pd.DataFrame) -> Dict[Tuple[str, str], float]:
    value_cols = [f'{metric}_{mode}' for metric in METRIC_ORDER for mode in MODE_ORDER]
    value_cols.append(AVG_TS_COL)
    lookup: Dict[Tuple[str, str], float] = {}
    for size_group, sub in wide_df.groupby('size_group', sort=False):
        for col in value_cols:
            series = sub[col].dropna()
            if series.empty:
                continue
            unique_vals = sorted(set(float(v) for v in series), reverse=True)
            if len(unique_vals) < 2:
                continue
            lookup[(size_group, col)] = float(unique_vals[1])
    return lookup


def build_latex_table(
    wide_df: pd.DataFrame,
    caption: str,
    label: str,
    precision: int = 2,
) -> str:
    bold_lookup = compute_bold_lookup(wide_df)
    underline_lookup = compute_underline_lookup(wide_df)
    cols = [f'{metric}_{mode}' for metric in METRIC_ORDER for mode in MODE_ORDER]
    cols.append(AVG_TS_COL)
    col_count = 2 + len(cols)

    lines: List[str] = []
    lines.append(r'\begin{table*}[t]')
    lines.append(r'\centering')
    lines.append(r'\small')
    lines.append(r'\setlength{\tabcolsep}{4pt}')
    lines.append(r'\resizebox{\textwidth}{!}{%')
    lines.append(r'\begin{tabular}{ll' + 'c' * len(cols) + '}')
    lines.append(r'\toprule')

    top_header = 'Category & Model'
    for metric in METRIC_ORDER:
        top_header += rf' & \multicolumn{{3}}{{c}}{{{metric}}}'
    top_header += rf' & \multicolumn{{1}}{{c}}{{{escape_latex(AVG_TS_COL)}}} \\'
    lines.append(top_header)

    cmidrules = []
    start = 3
    for _metric in METRIC_ORDER:
        end = start + 2
        cmidrules.append(rf'\cmidrule(lr){{{start}-{end}}}')
        start = end + 1
    cmidrules.append(rf'\cmidrule(lr){{{start}-{start}}}')
    lines.append(' '.join(cmidrules))

    sub_header = ' & & ' + ' & '.join(mode for _metric in METRIC_ORDER for mode in MODE_ORDER) + r' & \\'
    lines.append(sub_header)
    lines.append(r'\midrule')

    for size_group, sub in wide_df.groupby('size_group', sort=False):
        lines.append(rf'\multicolumn{{{col_count}}}{{l}}{{\textbf{{{escape_latex(size_group)}}}}} \\')
        for _, row in sub.iterrows():
            row_cells = [' ', escape_latex(row['display_model'])]
            for col in cols:
                value = row[col]
                best = False
                worst = False
                if pd.notna(value):
                    max_val = bold_lookup.get((row['size_group'], col))
                    min_val = underline_lookup.get((row['size_group'], col))
                    best = max_val is not None and abs(float(value) - max_val) < 1e-12
                    worst = min_val is not None and abs(float(value) - min_val) < 1e-12
                row_cells.append(format_value(value, best, worst, precision=precision))
            lines.append(' & '.join(row_cells) + r' \\')
        lines.append(r'\midrule')

    if lines[-1] == r'\midrule':
        lines.pop()
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}%')
    lines.append(r'}')
    lines.append(rf'\caption{{{escape_latex(caption)}}}')
    lines.append(rf'\label{{{escape_latex(label)}}}')
    lines.append(r'\end{table*}')
    return '\n'.join(lines) + '\n'


def export_outputs(
    input_csv: Path,
    output_csv: Path,
    output_tex: Path,
    grouping: str,
    caption: str,
    label: str,
    sort_by: str,
    ascending: bool,
    size_overrides: Dict[str, float],
    decimals: int,
) -> None:
    df = pd.read_csv(input_csv)
    print_missing_size_models(df, size_overrides)
    long_df = build_long_frame(df, grouping=grouping, size_overrides=size_overrides)
    wide_df = build_wide_frame(long_df, sort_by=sort_by, ascending=ascending)

    ordered_cols = [
        'size_group', 'size_group_sort', 'size_b', 'display_model', 'model_name',
        'config_inference_api', 'config_output_policy',
        *[f'{metric}_{mode}' for metric in METRIC_ORDER for mode in MODE_ORDER],
        AVG_TS_COL,
    ]
    wide_df = wide_df[ordered_cols]
    wide_df.to_csv(output_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    output_tex.write_text(
        build_latex_table(wide_df, caption=caption, label=label, precision=decimals),
        encoding='utf-8',
    )


def main() -> None:
    parser = argparse.ArgumentParser(description='Convert benchmark CSV into an organized CSV and LaTeX table.')
    parser.add_argument('input_csv', type=Path)
    parser.add_argument('--output-csv', type=Path, default=Path('organized_scores.csv'))
    parser.add_argument('--output-tex', type=Path, default=Path('organized_scores.tex'))
    parser.add_argument('--size-grouping', choices=['coarse', 'exact'], default='coarse')
    parser.add_argument(
        '--override-sizes',
        dest='override_sizes',
        action='append',
        default=[],
        help='Override model size, e.g. --override-size Llama-Primus-Merged:7B. Can be repeated.',
    )
    parser.add_argument(
        '--sort-by',
        default=AVG_TS_COL,
        help=f"Column used to sort models within each size group, e.g. {AVG_TS_COL} or Total_H.",
    )
    parser.add_argument(
        '--sort-order',
        choices=['asc', 'desc'],
        default='asc',
        help='Sort direction within each size group. Default asc puts the best model at the bottom for high-is-better metrics like Total_H.',
    )
    parser.add_argument(
        '--caption',
        default='Model results grouped by model size. Best values within each size group are bolded for each metric and mode.',
    )
    parser.add_argument('--label', default='tab:model-results-by-size')
    parser.add_argument(
        '--decimals',
        type=int,
        default=1,
        help='Number of decimal places to show in the LaTeX table.',
    )
    args = parser.parse_args()

    validate_sort_key(args.sort_by)
    size_overrides = parse_override_items(args.override_sizes)

    export_outputs(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        output_tex=args.output_tex,
        grouping=args.size_grouping,
        caption=args.caption,
        label=args.label,
        sort_by=args.sort_by,
        ascending=(args.sort_order == 'asc'),
        size_overrides=size_overrides,
        decimals=args.decimals,
    )


if __name__ == '__main__':
    main()
