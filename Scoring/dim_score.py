#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
kali-tool dimensional scoring pipeline — Method B normalization (per-dimension support)

Usage example (PowerShell):
python .\dim_score.py --models-folder ".\folder_that_stores_scoring_file" --output-folder ".\dim_score" --use-hf --hf-dataset "anonymous62567/kali-tools"
"""

import os
import json
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import sys

# Optional HF import
try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

def parse_args():
    p = argparse.ArgumentParser(description="Kali-tool dimensional scoring pipeline — Method B (per-dim support)")
    # NOTE: --base removed as requested
    p.add_argument("--subtools", "-s", type=str,
                   help="Path to subtool->title jsonl file", required=False)
    p.add_argument("--models-folder", "-m", type=str,
                   help="Folder containing model .jsonl files", required=True)
    p.add_argument("--output-folder", "-o", type=str,
                   help="Folder to write outputs to", required=True)
    p.add_argument("--use-hf", action="store_true",
                   help="Attempt to load HF dataset mapping (requires `datasets` lib and internet)")
    p.add_argument("--hf-dataset", type=str, default="anonymous62567/kali-tools",
                   help="HF dataset id to load for title->metapackages mapping (default: anonymous62567/kali-tools)")
    p.add_argument("--labels-file", type=str, default=None,
                   help="Optional path to a JSON or newline file with labels list; if omitted use embedded LABELS")
    p.add_argument("--no-tqdm", action="store_true", help="Disable tqdm progressbars")
    p.add_argument("--verbose", "-v", action="store_true", help="Verbose prints")
    return p.parse_args()

# ---------- Default LABELS (kept as original order) ----------
DEFAULT_LABELS = [
    "kali-tools-information-gathering","kali-tools-802-11","kali-tools-rfid","kali-tools-bluetooth","kali-tools-sdr","kali-tools-social-engineering",
    "kali-tools-vulnerability","kali-tools-fuzzing","kali-tools-web","kali-tools-database","kali-tools-voip",
    "kali-tools-exploitation", "kali-tools-sniffing-spoofing","kali-tools-hardware","kali-tools-wireless",
    "kali-tools-post-exploitation","kali-tools-passwords","kali-tools-windows-resources","kali-tools-crypto-stego",
    "kali-tools-reverse-engineering","kali-tools-forensics","kali-tools-reporting","kali-tools-gpu",
]

def load_labels_from_file(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Labels file not found: {path}")
    text = p.read_text(encoding="utf-8").strip()
    # try JSON list first
    try:
        obj = json.loads(text)
        if isinstance(obj, (list, tuple)):
            return [str(x).strip() for x in obj]
    except Exception:
        pass
    # fallback: newline or comma separated
    if "\n" in text:
        return [ln.strip() for ln in text.splitlines() if ln.strip()]
    if "," in text:
        return [x.strip() for x in text.split(",") if x.strip()]
    # single-line single label -> return as single element list
    return [text]

def resolve_title_from_toolname_factory(subtool_to_title, title_set):
    def resolve_title_from_toolname(name: str):
        """
        Candidate matching:
          - name
          - name.split('/',1)[0]
          - name.split('-',1)[0]
        Priority: subtool mapping -> direct title match (both normalized lower)
        Returns (title_or_None, matched_key_or_None) where returned title is normalized lower if from mapping.
        """
        if not name or not isinstance(name, str):
            return None, None
        name = name.strip()
        if not name:
            return None, None
        candidates = [name]
        if '/' in name:
            base_slash = name.split('/', 1)[0].strip()
            if base_slash not in candidates:
                candidates.append(base_slash)
        if '-' in name:
            base_dash = name.split('-', 1)[0].strip()
            if base_dash not in candidates:
                candidates.append(base_dash)
        for cand in candidates:
            cand_norm = cand.strip().lower()
            if cand_norm in subtool_to_title:
                # subtool_to_title value stored as lower title (or None)
                return subtool_to_title[cand_norm], cand_norm
            if cand_norm in title_set:
                return cand_norm, cand_norm
        return None, None
    return resolve_title_from_toolname

def metas_to_multihot(metas, labels):
    if not metas:
        return [0] * len(labels)
    s = set(metas)
    return [1 if lab in s else 0 for lab in labels]

def main():
    args = parse_args()

    # internal default base kept but not exposed as CLI
    BASE = Path(r"F:\3.vs_docs")

    SUBTOOLS = Path(args.subtools) if args.subtools else BASE / "1.kali" / "1.kali_tools_subtools.jsonl"
    MODELS_FOLDER = Path(args.models_folder)
    OUTPUT_FOLDER = Path(args.output_folder)

    USE_HF = bool(args.use_hf)
    HF_DATASET = args.hf_dataset

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    # labels
    if args.labels_file:
        LABELS = load_labels_from_file(args.labels_file)
    else:
        LABELS = DEFAULT_LABELS[:]
    LABELS_SET = set(LABELS)

    # ---------- Load subtool -> title (normalize keys to lower().strip()) ----------
    subtool_to_title = {}
    title_set = set()
    if SUBTOOLS.exists():
        with SUBTOOLS.open("r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    obj = json.loads(ln)
                except Exception:
                    continue
                sub = obj.get("subtool")
                title = obj.get("title")
                if sub is not None:
                    sub_k = str(sub).strip().lower()
                    title_v = str(title).strip() if title is not None else None
                    subtool_to_title[sub_k] = title_v.lower() if title_v is not None else None
                if title:
                    title_set.add(str(title).strip().lower())
        if args.verbose:
            print(f"[INFO] Loaded {len(subtool_to_title)} subtool mappings, {len(title_set)} unique titles (normalized).")
    else:
        if args.verbose:
            print(f"[WARN] subtools file not found at {SUBTOOLS}. Proceeding with empty mapping.")

    resolve_title_from_toolname = resolve_title_from_toolname_factory(subtool_to_title, title_set)

    # ---------- Load HF dataset title -> metapackages (filtered to LABELS only) ----------
    title_to_meta = {}
    if USE_HF:
        if not HF_AVAILABLE:
            print("[WARN] datasets lib not available; skipping HF load.")
        else:
            try:
                print(f"[INFO] loading {HF_DATASET} from HF (needs internet)...")
                ds = load_dataset(HF_DATASET, split="train")
                for rec in ds:
                    raw_title = rec.get("title") or rec.get("tool") or rec.get("package")
                    if raw_title is None:
                        continue
                    title_key = str(raw_title).strip().lower()
                    meta_field = rec.get("metadata") or rec.get("metapackages") or rec.get("meta")
                    metas = []
                    if isinstance(meta_field, dict):
                        raw = meta_field.get("metapackages") or meta_field.get("metapackage") or []
                        if isinstance(raw, str):
                            metas = [m.strip() for m in raw.replace(",", " ").split() if m.strip()]
                        elif isinstance(raw, (list, tuple)):
                            tmp = []
                            for it in raw:
                                if isinstance(it, str):
                                    tmp += [m.strip() for m in it.replace(",", " ").split() if m.strip()]
                                else:
                                    tmp.append(str(it).strip())
                            metas = tmp
                    elif isinstance(meta_field, (list, tuple)):
                        tmp = []
                        for it in meta_field:
                            if isinstance(it, str):
                                tmp += [m.strip() for m in it.replace(",", " ").split() if m.strip()]
                            else:
                                tmp.append(str(it).strip())
                        metas = tmp
                    elif isinstance(meta_field, str):
                        metas = [m.strip() for m in meta_field.replace(",", " ").split() if m.strip()]
                    else:
                        metas = []
                    # keep only labels we care about
                    filtered = [m for m in metas if isinstance(m, str) and m.strip() in LABELS_SET]
                    if filtered:
                        title_to_meta[title_key] = list(dict.fromkeys(filtered))
                print(f"[INFO] HF mapping loaded: {len(title_to_meta)} titles with LABELS metapackages.")
            except Exception as e:
                print("[WARN] HF load failed:", e)
                title_to_meta = {}
    else:
        if args.verbose:
            print("[INFO] HF load disabled by config; title->metapackages empty.")

    # ---------- Gather model files ----------
    model_files = sorted([p for p in Path(MODELS_FOLDER).glob("*.jsonl")])
    if not model_files:
        print(f"[WARN] No .jsonl files found in {MODELS_FOLDER}")
    else:
        print(f"[INFO] Found {len(model_files)} model files in {MODELS_FOLDER}")

    # Global counters (across all models)
    global_tag_sample_counts = Counter()
    # NOTE: we will still track missing internally for diagnostics, but per your request we WILL NOT write any missing_toolnames CSVs.
    missing_toolnames_all = []

    per_model_agg = {}

    model_iter = tqdm(model_files, desc="Models") if (not args.no_tqdm and model_files) else model_files
    for model_path in model_iter:
        model_name = model_path.stem
        if args.verbose:
            print(f"\n[PROCESS] {model_name}")
        raw_accum = np.zeros(len(LABELS), dtype=float)   # sum total_score for each label
        support = np.zeros(len(LABELS), dtype=int)       # number of samples for each label (per-model)
        sample_rows = []
        # per-model missing list retained in-memory but not written
        sample_count = 0
        matched_count = 0

        with model_path.open("r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    rec = json.loads(ln)
                except Exception:
                    continue
                sample_count += 1
                gt_cmd = rec.get("gt_cmd") or rec.get("gt") or rec.get("subtool") or rec.get("tool_name") or rec.get("pred_cmd")
                total_score = rec.get("total_score", rec.get("tool_score", 0.0))
                try:
                    total_score = float(total_score)
                except Exception:
                    total_score = 0.0
                custom_id = rec.get("custom_id") or rec.get("id") or None

                title, matched_key = resolve_title_from_toolname(gt_cmd)

                metas = []
                matched_here = False
                if title:
                    # title is normalized lower when returned from resolve_title; ensure key exists
                    metas = title_to_meta.get(title.lower().strip(), [])
                    matched_here = True
                else:
                    # fallback: maybe gt_cmd itself appears as title in HF mapping (normalize)
                    if isinstance(gt_cmd, str) and gt_cmd.strip().lower() in title_to_meta:
                        metas = title_to_meta.get(gt_cmd.strip().lower(), [])
                        title = gt_cmd.strip().lower()
                        matched_here = True

                if matched_here:
                    matched_count += 1
                else:
                    # record internally for diagnostics if needed, but DO NOT write missing CSVs per request
                    missing_toolnames_all.append({"model": model_name, "gt_cmd": gt_cmd, "matched_key": matched_key})

                vec = metas_to_multihot(metas, LABELS)
                vec_np = np.array(vec, dtype=int)

                # raw accumulation: add total_score to each label present in this sample
                raw_accum += total_score * vec_np

                # support: increment per label if this sample contains the label
                for i, v in enumerate(vec):
                    if v:
                        support[i] += 1
                        global_tag_sample_counts[LABELS[i]] += 1

                # prepare per-sample row
                row = {
                    "custom_id": custom_id,
                    "gt_cmd": gt_cmd,
                    "matched_key": matched_key,
                    "title": title,
                    "metapackages": ",".join(metas) if metas else "",
                    "total_score": float(total_score),
                }
                for i, lab in enumerate(LABELS):
                    row[lab] = int(vec[i])
                sample_rows.append(row)

        per_model_agg[model_name] = {"raw_accum": raw_accum, "support": support, "samples": sample_count, "matched": matched_count}

        # save per-sample CSV for this model
        df_samples = pd.DataFrame(sample_rows)
        out_sample = OUTPUT_FOLDER / f"per_model_per_sample_{model_name}.csv"
        df_samples.to_csv(out_sample, index=False, encoding="utf-8-sig")
        print(f"[SAVED] per-sample CSV: {out_sample} (samples: {sample_count}, matched: {matched_count})")

        # NOTE: missing_toolnames files are intentionally NOT saved per your request.

    # ---------- Aggregate per-model results (raw, support, norm) ----------
    rows = []
    for model_name, v in per_model_agg.items():
        raw = v["raw_accum"]
        support = v["support"]
        samples = v["samples"]
        matched = v["matched"]
        d = {"model": model_name, "samples_processed": samples, "matched_samples": matched}
        for i, lab in enumerate(LABELS):
            d[f"raw_{lab}"] = float(raw[i])
            d[f"support_{lab}"] = int(support[i])
            # normalized by support (method B). If support==0 -> 0.0
            d[f"norm_{lab}"] = float(raw[i] / support[i]) if support[i] > 0 else 0.0
        rows.append(d)

    df_agg = pd.DataFrame(rows).sort_values("model")
    agg_out = OUTPUT_FOLDER / "per_model_dimensional_scores.csv"
    df_agg.to_csv(agg_out, index=False, encoding="utf-8-sig")

    # ---------- Write per-model normalized percentage scores (compact names like kweb) ----------
    pct_df = pd.DataFrame()
    pct_df['model'] = df_agg['model']

    for lab in LABELS:
        src_col = f"norm_{lab}"
        short_name = lab.replace("kali-tools-", "")

        if src_col in df_agg.columns:
            pct_df[short_name] = (df_agg[src_col].astype(float) * 100).round(2)
        else:
            pct_df[short_name] = 0.00

    pct_out = OUTPUT_FOLDER / "per_model_dimensional_percent_scores.csv"
    pct_df.to_csv(pct_out, index=False, encoding="utf-8-sig")
    print(f"[SAVED] Per-model normalized percent scores (compact names) -> {pct_out}")

    print(f"[SAVED] Aggregated per-model dimensional scores (raw/support/norm) -> {agg_out}")

    # ---------- global tag/sample counts ----------
    tag_rows = [{"label": k, "sample_count": v} for k, v in global_tag_sample_counts.items()]
    if tag_rows:
        df_tag = pd.DataFrame(tag_rows).sort_values("sample_count", ascending=False)
    else:
        df_tag = pd.DataFrame(columns=["label", "sample_count"])
    tag_out = OUTPUT_FOLDER / "multilabel_tag_counts.csv"
    df_tag.to_csv(tag_out, index=False, encoding="utf-8-sig")
    print(f"[SAVED] Tag/sample counts -> {tag_out} (rows: {len(df_tag)})")

    # NOTE: per your request, we do NOT save missing_toolnames CSVs (including overall).

    # one-time debug: check mapping for python-faraday
    print("[DEBUG] title_to_meta.get('python-faraday') ->", title_to_meta.get("python-faraday"))

    print("\n[DONE] Outputs are in:", OUTPUT_FOLDER)

if __name__ == "__main__":
    main()
