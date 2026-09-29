#!/usr/bin/env python3
"""Rebuild the project-page explorer from selected released dataset records.

Run: python3 pages/tools/sync-examples.py
The selection manifest records split and sample ID; source fields are never rewritten.
"""
from pathlib import Path
import hashlib
import html
import json
import re
import shlex

ROOT = Path(__file__).resolve().parents[2]
PAGES = ROOT / "pages"
SELECTIONS = Path(__file__).with_name("example-selections.json")
SPLITS = {
    "test": "KaliBench_data/kalibench_verified_test_5000.jsonl",
    "train": "KaliBench_data/kalibench_verified_train_3504.jsonl",
}
WORD = re.compile(r'''(?:[^\s'"\\]+|\\.|'[^']*'|"(?:\\.|[^"\\])*")+''')


def tokenize(example):
    """Preserve exact shell spelling while aligning spans to released annotations."""
    command = example["ground_truth_command"]
    options = example["optional_args"]
    aliases = {alias.strip(): key for key in options for alias in key.split("|")}
    positionals = example["positional_args"]
    pieces = []
    positional_index = 0
    seen_options = set()
    expected = None
    last = 0
    for index, match in enumerate(WORD.finditer(command)):
        if match.start() > last:
            pieces.append({"text": command[last:match.start()], "kind": None})
        raw = match.group()
        values = shlex.split(raw)
        assert len(values) == 1, (example["custom_id"], raw)
        value = values[0]
        parts = None
        if index == 0:
            assert value == example["tool_name"], (example["custom_id"], "tool", value)
            kind = "tool"
        elif expected is not None:
            assert value == str(expected), (example["custom_id"], "option value", value, expected)
            expected = None
            kind = "values"
        elif value in aliases:
            key = aliases[value]
            seen_options.add(key)
            expected = options[key]
            kind = "keys"
        else:
            # Joined forms such as -T5 and --option=value preserve their spelling.
            for alias in sorted(aliases, key=len, reverse=True):
                key = aliases[alias]
                annotated = options[key]
                if annotated is None:
                    continue
                joiner = "=" if value.startswith(alias + "=") else ""
                if not joiner and not (len(alias) == 2 and alias.startswith("-") and not alias.startswith("--")):
                    continue
                prefix = alias + joiner
                if value.startswith(prefix) and value[len(prefix):] == str(annotated):
                    assert raw.startswith(prefix), (example["custom_id"], "quoted key", raw)
                    parts = [{"text": alias, "kind": "keys"}]
                    if joiner:
                        parts.append({"text": joiner, "kind": None})
                    parts.append({"text": raw[len(prefix):], "kind": "values"})
                    seen_options.add(key)
                    break
            if parts is None:
                assert positional_index < len(positionals), (example["custom_id"], "unannotated token", raw)
                assert value == str(positionals[positional_index]), (example["custom_id"], "positional", value, positionals)
                positional_index += 1
                kind = "arguments"
        pieces.extend(parts or [{"text": raw, "kind": kind}])
        last = match.end()
    if last < len(command):
        pieces.append({"text": command[last:], "kind": None})
    assert expected is None, (example["custom_id"], "missing option value")
    assert positional_index == len(positionals), (example["custom_id"], "missing positional")
    assert seen_options == set(options), (example["custom_id"], "unused options", set(options) - seen_options)
    assert "".join(piece["text"] for piece in pieces) == command
    return pieces


def escape(value):
    return html.escape(str(value), quote=True)


def render_default(example):
    classes = {"tool": "tool", "keys": "key", "values": "value", "arguments": "argument"}
    command = "".join(
        escape(token["text"]) if not token["kind"] else
        f'<span class="command-token token-{classes[token["kind"]]}{" is-emphasized" if token["kind"] == "tool" else ""}" data-token="{token["kind"]}">{escape(token["text"])}</span>'
        for token in example["tokens"]
    )
    rows = "".join(
        f'<tr><th scope="row"><code data-annotation="keys">{escape(" | ".join(key.split("|")))}</code></th>'
        f'<td><code data-annotation="values">{escape("null" if value is None else value)}</code>'
        + ('<span class="null-explanation">Flag without a value</span>' if value is None else '')
        + '</td></tr>'
        for key, value in example["optional_args"].items()
    )
    positionals = '<ol class="positional-values">' + ''.join(
        f'<li><code data-annotation="arguments">{escape(value)}</code></li>' for value in example["positional_args"]
    ) + '</ol>' if example["positional_args"] else '<span class="empty-arguments">None</span>'
    has_options = bool(example["optional_args"])
    aliases = any('|' in key for key in example["optional_args"])
    return f'''
          <div class="example-columns">
            <div class="example-query">
              <h3>Natural-language request</h3>
              <blockquote id="example-query">{escape(example["query"])}</blockquote>
              <p class="example-provenance">Released dataset example<br><a id="example-source" href="https://github.com/RISys-Lab/KaliBench/blob/main/{escape(example["source_path"])}">{"Test" if example["split"] == "test" else "Training"} split · {escape(example["custom_id"])}</a></p>
            </div>
            <div class="example-command">
              <h3>Ground-truth command</h3>
          <div class="anatomy-controls" role="group" aria-label="Explore command components" data-example-enhanced hidden>
            <button class="anatomy-button active" data-component="tool" aria-pressed="true">Tool</button>
            <button class="anatomy-button" data-component="keys" aria-pressed="false">Optional keys</button>
            <button class="anatomy-button" data-component="values" aria-pressed="false">Optional values</button>
            <button class="anatomy-button" data-component="arguments" aria-pressed="false">Positional arguments</button>
          </div>
              <pre class="command-line" tabindex="0" aria-label="Ground-truth command for {escape(example["tool_name"])}"><code id="example-command">{command}</code></pre>
          <p class="anatomy-description" id="anatomy-description" data-example-enhanced hidden><strong>Tool selection.</strong> The executable is <code>{escape(example['tool_name'])}</code>.</p>
              <div class="example-annotation-summary">
                <div><h4>Tool</h4><div id="example-tool"><code data-annotation="tool">{escape(example["tool_name"])}</code></div></div>
                <div><h4>Positional arguments</h4><div id="example-positionals">{positionals}</div></div>
              </div>
              <div class="argument-pairs">
                <h4>Optional arguments</h4>
                <table id="example-options-table" aria-label="Optional argument key/value pairs"{'' if has_options else ' hidden'}>
                  <thead><tr><th scope="col">Key / aliases</th><th scope="col">Value</th></tr></thead>
                  <tbody id="example-option-rows">{rows}</tbody>
                </table>
                <p id="example-no-options"{ ' hidden' if has_options else ''}>No optional arguments in this example.</p>
                <p id="example-alias-note"{'' if aliases else ' hidden'}>Names separated by | are aliases for the same key.</p>
              </div>
            </div>
          </div>'''


def main():
    selection = json.loads(SELECTIONS.read_text())
    rows = {split: {row['custom_id']: row for row in map(json.loads, (ROOT / source).read_text().splitlines())} for split, source in SPLITS.items()}
    examples = []
    for item in selection['examples']:
        source = rows[item['split']][item['custom_id']]
        example = {key: item[key] for key in ('phase', 'dimension', 'dimension_label', 'split', 'custom_id')}
        if 'dimension_short_label' in item:
            example['dimension_short_label'] = item['dimension_short_label']
        example.update({key: source[key] for key in ('tool_name', 'query', 'ground_truth_command', 'optional_args', 'positional_args')})
        example['source_path'] = SPLITS[item['split']]
        example['tokens'] = tokenize(example)
        examples.append(example)
    assert len(examples) == 23 and len({e['dimension'] for e in examples}) == 23
    assert len({e['ground_truth_command'] for e in examples}) == 23
    assert [sum(e['phase'] == phase['id'] for e in examples) for phase in selection['phases']] == [6, 5, 4, 4, 4]
    data = {'phases': selection['phases'], 'sources': {split: {'path': source, 'sha256': hashlib.sha256((ROOT / source).read_bytes()).hexdigest()} for split, source in SPLITS.items()}, 'examples': examples}
    (PAGES / 'assets/examples.json').write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    initial = examples[0]
    phase_options = ''.join(f'<option value="{escape(phase["id"])}">{i + 1}. {escape(phase["short_label"])}</option>' for i, phase in enumerate(selection['phases']))
    dimension_options = ''.join(f'<option value="{escape(e["dimension"])}">{escape(e.get("dimension_short_label", e["dimension_label"]))}</option>' for e in examples if e['phase'] == initial['phase'])
    # Escape HTML-sensitive characters so source strings cannot close the data script.
    embedded = json.dumps(data, ensure_ascii=False, separators=(',', ':')).replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e')
    component = f'''<!-- BEGIN EXAMPLE EXPLORER -->
        <figure class="command-example" id="examples" aria-labelledby="example-title">
          <figcaption><h3 id="example-title">Explore benchmark examples</h3><span>23 dimensions · 5 security phases</span></figcaption>
          <div class="example-filters" data-example-enhanced hidden>
            <div class="example-filter"><label for="example-phase">Security phase</label><select id="example-phase">{phase_options}</select></div>
            <div class="example-filter"><label for="example-dimension">Dimension</label><select id="example-dimension">{dimension_options}</select></div>
            <div class="example-pagination" role="group" aria-label="Browse benchmark examples">
              <button id="example-previous" type="button" aria-label="Previous example" disabled><svg class="icon previous-arrow" aria-hidden="true"><use href="#icon-arrow"></use></svg></button>
              <span id="example-counter" aria-hidden="true">1 / 23</span>
              <button id="example-next" type="button" aria-label="Next example"><svg class="icon" aria-hidden="true"><use href="#icon-arrow"></use></svg></button>
            </div>
          </div>
          {render_default(initial)}
          <p class="sr-only" id="example-status" aria-live="polite" aria-atomic="true"></p>
          <noscript><p class="example-noscript">This is one of 23 examples. Enable JavaScript to explore each dimension, or download all examples below.</p></noscript>
          <div class="example-note"><p>Representative dataset examples. Commands are displayed, never run.</p><a href="assets/examples.json" download>Download examples <span>(JSON)</span></a></div>
        </figure>
        <script type="application/json" id="benchmark-examples-data">{embedded}</script>
        <!-- END EXAMPLE EXPLORER -->'''
    index = PAGES / 'index.html'
    page = index.read_text()
    if '<!-- BEGIN EXAMPLE EXPLORER -->' in page:
        pattern = r'<!-- BEGIN EXAMPLE EXPLORER -->[\s\S]*?<!-- END EXAMPLE EXPLORER -->'
    else:
        pattern = r'<figure class="command-example"[\s\S]*?</figure>'
    page, count = re.subn(pattern, lambda _: component, page, count=1)
    assert count == 1, 'Example component not found'
    index.write_text(page)
    print(f'Rebuilt {len(examples)} source-backed examples, exact command tokens, and the static fallback.')


if __name__ == '__main__':
    main()
