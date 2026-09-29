#!/usr/bin/env node
// Refresh the no-JavaScript results and downloadable citation after edits.
// Run from any directory: node pages/tools/sync-results.cjs
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const root = path.resolve(__dirname, "..");
const script = fs.readFileSync(path.join(root, "script.js"), "utf8");
const data = script.split("document.documentElement")[0];
const rows = vm.runInNewContext(data + "\nMODEL_ROWS;");
const escape = (text) =>
  String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
const offsets = [0, 3, 6, 9, 12];
const body = [...rows]
  .sort((a, b) => b[4][0] - a[4][0] || b[4][12] - a[4][12])
  .map(([name, size, reasoning, ours, scores], index) => {
    const cells = offsets
      .map(
        (offset, i) =>
          `<td${i === 0 ? ' class="metric-selected"' : ""}>${scores[offset].toFixed(1)}</td>${i === 0 ? `<td>${escape(size)}</td>` : ""}`,
      )
      .join("");
    return `<tr${ours ? ' class="our-model"' : ""}><th scope="row"><span class="rank" aria-hidden="true">${String(index + 1).padStart(2, "0")}</span><span class="model-name">${escape(name)}</span>${reasoning ? '<span class="reasoning-mark" aria-label="Reasoning enabled">†</span>' : ""}${ours ? '<span class="ours-tag">Ours</span>' : ""}</th>${cells}</tr>`;
  })
  .join("\n");
const filename = path.join(root, "index.html");
let html = fs.readFileSync(filename, "utf8");
if (!/<tbody id="results-body">[\s\S]*?<\/tbody>/.test(html))
  throw new Error("Results table not found");
html = html.replace(
  /<tbody id="results-body">[\s\S]*?<\/tbody>/,
  `<tbody id="results-body">\n${body}\n</tbody>`,
);
fs.writeFileSync(filename, html);
const citation = html.match(/<code id="bibtex">([\s\S]*?)<\/code>/);
if (!citation) throw new Error("Citation not found");
fs.writeFileSync(
  path.join(root, "assets/kalibench.bib"),
  citation[1].trim() + "\n",
);
console.log(
  `Updated ${rows.length} static result rows and the BibTeX download.`,
);
