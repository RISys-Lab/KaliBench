"use strict";

// Source: authors' camera-ready reference draft, Table 1 (page 7).
// The draft PDF is intentionally excluded from the repository.
// Scores are transcribed as printed, in U / R / H order:
// exact correct, tool accuracy, optional-argument F1, positional-argument F1,
// total score, followed by average total score. All scores are percentages.
const MODEL_ROWS = [
  [
    "Foundation-Sec-Ins",
    "8",
    false,
    false,
    [
      14.7, 19.1, 56.2, 66.5, 92.5, 86.8, 33.5, 37.4, 75.6, 53.0, 60.0, 81.3,
      51.0, 63.3, 81.2, 65.2,
    ],
  ],
  [
    "Llama3.1-Ins",
    "8",
    false,
    false,
    [
      12.5, 18.2, 62.7, 61.7, 93.7, 95.4, 32.4, 37.2, 82.7, 53.5, 61.9, 85.8,
      49.2, 64.3, 88.0, 67.2,
    ],
  ],
  [
    "Llama-Primus-Base",
    "8",
    false,
    false,
    [
      15.8, 19.3, 60.3, 67.3, 90.7, 94.1, 36.6, 39.2, 80.8, 56.3, 61.9, 84.0,
      53.4, 63.9, 86.3, 67.9,
    ],
  ],
  [
    "Foundation-Sec-Rsn",
    "8",
    true,
    false,
    [
      13.3, 18.6, 66.6, 67.0, 92.8, 95.9, 34.8, 39.1, 85.5, 53.0, 57.9, 86.2,
      51.6, 63.3, 89.2, 68.0,
    ],
  ],
  [
    "Llama-Primus-Merged",
    "8",
    false,
    false,
    [
      16.3, 21.0, 65.2, 66.2, 93.4, 95.2, 36.7, 40.8, 84.1, 56.6, 63.2, 87.1,
      53.2, 65.8, 88.8, 69.3,
    ],
  ],
  [
    "RedSage-DPO",
    "8",
    false,
    false,
    [
      18.2, 25.4, 68.1, 66.8, 95.1, 95.7, 40.1, 46.1, 85.4, 56.7, 65.9, 88.5,
      54.5, 69.0, 89.9, 71.1,
    ],
  ],
  [
    "Qwen3",
    "8",
    false,
    false,
    [
      18.3, 24.1, 73.1, 68.9, 95.6, 96.3, 39.3, 43.4, 88.2, 58.5, 65.6, 90.5,
      55.5, 68.2, 91.6, 71.8,
    ],
  ],
  [
    "Qwen3",
    "8",
    true,
    false,
    [
      20.9, 27.7, 80.9, 70.3, 95.7, 96.2, 42.9, 47.2, 92.4, 59.9, 66.5, 93.4,
      57.7, 69.8, 94.0, 73.8,
    ],
  ],
  [
    "RedSage-Ins",
    "8",
    false,
    false,
    [
      19.7, 26.1, 67.2, 67.9, 94.6, 95.3, 42.4, 46.9, 85.0, 58.8, 66.4, 88.3,
      56.4, 69.3, 89.5, 71.7,
    ],
  ],
  [
    "RedSage-K (GRPO)",
    "8",
    true,
    true,
    [
      28.1, 33.0, 66.7, 76.5, 93.3, 93.1, 51.7, 53.7, 84.8, 73.9, 77.0, 88.4,
      67.4, 74.6, 88.8, 76.9,
    ],
  ],
  [
    "RedSage-K (SFT)",
    "8",
    false,
    true,
    [
      30.6, 34.4, 77.1, 78.1, 95.9, 97.1, 52.8, 55.2, 90.5, 66.9, 68.4, 92.0,
      65.9, 73.2, 93.2, 77.4,
    ],
  ],
  [
    "RedSage-K (SFT+GRPO)",
    "8",
    true,
    true,
    [
      32.2, 37.4, 69.4, 77.9, 95.0, 92.5, 56.1, 57.9, 86.5, 76.9, 80.9, 89.1,
      70.3, 77.9, 89.3, 79.2,
    ],
  ],
  [
    "Gemma-3-it",
    "27",
    false,
    false,
    [
      18.9, 24.0, 70.5, 73.4, 97.3, 87.0, 40.9, 44.2, 83.1, 60.2, 66.3, 88.4,
      58.1, 69.2, 86.2, 71.2,
    ],
  ],
  [
    "GPT-OSS",
    "20 / 3.6",
    true,
    false,
    [
      19.6, 24.6, 78.1, 72.3, 95.1, 95.9, 42.2, 45.6, 91.3, 57.1, 63.6, 92.0,
      57.2, 68.1, 93.0, 72.8,
    ],
  ],
  [
    "Mistral-Small-3.2",
    "24",
    false,
    false,
    [
      20.7, 27.4, 79.9, 70.6, 95.6, 96.2, 44.3, 48.2, 91.5, 62.6, 69.2, 93.2,
      59.2, 71.0, 93.7, 74.6,
    ],
  ],
  [
    "Qwen3",
    "32",
    false,
    false,
    [
      20.6, 27.7, 79.8, 73.5, 96.3, 95.5, 43.4, 48.9, 91.2, 62.8, 69.9, 93.4,
      59.9, 71.7, 93.4, 75.0,
    ],
  ],
  [
    "Qwen3",
    "32",
    true,
    false,
    [
      24.0, 30.9, 81.8, 72.4, 96.8, 95.2, 47.2, 51.6, 92.4, 65.3, 71.4, 94.0,
      61.6, 73.3, 93.9, 76.3,
    ],
  ],
  [
    "Llama-Primus-Ins",
    "70",
    false,
    false,
    [
      18.8, 25.0, 71.2, 72.8, 96.4, 92.1, 41.9, 46.3, 88.1, 59.6, 67.0, 90.2,
      58.1, 69.9, 90.1, 72.7,
    ],
  ],
  [
    "Llama-3.3-Ins",
    "70",
    false,
    false,
    [
      22.6, 28.7, 77.7, 72.1, 96.5, 94.1, 45.7, 49.9, 91.7, 61.9, 68.1, 92.5,
      59.9, 71.5, 92.7, 74.7,
    ],
  ],
  [
    "GPT-OSS",
    "120 / 5.1",
    true,
    false,
    [
      24.0, 32.0, 80.9, 73.6, 95.6, 96.0, 48.0, 52.9, 92.2, 61.4, 68.7, 92.7,
      61.0, 72.4, 93.6, 75.7,
    ],
  ],
  [
    "Qwen3-Coder-Next",
    "80 / 3",
    true,
    false,
    [
      26.2, 31.0, 79.9, 74.1, 96.8, 96.4, 48.8, 51.9, 91.4, 63.9, 70.7, 93.2,
      62.3, 73.1, 93.6, 76.3,
    ],
  ],
  [
    "Qwen2.5-Ins",
    "72",
    false,
    false,
    [
      25.4, 30.2, 80.6, 76.0, 97.4, 96.8, 47.7, 51.1, 91.5, 67.0, 72.2, 93.3,
      63.6, 73.6, 93.8, 77.0,
    ],
  ],
  [
    "DeepSeek-V3.2",
    "685 / 37",
    true,
    false,
    [
      33.0, 40.2, 84.5, 75.8, 98.3, 97.1, 57.0, 62.0, 93.5, 68.8, 75.4, 94.1,
      67.2, 78.6, 94.9, 80.2,
    ],
  ],
  [
    "GLM-5.2",
    "753",
    true,
    false,
    [
      41.3, 52.0, 74.8, 86.3, 94.2, 94.9, 75.3, 71.9, 88.4, 51.3, 78.8, 90.6,
      71.0, 81.7, 91.3, 81.3,
    ],
  ],
];

document.documentElement.classList.replace("no-js", "js");
document.querySelectorAll("[data-mode]").forEach((button) => {
  button.disabled = false;
});

const modeNames = ["Unrestricted", "Restricted", "Hinted"];
const metricOffsets = {
  exact: 0,
  tool: 3,
  optional: 6,
  positional: 9,
  total: 12,
};
const metricLabels = {
  exact: "Exact correct",
  tool: "Tool acc.",
  optional: "Optional F1",
  positional: "Positional F1",
  total: "Total score",
};
let selectedMode = 0;
let selectedMetric = "exact";
let showAllModels = false;
const status = document.querySelector("#interaction-status");

function makeElement(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function renderResults(announce = false) {
  const offset = metricOffsets[selectedMetric] + selectedMode;
  const sorted = [...MODEL_ROWS].sort(
    (a, b) =>
      b[4][offset] - a[4][offset] ||
      b[4][12 + selectedMode] - a[4][12 + selectedMode],
  );
  const visible = showAllModels ? sorted : sorted.slice(0, 8);
  const metricOrder = [
    selectedMetric,
    ...Object.keys(metricOffsets).filter((metric) => metric !== selectedMetric),
  ];
  const headerRow = document.querySelector(".results-table thead tr");
  const modelHeader = headerRow.querySelector("th");
  const sizeHeader = headerRow.querySelector('[data-column="size"]');
  const metricHeaders = Object.fromEntries(
    [...headerRow.querySelectorAll("[data-metric]")].map((header) => [
      header.dataset.metric,
      header,
    ]),
  );
  headerRow.replaceChildren(
    modelHeader,
    metricHeaders[selectedMetric],
    sizeHeader,
    ...metricOrder.slice(1).map((metric) => metricHeaders[metric]),
  );
  const fragment = document.createDocumentFragment();
  visible.forEach(([name, size, reasoning, ours, scores], index) => {
    const row = makeElement("tr", ours ? "our-model" : "");
    const modelCell = makeElement("th");
    modelCell.scope = "row";
    const rank = makeElement(
      "span",
      "rank",
      String(index + 1).padStart(2, "0"),
    );
    rank.setAttribute("aria-hidden", "true");
    modelCell.append(rank, makeElement("span", "model-name", name));
    if (reasoning) {
      const mark = makeElement("span", "reasoning-mark", "†");
      mark.setAttribute("aria-label", "Reasoning enabled");
      modelCell.append(mark);
    }
    if (ours) modelCell.append(makeElement("span", "ours-tag", "Ours"));
    row.append(modelCell);
    metricOrder.forEach((metric, metricIndex) => {
      const value = scores[metricOffsets[metric] + selectedMode];
      const cell = makeElement(
        "td",
        metric === selectedMetric ? "metric-selected" : "",
      );
      if (metric === "exact") {
        const score = makeElement("div", "score-cell");
        const track = makeElement("span", "score-mini");
        track.setAttribute("aria-hidden", "true");
        const bar = makeElement("i");
        bar.style.width = `${value}%`;
        track.append(bar);
        score.append(track, makeElement("span", "", value.toFixed(1)));
        cell.append(score);
      } else cell.textContent = value.toFixed(1);
      row.append(cell);
      if (metricIndex === 0) row.append(makeElement("td", "", size));
    });
    fragment.append(row);
  });
  document.querySelector("#results-body").replaceChildren(fragment);
  document.querySelector("#leaderboard-title").textContent =
    `${modeNames[selectedMode]} results`;
  document.querySelector("#table-count").textContent = showAllModels
    ? "All 24 models"
    : "Top 8 of 24";
  document.querySelectorAll("th[data-metric]").forEach((header) => {
    const active = header.dataset.metric === selectedMetric;
    header.textContent =
      metricLabels[header.dataset.metric] + (active ? " ↓" : "");
    header.classList.toggle("metric-selected", active);
    if (active) header.setAttribute("aria-sort", "descending");
    else header.removeAttribute("aria-sort");
  });
  if (announce)
    document.querySelector(".leaderboard .table-scroll").scrollLeft = 0;
  const showButton = document.querySelector("#show-all");
  showButton.textContent = showAllModels
    ? "Show top 8 models ↑"
    : "Show all 24 models ↓";
  showButton.setAttribute("aria-expanded", String(showAllModels));
  if (announce)
    status.textContent = `${modeNames[selectedMode]} results. Showing ${visible.length} of 24 models, sorted by ${metricLabels[selectedMetric]} from highest to lowest.`;
}

for (const button of document.querySelectorAll("[data-mode]")) {
  button.addEventListener("click", () => {
    selectedMode = Number(button.dataset.mode);
    document.querySelectorAll("[data-mode]").forEach((other) => {
      const active = other === button;
      other.classList.toggle("active", active);
      other.setAttribute("aria-pressed", String(active));
    });
    renderResults(true);
  });
}
document.querySelector("#sort-metric").addEventListener("change", (event) => {
  selectedMetric = event.target.value;
  renderResults(true);
});
document.querySelector("#show-all").addEventListener("click", () => {
  const button = document.querySelector("#show-all");
  const beforeTop = button.getBoundingClientRect().top;
  const collapsing = showAllModels;
  showAllModels = !showAllModels;
  renderResults(true);
  // Keep the focused control in view when a long table collapses.
  if (collapsing)
    window.scrollBy({
      top: button.getBoundingClientRect().top - beforeTop,
      behavior: "instant",
    });
});
renderResults();

// Clipboard API on HTTPS/localhost, with a fallback for locally opened pages.
async function copyText(text) {
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    /* Fall through to the local-file-compatible path. */
  }
  const previouslyFocused = document.activeElement;
  const helper = document.createElement("textarea");
  helper.value = text;
  helper.setAttribute("readonly", "");
  helper.style.cssText = "position:fixed;top:0;left:-9999px;opacity:0";
  document.body.append(helper);
  helper.select();
  let copied = false;
  try {
    copied = document.execCommand("copy");
  } catch {
    copied = false;
  }
  helper.remove();
  previouslyFocused?.focus({ preventScroll: true });
  return copied;
}
for (const button of document.querySelectorAll("[data-copy]")) {
  button.addEventListener("click", async () => {
    const text = document
      .getElementById(button.dataset.copy)
      .textContent.trim();
    const label = button.querySelector("span");
    const originalLabel = label.textContent;
    if (button.getAttribute("aria-disabled") === "true") return;
    button.setAttribute("aria-disabled", "true");
    const copied = await copyText(text);
    label.textContent = copied ? "Copied!" : "Select text to copy";
    status.textContent = copied
      ? "Copied to clipboard."
      : "Clipboard unavailable. Select the text and copy it manually.";
    window.setTimeout(() => {
      label.textContent = originalLabel;
      button.removeAttribute("aria-disabled");
    }, 2200);
  });
}

const menuToggle = document.querySelector(".menu-toggle");
const mobileNav = document.querySelector("#mobile-nav");
function closeMenu(returnFocus = false) {
  mobileNav.hidden = true;
  menuToggle.setAttribute("aria-expanded", "false");
  menuToggle.setAttribute("aria-label", "Open navigation");
  if (returnFocus) menuToggle.focus();
}
menuToggle.addEventListener("click", () => {
  const expanded = menuToggle.getAttribute("aria-expanded") === "true";
  mobileNav.hidden = expanded;
  menuToggle.setAttribute("aria-expanded", String(!expanded));
  menuToggle.setAttribute(
    "aria-label",
    expanded ? "Open navigation" : "Close navigation",
  );
});
mobileNav
  .querySelectorAll("a")
  .forEach((link) => link.addEventListener("click", () => closeMenu()));
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !mobileNav.hidden) closeMenu(true);
});
window.matchMedia("(min-width: 641px)").addEventListener("change", (event) => {
  if (event.matches) closeMenu();
});

if ("IntersectionObserver" in window) {
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        document.querySelectorAll(".desktop-nav a").forEach((link) => {
          if (link.hash === `#${entry.target.id}`)
            link.setAttribute("aria-current", "location");
          else link.removeAttribute("aria-current");
        });
      });
    },
    { rootMargin: "-15% 0px -65% 0px", threshold: 0 },
  );
  document
    .querySelectorAll("main section[id]")
    .forEach((section) => observer.observe(section));
}
