# KaliBench project page

A self-contained, responsive static project page for the NeurIPS 2026 Evaluations and Datasets Track paper. No build step, package installation, external font service, or backend is required.

## Preview locally

From the repository root:

```bash
python3 -m http.server 8765 --directory pages
```

Open <http://localhost:8765>. You can also open `index.html` directly; clipboard access depends on browser permissions.

## Publish to GitHub Pages

1. Commit `pages/` and `.github/workflows/pages.yml`, then merge them into `main`.
2. In the repository's **Settings → Pages → Build and deployment**, choose **GitHub Actions** as the source.
3. Run **Deploy project page** from the Actions tab on `main`, or push a change to `pages/` on `main`.

The included workflow uploads only `pages/` and deploys it with GitHub's official Pages actions. It also requires the `main` branch for manual runs. The expected public URL is <https://risys-lab.github.io/KaliBench/>. Relative asset paths support GitHub Pages project subdirectories.

The workflow is prepared locally; creating this page does not publish it or change repository settings. See [GitHub's custom-workflow documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages) for the hosting configuration.

## Content and sources

- **Paper:** the supplied camera-ready draft is a private content reference and is not distributed in this repository or the Pages site. The public arXiv version is in preparation; the page shows a noninteractive “coming soon” notice until its URL is available.
- **Pipeline:** `assets/pipeline.png` is a high-resolution crop of **Figure 2, page 3 of the supplied camera-ready reference**. It is deliberately not copied from the outdated root `assets/pipeline.png`.
- **Results:** all 24 model rows in `script.js` reproduce Table 1 on page 7. Each row stores 16 printed scores: the three modes for each of five metrics, followed by average total score. Ties in the selected metric are ordered by total score. Table 2's proprietary/scaffolded systems appear in a separate expandable section.
- **Command illustration:** the Nmap example is in Figure 1 and the released evaluation data. The illustrated ordering of optional flags is equivalent to the dataset command. The example never executes commands.
- **Taxonomy and split:** Section 3 of the supplied PDF. The older root coverage graphic is not used.
- **Acceptance:** the NeurIPS 2026 acceptance and track label were supplied by the project authors.
- **Citation:** `assets/kalibench.bib` and the inline citation use the supplied author list and acceptance information. Add proceedings identifiers when they are available.

This is the paper's evaluation snapshot, not a continuously updated leaderboard. The 41.3% finding applies to evaluated open-weight configurations; the additional proprietary/scaffolded results are labeled separately. The page also notes hinted-mode regressions from GRPO.

## Editing

- `index.html`: paper content, links, authors, metadata, and citation.
- `styles.css`: layout, colors, responsive rules, print styles, and reduced-motion support.
- `script.js`: command component selector (tool, optional keys, optional values, positional argument), mode/metric controls, model scores, clipboard controls, and mobile navigation.
- `guides/evaluation.html` and `guides/training.html`: self-contained guides adapted from the repository documentation, including environment setup.
- `tools/sync-results.cjs`: dependency-free helper to refresh the static results and downloadable citation after content edits.
- `assets/`: downloadable citation, current pipeline figure, the original `assets/logo.png` (also used as the favicon), social preview, and self-hosted fonts.

If the repository, hosting URL, or dataset moves, update links in `index.html`, the BibTeX URL in both locations, and the canonical/Open Graph URLs. The dataset link currently follows the existing repository documentation: `anonymous62567/KaliBench-Verified` on Hugging Face.

The fonts (DM Sans and IBM Plex Mono) are bundled with their SIL Open Font License files in `assets/fonts/`. The page loads no analytics or third-party scripts.

The three trained model labels are **RedSage-K (SFT+GRPO)**, **RedSage-K (SFT)**, and **RedSage-K (GRPO)**. A note below the results maps them to the paper’s original Kali-\* labels without changing any scores.

After editing the results in `script.js` or the inline BibTeX, run:

```bash
node pages/tools/sync-results.cjs
```

This refreshes all 24 unrestricted rows in the no-JavaScript fallback and `assets/kalibench.bib`. The interactive default shows the top 8; readers without JavaScript receive all 24. Keep the manually authored findings and training comparison aligned with any future result changes. The bundled guides are a publication snapshot; update their HTML when the corresponding `docs/` instructions change.

## Adding the public arXiv link

When the public arXiv URL is available, add a Paper link beside Code and Dataset in the hero, remove `.paper-status`, and replace the noninteractive `.resource-pending` block with a linked paper resource. Update the arXiv badge and paper entry in the root README. Table and appendix citations currently remain plain text; link them to the public paper if useful. Do not restore a bundled draft PDF. The two paper asset locations are covered by `.gitignore`.

## Verification

The final redesign was checked in Chromium at 320, 390, 768, 1024, and 1440 pixel widths. Verification covered all three modes and five metric sorts, expansion/collapse of 24 results, command highlighting, clipboard actions, mobile navigation, local links, the PDF-derived figure, and the no-JavaScript fallback. All 384 transcribed model values were compared with the supplied PDF. An independent critic completed two reviews; the second found no publication blockers. The redesigned page uses 18px desktop body text, 17px mobile body text, and at least 14px for interface text and metadata (excluding superscript affiliation markers).

Automated axe-core 4.10.3 checks reported no WCAG 2 A/AA, 2.1 AA, or 2.2 AA violations in the tested desktop/mobile default and expanded states, or in either guide. This is an automated check, not a claim of complete accessibility conformance. The deployed GitHub Pages site still needs a live smoke check after publishing.
