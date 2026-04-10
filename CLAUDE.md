# CLAUDE.md

## Project Overview

AI/ML training curriculum for **CodeTrain Africa** (Ghana), created by Speedykom Group GmbH, sponsored by GIZ. Target audience is MERN stack developers transitioning to Python and AI — code examples frequently draw JavaScript analogies and use Ghana-specific context (Accra Mall, MTN mobile money, etc.).

## Tech Stack

- **Python >= 3.12** (pinned for library stability, not 3.13)
- **uv** for package management (`uv sync` to install, `uv.lock` is committed)
- **Quarto** for authoring — `.qmd` files are the source of truth, `.ipynb` files are generated from them
- **Key libraries:** pandas, numpy, scikit-learn, torch, torchvision, matplotlib, seaborn, plotly
- **Dev dependency:** xgboost (in `[dependency-groups] dev`)

## Commands

```bash
uv sync                      # Install all dependencies
uv sync --group dev          # Install with dev dependencies (xgboost)
quarto render <file>.qmd     # Render a single .qmd to .ipynb + .html
quarto render                # Render entire project
python test_dataset.py       # Verify cassava dataset loads correctly
python 06-deep-learning/setup_gtsrb.py    # Download & organize GTSRB dataset
python 06-deep-learning/cassava_data.py   # Download cassava dataset from Kaggle
```

## File Structure

```
NN-module-name/              # Modules numbered 00-08, kebab-case names
├── intro_TOPIC1.qmd         # Quarto source (primary, edit these)
├── intro_TOPIC1.ipynb       # Generated notebook (do not edit directly)
├── intro_TOPIC1_files/      # Generated output assets (gitignored)
├── data/                    # Local datasets (gitignored, downloaded on-demand)
├── styles.css               # Optional per-module style override
└── *.py                     # Optional setup/download scripts
```

**Module progression:** 00-welcome → 01-setup → 02-version-control → 03-python-basics → 04-data-analysis → 05-machine-learning → 06-deep-learning → 07-agentic-ai → 08-resources

Modules are self-contained — no cross-module Python imports. Sequential ordering is implicit prerequisite only.

## Quarto Conventions

All `.qmd` files use this frontmatter pattern:

```yaml
---
title: "Module N: Topic"
subtitle: "Descriptive subtitle"
author: "Speedykom Group GmbH"
format:
  html:
    toc: true
    toc-depth: 3
    theme: cosmo
    code-fold: false
    code-block-bg: true
    code-block-border-left: "#31BAE9"
    css: styles.css
jupyter: python3
execute:
  eval: true          # false for setup guides and reports
  echo: true
  warning: false
  message: false
---
```

- Callout blocks: `::: {.callout-warning}`, `::: {.callout-note}`, `::: {.callout-tip}`, `::: {.callout-important}`
- Deep learning notebooks include Google Colab badge links pointing to `github.com/Speedykom/GIZ_GhanaAI_Training_Materials`
- Cell options use `#| code-fold: true/false` syntax

## Coding Conventions

- **snake_case** for variables and functions, **PascalCase** for classes, **UPPER_SNAKE_CASE** for constants
- Import order: stdlib → core data science (pandas, numpy) → ML frameworks (sklearn, torch) → visualization (matplotlib, seaborn) → utilities
- Educational tone: comments explain *why*, not just *what*
- F-strings for all string formatting
- Type hints used but optional (shown as "recommended for AI")
- No linter/formatter config — style is enforced by convention

## Brand / Styling

- Primary Blue: `#004367` (headings)
- Primary Teal: `#56A09A` (links, highlights)
- Accent border: `#31BAE9` (code blocks)
- Background: `#F8FAFC`
- Fonts: Inter (body), Fira Code (monospace)

## What's Gitignored

- `*.html`, `*.pdf` — rendered output
- `**/*_files/` — Quarto-generated asset directories
- `*.pth`, `*.h5`, `*.ckpt`, `*.bin` — model weights (except `cassava_model_full.pth` which is tracked)
- `cassava_dataset/`, module-level `data/` dirs — large datasets downloaded on-demand
- `.env` — Kaggle API credentials
- `08-resources/books/*.pdf` — copyrighted reference materials
- `/.quarto/` — Quarto cache

## Important Notes

- **Edit `.qmd` files, not `.ipynb`** — notebooks are generated from Quarto sources
- No CI/CD pipeline exists — rendering and testing are local only
- Datasets are not committed; students download them via setup scripts or Kaggle API
- Module 07 (agentic-ai) uses `eval: false` — it's a case study/guide, not executable notebooks
- The `_quarto.yml` in `00-welcome/` provides project-level Quarto defaults
