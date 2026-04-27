# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LTV Analyzer Advanced — a Streamlit web application for Customer Lifetime Value (LTV) analysis using Weibull survival modeling. Supports Japanese/English bilingual UI with multiple currency formats.

## Running the App

```bash
streamlit run app_pro.py
```

No `requirements.txt` exists. Key dependencies: `streamlit`, `pandas`, `numpy`, `scipy`, `matplotlib`, `plotly`, `python-pptx`, `lxml`.

## Architecture

Three files, no packages or subdirectories:

- **`app_pro.py`** (~3300 lines) — The entire Streamlit application in a single file. Contains:
  - CSS theme (dark mode, custom fonts: BIZ UDPGothic + IBM Plex Mono)
  - Statistical core: Kaplan-Meier estimation (`compute_km`), Weibull fitting (`fit_weibull`), LTV calculations (`ltv_inf`, `ltv_horizon`, `ltv_horizon_offset`, `ltv_horizon_spot`)
  - CSV ingestion with flexible column auto-mapping (Japanese/English column names)
  - Sidebar controls: language, currency, business type (subscription vs spot), billing cycle, dormancy, segment selection
  - Three built-in sample datasets generated inline (e-learning subscription, coworking subscription, D2C e-commerce spot)
  - Export: Excel, PowerPoint (via `pptx_export`), PDF
  - Segment analysis with per-segment Weibull fits

- **`lang.py`** — Internationalization module. `T(key, **kwargs)` function returns localized strings from a flat `_DICT` dictionary keyed by `{section}_{concept}` (e.g., `sidebar_cur_label`, `summary_ltv_inf`). Global language state managed by `set_lang()`/`get_lang()`. Also defines currency formatting (`fmt_c`, `cur_symbol`, `cur_decimal`) and business type/billing cycle constants.

- **`pptx_export.py`** — PowerPoint export using `python-pptx`. Expects a `.pptx` template file. Handles Japanese font discovery for matplotlib charts embedded in slides. Key function: `generate_pptx(...)`.

## Key Conventions

- All UI text uses `T('key_name')` from `lang.py` — never hardcode user-facing strings. Dictionary keys follow `{section}_{concept}` naming.
- Currency formatting always goes through `fmt_c(value, currency_code)`.
- Business types: `BIZ_SUBSCRIPTION` and `BIZ_SPOT` constants — logic branches significantly between these two modes.
- Billing cycles: `BILLING_CALENDAR_MONTHLY`, `BILLING_ANNUAL_365`, `BILLING_CUSTOM_DAYS`, `BILLING_FIXED_30`, `BILLING_DAILY_SPOT`.
- Streamlit caching: `@st.cache_data` is used on `compute_km`, `fit_weibull`, and `load_and_preprocess_csv`.
- Comments and internal variable names are primarily in Japanese.
- The Weibull survival function: `S(t) = exp(-(t/λ)^k)` where k = shape, λ = scale.

## Development Rules

### Version Management
- Version bump: `sed -i 's/vXXX/vYYY/g' app_pro.py`
- Verify: `grep "Intelligence" app_pro.py`
- Syntax check: `python3 -c "import ast; ast.parse(open('app_pro.py').read())"`

### PPTX Technical Notes
- solidFill must be insert(0) at rPr head or PowerPoint ignores it
- _copy_slide loses background; dark background (#0A0E14) must be set explicitly on cSld
- Japanese fonts via packages.txt (fonts-ipafont-gothic) on Streamlit Cloud
- Template must stay at 10 slides (never use output as new template)
- Template file: LTV-analyzer.pptx

### PPTX Slide Structure
S1 Title, S2 Analysis Summary (5 KPIs + conclusion), S3 Analysis Reliability (Survival/Weibull), S4 Provisional LTV Table (8 rows x 5 cols), S5 Transition Graph, S6 Segment Cover, S7 LTV∞ Comparison Bar Chart, S8 Segment Summary, S9 Top Pick Segment Detail, S10 Standard Segment Detail

### Segment Analysis
- Calculate once into `all_seg_results`, referenced by all outputs (app/Excel/PDF/PPTX)
- Independent recalculations are prohibited

### Sample Data
- 3 patterns: video learning / coworking / cosmetics EC
- n=10,000 each, single plan, 3 segment columns

### Fonts
- Japanese graphs: Noto Sans CJK JP (`/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc`)
- Streamlit Cloud: packages.txt with fonts-ipafont-gothic

### Terminology
- "trial" → "demo page" (unified)
