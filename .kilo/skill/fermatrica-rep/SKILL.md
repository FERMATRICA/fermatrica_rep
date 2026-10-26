---
name: fermatrica-rep
description: FERMATRICA_REP framework expertise — ModelRep, options/forecasts, decomposition, waterfall, efficiency curves, budget optimization, PPTX export, code conventions, and canonical workflow. Use when writing or reviewing code in the fermatrica_rep package, working with FERMATRICA reporting functions, or debugging reporting pipelines.
---

# FERMATRICA_REP Framework

FERMATRICA_REP is the **reporting and post-modeling** component of the FERMATRICA econometrics framework. It operates on already-built models (produced by the core `fermatrica` package) and provides:

1. **Model exploration** — fit vs actual plots, dynamic decomposition, waterfall charts, statistical outputs, efficiency curves, adstock visualization, transformation chain inspection
2. **Forecasting & options** — KPI forecasts for future periods under specified budget scenarios. Translating annual budgets into media variables, computing summaries, and calculating Return on Investment (ROI)
3. **Budget optimization** — algorithmic optimization of budget split: maximize KPI with known budget or minimize budget for target KPI (COBYLA, Nevergrad CMA-ES)
4. **Export** — XLSX data export, PPTX presentation generation with standardised slides

## Multi-Repo Architecture

| Repo | Purpose | Python package |
|------|---------|---------------|
| `fermatrica` (core) | Model definition, transformation, optimization, evaluation | `fermatrica` |
| `fermatrica_utils` | Utility functions (`StableClass`, `DotDict`, `groupby_eff`, `select_eff`) | `fermatrica_utils` |
| `fermatrica_rep` (this repo) | Reporting: decomposition, waterfall charts, PPTX export, options, optimization | `fermatrica_rep` |

## Code Organization (This Repo)

```
fermatrica_rep/
├── basics.py                  # Common utilities: coef_var_align, palette_fill
├── meta_model/
│   ├── model_rep.py           # ModelRep class — reporting settings (vis_dict, language, palettes, trans_dict)
│   ├── node.py                # ModelNode class — unit of computation (model + ds + model_rep) + _pack_results
│   ├── meta_model.py          # MetaModel class (multi-model DAG)
│   └── diagnostics.py         # MetaModelDiagnostics — validation on demand (node types, attrs, edges, merge effects)
├── analysis/
│   ├── __init__.py
│   ├── stats.py               # Metrics tables, predictor tables, VIF, stationarity tests
│   ├── fit.py                 # Fit/predict plots (single & multiple brands, volume/value)
│   ├── category.py            # Category model fit/predict
│   ├── decomposition.py       # Dynamic decomposition over time (extract_effect, decompose_main_plot)
│   ├── waterfall.py           # Period decomposition (waterfall charts, waterfall_data)
│   ├── curves.py              # Short-term efficiency curves (RHS-only, fast)
│   ├── curves_full.py         # Full efficiency curves (model-blind, slower, handles LHS)
│   ├── adstocks.py            # Adstock decay visualization data
│   ├── transformation.py      # Transformation chain visualization (raw -> transformed)
│   ├── elasticity_price.py    # Price elasticity calculations
│   └── ols.py                 # Advanced OLS diagnostics
├── options/
│   ├── __init__.py
│   ├── define.py              # Low-level data-loading functions (scopes_read, trans_dict_create, budget_ssn_create, media_ssn_apply, etc.) + deprecated OptionSettings
│   ├── scope.py               # OptionScope class — named application scope (+ find_scope, filter_scopes)
│   ├── option.py              # Option class — named budget scenario
│   ├── translators.py         # Standard budget->media translators (basic, basic_fix, etc.)
│   ├── calc.py                # Deprecated: option_report, option_report_multi_var — use ModelNode.compute() or parallel.compute_multi() instead
│   ├── calc_multi.py          # Deprecated: multi-option grid (option_report_multi_post, curves), grid construction (opt_grid, opt_grid_expand)
│   ├── parallel.py            # Parallel multi-option dispatcher (compute_multi, compute_grid) using MetaModel API
│   ├── objective.py           # Objective functions and constraint wrappers for optimization
│   ├── context.py             # ComputeContext + load_context (context.xlsx reader) + campaigns_to_options (derive scopes/allocations/seasonality from a campaign table)
│   └── optim.py               # Budget optimization (COBYLA, Nevergrad CMA-ES)
├── export/
│   ├── __init__.py
│   ├── basics.py              # PPTX chart styling (colours, fonts, gridlines)
│   ├── export_pptx.py         # Full PPTX presentation generation + per-component functions
│   ├── export_results.py      # XLSX export
│   └── slides/                # Individual slide generators
│       ├── __init__.py
│       ├── fit_predict.py
│       ├── decomposition.py
│       ├── waterfall.py
│       ├── incr_roi.py
│       ├── options.py
│       └── adstocks.py
└── res/
    ├── dict/
    │   └── vis_dict.xlsx      # Default visual dictionary
    └── (other resources)
```

**Dependency chain within fermatrica_rep**: `basics` -> `meta_model` -> `options` -> `export`

The options module further splits: `define` -> `translators` -> `calc` + `calc_multi` + `parallel` -> `objective` -> `optim`

Analysis functions (`analysis/`) are **independent** of the options pipeline but accept `Model`, `ModelNode`, `MetaModel`, or the deprecated `list` in their first (`model`) or `model_rep` parameters. Passing a `ModelNode` or `MetaModel` auto-resolves `ds`, `model_rep`, and `dt_pred` from the node(s).

## Key Concepts

### ModelRep (Model Reporting Settings)

`ModelRep` is the central configuration object. It defines **how** to report, separate from **what** to report (the `Model` object).

```python
model_rep = fmr.ModelRep(
    ds=ds,
    language="english",
    trans_dict=trans_dict,
    tools_list=["tv", "digital", "pr"],
    palette_sbr_names=("glasbey",),
    palette_tool_names=("glasbey",),
    adhoc_code=[adhoc_model],
    option_translate_long_fn=my_custom_fn,
    ssn_dict=seasonality_dict
)
```

Fields:
- `vis_dict` — visual dictionary (display names, language-agnostic keys, default from `res/dict/vis_dict.xlsx`)
- `language` — `"english"` or `"russian"` (affects default display names)
- `trans_dict` — translation dictionary (budget -> media variable), required for options/forecasts
- `palette_sbr` / `palette_tools` — colour mapping for brands and marketing tools (auto-assigned via `palette_fill`)
- `adhoc_code_src` — source code of adhoc modules (stored for PPTX export)
- `option_translate_long` — function for long-term translation
- `option_summary_adhoc` — optional adhoc summary function
- `aud_coefs` — audience/coefficient adjustments (optional)
- `ssn_dict` — seasonality data (optional, for option calculations)
- `media_prices` — per-scope inflation multipliers, optional, no constructor param. Wire it after construction with a plain assignment (`model_rep.media_prices = media_prices`) or via `MetaModel(compute_context=ctx)`.

Like `Model`, `ModelRep` extends `StableClass` — no new attributes after `__init__`.

### OptionScope

`OptionScope` defines **where** and **when** to apply a budget scenario. Named scopes are atomic (no inheritance), defined in `options/scope.py`. Like `Model`, it extends `StableClass` — no new attributes after `__init__`.

```python
scope = fmr.OptionScope(
    scope_name="brandx_2025",
    date_start="2025-01-01",
    date_end="2025-12-31",
    ref_date_start="2024-01-01",
    ref_date_end="2024-12-31",
    filters={"superbrand": ["BrandX"]},
    plan_period="year",
    inflation_multiplier=1.04,
    seasonality="brandx_default",
)
```

Key fields: `scope_name` (unique, referenced by budget rows), `date_start`/`date_end` (forecast period), `ref_date_start`/`ref_date_end` (reference period; `ref_date_end` defaults to `date_start - 1 day` in the scopes sheet, required in the Python constructor), `filters` (`{column: [values]}`), `plan_period` (`"exact"`, `"year"`, `"hy"`, `"quarter"`), `inflation_multiplier`, `seasonality` (key into `ModelRep.ssn_dict`, falls back to `scope_name`).

**Manipulation API** (replaces the by-hand comprehension idioms):

| Helper | Replaces | Semantics |
|--------|----------|-----------|
| `s.start_year` (property) | `s.date_start.year` | start year of the scope |
| `s.has_filter(col)` | `'col' in s.filters` | the scope filters on `col` |
| `s.filter(col, value)` | `list(s.filters[col])[0] == value` / `'col' in s.filters and s.filters[col] == [value]` | the scope's filter on `col` has `value` as its first (only) value |
| `s.match(col, value)` | `s.filter(col, value) or not s.has_filter(col)` | the scope covers `value` on `col` — explicit value match **or** no filter on `col` at all (national scope covers every value) |
| `s.copy(deep=True)` | `copy.deepcopy(s)` | new independent scope |
| `s.replace_dates(...)` | manual `s.date_start = ...; s.date_end = ...` on a copy | new scope with the given dates overridden (retro-period idiom) |
| `s.shift_years(delta=1)` | manual date rolling | new scope with all four dates rolled by `delta` calendar years |

Collection-level helpers in `fermatrica_rep.options.scope` (exported from the package root):

- `fmr.find_scope(scopes, scope_name)` — first scope matching `scope_name`, else `None`.
- `fmr.filter_scopes(scopes, **criteria)` — AND-composed filter over a list of scopes; returns a **new** list, input order preserved, never mutates the input, `None`/absent scopes → `[]`. Criteria (all optional):
  - `scope_name` — exact match; a list means any-of (e.g. p07's dual-scope `('brand_1_2026', 'category_2026')`)
  - `start_year` / `start_year_min` / `start_year_max` — `s.start_year ==/>=/<=` the value (plan/retro splits)
  - `has_filter` / `without_filter` — col present/absent in `s.filters` (regional vs national scopes)
  - `match` — a `(col, value)` tuple evaluated via `s.match(col, value)` (per-region curves keeping national scopes)

```python
scopes_plan = fmr.filter_scopes(ctx.scopes, start_year_min=plan_year)
scopes_retro = [s.replace_dates(date_start='2025-01-01', date_end='2025-12-31')
                for s in fmr.filter_scopes(ctx.scopes, start_year=2026)]
scopes_regional = fmr.filter_scopes(ctx.scopes, has_filter='region')
scopes_crv = fmr.filter_scopes(ctx.scopes, start_year_min=2026, match=('region', rgn))
```

### Option

`Option` is a named budget scenario. Lives in `options/option.py`.

```python
option = fmr.Option.from_excel("options.xlsx", name="option_1")
option = fmr.Option(name="option_1", budget_data={
    "brandx_2025": {"bdg_nat_tv": 100, "bdg_digital": 30},
})
```

`Option` helpers:

- `opt.copy(deep=True)` — new independent option (preserves `default_scope` / `text_flags`); use `ctx.options['zero'].copy()` instead of rebuilding a zero option by hand.
- `opt.replace_budget(scope, items)` — new option with `scope`'s budget items replaced; other scopes untouched.
- `opt.budget_for_scope(scope_name)` — `opt.budget_data.get(scope_name, {})`.

### ModelNode and MetaModel

`ModelNode` bundles one model level: `(Model, ds, ModelRep)`. `MetaModel` is a DAG of `ModelNode`s (single node auto-named `"model"`). Both provide `compute()` — the central entry point for translating budgets, running transform + predict, and summarizing results.

```python
node = fmr.ModelNode(model_path="path/to/model_dir/", language='english')  # or (model=..., ds=..., model_rep=...)
mm = fmr.MetaModel(nodes={"uba": node_uba, "sales": node_sales},
                   edges=[("uba", "sales", pred_merge_1)],
                   compute_context=ctx)
results = mm.compute(option, scopes=ctx.scopes)
```

- `compute()` returns `{scope_name: summary}` by default or `{scope_name: (ds, dt_pred, summary)}` with `return_full=True`; for a `MetaModel` it returns `{node_name: {scope_name: summary}}`. Scopes are processed sorted by `date_start`, each budget projected forward via `_translate_long`. `compute()` also stores `dt_pred` on the node.
- **No-arg compute** (`mm.compute()` / `node.compute()` with no option) runs a modelling-stage refresh: transform + predict (must already be fitted), merging predecessor `dt_pred` via edge merge functions.
- **compute_context**: `ctx = fmr.load_context("context.xlsx")` returns a frozen `ComputeContext` (`path`, `scopes`, `options`, `ssn_dict`, `media_prices`, `trans_dicts` keyed by node name). `MetaModel(compute_context=ctx)` wires each node's `ModelRep` (`trans_dict`/`ssn_dict`/`media_prices` when the node doesn't already have one). Scopes are **not** stored on the MetaModel — pass them to every computation entry point.
- **Save/load round-trip**: `mm.save(path=..., save_tag=..., include_context_path=...)` writes a self-contained `meta_<tag>/` dir (models, `model_rep.pkl.lzma`, `meta_conf.toml` with edges + `scope_names`, optional `context.xlsx` copy); `fmr.MetaModel.load(meta_dir)` restores it. `dt_pred` is not persisted — run the no-arg refresh after load.

### Deprecated

`OptionSettings`, `MetaModelConf`/`MetaModelObj`, the old `option_report*` wrappers, the old `export_*` wrappers (`export_option_table`, `export_option_detail`, `export_curves`, `export_adstocks`), and passing `list` of `Model`/`ModelRep` are deprecated. Use `OptionScope` + `Option` + `ModelNode.compute()` / `MetaModel.compute()` and the clean-named exports (`options_table`, `options_detail`, `curves`, `adstocks`). Old calls still work but emit `DeprecationWarning`.

### Translation (Budget → Media)

Translation bridges the gap between money (budgets) and media variables (OTS, impressions). Defined in a `translation` sheet in `context.xlsx`:

| Column | Example | Description |
|--------|---------|-------------|
| `item` | `bdg_tv` | Budget key (used in option dict) |
| `function` | `basic` | Translator function name (from `translators.py`) |
| `variable` | `brnd_tv_ots_clip` | Target media variable in dataset |
| `base_price` | `0.05` | Cost per OTS/impression (or formula) |
| `max_budget` | `100` | Maximum budget for curves calculation |

The `function` column references a translator from `translators.py` (or custom). **All highest-level translator functions must have the exact same signature** as `basic`. This is a hard requirement — the translation dispatcher in `ModelNode` calls all translators with the same set of arguments. The standard interface is:
```python
def translator_name(md, ds, model_rep, var_key, var_name, val, ds_mask, media_price=None):
    return ds
```
`tv_complex` is the only exception — it is a **worker function** with additional parameters (`clip_dur`, `tv_type`, `media_prefix`), not a highest-level translator. The top-level TV translators (`nat_tv_complex`, `reg_tv_complex`, etc.) wrap `tv_complex` and conform to the standard signature.

Translation functions (defined in `fermatrica_rep.options.translators`):
- `basic` — 1-to-1: money / price = media volume (with seasonality correction)
- `basic_fix` — set fixed value on masked rows (no price conversion)
- `price` — apply price ratio (current price / base price)
- `tv_complex` — worker function (extended params), wrapped by TV variants below
- `nat_tv_complex` — national TV variant (standard signature)
- `reg_tv_complex` — regional TV variant (standard signature)
- `nat_tv_sponsor_complex` — national TV sponsorship variant (standard signature)
- `brnd_nat_tv_complex` — brand-level national TV variant (standard signature)
Custom translators can be added via `adhoc_*` prefix in user code and passed to `ModelRep` via `adhoc_code` parameter. Custom translators must also follow the standard signature.

When a seasonality column (`opt_media_ssn_*`) exists in the dataset but contains NaN/None values over the ds_mask filter, `basic` and `tv_complex` emit a `fermatrica_warner.warning` with the `var_key` and the date range of the problematic rows.

### Decomposition

Two types of decomposition are available:

**Dynamic decomposition** (`decomposition.py`): Values broken down into effects over time. Uses `extract_effect()` on the model output, then `decompose_main_plot()` visualizes. Supports both volume and value decomposition, pretrain display, and category models.

**Waterfall decomposition** (`waterfall.py`): Aggregated effects for a specific time period as a waterfall chart. Uses `waterfall_data()` to prepare and `waterfall_plot()` to visualize. Supports absolute and relative sorting, positive/negative separation.

Both receive prepared data as `split_m_m` — a DataFrame produced by `extract_effect()`.

### Efficiency Curves

**Short-term curves** (`curves.py`): Fast, RHS-only formula-based approach. Use when your model has no LHS effects or when doing WIP analysis. Functions: `curves_simple_plot()`, `curves_simple_data()`.

**Full curves** (`curves_full.py`): Model-blind approach that calculates a grid of budget options and collects predictions. Handles any model design (RHS + LHS). Slower, uses multiprocessing. Functions: `curves_full_plot_short()`, `curves_full_plot_long()`. Input data from `parallel.compute_grid()`.

### Budget Optimization

Four modes:
1. **Maximize KPI with known budget**: `optimize_budget_local_cobyla()` or `optimize_budget_local_ncma()`
2. **Minimize budget for target KPI**: `optimize_target_local_cobyla()` or `optimize_target_local_ncma()`
3. **Parallel multi-run** (COBYLA only): `optimize_budget_local_cobyla_multi()` runs multiple budget sizes in parallel via ``multiprocessing.Pool.starmap``; `optimize_target_local_cobyla_multi()` runs multiple KPI targets in parallel. Returns ``dict[float, Option]`` keyed by budget/target value. Handles pickling failure with sequential fallback. New-API only (``MetaModel`` / ``OptionScope`` / ``Option``).
4. **Multi-node optimization**: Optimize against a weighted combination of multiple DAG nodes. Pass ``summary_nodes`` to any optimizer: ``{"sales": (0.7, scope_name), "awareness": (0.3, scope_name)}`` (a list ``[weight, scope_name]`` works too). Combined score = ``sum(w_i * score_i / baseline_i)``. ``None`` = single-sink behavior (backward compatible).

Algorithms:
- **COBYLA** — derivative-free constrained optimization (NLopt). Good for noisy objectives.
- **Nevergrad CMA-ES** — evolutionary strategy for rugged landscapes. Better for complex constraints.

COBYLA supports epochs (restarts from different starting points), convergence tolerances, and verbose output.

The ``summary_nodes`` parameter is available on all 4 public functions, 2 multi functions, and threads through to ``ObjectiveFunctionBudget`` and ``ConstraintFunctionTarget``. When set, the baseline is computed once pre-loop by ``_compute_baseline()`` which deep-copies ``node.ds``, runs ``MetaModel.compute()``, extracts per-node scores, and restores original ``ds``. Passing empty ``{}``, a missing node name, or a zero/NaN baseline raises ``FermatricaError``. The objective helpers ``_get_score_key(if_long, if_volume)`` and ``_compute_score(results, ...)`` in ``objective.py`` handle the score extraction logic.

### MetaModelDiagnostics (Validation on Demand)

Validates a `MetaModel` object at runtime (not at construction). Returns a structured `DiagnosticsReport`.

```python
diag = fmr.MetaModelDiagnostics(meta_model)
report = diag.run()
print(report.passed)   # True if all error-level checks pass
print(report.summary)  # Formatted multi-line report
```

Four checks: node types (`ModelNode` instances), node attributes (`model`, `ds`, `model_rep`, `dt_pred`), edge resolution (merge function lookup + signature), and merge effects (`compute()` with 2 options, compare `ds`/`dt_pred`). Check #4 skipped unless `options` and `scopes` provided (requires `len(options) >= 2`). `raise_on_first=True` raises on first error-level failure.

### PPTX Export

1. **Component-wise**: Individual slide functions for flexible assembly

Standard slides include fit & predict by superbrand, fit & predict by SKU/category, dynamic decomposition, waterfall decomposition, short/long-term curves, option summary tables, and detailed option reports.

Requires MS Office on Windows for proper functionality.

## Options / Scenarios Configuration

Options are defined in an Excel file (`context.xlsx`) with these sheets:

### Sheet: `scopes` — Named application scopes

| Column | Type | Description |
|--------|------|-------------|
| `scope_name` | str | Unique scope name referenced by allocation rows |
| `date_start` | date | Start of the forecast period |
| `date_end` | date | End of the forecast period |
| `ref_date_start` | date | Start of the reference period |
| `ref_date_end` | date | End of the reference period. If omitted, defaults to `date_start - 1 day` |
| `apply_vars` | str | Comma-separated column(s) to filter on (e.g., `"superbrand"`) |
| `target` | str | Comma-separated values for apply_vars (e.g., `"BrandX"`) |
| `plan_period` | str | `"exact"`, `"year"`, `"hy"`, `"quarter"` |
| `inflation_multiplier` | float | Default inflation for all tools in this scope |
| `seasonality` | str | Optional reference to a key in the seasonality dict. Falls back to `scope_name` if omitted |

### Sheet: `allocations` — Budget allocation scenarios (long format)

| Column | Type | Description |
|--------|------|-------------|
| `option` | str | Option name |
| `scope_name` | str | Scope name referencing `scopes.scope_name` |
| `item` | str | Budget key matching `translation.item` |
| `val` | float | Budget value (millions) |

### Sheet: `translation` — Budget-to-variable mapping

| Column | Type | Description |
|--------|------|-------------|
| `item` | str | Budget key (e.g., `"bdg_tv"`, `"bdg_digital"`) |
| `function` | str | Translator function name |
| `variable` | str | Target dataset variable |
| `base_price` | float | Media price (cost per unit) |
| `max_budget` | float | Maximum budget for curves (millions) |
| `item_group` | str | Group for budget allocation (auto-derived) |

### Sheet: `media_prices` — Per-tool inflation overrides (NEW, optional)

| Column | Type | Description |
|--------|------|-------------|
| `scope_name` | str | Scope name |
| `item` | str | Budget key |
| `multiplier` | float | Inflation multiplier (overrides scope's `inflation_multiplier`) |

### Sheet: `seasonality` — Seasonal coefficients (keyed by scope_name)

| Column | Type | Description |
|--------|------|-------------|
| `scope_name` | str | Scope name the seasonality applies to |
| `period` | int | Week/month number |
| `tv` | float | Seasonality coefficient for TV |
| ... | ... | Additional columns for each tool |

Note: column names are bare (no `bdg_`/`nobdg_` prefix). `budget_ssn_create()` strips prefixes automatically; `seasonality_read()` reads them as-is.

### Custom translators

Add `adhoc_*` functions to your code and reference them in the `function` column. Custom translators must follow the same fixed signature as `basic`:
```python
def adhoc_translator_name(md, ds, model_rep, var_key, var_name, val, ds_mask, media_price=None):
    return ds
```

## Code Conventions

- **Imports**: stdlib → third-party → fermatrica_utils → fermatrica → fermatrica_rep
- **Errors**: `fermatrica_error("message")` — never raw `raise`
- **Type annotations**: Python 3.10+ union syntax: `str | None`, not `Optional[str]`
- **Performance**: `@profile` from `line_profiler` (no-op when inactive)
- **Plotly**: `pio.templates.default = 'ggplot2'`, colours from `colorcet` via `palette_fill()`
- **Multiprocessing**: `ProcessPoolExecutor` in options/optim — check environment compatibility
- **StableClass**: `ModelRep` — no new attributes after `__init__`

## Canonical Reporting Workflow

```python
import pandas as pd
import fermatrica as fm
import fermatrica_rep as fmr

# 1. Load data and trained model
ds = pd.read_excel("data.xlsx", parse_dates=["date"])
model, ds, errors = fm.model_load_ext("path/to/model_dir/")

# Or auto-load via ModelNode (model_path)
node = fmr.ModelNode(model_path="path/to/model_dir/", language="english")

# 2. Transform and predict (if not already done)
pred, pred_raw, model = fm.fit_predict(ds, model, if_fit=False, return_full=True)

# 3. Create ModelRep (reporting settings)
model_rep = fmr.ModelRep(
    ds=ds,
    language="english",
    tools_list=["tv", "digital", "pr"],
)

# 4. Standard reports (exploration)
metrics = fmr.predictors_table(model)
fit_data = fmr.fit_main_data(model, pred, period="week")
fmr.fit_main_plot_vol(model, ds, pred)
decomp = fmr.decompose_main_plot(model, ds, pred)

# 5. Waterfall for specific period
split_m_m = fmr.extract_effect(ds, model)
fmr.waterfall_plot(split_m_m, brands=["brand_1"], model_rep=model_rep)

# 5. Load scopes, options, seasonality, media prices and translation in one call
ctx = fmr.load_context("context.xlsx")
scopes = ctx.scopes
options = ctx.options

# 6. Build a node / MetaModel and compute a forecast
node = fmr.ModelNode(model=model, ds=ds, model_rep=model_rep)
meta_model = fmr.MetaModel(nodes=node, compute_context=ctx)
opt_sum = meta_model.compute(options['predefined'], scopes=scopes)

# 7. Efficiency curves (full, model-blind)
options_grid = fmr.opt_grid(translation, budget_step=1)
results = fmr.compute_grid(meta_model, scopes=scopes, translation=translation)
fmr.curves_full_plot_short(results, meta_model)

# 8. Budget optimization
borders_dict = {"bdg_tv": [0, 100, 50], "bdg_digital": [0, 50, 20]}  # {var: [low, high, start]}
option = fmr.Option(name="opt1", budget_data={scope_name: {"bdg_tv": 50, "bdg_digital": 30}})
opt_result = fmr.optimize_budget_local_cobyla(
    meta_model=meta_model, scopes=scopes, budget_scope=scope_name,
    borders_dict=borders_dict, option=option, bdg_size=100
)

# Parallel multi-budget optimization (3 budget sizes, pool of workers)
opt_results = fmr.optimize_budget_local_cobyla_multi(
    meta_model=meta_model, scopes=scopes, budget_scope=scope_name,
    borders_dict=borders_dict, option=option, bdg_sizes=[50, 100, 200],
    epochs=3, iters_epoch=300,
)
# opt_results = {50.0: Option, 100.0: Option, 200.0: Option}

# Multi-node optimization (optimize against weighted combination of DAG nodes)
opt_multi = fmr.optimize_budget_local_cobyla(
    meta_model=meta_model, scopes=scopes, budget_scope=scope_name,
    borders_dict=borders_dict, option=option, bdg_size=100,
    summary_nodes={"sales": (0.7, scope_name), "awareness": (0.3, scope_name)},
)
```

## Common Pitfalls

1. **Forgetting to create ModelRep**: Most reporting functions require a ModelRep object. Create it before calling any plot/export functions.

2. **Not providing trans_dict for options**: Options and forecasts cannot be calculated without a translation dictionary. Wire it via `MetaModel(compute_context=ctx)` (from `fmr.load_context(...)`), or set `trans_dict` on the `ModelRep` before `MetaModel.compute()`.

3. **Using `curves.py` with LHS models**: Short-term curves (`curves.py`) are RHS-only. For models with significant LHS effects, use `curves_full.py` which is model-blind.

4. **Modifying ModelRep after init**: `ModelRep` extends `StableClass` — no new attributes after `__init__`. Set all fields during construction.

5. **Running PPTX export on non-Windows**: PPTX export requires MS Office on Windows. Operation on Linux/macOS is not guaranteed.

6. **Not handling multiprocessing contexts**: `compute_grid` and `optim.*` use multiprocessing. Avoid running them from environments that don't support it (e.g., some Jupyter notebook configurations).

7. **Calling plot functions without checking data ranges**: Decomposition and fit functions depend on the `listed` column having the expected values (`2`, `3`, `4`). Missing future periods (`listed=4`) will cause empty plots.

8. **Mixing models and ModelReps between brands**: When working with multi-brand panel models, ensure `model_rep.palette_sbr` includes all brands in the dataset. Use `palette_fill()` before calling plot functions.

9. **Not resetting the model context for options**: Option calculation internally transforms the dataset. If running multiple option scenarios sequentially, ensure the dataset is re-loaded or re-transformed between runs.

10. **Confusing short-term and long-term curves**: Short-term curves exclude adstock decay effects (immediate response only). Long-term curves include the full decay tail. Choose based on the business question being answered.
