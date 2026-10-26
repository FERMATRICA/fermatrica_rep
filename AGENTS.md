# FERMATRICA_REP — AI Instructions

> **⚠️ Maintenance reminder**: If you add new modules, rename directories, change conventions, or modify the project structure, update this file. The AI does not auto-detect staleness. Run `/fermatrica-rep-sync-instructions` to have the AI review and update this file against the current codebase.

## Framework Overview

FERMATRICA_REP is the **reporting and post-modeling** component of the FERMATRICA econometrics framework. It operates on already-built models (produced by the core `fermatrica` package) and provides:

1. **Model exploration** — fit vs actual plots, dynamic decomposition, waterfall charts, statistical outputs, efficiency curves, adstock visualization, transformation chain inspection
2. **Forecasting & options** — KPI forecasts for future periods under specified budget scenarios. Translating annual budgets into media variables, computing summaries, and calculating Return on Investment (ROI)
3. **Budget optimization** — algorithmic optimization of budget split: maximize KPI with known budget or minimize budget for target KPI (COBYLA, Nevergrad CMA-ES)
4. **Export** — XLSX data export, PPTX presentation generation with standardised slides

The core model repository (`fermatrica`) handles model definition, transformation, optimization, and evaluation. This repo handles everything after the model is built.

## Multi-Repo Architecture

FERMATRICA is developed across three repositories. This repo (`fermatrica_rep`) is the **reporting** component. When working on code that depends on or affects the other repos, check their READMEs for details. Cross-repo changes require coordination.

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
│   ├── elasticity_price.py    # Price elasticity (ModelNode + OptionScope + Option API, arc elasticity via centered differences on a price grid)
│   └── ols.py                 # Advanced OLS diagnostics
├── options/
│   ├── __init__.py
│   ├── define.py              # Low-level data-loading functions (scopes_read, trans_dict_create, budget_ssn_create, media_ssn_apply, etc.) + deprecated OptionSettings
│   ├── scope.py               # OptionScope class — named application scope
│   ├── option.py              # Option class — named budget scenario
│   ├── translators.py         # Standard budget->media translators (basic, basic_fix, etc.)
│   ├── calc.py                # Deprecated option_report and option_report_multi_var wrappers
│   ├── calc_multi.py          # Deprecated multi-option grid (option_report_multi_post, curves), grid construction functions (opt_grid, opt_grid_expand)
│   ├── parallel.py            # Parallel multi-option dispatcher (compute_multi, compute_grid) using MetaModel API
│   ├── objective.py           # Objective functions and constraint wrappers for optimization
│   ├── context.py             # ComputeContext + load_context (context.xlsx reader) + campaigns_to_options: derive scopes/allocations/seasonality from a campaign table (option | item | date_start | date_end | val) via date_range_to_periods; per-option scopes; optional context.xlsx write
│   └── optim.py               # Budget optimization (COBYLA, Nevergrad CMA-ES)
├── export/
│   ├── __init__.py
│   ├── basics.py              # PPTX chart styling (colours, fonts, gridlines) + resolve_scope_for_node / resolve_scopes_for_node (all non-zero scopes of a node's compute result)
│   ├── export_pptx.py         # Full PPTX presentation generation + per-component functions
│   ├── export_results.py      # XLSX export
│   ├── export_summary.py      # Per-(brand, entity) XLSX export: summary_data / export_summaries (one XLSX per entity: model setup/params/RHS/LHS/trans_path, brand stats, fit, dynamic decomposition, short/long curves, adstocks). Entities are derived from the brand's plan scopes' non-superbrand filters (p04 → one file per region, p08 → 4 region files + national brand.xlsx); pure-brand models fall back to the clone path (target-brand scope cloned, superbrand swapped, seasonality inherited). Partly-regional DAGs include national node curves via compute_grid(background_option=ctx.options['predefined']); sheets are entity-filtered per node (columns absent from a node's ds are skipped). Deep-copies the MetaModel before each compute_grid so the caller's mm is never mutated
│   └── slides/                # Individual slide generators
│       ├── __init__.py
│       ├── fit_predict.py
│       ├── decomposition.py
│       ├── waterfall.py
│       ├── incr_roi.py
│       ├── options.py
│       └── adstocks.py
├── pipeline/
│   ├── __init__.py           # exports generate_pipeline
│   └── generator.py          # generate_pipeline(): reads translation_* sheets, auto-generates scopes/allocations/media_prices/seasonality + per-node reporting scripts (incl. upstream predecessors) + adhoc stubs. Saves new XLSX (strips _template/_templ from name) to {project_root}/model_data/options/. Scripts are produced by substituting {{ TOKEN }} placeholders in pipeline/templates/reporting_template.py
└── res/
    ├── dict/
    │   └── vis_dict.xlsx      # Default visual dictionary (display names, language strings)
    └── (other resources)
```

**Dependency chain within fermatrica_rep**: `basics` -> `meta_model` -> `options` -> `export`

Analysis functions (`analysis/`) are **independent** of the options pipeline but accept `Model`, `ModelNode`, `MetaModel`, or the deprecated `list` in their first (`model`) or `model_rep` parameters. Passing a `ModelNode` or `MetaModel` auto-resolves `ds`, `model_rep`, and `dt_pred` from the node(s).

The options module further splits: `define` -> `translators` -> `calc` + `calc_multi` + `parallel` -> `objective` -> `optim`

The `pipeline` module is a code generator — it reads the `translation_*` sheets pattern and produces the remaining options infrastructure (XLSX sheets, reporting scripts, adhoc stubs). It depends on `fermatrica` (for `fermatrica_error`, `model_load_ext` via `dt_p.pkl.zstd` inspection) and produces output consumed by the rest of the framework at project setup time. Not part of the runtime chain.

`generate_pipeline()` saves generated XLSX as a new file (strips `_template`/`_templ` suffix from input name) to `{project_root}/model_data/options/`. Per-node reporting scripts automatically include all upstream DAG predecessors so that scripts closer to the sink node are fully self-contained.

## Local Configuration

Machine-specific settings are stored in `config.toml` (repo root, gitignored, created by each developer):

1. Copy `config_example.toml` to `config.toml`
2. Optionally set `[paths].samples` to a directory containing sample model projects used by integration tests
3. Set `[paths].python` to the Python executable path in your Conda environment
4. Set `[python].env_name` to your Conda environment name

The `[paths].python` field points to the Python interpreter the AI should invoke. When running Python commands (`python ...`, `pytest ...`), always prefix with this path:

```
C:\Users\...\anaconda3\envs\<env_name>\python.exe -m pytest tests/
```

The `[python].env_name` field is informational — the environment name corresponding to the interpreter.

Integration tests resolve the samples path from (highest to lowest):
- `--samples-path` pytest CLI flag
- `config.toml` `[paths].samples`

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
- `media_prices` — per-scope inflation multipliers, optional, no constructor param. Wire it after construction with a plain assignment (`model_rep.media_prices = media_prices` — the attribute is initialized to `None` in `__init__`, so `StableClass` allows reassignment). Persists automatically via the `model_rep.pkl.lzma` pickle on `ModelNode.save` / `MetaModel.save`.

Like `Model`, `ModelRep` extends `StableClass` — no new attributes after `__init__`.

### OptionScope

`OptionScope` defines **where** and **when** to apply a budget scenario. Named scopes are atomic (no inheritance), defined in `options/scope.py`.

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

Key fields:
- `scope_name` — unique name referenced by budget rows
- `date_start` / `date_end` — the forecast period
- `ref_date_start` / `ref_date_end` — the reference period (for growth calculation). `ref_date_end` defaults to `date_start - 1 day` if omitted (scopes sheet) or required (Python constructor)
- `filters` — dict `{column: [values]}` defining which dataset rows to target (e.g., `{"superbrand": ["BrandX"], "region": ["Moscow"]}`)
- `plan_period` — `"exact"`, `"year"`, `"hy"`, `"quarter"` — controls summary calculation
- `inflation_multiplier` — default inflation for all tools in this scope
- `seasonality` — optional reference to a key in the seasonality dict. Falls back to `scope_name` if `None`.

Like `Model`, `OptionScope` extends `StableClass` — no new attributes after `__init__`.

#### OptionScope manipulation API

`OptionScope` provides helpers that replace the common by-hand comprehension idioms in reporting scripts:

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

### Campaign-Based Options (campaigns_to_options)

`campaigns_to_options()` (in `options/context.py`) replaces the manual assembly
of `context.xlsx` sheets (`scopes`, `allocations`, `seasonality`) with a single
call that derives everything from a simple campaign table:
`option | item | date_start | date_end | val` (optional `target`).

- **Per-option scopes are the core design**: each option's campaigns form its
  own scope. Scope name = `{option}_{target}_{year}` (or `{option}_{year}`
  without a `target` column); scope dates are full calendar years
  (`date_start={y}-01-01`, `date_end={y}-12-31`, reference = previous year),
  `plan_period` / `inflation_multiplier` configurable, `seasonality=scope_name`.
- **allocations**: campaign budgets summed per (option, scope, item),
  split across years proportionally to calendar days. Rows with `val == 0`
  are skipped (absent rows mean no budget).
- **seasonality**: per-period budget shares per (scope, item) computed via
  `fermatrica_utils.date_range_to_periods` (day-weighted). All periods of the
  scope year are emitted (`0.0` where no budget) so `media_ssn_apply` never
  produces NaN; coefficients sum to 1.0 per scope-year.
- Returns a `ComputeContext`, so `MetaModel(compute_context=campaigns_to_options(...))`
  works directly. Optional `path=` writes a ready-to-load `context.xlsx`;
  optional `template_path=` (existing context with `translation_*` sheets) is
  copied first and only the campaign-derived sheets are overwritten.

```python
ctx = fmr.campaigns_to_options(campaigns, granularity='month',
                               template_path='context_template.xlsx',
                               path='context.xlsx')
mm = fmr.MetaModel(nodes=..., edges=..., compute_context=ctx)
```

`resolve_scopes_for_node(node_result, scopes)` (`export/basics.py`) returns
**all** scopes present in a node's compute result with non-zero prediction
(`resolve_scope_for_node` stays as a thin first-only wrapper). `options_detail`
uses the plural version, so per-option scopes produce one combined slide set
per option covering the union of its scopes. `options_table` detects per-scope
zero binding: scopes without a zero baseline are still reported with the
comparison cells filled with `'-'`, and scopes bound only to the zero option
are reported as well (zero baseline KPIs, unbound options as `'-'`) — p09
exports `options_table` with strict no-comparison.

### ModelNode

`ModelNode` bundles one model level: `(Model, ds, ModelRep)`. Lives in `meta_model/node.py`. Provides `compute()` — the central entry point for translating budgets, running transform + predict, and summarizing results.

```python
# Full construction with explicit ModelRep
node = fmr.ModelNode(model=model, ds=ds, model_rep=model_rep)

# Auto-create ModelRep from kwargs (language, trans_dict, etc.)
node = fmr.ModelNode(model=model, ds=ds, language='english', trans_dict=trans)

# From model path (auto-loads model + ds via model_load_ext)
node = fmr.ModelNode(model_path="path/to/model_dir/", language='english')

# Load from disk with ModelRep
node = fmr.ModelNode.load("path/to/model_dir/")

# Mutate dt_pred post-init (pre-declared in class body for StableClass)
node.dt_pred = pred

results = node.compute(option, scopes=[scope_1, scope_2])
```

Returns ``{scope_name: summary}`` by default, or ``{scope_name: (ds, dt_pred, summary)}`` when ``return_full=True``. Scopes are computed **cumulatively** — each scope's translation modifies the same dataset, so downstream merge functions get the final prediction. Scopes are processed sorted by ``OptionScope.date_start`` (not by ``Option.iter_scopes()`` order): each scope's budget is projected forward beyond its ``date_end`` by ``_translate_long``, so processing chronologically ensures the latest (target) scope's translation and long-term projection win over earlier-ending scopes.

`compute()` also stores `dt_pred` on the node (`self.dt_pred`), making it available for later analysis function calls without re-passing prediction data.

##### Static nodes (no translation)

A **static** node (`ModelNode(..., static=True)`) is a node whose model has no option-driven media RHS. Its `translate()` is a no-op and no `trans_dict` is required — `MetaModel(compute_context=...)` skips the translation-sheet requirement for it (with >1 sheets, a static node needs no matching `translation_<node>` sheet; `MetaModelDiagnostics` reports its missing `trans_dict` as info, not error). The node still **computes** per option: it may consume dynamic upstream predictions via DAG merges, so its output is not required to be immutable. `static` persists in `meta_conf.toml` under `[node_settings.<name>]` (like `scope_names`) and round-trips through `MetaModel.save`/`load` and `ModelNode.load`.

Every `ModelNode` must carry a model: construction with `model=None` and no `model_path` raises `FermatricaError`. The only exception is the internal `_display_only` bypass used by `_convert_to_metamodel` for visual-only nodes (a `ModelRep` carrying no model, constructed for plot dispatch and never computed).

##### No-arg compute (modelling-stage refresh)

`node.compute()` with no `option` runs a single refresh: `transform(set_start=False)` + `predict_ext`, stores the transformed dataset on `node.ds` and the prediction on `node.dt_pred`, and returns the `dt_pred` DataFrame. `scopes` / `return_full` are ignored on this path. The initial bootstrap `transform(set_start=True)` remains a core modelling step in `fermatrica`.

The model must already be **fitted** (`model.obj.models['main']` exists) before the refresh — `predict_ext` predicts with the fitted OLS/LME object and does not fit. `ModelNode._refresh()` raises `FermatricaError` ("Model is not fitted — run `fm.fit_predict(ds, model, if_fit=True)` before `node.compute()`") when `model.obj.models` is missing, empty, or has no `'main'` key. Models loaded from a saved directory (via `model_path=`) are fitted; a model freshly built from `model_conf.xlsx` is not, so fit it first with `fm.fit_predict(ds, model, if_fit=True)` (as the p06 notebooks do). When `model_path` is given, an explicit `ds` overrides the saved (post-transform) dataset — pass raw data so the no-arg refresh transforms from raw.

##### Saving

```python
node.save(path=..., save_tag=model_tag + '_dash')
```

`ModelNode.save()` is a thin wrapper over `model.save` that passes `node.ds` and `node.model_rep`, so both `dt_p.pkl.zstd` and `model_rep.pkl.lzma` are always written and `ModelNode.load` / `MetaModel.load` restore the reporting config. Mirrors `model.save` defaults (`path=''` → `code_py/model_data`, `save_tag=None` → timestamp).

`ModelNode.load()` restores the model + dataset + ModelRep. `model_rep.pkl.lzma` is **optional** at load: when absent (modelling-stage saves that predate reporting), a default `ModelRep` (English, palettes derived from `ds`) is auto-created and a warning is logged.

##### MetaModel save/load round-trip

`MetaModel.save()` writes the whole DAG as one self-contained, transferable folder that `MetaModel.load()` can restore without any re-setup:

```python
# Reporting stage: after building the DAG with reporting settings
mm = fmr.MetaModel(nodes={"aba": node_aba, "sales": node_sales},
                   edges=[("aba", "sales", pred_merge_1)])

# Persist everything (model, ds, ModelRep incl. trans_dict/ssn_dict/media_prices/adhoc)
meta_dir = mm.save(path="code_py/model_data", save_tag="dash",
                   include_context_path="code_py/model_data/options/context.xlsx")

# Any other script / reporting application:
mm2 = fmr.MetaModel.load(meta_dir)
opt_sum = mm2.compute(options["predefined"], scopes=scopes)   # dt_pred rebuilt on demand
```

Canonical folder layout (Option A — written by `MetaModel.save`):

```
meta_<save_tag>/
├── meta_conf.toml            # single config: [[graph]] edges (left, right, merge_fn)
│                             #   + [node_settings.<name>] scope_names / static (when set)
├── context.xlsx              # optional copy (include_context_path), no sheet rewriting
├── model_aba/                # ModelNode.save output: model_conf.xlsx,
│   ├── model_rep.pkl.lzma    #   model_objects.pkl.lzma, model_rep.pkl.lzma,
│   └── dt_p.pkl.zstd         #   dt_p.pkl.zstd
└── model_sales/
```

- **Single-node** MetaModels delegate to `ModelNode.save()` — a plain `model_<tag>/` dir carrying the same `meta_conf.toml` (graph empty, `[node_settings]` when set); `MetaModel.load` auto-detects it (node name `"model"`). The config and the copied options file land **inside** the node dir, so the single-node round-trip restores `scope_names` too.
- **`dt_pred` is not persisted** — run the no-arg `mm.compute()` / `node.compute()` refresh after load to rebuild predictions.
- **`scope_names`** (per-node) persist in `meta_conf.toml` under `[node_settings.<name>]` and are restored by `_load_multi()` (multi-node, meta-dir root) and `ModelNode.load()` (single-node, node dir). An explicitly empty set round-trips as an empty set (not `None`).
- **`include_context_path`** copies the context XLSX into the saved dir as `context.xlsx` (copy-only — options/scopes stay in the XLSX as the single source of truth).
- Merge functions are serialized to `merge_fn` as qualified names (`module.qualname`) — a callable that cannot be re-imported (lambda, closure, `__main__`) raises `FermatricaError` at save. Adhoc merge functions remain resolvable at compute time because `adhoc_code_src` is pickled into `model_rep.pkl.lzma` and re-imported by `_load_adhoc_from_node`.
- **Adhoc sources are stale-prone snapshots**: `adhoc_code_src` captures `inspect.getsource(...)` at `ModelRep` construction time, and loaded nodes re-import from that pickled snapshot — not from the live `code_py/adhoc/` files. Nodes of one DAG may have been saved at different development stages, so the same adhoc module can be pickled with different source per node; at compute time the first-imported snapshot wins process-wide (imports are idempotent on the module name), so stale or mixed adhoc code can run silently. After editing adhoc code, check that every node's `model_rep.adhoc_code_src` matches the current sources before relying on a saved MetaModel.
- **Guard rails**: `save_tag` and node names must be plain directory-safe names (path separators, `..`, and Windows-invalid characters raise `FermatricaError`); a `meta_<save_tag>/` dir that already exists raises instead of being silently overwritten; a failed multi-node save removes the partially written `meta_<tag>/` dir before re-raising.

##### Canonical modelling-stage flow

```python
# model, ds from modelling; model_rep is optional (auto default rep)
node = fmr.ModelNode(model=model, ds=dt_p)
node.compute()                      # transform + predict + store

# node-first analysis calls (model_rep resolved from the node)
metrics = fmr.metrics_group_table(node)   # or any stats function
fig = fmr.fit_mult_plot_vol(node)
split = fmr.extract_effect(node)
fig = fmr.decompose_sub_plot(split, brands=None, model_rep=node)

node.save(path=..., save_tag=model_tag)   # persists model + ds + model_rep
```

`summary_type` on `ModelConf` (`"sum"` default, `"mean"`, `"fin"`, `"mean_fin"`) drives how KPI aggregates are computed. `ModelNode._summarize()` aggregates the exact, long, and reference periods with the same method: `mean`/`mean_fin` use the mean, `fin` collapses to the final period, everything else sums. Reference aggregates (`ref_vol`, `ref_val`, `pred_ref_*`) follow the same method, so `growth_vol`/`growth_val` compare like with like. When a reference period straddles the observed/forecast boundary, the observed and predicted parts are pooled into one series and aggregated once (never the sum of two partial means), so a reference period that lies entirely in the forecast horizon yields a real number rather than `NaN`. The PPTX slides respect `summary_type` too: fit-predict yearly tables and option-comparison tables aggregate per year with the same method, render mean values < 500 with 2 decimals, and the options growth row compares against a like-aggregated reference period. Efficiency-curve axis titles follow the same rule: the long-term axis is labelled `Sum.` only for sum-type aggregation, while `mean`/`mean_fin` use the `Avg.` wording (`export/slides/incr_roi.py`), since mean-based KPIs report an average per period rather than a cumulative total.

#### Passing ModelNode to analysis functions

All public analysis functions accept `ModelNode` where they previously accepted `Model` or `ModelRep`:

```python
node = fmr.ModelNode(model=model, ds=ds, language='english')

# ds, dt_pred, model_rep auto-resolved from node
split_m_m = fmr.extract_effect(node)
fig = fmr.fit_main_plot_vol(node)
fig = fmr.waterfall_plot(split_m_m, brands=['brand_1'], model_rep=node)
fig = fmr.decompose_main_plot(split_m_m, brands=None, model_rep=node)
fig = fmr.curves_full_plot_short(curves_df, model_rep=node)

# dt_pred from node used automatically; explicit override still possible
fit_data = fmr.fit_main_data(node)               # uses node.dt_pred
fit_data = fmr.fit_main_data(node, dt_pred=pred) # overrides node.dt_pred
```

`ds`, `dt_pred`, and `model_rep` are extracted from the node when not explicitly provided. Explicit overrides take precedence over node values.

The stats family (`metrics_group_table`, `metrics_table`, `basic_metrics_table`, `predictors_table`, `vif_table`, `tests_table_ols`) also accepts `ModelNode` (dispatches on `node.dt_pred` / `node.model`) and `MetaModel` (returns `{node_name: table}`).

### MetaModel (New structure)

`MetaModel` is a DAG of `ModelNode`s. The ``nodes`` argument accepts either
a ``dict[str, ModelNode]`` or a single ``ModelNode`` (auto-named ``"model"``).

```python
# From pre-built nodes (dict):
mm = fmr.MetaModel(
    nodes={"bht": node_bht, "sales": node_sales},
    edges=[("bht", "sales", pred_merge_bht)],
)

# Single node (auto-named "model"):
mm = fmr.MetaModel(nodes=node_sales)

# Single node with custom name (dict form):
mm = fmr.MetaModel(nodes={"sales": node_sales})

# With compute_context: wires per-node trans_dict/ssn_dict/media_prices
# at construction. Scopes are NOT stored on the MetaModel — pass them
# to compute() explicitly (ctx.scopes) so different scopes can be applied
# to the same model.
ctx = fmr.load_context("context.xlsx")
mm = fmr.MetaModel(nodes={"uba": node_uba, "sales": node_sales},
                   edges=[("uba", "sales", pred_merge_1)],
                   compute_context=ctx)
```

#### compute_context wiring

`load_context(path)` (`fermatrica_rep.options.context`) returns a frozen `ComputeContext` — `path`, `scopes` (`list[OptionScope]`), `options` (`dict[str, Option]`), `ssn_dict`, `media_prices`, and `trans_dicts` (keyed by node name from `translation_<node>` sheets; a plain `translation` sheet is keyed under `""`). It replaces the old 4–5 loader calls (`scopes_read` + `Option.from_excel_all` + `seasonality_read` + `media_prices_read` + per-node `trans_dict_create`) and raises `FermatricaError` if the `scopes` or `allocations` sheet is missing.

`MetaModel(compute_context=...)` accepts a `ComputeContext` or a path string, **consumes** it at construction, and wires each node's `ModelRep` — explicit values already on a node always win:

- `trans_dict`: assigned only when `node.model_rep.trans_dict is None`, from `ctx.trans_dicts.get(node_name)`. When exactly **one** translation dict exists it is given to every node (covers the p05 shared `translation` sheet and single-node `"model"` samples p00–p04). When **more than one** sheet exists and a node has no matching `translation_<node>` sheet, a `FermatricaError` is raised at construction (never a silent no-translation). When **no** translation sheet exists at all, nothing is wired at construction (the node may legitimately run options-free) — but `ModelNode.translate()` / the option path of `compute()` raises `FermatricaError` at computation time if the option then carries budgets for the node's scopes, since a `trans_dict` may be attached to the model at any stage before compute.
- `ssn_dict` / `media_prices`: assigned only when the node's value is `None`.

Scopes are **not** part of the model: `MetaModel` stores no scope state (only per-node `scope_names` when explicitly set). Pass `scopes` to every computation entry point (`mm.compute(option, scopes=ctx.scopes)`, optimizers, `compute_grid`, `MetaModelDiagnostics.run`). The option path of `mm.compute()` raises `FermatricaError` when `scopes` is `None`; the no-arg refresh path ignores scopes.

#### Edge validation

When a `MetaModel` is constructed with `nodes` and `edges`, the following invariants are checked via `_validate()` and raise `FermatricaError`:

| Condition | Error |
|-----------|-------|
| Zero nodes | `"must contain at least one node"` |
| More than one node but no edges | `"no edges provided"` |
| Edge references a node name not in `nodes` | `"is not in MetaModel nodes"` |
| Self-loop (A -> A) | `"Self-loop detected"` |
| Duplicate edge (same left, right pair) | `"Duplicate edge"` |
| Node exists but is not mentioned in any edge | `"Nodes not referenced in any edge"` |
| Disconnected DAG (A->B and C->D with no path between them) | `"not a single connected DAG"` |

Single-node MetaModels (1 node, 0 edges) skip all edge checks — edges are only required when more than one node is present.

`MetaModel.load()` auto-detects the directory type:
- `meta_conf.toml` + subdirs → multi-model DAG via `_load_multi`
- `meta_conf.toml`, no subdirs → single model (node name `"model"`; the TOML is read by `ModelNode.load` to restore `scope_names`)
- No meta_conf, no subdirs → single model, node name `"model"`
- No meta_conf, has subdirs → error
- Legacy `meta_conf.xlsx`/`meta_model_conf.xlsx` is still read with a warning — re-save to upgrade to TOML

`_load_multi()` reads `meta_conf.toml` and loads each model + `model_rep.pkl.lzma` from subdirectories. The TOML format:

```toml
[[graph]]
left = "awareness"
right = "bht"

[[graph]]
left = "bht"
right = "sales"
merge_fn = "code_py.adhoc.reporting.pred_merge_bht"

[node_settings.bht]
scope_names = ["brand_1_2026"]
```

- `merge_fn` optional — absent = parallel independent branches
- Node folders resolve as `<name>` first, then `model_<name>` (the layout written by `MetaModel.save`); a leading `model_` prefix on graph names is stripped so saved nodes load with clean keys (`uba`/`sales`, not `model_uba`/`model_sales`)
- `model_rep.pkl.lzma` is **optional** per node — when absent a default `ModelRep` is auto-created from the dataset with a warning
- Missing node folder → `FermatricaError` (never silently skipped)
- `[node_settings.<name>]` tables (written by `MetaModel.save`) restore per-node `scope_names` when present

#### Passing MetaModel to analysis functions

When a `MetaModel` is passed to an analysis function, the result is a `dict` keyed by node name:

```python
mm = fmr.MetaModel.load("path/to/model_dir/")

# Returns dict[str, DataFrame] — one decomposition per node
split_dict = fmr.extract_effect(mm)

# Returns dict[str, Figure] — one plot per node
figs = fmr.decompose_main_plot(split_dict, brands=None, model_rep=mm)
figs = fmr.waterfall_plot(split_dict, brands=['brand_1'], model_rep=mm)

# curves_full_df must be dict[str, DataFrame] keyed by node name
figs = fmr.curves_full_plot_short(curves_dict, model_rep=mm)
```

For visual-only functions (`decompose_main_plot`, `waterfall_plot`, `curves_full_plot_short`, etc.), `MetaModel` is detected first and each node's `model_rep` is extracted. The `split_m_m` or `curves_full_df` argument must be a `dict` keyed by node name when `model_rep` is a `MetaModel`.

##### No-arg MetaModel.compute() (modelling-stage refresh)

`mm.compute()` with no `option` walks nodes in topological order, merges each predecessor's `dt_pred` via the edge-specific merge functions, and runs the no-arg `node.compute()` refresh on every node. Returns `{node_name: dt_pred}`. `allow_past` defaults to `True` on this path (the `None` sentinel resolves per path — the option path keeps `allow_past=False` semantics exactly); pass `mm.compute(allow_past=False)` to forward `False` to the merge functions. `scopes` / `return_full` are ignored.

#### Writing new analysis functions

New analysis functions that accept `model` or `model_rep` should import and use the converter function from `fermatrica_rep.meta_model.meta_model`:

```python
from fermatrica_rep.meta_model.meta_model import _convert_to_metamodel

def my_analysis_fn(model, ds=None, model_rep=None, dt_pred=None):
    mm = _convert_to_metamodel(model, ds=ds, model_rep=model_rep, dt_pred=dt_pred)
    results = {}
    for name, node in mm.nodes.items():
        results[name] = _worker(node.model, node.ds, node.model_rep, node.dt_pred)

    if isinstance(model, MetaModel):
        return results                          # dict keyed by node name
    if isinstance(model, list):
        return list(results.values())           # list (backward compat)
    return list(results.values())[0]            # scalar

def _worker(model, ds, model_rep, dt_pred):
    ...
```

`_convert_to_metamodel` converts any input type (`Model`, `ModelNode`, `MetaModel`, `ModelRep`, or deprecated `list`) into a `MetaModel`. The wrapper then iterates `mm.nodes.items()`, passes the extracted `(model, ds, model_rep, dt_pred)` to the worker, and dispatches the return type based on the original input type. When `model_rep` is `None`, `_convert_to_metamodel` auto-creates a default `ModelRep` (derived from `ds`, falling back to `dt_pred`, then an empty frame) so `model_rep` is optional in analysis calls.

When a `list` of models is passed alongside a `merge_fn_pattern` string (e.g., ``"code_py.adhoc.reporting.pred_merge_{i}"``), sequential edges are created between consecutive list elements. The converter also broadcasts scalar ``ds``, ``model_rep``, and ``dt_pred`` across list elements. Without the pattern (default ``None``), only single-element lists are accepted — multi-element lists raise ``FermatricaError`` because edges are required when more than one node is present.

For visual-only functions where `model_rep` is the primary dispatch parameter, pass `model_rep` as the first argument to `_convert_to_metamodel` (it accepts `ModelRep` directly):

```python
def my_plot_fn(split_data, model_rep):
    mm = _convert_to_metamodel(model_rep)
    results = {}
    for name, node in mm.nodes.items():
        results[name] = _worker(split_data[name], node.model_rep)
    ...
```

The old normalizers `_normalize_model_arg` and `_normalize_modelrep_arg` are deprecated — they still work but emit `DeprecationWarning`. New code should use `_convert_to_metamodel` instead.

### MetaModelDiagnostics (Validation on Demand)

`MetaModelDiagnostics` validates a `MetaModel` object at runtime (not at construction). Returns a structured `DiagnosticsReport`.

```python
diag = fmr.MetaModelDiagnostics(meta_model)
report = diag.run()
print(report.passed)   # True if all error-level checks pass
print(report.summary)  # Formatted multi-line report
```

Four checks run by default:

| # | Check | Description |
|---|-------|-------------|
| 1 | `node_type` | Every node in the DAG is a `ModelNode` instance |
| 2 | `node_attrs` | Critical attributes present: `model`, `ds`, `model_rep` (incl. `vis_dict`, `trans_dict`, `language`, `adhoc_code_src`), `dt_pred` |
| 3 | `edge_resolution` | Merge functions resolve (callable, string eval, or dotted name lookup) and accept `ds_cur`/`dt_pred_prev` kwargs |
| 4 | `merge_effects` | Run `compute()` with 2 different options, compare `ds`/`dt_pred` between runs. Skipped unless `options` and `scopes` provided (requires `len(options) >= 2`). Includes per-edge propagation attribution: when a predecessor's `dt_pred` differs between the two options, each merge is invoked directly with both predictions and the outputs are compared. A merge that writes nothing despite an upstream signal is an error (`merge_edge`); a merge that writes only constant columns is a warning. This catches broken edges masked by other working edges in fan-in DAGs. |

Options for check #4:

```python
report = diag.run(options=[opt_a, opt_b], scopes=[scope_1])
```

`run(options=...)` accepts a `list` or a `dict` (e.g. `ctx.options` — dict values are used in insertion order). Pass `scopes` explicitly (`diag.run(options=ctx.options, scopes=ctx.scopes)`) — scopes are a computation argument, not stored on the MetaModel.

Error policy: `raise_on_first=True` raises `FermatricaError` on the first **error-level** failure. Default `False` collects all results.

### Deprecated

`OptionSettings` is deprecated. Use `OptionScope` + `Option` + `ModelNode.compute()` instead. `MetaModelConf` and `MetaModelObj` have been removed — use `MetaModel` with `nodes` + `edges` directly, or `load()`.

Passing `list` of `Model` or `ModelRep` objects to analysis functions is deprecated. Use `MetaModel` or `ModelNode` instead. The old list API still works but emits a `DeprecationWarning`.

`calc.option_report()` and `calc_multi.option_report_multi_post()` are deprecated. Use `ModelNode.compute()` / `MetaModel.compute()` or `parallel.compute_multi()` / `parallel.compute_grid()` instead.

The old `export_*` wrappers (`export_option_table()`, `export_option_detail()`, `export_curves()`, `export_adstocks()`) are deprecated. Use the new clean-named functions (`options_table()`, `options_detail()`, `curves()`, `adstocks()`) with `MetaModel`, `OptionScope`, `Option` instead. Old wrappers still work but emit `DeprecationWarning`.

Passing `Model`, `ModelRep`, `OptionSettings`, and raw dicts to `export_*` wrappers is deprecated — these wrappers auto-convert old types to new internally via `_convert_to_metamodel` and `_convert_old_options`.

### Translation (Budget → Media)

Translation bridges the gap between money (budgets) and media variables (OTS, impressions). Defined in a `translation` sheet in `context.xlsx`:

| Column | Example | Description |
|--------|---------|-------------|
| `item` | `bdg_tv` | Budget key (used in option dict) |
| `function` | `basic` | Translator function name (from `translators.py`) |
| `variable` | `brnd_tv_ots_clip` | Target media variable in dataset |
| `base_price` | `0.05` | Cost per OTS/impression (or formula) |
| `max_budget` | `100` | Maximum budget for curves calculation |

The `function` column references a translator from `translators.py` (or custom). **All highest-level translator functions must have the exact same signature** as `basic` — the translation dispatcher in `ModelNode` calls all translators with a fixed set of arguments. Do not add, remove, or rename parameters in top-level translators. The standard interface is:
```python
def translator_name(md, ds, model_rep, var_key, var_name, val, ds_mask, media_price=None):
    # Apply val to ds[var_name] where ds_mask is True
    return ds
```

Media price resolution: `ModelNode` computes `final_price = trans_dict['base_price'][item] * media_prices[scope][item]` (falls back to scope's `inflation_multiplier`, then `1.0`) and passes it to the translator via the `media_price` parameter. Translators no longer look up prices from `md.trans_path_df`.

**Backward compatibility for legacy models** whose prices live in `model.conf.trans_path_df["price"]` rather than in the XLSX translation sheet: the utility function `fermatrica_rep.options.define._enrich_trans_dict_from_model(model_conf, trans_dict)` reads `trans_path_df` and maps prices into `trans_dict["base_price"]` by matching `trans_path_df["variable"]` to `trans_dict["variable"][item]`. It is automatically called in all deprecated code paths (`option_report()`, `export_option_table()`, `export_option_detail()`, `export_curves()`, `export_adstocks()`). The deprecated `export_*` wrappers also call `fermatrica_rep.options.define._convert_old_options(opt_set, options_m)` to convert legacy `OptionSettings` and raw dicts into `list[OptionScope]` + `list[Option]`. New-API users should set `base_price` directly in their XLSX translation sheet.

Translation functions (defined in `fermatrica_rep.options.translators`):
- `basic` — 1-to-1: money / price = media volume (with seasonality correction; falls back to uniform if seasonality column missing)
- `basic_fix` — set fixed value on masked rows (no price conversion)
- `price` — apply price ratio (current price / base price)
- `tv_complex` — **worker function** with extra parameters (`clip_dur`, `tv_type`, `media_prefix`), NOT a highest-level translator
- `nat_tv_complex` — national TV variant (standard signature, wraps `tv_complex`)
- `reg_tv_complex` — regional TV variant (standard signature, wraps `tv_complex`)
- `nat_tv_sponsor_complex` — national TV sponsorship variant (standard signature, wraps `tv_complex`)
- `brnd_nat_tv_complex` — brand-level national TV variant (standard signature, wraps `tv_complex`)
Custom translators can be added via `adhoc_*` prefix in user code and passed to `ModelRep` via `adhoc_code` parameter. Custom translators must also follow the standard signature.

When a seasonality column (`opt_media_ssn_*`) exists in the dataset but contains NaN/None values over the ds_mask filter, `basic` and `tv_complex` emit a `fermatrica_warner.warning` with the `var_key` and the date range of the problematic rows.

### Seasonality

Seasonality coefficients distribute yearly budget across time periods (months, weeks). They are consumed by translators like `basic` which look up `opt_media_ssn_<item>` columns in the dataset and multiply the budget value by the coefficient.

#### Data Sources

Three ways to populate `ModelRep.ssn_dict`:

1. **`budget_ssn_create()`** (`options/define.py`) — reads period-level XLSX sheets, produces both `budget_dict` (yearly totals) and `seasonality_dict` (normalized distributions). Strips `bdg_`/`nobdg_` prefixes from column names. Keyed by option name.

2. **`media_ssn_create()`** (`options/define.py`) — loads sheets from XLSX as raw DataFrames. Keyed by sheet name. No prefix stripping.

3. **`seasonality_read()`** (`options/define.py`) — reads a `seasonality` sheet from `options.xlsx` with a `scope_name` column. Groups by scope, drops `scope_name` column. Keyed by scope name. Raw values (un-normalized).

The seasonality dict is stored on `ModelRep`:
```python
model_rep = fmr.ModelRep(..., ssn_dict=seasonality_dict)
```

#### Application Flow

1. **`ModelNode.translate()`** (called from `compute()`) iterates scopes. For each scope, it calls:
   
2. **`ModelNode._apply_media_ssn()`** (`meta_model/node.py`) — looks up the seasonality DataFrame from `model_rep.ssn_dict` using `scope.seasonality` (falls back to `scope.scope_name`). Calls:
   
3. **`media_ssn_apply()`** (`options/define.py`) — the core function that merges seasonality coefficients into the dataset. Steps:

   a. **Column rename** — all non-key columns get `opt_media_ssn_` prefix
   b. **`_seasonality_to_dates()`** — converts any period format (`month`, `week_number`, `day`, `date`, `period`) to a uniform `{date, period, opt_media_ssn_*}` format. If no `year` column, cycles the data across all years in the scope's date range
   c. **`_normalize_seasonality()`** — for budget columns only (`bdg_` items), divides each value by its calendar year's total, making coefficients sum to 1.0 per year. Non-budget columns untouched. Already-normalized data (from `budget_ssn_create`) is a no-op (division by 1.0)
   d. **`period_align()`** (from `fermatrica_utils`) — if seasonality granularity differs from dataset granularity, converts via even redistribution:
      - Upsampling (e.g., month → week): divides each coefficient by the count of sub-periods (e.g., monthly 0.08 → 4 weeks of 0.02 each)
      - Downsampling (e.g., week → month): sums coefficients
      - Budget columns use `"sum"` aggregation (divided on upsampling), non-budget use `"first"` (not divided)
   e. **Scope rescaling** — if `date_start`/`date_end` cover only part of a year (e.g., 6 months), budget columns are renormalized to sum to 1.0 over the scope period (a no-op for full-year scopes since sum already = 1.0)
   f. **Merge** — joins `dt_ssn` with the main dataset on `date`

4. **Translator consumption** — e.g., `basic` reads `opt_media_ssn_<item>` and computes:
   ```python
   media_var = ssn_coefficient * yearly_budget * 1e6 / media_price
   ```

#### Key Design Details

- **Granularity inference** for `period` column (from `seasonality_read`): max period value ≤ 12 → month, else week. Configure via the first column name (e.g., `month`, `week_number`).
- **`bdg_` vs `nobdg_`**: budget columns (`bdg_` prefix in `trans_dict`) get normalized to sum=1 per year and use `"sum"` aggregation during period alignment. Non-budget columns (`nobdg_` prefix) are not normalized and use `"first"` aggregation. The distinction is derived from `trans_dict` in `ModelNode._build_bdg_ssn_cols()`.
- **Missing `period` column** in the dataset: inferred heuristically from the seasonality first column name (`"month"` → `"month"`, `"week"` in name → `"week"`, etc.).

### Decomposition

Two types of decomposition are available:

**Dynamic decomposition** (`decomposition.py`): Values broken down into effects over time. Uses `extract_effect()` on the model output, then `decompose_main_plot()` visualizes. Supports both volume and value decomposition, pretrain display, and category models.

**Waterfall decomposition** (`waterfall.py`): Aggregated effects for a specific time period as a waterfall chart. Uses `waterfall_data()` to prepare and `waterfall_plot()` to visualize. Supports absolute and relative sorting, positive/negative separation.

Both receive prepared data as `split_m_m` — a DataFrame produced by `extract_effect()`.

Both `extract_effect` and `decompose_main_plot` accept `ModelNode | MetaModel | list` in their `model`/`model_rep` parameters. `waterfall_plot` and `waterfall_data` accept `ModelNode | MetaModel | list` in their `model_rep` parameter.

#### Whole-DAG decomposition (`extract_effect(..., whole_graph=...)`)

By default `extract_effect(mm)` decomposes each MetaModel node **independently**: a
predecessor's whole output appears as a single factor in the successor's split (p10
sales shows `02_Category` = `coef_sales * softmax(category_predicted)`). The opt-in
keyword-only `whole_graph: bool | str = False` expands each node's **additive
single-link** factors transitively through the DAG:

- `False` (default) — current behavior, unchanged.
- `True` — returns `{node_name: split_m_m}`, every node's frame expanded down to the
  leaf media/intercept factors. Nodes without predecessors are unchanged.
- `<node_name>` (str) — returns a single `split_m_m` for that node, expanded the same
  way. Unknown node name raises `FermatricaError`.

Attribution is a **proportional split**: the successor's link factor value per row is
split as `link_value · e_i / pred_P`, where `e_i` is the predecessor factor's effect
and `pred_P` is the predecessor's prediction (rowsum of its wide effects). Expanded
variables are named `<link display>.<factor display>` (e.g. `Category.TV`), and by
construction `Σ expanded = link_value` per row, so the "decomposition sums equal
prediction" invariant holds exactly. Multi-hop chains flatten via recursion;
operator-bearing (interaction) terms that reference a link stay single factors with a
`logging.warning`. The MetaModel must be computed first — `extract_effect(mm,
whole_graph=True)` before `mm.compute()` raises `FermatricaError` (every node's
`dt_pred` must carry `predicted`).

Whole-DAG usage is a pure composition: the expanded frames are plain `split_m_m`, so
`extract_effect(mm, whole_graph=True)` feeds `decompose_main_plot(split,
model_rep=mm)` and `waterfall_plot(...)` unchanged, and
`extract_effect(mm, whole_graph='sales')` feeds the per-node plot functions with a
single frame. Export threading: `options_detail(..., whole_graph=...)` (PPTX, both
option and retro branches, forwarded by the deprecated `export_option_detail` shim)
and `summary_data(...)` / `export_summaries(..., whole_graph=True)` (XLSX
`decomposition` / `decomposition_rub` sheets). All default to `False` — backward
compatible. The pipeline generator's `reporting_template.py:107` can opt in via
`extract_effect(mm, whole_graph=last_node)` without package changes.

### Filtering interface (analysis & export)

Analysis and export functions that previously took ad-hoc filter parameters
(`brands`, `bs_key_filter`, `target_superbrand`, `superbrand`, `scope_filters`)
now expose one systemic two-tier interface:

- `brands: list | None` — simple filter values (the existing `brands` param, optional where it was required)
- `brands_var: str = 'superbrand'` — column the simple filter values apply to (covers the old `bs_key_filter` case via `brands_var='bs_key'`)
- `filters: dict[str, list] | None` — complex filter `{column: [values]}` (the old `scope_filters` / `OptionScope.filters` shape)

Semantics:

- `brands` + `brands_var` and `filters` collapse into one dict and are applied as AND masks. Same key in both → intersection (simple filter wins on its own variable).
- `brands=None` → no simple filter (all brands). Plot titles keep using the effective `brands` list (shows the filtered values for a non-default `brands_var`).
- Missing filter columns are skipped with a `logging.warning` (no crash).
- `bs_key_filter`, `target_superbrand`, `superbrand` are removed from all new-API signatures; `model.conf.target_superbrand` (a model attribute) stays everywhere. The frozen legacy shims (`slides.*.create_legacy`, deprecated `export_*` wrappers) keep their old signatures unchanged and translate internally.

Example (all three spellings are equivalent for the default variable):

```python
waterfall_data(split_m_m, brands=['brand_1'])
waterfall_data(split_m_m, brands=['brand_1'], brands_var='superbrand')
waterfall_data(split_m_m, filters={'superbrand': ['brand_1']})

# old bs_key_filter behavior is reproduced via brands_var:
waterfall_data(split_m_m, brands=[1, 2], brands_var='bs_key')
# or the complex form:
waterfall_data(split_m_m, filters={'bs_key': [1, 2]})
```

Shared helpers live in `fermatrica_rep/basics.py`: `_merge_filters`, `_filter_df`, `_auto_brand_bs_keys` (the last centralizes the previously duplicated `dt_pred.loc[dt_pred['superbrand'] == model.conf.target_superbrand, 'bs_key']` derivation).

Function-specific notes:

- `extract_effect` applies the filters to the **output split** (post-computation); `decompose_main_plot` / `decompose_sub_plot` / `waterfall_plot` / `waterfall_data` filter before the date/listed mask.
- `adstocks_data`: when no filter param is given, defaults to `{'superbrand': [model.conf.target_superbrand]}` (adstock impulse must be computed on one brand).
- `options_table` / `options_detail` use **user-filters-replace-default** semantics: all three params None → auto `{'bs_key': target-brand bs_keys}` (current behavior); any user filter param → only the user's filters. The structural per-node brand mask (`brands=[node.model.conf.target_superbrand]`) at slide level stays — a user superbrand filter that disagrees may AND-empty (documented edge case).
- Waterfall slide: the scope-union derivation (`_union_scope_filters`) stays and is ANDed (same-key intersection) with the user filters.

### Efficiency Curves

**Short-term curves** (`curves.py`): Fast, RHS-only formula-based approach. Use when your model has no LHS effects or when doing WIP analysis. Functions: `curves_simple_plot()`, `curves_simple_data()`.

**Full curves** (`curves_full.py`): Model-blind approach that calculates a grid of budget options and collects predictions. Handles any model design (RHS + LHS). Slower, uses multiprocessing. Functions: `curves_full_plot_short()`, `curves_full_plot_long()`. Input data from `parallel.compute_grid()`.

**Per-region curves with national-model fixed costs**: when one DAG node's scope is a regional sub-scope (e.g. Cairo) and another node's scope is the national parent, the national node must receive the varying regional grid items **plus** the other regions' costs held fixed. Pass ``background_option`` (keyword-only, accepted by `parallel.compute_grid()`, `export_pptx.curves()`, and `slides.incr_roi.create()`) as:
- ``None`` (default) — every scope receives only the grid items (current behavior).
- ``dict[str, dict[str, float]]`` — raw additions ``{scope_name: {item: value}}`` merged (added) into that scope's budget items in every grid Option; every scope name must be among ``scopes``.
- ``Option`` — a base budget scenario (e.g. ``options['predefined']``); the background for each parent scope ``n`` is the cross-scope diff ``base[n] - base[leaf]`` where ``leaf`` is the grid's regional scope whose ``filters`` strictly contain ``n.filters`` (p08: ``brand_1_2026`` minus ``brand_1_cairo_2026`` = other regions' ``bdg_reg_*``; shared national items cancel to 0). Leaf scopes (with a ``region`` filter) get no background. Ambiguous leaf matching and non-numeric item values raise ``FermatricaError``; items where the leaf spends more than the parent are excluded with a warning.

The merged national scope therefore sees ``grid_items + other_regions``, so the national model varies with the region while holding other regions fixed. Nodes that need different budget compositions must use disjoint scope names. The output already returns one DataFrame per node, so the national node's curve is an extra key (e.g. ``curves_full_data['uba']``), non-breaking for existing callers.

### Budget Optimization

Four modes:
1. **Maximize KPI with known budget**: `optimize_budget_local_cobyla()` or `optimize_budget_local_ncma()`
2. **Minimize budget for target KPI**: `optimize_target_local_cobyla()` or `optimize_target_local_ncma()`
3. **Parallel multi-run** (COBYLA only): `optimize_budget_local_cobyla_multi()` runs multiple budget sizes in parallel via ``multiprocessing.Pool.starmap``; `optimize_target_local_cobyla_multi()` runs multiple KPI targets in parallel. Returns ``dict[float, Option]`` keyed by budget/target value. Handles pickling failure with sequential fallback. New-API only (``MetaModel`` / ``OptionScope`` / ``Option``).
4. **Multi-node optimization**: Optimize against a weighted combination of multiple DAG nodes. Pass ``summary_nodes`` to any optimizer as ``{node_name: (weight, scope_name)}``, e.g. ``{"sales": (0.7, "brand_1_2026"), "awareness": (0.3, "brand_1_2026")}`` (a list ``[weight, scope_name]`` works too). Combined score = ``sum(w_i * score_i / baseline_i)``. ``None`` = single-sink behavior (backward compatible).

Algorithms:
- **COBYLA** — derivative-free constrained optimization (NLopt). Good for noisy objectives.
- **Nevergrad CMA-ES** — evolutionary strategy for rugged landscapes. Better for complex constraints.

COBYLA supports epochs (restarts from different starting points), convergence tolerances, and verbose output.

The ``summary_nodes`` parameter is available on all 4 public functions, 2 multi functions, and threads through to ``ObjectiveFunctionBudget`` and ``ConstraintFunctionTarget``. When set, the baseline is computed once pre-loop by ``_compute_baseline()`` which deep-copies ``node.ds``, runs ``MetaModel.compute()``, extracts per-node scores, and restores original ``ds``. Passing empty ``{}``, a missing node name, or a zero/NaN baseline raises ``FermatricaError``. The objective helpers ``_get_score_key(if_long, if_volume)`` and ``_compute_score(results, ...)`` in ``objective.py`` handle the score extraction logic.

### PPTX Export

Two approaches:
1. **Component-wise** (recommended): Individual slide functions for flexible assembly, using `MetaModel` / `OptionScope` / `Option` API.

New clean API (primary): shorter names, new-API-only:

```python
# Option summary table
fmr.options_table(prs, meta_model=meta_model, scopes=scopes, options=options, ...)

# Detailed per-option slides (uses deep-copy + MetaModel.compute() per option)
fmr.options_detail(prs, meta_model=meta_model, scopes=scopes, options=options, ...)

# Hierarchical per-SKU reporting — report every prefix level of sku_var (coarse -> fine),
# skipping levels already covered by the brand-level group_var (e.g. ['superbrand'])
fmr.options_detail(prs, meta_model=meta_model, scopes=scopes, options=options,
                   sku_var=['superbrand', 'master', 'form', 'region'], if_sku_hierarchy=True)

# Retro analysis — options=None + listed_filter controls which periods to show
fmr.options_detail(prs, meta_model=meta_model, scopes=scopes, options=None, listed_filter=(2, 3))

# Efficiency curves (incremental KPI, profit, ROI)
fmr.curves(prs, meta_model=meta_model, scopes=scopes, translation=translation, ...)

# Adstock curves
fmr.adstocks(prs, meta_model=meta_model, cln_meas=cln_meas, ...)
```

Old `export_*` wrappers are deprecated thin shims that convert old types (Model, ModelRep, OptionSettings) to new (MetaModel, OptionScope, Option) and call the new functions. They emit `DeprecationWarning`:

```python
# Deprecated — use fmr.options_table() instead
fmr.export_option_table(prs, model=model, model_rep=model_rep, ...)

# Deprecated — use fmr.options_detail() instead
fmr.export_option_detail(prs, model=model, model_rep=model_rep, ...)

# Deprecated — use fmr.curves() instead
fmr.export_curves(prs, model=model, model_rep=model_rep, ...)

# Deprecated — use fmr.adstocks() instead
fmr.export_adstocks(prs, model=model, model_rep=model_rep, ...)
```

Slide generators in `export/slides/` also accept `ModelNode | MetaModel` for `model` and `model_rep` parameters:

```python
# fit_predict.py, decomposition.py, waterfall.py, adstocks.py, incr_roi.py
slides.fit_predict.create(prs, model=model_node, model_rep=model_node, ...)
slides.decomposition.create(prs, model=meta_model, model_rep=meta_model, ...)
slides.incr_roi.create(prs, model=meta_model, scopes=scopes, ...)
```

Standard slides include:
- Fit & predict by superbrand
- Fit & predict by SKU/category
- Dynamic decomposition (superbrand)
- Dynamic decomposition (SKU/category)
- Waterfall decomposition
- Short/long-term curves (incremental KPI volume/value, ROI)
- Option summary tables
- Detailed option reports

Requires MS Office on Windows for proper functionality.

## Reporting Pipeline (Canonical Workflow)

```python
import pandas as pd
import fermatrica as fm
import fermatrica_rep as fmr

# 1. Load data and trained model
ds = pd.read_excel("data.xlsx", parse_dates=["date"])
model, ds, errors = fm.model_load_ext("path/to/model_dir/")

# 2. Transform and predict (if not already done)
pred, pred_raw, model = fm.fit_predict(ds, model, if_fit=False, return_full=True)

# 3. Create ModelRep (reporting settings) — OR skip and let ModelNode auto-create it
model_rep = fmr.ModelRep(
    ds=ds,
    language="english",
    trans_dict=fmr.trans_dict_create(path="options.xlsx", sheet="translation"),
    tools_list=["tv", "digital", "pr"],
)

# 4. Standard reports (exploration)
# Pass ModelNode to auto-resolve ds/dt_pred/model_rep
node = fmr.ModelNode(model=model, ds=ds, model_rep=model_rep, dt_pred=pred)

metrics = fmr.predictors_table(node)
fit_data = fmr.fit_main_data(node)
fmr.fit_main_plot_vol(node)
fmr.decompose_main_plot(node)

# Or continue with old explicit API (Model/ds/model_rep separately)
split_m_m = fmr.extract_effect(model, ds, model_rep)
fmr.waterfall_plot(split_m_m, brands=["brand_1"], model_rep=model_rep)

# 6. Load scopes, options, seasonality, media prices and translation
#    in one call via ComputeContext (replaces 4–5 separate loader calls)
ctx = fmr.load_context("context.xlsx")
scopes = ctx.scopes
options = ctx.options
option = ctx.options["option_1"]   # or fmr.Option.from_excel("options.xlsx", name="option_1")

# 7. Build ModelNode and compute forecast
#    (trans_dict/ssn_dict/media_prices wired via MetaModel(compute_context=ctx))
node = fmr.ModelNode(model=model, ds=ds, model_rep=model_rep)
opt_sum = node.compute(option, scopes=scopes)
# opt_sum = {"scope_name": summary}

# 8. Multi-model: MetaModel(compute_context=ctx) + MetaModel.compute()
meta_model = fmr.MetaModel(nodes=node, compute_context=ctx)
opt_sum = meta_model.compute(option, scopes=ctx.scopes)
# opt_sum = {"node_name": {"scope_name": summary}}

# 9. Efficiency curves (full, model-blind)
options_grid = fmr.opt_grid(translation, budget_step=1)
results = fmr.compute_grid(meta_model, scopes=[scope], translation=translation)
fmr.curves_full_plot_short(results, meta_model)

# 10. Budget optimization
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

# 11. Export to PPTX (new API)
fmr.export_pptx.options_table(prs, meta_model=meta_model, scopes=scopes, options=options)

fmr.export_pptx.options_detail(prs, meta_model=meta_model, scopes=scopes,
                                      options=options, sku_var=['region'])

fmr.export_pptx.curves(prs, meta_model=meta_model, scopes=scopes,
                               translation=translation, budget_step=5)

fmr.export_pptx.adstocks(prs, meta_model=meta_model, cln_meas=['tv_grp'])
```

### Data conventions

- `listed` column: `2` = train, `3` = test, `4` = future (NaN target) — same as core
- `superbrand` = brand identifier string
- `bs_key` = numeric group identifier for panel models
- `date` = datetime column
- `pred` and `pred_raw` are the outputs of `fm.fit_predict()`
- `split_m_m` = prepared decomposition data from `extract_effect()`

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

### Sheet: `seasonality` — Seasonal coefficients (NEW, keyed by scope_name)

| Column | Type | Description |
|--------|------|-------------|
| `scope_name` | str | Scope name the seasonality applies to |
| `period` | int | Week/month number |
| `tv` | float | Seasonality coefficient for TV |
| ... | ... | Additional columns for each tool |

Note: column names are bare (no `bdg_`/`nobdg_` prefix). The `budget_ssn_create()` path strips prefixes automatically; `seasonality_read()` reads them as-is.

### Custom translators

Add `adhoc_*` functions to your code and reference them in the `function` column:
```python
def adhoc_translator_name(md, ds, model_rep, var_key, var_name, val, ds_mask, media_price=None):
    # Custom logic
    return ds
```

## Code Conventions

### Imports

```python
# Standard library
import copy, logging, os

# Third-party
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from line_profiler import profile

# fermatrica_utils
from fermatrica_utils import groupby_eff, StableClass

# fermatrica internal
from fermatrica import Model, fermatrica_error
from fermatrica_rep.meta_model.model_rep import ModelRep
```

Order: standard library → third-party → fermatrica_utils → fermatrica → fermatrica_rep

### Error handling

Use the framework's error function, never raw `raise`:
```python
fermatrica_error("Descriptive message about what went wrong")
```

Non-fatal issues: use the `fermatrica` logger at WARNING level.

### Type annotations

Use Python 3.10+ union syntax: `str | None`, `pd.DataFrame | None`. Not `Optional[str]`.

### Performance profiling

Functions are decorated with `@profile` from `line_profiler`. Leave these in place — they are no-ops when profiling is not active.

### Plotly conventions

- Default template: `pio.templates.default = 'ggplot2'` (set at module level)
- Functions return plotly Figure objects — the caller handles display (`.show()`) or export
- Colour palettes from `colorcet` (typically `'glasbey'`) via `palette_fill()`
- Plot configurations use `colorcet` for colour assignment, applied in `basics.palette_fill()`

### Multiprocessing

Options calculation and budget optimization use `concurrent.futures.ProcessPoolExecutor` for speed. Ensure the runtime environment supports multiprocessing (avoid on certain interactive/notebook contexts).

### ModelRep extends StableClass

No new attributes can be added after `__init__`. Set all attributes during construction.

## Testing Conventions

### Running tests

xdist is the enforced default: `addopts = "-n 4"` in `pyproject.toml`, so every
`pytest` run uses 4 workers automatically. Requires `pytest-xdist`
(`pip install -r requirements-dev.txt`). Override with an explicit `-n`
(e.g. `-n 0` for the debug mode, which needs a single process).

```bash
# All tests (already parallel via addopts "-n 4")
pytest tests/

# Options module tests
pytest tests/test_options/ -v

# Single test file
pytest tests/test_options/test_budget_ssn_create.py -v

# With debug mode (breaks at assertion failures) — single process required
pytest tests/ --if-debug -n 0

# Verbose with shortened traceback
pytest tests/test_options/ -v --tb=short
```

### Test patterns

- **YAML-driven parametrization**: Test cases in `tests/test_options/test_cases/*.yaml` with custom constructors (`!pd.DataFrame`, `!pd.Series`, `!np.array`). Use `parametrize_fun()` and `parametrize_class()` helpers from `tests/basics/basics.py`.
- **Expected error testing**: `exp_err` field — `pytest.raises(FermatricaError)` or `nullcontext()`
- **`if_debug` fixture**: When `True`, drops into `breakpoint()` on assertion failure. Set via `pytest --if-debug`.

### What to test when changing code

| Change location | What tests to run |
|----------------|-------------------|
| `fermatrica_rep/options/define.py` | `tests/test_options/test_budget_ssn_create.py` |
| `fermatrica_rep/options/translators.py` | `tests/test_options/` (if tests exist) |
| `fermatrica_rep/options/calc.py` | `tests/test_options/` (if tests exist) |
| `fermatrica_rep/options/optim.py` | `tests/test_options/` (if tests exist) |
| `fermatrica_rep/options/context.py` | `tests/test_options/test_campaigns.py`, `tests/test_compute_context.py` (depends on `fermatrica_utils.date_range_to_periods` — reinstall `fermatrica_utils` after changing it) |
| `fermatrica_rep/basics.py` | Manual inspection (no dedicated tests yet) |
| `fermatrica_rep/analysis/stats.py` | Manual inspection |
| `fermatrica_rep/meta_model/node.py` | `tests/test_refactored.py` (ModelNode + analysis function integration), `tests/test_node_internals.py` (static flag, translate no-op) |
| `fermatrica_rep/meta_model/diagnostics.py` | `tests/test_diagnostics.py` (incl. static-node expected-invariant downgrade) |
| `fermatrica_rep/meta_model/meta_model.py` | `tests/test_meta_model.py`, `tests/test_meta_model_save.py` (static persistence), `tests/test_integration/test_p10_pipeline.py` (static-node DAG, p10 sample) |
| `fermatrica_rep/analysis/` (any) | `tests/test_refactored.py::TestAnalysisFunctionModelNode` (ModelNode/MetaModel interfaces) |
| `fermatrica_rep/export/` | `tests/test_export.py`, `tests/test_export_slides.py` |
| `fermatrica_rep/export/export_summary.py` | `tests/test_integration/test_export_summary.py`, `tests/test_integration/test_export_summary_regional.py` |
| Any new module | Add test cases + YAML files in `tests/` |

## Common Mistakes to Avoid

1. **Forgetting to create ModelRep**: Most reporting functions require a ModelRep object. Create it before calling any plot/export functions.

2. **Not providing trans_dict for options**: Options and forecasts cannot be calculated without a translation dictionary. Create it via `trans_dict_create()` before `MetaModel.compute()` or `ModelNode.compute()`.

3. **Using `curves.py` with LHS models**: Short-term curves (`curves.py`) are RHS-only. For models with significant LHS effects, use `curves_full.py` which is model-blind.

4. **Modifying ModelRep after init**: `ModelRep` extends `StableClass` — no new attributes after `__init__`. Set all fields during construction.

5. **Running PPTX export on non-Windows**: PPTX export requires MS Office on Windows. Operation on Linux/macOS is not guaranteed.

6. **Not handling multiprocessing contexts**: `compute_grid` and `optim.*` use multiprocessing. Avoid running them from environments that don't support it (e.g., some Jupyter notebook configurations).

7. **Calling plot functions without checking data ranges**: Decomposition and fit functions depend on the `listed` column having the expected values (`2`, `3`, `4`). Missing future periods (`listed=4`) will cause empty plots.

8. **Mixing models and ModelReps between brands**: When working with multi-brand panel models, ensure `model_rep.palette_sbr` includes all brands in the dataset. Use `palette_fill()` before calling plot functions.

9. **Not resetting the model context for options**: Option calculation internally transforms the dataset. If running multiple option scenarios sequentially, ensure the dataset is re-loaded or re-transformed between runs.

10. **Mixing granularities in seasonality data**: When the seasonality DataFrame's granularity differs from the main dataset's `period` column, `media_ssn_apply()` uses `period_align()` to reweight coefficients (dividing by sub-period count on upsampling, summing on downsampling). Budget columns (`bdg_` in `trans_dict`) are identified via `ModelNode._build_bdg_ssn_cols()` and get `"sum"` aggregation; non-budget columns get `"first"`. If the bdg/nobdg mapping is missing, budget coefficients won't be properly reweighted.

11. **Confusing short-term and long-term curves**: Short-term curves exclude adstock decay effects (immediate response only). Long-term curves include the full decay tail. Choose based on the business question being answered.

12. **Using old export API**: The old PPTX export signatures (`export_option_table(model=..., model_rep=..., opt_set=..., options_m=...)`, etc.) are deprecated. Use `MetaModel`, `OptionScope`, and `Option` instead. Old calls still work but emit `DeprecationWarning`. The new API uses `meta_model`, `scopes`, and `options` as primary parameters. See the canonical workflow for examples.
