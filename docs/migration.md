# Migration Guide: Old → New Reporting Pipeline

This guide covers what changed from the old `OptionSettings`-based reporting pipeline to the new `OptionScope` + `Option` + `ModelNode` + `MetaModel` architecture. It assumes you are familiar with the old API.

## Summary of Changes

| Old | New |
|-----|-----|
| `OptionSettings(target, date_start, date_end, apply_vars, ...)` | `OptionScope(scope_name, date_start, date_end, ref_date_start, ref_date_end, filters={...}, ...)` |
| Budget as raw dicts or row-wise Excel (per option) | `Option.from_excel()` / `Option.from_excel_all()` — long `allocations` format with a `scope_name` column |
| `option_report(model, ds, model_rep, option_dict, option_settings)` | `node.compute(option, scopes=[...])` or `mm.compute(option, scopes=[...])` |
| `option_report_multi_var(...)` | `MetaModel.compute()` / `parallel.compute_multi()` |
| `option_report_multi_post(...)` | `parallel.compute_grid()` (returns one DataFrame per DAG node) |
| `MetaModel(path=..., dss=..., models=..., model_reps=...)` | `MetaModel(nodes={}, edges=[...])` or `MetaModel.load(path)` or `MetaModel(nodes=node)` |
| `MetaModelConf`, `MetaModelObj` | Removed — logic folded into `MetaModel._load_multi()` |
| Translator price from `md.trans_path_df` | Price resolved in `ModelNode`: `base_price * media_prices[scope][item]` (falls back to scope `inflation_multiplier`, then `1.0`) |
| `apply_vars` + `target` parallel lists | `filters: dict[str, list[str]]` |
| `long_term_type`, `if_ah` parameters | Removed |
| `ref_date_end` silently defaults to `date_end` | Defaults to `date_start - 1 day` when blank in the `scopes` sheet. **Required** in the Python `OptionScope` constructor |
| Seasonality keyed by option name via `text_flags` | Keyed by scope via `seasonality` column (fallback to `scope_name`) |
| `export_option_table()` / `export_option_detail()` / `export_curves()` / `export_adstocks()` | `options_table()` / `options_detail()` / `curves()` / `adstocks()` with `MetaModel` + `OptionScope` + `Option` |
| `optimize_budget_local_cobyla(ds=..., model=..., model_rep=..., opt_set=...)` | `optimize_budget_local_cobyla(meta_model=..., scopes=..., budget_scope=..., borders_dict=..., option=Option(...))` |

---

## New Objects

### OptionScope

Defines **where** and **when** to apply a budget scenario. Atomic — no inheritance.

```python
from fermatrica_rep import OptionScope

# Single filter (one brand)
scope = OptionScope(
    scope_name="brandx_2025",
    date_start="2025-01-01",
    date_end="2025-12-31",
    ref_date_start="2024-01-01",
    ref_date_end="2024-12-31",
    filters={"superbrand": ["BrandX"]},
    plan_period="year",
    inflation_multiplier=1.04,
    seasonality="brandx_default",  # optional
)

# Multi-filter (brand + region)
scope_msk = OptionScope(
    scope_name="brandx_msk_2025",
    date_start="2025-01-01",
    date_end="2025-12-31",
    ref_date_start="2024-01-01",
    ref_date_end="2024-12-31",
    filters={"superbrand": ["BrandX"], "region": ["Moscow"]},
    plan_period="year",
    inflation_multiplier=1.04,
)
```

Key facts:

- All five date/name fields (`scope_name`, `date_start`, `date_end`, `ref_date_start`, `ref_date_end`) are **required** in the Python constructor. Only in the Excel `scopes` sheet is `ref_date_end` optional — a blank value defaults to `date_start - 1 day`.
- `plan_period` defaults to `"year"` (`"exact"`, `"hy"`, `"quarter"` are the alternatives).
- `inflation_multiplier` defaults to `1.0`.
- `seasonality` defaults to `None` — the translator then looks up `scope_name` in `model_rep.ssn_dict`.

**Manipulation helpers** (replace the by-hand comprehension idioms):

- Per scope: `s.start_year` (property), `s.has_filter(col)`, `s.filter(col, value)` (explicit first-value match), `s.match(col, value)` (explicit value **or** no filter on `col` — national scopes), `s.copy(deep=True)`, `s.replace_dates(date_start=..., date_end=...)` (retro periods), `s.shift_years(delta=1)`.
- Collection level (`options/scope.py`, exported from the package root): `fmr.find_scope(scopes, name)` (first match or `None`) and `fmr.filter_scopes(scopes, **criteria)` — AND-composed, order-preserving, non-mutating filter. Criteria: `scope_name` (str or list = any-of), `start_year`, `start_year_min`, `start_year_max`, `has_filter`, `without_filter`, `match=(col, value)`.

```python
scopes_plan = fmr.filter_scopes(ctx.scopes, start_year_min=2026)
scopes_retro = [s.replace_dates(date_start='2025-01-01', date_end='2025-12-31')
                for s in fmr.filter_scopes(ctx.scopes, start_year=2026)]
scopes_regional = fmr.filter_scopes(ctx.scopes, has_filter='region')
scopes_crv = fmr.filter_scopes(ctx.scopes, start_year_min=2026, match=('region', rgn))
```

**Migration from `OptionSettings`:**

```python
# OLD
opt_set = fmr.OptionSettings(
    target=["BrandX"],
    date_start="2025-01-01",
    date_end="2025-12-31",
    ref_date_start="2024-01-01",
    ref_date_end="2024-12-31",
    apply_vars=("superbrand",),
    plan_period="year",
)

# NEW
scope = fmr.OptionScope(
    scope_name="brandx_2025",
    date_start="2025-01-01",
    date_end="2025-12-31",
    ref_date_start="2024-01-01",
    ref_date_end="2024-12-31",
    filters={"superbrand": ["BrandX"]},
    plan_period="year",
)
```

### Option

A named budget scenario. Loads from the `options.xlsx` `allocations` sheet or is constructed in Python.

```python
from fermatrica_rep import Option

# From Excel (single option by name)
option = fmr.Option.from_excel("options.xlsx", name="option_1")

# From Excel (all options at once) — the standard approach used by samples
options = fmr.Option.from_excel_all("options.xlsx")   # {"option_1": Option, "zero": Option, ...}
option = options["predefined"]

# From Python dict
option = fmr.Option(name="option_1", budget_data={
    "brandx_2025": {"bdg_nat_tv": 100, "bdg_digital": 30},
    "brandx_msk_2025": {"bdg_reg_tv": 20},
})
```

`from_excel_all()` reads the `allocations` sheet once and returns a `dict[str, Option]` keyed by option name. It requires the columns `option`, `scope_name`, `item`, `val`.

`Option` manipulation helpers:

- `opt.copy(deep=True)` — new independent option (preserves `default_scope` / `text_flags`); use `ctx.options['zero'].copy()` instead of rebuilding a zero option by hand.
- `opt.replace_budget(scope, items)` — new option with `scope`'s budget items replaced.
- `opt.budget_for_scope(scope_name)` — `opt.budget_data.get(scope_name, {})`.

**Migration from old budget dict:**

```python
# OLD
option = {"bdg_nat_tv": 100, "bdg_digital": 30, "text_flags": "name_option_1"}

# NEW
option = fmr.Option(name="option_1", budget_data={
    "brandx_2025": {"bdg_nat_tv": 100, "bdg_digital": 30},
})
```

Note: `text_flags` still exists as a legacy attribute on `Option` (it is parsed from an allocations `text_flags` row and preserved for backward compatibility), but it **no longer drives seasonality naming**. Seasonality is referenced via `OptionScope.seasonality`.

### ModelNode

Bundles one model level: `(Model, ds, ModelRep)`. The central compute entry point.

```python
from fermatrica_rep import ModelNode

# From objects
node = fmr.ModelNode(model=model, ds=ds, model_rep=model_rep)

# From model path (auto-loads model + ds via model_load_ext; ModelRep is
# auto-created from the remaining kwargs)
node = fmr.ModelNode(model_path="path/to/model_dir/",
                     language="english",
                     trans_dict=trans_dict,
                     adhoc_code=[code_py.adhoc.translators],
                     ssn_dict=ssn_dict,
                     missed_stop=True,
                     scope_names={"brandx_2025", "brandx_2026"})

# From disk (loads model_rep.pkl.lzma in addition to model + ds)
node = fmr.ModelNode.load("path/to/model_dir/")

# Compute
results = node.compute(
    option,
    scopes=scopes,
    allow_past=False,
    if_exact=False,
    return_full=True,
)
# Default (return_full=False): {"scope_name": summary}
# return_full=True:            {"scope_name": (ds, dt_pred, summary)}
```

Constructor notes:

- `model_path` loads the model and dataset via `model_load_ext`; `missed_stop` is forwarded to it. `model_path` and `model` cannot be used together.
- `scope_names` (a `set[str]`) restricts which scopes this node participates in during `translate()` / `compute()`. Useful in multi-node DAGs where nodes use disjoint scope sets (e.g. per-brand UBA node vs per-region sales node, as in p08).
- Any extra keyword argument (`language`, `trans_dict`, `adhoc_code`, `ssn_dict`, `palette_sbr_names`, ...) is forwarded to `ModelRep(ds, **kwargs)` when `model_rep` is not given explicitly.

`compute()`:

- With an `option`: translates budgets per scope (cumulatively, sorted by `date_start`), runs one transform + predict, and derives per-scope summaries. `allow_past=True` also rewrites historical rows (listed ≤ 3); `if_exact=True` skips the long-term (post-`date_end`) projection. `return_full=True` returns `(ds, dt_pred, summary)` tuples.
- No-arg `node.compute()` runs a refresh (`transform(set_start=False)` + `predict_ext`) and returns the `dt_pred` DataFrame. The model must already be **fitted**.
- `self.ds` and `self.dt_pred` are always updated, so later analysis calls (`fit_main_data`, `extract_effect`, ...) pick up the prediction automatically.

**Migration from `option_report()`:**

```python
# OLD
ds, dt_pred, summary = fmr.option_report(
    model, ds, model_rep,
    option_dict, option_settings,
    if_exact=False,
)

# NEW
node = fmr.ModelNode(model=model, ds=ds, model_rep=model_rep)
results = node.compute(option, scopes=scopes, return_full=True)
# access a specific scope result:
ds, dt_pred, summary = results["brandx_2025"]
```

### MetaModel

DAG of `ModelNode`s. Construction paths:

```python
from fermatrica_rep import MetaModel

# From pre-built nodes
mm = fmr.MetaModel(
    nodes={"bht": node_bht, "sales": node_sales},
    edges=[("bht", "sales", pred_merge_bht)],  # (left, right, merge_fn)
)

# Universal load (auto-detects single model vs DAG)
mm = fmr.MetaModel.load("path/to/model_dir/")

# Wrap a single ModelNode (auto-named "model")
mm = fmr.MetaModel(nodes=node)

# Or with a custom name (dict form):
mm = fmr.MetaModel(nodes={"sales": node})
```

Compute (with an `option`):

```python
results = mm.compute(option, scopes=scopes, allow_past=True)
# Default (return_full=False): {"node_name": {"scope_name": summary}}
# return_full=True:            {"node_name": {"scope_name": (ds, dt_pred, summary)}}
```

- Translation is two-phase: all nodes are translated in topological order first, then each node's predecessors' `dt_pred` is merged via the edge merge functions before its single transform + predict + summarize.
- No-arg `mm.compute()` walks nodes in topological order, merges predecessor `dt_pred`, runs the no-arg node refresh on each, and returns `{node_name: dt_pred}`. `allow_past` defaults to `True` on that path.

Save / load round-trip:

```python
meta_dir = mm.save(path="code_py/model_data",
                   save_tag="dash",
                   include_context_path="code_py/model_data/options/context.xlsx")

mm2 = fmr.MetaModel.load(meta_dir)
opt_sum = mm2.compute(options["predefined"], scopes=scopes)
```

- `save()` writes `meta_<save_tag>/` with a single `meta_conf.toml` (`[[graph]]` edges + `[node_settings.<name>]` scope_names), one `model_<node_name>/` subdirectory per node, and an optional copy of the context XLSX as `context.xlsx`. Single-node MetaModels delegate to `ModelNode.save()` — a plain `model_<save_tag>/` directory carrying the same `meta_conf.toml` (auto-detected by `MetaModel.load`).
- `dt_pred` is **not** persisted — run the no-arg refresh (`mm.compute()`) after load.
- `ssn_dict`, `media_prices`, `adhoc_code_src`, `trans_dict` persist automatically inside each node's `model_rep.pkl.lzma`.

**`MetaModelConf` and `MetaModelObj` are removed.** The graph is read from `meta_conf.toml` and converted to `edges` inside `_load_multi()`.

### ComputeContext / load_context

`fmr.load_context(path)` replaces the 4–5 separate loader calls (`scopes_read` + `Option.from_excel_all` + `seasonality_read` + `media_prices_read` + per-node `trans_dict_create`) with a single call that returns a `ComputeContext`:

```python
ctx = fmr.load_context("context.xlsx")

ctx.path              # str — the context XLSX path
ctx.scopes            # list[OptionScope]
ctx.options           # dict[str, Option]  (all options from the allocations sheet)
ctx.ssn_dict          # dict[str, pd.DataFrame]  ({} when the sheet is absent)
ctx.media_prices      # dict[str, dict[str, float]]  ({} when the sheet is absent)
ctx.trans_dicts       # dict[str, dict] — keyed by node name
```

- `trans_dicts` is discovered from the workbook: every `translation_<node>` sheet is built via `trans_dict_create` and keyed by the node-name suffix (`translation_uba` → key `"uba"`). A plain `translation` sheet (no suffix, p00–p05) is keyed under `""`.
- The `scopes` and `allocations` sheets are **mandatory** — `FermatricaError` is raised if either is missing. `seasonality` / `media_prices` are optional (empty dicts when absent).
- `ComputeContext` is a frozen dataclass; it is meant to be consumed, not mutated.

### MetaModel(compute_context=...)

Pass `compute_context=` (a `ComputeContext` or a path string) to `MetaModel` and the per-node wiring happens automatically at construction — you no longer pass `trans_dict=`, `ssn_dict=`, or `options_path=` to each `ModelNode`:

```python
ctx = fmr.load_context("context.xlsx")

node_uba = fmr.ModelNode(model_path=pth_uba, adhoc_code=adhoc, language="english")
node_sales = fmr.ModelNode(model_path=pth_sales, adhoc_code=adhoc, language="english")

mm = fmr.MetaModel(
    nodes={"uba": node_uba, "sales": node_sales},
    edges=[("uba", "sales", pred_merge_1)],
    compute_context=ctx,          # wires trans_dict / ssn_dict / media_prices per node
)
```

Wiring rules (explicit values already on a node always win):

- `trans_dict` — assigned only when `node.model_rep.trans_dict is None`. Looked up by node name from `ctx.trans_dicts`; when exactly **one** translation dict is available (e.g. the single shared `translation` sheet used by p00–p05), that dict is given to every node. When **more than one** sheet exists and a node has no matching `translation_<node>` sheet, a `FermatricaError` is raised at construction — never a silent no-translation. Name the node to match a sheet (e.g. `MetaModel(nodes={"uba": node}, compute_context=ctx)`) or pass `trans_dict` explicitly. When **no** translation sheet exists, nothing is wired at construction — `translate()` / the option path of `compute()` raises `FermatricaError` at computation time if the option then carries budgets for the node's scopes (a `trans_dict` may be attached to the model at any stage before compute).
- `ssn_dict` — assigned only when `node.model_rep.ssn_dict is None`.
- `media_prices` — assigned only when `node.model_rep.media_prices is None`.
- The context is **consumed** at construction — `MetaModel` stores no scope state (scopes are a computation argument, not model state).

Scopes are **not** part of the model: `mm.compute(option)` requires `scopes=` on the option path (raising `FermatricaError` when `None`), so the same model can be computed under different scope sets:

```python
opt_sum = mm.compute(ctx.options["predefined"], scopes=ctx.scopes, allow_past=True)
```

The no-arg refresh path (`mm.compute()`) ignores scopes, as before.

### ModelRep (new fields)

```python
model_rep = fmr.ModelRep(
    ds=ds,
    language="english",
    trans_dict=trans_dict,
    ssn_dict=seasonality,          # {scope_name: DataFrame}, optional
    adhoc_code=[...],              # optional custom translators / merge fns
)

model_rep.media_prices = media_prices   # {scope_name: {item: multiplier}}, optional
```

- `ssn_dict` is a constructor parameter; `media_prices` is **not** — assign it after construction (`ModelRep` extends `StableClass`, but both attributes are pre-declared in the class body, so reassignment is allowed).
- Both persist via the `model_rep.pkl.lzma` pickle on `ModelNode.save` / `MetaModel.save`.

---

## MetaModelDiagnostics (validation on demand)

```python
d = fmr.MetaModelDiagnostics(mm)
report = d.run(options=ctx.options, scopes=ctx.scopes, raise_on_first=False)

print(report.passed)   # True if no error-level failures
if report.warnings:
    print(report.warnings_summary)
```

`run()` accepts `options` as a `list` or a `dict` (e.g. `ctx.options` — dict values are used in insertion order). Pass `scopes` explicitly (`d.run(options=ctx.options, scopes=ctx.scopes)`) — scopes are a computation argument, not stored on the MetaModel.

Runs four checks:

1. `node_type` — every node is a `ModelNode`.
2. `node_attrs` — `model`, `ds`, `model_rep` (incl. `vis_dict`, `trans_dict`, `language`, `adhoc_code_src`), `dt_pred` present.
3. `edge_resolution` — merge functions resolve (callable, string, or dotted name) and accept `ds_cur` / `dt_pred_prev` kwargs.
4. `merge_effects` — runs `compute()` with 2 different options and compares `ds` / `dt_pred`. Skipped unless `options` and `scopes` are provided and `len(options) >= 2`.

`raise_on_first=True` raises `FermatricaError` on the first error-level failure.

---

## Excel Format Changes

### context.xlsx

The `context.xlsx` file used by the samples contains up to 5 sheets:

#### 1. `scopes` (NEW — required)

| scope_name   | date_start | date_end   | ref_date_start | ref_date_end | apply_vars         | target           | plan_period | inflation_multiplier | seasonality     |
|--------------|------------|------------|----------------|--------------|--------------------|------------------|-------------|----------------------|-----------------|
| brandx_2025  | 2025-01-01 | 2025-12-31 | 2024-01-01     | 2024-12-31   | superbrand         | BrandX           | year        | 1.04                 | brandx_default  |
| brandx_msk   | 2025-01-01 | 2025-12-31 | 2024-01-01     | 2024-12-31   | superbrand, region | BrandX, Moscow   | year        | 1.04                 |                 |

- The `ref_date_end` **column must exist** but may be blank — blank values default to `date_start - 1 day`.
- `seasonality` is optional — falls back to `scope_name`.
- `apply_vars` and `target` are comma-separated in Excel; zipped into a `filters` dict by `scopes_read()`.
- `plan_period` defaults to `"year"` when the column is absent or the cell is empty.
- `inflation_multiplier` defaults to `1.0` when the column is absent or the cell is empty. (Empty cells previously leaked `NaN` into `final_price` via `scope.inflation_multiplier` in `ModelNode._translate_budget` — fixed by NaN-guarding both fields in `scopes_read()`.)

Read with `fmr.scopes_read(path)` (sheet `"scopes"` by default).

#### 2. `allocations` (RENAMED — was `budget`, now long format with scope)

> **Sheet rename**: The sheet was renamed from `budget` to `allocations`. The default in `Option.from_excel()` changed accordingly. Pass `sheet="budget"` explicitly for backwards compatibility with old files.

| option   | scope_name    | item        | val |
|----------|---------------|-------------|-----|
| option_1 | brandx_2025   | bdg_nat_tv  | 100 |
| option_1 | brandx_msk    | bdg_reg_tv  | 20  |
| option_1 | brandx_spb    | bdg_reg_tv  | 25  |

- `scope_name` is optional per row — defaults to the option's `default_scope` (set in `Option.from_excel()`).
- Different targets (regions, SKUs) use different rows with different scopes — no override columns.
- An optional `text_flags` row (with a `val` string like `"name_option_1"`) is still parsed into `Option.text_flags` for legacy compatibility.

**Migration from old budget sheet:**

```python
# OLD format (columnar):
option   | bdg_nat_tv | bdg_reg_tv
option_1 | 100        | 45

# NEW format (long):
option   | scope_name   | item       | val
option_1 | brandx_2025  | bdg_nat_tv | 100
option_1 | brandx_msk   | bdg_reg_tv | 20
option_1 | brandx_spb   | bdg_reg_tv | 25
```

#### 3. `translation` (CHANGED — `price` → `base_price`)

| item       | function | variable        | base_price | max_budget |
|------------|----------|-----------------|------------|------------|
| bdg_nat_tv | basic    | nat_tv_ots      | 0.05       | 100        |
| bdg_digital| basic    | digital_imp     | 0.003      | 50         |

- Renamed `price` column to `base_price` (media cost per unit). `trans_dict_create()` still accepts a legacy `price` column and renames it automatically.
- Translators no longer read price from `md.trans_path_df` — price is resolved in `ModelNode` and passed as the `media_price` kwarg.
- **Multi-node DAGs use one sheet per node**: e.g. `translation_aba`, `translation_consideration`, `translation_sales` (p06), `translation_uba`, `translation_sales` (p08). Load each with `fmr.trans_dict_create(path=options_path, sheet="translation_<node>")` and pass the matching dict to the corresponding node.

#### 4. `media_prices` (NEW — optional)

| scope_name   | item       | multiplier |
|--------------|------------|------------|
| brandx_2025  | bdg_nat_tv | 1.04       |
| brandx_2025  | bdg_digital| 1.02       |

- Read with `fmr.media_prices_read(path)` — returns `{scope_name: {item: multiplier}}`, or `{}` if the sheet is absent.
- Per-tool inflation override. Falls back to scope's `inflation_multiplier`, then `1.0`.
- Final price = `base_price * media_prices[scope_name][item]`.
- Attach after construction: `model_rep.media_prices = media_prices`.

#### 5. `seasonality` (CHANGED — keyed by scope_name, bare column names)

| scope_name     | period | tv      | digital | ... |
|----------------|--------|---------|---------|-----|
| brandx_default | 1      | 0.10    | 0.12    | ... |
| brandx_default | 2      | 0.08    | 0.09    | ... |

- Read with `fmr.seasonality_read(path)` — groups rows by `scope_name` and returns `{scope_name: DataFrame}` (or `{}` if the sheet is absent).
- **Column names are bare tool names** (`tv`, `digital`, ...) — no `bdg_`/`nobdg_` prefix. `ModelNode._build_bdg_ssn_cols()` strips the prefix from `trans_dict` items to match them.
- Seasonality is keyed by `scope_name` (column), not option name.
- `OptionScope.seasonality` references a key; falls back to `scope_name`.
- Multiple scopes can share the same seasonality profile (many:1).

---

## meta_conf.toml — Model Graph

The single `meta_conf.toml` defines the DAG structure and per-node settings:

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

- `merge_fn` is optional — an absent `merge_fn` means the nodes run independently (parallel branches). When present it is a function name resolved from `code_py.adhoc.reporting` or from adhoc code loaded via `ModelRep.adhoc_code_src`.
- Execution order is a topological sort of the edges (Kahn's algorithm) — there is **no `order` key**.
- Node folders are named `<name>` or `model_<name>` (the layout written by `MetaModel.save`); a leading `model_` prefix on graph names is stripped on load.
- `model_rep.pkl.lzma` is optional per node — when absent a default `ModelRep` is auto-created with a warning.
- Legacy `meta_conf.xlsx` / `meta_model_conf.xlsx` (`model_graph` sheet) is still read on load with a warning — re-save to upgrade.

**Migration from old positional convention:**

```python
# OLD: merge function discovered by positional index
fun_name = 'code_py.adhoc.reporting.pred_merge_' + str(i)

# NEW: merge function referenced explicitly by edge
# meta_conf.toml graph table: bht | sales | pred_merge_bht
```

The positional convention survives only inside the deprecated `export_*` wrappers, which convert a `list` of models via `_convert_to_metamodel(..., merge_fn_pattern="code_py.adhoc.reporting.pred_merge_{i}")`.

Merge function signature unchanged:
```python
def pred_merge_bht(ds_cur: pd.DataFrame, dt_pred_prev: pd.DataFrame, allow_past: bool = False) -> pd.DataFrame:
    return ds_cur
```

---

## Budget Optimization (NEW API)

The six public optimizers take `meta_model`, `scopes`, `budget_scope`, `borders_dict`, and an `Option` as the base scenario — the two COBYLA / two Nevergrad CMA-ES functions plus their `*_multi` parallel variants (`optimize_budget_local_cobyla`, `optimize_target_local_cobyla`, `optimize_budget_local_ncma`, `optimize_target_local_ncma`, `optimize_budget_local_cobyla_multi`, `optimize_target_local_cobyla_multi`). `borders_dict` is `{var: [low, high, start]}`.

```python
borders_dict = {"bdg_tv": [0, 500, 200], "bdg_digital": [0, 100, 30]}

# Maximize KPI with a known total budget
opt = fmr.optimize_budget_local_cobyla(
    meta_model=mm, scopes=scopes, budget_scope="brand_1_2026",
    borders_dict=borders_dict, option=options["zero"],
    bdg_size=300, epochs=3, iters_epoch=300,
    if_exact=False, if_volume=True, warn_nan_seasonality=False,
)

# Minimize budget for a target KPI
opt = fmr.optimize_target_local_cobyla(
    meta_model=mm, scopes=scopes, budget_scope="brand_1_2026",
    borders_dict=borders_dict, option=options["zero"],
    trg=500_000_000, trg_tol_ratio=0.001, if_exact=True, if_volume=True,
)

# Parallel multi-run: dict[float, Option] keyed by budget / target
res = fmr.optimize_budget_local_cobyla_multi(
    meta_model=mm, scopes=scopes, budget_scope="brand_1_2026",
    borders_dict=borders_dict, option=options["zero"],
    bdg_sizes=[100, 200, 300], cores=4,
)

# Multi-node scoring: weighted combination of DAG nodes
opt = fmr.optimize_budget_local_cobyla(
    ..., summary_nodes={"aba": (0.25, "brand_1_2026"),
                        "consideration": (0.25, "brand_1_2026"),
                        "sales": (0.5, "brand_1_2026")},
)
```

- `summary_nodes` is `{node_name: (weight, scope_name)}` — the combined score is `sum(w_i * score_i / baseline_i)`. `None` = single-sink behavior. A list value `[weight, scope_name]` works too.
- The old signature (`ds=`, `model=`, `model_rep=`, `opt_set=`) still works but emits a `DeprecationWarning`.

---

## Efficiency Curves (compute_grid)

The full (model-blind) curves path is now `parallel.compute_grid()`, which returns one DataFrame per DAG node.

```python
curves_full_data = fmr.compute_grid(
    mm,
    scopes_crv,                 # e.g. [scope] for the target scope
    budget_step=5,
    bdg_max=300,
    cores=14,
    if_exact=True,
    allow_past=True,
    background_option=options["predefined"],   # optional
)
# -> {"sales": DataFrame, "uba": DataFrame, ...}

figs = fmr.curves_full_plot_short(curves_full_data["sales"], model_rep=node_sales)
figs = fmr.curves_full_plot_long(curves_full_data["sales"], model_rep=node_sales)
```

- `background_option` (keyword-only) supplies fixed additions merged into every grid option — `None`, a raw dict `{scope_name: {item: value}}`, or an `Option` (per-region curves where the national node must see the other regions' costs held fixed, p08).
- `translation` may be omitted — it falls back to the sink node's `model_rep.trans_dict`.

---

## PPTX Export

Old wrappers `export_option_table`, `export_option_detail`, `export_curves`, `export_adstocks` are deprecated shims that emit a `DeprecationWarning`. Use the new clean functions (exported both at `fmr.*` and `fmr.export_pptx.*`):

```python
node.model_rep, prs = export_pptx.set_config(node.model_rep, template_name="blanc")

# Option summary table
prs = export_pptx.options_table(prs, meta_model=mm, scopes=scopes, options=options, if_exact=True)

# Detailed per-option slides
prs = export_pptx.options_detail(prs, meta_model=mm, scopes=scopes, options=options,
                                 sku_var=["superbrand", "master", "form"],
                                 period="month", if_exact=True, if_volume=True,
                                 if_sku_fit_predict=True, if_sku_decompose=True,
                                 if_sku_waterfall=True)

# Retro analysis (options=None): show observed periods only
prs = export_pptx.options_detail(prs, meta_model=mm, scopes=scopes, options=None,
                                 sku_var=["superbrand"], period="month",
                                 listed_filter=(2, 3), if_volume=True)

# SKU hierarchy + simultaneous (summary) scopes
prs = export_pptx.options_detail(prs, meta_model=mm, scopes=scopes, options=options,
                                 sku_var=["superbrand", "region", "master", "form"],
                                 if_sku_hierarchy=True, summary_scopes=scopes_regional)

# Efficiency curves
prs = export_pptx.curves(prs, meta_model=mm, scopes=scopes_crv,
                         budget_step=5, bdg_max=300, if_exact=True, cores=14,
                         background_option=options["predefined"])

# Adstock curves
prs = export_pptx.adstocks(prs, meta_model=mm, cln_meas=cln_meas,
                           cln_dim=cln_dim, n=52)

prs.save("output/_report.pptx")
```

New parameters worth knowing:

- `options_detail`: `options=None` enables retro analysis; `listed_filter=(2, 3)` excludes the future period; `if_sku_hierarchy=True` reports every prefix level of `sku_var` (skipping levels covered by the brand-level `group_var`); `summary_scopes` aggregates simultaneous scopes (e.g. per-region) into one slide set per option.
- `curves`: `background_option` behaves exactly like `compute_grid()`'s.

---

## Full Workflow Example

### Single-level model

```python
import fermatrica_rep as fmr

# Load everything from the context file in one call
ctx = fmr.load_context("context.xlsx")

# Build node and compute (trans_dict/ssn_dict/media_prices wired via compute_context)
mm = fmr.MetaModel(nodes=node, compute_context=ctx)
results = mm.compute(ctx.options["predefined"], scopes=ctx.scopes, return_full=True)

for scope_name, (ds, dt_pred, summary) in results["model"].items():
    print(f"{scope_name}: vol={summary['pred_exact_vol']:.0f}, growth={summary['growth_vol']:.2f}")
```

### Multi-level DAG model

```python
# Load multi-level model from disk
meta_model = fmr.MetaModel.load("path/to/model_dir/")

# Compute across all nodes
results = meta_model.compute(option, scopes=scopes)

# Access specific node + scope (default return: {node: {scope: summary}})
sales_summary = results["sales"]["brandx_2025"]
```

### Adding seasonality and media prices

```python
# Load seasonality and media prices
seasonality = fmr.seasonality_read(options_path)
media_prices = fmr.media_prices_read(options_path)

# Attach to ModelRep (ssn_dict is a constructor param; media_prices is not)
model_rep = fmr.ModelRep(ds=ds, trans_dict=trans_dict, ssn_dict=seasonality)
model_rep.media_prices = media_prices

node = fmr.ModelNode(model=model, ds=ds, model_rep=model_rep)
results = node.compute(option, scopes=scopes)
```

Alternatively pass `ssn_dict=...` directly to `ModelNode(model_path=..., ssn_dict=ssn_dict)` as the samples do.

### Code generation (pipeline)

`fmr.generate_pipeline()` reads the `translation_*` sheets pattern and auto-generates the remaining options infrastructure (scopes / allocations / media_prices / seasonality sheets plus per-node reporting scripts) into a new XLSX under `{project_root}/model_data/options/`. The generated scripts follow the new `ModelNode` / `MetaModel` API shown above.

---

## Price Elasticity (elasticity_data)

`elasticity_data` was refactored to the new API. Old `OptionSettings`, raw dict options, `option_report_multi_var`, `targets_new`, and `cores` are removed.

**Migration:**

```python
# OLD
opt_set = fmr.OptionSettings(
    target=["BrandX"],
    date_start="2025-01-01",
    date_end="2025-12-31",
    ref_date_start="2024-01-01",
    ref_date_end="2024-12-31",
    apply_vars=["superbrand"],
    plan_period="exact",
)
option = {"bdg_tv": 50, "bdg_digital": 30, "bdg_pr": 20, "price": 1.0}
model_rep = fmr.ModelRep(ds, trans_dict=trans_dict)

result = elasticity_data(
    model=model, model_rep=model_rep, ds=ds,
    opt_set=opt_set, option=option,
    step=0.1, price_range=(-0.5, 0.5),
    cores=1, if_exact=True,
)

# NEW
scope = fmr.OptionScope(
    scope_name="brandx_2025",
    date_start="2025-01-01",
    date_end="2025-12-31",
    ref_date_start="2024-01-01",
    ref_date_end="2024-12-31",
    filters={"superbrand": ["BrandX"]},
    plan_period="exact",
)
option = fmr.Option(name="base", budget_data={
    "brandx_2025": {"bdg_tv": 50, "bdg_digital": 30, "bdg_pr": 20},
})
node = fmr.ModelNode(model=model, ds=ds, model_rep=model_rep)

result = fmr.elasticity_data(
    model=node, scope=scope, option=option,
    step=.01, price_range=(-0.25, 0.26),
    price_name_opt="nobdg_price", var="exact", if_exact=True,
)
```

Key changes:

- `OptionSettings` parameters become `OptionScope` — scope name, filters dict, and plan_period are all in one object.
- Budget dict becomes `Option.budget_data` keyed by scope name.
- `model` takes `ModelNode | MetaModel` directly; bare `Model` + `ds` + `model_rep` overrides are removed.
- `targets_new` (cross-elasticity) is removed — use a separate `OptionScope` for summarization.
- `cores` is removed — always sequential.
- `price_name_opt` names the price item inside the option budget data (default `"nobdg_price"`); `var` selects `"exact"` or `"long"` prediction for summarizing.
- If `model` is a `MetaModel`, the return type is `dict[str, pd.DataFrame]` keyed by node name.

---

## Removed APIs

Truly removed (no shim — code using these breaks):

| Removed | Replacement |
|---------|-------------|
| `MetaModelConf` | Graph DataFrame read directly in `MetaModel._load_multi()` |
| `MetaModelObj` | Model/ds/ModelRep dicts built inline in `MetaModel._load_multi()` |
| `long_term_type` parameter | `if_exact` (True/False) replaces it |
| `if_ah` parameter | Removed — never implemented |
| `apply_vars` / `target` on OptionScope | `filters` dict |
| `text_flags` / `name_` convention | `OptionScope.seasonality` |
| `md.trans_path_df` for price | `base_price * media_prices` resolved in `ModelNode` |
| `elasticity_data(opt_set, option_dict, targets_new, cores)` | `elasticity_data(model, scope, option)` — plus `price_name_opt`, `var` |

Deprecated — still present as working shims that emit `DeprecationWarning`:

| Deprecated | Replacement |
|------------|-------------|
| `OptionSettings` | `OptionScope` |
| `option_report()` | `ModelNode.compute()` or `MetaModel.compute()` |
| `option_report_multi_var()` | `ModelNode.compute()` / `MetaModel.compute()` / `parallel.compute_multi()` |
| `option_report_multi_post()` | `parallel.compute_grid()` |
| `export_option_table()` | `export_pptx.options_table()` |
| `export_option_detail()` | `export_pptx.options_detail()` |
| `export_curves()` | `export_pptx.curves()` |
| `export_adstocks()` | `export_pptx.adstocks()` |

All deprecated entry points still work but emit a `DeprecationWarning`: `option_report()`, `option_report_multi_var()`, `option_report_multi_post()`, `export_option_table()`, `export_option_detail()`, `export_curves()`, `export_adstocks()`, and the `ds=, model=, model_rep=, opt_set=` signatures of the optimizers. They convert old types internally via `_convert_to_metamodel` and `_convert_old_options`, and they enrich legacy models' `trans_dict["base_price"]` from `model.conf.trans_path_df` via `_enrich_trans_dict_from_model`.
