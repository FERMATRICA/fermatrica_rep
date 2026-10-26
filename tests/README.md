# FERMATRICA_REP Tests

1052 tests (873 unit + 179 integration) covering reporting, analysis, options, and the export pipeline.

---

## 1. Basic Concepts

### Test Layers

Tests are organized into three layers:

| Layer | Location | Data | Depends On | Gate |
|-------|----------|------|------------|------|
| **Unit** | tests/test_*.py | Synthetic in-memory | Nothing external | Always runs |
| **Static fixtures** | tests/fixtures/ | .xlsx shipped in repo | Self-contained | Always runs |
| **Integration** | tests/test_integration/ | Real models from a local samples directory | `--samples-path`, env var, or root config.toml | Auto-skipped |

- **Layer 1**: Synthetic DataFrames/Series test logic in isolation. Fastest and most reliable.
- **Layer 2**: Two shipped xlsx files (options.xlsx, seasonality.xlsx from p01_sample) verify Excel parsers.
- **Layer 3**: Full pipeline against real trained models via fm.model_load_ext(). Gated behind @pytest.mark.integration.

### Fixtures (conftest.py, root)

| Fixture | Scope | Purpose |
|---------|-------|---------|
| split_m_m | function | Synthetic 12-month decomposition data (1 brand, 4 vars) |
| basic_model_rep | function | ModelRep built on split_m_m, language=english |
| fixtures_path | session | Path to tests/fixtures/ |
| options_xlsx_path | session | Path to fixtures/options.xlsx |
| seasonality_xlsx_path | session | Path to fixtures/seasonality.xlsx |
| samples_path | session | Samples directory path (from --samples-path, env var, or config.toml) |
| if_debug | function | Boolean for breakpoint debugging (default False) |

### Fixtures (conftest.py, test_integration/)

| Fixture | Scope | Purpose |
|---------|-------|---------|
| p00_model_ds | session | Loads p00 OLS model + dataset (no adhoc) |
| p01_model_ds | session | Loads p01 LME panel model + dataset (adhoc_unparam_fin) |
| p02_model_ds | session | Loads p02 SKU-level panel model + dataset |
| p03_model_ds | session | Loads p03 ABA monthly panel model + dataset |
| p04_model_ds | session | Loads p04 regional panel model + dataset |
| p05_models_ds | session | Loads p05 two-level (BHT + Sales) models + datasets |

Model discovery for p00-p05 uses `_find_latest_model()` — picks the most recent `model_*` directory. Adhoc code is activated on `sys.path` and cached modules are evicted between runs.

### Markers

| Marker | Purpose |
|--------|---------|
| integration | Tests requiring --samples-path or tests/config.toml |

---

## 2. Test Approach

### Coverage by module

| Module | Tested | Not tested |
|--------|--------|------------|
| basics.py | palette_fill, coef_var_align | — |
| stats.py | basic_metrics_table, metrics_group_table | Full metrics_table (needs real model) |
| transformation.py | transformation_main_plot, transformation_plot | — |
| waterfall.py | waterfall_data, waterfall_plot | _waterfall_data_worker (indirect) |
| options/define.py | trans_dict_create, budget_dict_create, media_ssn_create/apply, option_settings_list_create | budget_ssn_create (tested in separate file) |
| options/translators.py | basic, basic_fix, price | ... |
| options/calc.py | option_report (integration only) | Direct unit tests |
| curves.py | curves_simple_data, curves_simple_plot (integration, 3 samples) | — |
| curves_full.py | curves_full_plot_short, curves_full_plot_long (integration, 2 samples) | — |
| decomposition.py | decompose_main_plot (integration, 5 samples) | extract_effect unit tests |
| fit.py | fit_main_plot_vol, fit_mult_plot_vol, fit_main_data (integration, 5 samples) | — |
| category.py | category_data, category_main_plot, category_err_plot, category_plot (unit) | — |
| adstocks.py | adstocks_data (integration, p01) | — |
| elasticity_price.py | elasticity_data (integration, p01) | — |
| analysis/ols.py | get_model_ds, trans, rhs_comparison, impacts_comparison (unit + integration) | — |
| export/* | — | Requires MS Office on Windows |
| options/optim.py | — | Requires real model, slow |
| meta_model/ | ModelRep, ModelNode, MetaModel, MetaModelDiagnostics | — |

### Integration test coverage by sample

| Sample | Type | Pipeline | Curves | Curves Full | Decomp | Fit | Others |
|--------|------|----------|--------|-------------|--------|-----|--------|
| p00 | OLS, 1 brand | test_p00_pipeline | test_curves | test_curves_full | test_decomposition | test_fit | test_analysis_ols |
| p01 | Panel LME, 3 brands | test_p01_pipeline | test_curves | test_curves_full | test_decomposition | test_fit | test_adstocks, test_elasticity |
| p02 | Panel per SKU, 13 SKUs | test_p02_pipeline | test_curves | — | test_decomposition | test_fit | — |
| p03 | ABA monthly, 3 brands | test_p03_pipeline | test_curves | — | test_decomposition | test_fit | — |
| p04 | Regional panel, 4 regions | test_p04_pipeline | — | — | test_decomposition | test_fit | — |
| p05 | Two-level (BHT→Sales) | test_p05_pipeline | — | — | — | — | — |

### Synthetic data conventions

Unit test DataFrames mimic the real dataset:
- **date**: pd.date_range (MS=monthly, W=weekly)
- **superbrand**: String brand id (brand_1, brand_x)
- **bs_key**: Numeric panel group id
- **listed**: 2=train, 3=test, 4=future
- **observed/predicted**: Numeric targets (for metrics)
- **variable/value**: Decomposition outputs (for waterfall)

### YAML parametrization

test_budget_ssn_create.py loads test cases from `test_cases/budget_ssn_create_stability.yaml` for regression testing. Helpers in `tests/basics/basics.py` support `!pd.DataFrame`, `!pd.Series`, `!np.array` tags for future parametrized tests.

---

## 3. Test Groups

### test_basics.py — Utility functions (25 tests)
Tests for `fermatrica_rep.basics`:
- **TestPaletteFill** (11 tests): colourcet palette assignment. Existing colours preserved, no duplicates, empty input, 20+ entities.
- **TestCoefVarAlignOne** (7 tests): Single-var coefficient matching. Exact, regex fallback, word boundary, no-match → None.
- **TestCoefVarAlign** (7 tests): Batch alignment. Verifies coef_name/coef_value columns, empty input, non-mutation of originals.

### test_stats.py — Metrics tables (14 tests)
Tests for `fermatrica_rep.stats`:
- **TestBasicMetricsTable** (10 tests): R2, RMSE, MAPE. Train/test, MAPE threshold column, empty data.
- **TestMetricsGroupTable** (3 tests): Per-group metrics. Verifies r_sq, rmse, smape, pc columns, one row per brand.
- **TestMetricsTable** (1 test): Error when no real model provided.

### test_transformation.py — Transformation plots (12 tests)
Tests for `fermatrica_rep.transformation`:
- **TestTransformationMainPlot** (8 tests): Plotly Figure, two traces, secondary y-axis, title, height, if_main=False, listed=1/4.
- **TestTransformationPlot** (4 tests): Chain decomposition via sub=True/False.

### test_waterfall.py — Waterfall decomposition (12 tests)
Tests for `fermatrica_rep.waterfall`, using shared `split_m_m` and `basic_model_rep` from conftest:
- **TestWaterfallData** (8 tests): Date/brand filtering, absolute/signed sort, multi-brand, pretrain, list input.
- **TestWaterfallPlot** (4 tests): Returns go.Figure (bar & native), list input, horizontal reference line.

### test_category.py — Category fit/predict plots (18 tests)
Tests for `fermatrica_rep.category`:
- **TestCategoryData** (8 tests): Data returns, columns, period (month/week/year), err_int, show_future filtering.
- **TestCategoryMainPlot** (7 tests): Returns go.Figure, 3 traces, trace names, preserves existing traces, row/col passthrough, empty data raises.
- **TestCategoryErrPlot** (7 tests): Returns go.Figure, 3 traces, error column, err_int hlines, preserves existing traces, empty data raises.
- **TestCategoryPlot** (4 tests): Returns go.Figure, has traces, two subplots, err_int passthrough.

### test_analysis_ols.py — Advanced OLS diagnostics (21 tests)
Tests for `fermatrica_rep.analysis.ols`. Uses p00 real model, statsmodels fixtures, and synthetic data:
- **TestGetModelDs** (7 tests): Returns DataFrame, expected columns, Intercept=1, excludes listed=1, finite values.
- **TestTrans** (3 tests): Returns DataFrame, adds lag columns, transformed values correct.
- **TestRhsComparison** (5 tests): Returns DataFrame, expected columns, sign change detection.
- **TestImpactsComparison** (4 tests): Returns DataFrame, expected columns, impacts sum to 100.

### test_diagnostics.py — MetaModel diagnostics (45 tests)
Tests for `fermatrica_rep.meta_model.diagnostics`:
- **TestCheckResult** (3 tests): Construction, details dict, level enum values.
- **TestDiagnosticsReport** (6 tests): `.passed` property (error-only, ignores warnings, empty), `.summary` formatting (errors/warnings/info counts, skipped).
- **TestMetaModelDiagnosticsIntegration** (3 tests): `.run()` returns report, skips merge when no options, skips merge with <2 options.
- **TestCheckNodeTypes** (4 tests): Valid ModelNode passes, non-ModelNode fails with details, `raise_on_first=True` raises, `raise_on_first` only on error-level.
- **TestCheckNodeAttrs** (14 tests): model/None error, empty ds error, None model_rep error, empty vis_dict error, None trans_dict warning, invalid language error, dt_pred None/present/wrong-type, adhoc dir missing/comment-only/has-code-no-src warning/has-code-with-src passes.
- **TestCheckEdgeResolution** (7 tests): No edges info, None merge_fn info, callable passes, string resolves via eval, unresolvable string error, non-callable/non-string error, uninspectable signature warning.
- **TestCheckMergeEffects** (8 tests): Identical options skipped, propagation detected via ds+dt_pred comparison, state restoration on compute crash, None prediction skip, all columns identical warning, float/non-float column diff warnings, single-node effects.

### test_options/test_define.py — Options config parsers (32 tests)
Tests for `fermatrica_rep.options.define`, loading from `tests/fixtures/`:
- **TestTransDictCreate** (6 tests): 9 translation items, bdg_tv→variable mapping, error on missing sheet/file.
- **TestBudgetDictCreate** (8 tests): main/current sheets, VAT rate, DataFrame/dict input, error paths.
- **TestMediaSsnCreate** (5 tests): Seasonality DataFrames, sheet filtering, error paths.
- **TestOptionSettingsListCreate** (5 tests): Single/multi/partial year windows, options_dict merge, plan_period.
- **TestMediaSsnApply** (8 tests): Seasonality application, old column cleanup, date_start boundary, weekly.

### test_options/test_translators.py — Budget translators (14 tests)
Tests for `fermatrica_rep.options.translators`:
- **TestBasicFix** (6 tests): Fixed-value, masked/all/no-masked, float/zero vals.
- **TestPrice** (5 tests): Price ratio, increase, zero, custom var name.
- **TestBasic** (3 tests): Full translator + seasonality correction, default price, long-period.

### test_options/test_budget_ssn_create.py — Budget & seasonality creation (19+ tests)
- **TestBudgetSsnCreateFromPath** (6 tests): Excel load. All sheets, specific sheets, VAT, year column.
- **TestBudgetSsnCreateFromDict** (2 tests): Dict of DataFrames. Multi-year, VAT.
- **TestBudgetSsnCreateEdgeCases** (6 tests): No input, invalid path, no budget/yr columns, empty data.
- **TestBudgetSsnCreateSeasonality** (3 tests): Normalization, multi-year separate, zero budget.
- **TestBudgetSsnCreateIntegration** (1 test): Full multi-year multi-option workflow.
- **TestBudgetSsnCreateStability** (parametrized): YAML regression tests (5+ cases from test_cases/budget_ssn_create_stability.yaml).

### test_integration/test_p00_pipeline.py — p00 full pipeline (5 tests)
End-to-end on simplest sample (OLS, 1 brand, no adhoc code). Verifies model load, columns, option_report, extract_effect, waterfall_data.

### test_integration/test_p01_pipeline.py — p01 full pipeline (4 tests)
End-to-end on panel sample (LME, 3 brands, adhoc_unparam_fin). Verifies model load with sys.path adhoc code, 3 brands present, future predictions, multi-brand decomposition.

### test_integration/test_p02_pipeline.py — p02 full pipeline (11 tests)
End-to-end on SKU-level panel (LME, 13 SKUs, 3 brands). Verifies model load, multi-SKU dataset, option_report (zero vs predefined), extract_effect (multi-SKU, value sum matches prediction), waterfall_data ratio sum, edge cases (nonexistent brand, future-only dates).

### test_integration/test_p03_pipeline.py — p03 full pipeline (10 tests)
End-to-end on ABA target panel (monthly, LME, 3 brands). Verifies ABA target, 3 brands, option_report with ABA predictions, extract_effect, waterfall_data, edge cases.

### test_integration/test_p04_pipeline.py — p04 full pipeline (7 tests)
End-to-end on regional panel (LME, 4 geographic regions). Verifies regional model load, 4 regions, per-master option_report with media_prices, extract_effect with master column, edge cases.

### test_integration/test_p05_pipeline.py — p05 two-level pipeline (6 tests)
End-to-end on chained two-level panel (BHT → Sales). Verifies BHT and Sales model loads, pred_merge_1 importable, full chained option_report, extract_effect on both levels.

### test_integration/test_fit.py — Fit plots (18 tests)
Tests for `fermatrica_rep.fit` across 5 samples (p00-p04):
- **TestP00Fit** (6 tests): fit_main_plot_vol, fit_main_data wide/long consistency, period week/quarter, no future, zero err_int.
- **TestP01Fit** (3 tests): fit_main_plot_vol single brand, fit_mult_plot_vol 3 brands with/without future.
- **TestP02Fit** (3 tests): fit_main_plot_vol, fit_mult_plot_vol with/without future (SKU-level).
- **TestP03Fit** (1 test): fit_main_plot_vol with ABA target.
- **TestP04Fit** (2 tests): fit_main_plot_vol single region, fit_mult_plot_vol 4 regions.

### test_integration/test_decomposition.py — Decomposition plots (14 tests)
Tests for `fermatrica_rep.decomposition` across 5 samples (p00-p04):
- **TestP00Decomp** (3 tests): decompose_main_plot, contour_line, show_future=False.
- **TestP01Decomp** (4 tests): decompose_main_plot single brand, multi-brand, weekly period, brands=None.
- **TestP02Decomp** (2 tests): decompose_main_plot single brand, multi-brand (SKU-level).
- **TestP03Decomp** (1 test): decompose_main_plot with ABA model.
- **TestP04Decomp** (2 tests): decompose_main_plot single region, multi-region.

### test_integration/test_curves.py — Short-term efficiency curves (8 tests)
Tests for `fermatrica_rep.curves` across 3 samples (p00-p03):
- **TestP00Curves** (2 tests): curves_simple_data and curves_simple_plot (OLS).
- **TestP01Curves** (2 tests): curves_simple_data and curves_simple_plot (panel LME).
- **TestP02Curves** (2 tests): curves_simple_data and curves_simple_plot (SKU panel).
- **TestP03Curves** (2 tests): curves_simple_data and curves_simple_plot (ABA).
- NOTE: No TestP04Curves (p04 uses media_prices translation not supported by curves_simple_data).

### test_integration/test_curves_full.py — Full efficiency curves (2 tests)
Tests for `fermatrica_rep.curves_full` across 2 samples (p00-p01):
- **TestP00CurvesFull** (1 test): Full pipeline compute_grid → curves_full_plot_short → curves_full_plot_long.
- **TestP01CurvesFull** (1 test): Full curves pipeline on panel LME model.

### test_integration/test_adstocks.py — Adstock visualization (1 test)
- **TestP01Adstocks** (1 test): adstocks_data returns non-empty DataFrame with non-increasing decay weights.

### test_integration/test_elasticity.py — Price elasticity (1 test)
- **TestP01Elasticity** (1 test): elasticity_data returns non-empty DataFrame with finite values.

---

## 4. How to Run

### Prerequisites

- Python 3.11+ with `fermatrica`, `fermatrica_rep`, `fermatrica_utils`, `pandas`, `numpy`, `plotly`, `openpyxl`, `pyyaml`, `statsmodels` installed
- `pytest-xdist` is **required** — it is the enforced default runner (`addopts = "-n 4"` in `pyproject.toml`). Install it via `pip install -r requirements-dev.txt`
- Integration tests additionally require a directory with sample models (p00-p10). Sample models are not shipped with this repository.
- The samples path is configured via `config.toml` (repo root, copy from `config_example.toml`) or the `--samples-path` CLI flag

### Unit tests (fast, no dependencies)
```bash
pytest tests/ -v
pytest tests/ -m "not integration" -v
```
Result: 873 passed, 179 deselected.

### Single test file / class / method
```bash
pytest tests/test_waterfall.py -v
pytest tests/test_waterfall.py::TestWaterfallData -v
pytest tests/test_waterfall.py::TestWaterfallData::test_brand_filtering -v
```

### Integration tests
```bash
# CLI flag
pytest tests/ --samples-path /path/to/samples -m integration -v

pytest tests/ -m integration -v

# Config file (no CLI flags or env vars needed)
# Edit config.toml (repo root) with your local path, then run:
pytest tests/ -m integration -v
```
Result: 179 passed (integration tests only, with --samples-path).

### Full suite
```bash
pytest tests/ --samples-path /path/to/samples -v
```
Result: 1052 tests (1049 passed, 3 skipped).

### Parallel runs (pytest-xdist, enforced by default)

**Every `pytest` invocation in this repo runs with xdist automatically**
(`addopts = "-n 4"` in `pyproject.toml`). `pytest tests/` already uses 4
workers — no flag needed. Override with an explicit `-n` on the command line
(e.g. `-n 0` for a single-process run, `-n auto` for one worker per CPU).

The suite is safe to run in parallel — tests write only to their own `tmp_path`
directories, and the adhoc `sys.path`/`sys.modules` state is per-process. Session
fixtures load each sample model once per worker (the samples are small).

```bash
pip install -r requirements-dev.txt   # pytest-xdist (required)
pytest tests/                         # already -n 4 via addopts
pytest tests/ -n 0                    # single-process override (e.g. --if-debug)
pytest tests/ -n auto                 # one worker per CPU
```

On this machine (16 cores) the full suite runs in ~10 minutes with `-n 4`
(measured: 9:51). The heaviest single test (brand summary XLSX export) runs a
sequential in-process grid, so it does not parallelize internally — xdist
overlaps it with the rest of the suite instead.

### Debug mode
```bash
pytest tests/ --if-debug -n 0
```
Drops into breakpoint() on assertion failures for interactive debugging.
Pass `-n 0` — breakpoints cannot be interacted with inside xdist workers.

### VS Code GUI

Create `config.toml` at repo root (copy from `config_example.toml`) with your
samples path set. Integration tests then appear and run in the Test Explorer
without any workspace settings:
```toml
[paths]
samples = "C:\\path\\to\\samples"
```

---

## 5. TODO

### High priority
- [ ] Unit tests for options/calc.py (option_report) — currently integration-only
- [ ] Additional translator tests: basic_fix, price, tv_complex
- [ ] export/* tests — requires MS Office on Windows
- [ ] options/optim.py tests — requires real model, slow

### Medium priority
- [ ] Coverage reporting (pytest-cov)
- [ ] Unit tests for extract_effect (currently integration-only)
- [ ] Slow test markers for tiered CI

### Low priority
- [ ] Property-based tests (hypothesis)
