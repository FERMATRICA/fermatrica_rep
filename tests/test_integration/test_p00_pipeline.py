"""Integration tests against p00_sample (time series, OLS, 1 brand).

Tests the full reporting pipeline on the simplest available sample.
The sample carries an adhoc translator module (code_py.adhoc.translators)
referenced by two dotted rows in the context.xlsx translation sheet.
"""
import numpy as np
import pandas as pd
import pytest
from datetime import date

from tests.test_integration.conftest import _activate_adhoc_path


@pytest.mark.integration
class TestP00Pipeline:
    """Full reporting pipeline tests against p00_sample (time series OLS)."""

    BRAND = "brand_x"

    @pytest.fixture
    def p00_setup(self, p00_model_ds, samples_path):
        """Build reporting config from model and dataset (function-scoped)."""
        import fermatrica_rep as fmr

        model, ds = p00_model_ds
        ds = ds.copy()

        root = samples_path / "p00_sample"
        opt_path = str(root / "code_py" / "model_data" / "options" / "options.xlsx")
        ssn_path = str(root / "code_py" / "model_data" / "options" / "seasonality.xlsx")

        trans_dict = fmr.trans_dict_create(path=opt_path, sheet="translation")
        options_m = fmr.budget_dict_create(path=opt_path, sheet="main", vat_rate=0.0)
        model_rep = fmr.ModelRep(ds, trans_dict=trans_dict, language="english")
        ds = fmr.media_ssn_apply(ds, path=ssn_path, sheet="data")

        return {
            "model": model, "ds": ds, "model_rep": model_rep,
            "options_m": options_m,
        }

    def test_model_loads(self, p00_setup):
        """Model object is not None and has expected structure."""
        model = p00_setup["model"]
        assert model.obj is not None
        assert model.conf is not None
        assert model.conf.Y_var is not None

    def test_dataset_has_expected_columns(self, p00_setup):
        """Dataset contains standard columns expected by reporting functions."""
        ds = p00_setup["ds"]
        for col in ["date", "superbrand", "listed"]:
            assert col in ds.columns, f"Missing column: {col}"

    def test_option_report_runs(self, p00_setup):
        """option_report produces predictions with observed + predicted columns."""
        import fermatrica_rep as fmr

        model = p00_setup["model"]
        ds = p00_setup["ds"]
        model_rep = p00_setup["model_rep"]
        options_m = p00_setup["options_m"]

        opt_set = fmr.OptionSettings(
            target=[[self.BRAND]],
            date_start=date(2018, 1, 1),
            date_end=date(2018, 12, 31),
            ref_date_start=date(2017, 1, 1),
            ref_date_end=date(2017, 12, 31),
            apply_vars=["superbrand"],
            plan_period="exact",
        )

        ds_out, dt_pred, opt_sum = fmr.option_report(
            model, ds, model_rep, options_m["predefined"], opt_set, if_exact=False,
        )

        assert ds_out is not None
        assert dt_pred is not None
        assert "observed" in dt_pred.columns
        assert "predicted" in dt_pred.columns
        assert len(dt_pred) > 0

        # opt_sum should contain long-term keys when if_exact=False
        assert "pred_long_vol" in opt_sum
        assert "pred_long_val" in opt_sum

    def test_option_report_summary_structure(self, p00_setup):
        """opt_sum contains expected keys depending on if_exact mode."""
        import fermatrica_rep as fmr

        model = p00_setup["model"]
        ds = p00_setup["ds"]
        model_rep = p00_setup["model_rep"]
        options_m = p00_setup["options_m"]

        opt_set = fmr.OptionSettings(
            target=[[self.BRAND]],
            date_start=date(2018, 1, 1),
            date_end=date(2018, 12, 31),
            ref_date_start=date(2017, 1, 1),
            ref_date_end=date(2017, 12, 31),
            apply_vars=["superbrand"],
            plan_period="exact",
        )

        # Test if_exact=False (long-term)
        _, _, opt_sum_long = fmr.option_report(
            model, ds, model_rep, options_m["predefined"], opt_set, if_exact=False,
        )
        assert "pred_long_vol" in opt_sum_long
        assert "pred_long_val" in opt_sum_long

        # Test if_exact=True (exact period)
        _, _, opt_sum_exact = fmr.option_report(
            model, ds, model_rep, options_m["predefined"], opt_set, if_exact=True,
        )
        assert "pred_exact_0" in opt_sum_exact
        assert "pred_exact_vol" in opt_sum_exact["pred_exact_0"]
        assert "pred_exact_val" in opt_sum_exact["pred_exact_0"]

    def test_option_report_zero_vs_predefined_budget(self, p00_setup):
        """Option with predefined (non-zero) budget yields higher predicted than zero budget."""
        import fermatrica_rep as fmr

        model = p00_setup["model"]
        ds = p00_setup["ds"]
        model_rep = p00_setup["model_rep"]
        options_m = p00_setup["options_m"]

        opt_set = fmr.OptionSettings(
            target=[[self.BRAND]],
            date_start=date(2018, 1, 1),
            date_end=date(2018, 12, 31),
            ref_date_start=date(2017, 1, 1),
            ref_date_end=date(2017, 12, 31),
            apply_vars=["superbrand"],
            plan_period="exact",
        )

        _, dt_pred_zero, opt_sum_zero = fmr.option_report(
            model, ds, model_rep, options_m["zero"], opt_set, if_exact=False,
        )
        _, dt_pred_predefined, opt_sum_predefined = fmr.option_report(
            model, ds, model_rep, options_m["predefined"], opt_set, if_exact=False,
        )

        # Non-zero budget should produce higher predicted KPI
        pred_total_zero = opt_sum_zero["pred_long_vol"]
        pred_total_predefined = opt_sum_predefined["pred_long_vol"]

        assert pred_total_predefined >= pred_total_zero, (
            f"Predefined budget should produce >= predicted KPI than zero budget "
            f"(zero={pred_total_zero:.2f}, predefined={pred_total_predefined:.2f})"
        )

    def test_option_report_mismatched_keys_handled(self, p00_setup):
        """option_report handles unknown budget keys without crashing."""
        import fermatrica_rep as fmr

        model = p00_setup["model"]
        ds = p00_setup["ds"]
        model_rep = p00_setup["model_rep"]

        opt_set = fmr.OptionSettings(
            target=[[self.BRAND]],
            date_start=date(2018, 1, 1),
            date_end=date(2018, 12, 31),
            ref_date_start=date(2017, 1, 1),
            ref_date_end=date(2017, 12, 31),
            apply_vars=["superbrand"],
            plan_period="exact",
        )

        bad_option = {"bdg_nonexistent": 100}

        ds_out, dt_pred, opt_sum = fmr.option_report(
            model, ds, model_rep, bad_option, opt_set, if_exact=False,
        )

        assert dt_pred is not None
        assert "observed" in dt_pred.columns

    def test_extract_effect_runs(self, p00_setup):
        """extract_effect produces decomposition with variable and value columns."""
        import fermatrica_rep as fmr

        model = p00_setup["model"]
        ds = p00_setup["ds"]
        model_rep = p00_setup["model_rep"]

        split_m_m = fmr.extract_effect(model, ds, model_rep)

        assert split_m_m is not None
        assert len(split_m_m) > 0
        assert "variable" in split_m_m.columns
        assert "value" in split_m_m.columns

    def test_extract_effect_no_nan_inf(self, p00_setup):
        """extract_effect produces finite values in the value column."""
        import fermatrica_rep as fmr

        model = p00_setup["model"]
        ds = p00_setup["ds"]
        model_rep = p00_setup["model_rep"]

        split_m_m = fmr.extract_effect(model, ds, model_rep)

        assert np.all(np.isfinite(split_m_m["value"])), (
            "extract_effect produced NaN or Inf values in decomposition"
        )

    def test_extract_effect_value_sum(self, p00_setup):
        """extract_effect decomposition values sum equals model prediction per period."""
        import fermatrica as fm
        import fermatrica_rep as fmr

        model = p00_setup["model"]
        ds = p00_setup["ds"]
        model_rep = p00_setup["model_rep"]

        split_m_m = fmr.extract_effect(model, ds, model_rep)

        sum_per_period = split_m_m.groupby(["superbrand", "bs_key", "date"])["value"].sum().reset_index()
        sum_per_period.columns = ["superbrand", "bs_key", "date", "decomposed_total"]

        # Use fit_predict for proper model-specific prediction
        pred_values, _, _ = fm.fit_predict(ds, model, if_fit=False, return_full=True)
        ds_pred = ds[["superbrand", "bs_key", "date"]].copy()
        ds_pred["predicted"] = pred_values.values
        ds_pred = ds_pred.dropna(subset=["predicted"])
        pred_agg = ds_pred.groupby(["superbrand", "bs_key", "date"], as_index=False)["predicted"].sum()

        merged = sum_per_period.merge(pred_agg, on=["superbrand", "bs_key", "date"], how="inner")

        for _, row in merged.iterrows():
            if abs(row["predicted"]) > 1e-10:
                rel_diff = abs(row["decomposed_total"] - row["predicted"]) / abs(row["predicted"])
                assert rel_diff < 1e-6, (
                    f"Decomposition [{row['superbrand']}, {row['date']}]: "
                    f"decomposed={row['decomposed_total']:.4f}, "
                    f"predicted={row['predicted']:.4f}, "
                    f"rel_diff={rel_diff:.4%}"
                )

    def test_extract_effect_detail_mode(self, p00_setup):
        """extract_effect with if_detail=True produces more variable categories."""
        import fermatrica_rep as fmr

        model = p00_setup["model"]
        ds = p00_setup["ds"]
        model_rep = p00_setup["model_rep"]

        split_basic = fmr.extract_effect(model, ds, model_rep, if_detail=False)
        split_detail = fmr.extract_effect(model, ds, model_rep, if_detail=True)

        # Detail mode should produce at least as many unique variables
        n_vars_basic = split_basic["variable"].nunique()
        n_vars_detail = split_detail["variable"].nunique()

        assert n_vars_detail >= n_vars_basic, (
            f"if_detail=True produced fewer variables ({n_vars_detail}) "
            f"than if_detail=False ({n_vars_basic})"
        )

    def test_waterfall_data_runs(self, p00_setup):
        """waterfall_data processes extract_effect output into summary."""
        import fermatrica_rep as fmr

        model = p00_setup["model"]
        ds = p00_setup["ds"]
        model_rep = p00_setup["model_rep"]

        split_m_m = fmr.extract_effect(model, ds, model_rep)
        wf = fmr.waterfall_data(
            split_m_m,
            brands=[self.BRAND],
            date_start="2017-01-01",
            date_end="2017-12-31",
        )

        assert len(wf) > 0
        assert "variable" in wf.columns
        assert "value" in wf.columns

        # Ratio column should sum to ~100%
        assert abs(wf["ratio"].sum() - 100.0) < 0.1, (
            f"waterfall_data ratio column sums to {wf['ratio'].sum():.2f}%, expected ~100%"
        )

        # Cumulative sum at last row should equal total sum
        assert abs(wf["value_cumsum"].iloc[-1] - wf["value"].sum()) < 1e-6, (
            f"waterfall_data cumsum ({wf['value_cumsum'].iloc[-1]:.4f}) "
            f"does not match value sum ({wf['value'].sum():.4f})"
        )

    def test_edge_case_future_dates_only(self, p00_setup):
        """option_report with future dates beyond train/test: forecast rows are listed=4."""
        import fermatrica_rep as fmr

        model = p00_setup["model"]
        ds = p00_setup["ds"]
        model_rep = p00_setup["model_rep"]
        options_m = p00_setup["options_m"]

        max_date = ds["date"].max()
        # Use the last year of the dataset which has listed=4 (future) rows
        opt_set = fmr.OptionSettings(
            target=[[self.BRAND]],
            date_start=max_date - pd.Timedelta(days=180),
            date_end=max_date,
            ref_date_start=date(2017, 1, 1),
            ref_date_end=date(2017, 12, 31),
            apply_vars=["superbrand"],
            plan_period="exact",
        )

        _, dt_pred, _ = fmr.option_report(
            model, ds, model_rep, options_m["predefined"], opt_set, if_exact=False,
        )

        future_start = max_date - pd.Timedelta(days=180)
        forecast = dt_pred[
            (dt_pred["date"] >= future_start) &
            (dt_pred["date"] <= max_date)
        ]

        assert len(forecast) > 0, "No rows in the forecast period"
        assert set(forecast["listed"].unique()).issubset({4}), (
            f"Forecast rows should be listed=4, got {forecast['listed'].unique()}"
        )

    def test_edge_case_allow_past(self, p00_setup):
        """option_report with allow_past=True and past dates runs without error."""
        import fermatrica_rep as fmr

        model = p00_setup["model"]
        ds = p00_setup["ds"]
        model_rep = p00_setup["model_rep"]
        options_m = p00_setup["options_m"]

        opt_set = fmr.OptionSettings(
            target=[[self.BRAND]],
            date_start=date(2016, 1, 1),
            date_end=date(2016, 12, 31),
            ref_date_start=date(2015, 1, 1),
            ref_date_end=date(2015, 12, 31),
            apply_vars=["superbrand"],
            plan_period="exact",
        )

        _, dt_pred, _ = fmr.option_report(
            model, ds, model_rep, options_m["predefined"], opt_set, if_exact=False,
            allow_past=True,
        )

        assert dt_pred is not None
        assert len(dt_pred) > 0


@pytest.mark.integration
class TestP00NewApi:
    """New-API (Option/OptionScope/ModelNode/MetaModel) tests against p00_sample.

    Regression coverage for the refactored reporting flow: per-scope option
    computation via MetaModel.compute and full-horizon seasonality on weekly data
    (monthly seasonality merged through the `period` column — the p00-specific trap,
    where stale saved models lack `period` and produce NaN on future rows).
    """

    BRAND = "brand_x"
    SCOPES = ["brand_x_2018", "brand_x_2017"]

    @pytest.fixture
    def p00_new_setup(self, p00_model_ds, samples_path):
        """Build a single-node MetaModel from context.xlsx (function-scoped)."""
        import fermatrica_rep as fmr

        model, ds = p00_model_ds

        root = samples_path / "p00_sample"
        opt_path = str(root / "code_py" / "model_data" / "options" / "context.xlsx")

        # p00's translation sheet references code_py.adhoc.translators.adhoc_basic
        # for bdg_tv_nat / bdg_dgt_da, so the adhoc module must be importable
        # and registered on the node's ModelRep for the dotted dispatch to work.
        _activate_adhoc_path(root)
        import code_py.adhoc.translators

        scopes = fmr.scopes_read(opt_path)
        options = fmr.Option.from_excel_all(opt_path)
        ssn_dict = fmr.seasonality_read(path=opt_path)
        trans_dict = fmr.trans_dict_create(path=opt_path, sheet="translation")

        node = fmr.ModelNode(model=model, ds=ds, trans_dict=trans_dict,
                             language='english', ssn_dict=ssn_dict,
                             adhoc_code=[code_py.adhoc.translators],
                             scope_names={s.scope_name for s in scopes})
        mm = fmr.MetaModel(nodes=node)

        return {"mm": mm, "scopes": scopes, "options": options, "node": node}

    def test_context_parse(self, p00_new_setup):
        """context.xlsx parses into 2 scopes and zero/predefined options."""
        scopes = p00_new_setup["scopes"]
        options = p00_new_setup["options"]

        assert len(scopes) == 2
        assert {s.scope_name for s in scopes} == set(self.SCOPES)
        assert set(options.keys()) == {"zero", "predefined"}

        for s in scopes:
            assert s.filters["superbrand"] == [self.BRAND]
            items = options["predefined"].budget_for_scope(s.scope_name)
            assert items, f"no budget items for {s.scope_name}"

    def test_meta_model_compute_per_scope(self, p00_new_setup):
        """MetaModel.compute yields per-scope pred_exact_vol > 0 and finite."""
        mm = p00_new_setup["mm"]
        scopes = p00_new_setup["scopes"]
        options = p00_new_setup["options"]

        result = mm.compute(options["predefined"], scopes=scopes, allow_past=True)["model"]

        for s in scopes:
            summary = result[s.scope_name]
            assert summary["pred_exact_vol"] > 0, (
                f"{s.scope_name} pred_exact_vol must be > 0, got {summary['pred_exact_vol']}"
            )
            assert np.isfinite(summary["pred_exact_vol"])
            assert np.isfinite(summary["pred_long_vol"])

    def test_seasonality_covers_horizon(self, p00_new_setup):
        """No NaN opt_media_ssn on listed==4 rows (monthly seasonality on weekly data)."""
        mm = p00_new_setup["mm"]
        scopes = p00_new_setup["scopes"]
        options = p00_new_setup["options"]

        mm.compute(options["predefined"], scopes=scopes, allow_past=True)
        node = mm.nodes["model"]

        ssn_cols = [c for c in node.ds.columns if c.startswith("opt_media_ssn")]
        assert ssn_cols, "no opt_media_ssn columns after compute"

        future = node.ds[node.ds["listed"] == 4]
        assert future[ssn_cols].isna().sum().sum() == 0, (
            "NaN media seasonality inside the forecast horizon"
        )

        beyond_plan_year = future[future["date"].dt.year > 2018]
        assert len(beyond_plan_year) > 0
        assert beyond_plan_year[ssn_cols].isna().sum().sum() == 0, (
            "NaN media seasonality beyond the plan year"
        )

        # historical years before the earliest scope (2017) must stay NaN,
        # except for the 2016 week boundary row the 2017 scope window includes
        historical = node.ds[(node.ds["date"].dt.year < 2016) & (node.ds["listed"].isin([1, 2]))]
        assert historical[ssn_cols].isna().all().all(), (
            "historical years must keep NaN seasonality"
        )
