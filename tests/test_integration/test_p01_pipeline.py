"""Integration tests against p01_sample (panel per brand, LME, 3 brands).

Tests the reporting pipeline on a panel model that requires adhoc code
(code_py.adhoc.model.adhoc_unparam_fin) to be importable during model loading.
"""
import numpy as np
import pandas as pd
import pytest
from datetime import date


@pytest.mark.integration
class TestP01Pipeline:
    """Full reporting pipeline tests against p01_sample (panel LME)."""

    BRAND = "brand_1"

    @pytest.fixture
    def p01_setup(self, p01_model_ds, samples_path):
        """Build reporting config from model and dataset (function-scoped)."""
        import fermatrica_rep as fmr

        model, ds = p01_model_ds
        ds = ds.copy()

        root = samples_path / "p01_sample"
        opt_path = str(root / "code_py" / "model_data" / "options" / "options.xlsx")
        ssn_path = str(root / "code_py" / "model_data" / "options" / "seasonality.xlsx")

        trans_dict = fmr.trans_dict_create(path=opt_path, sheet="translation")
        options_m = fmr.budget_dict_create(path=opt_path, sheet="main", vat_rate=0.0)
        options_c = fmr.budget_dict_create(path=opt_path, sheet="current", vat_rate=0.0)
        model_rep = fmr.ModelRep(ds, trans_dict=trans_dict, language="english")
        ds = fmr.media_ssn_apply(ds, path=ssn_path, sheet="data")

        return {
            "model": model, "ds": ds, "model_rep": model_rep,
            "options_m": options_m, "options_c": options_c,
        }

    def test_option_report_zero_vs_predefined_budget(self, p01_setup):
        """Option with predefined (non-zero) budget yields higher predicted than zero budget."""
        import fermatrica_rep as fmr

        model = p01_setup["model"]
        ds = p01_setup["ds"]
        model_rep = p01_setup["model_rep"]
        options_m = p01_setup["options_m"]

        opt_set = fmr.OptionSettings(
            target=[[self.BRAND]],
            date_start=date(2025, 6, 30),
            date_end=date(2025, 12, 31),
            ref_date_start=date(2024, 1, 1),
            ref_date_end=date(2024, 12, 31),
            apply_vars=["superbrand"],
            plan_period="exact",
        )

        _, _, opt_sum_zero = fmr.option_report(
            model, ds, model_rep, options_m["zero"], opt_set, if_exact=False,
        )
        _, _, opt_sum_predefined = fmr.option_report(
            model, ds, model_rep, options_m["predefined"], opt_set, if_exact=False,
        )

        pred_total_zero = opt_sum_zero["pred_long_vol"]
        pred_total_predefined = opt_sum_predefined["pred_long_vol"]

        assert pred_total_predefined >= pred_total_zero, (
            f"Predefined budget should produce >= predicted KPI than zero budget "
            f"(zero={pred_total_zero:.2f}, predefined={pred_total_predefined:.2f})"
        )

    def test_extract_effect_multi_brand(self, p01_setup):
        """option_report produces predictions with future period (listed=4)."""
        import fermatrica_rep as fmr

        model = p01_setup["model"]
        ds = p01_setup["ds"]
        model_rep = p01_setup["model_rep"]
        options_c = p01_setup["options_c"]

        opt_set = fmr.OptionSettings(
            target=[[self.BRAND]],
            date_start=date(2025, 6, 30),
            date_end=date(2025, 12, 31),
            ref_date_start=date(2024, 1, 1),
            ref_date_end=date(2024, 12, 31),
            apply_vars=["superbrand"],
            plan_period="exact",
        )

        _, dt_pred, opt_sum = fmr.option_report(
            model, ds, model_rep, options_c["predefined"], opt_set, if_exact=False,
        )

        assert dt_pred is not None
        assert "predicted" in dt_pred.columns
        assert dt_pred["listed"].isin([4]).any(), "No future predictions in output"

        # opt_sum should contain long-term keys when if_exact=False
        assert "pred_long_vol" in opt_sum
        assert "pred_long_val" in opt_sum

    def test_extract_effect_multi_brand(self, p01_setup):
        """extract_effect produces decomposition covering all 3 brands."""
        import fermatrica_rep as fmr

        model = p01_setup["model"]
        ds = p01_setup["ds"]
        model_rep = p01_setup["model_rep"]

        split_m_m = fmr.extract_effect(model, ds, model_rep)

        assert split_m_m is not None
        assert len(split_m_m) > 0
        assert "variable" in split_m_m.columns
        assert "value" in split_m_m.columns
        brands = split_m_m["superbrand"].unique()
        assert len(brands) == 3, f"Expected 3 brands, got {brands}"

    def test_extract_effect_no_nan_inf(self, p01_setup):
        """extract_effect produces finite values in the value column."""
        import fermatrica_rep as fmr

        model = p01_setup["model"]
        ds = p01_setup["ds"]
        model_rep = p01_setup["model_rep"]

        split_m_m = fmr.extract_effect(model, ds, model_rep)

        assert np.all(np.isfinite(split_m_m["value"])), (
            "extract_effect produced NaN or Inf values"
        )

    def test_extract_effect_value_sum(self, p01_setup):
        """extract_effect decomposition values sum equals model prediction per period."""
        import fermatrica_rep as fmr

        model = p01_setup["model"]
        ds = p01_setup["ds"]
        model_rep = p01_setup["model_rep"]

        split_m_m = fmr.extract_effect(model, ds, model_rep)

        sum_per_period = split_m_m.groupby(["superbrand", "bs_key", "date"])["value"].sum().reset_index()
        sum_per_period.columns = ["superbrand", "bs_key", "date", "decomposed_total"]

        import fermatrica as fm
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
                    f"predicted={row['predicted']:.4f}, rel_diff={rel_diff:.4%}"
                )

    def test_waterfall_data_ratio(self, p01_setup):
        """waterfall_data ratio column sums to ~100% and cumsum is consistent."""
        import fermatrica_rep as fmr

        model = p01_setup["model"]
        ds = p01_setup["ds"]
        model_rep = p01_setup["model_rep"]

        split_m_m = fmr.extract_effect(model, ds, model_rep)
        wf = fmr.waterfall_data(
            split_m_m,
            brands=[self.BRAND],
            date_start="2024-01-01",
            date_end="2024-12-31",
        )

        assert len(wf) > 0
        assert abs(wf["ratio"].sum() - 100.0) < 0.1, (
            f"ratio column sums to {wf['ratio'].sum():.2f}%, expected ~100%"
        )
        assert abs(wf["value_cumsum"].iloc[-1] - wf["value"].sum()) < 1e-6

    def test_edge_case_nonexistent_brand(self, p01_setup):
        """option_report with nonexistent brand target should handle gracefully."""
        import fermatrica_rep as fmr

        model = p01_setup["model"]
        ds = p01_setup["ds"]
        model_rep = p01_setup["model_rep"]
        options_c = p01_setup["options_c"]

        opt_set = fmr.OptionSettings(
            target=[["nonexistent_brand"]],
            date_start=date(2025, 6, 30),
            date_end=date(2025, 12, 31),
            ref_date_start=date(2024, 1, 1),
            ref_date_end=date(2024, 12, 31),
            apply_vars=["superbrand"],
            plan_period="exact",
        )

        ds_out, dt_pred, opt_sum = fmr.option_report(
            model, ds, model_rep, options_c["predefined"], opt_set, if_exact=False,
        )

        assert dt_pred is not None

    def test_edge_case_future_only(self, p01_setup):
        """option_report with dates entirely in future: forecast rows are listed=4."""
        import fermatrica_rep as fmr

        model = p01_setup["model"]
        ds = p01_setup["ds"]
        model_rep = p01_setup["model_rep"]
        options_c = p01_setup["options_c"]

        opt_set = fmr.OptionSettings(
            target=[[self.BRAND]],
            date_start=date(2030, 1, 1),
            date_end=date(2030, 12, 31),
            ref_date_start=date(2024, 1, 1),
            ref_date_end=date(2024, 12, 31),
            apply_vars=["superbrand"],
            plan_period="exact",
        )

        _, dt_pred, _ = fmr.option_report(
            model, ds, model_rep, options_c["predefined"], opt_set, if_exact=False,
        )

        assert dt_pred is not None
        forecast = dt_pred[
            (dt_pred["date"] >= pd.Timestamp("2030-01-01")) &
            (dt_pred["date"] <= pd.Timestamp("2030-12-31"))
        ]
        assert len(forecast) > 0, "No rows in the forecast period"
        assert set(forecast["listed"].unique()).issubset({4}), (
            f"Forecast rows should be listed=4, got {forecast['listed'].unique()}"
        )


@pytest.mark.integration
class TestP01NewApi:
    """New-API (Option/OptionScope/ModelNode/MetaModel) tests against p01_sample.

    Regression coverage for the refactored reporting flow: per-scope option
    computation via MetaModel.compute and full-horizon seasonality on weekly data
    (monthly seasonality merged through the `period` column — the p01-specific trap).
    """

    BRAND = "brand_1"
    SCOPES = ["brand_1_2025", "brand_1_2026", "brand_1_2025_retro"]

    @pytest.fixture
    def p01_new_setup(self, p01_model_ds, samples_path):
        """Build a single-node MetaModel from context.xlsx (function-scoped)."""
        import fermatrica_rep as fmr
        import code_py.adhoc.translators

        model, ds = p01_model_ds

        root = samples_path / "p01_sample"
        opt_path = str(root / "code_py" / "model_data" / "options" / "context.xlsx")

        scopes = fmr.scopes_read(opt_path)
        options = fmr.Option.from_excel_all(opt_path)
        ssn_dict = fmr.seasonality_read(path=opt_path)
        trans_dict = fmr.trans_dict_create(path=opt_path, sheet="translation")

        node = fmr.ModelNode(model=model, ds=ds, trans_dict=trans_dict,
                             adhoc_code=[code_py.adhoc.translators],
                             language='english', ssn_dict=ssn_dict,
                             scope_names={s.scope_name for s in scopes})
        mm = fmr.MetaModel(nodes=node)

        return {"mm": mm, "scopes": scopes, "options": options, "node": node}

    def test_context_parse(self, p01_new_setup):
        """context.xlsx parses into 3 scopes and zero/predefined options."""
        scopes = p01_new_setup["scopes"]
        options = p01_new_setup["options"]

        assert len(scopes) == 3
        assert {s.scope_name for s in scopes} == set(self.SCOPES)
        assert set(options.keys()) == {"zero", "predefined"}

        for s in scopes:
            assert s.filters["superbrand"] == [self.BRAND]
            items = options["predefined"].budget_for_scope(s.scope_name)
            assert items, f"no budget items for {s.scope_name}"

    def test_meta_model_compute_per_scope(self, p01_new_setup):
        """MetaModel.compute yields per-scope pred_exact_vol > 0 and finite."""
        mm = p01_new_setup["mm"]
        scopes = p01_new_setup["scopes"]
        options = p01_new_setup["options"]

        result = mm.compute(options["predefined"], scopes=scopes, allow_past=True)["model"]

        for s in scopes:
            summary = result[s.scope_name]
            assert summary["pred_exact_vol"] > 0, (
                f"{s.scope_name} pred_exact_vol must be > 0, got {summary['pred_exact_vol']}"
            )
            assert np.isfinite(summary["pred_exact_vol"])
            assert np.isfinite(summary["pred_long_vol"])

    def test_seasonality_covers_horizon(self, p01_new_setup):
        """No NaN opt_media_ssn for listed==4 rows (monthly seasonality on weekly data)."""
        mm = p01_new_setup["mm"]
        scopes = p01_new_setup["scopes"]
        options = p01_new_setup["options"]

        mm.compute(options["predefined"], scopes=scopes, allow_past=True)
        node = mm.nodes["model"]

        ssn_cols = [c for c in node.ds.columns if c.startswith("opt_media_ssn")]
        assert ssn_cols, "no opt_media_ssn columns after compute"

        future = node.ds[node.ds["listed"] == 4]
        assert future[ssn_cols].isna().sum().sum() == 0, (
            "NaN media seasonality inside the forecast horizon"
        )

        beyond_plan_year = future[future["date"].dt.year > 2026]
        assert len(beyond_plan_year) > 0
        assert beyond_plan_year[ssn_cols].isna().sum().sum() == 0, (
            "NaN media seasonality beyond the plan year"
        )

        # historical years before the plan horizon must stay NaN
        # (no year-invariance stamping over history)
        historical = node.ds[node.ds["listed"].isin([1, 2])]
        assert historical[ssn_cols].isna().all().all(), (
            "historical years must keep NaN seasonality"
        )
