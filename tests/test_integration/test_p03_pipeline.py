"""Integration tests against p03_sample (ABA target, monthly, LME, 3 brands).

Tests the reporting pipeline on a brand awareness (ABA) model with monthly
frequency and BHT multiplier transformation.
"""
import numpy as np
import pandas as pd
import pytest
from datetime import date


@pytest.mark.integration
class TestP03Pipeline:
    """Reporting pipeline tests against p03_sample (ABA monthly panel)."""

    BRAND = "brand_1"

    @pytest.fixture
    def p03_setup(self, p03_model_ds, samples_path):
        """Build reporting config from model and dataset (function-scoped)."""
        import fermatrica_rep as fmr

        model, ds = p03_model_ds
        ds = ds.copy()

        root = samples_path / "p03_sample"
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

    def test_model_loads(self, p03_setup):
        """Model object is not None and target is ABA."""
        model = p03_setup["model"]
        assert model.obj is not None
        assert model.conf is not None
        assert "aba" in model.conf.Y_var

    def test_dataset_has_three_brands(self, p03_setup):
        """Dataset contains all 3 brands."""
        ds = p03_setup["ds"]
        brands = set(ds["superbrand"].unique())
        expected = {"brand_1", "brand_2", "brand_3"}
        missing = expected - brands
        assert not missing, f"Missing brands: {missing}"

    def test_option_report_target_is_aba(self, p03_setup):
        """option_report produces predictions for ABA target."""
        import fermatrica_rep as fmr

        model = p03_setup["model"]
        ds = p03_setup["ds"]
        model_rep = p03_setup["model_rep"]
        options_m = p03_setup["options_m"]

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
            model, ds, model_rep, options_m["predefined"], opt_set, if_exact=False,
        )

        assert dt_pred is not None
        assert "observed" in dt_pred.columns
        assert "predicted" in dt_pred.columns
        assert dt_pred["listed"].isin([4]).any(), "No future predictions in output"

        assert "pred_long_vol" in opt_sum
        assert "pred_long_val" in opt_sum

    def test_option_report_zero_vs_predefined_budget(self, p03_setup):
        """Option with predefined (non-zero) budget yields higher predicted than zero."""
        import fermatrica_rep as fmr

        model = p03_setup["model"]
        ds = p03_setup["ds"]
        model_rep = p03_setup["model_rep"]
        options_m = p03_setup["options_m"]

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

    def test_extract_effect_monthly(self, p03_setup):
        """extract_effect produces decomposition for ABA model."""
        import fermatrica_rep as fmr

        model = p03_setup["model"]
        ds = p03_setup["ds"]
        model_rep = p03_setup["model_rep"]

        split_m_m = fmr.extract_effect(model, ds, model_rep)

        assert split_m_m is not None
        assert len(split_m_m) > 0
        assert "variable" in split_m_m.columns
        assert "value" in split_m_m.columns
        brands = split_m_m["superbrand"].unique()
        assert len(brands) == 3, f"Expected 3 brands, got {brands}"

    def test_extract_effect_no_nan_inf(self, p03_setup):
        """extract_effect produces finite values in the value column."""
        import fermatrica_rep as fmr

        model = p03_setup["model"]
        ds = p03_setup["ds"]
        model_rep = p03_setup["model_rep"]

        split_m_m = fmr.extract_effect(model, ds, model_rep)

        assert np.all(np.isfinite(split_m_m["value"])), (
            "extract_effect produced NaN or Inf values"
        )

    def test_extract_effect_value_sum(self, p03_setup):
        """extract_effect decomposition values sum equals model prediction per period."""
        import fermatrica_rep as fmr

        model = p03_setup["model"]
        ds = p03_setup["ds"]
        model_rep = p03_setup["model_rep"]

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

    def test_waterfall_data_ratio(self, p03_setup):
        """waterfall_data ratio column sums to ~100% and cumsum is consistent."""
        import fermatrica_rep as fmr

        model = p03_setup["model"]
        ds = p03_setup["ds"]
        model_rep = p03_setup["model_rep"]

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

    def test_edge_case_future_only(self, p03_setup):
        """option_report with dates entirely in future: forecast rows are listed=4."""
        import fermatrica_rep as fmr

        model = p03_setup["model"]
        ds = p03_setup["ds"]
        model_rep = p03_setup["model_rep"]
        options_m = p03_setup["options_m"]

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
            model, ds, model_rep, options_m["predefined"], opt_set, if_exact=False,
        )

        assert dt_pred is not None
        forecast = dt_pred[
            (dt_pred["date"] >= pd.Timestamp("2030-01-01")) &
            (dt_pred["date"] <= pd.Timestamp("2030-12-31"))
        ]
        assert len(forecast) > 0, "No rows in the forecast period"
        assert set(forecast["listed"].unique()).issubset({4})

    def test_edge_case_nonexistent_brand(self, p03_setup):
        """option_report with nonexistent brand should handle gracefully."""
        import fermatrica_rep as fmr

        model = p03_setup["model"]
        ds = p03_setup["ds"]
        model_rep = p03_setup["model_rep"]
        options_m = p03_setup["options_m"]

        opt_set = fmr.OptionSettings(
            target=[["nonexistent_brand"]],
            date_start=date(2025, 6, 30),
            date_end=date(2025, 12, 31),
            ref_date_start=date(2024, 1, 1),
            ref_date_end=date(2024, 12, 31),
            apply_vars=["superbrand"],
            plan_period="exact",
        )

        _, dt_pred, _ = fmr.option_report(
            model, ds, model_rep, options_m["predefined"], opt_set, if_exact=False,
        )
        assert dt_pred is not None


@pytest.mark.integration
class TestP03NewApi:
    """New-API (Option/OptionScope/ModelNode/MetaModel) tests against p03_sample.

    Regression coverage for the refactored reporting flow on the monthly ABA
    model: per-scope option computation via MetaModel.compute and full-horizon
    media seasonality on future (listed==4) rows.
    """

    SCOPE_NAMES = ["brand_1_2025", "brand_1_2026", "brand_1_2025_retro"]

    @pytest.fixture
    def p03_new_setup(self, p03_model_ds, samples_path):
        """Build a single-node MetaModel from context.xlsx (function-scoped)."""
        import fermatrica_rep as fmr
        import code_py.adhoc.translators

        model, ds = p03_model_ds

        root = samples_path / "p03_sample"
        opt_path = str(root / "code_py" / "model_data" / "options" / "context.xlsx")

        scopes = fmr.scopes_read(opt_path)
        options = fmr.Option.from_excel_all(opt_path)
        ssn_dict = fmr.seasonality_read(path=opt_path)
        trans_dict = fmr.trans_dict_create(path=opt_path, sheet="translation")

        node = fmr.ModelNode(model=model, ds=ds, trans_dict=trans_dict,
                             adhoc_code=[code_py.adhoc.translators],
                             language='english', ssn_dict=ssn_dict)
        mm = fmr.MetaModel(nodes=node)

        return {"mm": mm, "scopes": scopes, "options": options, "node": node}

    def test_context_parse(self, p03_new_setup):
        """context.xlsx parses into 3 scopes and zero/predefined options."""
        scopes = p03_new_setup["scopes"]
        options = p03_new_setup["options"]

        assert len(scopes) == 3
        assert {s.scope_name for s in scopes} == set(self.SCOPE_NAMES)
        assert set(options.keys()) == {"zero", "predefined"}

    def test_meta_model_compute_per_scope(self, p03_new_setup):
        """MetaModel.compute yields per-scope pred_exact_vol > 0 and finite."""
        mm = p03_new_setup["mm"]
        scopes = p03_new_setup["scopes"]
        options = p03_new_setup["options"]

        result = mm.compute(options["predefined"], scopes=scopes, allow_past=True)["model"]

        for scope_name in self.SCOPE_NAMES:
            summary = result[scope_name]
            assert summary["pred_exact_vol"] > 0, (
                f"{scope_name} pred_exact_vol must be > 0, got {summary['pred_exact_vol']}"
            )
            assert np.isfinite(summary["pred_exact_vol"])
            assert np.isfinite(summary["pred_long_vol"])

    def test_no_nan_seasonality_future(self, p03_new_setup):
        """No NaN opt_media_ssn for listed==4 rows across the forecast horizon."""
        mm = p03_new_setup["mm"]
        scopes = p03_new_setup["scopes"]
        options = p03_new_setup["options"]

        mm.compute(options["predefined"], scopes=scopes, allow_past=True)
        node = mm.nodes["model"]

        ssn_cols = [c for c in node.ds.columns if c.startswith("opt_media_ssn")]
        assert ssn_cols, "no opt_media_ssn columns after compute"

        future = node.ds[node.ds["listed"] == 4]
        assert len(future) > 0
        assert future[ssn_cols].isna().sum().sum() == 0, (
            "NaN media seasonality inside the forecast horizon"
        )

    def test_compute_grid_runs(self, p03_new_setup):
        """compute_grid produces a finite, positive curve for brand_1_2026."""
        import fermatrica_rep as fmr

        mm = p03_new_setup["mm"]
        scopes = p03_new_setup["scopes"]

        scopes_crv = [s for s in scopes if s.scope_name == "brand_1_2026"]
        curves_full_data = fmr.compute_grid(
            mm, scopes_crv, budget_step=25, bdg_max=100,
            cores=1, if_exact=True, allow_past=True, background_option=None,
        )
        df = curves_full_data["model"]
        assert len(df) > 0
        assert df["pred_exact_vol"].notna().all()
        assert (df["pred_exact_vol"] > 0).all()
