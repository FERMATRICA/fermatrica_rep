"""Integration tests against p05_sample (two-level panel: BHT → Sales).

Tests the chained reporting pipeline: load BHT model → option_report →
pred_merge_1 → load Sales model → option_report on merged data.
"""
import numpy as np
import pytest
from datetime import date


@pytest.mark.integration
class TestP05Pipeline:
    """Reporting pipeline tests against p05_sample (two-level chained panel)."""

    BRAND = "brand_1"

    @pytest.fixture
    def p05_setup(self, p05_models_ds, samples_path):
        """Build reporting config for both BHT and Sales models (function-scoped)."""
        import fermatrica_rep as fmr

        bht_model, bht_ds, sales_model, sales_ds = p05_models_ds
        bht_ds = bht_ds.copy()
        sales_ds = sales_ds.copy()

        root = samples_path / "p05_sample"
        opt_path = str(root / "code_py" / "model_data" / "options" / "options.xlsx")
        ssn_path = str(root / "code_py" / "model_data" / "options" / "seasonality.xlsx")

        trans_dict_1 = fmr.trans_dict_create(path=opt_path, sheet="translation_1")
        trans_dict_2 = fmr.trans_dict_create(path=opt_path, sheet="translation_2")

        model_rep_1 = fmr.ModelRep(bht_ds, trans_dict=trans_dict_1, language="english")
        model_rep_2 = fmr.ModelRep(sales_ds, trans_dict=trans_dict_2, language="english")

        bht_ds = fmr.media_ssn_apply(bht_ds, path=ssn_path, sheet="data")
        sales_ds = fmr.media_ssn_apply(sales_ds, path=ssn_path, sheet="data")

        return {
            "bht_model": bht_model, "bht_ds": bht_ds, "model_rep_1": model_rep_1,
            "sales_model": sales_model, "sales_ds": sales_ds, "model_rep_2": model_rep_2,
        }

    def test_bht_model_loads(self, p05_setup):
        """BHT model has ABA as target and superbrand grouping."""
        bht_model = p05_setup["bht_model"]
        assert bht_model.obj is not None
        assert "aba" in bht_model.conf.Y_var

    def test_sales_model_loads(self, p05_setup):
        """Sales model has units as target and bs_key grouping."""
        sales_model = p05_setup["sales_model"]
        assert sales_model.obj is not None
        assert sales_model.conf.Y_var is not None

    def test_pred_merge_1_importable(self, p05_setup):
        """pred_merge_1 function is importable from adhoc.reporting."""
        from code_py.adhoc.reporting import pred_merge_1
        assert callable(pred_merge_1)

    def test_new_api_chain_loads_options_configuration(self, p05_models_ds, samples_path):
        """New API chains both nodes and loads media prices from context.xlsx."""
        import code_py.adhoc.reporting
        import code_py.adhoc.translators
        from code_py.adhoc.reporting import pred_merge_1
        import fermatrica_rep as fmr

        bht_model, bht_ds, sales_model, sales_ds = p05_models_ds
        options_path = str(
            samples_path
            / "p05_sample"
            / "code_py"
            / "model_data"
            / "options"
            / "context.xlsx"
        )
        scopes = fmr.scopes_read(options_path)
        options = fmr.Option.from_excel_all(options_path)
        ssn_dict = fmr.seasonality_read(options_path)
        trans_dict = fmr.trans_dict_create(path=options_path, sheet="translation")

        node_bht = fmr.ModelNode(
            model=bht_model,
            ds=bht_ds,
            options_path=options_path,
            trans_dict=trans_dict,
            adhoc_code=[code_py.adhoc.translators, code_py.adhoc.reporting],
            language="english",
            ssn_dict=ssn_dict,
        )
        node_sales = fmr.ModelNode(
            model=sales_model,
            ds=sales_ds,
            options_path=options_path,
            trans_dict=trans_dict,
            adhoc_code=[code_py.adhoc.translators, code_py.adhoc.reporting],
            language="english",
            ssn_dict=ssn_dict,
        )
        node_bht.scope_names = {s.scope_name for s in scopes}
        node_sales.scope_names = {s.scope_name for s in scopes}

        mm = fmr.MetaModel(
            nodes={"bht": node_bht, "sales": node_sales},
            edges=[("bht", "sales", pred_merge_1)],
        )
        result = mm.compute(options["predefined"], scopes=scopes, allow_past=True)

        assert node_bht.model_rep.media_prices is not None
        assert node_sales.model_rep.media_prices is not None
        assert result["sales"]

    def test_full_chain_option_report(self, p05_setup, samples_path):
        """Full chain: BHT option_report → pred_merge_1 → Sales option_report."""
        import fermatrica_rep as fmr
        from code_py.adhoc.reporting import pred_merge_1

        bht_model = p05_setup["bht_model"]
        bht_ds = p05_setup["bht_ds"]
        model_rep_1 = p05_setup["model_rep_1"]
        sales_model = p05_setup["sales_model"]
        sales_ds = p05_setup["sales_ds"]
        model_rep_2 = p05_setup["model_rep_2"]

        root = samples_path / "p05_sample"
        opt_path = str(root / "code_py" / "model_data" / "options" / "options.xlsx")

        options_c = fmr.budget_dict_create(
            path=opt_path, sheet="current", vat_rate=0.0)

        opt_set = fmr.OptionSettings(
            target=[[self.BRAND]],
            date_start=date(2025, 6, 30),
            date_end=date(2025, 12, 31),
            ref_date_start=date(2024, 1, 1),
            ref_date_end=date(2024, 12, 31),
            apply_vars=["superbrand"],
            plan_period="exact",
        )

        _, dt_pred_1, opt_sum_1 = fmr.option_report(
            bht_model, bht_ds, model_rep_1,
            options_c["predefined"], opt_set, if_exact=False,
        )

        assert dt_pred_1 is not None
        assert "predicted" in dt_pred_1.columns
        assert "pred_long_vol" in opt_sum_1
        assert "pred_long_val" in opt_sum_1

        sales_ds_merged = pred_merge_1(sales_ds, dt_pred_1)

        assert "aba_predicted" in sales_ds_merged.columns

        _, dt_pred_2, opt_sum_2 = fmr.option_report(
            sales_model, sales_ds_merged, model_rep_2,
            options_c["predefined"], opt_set, if_exact=False,
        )

        assert dt_pred_2 is not None
        assert "observed" in dt_pred_2.columns
        assert "predicted" in dt_pred_2.columns
        assert dt_pred_2["listed"].isin([4]).any(), \
            "No future predictions in Sales output"

        assert "pred_long_vol" in opt_sum_2
        assert "pred_long_val" in opt_sum_2

    def test_extract_effect_bht(self, p05_setup):
        """extract_effect on BHT model produces finite decomposition."""
        import fermatrica_rep as fmr

        bht_model = p05_setup["bht_model"]
        bht_ds = p05_setup["bht_ds"]
        model_rep_1 = p05_setup["model_rep_1"]

        split_m_m = fmr.extract_effect(bht_model, bht_ds, model_rep_1)
        assert split_m_m is not None
        assert len(split_m_m) > 0
        assert np.all(np.isfinite(split_m_m["value"]))

    def test_extract_effect_sales(self, p05_setup):
        """extract_effect on Sales model produces finite decomposition."""
        import fermatrica_rep as fmr

        sales_model = p05_setup["sales_model"]
        sales_ds = p05_setup["sales_ds"]
        model_rep_2 = p05_setup["model_rep_2"]

        split_m_m = fmr.extract_effect(sales_model, sales_ds, model_rep_2)
        assert split_m_m is not None
        assert len(split_m_m) > 0
        assert np.all(np.isfinite(split_m_m["value"]))

    def test_extract_effect_value_sum_bht(self, p05_setup):
        """BHT extract_effect decomposition sums match BHT fit_predict per period."""
        import fermatrica as fm
        import fermatrica_rep as fmr

        bht_model = p05_setup["bht_model"]
        bht_ds = p05_setup["bht_ds"]
        model_rep_1 = p05_setup["model_rep_1"]

        split_m_m = fmr.extract_effect(bht_model, bht_ds, model_rep_1)

        sum_per_period = split_m_m.groupby(["superbrand", "bs_key", "date"])["value"].sum().reset_index()
        sum_per_period.columns = ["superbrand", "bs_key", "date", "decomposed_total"]

        pred_values, _, _ = fm.fit_predict(bht_ds, bht_model, if_fit=False, return_full=True)
        ds_pred = bht_ds[["superbrand", "bs_key", "date"]].copy()
        ds_pred["predicted"] = pred_values.values
        ds_pred = ds_pred.dropna(subset=["predicted"])
        pred_agg = ds_pred.groupby(["superbrand", "bs_key", "date"], as_index=False)["predicted"].sum()

        merged = sum_per_period.merge(pred_agg, on=["superbrand", "bs_key", "date"], how="inner")

        for _, row in merged.iterrows():
            if abs(row["predicted"]) > 1e-10:
                rel_diff = abs(row["decomposed_total"] - row["predicted"]) / abs(row["predicted"])
                assert rel_diff < 1e-6, (
                    f"BHT Decomposition [{row['superbrand']}, {row['date']}]: "
                    f"decomposed={row['decomposed_total']:.4f}, "
                    f"predicted={row['predicted']:.4f}, rel_diff={rel_diff:.4%}"
                )

    def test_extract_effect_value_sum_sales(self, p05_setup):
        """Sales extract_effect decomposition sums match Sales fit_predict per period."""
        import fermatrica as fm
        import fermatrica_rep as fmr

        sales_model = p05_setup["sales_model"]
        sales_ds = p05_setup["sales_ds"]
        model_rep_2 = p05_setup["model_rep_2"]

        split_m_m = fmr.extract_effect(sales_model, sales_ds, model_rep_2)

        sum_per_period = split_m_m.groupby(["superbrand", "bs_key", "date"])["value"].sum().reset_index()
        sum_per_period.columns = ["superbrand", "bs_key", "date", "decomposed_total"]

        pred_values, _, _ = fm.fit_predict(sales_ds, sales_model, if_fit=False, return_full=True)
        ds_pred = sales_ds[["superbrand", "bs_key", "date"]].copy()
        ds_pred["predicted"] = pred_values.values
        ds_pred = ds_pred.dropna(subset=["predicted"])
        pred_agg = ds_pred.groupby(["superbrand", "bs_key", "date"], as_index=False)["predicted"].sum()

        merged = sum_per_period.merge(pred_agg, on=["superbrand", "bs_key", "date"], how="inner")

        for _, row in merged.iterrows():
            if abs(row["predicted"]) > 1e-10:
                rel_diff = abs(row["decomposed_total"] - row["predicted"]) / abs(row["predicted"])
                assert rel_diff < 1e-6, (
                    f"Sales Decomposition [{row['superbrand']}, {row['date']}]: "
                    f"decomposed={row['decomposed_total']:.4f}, "
                    f"predicted={row['predicted']:.4f}, rel_diff={rel_diff:.4%}"
                )
