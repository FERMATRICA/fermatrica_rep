"""Integration tests against p04_sample (panel per region, LME, 4 regions).

Tests the reporting pipeline on a geographic panel model with regional
media prices and per-master budget loop.
"""
import numpy as np
import pandas as pd
import pytest
from datetime import date


@pytest.mark.integration
class TestP04Pipeline:
    """Reporting pipeline tests against p04_sample (panel per region)."""

    BRAND = "brand_1"
    MASTERS = ["Moscow", "Paris", "Warsaw", "Cairo"]

    @pytest.fixture
    def p04_setup(self, p04_model_ds, samples_path):
        """Build reporting config with per-master budgets and media prices."""
        import pandas as pd
        import fermatrica_rep as fmr

        model, ds = p04_model_ds
        ds = ds.copy()

        root = samples_path / "p04_sample"
        opt_path = str(root / "code_py" / "model_data" / "options" / "options.xlsx")
        ssn_path = str(root / "code_py" / "model_data" / "options" / "seasonality.xlsx")

        trans_dict = fmr.trans_dict_create(path=opt_path, sheet="translation")
        model_rep = fmr.ModelRep(ds, trans_dict=trans_dict, language="english")
        ds = fmr.media_ssn_apply(ds, path=ssn_path, sheet="data")

        options_m_raw = pd.read_excel(opt_path, sheet_name="main")
        options_c_raw = pd.read_excel(opt_path, sheet_name="current")

        options_m = {}
        options_c = {}
        for ms in options_m_raw["text_flags"].unique():
            if pd.isna(ms):
                continue
            options_m[ms] = fmr.budget_dict_create(
                dct=options_m_raw[options_m_raw["text_flags"] == ms])
        for ms in options_c_raw["text_flags"].unique():
            if pd.isna(ms):
                continue
            options_c[ms] = fmr.budget_dict_create(
                dct=options_c_raw[options_c_raw["text_flags"] == ms])

        media_prices = pd.read_excel(opt_path, sheet_name="media_prices")
        media_prices = pd.melt(
            media_prices, id_vars="master", var_name="variable", value_name="price")

        return {
            "model": model, "ds": ds, "model_rep": model_rep,
            "options_m": options_m, "options_c": options_c,
            "media_prices": media_prices,
        }

    def test_model_loads(self, p04_setup):
        """Model object is not None and groups by master."""
        model = p04_setup["model"]
        assert model.obj is not None
        assert model.conf is not None
        assert model.conf.Y_var is not None

    def test_dataset_has_four_regions(self, p04_setup):
        """Dataset contains all 4 geographic regions."""
        ds = p04_setup["ds"]
        regions = set(ds["master"].unique())
        expected = set(self.MASTERS)
        missing = expected - regions
        assert not missing, f"Missing regions: {missing}"

    def test_option_report_per_master(self, p04_setup):
        """option_report per region with media_prices trans_path_df."""
        import fermatrica_rep as fmr

        model = p04_setup["model"]
        ds = p04_setup["ds"]
        model_rep = p04_setup["model_rep"]
        options_c = p04_setup["options_c"]
        media_prices = p04_setup["media_prices"]

        for ms in self.MASTERS:
            model.conf.trans_path_df = media_prices[media_prices["master"] == ms]

            opt_set = fmr.OptionSettings(
                target=[[self.BRAND], [ms]],
                date_start=date(2025, 6, 30),
                date_end=date(2025, 12, 31),
                ref_date_start=date(2024, 1, 1),
                ref_date_end=date(2024, 12, 31),
                apply_vars=["superbrand", "master"],
                plan_period="exact",
            )

            _, dt_pred, opt_sum = fmr.option_report(
                model, ds, model_rep,
                options_c[ms]["predefined"], opt_set, if_exact=False,
            )

            assert dt_pred is not None, f"No prediction for {ms}"
            assert "observed" in dt_pred.columns
            assert "predicted" in dt_pred.columns
            assert dt_pred["listed"].isin([4]).any(), f"No future predictions for {ms}"

            assert "pred_long_vol" in opt_sum, f"Missing pred_long_vol for {ms}"
            assert "pred_long_val" in opt_sum, f"Missing pred_long_val for {ms}"

    def test_extract_effect_regional(self, p04_setup):
        """extract_effect produces decomposition with master column."""
        import fermatrica_rep as fmr

        model = p04_setup["model"]
        ds = p04_setup["ds"]
        model_rep = p04_setup["model_rep"]

        split_m_m = fmr.extract_effect(model, ds, model_rep)

        assert split_m_m is not None
        assert len(split_m_m) > 0
        assert "variable" in split_m_m.columns
        assert "value" in split_m_m.columns
        assert "master" in split_m_m.columns
        regions_in_result = set(split_m_m["master"].unique())
        expected_regions = set(self.MASTERS)
        missing = expected_regions - regions_in_result
        assert not missing, f"Missing regions in decomposition: {missing}"

    def test_extract_effect_no_nan_inf(self, p04_setup):
        """extract_effect produces finite values."""
        import fermatrica_rep as fmr

        model = p04_setup["model"]
        ds = p04_setup["ds"]
        model_rep = p04_setup["model_rep"]

        split_m_m = fmr.extract_effect(model, ds, model_rep)
        assert np.all(np.isfinite(split_m_m["value"]))

    def test_extract_effect_value_sum(self, p04_setup):
        """extract_effect decomposition values sum equals model prediction per period."""
        import fermatrica_rep as fmr

        model = p04_setup["model"]
        ds = p04_setup["ds"]
        model_rep = p04_setup["model_rep"]

        split_m_m = fmr.extract_effect(model, ds, model_rep)

        sum_per_period = split_m_m.groupby(["superbrand", "bs_key", "date", "master"])["value"].sum().reset_index()
        sum_per_period.columns = ["superbrand", "bs_key", "date", "master", "decomposed_total"]

        import fermatrica as fm
        pred_values, _, _ = fm.fit_predict(ds, model, if_fit=False, return_full=True)
        ds_pred = ds[["superbrand", "bs_key", "date", "master"]].copy()
        ds_pred["predicted"] = pred_values.values
        ds_pred = ds_pred.dropna(subset=["predicted"])
        pred_agg = ds_pred.groupby(["superbrand", "bs_key", "date", "master"], as_index=False)["predicted"].sum()

        merged = sum_per_period.merge(pred_agg, on=["superbrand", "bs_key", "date", "master"], how="inner")

        for _, row in merged.iterrows():
            if abs(row["predicted"]) > 1e-10:
                rel_diff = abs(row["decomposed_total"] - row["predicted"]) / abs(row["predicted"])
                assert rel_diff < 1e-6, (
                    f"Decomp [{row['superbrand']}, {row['master']}, {row['date']}]: "
                    f"decomposed={row['decomposed_total']:.4f}, "
                    f"predicted={row['predicted']:.4f}, rel_diff={rel_diff:.4%}"
                )

    def test_edge_case_future_only(self, p04_setup):
        """option_report with far-future dates: forecast rows are listed=4."""
        import fermatrica_rep as fmr

        model = p04_setup["model"]
        ds = p04_setup["ds"]
        model_rep = p04_setup["model_rep"]
        options_c = p04_setup["options_c"]
        media_prices = p04_setup["media_prices"]

        ms = self.MASTERS[0]
        model.conf.trans_path_df = media_prices[media_prices["master"] == ms]

        opt_set = fmr.OptionSettings(
            target=[[self.BRAND], [ms]],
            date_start=date(2030, 1, 1),
            date_end=date(2030, 12, 31),
            ref_date_start=date(2024, 1, 1),
            ref_date_end=date(2024, 12, 31),
            apply_vars=["superbrand", "master"],
            plan_period="exact",
        )

        _, dt_pred, _ = fmr.option_report(
            model, ds, model_rep, options_c[ms]["predefined"], opt_set, if_exact=False,
        )

        assert dt_pred is not None
        forecast = dt_pred[
            (dt_pred["date"] >= pd.Timestamp("2030-01-01")) &
            (dt_pred["date"] <= pd.Timestamp("2030-12-31"))
        ]
        assert len(forecast) > 0, "No rows in the forecast period"
        assert set(forecast["listed"].unique()).issubset({4})
