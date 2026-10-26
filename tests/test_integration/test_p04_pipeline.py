"""Integration tests against p04_sample (panel per region, LME, 4 regions).

Tests the reporting pipeline on a geographic panel model with regional
media prices and per-master budget loop.
"""
import numpy as np
import pandas as pd
import pytest
from datetime import date

from tests.test_integration.conftest import (
    _activate_adhoc_path,
    _deactivate_adhoc_path,
)


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


@pytest.mark.integration
class TestP04NewApi:
    """New-API (Option/OptionScope/ModelNode/MetaModel) tests against p04_sample.

    Regression coverage for the refactored reporting flow: capitalized scope
    targets matching the data `master` column, full-horizon seasonality, and
    per-region option computation via MetaModel.compute / compute_grid.
    """

    BRAND = "brand_1"
    MASTERS = ["Moscow", "Paris", "Warsaw", "Cairo"]

    @pytest.fixture
    def p04_new_setup(self, p04_model_ds, samples_path):
        """Build a single-node MetaModel from context.xlsx (function-scoped)."""
        import fermatrica_rep as fmr

        model, ds = p04_model_ds

        root = samples_path / "p04_sample"
        _activate_adhoc_path(root)
        try:
            import code_py.adhoc.reporting
            import code_py.adhoc.translators
            from code_py.adhoc.reporting import pre_reporting_new

            opt_path = str(root / "code_py" / "model_data" / "options" / "context.xlsx")

            scopes = fmr.scopes_read(opt_path)
            options = fmr.Option.from_excel_all(opt_path)
            ssn_dict = fmr.seasonality_read(path=opt_path)
            media_prices = fmr.media_prices_read(path=opt_path)
            trans_dict = fmr.trans_dict_create(path=opt_path, sheet="translation")

            node = fmr.ModelNode(model=model, ds=ds, trans_dict=trans_dict,
                                 adhoc_code=[code_py.adhoc.translators, code_py.adhoc.reporting],
                                 language='english', ssn_dict=ssn_dict,
                                 scope_names={s.scope_name for s in scopes})
            pre_reporting_new(node)
            node.model_rep.media_prices = media_prices
            mm = fmr.MetaModel(nodes=node)

            return {"mm": mm, "scopes": scopes, "options": options, "node": node}
        finally:
            _deactivate_adhoc_path(root)

    def test_context_parse_capitalized_targets(self, p04_new_setup):
        """scopes/allocations parse with capitalized targets (case-bug regression)."""
        scopes = p04_new_setup["scopes"]
        options = p04_new_setup["options"]

        assert len(scopes) == 8
        assert set(options.keys()) == {"zero", "predefined"}

        for r in self.MASTERS:
            s = [x for x in scopes if x.scope_name == f"brand_1_{r.lower()}_2026"]
            assert len(s) == 1
            assert s[0].filters["master"] == [r], (
                f"scope target for {r} must be capitalized, got {s[0].filters['master']}"
            )
            items = options["predefined"].budget_for_scope(s[0].scope_name)
            assert items, f"no budget items for {s[0].scope_name}"
            assert "nobdg_price" in items
            assert "nobdg_price_index" in items

    def test_meta_model_compute_per_region(self, p04_new_setup):
        """MetaModel.compute yields per-region pred_exact_vol > 0 (case-bug regression)."""
        mm = p04_new_setup["mm"]
        scopes = p04_new_setup["scopes"]
        options = p04_new_setup["options"]

        result = mm.compute(options["predefined"], scopes=scopes, allow_past=True)["model"]

        for r in self.MASTERS:
            scope_name = f"brand_1_{r.lower()}_2026"
            summary = result[scope_name]
            assert summary["pred_exact_vol"] > 0, (
                f"{r} pred_exact_vol must be > 0, got {summary['pred_exact_vol']}"
            )
            assert np.isfinite(summary["pred_exact_vol"])
            assert np.isfinite(summary["pred_long_vol"])

    def test_seasonality_covers_horizon(self, p04_new_setup):
        """No NaN opt_media_ssn for listed==4 rows beyond the plan year."""
        mm = p04_new_setup["mm"]
        scopes = p04_new_setup["scopes"]
        options = p04_new_setup["options"]

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

        # future-horizon expansion: historical years before the scope start
        # must stay NaN (no year-invariance stamping over history)
        historical = node.ds[node.ds["listed"].isin([1, 2])]
        assert historical[ssn_cols].isna().all().all(), (
            "historical years must keep NaN seasonality"
        )

    def test_compute_grid_runs(self, p04_new_setup):
        """compute_grid produces a finite, positive curve for one region."""
        import fermatrica_rep as fmr

        mm = p04_new_setup["mm"]
        scopes = p04_new_setup["scopes"]
        options = p04_new_setup["options"]

        scopes_crv = [s for s in scopes if s.scope_name == "brand_1_moscow_2026"]
        curves_full_data = fmr.compute_grid(
            mm, scopes_crv, budget_step=25, bdg_max=100,
            cores=1, if_exact=True, allow_past=True, background_option=None,
        )
        df = curves_full_data["model"]
        assert len(df) > 0
        assert df["pred_exact_vol"].notna().all()
        assert (df["pred_exact_vol"] > 0).all()
