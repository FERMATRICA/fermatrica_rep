"""Integration tests for curves_full.py: full efficiency curves pipeline.

Tests that model-blind efficiency curves produce valid plotly figures
via option_report_multi_post → curves_full_plot_short/curves_full_plot_long.
Heavier than short-term curves: generates a grid of budget scenarios.
"""
import numpy as np
import pytest
from datetime import date


@pytest.mark.integration
class TestP00CurvesFull:
    """Full curve tests against p00_sample (OLS, 1 brand)."""

    BRAND = "brand_x"

    @pytest.fixture
    def setup(self, p00_model_ds, samples_path):
        """Build ModelRep and opt_set for curves_full calculation."""
        import fermatrica_rep as fmr

        model, ds = p00_model_ds
        ds = ds.copy()

        root = samples_path / "p00_sample"
        opt_path = str(root / "code_py" / "model_data" / "options" / "options.xlsx")
        ssn_path = str(root / "code_py" / "model_data" / "options" / "seasonality.xlsx")

        trans_dict = fmr.trans_dict_create(path=opt_path, sheet="translation")
        model_rep = fmr.ModelRep(ds, trans_dict=trans_dict, language="english")
        ds = fmr.media_ssn_apply(ds, path=ssn_path, sheet="data")

        opt_set = fmr.OptionSettings(
            target=[[self.BRAND]],
            date_start=date(2018, 1, 1),
            date_end=date(2018, 12, 31),
            ref_date_start=date(2017, 1, 1),
            ref_date_end=date(2017, 12, 31),
            apply_vars=["superbrand"],
            plan_period="exact",
        )

        return {"model": model, "ds": ds, "model_rep": model_rep, "opt_set": opt_set}

    def test_curves_full_pipeline(self, setup):
        """Full curves pipeline produces dict of plotly figures."""
        import plotly.graph_objects as go
        from fermatrica_rep.options.calc_multi import option_report_multi_post
        from fermatrica_rep.curves_full import curves_full_plot_short, curves_full_plot_long

        curves_full_df = option_report_multi_post(
            model=setup["model"],
            ds=setup["ds"],
            model_rep=setup["model_rep"],
            opt_set=setup["opt_set"],
            budget_step=10,
            bdg_max=100,
            cores=10,
            if_exact=True,
        )

        assert curves_full_df is not None
        assert len(curves_full_df) > 0
        # Verify pred columns exist and are finite
        for col in ["pred_exact_vol", "pred_long_vol"]:
            assert col in curves_full_df.columns, f"Missing column: {col}"
            assert np.all(np.isfinite(curves_full_df[col].values)), f"{col} contains NaN/Inf"

        short_figs = curves_full_plot_short(curves_full_df, setup["model_rep"])
        assert isinstance(short_figs, dict)
        assert len(short_figs) > 0
        for name, fig in short_figs.items():
            assert isinstance(fig, go.Figure), f"short '{name}' not a Figure"

        for name, fig in short_figs.items():
            assert isinstance(fig, go.Figure)
            assert len(fig.data) > 0, f"short '{name}' has no traces"
            for t in fig.data:
                assert len(t.x) > 0 and len(t.y) > 0, f"short '{name}' trace '{t.name}' has empty data"

        long_figs = curves_full_plot_long(curves_full_df, setup["model_rep"])
        assert isinstance(long_figs, dict)
        assert len(long_figs) > 0
        for name, fig in long_figs.items():
            assert isinstance(fig, go.Figure), f"long '{name}' not a Figure"

        for name, fig in long_figs.items():
            assert isinstance(fig, go.Figure)
            assert len(fig.data) > 0, f"long '{name}' has no traces"
            for t in fig.data:
                assert len(t.x) > 0 and len(t.y) > 0, f"long '{name}' trace '{t.name}' has empty data"


@pytest.mark.integration
class TestP01CurvesFull:
    """Full curve tests against p01_sample (panel LME, 3 brands)."""

    BRAND = "brand_1"

    @pytest.fixture
    def setup(self, p01_model_ds, samples_path):
        """Build ModelRep and opt_set for curves_full calculation."""
        import fermatrica_rep as fmr

        model, ds = p01_model_ds
        ds = ds.copy()

        root = samples_path / "p01_sample"
        opt_path = str(root / "code_py" / "model_data" / "options" / "options.xlsx")
        ssn_path = str(root / "code_py" / "model_data" / "options" / "seasonality.xlsx")

        trans_dict = fmr.trans_dict_create(path=opt_path, sheet="translation")
        model_rep = fmr.ModelRep(ds, trans_dict=trans_dict, language="english")
        ds = fmr.media_ssn_apply(ds, path=ssn_path, sheet="data")

        opt_set = fmr.OptionSettings(
            target=[[self.BRAND]],
            date_start=date(2025, 6, 30),
            date_end=date(2025, 12, 31),
            ref_date_start=date(2024, 1, 1),
            ref_date_end=date(2024, 12, 31),
            apply_vars=["superbrand"],
            plan_period="exact",
        )

        return {"model": model, "ds": ds, "model_rep": model_rep, "opt_set": opt_set}

    def test_curves_full_pipeline(self, setup):
        """Full curves pipeline produces dict of plotly figures."""
        import plotly.graph_objects as go
        from fermatrica_rep.options.calc_multi import option_report_multi_post
        from fermatrica_rep.curves_full import curves_full_plot_short, curves_full_plot_long

        curves_full_df = option_report_multi_post(
            model=setup["model"],
            ds=setup["ds"],
            model_rep=setup["model_rep"],
            opt_set=setup["opt_set"],
            budget_step=10,
            bdg_max=100,
            cores=10,
            if_exact=True,
        )

        assert curves_full_df is not None
        assert len(curves_full_df) > 0
        # Verify pred columns exist and are finite (not "predicted")
        for col in ["pred_exact_vol", "pred_long_vol"]:
            assert col in curves_full_df.columns, f"Missing column: {col}"
            assert np.all(np.isfinite(curves_full_df[col].values)), f"{col} contains NaN/Inf"

        short_figs = curves_full_plot_short(curves_full_df, setup["model_rep"])
        assert isinstance(short_figs, dict)
        assert len(short_figs) > 0
        for name, fig in short_figs.items():
            assert isinstance(fig, go.Figure), f"short '{name}' not a Figure"

        for name, fig in short_figs.items():
            assert isinstance(fig, go.Figure)
            assert len(fig.data) > 0, f"short '{name}' has no traces"
            for t in fig.data:
                assert len(t.x) > 0 and len(t.y) > 0, f"short '{name}' trace '{t.name}' has empty data"

        long_figs = curves_full_plot_long(curves_full_df, setup["model_rep"])
        assert isinstance(long_figs, dict)
        assert len(long_figs) > 0
        for name, fig in long_figs.items():
            assert isinstance(fig, go.Figure), f"long '{name}' not a Figure"

        for name, fig in long_figs.items():
            assert isinstance(fig, go.Figure)
            assert len(fig.data) > 0, f"long '{name}' has no traces"
            for t in fig.data:
                assert len(t.x) > 0 and len(t.y) > 0, f"long '{name}' trace '{t.name}' has empty data"
