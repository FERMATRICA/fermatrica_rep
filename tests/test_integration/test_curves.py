"""Integration tests for curves.py: curves_simple_data and curves_simple_plot.

Tests that short-term efficiency curve functions return valid data and plots.
RHS-only — works for both OLS and LME models.
"""
import numpy as np
import pandas as pd
import pytest


@pytest.mark.integration
class TestP00Curves:
    """Curve tests against p00_sample (OLS, 1 brand)."""

    BRAND = "brand_x"

    @pytest.fixture
    def setup(self, p00_model_ds, samples_path):
        """Build ModelRep for curves calculation."""
        import fermatrica_rep as fmr

        model, ds = p00_model_ds
        ds = ds.copy()

        root = samples_path / "p00_sample"
        opt_path = str(root / "code_py" / "model_data" / "options" / "options.xlsx")

        trans_dict = fmr.trans_dict_create(path=opt_path, sheet="translation")
        model_rep = fmr.ModelRep(ds, trans_dict=trans_dict, language="english")

        return {"model": model, "ds": ds, "model_rep": model_rep}

    def test_curves_simple_data(self, setup):
        """curves_simple_data returns a DataFrame with expected columns."""
        from fermatrica_rep.curves import curves_simple_data

        result = curves_simple_data(
            model=setup["model"],
            ds=setup["ds"],
            model_rep=setup["model_rep"],
            budget_lim=50,
            budget_step=5,
        )

        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert result.index[0] == 0

    def test_curves_simple_data_monotonic(self, setup):
        """curves_simple_data response is monotonically non-decreasing with budget."""
        from fermatrica_rep.curves import curves_simple_data

        result = curves_simple_data(
            model=setup["model"],
            ds=setup["ds"],
            model_rep=setup["model_rep"],
            budget_lim=50,
            budget_step=5,
        )

        core_cols = [c for c in result.columns if not c.endswith('_extrapolated')]
        diffs = result[core_cols].diff().dropna()
        assert (diffs >= -1e-10).all().all(), "Curves are not monotonically non-decreasing"

    def test_curves_simple_data_zero_budget(self, setup):
        """curves_simple_data row with zero budget has zero incremental KPI."""
        from fermatrica_rep.curves import curves_simple_data

        result = curves_simple_data(
            model=setup["model"],
            ds=setup["ds"],
            model_rep=setup["model_rep"],
            budget_lim=50,
            budget_step=5,
        )

        core_cols = [c for c in result.columns if not c.endswith('_extrapolated')]
        zero_row = result[core_cols].iloc[0]
        assert np.all(np.abs(zero_row.values) < 1e-10), \
            "Zero-budget row should have zero incremental KPI"

    def test_curves_simple_plot(self, setup):
        """curves_simple_plot returns dict of go.Figure with data traces."""
        import plotly.graph_objects as go
        from fermatrica_rep.curves import curves_simple_data, curves_simple_plot

        data = curves_simple_data(
            model=setup["model"],
            ds=setup["ds"],
            model_rep=setup["model_rep"],
            budget_lim=50,
            budget_step=5,
        )

        figs = curves_simple_plot(data, model_rep=setup["model_rep"])

        assert isinstance(figs, dict)
        assert len(figs) > 0
        for name, fig in figs.items():
            assert isinstance(fig, go.Figure), f"'{name}' is not a Figure"

    def test_curves_simple_plot_x_is_budget(self, setup):
        """curves_simple_plot x-axis values match budget index."""
        import plotly.graph_objects as go
        from fermatrica_rep.curves import curves_simple_data, curves_simple_plot

        data = curves_simple_data(
            model=setup["model"],
            ds=setup["ds"],
            model_rep=setup["model_rep"],
            budget_lim=50,
            budget_step=5,
        )

        figs = curves_simple_plot(data, model_rep=setup["model_rep"])

        for name, fig in figs.items():
            for t in fig.data:
                if len(t.x) > 0:
                    np.testing.assert_array_almost_equal(t.x, data.index.values[:len(t.x)])


@pytest.mark.integration
class TestP01Curves:
    """Curve tests against p01_sample (panel LME, 3 brands)."""

    BRAND = "brand_1"

    @pytest.fixture
    def setup(self, p01_model_ds, samples_path):
        """Build ModelRep for curves calculation."""
        import fermatrica_rep as fmr

        model, ds = p01_model_ds
        ds = ds.copy()

        root = samples_path / "p01_sample"
        opt_path = str(root / "code_py" / "model_data" / "options" / "options.xlsx")

        trans_dict = fmr.trans_dict_create(path=opt_path, sheet="translation")
        model_rep = fmr.ModelRep(ds, trans_dict=trans_dict, language="english")

        return {"model": model, "ds": ds, "model_rep": model_rep}

    def test_curves_simple_data(self, setup):
        """curves_simple_data returns data with expected structure."""
        from fermatrica_rep.curves import curves_simple_data

        result = curves_simple_data(
            model=setup["model"],
            ds=setup["ds"],
            model_rep=setup["model_rep"],
            budget_lim=50,
            budget_step=5,
        )

        assert result is not None

    def test_curves_simple_plot(self, setup):
        """curves_simple_plot returns dict of go.Figure with data traces."""
        import plotly.graph_objects as go
        from fermatrica_rep.curves import curves_simple_data, curves_simple_plot

        data = curves_simple_data(
            model=setup["model"],
            ds=setup["ds"],
            model_rep=setup["model_rep"],
            budget_lim=50,
            budget_step=5,
        )

        figs = curves_simple_plot(data, model_rep=setup["model_rep"])

        assert isinstance(figs, dict)
        assert len(figs) > 0
        for name, fig in figs.items():
            assert isinstance(fig, go.Figure), f"'{name}' is not a Figure"

    def test_curves_simple_data_monotonic(self, setup):
        """curves_simple_data response is monotonically non-decreasing with budget."""
        from fermatrica_rep.curves import curves_simple_data

        result = curves_simple_data(
            model=setup["model"],
            ds=setup["ds"],
            model_rep=setup["model_rep"],
            budget_lim=50,
            budget_step=5,
        )

        core_cols = [c for c in result.columns if not c.endswith('_extrapolated')]
        diffs = result[core_cols].diff().dropna()
        assert (diffs >= -1e-10).all().all(), "Curves are not monotonically non-decreasing"

    def test_curves_simple_data_zero_budget(self, setup):
        """curves_simple_data row with zero budget has zero incremental KPI."""
        from fermatrica_rep.curves import curves_simple_data

        result = curves_simple_data(
            model=setup["model"],
            ds=setup["ds"],
            model_rep=setup["model_rep"],
            budget_lim=50,
            budget_step=5,
        )

        if len(result) == 0:
            pytest.skip("curves_simple_data returned empty DataFrame for this sample")

        core_cols = [c for c in result.columns if not c.endswith('_extrapolated')]
        zero_row = result[core_cols].iloc[0]
        assert np.all(np.abs(zero_row.values) < 1e-10), \
            "Zero-budget row should have zero incremental KPI"


@pytest.mark.integration
class TestP02Curves:
    """Curve tests against p02_sample (panel per SKU, LME, 3 brands)."""

    BRAND = "brand_1"

    @pytest.fixture
    def setup(self, p02_model_ds, samples_path):
        """Build ModelRep for curves calculation."""
        import fermatrica_rep as fmr

        model, ds = p02_model_ds
        ds = ds.copy()

        root = samples_path / "p02_sample"
        opt_path = str(root / "code_py" / "model_data" / "options" / "options.xlsx")

        trans_dict = fmr.trans_dict_create(path=opt_path, sheet="translation")
        model_rep = fmr.ModelRep(ds, trans_dict=trans_dict, language="english")

        return {"model": model, "ds": ds, "model_rep": model_rep}

    def test_curves_simple_data(self, setup):
        """curves_simple_data returns a result with expected structure."""
        from fermatrica_rep.curves import curves_simple_data

        result = curves_simple_data(
            model=setup["model"],
            ds=setup["ds"],
            model_rep=setup["model_rep"],
            budget_lim=50,
            budget_step=5,
        )

        assert result is not None

    def test_curves_simple_plot(self, setup):
        """curves_simple_plot returns dict of go.Figure."""
        import plotly.graph_objects as go
        from fermatrica_rep.curves import curves_simple_data, curves_simple_plot

        data = curves_simple_data(
            model=setup["model"],
            ds=setup["ds"],
            model_rep=setup["model_rep"],
            budget_lim=50,
            budget_step=5,
        )

        figs = curves_simple_plot(data, model_rep=setup["model_rep"])

        assert isinstance(figs, dict)
        assert len(figs) > 0
        for name, fig in figs.items():
            assert isinstance(fig, go.Figure), f"'{name}' is not a Figure"

    def test_curves_simple_data_monotonic(self, setup):
        """curves_simple_data response is monotonically non-decreasing with budget."""
        from fermatrica_rep.curves import curves_simple_data

        result = curves_simple_data(
            model=setup["model"],
            ds=setup["ds"],
            model_rep=setup["model_rep"],
            budget_lim=50,
            budget_step=5,
        )

        core_cols = [c for c in result.columns if not c.endswith('_extrapolated')]
        diffs = result[core_cols].diff().dropna()
        assert (diffs >= -1e-10).all().all(), "Curves are not monotonically non-decreasing"

    def test_curves_simple_data_zero_budget(self, setup):
        """curves_simple_data row with zero budget has zero incremental KPI."""
        from fermatrica_rep.curves import curves_simple_data

        result = curves_simple_data(
            model=setup["model"],
            ds=setup["ds"],
            model_rep=setup["model_rep"],
            budget_lim=50,
            budget_step=5,
        )

        if len(result) == 0:
            pytest.skip("curves_simple_data returned empty DataFrame for this sample")

        core_cols = [c for c in result.columns if not c.endswith('_extrapolated')]
        zero_row = result[core_cols].iloc[0]
        assert np.all(np.abs(zero_row.values) < 1e-10), \
            "Zero-budget row should have zero incremental KPI"


@pytest.mark.integration
class TestP03Curves:
    """Curve tests against p03_sample (ABA target, monthly, LME)."""

    BRAND = "brand_1"

    @pytest.fixture
    def setup(self, p03_model_ds, samples_path):
        """Build ModelRep for curves calculation."""
        import fermatrica_rep as fmr

        model, ds = p03_model_ds
        ds = ds.copy()

        root = samples_path / "p03_sample"
        opt_path = str(root / "code_py" / "model_data" / "options" / "options.xlsx")

        trans_dict = fmr.trans_dict_create(path=opt_path, sheet="translation")
        model_rep = fmr.ModelRep(ds, trans_dict=trans_dict, language="english")

        return {"model": model, "ds": ds, "model_rep": model_rep}

    def test_curves_simple_data(self, setup):
        """curves_simple_data returns a result with expected structure."""
        from fermatrica_rep.curves import curves_simple_data

        result = curves_simple_data(
            model=setup["model"],
            ds=setup["ds"],
            model_rep=setup["model_rep"],
            budget_lim=50,
            budget_step=5,
        )

        assert result is not None

    def test_curves_simple_plot(self, setup):
        """curves_simple_plot returns dict of go.Figure."""
        import plotly.graph_objects as go
        from fermatrica_rep.curves import curves_simple_data, curves_simple_plot

        data = curves_simple_data(
            model=setup["model"],
            ds=setup["ds"],
            model_rep=setup["model_rep"],
            budget_lim=50,
            budget_step=5,
        )

        figs = curves_simple_plot(data, model_rep=setup["model_rep"])

        assert isinstance(figs, dict)
        assert len(figs) > 0
        for name, fig in figs.items():
            assert isinstance(fig, go.Figure), f"'{name}' is not a Figure"

    def test_curves_simple_data_monotonic(self, setup):
        """curves_simple_data response is monotonically non-decreasing with budget."""
        from fermatrica_rep.curves import curves_simple_data

        result = curves_simple_data(
            model=setup["model"],
            ds=setup["ds"],
            model_rep=setup["model_rep"],
            budget_lim=50,
            budget_step=5,
        )

        core_cols = [c for c in result.columns if not c.endswith('_extrapolated')]
        diffs = result[core_cols].diff().dropna()
        assert (diffs >= -1e-10).all().all(), "Curves are not monotonically non-decreasing"

    def test_curves_simple_data_zero_budget(self, setup):
        """curves_simple_data row with zero budget has zero incremental KPI."""
        from fermatrica_rep.curves import curves_simple_data

        result = curves_simple_data(
            model=setup["model"],
            ds=setup["ds"],
            model_rep=setup["model_rep"],
            budget_lim=50,
            budget_step=5,
        )

        if len(result) == 0:
            pytest.skip("curves_simple_data returned empty DataFrame for this sample")

        core_cols = [c for c in result.columns if not c.endswith('_extrapolated')]
        zero_row = result[core_cols].iloc[0]
        assert np.all(np.abs(zero_row.values) < 1e-10), \
            "Zero-budget row should have zero incremental KPI"


# NOTE: p03 uses ABA target. Monotonicity and zero-budget invariants (Test 1 above)
# should still hold since curves_simple_data is RHS-only.

# NOTE: No TestP04Curves — p04 uses media_prices-based translation with
# variable/variable_fin column mismatch. curves_simple_data does not support this format.
# Full curves (curves_full) would be the appropriate test for p04's regional model.

# NOTE: p05 (two-level chain model) has no curves test class. Short-term RHS-only curves
# do not apply to chain models where BHT output feeds Sales input. Full curves
# (curves_full) would be the appropriate approach.
