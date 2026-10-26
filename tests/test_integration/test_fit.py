"""Integration tests for fit.py: fit_main_plot_vol and fit_mult_plot_vol.

Tests that fit plot functions return valid Plotly figures against real model data.
"""
import pytest
import numpy as np
from datetime import date


def _check_figure_traces(fig, min_traces=1):
    """Verify all traces have non-empty x/y data."""
    assert len(fig.data) >= min_traces
    for t in fig.data:
        assert len(t.x) > 0, f"Trace '{t.name}' has empty x"
        assert len(t.y) > 0, f"Trace '{t.name}' has empty y"


def _check_observed_predicted_same_length(fig):
    """Verify observed and predicted traces have same length."""
    observed_traces = [t for t in fig.data if 'observed' in (t.name or '').lower()]
    predicted_traces = [t for t in fig.data if 'predicted' in (t.name or '').lower()]
    for obs_t in observed_traces:
        for pred_t in predicted_traces:
            assert len(obs_t.x) == len(pred_t.x)


@pytest.mark.integration
class TestP00Fit:
    """fit plot tests against p00_sample (time series OLS, 1 brand)."""

    BRAND = "brand_x"

    @pytest.fixture
    def setup(self, p00_model_ds, samples_path):
        """Run option_report to produce dt_pred for fit plots."""
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

        opt_set = fmr.OptionSettings(
            target=[[self.BRAND]],
            date_start=date(2018, 1, 1),
            date_end=date(2018, 12, 31),
            ref_date_start=date(2017, 1, 1),
            ref_date_end=date(2017, 12, 31),
            apply_vars=["superbrand"],
            plan_period="exact",
        )

        _, dt_pred, _ = fmr.option_report(
            model, ds, model_rep, options_m["predefined"], opt_set, if_exact=False,
        )

        return {"model": model, "dt_pred": dt_pred, "model_rep": model_rep}

    def test_fit_main_plot_vol(self, setup):
        """fit_main_plot_vol returns a go.Figure with data traces."""
        import plotly.graph_objects as go
        from fermatrica_rep.analysis.fit import fit_main_plot_vol

        fig = fit_main_plot_vol(
            setup["model"], setup["dt_pred"], setup["model_rep"], period="month",
        )

        assert isinstance(fig, go.Figure)
        _check_figure_traces(fig)

    def test_fit_main_data_wide_long_consistency(self, setup):
        """fit_main_data wide and long DataFrames have matching totals."""
        from fermatrica_rep.analysis.fit import fit_main_data

        wide, long = fit_main_data(
            setup["model"], setup["dt_pred"], period="month",
        )

        # Sum observed in wide should match long's observed total
        wide_obs = wide["observed"].sum()
        long_obs = long[long["variable"] == "observed"]["value"].sum()
        assert abs(wide_obs - long_obs) < 1e-6, (
            f"Wide observed sum ({wide_obs:.2f}) != long observed sum ({long_obs:.2f})"
        )

        wide_pred = wide["predicted"].sum()
        long_pred = long[long["variable"] == "predicted"]["value"].sum()
        assert abs(wide_pred - long_pred) < 1e-6, (
            f"Wide predicted sum ({wide_pred:.2f}) != long predicted sum ({long_pred:.2f})"
        )

    def test_edge_case_period_week(self, setup):
        """fit_main_plot_vol with weekly period."""
        import plotly.graph_objects as go
        from fermatrica_rep.analysis.fit import fit_main_plot_vol

        fig = fit_main_plot_vol(
            setup["model"], setup["dt_pred"], setup["model_rep"], period="week",
        )
        assert isinstance(fig, go.Figure)
        _check_figure_traces(fig)

    def test_edge_case_period_quarter(self, setup):
        """fit_main_plot_vol with quarterly period."""
        import plotly.graph_objects as go
        from fermatrica_rep.analysis.fit import fit_main_plot_vol

        fig = fit_main_plot_vol(
            setup["model"], setup["dt_pred"], setup["model_rep"], period="quarter",
        )
        assert isinstance(fig, go.Figure)
        _check_figure_traces(fig)

    def test_edge_case_no_future(self, setup):
        """fit_main_plot_vol with show_future=False."""
        import plotly.graph_objects as go
        from fermatrica_rep.analysis.fit import fit_main_plot_vol

        fig = fit_main_plot_vol(
            setup["model"], setup["dt_pred"], setup["model_rep"],
            period="month", show_future=False,
        )
        assert isinstance(fig, go.Figure)
        _check_figure_traces(fig)

    def test_edge_case_zero_err_int(self, setup):
        """fit_main_plot_vol with err_int=0.0 (collapsed error band)."""
        import plotly.graph_objects as go
        from fermatrica_rep.analysis.fit import fit_main_plot_vol

        fig = fit_main_plot_vol(
            setup["model"], setup["dt_pred"], setup["model_rep"],
            period="month", err_int=0.0,
        )
        assert isinstance(fig, go.Figure)
        _check_figure_traces(fig)

    def test_fit_golden(self, setup, if_debug):
        """fit_main_plot_vol matches golden expected trace values (p00 OLS)."""
        import os
        import yaml
        from fermatrica_rep.analysis.fit import fit_main_plot_vol
        from tests.test_integration.helpers import compare_plot_golden

        pth = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', 'test_cases', 'fit_golden_p00.yaml')
        if not os.path.exists(pth):
            pytest.skip("Golden YAML file not found")

        with open(pth, mode='rt', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f)

        fig = fit_main_plot_vol(
            setup["model"], setup["dt_pred"], setup["model_rep"],
            period="month", show_future=True,
        )

        for item in yaml_data:
            if item.get('sample') == 'p00':
                expected = item.get('expected', {})
                try:
                    compare_plot_golden(fig, expected)
                except AssertionError:
                    if if_debug:
                        import traceback
                        traceback.print_exc()
                        breakpoint()
                        # Inspect actual values and update YAML
                        for t in fig.data:
                            print(f"Trace '{t.name}': y_sum={sum(t.y):.4f}, y={list(t.y)}")
                    else:
                        raise
                else:
                    break


@pytest.mark.integration
class TestP01Fit:
    """fit plot tests against p01_sample (panel LME, 3 brands)."""

    BRAND = "brand_1"

    @pytest.fixture
    def setup(self, p01_model_ds, samples_path):
        """Run option_report to produce dt_pred for fit plots."""
        import fermatrica_rep as fmr

        model, ds = p01_model_ds
        ds = ds.copy()

        root = samples_path / "p01_sample"
        opt_path = str(root / "code_py" / "model_data" / "options" / "options.xlsx")
        ssn_path = str(root / "code_py" / "model_data" / "options" / "seasonality.xlsx")

        trans_dict = fmr.trans_dict_create(path=opt_path, sheet="translation")
        options_c = fmr.budget_dict_create(path=opt_path, sheet="current", vat_rate=0.0)
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

        _, dt_pred, _ = fmr.option_report(
            model, ds, model_rep, options_c["predefined"], opt_set, if_exact=False,
        )

        return {"model": model, "dt_pred": dt_pred, "model_rep": model_rep}

    def test_fit_main_plot_vol(self, setup):
        """fit_main_plot_vol with single brand returns go.Figure."""
        import plotly.graph_objects as go
        from fermatrica_rep.analysis.fit import fit_main_plot_vol

        fig = fit_main_plot_vol(
            setup["model"], setup["dt_pred"], setup["model_rep"],
            period="month", show_future=True,
        )

        assert isinstance(fig, go.Figure)
        _check_figure_traces(fig)

    def test_fit_mult_plot_vol(self, setup):
        """fit_mult_plot_vol with 3 brands returns go.Figure with multiple traces."""
        import plotly.graph_objects as go
        from fermatrica_rep.analysis.fit import fit_mult_plot_vol

        fig = fit_mult_plot_vol(
            setup["model"], setup["dt_pred"], setup["model_rep"],
            period="month", show_future=True, group_var=["superbrand"],
        )

        assert isinstance(fig, go.Figure)
        # Should have at least 2 traces per brand (observed + predicted) = 6
        _check_figure_traces(fig, min_traces=4)

    def test_fit_mult_plot_vol_no_future(self, setup):
        """fit_mult_plot_vol with show_future=False."""
        import plotly.graph_objects as go
        from fermatrica_rep.analysis.fit import fit_mult_plot_vol

        fig = fit_mult_plot_vol(
            setup["model"], setup["dt_pred"], setup["model_rep"],
            period="month", show_future=False, group_var=["superbrand"],
        )

        assert isinstance(fig, go.Figure)
        _check_figure_traces(fig, min_traces=4)

    def test_fit_main_data_wide_long_consistency(self, setup):
        """fit_main_data wide and long DataFrames have matching totals."""
        from fermatrica_rep.analysis.fit import fit_main_data

        wide, long = fit_main_data(
            setup["model"], setup["dt_pred"], period="month",
        )

        wide_obs = wide["observed"].sum()
        long_obs = long[long["variable"] == "observed"]["value"].sum()
        assert abs(wide_obs - long_obs) < 1e-6, (
            f"Wide observed sum ({wide_obs:.2f}) != long observed sum ({long_obs:.2f})"
        )

        wide_pred = wide["predicted"].sum()
        long_pred = long[long["variable"] == "predicted"]["value"].sum()
        assert abs(wide_pred - long_pred) < 1e-6, (
            f"Wide predicted sum ({wide_pred:.2f}) != long predicted sum ({long_pred:.2f})"
        )

    def test_fit_golden(self, setup, if_debug):
        """fit_main_plot_vol matches golden expected trace values (p01 LME)."""
        import os
        import yaml
        from fermatrica_rep.analysis.fit import fit_main_plot_vol
        from tests.test_integration.helpers import compare_plot_golden

        pth = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', 'test_cases', 'fit_golden_p01.yaml')
        if not os.path.exists(pth):
            pytest.skip("Golden YAML file not found")

        with open(pth, mode='rt', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f)

        fig = fit_main_plot_vol(
            setup["model"], setup["dt_pred"], setup["model_rep"],
            period="month", show_future=True,
        )

        for item in yaml_data:
            if item.get('sample') == 'p01':
                expected = item.get('expected', {})
                try:
                    compare_plot_golden(fig, expected)
                except AssertionError:
                    if if_debug:
                        import traceback
                        traceback.print_exc()
                        breakpoint()
                        for t in fig.data:
                            print(f"Trace '{t.name}': y_sum={sum(t.y):.4f}")
                    else:
                        raise
                else:
                    break


@pytest.mark.integration
class TestP02Fit:
    """fit plot tests against p02_sample (panel per SKU, LME, 3 brands)."""

    BRAND = "brand_1"

    @pytest.fixture
    def setup(self, p02_model_ds, samples_path):
        """Run option_report to produce dt_pred for fit plots."""
        import fermatrica_rep as fmr

        model, ds = p02_model_ds
        ds = ds.copy()

        root = samples_path / "p02_sample"
        opt_path = str(root / "code_py" / "model_data" / "options" / "options.xlsx")
        ssn_path = str(root / "code_py" / "model_data" / "options" / "seasonality.xlsx")

        trans_dict = fmr.trans_dict_create(path=opt_path, sheet="translation")
        options_m = fmr.budget_dict_create(path=opt_path, sheet="main", vat_rate=0.0)
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

        _, dt_pred, _ = fmr.option_report(
            model, ds, model_rep, options_m["predefined"], opt_set, if_exact=False,
        )

        return {"model": model, "dt_pred": dt_pred, "model_rep": model_rep}

    def test_fit_main_plot_vol(self, setup):
        """fit_main_plot_vol with single brand returns go.Figure."""
        import plotly.graph_objects as go
        from fermatrica_rep.analysis.fit import fit_main_plot_vol

        fig = fit_main_plot_vol(
            setup["model"], setup["dt_pred"], setup["model_rep"],
            period="month", show_future=True,
        )

        assert isinstance(fig, go.Figure)
        _check_figure_traces(fig)

    def test_fit_mult_plot_vol(self, setup):
        """fit_mult_plot_vol with 3 brands returns go.Figure."""
        import plotly.graph_objects as go
        from fermatrica_rep.analysis.fit import fit_mult_plot_vol

        fig = fit_mult_plot_vol(
            setup["model"], setup["dt_pred"], setup["model_rep"],
            period="month", show_future=True, group_var=["superbrand"],
        )

        assert isinstance(fig, go.Figure)
        _check_figure_traces(fig, min_traces=4)

    def test_fit_mult_plot_vol_no_future(self, setup):
        """fit_mult_plot_vol with show_future=False."""
        import plotly.graph_objects as go
        from fermatrica_rep.analysis.fit import fit_mult_plot_vol

        fig = fit_mult_plot_vol(
            setup["model"], setup["dt_pred"], setup["model_rep"],
            period="month", show_future=False, group_var=["superbrand"],
        )
        assert isinstance(fig, go.Figure)
        _check_figure_traces(fig, min_traces=4)

    def test_fit_main_data_wide_long_consistency(self, setup):
        """fit_main_data wide and long DataFrames have matching totals."""
        from fermatrica_rep.analysis.fit import fit_main_data

        wide, long = fit_main_data(
            setup["model"], setup["dt_pred"], period="month",
        )

        wide_obs = wide["observed"].sum()
        long_obs = long[long["variable"] == "observed"]["value"].sum()
        assert abs(wide_obs - long_obs) < 1e-6, (
            f"Wide observed sum ({wide_obs:.2f}) != long observed sum ({long_obs:.2f})"
        )

        wide_pred = wide["predicted"].sum()
        long_pred = long[long["variable"] == "predicted"]["value"].sum()
        assert abs(wide_pred - long_pred) < 1e-6, (
            f"Wide predicted sum ({wide_pred:.2f}) != long predicted sum ({long_pred:.2f})"
        )


@pytest.mark.integration
class TestP03Fit:
    """fit plot tests against p03_sample (ABA target, monthly, LME, 3 brands)."""

    BRAND = "brand_1"

    @pytest.fixture
    def setup(self, p03_model_ds, samples_path):
        """Run option_report to produce dt_pred for ABA fit plots."""
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

        opt_set = fmr.OptionSettings(
            target=[[self.BRAND]],
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

        return {"model": model, "dt_pred": dt_pred, "model_rep": model_rep}

    def test_fit_main_plot_vol(self, setup):
        """fit_main_plot_vol with ABA target returns go.Figure."""
        import plotly.graph_objects as go
        from fermatrica_rep.analysis.fit import fit_main_plot_vol

        fig = fit_main_plot_vol(
            setup["model"], setup["dt_pred"], setup["model_rep"],
            period="month", show_future=True,
        )

        assert isinstance(fig, go.Figure)
        _check_figure_traces(fig)

    def test_fit_main_data_wide_long_consistency(self, setup):
        """fit_main_data wide and long DataFrames have matching totals."""
        from fermatrica_rep.analysis.fit import fit_main_data

        wide, long = fit_main_data(
            setup["model"], setup["dt_pred"], period="month",
        )

        wide_obs = wide["observed"].sum()
        long_obs = long[long["variable"] == "observed"]["value"].sum()
        assert abs(wide_obs - long_obs) < 1e-6, (
            f"Wide observed sum ({wide_obs:.2f}) != long observed sum ({long_obs:.2f})"
        )

        wide_pred = wide["predicted"].sum()
        long_pred = long[long["variable"] == "predicted"]["value"].sum()
        assert abs(wide_pred - long_pred) < 1e-6, (
            f"Wide predicted sum ({wide_pred:.2f}) != long predicted sum ({long_pred:.2f})"
        )


@pytest.mark.integration
class TestP04Fit:
    """fit plot tests against p04_sample (panel per region, LME, 4 regions)."""

    BRAND = "brand_1"

    @pytest.fixture
    def setup(self, p04_model_ds, samples_path):
        """Run option_report to produce dt_pred for regional fit plots."""
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
        options_m = {}
        for ms in options_m_raw["text_flags"].unique():
            if pd.isna(ms):
                continue
            options_m[ms] = fmr.budget_dict_create(
                dct=options_m_raw[options_m_raw["text_flags"] == ms])

        media_prices = pd.read_excel(opt_path, sheet_name="media_prices")
        media_prices = pd.melt(
            media_prices, id_vars="master", var_name="variable", value_name="price")

        ms = "Moscow"
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

        _, dt_pred, _ = fmr.option_report(
            model, ds, model_rep,
            options_m[ms]["predefined"], opt_set, if_exact=False,
        )

        return {"model": model, "dt_pred": dt_pred, "model_rep": model_rep}

    def test_fit_main_plot_vol(self, setup):
        """fit_main_plot_vol with single region returns go.Figure."""
        import plotly.graph_objects as go
        from fermatrica_rep.analysis.fit import fit_main_plot_vol

        fig = fit_main_plot_vol(
            setup["model"], setup["dt_pred"], setup["model_rep"],
            period="month", show_future=True,
        )

        assert isinstance(fig, go.Figure)
        _check_figure_traces(fig)

    def test_fit_mult_plot_vol(self, setup):
        """fit_mult_plot_vol with 4 regions returns go.Figure."""
        import plotly.graph_objects as go
        from fermatrica_rep.analysis.fit import fit_mult_plot_vol

        fig = fit_mult_plot_vol(
            setup["model"], setup["dt_pred"], setup["model_rep"],
            period="month", show_future=True, group_var=["superbrand", "master"],
        )

        assert isinstance(fig, go.Figure)
        _check_figure_traces(fig, min_traces=4)

    def test_fit_main_data_wide_long_consistency(self, setup):
        """fit_main_data wide and long DataFrames have matching totals."""
        from fermatrica_rep.analysis.fit import fit_main_data

        wide, long = fit_main_data(
            setup["model"], setup["dt_pred"], period="month",
        )

        wide_obs = wide["observed"].sum()
        long_obs = long[long["variable"] == "observed"]["value"].sum()
        assert abs(wide_obs - long_obs) < 1e-6, (
            f"Wide observed sum ({wide_obs:.2f}) != long observed sum ({long_obs:.2f})"
        )

        wide_pred = wide["predicted"].sum()
        long_pred = long[long["variable"] == "predicted"]["value"].sum()
        assert abs(wide_pred - long_pred) < 1e-6, (
            f"Wide predicted sum ({wide_pred:.2f}) != long predicted sum ({long_pred:.2f})"
        )
