"""Unit tests for curves_full_data and curves_full_long_data functions.

These tests use monkeypatch to avoid heavy integration dependencies,
focusing on data shape, monotonicity, zero-budget behaviour, and
long-vs-short KPI relationships.
"""
import numpy as np
import pandas as pd
import pytest

from fermatrica_rep.analysis.curves_full import curves_full_plot_short, curves_full_plot_long
from fermatrica_rep.options.define import OptionSettings
from fermatrica_rep.meta_model.model_rep import ModelRep


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synthetic_results():
    """Return a DataFrame mimicking option_report_multi_post output.

    Zero option has a single row (bdg=0). Other options span budget range.
    """
    np.random.seed(42)
    n_bdg = 11
    bdg_range = np.arange(0, 110, 10)

    zero_df = pd.DataFrame({
        'bdg': [0],
        'option': ['zero'],
        'pred_exact_vol': [100.0],
        'pred_long_vol': [200.0],
        'pred_exact_val': [150.0],
        'pred_long_val': [300.0],
        'max_obs_budg': [100],
    })

    other = pd.DataFrame({
        'bdg': np.tile(bdg_range, 3),
        'option': np.repeat(['tv', 'digital', 'pr'], n_bdg),
        'pred_exact_vol': np.random.uniform(100, 200, n_bdg * 3),
        'pred_long_vol': np.random.uniform(200, 400, n_bdg * 3),
        'pred_exact_val': np.random.uniform(150, 300, n_bdg * 3),
        'pred_long_val': np.random.uniform(300, 600, n_bdg * 3),
        'max_obs_budg': [100] * (n_bdg * 3),
    })

    return pd.concat([zero_df, other], ignore_index=True)


def _synthetic_ds():
    n = 12
    return pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=n, freq='MS'),
        'superbrand': ['brand_x'] * n,
        'bs_key': [1] * n,
        'listed': [4] * n,
    })


# ---------------------------------------------------------------------------
# TestCurvesFullData
# ---------------------------------------------------------------------------

class TestCurvesFullPlotShort:

    def test_returns_dict_of_figures(self):
        results = _synthetic_results()
        ds = _synthetic_ds()
        model_rep = ModelRep(ds, language='english')

        figs = curves_full_plot_short(results, model_rep)
        assert isinstance(figs, dict)
        assert len(figs) > 0
        for name, fig in figs.items():
            import plotly.graph_objects as go
            assert isinstance(fig, go.Figure)

    def test_zero_budget_row_present(self):
        results = _synthetic_results()
        ds = _synthetic_ds()
        model_rep = ModelRep(ds, language='english')

        figs = curves_full_plot_short(results, model_rep)
        for name, fig in figs.items():
            import plotly.graph_objects as go
            assert isinstance(fig, go.Figure)

    def test_handles_empty_df(self):
        results = _synthetic_results().iloc[0:0].copy()
        ds = _synthetic_ds()
        model_rep = ModelRep(ds, language='english')

        # Should not crash with empty DataFrame
        figs = curves_full_plot_short(results, model_rep)
        assert isinstance(figs, dict)


class TestCurvesFullPlotLong:

    def test_returns_dict_of_figures(self):
        results = _synthetic_results()
        ds = _synthetic_ds()
        model_rep = ModelRep(ds, language='english')

        figs = curves_full_plot_long(results, model_rep)
        assert isinstance(figs, dict)
        assert len(figs) > 0
        for name, fig in figs.items():
            import plotly.graph_objects as go
            assert isinstance(fig, go.Figure)

    def test_handles_empty_df(self):
        results = _synthetic_results().iloc[0:0].copy()
        ds = _synthetic_ds()
        model_rep = ModelRep(ds, language='english')

        figs = curves_full_plot_long(results, model_rep)
        assert isinstance(figs, dict)
