"""Unit tests for the adstock slide series filter (export/slides/adstocks.py).

The adstock slide must plot only columns referenced by an active model-formula
token. Adstock columns of variables present in the dataset but absent from the
formula resolve to no display_var — previously raising IndexError in the
display-name lookup — and are now excluded from the chart.
"""
import types

import pandas as pd
import pytest

import fermatrica_rep as fmr
from fermatrica_rep.export import export_pptx
from fermatrica_rep.export.slides import adstocks as slides_adstocks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model_rhs():
    """Active and inactive formula tokens, mirroring a real MMM RHS sheet."""
    return pd.DataFrame({
        "token": [
            "brnd_tv_lag_adstockpd",
            "dgt_perform_adstockpd",
            "war_dwblp",
            "I(dgt_posts_adstockd + dgt_banners_adstockd * .2)",
        ],
        "if_active": [1, 0, 1, 1],
        "display_var": ["tv", "perform", "war", "digital"],
    })


def _trace():
    """Trace ``adstocks_data`` could return: two directly active columns, one
    inactive-token column (dgt_perform), and two columns nested inside an
    active ``I(...)`` token."""
    return pd.DataFrame({
        "brnd_tv_lag_adstockpd": [1.0, 0.5, 0.25],
        "dgt_perform_adstockpd": [1.0, 0.7, 0.49],
        "war_dwblp": [1.0, 0.3, 0.09],
        "dgt_posts_adstockd": [1.0, 0.8, 0.64],
        "dgt_banners_adstockd": [1.0, 0.6, 0.36],
    })


def _node_and_prs(model_rhs, palette_tools=None):
    """ModelNode-like stub bundling a real ModelRep + presentation template."""
    conf = types.SimpleNamespace(
        model_rhs=model_rhs,
        target_superbrand="poetti",
        period_var="month",
    )
    model_rep = fmr.ModelRep(
        pd.DataFrame({"date": pd.date_range("2024-01-01", periods=3, freq="MS")}),
        language="english",
    )
    model_rep, prs = export_pptx.set_config(model_rep, template_name="OKKAM_2023")
    if palette_tools is not None:
        model_rep.palette_tools = palette_tools
    node = types.SimpleNamespace(model=types.SimpleNamespace(conf=conf), model_rep=model_rep)
    return node, prs


def _chart_series(prs):
    """Series names of the chart on the last slide (empty when no chart)."""
    for shp in prs.slides[len(prs.slides) - 1].shapes:
        if shp.has_chart:
            return [sr.name for sr in shp.chart.plots[0].series]
    return []


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAdstocksSlideSeries:

    def test_keeps_only_active_formula_columns(self, monkeypatch):
        node, prs = _node_and_prs(_model_rhs())
        monkeypatch.setattr(slides_adstocks, "adstocks_data", lambda **kw: _trace())

        slides_adstocks.create(
            prs, node,
            ds=pd.DataFrame(),
            cln_meas=["brnd_tv_lag", "dgt_perform", "war", "dgt_posts", "dgt_banners"],
            cln_dim=(),
            n=3,
        )

        assert _chart_series(prs) == [
            "brnd_tv_lag_adstockpd",
            "war_dwblp",
            "dgt_posts_adstockd",
            "dgt_banners_adstockd",
        ]

    def test_renames_to_display_var_when_in_palette(self, monkeypatch):
        node, prs = _node_and_prs(
            _model_rhs(), palette_tools={"tv": "#FF0000", "war": "#00FF00"},
        )
        monkeypatch.setattr(slides_adstocks, "adstocks_data", lambda **kw: _trace())

        slides_adstocks.create(
            prs, node,
            ds=pd.DataFrame(),
            cln_meas=["brnd_tv_lag", "dgt_perform", "war", "dgt_posts", "dgt_banners"],
            cln_dim=(),
            n=3,
        )

        # renamed to display_var where the palette covers them, raw names otherwise
        assert _chart_series(prs) == [
            "tv",
            "war",
            "dgt_posts_adstockd",
            "dgt_banners_adstockd",
        ]

    def test_no_chart_when_every_column_is_non_formula(self, monkeypatch):
        node, prs = _node_and_prs(_model_rhs())
        monkeypatch.setattr(
            slides_adstocks, "adstocks_data",
            lambda **kw: pd.DataFrame({"dgt_perform_adstockpd": [1.0, 0.7]}),
        )

        slides_adstocks.create(
            prs, node,
            ds=pd.DataFrame(),
            cln_meas=["dgt_perform"],
            cln_dim=(),
            n=2,
        )

        assert _chart_series(prs) == []
