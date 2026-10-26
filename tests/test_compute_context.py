"""Unit tests for ComputeContext, load_context, and MetaModel compute_context wiring."""


import numpy as np
import pandas as pd
import pytest

from fermatrica.basics.basics import FermatricaError
from fermatrica_rep.meta_model.node import ModelNode
from fermatrica_rep.meta_model.meta_model import MetaModel
from fermatrica_rep.meta_model.model_rep import ModelRep
from fermatrica_rep.options.context import ComputeContext, load_context
from fermatrica_rep.options.option import Option
from fermatrica_rep.options.scope import OptionScope


class MockConf:
    price_var = None
    summary_type = "sum"
    Y_var = "y"
    model_type = "OLS"
    params: dict = {}


class MockModel:
    def __init__(self):
        self.conf = MockConf()
        self.obj = type("Obj", (), {"models": {}})()


def _basic_ds(n=10):
    return pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=n, freq="D"),
        "superbrand": ["x"] * n,
        "bs_key": [1] * n,
        "listed": [4] * n,
        "predicted": [100.0] * n,
        "observed": [90.0] * n,
    })


def _scopes_df():
    return pd.DataFrame({
        "scope_name": ["s1", "s2"],
        "date_start": ["2025-01-01", "2026-01-01"],
        "date_end": ["2025-12-31", "2026-12-31"],
        "ref_date_start": ["2024-01-01", "2025-01-01"],
        "ref_date_end": ["2024-12-31", "2025-12-31"],
        "apply_vars": ["superbrand", "superbrand"],
        "target": ["x", "x"],
        "plan_period": ["year", "year"],
        "inflation_multiplier": [1.0, 1.0],
        "seasonality": ["", ""],
    })


def _allocations_df():
    return pd.DataFrame({
        "option": ["predefined", "predefined", "zero", "zero"],
        "scope_name": ["s1", "s2", "s1", "s2"],
        "item": ["bdg_tv", "bdg_tv", "bdg_tv", "bdg_tv"],
        "val": [50, 50, 0, 0],
    })


def _media_prices_df():
    return pd.DataFrame({
        "scope_name": ["s1", "s1"],
        "item": ["bdg_tv", "bdg_digital"],
        "multiplier": [1.1, 1.2],
    })


def _seasonality_df():
    return pd.DataFrame({
        "scope_name": ["s1", "s1", "s2", "s2"],
        "month": [1, 2, 1, 2],
        "tv": [0.5, 0.5, 0.5, 0.5],
    })


def _translation_df(var="tv"):
    return pd.DataFrame({
        "item": ["bdg_tv", "bdg_digital"],
        "function": ["basic", "basic"],
        "variable": [var, var + "_digital"],
        "base_price": [1.0, 2.0],
    })


def _write_options(tmp_path, include_scopes=True, include_allocations=True,
                   include_media=True, include_ssn=True, include_translation=True,
                   sheets=("a", "b"), single_translation=False):
    xlsx_path = tmp_path / "options.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        if include_scopes:
            _scopes_df().to_excel(writer, sheet_name="scopes", index=False)
        if include_allocations:
            _allocations_df().to_excel(writer, sheet_name="allocations", index=False)
        if include_media:
            _media_prices_df().to_excel(writer, sheet_name="media_prices", index=False)
        if include_ssn:
            _seasonality_df().to_excel(writer, sheet_name="seasonality", index=False)
        if include_translation:
            if single_translation:
                _translation_df().to_excel(writer, sheet_name="translation", index=False)
            else:
                for n in sheets:
                    _translation_df("tv_" + n).to_excel(
                        writer, sheet_name=f"translation_{n}", index=False)
    return str(xlsx_path)


def _node(name="", trans_dict=None, static=False):
    ds = _basic_ds()
    if trans_dict is None:
        return ModelNode(model=MockModel(), ds=ds, static=static)
    return ModelNode(model=MockModel(), ds=ds, trans_dict=trans_dict, static=static)


class TestLoadContext:

    def test_populates_all_fields(self, tmp_path):
        path = _write_options(tmp_path)
        ctx = load_context(path)

        assert isinstance(ctx, ComputeContext)
        assert ctx.path == path
        assert isinstance(ctx.scopes, list)
        assert len(ctx.scopes) == 2
        assert all(isinstance(s, OptionScope) for s in ctx.scopes)
        assert isinstance(ctx.options, dict)
        assert set(ctx.options.keys()) == {"predefined", "zero"}
        assert all(isinstance(o, Option) for o in ctx.options.values())
        assert set(ctx.ssn_dict.keys()) == {"s1", "s2"}
        assert ctx.media_prices == {"s1": {"bdg_tv": 1.1, "bdg_digital": 1.2}}
        assert set(ctx.trans_dicts.keys()) == {"a", "b"}
        assert ctx.trans_dicts["a"]["function"]["bdg_tv"] == "basic"
        assert ctx.trans_dicts["b"]["variable"]["bdg_digital"] == "tv_b_digital"

    def test_plain_translation_sheet_keyed_empty(self, tmp_path):
        path = _write_options(tmp_path, single_translation=True)
        ctx = load_context(path)
        assert set(ctx.trans_dicts.keys()) == {""}
        assert ctx.trans_dicts[""]["function"]["bdg_tv"] == "basic"

    def test_missing_scopes_sheet_raises(self, tmp_path):
        path = _write_options(tmp_path, include_scopes=False)
        with pytest.raises(FermatricaError, match="scopes"):
            load_context(path)

    def test_missing_allocations_sheet_raises(self, tmp_path):
        path = _write_options(tmp_path, include_allocations=False)
        with pytest.raises(FermatricaError, match="allocations"):
            load_context(path)

    def test_optional_sheets_absent_give_empty_dicts(self, tmp_path):
        path = _write_options(tmp_path, include_media=False, include_ssn=False)
        ctx = load_context(path)
        assert ctx.media_prices == {}
        assert ctx.ssn_dict == {}

    def test_repr_is_compact(self, tmp_path):
        path = _write_options(tmp_path)
        ctx = load_context(path)
        s = repr(ctx)
        assert s.startswith("ComputeContext(")
        assert "'s1'" in s and "'predefined'" in s
        assert "trans_dicts=['a', 'b']" in s
        assert "basic" not in s

    def test_str_contains_sections(self, tmp_path):
        path = _write_options(tmp_path)
        ctx = load_context(path)
        s = str(ctx)
        assert s.startswith("ComputeContext: ")
        assert "s1: 2025-01-01 -> 2025-12-31" in s
        assert "Options: predefined, zero" in s
        assert "Seasonality: s1, s2" in s
        assert "Media prices: s1" in s
        assert "Translation sheets: a, b" in s

    def test_str_empty_collections_show_none(self, tmp_path):
        path = _write_options(tmp_path, include_media=False, include_ssn=False,
                               include_translation=False)
        ctx = load_context(path)
        s = str(ctx)
        assert "Seasonality: (none)" in s
        assert "Media prices: (none)" in s
        assert "Translation sheets: (none)" in s
        assert "Options: predefined, zero" in s


class TestMetaModelComputeContext:

    def test_wires_trans_ssn_media_prices(self, tmp_path):
        path = _write_options(tmp_path)
        ctx = load_context(path)
        nodes = {"a": _node("a"), "b": _node("b")}
        mm = MetaModel(nodes=nodes, edges=[("a", "b", None)], compute_context=ctx)

        assert not hasattr(mm, "scopes")
        assert mm.nodes["a"].model_rep.trans_dict == ctx.trans_dicts["a"]
        assert mm.nodes["b"].model_rep.trans_dict == ctx.trans_dicts["b"]
        assert mm.nodes["a"].model_rep.ssn_dict == ctx.ssn_dict
        assert mm.nodes["b"].model_rep.media_prices == ctx.media_prices

    def test_path_form_equivalence(self, tmp_path):
        path = _write_options(tmp_path)
        nodes = {"a": _node("a"), "b": _node("b")}
        mm = MetaModel(nodes=nodes, edges=[("a", "b", None)], compute_context=path)
        ctx = load_context(path)
        assert mm.nodes["a"].model_rep.trans_dict == ctx.trans_dicts["a"]
        assert mm.nodes["b"].model_rep.trans_dict == ctx.trans_dicts["b"]

    def test_single_translation_sheet_fallback(self, tmp_path):
        path = _write_options(tmp_path, single_translation=True)
        ctx = load_context(path)
        nodes = {"a": _node("a"), "b": _node("b")}
        mm = MetaModel(nodes=nodes, edges=[("a", "b", None)], compute_context=ctx)

        shared = next(iter(ctx.trans_dicts.values()))
        assert mm.nodes["a"].model_rep.trans_dict == shared
        assert mm.nodes["b"].model_rep.trans_dict == shared

    def test_explicit_trans_dict_preserved(self, tmp_path):
        path = _write_options(tmp_path)
        ctx = load_context(path)
        explicit = {"function": {"bdg_tv": "basic"}, "variable": {"bdg_tv": "custom"}}
        nodes = {"a": _node("a", trans_dict=explicit), "b": _node("b")}
        mm = MetaModel(nodes=nodes, edges=[("a", "b", None)], compute_context=ctx)

        assert mm.nodes["a"].model_rep.trans_dict == explicit
        assert mm.nodes["b"].model_rep.trans_dict == ctx.trans_dicts["b"]

    def test_unmatched_node_raises_with_multiple_sheets(self, tmp_path):
        path = _write_options(tmp_path)
        ctx = load_context(path)
        nodes = {"z": _node("z"), "b": _node("b")}
        with pytest.raises(FermatricaError, match="translation"):
            MetaModel(nodes=nodes, edges=[("z", "b", None)], compute_context=ctx)

    def test_unmatched_node_ok_with_single_sheet(self, tmp_path):
        path = _write_options(tmp_path, single_translation=True)
        ctx = load_context(path)
        nodes = {"z": _node("z"), "b": _node("b")}
        mm = MetaModel(nodes=nodes, edges=[("z", "b", None)], compute_context=ctx)
        shared = next(iter(ctx.trans_dicts.values()))
        assert mm.nodes["z"].model_rep.trans_dict == shared
        assert mm.nodes["b"].model_rep.trans_dict == shared

    def test_static_node_skips_trans_dict_wiring(self, tmp_path):
        path = _write_options(tmp_path)
        ctx = load_context(path)
        nodes = {"a": _node("a", static=True), "b": _node("b")}
        mm = MetaModel(nodes=nodes, edges=[("a", "b", None)], compute_context=ctx)

        assert mm.nodes["a"].model_rep.trans_dict is None
        assert mm.nodes["b"].model_rep.trans_dict == ctx.trans_dicts["b"]

    def test_static_node_without_matching_sheet_does_not_raise(self, tmp_path):
        path = _write_options(tmp_path)
        ctx = load_context(path)
        nodes = {"z": _node("z", static=True), "b": _node("b")}
        mm = MetaModel(nodes=nodes, edges=[("z", "b", None)], compute_context=ctx)

        assert mm.nodes["z"].model_rep.trans_dict is None
        assert mm.nodes["b"].model_rep.trans_dict == ctx.trans_dicts["b"]

    def test_single_sheet_fallback_skips_static_node(self, tmp_path):
        path = _write_options(tmp_path, single_translation=True)
        ctx = load_context(path)
        nodes = {"a": _node("a", static=True), "b": _node("b")}
        mm = MetaModel(nodes=nodes, edges=[("a", "b", None)], compute_context=ctx)

        shared = next(iter(ctx.trans_dicts.values()))
        assert mm.nodes["a"].model_rep.trans_dict is None
        assert mm.nodes["b"].model_rep.trans_dict == shared

    def test_explicit_ssn_preserved(self, tmp_path):
        path = _write_options(tmp_path)
        ctx = load_context(path)
        ds = _basic_ds()
        mr = ModelRep(ds)
        mr.ssn_dict = {"custom": pd.DataFrame({"month": [1], "tv": [1.0]})}
        node_a = ModelNode(model=MockModel(), ds=ds, model_rep=mr)
        node_b = _node("b")
        mm = MetaModel(nodes={"a": node_a, "b": node_b},
                       edges=[("a", "b", None)], compute_context=ctx)

        assert set(mm.nodes["a"].model_rep.ssn_dict.keys()) == {"custom"}
        pd.testing.assert_frame_equal(
            mm.nodes["a"].model_rep.ssn_dict["custom"],
            pd.DataFrame({"month": [1], "tv": [1.0]}),
        )
        assert mm.nodes["b"].model_rep.ssn_dict == ctx.ssn_dict

    def test_compute_raises_without_any_translation_sheet(self, tmp_path):
        path = _write_options(tmp_path, include_translation=False)
        ctx = load_context(path)
        assert ctx.trans_dicts == {}

        nodes = {"a": _node("a"), "b": _node("b")}
        mm = MetaModel(nodes=nodes, edges=[("a", "b", None)], compute_context=ctx)
        assert mm.nodes["a"].model_rep.trans_dict is None
        assert mm.nodes["b"].model_rep.trans_dict is None

        with pytest.raises(FermatricaError, match="trans_dict"):
            mm.compute(ctx.options["predefined"], scopes=ctx.scopes)

    def test_compute_raises_when_scopes_missing(self):
        mm = MetaModel(nodes=_node())
        option = Option(name="opt1", budget_data={"s1": {"bdg_tv": 10}})
        with pytest.raises(FermatricaError, match="scopes"):
            mm.compute(option)

    def test_compute_requires_explicit_scopes(self, monkeypatch):
        mm = MetaModel(nodes=_node())
        option = Option(name="opt1", budget_data={"s1": {"bdg_tv": 10}})
        scopes = load_context_path_scopes = None

        from fermatrica_rep import meta_model as mm_mod
        translate_calls = []

        def fake_translate(self, option, scopes, allow_past=False, if_exact=False):
            translate_calls.append(scopes)
            self.ds = self.ds.copy()

        monkeypatch.setattr(mm_mod.meta_model.ModelNode, "translate", fake_translate)
        monkeypatch.setattr(mm_mod.meta_model.ModelNode, "compute_after_translate",
                            lambda self, option, scopes=None, return_full=False: {"s1": {}})

        result = mm.compute(option, scopes=["s1"])
        assert translate_calls and translate_calls[0] == ["s1"]
        assert result["model"] == {"s1": {}}
