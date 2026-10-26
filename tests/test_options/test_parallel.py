"""Unit tests for parallel.py: _last_node_name, compute_multi, compute_grid."""


import logging

import numpy as np
import pandas as pd
import pytest

from fermatrica.basics.basics import FermatricaError
from fermatrica_rep.meta_model.meta_model import MetaModel
from fermatrica_rep.meta_model.node import ModelNode
from fermatrica_rep.meta_model.model_rep import ModelRep
from fermatrica_rep.options.option import Option
from fermatrica_rep.options.scope import OptionScope
from fermatrica_rep.options.parallel import (
    _last_node_name,
    _strip_meta_model_callables,
    compute_multi,
    compute_grid,
    _summarize_grid_results,
)
from fermatrica_rep.options.calc_multi import opt_grid


def _synthetic_ds():
    n = 12
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n, freq="MS"),
        "superbrand": ["brand_x"] * n,
        "bs_key": [1] * n,
        "listed": [4] * n,
        "tv_ots": [0.0] * n,
        "digital_imp": [0.0] * n,
    })


def _sample_trans_dict():
    return pd.DataFrame({
        "max_budg": [100, 50],
        "function": ["basic", "basic"],
        "variable": ["tv_ots", "digital_imp"],
        "base_price": [0.05, 0.02],
    }, index=pd.Index(["bdg_tv", "bdg_digital"], name="item"))


class _MockModel:
    obj = None

    class Conf:
        price_var = "price"
        summary_type = "sum"

    def __init__(self):
        self.obj = self.Obj()
        self.conf = self.Conf()

    class Obj:
        transform_lhs_fn = None
        custom_predict_fn = None


def _build_meta_model():
    ds = _synthetic_ds()
    trans_dict = _sample_trans_dict()
    model_rep = ModelRep(ds, trans_dict=trans_dict, language="english")
    model = _MockModel()
    node = ModelNode(model=model, ds=ds, model_rep=model_rep)
    return MetaModel(nodes=node)


def _build_scope():
    return OptionScope(
        scope_name="test_scope",
        date_start="2024-01-01",
        date_end="2024-12-31",
        ref_date_start="2023-01-01",
        ref_date_end="2023-12-31",
        filters={"superbrand": ["brand_x"]},
        plan_period="exact",
    )


class TestLastNodeName:

    def test_single_node_returns_model(self):
        mm = _build_meta_model()
        assert _last_node_name(mm) == "model"

    def test_multi_node_chain(self):
        ds = _synthetic_ds()
        trans_dict = _sample_trans_dict()
        model_rep = ModelRep(ds, trans_dict=trans_dict, language="english")
        n1 = ModelNode(model=_MockModel(), ds=ds, model_rep=model_rep)
        n2 = ModelNode(model=_MockModel(), ds=ds.copy(), model_rep=model_rep)
        mm = MetaModel(nodes={"awareness": n1, "sales": n2}, edges=[("awareness", "sales", None)])
        assert _last_node_name(mm) == "sales"

    def test_parallel_sinks_raises(self):
        ds = _synthetic_ds()
        trans_dict = _sample_trans_dict()
        model_rep = ModelRep(ds, trans_dict=trans_dict, language="english")
        n0 = ModelNode(model=_MockModel(), ds=ds.copy(), model_rep=model_rep)
        n1 = ModelNode(model=_MockModel(), ds=ds, model_rep=model_rep)
        n2 = ModelNode(model=_MockModel(), ds=ds.copy(), model_rep=model_rep)
        mm = MetaModel(
            nodes={"source": n0, "brand_a": n1, "brand_b": n2},
            edges=[("source", "brand_a", None), ("source", "brand_b", None)],
        )
        with pytest.raises(FermatricaError, match="sink nodes"):
            _last_node_name(mm)


class TestStripMetaModelCallables:

    def test_strips_callables(self):
        mm = _build_meta_model()
        mm.nodes["model"].model.obj.transform_lhs_fn = lambda x: x
        mm.nodes["model"].model.obj.custom_predict_fn = lambda x: x

        stripped = _strip_meta_model_callables(mm)

        for node in stripped.nodes.values():
            assert node.model.obj.transform_lhs_fn is None
            assert node.model.obj.custom_predict_fn is None

    def test_picklable_after_strip(self):
        import pickle
        mm = _build_meta_model()
        stripped = _strip_meta_model_callables(mm)
        data = pickle.dumps(stripped)
        assert data is not None


class TestComputeMany:

    def test_sequential_execution(self):
        mm = _build_meta_model()
        scope = _build_scope()

        def fake_compute(self, option, scopes=None, allow_past=False, if_exact=False, *, return_full=False):
            return {"model": {scope.scope_name: {"pred_exact_vol": 42}}}

        original = mm.compute
        mm.compute = fake_compute.__get__(mm, MetaModel)

        try:
            option = Option(name="opt1", budget_data={scope.scope_name: {"bdg_tv": 10}})
            results = compute_multi(mm, [option], [scope], cores=1, if_exact=True)
        finally:
            mm.compute = original

        assert len(results) == 1
        assert results[0] is not None
        assert results[0]["model"][scope.scope_name]["pred_exact_vol"] == 42

    def test_failed_option_yields_none(self):
        mm = _build_meta_model()
        scope = _build_scope()

        def fail_compute(self, option, scopes=None, allow_past=False, if_exact=False, *, return_full=False):
            raise RuntimeError("simulated failure")

        original = mm.compute
        mm.compute = fail_compute.__get__(mm, MetaModel)

        try:
            option = Option(name="opt1", budget_data={scope.scope_name: {"bdg_tv": 10}})
            results = compute_multi(mm, [option], [scope], cores=1, if_exact=True)
        finally:
            mm.compute = original

        assert results[0] is None

    def test_dict_input_returns_dict(self):
        mm = _build_meta_model()
        scope = _build_scope()

        def fake_compute(self, option, scopes=None, allow_past=False, if_exact=False, *, return_full=False):
            return {"model": {scope.scope_name: {"pred_exact_vol": 42}}}

        original = mm.compute
        mm.compute = fake_compute.__get__(mm, MetaModel)

        try:
            opts = {
                "opt_a": Option(name="opt_a", budget_data={scope.scope_name: {"bdg_tv": 10}}),
                "opt_b": Option(name="opt_b", budget_data={scope.scope_name: {"bdg_tv": 20}}),
            }
            results = compute_multi(mm, opts, [scope], cores=1, if_exact=True)
        finally:
            mm.compute = original

        assert isinstance(results, dict)
        assert list(results.keys()) == ["opt_a", "opt_b"]
        assert results["opt_a"] is not None
        assert results["opt_b"] is not None
        assert results["opt_a"]["model"][scope.scope_name]["pred_exact_vol"] == 42
        assert results["opt_b"]["model"][scope.scope_name]["pred_exact_vol"] == 42

    def test_empty_dict_returns_empty_dict(self):
        mm = _build_meta_model()
        scope = _build_scope()
        results = compute_multi(mm, {}, [scope], cores=1, if_exact=True)

        assert isinstance(results, dict)
        assert results == {}


class TestSummarizeGridResults:

    def test_summarize_flat_keys(self):
        mm = _build_meta_model()
        scope = _build_scope()

        results = [
            {"model": {scope.scope_name:
                {"pred_exact_vol": 100, "pred_exact_val": 150,
                 "pred_long_vol": 200, "pred_long_val": 250}
            }},
            {"model": {scope.scope_name:
                {"pred_exact_vol": 300, "pred_exact_val": 350,
                 "pred_long_vol": 400, "pred_long_val": 450}
            }},
        ]

        td = _sample_trans_dict()
        grid = opt_grid(td, budget_step=10)

        result = _summarize_grid_results(results, grid, mm, grid_type="curves")

        assert isinstance(result, dict)
        assert "model" in result
        df = result["model"]
        assert isinstance(df, pd.DataFrame)
        assert "pred_exact_vol" in df.columns
        assert "pred_long_vol" in df.columns
        assert "bdg" in df.columns
        assert "option" in df.columns
        assert len(df) == 2
        assert df["pred_exact_vol"].iloc[0] == 100

    def test_empty_results(self):
        mm = _build_meta_model()
        td = _sample_trans_dict()
        grid = opt_grid(td, budget_step=10)

        result = _summarize_grid_results([], grid, mm)
        assert isinstance(result, dict)
        assert len(result) == 0


class TestComputeGrid:

    def test_returns_dataframe(self, monkeypatch):
        mm = _build_meta_model()
        scope = _build_scope()
        td = _sample_trans_dict()

        def mock_compute_multi(meta_model, options, scopes, cores=None,
                              allow_past=False, if_exact=False):
            if isinstance(options, dict):
                option_names = list(options.keys())
                n = len(option_names)
            else:
                option_names = None
                n = len(options)
            raw = [
                {"model": {scope.scope_name:
                    {"pred_exact_vol": float(100 + i * 10),
                     "pred_exact_val": float(150 + i * 10),
                     "pred_long_vol": float(200 + i * 10),
                     "pred_long_val": float(250 + i * 10)}
                }}
                for i in range(n)
            ]
            if option_names is not None:
                return dict(zip(option_names, raw))
            return raw

        monkeypatch.setattr(
            "fermatrica_rep.options.parallel.compute_multi",
            mock_compute_multi,
        )

        result = compute_grid(
            meta_model=mm,
            scopes=[scope],
            translation=td,
            budget_step=10,
            cores=1,
            if_exact=True,
        )

        assert isinstance(result, dict)
        assert "model" in result
        df = result["model"]
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        expected = {"pred_exact_vol", "pred_exact_val", "pred_long_vol",
                    "pred_long_val", "bdg", "option"}
        assert expected.issubset(set(df.columns))

    def test_zero_option_present(self, monkeypatch):
        mm = _build_meta_model()
        scope = _build_scope()
        td = _sample_trans_dict()

        def mock_compute_multi(meta_model, options, scopes, cores=None,
                              allow_past=False, if_exact=False):
            if isinstance(options, dict):
                option_names = list(options.keys())
                n = len(option_names)
            else:
                option_names = None
                n = len(options)
            raw = [
                {"model": {scope.scope_name:
                    {"pred_exact_vol": 100.0, "pred_exact_val": 150.0,
                     "pred_long_vol": 200.0, "pred_long_val": 250.0}
                }}
                for _ in range(n)
            ]
            if option_names is not None:
                return dict(zip(option_names, raw))
            return raw

        monkeypatch.setattr(
            "fermatrica_rep.options.parallel.compute_multi",
            mock_compute_multi,
        )

        result = compute_grid(
            meta_model=mm,
            scopes=[scope],
            translation=td,
            budget_step=10,
            cores=1,
            if_exact=True,
        )

        assert "zero" in result["model"]["option"].values


def _make_scope(name, year, filters):
    return OptionScope(
        scope_name=name,
        date_start=f"{year}-01-01",
        date_end=f"{year}-12-31",
        ref_date_start=f"{year - 1}-01-01",
        ref_date_end=f"{year - 1}-12-31",
        filters=filters,
        plan_period="year",
    )


def _leaf_parent_scopes():
    leaf = _make_scope(
        "cairo_2026", 2026,
        {"superbrand": ["brand_1"], "region": ["Cairo"]},
    )
    parent = _make_scope(
        "brand_1_2026", 2026,
        {"superbrand": ["brand_1"]},
    )
    return leaf, parent


def _grid_items(grid, idx):
    row = grid.iloc[idx]
    return {
        col: float(row[col])
        for col in grid.columns
        if col.startswith("bdg_")
    }


def _capture_mock(monkeypatch, captured):
    def mock_compute_multi(meta_model, options, scopes, cores=None,
                           allow_past=False, if_exact=False):
        captured["options"] = options
        if isinstance(options, dict):
            option_names = list(options.keys())
            n = len(option_names)
        else:
            option_names = None
            n = len(options)
        raw = [
            {"model": {scopes[0].scope_name:
                {"pred_exact_vol": float(100 + i * 10),
                 "pred_exact_val": float(150 + i * 10),
                 "pred_long_vol": float(200 + i * 10),
                 "pred_long_val": float(250 + i * 10)}
            }}
            for i in range(n)
        ]
        if option_names is not None:
            return dict(zip(option_names, raw))
        return raw

    monkeypatch.setattr(
        "fermatrica_rep.options.parallel.compute_multi",
        mock_compute_multi,
    )


class TestComputeGridBackgroundOption:

    def test_raw_dict_background_option_merged(self, monkeypatch):
        mm = _build_meta_model()
        leaf, parent = _leaf_parent_scopes()
        scopes = [leaf, parent]
        td = _sample_trans_dict()
        captured = {}
        _capture_mock(monkeypatch, captured)

        compute_grid(
            meta_model=mm,
            scopes=scopes,
            translation=td,
            budget_step=10,
            cores=1,
            if_exact=True,
            background_option={"brand_1_2026": {"bdg_reg_tv": 300}},
        )

        options = captured["options"]
        grid = opt_grid(td, budget_step=10)
        assert len(options) == len(grid)
        for i, opt in enumerate(options.values()):
            expected_items = _grid_items(grid, i)
            assert opt.budget_data["cairo_2026"] == expected_items
            expected_parent = dict(expected_items)
            expected_parent["bdg_reg_tv"] = 300.0
            assert opt.budget_data["brand_1_2026"] == expected_parent

    def test_option_background_option_cross_scope_diff(self, monkeypatch):
        mm = _build_meta_model()
        leaf, parent = _leaf_parent_scopes()
        scopes = [leaf, parent]
        td = _sample_trans_dict()
        captured = {}
        _capture_mock(monkeypatch, captured)

        base = Option(
            name="predefined",
            budget_data={
                "cairo_2026": {"bdg_tv": 200, "bdg_reg_tv": 100},
                "brand_1_2026": {"bdg_tv": 200, "bdg_reg_tv": 400},
            },
        )

        compute_grid(
            meta_model=mm,
            scopes=scopes,
            translation=td,
            budget_step=10,
            cores=1,
            if_exact=True,
            background_option=base,
        )

        options = captured["options"]
        grid = opt_grid(td, budget_step=10)
        assert len(options) == len(grid)
        for i, opt in enumerate(options.values()):
            expected_items = _grid_items(grid, i)
            assert opt.budget_data["cairo_2026"] == expected_items
            expected_parent = dict(expected_items)
            expected_parent["bdg_reg_tv"] = expected_items.get("bdg_reg_tv", 0.0) + 300.0
            assert opt.budget_data["brand_1_2026"] == expected_parent

    def test_background_option_scope_not_in_scopes_raises(self, monkeypatch):
        mm = _build_meta_model()
        leaf, parent = _leaf_parent_scopes()
        scopes = [leaf, parent]
        td = _sample_trans_dict()
        _capture_mock(monkeypatch, {})

        with pytest.raises(FermatricaError, match="not among the scopes"):
            compute_grid(
                meta_model=mm,
                scopes=scopes,
                translation=td,
                budget_step=10,
                cores=1,
                if_exact=True,
                background_option={"unknown_scope": {"bdg_tv": 100}},
            )

    def test_background_option_non_numeric_item_raises(self, monkeypatch):
        mm = _build_meta_model()
        leaf, parent = _leaf_parent_scopes()
        scopes = [leaf, parent]
        td = _sample_trans_dict()
        _capture_mock(monkeypatch, {})

        with pytest.raises(FermatricaError, match="must be numeric"):
            compute_grid(
                meta_model=mm,
                scopes=scopes,
                translation=td,
                budget_step=10,
                cores=1,
                if_exact=True,
                background_option={"brand_1_2026": {"bdg_reg_tv": "300"}},
            )

    def test_multiple_leaves_raises(self, monkeypatch):
        mm = _build_meta_model()
        parent = _make_scope("brand_1_2026", 2026, {"superbrand": ["brand_1"]})
        leaf1 = _make_scope(
            "cairo_2026", 2026,
            {"superbrand": ["brand_1"], "region": ["Cairo"]},
        )
        leaf2 = _make_scope(
            "alex_2026", 2026,
            {"superbrand": ["brand_1"], "region": ["Alex"]},
        )
        scopes = [parent, leaf1, leaf2]
        td = _sample_trans_dict()
        _capture_mock(monkeypatch, {})

        base = Option(
            name="predefined",
            budget_data={
                "brand_1_2026": {"bdg_tv": 200, "bdg_reg_tv": 400},
                "cairo_2026": {"bdg_tv": 200, "bdg_reg_tv": 100},
                "alex_2026": {"bdg_tv": 200, "bdg_reg_tv": 100},
            },
        )

        with pytest.raises(FermatricaError, match="explicit background_option"):
            compute_grid(
                meta_model=mm,
                scopes=scopes,
                translation=td,
                budget_step=10,
                cores=1,
                if_exact=True,
                background_option=base,
            )

    def test_negative_diff_warns_and_drops(self, monkeypatch, caplog):
        mm = _build_meta_model()
        leaf, parent = _leaf_parent_scopes()
        scopes = [leaf, parent]
        td = _sample_trans_dict()
        captured = {}
        _capture_mock(monkeypatch, captured)

        base = Option(
            name="predefined",
            budget_data={
                "cairo_2026": {"bdg_tv": 200, "bdg_reg_tv": 500},
                "brand_1_2026": {"bdg_tv": 200, "bdg_reg_tv": 400},
            },
        )

        with caplog.at_level(logging.WARNING):
            compute_grid(
                meta_model=mm,
                scopes=scopes,
                translation=td,
                budget_step=10,
                cores=1,
                if_exact=True,
                background_option=base,
            )

        assert any("spends less" in rec.message for rec in caplog.records)

        options = captured["options"]
        grid = opt_grid(td, budget_step=10)
        assert len(options) == len(grid)
        for i, opt in enumerate(options.values()):
            expected_items = _grid_items(grid, i)
            assert opt.budget_data["cairo_2026"] == expected_items
            assert opt.budget_data["brand_1_2026"] == expected_items

    def test_summarize_two_nodes(self):
        ds = _synthetic_ds()
        trans_dict = _sample_trans_dict()
        model_rep = ModelRep(ds, trans_dict=trans_dict, language="english")
        n_sales = ModelNode(model=_MockModel(), ds=ds, model_rep=model_rep)
        n_uba = ModelNode(model=_MockModel(), ds=ds.copy(), model_rep=model_rep)
        mm = MetaModel(
            nodes={"sales": n_sales, "uba": n_uba},
            edges=[("uba", "sales", None)],
        )

        scope_sales = _build_scope()
        scope_uba = _make_scope(
            "national_2026", 2026,
            {"superbrand": ["brand_1"]},
        )

        results = [
            {
                "sales": {scope_sales.scope_name:
                    {"pred_exact_vol": 100, "pred_exact_val": 150,
                     "pred_long_vol": 200, "pred_long_val": 250}},
                "uba": {scope_uba.scope_name:
                    {"pred_exact_vol": 50, "pred_exact_val": 60,
                     "pred_long_vol": 70, "pred_long_val": 80}},
            },
            {
                "sales": {scope_sales.scope_name:
                    {"pred_exact_vol": 110, "pred_exact_val": 160,
                     "pred_long_vol": 210, "pred_long_val": 260}},
                "uba": {scope_uba.scope_name:
                    {"pred_exact_vol": 55, "pred_exact_val": 65,
                     "pred_long_vol": 75, "pred_long_val": 85}},
            },
        ]

        td = _sample_trans_dict()
        grid = opt_grid(td, budget_step=10)

        result = _summarize_grid_results(results, grid, mm, grid_type="curves")

        assert isinstance(result, dict)
        assert set(result.keys()) == {"sales", "uba"}
        for node_name, df in result.items():
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 2
            assert "pred_exact_vol" in df.columns
            assert "bdg" in df.columns
            assert "option" in df.columns
