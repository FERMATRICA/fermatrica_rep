"""Unit tests for OptionScope, Option, ModelNode, and MetaModel (DAG-based).

All tests use synthetic data — no external sample models required.
"""


import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.formula.api import ols
from fermatrica_utils import select_eff

from fermatrica_rep.options.scope import OptionScope
from fermatrica_rep.options.option import Option
from fermatrica_rep.meta_model.node import ModelNode, _pack_results
from fermatrica_rep.meta_model.meta_model import _convert_to_metamodel
from fermatrica_rep.meta_model.model_rep import ModelRep
from fermatrica_rep.meta_model.meta_model import MetaModel
from fermatrica_rep.analysis.decomposition import extract_effect
from fermatrica_rep.analysis.waterfall import waterfall_plot
from fermatrica_rep.analysis.fit import fit_main_data, fit_mult_plot_vol


class TestOptionScope:

    def test_basic_construction(self):
        scope = OptionScope(
            scope_name="brandx_2025",
            date_start="2025-01-01",
            date_end="2025-12-31",
            ref_date_start="2024-01-01",
            ref_date_end="2024-12-31",
            filters={"superbrand": ["BrandX"]},
            plan_period="year",
            inflation_multiplier=1.04,
        )
        assert scope.scope_name == "brandx_2025"
        assert scope.plan_period == "year"
        assert scope.inflation_multiplier == 1.04
        assert pd.notna(scope.date_start)
        assert pd.notna(scope.date_end)

    def test_build_mask_date_range(self):
        scope = OptionScope(
            scope_name="test",
            date_start="2025-06-01",
            date_end="2025-06-30",
            ref_date_start="2024-01-01",
            ref_date_end="2024-12-31",
        )
        ds = pd.DataFrame({
            "date": pd.date_range("2025-01-01", "2025-12-31", freq="D"),
            "listed": [4] * 365,
        })
        mask = scope.build_mask(ds, allow_past=True)
        assert mask.sum() == 30

    def test_build_mask_entity_filter(self):
        scope = OptionScope(
            scope_name="test",
            date_start="2025-01-01",
            date_end="2025-12-31",
            ref_date_start="2024-01-01",
            ref_date_end="2024-12-31",
            filters={"superbrand": ["BrandX"]},
        )
        ds = pd.DataFrame({
            "date": pd.date_range("2025-01-01", periods=10, freq="MS"),
            "superbrand": ["BrandX"] * 5 + ["BrandY"] * 5,
            "listed": [4] * 10,
        })
        mask = scope.build_mask(ds, allow_past=True)
        assert mask.sum() == 5

    def test_build_mask_disallow_past(self):
        scope = OptionScope(
            scope_name="test",
            date_start="2025-01-01",
            date_end="2025-12-31",
            ref_date_start="2024-01-01",
            ref_date_end="2024-12-31",
        )
        ds = pd.DataFrame({
            "date": pd.date_range("2025-06-01", periods=10, freq="MS"),
            "listed": [2, 3, 2, 3, 4, 4, 4, 4, 4, 4],
        })
        mask = scope.build_mask(ds, allow_past=False)
        assert mask.sum() == 3

    def test_build_entity_mask(self):
        scope = OptionScope(
            scope_name="test",
            date_start="2025-01-01",
            date_end="2025-12-31",
            ref_date_start="2024-01-01",
            ref_date_end="2024-12-31",
            filters={"superbrand": ["A", "B"]},
        )
        ds = pd.DataFrame({
            "date": pd.date_range("2025-01-01", periods=6, freq="MS"),
            "superbrand": ["A", "B", "C", "A", "B", "C"],
            "listed": [4] * 6,
        })
        mask = scope.build_entity_mask(ds)
        assert mask.sum() == 4

    def test_empty_scope_name_raises(self):
        from fermatrica.basics.basics import FermatricaError

        with pytest.raises(FermatricaError):
            OptionScope(
                scope_name="",
                date_start="2025-01-01",
                date_end="2025-12-31",
                ref_date_start="2024-01-01",
                ref_date_end="2024-12-31",
            )

    def test_filters_multi_column(self):
        scope = OptionScope(
            scope_name="test",
            date_start="2025-01-01",
            date_end="2025-12-31",
            ref_date_start="2024-01-01",
            ref_date_end="2024-12-31",
            filters={"superbrand": ["A"], "region": ["NY"]},
        )
        ds = pd.DataFrame({
            "date": pd.date_range("2025-01-01", periods=6, freq="MS"),
            "superbrand": ["A", "A", "B", "A", "A", "B"],
            "region": ["NY", "LA", "NY", "NY", "NY", "LA"],
            "listed": [4] * 6,
        })
        mask = scope.build_mask(ds, allow_past=True)
        assert mask.sum() == 3

    def test_plan_period_values(self):
        for pp in ("exact", "hy", "quarter"):
            scope = OptionScope(
                scope_name="test",
                date_start="2025-01-01",
                date_end="2025-12-31",
                ref_date_start="2024-01-01",
                ref_date_end="2024-12-31",
                plan_period=pp,
            )
            assert scope.plan_period == pp

    def test_ref_date_end_is_required(self):
        with pytest.raises(TypeError):
            OptionScope(
                scope_name="test",
                date_start="2025-01-01",
                date_end="2025-12-31",
                ref_date_start="2024-01-01",
            )


class TestOption:

    def test_basic_construction(self):
        opt = Option(name="option_1", default_scope="brandx_2025")
        assert opt.name == "option_1"
        assert opt.default_scope == "brandx_2025"
        assert opt.all_scopes() == []

    def test_budget_data_direct(self):
        opt = Option(
            name="option_1",
            budget_data={
                "brandx_2025": {"bdg_nat_tv": 100, "bdg_digital": 50},
            },
        )
        assert opt.budget_for_scope("brandx_2025") == {
            "bdg_nat_tv": 100,
            "bdg_digital": 50,
        }
        assert opt.all_scopes() == ["brandx_2025"]

    def test_iter_scopes(self):
        opt = Option(
            name="option_1",
            budget_data={
                "scope_a": {"bdg_tv": 10},
                "scope_b": {"bdg_digital": 20},
            },
        )
        scopes = list(opt.iter_scopes())
        assert len(scopes) == 2
        names = [s[0] for s in scopes]
        assert "scope_a" in names
        assert "scope_b" in names

    def test_empty_budget_for_unknown_scope(self):
        opt = Option(name="test")
        assert opt.budget_for_scope("nonexistent") == {}


class TestModelNode:

    @pytest.fixture
    def mock_model(self):
        class MockConf:
            price_var = None
            summary_type = "sum"
            params: dict = {}

        class MockObj:
            models = {"main": None}

        class MockModel:
            def __init__(self):
                self.conf = MockConf()
                self.obj = MockObj()

        return MockModel()

    @pytest.fixture
    def basic_ds(self):
        return pd.DataFrame({
            "date": pd.date_range("2025-01-01", periods=30, freq="D"),
            "listed": [4] * 30,
            "predicted": [100.0] * 30,
            "observed": [90.0] * 30,
        })

    @pytest.fixture
    def basic_model_rep(self, basic_ds):
        return ModelRep(basic_ds, language="english")

    def test_node_construction(self, mock_model, basic_ds, basic_model_rep):
        node = ModelNode(
            model=mock_model,
            ds=basic_ds,
            model_rep=basic_model_rep,
        )
        assert node.model is mock_model
        assert len(node.ds) == 30

    def test_node_compute_no_scopes(self, mock_model, basic_ds, basic_model_rep, monkeypatch):
        node = ModelNode(
            model=mock_model,
            ds=basic_ds,
            model_rep=basic_model_rep,
        )
        monkeypatch.setattr(
            "fermatrica_rep.meta_model.node.transform",
            lambda **kw: (mock_model, basic_ds),
        )
        monkeypatch.setattr(
            "fermatrica_rep.meta_model.node.predict_ext",
            lambda model, ds: pd.DataFrame({
                "date": basic_ds["date"],
                "listed": basic_ds["listed"],
                "predicted": [100.0] * len(basic_ds),
                "observed": [90.0] * len(basic_ds),
            }),
        )
        opt = Option(name="test", budget_data={})
        results = node.compute(opt)
        assert results == {}

    def test_node_compute_with_scope(self, mock_model, basic_ds, basic_model_rep):
        basic_ds["predicted"] = 100.0
        basic_ds["observed"] = 90.0

        node = ModelNode(
            model=mock_model,
            ds=basic_ds,
            model_rep=basic_model_rep,
        )

        scope = OptionScope(
            scope_name="test_scope",
            date_start="2025-01-01",
            date_end="2025-01-31",
            ref_date_start="2024-12-01",
            ref_date_end="2024-12-31",
        )

        opt = Option(
            name="test",
            budget_data={"test_scope": {}},
        )

        matching = node._find_scope([scope], "test_scope")
        assert matching is not None
        assert matching.scope_name == "test_scope"

    def test_node_find_scope_none(self, mock_model, basic_ds, basic_model_rep):
        node = ModelNode(
            model=mock_model,
            ds=basic_ds,
            model_rep=basic_model_rep,
        )
        assert node._find_scope(None, "any") is None

    def test_node_find_scope_missing(self, mock_model, basic_ds, basic_model_rep):
        node = ModelNode(
            model=mock_model,
            ds=basic_ds,
            model_rep=basic_model_rep,
        )
        scope = OptionScope(
            scope_name="existing",
            date_start="2025-01-01",
            date_end="2025-12-31",
            ref_date_start="2024-01-01",
            ref_date_end="2024-12-31",
        )
        assert node._find_scope([scope], "missing") is None

    def test_node_from_model_path(self, mock_model, basic_ds, monkeypatch):
        def _mock_load_ext(path, missed_stop=False):
            return mock_model, basic_ds, ""

        monkeypatch.setattr(
            "fermatrica_rep.meta_model.node.model_load_ext",
            _mock_load_ext,
        )
        node = ModelNode(model_path="/fake/path/", language="english")
        assert node.model is mock_model
        assert len(node.ds) == 30
        assert node.model_rep.language == "english"

    def test_node_loads_media_prices_from_options_path(self, mock_model, basic_ds, tmp_path):
        options_path = tmp_path / "options.xlsx"
        media_prices = pd.DataFrame({
            "scope_name": ["scope_a"],
            "item": ["bdg_tv"],
            "multiplier": [1.5],
        })
        with pd.ExcelWriter(options_path, engine="openpyxl") as writer:
            media_prices.to_excel(writer, sheet_name="media_prices", index=False)

        node = ModelNode(
            model=mock_model,
            ds=basic_ds,
            options_path=str(options_path),
        )

        assert node.model_rep.media_prices == {"scope_a": {"bdg_tv": 1.5}}

    def test_node_preserves_explicit_media_prices(self, mock_model, basic_ds, basic_model_rep, tmp_path):
        options_path = tmp_path / "options.xlsx"
        media_prices = pd.DataFrame({
            "scope_name": ["scope_a"],
            "item": ["bdg_tv"],
            "multiplier": [1.5],
        })
        with pd.ExcelWriter(options_path, engine="openpyxl") as writer:
            media_prices.to_excel(writer, sheet_name="media_prices", index=False)

        basic_model_rep.media_prices = {"scope_a": {"bdg_tv": 3.0}}
        node = ModelNode(
            model=mock_model,
            ds=basic_ds,
            model_rep=basic_model_rep,
            options_path=str(options_path),
        )

        assert node.model_rep.media_prices == {"scope_a": {"bdg_tv": 3.0}}

    def test_node_from_model_path_with_explicit_ds(self, mock_model, basic_ds, monkeypatch):
        ds_override = basic_ds.iloc[:10].copy()

        def _mock_load_ext(path, missed_stop=False):
            return mock_model, basic_ds, ""

        monkeypatch.setattr(
            "fermatrica_rep.meta_model.node.model_load_ext",
            _mock_load_ext,
        )
        node = ModelNode(model_path="/fake/path/", ds=ds_override, language="english")
        assert node.model is mock_model
        assert len(node.ds) == 10
        assert node.model_rep.language == "english"

    def test_node_path_and_model_error(self, mock_model, basic_ds):
        from fermatrica.basics.basics import FermatricaError

        with pytest.raises(FermatricaError):
            ModelNode(model=mock_model, model_path="/fake/path/", ds=basic_ds)

    def test_node_model_none_with_ds_raises(self, basic_ds, basic_model_rep):
        from fermatrica.basics.basics import FermatricaError

        with pytest.raises(FermatricaError, match="requires a model"):
            ModelNode(model=None, ds=basic_ds, model_rep=basic_model_rep)


class TestMetaModelDAG:

    class _MockModel:
        def __init__(self, name):
            self.conf = self
            self.obj = self
            self.name = name
            self.price_var = None
            self.summary_type = "sum"

    def _mock_model_rep(self, ds):
        return ModelRep(ds, language="english")

    def test_topological_order_chain(self):
        ds = pd.DataFrame({
            "date": pd.date_range("2025-01-01", periods=10, freq="D"),
            "listed": [4] * 10,
            "predicted": [100.0] * 10,
            "observed": [90.0] * 10,
            "superbrand": ["BrandX"] * 10,
        })

        models = {
            "awareness": self._MockModel("awareness"),
            "bht": self._MockModel("bht"),
            "sales": self._MockModel("sales"),
        }
        dss = {"awareness": ds.copy(), "bht": ds.copy(), "sales": ds.copy()}
        mrs = {
            "awareness": self._mock_model_rep(ds),
            "bht": self._mock_model_rep(ds),
            "sales": self._mock_model_rep(ds),
        }

        nodes = {
            name: ModelNode(model=model, ds=dss[name], model_rep=mrs[name])
            for name, model in models.items()
        }

        mm = MetaModel(nodes=nodes, edges=[("awareness", "bht", None), ("bht", "sales", None)])

        order = mm._topological_order()
        assert order == ["awareness", "bht", "sales"]

    def test_topological_order_parallel(self):
        ds = pd.DataFrame({
            "date": pd.date_range("2025-01-01", periods=10, freq="D"),
            "listed": [4] * 10,
            "predicted": [100.0] * 10,
            "observed": [90.0] * 10,
        })
        models = {
            "a": self._MockModel("a"),
            "b": self._MockModel("b"),
            "output": self._MockModel("output"),
        }
        dss = {"a": ds.copy(), "b": ds.copy(), "output": ds.copy()}
        mrs = {
            "a": self._mock_model_rep(ds),
            "b": self._mock_model_rep(ds),
            "output": self._mock_model_rep(ds),
        }

        nodes = {
            name: ModelNode(model=model, ds=dss[name], model_rep=mrs[name])
            for name, model in models.items()
        }

        mm = MetaModel(nodes=nodes, edges=[("a", "output", None), ("b", "output", None)])

        order = mm._topological_order()
        assert "a" in order
        assert "b" in order

    def test_cycle_detection(self):
        from fermatrica.basics.basics import FermatricaError

        ds = pd.DataFrame({
            "date": pd.date_range("2025-01-01", periods=10, freq="D"),
            "listed": [4] * 10,
        })
        models = {
            "a": self._MockModel("a"),
            "b": self._MockModel("b"),
        }
        dss = {"a": ds.copy(), "b": ds.copy()}
        mrs = {
            "a": self._mock_model_rep(ds),
            "b": self._mock_model_rep(ds),
        }

        nodes = {
            name: ModelNode(model=model, ds=dss[name], model_rep=mrs[name])
            for name, model in models.items()
        }

        mm = MetaModel(nodes=nodes, edges=[("a", "b", None), ("b", "a", None)])

        with pytest.raises(FermatricaError):
            mm._topological_order()

    def test_predecessors(self):
        ds = pd.DataFrame({
            "date": pd.date_range("2025-01-01", periods=5, freq="D"),
            "listed": [4] * 5,
        })
        models = {"a": self._MockModel("a"), "b": self._MockModel("b")}
        dss = {"a": ds.copy(), "b": ds.copy()}
        mrs = {"a": self._mock_model_rep(ds), "b": self._mock_model_rep(ds)}

        nodes = {
            name: ModelNode(model=model, ds=dss[name], model_rep=mrs[name])
            for name, model in models.items()
        }

        mm = MetaModel(nodes=nodes, edges=[("a", "b", None)])

        assert mm._predecessors("b") == ["a"]
        assert mm._predecessors("a") == []


class _MockConf:
    pass


class _MockModel:
    pass


class TestAnalysisFunctionModelNode:
    """Integration tests: analysis functions accepting ModelNode / MetaModel."""

    @pytest.fixture
    def synthetic_model_ds(self):
        np.random.seed(42)
        n = 20
        x1 = np.random.uniform(1, 10, n)
        x2 = np.random.uniform(10, 100, n)
        y = 3.0 * x1 + 0.5 * x2 + 10.0 + np.random.normal(0, 1, n)
        ds = pd.DataFrame({
            'x1': x1, 'x2': x2, 'y': y,
            'date': pd.date_range('2024-01-01', periods=n, freq='D'),
            'superbrand': ['brand_x'] * n,
            'bs_key': [1] * n,
            'listed': [2] * n,
            'const': [1.0] * n,
            'price_dummy': [1.0] * n,
        })
        ols_result = ols('y ~ x1 + x2', data=ds).fit()

        conf = _MockConf()
        conf.model_type = 'OLS'
        conf.model_rhs = pd.DataFrame({
            'token': ['x1', 'x2'],
            'display_var': ['x1_disp', 'x2_disp'],
            'if_active': [1, 1],
        })
        conf.model_lhs = None
        conf.price_var = 'price_dummy'
        conf.bs_key = ['superbrand']
        conf.Y_var = 'y'
        conf.params = pd.DataFrame({
            'variable': ['', ''],
            'fun': ['coef', 'coef'],
            'arg': ['', ''],
            'value': [0.0, 0.0],
        })
        conf.display_var_base = 'base'

        model = _MockModel()
        model.conf = conf
        model.obj = type('Obj', (), {'models': {'main': ols_result}})()

        ds = select_eff(ds, ['date', 'superbrand', 'bs_key', 'listed',
                             'x1', 'x2', 'y', 'const', 'price_dummy'])

        return model, ds

    @pytest.fixture
    def model_rep(self, synthetic_model_ds):
        _, ds = synthetic_model_ds
        return ModelRep(ds, language='english')

    def test_auto_create_modelrep_from_kwargs(self, synthetic_model_ds):
        model, ds = synthetic_model_ds
        node = ModelNode(model, ds, language='english')
        assert isinstance(node.model_rep, ModelRep)
        assert node.model_rep.language == 'english'

    def test_explicit_modelrep_used(self, synthetic_model_ds, model_rep):
        model, ds = synthetic_model_ds
        node = ModelNode(model, ds, model_rep=model_rep)
        assert node.model_rep is model_rep

    def test_dt_pred_mutable_post_init(self, synthetic_model_ds, model_rep):
        model, ds = synthetic_model_ds
        node = ModelNode(model, ds, model_rep=model_rep)
        pred = pd.DataFrame({'date': [], 'predicted': []})
        node.dt_pred = pred
        assert node.dt_pred is pred

    def test_extract_effect_with_modelnode(self, synthetic_model_ds, model_rep):
        model, ds = synthetic_model_ds
        node = ModelNode(model, ds, model_rep=model_rep)
        result = extract_effect(node)
        assert isinstance(result, pd.DataFrame)
        assert 'variable' in result.columns
        assert 'value' in result.columns

    def test_extract_effect_with_metamodel_returns_dict(self, synthetic_model_ds, model_rep):
        model, ds = synthetic_model_ds
        node = ModelNode(model, ds, model_rep=model_rep)
        mm = MetaModel(nodes={"sales": node})
        result = extract_effect(mm)
        assert isinstance(result, dict)
        assert 'sales' in result
        assert isinstance(result['sales'], pd.DataFrame)

    def test_extract_effect_old_list_api_still_works(self, synthetic_model_ds, model_rep):
        model, ds = synthetic_model_ds
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            result = extract_effect([model], [ds], [model_rep])
            deprecations = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecations) >= 1
            assert 'list of models' in str(deprecations[0].message).lower()
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], pd.DataFrame)

    def test_extract_effect_modelnode_ds_override(self, synthetic_model_ds, model_rep):
        model, ds = synthetic_model_ds
        node = ModelNode(model, ds, model_rep=model_rep)
        result_via_node = extract_effect(node)
        result_explicit = extract_effect(model, ds, model_rep)
        assert len(result_via_node) == len(result_explicit)

    def test_convert_to_metamodel_from_model(self, synthetic_model_ds, model_rep):
        """_convert_to_metamodel wraps a bare Model + ds + model_rep into MetaModel."""
        model, ds = synthetic_model_ds
        mm = _convert_to_metamodel(model, ds=ds, model_rep=model_rep)
        assert isinstance(mm, MetaModel)
        assert "model" in mm.nodes
        assert mm.nodes["model"].model is model
        assert mm.nodes["model"].ds.equals(ds)
        assert mm.nodes["model"].model_rep is model_rep

    def test_convert_to_metamodel_from_modelnode(self, synthetic_model_ds, model_rep):
        """_convert_to_metamodel wraps ModelNode into a single-node MetaModel."""
        model, ds = synthetic_model_ds
        node = ModelNode(model, ds, model_rep=model_rep)
        mm = _convert_to_metamodel(node)
        assert isinstance(mm, MetaModel)
        assert "model" in mm.nodes
        assert mm.nodes["model"] is node

    def test_convert_to_metamodel_from_metamodel(self, synthetic_model_ds, model_rep):
        """_convert_to_metamodel returns MetaModel as-is."""
        model, ds = synthetic_model_ds
        node = ModelNode(model, ds, model_rep=model_rep)
        mm_in = MetaModel(nodes={"sales": node})
        mm_out = _convert_to_metamodel(mm_in)
        assert mm_out is mm_in

    def test_convert_to_metamodel_from_list(self, synthetic_model_ds, model_rep):
        """_convert_to_metamodel wraps list with DeprecationWarning."""
        model, ds = synthetic_model_ds
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            mm = _convert_to_metamodel([model], ds=ds, model_rep=model_rep)
            deprecations = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecations) >= 1
            assert 'list of models' in str(deprecations[0].message).lower()
        assert isinstance(mm, MetaModel)
        assert "model_0" in mm.nodes

    def test_convert_to_metamodel_from_modelrep(self, model_rep):
        """_convert_to_metamodel wraps ModelRep into a MetaModel."""
        mm = _convert_to_metamodel(model_rep)
        assert isinstance(mm, MetaModel)
        assert "model" in mm.nodes
        assert mm.nodes["model"].model_rep is model_rep

    def test_convert_to_metamodel_modelnode_ds_override(self, synthetic_model_ds, model_rep):
        """_convert_to_metamodel overrides ds on ModelNode when provided."""
        model, ds = synthetic_model_ds
        node = ModelNode(model, ds, model_rep=model_rep)
        ds_override = ds.iloc[:5].copy()
        mm = _convert_to_metamodel(node, ds=ds_override)
        assert len(mm.nodes["model"].ds) == 5

    def test_convert_to_metamodel_modelnode_dt_pred_override(self, synthetic_model_ds, model_rep):
        """_convert_to_metamodel overrides dt_pred on ModelNode when provided."""
        model, ds = synthetic_model_ds
        node = ModelNode(model, ds, model_rep=model_rep)
        pred = pd.DataFrame({"date": [], "predicted": []})
        mm = _convert_to_metamodel(node, dt_pred=pred)
        assert mm.nodes["model"].dt_pred is pred

    def test_convert_to_metamodel_list_broadcast(self, synthetic_model_ds, model_rep):
        """_convert_to_metamodel broadcasts ds/model_rep across list elements."""
        model, ds = synthetic_model_ds
        model2_conf = type('Conf', (), {'model_type': 'OLS', 'model_rhs': model.conf.model_rhs,
                                        'price_var': 'price_dummy', 'Y_var': 'y',
                                        'bs_key': ['superbrand'], 'display_var_base': 'base',
                                        'model_lhs': None, 'params': model.conf.params})
        model2 = type('Model', (), {})
        model2.conf = model2_conf
        
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            mm = _convert_to_metamodel(
                [model, model], ds=ds, model_rep=model_rep,
                merge_fn_pattern="code_py.adhoc.reporting.pred_merge_{i}",
            )
        
        assert len(mm.nodes) == 2
        assert "model_0" in mm.nodes
        assert "model_1" in mm.nodes
        assert mm.nodes["model_0"].ds.equals(ds)
        assert mm.nodes["model_1"].ds.equals(ds)

    def test_pack_results_with_names_returns_dict(self):
        """_pack_results returns dict when names is a list."""
        result = _pack_results(['a', 'b'], ['x', 'y'])
        assert result == {'x': 'a', 'y': 'b'}

    def test_pack_results_single_no_names_returns_scalar(self):
        """_pack_results unwraps single-element list when names is None."""
        result = _pack_results(['only'], None)
        assert result == 'only'

    def test_pack_results_multi_no_names_returns_list(self):
        """_pack_results returns list when names is None and len > 1."""
        result = _pack_results(['a', 'b'], None)
        assert result == ['a', 'b']

    def test_pack_results_force_list_with_true_names(self):
        """_pack_results forces list return when names is True (old list API)."""
        result = _pack_results(['only'], True)
        assert result == ['only']

    def test_fit_main_data_with_modelnode(self, synthetic_model_ds, model_rep):
        model, ds = synthetic_model_ds
        pred = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=20, freq='D'),
            'listed': [2] * 20,
            'predicted': [100.0] * 20,
            'observed': [95.0] * 20,
            'price_dummy': [1.0] * 20,
        })
        node = ModelNode(model, ds, model_rep=model_rep)
        node.dt_pred = pred
        ds_out, ds_melt = fit_main_data(node)
        assert isinstance(ds_out, pd.DataFrame)
        assert isinstance(ds_melt, pd.DataFrame)

    def test_fit_main_data_with_modelnode_dt_pred_override(self, synthetic_model_ds, model_rep):
        model, ds = synthetic_model_ds
        pred_stored = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=20, freq='D'),
            'listed': [2] * 20,
            'predicted': [1.0] * 20,
            'observed': [0.5] * 20,
            'price_dummy': [1.0] * 20,
        })
        pred_override = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=20, freq='D'),
            'listed': [2] * 20,
            'predicted': [100.0] * 20,
            'observed': [95.0] * 20,
            'price_dummy': [1.0] * 20,
        })
        node = ModelNode(model, ds, model_rep=model_rep, dt_pred=pred_stored)
        ds_out, _ = fit_main_data(node, dt_pred=pred_override)
        assert ds_out['predicted'].sum() > ds_out['observed'].sum()

    def test_extract_effect_ds_none_when_modelnode_uses_node_ds(self, synthetic_model_ds, model_rep):
        model, ds = synthetic_model_ds
        node = ModelNode(model, ds, model_rep=model_rep)
        result = extract_effect(node)
        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    def test_node_compute_no_arg_refresh(self, synthetic_model_ds, model_rep, monkeypatch):
        """No-arg compute() returns dt_pred and updates node.ds / node.dt_pred."""
        model, ds = synthetic_model_ds
        node = ModelNode(model, ds, model_rep=model_rep)

        def fake_transform(ds, model, set_start=False, if_by_ref=True):
            return model, ds

        pred = pd.DataFrame({
            'date': ds['date'], 'superbrand': ds['superbrand'],
            'listed': [2] * len(ds), 'predicted': [1.0] * len(ds),
            'observed': [0.5] * len(ds), 'price_dummy': [1.0] * len(ds),
        })

        def fake_predict(model, ds):
            return pred

        monkeypatch.setattr('fermatrica_rep.meta_model.node.transform', fake_transform)
        monkeypatch.setattr('fermatrica_rep.meta_model.node.predict_ext', fake_predict)

        node.dt_pred = None
        result = node.compute()
        assert result is pred
        assert node.dt_pred is pred

    def test_node_compute_option_returns_scope_summaries(self, synthetic_model_ds, model_rep, monkeypatch):
        """compute(option) still returns {scope_name: summary}."""
        model, ds = synthetic_model_ds
        node = ModelNode(model, ds, model_rep=model_rep)

        def fake_transform(ds, model, set_start=False, if_by_ref=True):
            return model, ds

        pred = pd.DataFrame({
            'date': ds['date'], 'superbrand': ds['superbrand'],
            'listed': [2] * len(ds), 'predicted': [1.0] * len(ds),
            'observed': [0.5] * len(ds), 'price_dummy': [1.0] * len(ds),
        })

        def fake_predict(model, ds):
            return pred

        monkeypatch.setattr('fermatrica_rep.meta_model.node.transform', fake_transform)
        monkeypatch.setattr('fermatrica_rep.meta_model.node.predict_ext', fake_predict)

        scope = OptionScope(
            scope_name='test_scope',
            date_start='2024-01-01', date_end='2024-12-31',
            ref_date_start='2023-01-01', ref_date_end='2023-12-31',
        )
        opt = Option(name='test', budget_data={'test_scope': {}})
        results = node.compute(opt, scopes=[scope])
        assert isinstance(results, dict)
        assert 'test_scope' in results
        assert 'pred_exact_vol' in results['test_scope']

    def test_metamodel_compute_no_arg_refresh(self, synthetic_model_ds, model_rep, monkeypatch):
        """MetaModel.compute() no-arg refreshes in topological order and returns {name: dt_pred}."""
        model, ds = synthetic_model_ds
        n_a = ModelNode(model, ds, model_rep=model_rep)
        n_b = ModelNode(model, ds, model_rep=model_rep)
        calls = []

        def merge_fn(ds_cur, dt_pred_prev, allow_past):
            calls.append(('b', allow_past, dt_pred_prev))
            return ds_cur

        mm = MetaModel(nodes={'a': n_a, 'b': n_b}, edges=[('a', 'b', merge_fn)])

        def fake_transform(ds, model, set_start=False, if_by_ref=True):
            return model, ds

        def fake_predict(model, ds):
            return pd.DataFrame({
                'date': ds['date'], 'superbrand': ds['superbrand'],
                'listed': [2] * len(ds), 'predicted': [1.0] * len(ds),
                'observed': [0.5] * len(ds), 'price_dummy': [1.0] * len(ds),
            })

        monkeypatch.setattr('fermatrica_rep.meta_model.node.transform', fake_transform)
        monkeypatch.setattr('fermatrica_rep.meta_model.node.predict_ext', fake_predict)

        results = mm.compute()
        assert set(results.keys()) == {'a', 'b'}
        assert isinstance(results['a'], pd.DataFrame)
        assert isinstance(results['b'], pd.DataFrame)
        assert calls and calls[0][0] == 'b'
        assert calls[0][1] is True
        assert calls[0][2] is results['a']

    def test_metamodel_compute_no_arg_allow_past_false(self, synthetic_model_ds, model_rep, monkeypatch):
        """MetaModel.compute(allow_past=False) forwards False to the merge fn."""
        model, ds = synthetic_model_ds
        n_a = ModelNode(model, ds, model_rep=model_rep)
        n_b = ModelNode(model, ds, model_rep=model_rep)
        calls = []

        def merge_fn(ds_cur, dt_pred_prev, allow_past):
            calls.append(allow_past)
            return ds_cur

        mm = MetaModel(nodes={'a': n_a, 'b': n_b}, edges=[('a', 'b', merge_fn)])

        def fake_transform(ds, model, set_start=False, if_by_ref=True):
            return model, ds

        def fake_predict(model, ds):
            return pd.DataFrame({
                'date': ds['date'], 'superbrand': ds['superbrand'],
                'listed': [2] * len(ds), 'predicted': [1.0] * len(ds),
                'observed': [0.5] * len(ds), 'price_dummy': [1.0] * len(ds),
            })

        monkeypatch.setattr('fermatrica_rep.meta_model.node.transform', fake_transform)
        monkeypatch.setattr('fermatrica_rep.meta_model.node.predict_ext', fake_predict)

        mm.compute(allow_past=False)
        assert calls == [False]

    def test_node_save_passes_ds_and_model_rep(self, synthetic_model_ds, model_rep):
        """ModelNode.save() forwards node.ds and node.model_rep to model.save."""
        model, ds = synthetic_model_ds
        called = {}

        class SaveRecorder:
            def save(self, ds, model_rep=None, path='', save_tag=None):
                called['ds'] = ds
                called['model_rep'] = model_rep
                called['path'] = path
                called['save_tag'] = save_tag
                return '/saved'

        node = ModelNode(model=SaveRecorder(), ds=ds, model_rep=model_rep)
        node.save(path='/x', save_tag='tag')
        assert called['ds'] is node.ds
        assert called['model_rep'] is model_rep
        assert called['path'] == '/x'
        assert called['save_tag'] == 'tag'

    def test_convert_to_metamodel_auto_rep_from_dt_pred(self, synthetic_model_ds):
        """_convert_to_metamodel(model, dt_pred=pred) builds a default ModelRep from dt_pred."""
        model, ds = synthetic_model_ds
        pred = pd.DataFrame({
            'date': ds['date'], 'superbrand': ds['superbrand'],
            'listed': [2] * len(ds), 'predicted': [1.0] * len(ds),
            'observed': [0.5] * len(ds),
        })
        mm = _convert_to_metamodel(model, dt_pred=pred)
        rep = mm.nodes['model'].model_rep
        assert isinstance(rep, ModelRep)
        assert 'brand_x' in rep.palette_sbr

    def test_fit_mult_plot_vol_without_model_rep(self, synthetic_model_ds):
        """fit_mult_plot_vol(model, dt_pred) runs with the auto default rep."""
        model, ds = synthetic_model_ds
        pred = pd.DataFrame({
            'date': ds['date'], 'superbrand': ds['superbrand'],
            'listed': [2] * len(ds), 'predicted': [1.0] * len(ds),
            'observed': [0.5] * len(ds), 'price_dummy': [1.0] * len(ds),
        })
        fig = fit_mult_plot_vol(model, dt_pred=pred)
        assert fig is not None
