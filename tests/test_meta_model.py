"""Unit tests for MetaModel — multi-model aggregate container.

Tests cover construction from dicts, brand list derivation,
error handling for empty models, model dispatch, from_node wrapping,
and universal load detection.
"""
import pandas as pd
import pytest

from fermatrica.basics.basics import FermatricaError
from fermatrica_rep.meta_model.meta_model import MetaModel
from fermatrica_rep.meta_model.node import ModelNode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _MockModel:
    def __init__(self, name):
        self.conf = _MockConf(name)
        self.obj = _MockObj()

    @property
    def brands(self):
        return getattr(self.conf, 'brands', [])


class _MockConf:
    def __init__(self, name):
        self.name = name
        self.brands = []
        self.Y_var = 'kpi'


class _MockObj:
    pass


def _mock_model_rep():
    class MockRep:
        language = 'english'
        pass
    return MockRep()


# ---------------------------------------------------------------------------
# TestMetaModelConstruction
# ---------------------------------------------------------------------------

class TestMetaModelConstruction:

    def test_single_model(self):
        models = {'node_a': _MockModel('node_a')}
        dss = {'node_a': pd.DataFrame({'date': pd.date_range('2024-01-01', periods=3, freq='MS'), 'listed': [4] * 3})}
        mrs = {'node_a': _mock_model_rep()}

        nodes = {
            name: ModelNode(model=model, ds=dss[name], model_rep=mrs[name])
            for name, model in models.items()
        }
        mm = MetaModel(nodes=nodes)

        assert set(mm.nodes.keys()) == {'node_a'}
        assert mm.nodes['node_a'].model is models['node_a']

    def test_single_node_passed_directly(self):
        node = ModelNode(
            model=_MockModel("test"),
            ds=pd.DataFrame({"listed": [4]}),
            model_rep=_mock_model_rep(),
        )
        mm = MetaModel(nodes=node)
        assert list(mm.nodes.keys()) == ["model"]
        assert mm.nodes["model"] is node
        assert len(mm.edges) == 0

    def test_multiple_models(self):
        models = {
            'bht': _MockModel('bht'),
            'sales': _MockModel('sales'),
        }
        dss = {
            'bht': pd.DataFrame({'date': pd.date_range('2024-01-01', periods=3, freq='MS'), 'listed': [4] * 3}),
            'sales': pd.DataFrame({'date': pd.date_range('2024-06-01', periods=3, freq='MS'), 'listed': [4] * 3}),
        }
        mrs = {'bht': _mock_model_rep(), 'sales': _mock_model_rep()}

        nodes = {
            name: ModelNode(model=model, ds=dss[name], model_rep=mrs[name])
            for name, model in models.items()
        }
        mm = MetaModel(nodes=nodes, edges=[("bht", "sales", None)])

        assert set(mm.nodes.keys()) == {'bht', 'sales'}

    def test_brand_list_derivation(self):
        m1 = _MockModel('m1')
        m1.conf.brands = ['brand_a']
        m2 = _MockModel('m2')
        m2.conf.brands = ['brand_b']

        models = {'m1': m1, 'm2': m2}
        dss = {'m1': pd.DataFrame({'listed': [4]}), 'm2': pd.DataFrame({'listed': [4]})}
        mrs = {'m1': _mock_model_rep(), 'm2': _mock_model_rep()}

        nodes = {
            name: ModelNode(model=model, ds=dss[name], model_rep=mrs[name])
            for name, model in models.items()
        }
        mm = MetaModel(nodes=nodes, edges=[("m1", "m2", None)])

        expected_brands = {'brand_a', 'brand_b'}
        actual_brands = set()
        for name in mm.nodes:
            actual_brands.update(mm.nodes[name].model.conf.brands)
        assert actual_brands == expected_brands


# ---------------------------------------------------------------------------
# TestMetaModelDispatch
# ---------------------------------------------------------------------------

class TestMetaModelDispatch:

    def test_get_model_for_brand(self):
        m1 = _MockModel('m1')
        m2 = _MockModel('m2')

        models = {'m1': m1, 'm2': m2}
        dss = {'m1': pd.DataFrame({'listed': [4]}), 'm2': pd.DataFrame({'listed': [4]})}
        mrs = {'m1': _mock_model_rep(), 'm2': _mock_model_rep()}

        nodes = {
            name: ModelNode(model=model, ds=dss[name], model_rep=mrs[name])
            for name, model in models.items()
        }
        mm = MetaModel(nodes=nodes, edges=[("m1", "m2", None)])
        assert mm.nodes['m1'].model is m1

    def test_missing_brand_returns_none(self):
        m1 = _MockModel('m1')
        models = {'m1': m1}
        dss = {'m1': pd.DataFrame({'listed': [4]})}
        mrs = {'m1': _mock_model_rep()}

        nodes = {
            name: ModelNode(model=model, ds=dss[name], model_rep=mrs[name])
            for name, model in models.items()
        }
        mm = MetaModel(nodes=nodes)
        assert mm.nodes.get('nonexistent') is None

    def test_all_brands_iterable(self):
        models = {
            'bht': _MockModel('bht'),
            'sales': _MockModel('sales'),
        }
        dss = {'bht': pd.DataFrame({'listed': [4]}), 'sales': pd.DataFrame({'listed': [4]})}
        mrs = {'bht': _mock_model_rep(), 'sales': _mock_model_rep()}

        nodes = {
            name: ModelNode(model=model, ds=dss[name], model_rep=mrs[name])
            for name, model in models.items()
        }
        mm = MetaModel(nodes=nodes, edges=[("bht", "sales", None)])
        node_names = list(mm.nodes.keys())
        assert 'bht' in node_names
        assert 'sales' in node_names


# ---------------------------------------------------------------------------
# TestMetaModelFromNode
# ---------------------------------------------------------------------------

class TestMetaModelFromNode:

    def test_from_node_with_explicit_name(self):
        node = ModelNode(
            model=_MockModel("test"),
            ds=pd.DataFrame({"listed": [4]}),
            model_rep=_mock_model_rep(),
        )
        mm = MetaModel.from_node(node, name="sales")
        assert list(mm.nodes.keys()) == ["sales"]
        assert mm.nodes["sales"] is node
        assert len(mm.edges) == 0

    def test_from_node_default_name(self):
        node = ModelNode(
            model=_MockModel("test"),
            ds=pd.DataFrame({"listed": [4]}),
            model_rep=_mock_model_rep(),
        )
        mm = MetaModel.from_node(node)
        assert list(mm.nodes.keys()) == ["model"]

    def test_from_node_rejects_empty_name(self):
        node = ModelNode(
            model=_MockModel("test"),
            ds=pd.DataFrame({"listed": [4]}),
            model_rep=_mock_model_rep(),
        )
        with pytest.raises(FermatricaError):
            MetaModel.from_node(node, name="")

    def test_from_node_rejects_whitespace_name(self):
        node = ModelNode(
            model=_MockModel("test"),
            ds=pd.DataFrame({"listed": [4]}),
            model_rep=_mock_model_rep(),
        )
        with pytest.raises(FermatricaError):
            MetaModel.from_node(node, name="   ")


# ---------------------------------------------------------------------------
# TestMetaModelLoad
# ---------------------------------------------------------------------------

class TestMetaModelLoad:

    def test_load_detects_multi_model(self, tmp_path, monkeypatch):
        (tmp_path / "meta_conf.toml").touch()
        (tmp_path / "a").mkdir()

        mock_node = ModelNode(
            model=_MockModel("test"),
            ds=pd.DataFrame({"listed": [4]}),
            model_rep=_mock_model_rep(),
        )
        expected = MetaModel(
            nodes={"a": mock_node},
            edges=[],
        )
        monkeypatch.setattr(MetaModel, "_load_multi", lambda p: expected)

        result = MetaModel.load(str(tmp_path))
        assert result is expected

    def test_load_detects_single_model(self, tmp_path, monkeypatch):
        mock_node = ModelNode(
            model=_MockModel("test"),
            ds=pd.DataFrame({"listed": [4]}),
            model_rep=_mock_model_rep(),
        )
        monkeypatch.setattr(ModelNode, "load", lambda p: mock_node)

        mm = MetaModel.load(str(tmp_path))
        assert list(mm.nodes.keys()) == ["model"]
        assert mm.nodes["model"] is mock_node

    def test_load_raises_on_subdirs_without_meta_conf(self, tmp_path):
        (tmp_path / "bht").mkdir()
        (tmp_path / "sales").mkdir()

        with pytest.raises(FermatricaError, match="looks like a multi-model DAG"):
            MetaModel.load(str(tmp_path))

    def test_load_raises_on_nonexistent_path(self):
        with pytest.raises(FermatricaError, match="Path not found"):
            MetaModel.load("nonexistent_directory_xyz")

    def test_load_raises_on_file_not_directory(self, tmp_path):
        file_path = tmp_path / "some_file.txt"
        file_path.touch()

        with pytest.raises(FermatricaError, match="Path is not a directory"):
            MetaModel.load(str(file_path))


# ---------------------------------------------------------------------------
# TestMetaModelEdgeValidation
# ---------------------------------------------------------------------------

class TestMetaModelEdgeValidation:

    def _node(self, name="n"):
        return ModelNode(
            model=_MockModel(name),
            ds=pd.DataFrame({"listed": [4]}),
            model_rep=_mock_model_rep(),
        )

    def test_multi_node_no_edges_raises(self):
        with pytest.raises(FermatricaError, match="no edges provided"):
            MetaModel(nodes={"a": self._node("a"), "b": self._node("b")})

    def test_edge_left_not_in_nodes_raises(self):
        with pytest.raises(FermatricaError, match="Edge left node 'missing' is not in MetaModel nodes"):
            MetaModel(
                nodes={"a": self._node("a"), "b": self._node("b")},
                edges=[("missing", "a", None), ("a", "b", None)],
            )

    def test_edge_right_not_in_nodes_raises(self):
        with pytest.raises(FermatricaError, match="Edge right node 'missing' is not in MetaModel nodes"):
            MetaModel(
                nodes={"a": self._node("a"), "b": self._node("b")},
                edges=[("a", "missing", None), ("a", "b", None)],
            )

    def test_self_loop_raises(self):
        with pytest.raises(FermatricaError, match="Self-loop detected"):
            MetaModel(
                nodes={"a": self._node("a"), "b": self._node("b")},
                edges=[("a", "a", None), ("a", "b", None)],
            )

    def test_duplicate_edge_raises(self):
        with pytest.raises(FermatricaError, match="Duplicate edge"):
            MetaModel(
                nodes={"a": self._node("a"), "b": self._node("b")},
                edges=[("a", "b", None), ("a", "b", None)],
            )

    def test_orphan_node_raises(self):
        with pytest.raises(FermatricaError, match="Nodes not referenced in any edge"):
            MetaModel(
                nodes={"a": self._node("a"), "b": self._node("b"), "orphan": self._node("o")},
                edges=[("a", "b", None)],
            )

    def test_disconnected_components_raises(self):
        with pytest.raises(FermatricaError, match="not a single connected DAG"):
            MetaModel(
                nodes={"a": self._node("a"), "b": self._node("b"), "c": self._node("c"), "d": self._node("d")},
                edges=[("a", "b", None), ("c", "d", None)],
            )

    def test_empty_nodes_raises(self):
        with pytest.raises(FermatricaError, match="must contain at least one node"):
            MetaModel(nodes={})

    def test_single_node_no_edges_ok(self):
        mm = MetaModel(nodes=self._node())
        assert len(mm.nodes) == 1
        assert len(mm.edges) == 0

    def test_valid_dag_passes_validation(self):
        mm = MetaModel(
            nodes={"awareness": self._node("a"), "bht": self._node("b"), "sales": self._node("s")},
            edges=[("awareness", "bht", None), ("bht", "sales", None)],
        )
        assert len(mm.nodes) == 3
        assert len(mm.edges) == 2
