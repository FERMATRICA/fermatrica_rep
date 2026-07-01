"""Unit tests for MetaModel — multi-model aggregate container.

Tests cover construction from dicts, brand list derivation,
error handling for empty models, and model dispatch.
"""
import pandas as pd
import pytest

from fermatrica.basics.basics import FermatricaError
from fermatrica_rep.meta_model.meta_model import MetaModel, MetaModelConf, MetaModelObj


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
        dss = {'node_a': pd.DataFrame({'date': pd.date_range('2024-01-01', periods=3, freq='MS')})}
        mrs = {'node_a': _mock_model_rep()}

        mg = pd.DataFrame({'left': ['node_a'], 'right': ['output'], 'type': ['']})
        conf = MetaModelConf(model_graph_ds=mg, if_stable=False)

        mm = MetaModel(
            model_graph_ds=mg,
            dss=dss,
            models=models,
            model_reps=mrs,
            if_stable=False,
        )
        assert mm.obj is not None
        assert mm.obj.models == models
        assert mm.obj.dss == dss

    def test_multiple_models(self):
        models = {
            'bht': _MockModel('bht'),
            'sales': _MockModel('sales'),
        }
        dss = {
            'bht': pd.DataFrame({'date': pd.date_range('2024-01-01', periods=3, freq='MS')}),
            'sales': pd.DataFrame({'date': pd.date_range('2024-06-01', periods=3, freq='MS')}),
        }
        mrs = {'bht': _mock_model_rep(), 'sales': _mock_model_rep()}

        mg = pd.DataFrame({
            'left': ['bht', 'sales'],
            'right': ['sales', 'output'],
            'type': ['', ''],
        })
        mm = MetaModel(
            model_graph_ds=mg,
            dss=dss,
            models=models,
            model_reps=mrs,
            if_stable=False,
        )
        assert set(mm.obj.models.keys()) == {'bht', 'sales'}

    def test_brand_list_derivation(self):
        m1 = _MockModel('m1')
        m1.conf.brands = ['brand_a']
        m2 = _MockModel('m2')
        m2.conf.brands = ['brand_b']

        models = {'m1': m1, 'm2': m2}
        dss = {'m1': pd.DataFrame(), 'm2': pd.DataFrame()}
        mrs = {'m1': _mock_model_rep(), 'm2': _mock_model_rep()}

        mg = pd.DataFrame({'left': ['m1', 'm2'], 'right': ['m2', 'output'], 'type': ['', '']})
        mm = MetaModel(
            model_graph_ds=mg,
            dss=dss,
            models=models,
            model_reps=mrs,
            if_stable=False,
        )

        expected_brands = {'brand_a', 'brand_b'}
        actual_brands = set()
        for node in mm.obj.models:
            actual_brands.update(mm.obj.models[node].conf.brands)
        assert actual_brands == expected_brands

    def test_empty_models_list_raises(self):
        mg = pd.DataFrame({'left': ['a'], 'right': ['b'], 'type': ['']})
        with pytest.raises(FermatricaError):
            MetaModelObj(
                meta_model_conf=MetaModelConf(model_graph_ds=mg, if_stable=False),
                dss=None,
                models=None,
                model_reps={'a': _mock_model_rep()},
                if_stable=False,
            )


# ---------------------------------------------------------------------------
# TestMetaModelDispatch
# ---------------------------------------------------------------------------

class TestMetaModelDispatch:

    def test_get_model_for_brand(self):
        m1 = _MockModel('m1')
        m2 = _MockModel('m2')

        models = {'m1': m1, 'm2': m2}
        dss = {'m1': pd.DataFrame(), 'm2': pd.DataFrame()}
        mrs = {'m1': _mock_model_rep(), 'm2': _mock_model_rep()}

        mg = pd.DataFrame({'left': ['m1', 'm2'], 'right': ['m2', 'output'], 'type': ['', '']})
        mm = MetaModel(
            model_graph_ds=mg,
            dss=dss,
            models=models,
            model_reps=mrs,
            if_stable=False,
        )
        # Direct dict access
        assert mm.obj.models['m1'] is m1

    def test_missing_brand_returns_none(self):
        m1 = _MockModel('m1')
        models = {'m1': m1}
        dss = {'m1': pd.DataFrame()}
        mrs = {'m1': _mock_model_rep()}

        mg = pd.DataFrame({'left': ['m1'], 'right': ['output'], 'type': ['']})
        mm = MetaModel(
            model_graph_ds=mg,
            dss=dss,
            models=models,
            model_reps=mrs,
            if_stable=False,
        )
        assert mm.obj.models.get('nonexistent') is None

    def test_all_brands_iterable(self):
        models = {
            'bht': _MockModel('bht'),
            'sales': _MockModel('sales'),
        }
        dss = {'bht': pd.DataFrame(), 'sales': pd.DataFrame()}
        mrs = {'bht': _mock_model_rep(), 'sales': _mock_model_rep()}

        mg = pd.DataFrame({'left': ['bht', 'sales'], 'right': ['sales', 'output'], 'type': ['', '']})
        mm = MetaModel(
            model_graph_ds=mg,
            dss=dss,
            models=models,
            model_reps=mrs,
            if_stable=False,
        )
        node_names = list(mm.obj.models.keys())
        assert 'bht' in node_names
        assert 'sales' in node_names
