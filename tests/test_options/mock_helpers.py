import numpy as np
import pandas as pd


class _MockCompute:
    """Picklable mock for MetaModel.compute. Single-node variant."""
    def __init__(self, scope_name, kpi_multiplier=2.0, coefs=None):
        self.scope_name = scope_name
        self.kpi_multiplier = kpi_multiplier
        self.coefs = coefs

    def __call__(self, option, scopes=None, **kwargs):
        items = option.budget_data.get(self.scope_name, {})
        if self.coefs is not None:
            kpi = sum(items.get(k, 0) * c for k, c in self.coefs.items())
        else:
            kpi = sum(items.values()) * self.kpi_multiplier
        return {"model": {self.scope_name: {
            "pred_exact_vol": kpi,
            "pred_exact_val": kpi,
            "pred_long_vol": kpi,
            "pred_long_val": kpi,
            "pred_ref_vol": 100, "ref_vol": 50, "growth_vol": 2.0,
        }}}


class _MockComputeMulti:
    """Picklable mock for MetaModel.compute. Two-node variant (category -> sales)."""
    def __init__(self, cat_scope, sales_scope, cat_kpi=1.0, sales_kpi=2.0):
        self.cat_scope = cat_scope
        self.sales_scope = sales_scope
        self.cat_kpi = cat_kpi
        self.sales_kpi = sales_kpi

    def __call__(self, option, scopes=None, **kwargs):
        cat_bdg = sum(option.budget_data.get(self.cat_scope, {}).values())
        sales_bdg = sum(option.budget_data.get(self.sales_scope, {}).values())
        return {
            "category": {self.cat_scope: {
                "pred_exact_vol": cat_bdg * self.cat_kpi,
                "pred_long_vol": cat_bdg * self.cat_kpi,
                "pred_exact_val": cat_bdg * self.cat_kpi,
                "pred_long_val": cat_bdg * self.cat_kpi,
            }},
            "sales": {self.sales_scope: {
                "pred_exact_vol": sales_bdg * self.sales_kpi,
                "pred_long_vol": sales_bdg * self.sales_kpi,
                "pred_exact_val": sales_bdg * self.sales_kpi,
                "pred_long_val": sales_bdg * self.sales_kpi,
            }},
        }


def _make_mock_compute(scope_name, kpi_multiplier=2.0, coefs=None):
    return _MockCompute(scope_name, kpi_multiplier, coefs)


def _make_mock_meta_model(scope_name, kpi_multiplier=2.0, coefs=None):
    """Single-node MagicMock MetaModel."""
    from unittest.mock import MagicMock
    from fermatrica_rep.meta_model.meta_model import MetaModel
    from fermatrica_rep.meta_model.node import ModelNode

    mm = MagicMock(spec=MetaModel)
    mm.nodes = {"model": MagicMock(spec=ModelNode)}
    mm.compute = _make_mock_compute(scope_name, kpi_multiplier, coefs)
    return mm


def _make_mock_meta_model_multi(cat_scope, sales_scope, cat_kpi=1.0, sales_kpi=2.0):
    """Two-node MagicMock MetaModel (category -> sales)."""
    from unittest.mock import MagicMock
    from fermatrica_rep.meta_model.meta_model import MetaModel
    from fermatrica_rep.meta_model.node import ModelNode

    mm = MagicMock(spec=MetaModel)
    mm.nodes = {
        "category": MagicMock(spec=ModelNode),
        "sales": MagicMock(spec=ModelNode),
    }
    mm.compute = _MockComputeMulti(cat_scope, sales_scope, cat_kpi, sales_kpi)
    return mm


class _MockNcmaModel:
    """Minimal model object for NCMA tests (no real model needed)."""
    class Obj:
        transform_lhs_fn = None
        custom_predict_fn = None
    obj = Obj()

    class Conf:
        price_var = 'price'
        summary_type = 'sum'
        trans_path_df = pd.DataFrame({'variable': ['x1'], 'price': [1.0]})
    conf = Conf()


def _make_ncma_meta_model(scope_name, mock_model, kpi_multiplier=2.0):
    """Real MetaModel wrapping a mock model for NCMA multiprocessing."""
    from fermatrica_rep.meta_model.meta_model import MetaModel
    from fermatrica_rep.meta_model.node import ModelNode

    node = ModelNode(model=mock_model, ds=pd.DataFrame(), model_rep=None)
    meta = MetaModel(nodes={"model": node})
    meta.compute = _make_mock_compute(scope_name, kpi_multiplier)
    return meta
