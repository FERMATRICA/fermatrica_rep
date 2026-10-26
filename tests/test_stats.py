import pytest
import pandas as pd
import numpy as np

from statsmodels.formula.api import ols
from fermatrica.model.model import Model

from fermatrica_rep.analysis.stats import (
    basic_metrics_table,
    metrics_group_table,
    metrics_table,
    predictors_table,
    tests_table_ols as _tests_table_ols,
    vif_table,
)
from fermatrica_rep.meta_model.node import ModelNode
from fermatrica_rep.meta_model.meta_model import MetaModel
from fermatrica_rep.meta_model.model_rep import ModelRep


class TestBasicMetricsTable:

    @pytest.fixture
    def dt_pred(self):
        np.random.seed(42)
        n_train = 50
        n_test = 10
        observed_train = np.random.uniform(100, 200, n_train)
        predicted_train = observed_train + np.random.normal(0, 5, n_train)
        observed_test = np.random.uniform(100, 200, n_test)
        predicted_test = observed_test + np.random.normal(0, 8, n_test)
        return pd.DataFrame({
            'listed': [2] * n_train + [3] * n_test,
            'observed': np.concatenate([observed_train, observed_test]),
            'predicted': np.concatenate([predicted_train, predicted_test])
        })

    def test_returns_dataframe(self, dt_pred):
        result = basic_metrics_table(dt_pred)
        assert isinstance(result, pd.DataFrame)

    def test_has_r2_train(self, dt_pred):
        result = basic_metrics_table(dt_pred)
        assert any('R^2' in str(v) for v in result['Metrics'].values)

    def test_has_rmse_train(self, dt_pred):
        result = basic_metrics_table(dt_pred)
        assert any('RMSE' in str(v) for v in result['Metrics'].values)

    def test_has_mape_train(self, dt_pred):
        result = basic_metrics_table(dt_pred)
        assert any('MAPE' in str(v) for v in result['Metrics'].values)

    def test_has_test_metrics_when_test_data_present(self, dt_pred):
        result = basic_metrics_table(dt_pred)
        metrics_str = ' '.join(result['Metrics'].values)
        assert 'RMSE test' in metrics_str or 'MAPE test' in metrics_str

    def test_no_test_metrics_when_no_test_data(self, dt_pred):
        train_only = dt_pred[dt_pred['listed'] == 2].copy()
        result = basic_metrics_table(train_only)
        metrics_str = ' '.join(result['Metrics'].values)
        assert 'test' not in metrics_str

    def test_mape_threshold_adds_adj_column(self, dt_pred):
        result = basic_metrics_table(dt_pred, mape_threshold=0.3)
        metrics_str = ' '.join(result['Metrics'].values)
        assert 'MAPE adj' in metrics_str

    def test_no_mape_adj_without_threshold(self, dt_pred):
        result = basic_metrics_table(dt_pred)
        metrics_str = ' '.join(result['Metrics'].values)
        assert 'MAPE adj' not in metrics_str

    def test_has_statistics_column(self, dt_pred):
        result = basic_metrics_table(dt_pred)
        assert 'Statictics' in result.columns

    def test_invalid_empty_data_raises(self):
        empty = pd.DataFrame(columns=['listed', 'observed', 'predicted'])
        with pytest.raises(Exception):
            basic_metrics_table(empty)

    def test_golden_r2_train(self, dt_pred):
        result = basic_metrics_table(dt_pred)
        r2_str = result.loc[result['Metrics'] == 'R^2 train', 'Statictics'].iloc[0]
        assert float(r2_str) == pytest.approx(0.9752, abs=1e-4)

    def test_golden_rmse_train(self, dt_pred):
        result = basic_metrics_table(dt_pred)
        rmse_str = result.loc[result['Metrics'] == 'RMSE train', 'Statictics'].iloc[0]
        assert float(rmse_str) == pytest.approx(4.5474, abs=1e-4)

    def test_golden_rmse_test(self, dt_pred):
        result = basic_metrics_table(dt_pred)
        rmse_str = result.loc[result['Metrics'] == 'RMSE test', 'Statictics'].iloc[0]
        assert float(rmse_str) == pytest.approx(6.5843, abs=1e-4)

    def test_golden_mape_train(self, dt_pred):
        result = basic_metrics_table(dt_pred)
        mape_str = result.loc[result['Metrics'] == 'MAPE train', 'Statictics'].iloc[0]
        assert float(mape_str) == pytest.approx(2.57, abs=1e-2)

    def test_golden_mape_test(self, dt_pred):
        result = basic_metrics_table(dt_pred)
        mape_str = result.loc[result['Metrics'] == 'MAPE test', 'Statictics'].iloc[0]
        assert float(mape_str.strip()) == pytest.approx(2.95, abs=1e-2)

    def test_golden_mape_adj_train(self, dt_pred):
        result = basic_metrics_table(dt_pred, mape_threshold=0.3)
        mape_str = result.loc[result['Metrics'] == 'MAPE adj train', 'Statictics'].iloc[0]
        assert float(mape_str) == pytest.approx(2.57, abs=1e-2)

    def test_golden_mape_ratio(self, dt_pred):
        result = basic_metrics_table(dt_pred, mape_threshold=0.3)
        ratio_str = result.loc[result['Metrics'] == 'MAPE test/train - 1', 'Statictics'].iloc[0]
        assert float(ratio_str) == pytest.approx(0.15, abs=1e-2)


class TestMetricsGroupTable:

    @pytest.fixture
    def dt_pred(self):
        np.random.seed(42)
        groups = ['brand_a', 'brand_b']
        data = []
        for g in groups:
            for d in pd.date_range('2024-01-01', periods=20, freq='MS'):
                obs = np.random.uniform(100, 200)
                pred = obs + np.random.normal(0, 5)
                data.append({
                    'market': 'market_1',
                    'superbrand': g,
                    'date': d,
                    'listed': 2,
                    'observed': obs,
                    'predicted': pred
                })
        return pd.DataFrame(data)

    def test_returns_dataframe(self, dt_pred):
        result = metrics_group_table(dt_pred, group_vars=['market', 'superbrand'])
        assert isinstance(result, pd.DataFrame)

    def test_has_expected_columns(self, dt_pred):
        result = metrics_group_table(dt_pred, group_vars=['market', 'superbrand'])
        assert 'r_sq' in result.columns
        assert 'rmse' in result.columns
        assert 'smape' in result.columns
        assert 'pc' in result.columns

    def test_one_row_per_group(self, dt_pred):
        result = metrics_group_table(dt_pred)
        assert len(result) == 2
        assert set(result.index.get_level_values('superbrand')) == {'brand_a', 'brand_b'}

    def test_golden_r_sq_brand_a(self, dt_pred):
        result = metrics_group_table(dt_pred, group_vars=['market', 'superbrand'])
        assert float(result.loc[('market_1', 'brand_a'), 'r_sq'].iloc[0]) == pytest.approx(0.966381, abs=1e-5)

    def test_golden_r_sq_brand_b(self, dt_pred):
        result = metrics_group_table(dt_pred, group_vars=['market', 'superbrand'])
        assert float(result.loc[('market_1', 'brand_b'), 'r_sq'].iloc[0]) == pytest.approx(0.966364, abs=1e-5)

    def test_golden_rmse_brand_a(self, dt_pred):
        result = metrics_group_table(dt_pred, group_vars=['market', 'superbrand'])
        assert float(result.loc[('market_1', 'brand_a'), 'rmse'].iloc[0]) == pytest.approx(4.341551, abs=1e-5)

    def test_golden_rmse_brand_b(self, dt_pred):
        result = metrics_group_table(dt_pred, group_vars=['market', 'superbrand'])
        assert float(result.loc[('market_1', 'brand_b'), 'rmse'].iloc[0]) == pytest.approx(5.783333, abs=1e-5)

    def test_r_sq_in_expected_range(self, dt_pred):
        result = metrics_group_table(dt_pred, group_vars=['market', 'superbrand'])
        for val in result['r_sq']:
            assert 0.0 <= float(val) <= 1.0

    def test_rmse_non_negative(self, dt_pred):
        result = metrics_group_table(dt_pred, group_vars=['market', 'superbrand'])
        for val in result['rmse']:
            assert float(val) >= 0.0

    def test_smape_in_percentage_range(self, dt_pred):
        result = metrics_group_table(dt_pred, group_vars=['market', 'superbrand'])
        for val in result['smape']:
            assert 0.0 <= float(val) <= 200.0


class TestMetricsTable:

    def test_requires_real_model(self):
        dt_pred = pd.DataFrame({
            'listed': [2, 2],
            'observed': [100.0, 101.0],
            'predicted': [101.0, 102.0]
        })
        with pytest.raises(AttributeError):
            metrics_table(model=None, dt_pred=dt_pred)


class TestStatsNodeDispatch:
    """Stats functions accept ModelNode / MetaModel (node-level dispatch)."""

    @pytest.fixture
    def synthetic_model(self):
        np.random.seed(42)
        n = 40
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

        class _Conf:
            pass

        conf = _Conf()
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
            'variable': ['x1', 'x2'],
            'fun': ['coef', 'coef'],
            'arg': ['', ''],
            'value': [0.0, 0.0],
            'if_active': [1, 1],
        })
        conf.display_var_base = 'base'
        conf.summary_type = 'sum'

        model = Model.__new__(Model)
        model.conf = conf
        model.obj = type('Obj', (), {'models': {'main': ols_result}})()
        return model, ds, ols_result

    @pytest.fixture
    def dt_pred(self, synthetic_model):
        _, ds, _ = synthetic_model
        n = len(ds)
        return pd.DataFrame({
            'date': ds['date'],
            'listed': [2] * 30 + [3] * (n - 30),
            'observed': ds['y'],
            'predicted': ds['y'] + 1.0,
            'superbrand': ['brand_x'] * n,
            'market': ['market_1'] * n,
            'price_dummy': [1.0] * n,
        })

    @pytest.fixture
    def node(self, synthetic_model, dt_pred):
        model, ds, _ = synthetic_model
        node = ModelNode(model=model, ds=ds, model_rep=ModelRep(ds, language='english'))
        node.dt_pred = dt_pred
        return node

    def test_metrics_group_table_node(self, node):
        assert metrics_group_table(node).equals(metrics_group_table(node.dt_pred))

    def test_predictors_table_node(self, node):
        assert predictors_table(node).equals(predictors_table(node.model))

    def test_vif_table_node(self, node):
        assert vif_table(node).equals(vif_table(node.model.obj.models['main']))

    def test_basic_metrics_table_node(self, node):
        assert basic_metrics_table(node).equals(basic_metrics_table(node.dt_pred))

    def test_metrics_table_node(self, node):
        assert metrics_table(node).equals(metrics_table(node.model, node.dt_pred))

    def test_tests_table_ols_node(self, node):
        assert _tests_table_ols(node).equals(_tests_table_ols(node.model, node.dt_pred))

    def test_metamodel_dispatch_returns_dict(self, node):
        mm = MetaModel(nodes={'sales': node})
        for fn in (metrics_group_table, basic_metrics_table, predictors_table,
                   vif_table, metrics_table, _tests_table_ols):
            result = fn(mm)
            assert isinstance(result, dict)
            assert set(result.keys()) == {'sales'}

    def test_metamodel_dispatch_matches_node(self, node):
        mm = MetaModel(nodes={'sales': node})
        assert metrics_group_table(mm)['sales'].equals(metrics_group_table(node))
        assert predictors_table(mm)['sales'].equals(predictors_table(node))
        assert vif_table(mm)['sales'].equals(vif_table(node))
        assert basic_metrics_table(mm)['sales'].equals(basic_metrics_table(node))
        assert metrics_table(mm)['sales'].equals(metrics_table(node))
        assert _tests_table_ols(mm)['sales'].equals(_tests_table_ols(node))
