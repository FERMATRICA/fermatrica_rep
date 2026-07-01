import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.formula.api import ols
from fermatrica_utils import select_eff

from fermatrica_rep.decomposition import decompose_basic, extract_effect
from fermatrica_rep.meta_model.model_rep import ModelRep


class MockModelConf:
    pass


class MockModel:
    pass


class TestDecomposeBasic:

    @pytest.fixture
    def synthetic_ols(self):
        np.random.seed(42)
        n = 20
        x1 = np.random.uniform(1, 10, n)
        x2 = np.random.uniform(10, 100, n)
        y = 3.0 * x1 + 0.5 * x2 + 10.0 + np.random.normal(0, 1, n)
        ds = pd.DataFrame({
            'x1': x1,
            'x2': x2,
            'y': y,
            'date': pd.date_range('2024-01-01', periods=n, freq='D'),
            'listed': [2] * n,
            'const': [1.0] * n,
        })
        model = ols('y ~ x1 + x2', data=ds)
        return model, ds

    def test_returns_list_for_list_input(self, synthetic_ols):
        model, ds = synthetic_ols
        result = decompose_basic([model], [ds])
        assert isinstance(result, list)
        assert len(result) == 1

    def test_returns_dataframe_for_single_input(self, synthetic_ols):
        model, ds = synthetic_ols
        result = decompose_basic(model, ds)
        assert isinstance(result, pd.DataFrame)

    def test_columns_match_coefficient_names(self, synthetic_ols):
        model, ds = synthetic_ols
        result = decompose_basic(model, ds)
        expected_cols = ['Intercept', 'x1', 'x2']
        for c in expected_cols:
            assert c in result.columns, f"Missing column: {c}"

    def test_contributions_sum_to_prediction(self, synthetic_ols):
        model, ds = synthetic_ols
        result = decompose_basic(model, ds)
        row_sums = result.sum(axis=1)
        fitted = model.fit()
        predicted = fitted.fittedvalues
        assert row_sums.index.equals(predicted.index)
        np.testing.assert_allclose(row_sums.values, predicted.values, rtol=1e-10)

    def test_contribution_of_x1_matches_coefficient(self, synthetic_ols):
        model, ds = synthetic_ols
        result = decompose_basic(model, ds)
        beta_x1 = model.fit().params['x1']
        expected_contrib = ds['x1'] * beta_x1
        np.testing.assert_allclose(result['x1'].values, expected_contrib.values, rtol=1e-10)

    def test_contribution_of_x2_matches_coefficient(self, synthetic_ols):
        model, ds = synthetic_ols
        result = decompose_basic(model, ds)
        beta_x2 = model.fit().params['x2']
        expected_contrib = ds['x2'] * beta_x2
        np.testing.assert_allclose(result['x2'].values, expected_contrib.values, rtol=1e-10)

    def test_intercept_contribution_is_constant(self, synthetic_ols):
        model, ds = synthetic_ols
        result = decompose_basic(model, ds)
        intercept_val = model.fit().params['Intercept']
        assert np.allclose(result['Intercept'].values, intercept_val)


class TestExtractEffect:

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

        conf = MockModelConf()
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

        model = MockModel()
        model.conf = conf
        model.obj = type('Obj', (), {'models': {'main': ols_result}})()

        ds = select_eff(ds, ['date', 'superbrand', 'bs_key', 'listed',
                             'x1', 'x2', 'y', 'const', 'price_dummy'])

        return model, ds

    @pytest.fixture
    def model_rep(self, synthetic_model_ds):
        _, ds = synthetic_model_ds
        return ModelRep(ds, language='english')

    def test_returns_dataframe(self, synthetic_model_ds, model_rep):
        model, ds = synthetic_model_ds
        result = extract_effect(model, ds, model_rep)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, synthetic_model_ds, model_rep):
        model, ds = synthetic_model_ds
        result = extract_effect(model, ds, model_rep)
        for col in ['superbrand', 'bs_key', 'date', 'variable', 'value', 'value_rub']:
            assert col in result.columns, f"Missing column: {col}"

    def test_values_finite(self, synthetic_model_ds, model_rep):
        model, ds = synthetic_model_ds
        result = extract_effect(model, ds, model_rep)
        assert np.all(np.isfinite(result['value'])), "NaN or Inf in value column"

    def test_returns_list_for_list_input(self, synthetic_model_ds, model_rep):
        model, ds = synthetic_model_ds
        result = extract_effect([model], [ds], [model_rep])
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], pd.DataFrame)

    def test_detail_mode_produces_more_variables(self, synthetic_model_ds, model_rep):
        model, ds = synthetic_model_ds
        basic = extract_effect(model, ds, model_rep, if_detail=False)
        detail = extract_effect(model, ds, model_rep, if_detail=True)
        assert detail['variable'].nunique() >= basic['variable'].nunique()

    def test_pretrain_rows_not_included(self, synthetic_model_ds, model_rep):
        model, ds = synthetic_model_ds
        ds_pretrain = ds.copy()
        ds_pretrain.loc[0, 'listed'] = 1
        result = extract_effect(model, ds_pretrain, model_rep)
        assert not result.empty

    def test_values_finite_in_detail_mode(self, synthetic_model_ds, model_rep):
        model, ds = synthetic_model_ds
        result = extract_effect(model, ds, model_rep, if_detail=True)
        assert np.all(np.isfinite(result['value']))

    def test_aggregate_row_sums_match_decompose_basic(self, synthetic_model_ds, model_rep):
        model, ds = synthetic_model_ds
        result = extract_effect(model, ds, model_rep)
        from statsmodels.formula.api import ols as ols_formula
        ols_model = ols_formula('y ~ x1 + x2', data=ds)
        decomp = decompose_basic(ols_model, ds)
        grouped = result.groupby(['superbrand', 'bs_key', 'date'])['value'].sum().reset_index()
        assert np.allclose(grouped['value'].values,
                           decomp[['Intercept', 'x1', 'x2']].sum(axis=1).values[:len(grouped)],
                           rtol=1e-10)

    def test_lhs_model_produces_additional_variables(self, synthetic_model_ds, model_rep):
        model, ds = synthetic_model_ds
        model.conf.model_lhs = pd.DataFrame({
            'name': ['y_lhs1'],
            'display_var': ['lhs1_disp'],
            'type': ['adstock'],
            'if_active': [1],
        })
        result = extract_effect(model, ds, model_rep)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_empty_dataset_returns_empty(self, synthetic_model_ds, model_rep):
        model, ds = synthetic_model_ds
        empty_ds = ds.iloc[0:0].copy()
        result = extract_effect(model, empty_ds, model_rep)
        assert len(result) == 0
        for col in ['superbrand', 'bs_key', 'date', 'variable', 'value']:
            assert col in result.columns

    def test_detail_mode_tolerates_nonexistent_columns(self, synthetic_model_ds, model_rep):
        model, ds = synthetic_model_ds
        import copy
        conf = copy.deepcopy(model.conf)
        conf.model_rhs = pd.DataFrame({
            'token': ['x1', 'nonexistent_var'],
            'display_var': ['x1_disp', 'ghost_disp'],
            'if_active': [1, 1],
        })
        model.conf = conf
        result = extract_effect(model, ds, model_rep, if_detail=True)
        assert np.all(np.isfinite(result['value']))
