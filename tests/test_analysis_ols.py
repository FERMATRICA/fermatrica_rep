"""
Tests for analysis/ols.py: get_model_ds, trans, rhs_comparison, impacts_comparison.

Uses real FERMATRICA sample models loaded with fm.model_load_ext.
"""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from fermatrica_rep.analysis.ols import get_model_ds, trans, rhs_comparison, impacts_comparison


P00_MODEL_TAG = "2026-08-04_222911"


class TestGetModelDs:

    @pytest.fixture(scope="class")
    def model_and_ds(self, samples_path):
        import fermatrica as fm
        from pathlib import Path
        from tests.test_integration.conftest import _find_latest_model

        root = samples_path / "p00_sample"
        model_dir = root / "code_py" / "model_data"
        model_path = model_dir / f"model_{P00_MODEL_TAG}"
        if not model_path.is_dir():
            model_path = Path(_find_latest_model(model_dir))

        model, ds, _ = fm.model_load_ext(str(model_path), missed_stop=True)
        return model, ds

    @pytest.fixture
    def simple_ols(self, model_and_ds):
        return model_and_ds[0].obj.models['main']

    @pytest.fixture
    def ds(self, model_and_ds):
        return model_and_ds[1].copy()

    def test_returns_dataframe(self, simple_ols, ds):
        result = get_model_ds(simple_ols, ds)
        assert isinstance(result, pd.DataFrame)

    def test_has_expected_columns(self, simple_ols, ds):
        result = get_model_ds(simple_ols, ds)
        for col in ["date", "listed", "Intercept"]:
            assert col in result.columns, f"Missing column: {col}"
        for p in simple_ols.params.index:
            assert p in result.columns, f"Missing model param column: {p}"

    def test_has_intercept_column(self, simple_ols, ds):
        result = get_model_ds(simple_ols, ds)
        assert "Intercept" in result.columns

    def test_intercept_value_is_one(self, simple_ols, ds):
        result = get_model_ds(simple_ols, ds)
        assert (result["Intercept"] == 1.0).all()

    def test_excludes_listed_1(self, simple_ols, ds):
        ds_with_pretrain = pd.concat([
            ds,
            pd.DataFrame({
                "date": pd.date_range("2023-01-01", periods=5, freq="D"),
                "listed": [1] * 5,
                "x1": [0] * 5,
                "x2": [0] * 5,
            })
        ], ignore_index=True)
        result = get_model_ds(simple_ols, ds_with_pretrain)
        assert not (result["listed"] == 1).any(), "Listed=1 rows should be excluded"

    def test_finite_values(self, simple_ols, ds):
        result = get_model_ds(simple_ols, ds)
        num_cols = result.select_dtypes(include=[np.number]).columns
        assert np.all(np.isfinite(result[num_cols].values))

    def test_row_count_excludes_listed_1(self, simple_ols, ds):
        ds_with_pretrain = pd.concat([
            ds,
            pd.DataFrame({
                "date": pd.date_range("2023-01-01", periods=5, freq="D"),
                "listed": [1] * 5,
                "x1": [0] * 5,
                "x2": [0] * 5,
            })
        ], ignore_index=True)
        result = get_model_ds(simple_ols, ds_with_pretrain)
        listed_not_1 = ds_with_pretrain[ds_with_pretrain["listed"] != 1]
        assert len(result) == len(listed_not_1)


class TestTrans:

    @pytest.fixture
    def params_df(self):
        return pd.DataFrame({
            "variable": ["x1", "x1", "x2", "x2"],
            "fun": ["lag", "lag", "lag", "lag"],
            "arg": ["n", "cval", "n", "cval"],
            "value": [2, 0.0, 3, 0.0],
            "index_vars": [np.nan, np.nan, np.nan, np.nan],
        })

    @pytest.fixture
    def ds(self):
        return pd.DataFrame({
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "x2": [10.0, 20.0, 30.0, 40.0, 50.0],
            "date": pd.date_range("2024-01-01", periods=5, freq="D"),
            "listed": [2] * 5,
            "const": [1.0] * 5,
        })

    def test_returns_dataframe(self, ds, params_df):
        result = trans(ds, params_df)
        assert isinstance(result, pd.DataFrame)

    def test_adds_transformed_columns(self, ds, params_df):
        result = trans(ds, params_df)
        for col in ["x1_lag", "x2_lag"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_transformed_values_correct(self, ds, params_df):
        result = trans(ds, params_df)
        pd.testing.assert_series_equal(
            result["x1_lag"], ds["x1"].shift(2).fillna(0.0), check_names=False
        )
        pd.testing.assert_series_equal(
            result["x2_lag"], ds["x2"].shift(3).fillna(0.0), check_names=False
        )


class TestRhsComparison:

    @pytest.fixture
    def models(self):
        np.random.seed(42)
        n = 30
        x1 = np.random.uniform(0, 10, n)
        x2 = np.random.uniform(0, 5, n)
        y = 2 * x1 + 3 * x2 + 5 + np.random.normal(0, 0.5, n)
        X = sm.add_constant(pd.DataFrame({"x1": x1, "x2": x2}))
        model_init = sm.OLS(y, X).fit()

        y_upd = 2.1 * x1 + 2.8 * x2 + 4.5 + np.random.normal(0, 0.5, n)
        model_upd = sm.OLS(y_upd, X).fit()
        return model_init, model_upd

    def test_returns_dataframe(self, models):
        result = rhs_comparison(*models)
        assert isinstance(result, pd.DataFrame)

    def test_has_expected_columns(self, models):
        result = rhs_comparison(*models)
        assert "index" in result.columns
        assert "coeff_init" in result.columns
        assert "coeff_upd" in result.columns
        assert "coeff_diff_prc" in result.columns
        assert "sign_change" in result.columns

    def test_has_model_parameters_as_rows(self, models):
        result = rhs_comparison(*models)
        expected = ["const", "x1", "x2"]
        for p in expected:
            assert p in result["index"].values, f"Missing parameter: {p}"

    def test_sign_change_ok_when_signs_match(self, models):
        result = rhs_comparison(*models)
        assert (result["sign_change"] == "OK").all()

    def test_sign_change_detected(self, models):
        model_init, model_upd = models
        params_upd = -model_upd.params
        model_upd.params[:] = params_upd.values
        # Force sign reversal on a copy to avoid modifying fixture
        np.random.seed(1)
        n = 30
        x1 = np.random.uniform(0, 10, n)
        x2 = np.random.uniform(0, 5, n)
        y = -2 * x1 + 3 * x2 + 5 + np.random.normal(0, 0.5, n)
        X = sm.add_constant(pd.DataFrame({"x1": x1, "x2": x2}))
        model_init2 = sm.OLS(y, X).fit()

        y2 = 2 * x1 + 3 * x2 + 5 + np.random.normal(0, 0.5, n)
        model_upd2 = sm.OLS(y2, X).fit()
        result = rhs_comparison(model_init2, model_upd2)
        sign_changed = result[result["sign_change"] == "Sign_changed!"]
        assert "x1" in sign_changed["index"].values


class TestImpactsComparison:

    @pytest.fixture
    def models_and_ds(self):
        np.random.seed(42)
        n = 30
        x1 = np.random.uniform(0, 10, n)
        x2 = np.random.uniform(0, 5, n)
        y = 2 * x1 + 3 * x2 + 5 + np.random.normal(0, 0.5, n)
        X = sm.add_constant(pd.DataFrame({"x1": x1, "x2": x2}))
        model_init = sm.OLS(y, X).fit()

        y_upd = 2.1 * x1 + 2.8 * x2 + 4.5 + np.random.normal(0, 0.5, n)
        model_upd = sm.OLS(y_upd, X).fit()

        ds_init = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "listed": [2] * n,
            "const": 1.0,
            "x1": x1,
            "x2": x2,
        })
        ds_upd = ds_init.copy()
        return model_init, model_upd, ds_init, ds_upd

    def test_returns_dataframe(self, models_and_ds):
        result = impacts_comparison(*models_and_ds)
        assert isinstance(result, pd.DataFrame)

    def test_has_expected_columns(self, models_and_ds):
        result = impacts_comparison(*models_and_ds)
        for col in ["imp_init", "imp_upd", "diff_prc"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_has_model_parameters_as_index(self, models_and_ds):
        result = impacts_comparison(*models_and_ds)
        expected = ["const", "x1", "x2"]
        for p in expected:
            assert p in result.index, f"Missing index: {p}"

    def test_impacts_sum_to_100(self, models_and_ds):
        result = impacts_comparison(*models_and_ds)
        assert abs(result["imp_init"].sum() - 100) < 1e-10
        assert abs(result["imp_upd"].sum() - 100) < 1e-10
