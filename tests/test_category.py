import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pytest

from fermatrica_rep.analysis.category import category_data, category_main_plot, category_err_plot, category_plot
from fermatrica_rep.meta_model.node import ModelNode
from fermatrica_rep.meta_model.meta_model import MetaModel


class MockConf:
    Y_var = "observed"


class MockModel:
    conf = MockConf()


class TestCategoryData:

    @pytest.fixture
    def ds(self):
        """Synthetic dataset with columns required by category_data()."""
        np.random.seed(42)
        rows = []
        for d in pd.date_range("2024-01-01", periods=12, freq="MS"):
            obs = np.random.uniform(50, 150)
            rows.append({
                "date": d,
                "pred_mrk": obs * np.random.uniform(0.9, 1.1),
                "observed": obs,
                "listed": 2,
            })
        return pd.DataFrame(rows)

    @pytest.fixture
    def model(self):
        return MockModel()

    def test_returns_dataframe(self, model, ds):
        result = category_data(model=model, ds=ds, period="month")
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_expected_columns(self, model, ds):
        result = category_data(model=model, ds=ds, period="month")
        for col in ["date", "observed", "predicted", "observed_lower", "observed_upper"]:
            assert col in result.columns

    def test_finite_values(self, model, ds):
        result = category_data(model=model, ds=ds, period="month")
        num_cols = result.select_dtypes(include=[np.number]).columns
        assert np.all(np.isfinite(result[num_cols].values))

    def test_observed_lower_upper(self, model, ds):
        result = category_data(model=model, ds=ds, period="month", err_int=0.1)
        assert np.all(result["observed_lower"] <= result["observed"])
        assert np.all(result["observed_upper"] >= result["observed"])

    def test_show_future_filters(self, model, ds):
        ds_future = pd.concat([ds, pd.DataFrame({
            "date": pd.date_range("2025-01-01", periods=3, freq="MS"),
            "pred_mrk": [100.0] * 3,
            "observed": [100.0] * 3,
            "listed": [4] * 3,
        })], ignore_index=True)
        result_no_future = category_data(model=model, ds=ds_future, period="month", show_future=False)
        result_future = category_data(model=model, ds=ds_future, period="month", show_future=True)
        assert len(result_no_future) < len(result_future)

    def test_period_week(self, model, ds):
        result = category_data(model=model, ds=ds, period="week")
        assert isinstance(result, pd.DataFrame)

    def test_period_year(self, model, ds):
        result = category_data(model=model, ds=ds, period="year")
        assert isinstance(result, pd.DataFrame)

    def test_err_int_custom(self, model, ds):
        result = category_data(model=model, ds=ds, period="month", err_int=0.05)
        assert "observed_lower" in result.columns


class TestCategoryMainPlot:

    @pytest.fixture
    def ds_prepared(self):
        """Data matching the output shape of category_data()."""
        return pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=6, freq="MS"),
            "observed": [100.0, 110.0, 105.0, 120.0, 115.0, 130.0],
            "predicted": [98.0, 108.0, 107.0, 118.0, 113.0, 128.0],
            "observed_lower": [90.0, 99.0, 94.5, 108.0, 103.5, 117.0],
            "observed_upper": [110.0, 121.0, 115.5, 132.0, 126.5, 143.0],
        })

    def test_returns_go_figure(self, ds_prepared):
        fig = make_subplots(rows=1, cols=1)
        result = category_main_plot(ds=ds_prepared, fig=fig)
        assert isinstance(result, go.Figure)

    def test_adds_three_traces(self, ds_prepared):
        fig = make_subplots(rows=1, cols=1)
        result = category_main_plot(ds=ds_prepared, fig=fig)
        assert len(result.data) == 3

    def test_trace_names(self, ds_prepared):
        fig = make_subplots(rows=1, cols=1)
        result = category_main_plot(ds=ds_prepared, fig=fig)
        names = [t.name for t in result.data]
        assert "Observed" in names
        assert "Predicted" in names

    def test_preserves_existing_traces(self, ds_prepared):
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Scatter(x=[1], y=[1], name="preexisting"))
        result = category_main_plot(ds=ds_prepared, fig=fig)
        assert len(result.data) == 4
        assert result.data[0].name == "preexisting"

    def test_row_col_passed_to_add_trace(self, ds_prepared):
        fig = make_subplots(rows=2, cols=1)
        result = category_main_plot(ds=ds_prepared, fig=fig, row_n=2, col_n=1)
        assert isinstance(result, go.Figure)

    def test_empty_data_raises(self):
        empty = pd.DataFrame(columns=["date", "observed", "predicted", "observed_lower", "observed_upper"])
        fig = go.Figure()
        with pytest.raises(Exception):
            category_main_plot(ds=empty, fig=fig)


class TestCategoryErrPlot:

    @pytest.fixture
    def ds_prepared(self):
        return pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=6, freq="MS"),
            "observed": [100.0, 110.0, 105.0, 120.0, 115.0, 130.0],
            "predicted": [98.0, 108.0, 107.0, 118.0, 113.0, 128.0],
            "observed_lower": [90.0, 99.0, 94.5, 108.0, 103.5, 117.0],
            "observed_upper": [110.0, 121.0, 115.5, 132.0, 126.5, 143.0],
        })

    def test_returns_go_figure(self, ds_prepared):
        fig = make_subplots(rows=1, cols=1)
        result = category_err_plot(ds=ds_prepared, fig=fig)
        assert isinstance(result, go.Figure)

    def test_adds_three_traces(self, ds_prepared):
        fig = make_subplots(rows=1, cols=1)
        result = category_err_plot(ds=ds_prepared, fig=fig)
        assert len(result.data) == 3

    def test_has_error_column(self, ds_prepared):
        fig = make_subplots(rows=1, cols=1)
        result = category_err_plot(ds=ds_prepared, fig=fig)
        names = [t.name for t in result.data]
        assert "Prediction Error" in names

    def test_err_int_affects_horizontal_lines(self, ds_prepared):
        fig = make_subplots(rows=1, cols=1)
        result = category_err_plot(ds=ds_prepared, fig=fig, err_int=0.2)
        y_vals = []
        for t in result.data:
            if t.name == "":
                y_vals.extend(t.y)
        assert any(abs(y - 0.2) < 1e-9 for y in y_vals)
        assert any(abs(y + 0.2) < 1e-9 for y in y_vals)

    def test_preserves_existing_traces(self, ds_prepared):
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Scatter(x=[1], y=[1], name="preexisting"))
        result = category_err_plot(ds=ds_prepared, fig=fig)
        assert len(result.data) == 4

    def test_empty_data_raises(self):
        empty = pd.DataFrame(columns=["date", "observed", "predicted", "observed_lower", "observed_upper"])
        fig = go.Figure()
        with pytest.raises(Exception):
            category_err_plot(ds=empty, fig=fig)


class TestCategoryPlot:

    @pytest.fixture
    def ds(self):
        np.random.seed(42)
        rows = []
        for d in pd.date_range("2024-01-01", periods=12, freq="MS"):
            obs = np.random.uniform(50, 150)
            rows.append({
                "date": d,
                "pred_mrk": obs * np.random.uniform(0.9, 1.1),
                "observed": obs,
                "listed": 2,
            })
        return pd.DataFrame(rows)

    @pytest.fixture
    def model(self):
        return MockModel()

    def test_returns_go_figure(self, model, ds):
        result = category_plot(model=model, ds=ds, period="month")
        assert isinstance(result, go.Figure)

    def test_has_data_traces(self, model, ds):
        result = category_plot(model=model, ds=ds, period="month")
        assert len(result.data) > 0

    def test_two_subplots(self, model, ds):
        result = category_plot(model=model, ds=ds, period="month")
        assert result._get_subplot_rows_columns() is not None

    def test_err_int_passed_through(self, model, ds):
        result = category_plot(model=model, ds=ds, period="month", err_int=0.05)
        assert isinstance(result, go.Figure)

    def test_metamodel_returns_dict(self, model, ds):
        node = ModelNode(model=model, ds=ds)
        mm = MetaModel(nodes={"cat": node})
        result = category_plot(mm)
        assert isinstance(result, dict)
        assert "cat" in result
        assert isinstance(result["cat"], go.Figure)

