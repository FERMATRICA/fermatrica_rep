import pytest
import pandas as pd
import plotly.graph_objects as go

from fermatrica_rep.analysis.waterfall import waterfall_data, waterfall_plot, _waterfall_data_worker


class TestWaterfallData:

    def test_date_filtering_outside_range(self, split_m_m):
        result = waterfall_data(split_m_m, brands=['brand_1'],
                                date_start='2025-01-01', date_end='2025-12-31')
        assert len(result) == 0

    def test_brand_filtering(self, split_m_m):
        result = waterfall_data(split_m_m, brands=['other_brand'],
                                date_start='2024-01-01', date_end='2024-12-31')
        assert len(result) == 0

    def test_absolute_sort_descending(self, split_m_m):
        result = waterfall_data(split_m_m, brands=['brand_1'],
                                date_start='2024-01-01', date_end='2024-12-31',
                                absolute_sort=True)
        abs_vals = result['value'].abs().values
        for i in range(len(abs_vals) - 1):
            assert abs_vals[i] >= abs_vals[i + 1]

    def test_signed_sort_descending(self, split_m_m):
        result = waterfall_data(split_m_m, brands=['brand_1'],
                                date_start='2024-01-01', date_end='2024-12-31',
                                absolute_sort=False)
        vals = result['value'].values
        for i in range(len(vals) - 1):
            assert vals[i] >= vals[i + 1]

    def test_multiple_brands_mixed(self, split_m_m):
        mixed = pd.concat([
            split_m_m,
            pd.DataFrame({
                'date': pd.date_range('2024-01-01', periods=5, freq='MS'),
                'superbrand': ['brand_2'] * 5,
                'bs_key': [1] * 5,
                'listed': [2] * 5,
                'variable': ['tv', 'digital', 'pr', 'base', 'other'],
                'value': [5.0, 3.0, 1.0, 2.0, 4.0]
            })
        ], ignore_index=True)
        result = waterfall_data(mixed, brands=['brand_1', 'brand_2'],
                                date_start='2024-01-01', date_end='2024-12-31')
        assert len(result) > 0

    def test_pretrain_excluded(self, split_m_m):
        with_pretrain = pd.concat([
            split_m_m,
            pd.DataFrame({
                'date': pd.date_range('2023-01-01', periods=3, freq='MS'),
                'superbrand': ['brand_1'] * 3,
                'bs_key': [1] * 3,
                'listed': [1] * 3,
                'variable': ['pretrain_var'] * 3,
                'value': [100.0, 200.0, 300.0]
            })
        ], ignore_index=True)
        result_no_pretrain = waterfall_data(with_pretrain, brands=['brand_1'],
                                            date_start='2023-01-01', date_end='2024-12-31',
                                            show_pretrain=False)
        result_with_pretrain = waterfall_data(with_pretrain, brands=['brand_1'],
                                             date_start='2023-01-01', date_end='2024-12-31',
                                             show_pretrain=True)
        assert 'pretrain_var' not in result_no_pretrain['variable'].values
        assert 'pretrain_var' in result_with_pretrain['variable'].values

    def test_abs_column_present(self, split_m_m):
        result = waterfall_data(split_m_m, brands=['brand_1'],
                                date_start='2024-01-01', date_end='2024-12-31')
        assert 'abs' in result.columns

    def test_list_input(self, split_m_m):
        result = waterfall_data([split_m_m, split_m_m], brands=['brand_1'],
                                date_start='2024-01-01', date_end='2024-12-31')
        assert isinstance(result, list)
        assert len(result) == 2

    def test_nonexistent_brand_returns_empty(self, split_m_m):
        result = waterfall_data(split_m_m, brands=['nonexistent_brand'],
                                date_start='2024-01-01', date_end='2024-12-31')
        assert len(result) == 0

    def test_date_range_before_data_returns_empty(self, split_m_m):
        result = waterfall_data(split_m_m, brands=['brand_1'],
                                date_start='2020-01-01', date_end='2020-12-31')
        assert len(result) == 0


class TestWaterfallPlot:

    def test_returns_go_figure_with_expected_traces(self, split_m_m, basic_model_rep):
        fig = waterfall_plot(split_m_m, brands=['brand_1'], model_rep=basic_model_rep,
                             date_start='2024-01-01', date_end='2024-12-31',
                             pos_neg=False)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2

    def test_returns_go_figure_native(self, split_m_m, basic_model_rep):
        fig = waterfall_plot(split_m_m, brands=['brand_1'], model_rep=basic_model_rep,
                             date_start='2024-01-01', date_end='2024-12-31',
                             pos_neg=True)
        assert isinstance(fig, go.Figure)

    def test_list_input(self, split_m_m, basic_model_rep):
        figs = waterfall_plot([split_m_m], brands=['brand_1'], model_rep=[basic_model_rep],
                              date_start='2024-01-01', date_end='2024-12-31')
        assert isinstance(figs, list)

    def test_has_hline_at_100(self, split_m_m, basic_model_rep):
        fig = waterfall_plot(split_m_m, brands=['brand_1'], model_rep=basic_model_rep,
                             date_start='2024-01-01', date_end='2024-12-31')
        shapes = fig.layout.shapes
        assert len(shapes) > 0, "Expected at least one shape (hline) in waterfall plot"
        hlines_at_100 = [
            s for s in shapes
            if (hasattr(s, 'y0') and s.y0 == 100) or (hasattr(s, 'y1') and s.y1 == 100)
        ]
        assert len(hlines_at_100) >= 1, "Expected at least one horizontal line at y=100 in waterfall plot"

    def test_layout_has_xaxis_title(self, split_m_m, basic_model_rep):
        fig = waterfall_plot(split_m_m, brands=['brand_1'], model_rep=basic_model_rep,
                             date_start='2024-01-01', date_end='2024-12-31')
        assert fig.layout.xaxis.title.text is not None
        assert len(fig.layout.xaxis.title.text) > 0
