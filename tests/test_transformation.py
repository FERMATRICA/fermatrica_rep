import numpy as np
import pytest
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from fermatrica_rep.analysis.transformation import transformation_main_plot, transformation_plot


class TestTransformationMainPlot:

    @pytest.fixture
    def ds(self):
        return pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10, freq='MS'),
            'listed': [2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
            'raw_var': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            'raw_var_adstock': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        })

    def test_returns_subplot_figure(self, ds):
        fig = transformation_main_plot(ds, 'raw_var', 'raw_var_adstock')
        assert isinstance(fig, go.Figure)
        assert hasattr(fig, 'data')

    def test_two_traces_added(self, ds):
        fig = transformation_main_plot(ds, 'raw_var', 'raw_var_adstock')
        assert len(fig.data) >= 2

    def test_has_secondary_yaxis(self, ds):
        fig = transformation_main_plot(ds, 'raw_var', 'raw_var_adstock')
        assert 'yaxis2' in fig.layout or any(
            'yaxis2' in str(trace.yaxis) for trace in fig.data
        )

    def test_title_contains_transformation_name(self, ds):
        fig = transformation_main_plot(ds, 'raw_var', 'raw_var_adstock')
        title_parts = ' '.join('raw_var_adstock'.replace('raw_var', '').split('_'))
        assert title_parts in (fig.layout.title.text or '')

    def test_height_param(self, ds):
        fig = transformation_main_plot(ds, 'raw_var', 'raw_var_adstock', height=500)
        assert fig.layout.height == 500

    def test_default_height_300(self, ds):
        fig = transformation_main_plot(ds, 'raw_var', 'raw_var_adstock')
        assert fig.layout.height == 300

    def test_if_main_false_skips_axes_styling(self, ds):
        fig = transformation_main_plot(ds, 'raw_var', 'raw_var_adstock', if_main=False)
        assert fig.layout.height == 300

    def test_first_trace_y_matches_raw_var(self, ds):
        fig = transformation_main_plot(ds, 'raw_var', 'raw_var_adstock')
        tmp = ds.loc[ds['listed'].isin([2, 3])].groupby('date')[['raw_var', 'raw_var_adstock']].sum()
        np.testing.assert_array_almost_equal(fig.data[0].y, tmp['raw_var'].values)

    def test_second_trace_y_matches_transformed_var(self, ds):
        fig = transformation_main_plot(ds, 'raw_var', 'raw_var_adstock')
        tmp = ds.loc[ds['listed'].isin([2, 3])].groupby('date')[['raw_var', 'raw_var_adstock']].sum()
        np.testing.assert_array_almost_equal(fig.data[1].y, tmp['raw_var_adstock'].values)

    def test_trace_x_is_datetime_from_date(self, ds):
        fig = transformation_main_plot(ds, 'raw_var', 'raw_var_adstock')
        tmp = ds.loc[ds['listed'].isin([2, 3])].groupby('date')[['raw_var', 'raw_var_adstock']].sum()
        np.testing.assert_array_equal(fig.data[0].x, tmp.index.values)

    def test_works_with_listed_2_and_3_only(self, ds):
        ds_extended = pd.concat([
            ds,
            pd.DataFrame({
                'date': pd.date_range('2025-01-01', periods=2, freq='MS'),
                'listed': [1, 4],
                'raw_var': [100.0, 200.0],
                'raw_var_adstock': [50.0, 100.0]
            })
        ], ignore_index=True)
        fig = transformation_main_plot(ds_extended, 'raw_var', 'raw_var_adstock')
        assert isinstance(fig, go.Figure)


class TestTransformationPlot:

    @pytest.fixture
    def ds(self):
        return pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10, freq='MS'),
            'listed': [2] * 10,
            'raw_var': list(range(1, 11)),
            'raw_var_trans_adstock': [x * 0.5 for x in range(1, 11)]
        })

    def test_sub_false_returns_single_dict(self, ds):
        result = transformation_plot(ds, 'raw_var', 'raw_var_trans_adstock', sub=False)
        assert isinstance(result, dict)
        assert 'main' in result

    def test_sub_true_returns_chain(self, ds):
        ds['raw_var_trans'] = [x * 0.8 for x in range(1, 11)]
        result = transformation_plot(ds, 'raw_var', 'raw_var_trans_adstock', sub=True)
        assert isinstance(result, dict)
        assert len(result) >= 1

    def test_chain_contains_only_plotly_figures(self, ds):
        result = transformation_plot(ds, 'raw_var', 'raw_var_trans_adstock', sub=False)
        assert isinstance(result['main'], go.Figure)

    def test_short_transformation_name(self, ds):
        ds['raw_var_adj'] = [x * 0.9 for x in range(1, 11)]
        result = transformation_plot(ds, 'raw_var', 'raw_var_adj', sub=False)
        assert 'main' in result

    def test_sub_false_trace_y_matches_raw_var(self, ds):
        ds['raw_var_trans'] = [x * 0.8 for x in range(1, 11)]
        ds['raw_var_trans_adstock'] = [x * 0.5 for x in range(1, 11)]
        result = transformation_plot(ds, 'raw_var', 'raw_var_trans_adstock', sub=False)
        fig = result['main']
        tmp = ds.loc[ds['listed'].isin([2, 3])].groupby('date')[['raw_var', 'raw_var_trans_adstock']].sum()
        np.testing.assert_array_almost_equal(fig.data[0].y, tmp['raw_var'].values)

    def test_sub_true_chain_contains_intermediate_var(self, ds):
        ds['raw_var_trans'] = [x * 0.8 for x in range(1, 11)]
        ds['raw_var_trans_adstock'] = [x * 0.5 for x in range(1, 11)]
        result = transformation_plot(ds, 'raw_var', 'raw_var_trans_adstock', sub=True)
        assert 0 in result
        assert isinstance(result[0], go.Figure)

    def test_sub_true_chain_first_link_plots_raw_vs_intermediate(self, ds):
        ds['raw_var_trans'] = [x * 0.8 for x in range(1, 11)]
        ds['raw_var_trans_adstock'] = [x * 0.5 for x in range(1, 11)]
        result = transformation_plot(ds, 'raw_var', 'raw_var_trans_adstock', sub=True)
        fig = result[0]
        init_var = 'raw_var'
        fin_var = 'raw_var_trans'
        tmp = ds.loc[ds['listed'].isin([2, 3])].groupby('date')[[init_var, fin_var]].sum()
        np.testing.assert_array_almost_equal(fig.data[0].y, tmp[init_var].values)
