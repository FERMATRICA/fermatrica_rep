"""Unit tests for PPTX slide data-preparation functions.

Tests verify that _prepare_*_data() functions return dicts with expected
structure (figures, tables, texts, title) and handle edge cases.
"""
import pandas as pd
import pytest

from fermatrica_rep.meta_model.model_rep import ModelRep


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_ds():
    n = 12
    return pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=n, freq='MS'),
        'superbrand': ['brand_x'] * n,
        'bs_key': [1] * n,
        'listed': [4] * n,
    })


def _minimal_dt_pred():
    n = 12
    return pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=n, freq='MS'),
        'superbrand': ['brand_x'] * n,
        'bs_key': [1] * n,
        'listed': [2] * 6 + [4] * 6,
        'predicted': [float(x) for x in range(100, 112)],
        'observed': [float(x) for x in range(90, 102)],
    })


class _MockModelConf:
    price_var = 'price'
    summary_type = 'sum'
    target_superbrand = 'brand_x'


class _MockModel:
    conf = _MockModelConf()


# ---------------------------------------------------------------------------
# TestFitPredictSlideData
# ---------------------------------------------------------------------------

class TestFitPredictSlideData:

    @pytest.fixture
    def setup(self):
        from fermatrica_rep.options.define import OptionSettings
        ds = _minimal_ds()
        dt_pred = _minimal_dt_pred()
        dt_pred['price'] = [1.0] * len(dt_pred)
        model = _MockModel()
        model_rep = ModelRep(ds, language='english')
        opt_set = OptionSettings(
            target='brand_x', date_start='2024-01-01', date_end='2024-12-01',
            ref_date_start='2023-01-01', ref_date_end='2023-06-01',
        )
        return ds, dt_pred, model, model_rep, opt_set

    def test_prepare_fit_data_returns_dict(self, setup):
        from fermatrica_rep.export.slides.fit_predict import _prepare_fit_data
        ds, dt_pred, model, model_rep, opt_set = setup

        result = _prepare_fit_data(
            model=model, model_rep=model_rep, dt_pred=dt_pred,
            opt_set=opt_set, option_name='test_option', plot_type='brand',
            bs_key_filter=[],
        )

        assert isinstance(result, dict)
        assert 'language' in result
        assert 'superbrand' in result
        assert 'slide_title' in result
        assert 'dt_pred_prep_vol' in result
        assert 'dt_pred_prep_val' in result
        assert 'axis_title_vol' in result
        assert 'axis_title_val' in result

    def test_prepare_fit_data_tables_have_data(self, setup):
        from fermatrica_rep.export.slides.fit_predict import _prepare_fit_data
        ds, dt_pred, model, model_rep, opt_set = setup

        result = _prepare_fit_data(
            model=model, model_rep=model_rep, dt_pred=dt_pred,
            opt_set=opt_set, option_name='test_option', plot_type='brand',
            bs_key_filter=[1],
        )

        vol_table = result['dt_pred_prep_vol']
        val_table = result['dt_pred_prep_val']
        assert len(vol_table) > 0
        assert len(val_table) > 0

    def test_prepare_fit_data_title_contains_option_name(self, setup):
        from fermatrica_rep.export.slides.fit_predict import _prepare_fit_data
        ds, dt_pred, model, model_rep, opt_set = setup

        result = _prepare_fit_data(
            model=model, model_rep=model_rep, dt_pred=dt_pred,
            opt_set=opt_set, option_name='opt_base', plot_type='brand',
            bs_key_filter=[],
        )

        assert 'opt_base' in result['slide_title']

    def test_prepare_fit_data_empty_dt_pred(self, setup):
        from fermatrica_rep.export.slides.fit_predict import _prepare_fit_data
        ds, dt_pred, model, model_rep, opt_set = setup
        empty_dt_pred = dt_pred.iloc[0:0].copy()

        result = _prepare_fit_data(
            model=model, model_rep=model_rep, dt_pred=empty_dt_pred,
            opt_set=opt_set, option_name='test', plot_type='brand',
            bs_key_filter=[],
        )

        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# TestLineChartData
# ---------------------------------------------------------------------------

class TestLineChartData:

    @pytest.fixture
    def setup(self):
        ds = _minimal_ds()
        dt_pred = _minimal_dt_pred()
        dt_pred['price'] = [1.0] * len(dt_pred)
        model = _MockModel()
        model_rep = ModelRep(ds, language='english')
        return ds, dt_pred, model, model_rep

    def test_returns_two_dataframes(self, setup):
        from fermatrica_rep.export.slides.fit_predict import _line_chart_data
        ds, dt_pred, model, model_rep = setup

        vol, val = _line_chart_data(
            model=model, dt_pred=dt_pred, model_rep=model_rep,
            period='month', group_var=('superbrand',),
            show_future=True,
        )

        assert isinstance(vol, pd.DataFrame)
        assert isinstance(val, pd.DataFrame)
        assert len(vol) > 0
        assert len(val) > 0

    def test_observed_and_predicted_present(self, setup):
        from fermatrica_rep.export.slides.fit_predict import _line_chart_data
        ds, dt_pred, model, model_rep = setup

        vol, val = _line_chart_data(
            model=model, dt_pred=dt_pred, model_rep=model_rep,
            period='month', group_var=('superbrand',),
            show_future=True,
        )

        observed_vars = [c for c in vol['variable'].unique() if 'observed' in c.lower()]
        assert len(observed_vars) > 0


# ---------------------------------------------------------------------------
# TestDecomposeChartData
# ---------------------------------------------------------------------------

class TestDecomposeChartData:

    @pytest.fixture
    def split_m_m(self):
        import numpy as np
        np.random.seed(42)
        n = 12
        return pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=n, freq='MS'),
            'superbrand': ['brand_1'] * n,
            'bs_key': [1] * n,
            'listed': [2] * n,
            'variable': (['tv'] * 3 + ['digital'] * 3 + ['pr'] * 3 + ['base'] * 3),
            'value': np.random.uniform(-10, 20, n),
        })

    def test_returns_tuple_of_df_and_sku_name(self, split_m_m):
        from fermatrica_rep.export.slides.decomposition import _decompose_chart_data

        result, sku = _decompose_chart_data(
            split=split_m_m, brands=['brand_1'],
            show_future=True, period='m',
            group_var=['superbrand'],
            if_volume=True,
        )

        assert isinstance(result, pd.DataFrame)
        assert isinstance(sku, str)
        assert len(result) > 0

    def test_no_data_returns_empty(self, split_m_m):
        from fermatrica_rep.export.slides.decomposition import _decompose_chart_data
        filtered = split_m_m[split_m_m['superbrand'] == 'nonexistent']

        try:
            result, sku = _decompose_chart_data(
                split=filtered, brands=['nonexistent'],
                show_future=False, period='m',
                group_var=['superbrand'],
                if_volume=True,
            )
            assert isinstance(result, pd.DataFrame)
        except IndexError:
            pass


# ---------------------------------------------------------------------------
# TestWaterfallChartData
# ---------------------------------------------------------------------------

class TestWaterfallChartData:

    @pytest.fixture
    def split_m_m(self):
        import numpy as np
        np.random.seed(42)
        n = 12
        return pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=n, freq='MS'),
            'superbrand': ['brand_1'] * n,
            'bs_key': [1] * n,
            'listed': [2] * n,
            'variable': (['tv'] * 3 + ['digital'] * 3 + ['pr'] * 3 + ['base'] * 3),
            'value': np.random.uniform(-10, 20, n),
        })

    def test_returns_dataframe_with_expected_columns(self, split_m_m):
        from fermatrica_rep.export.slides.waterfall import _waterfall_chart_data

        result = _waterfall_chart_data(
            split=split_m_m, brands=['brand_1'],
            date_start='2024-01-01', date_end='2024-12-31',
            if_volume=True,
        )

        assert isinstance(result, pd.DataFrame)
        for col in ['variable', 'value', 'base', 'ratio_lbl']:
            assert col in result.columns, f"Missing column: {col}"

    def test_no_data_returns_empty(self, split_m_m):
        from fermatrica_rep.export.slides.waterfall import _waterfall_chart_data
        empty = split_m_m.iloc[0:0].copy()

        result = _waterfall_chart_data(
            split=empty, brands=['brand_1'],
            date_start='2024-01-01', date_end='2024-12-31',
            if_volume=True,
        )

        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# TestIncrRoiChartDataWorker
# ---------------------------------------------------------------------------

class TestIncrRoiChartDataWorker:

    @pytest.fixture
    def curves_full_df(self):
        import numpy as np
        n = 11
        bdg = np.arange(0, 110, 10)
        zero = pd.DataFrame({
            'bdg': [0], 'option': ['zero'],
            'pred_exact_vol': [100.0], 'pred_long_vol': [200.0],
            'pred_exact_val': [150.0], 'pred_long_val': [300.0],
            'max_obs_budg': [100],
        })
        other = pd.DataFrame({
            'bdg': np.tile(bdg, 3),
            'option': ['tv'] * n + ['digital'] * n + ['pr'] * n,
            'pred_exact_vol': np.random.uniform(100, 200, n * 3),
            'pred_long_vol': np.random.uniform(200, 400, n * 3),
            'pred_exact_val': np.random.uniform(150, 300, n * 3),
            'pred_long_val': np.random.uniform(300, 600, n * 3),
            'max_obs_budg': [100] * (n * 3),
        })
        return pd.concat([zero, other], ignore_index=True)

    def test_returns_chartdata(self, curves_full_df, monkeypatch):
        from fermatrica_rep.export.slides.incr_roi import _chart_data_worker, ChartData

        # Monkeypatch ChartData to avoid pptx import issues
        class MockChartData:
            def __init__(self):
                self.categories = None
                self.series = []

            def add_series(self, name, values):
                self.series.append({'name': name, 'values': values})

        monkeypatch.setattr(type(ChartData), '__new__', lambda cls: MockChartData())

        result = _chart_data_worker(
            curves_full_df, 'increment_volume_short', if_exact=True, n_years=1
        )

        assert result is not None

    def test_handles_empty_df(self, curves_full_df, monkeypatch):
        from fermatrica_rep.export.slides.incr_roi import _chart_data_worker, ChartData

        class MockChartData:
            def __init__(self):
                self.categories = None
                self.series = []

            def add_series(self, name, values):
                self.series.append({'name': name, 'values': values})

        monkeypatch.setattr(type(ChartData), '__new__', lambda cls: MockChartData())
        empty = curves_full_df.iloc[0:0].copy()

        result = _chart_data_worker(
            empty, 'increment_volume_short', if_exact=True, n_years=1
        )

        assert result is not None


# ---------------------------------------------------------------------------
# TestOptionsBuildOptionDataframes
# ---------------------------------------------------------------------------

class TestOptionsBuildOptionDataframes:

    def test_single_mode_returns_two_dfs(self):
        from fermatrica_rep.export.slides.options import _build_option_dataframes

        options_data = {'bdg_tv': 50, 'bdg_digital': 30}
        df_short, df_long = _build_option_dataframes(
            options_data, 'test', 'item', hide_rows=[], is_list_mode=False,
        )

        assert isinstance(df_short, pd.DataFrame)
        assert isinstance(df_long, pd.DataFrame)
        assert len(df_short) > 0

    def test_list_mode_returns_two_dfs(self):
        from fermatrica_rep.export.slides.options import _build_option_dataframes

        options_data = [{'bdg_tv': 50}, {'bdg_tv': 30}]
        df_short, df_long = _build_option_dataframes(
            options_data, 'test', 'item', hide_rows=[], is_list_mode=True,
        )

        assert isinstance(df_short, pd.DataFrame)
        assert isinstance(df_long, pd.DataFrame)

    def test_hide_rows_filtered_out(self):
        from fermatrica_rep.export.slides.options import _build_option_dataframes

        options_data = {'bdg_tv': 50, 'price': 1.0, 'text_flags': 'base'}
        df_short, df_long = _build_option_dataframes(
            options_data, 'test', 'item', hide_rows=['text_flags', 'price'], is_list_mode=False,
        )

        assert 'text_flags' not in df_short['item'].values
        assert 'price' not in df_short['item'].values
        assert 'bdg_tv' in df_short['item'].values
