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
            filters={'bs_key': []},
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
            filters={'bs_key': [1]},
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
            filters={'bs_key': []},
        )

        assert 'opt_base' in result['slide_title']

    def test_prepare_fit_data_empty_dt_pred(self, setup):
        from fermatrica_rep.export.slides.fit_predict import _prepare_fit_data
        ds, dt_pred, model, model_rep, opt_set = setup
        empty_dt_pred = dt_pred.iloc[0:0].copy()

        result = _prepare_fit_data(
            model=model, model_rep=model_rep, dt_pred=empty_dt_pred,
            opt_set=opt_set, option_name='test', plot_type='brand',
            filters={'bs_key': []},
        )

        assert isinstance(result, dict)

    def test_prepare_fit_data_default_sums(self, setup):
        from fermatrica_rep.export.slides.fit_predict import _prepare_fit_data
        ds, dt_pred, model, model_rep, opt_set = setup

        result = _prepare_fit_data(
            model=model, model_rep=model_rep, dt_pred=dt_pred,
            opt_set=opt_set, option_name='test_option', plot_type='brand',
            filters={'bs_key': [1]},
        )

        vol_table = result['dt_pred_prep_vol']
        pred_col = [c for c in vol_table.columns if 'predicted' in c.lower()][0]
        expected = '{:,.0f}'.format(dt_pred['predicted'].sum()).replace(',', ' ')
        assert vol_table[pred_col].iloc[0] == expected

    def test_prepare_fit_data_means_with_summary_type(self, setup):
        from fermatrica_rep.export.slides.fit_predict import _prepare_fit_data
        ds, dt_pred, model, model_rep, opt_set = setup
        model.conf.summary_type = 'mean'

        result = _prepare_fit_data(
            model=model, model_rep=model_rep, dt_pred=dt_pred,
            opt_set=opt_set, option_name='test_option', plot_type='brand',
            filters={'bs_key': [1]},
        )

        vol_table = result['dt_pred_prep_vol']
        pred_col = [c for c in vol_table.columns if 'predicted' in c.lower()][0]
        expected = f"{dt_pred['predicted'].mean():,.2f}".replace(',', ' ')
        assert vol_table[pred_col].iloc[0] == expected

    def test_prepare_fit_data_means_large_stays_integer(self, setup):
        from fermatrica_rep.export.slides.fit_predict import _prepare_fit_data
        ds, dt_pred, model, model_rep, opt_set = setup
        model.conf.summary_type = 'mean'
        dt_pred['predicted'] = [500.0 + x for x in range(len(dt_pred))]

        result = _prepare_fit_data(
            model=model, model_rep=model_rep, dt_pred=dt_pred,
            opt_set=opt_set, option_name='test_option', plot_type='brand',
            filters={'bs_key': [1]},
        )

        vol_table = result['dt_pred_prep_vol']
        pred_col = [c for c in vol_table.columns if 'predicted' in c.lower()][0]
        expected = '{:,.0f}'.format(dt_pred['predicted'].mean()).replace(',', ' ')
        assert vol_table[pred_col].iloc[0] == expected

    def test_prepare_fit_data_err_dash_when_observed_zero(self, setup):
        from fermatrica_rep.export.slides.fit_predict import _prepare_fit_data
        ds, dt_pred, model, model_rep, opt_set = setup
        dt_pred['observed'] = 0.0
        dt_pred['predicted'] = 0.0

        result = _prepare_fit_data(
            model=model, model_rep=model_rep, dt_pred=dt_pred,
            opt_set=opt_set, option_name='test_option', plot_type='brand',
            filters={'bs_key': [1]},
        )

        vol_table = result['dt_pred_prep_vol']
        err_col = [c for c in vol_table.columns if 'err' in c.lower()][0]
        assert vol_table[err_col].iloc[0] == '-'


# ---------------------------------------------------------------------------
# TestFitPredictPaneLayout — when `if_value_applicable` is False the value pane
# is omitted and the volume chart spans the slide; the single per-year volume
# table must stay clear of the always-centred option budget table that
# options_detail() adds, while retro slides (no option) keep it centred.
# ---------------------------------------------------------------------------

class TestFitPredictPaneLayout:

    @staticmethod
    def _model_rep(ds):
        mr = ModelRep(ds, language='english')
        mr.pptx_cnf = {
            'slide_height': 6858000,
            'slide_width': 12192000,
            'font_size_gross': 400000,
            'font_family_header': 'Arial',
            'font_size_main': 200000,
            'font_family_body': 'Arial',
            'font_size_small': 120000,
            'Blank_slide': 'Blank_slide',
            'mock_text': '',
        }
        return mr

    def _build_slide(self, if_value, with_option):
        from types import SimpleNamespace
        from pptx import Presentation
        from fermatrica_rep.export.slides import fit_predict
        from fermatrica_rep.options.scope import OptionScope
        from fermatrica_rep.options.option import Option

        ds = _minimal_ds()
        dt_pred = _minimal_dt_pred()
        dt_pred['price'] = [1.0] * len(dt_pred)
        model_rep = self._model_rep(ds)
        conf = SimpleNamespace(
            target_superbrand='brand_x', price_var='price', summary_type='sum',
            kpi_name='sales' if if_value else 'brand_awareness',
            currency='RUB', if_value_applicable=if_value,
            display_units='packs' if if_value else 'pct',
        )
        node = SimpleNamespace(model=SimpleNamespace(conf=conf), model_rep=model_rep,
                               dt_pred=dt_pred)
        scope = OptionScope(
            scope_name='brand_x_2024', date_start='2024-01-01', date_end='2024-12-31',
            ref_date_start='2023-01-01', ref_date_end='2023-12-31',
        )
        option = (Option('opt1', budget_data={'brand_x_2024': {'bdg_tv': 100.0, 'bdg_digital': 30.0}})
                  if with_option else None)

        prs = Presentation()
        fit_predict.create(prs, node, dt_pred=dt_pred, option_name='opt1',
                           opt_set=scope, option=option, period='month',
                           group_var=['superbrand'], plot_type='brand')
        return prs.slides[0]

    @staticmethod
    def _tables(slide):
        return [sh for sh in slide.shapes if sh.has_table]

    def test_no_value_pane_prediction_table_avoids_option_table(self):
        slide = self._build_slide(if_value=False, with_option=True)
        tables = self._tables(slide)
        vol = [sh for sh in tables if len(sh.table.columns) == 4]
        opt = [sh for sh in tables if len(sh.table.columns) == 2]
        assert len(vol) == 1 and len(opt) == 1

        vol, opt = vol[0], opt[0]
        no_overlap = (vol.left + vol.width <= opt.left
                      or opt.left + opt.width <= vol.left)
        assert no_overlap, (
            f"volume table [{vol.left}, {vol.left + vol.width}] overlaps "
            f"option table [{opt.left}, {opt.left + opt.width}]"
        )

    def test_no_value_retro_keeps_prediction_table_centred(self):
        slide = self._build_slide(if_value=False, with_option=False)
        vol = [sh for sh in self._tables(slide) if len(sh.table.columns) == 4]
        assert len(vol) == 1
        vol = vol[0]
        slide_width = self._model_rep(_minimal_ds()).pptx_cnf['slide_width']

        assert abs((vol.left + vol.width / 2) - slide_width / 2) <= 2


# ---------------------------------------------------------------------------
# TestFitPredictWordingTitles — wording-module routing of slide titles:
# KPI-specific rows win over the sales base and the ***cur token resolves
# to the localized currency label.
# ---------------------------------------------------------------------------

class TestFitPredictWordingTitles:

    @pytest.fixture
    def setup(self):
        from types import SimpleNamespace
        ds = _minimal_ds()
        dt_pred = _minimal_dt_pred()
        dt_pred['price'] = [1.0] * len(dt_pred)
        model_rep = ModelRep(ds, language='english')
        return ds, dt_pred, model_rep

    def _run_prepare(self, setup, conf, option_name='opt_1'):
        from types import SimpleNamespace
        from fermatrica_rep.export.slides.fit_predict import _prepare_fit_data
        ds, dt_pred, model_rep = setup
        model = SimpleNamespace(conf=conf)
        return _prepare_fit_data(
            model=model, model_rep=model_rep, dt_pred=dt_pred,
            opt_set=None, option_name=option_name, plot_type='brand',
        )

    def test_brand_awareness_title_replaces_sales(self, setup):
        from types import SimpleNamespace
        conf = SimpleNamespace(
            target_superbrand='brand_x',
            price_var='price',
            summary_type='sum',
            kpi_name='brand_awareness',
            currency='RUB',
            if_value_applicable=False,
            display_units='pct',
        )

        result = self._run_prepare(setup, conf, option_name='opt_ba')

        slide_title = result['slide_title']
        assert 'Brand Awareness' in slide_title
        assert 'Sales' not in slide_title

    def test_sales_title_ends_with_currency(self, setup):
        from types import SimpleNamespace
        conf = SimpleNamespace(
            target_superbrand='brand_x',
            price_var='price',
            summary_type='sum',
        )

        result = self._run_prepare(setup, conf)

        assert result['slide_title'].endswith('RUB')

    def test_brand_retro_title_russian_tokens_resolved(self, setup):
        from types import SimpleNamespace
        from fermatrica_rep.export.slides.fit_predict import _prepare_fit_data
        ds, dt_pred, model_rep = setup
        model_rep.language = 'russian'
        conf = SimpleNamespace(
            target_superbrand='brand_x',
            price_var='price',
            summary_type='sum',
            kpi_name='sales',
            currency='RUB',
            if_value_applicable=True,
            display_units='packs',
        )
        model = SimpleNamespace(conf=conf)

        result = _prepare_fit_data(
            model=model, model_rep=model_rep, dt_pred=dt_pred,
            opt_set=None, option_name='opt_retro', plot_type='brand_retro',
        )

        title = result['slide_title']
        # the retro branch routes through predict_table.title with no tokens left
        assert '***' not in title
        assert 'Продажи' in title
        assert 'руб.' in title


# ---------------------------------------------------------------------------
# TestNumKpiPrep — slide-title KPI formatter: sub-thousand values (e.g. an
# awareness level ~27.6) must render as a plain 2-decimal number instead of '0',
# while the thousand/million/billion branches stay byte-identical.
# ---------------------------------------------------------------------------

class TestNumKpiPrep:

    @pytest.fixture
    def model_rep(self):
        return ModelRep(_minimal_ds(), language='english')

    def _short_unit(self, model_rep, variable):
        df = model_rep.vis_dict
        return df.loc[(df['section'] == 'shared') & (df['variable'] == variable),
                      model_rep.language].iloc[0]

    @pytest.mark.parametrize('raw, expected', [
        ('0', '0'),
        ('27.60', '27.60'),
        ('999.99', '999.99'),
        ('0.40', '0.40'),
    ])
    def test_english_sub_thousand(self, model_rep, raw, expected):
        from fermatrica_rep.export.slides.fit_predict import _num_kpi_prep
        assert _num_kpi_prep(raw, model_rep) == expected

    @pytest.mark.parametrize('raw, expected', [
        ('27.60', '27,60'),
        ('999.99', '999,99'),
        ('0', '0'),
    ])
    def test_russian_sub_thousand(self, model_rep, raw, expected):
        from fermatrica_rep.export.slides.fit_predict import _num_kpi_prep
        model_rep.language = 'russian'
        assert _num_kpi_prep(raw, model_rep) == expected

    def test_rounds_to_zero_collapses(self, model_rep):
        from fermatrica_rep.export.slides.fit_predict import _num_kpi_prep
        # 0.004 rounds to 0.00 -> must not render '0.00'
        assert _num_kpi_prep('0.004', model_rep) == '0'

    def test_negative_formats_with_minus(self, model_rep):
        from fermatrica_rep.export.slides.fit_predict import _num_kpi_prep
        assert _num_kpi_prep('-27.60', model_rep) == '-27.60'

    def test_thousand_branch_unchanged(self, model_rep):
        from fermatrica_rep.export.slides.fit_predict import _num_kpi_prep
        expected = '2 ' + self._short_unit(model_rep, 'thousand_short')
        assert _num_kpi_prep('1500', model_rep) == expected

    def test_million_branch_unchanged(self, model_rep):
        from fermatrica_rep.export.slides.fit_predict import _num_kpi_prep
        expected = '1.5 ' + self._short_unit(model_rep, 'million_short')
        assert _num_kpi_prep('1500000', model_rep) == expected


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
# TestDecompositionWordingCaption — decomposition chart captions resolve via
# the wording module: the KPI-specific plot_efficiency override row wins over
# the sales base and the ***units token resolves to the localized units label.
# ---------------------------------------------------------------------------

class TestDecompositionWordingCaption:

    @pytest.fixture
    def setup(self):
        from types import SimpleNamespace
        ds = _minimal_ds()
        model_rep = ModelRep(ds, language='english')
        conf = SimpleNamespace(
            kpi_name='brand_awareness',
            currency='RUB',
            if_value_applicable=False,
            display_units='pct',
        )
        return model_rep, conf

    def test_brand_awareness_caption_uses_pct_units(self, setup):
        from fermatrica_rep.export import wording
        model_rep, conf = setup

        units_label = wording.units_label(model_rep, conf)
        caption = wording.text(model_rep, conf, section='plot_efficiency',
                               variable='factors_title')
        caption = caption.replace('***units', units_label)

        assert units_label == '%'
        assert 'Brand Awareness' in caption
        assert '%' in caption
        assert 'Sales' not in caption


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
# TestIncrRoiSlideWorkerBdgMaxNone — bdg_max=None must not crash the curves
# slides; the axis skip is derived from the computed grid's max budget.
# ---------------------------------------------------------------------------

class TestIncrRoiSlideWorkerBdgMaxNone:

    @pytest.fixture
    def model_rep(self):
        mr = ModelRep(_minimal_ds(), language='english')
        mr.pptx_cnf = {
            'slide_height': 6858000,
            'slide_width': 12192000,
            'font_size_gross': 400000,
            'font_family_header': 'Arial',
            'font_size_main': 200000,
            'font_family_body': 'Arial',
            'font_size_small': 120000,
            'mock_text': '',
        }
        return mr

    @pytest.fixture
    def curves_full_data(self):
        import numpy as np
        n = 10
        bdg = np.arange(10, 110, 10)
        zero = pd.DataFrame({
            'bdg': [0], 'option': ['zero'],
            'pred_exact_vol': [100.0], 'pred_long_vol': [200.0],
            'pred_exact_val': [150.0], 'pred_long_val': [300.0],
            'max_obs_budg': [100],
        })
        other = pd.DataFrame({
            'bdg': np.tile(bdg, 2),
            'option': ['tv'] * n + ['digital'] * n,
            'pred_exact_vol': np.random.uniform(100, 200, n * 2),
            'pred_long_vol': np.random.uniform(200, 400, n * 2),
            'pred_exact_val': np.random.uniform(150, 300, n * 2),
            'pred_long_val': np.random.uniform(300, 600, n * 2),
            'max_obs_budg': [100] * (n * 2),
        })
        return pd.concat([zero, other], ignore_index=True)

    def test_bdg_max_none_reaches_axis_as_number(self, model_rep, curves_full_data, monkeypatch):
        from pptx import Presentation
        from fermatrica_rep.export.slides import incr_roi
        from fermatrica_rep.options.scope import OptionScope

        scope = OptionScope(
            scope_name='scope_2024',
            date_start='2024-01-01',
            date_end='2024-12-31',
            ref_date_start='2023-01-01',
            ref_date_end='2023-12-31',
        )

        received = []

        def fake_axis(chart, model_rep, **kwargs):
            received.append(kwargs['bdg_max'])
            return chart

        monkeypatch.setattr(incr_roi, '_set_chart_axis', fake_axis)

        prs = Presentation()
        slide = prs.slides.add_slide(prs.slide_layouts[0])

        slide = incr_roi._slide_vis_worker(
            slide, model_rep=model_rep, scope=scope,
            curves_full_data=curves_full_data, plot_type='short',
            year_end=2024, budget_step=5, bdg_max=None,
        )

        assert len(received) == 3
        assert all(b is not None for b in received)
        assert all(b == float(curves_full_data['bdg'].max()) for b in received)

    def test_bdg_max_number_untouched(self, model_rep, curves_full_data, monkeypatch):
        from pptx import Presentation
        from fermatrica_rep.export.slides import incr_roi
        from fermatrica_rep.options.scope import OptionScope

        scope = OptionScope(
            scope_name='scope_2024',
            date_start='2024-01-01',
            date_end='2024-12-31',
            ref_date_start='2023-01-01',
            ref_date_end='2023-12-31',
        )

        received = []

        def fake_axis(chart, model_rep, **kwargs):
            received.append(kwargs['bdg_max'])
            return chart

        monkeypatch.setattr(incr_roi, '_set_chart_axis', fake_axis)

        prs = Presentation()
        slide = prs.slides.add_slide(prs.slide_layouts[0])

        slide = incr_roi._slide_vis_worker(
            slide, model_rep=model_rep, scope=scope,
            curves_full_data=curves_full_data, plot_type='short',
            year_end=2024, budget_step=5, bdg_max=50,
        )

        assert received == [50, 50, 50]


# ---------------------------------------------------------------------------
# TestOptionsTableZeroBaseline — per-scope zero binding (p09 per-option scopes)
# ---------------------------------------------------------------------------

def _opt_scope(name, year):
    from fermatrica_rep.options.scope import OptionScope
    return OptionScope(
        scope_name=name,
        date_start=f'{year}-01-01',
        date_end=f'{year}-12-31',
        ref_date_start=f'{year - 1}-01-01',
        ref_date_end=f'{year - 1}-12-31',
        filters={'superbrand': ['brand_1']},
    )


def _opt_summary(vol, val, growth=1.2, ref_vol=1000.0):
    return {
        'pred_exact_vol': vol,
        'pred_exact_val': val,
        'pred_long_vol': vol * 2.0,
        'pred_long_val': val * 2.0,
        'growth_vol': growth,
        'ref_vol': ref_vol,
    }


class TestOptionsTableZeroBaseline:

    @pytest.fixture
    def fakes(self):
        from types import SimpleNamespace
        model_rep = ModelRep(_minimal_ds(), language='english')
        node = SimpleNamespace(
            model_rep=model_rep,
            model=SimpleNamespace(conf=SimpleNamespace(summary_type='sum')),
        )
        meta_model = SimpleNamespace(nodes={'sales': node})
        return model_rep, meta_model

    def _metric_cell(self, model_rep, calc_df, metric_var, variant):
        """Look up one cell of the calc table by metric variable + option column."""
        from fermatrica_rep.export import wording
        metric = wording.text(model_rep, None, section='option_compare_sales', variable=metric_var)
        sales_col = wording.text(model_rep, None, section='option_compare', variable='sales_title')
        return calc_df.loc[calc_df[sales_col] == metric, variant].iloc[0]

    def _growth_cell(self, model_rep, calc_df, pred_year_start, variant):
        from fermatrica_rep.export import wording
        metric = wording.text(model_rep, None, section='option_compare_sales',
                              variable='grw').replace("***", str(pred_year_start - 1))
        sales_col = wording.text(model_rep, None, section='option_compare', variable='sales_title')
        return calc_df.loc[calc_df[sales_col] == metric, variant].iloc[0]

    # ------------------------- _get_nonzero_scopes ------------------------- #

    def test_nonzero_scopes_disjoint_per_option(self, fakes):
        from fermatrica_rep.export.slides.options import _get_nonzero_scopes
        scopes = [_opt_scope('predefined_brand_1_2025', 2025),
                  _opt_scope('zero_brand_1_2025', 2025),
                  _opt_scope('predefined_brand_1_2026', 2026)]
        zero_res = {'sales': {'zero_brand_1_2025': _opt_summary(10.0, 100.0)}}
        predef_res = {'sales': {'predefined_brand_1_2025': _opt_summary(500.0, 5000.0),
                                'predefined_brand_1_2026': _opt_summary(600.0, 6000.0)}}
        name_to_idx = {'zero': 0, 'predefined': 1}
        relevant = _get_nonzero_scopes([zero_res, predef_res], scopes, 'sales', name_to_idx)
        # all six p09-like scopes are reported: the zero-only scope carries
        # zero_available=True, the predefined-only scopes zero_available=False
        assert [(s, za) for _, s, za in relevant] == [
            ('predefined_brand_1_2025', False),
            ('zero_brand_1_2025', True),
            ('predefined_brand_1_2026', False),
        ]

    def test_nonzero_scopes_shared_zero_bound(self, fakes):
        from fermatrica_rep.export.slides.options import _get_nonzero_scopes
        scopes = [_opt_scope('brand_1_2025', 2025), _opt_scope('brand_1_2026', 2026)]
        zero_res = {'sales': {'brand_1_2025': _opt_summary(10.0, 100.0),
                              'brand_1_2026': _opt_summary(11.0, 110.0)}}
        opt_res = {'sales': {'brand_1_2025': _opt_summary(500.0, 5000.0),
                             'brand_1_2026': _opt_summary(600.0, 6000.0)}}
        relevant = _get_nonzero_scopes([zero_res, opt_res], scopes, 'sales', {'zero': 0, 'opt': 1})
        assert [(s, za) for _, s, za in relevant] == [
            ('brand_1_2025', True),
            ('brand_1_2026', True),
        ]

    def test_nonzero_scopes_no_zero_entry(self, fakes):
        from fermatrica_rep.export.slides.options import _get_nonzero_scopes
        scopes = [_opt_scope('predefined_brand_1_2025', 2025),
                  _opt_scope('zero_brand_1_2025', 2025)]
        predef_res = {'sales': {'predefined_brand_1_2025': _opt_summary(500.0, 5000.0)}}
        relevant = _get_nonzero_scopes([predef_res], scopes, 'sales', {'predefined': 0})
        assert [(s, za) for _, s, za in relevant] == [('predefined_brand_1_2025', False)]

    def test_nonzero_scopes_zero_result_none(self, fakes):
        from fermatrica_rep.export.slides.options import _get_nonzero_scopes
        scopes = [_opt_scope('brand_1_2026', 2026)]
        opt_res = {'sales': {'brand_1_2026': _opt_summary(500.0, 5000.0)}}
        relevant = _get_nonzero_scopes([None, opt_res], scopes, 'sales', {'zero': 0, 'opt': 1})
        assert [(s, za) for _, s, za in relevant] == [('brand_1_2026', False)]

    # ------------------------- _tables_data_worker ------------------------- #

    def test_worker_zero_missing_for_scope(self, fakes):
        from fermatrica_rep.options.option import Option
        from fermatrica_rep.export.slides.options import _tables_data_worker
        model_rep, meta_model = fakes
        scopes = [_opt_scope('predefined_brand_1_2025', 2025), _opt_scope('zero_brand_1_2025', 2025)]
        name_to_idx = {'zero': 0, 'predefined': 1}
        results = [
            {'sales': {'zero_brand_1_2025': _opt_summary(10.0, 100.0)}},
            {'sales': {'predefined_brand_1_2025': _opt_summary(500.0, 5000.0)}},
        ]
        options = {
            'zero': Option('zero', budget_data={'zero_brand_1_2025': {'bdg_tv': 0.0}}),
            'predefined': Option('predefined',
                                 budget_data={'predefined_brand_1_2025': {'bdg_tv': 200.0, 'bdg_ooh': 80.0}}),
        }
        opt_df, calc_df = _tables_data_worker(
            meta_model=meta_model, scopes=scopes, options=options,
            options_names_subset=['predefined'], pred_year_start=2025, pred_year_end=2026,
            target_scope_name='predefined_brand_1_2025', if_exact=True,
            hide_rows=['text_flags', 'google', 'price'], summary_node='sales', cores=1,
            precomputed_results=results, name_to_idx=name_to_idx, zero_available=False,
        )
        assert not opt_df.empty and not calc_df.empty
        assert (opt_df['zero_short'] == 0).all()
        assert sorted(opt_df['predefined_short'].round(0).tolist()) == [80.0, 200.0, 280.0]
        assert self._metric_cell(model_rep, calc_df, 'res', 'predefined_short') == '500.00'
        assert self._metric_cell(model_rep, calc_df, 'res_rub_progress_distr', 'predefined_short') == '-'
        assert self._metric_cell(model_rep, calc_df, 'profit', 'predefined_short') == '-'
        assert self._metric_cell(model_rep, calc_df, 'romi', 'predefined_short') == '-'
        assert self._growth_cell(model_rep, calc_df, 2025, 'predefined_short') == '20.00%'
        assert self._growth_cell(model_rep, calc_df, 2025, 'predefined_long') == '-'
        assert self._metric_cell(model_rep, calc_df, 'res', 'zero_short') == '-'

    def test_worker_zero_bound_classic(self, fakes):
        from fermatrica_rep.options.option import Option
        from fermatrica_rep.export.slides.options import _tables_data_worker
        model_rep, meta_model = fakes
        scopes = [_opt_scope('brand_1_2026', 2026)]
        name_to_idx = {'zero': 0, 'predefined': 1}
        results = [
            {'sales': {'brand_1_2026': _opt_summary(10.0, 100.0)}},
            {'sales': {'brand_1_2026': _opt_summary(500.0, 5000.0)}},
        ]
        options = {
            'zero': Option('zero', budget_data={'brand_1_2026': {'bdg_tv': 100.0, 'bdg_ooh': 50.0}}),
            'predefined': Option('predefined', budget_data={'brand_1_2026': {'bdg_tv': 200.0, 'bdg_ooh': 80.0}}),
        }
        opt_df, calc_df = _tables_data_worker(
            meta_model=meta_model, scopes=scopes, options=options,
            options_names_subset=['predefined'], pred_year_start=2026, pred_year_end=2026,
            target_scope_name='brand_1_2026', if_exact=True,
            hide_rows=['text_flags', 'google', 'price'], summary_node='sales', cores=1,
            precomputed_results=results, name_to_idx=name_to_idx, zero_available=True,
        )
        total_row = opt_df.loc[opt_df['Costs, Mln. RUB'] == 'Total'].iloc[0]
        assert total_row['zero_short'] == 150.0 and total_row['predefined_short'] == 280.0
        assert self._metric_cell(model_rep, calc_df, 'res_rub_progress_distr', 'predefined_short') != '-'
        assert self._growth_cell(model_rep, calc_df, 2026, 'zero_long') != '-'

    def test_worker_option_result_missing(self, fakes):
        from fermatrica_rep.options.option import Option
        from fermatrica_rep.export.slides.options import _tables_data_worker
        model_rep, meta_model = fakes
        scopes = [_opt_scope('brand_1_2026', 2026)]
        name_to_idx = {'zero': 0, 'predefined': 1}
        # option has no budget and no result for the scope
        results = [{'sales': {'brand_1_2026': _opt_summary(10.0, 100.0)}}, None]
        options = {
            'zero': Option('zero', budget_data={'brand_1_2026': {'bdg_tv': 100.0}}),
            'predefined': Option('predefined', budget_data={}),
        }
        opt_df, calc_df = _tables_data_worker(
            meta_model=meta_model, scopes=scopes, options=options,
            options_names_subset=['predefined'], pred_year_start=2026, pred_year_end=2026,
            target_scope_name='brand_1_2026', if_exact=True,
            hide_rows=['text_flags', 'google', 'price'], summary_node='sales', cores=1,
            precomputed_results=results, name_to_idx=name_to_idx, zero_available=True,
        )
        assert not opt_df.empty and not calc_df.empty
        assert (opt_df['predefined_short'].fillna(0) == 0).all()
        assert self._metric_cell(model_rep, calc_df, 'res', 'predefined_short') == '-'
        assert self._metric_cell(model_rep, calc_df, 'res_rub_progress_distr', 'predefined_short') == '-'

    def test_worker_total_row_counts_budget_items_only(self, fakes):
        from fermatrica_rep.options.option import Option
        from fermatrica_rep.export.slides.options import _tables_data_worker
        model_rep, meta_model = fakes
        scopes = [_opt_scope('brand_1_2026', 2026)]
        name_to_idx = {'zero': 0, 'predefined': 1}
        results = [
            {'sales': {'brand_1_2026': _opt_summary(10.0, 100.0)}},
            {'sales': {'brand_1_2026': _opt_summary(500.0, 5000.0)}},
        ]
        # the option carries a non-budget item (nobdg_price) that is a visible
        # row; the Total row must count budget (bdg_) items only — summing
        # costs and prices together would be meaningless
        options = {
            'zero': Option('zero', budget_data={'brand_1_2026': {'bdg_tv': 100.0, 'nobdg_price': 230.0}}),
            'predefined': Option('predefined', budget_data={'brand_1_2026': {'bdg_tv': 200.0, 'nobdg_price': 240.0}}),
        }
        opt_df, calc_df = _tables_data_worker(
            meta_model=meta_model, scopes=scopes, options=options,
            options_names_subset=['predefined'], pred_year_start=2026, pred_year_end=2026,
            target_scope_name='brand_1_2026', if_exact=True,
            hide_rows=['text_flags', 'google', 'price'], summary_node='sales', cores=1,
            precomputed_results=results, name_to_idx=name_to_idx, zero_available=True,
        )
        total_row = opt_df.loc[opt_df['Costs, Mln. RUB'] == 'Total'].iloc[0]
        # Total = budget spend only: nobdg_price (230/240) is excluded
        assert total_row['zero_short'] == 100.0
        assert total_row['predefined_short'] == 200.0
        assert total_row['predefined_long'] == 200.0

    # ------------------- wording routing & money-row gating ------------------- #

    def _run_worker(self, model_rep, meta_model, model_conf=None):
        from fermatrica_rep.options.option import Option
        from fermatrica_rep.export.slides.options import _tables_data_worker
        scopes = [_opt_scope('brand_1_2026', 2026)]
        name_to_idx = {'zero': 0, 'predefined': 1}
        results = [
            {'sales': {'brand_1_2026': _opt_summary(10.0, 100.0)}},
            {'sales': {'brand_1_2026': _opt_summary(500.0, 5000.0)}},
        ]
        options = {
            'zero': Option('zero', budget_data={'brand_1_2026': {'bdg_tv': 100.0}}),
            'predefined': Option('predefined', budget_data={'brand_1_2026': {'bdg_tv': 200.0}}),
        }
        return _tables_data_worker(
            meta_model=meta_model, scopes=scopes, options=options,
            options_names_subset=['predefined'], pred_year_start=2026, pred_year_end=2026,
            target_scope_name='brand_1_2026', if_exact=True,
            hide_rows=['text_flags', 'google', 'price'], summary_node='sales', cores=1,
            precomputed_results=results, name_to_idx=name_to_idx, zero_available=True,
            model_conf=model_conf,
        )

    def test_worker_default_sales_all_rows_currency_resolved(self, fakes):
        model_rep, meta_model = fakes

        opt_df, calc_df = self._run_worker(model_rep, meta_model)

        # the ***cur token in the dictionary resolves to RUB for the default
        # sales conf (costs header ends RUB, not a literal token)
        assert opt_df.columns[0] == 'Costs, Mln. RUB'
        assert calc_df.columns[0] == 'Sales'
        assert list(calc_df['Sales']) == [
            'Volume Sales, Units', 'Value Sales, RUB', 'Incremental Sales, RUB',
            'Profit, RUB', 'ROMI', 'Volume growth to 2025',
        ]
        assert not calc_df.astype(str).apply(lambda col: col.str.contains(r'\*\*\*cur')).any().any()

    def test_worker_value_not_applicable_drops_value_rows(self, fakes):
        from types import SimpleNamespace
        model_rep, meta_model = fakes
        conf = SimpleNamespace(kpi_name='brand_awareness', currency='RUB',
                               if_value_applicable=False, display_units='pct')

        opt_df, calc_df = self._run_worker(model_rep, meta_model, model_conf=conf)

        # value/profit/ROMI rows are absent; the remaining labels come from the
        # brand_awareness overrides (volume + growth only)
        assert calc_df.columns[0] == 'Brand Awareness'
        assert list(calc_df['Brand Awareness']) == ['Brand Awareness, %', 'Brand Awareness growth to 2025']
        assert len(calc_df) == 2


# ---------------------------------------------------------------------------
# TestIncrRoiWordingAxes — curves (incr_roi) axis titles resolve via the wording
# module (currency token -> RUB, KPI override -> awareness %), and money-less
# confs render only the incremental chart (no profit/ROI charts, no dispUnits).
# ---------------------------------------------------------------------------

def _incr_roi_model_rep():
    mr = ModelRep(_minimal_ds(), language='english')
    mr.pptx_cnf = {
        'slide_height': 6858000,
        'slide_width': 12192000,
        'font_size_gross': 400000,
        'font_family_header': 'Arial',
        'font_size_main': 200000,
        'font_family_body': 'Arial',
        'font_size_small': 120000,
        'mock_text': '',
    }
    return mr


def _incr_roi_curves_data():
    import numpy as np
    np.random.seed(7)
    n = 10
    bdg = np.arange(10, 110, 10)
    zero = pd.DataFrame({
        'bdg': [0], 'option': ['zero'],
        'pred_exact_vol': [100.0], 'pred_long_vol': [200.0],
        'pred_exact_val': [150.0], 'pred_long_val': [300.0],
        'max_obs_budg': [100],
    })
    other = pd.DataFrame({
        'bdg': np.tile(bdg, 2),
        'option': ['tv'] * n + ['digital'] * n,
        'pred_exact_vol': np.random.uniform(100, 200, n * 2),
        'pred_long_vol': np.random.uniform(200, 400, n * 2),
        'pred_exact_val': np.random.uniform(150, 300, n * 2),
        'pred_long_val': np.random.uniform(300, 600, n * 2),
        'max_obs_budg': [100] * (n * 2),
    })
    return pd.concat([zero, other], ignore_index=True)


class TestIncrRoiWordingAxes:

    def test_sales_incr_axis_ends_with_currency(self, basic_model_rep):
        from types import SimpleNamespace
        from fermatrica_rep.export import wording
        conf = SimpleNamespace(kpi_name='sales', currency='RUB',
                               if_value_applicable=True, display_units='packs')
        title = wording.text(basic_model_rep, conf, section='efficiency_short',
                             variable='incr_sales_axis_y')
        title = title.replace('***', '2024')
        assert title.endswith('RUB')
        assert '***cur' not in title

    def test_sales_cost_axis_ends_with_currency(self, basic_model_rep):
        from types import SimpleNamespace
        from fermatrica_rep.export import wording
        conf = SimpleNamespace(kpi_name='sales', currency='RUB',
                               if_value_applicable=True, display_units='packs')
        title = wording.text(basic_model_rep, conf, section='efficiency_short',
                             variable='axis_x')
        title = title.replace('***', '2024')
        assert title.endswith('RUB')
        assert '***cur' not in title

    def test_brand_awareness_incr_axis_uses_pct(self, basic_model_rep):
        from types import SimpleNamespace
        from fermatrica_rep.export import wording
        conf = SimpleNamespace(kpi_name='brand_awareness', currency='RUB',
                               if_value_applicable=False, display_units='pct')
        title = wording.text(basic_model_rep, conf, section='efficiency_short',
                             variable='incr_sales_axis_y')
        assert '%' in title
        assert '***cur' not in title

    def _run_worker(self, model_conf, plot_type='short', if_exact=True):
        from pptx import Presentation
        from fermatrica_rep.export.slides import incr_roi
        from fermatrica_rep.options.scope import OptionScope

        scope = OptionScope(
            scope_name='scope_2024',
            date_start='2024-01-01',
            date_end='2024-12-31',
            ref_date_start='2023-01-01',
            ref_date_end='2023-12-31',
        )
        prs = Presentation()
        slide = prs.slides.add_slide(prs.slide_layouts[0])
        slide = incr_roi._slide_vis_worker(
            slide, model_rep=_incr_roi_model_rep(), model_conf=model_conf,
            scope=scope, curves_full_data=_incr_roi_curves_data(),
            plot_type=plot_type, year_end=2024, budget_step=5, bdg_max=None,
            if_exact=if_exact,
        )
        return [sh for sh in slide.shapes if sh.has_chart]

    def test_worker_money_less_renders_single_incremental_chart(self):
        from types import SimpleNamespace
        conf = SimpleNamespace(kpi_name='brand_awareness', currency='RUB',
                               if_value_applicable=False, display_units='pct')

        charts = self._run_worker(conf)

        assert len(charts) == 1
        axis_title = charts[0].chart.value_axis.axis_title.text_frame.text
        assert '%' in axis_title
        assert '***cur' not in axis_title
        assert charts[0].chart.has_legend is True

    def test_worker_value_applicable_renders_three_charts(self):
        from types import SimpleNamespace
        conf = SimpleNamespace(kpi_name='sales', currency='RUB',
                               if_value_applicable=True, display_units='packs')

        charts = self._run_worker(conf)

        assert len(charts) == 3
        axis_title = charts[0].chart.value_axis.axis_title.text_frame.text
        assert 'RUB' in axis_title
        assert '***cur' not in axis_title
        assert charts[0].chart.has_legend is False
        assert charts[2].chart.has_legend is True

    def test_long_axis_labelled_sum_for_sum_aggregation(self):
        from types import SimpleNamespace
        conf = SimpleNamespace(kpi_name='brand_awareness', currency='RUB',
                               if_value_applicable=False, display_units='pct',
                               summary_type='sum')

        charts = self._run_worker(conf, plot_type='long', if_exact=True)

        axis_title = charts[0].chart.value_axis.axis_title.text_frame.text
        assert 'Sum.' in axis_title

    def test_long_axis_labelled_avg_for_mean_aggregation(self):
        from types import SimpleNamespace
        conf = SimpleNamespace(kpi_name='brand_awareness', currency='RUB',
                               if_value_applicable=False, display_units='pct',
                               summary_type='mean')

        charts = self._run_worker(conf, plot_type='long', if_exact=True)

        axis_title = charts[0].chart.value_axis.axis_title.text_frame.text
        assert 'Avg.' in axis_title
        assert 'Sum.' not in axis_title

    def test_long_axis_labelled_avg_for_mean_fin_aggregation(self):
        from types import SimpleNamespace
        conf = SimpleNamespace(kpi_name='brand_awareness', currency='RUB',
                               if_value_applicable=False, display_units='pct',
                               summary_type='mean_fin')

        charts = self._run_worker(conf, plot_type='long', if_exact=True)

        axis_title = charts[0].chart.value_axis.axis_title.text_frame.text
        assert 'Avg.' in axis_title
        assert 'Sum.' not in axis_title

