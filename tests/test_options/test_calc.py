import copy

import numpy as np
import pandas as pd
import pytest

from fermatrica_rep.options.calc import option_report, option_summarize, option_summarize_exact
from fermatrica_rep.options.define import OptionSettings
from fermatrica_rep.meta_model.model_rep import ModelRep
from fermatrica.basics.basics import FermatricaError


class MockModelConf:
    pass


class TestOptionReportArgumentValidation:

    def test_error_mixed_list_and_nonlist_settings(self):
        with pytest.raises(FermatricaError) as exc_info:
            option_report(
                model=None,
                ds=None,
                model_rep=None,
                option_dict={'bdg_tv': 100},
                option_settings=['not', 'a', 'list'],
            )
        assert 'Option could not be calculated' in str(exc_info.value)

    def test_error_list_option_dict_without_exact(self):
        with pytest.raises(FermatricaError) as exc_info:
            option_report(
                model=None,
                ds=None,
                model_rep=None,
                option_dict=[{'bdg_tv': 100}],
                option_settings=['opt_settings'],
                if_exact=False,
            )
        assert "if_exact=True" in str(exc_info.value)


class TestOptionSummarizeExact:

    @pytest.fixture
    def model_conf(self):
        mc = MockModelConf()
        mc.price_var = ''
        mc.summary_type = 'sum'
        return mc

    @pytest.fixture
    def dt_pred(self):
        return pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=12, freq='MS'),
            'superbrand': ['brand_x'] * 12,
            'listed': [2] * 12,
            'predicted': np.arange(100, 112, dtype=float),
            'observed': np.arange(90, 102, dtype=float),
        })

    @pytest.fixture
    def opt_settings(self):
        return OptionSettings(
            target='brand_x', date_start='2024-01-01', date_end='2024-12-01',
            ref_date_start='2023-01-01', ref_date_end='2023-12-01'
        )

    def test_pred_exact_vol_matches_sum(self, model_conf, dt_pred, opt_settings):
        from fermatrica_rep.options.calc import _align_option_settings
        opt_settings = _align_option_settings(opt_settings)
        result = option_summarize_exact(model_conf, opt_settings, dt_pred)
        expected_sum = dt_pred.loc[
            (dt_pred['listed'].isin([2, 3, 4])) &
            (dt_pred['date'] >= opt_settings.date_start) &
            (dt_pred['date'] <= opt_settings.date_end) &
            (dt_pred['superbrand'].isin(opt_settings.target[0])),
            'predicted'
        ].sum()
        assert np.isclose(result['pred_exact_vol'], expected_sum)

    def test_pred_exact_val_with_price(self, dt_pred, opt_settings):
        from fermatrica_rep.options.calc import _align_option_settings
        opt_settings = _align_option_settings(opt_settings)
        mc = MockModelConf()
        mc.price_var = 'price'
        mc.summary_type = 'sum'
        dt_pred['price'] = 1.5
        result = option_summarize_exact(mc, opt_settings, dt_pred)
        mask = (
            (dt_pred['listed'].isin([2, 3, 4])) &
            (dt_pred['date'] >= opt_settings.date_start) &
            (dt_pred['date'] <= opt_settings.date_end) &
            (dt_pred['superbrand'].isin(opt_settings.target[0]))
        )
        expected_val = (dt_pred.loc[mask, 'predicted'] * dt_pred.loc[mask, 'price']).sum()
        assert np.isclose(result['pred_exact_val'], expected_val)

    def test_fin_summary_type(self, model_conf, dt_pred, opt_settings):
        from fermatrica_rep.options.calc import _align_option_settings
        opt_settings = _align_option_settings(opt_settings)
        model_conf.summary_type = 'fin'
        result = option_summarize_exact(model_conf, opt_settings, dt_pred)
        date_end = dt_pred.loc[dt_pred['listed'].isin([2, 3, 4]), 'date'].unique()
        dlt = np.absolute(date_end - opt_settings.date_end)
        closest_date = date_end[np.argmin(dlt)]
        mask = (
            (dt_pred['listed'].isin([2, 3, 4])) &
            (dt_pred['date'] == closest_date) &
            (dt_pred['superbrand'].isin(opt_settings.target[0]))
        )
        expected = dt_pred.loc[mask, 'predicted'].sum()
        assert np.isclose(result['pred_exact_vol'], expected)

    def test_empty_dt_pred_returns_zero(self, model_conf, opt_settings):
        from fermatrica_rep.options.calc import _align_option_settings
        opt_settings = _align_option_settings(opt_settings)
        dt_pred = pd.DataFrame(columns=['date', 'superbrand', 'listed', 'predicted', 'observed'])
        result = option_summarize_exact(model_conf, opt_settings, dt_pred)
        assert np.isclose(result['pred_exact_vol'], 0.0)


class TestOptionSummarize:

    @pytest.fixture
    def model_conf(self):
        mc = MockModelConf()
        mc.price_var = ''
        mc.summary_type = 'sum'
        return mc

    @pytest.fixture
    def model_rep(self):
        ds_dummy = pd.DataFrame({'date': pd.date_range('2024-01-01', periods=12, freq='MS'),
                                  'superbrand': ['brand_x'] * 12})
        mr = ModelRep(ds_dummy, language='english')
        mr.option_summary_adhoc = None
        return mr

    @pytest.fixture
    def dt_pred(self):
        return pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=24, freq='MS'),
            'superbrand': ['brand_x'] * 24,
            'listed': [2] * 12 + [4] * 12,
            'predicted': np.arange(100, 124, dtype=float),
            'observed': np.arange(90, 114, dtype=float),
        })

    @pytest.fixture
    def opt_settings(self):
        return OptionSettings(
            target='brand_x', date_start='2024-01-01', date_end='2024-12-01',
            ref_date_start='2023-01-01', ref_date_end='2023-06-01'
        )

    def test_pred_long_vol_longer_than_exact(self, model_conf, model_rep, dt_pred, opt_settings):
        from fermatrica_rep.options.calc import _align_option_settings
        opt_settings = _align_option_settings(opt_settings)
        result = option_summarize(model_conf, model_rep, {'bdg_x': 100}, opt_settings, dt_pred)
        assert result['pred_long_vol'] >= result['pred_exact_0']['pred_exact_vol']

    def test_growth_vol_ratio(self, model_conf, model_rep, dt_pred, opt_settings):
        from fermatrica_rep.options.calc import _align_option_settings
        opt_settings = _align_option_settings(opt_settings)
        result = option_summarize(model_conf, model_rep, {'bdg_x': 100}, opt_settings, dt_pred)
        expected_growth = result['pred_exact_0']['pred_exact_vol'] / result['ref_vol']
        assert np.isclose(result['growth_vol'], expected_growth)

    def test_ref_vol_uses_observed_and_predicted(self, model_conf, model_rep, dt_pred, opt_settings):
        from fermatrica_rep.options.calc import _align_option_settings
        opt_settings = _align_option_settings(opt_settings)
        result = option_summarize(model_conf, model_rep, {'bdg_x': 100}, opt_settings, dt_pred)
        date_obs_max = dt_pred.loc[dt_pred['listed'].isin([2, 3]), 'date'].max()
        mask_ref = (
            (dt_pred['listed'].isin([2, 3, 4])) &
            (dt_pred['date'] >= opt_settings.ref_date_start) &
            (dt_pred['date'] <= opt_settings.ref_date_end) &
            (dt_pred['superbrand'].isin(opt_settings.target[0]))
        )
        mask_obs = mask_ref & (dt_pred['date'] <= date_obs_max)
        mask_pred = mask_ref & (dt_pred['date'] > date_obs_max)
        expected = dt_pred.loc[mask_obs, 'observed'].sum() + dt_pred.loc[mask_pred, 'predicted'].sum()
        assert np.isclose(result['ref_vol'], expected)


class TestOptionReport:

    @pytest.fixture
    def synthetic_setup(self):
        n = 12
        ds = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=n, freq='MS'),
            'superbrand': ['brand_x'] * n,
            'bs_key': [1] * n,
            'listed': [4] * n,
            'x1': [0.0] * n,
            'x2': [0.0] * n,
        })
        mc = MockModelConf()
        mc.price_var = 'price'
        mc.summary_type = 'sum'

        class MockModel:
            conf = mc

        model = MockModel()
        trans_dict = {
            'function': {'bdg_x1': 'basic_fix', 'bdg_x2': 'basic_fix'},
            'variable': {'bdg_x1': 'x1', 'bdg_x2': 'x2'},
        }
        model_rep = ModelRep(ds, trans_dict=trans_dict, language='english')
        opt_set = OptionSettings(
            target='brand_x', date_start='2024-01-01', date_end='2024-12-01',
            ref_date_start='2023-01-01', ref_date_end='2023-06-01'
        )
        return ds, model, model_rep, opt_set

    def test_zero_budget_via_basic_fix(self, synthetic_setup, monkeypatch):
        ds, model, model_rep, opt_set = synthetic_setup

        def mock_predict_ext(model, ds):
            return pd.DataFrame({
                'date': ds['date'],
                'superbrand': ds['superbrand'],
                'listed': ds['listed'],
                'predicted': [10.0] * len(ds),
                'observed': [10.0] * len(ds),
                'price': [1.0] * len(ds),
            })

        monkeypatch.setattr('fermatrica_rep.options.calc.transform', lambda ds, model, **kw: (model, ds))
        monkeypatch.setattr('fermatrica_rep.options.calc.predict_ext', mock_predict_ext)

        ds_out, dt_pred, opt_sum = option_report(
            model=model, ds=ds, model_rep=model_rep,
            option_dict={'bdg_x1': 0, 'bdg_x2': 0},
            option_settings=opt_set, if_exact=True
        )

        assert np.isclose(opt_sum['pred_exact_0']['pred_exact_vol'], 10.0 * 12)

    def test_nonzero_budget_increases_kpi(self, synthetic_setup, monkeypatch):
        ds, model, model_rep, opt_set = synthetic_setup

        def mock_predict_ext(model, ds):
            predicted = ds['x1'] * 2.0 + ds['x2'] * 1.0 + 10.0
            return pd.DataFrame({
                'date': ds['date'],
                'superbrand': ds['superbrand'],
                'listed': ds['listed'],
                'predicted': predicted,
                'observed': predicted,
                'price': [1.0] * len(ds),
            })

        monkeypatch.setattr('fermatrica_rep.options.calc.transform', lambda ds, model, **kw: (model, ds))
        monkeypatch.setattr('fermatrica_rep.options.calc.predict_ext', mock_predict_ext)

        _, _, opt_sum_zero = option_report(
            model=model, ds=ds, model_rep=model_rep,
            option_dict={'bdg_x1': 0, 'bdg_x2': 0},
            option_settings=opt_set, if_exact=True
        )
        _, _, opt_sum_nonzero = option_report(
            model=model, ds=ds, model_rep=model_rep,
            option_dict={'bdg_x1': 10, 'bdg_x2': 20},
            option_settings=opt_set, if_exact=True
        )

        assert opt_sum_nonzero['pred_exact_0']['pred_exact_vol'] > opt_sum_zero['pred_exact_0']['pred_exact_vol']

    def test_if_exact_structure(self, synthetic_setup, monkeypatch):
        ds, model, model_rep, opt_set = synthetic_setup

        def mock_predict_ext(model, ds):
            return pd.DataFrame({
                'date': ds['date'], 'superbrand': ds['superbrand'],
                'listed': ds['listed'], 'predicted': [10.0] * len(ds),
                'observed': [10.0] * len(ds),
                'price': [1.0] * len(ds),
            })

        monkeypatch.setattr('fermatrica_rep.options.calc.transform', lambda ds, model, **kw: (model, ds))
        monkeypatch.setattr('fermatrica_rep.options.calc.predict_ext', mock_predict_ext)

        _, _, opt_sum = option_report(
            model=model, ds=ds, model_rep=model_rep,
            option_dict={'bdg_x1': 5, 'bdg_x2': 5},
            option_settings=opt_set, if_exact=True
        )

        assert 'pred_exact_0' in opt_sum
        assert 'pred_exact_vol' in opt_sum['pred_exact_0']
        assert 'pred_long_vol' in opt_sum
        assert 'pred_ref_vol' in opt_sum
        assert 'ref_vol' in opt_sum
        assert 'growth_vol' in opt_sum

    def test_if_false_dispatches_to_long_translation(self, synthetic_setup, monkeypatch):
        ds, model, model_rep, opt_set = synthetic_setup

        translate_called = False
        original_translate = model_rep.option_translate_long

        def tracking_long(*args, **kwargs):
            nonlocal translate_called
            translate_called = True
            return original_translate(*args, **kwargs)

        model_rep.option_translate_long = tracking_long

        def mock_predict_ext(model, ds):
            return pd.DataFrame({
                'date': ds['date'], 'superbrand': ds['superbrand'],
                'listed': ds['listed'], 'predicted': [10.0] * len(ds),
                'observed': [10.0] * len(ds),
                'price': [1.0] * len(ds),
            })

        monkeypatch.setattr('fermatrica_rep.options.calc.transform', lambda ds, model, **kw: (model, ds))
        monkeypatch.setattr('fermatrica_rep.options.calc.predict_ext', mock_predict_ext)

        option_report(
            model=model, ds=ds, model_rep=model_rep,
            option_dict={'bdg_x1': 5},
            option_settings=opt_set, if_exact=False
        )

        assert translate_called

    def test_unknown_budget_key_does_not_crash(self, synthetic_setup, monkeypatch):
        ds, model, model_rep, opt_set = synthetic_setup

        def mock_predict_ext(model, ds):
            return pd.DataFrame({
                'date': ds['date'], 'superbrand': ds['superbrand'],
                'listed': ds['listed'], 'predicted': [10.0] * len(ds),
                'observed': [10.0] * len(ds), 'price': [1.0] * len(ds),
            })

        monkeypatch.setattr('fermatrica_rep.options.calc.transform', lambda ds, model, **kw: (model, ds))
        monkeypatch.setattr('fermatrica_rep.options.calc.predict_ext', mock_predict_ext)

        _, _, opt_sum = option_report(
            model=model, ds=ds, model_rep=model_rep,
            option_dict={'bdg_nonexistent': 100},
            option_settings=opt_set, if_exact=True
        )
        assert opt_sum['pred_exact_0']['pred_exact_vol'] > 0

    def test_golden_kpi_with_basic_translator(self, monkeypatch):
        n = 12
        ds = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=n, freq='MS'),
            'superbrand': ['brand_x'] * n,
            'bs_key': [1] * n,
            'listed': [4] * n,
            'x1': [0.0] * n,
            'x2': [0.0] * n,
            'opt_media_ssn_x1': [1.0 / n] * n,
        })

        mc = MockModelConf()
        mc.price_var = 'price'
        mc.summary_type = 'sum'
        mc.trans_path_df = pd.DataFrame({
            'variable': ['x1'],
            'price': [1.0]
        })

        class MockModel:
            conf = mc

        model = MockModel()

        def mock_predict_ext(model, ds):
            predicted = ds['x1'] * 3.0 + 10.0
            return pd.DataFrame({
                'date': ds['date'], 'superbrand': ds['superbrand'],
                'listed': ds['listed'], 'predicted': predicted,
                'observed': predicted, 'price': [1.0] * len(ds),
            })

        monkeypatch.setattr('fermatrica_rep.options.calc.transform', lambda ds, model, **kw: (model, ds))
        monkeypatch.setattr('fermatrica_rep.options.calc.predict_ext', mock_predict_ext)

        trans_dict = {
            'function': {'bdg_x1': 'basic'},
            'variable': {'bdg_x1': 'x1'},
        }
        model_rep = ModelRep(ds, trans_dict=trans_dict, language='english')
        opt_set = OptionSettings(
            target='brand_x', date_start='2024-01-01', date_end='2024-12-01',
            ref_date_start='2023-01-01', ref_date_end='2023-06-01'
        )

        _, _, opt_sum_zero = option_report(
            model=model, ds=ds, model_rep=model_rep,
            option_dict={'bdg_x1': 0},
            option_settings=opt_set, if_exact=True
        )
        _, _, opt_sum = option_report(
            model=model, ds=ds, model_rep=model_rep,
            option_dict={'bdg_x1': 100},
            option_settings=opt_set, if_exact=True
        )

        zero_vol = opt_sum_zero['pred_exact_0']['pred_exact_vol']
        nonzero_vol = opt_sum['pred_exact_0']['pred_exact_vol']
        assert nonzero_vol > zero_vol
        assert nonzero_vol > 1e6
        assert np.isfinite(nonzero_vol)
        assert np.isclose(opt_sum_zero['pred_exact_0']['pred_exact_vol'], n * 10.0, rtol=1e-6)


# ---------------------------------------------------------------------------
# TestOptionReportEdgeCases — zero-budget across modes
# ---------------------------------------------------------------------------

class TestOptionReportEdgeCases:

    @pytest.fixture
    def setup(self):
        n = 12
        ds = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=n, freq='MS'),
            'superbrand': ['brand_x'] * n,
            'bs_key': [1] * n,
            'listed': [4] * n,
            'x1': [0.0] * n,
            'x2': [0.0] * n,
            'opt_media_ssn_x1': [1.0 / n] * n,
        })
        mc = MockModelConf()
        mc.price_var = 'price'
        mc.summary_type = 'sum'
        mc.trans_path_df = pd.DataFrame({
            'variable': ['x1'],
            'price': [1.0]
        })

        class MockModel:
            conf = mc

        model = MockModel()

        trans_dict = {
            'function': {'bdg_x1': 'basic'},
            'variable': {'bdg_x1': 'x1'},
        }
        model_rep = ModelRep(ds, trans_dict=trans_dict, language='english')
        return ds, model, model_rep

    @staticmethod
    def _mock_predict(model, ds):
        predicted = ds['x1'] * 3.0 + 10.0
        return pd.DataFrame({
            'date': ds['date'], 'superbrand': ds['superbrand'],
            'listed': ds['listed'], 'predicted': predicted,
            'observed': predicted, 'price': [1.0] * len(ds),
        })

    def test_zero_budget_if_exact_true(self, setup, monkeypatch):
        ds, model, model_rep = setup
        monkeypatch.setattr('fermatrica_rep.options.calc.transform', lambda ds, model, **kw: (model, ds))
        monkeypatch.setattr('fermatrica_rep.options.calc.predict_ext', self._mock_predict)

        opt_set = OptionSettings(
            target='brand_x', date_start='2024-01-01', date_end='2024-12-01',
            ref_date_start='2023-01-01', ref_date_end='2023-06-01'
        )
        _, _, opt_sum = option_report(
            model=model, ds=ds, model_rep=model_rep,
            option_dict={'bdg_x1': 0}, option_settings=opt_set, if_exact=True,
        )
        assert np.isclose(opt_sum['pred_exact_0']['pred_exact_vol'], 120.0, rtol=1e-6)

    def test_zero_budget_if_exact_false(self, setup, monkeypatch):
        ds, model, model_rep = setup
        monkeypatch.setattr('fermatrica_rep.options.calc.transform', lambda ds, model, **kw: (model, ds))
        monkeypatch.setattr('fermatrica_rep.options.calc.predict_ext', self._mock_predict)

        opt_set = OptionSettings(
            target='brand_x', date_start='2024-01-01', date_end='2024-12-01',
            ref_date_start='2023-01-01', ref_date_end='2023-06-01'
        )
        _, _, opt_sum = option_report(
            model=model, ds=ds, model_rep=model_rep,
            option_dict={'bdg_x1': 0}, option_settings=opt_set, if_exact=False,
        )
        assert 'pred_long_vol' in opt_sum

    def test_zero_budget_long_term_first(self, setup, monkeypatch):
        ds, model, model_rep = setup
        monkeypatch.setattr('fermatrica_rep.options.calc.transform', lambda ds, model, **kw: (model, ds))
        monkeypatch.setattr('fermatrica_rep.options.calc.predict_ext', self._mock_predict)

        opt_set = OptionSettings(
            target='brand_x', date_start='2024-01-01', date_end='2024-12-01',
            ref_date_start='2023-01-01', ref_date_end='2023-06-01',
        )
        opt_set.long_term_type = 'first'
        _, _, opt_sum = option_report(
            model=model, ds=ds, model_rep=model_rep,
            option_dict={'bdg_x1': 0}, option_settings=opt_set, if_exact=False,
        )
        assert 'pred_long_vol' in opt_sum
        assert np.isfinite(opt_sum['pred_long_vol'])

    def test_zero_budget_long_term_last(self, setup, monkeypatch):
        ds, model, model_rep = setup
        monkeypatch.setattr('fermatrica_rep.options.calc.transform', lambda ds, model, **kw: (model, ds))
        monkeypatch.setattr('fermatrica_rep.options.calc.predict_ext', self._mock_predict)

        opt_set = OptionSettings(
            target='brand_x', date_start='2024-01-01', date_end='2024-12-01',
            ref_date_start='2023-01-01', ref_date_end='2023-06-01',
        )
        opt_set.long_term_type = 'last'
        _, _, opt_sum = option_report(
            model=model, ds=ds, model_rep=model_rep,
            option_dict={'bdg_x1': 0}, option_settings=opt_set, if_exact=False,
        )
        assert 'pred_long_vol' in opt_sum
        assert np.isfinite(opt_sum['pred_long_vol'])

    def test_zero_budget_allow_past_true(self, setup, monkeypatch):
        ds, model, model_rep = setup
        monkeypatch.setattr('fermatrica_rep.options.calc.transform', lambda ds, model, **kw: (model, ds))
        monkeypatch.setattr('fermatrica_rep.options.calc.predict_ext', self._mock_predict)

        opt_set = OptionSettings(
            target='brand_x', date_start='2023-01-01', date_end='2023-06-01',
            ref_date_start='2022-01-01', ref_date_end='2022-06-01',
        )
        _, _, opt_sum = option_report(
            model=model, ds=ds, model_rep=model_rep,
            option_dict={'bdg_x1': 0}, option_settings=opt_set, if_exact=True,
            allow_past=True,
        )
        assert 'pred_exact_0' in opt_sum
        assert np.isfinite(opt_sum['pred_exact_0']['pred_exact_vol'])

    def test_all_unknown_budget_keys_equals_zero(self, setup, monkeypatch):
        ds, model, model_rep = setup
        monkeypatch.setattr('fermatrica_rep.options.calc.transform', lambda ds, model, **kw: (model, ds))
        monkeypatch.setattr('fermatrica_rep.options.calc.predict_ext', self._mock_predict)

        opt_set = OptionSettings(
            target='brand_x', date_start='2024-01-01', date_end='2024-12-01',
            ref_date_start='2023-01-01', ref_date_end='2023-06-01',
        )
        _, _, opt_sum_zero = option_report(
            model=model, ds=ds, model_rep=model_rep,
            option_dict={'bdg_x1': 0}, option_settings=opt_set, if_exact=True,
        )
        _, _, opt_sum_unknown = option_report(
            model=model, ds=ds, model_rep=model_rep,
            option_dict={'bdg_unknown': 999}, option_settings=opt_set, if_exact=True,
        )
        assert np.isclose(
            opt_sum_unknown['pred_exact_0']['pred_exact_vol'],
            opt_sum_zero['pred_exact_0']['pred_exact_vol'],
            rtol=1e-6,
        )
