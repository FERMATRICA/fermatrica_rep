import pytest
import pandas as pd
import numpy as np

from fermatrica_rep.options.define import (trans_dict_create, budget_dict_create,
    media_ssn_create, media_ssn_apply, option_settings_list_create, OptionSettings)
from fermatrica.basics.basics import FermatricaError


class TestTransDictCreate:

    def test_from_path(self, options_xlsx_path):
        result = trans_dict_create(path=options_xlsx_path,
                                   sheet='translation')
        assert isinstance(result, dict)
        assert 'bdg_tv' in result['variable']
        assert 'bdg_ooh' in result['variable']
        assert 'bdg_pos' in result['variable']
        assert result['function']['bdg_tv'] == 'basic'
        assert result['variable']['bdg_tv'] == 'brnd_tv_ots_clip'
        assert result['max_budg']['bdg_tv'] == 500

    def test_from_path_non_existent_sheet(self, options_xlsx_path):
        with pytest.raises(FermatricaError, match='translation rules'):
            trans_dict_create(path=options_xlsx_path,
                              sheet='nonexistent')

    def test_from_path_non_existent_file(self):
        with pytest.raises(FileNotFoundError):
            trans_dict_create(path=r'C:\nonexistent_file.xlsx')

    def test_no_input_raises(self):
        with pytest.raises(FermatricaError, match='translation rules'):
            trans_dict_create()

    def test_na_path_raises(self):
        with pytest.raises(FermatricaError, match='translation rules'):
            trans_dict_create(path=None)

    def test_all_9_items_in_variable_subdict(self, options_xlsx_path):
        result = trans_dict_create(path=options_xlsx_path,
                                   sheet='translation')
        assert len(result['variable']) == 9
        assert len(result['function']) == 9


class TestBudgetDictCreate:

    def test_from_path_main(self, options_xlsx_path):
        result = budget_dict_create(path=options_xlsx_path,
                                    sheet='main', vat_rate=0.0)
        assert isinstance(result, dict)
        assert 'zero' in result
        assert 'predefined' in result
        assert result['zero']['bdg'] == 0.0
        assert np.isclose(result['predefined']['bdg'], 200 + 30 + 20 + 30 + 30 + 20 + 10 + 10 + 20)

    def test_from_path_current(self, options_xlsx_path):
        result = budget_dict_create(path=options_xlsx_path,
                                    sheet='current', vat_rate=0.0)
        assert 'predefined' in result
        assert result['predefined']['bdg_tv'] == 200

    def test_from_path_with_vat(self, options_xlsx_path):
        result_no_vat = budget_dict_create(path=options_xlsx_path,
                                           sheet='main', vat_rate=0.0)
        result_with_vat = budget_dict_create(path=options_xlsx_path,
                                             sheet='main', vat_rate=0.20)
        for opt in result_no_vat:
            for col in result_no_vat[opt]:
                if col.startswith('bdg_'):
                    expected = result_no_vat[opt][col] * 1.20
                    assert np.isclose(result_with_vat[opt][col], expected)

    def test_from_path_non_existent_sheet(self, options_xlsx_path):
        with pytest.raises(FermatricaError, match='budget options'):
            budget_dict_create(path=options_xlsx_path,
                               sheet='nonexistent')

    def test_from_path_non_existent_file(self):
        with pytest.raises(FileNotFoundError):
            budget_dict_create(path=r'C:\nonexistent.xlsx')

    def test_from_dataframe(self):
        df = pd.DataFrame({
            'option': ['opt_a', 'opt_b'],
            'bdg_tv': [100, 200],
            'bdg_radio': [50, 0]
        })
        result = budget_dict_create(dct=df, vat_rate=0.0)
        assert 'opt_a' in result
        assert 'opt_b' in result
        assert result['opt_a']['bdg'] == 150
        assert result['opt_b']['bdg'] == 200

    def test_from_dict(self):
        dct = {
            'opt_a': {'bdg_tv': 100, 'bdg_radio': 50},
            'opt_b': {'bdg_tv': 200}
        }
        result = budget_dict_create(dct=dct, vat_rate=0.0)
        assert 'opt_a' in result
        assert 'opt_b' in result

    def test_no_input_raises(self):
        with pytest.raises(FermatricaError, match='budget options'):
            budget_dict_create()


class TestMediaSsnCreate:

    def test_from_path(self, seasonality_xlsx_path):
        result = media_ssn_create(path=seasonality_xlsx_path)
        assert isinstance(result, dict)
        assert 'data' in result
        ssn = result['data']
        assert 'month' in ssn.columns
        assert 'tv' in ssn.columns
        assert 'ooh' in ssn.columns
        assert len(ssn) == 13

    def test_sheets_filter_list(self, seasonality_xlsx_path):
        result = media_ssn_create(path=seasonality_xlsx_path,
                                  sheets=['data'])
        assert 'data' in result

    def test_sheets_filter_no_match(self, seasonality_xlsx_path):
        with pytest.raises(FermatricaError):
            media_ssn_create(path=seasonality_xlsx_path,
                             sheets=['nonexistent'])

    def test_non_existent_file(self):
        with pytest.raises(FermatricaError, match='media seasonality'):
            media_ssn_create(path=r'C:\nonexistent.xlsx')

    def test_empty_path_raises(self):
        with pytest.raises(FermatricaError, match='media seasonality'):
            media_ssn_create(path='')


class TestOptionSettingsListCreate:

    def test_single_year(self):
        result = option_settings_list_create(
            target='brand_1',
            date_start='2025-01-01',
            date_end='2025-12-31',
            ref_date_start='2024-01-01',
            ref_date_end='2024-12-31',
            plan_period='year'
        )
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], OptionSettings)

    def test_multi_year_split(self):
        result = option_settings_list_create(
            target='brand_1',
            date_start='2025-01-01',
            date_end='2026-12-31',
            ref_date_start='2024-01-01',
            ref_date_end='2024-12-31',
            split_period='year'
        )
        assert len(result) == 2
        assert result[0].date_start == pd.Timestamp('2025-01-01')
        assert result[0].date_end == pd.Timestamp('2025-12-31')
        assert result[1].date_start == pd.Timestamp('2026-01-01')
        assert result[1].date_end == pd.Timestamp('2026-12-31')

    def test_partial_first_last_year(self):
        result = option_settings_list_create(
            target='brand_1',
            date_start='2025-06-01',
            date_end='2027-03-31',
            ref_date_start='2024-01-01',
            ref_date_end='2024-12-31',
            split_period='year'
        )
        assert len(result) == 3
        assert result[0].date_start == pd.Timestamp('2025-06-01')
        assert result[0].date_end == pd.Timestamp('2025-12-31')
        assert result[1].date_start == pd.Timestamp('2026-01-01')
        assert result[1].date_end == pd.Timestamp('2026-12-31')
        assert result[2].date_start == pd.Timestamp('2027-01-01')
        assert result[2].date_end == pd.Timestamp('2027-03-31')

    def test_with_options_dict(self):
        opt_dict = {2025: {'bdg_tv': 100}, 2026: {'bdg_tv': 200}}
        result = option_settings_list_create(
            target='brand_1',
            date_start='2025-01-01',
            date_end='2026-12-31',
            ref_date_start='2024-01-01',
            ref_date_end='2024-12-31',
            split_period='year',
            options_dict=opt_dict
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert len(result[0]) == 2
        assert len(result[1]) == 2

    def test_target_string_converted_to_list(self):
        result = option_settings_list_create(
            target='brand_1',
            date_start='2025-01-01',
            date_end='2025-12-31',
            ref_date_start='2024-01-01',
            ref_date_end='2024-12-31'
        )
        assert result[0].target == ['brand_1']

    def test_plan_period_non_default(self):
        result = option_settings_list_create(
            target='brand_1',
            date_start='2025-01-01',
            date_end='2025-12-31',
            ref_date_start='2024-01-01',
            ref_date_end='2024-12-31',
            plan_period='exact'
        )
        assert result[0].plan_period == 'exact'


class TestMediaSsnApply:

    def test_apply_from_path(self, seasonality_xlsx_path):
        ds = pd.DataFrame({
            'date': pd.date_range('2025-01-01', periods=12, freq='MS'),
            'value': range(12)
        })
        result = media_ssn_apply(ds=ds,
                                 path=seasonality_xlsx_path,
                                 sheet='data')
        assert 'opt_media_ssn_tv' in result.columns
        assert 'opt_media_ssn_ooh' in result.columns
        assert len(result) == 12

    def test_apply_from_dataframe(self):
        ds = pd.DataFrame({
            'date': pd.date_range('2025-01-01', periods=12, freq='MS'),
            'value': range(12)
        })
        dt_ssn = pd.DataFrame({
            'month': range(1, 13),
            'tv': [0.1] * 12,
            'ooh': [0.05] * 12
        })
        result = media_ssn_apply(ds=ds, dt_ssn=dt_ssn)
        assert 'opt_media_ssn_tv' in result.columns
        assert 'opt_media_ssn_ooh' in result.columns
        assert np.allclose(result['opt_media_ssn_tv'], 0.1)

    def test_apply_removes_old_ssn_columns(self):
        ds = pd.DataFrame({
            'date': pd.date_range('2025-01-01', periods=12, freq='MS'),
            'value': range(12),
            'opt_media_ssn_old': [1.0] * 12
        })
        dt_ssn = pd.DataFrame({
            'month': range(1, 13),
            'tv': [0.1] * 12
        })
        result = media_ssn_apply(ds=ds, dt_ssn=dt_ssn)
        assert 'opt_media_ssn_old' not in result.columns
        assert 'opt_media_ssn_tv' in result.columns

    def test_date_start_preserves_outside_values(self):
        ds = pd.DataFrame({
            'date': pd.date_range('2025-01-01', periods=12, freq='MS'),
            'value': range(12),
            'opt_media_ssn_tv': [0.5] * 12
        })
        dt_ssn = pd.DataFrame({
            'month': range(1, 13),
            'tv': [0.08] * 12
        })
        result = media_ssn_apply(ds=ds, dt_ssn=dt_ssn, date_start='2025-03-01')
        jan_mask = result['date'] < '2025-03-01'
        assert np.allclose(result.loc[jan_mask, 'opt_media_ssn_tv'], 0.5)
        mar_mask = result['date'] >= '2025-03-01'
        assert np.allclose(result.loc[mar_mask, 'opt_media_ssn_tv'], 0.08)

    def test_with_sample_seasonality(self, seasonality_xlsx_path):
        ds = pd.DataFrame({
            'date': pd.date_range('2025-01-01', periods=12, freq='MS'),
            'value': range(12)
        })
        result = media_ssn_apply(ds=ds,
                                 path=seasonality_xlsx_path,
                                 sheet='data')
        jan_val = result.loc[result['date'] == '2025-01-01', 'opt_media_ssn_tv'].iloc[0]
        assert np.isclose(jan_val, 0.051852, atol=1e-5)

    def test_no_input_raises(self):
        with pytest.raises(FermatricaError, match='media seasonality'):
            media_ssn_apply(ds=pd.DataFrame())

    def test_with_week_seasonality(self):
        import calendar
        weeks_2025 = list(pd.date_range('2025-01-01', '2025-12-31', freq='W'))
        ds = pd.DataFrame({'date': weeks_2025, 'value': range(len(weeks_2025))})
        dt_ssn = pd.DataFrame({
            'week_number': list(range(1, 53)),
            'tv': [0.02] * 52,
            'ooh': [0.01] * 52
        })
        result = media_ssn_apply(ds=ds, dt_ssn=dt_ssn)
        assert 'opt_media_ssn_tv' in result.columns
