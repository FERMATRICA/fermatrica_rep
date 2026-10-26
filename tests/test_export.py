import copy

import numpy as np
import pandas as pd
import pytest

from fermatrica_rep.export.basics import _extract_options_data, resolve_scopes_for_node, resolve_scope_for_node
from fermatrica_rep.options.define import OptionSettings
from fermatrica_rep.options.scope import OptionScope


def _scope(name, date_start='2026-01-01', date_end='2026-12-31'):
    return OptionScope(
        scope_name=name,
        date_start=date_start,
        date_end=date_end,
        ref_date_start='2025-01-01',
        ref_date_end='2025-12-31',
        filters={'superbrand': ['brand_1']},
    )


class TestResolveScopesForNode:

    def _scopes(self):
        return [
            _scope('predefined_brand_1_2025', '2025-01-01', '2025-12-31'),
            _scope('predefined_brand_1_2026', '2026-01-01', '2026-12-31'),
            _scope('zero_brand_1_2026', '2026-01-01', '2026-12-31'),
        ]

    def test_returns_all_nonzero_scopes_in_order(self):
        node_result = {
            'predefined_brand_1_2025': {'pred_exact_vol': 100.0},
            'predefined_brand_1_2026': {'pred_exact_vol': 200.0},
            'zero_brand_1_2026': {'pred_exact_vol': 0.0},
        }
        resolved = resolve_scopes_for_node(node_result, self._scopes())
        assert [name for _, name in resolved] == [
            'predefined_brand_1_2025', 'predefined_brand_1_2026',
        ]

    def test_handles_return_full_tuples(self):
        scopes = self._scopes()
        node_result = {
            'predefined_brand_1_2026': ('ds', 'dt_pred', {'pred_exact_vol': 50.0}),
        }
        resolved = resolve_scopes_for_node(node_result, scopes)
        assert len(resolved) == 1
        scope, name = resolved[0]
        assert name == 'predefined_brand_1_2026'
        assert scope is scopes[1]

    def test_empty_when_all_zero(self):
        node_result = {
            'predefined_brand_1_2025': {'pred_exact_vol': 0.0},
            'predefined_brand_1_2026': {'pred_exact_vol': 0.0},
        }
        assert resolve_scopes_for_node(node_result, self._scopes()) == []

    def test_none_and_missing_skipped(self):
        node_result = {
            'predefined_brand_1_2026': {'pred_exact_vol': 7.0},
            'zero_brand_1_2026': None,
        }
        resolved = resolve_scopes_for_node(node_result, self._scopes())
        assert [name for _, name in resolved] == ['predefined_brand_1_2026']

    def test_scope_filtered_by_option_returns_only_its_scopes(self):
        # per-option scopes: the option's compute result covers only its own scopes
        node_result = {
            'predefined_brand_1_2025': {'pred_exact_vol': 100.0},
            'predefined_brand_1_2026': {'pred_exact_vol': 200.0},
        }
        resolved = resolve_scopes_for_node(node_result, self._scopes())
        assert [name for _, name in resolved] == [
            'predefined_brand_1_2025', 'predefined_brand_1_2026',
        ]

    def test_singular_wrapper_returns_first(self):
        node_result = {
            'predefined_brand_1_2025': {'pred_exact_vol': 0.0},
            'predefined_brand_1_2026': {'pred_exact_vol': 200.0},
        }
        scope, name = resolve_scope_for_node(node_result, self._scopes())
        assert name == 'predefined_brand_1_2026'

    def test_singular_wrapper_falls_back_to_first_scope(self):
        node_result = {}
        scope, name = resolve_scope_for_node(node_result, self._scopes())
        assert name == 'predefined_brand_1_2025'


class TestExtractOptionsData:

    def test_single_option_key(self):
        options_m = {'base': {'bdg_tv': 100, 'bdg_digital': 50}}
        result = _extract_options_data(options_m, 'base', opt_set=None)
        assert result == options_m['base']

    def test_list_mode_extracts_years(self):
        opt_set = OptionSettings(
            target='brand_x', date_start='2024-01-01', date_end='2025-12-31',
            ref_date_start='2023-01-01', ref_date_end='2023-12-01',
            plan_period='year',
        )
        opt_set_list = [opt_set]
        options_m = {
            'multi_year': {
                2024: {'bdg_tv': 100},
                2025: {'bdg_tv': 150},
                2026: {'bdg_tv': 200},
            }
        }
        result = _extract_options_data(options_m, 'multi_year', opt_set_list, is_list_mode=True)
        assert len(result) == 2
        assert {'bdg_tv': 100} in result
        assert {'bdg_tv': 150} in result

    def test_list_mode_empty_when_no_years_match(self):
        opt_set = OptionSettings(
            target='brand_x', date_start='2024-01-01', date_end='2024-12-31',
            ref_date_start='2023-01-01', ref_date_end='2023-12-01',
            plan_period='year',
        )
        opt_set_list = [opt_set]
        options_m = {
            'future': {
                2025: {'bdg_tv': 100},
                2026: {'bdg_tv': 150},
            }
        }
        result = _extract_options_data(options_m, 'future', opt_set_list, is_list_mode=True)
        assert len(result) == 0

    def test_non_list_mode_returns_full_dict(self):
        opt_set = OptionSettings(
            target='brand_x', date_start='2024-01-01', date_end='2024-12-31',
            ref_date_start='2023-01-01', ref_date_end='2023-12-01',
            plan_period='year',
        )
        options_m = {'base': {'bdg_tv': 100}}
        result = _extract_options_data(options_m, 'base', opt_set, is_list_mode=False)
        assert result == options_m['base']


class TestGetData:

    @pytest.fixture
    def simple_model_conf(self):
        class MockModelConf:
            pass
        mc = MockModelConf()
        mc.model_rhs = pd.DataFrame({
            'token': ['x1', 'x2'],
            'display_var': ['x1_disp', 'x2_disp'],
            'if_active': [1, 1],
        })
        mc.params = pd.DataFrame({
            'variable': ['raw_var1', 'raw_var2'],
            'fun': ['adstock', 'adstock'],
            'arg': ['0.5', '0.3'],
            'value': [0.5, 0.3],
        })
        mc.Y_var = 'y'
        return mc

    @pytest.fixture
    def simple_ds(self):
        return pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10, freq='MS'),
            'listed': [2] * 5 + [3] * 5,
            'y': range(10, 20),
            'raw_var1': range(1, 11),
            'raw_var2': range(10, 0, -1),
        })

    def test_get_data_returns_dataframe(self, simple_model_conf, simple_ds):
        from fermatrica_rep.export.export_results import get_data

        class MockModel:
            conf = simple_model_conf

        model = MockModel()
        result = get_data(model, simple_ds)
        assert isinstance(result, pd.DataFrame)

    def test_get_data_has_date_index(self, simple_model_conf, simple_ds):
        from fermatrica_rep.export.export_results import get_data

        class MockModel:
            conf = simple_model_conf

        model = MockModel()
        result = get_data(model, simple_ds)
        assert result.index.name == 'date' or 'date' in result.index.names

    def test_get_data_excludes_listed_1(self, simple_model_conf, simple_ds):
        from fermatrica_rep.export.export_results import get_data

        class MockModel:
            conf = simple_model_conf

        model = MockModel()
        simple_ds.loc[0, 'listed'] = 1
        result = get_data(model, simple_ds)
        assert len(result) == 9

    def test_get_data_detailed_returns_without_listed_1(self, simple_model_conf, simple_ds):
        from fermatrica_rep.export.export_results import get_data

        class MockModel:
            conf = simple_model_conf

        model = MockModel()
        simple_ds.loc[0, 'listed'] = 1
        result = get_data(model, simple_ds, return_detailed=True)
        assert len(result) == 9

    def test_decompose_detailed_returns_train_data(self, simple_model_conf, simple_ds):
        from fermatrica_rep.export.export_results import decompose_detailed
        import statsmodels.api as sm
        from statsmodels.formula.api import ols

        ols_model = ols('y ~ raw_var1 + raw_var2', data=simple_ds).fit()

        class MockModel:
            conf = simple_model_conf
            obj = type('Obj', (), {'models': {'main': ols_model}})()

        model = MockModel()
        result = decompose_detailed(model, simple_ds, target_var='y')
        assert isinstance(result, pd.DataFrame)
        assert 'y' in result.columns or 'date' in result.columns


class TestGetDataEdgeCases:

    @pytest.fixture
    def empty_model_conf(self):
        class MockModelConf:
            pass
        mc = MockModelConf()
        mc.model_rhs = pd.DataFrame({'token': [], 'display_var': [], 'if_active': []})
        mc.params = pd.DataFrame({'variable': [], 'fun': [], 'arg': [], 'value': []})
        mc.Y_var = 'y'
        return mc

    def test_empty_ds_returns_dataframe(self, empty_model_conf):
        from fermatrica_rep.export.export_results import get_data

        class MockModel:
            conf = empty_model_conf

        model = MockModel()
        ds = pd.DataFrame({'date': pd.date_range('2024-01-01', periods=0), 'listed': [], 'y': []})
        result = get_data(model, ds)
        assert isinstance(result, pd.DataFrame)

    def test_single_row(self, empty_model_conf):
        from fermatrica_rep.export.export_results import get_data

        class MockModel:
            conf = empty_model_conf

        model = MockModel()
        ds = pd.DataFrame({
            'date': [pd.Timestamp('2024-01-01')], 'listed': [2], 'y': [1.0]
        })
        result = get_data(model, ds)
        assert len(result) == 1

    def test_all_zero_values(self):
        from fermatrica_rep.export.export_results import get_data

        class MockModelConf:
            pass
        mc = MockModelConf()
        mc.model_rhs = pd.DataFrame({
            'token': ['x1', 'x2'], 'display_var': ['x1', 'x2'], 'if_active': [1, 1]
        })
        mc.params = pd.DataFrame({
            'variable': ['raw1', 'raw2'], 'fun': ['', ''], 'arg': ['', ''], 'value': [0.0, 0.0]
        })
        mc.Y_var = 'y'

        class MockModel:
            conf = mc

        model = MockModel()
        ds = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5, freq='MS'),
            'listed': [2] * 5, 'y': [0.0] * 5,
            'raw1': [0.0] * 5, 'raw2': [0.0] * 5,
        })
        result = get_data(model, ds)
        assert np.all(np.isfinite(result.values))
