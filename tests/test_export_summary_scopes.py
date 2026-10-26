"""Tests for `summary_scopes` aggregation in options_detail export.

Covers the private helpers introduced for aggregating simultaneous
(per-region / per-segment) OptionScopes into a single top-level slide set:
`_scope_group_meta`, `_sum_budgets`, `_union_scope_filters`,
`_find_scope_for_row`, and the waterfall `filters` mask.
"""
import pandas as pd

from fermatrica_rep.options.option import Option
from fermatrica_rep.options.scope import OptionScope
from fermatrica_rep.export.export_pptx import (
    _scope_group_meta,
    _sum_budgets,
    _union_scope_filters,
    _find_scope_for_row,
)
from fermatrica_rep.export.slides.waterfall import _waterfall_chart_data


def _scope(name, filters, date_start='2026-01-01', date_end='2026-12-31'):
    return OptionScope(
        scope_name=name,
        date_start=date_start,
        date_end=date_end,
        ref_date_start='2025-01-01',
        ref_date_end='2025-12-31',
        filters=filters,
    )


class TestScopeGroupMeta:

    def test_single_regional_scope(self):
        scope = _scope('moscow', {'superbrand': ['brand_1'], 'region': ['Moscow']},
                       '2026-01-01', '2026-12-31')
        meta = _scope_group_meta([scope])
        assert meta['group_var'] == ['superbrand', 'region']
        assert meta['target_vals'] == ['brand_1', 'Moscow']
        assert meta['date_start'] == pd.Timestamp('2026-01-01')
        assert meta['date_end'] == pd.Timestamp('2026-12-31')

    def test_two_regional_scopes_union(self):
        s1 = _scope('moscow', {'superbrand': ['brand_1'], 'region': ['Moscow']},
                    '2026-01-01', '2026-06-30')
        s2 = _scope('spb', {'superbrand': ['brand_1'], 'region': ['SPb']},
                    '2026-07-01', '2026-12-31')
        meta = _scope_group_meta([s1, s2])
        assert meta['group_var'] == ['superbrand']
        assert meta['target_vals'] == ['brand_1']
        assert meta['date_start'] == pd.Timestamp('2026-01-01')
        assert meta['date_end'] == pd.Timestamp('2026-12-31')

    def test_empty_filters(self):
        s1 = _scope('moscow', {})
        s2 = _scope('spb', {})
        meta = _scope_group_meta([s1, s2])
        assert meta['group_var'] == ['superbrand']
        assert meta['target_vals'] == []

    def test_no_common_const_keys_falls_back_to_superbrand(self):
        s1 = _scope('moscow', {'region': ['Moscow']})
        s2 = _scope('spb', {'region': ['SPb']})
        meta = _scope_group_meta([s1, s2])
        assert meta['group_var'] == ['superbrand']
        assert meta['target_vals'] == []


class TestSumBudgets:

    def test_sums_items_across_scopes(self):
        option = Option(name='opt', budget_data={
            'moscow': {'bdg_tv': 500, 'bdg_digital': 100},
            'spb': {'bdg_tv': 300, 'bdg_reg_tv': 400},
        })
        scopes = [_scope('moscow', {}), _scope('spb', {})]
        result = _sum_budgets(option, scopes)
        assert result == {'bdg_tv': 800.0, 'bdg_digital': 100.0, 'bdg_reg_tv': 400.0}

    def test_missing_item_in_scope_counts_zero(self):
        option = Option(name='opt', budget_data={
            'moscow': {'bdg_tv': 500},
            'spb': {'bdg_digital': 100},
        })
        scopes = [_scope('moscow', {}), _scope('spb', {})]
        result = _sum_budgets(option, scopes)
        assert result == {'bdg_tv': 500.0, 'bdg_digital': 100.0}

    def test_dict_passed_through_unchanged(self):
        scopes = [_scope('moscow', {}), _scope('spb', {})]
        option_dict = {'bdg_tv': 100}
        assert _sum_budgets(option_dict, scopes) is option_dict

    def test_none_passed_through_unchanged(self):
        scopes = [_scope('moscow', {})]
        assert _sum_budgets(None, scopes) is None


class TestUnionScopeFilters:

    def test_union_without_duplicates(self):
        s1 = _scope('moscow', {'superbrand': ['brand_1'], 'region': ['Moscow']})
        s2 = _scope('spb', {'superbrand': ['brand_1'], 'region': ['SPb', 'Moscow']})
        result = _union_scope_filters([s1, s2])
        assert result == {'superbrand': ['brand_1'], 'region': ['Moscow', 'SPb']}

    def test_disjoint_filter_keys(self):
        s1 = _scope('moscow', {'region': ['Moscow']})
        s2 = _scope('national', {'channel': ['tv']})
        result = _union_scope_filters([s1, s2])
        assert result == {'region': ['Moscow'], 'channel': ['tv']}


class TestFindScopeForRow:

    def test_row_matches_its_scope(self):
        s_moscow = _scope('moscow', {'superbrand': ['brand_1'], 'region': ['Moscow']})
        s_paris = _scope('paris', {'superbrand': ['brand_1'], 'region': ['Paris']})
        row = pd.Series({'superbrand': 'brand_1', 'region': 'Paris'})
        result = _find_scope_for_row([s_moscow, s_paris], row, ['superbrand', 'region'])
        assert result is s_paris

    def test_unmatched_row_falls_back_to_first_scope(self):
        s_moscow = _scope('moscow', {'superbrand': ['brand_1'], 'region': ['Moscow']})
        s_paris = _scope('paris', {'superbrand': ['brand_1'], 'region': ['Paris']})
        row = pd.Series({'superbrand': 'brand_1', 'region': 'Berlin'})
        result = _find_scope_for_row([s_moscow, s_paris], row, ['superbrand', 'region'])
        assert result is s_moscow


class TestWaterfallChartDataScopeFilters:

    def test_scope_filters_restrict_rows(self):
        n = 12
        split = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=n, freq='MS'),
            'superbrand': ['brand_1'] * n,
            'region': ['Moscow'] * 6 + ['Paris'] * 6,
            'bs_key': [1] * n,
            'listed': [2] * n,
            'variable': ['tv'] * 6 + ['digital'] * 6,
            'value': list(range(1, 7)) + list(range(7, 13)),
        })
        result = _waterfall_chart_data(split=split, brands=['brand_1'],
                                       date_start='2024-01-01', date_end='2024-12-31',
                                       if_volume=True,
                                       filters={'region': ['Paris']})
        assert set(result['variable']) == {'digital'}
        assert result['value'].iloc[0] == 100.0

    def test_scope_filters_none_is_noop(self):
        n = 12
        split = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=n, freq='MS'),
            'superbrand': ['brand_1'] * n,
            'region': ['Moscow'] * 6 + ['Paris'] * 6,
            'bs_key': [1] * n,
            'listed': [2] * n,
            'variable': ['tv'] * n,
            'value': list(range(1, n + 1)),
        })
        with_filter = _waterfall_chart_data(split=split, brands=['brand_1'],
                                            date_start='2024-01-01', date_end='2024-12-31',
                                            if_volume=True,
                                            filters={'region': ['Paris']})
        without_filter = _waterfall_chart_data(split=split, brands=['brand_1'],
                                               date_start='2024-01-01', date_end='2024-12-31',
                                               if_volume=True,
                                               filters=None)
        assert len(with_filter) == 1
        assert len(without_filter) == 1
        assert with_filter['value'].iloc[0] == 100.0
        assert without_filter['value'].iloc[0] == 100.0
