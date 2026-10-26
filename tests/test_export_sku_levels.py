"""Tests for hierarchical per-SKU reporting in options_detail.

Covers the private helpers `_sku_levels_expand` and `_iter_sku_level_keys`
introduced for reporting every prefix level of `sku_var`.
"""
import numpy as np
import pandas as pd

from fermatrica_rep.export.export_pptx import (
    _sku_levels_expand,
    _iter_sku_level_keys,
)


SKU_VAR = ['superbrand', 'master', 'form', 'region']


class TestSkuLevelsExpand:

    def test_hierarchy_off_returns_bottom_level(self):
        result = _sku_levels_expand(SKU_VAR, ['superbrand'], False)
        assert result == [SKU_VAR]

    def test_hierarchy_on_skips_covered_level(self):
        result = _sku_levels_expand(SKU_VAR, ['superbrand'], True)
        assert result == [
            ['superbrand', 'master'],
            ['superbrand', 'master', 'form'],
            ['superbrand', 'master', 'form', 'region'],
        ]

    def test_hierarchy_on_no_superbrand_keeps_all(self):
        sku_var = ['master', 'form', 'region']
        result = _sku_levels_expand(sku_var, ['superbrand'], True)
        assert result == [
            ['master'],
            ['master', 'form'],
            ['master', 'form', 'region'],
        ]

    def test_hierarchy_on_partial_group_var_coverage(self):
        result = _sku_levels_expand(SKU_VAR, ['superbrand', 'region'], True)
        assert result == [
            ['superbrand', 'master'],
            ['superbrand', 'master', 'form'],
            ['superbrand', 'master', 'form', 'region'],
        ]

    def test_hierarchy_on_single_column_covered_falls_back(self):
        result = _sku_levels_expand(['superbrand'], ['superbrand'], True)
        assert result == [['superbrand']]

    def test_sku_var_none_returns_empty(self):
        assert _sku_levels_expand(None, ['superbrand'], True) == []
        assert _sku_levels_expand([], ['superbrand'], True) == []


class TestIterSkuLevelKeys:

    def _df(self):
        return pd.DataFrame({
            'superbrand': ['brand_1', 'brand_1', 'brand_1', 'brand_2', 'brand_1'],
            'master': ['M1', 'M1', 'M2', 'M9', 'M2'],
            'form': ['A', 'B', 'C', 'A', 'D'],
            'region': ['Moscow', 'SPb', 'Kazan', 'Moscow', 'Vlad'],
            'bs_key': [1, 2, 3, 4, 5],
        })

    def test_intermediate_level_combos_and_bs_keys(self):
        dt_pred = self._df()
        results = list(_iter_sku_level_keys(dt_pred, 'brand_1', ['superbrand', 'master']))
        assert len(results) == 2
        combos = [tuple(row[['superbrand', 'master']]) for row, _ in results]
        assert combos == [('brand_1', 'M1'), ('brand_1', 'M2')]
        bs_key_sets = [set(bs_keys) for _, bs_keys in results]
        assert bs_key_sets == [{1, 2}, {3, 5}]

    def test_rows_restricted_to_target_superbrand(self):
        dt_pred = self._df()
        results = list(_iter_sku_level_keys(dt_pred, 'brand_1', ['master']))
        combos = [row['master'] for row, _ in results]
        assert combos == ['M1', 'M2']
        assert all(isinstance(bs_keys, np.ndarray) for _, bs_keys in results)
        assert set(results[0][1]) == {1, 2}
        assert set(results[1][1]) == {3, 5}

    def test_single_combo_level(self):
        dt_pred = pd.DataFrame({
            'superbrand': ['brand_1', 'brand_1'],
            'master': ['M1', 'M1'],
            'bs_key': [1, 2],
        })
        results = list(_iter_sku_level_keys(dt_pred, 'brand_1', ['superbrand', 'master']))
        assert len(results) == 1
        assert set(results[0][1]) == {1, 2}

    def test_bs_keys_do_not_leak_other_superbrand(self):
        dt_pred = pd.DataFrame({
            'superbrand': ['brand_1', 'brand_1', 'brand_2'],
            'master': ['M1', 'M1', 'M1'],
            'bs_key': [1, 2, 99],
        })
        results = list(_iter_sku_level_keys(dt_pred, 'brand_1', ['master']))
        assert len(results) == 1
        assert set(results[0][1]) == {1, 2}
