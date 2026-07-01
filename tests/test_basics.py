import copy
import pytest
import pandas as pd
import numpy as np

from fermatrica_rep.basics import palette_fill, _coef_var_align_one, coef_var_align


class TestPaletteFill:

    def test_empty_initial(self):
        result = palette_fill({}, ['brand_a', 'brand_b'])
        assert len(result) == 2
        assert 'brand_a' in result
        assert 'brand_b' in result

    def test_existing_items_preserved(self):
        old = {'brand_a': '#000000'}
        result = palette_fill(old, ['brand_b', 'brand_c'])
        assert result['brand_a'] == '#000000'
        assert 'brand_b' in result
        assert 'brand_c' in result

    def test_duplicate_new_skipped(self):
        old = {'brand_a': '#000000', 'brand_b': '#111111'}
        result = palette_fill(old, ['brand_a', 'brand_c'])
        assert len(result) == 3
        assert result['brand_a'] == '#000000'
        assert result['brand_b'] == '#111111'
        assert 'brand_c' in result

    def test_new_items_empty_list(self):
        old = {'brand_a': '#000000'}
        result = palette_fill(old, [])
        assert result == old

    def test_single_palette_name_string(self):
        result = palette_fill({}, ['brand_a'], palette_names='glasbey')
        assert 'brand_a' in result

    def test_tuple_palette_names(self):
        result = palette_fill({}, ['brand_a'], palette_names=('glasbey',))
        assert 'brand_a' in result

    def test_list_new_items(self):
        result = palette_fill({'a': '#000000'}, ['b', 'c'])
        assert result['a'] == '#000000'
        assert 'b' in result
        assert 'c' in result

    def test_dict_input_not_allowed_as_new(self):
        result = palette_fill({}, ['b'])
        assert 'b' in result

    def test_palette_no_duplicate_colors(self):
        old = {'a': '#000000', 'b': '#111111'}
        _palette = ['#000000', '#111111', '#222222', '#333333']
        result = palette_fill(old, ['c', 'd'], palette_names=('glasbey',))
        assigned = list(result.values())
        assert assigned.count('#000000') == 1
        assert assigned.count('#111111') == 1

    def test_many_entities(self):
        entities = [f'brand_{i}' for i in range(20)]
        result = palette_fill({}, entities)
        assert len(result) == 20
        for e in entities:
            assert e in result

    def test_colors_are_distinct_for_new_items(self):
        result = palette_fill({}, ['a', 'b', 'c', 'd'])
        vals = list(result.values())
        assert len(set(vals)) == 4


class TestCoefVarAlignOne:

    def test_exact_match(self):
        coefs = pd.Series([0.5, 0.3], index=['var_x', 'var_y'])
        result = _coef_var_align_one('var_x', coefs)
        assert result is not None
        assert result.index[0] == 'var_x'
        assert result.values[0] == 0.5

    def test_regex_fallback_one_match(self):
        coefs = pd.Series([0.5, 0.3], index=['adstock-var_x', 'var_y'])
        result = _coef_var_align_one('var_x', coefs)
        assert result is not None
        assert 'var_x' in result.index[0]

    def test_regex_fallback_multiple_matches_picks_first(self):
        coefs = pd.Series([0.1, 0.2, 0.3], index=['prefix-var_x', 'var_x-suffix', 'var_y'])
        result = _coef_var_align_one('var_x', coefs)
        assert result is not None
        assert len(result) == 1

    def test_no_match_returns_none(self):
        coefs = pd.Series([0.5, 0.3], index=['var_a', 'var_b'])
        result = _coef_var_align_one('var_x', coefs)
        assert result is None

    def test_empty_coefs_returns_none(self):
        coefs = pd.Series(dtype=float)
        result = _coef_var_align_one('var_x', coefs)
        assert result is None

    def test_substring_not_matched_as_word(self):
        coefs = pd.Series([0.5], index=['x_variable'])
        result = _coef_var_align_one('var', coefs)
        assert result is None

    def test_word_boundary_respected(self):
        coefs = pd.Series([0.5, 0.3], index=['my-var-123', 'notvar456'])
        result = _coef_var_align_one('var', coefs)
        assert result is not None
        assert 'var' in result.index[0]
        assert 'notvar456' not in result.index[0]


class TestCoefVarAlign:

    def test_basic_alignment(self):
        params = pd.DataFrame({
            'variable_fin': ['var_x', 'var_y', 'var_z']
        })
        coefs = pd.Series([0.5, 0.3, 0.1], index=['var_x', 'var_y', 'var_w'])
        result = coef_var_align(params, coefs)
        assert 'var_x' in result['coef_name'].values
        assert 'var_y' in result['coef_name'].values
        assert 'var_z' not in result['coef_name'].values

    def test_returns_only_matched_rows(self):
        params = pd.DataFrame({
            'variable_fin': ['var_x', 'var_y', 'var_z']
        })
        coefs = pd.Series([0.5, 0.3], index=['var_x', 'var_y'])
        result = coef_var_align(params, coefs)
        assert len(result) == 2

    def test_coef_value_populated(self):
        params = pd.DataFrame({
            'variable_fin': ['var_x', 'var_y']
        })
        coefs = pd.Series([0.5, 0.3], index=['var_x', 'var_y'])
        result = coef_var_align(params, coefs)
        assert np.isclose(result.loc[result['coef_name'] == 'var_x', 'coef_value'].values[0], 0.5)
        assert np.isclose(result.loc[result['coef_name'] == 'var_y', 'coef_value'].values[0], 0.3)

    def test_empty_params_returns_empty(self):
        params = pd.DataFrame({'variable_fin': pd.Series(dtype=str)})
        coefs = pd.Series([0.5], index=['var_x'])
        result = coef_var_align(params, coefs)
        assert len(result) == 0

    def test_empty_coefs_returns_empty(self):
        params = pd.DataFrame({'variable_fin': ['var_x', 'var_y']})
        coefs = pd.Series(dtype=float)
        result = coef_var_align(params, coefs)
        assert len(result) == 0

    def test_regex_fallback_in_alignment(self):
        params = pd.DataFrame({
            'variable_fin': ['var_x', 'var_y']
        })
        coefs = pd.Series([0.5, 0.3], index=['adstock-var_x', 'adstock-var_y'])
        result = coef_var_align(params, coefs)
        assert len(result) == 2

    def test_original_params_not_mutated(self):
        params = pd.DataFrame({'variable_fin': ['var_x']})
        coefs = pd.Series([0.5], index=['var_x'])
        params_copy = params.copy()
        coef_var_align(params, coefs)
        pd.testing.assert_frame_equal(params, params_copy)
