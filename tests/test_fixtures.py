"""Smoke tests for shared fixtures in tests/conftest.py.

Ensures shared fixtures produce expected structure. If these break,
many downstream tests will also fail.
"""
import numpy as np
import pandas as pd
import pytest

from fermatrica_rep.meta_model.model_rep import ModelRep


class TestSplitMMFixture:

    def test_has_expected_columns(self, split_m_m):
        for col in ['superbrand', 'bs_key', 'date', 'variable', 'value', 'listed']:
            assert col in split_m_m.columns

    def test_row_count(self, split_m_m):
        assert len(split_m_m) == 12

    def test_expected_variables(self, split_m_m):
        assert set(split_m_m['variable'].unique()) == {'tv', 'digital', 'pr', 'base'}

    def test_finite_values(self, split_m_m):
        assert np.all(np.isfinite(split_m_m['value']))

    def test_single_superbrand(self, split_m_m):
        assert split_m_m['superbrand'].nunique() == 1
        assert split_m_m['superbrand'].iloc[0] == 'brand_1'


class TestBasicModelRepFixture:

    def test_language_is_english(self, basic_model_rep):
        assert basic_model_rep.language == 'english'

    def test_vis_dict_loaded(self, basic_model_rep):
        assert basic_model_rep.vis_dict is not None
        assert isinstance(basic_model_rep.vis_dict, pd.DataFrame)
        assert len(basic_model_rep.vis_dict) > 0
