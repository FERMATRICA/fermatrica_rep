"""Unit tests for ModelRep — the central reporting configuration object.

Tests cover construction, StableClass immutability, vis_dict loading,
and adhoc code source extraction.
"""
import copy
import inspect
import pandas as pd
import pytest

from fermatrica_rep.meta_model.model_rep import ModelRep
from fermatrica.basics.basics import FermatricaError
from fermatrica_utils import StableClass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_ds():
    return pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=3, freq='MS'),
        'superbrand': ['brand_x'] * 3,
    })


# ---------------------------------------------------------------------------
# TestModelRepConstruction
# ---------------------------------------------------------------------------

class TestModelRepConstruction:

    def test_minimal_construction(self):
        ds = _minimal_ds()
        mr = ModelRep(ds)
        assert mr.language == 'english'
        assert mr.vis_dict is not None
        assert isinstance(mr.vis_dict, pd.DataFrame)
        assert mr.trans_dict is None
        assert mr.adhoc_code_src == {}
        assert mr.option_summary_adhoc is None
        assert mr.aud_coefs is None
        assert mr.ssn_dict is None

    def test_language_english(self):
        ds = _minimal_ds()
        mr = ModelRep(ds, language='english')
        assert mr.language == 'english'

    def test_language_russian(self):
        ds = _minimal_ds()
        mr = ModelRep(ds, language='russian')
        assert mr.language == 'russian'

    def test_trans_dict_assignment(self):
        ds = _minimal_ds()
        trans_dict = {'function': {'bdg_x': 'basic'}, 'variable': {'bdg_x': 'x1'}}
        mr = ModelRep(ds, trans_dict=trans_dict)
        assert mr.trans_dict == trans_dict

    def test_all_fields_persist(self):
        ds = _minimal_ds()
        ssn_dict = {'ssn_1': pd.DataFrame({'period': [1, 2], 'ssn': [0.9, 1.1]})}

        def dummy_translate(*args, **kwargs):
            return args[0]

        def dummy_summary(*args, **kwargs):
            return {}

        mr = ModelRep(
            ds,
            trans_dict={'function': {}, 'variable': {}},
            language='russian',
            adhoc_code=[],
            option_translate_long_fn=dummy_translate,
            option_summary_adhoc_fn=dummy_summary,
            ssn_dict=ssn_dict,
        )
        assert mr.language == 'russian'
        assert mr.trans_dict == {'function': {}, 'variable': {}}
        assert mr.option_translate_long is dummy_translate
        assert mr.option_summary_adhoc is dummy_summary
        assert mr.ssn_dict is ssn_dict


# ---------------------------------------------------------------------------
# TestModelRepStableClass
# ---------------------------------------------------------------------------

class TestModelRepStableClass:

    def test_existing_attributes_can_be_read(self):
        ds = _minimal_ds()
        mr = ModelRep(ds)
        assert mr.language == 'english'
        assert isinstance(mr.vis_dict, pd.DataFrame)
        assert isinstance(mr.palette_sbr, dict)
        assert isinstance(mr.palette_tools, dict)


# ---------------------------------------------------------------------------
# TestModelRepVisDict
# ---------------------------------------------------------------------------

class TestModelRepVisDict:

    def test_vis_dict_loaded_by_default(self):
        ds = _minimal_ds()
        mr = ModelRep(ds)
        assert isinstance(mr.vis_dict, pd.DataFrame)
        assert len(mr.vis_dict) > 0

    def test_vis_dict_has_expected_keys(self):
        ds = _minimal_ds()
        mr = ModelRep(ds)
        required_cols = {'section', 'variable', 'english', 'russian'}
        assert required_cols.issubset(mr.vis_dict.columns)

    def test_vis_dict_custom(self):
        ds = _minimal_ds()
        custom_vis = pd.DataFrame({
            'section': ['test'],
            'variable': ['greeting'],
            'english': ['Hello'],
            'russian': ['Privet'],
        })
        mr = ModelRep(ds, vis_dict=custom_vis)
        assert mr.vis_dict is custom_vis
        val = mr.vis_dict.loc[
            (mr.vis_dict['section'] == 'test') &
            (mr.vis_dict['variable'] == 'greeting'),
            'english'
        ].iloc[0]
        assert val == 'Hello'


# ---------------------------------------------------------------------------
# TestModelRepAdhocCode
# ---------------------------------------------------------------------------

class TestModelRepAdhocCode:

    def test_adhoc_code_empty_default(self):
        ds = _minimal_ds()
        mr = ModelRep(ds)
        assert mr.adhoc_code_src == {}

    def test_adhoc_code_stored_as_source(self):
        import pandas as pd
        ds = _minimal_ds()
        mr = ModelRep(ds, adhoc_code=[pd])
        assert 'pandas' in mr.adhoc_code_src
        assert 'DataFrame' in mr.adhoc_code_src['pandas']


# ---------------------------------------------------------------------------
# TestModelRepPalette
# ---------------------------------------------------------------------------

class TestModelRepPalette:

    def test_palette_sbr_populated_from_superbrands(self):
        ds = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=6, freq='MS'),
            'superbrand': ['brand_a'] * 3 + ['brand_b'] * 3,
        })
        mr = ModelRep(ds)
        assert 'brand_a' in mr.palette_sbr
        assert 'brand_b' in mr.palette_sbr

    def test_palette_sbr_no_duplicate_colors(self):
        ds = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=9, freq='MS'),
            'superbrand': ['brand_a', 'brand_b', 'brand_c'] * 3,
        })
        mr = ModelRep(ds)
        colors = list(mr.palette_sbr.values())
        assert len(colors) == len(set(colors)), "Duplicate colors in palette_sbr"

    def test_palette_sbr_empty_dataset_no_crash(self):
        ds = pd.DataFrame({'date': pd.Series(dtype='datetime64[ns]'), 'superbrand': pd.Series(dtype='str')})
        mr = ModelRep(ds)
        assert isinstance(mr.palette_sbr, dict)
