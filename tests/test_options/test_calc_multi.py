"""Unit tests for calc_multi.py: opt_grid and option_report_multi_post.

Tests cover grid shape, budget step scaling, column integrity,
edge cases, and multiprocessing behaviour via monkeypatch.
"""
import numpy as np
import pandas as pd
import pytest

from fermatrica.basics.basics import FermatricaError
from fermatrica_rep.options.calc_multi import opt_grid, option_report_multi_post
from fermatrica_rep.options.define import OptionSettings
from fermatrica_rep.meta_model.model_rep import ModelRep


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_trans_dict():
    return pd.DataFrame({
        'max_budg': [100, 50],
        'function': ['basic', 'basic'],
        'variable': ['tv_ots', 'digital_imp'],
        'price': [0.05, 0.02],
    }, index=pd.Index(['bdg_tv', 'bdg_digital'], name='item'))


def _synthetic_ds():
    n = 12
    return pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=n, freq='MS'),
        'superbrand': ['brand_x'] * n,
        'bs_key': [1] * n,
        'listed': [4] * n,
        'tv_ots': [0.0] * n,
        'digital_imp': [0.0] * n,
    })


# ---------------------------------------------------------------------------
# TestOptGrid
# ---------------------------------------------------------------------------

class TestOptGrid:

    def test_grid_shape(self):
        td = _sample_trans_dict()
        grid = opt_grid(td, budget_step=5)
        assert isinstance(grid, pd.DataFrame)
        assert len(grid) > 0
        # Columns should match variable keys
        expected_cols = list(td['variable'].keys())
        assert all(c in grid.columns for c in expected_cols)

    def test_grid_budget_step_scaling(self):
        td = _sample_trans_dict()
        grid_5 = opt_grid(td, budget_step=5)
        grid_10 = opt_grid(td, budget_step=10)
        # budget_step=10 should produce roughly half the rows
        assert len(grid_10) < len(grid_5)

    def test_grid_column_names(self):
        td = _sample_trans_dict()
        grid = opt_grid(td, budget_step=5)
        expected = {'bdg_tv', 'bdg_digital'}
        assert expected.issubset(set(grid.columns))

    def test_grid_all_values_in_bounds(self):
        td = _sample_trans_dict()
        grid = opt_grid(td, budget_step=5)
        for col in ['bdg_tv', 'bdg_digital']:
            max_b = td.loc[col, 'max_budg']
            assert (grid[col] >= 0).all()
            assert (grid[col] <= max_b).all()

    def test_grid_empty_trans_dict_raises(self):
        with pytest.raises((FermatricaError, KeyError)):
            opt_grid(pd.DataFrame(), budget_step=5)


# ---------------------------------------------------------------------------
# TestOptionReportMultiPost (via _opt_grid_summarize)
# ---------------------------------------------------------------------------

class TestOptionReportMultiPost:

    def test_grid_has_expected_structure(self):
        td = _sample_trans_dict()
        grid = opt_grid(td, budget_step=10)
        assert isinstance(grid, pd.DataFrame)
        assert 'option' in grid.columns
        assert 'bdg' in grid.columns
        bdg_cols = [c for c in grid.columns if c.startswith('bdg_')]
        assert len(bdg_cols) > 0

    def test_grid_zero_option_present(self):
        td = _sample_trans_dict()
        grid = opt_grid(td, budget_step=10)
        assert 'zero' in grid['option'].values

    @pytest.fixture
    def multi_setup(self):
        ds = _synthetic_ds()
        trans_dict = _sample_trans_dict()
        model_rep = ModelRep(ds, trans_dict=trans_dict, language='english')
        opt_set = OptionSettings(
            target='brand_x', date_start='2024-01-01', date_end='2024-12-01',
            ref_date_start='2023-01-01', ref_date_end='2023-06-01'
        )
        return ds, model_rep, opt_set, trans_dict

    def test_multi_post_returns_dataframe(self, multi_setup, monkeypatch):
        ds, model_rep, opt_set, translation = multi_setup

        mc = _MockModelConf()
        model = _MockModel(mc)

        def mock_multi_var(model, ds, model_rep, options_dict, opt_set, cores=1, if_exact=False):
            return {
                i: {
                    'pred_exact_0': {'pred_exact_vol': float(100 + i), 'pred_exact_val': float(150 + i)},
                    'pred_long_vol': float(100 + i), 'pred_long_val': float(150 + i),
                }
                for i in range(len(options_dict))
            }

        monkeypatch.setattr('fermatrica_rep.options.calc.option_report_multi_var', mock_multi_var)

        result = option_report_multi_post(
            model=model, ds=ds, model_rep=model_rep,
            opt_set=opt_set, translation=translation,
            budget_step=10, cores=1, if_exact=True,
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        expected_cols = {'pred_exact_vol', 'pred_exact_val', 'pred_long_vol', 'pred_long_val', 'bdg', 'option'}
        assert expected_cols.issubset(set(result.columns))

    def test_multi_post_zero_option_present(self, multi_setup, monkeypatch):
        ds, model_rep, opt_set, translation = multi_setup

        mc = _MockModelConf()
        model = _MockModel(mc)

        def mock_multi_var(model, ds, model_rep, options_dict, opt_set, cores=1, if_exact=False):
            return {
                i: {
                    'pred_exact_0': {'pred_exact_vol': 0.0, 'pred_exact_val': 0.0},
                    'pred_long_vol': 0.0, 'pred_long_val': 0.0,
                }
                for i in range(len(options_dict))
            }

        monkeypatch.setattr('fermatrica_rep.options.calc.option_report_multi_var', mock_multi_var)

        result = option_report_multi_post(
            model=model, ds=ds, model_rep=model_rep,
            opt_set=opt_set, translation=translation,
            budget_step=10, cores=1, if_exact=True,
        )

        assert 'zero' in result['option'].values

    def test_multi_post_finite_values(self, multi_setup, monkeypatch):
        ds, model_rep, opt_set, translation = multi_setup

        mc = _MockModelConf()
        model = _MockModel(mc)

        def mock_multi_var(model, ds, model_rep, options_dict, opt_set, cores=1, if_exact=False):
            return {
                i: {
                    'pred_exact_0': {'pred_exact_vol': float(100 + i), 'pred_exact_val': float(150 + i)},
                    'pred_long_vol': float(100 + i), 'pred_long_val': float(150 + i),
                }
                for i in range(len(options_dict))
            }

        monkeypatch.setattr('fermatrica_rep.options.calc.option_report_multi_var', mock_multi_var)

        result = option_report_multi_post(
            model=model, ds=ds, model_rep=model_rep,
            opt_set=opt_set, translation=translation,
            budget_step=10, cores=1, if_exact=True,
        )

        num_cols = result.select_dtypes(include=[np.number]).columns
        assert np.all(np.isfinite(result[num_cols].values))

    def test_multi_post_handles_empty_grid(self, multi_setup, monkeypatch):
        ds, model_rep, opt_set, translation = multi_setup

        mc = _MockModelConf()
        model = _MockModel(mc)

        empty_translation = pd.DataFrame({
            'max_budg': [0, 0],
            'function': ['basic', 'basic'],
            'variable': ['tv_ots', 'digital_imp'],
            'price': [0.05, 0.02],
        }, index=pd.Index(['bdg_tv', 'bdg_digital'], name='item'))

        # opt_grid still produces the "zero" row; mock must match the grid size
        def mock_multi_var(model, ds, model_rep, options_dict, opt_set, cores=1, if_exact=False):
            return {
                i: {
                    'pred_exact_0': {'pred_exact_vol': 0.0, 'pred_exact_val': 0.0},
                    'pred_long_vol': 0.0, 'pred_long_val': 0.0,
                }
                for i in range(len(options_dict))
            }

        monkeypatch.setattr('fermatrica_rep.options.calc.option_report_multi_var', mock_multi_var)

        result = option_report_multi_post(
            model=model, ds=ds, model_rep=model_rep,
            opt_set=opt_set, translation=empty_translation,
            budget_step=10, cores=1, if_exact=True,
        )

        assert isinstance(result, pd.DataFrame)
        # With all max_budg=0, only the "zero" row exists
        assert 'zero' in result['option'].values


class _MockModelConf:
    price_var = 'price'
    summary_type = 'sum'
    trans_path_df = pd.DataFrame({'variable': ['x1'], 'price': [1.0]})


class _MockModel:
    obj = None

    def __init__(self, conf):
        self.conf = conf
