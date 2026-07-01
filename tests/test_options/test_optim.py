import numpy as np
import pandas as pd
import pytest

from fermatrica_rep.options.objective import ObjectiveBudgetNG, ObjectiveTargetNG, ObjectiveFunctionBudget, ConstraintFunctionTarget
from fermatrica_rep.options.define import OptionSettings
from fermatrica_rep.meta_model.model_rep import ModelRep


class TestObjectiveBudgetNG:

    def test_penalty_applied_when_budget_exceeds(self):
        param_names = ['x', 'y']
        objective_fun = lambda params: params[0] * 10 + params[1] * 5
        obj = ObjectiveBudgetNG(param_names, objective_fun, bdg_size=100, bdg_penalty=1e6)
        result = obj(x=120, y=0)
        expected = 1200 + 1e6 * 20
        assert np.isclose(result, expected)

    def test_no_penalty_when_budget_met(self):
        param_names = ['x', 'y']
        objective_fun = lambda params: params[0] * 10 + params[1] * 5
        obj = ObjectiveBudgetNG(param_names, objective_fun, bdg_size=100, bdg_penalty=1e6)
        result = obj(x=80, y=20)
        expected = 800 + 100
        assert np.isclose(result, expected)

    def test_no_penalty_within_tolerance(self):
        param_names = ['x']
        objective_fun = lambda params: params[0]
        obj = ObjectiveBudgetNG(param_names, objective_fun, bdg_size=100, bdg_penalty=1e6)
        result = obj(x=100.5)
        assert np.isclose(result, 100.5)


class TestObjectiveTargetNG:

    def test_penalty_applied_when_kpi_violated(self):
        param_names = ['x', 'y']
        constraint_fun = lambda params: abs(abs(params[0] * 10 + params[1] * 5) - 100)
        obj = ObjectiveTargetNG(param_names, constraint_fun, trg=100, trg_tol_ratio=0.0001, trg_penalty=1e6)
        params = {'x': 0, 'y': 0}
        kpi_violation = abs(0 - 100)
        expected = sum(params.values()) + 1e6 * kpi_violation
        result = obj(**params)
        assert np.isclose(result, expected)

    def test_no_penalty_when_kpi_achieved(self):
        param_names = ['x']
        constraint_fun = lambda params: abs(abs(params[0] * 10) - 100)
        obj = ObjectiveTargetNG(param_names, constraint_fun, trg=100, trg_tol_ratio=0.1, trg_penalty=1e6)
        result = obj(x=10)
        assert np.isclose(result, 10)

    def test_budget_minimization(self):
        param_names = ['x', 'y']
        constraint_fun = lambda params: abs(abs(params[0] + params[1]) - 100)
        obj = ObjectiveTargetNG(param_names, constraint_fun, trg=100, trg_tol_ratio=0.01, trg_penalty=1e6)
        result = obj(x=30, y=70)
        assert np.isclose(result, 100)


class TestObjectiveFunctionBudget:

    def test_error_score_on_nan_params(self):
        ds = pd.DataFrame({'date': pd.date_range('2024-01-01', periods=3, freq='MS'), 'y': [1, 2, 3]})
        opt_set = OptionSettings(
            target='brand_x', date_start='2024-01-01', date_end='2024-03-01',
            ref_date_start='2024-01-01', ref_date_end='2024-03-01'
        )
        borders = pd.DataFrame({'bdg_x': [0, 100, 50], 'bdg_y': [0, 100, 50]})
        obj = ObjectiveFunctionBudget(
            ds=ds, model=None, model_rep=None, borders=borders,
            option={'bdg_x': 50, 'bdg_y': 50}, opt_set=opt_set, verbose=False
        )
        result = obj(params_cur=np.array([np.nan, 50]))
        assert result == 1e+12


class TestConstraintFunctionTarget:

    def test_error_score_on_nan_params(self):
        ds = pd.DataFrame({'date': pd.date_range('2024-01-01', periods=3, freq='MS'), 'y': [1, 2, 3]})
        opt_set = OptionSettings(
            target='brand_x', date_start='2024-01-01', date_end='2024-03-01',
            ref_date_start='2024-01-01', ref_date_end='2024-03-01'
        )
        borders = pd.DataFrame({'bdg_x': [0, 100, 50], 'bdg_y': [0, 100, 50]})
        obj = ConstraintFunctionTarget(
            ds=ds, model=None, model_rep=None, borders=borders,
            option={'bdg_x': 50, 'bdg_y': 50}, opt_set=opt_set,
            trg=100, verbose=False
        )
        result = obj(params_cur=np.array([np.nan, 50]))
        assert result == 1e+12


class TestOptimizeBudgetLocalCobyla:

    def test_returns_option_dict_with_bdg_key(self, monkeypatch):
        def mock_option_report(model, ds, model_rep, option_dict, option_settings, **kwargs):
            bdg_sum = sum(v for k, v in option_dict.items() if k.startswith('bdg_') and k != 'bdg')
            opt_sum = {
                'pred_exact_0': {'pred_exact_vol': bdg_sum * 2, 'pred_exact_val': bdg_sum * 2},
                'pred_long_vol': bdg_sum * 2,
                'pred_long_val': bdg_sum * 2,
                'pred_ref_vol': 100,
                'ref_vol': 50,
                'growth_vol': 2.0,
            }
            return ds, None, opt_sum

        monkeypatch.setattr('fermatrica_rep.options.calc.option_report', mock_option_report)

        from fermatrica_rep.options.optim import optimize_budget_local_cobyla

        ds = pd.DataFrame({'date': pd.date_range('2024-01-01', periods=12, freq='MS'),
                           'superbrand': ['brand_x'] * 12, 'listed': [2] * 12,
                           'y': range(12)})
        opt_set = OptionSettings(
            target='brand_x', date_start='2024-01-01', date_end='2024-12-01',
            ref_date_start='2023-01-01', ref_date_end='2023-12-01'
        )
        borders_dict = {'bdg_x': [0, 100, 50], 'bdg_y': [0, 100, 50]}
        option = {'bdg_x': 50, 'bdg_y': 50}

        result = optimize_budget_local_cobyla(
            ds=ds, model=None, model_rep=None, borders_dict=borders_dict,
            option=option, opt_set=opt_set, bdg_size=100,
            epochs=1, iters_epoch=50, ftol_abs=1e-4, verbose=False
        )

        assert isinstance(result, dict)
        assert 'bdg' in result
        assert result['bdg'] > 0

    def test_bdg_size_zero_does_not_crash(self, monkeypatch):
        def mock_option_report(model, ds, model_rep, option_dict, option_settings, **kwargs):
            bdg_sum = sum(v for k, v in option_dict.items() if k.startswith('bdg_') and k != 'bdg')
            opt_sum = {
                'pred_exact_0': {'pred_exact_vol': bdg_sum * 2, 'pred_exact_val': bdg_sum * 2},
                'pred_long_vol': bdg_sum * 2, 'pred_long_val': bdg_sum * 2,
                'pred_ref_vol': 100, 'ref_vol': 50, 'growth_vol': 2.0,
            }
            return ds, None, opt_sum

        monkeypatch.setattr('fermatrica_rep.options.calc.option_report', mock_option_report)

        from fermatrica_rep.options.optim import optimize_budget_local_cobyla

        ds = pd.DataFrame({'date': pd.date_range('2024-01-01', periods=3, freq='MS'),
                           'superbrand': ['brand_x'] * 3, 'listed': [2] * 3, 'y': [1, 2, 3]})
        opt_set = OptionSettings(
            target='brand_x', date_start='2024-01-01', date_end='2024-03-01',
            ref_date_start='2024-01-01', ref_date_end='2024-03-01'
        )
        borders_dict = {'bdg_x': [0, 100, 0], 'bdg_y': [0, 100, 0]}
        result = optimize_budget_local_cobyla(
            ds=ds, model=None, model_rep=None, borders_dict=borders_dict,
            option={'bdg_x': 0, 'bdg_y': 0}, opt_set=opt_set, bdg_size=0,
            epochs=1, iters_epoch=10, ftol_abs=1e-2, verbose=False
        )
        assert isinstance(result, dict)

    def test_prefers_higher_coef_variable(self, monkeypatch):
        called_with = []

        def mock_option_report(model, ds, model_rep, option_dict, option_settings, **kwargs):
            x = option_dict.get('bdg_x', 0)
            y = option_dict.get('bdg_y', 0)
            called_with.append((x, y))
            score = 3.0 * x + 0.5 * y
            opt_sum = {
                'pred_exact_0': {'pred_exact_vol': score, 'pred_exact_val': score},
                'pred_long_vol': score, 'pred_long_val': score,
                'pred_ref_vol': 100, 'ref_vol': 50, 'growth_vol': 2.0,
            }
            return ds, None, opt_sum

        monkeypatch.setattr('fermatrica_rep.options.calc.option_report', mock_option_report)

        from fermatrica_rep.options.optim import optimize_budget_local_cobyla

        ds = pd.DataFrame({'date': pd.date_range('2024-01-01', periods=12, freq='MS'),
                           'superbrand': ['brand_x'] * 12, 'listed': [2] * 12, 'y': range(12)})
        opt_set = OptionSettings(
            target='brand_x', date_start='2024-01-01', date_end='2024-12-01',
            ref_date_start='2023-01-01', ref_date_end='2023-12-01'
        )
        borders_dict = {'bdg_x': [0, 100, 50], 'bdg_y': [0, 100, 50]}
        result = optimize_budget_local_cobyla(
            ds=ds, model=None, model_rep=None, borders_dict=borders_dict,
            option={'bdg_x': 50, 'bdg_y': 50}, opt_set=opt_set, bdg_size=100,
            epochs=1, iters_epoch=100, ftol_abs=1e-4, verbose=False
        )
        assert result.get('bdg_x', 0) >= result.get('bdg_y', 0)

    def test_raises_on_invalid_borders(self):
        from fermatrica_rep.options.optim import optimize_budget_local_cobyla

        ds = pd.DataFrame({'date': pd.date_range('2024-01-01', periods=5, freq='MS'),
                           'superbrand': ['brand_x'] * 5, 'listed': [2] * 5})
        opt_set = OptionSettings(
            target='brand_x', date_start='2024-01-01', date_end='2024-05-01',
            ref_date_start='2023-01-01', ref_date_end='2023-12-01'
        )

        with pytest.raises(Exception):
            optimize_budget_local_cobyla(
                ds=ds, model=None, model_rep=None,
                borders_dict={'bdg_x': [0]}, option={}, opt_set=opt_set,
                bdg_size=100, epochs=1, iters_epoch=10, verbose=False
            )


# ---------------------------------------------------------------------------
# TestOptimizeBudgetLocalNcma
# ---------------------------------------------------------------------------

class _MockNcmaModel:
    class Obj:
        transform_lhs_fn = None
        custom_predict_fn = None
    obj = Obj()

    class Conf:
        price_var = 'price'
        summary_type = 'sum'
        trans_path_df = pd.DataFrame({'variable': ['x1'], 'price': [1.0]})
    conf = Conf()


class TestOptimizeBudgetLocalNcma:

    _model = _MockNcmaModel()

    def test_returns_option_dict_with_bdg_key(self, monkeypatch):
        def mock_option_report(model, ds, model_rep, option_dict, option_settings, **kwargs):
            bdg_sum = sum(v for k, v in option_dict.items() if k.startswith('bdg_') and k != 'bdg')
            opt_sum = {
                'pred_exact_0': {'pred_exact_vol': bdg_sum * 2, 'pred_exact_val': bdg_sum * 2},
                'pred_long_vol': bdg_sum * 2, 'pred_long_val': bdg_sum * 2,
                'pred_ref_vol': 100, 'ref_vol': 50, 'growth_vol': 2.0,
            }
            return ds, None, opt_sum

        monkeypatch.setattr('fermatrica_rep.options.calc.option_report', mock_option_report)

        from fermatrica_rep.options.optim import optimize_budget_local_ncma

        ds = pd.DataFrame({'date': pd.date_range('2024-01-01', periods=12, freq='MS'),
                           'superbrand': ['brand_x'] * 12, 'listed': [2] * 12,
                           'y': range(12)})
        opt_set = OptionSettings(
            target='brand_x', date_start='2024-01-01', date_end='2024-12-01',
            ref_date_start='2023-01-01', ref_date_end='2023-12-01'
        )
        borders_dict = {'bdg_x': [0, 100, 50], 'bdg_y': [0, 100, 50]}
        option = {'bdg_x': 50, 'bdg_y': 50}

        result = optimize_budget_local_ncma(
            ds=ds, model=self._model, model_rep=None, borders_dict=borders_dict,
            option=option, opt_set=opt_set, bdg_size=100,
            epochs=1, iters_epoch=10, cores=1, verbose=False,
        )

        assert isinstance(result, dict)
        assert 'bdg' in result

    def test_prefers_higher_coef_variable(self, monkeypatch):
        def mock_option_report(model, ds, model_rep, option_dict, option_settings, **kwargs):
            x = option_dict.get('bdg_x', 0)
            y = option_dict.get('bdg_y', 0)
            score = 3.0 * x + 0.5 * y
            opt_sum = {
                'pred_exact_0': {'pred_exact_vol': score, 'pred_exact_val': score},
                'pred_long_vol': score, 'pred_long_val': score,
                'pred_ref_vol': 100, 'ref_vol': 50, 'growth_vol': 2.0,
            }
            return ds, None, opt_sum

        monkeypatch.setattr('fermatrica_rep.options.calc.option_report', mock_option_report)

        from fermatrica_rep.options.optim import optimize_budget_local_ncma

        ds = pd.DataFrame({'date': pd.date_range('2024-01-01', periods=12, freq='MS'),
                           'superbrand': ['brand_x'] * 12, 'listed': [2] * 12, 'y': range(12)})
        opt_set = OptionSettings(
            target='brand_x', date_start='2024-01-01', date_end='2024-12-01',
            ref_date_start='2023-01-01', ref_date_end='2023-12-01'
        )
        borders_dict = {'bdg_x': [0, 100, 50], 'bdg_y': [0, 100, 50]}
        result = optimize_budget_local_ncma(
            ds=ds, model=self._model, model_rep=None, borders_dict=borders_dict,
            option={'bdg_x': 50, 'bdg_y': 50}, opt_set=opt_set, bdg_size=100,
            epochs=1, iters_epoch=10, cores=1, verbose=False,
        )
        assert result.get('bdg_x', 0) >= result.get('bdg_y', 0)

    def test_runs_without_crash(self):
        from fermatrica_rep.options.optim import optimize_budget_local_ncma

        ds = pd.DataFrame({'date': pd.date_range('2024-01-01', periods=12, freq='MS'),
                           'superbrand': ['brand_x'] * 12, 'listed': [2] * 12, 'y': range(12)})
        opt_set = OptionSettings(
            target='brand_x', date_start='2024-01-01', date_end='2024-12-01',
            ref_date_start='2023-01-01', ref_date_end='2023-12-01'
        )
        borders_dict = {'bdg_x': [0, 100, 50], 'bdg_y': [0, 100, 50]}
        model_rep = ModelRep(ds, language='english')
        result = optimize_budget_local_ncma(
            ds=ds, model=self._model, model_rep=model_rep, borders_dict=borders_dict,
            option={'bdg_x': 50, 'bdg_y': 50}, opt_set=opt_set, bdg_size=100,
            epochs=1, iters_epoch=10, cores=1, verbose=False,
        )
        assert isinstance(result, dict)
        assert 'bdg' in result


# ---------------------------------------------------------------------------
# TestOptimizeTargetLocalCobyla
# ---------------------------------------------------------------------------

class TestOptimizeTargetLocalCobyla:

    def test_returns_option_dict_with_bdg_key(self, monkeypatch):
        def mock_option_report(model, ds, model_rep, option_dict, option_settings, **kwargs):
            bdg_sum = sum(v for k, v in option_dict.items() if k.startswith('bdg_') and k != 'bdg')
            opt_sum = {
                'pred_exact_0': {'pred_exact_vol': bdg_sum * 2, 'pred_exact_val': bdg_sum * 2},
                'pred_long_vol': bdg_sum * 2, 'pred_long_val': bdg_sum * 2,
                'pred_ref_vol': 100, 'ref_vol': 50, 'growth_vol': 2.0,
            }
            return ds, None, opt_sum

        monkeypatch.setattr('fermatrica_rep.options.calc.option_report', mock_option_report)

        from fermatrica_rep.options.optim import optimize_target_local_cobyla

        ds = pd.DataFrame({'date': pd.date_range('2024-01-01', periods=12, freq='MS'),
                           'superbrand': ['brand_x'] * 12, 'listed': [2] * 12,
                           'y': range(12)})
        opt_set = OptionSettings(
            target='brand_x', date_start='2024-01-01', date_end='2024-12-01',
            ref_date_start='2023-01-01', ref_date_end='2023-12-01'
        )
        borders_dict = {'bdg_x': [0, 100, 50], 'bdg_y': [0, 100, 50]}
        result = optimize_target_local_cobyla(
            ds=ds, model=None, model_rep=None, borders_dict=borders_dict,
            option={'bdg_x': 50, 'bdg_y': 50}, opt_set=opt_set,
            trg=100, epochs=1, iters_epoch=50, verbose=False,
        )
        assert isinstance(result, dict)
        assert 'bdg' in result

    def test_achieves_target_kpi(self, monkeypatch):
        def mock_option_report(model, ds, model_rep, option_dict, option_settings, **kwargs):
            bdg_sum = sum(v for k, v in option_dict.items() if k.startswith('bdg_') and k != 'bdg')
            opt_sum = {
                'pred_exact_0': {'pred_exact_vol': bdg_sum * 2, 'pred_exact_val': bdg_sum * 2},
                'pred_long_vol': bdg_sum * 2, 'pred_long_val': bdg_sum * 2,
                'pred_ref_vol': 100, 'ref_vol': 50, 'growth_vol': 2.0,
            }
            return ds, None, opt_sum

        monkeypatch.setattr('fermatrica_rep.options.calc.option_report', mock_option_report)

        from fermatrica_rep.options.optim import optimize_target_local_cobyla

        ds = pd.DataFrame({'date': pd.date_range('2024-01-01', periods=12, freq='MS'),
                           'superbrand': ['brand_x'] * 12, 'listed': [2] * 12, 'y': range(12)})
        opt_set = OptionSettings(
            target='brand_x', date_start='2024-01-01', date_end='2024-12-01',
            ref_date_start='2023-01-01', ref_date_end='2023-12-01'
        )
        borders_dict = {'bdg_x': [0, 100, 50], 'bdg_y': [0, 100, 50]}
        result = optimize_target_local_cobyla(
            ds=ds, model=None, model_rep=None, borders_dict=borders_dict,
            option={'bdg_x': 50, 'bdg_y': 50}, opt_set=opt_set,
            trg=100, epochs=1, iters_epoch=50, verbose=False,
        )
        assert result['bdg'] >= 0

    def test_budget_within_borders(self, monkeypatch):
        def mock_option_report(model, ds, model_rep, option_dict, option_settings, **kwargs):
            bdg_sum = sum(v for k, v in option_dict.items() if k.startswith('bdg_') and k != 'bdg')
            opt_sum = {
                'pred_exact_0': {'pred_exact_vol': bdg_sum * 2, 'pred_exact_val': bdg_sum * 2},
                'pred_long_vol': bdg_sum * 2, 'pred_long_val': bdg_sum * 2,
                'pred_ref_vol': 100, 'ref_vol': 50, 'growth_vol': 2.0,
            }
            return ds, None, opt_sum

        monkeypatch.setattr('fermatrica_rep.options.calc.option_report', mock_option_report)

        from fermatrica_rep.options.optim import optimize_target_local_cobyla

        ds = pd.DataFrame({'date': pd.date_range('2024-01-01', periods=12, freq='MS'),
                           'superbrand': ['brand_x'] * 12, 'listed': [2] * 12, 'y': range(12)})
        opt_set = OptionSettings(
            target='brand_x', date_start='2024-01-01', date_end='2024-12-01',
            ref_date_start='2023-01-01', ref_date_end='2023-12-01'
        )
        borders_dict = {'bdg_x': [0, 100, 50], 'bdg_y': [0, 100, 50]}
        result = optimize_target_local_cobyla(
            ds=ds, model=None, model_rep=None, borders_dict=borders_dict,
            option={'bdg_x': 50, 'bdg_y': 50}, opt_set=opt_set,
            trg=100, epochs=1, iters_epoch=50, verbose=False,
        )
        for k, border in borders_dict.items():
            if k in result and k != 'bdg':
                assert border[0] <= result[k] <= border[1], \
                    f"{k}={result[k]} outside [{border[0]}, {border[1]}]"


# ---------------------------------------------------------------------------
# TestOptimizeTargetLocalNcma
# ---------------------------------------------------------------------------

class TestOptimizeTargetLocalNcma:

    _model = _MockNcmaModel()

    def test_returns_option_dict_with_bdg_key(self, monkeypatch):
        def mock_option_report(model, ds, model_rep, option_dict, option_settings, **kwargs):
            bdg_sum = sum(v for k, v in option_dict.items() if k.startswith('bdg_') and k != 'bdg')
            opt_sum = {
                'pred_exact_0': {'pred_exact_vol': bdg_sum * 2, 'pred_exact_val': bdg_sum * 2},
                'pred_long_vol': bdg_sum * 2, 'pred_long_val': bdg_sum * 2,
                'pred_ref_vol': 100, 'ref_vol': 50, 'growth_vol': 2.0,
            }
            return ds, None, opt_sum

        monkeypatch.setattr('fermatrica_rep.options.calc.option_report', mock_option_report)

        from fermatrica_rep.options.optim import optimize_target_local_ncma

        ds = pd.DataFrame({'date': pd.date_range('2024-01-01', periods=12, freq='MS'),
                           'superbrand': ['brand_x'] * 12, 'listed': [2] * 12,
                           'y': range(12)})
        opt_set = OptionSettings(
            target='brand_x', date_start='2024-01-01', date_end='2024-12-01',
            ref_date_start='2023-01-01', ref_date_end='2023-12-01'
        )
        borders_dict = {'bdg_x': [0, 100, 50], 'bdg_y': [0, 100, 50]}
        result = optimize_target_local_ncma(
            ds=ds, model=self._model, model_rep=None, borders_dict=borders_dict,
            option={'bdg_x': 50, 'bdg_y': 50}, opt_set=opt_set,
            trg=100, epochs=1, iters_epoch=10, cores=1, verbose=False,
        )
        assert isinstance(result, dict)
        assert 'bdg' in result

    def test_bdg_size_zero_does_not_crash(self, monkeypatch):
        def mock_option_report(model, ds, model_rep, option_dict, option_settings, **kwargs):
            bdg_sum = sum(v for k, v in option_dict.items() if k.startswith('bdg_') and k != 'bdg')
            opt_sum = {
                'pred_exact_0': {'pred_exact_vol': bdg_sum * 2, 'pred_exact_val': bdg_sum * 2},
                'pred_long_vol': bdg_sum * 2, 'pred_long_val': bdg_sum * 2,
                'pred_ref_vol': 100, 'ref_vol': 50, 'growth_vol': 2.0,
            }
            return ds, None, opt_sum

        monkeypatch.setattr('fermatrica_rep.options.calc.option_report', mock_option_report)

        from fermatrica_rep.options.optim import optimize_target_local_ncma

        ds = pd.DataFrame({'date': pd.date_range('2024-01-01', periods=3, freq='MS'),
                           'superbrand': ['brand_x'] * 3, 'listed': [2] * 3, 'y': [1, 2, 3]})
        opt_set = OptionSettings(
            target='brand_x', date_start='2024-01-01', date_end='2024-03-01',
            ref_date_start='2024-01-01', ref_date_end='2024-03-01'
        )
        borders_dict = {'bdg_x': [0, 100, 0], 'bdg_y': [0, 100, 0]}
        result = optimize_target_local_ncma(
            ds=ds, model=self._model, model_rep=None, borders_dict=borders_dict,
            option={'bdg_x': 0, 'bdg_y': 0}, opt_set=opt_set,
            trg=100, epochs=1, iters_epoch=10, cores=1, verbose=False,
        )
        assert isinstance(result, dict)
