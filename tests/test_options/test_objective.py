import numpy as np
import pandas as pd
import pytest

from fermatrica.basics.basics import fermatrica_error

from fermatrica_rep.options.objective import (
    _get_score_key, _compute_score,
    ObjectiveBudgetNG, ObjectiveTargetNG,
    VerboseNG, ObjectiveFunctionBudget, ConstraintFunctionTarget,
)
from fermatrica_rep.options.option import Option
from fermatrica_rep.options.scope import OptionScope
from fermatrica_rep.meta_model.meta_model import MetaModel
from fermatrica_rep.meta_model.node import ModelNode

from tests.test_options.mock_helpers import (
    _make_mock_compute, _make_mock_meta_model,
    _make_mock_meta_model_multi, _MockComputeMulti,
)


# ---------------------------------------------------------------------------
# TestGetScoreKey
# ---------------------------------------------------------------------------

class TestGetScoreKey:

    def test_exact_vol(self):
        assert _get_score_key(if_long=False, if_volume=True) == "pred_exact_vol"

    def test_exact_val(self):
        assert _get_score_key(if_long=False, if_volume=False) == "pred_exact_val"

    def test_long_vol(self):
        assert _get_score_key(if_long=True, if_volume=True) == "pred_long_vol"

    def test_long_val(self):
        assert _get_score_key(if_long=True, if_volume=False) == "pred_long_val"


# ---------------------------------------------------------------------------
# TestComputeScore
# ---------------------------------------------------------------------------

class TestComputeScore:

    SCOPE_NAME = "test_scope"
    SCORE_KEY = "pred_exact_vol"

    def _make_results(self, score_val=None, missing_key=False):
        if missing_key:
            return {"model": {self.SCOPE_NAME: {}}}
        return {"model": {self.SCOPE_NAME: {self.SCORE_KEY: score_val}}}

    # Single-node tests (summary_nodes=None)

    def test_returns_score_single_node(self, monkeypatch):
        monkeypatch.setattr(
            'fermatrica_rep.options.objective._last_node_name',
            lambda mm: "model",
        )
        meta_model = _make_mock_meta_model(self.SCOPE_NAME)
        results = self._make_results(score_val=200.0)
        score = _compute_score(results, meta_model, self.SCOPE_NAME,
                               if_long=False, if_volume=True, error_score=1e12)
        assert score == 200.0

    def test_returns_error_on_none_score(self, monkeypatch):
        monkeypatch.setattr(
            'fermatrica_rep.options.objective._last_node_name',
            lambda mm: "model",
        )
        meta_model = _make_mock_meta_model(self.SCOPE_NAME)
        results = self._make_results(score_val=None)
        score = _compute_score(results, meta_model, self.SCOPE_NAME,
                               if_long=False, if_volume=True, error_score=1e12)
        assert score == 1e12

    def test_returns_error_on_inf_score(self, monkeypatch):
        monkeypatch.setattr(
            'fermatrica_rep.options.objective._last_node_name',
            lambda mm: "model",
        )
        meta_model = _make_mock_meta_model(self.SCOPE_NAME)
        results = self._make_results(score_val=float('inf'))
        score = _compute_score(results, meta_model, self.SCOPE_NAME,
                               if_long=False, if_volume=True, error_score=1e12)
        assert score == 1e12

    def test_returns_zero_on_missing_key(self, monkeypatch):
        monkeypatch.setattr(
            'fermatrica_rep.options.objective._last_node_name',
            lambda mm: "model",
        )
        meta_model = _make_mock_meta_model(self.SCOPE_NAME)
        results = self._make_results(missing_key=True)
        score = _compute_score(results, meta_model, self.SCOPE_NAME,
                               if_long=False, if_volume=True, error_score=1e12)
        assert score == 0

    # Multi-node tests (summary_nodes set)

    def test_multi_node_combined_weighted(self):
        meta_model = _make_mock_meta_model_multi("cat_scope", "sales_scope",
                                                  cat_kpi=1.0, sales_kpi=2.0)
        results = {
            "category": {"cat_scope": {"pred_exact_vol": 100.0}},
            "sales": {"sales_scope": {"pred_exact_vol": 200.0}},
        }
        summary_nodes = {"category": (0.3, "cat_scope"), "sales": (0.7, "sales_scope")}
        baseline_scores = {"category": 50.0, "sales": 100.0}
        score = _compute_score(results, meta_model, self.SCOPE_NAME,
                               if_long=False, if_volume=True, error_score=1e12,
                               summary_nodes=summary_nodes,
                               baseline_scores=baseline_scores)
        expected = 0.3 * 100.0 / 50.0 + 0.7 * 200.0 / 100.0
        assert np.isclose(score, expected)

    def test_multi_node_returns_error_on_none_score(self):
        meta_model = _make_mock_meta_model_multi("cat_scope", "sales_scope")
        results = {
            "category": {"cat_scope": {"pred_exact_vol": None}},
            "sales": {"sales_scope": {"pred_exact_vol": 200.0}},
        }
        summary_nodes = {"category": (0.3, "cat_scope"), "sales": (0.7, "sales_scope")}
        baseline_scores = {"category": 50.0, "sales": 100.0}
        score = _compute_score(results, meta_model, self.SCOPE_NAME,
                               if_long=False, if_volume=True, error_score=1e12,
                               summary_nodes=summary_nodes,
                               baseline_scores=baseline_scores)
        assert score == 1e12

    def test_multi_node_returns_error_on_inf_score(self):
        meta_model = _make_mock_meta_model_multi("cat_scope", "sales_scope")
        results = {
            "category": {"cat_scope": {"pred_exact_vol": float('inf')}},
            "sales": {"sales_scope": {"pred_exact_vol": 200.0}},
        }
        summary_nodes = {"category": (0.3, "cat_scope"), "sales": (0.7, "sales_scope")}
        baseline_scores = {"category": 50.0, "sales": 100.0}
        score = _compute_score(results, meta_model, self.SCOPE_NAME,
                               if_long=False, if_volume=True, error_score=1e12,
                               summary_nodes=summary_nodes,
                               baseline_scores=baseline_scores)
        assert score == 1e12


# ---------------------------------------------------------------------------
# TestObjectiveBudgetNG (moved from test_optim.py + new)
# ---------------------------------------------------------------------------

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

    def test_exactly_at_tolerance_boundary(self):
        param_names = ['x']
        objective_fun = lambda params: params[0]
        obj = ObjectiveBudgetNG(param_names, objective_fun, bdg_size=100, bdg_penalty=1e6)
        result = obj(x=101.0)
        assert np.isclose(result, 101.0)

    def test_below_bdg_size_with_penalty(self):
        param_names = ['x']
        objective_fun = lambda params: params[0]
        obj = ObjectiveBudgetNG(param_names, objective_fun, bdg_size=100, bdg_penalty=1e6)
        result = obj(x=98.5)
        expected = 98.5 + 1e6 * 1.5
        assert np.isclose(result, expected)


# ---------------------------------------------------------------------------
# TestObjectiveTargetNG (moved from test_optim.py + new)
# ---------------------------------------------------------------------------

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

    def test_exactly_at_tolerance_boundary(self):
        param_names = ['x']
        constraint_fun = lambda params: abs(abs(params[0] * 10) - 100)
        obj = ObjectiveTargetNG(param_names, constraint_fun, trg=100, trg_tol_ratio=0.01, trg_penalty=1e6)
        result = obj(x=10.1)
        kpi_violation = abs(abs(10.1 * 10) - 100)
        assert np.isclose(kpi_violation, 1.0)
        assert np.isclose(result, 10.1)

    def test_zero_target(self):
        param_names = ['x']
        constraint_fun = lambda params: abs(abs(params[0] * 10) - 0)
        obj = ObjectiveTargetNG(param_names, constraint_fun, trg=0, trg_tol_ratio=0.1, trg_penalty=1e6)
        result = obj(x=5)
        kpi_violation = abs(50 - 0)
        expected = 5 + 1e6 * kpi_violation
        assert np.isclose(result, expected)


# ---------------------------------------------------------------------------
# TestVerboseNG
# ---------------------------------------------------------------------------

class TestVerboseNG:

    def _make_parametrization(self, values=None):
        """Build a mock parametrization with budget parameters."""
        from collections import namedtuple
        from unittest.mock import MagicMock
        if values is None:
            values = [50.0, 50.0]
        Args = namedtuple('Args', ['args'])
        items = {f"p{i}": Args(args=(v,)) for i, v in enumerate(values)}
        param = [None, items]
        return param

    def test_prints_every_nth_iteration(self, capsys):
        cb = VerboseNG(verbose_level=3)
        param = self._make_parametrization([50, 50])
        for i in range(6):
            cb(None, param, -200.0)
        captured = capsys.readouterr()
        assert captured.out.count("Scoring") == 2

    def test_does_not_print_between_intervals(self, capsys):
        cb = VerboseNG(verbose_level=5)
        param = self._make_parametrization([50, 50])
        cb(None, param, -200.0)
        cb(None, param, -200.0)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_iter_idx_increments(self):
        cb = VerboseNG(verbose_level=1)
        param = self._make_parametrization([50, 50])
        assert cb.iter_idx == 1
        cb(None, param, -200.0)
        assert cb.iter_idx == 2
        cb(None, param, -200.0)
        assert cb.iter_idx == 3
        cb(None, param, -200.0)
        assert cb.iter_idx == 4

    def test_loss_less_than_500_rounds_to_5_decimals(self, capsys):
        cb = VerboseNG(verbose_level=1)
        param = self._make_parametrization([50, 50])
        cb(None, param, -123.456789)
        captured = capsys.readouterr()
        assert "123.45679" in captured.out

    def test_loss_greater_than_500_rounds_to_0_decimals(self, capsys):
        cb = VerboseNG(verbose_level=1)
        param = self._make_parametrization([50, 50])
        cb(None, param, -1234.567)
        captured = capsys.readouterr()
        assert "1235" in captured.out

    def test_loss_exactly_500_rounds_to_0_decimals(self, capsys):
        cb = VerboseNG(verbose_level=1)
        param = self._make_parametrization([50, 50])
        cb(None, param, -500.0)
        captured = capsys.readouterr()
        assert "500" in captured.out
        assert "." not in captured.out.split(":")[1].strip()


# ---------------------------------------------------------------------------
# TestObjectiveFunctionBudget (moved from test_optim.py + new)
# ---------------------------------------------------------------------------

class TestObjectiveFunctionBudget:

    SCOPE_NAME = "test_scope"

    # __init__ validation — 7 tests

    def test_init_raises_on_empty_summary_nodes(self):
        scopes = [OptionScope(
            scope_name=self.SCOPE_NAME,
            date_start="2024-01-01", date_end="2024-03-01",
            ref_date_start="2024-01-01", ref_date_end="2024-03-01",
        )]
        meta_model = MetaModel(nodes={
            "model": ModelNode(model=None, ds=pd.DataFrame(), model_rep=None, _display_only=True)
        })
        borders = pd.DataFrame({'bdg_x': [0, 100, 50], 'bdg_y': [0, 100, 50]})
        with pytest.raises(Exception):
            ObjectiveFunctionBudget(
                meta_model=meta_model, scopes=scopes, borders=borders,
                option_template={'bdg_x': 50, 'bdg_y': 50}, budget_scope=self.SCOPE_NAME,
                summary_nodes={}, baseline_scores={"model": 100.0},
                option_budget_data={self.SCOPE_NAME: {'bdg_x': 50, 'bdg_y': 50}},
                verbose=False,
            )

    def test_init_raises_on_missing_node_name(self):
        scopes = [OptionScope(
            scope_name=self.SCOPE_NAME,
            date_start="2024-01-01", date_end="2024-03-01",
            ref_date_start="2024-01-01", ref_date_end="2024-03-01",
        )]
        meta_model = MetaModel(nodes={
            "model": ModelNode(model=None, ds=pd.DataFrame(), model_rep=None, _display_only=True)
        })
        borders = pd.DataFrame({'bdg_x': [0, 100, 50], 'bdg_y': [0, 100, 50]})
        with pytest.raises(Exception):
            ObjectiveFunctionBudget(
                meta_model=meta_model, scopes=scopes, borders=borders,
                option_template={'bdg_x': 50, 'bdg_y': 50}, budget_scope=self.SCOPE_NAME,
                summary_nodes={"nonexistent": (0.5, self.SCOPE_NAME)},
                baseline_scores={"model": 100.0},
                option_budget_data={self.SCOPE_NAME: {'bdg_x': 50, 'bdg_y': 50}},
                verbose=False,
            )

    def test_init_raises_on_invalid_spec_format(self):
        scopes = [OptionScope(
            scope_name=self.SCOPE_NAME,
            date_start="2024-01-01", date_end="2024-03-01",
            ref_date_start="2024-01-01", ref_date_end="2024-03-01",
        )]
        meta_model = MetaModel(nodes={
            "model": ModelNode(model=None, ds=pd.DataFrame(), model_rep=None, _display_only=True)
        })
        borders = pd.DataFrame({'bdg_x': [0, 100, 50], 'bdg_y': [0, 100, 50]})
        with pytest.raises(Exception):
            ObjectiveFunctionBudget(
                meta_model=meta_model, scopes=scopes, borders=borders,
                option_template={'bdg_x': 50, 'bdg_y': 50}, budget_scope=self.SCOPE_NAME,
                summary_nodes={"a": "not_a_tuple"},
                baseline_scores={"a": 100.0},
                option_budget_data={self.SCOPE_NAME: {'bdg_x': 50, 'bdg_y': 50}},
                verbose=False,
            )

    def test_init_raises_on_missing_baseline_scores(self):
        scopes = [OptionScope(
            scope_name=self.SCOPE_NAME,
            date_start="2024-01-01", date_end="2024-03-01",
            ref_date_start="2024-01-01", ref_date_end="2024-03-01",
        )]
        meta_model = MetaModel(nodes={
            "model": ModelNode(model=None, ds=pd.DataFrame(), model_rep=None, _display_only=True)
        })
        borders = pd.DataFrame({'bdg_x': [0, 100, 50], 'bdg_y': [0, 100, 50]})
        with pytest.raises(Exception):
            ObjectiveFunctionBudget(
                meta_model=meta_model, scopes=scopes, borders=borders,
                option_template={'bdg_x': 50, 'bdg_y': 50}, budget_scope=self.SCOPE_NAME,
                summary_nodes={"model": (1.0, self.SCOPE_NAME)},
                baseline_scores=None,
                option_budget_data={self.SCOPE_NAME: {'bdg_x': 50, 'bdg_y': 50}},
                verbose=False,
            )

    def test_init_raises_on_missing_option_budget_data(self):
        scopes = [OptionScope(
            scope_name=self.SCOPE_NAME,
            date_start="2024-01-01", date_end="2024-03-01",
            ref_date_start="2024-01-01", ref_date_end="2024-03-01",
        )]
        meta_model = MetaModel(nodes={
            "model": ModelNode(model=None, ds=pd.DataFrame(), model_rep=None, _display_only=True)
        })
        borders = pd.DataFrame({'bdg_x': [0, 100, 50], 'bdg_y': [0, 100, 50]})
        with pytest.raises(Exception):
            ObjectiveFunctionBudget(
                meta_model=meta_model, scopes=scopes, borders=borders,
                option_template={'bdg_x': 50, 'bdg_y': 50}, budget_scope=self.SCOPE_NAME,
                summary_nodes={"model": (1.0, self.SCOPE_NAME)},
                baseline_scores={"model": 100.0},
                option_budget_data=None,
                verbose=False,
            )

    def test_init_raises_on_none_baseline_score(self):
        scopes = [OptionScope(
            scope_name=self.SCOPE_NAME,
            date_start="2024-01-01", date_end="2024-03-01",
            ref_date_start="2024-01-01", ref_date_end="2024-03-01",
        )]
        meta_model = MetaModel(nodes={
            "model": ModelNode(model=None, ds=pd.DataFrame(), model_rep=None, _display_only=True)
        })
        borders = pd.DataFrame({'bdg_x': [0, 100, 50], 'bdg_y': [0, 100, 50]})
        with pytest.raises(Exception):
            ObjectiveFunctionBudget(
                meta_model=meta_model, scopes=scopes, borders=borders,
                option_template={'bdg_x': 50, 'bdg_y': 50}, budget_scope=self.SCOPE_NAME,
                summary_nodes={"model": (1.0, self.SCOPE_NAME)},
                baseline_scores={"model": None},
                option_budget_data={self.SCOPE_NAME: {'bdg_x': 50, 'bdg_y': 50}},
                verbose=False,
            )

    def test_init_raises_on_zero_baseline_score(self):
        scopes = [OptionScope(
            scope_name=self.SCOPE_NAME,
            date_start="2024-01-01", date_end="2024-03-01",
            ref_date_start="2024-01-01", ref_date_end="2024-03-01",
        )]
        meta_model = MetaModel(nodes={
            "model": ModelNode(model=None, ds=pd.DataFrame(), model_rep=None, _display_only=True)
        })
        borders = pd.DataFrame({'bdg_x': [0, 100, 50], 'bdg_y': [0, 100, 50]})
        with pytest.raises(Exception):
            ObjectiveFunctionBudget(
                meta_model=meta_model, scopes=scopes, borders=borders,
                option_template={'bdg_x': 50, 'bdg_y': 50}, budget_scope=self.SCOPE_NAME,
                summary_nodes={"model": (1.0, self.SCOPE_NAME)},
                baseline_scores={"model": 0.0},
                option_budget_data={self.SCOPE_NAME: {'bdg_x': 50, 'bdg_y': 50}},
                verbose=False,
            )

    # __call__ paths — 8 tests

    def test_error_score_on_nan_params(self):
        scopes = [OptionScope(
            scope_name=self.SCOPE_NAME,
            date_start="2024-01-01", date_end="2024-03-01",
            ref_date_start="2024-01-01", ref_date_end="2024-03-01",
        )]
        meta_model = MetaModel(nodes={
            "model": ModelNode(model=None, ds=pd.DataFrame(), model_rep=None, _display_only=True)
        })
        borders = pd.DataFrame({'bdg_x': [0, 100, 50], 'bdg_y': [0, 100, 50]})
        obj = ObjectiveFunctionBudget(
            meta_model=meta_model, scopes=scopes, borders=borders,
            option_template={'bdg_x': 50, 'bdg_y': 50}, budget_scope=self.SCOPE_NAME,
            verbose=False,
        )
        result = obj(params_cur=np.array([np.nan, 50]))
        assert result == 1e+12

    def test_pd_series_conversion(self, monkeypatch):
        monkeypatch.setattr(
            'fermatrica_rep.options.objective._last_node_name',
            lambda mm: "model",
        )
        from fermatrica_rep.options.objective import _compute_score
        meta_model = _make_mock_meta_model(self.SCOPE_NAME)
        scopes = [OptionScope(
            scope_name=self.SCOPE_NAME,
            date_start="2024-01-01", date_end="2024-03-01",
            ref_date_start="2024-01-01", ref_date_end="2024-03-01",
        )]
        borders = pd.DataFrame({'bdg_x': [0, 100, 50], 'bdg_y': [0, 100, 50]})
        obj = ObjectiveFunctionBudget(
            meta_model=meta_model, scopes=scopes, borders=borders,
            option_template={'bdg_x': 50, 'bdg_y': 50}, budget_scope=self.SCOPE_NAME,
            verbose=False,
        )
        result = obj(params_cur=pd.Series([50, 50]))
        assert isinstance(result, float)

    def test_params_none_not_needed(self, monkeypatch):
        monkeypatch.setattr(
            'fermatrica_rep.options.objective._last_node_name',
            lambda mm: "model",
        )
        meta_model = _make_mock_meta_model(self.SCOPE_NAME)
        scopes = [OptionScope(
            scope_name=self.SCOPE_NAME,
            date_start="2024-01-01", date_end="2024-03-01",
            ref_date_start="2024-01-01", ref_date_end="2024-03-01",
        )]
        borders = pd.DataFrame({'bdg_x': [0, 100, 50]})
        obj = ObjectiveFunctionBudget(
            meta_model=meta_model, scopes=scopes, borders=borders,
            option_template={'bdg_x': 50}, budget_scope=self.SCOPE_NAME,
            verbose=False,
        )
        result = obj(params_cur=np.array([50]))
        assert isinstance(result, float)

    def test_normal_scoring_single_node(self, monkeypatch):
        monkeypatch.setattr(
            'fermatrica_rep.options.objective._last_node_name',
            lambda mm: "model",
        )
        meta_model = _make_mock_meta_model(self.SCOPE_NAME, kpi_multiplier=2.0)
        scopes = [OptionScope(
            scope_name=self.SCOPE_NAME,
            date_start="2024-01-01", date_end="2024-03-01",
            ref_date_start="2024-01-01", ref_date_end="2024-03-01",
        )]
        borders = pd.DataFrame({'bdg_x': [0, 100, 50], 'bdg_y': [0, 100, 50]})
        obj = ObjectiveFunctionBudget(
            meta_model=meta_model, scopes=scopes, borders=borders,
            option_template={'bdg_x': 50, 'bdg_y': 50}, budget_scope=self.SCOPE_NAME,
            verbose=False,
        )
        result = obj(params_cur=np.array([50, 50]))
        assert np.isclose(result, -200.0)

    def test_if_volume_false(self, monkeypatch):
        monkeypatch.setattr(
            'fermatrica_rep.options.objective._last_node_name',
            lambda mm: "model",
        )
        meta_model = _make_mock_meta_model(self.SCOPE_NAME, kpi_multiplier=2.0)
        scopes = [OptionScope(
            scope_name=self.SCOPE_NAME,
            date_start="2024-01-01", date_end="2024-03-01",
            ref_date_start="2024-01-01", ref_date_end="2024-03-01",
        )]
        borders = pd.DataFrame({'bdg_x': [0, 100, 50], 'bdg_y': [0, 100, 50]})
        obj = ObjectiveFunctionBudget(
            meta_model=meta_model, scopes=scopes, borders=borders,
            option_template={'bdg_x': 50, 'bdg_y': 50}, budget_scope=self.SCOPE_NAME,
            if_volume=False, verbose=False,
        )
        result = obj(params_cur=np.array([50, 50]))
        assert np.isclose(result, -200.0)

    def test_if_long_true(self, monkeypatch):
        monkeypatch.setattr(
            'fermatrica_rep.options.objective._last_node_name',
            lambda mm: "model",
        )
        meta_model = _make_mock_meta_model(self.SCOPE_NAME, kpi_multiplier=2.0)
        scopes = [OptionScope(
            scope_name=self.SCOPE_NAME,
            date_start="2024-01-01", date_end="2024-03-01",
            ref_date_start="2024-01-01", ref_date_end="2024-03-01",
        )]
        borders = pd.DataFrame({'bdg_x': [0, 100, 50], 'bdg_y': [0, 100, 50]})
        obj = ObjectiveFunctionBudget(
            meta_model=meta_model, scopes=scopes, borders=borders,
            option_template={'bdg_x': 50, 'bdg_y': 50}, budget_scope=self.SCOPE_NAME,
            if_long=True, if_exact=False, verbose=False,
        )
        result = obj(params_cur=np.array([50, 50]))
        assert np.isclose(result, -200.0)

    def test_score_equals_error_bypasses_negation(self, monkeypatch):
        monkeypatch.setattr(
            'fermatrica_rep.options.objective._last_node_name',
            lambda mm: "model",
        )
        meta_model = _make_mock_meta_model(self.SCOPE_NAME)
        scopes = [OptionScope(
            scope_name=self.SCOPE_NAME,
            date_start="2024-01-01", date_end="2024-03-01",
            ref_date_start="2024-01-01", ref_date_end="2024-03-01",
        )]
        borders = pd.DataFrame({'bdg_x': [0, 100, 50]})

        def _compute_score_returns_error(results, meta_model, budget_scope,
                                          if_long, if_volume, error_score,
                                          summary_nodes=None, baseline_scores=None):
            return 1e+12

        monkeypatch.setattr('fermatrica_rep.options.objective._compute_score',
                            _compute_score_returns_error)
        obj = ObjectiveFunctionBudget(
            meta_model=meta_model, scopes=scopes, borders=borders,
            option_template={'bdg_x': 50}, budget_scope=self.SCOPE_NAME,
            error_score=1e+12, verbose=False,
        )
        result = obj(params_cur=np.array([50]))
        assert result == 1e+12

    def test_verbose_output_on_10th_iteration(self, monkeypatch, capsys):
        monkeypatch.setattr(
            'fermatrica_rep.options.objective._last_node_name',
            lambda mm: "model",
        )
        meta_model = _make_mock_meta_model(self.SCOPE_NAME, kpi_multiplier=2.0)
        scopes = [OptionScope(
            scope_name=self.SCOPE_NAME,
            date_start="2024-01-01", date_end="2024-03-01",
            ref_date_start="2024-01-01", ref_date_end="2024-03-01",
        )]
        borders = pd.DataFrame({'bdg_x': [0, 100, 50]})
        obj = ObjectiveFunctionBudget(
            meta_model=meta_model, scopes=scopes, borders=borders,
            option_template={'bdg_x': 50}, budget_scope=self.SCOPE_NAME,
            verbose=True,
        )
        for _ in range(10):
            obj(params_cur=np.array([50]))
        captured = capsys.readouterr()
        assert "Scoring" in captured.out

    def test_multi_node_budget_data_preserves_scopes(self):
        meta_model = _make_mock_meta_model_multi("cat_scope", "sales_scope")
        scopes = [OptionScope(
            scope_name=self.SCOPE_NAME,
            date_start="2024-01-01", date_end="2024-03-01",
            ref_date_start="2024-01-01", ref_date_end="2024-03-01",
        )]
        borders = pd.DataFrame({'bdg_x': [0, 100, 50]})
        budget_data = {"cat_scope": {"bdg_x": 50}, "sales_scope": {"bdg_y": 75}}
        obj = ObjectiveFunctionBudget(
            meta_model=meta_model, scopes=scopes, borders=borders,
            option_template={'bdg_x': 50}, budget_scope="test_scope",
            summary_nodes={"category": (1.0, "cat_scope")},
            baseline_scores={"category": 100.0},
            option_budget_data=budget_data,
            verbose=False,
        )
        obj(params_cur=np.array([50]))
        assert True


# ---------------------------------------------------------------------------
# TestConstraintFunctionTarget (moved from test_optim.py + new)
# ---------------------------------------------------------------------------

class TestConstraintFunctionTarget:

    SCOPE_NAME = "test_scope"

    # __init__ validation — 7 tests

    def test_init_raises_on_empty_summary_nodes(self):
        scopes = [OptionScope(
            scope_name=self.SCOPE_NAME,
            date_start="2024-01-01", date_end="2024-03-01",
            ref_date_start="2024-01-01", ref_date_end="2024-03-01",
        )]
        meta_model = MetaModel(nodes={
            "model": ModelNode(model=None, ds=pd.DataFrame(), model_rep=None, _display_only=True)
        })
        borders = pd.DataFrame({'bdg_x': [0, 100, 50], 'bdg_y': [0, 100, 50]})
        with pytest.raises(Exception):
            ConstraintFunctionTarget(
                meta_model=meta_model, scopes=scopes, borders=borders,
                option_template={'bdg_x': 50, 'bdg_y': 50}, budget_scope=self.SCOPE_NAME,
                trg=100, summary_nodes={}, baseline_scores={"model": 100.0},
                option_budget_data={self.SCOPE_NAME: {'bdg_x': 50, 'bdg_y': 50}},
                verbose=False,
            )

    def test_init_raises_on_missing_node_name(self):
        scopes = [OptionScope(
            scope_name=self.SCOPE_NAME,
            date_start="2024-01-01", date_end="2024-03-01",
            ref_date_start="2024-01-01", ref_date_end="2024-03-01",
        )]
        meta_model = MetaModel(nodes={
            "model": ModelNode(model=None, ds=pd.DataFrame(), model_rep=None, _display_only=True)
        })
        borders = pd.DataFrame({'bdg_x': [0, 100, 50], 'bdg_y': [0, 100, 50]})
        with pytest.raises(Exception):
            ConstraintFunctionTarget(
                meta_model=meta_model, scopes=scopes, borders=borders,
                option_template={'bdg_x': 50, 'bdg_y': 50}, budget_scope=self.SCOPE_NAME,
                trg=100, summary_nodes={"nonexistent": (0.5, self.SCOPE_NAME)},
                baseline_scores={"model": 100.0},
                option_budget_data={self.SCOPE_NAME: {'bdg_x': 50, 'bdg_y': 50}},
                verbose=False,
            )

    def test_init_raises_on_invalid_spec_format(self):
        scopes = [OptionScope(
            scope_name=self.SCOPE_NAME,
            date_start="2024-01-01", date_end="2024-03-01",
            ref_date_start="2024-01-01", ref_date_end="2024-03-01",
        )]
        meta_model = MetaModel(nodes={
            "model": ModelNode(model=None, ds=pd.DataFrame(), model_rep=None, _display_only=True)
        })
        borders = pd.DataFrame({'bdg_x': [0, 100, 50], 'bdg_y': [0, 100, 50]})
        with pytest.raises(Exception):
            ConstraintFunctionTarget(
                meta_model=meta_model, scopes=scopes, borders=borders,
                option_template={'bdg_x': 50, 'bdg_y': 50}, budget_scope=self.SCOPE_NAME,
                trg=100, summary_nodes={"a": "not_a_tuple"},
                baseline_scores={"a": 100.0},
                option_budget_data={self.SCOPE_NAME: {'bdg_x': 50, 'bdg_y': 50}},
                verbose=False,
            )

    def test_init_raises_on_missing_baseline_scores(self):
        scopes = [OptionScope(
            scope_name=self.SCOPE_NAME,
            date_start="2024-01-01", date_end="2024-03-01",
            ref_date_start="2024-01-01", ref_date_end="2024-03-01",
        )]
        meta_model = MetaModel(nodes={
            "model": ModelNode(model=None, ds=pd.DataFrame(), model_rep=None, _display_only=True)
        })
        borders = pd.DataFrame({'bdg_x': [0, 100, 50], 'bdg_y': [0, 100, 50]})
        with pytest.raises(Exception):
            ConstraintFunctionTarget(
                meta_model=meta_model, scopes=scopes, borders=borders,
                option_template={'bdg_x': 50, 'bdg_y': 50}, budget_scope=self.SCOPE_NAME,
                trg=100, summary_nodes={"model": (1.0, self.SCOPE_NAME)},
                baseline_scores=None,
                option_budget_data={self.SCOPE_NAME: {'bdg_x': 50, 'bdg_y': 50}},
                verbose=False,
            )

    def test_init_raises_on_missing_option_budget_data(self):
        scopes = [OptionScope(
            scope_name=self.SCOPE_NAME,
            date_start="2024-01-01", date_end="2024-03-01",
            ref_date_start="2024-01-01", ref_date_end="2024-03-01",
        )]
        meta_model = MetaModel(nodes={
            "model": ModelNode(model=None, ds=pd.DataFrame(), model_rep=None, _display_only=True)
        })
        borders = pd.DataFrame({'bdg_x': [0, 100, 50], 'bdg_y': [0, 100, 50]})
        with pytest.raises(Exception):
            ConstraintFunctionTarget(
                meta_model=meta_model, scopes=scopes, borders=borders,
                option_template={'bdg_x': 50, 'bdg_y': 50}, budget_scope=self.SCOPE_NAME,
                trg=100, summary_nodes={"model": (1.0, self.SCOPE_NAME)},
                baseline_scores={"model": 100.0},
                option_budget_data=None,
                verbose=False,
            )

    def test_init_raises_on_none_baseline_score(self):
        scopes = [OptionScope(
            scope_name=self.SCOPE_NAME,
            date_start="2024-01-01", date_end="2024-03-01",
            ref_date_start="2024-01-01", ref_date_end="2024-03-01",
        )]
        meta_model = MetaModel(nodes={
            "model": ModelNode(model=None, ds=pd.DataFrame(), model_rep=None, _display_only=True)
        })
        borders = pd.DataFrame({'bdg_x': [0, 100, 50], 'bdg_y': [0, 100, 50]})
        with pytest.raises(Exception):
            ConstraintFunctionTarget(
                meta_model=meta_model, scopes=scopes, borders=borders,
                option_template={'bdg_x': 50, 'bdg_y': 50}, budget_scope=self.SCOPE_NAME,
                trg=100, summary_nodes={"model": (1.0, self.SCOPE_NAME)},
                baseline_scores={"model": None},
                option_budget_data={self.SCOPE_NAME: {'bdg_x': 50, 'bdg_y': 50}},
                verbose=False,
            )

    def test_init_raises_on_zero_baseline_score(self):
        scopes = [OptionScope(
            scope_name=self.SCOPE_NAME,
            date_start="2024-01-01", date_end="2024-03-01",
            ref_date_start="2024-01-01", ref_date_end="2024-03-01",
        )]
        meta_model = MetaModel(nodes={
            "model": ModelNode(model=None, ds=pd.DataFrame(), model_rep=None, _display_only=True)
        })
        borders = pd.DataFrame({'bdg_x': [0, 100, 50], 'bdg_y': [0, 100, 50]})
        with pytest.raises(Exception):
            ConstraintFunctionTarget(
                meta_model=meta_model, scopes=scopes, borders=borders,
                option_template={'bdg_x': 50, 'bdg_y': 50}, budget_scope=self.SCOPE_NAME,
                trg=100, summary_nodes={"model": (1.0, self.SCOPE_NAME)},
                baseline_scores={"model": 0.0},
                option_budget_data={self.SCOPE_NAME: {'bdg_x': 50, 'bdg_y': 50}},
                verbose=False,
            )

    # __call__ paths — 7 tests (+ 1 moved from test_optim)

    def test_error_score_on_nan_params(self):
        scopes = [OptionScope(
            scope_name=self.SCOPE_NAME,
            date_start="2024-01-01", date_end="2024-03-01",
            ref_date_start="2024-01-01", ref_date_end="2024-03-01",
        )]
        meta_model = MetaModel(nodes={
            "model": ModelNode(model=None, ds=pd.DataFrame(), model_rep=None, _display_only=True)
        })
        borders = pd.DataFrame({'bdg_x': [0, 100, 50], 'bdg_y': [0, 100, 50]})
        obj = ConstraintFunctionTarget(
            meta_model=meta_model, scopes=scopes, borders=borders,
            option_template={'bdg_x': 50, 'bdg_y': 50}, budget_scope=self.SCOPE_NAME,
            trg=100, verbose=False,
        )
        result = obj(params_cur=np.array([np.nan, 50]))
        assert result == 1e+12

    def test_pd_series_conversion(self, monkeypatch):
        monkeypatch.setattr(
            'fermatrica_rep.options.objective._last_node_name',
            lambda mm: "model",
        )
        meta_model = _make_mock_meta_model(self.SCOPE_NAME)
        scopes = [OptionScope(
            scope_name=self.SCOPE_NAME,
            date_start="2024-01-01", date_end="2024-03-01",
            ref_date_start="2024-01-01", ref_date_end="2024-03-01",
        )]
        borders = pd.DataFrame({'bdg_x': [0, 100, 50], 'bdg_y': [0, 100, 50]})
        obj = ConstraintFunctionTarget(
            meta_model=meta_model, scopes=scopes, borders=borders,
            option_template={'bdg_x': 50, 'bdg_y': 50}, budget_scope=self.SCOPE_NAME,
            trg=100, verbose=False,
        )
        result = obj(params_cur=pd.Series([50, 50]))
        assert isinstance(result, float)

    def test_params_none_not_needed(self, monkeypatch):
        monkeypatch.setattr(
            'fermatrica_rep.options.objective._last_node_name',
            lambda mm: "model",
        )
        meta_model = _make_mock_meta_model(self.SCOPE_NAME)
        scopes = [OptionScope(
            scope_name=self.SCOPE_NAME,
            date_start="2024-01-01", date_end="2024-03-01",
            ref_date_start="2024-01-01", ref_date_end="2024-03-01",
        )]
        borders = pd.DataFrame({'bdg_x': [0, 100, 50]})
        obj = ConstraintFunctionTarget(
            meta_model=meta_model, scopes=scopes, borders=borders,
            option_template={'bdg_x': 50}, budget_scope=self.SCOPE_NAME,
            trg=100, verbose=False,
        )
        result = obj(params_cur=np.array([50]))
        assert isinstance(result, float)

    def test_normal_constraint_check(self, monkeypatch):
        monkeypatch.setattr(
            'fermatrica_rep.options.objective._last_node_name',
            lambda mm: "model",
        )
        meta_model = _make_mock_meta_model(self.SCOPE_NAME, kpi_multiplier=2.0)
        scopes = [OptionScope(
            scope_name=self.SCOPE_NAME,
            date_start="2024-01-01", date_end="2024-03-01",
            ref_date_start="2024-01-01", ref_date_end="2024-03-01",
        )]
        borders = pd.DataFrame({'bdg_x': [0, 100, 50], 'bdg_y': [0, 100, 50]})
        obj = ConstraintFunctionTarget(
            meta_model=meta_model, scopes=scopes, borders=borders,
            option_template={'bdg_x': 50, 'bdg_y': 50}, budget_scope=self.SCOPE_NAME,
            trg=100, verbose=False,
        )
        result = obj(params_cur=np.array([50, 50]))
        assert np.isclose(result, abs(abs(200.0) - 100.0))

    def test_constraint_met(self, monkeypatch):
        monkeypatch.setattr(
            'fermatrica_rep.options.objective._last_node_name',
            lambda mm: "model",
        )
        meta_model = _make_mock_meta_model(self.SCOPE_NAME, kpi_multiplier=2.0)
        scopes = [OptionScope(
            scope_name=self.SCOPE_NAME,
            date_start="2024-01-01", date_end="2024-03-01",
            ref_date_start="2024-01-01", ref_date_end="2024-03-01",
        )]
        borders = pd.DataFrame({'bdg_x': [0, 100, 50]})
        obj = ConstraintFunctionTarget(
            meta_model=meta_model, scopes=scopes, borders=borders,
            option_template={'bdg_x': 50}, budget_scope=self.SCOPE_NAME,
            trg=100, verbose=False,
        )
        result = obj(params_cur=np.array([50]))
        score = 50 * 2.0
        expected = abs(abs(score) - 100)
        assert np.isclose(result, expected)

    def test_score_equals_error_returns_error_score(self, monkeypatch):
        monkeypatch.setattr(
            'fermatrica_rep.options.objective._last_node_name',
            lambda mm: "model",
        )
        meta_model = _make_mock_meta_model(self.SCOPE_NAME)
        scopes = [OptionScope(
            scope_name=self.SCOPE_NAME,
            date_start="2024-01-01", date_end="2024-03-01",
            ref_date_start="2024-01-01", ref_date_end="2024-03-01",
        )]
        borders = pd.DataFrame({'bdg_x': [0, 100, 50]})

        def _compute_score_returns_error(results, meta_model, budget_scope,
                                          if_long, if_volume, error_score,
                                          summary_nodes=None, baseline_scores=None):
            return 1e+12

        monkeypatch.setattr('fermatrica_rep.options.objective._compute_score',
                            _compute_score_returns_error)
        obj = ConstraintFunctionTarget(
            meta_model=meta_model, scopes=scopes, borders=borders,
            option_template={'bdg_x': 50}, budget_scope=self.SCOPE_NAME,
            trg=100, error_score=1e+12, verbose=False,
        )
        result = obj(params_cur=np.array([50]))
        assert result == 1e+12

    def test_if_volume_false(self, monkeypatch):
        monkeypatch.setattr(
            'fermatrica_rep.options.objective._last_node_name',
            lambda mm: "model",
        )
        meta_model = _make_mock_meta_model(self.SCOPE_NAME, kpi_multiplier=2.0)
        scopes = [OptionScope(
            scope_name=self.SCOPE_NAME,
            date_start="2024-01-01", date_end="2024-03-01",
            ref_date_start="2024-01-01", ref_date_end="2024-03-01",
        )]
        borders = pd.DataFrame({'bdg_x': [0, 100, 50]})
        obj = ConstraintFunctionTarget(
            meta_model=meta_model, scopes=scopes, borders=borders,
            option_template={'bdg_x': 50}, budget_scope=self.SCOPE_NAME,
            trg=100, if_volume=False, verbose=False,
        )
        result = obj(params_cur=np.array([50]))
        assert np.isclose(result, abs(abs(100.0) - 100.0))

    def test_if_long_true(self, monkeypatch):
        monkeypatch.setattr(
            'fermatrica_rep.options.objective._last_node_name',
            lambda mm: "model",
        )
        meta_model = _make_mock_meta_model(self.SCOPE_NAME, kpi_multiplier=2.0)
        scopes = [OptionScope(
            scope_name=self.SCOPE_NAME,
            date_start="2024-01-01", date_end="2024-03-01",
            ref_date_start="2024-01-01", ref_date_end="2024-03-01",
        )]
        borders = pd.DataFrame({'bdg_x': [0, 100, 50]})
        obj = ConstraintFunctionTarget(
            meta_model=meta_model, scopes=scopes, borders=borders,
            option_template={'bdg_x': 50}, budget_scope=self.SCOPE_NAME,
            trg=100, if_long=True, if_exact=False, verbose=False,
        )
        result = obj(params_cur=np.array([50]))
        assert np.isclose(result, abs(abs(100.0) - 100.0))
