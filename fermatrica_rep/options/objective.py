import datetime
import math
import numpy as np
import pandas as pd

from fermatrica.basics.basics import fermatrica_error
from fermatrica_utils import dict_multifill

from fermatrica_rep.meta_model.meta_model import MetaModel, _last_node_name
from fermatrica_rep.options.option import Option
from fermatrica_rep.options.scope import OptionScope


def _get_score_key(if_long: bool, if_volume: bool) -> str:
    if if_long:
        return "pred_long_vol" if if_volume else "pred_long_val"
    return "pred_exact_vol" if if_volume else "pred_exact_val"


def _compute_score(results, meta_model, budget_scope, if_long, if_volume,
                   error_score, summary_nodes=None, baseline_scores=None):
    score_key = _get_score_key(if_long, if_volume)

    if summary_nodes is None:
        last_node = _last_node_name(meta_model)
        score_val = results[last_node][budget_scope].get(score_key, 0)
        if score_val is None or math.isinf(score_val):
            return error_score
        return score_val

    combined = 0.0
    for node_name, (weight, node_scope) in summary_nodes.items():
        score_val = results[node_name][node_scope].get(score_key, 0)
        if score_val is None or math.isinf(score_val):
            return error_score
        combined += weight * score_val / baseline_scores[node_name]
    return combined


def _combine_trg(trg: float | dict[str, float], summary_nodes, baseline_scores):
    if summary_nodes is None:
        if isinstance(trg, dict):
            fermatrica_error("trg must be a float when summary_nodes is None")
        return float(trg)

    if not isinstance(trg, dict):
        fermatrica_error(
            "trg must be a dict[str, float] when summary_nodes is set. "
            "Provide per-node targets, e.g. {'sales': 200, 'awareness': 35}."
        )

    combined = 0.0
    for node_name, (weight, _node_scope) in summary_nodes.items():
        node_trg = trg.get(node_name)
        if node_trg is None:
            fermatrica_error(
                f"trg dict missing key '{node_name}'. "
                f"Required keys: {list(summary_nodes.keys())}"
            )
        if not isinstance(node_trg, (int, float)):
            fermatrica_error(f"trg['{node_name}'] must be a number, got {type(node_trg)}")

        bs = baseline_scores[node_name]
        if bs == 0 or math.isnan(bs):
            fermatrica_error(
                f"Baseline score for node '{node_name}' is zero/NaN. Got: {bs}"
            )
        combined += weight * node_trg / bs
    return combined


class ObjectiveBudgetNG:
    """
    Wrapper for Nevergrad objective function that applies a budget
    constraint penalty.

    Used with CMA-ES optimization via Nevergrad. Wraps an underlying
    ``objective_fun`` and adds a large penalty when the total budget
    deviates from ``bdg_size`` by more than 1.0 unit. This enforces the
    budget equality constraint through the objective itself rather than
    through a separate constraint mechanism.

    Parameters
    ----------
    param_names : list[str]
        Ordered list of parameter names (budget item keys).
    objective_fun : callable
        Underlying objective function with signature
        ``f(params: list[float]) -> float``. Should return the KPI
        score (higher is better); the wrapper handles sign flipping.
    bdg_size : float
        Target total budget size. The sum of all parameters must equal
        this value.
    bdg_penalty : float, optional
        Penalty multiplier applied to budget violations.
        Default ``1e6``.
    """

    def __init__(self
                    , param_names
                    , objective_fun
                    , bdg_size
                    , bdg_penalty=1e6):

        self.param_names = param_names
        self.objective_fun = objective_fun
        self.bdg_size = bdg_size
        self.bdg_penalty = bdg_penalty

    def __call__(self, **params_dict):
        params = [params_dict[par] for par in self.param_names]

        # Apply budget constraint penalty
        total = sum(params)
        budget_violation = abs(total - self.bdg_size)

        if budget_violation > 1.0:  # tolerance
            return self.objective_fun(params) + self.bdg_penalty * budget_violation

        return self.objective_fun(params)


class ObjectiveTargetNG:
    """
    Wrapper for Nevergrad objective function that minimizes budget while
    achieving a target KPI.

    Used with CMA-ES optimization via Nevergrad. Minimizes the total
    budget (sum of parameters) while penalizing deviations from the
    target KPI. The underlying ``constraint_fun`` returns
    ``abs(abs(score) - target)`` — the objective adds a large penalty
    when this exceeds the tolerance.

    Parameters
    ----------
    param_names : list[str]
        Ordered list of parameter names (budget item keys).
    constraint_fun : callable
        Constraint function with signature
        ``f(params: list[float]) -> float``. Returns the KPI violation
        as ``abs(abs(score) - target)``.
    trg : float
        Combined target value. With ``summary_nodes`` this is
        ``sum(weight * raw_target / baseline)``.
    trg_tol_ratio : float, optional
        Tolerance ratio relative to target. The KPI must be within
        ``trg_tol_ratio * trg`` of the target.
        Default ``0.0001``.
    trg_penalty : float, optional
        Penalty multiplier for KPI violations. Default ``1e6``.
    """

    def __init__(self
                    , param_names
                    , constraint_fun
                    , trg: float
                    , trg_tol_ratio: float = 0.0001
                    , trg_penalty: float = 1e6):

        self.param_names = param_names
        self.constraint_fun = constraint_fun
        self.trg = trg
        self.trg_tol = trg_tol_ratio * trg
        self.trg_penalty = trg_penalty

    def __call__(self, **params_dict):
        params = [params_dict[par] for par in self.param_names]

        # Objective: minimize budget (sum of params)
        budget = sum(params)

        # Constraint: achieve target KPI
        # constraint_fun returns abs(abs(score) - self.trg)
        kpi_violation = self.constraint_fun(params)

        if kpi_violation > self.trg_tol:
            return budget + self.trg_penalty * kpi_violation

        return budget


class VerboseNG:
    """
    Nevergrad callback to print optimization progress.

    Prints current budget, score, iteration step, and timestamp at
    regular intervals during CMA-ES optimization. Registered as a
    ``tell`` callback on the Nevergrad optimizer.

    Parameters
    ----------
    verbose_level : int, optional
        Print every Nth iteration. Default ``10``.
    """
    def __init__(self
                 , verbose_level: int = 10):

        self.verbose_level = verbose_level
        self.iter_idx = 1

    def __call__(self
                 , optimizer
                 , parametrization
                 , loss: float
                 , *args
                 , **kwargs):

        if self.iter_idx % self.verbose_level == 0:

            score_print = -(round(loss, 5) if abs(loss) < 500 else round(loss, 0))
            params_cur = [x[1].args[0] for x in parametrization[1].items()]

            print('Total budget current : ' + str(sum(params_cur)))
            print('Scoring : ' + str(score_print) + ' ; step : ' + str(
                self.iter_idx) + ' : ' + datetime.datetime.now().isoformat(sep=" ", timespec="seconds"))

        self.iter_idx += 1


class ObjectiveFunctionBudget:
    """
    Objective function callable class for budget optimization with NLopt
    COBYLA.

    Evaluates a budget allocation by running it through the MetaModel,
    computing the predicted KPI, and returning the negative KPI (for
    minimization). A new model evaluation is executed each time the
    optimizer evaluates a parameter vector.

    Parameters
    ----------
    meta_model : MetaModel
        MetaModel instance containing the DAG of ModelNodes.
    scopes : list[OptionScope]
        List of OptionScope objects defining target periods and entities.
    borders : pd.DataFrame
        DataFrame of lower and upper bounds for each variable to vary.
        Expected columns correspond to budget item names.
    option_template : dict[str, float]
        Base budget option as a flat dict. Optimized values are inserted
        into a copy of this template.
    budget_scope : str
        Key into ``option.budget_data`` for the budget template.
    if_exact : bool, optional
        Apply only to the specific time period without forward
        projection. Default ``False``.
    if_long : bool or None, optional
        Optimize long-term or short-term KPI. If ``None``, uses
        ``not if_exact``.
    if_volume : bool, optional
        Optimize volume (``True``) or value (``False``) KPI.
        Default ``True``.
    error_score : float, optional
        Large value returned when the model evaluation fails.
        Default ``1e+12``.
    verbose : bool, optional
        Print progress info every 10 iterations. Default ``True``.
    """

    def __init__(self
                 , meta_model: MetaModel
                 , scopes: list[OptionScope]
                 , borders: pd.DataFrame
                 , option_template: dict[str, float]
                 , budget_scope: str
                 , if_exact: bool = False
                 , if_long: bool | None = None
                 , if_volume: bool = True
                 , error_score: float = 1e+12
                 , verbose: bool = True
                 , summary_nodes: dict[str, tuple[float, str]] | None = None
                 , baseline_scores: dict[str, float] | None = None
                 , option_budget_data: dict[str, dict[str, float]] | None = None
                 ):
        """
        :param meta_model: MetaModel instance
        :param scopes: list of OptionScope objects
        :param borders: dataframe of low and upper borders for every variable to vary
        :param option_template: base budget option / scenario as flat dict. Optimized values to be inserted
        :param budget_scope: key into ``option.budget_data`` for the budget template
        :param if_exact: apply only to the specific time period, without next years
        :param if_long: optimize long or short KPI. If `None`, `if_exact` is used
        :param if_volume: optimize volume or value KPI
        :param error_score: extremely big (or small) value to be used as score if fit_predict returns None (error)
        :param verbose: print diagnostic / progress info
        :param summary_nodes: dict mapping node name to ``(weight, scope)`` for multi-node scoring
        :param baseline_scores: pre-computed baseline scores per node
        :param option_budget_data: full ``option.budget_data`` preserving other scopes; required when ``summary_nodes`` is set
        """
        self.meta_model = meta_model
        self.scopes = scopes
        self.borders = borders
        self.option_template = option_template
        self.budget_scope = budget_scope
        self.if_exact = if_exact
        self.if_long = if_long if if_long is not None else not if_exact
        self.if_volume = if_volume
        self.error_score = error_score
        self.verbose = verbose
        self.iter_idx = 1
        self.summary_nodes = summary_nodes
        self._baseline_scores = baseline_scores
        self._option_budget_data = option_budget_data

        if summary_nodes is not None:
            if not summary_nodes:
                fermatrica_error("summary_nodes is empty")
            for node_name, spec in summary_nodes.items():
                if node_name not in meta_model.nodes:
                    fermatrica_error(
                        f"summary_nodes key '{node_name}' is not in meta_model.nodes. "
                        f"Available nodes: {list(meta_model.nodes.keys())}"
                    )
                if not isinstance(spec, (tuple, list)) or len(spec) != 2:
                    fermatrica_error(
                        f"summary_nodes['{node_name}'] must be a tuple of (weight, scope). "
                        f"Got: {spec}"
                    )
            if baseline_scores is None:
                fermatrica_error(
                    "baseline_scores must be provided when summary_nodes is set. "
                    "Call _compute_baseline() first."
                )
            if option_budget_data is None:
                fermatrica_error(
                    "option_budget_data must be provided when summary_nodes is set. "
                    "Pass option.budget_data to preserve all scopes."
                )
            for node_name in summary_nodes:
                bs = baseline_scores.get(node_name)
                if bs is None or (isinstance(bs, (int, float)) and (bs == 0 or math.isnan(bs))):
                    fermatrica_error(
                        f"Baseline score for node '{node_name}' is zero/NaN/None. "
                        f"Got: {bs}"
                    )

    def __call__(self
                 , params_cur: list | np.ndarray | pd.Series | None = None
                 , solution_idx=None):
        """
        Objective function with environment variables from instance.

        :param params_cur: current params vector
        :param solution_idx: not used explicitly
        :return: current score
        """

        if np.isnan(params_cur).any():
            return self.error_score

        if isinstance(params_cur, pd.Series):
            params_cur = params_cur.values

        if params_cur is not None:
            option_dict = dict_multifill(self.option_template.copy(), self.borders.columns, params_cur)
        else:
            option_dict = self.option_template

        if self.summary_nodes is not None and self._option_budget_data is not None:
            budget_data = dict(self._option_budget_data)
            budget_data[self.budget_scope] = option_dict
        else:
            budget_data = {self.budget_scope: option_dict}

        option = Option(name="optim_iter", budget_data=budget_data)

        results = self.meta_model.compute(option, scopes=self.scopes, if_exact=self.if_exact)
        score = _compute_score(results, self.meta_model, self.budget_scope, self.if_long,
                               self.if_volume, self.error_score,
                               self.summary_nodes, self._baseline_scores)

        if score == self.error_score:
            pass
        else:
            score = -score

        if self.verbose:
            if self.iter_idx % 10 == 0:
                score_print = round(abs(score), 5) if abs(score) < 500 else round(abs(score), 0)
                print('Total budget current : ' + str(sum(params_cur)))
                print('Scoring : ' + str(score_print) + ' ; step : ' + str(
                    self.iter_idx) + ' : ' + datetime.datetime.now().isoformat(sep=" ", timespec="seconds"))
            self.iter_idx += 1

        return score


class ConstraintFunctionTarget:
    """
    Inequality constraint function callable class for target optimization
    with NLopt COBYLA.

    Evaluates a budget allocation by running it through the MetaModel and
    returning ``abs(abs(score) - target)`` — the absolute distance from
    the target KPI. The COBYLA optimizer treats this as an inequality
    constraint to be kept within tolerance.

    Parameters
    ----------
    meta_model : MetaModel
        MetaModel instance containing the DAG of ModelNodes.
    scopes : list[OptionScope]
        List of OptionScope objects defining target periods and entities.
    borders : pd.DataFrame
        DataFrame of lower and upper bounds for each variable to vary.
    option_template : dict[str, float]
        Base budget option as a flat dict. Optimized values are inserted
        into a copy of this template.
    budget_scope : str
        Key into ``option.budget_data`` for the budget template.
    trg : float, optional
        Target KPI value to achieve. Default ``100``.
    if_exact : bool, optional
        Apply only to the specific time period without forward
        projection. Default ``False``.
    if_long : bool or None, optional
        Optimize long-term or short-term KPI. If ``None``, uses
        ``not if_exact``.
    if_volume : bool, optional
        Optimize volume (``True``) or value (``False``) KPI.
        Default ``True``.
    error_score : float, optional
        Large value returned when the model evaluation fails.
        Default ``1e+12``.
    verbose : bool, optional
        Print progress info every 10 iterations. Default ``True``.
    """

    def __init__(self
                 , meta_model: MetaModel
                 , scopes: list[OptionScope]
                 , borders: pd.DataFrame
                 , option_template: dict[str, float]
                 , budget_scope: str
                 , trg: float = 100
                 , if_exact: bool = False
                 , if_long: bool | None = None
                 , if_volume: bool = True
                 , error_score: float = 1e+12
                 , verbose: bool = True
                 , summary_nodes: dict[str, tuple[float, str]] | None = None
                 , baseline_scores: dict[str, float] | None = None
                 , option_budget_data: dict[str, dict[str, float]] | None = None):
        """
        :param meta_model: MetaModel instance
        :param scopes: list of OptionScope objects
        :param borders: dataframe of low and upper borders for every variable to vary
        :param option_template: base budget option / scenario as flat dict. Optimized values to be inserted
        :param budget_scope: key into ``option.budget_data`` for the budget template
        :param trg: combined target value. When ``summary_nodes`` is set this is
            ``sum(weight * raw_target / baseline)``, pre-computed by the caller
            via ``_combine_trg``.
        :param if_exact: apply only to the specific time period, without next years
        :param if_long: optimize long or short KPI. If `None`, `if_exact` is used
        :param if_volume: optimize volume or value KPI
        :param error_score: extremely big (or small) value to be used as score if fit_predict returns None (error)
        :param verbose: print diagnostic / progress info
        :param summary_nodes: dict mapping node name to ``(weight, scope)`` for multi-node scoring
        :param baseline_scores: pre-computed baseline scores per node
        :param option_budget_data: full ``option.budget_data`` preserving other scopes; required when ``summary_nodes`` is set
        """
        self.meta_model = meta_model
        self.scopes = scopes
        self.borders = borders
        self.option_template = option_template
        self.budget_scope = budget_scope
        self.trg = trg
        self.if_exact = if_exact
        self.if_long = if_long if if_long is not None else not if_exact
        self.if_volume = if_volume
        self.error_score = error_score
        self.verbose = verbose
        self.iter_idx = 1
        self.summary_nodes = summary_nodes
        self._baseline_scores = baseline_scores
        self._option_budget_data = option_budget_data

        if summary_nodes is not None:
            if not summary_nodes:
                fermatrica_error("summary_nodes is empty")
            for node_name, spec in summary_nodes.items():
                if node_name not in meta_model.nodes:
                    fermatrica_error(
                        f"summary_nodes key '{node_name}' is not in meta_model.nodes. "
                        f"Available nodes: {list(meta_model.nodes.keys())}"
                    )
                if not isinstance(spec, (tuple, list)) or len(spec) != 2:
                    fermatrica_error(
                        f"summary_nodes['{node_name}'] must be a tuple of (weight, scope). "
                        f"Got: {spec}"
                    )
            if baseline_scores is None:
                fermatrica_error(
                    "baseline_scores must be provided when summary_nodes is set. "
                    "Call _compute_baseline() first."
                )
            if option_budget_data is None:
                fermatrica_error(
                    "option_budget_data must be provided when summary_nodes is set. "
                    "Pass option.budget_data to preserve all scopes."
                )
            for node_name in summary_nodes:
                bs = baseline_scores.get(node_name)
                if bs is None or (isinstance(bs, (int, float)) and (bs == 0 or math.isnan(bs))):
                    fermatrica_error(
                        f"Baseline score for node '{node_name}' is zero/NaN/None. "
                        f"Got: {bs}"
                    )

    def __call__(self
                 , params_cur: list | np.ndarray | pd.Series | None = None
                 , solution_idx=None):
        """
        Inequality constraints function with environment variables from instance.

        :param params_cur: current params vector
        :param solution_idx: not used explicitly
        :return: current score
        """

        if np.isnan(params_cur).any():
            return self.error_score

        if isinstance(params_cur, pd.Series):
            params_cur = params_cur.values

        if params_cur is not None:
            option_dict = dict_multifill(self.option_template.copy(), self.borders.columns, params_cur)
        else:
            option_dict = self.option_template

        if self.summary_nodes is not None and self._option_budget_data is not None:
            budget_data = dict(self._option_budget_data)
            budget_data[self.budget_scope] = option_dict
        else:
            budget_data = {self.budget_scope: option_dict}

        option = Option(name="optim_iter", budget_data=budget_data)

        results = self.meta_model.compute(option, scopes=self.scopes, if_exact=self.if_exact)
        score = _compute_score(results, self.meta_model, self.budget_scope, self.if_long,
                               self.if_volume, self.error_score,
                               self.summary_nodes, self._baseline_scores)

        if score == self.error_score:
            return self.error_score

        if self.verbose:
            if self.iter_idx % 10 == 0:
                score_print = round(score, 5) if abs(score) < 500 else round(score, 0)
                print('Total budget current : ' + str(sum(params_cur)))
                print('Scoring : ' + str(score_print) + ' ; step : ' + str(
                    self.iter_idx) + ' : ' + datetime.datetime.now().isoformat(sep=" ", timespec="seconds"))
            self.iter_idx += 1

        return abs(abs(score) - self.trg)