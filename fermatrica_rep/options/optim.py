"""
Algorithmic optimization of the budget split: either maximising KPI with
known budget or minimising budget with known KPI.

As for now COBYLA constrained optimization by local approximations is used
as fairly well suited for optimization with complex and not known in advance
model structure.

Beware! Multiprocessing calculations are used.
"""


import copy
import logging
import math
import warnings
from contextlib import nullcontext

import pandas as pd
import numpy as np

import multiprocessing
import pickle
import nlopt
import nevergrad as ng
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor

from fermatrica_utils import dict_multifill
from fermatrica.basics.basics import fermatrica_error

from fermatrica_rep.meta_model.meta_model import MetaModel
from fermatrica_rep.meta_model.node import ModelNode
from fermatrica_rep.options.scope import OptionScope
from fermatrica_rep.options.option import Option
from fermatrica_rep.options.parallel import _strip_meta_model_callables
from fermatrica_rep.options.objective import (
    ConstraintFunctionTarget, ObjectiveBudgetNG, ObjectiveFunctionBudget,
    ObjectiveTargetNG, VerboseNG, _combine_trg, _get_score_key,
)


def _prepare_borders(
    borders_dict: dict,
    bdg_size: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, float | None]:
    """
    Split borders into variable/fixed and optionally adjust bdg_size.

    Returns (variable_borders, fixed_borders, adjusted_bdg_size).

    When bdg_size is None (target functions), skips budget adjustment.
    When bdg_size is a float (budget functions), subtracts fixed totals
    and rescales start values.
    """
    borders = pd.DataFrame(borders_dict).astype(float)

    borders_fixed = borders[list(borders.columns[borders.nunique() == 1].values)]
    borders = borders[list(borders.columns[borders.nunique() != 1].values)]

    if bdg_size is not None:
        bdg_size -= borders_fixed.iloc[2].sum()

        if bdg_size <= borders.iloc[2].sum():
            borders.iloc[2] = borders.iloc[2] * bdg_size / borders.iloc[2].sum() * 0.9

        tmp = borders.iloc[2]
        for i in tmp.index:
            if tmp[i] < borders.iloc[0][i]:
                tmp[i] = borders.iloc[0][i]
        borders.iloc[2] = tmp

    return borders, borders_fixed, bdg_size


def _compute_baseline(meta_model, option_budget_data, budget_scope, scopes,
                      if_exact, if_long, if_volume, summary_nodes):
    """Deep-copy ds, compute baseline scores, restore ds."""
    original_ds = {name: copy.deepcopy(node.ds) for name, node in meta_model.nodes.items()}
    try:
        baseline_budget_data = dict(option_budget_data)
        baseline_option = Option(name="baseline", budget_data=baseline_budget_data)
        results = meta_model.compute(baseline_option, scopes=scopes, if_exact=if_exact)
        baseline_scores = {}
        score_key = _get_score_key(if_long, if_volume)
        for node_name in summary_nodes:
            _weight, node_scope = summary_nodes[node_name]
            score = results[node_name][node_scope].get(score_key, 0)
            baseline_scores[node_name] = score
    finally:
        for name, node in meta_model.nodes.items():
            node.ds = original_ds[name]
    return baseline_scores


def _optimize_budget_local_cobyla_impl(
    meta_model: MetaModel,
    scopes: list[OptionScope],
    budget_scope: str,
    borders_dict: dict,
    option: Option,
    bdg_size: float = 100,
    if_exact: bool = False,
    if_long: bool | None = None,
    if_volume: bool = True,
    epochs: int = 3,
    iters_epoch: int = 300,
    error_score: float = 1e+12,
    ftol_abs: float = .01,
    xtol_abs: float | None = None,
    verbose: bool = True,
    summary_nodes: dict[str, tuple[float, str]] | None = None,
) -> Option:

    if if_long is None:
        if_long = not if_exact

    option_template = dict(option.budget_data.get(budget_scope, {}))
    if not option_template:
        fermatrica_error(
            f"budget_scope '{budget_scope}' not found in option.budget_data. "
            f"Available scopes: {list(option.budget_data.keys())}"
        )

    borders, borders_fixed, bdg_size = _prepare_borders(borders_dict, bdg_size)

    option_template = dict_multifill(option_template, borders_fixed.columns, borders_fixed.iloc[2])

    baseline_scores = None
    if summary_nodes is not None:
        baseline_scores = _compute_baseline(meta_model, option.budget_data, budget_scope,
                                            scopes, if_exact, if_long, if_volume,
                                            summary_nodes)
        for node_name in summary_nodes:
            score = baseline_scores.get(node_name)
            if score is None or (isinstance(score, (int, float)) and (score == 0 or math.isnan(score))):
                fermatrica_error(f"Baseline score is zero/NaN for node '{node_name}'")

    for i in range(epochs):

        objective_fun = ObjectiveFunctionBudget(meta_model=meta_model
                                               , scopes=scopes
                                               , budget_scope=budget_scope
                                               , option_template=option_template
                                               , if_exact=if_exact
                                               , if_long=if_long
                                               , if_volume=if_volume
                                               , verbose=verbose
                                               , error_score=error_score
                                               , borders=borders
                                               , summary_nodes=summary_nodes
                                               , baseline_scores=baseline_scores
                                               , option_budget_data=option.budget_data)

        start_values = borders.iloc[2].values

        opt_obj = nlopt.opt(nlopt.LN_COBYLA, len(start_values))

        opt_obj.set_min_objective(objective_fun)
        opt_obj.set_lower_bounds(borders.iloc[0].values)
        opt_obj.set_upper_bounds(borders.iloc[1].values)

        def constraint(params_cur: list | np.ndarray | pd.Series | None = None
                       , solution_idx=None):
            return sum(params_cur) - bdg_size

        opt_obj.add_inequality_constraint(constraint, 1)

        opt_obj.set_ftol_abs(ftol_abs)
        if xtol_abs is not None:
            opt_obj.set_xtol_abs(xtol_abs)

        opt_obj.set_maxeval(iters_epoch)
        opt_obj.set_maxtime(0)

        try:
            rtrn = opt_obj.optimize(start_values)
        except (nlopt.RoundoffLimited, ValueError):
            rtrn = start_values
            logging.warning("Epoche terminated unsuccessfully due to ROUNDOFF error")

        print('Epoche ' + str(i + 1) + ' finished')

        if (isinstance(rtrn, np.ndarray)) or (isinstance(rtrn, list)):
            option_template = dict_multifill(option_template, list(borders.keys()), rtrn)
            borders.iloc[2] = rtrn
        else:
            logging.warning("Epoche did not terminate successfully")

    return option.replace_budget(budget_scope, option_template)


def _optimize_budget_local_ncma_impl(
    meta_model: MetaModel,
    scopes: list[OptionScope],
    budget_scope: str,
    borders_dict: dict,
    option: Option,
    bdg_size: float = 100,
    if_exact: bool = False,
    if_long: bool | None = None,
    if_volume: bool = True,
    epochs: int = 1,
    iters_epoch: int = 1000,
    error_score: float = 1e+12,
    cores: int = 4,
    verbose: bool = True,
    summary_nodes: dict[str, tuple[float, str]] | None = None,
) -> Option:

    if if_long is None:
        if_long = not if_exact

    option_template = dict(option.budget_data.get(budget_scope, {}))
    if not option_template:
        fermatrica_error(
            f"budget_scope '{budget_scope}' not found in option.budget_data. "
            f"Available scopes: {list(option.budget_data.keys())}"
        )

    borders, borders_fixed, bdg_size = _prepare_borders(borders_dict, bdg_size)

    option_template = dict_multifill(option_template, borders_fixed.columns, borders_fixed.iloc[2])

    baseline_scores = None
    if summary_nodes is not None:
        baseline_scores = _compute_baseline(meta_model, option.budget_data, budget_scope,
                                            scopes, if_exact, if_long, if_volume,
                                            summary_nodes)
        for node_name in summary_nodes:
            score = baseline_scores.get(node_name)
            if score is None or (isinstance(score, (int, float)) and (score == 0 or math.isnan(score))):
                fermatrica_error(f"Baseline score is zero/NaN for node '{node_name}'")

    if cores > (cpu_count() - 1):
        cores = cpu_count() - 1
        logging.warning('Cores number for parallel computing is set too high for this computer. '
                        f'Reset to {cores}')

    param_names = list(borders.columns)
    lwr = borders.iloc[0].values
    upr = borders.iloc[1].values
    start_values = borders.iloc[2].values

    lwr = np.nan_to_num(lwr, nan=-1e+8)
    upr = np.nan_to_num(upr, nan=1e+8)

    meta_model_iter = _strip_meta_model_callables(meta_model)

    executor_ctx = nullcontext(None) if cores <= 1 else ProcessPoolExecutor(max_workers=cores)

    with executor_ctx as executor:

        for epoch in range(epochs):

            objective_fun = ObjectiveFunctionBudget(meta_model=meta_model_iter
                                                   , scopes=scopes
                                                   , budget_scope=budget_scope
                                                   , option_template=option_template
                                                   , if_exact=if_exact
                                                   , if_long=if_long
                                                   , if_volume=if_volume
                                                   , verbose=False
                                                   , error_score=error_score
                                                   , borders=borders
                                                   , summary_nodes=summary_nodes
                                                   , baseline_scores=baseline_scores
                                                   , option_budget_data=option.budget_data
                                                   )

            search_space = ng.p.Instrumentation(**{
                name: ng.p.Scalar(lower=low, upper=high, init=val)
                for name, low, high, val in zip(param_names, lwr, upr, start_values)
            })

            objective_ng = ObjectiveBudgetNG(param_names=param_names
                                            , objective_fun=objective_fun
                                            , bdg_size=bdg_size)

            optimizer = ng.optimizers.CMA(
                parametrization=search_space,
                budget=iters_epoch,
                num_workers=cores,
            )

            if verbose:
                verbose_cb = VerboseNG(10)
                optimizer.register_callback('tell', verbose_cb)

            try:
                rtrn = optimizer.minimize(objective_ng
                                        , executor=executor
                                        , verbosity=0)
                best_params = rtrn.value

            except Exception as e:
                logging.warning(f"Epoch {epoch + 1} did not terminate successfully: {e}")
                best_params = start_values
                print('Epoch ' + str(epoch + 1) + ' finished')
                continue

            print('Epoch ' + str(epoch + 1) + ' finished')

            if isinstance(best_params, tuple):
                params = [best_params[1][par] for par in param_names]

                total = sum(params)
                if total > 0:
                    params = [p * bdg_size / total for p in params]

                option_template = dict_multifill(option_template, param_names, params)
                borders.iloc[2] = params
                start_values = np.array(params)
            else:
                logging.warning("Epoch did not terminate successfully")

    return option.replace_budget(budget_scope, option_template)


def _optimize_target_local_cobyla_impl(
    meta_model: MetaModel,
    scopes: list[OptionScope],
    budget_scope: str,
    borders_dict: dict,
    option: Option,
    trg: float | dict[str, float] = 100,
    trg_tol_ratio: float = 0.0001,
    if_exact: bool = False,
    if_long: bool | None = None,
    if_volume: bool = True,
    epochs: int = 3,
    iters_epoch: int = 300,
    error_score: float = 1e+12,
    ftol_abs: float = .01,
    xtol_abs: float | None = None,
    verbose: bool = True,
    summary_nodes: dict[str, tuple[float, str]] | None = None,
) -> Option:

    if if_long is None:
        if_long = not if_exact

    option_template = dict(option.budget_data.get(budget_scope, {}))
    if not option_template:
        fermatrica_error(
            f"budget_scope '{budget_scope}' not found in option.budget_data. "
            f"Available scopes: {list(option.budget_data.keys())}"
        )

    borders, borders_fixed, _ = _prepare_borders(borders_dict)

    option_template = dict_multifill(option_template, borders_fixed.columns, borders_fixed.iloc[2])

    baseline_scores = None
    if summary_nodes is not None:
        baseline_scores = _compute_baseline(meta_model, option.budget_data, budget_scope,
                                            scopes, if_exact, if_long, if_volume,
                                            summary_nodes)
        for node_name in summary_nodes:
            score = baseline_scores.get(node_name)
            if score is None or (isinstance(score, (int, float)) and (score == 0 or math.isnan(score))):
                fermatrica_error(f"Baseline score is zero/NaN for node '{node_name}'")

    combined_trg = _combine_trg(trg, summary_nodes, baseline_scores)

    for i in range(epochs):

        def objective_fun(params_cur: list | np.ndarray | pd.Series | None = None
                          , solution_idx=None):
            return sum(params_cur)

        constraint_fun = ConstraintFunctionTarget(meta_model=meta_model
                                                 , scopes=scopes
                                                 , budget_scope=budget_scope
                                                 , option_template=option_template
                                                 , if_exact=if_exact
                                                 , if_long=if_long
                                                 , if_volume=if_volume
                                                 , error_score=error_score
                                                 , verbose=verbose
                                                 , borders=borders
                                                 , trg=combined_trg
                                                 , summary_nodes=summary_nodes
                                                 , baseline_scores=baseline_scores
                                                 , option_budget_data=option.budget_data)

        start_values = borders.iloc[2].values

        opt_obj = nlopt.opt(nlopt.LN_COBYLA, len(start_values))

        opt_obj.set_min_objective(objective_fun)
        opt_obj.set_lower_bounds(borders.iloc[0].values)
        opt_obj.set_upper_bounds(borders.iloc[1].values)

        opt_obj.add_inequality_constraint(constraint_fun, trg_tol_ratio * combined_trg)

        opt_obj.set_ftol_abs(ftol_abs)
        if xtol_abs is not None:
            opt_obj.set_xtol_abs(xtol_abs)

        opt_obj.set_maxeval(iters_epoch)
        opt_obj.set_maxtime(0)

        try:
            rtrn = opt_obj.optimize(start_values)
        except (nlopt.RoundoffLimited, ValueError):
            rtrn = start_values
            logging.warning("Epoche terminated unsuccessfully due to ROUNDOFF error")

        print('Epoche ' + str(i + 1) + ' finished')

        if (isinstance(rtrn, np.ndarray)) or (isinstance(rtrn, list)):
            option_template = dict_multifill(option_template, list(borders.keys()), rtrn)
            borders.iloc[2] = rtrn
        else:
            logging.warning("Epoche did not terminate successfully")

    return option.replace_budget(budget_scope, option_template)


def _optimize_target_local_ncma_impl(
    meta_model: MetaModel,
    scopes: list[OptionScope],
    budget_scope: str,
    borders_dict: dict,
    option: Option,
    trg: float | dict[str, float] = 100,
    trg_tol_ratio: float = 0.0001,
    if_exact: bool = False,
    if_long: bool | None = None,
    if_volume: bool = True,
    epochs: int = 1,
    iters_epoch: int = 1000,
    error_score: float = 1e+12,
    cores: int = 4,
    verbose: bool = True,
    summary_nodes: dict[str, tuple[float, str]] | None = None,
) -> Option:

    if if_long is None:
        if_long = not if_exact

    option_template = dict(option.budget_data.get(budget_scope, {}))
    if not option_template:
        fermatrica_error(
            f"budget_scope '{budget_scope}' not found in option.budget_data. "
            f"Available scopes: {list(option.budget_data.keys())}"
        )

    borders, borders_fixed, _ = _prepare_borders(borders_dict)

    option_template = dict_multifill(option_template, borders_fixed.columns, borders_fixed.iloc[2])

    baseline_scores = None
    if summary_nodes is not None:
        baseline_scores = _compute_baseline(meta_model, option.budget_data, budget_scope,
                                            scopes, if_exact, if_long, if_volume,
                                            summary_nodes)
        for node_name in summary_nodes:
            score = baseline_scores.get(node_name)
            if score is None or (isinstance(score, (int, float)) and (score == 0 or math.isnan(score))):
                fermatrica_error(f"Baseline score is zero/NaN for node '{node_name}'")

    combined_trg = _combine_trg(trg, summary_nodes, baseline_scores)

    if cores > (cpu_count() - 1):
        cores = cpu_count() - 1
        logging.warning('Cores number for parallel computing is set too high for this computer. '
                        f'Reset to {cores}')

    param_names = list(borders.columns)
    lwr = borders.iloc[0].values
    upr = borders.iloc[1].values
    start_values = borders.iloc[2].values

    lwr = np.nan_to_num(lwr, nan=-1e+8)
    upr = np.nan_to_num(upr, nan=1e+8)

    meta_model_iter = _strip_meta_model_callables(meta_model)

    executor_ctx = nullcontext(None) if cores <= 1 else ProcessPoolExecutor(max_workers=cores)

    with executor_ctx as executor:

        for epoch in range(epochs):

            constraint_fun = ConstraintFunctionTarget(meta_model=meta_model_iter
                                                     , scopes=scopes
                                                     , budget_scope=budget_scope
                                                     , option_template=option_template
                                                     , if_exact=if_exact
                                                     , if_long=if_long
                                                     , if_volume=if_volume
                                                     , error_score=error_score
                                                     , verbose=False
                                                     , borders=borders
                                                     , trg=combined_trg
                                                     , summary_nodes=summary_nodes
                                                     , baseline_scores=baseline_scores
                                                     , option_budget_data=option.budget_data)

            search_space = ng.p.Instrumentation(**{
                name: ng.p.Scalar(lower=low, upper=high, init=val)
                for name, low, high, val in zip(param_names, lwr, upr, start_values)
            })

            objective_ng = ObjectiveTargetNG(param_names=param_names
                                            , constraint_fun=constraint_fun
                                            , trg=combined_trg
                                            , trg_tol_ratio=trg_tol_ratio)

            optimizer = ng.optimizers.CMA(
                parametrization=search_space,
                budget=iters_epoch,
                num_workers=cores,
            )

            if verbose:
                verbose_cb = VerboseNG(10)
                optimizer.register_callback('tell', verbose_cb)

            try:
                rtrn = optimizer.minimize(objective_ng
                                        , executor=executor
                                        , verbosity=0)
                best_params = rtrn.value

            except Exception as e:
                logging.warning(f"Epoch {epoch + 1} did not terminate successfully: {e}")
                best_params = start_values
                print('Epoch ' + str(epoch + 1) + ' finished')
                continue

            print('Epoch ' + str(epoch + 1) + ' finished')

            if isinstance(best_params, tuple):
                params = [best_params[1][par] for par in param_names]

                option_template = dict_multifill(option_template, param_names, params)
                borders.iloc[2] = params
                start_values = np.array(params)
            else:
                logging.warning("Epoch did not terminate successfully")

    return option.replace_budget(budget_scope, option_template)


# ---------------------------------------------------------------------------
# Parallel COBYLA multi-optimization workers (module-level, picklable)
# ---------------------------------------------------------------------------


def _cobyla_budget_worker(
    meta_model: MetaModel,
    scopes: list[OptionScope],
    budget_scope: str,
    borders_dict: dict,
    option: Option,
    bdg_size: float,
    if_exact: bool,
    if_long: bool | None,
    if_volume: bool,
    epochs: int,
    iters_epoch: int,
    error_score: float,
    ftol_abs: float,
    xtol_abs: float | None,
    verbose: bool,
    warn_nan_seasonality: bool,
    summary_nodes: dict[str, tuple[float, str]] | None = None,
) -> Option:

    import fermatrica_rep.options.translators as _translators_mod
    _saved = _translators_mod.WARN_NAN_SEASONALITY
    _translators_mod.WARN_NAN_SEASONALITY = warn_nan_seasonality
    try:
        return _optimize_budget_local_cobyla_impl(
            meta_model=meta_model, scopes=scopes, budget_scope=budget_scope,
            borders_dict=borders_dict, option=option,
            bdg_size=bdg_size, if_exact=if_exact, if_long=if_long,
            if_volume=if_volume, epochs=epochs, iters_epoch=iters_epoch,
            error_score=error_score, ftol_abs=ftol_abs, xtol_abs=xtol_abs,
            verbose=verbose, summary_nodes=summary_nodes,
        )
    finally:
        _translators_mod.WARN_NAN_SEASONALITY = _saved


def _cobyla_target_worker(
    meta_model: MetaModel,
    scopes: list[OptionScope],
    budget_scope: str,
    borders_dict: dict,
    option: Option,
    trg: float | dict[str, float],
    trg_tol_ratio: float,
    if_exact: bool,
    if_long: bool | None,
    if_volume: bool,
    epochs: int,
    iters_epoch: int,
    error_score: float,
    ftol_abs: float,
    xtol_abs: float | None,
    verbose: bool,
    warn_nan_seasonality: bool,
    summary_nodes: dict[str, tuple[float, str]] | None = None,
) -> Option:

    import fermatrica_rep.options.translators as _translators_mod
    _saved = _translators_mod.WARN_NAN_SEASONALITY
    _translators_mod.WARN_NAN_SEASONALITY = warn_nan_seasonality
    try:
        return _optimize_target_local_cobyla_impl(
            meta_model=meta_model, scopes=scopes, budget_scope=budget_scope,
            borders_dict=borders_dict, option=option,
            trg=trg, trg_tol_ratio=trg_tol_ratio,
            if_exact=if_exact, if_long=if_long,
            if_volume=if_volume, epochs=epochs, iters_epoch=iters_epoch,
            error_score=error_score, ftol_abs=ftol_abs, xtol_abs=xtol_abs,
            verbose=verbose, summary_nodes=summary_nodes,
        )
    finally:
        _translators_mod.WARN_NAN_SEASONALITY = _saved


def optimize_budget_local_cobyla_multi(
    meta_model: MetaModel,
    scopes: list[OptionScope],
    budget_scope: str,
    borders_dict: dict,
    option: Option,
    bdg_sizes: list[float],
    *,
    if_exact: bool = False,
    if_long: bool | None = None,
    if_volume: bool = True,
    epochs: int = 3,
    iters_epoch: int = 300,
    error_score: float = 1e+12,
    ftol_abs: float = .01,
    xtol_abs: float | None = None,
    verbose: bool = True,
    cores: int | None = None,
    warn_nan_seasonality: bool = True,
    summary_nodes: dict[str, tuple[float, str]] | None = None,
) -> dict[float, Option]:
    """
    Run multiple independent COBYLA budget optimizations in parallel.

    Each value in ``bdg_sizes`` spawns a separate COBYLA optimization
    with that fixed total budget. Optimizations are dispatched across
    ``cores`` worker processes via ``multiprocessing.Pool.starmap``.

    For per-budget behavior, see ``optimize_budget_local_cobyla``.

    Parameters
    ----------
    meta_model : MetaModel
        MetaModel instance containing the DAG of ModelNodes.
    scopes : list[OptionScope]
        List of OptionScope objects defining target periods and entities.
    budget_scope : str
        Which scope the budgets apply to.
    borders_dict : dict
        Dictionary of lower and upper bounds for each variable to vary.
    option : Option
        Base budget option. Must be an ``Option`` instance (dict not
        supported in multi mode).
    bdg_sizes : list[float]
        List of total budget values to optimize for. Must be non-empty
        and contain unique values.
    if_exact : bool, optional
        Apply only to the specific time period. Default ``False``.
    if_long : bool or None, optional
        Optimize long-term or short-term KPI. ``None`` → ``not if_exact``.
    if_volume : bool, optional
        Optimize volume (``True``) or value (``False``). Default ``True``.
    epochs : int, optional
        Number of restart epochs per budget. Default ``3``.
    iters_epoch : int, optional
        Max evaluations per epoch. Default ``300``.
    error_score : float, optional
        Large value used when model evaluation fails. Default ``1e+12``.
    ftol_abs : float, optional
        Early-stop threshold for objective change. Default ``0.01``.
    xtol_abs : float or None, optional
        Early-stop threshold for parameter change. Default ``None``.
    verbose : bool, optional
        Print progress. With ``cores > 1``, output from different workers
        interleaves. Default ``True``.
    cores : int or None, optional
        Number of worker processes. ``None`` → ``cpu_count() - 1`` (min 1).
        Set to ``1`` for sequential execution (no pickling).
    warn_nan_seasonality : bool, optional
        Warn about NaN seasonality during translation. Default ``True``.
    summary_nodes : dict[str, tuple[float, str]] or None, optional
        Multi-node scoring weights per node as ``(weight, scope)``.
        ``None`` = single sink node behavior. Default ``None``.

    Returns
    -------
    dict[float, Option]
        Mapping ``{bdg_size: optimized_Option}``.
    """
    if not isinstance(bdg_sizes, list) or len(bdg_sizes) == 0:
        fermatrica_error("bdg_sizes must be a non-empty list of floats")
    if len(set(bdg_sizes)) != len(bdg_sizes):
        fermatrica_error("bdg_sizes must contain unique values")
    if isinstance(option, dict):
        fermatrica_error("Passing dict for 'option' is not supported. Use Option() instead.")

    if if_long is None:
        if_long = not if_exact

    if cores is None:
        cores = max(1, multiprocessing.cpu_count() - 1)

    try:
        stripped = _strip_meta_model_callables(meta_model)
    except (pickle.PicklingError, TypeError, AttributeError):
        stripped = meta_model

    raw = None

    if cores > 1 and len(bdg_sizes) > 1:
        try:
            args = [
                (stripped, scopes, budget_scope, borders_dict, option, bdg_size,
                 if_exact, if_long, if_volume, epochs, iters_epoch, error_score,
                 ftol_abs, xtol_abs, verbose, warn_nan_seasonality, summary_nodes)
                for bdg_size in bdg_sizes
            ]
            with multiprocessing.Pool(cores) as pool:
                raw = pool.starmap(_cobyla_budget_worker, args)
        except (pickle.PicklingError, TypeError, AttributeError) as e:
            logging.warning(
                f"Multiprocessing pickling failed ({e}). Falling back to sequential."
            )

    if raw is None:
        raw = []
        for bdg_size in bdg_sizes:
            raw.append(
                _cobyla_budget_worker(
                    stripped, scopes, budget_scope, borders_dict, option, bdg_size,
                    if_exact, if_long, if_volume, epochs, iters_epoch, error_score,
                    ftol_abs, xtol_abs, verbose, warn_nan_seasonality, summary_nodes,
                )
            )

    return dict(zip(bdg_sizes, raw))


def optimize_target_local_cobyla_multi(
    meta_model: MetaModel,
    scopes: list[OptionScope],
    budget_scope: str,
    borders_dict: dict,
    option: Option,
    trg_list: list[float | dict[str, float]],
    *,
    trg_tol_ratio: float = 0.0001,
    if_exact: bool = False,
    if_long: bool | None = None,
    if_volume: bool = True,
    epochs: int = 3,
    iters_epoch: int = 300,
    error_score: float = 1e+12,
    ftol_abs: float = .01,
    xtol_abs: float | None = None,
    verbose: bool = True,
    cores: int | None = None,
    warn_nan_seasonality: bool = True,
    summary_nodes: dict[str, tuple[float, str]] | None = None,
) -> dict[float, Option]:
    """
    Run multiple independent COBYLA target optimizations in parallel.

    Each value in ``trg_list`` spawns a separate COBYLA optimization
    targeting that KPI value with minimum budget. Optimizations are
    dispatched across ``cores`` worker processes.

    For per-target behavior, see ``optimize_target_local_cobyla``.

    Parameters
    ----------
    meta_model : MetaModel
        MetaModel instance containing the DAG of ModelNodes.
    scopes : list[OptionScope]
        List of OptionScope objects defining target periods and entities.
    budget_scope : str
        Which scope the budgets apply to.
    borders_dict : dict
        Dictionary of lower and upper bounds for each variable to vary.
    option : Option
        Base budget option. Must be an ``Option`` instance.
    trg_list : list[float | dict[str, float]]
        List of target KPI values to minimize budget for. Must be
        non-empty and contain unique values. When ``summary_nodes`` is
        set, each element must be a ``dict`` with per-node raw targets.
    trg_tol_ratio : float, optional
        Tolerance ratio for the target KPI. Default ``0.0001``.
    if_exact : bool, optional
        Apply only to the specific time period. Default ``False``.
    if_long : bool or None, optional
        Optimize long-term or short-term KPI. ``None`` → ``not if_exact``.
    if_volume : bool, optional
        Optimize volume (``True``) or value (``False``). Default ``True``.
    epochs : int, optional
        Number of restart epochs per target. Default ``3``.
    iters_epoch : int, optional
        Max evaluations per epoch. Default ``300``.
    error_score : float, optional
        Large value used when model evaluation fails. Default ``1e+12``.
    ftol_abs : float, optional
        Early-stop threshold for objective change. Default ``0.01``.
    xtol_abs : float or None, optional
        Early-stop threshold for parameter change. Default ``None``.
    verbose : bool, optional
        Print progress. With ``cores > 1``, output interleaves.
        Default ``True``.
    cores : int or None, optional
        Number of worker processes. ``None`` → ``cpu_count() - 1`` (min 1).
    warn_nan_seasonality : bool, optional
        Warn about NaN seasonality during translation. Default ``True``.
    summary_nodes : dict[str, tuple[float, str]] or None, optional
        Multi-node scoring weights per node as ``(weight, scope)``.
        ``None`` = single sink node behavior. Default ``None``.

    Returns
    -------
    dict[float, Option]
        Mapping ``{combined_trg: optimized_Option}``. When
        ``summary_nodes`` is set, keys are the combined float targets.
    """
    if not isinstance(trg_list, list) or len(trg_list) == 0:
        fermatrica_error("trg_list must be a non-empty list of floats or dicts with unique values")
    if len(set(
        tuple(sorted(t.items())) if isinstance(t, dict) else t
        for t in trg_list
    )) != len(trg_list):
        fermatrica_error("trg_list must contain unique values")
    if isinstance(option, dict):
        fermatrica_error("Passing dict for 'option' is not supported. Use Option() instead.")

    if if_long is None:
        if_long = not if_exact

    if cores is None:
        cores = max(1, multiprocessing.cpu_count() - 1)

    pre_baseline = None
    if summary_nodes is not None:
        pre_baseline = _compute_baseline(meta_model, option.budget_data, budget_scope,
                                          scopes, if_exact, if_long, if_volume,
                                          summary_nodes)

    try:
        stripped = _strip_meta_model_callables(meta_model)
    except (pickle.PicklingError, TypeError, AttributeError):
        stripped = meta_model

    raw = None

    if cores > 1 and len(trg_list) > 1:
        try:
            args = [
                (stripped, scopes, budget_scope, borders_dict, option, trg,
                 trg_tol_ratio, if_exact, if_long, if_volume, epochs,
                 iters_epoch, error_score, ftol_abs, xtol_abs, verbose,
                 warn_nan_seasonality, summary_nodes)
                for trg in trg_list
            ]
            with multiprocessing.Pool(cores) as pool:
                raw = pool.starmap(_cobyla_target_worker, args)
        except (pickle.PicklingError, TypeError, AttributeError) as e:
            logging.warning(
                f"Multiprocessing pickling failed ({e}). Falling back to sequential."
            )

    if raw is None:
        raw = []
        for trg in trg_list:
            raw.append(
                _cobyla_target_worker(
                    stripped, scopes, budget_scope, borders_dict, option, trg,
                    trg_tol_ratio, if_exact, if_long, if_volume, epochs,
                    iters_epoch, error_score, ftol_abs, xtol_abs, verbose,
                    warn_nan_seasonality, summary_nodes,
                )
            )

    keys = [_combine_trg(trg, summary_nodes, pre_baseline) for trg in trg_list]
    return dict(zip(keys, raw))


def optimize_budget_local_cobyla(
    meta_model: MetaModel | None = None,
    scopes: list[OptionScope] | None = None,
    budget_scope: str | None = None,
    borders_dict: dict | None = None,
    option: Option | dict | None = None,
    bdg_size: float = 100,
    if_exact: bool = False,
    if_long: bool | None = None,
    if_volume: bool = True,
    epochs: int = 3,
    iters_epoch: int = 300,
    error_score: float = 1e+12,
    ftol_abs: float = .01,
    xtol_abs: float | None = None,
    verbose: bool = True,
    warn_nan_seasonality: bool = True,
    summary_nodes: dict[str, tuple[float, str]] | None = None,

    ds=None,
    model=None,
    model_rep=None,
    opt_set=None,
) -> Option | dict:
    """
    Maximize KPI with a known total budget via COBYLA local optimization.

    Uses NLopt's COBYLA (Constrained Optimization BY Linear
    Approximations) algorithm, which is derivative-free and handles
    inequality constraints natively. A budget equality constraint is
    enforced via an inequality constraint: ``sum(params) - bdg_size
    <= 0`` and ``-(sum(params) - bdg_size) <= 0``.

    Runs multiple **epochs** from different starting points to help
    escape local optima. Each epoch uses the previous epoch's best
    parameters as its starting point.

    For more background:
    https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#cobyla-constrained-optimization-by-linear-approximations

    Parameters
    ----------
    meta_model : MetaModel
        MetaModel instance containing the DAG of ModelNodes.
    scopes : list[OptionScope]
        List of OptionScope objects defining target periods and entities.
    budget_scope : str
        Which scope the budgets apply to.
    borders_dict : dict
        Dictionary of lower and upper bounds for each variable to vary,
        formatted as ``{var: [low, high, start]}`` where ``low`` and
        ``high`` are the bounds and ``start`` is the initial guess.
    option : Option or dict
        Base budget option. If ``Option``, returns ``Option``. If ``dict``
        (deprecated), returns ``dict``.
    bdg_size : float, optional
        Fixed total budget size. The optimizer changes the split while
        keeping the total constant. Default ``100``.
    if_exact : bool, optional
        Apply only to the specific time period without forward
        projection. Default ``False``.
    if_long : bool or None, optional
        Optimize long-term or short-term KPI. If ``None``, uses
        ``not if_exact``.
    if_volume : bool, optional
        Optimize volume (``True``) or value (``False``) KPI.
        Default ``True``.
    epochs : int, optional
        Number of restart epochs. Default ``3``.
    iters_epoch : int, optional
        Maximum objective evaluations per epoch. Default ``300``.
    error_score : float, optional
        Large value used when model evaluation fails.
        Default ``1e+12``.
    ftol_abs : float, optional
        Early-stop threshold: minimum absolute objective change per
        iteration. Default ``0.01``.
    xtol_abs : float or None, optional
        Early-stop threshold: minimum absolute parameter change per
        iteration. Default ``None`` (disabled).
    verbose : bool, optional
        Print progress information. Default ``True``.
    summary_nodes : dict[str, tuple[float, str]] or None, optional
        Multi-node scoring weights per node as ``(weight, scope)``.
        ``None`` = single sink node behavior. Default ``None``.

    Returns
    -------
    Option or dict
        Optimized budget option. Returns ``Option`` when ``option`` is
        ``Option``, or ``dict`` with a ``"bdg"`` key when ``option`` is
        ``dict`` (deprecated).
    """

    _old_api = any(x is not None for x in (ds, model, model_rep, opt_set))
    _option_is_dict = isinstance(option, dict)

    if not _old_api and any(x is None for x in (meta_model, scopes, budget_scope, borders_dict, option)):
        fermatrica_error(
            "optimize_budget_local_cobyla requires meta_model, scopes, budget_scope, borders_dict, option. "
            "Or use deprecated params: ds, model, model_rep, opt_set."
        )

    if _old_api:
        from fermatrica_rep.options.define import _option_settings_to_scope

        warnings.warn(
            "Passing ds=, model=, model_rep=, opt_set= to optimize_budget_local_cobyla() is deprecated. "
            "Use meta_model=, scopes=, budget_scope=, option=Option() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        scopes = [_option_settings_to_scope(opt_set)]
        budget_scope = scopes[0].scope_name
        node = ModelNode(model=model, ds=ds, model_rep=model_rep)
        meta_model = MetaModel(nodes=node)

    if _option_is_dict:
        warnings.warn(
            "Passing dict for 'option' is deprecated. Use Option() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        bdg_items = {
            k: float(v) for k, v in option.items()
            if k.startswith('bdg_') and k != 'bdg'
        }
        option = Option(name="optim_option", budget_data={budget_scope: bdg_items})

    import fermatrica_rep.options.translators as _translators_mod
    _saved = _translators_mod.WARN_NAN_SEASONALITY
    _translators_mod.WARN_NAN_SEASONALITY = warn_nan_seasonality
    try:
        result = _optimize_budget_local_cobyla_impl(
            meta_model=meta_model, scopes=scopes, budget_scope=budget_scope,
            borders_dict=borders_dict, option=option,
            bdg_size=bdg_size, if_exact=if_exact, if_long=if_long,
            if_volume=if_volume, epochs=epochs, iters_epoch=iters_epoch,
            error_score=error_score, ftol_abs=ftol_abs, xtol_abs=xtol_abs,
            verbose=verbose, summary_nodes=summary_nodes,
        )
    finally:
        _translators_mod.WARN_NAN_SEASONALITY = _saved

    if _option_is_dict:
        result_dict = dict(result.budget_data.get(budget_scope, {}))
        result_dict['bdg'] = sum(
            v for k, v in result_dict.items()
            if k.startswith('bdg_') and k != 'bdg'
        )
        return result_dict

    return result


def optimize_budget_local_ncma(
    meta_model: MetaModel | None = None,
    scopes: list[OptionScope] | None = None,
    budget_scope: str | None = None,
    borders_dict: dict | None = None,
    option: Option | dict | None = None,
    bdg_size: float = 100,
    if_exact: bool = False,
    if_long: bool | None = None,
    if_volume: bool = True,
    epochs: int = 1,
    iters_epoch: int = 1000,
    error_score: float = 1e+12,
    cores: int = 4,
    verbose: bool = True,
    warn_nan_seasonality: bool = True,
    summary_nodes: dict[str, tuple[float, str]] | None = None,

    ds=None,
    model=None,
    model_rep=None,
    opt_set=None,
) -> Option | dict:
    """
    Maximize KPI with a known total budget via CMA-ES using Nevergrad.

    Uses CMA-ES (Covariance Matrix Adaptation Evolution Strategy), a
    derivative-free global optimizer that maintains a search distribution
    and adapts it over generations. Better suited for rugged landscapes
    with complex constraints than COBYLA.

    Budget equality is enforced through the ``ObjectiveBudgetNG``
    wrapper which adds a penalty to the objective when the total budget
    deviates from ``bdg_size``.

    Runs multiple **epochs** from different starting points. Uses
    ``ProcessPoolExecutor`` for parallel model evaluations within each
    CMA-ES generation.

    Parameters
    ----------
    meta_model : MetaModel
        MetaModel instance (must be pickle-safe for multiprocessing).
    scopes : list[OptionScope]
        List of OptionScope objects defining target periods and entities.
    budget_scope : str
        Which scope the budgets apply to.
    borders_dict : dict
        Dictionary of lower and upper bounds for each variable to vary,
        formatted as ``{var: [low, high, start]}``.
    option : Option or dict
        Base budget option. If ``Option``, returns ``Option``. If ``dict``
        (deprecated), returns ``dict``.
    bdg_size : float, optional
        Fixed total budget size. Default ``100``.
    if_exact : bool, optional
        Apply only to the specific time period without forward
        projection. Default ``False``.
    if_long : bool or None, optional
        Optimize long-term or short-term KPI. If ``None``, uses
        ``not if_exact``.
    if_volume : bool, optional
        Optimize volume (``True``) or value (``False``) KPI.
        Default ``True``.
    epochs : int, optional
        Number of restart epochs. Default ``1``.
    iters_epoch : int, optional
        Budget (total evaluations) per epoch. Default ``1000``.
    error_score : float, optional
        Large value used when model evaluation fails.
        Default ``1e+12``.
    cores : int, optional
        Number of CPU cores for parallel evaluation. Default ``4``.
    verbose : bool, optional
        Print progress information. Default ``True``.
    summary_nodes : dict[str, tuple[float, str]] or None, optional
        Multi-node scoring weights per node as ``(weight, scope)``.
        ``None`` = single sink node behavior. Default ``None``.

    Returns
    -------
    Option or dict
        Optimized budget option. Returns ``Option`` when ``option`` is
        ``Option``, or ``dict`` with a ``"bdg"`` key when ``option`` is
        ``dict`` (deprecated).
    """

    _old_api = any(x is not None for x in (ds, model, model_rep, opt_set))
    _option_is_dict = isinstance(option, dict)

    if not _old_api and any(x is None for x in (meta_model, scopes, budget_scope, borders_dict, option)):
        fermatrica_error(
            "optimize_budget_local_ncma requires meta_model, scopes, budget_scope, borders_dict, option. "
            "Or use deprecated params: ds, model, model_rep, opt_set."
        )

    if _old_api:
        from fermatrica_rep.options.define import _option_settings_to_scope

        warnings.warn(
            "Passing ds=, model=, model_rep=, opt_set= to optimize_budget_local_ncma() is deprecated. "
            "Use meta_model=, scopes=, budget_scope=, option=Option() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        scopes = [_option_settings_to_scope(opt_set)]
        budget_scope = scopes[0].scope_name
        node = ModelNode(model=model, ds=ds, model_rep=model_rep)
        meta_model = MetaModel(nodes=node)

    if _option_is_dict:
        warnings.warn(
            "Passing dict for 'option' is deprecated. Use Option() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        bdg_items = {
            k: float(v) for k, v in option.items()
            if k.startswith('bdg_') and k != 'bdg'
        }
        option = Option(name="optim_option", budget_data={budget_scope: bdg_items})

    import fermatrica_rep.options.translators as _translators_mod
    _saved = _translators_mod.WARN_NAN_SEASONALITY
    _translators_mod.WARN_NAN_SEASONALITY = warn_nan_seasonality
    try:
        result = _optimize_budget_local_ncma_impl(
            meta_model=meta_model, scopes=scopes, budget_scope=budget_scope,
            borders_dict=borders_dict, option=option,
            bdg_size=bdg_size, if_exact=if_exact, if_long=if_long,
            if_volume=if_volume, epochs=epochs, iters_epoch=iters_epoch,
            error_score=error_score, cores=cores, verbose=verbose,
            summary_nodes=summary_nodes,
        )
    finally:
        _translators_mod.WARN_NAN_SEASONALITY = _saved

    if _option_is_dict:
        result_dict = dict(result.budget_data.get(budget_scope, {}))
        result_dict['bdg'] = sum(
            v for k, v in result_dict.items()
            if k.startswith('bdg_') and k != 'bdg'
        )
        return result_dict

    return result


def optimize_target_local_cobyla(
    meta_model: MetaModel | None = None,
    scopes: list[OptionScope] | None = None,
    budget_scope: str | None = None,
    borders_dict: dict | None = None,
    option: Option | dict | None = None,
    trg: float | dict[str, float] = 100,
    trg_tol_ratio: float = 0.0001,
    if_exact: bool = False,
    if_long: bool | None = None,
    if_volume: bool = True,
    epochs: int = 3,
    iters_epoch: int = 300,
    error_score: float = 1e+12,
    ftol_abs: float = .01,
    xtol_abs: float | None = None,
    verbose: bool = True,
    warn_nan_seasonality: bool = True,
    summary_nodes: dict[str, tuple[float, str]] | None = None,

    ds=None,
    model=None,
    model_rep=None,
    opt_set=None,
) -> Option | dict:
    """
    Minimize budget size to achieve a target KPI via COBYLA local
    optimization.

    Uses NLopt's COBYLA algorithm. The objective is simply the sum of
    parameters (total budget), and the KPI target is enforced as an
    inequality constraint via ``ConstraintFunctionTarget``.

    Runs multiple **epochs** from different starting points.

    Parameters
    ----------
    meta_model : MetaModel
        MetaModel instance containing the DAG of ModelNodes.
    scopes : list[OptionScope]
        List of OptionScope objects defining target periods and entities.
    budget_scope : str
        Which scope the budgets apply to.
    borders_dict : dict
        Dictionary of lower and upper bounds for each variable to vary,
        formatted as ``{var: [low, high, start]}``.
    option : Option or dict
        Base budget option. If ``Option``, returns ``Option``. If ``dict``
        (deprecated), returns ``dict``.
    trg : float or dict[str, float], optional
        Target KPI value(s) to achieve with minimum budget. When
        ``summary_nodes`` is ``None``, a float (raw KPI value for the
        sink node). When ``summary_nodes`` is set, a dict of per-node
        raw KPI targets, e.g. ``{"sales": 200, "awareness": 35}``.
        Default ``100``.
    trg_tol_ratio : float, optional
        Tolerance ratio for the target KPI. The constraint considers
        the KPI satisfied when within ``trg_tol_ratio * combined_trg``
        of the target. Default ``0.0001``.
    if_exact : bool, optional
        Apply only to the specific time period without forward
        projection. Default ``False``.
    if_long : bool or None, optional
        Optimize long-term or short-term KPI. If ``None``, uses
        ``not if_exact``.
    if_volume : bool, optional
        Optimize volume (``True``) or value (``False``) KPI.
        Default ``True``.
    epochs : int, optional
        Number of restart epochs. Default ``3``.
    iters_epoch : int, optional
        Maximum objective evaluations per epoch. Default ``300``.
    error_score : float, optional
        Large value used when model evaluation fails.
        Default ``1e+12``.
    ftol_abs : float, optional
        Early-stop threshold: minimum absolute objective change per
        iteration. Default ``0.01``.
    xtol_abs : float or None, optional
        Early-stop threshold: minimum absolute parameter change per
        iteration. Default ``None`` (disabled).
    verbose : bool, optional
        Print progress information. Default ``True``.
    summary_nodes : dict[str, tuple[float, str]] or None, optional
        Multi-node scoring weights per node as ``(weight, scope)``.
        ``None`` = single sink node behavior. Default ``None``.

    Returns
    -------
    Option or dict
        Optimized budget option. Returns ``Option`` when ``option`` is
        ``Option``, or ``dict`` with a ``"bdg"`` key when ``option`` is
        ``dict`` (deprecated).
    """

    _old_api = any(x is not None for x in (ds, model, model_rep, opt_set))
    _option_is_dict = isinstance(option, dict)

    if not _old_api and any(x is None for x in (meta_model, scopes, budget_scope, borders_dict, option)):
        fermatrica_error(
            "optimize_target_local_cobyla requires meta_model, scopes, budget_scope, borders_dict, option. "
            "Or use deprecated params: ds, model, model_rep, opt_set."
        )

    if _old_api:
        from fermatrica_rep.options.define import _option_settings_to_scope

        warnings.warn(
            "Passing ds=, model=, model_rep=, opt_set= to optimize_target_local_cobyla() is deprecated. "
            "Use meta_model=, scopes=, budget_scope=, option=Option() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        scopes = [_option_settings_to_scope(opt_set)]
        budget_scope = scopes[0].scope_name
        node = ModelNode(model=model, ds=ds, model_rep=model_rep)
        meta_model = MetaModel(nodes=node)

    if _option_is_dict:
        warnings.warn(
            "Passing dict for 'option' is deprecated. Use Option() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        bdg_items = {
            k: float(v) for k, v in option.items()
            if k.startswith('bdg_') and k != 'bdg'
        }
        option = Option(name="optim_option", budget_data={budget_scope: bdg_items})

    import fermatrica_rep.options.translators as _translators_mod
    _saved = _translators_mod.WARN_NAN_SEASONALITY
    _translators_mod.WARN_NAN_SEASONALITY = warn_nan_seasonality
    try:
        result = _optimize_target_local_cobyla_impl(
            meta_model=meta_model, scopes=scopes, budget_scope=budget_scope,
            borders_dict=borders_dict, option=option,
            trg=trg, trg_tol_ratio=trg_tol_ratio,
            if_exact=if_exact, if_long=if_long,
            if_volume=if_volume, epochs=epochs, iters_epoch=iters_epoch,
            error_score=error_score, ftol_abs=ftol_abs, xtol_abs=xtol_abs,
            verbose=verbose, summary_nodes=summary_nodes,
        )
    finally:
        _translators_mod.WARN_NAN_SEASONALITY = _saved

    if _option_is_dict:
        result_dict = dict(result.budget_data.get(budget_scope, {}))
        result_dict['bdg'] = sum(
            v for k, v in result_dict.items()
            if k.startswith('bdg_') and k != 'bdg'
        )
        return result_dict

    return result


def optimize_target_local_ncma(
    meta_model: MetaModel | None = None,
    scopes: list[OptionScope] | None = None,
    budget_scope: str | None = None,
    borders_dict: dict | None = None,
    option: Option | dict | None = None,
    trg: float | dict[str, float] = 100,
    trg_tol_ratio: float = 0.0001,
    if_exact: bool = False,
    if_long: bool | None = None,
    if_volume: bool = True,
    epochs: int = 1,
    iters_epoch: int = 1000,
    error_score: float = 1e+12,
    cores: int = 4,
    verbose: bool = True,
    warn_nan_seasonality: bool = True,
    summary_nodes: dict[str, tuple[float, str]] | None = None,

    ds=None,
    model=None,
    model_rep=None,
    opt_set=None,
) -> Option | dict:
    """
    Minimize budget size to achieve a target KPI via CMA-ES using
    Nevergrad.

    Uses CMA-ES with the ``ObjectiveTargetNG`` wrapper which minimizes
    the total budget while adding a penalty for KPI violations. Parallel
    model evaluations use ``ProcessPoolExecutor`` within each CMA-ES
    generation.

    Parameters
    ----------
    meta_model : MetaModel
        MetaModel instance (must be pickle-safe for multiprocessing).
    scopes : list[OptionScope]
        List of OptionScope objects defining target periods and entities.
    budget_scope : str
        Which scope the budgets apply to.
    borders_dict : dict
        Dictionary of lower and upper bounds for each variable to vary,
        formatted as ``{var: [low, high, start]}``.
    option : Option or dict
        Base budget option. If ``Option``, returns ``Option``. If ``dict``
        (deprecated), returns ``dict``.
    trg : float or dict[str, float], optional
        Target KPI value(s) to achieve with minimum budget. When
        ``summary_nodes`` is ``None``, a float (raw KPI value for the
        sink node). When ``summary_nodes`` is set, a dict of per-node
        raw KPI targets, e.g. ``{"sales": 200, "awareness": 35}``.
        Default ``100``.
    trg_tol_ratio : float, optional
        Tolerance ratio for the target KPI. Default ``0.0001``.
    if_exact : bool, optional
        Apply only to the specific time period without forward
        projection. Default ``False``.
    if_long : bool or None, optional
        Optimize long-term or short-term KPI. If ``None``, uses
        ``not if_exact``.
    if_volume : bool, optional
        Optimize volume (``True``) or value (``False``) KPI.
        Default ``True``.
    epochs : int, optional
        Number of restart epochs. Default ``1``.
    iters_epoch : int, optional
        Budget (total evaluations) per epoch. Default ``1000``.
    error_score : float, optional
        Large value used when model evaluation fails.
        Default ``1e+12``.
    cores : int, optional
        Number of CPU cores for parallel evaluation. Default ``4``.
    verbose : bool, optional
        Print progress information. Default ``True``.
    summary_nodes : dict[str, tuple[float, str]] or None, optional
        Multi-node scoring weights per node as ``(weight, scope)``.
        ``None`` = single sink node behavior. Default ``None``.

    Returns
    -------
    Option or dict
        Optimized budget option. Returns ``Option`` when ``option`` is
        ``Option``, or ``dict`` with a ``"bdg"`` key when ``option`` is
        ``dict`` (deprecated).
    """

    _old_api = any(x is not None for x in (ds, model, model_rep, opt_set))
    _option_is_dict = isinstance(option, dict)

    if not _old_api and any(x is None for x in (meta_model, scopes, budget_scope, borders_dict, option)):
        fermatrica_error(
            "optimize_target_local_ncma requires meta_model, scopes, budget_scope, borders_dict, option. "
            "Or use deprecated params: ds, model, model_rep, opt_set."
        )

    if _old_api:
        from fermatrica_rep.options.define import _option_settings_to_scope

        warnings.warn(
            "Passing ds=, model=, model_rep=, opt_set= to optimize_target_local_ncma() is deprecated. "
            "Use meta_model=, scopes=, budget_scope=, option=Option() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        scopes = [_option_settings_to_scope(opt_set)]
        budget_scope = scopes[0].scope_name
        node = ModelNode(model=model, ds=ds, model_rep=model_rep)
        meta_model = MetaModel(nodes=node)

    if _option_is_dict:
        warnings.warn(
            "Passing dict for 'option' is deprecated. Use Option() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        bdg_items = {
            k: float(v) for k, v in option.items()
            if k.startswith('bdg_') and k != 'bdg'
        }
        option = Option(name="optim_option", budget_data={budget_scope: bdg_items})

    import fermatrica_rep.options.translators as _translators_mod
    _saved = _translators_mod.WARN_NAN_SEASONALITY
    _translators_mod.WARN_NAN_SEASONALITY = warn_nan_seasonality
    try:
        result = _optimize_target_local_ncma_impl(
            meta_model=meta_model, scopes=scopes, budget_scope=budget_scope,
            borders_dict=borders_dict, option=option,
            trg=trg, trg_tol_ratio=trg_tol_ratio,
            if_exact=if_exact, if_long=if_long,
            if_volume=if_volume, epochs=epochs, iters_epoch=iters_epoch,
            error_score=error_score, cores=cores, verbose=verbose,
            summary_nodes=summary_nodes,
        )
    finally:
        _translators_mod.WARN_NAN_SEASONALITY = _saved

    if _option_is_dict:
        result_dict = dict(result.budget_data.get(budget_scope, {}))
        result_dict['bdg'] = sum(
            v for k, v in result_dict.items()
            if k.startswith('bdg_') and k != 'bdg'
        )
        return result_dict

    return result
