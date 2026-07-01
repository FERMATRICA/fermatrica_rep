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
import pandas as pd
import numpy as np

import nlopt
import nevergrad as ng
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor

from fermatrica_utils import dict_multifill

from fermatrica import Model

from fermatrica_rep.options.define import OptionSettings
from fermatrica_rep.options.objective import ConstraintFunctionTarget, ObjectiveBudgetNG, ObjectiveFunctionBudget, ObjectiveTargetNG, VerboseNG


def optimize_budget_local_cobyla(ds: pd.DataFrame | list
                                 , model: "Model | list"
                                 , model_rep: "ModelRep | list"
                                 , borders_dict: dict
                                 , option: dict
                                 , opt_set: "OptionSettings | list"
                                 , bdg_size: float = 100
                                 , if_exact: bool = False
                                 , if_long: bool | None = None
                                 , if_volume: bool = True
                                 , epochs: int = 3
                                 , iters_epoch: int = 300
                                 , error_score: float = 1e+12
                                 , ftol_abs: float = .01
                                 , xtol_abs: float | None = None
                                 , verbose: bool = True
                                 ):
    """
    Maximize KPI with known budget via COBYLA constrained optimization by local approximations algorithm (local).
    COBYLA is derivative-free, so no analytical gradient is required.

    Epochs are required to shuffle params a bit and to help algo get out from local optimum (sometimes)
    and to speed up calculation.

    Early stop threshold is defined as minimum absolute score gain per iteration. However, algo doesn't
    respect it directly, so don't be upset to see it working with much lesser gain per iteration.

    For more hints about COBYLA algorithm check docs of `fermatrica.model.optim.optimize_local_cobyla`
    or https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#cobyla-constrained-optimization-by-linear-approximations
    and Powell's articles mentioned via link.

    :param ds: dataset or list of datasets
    :param model: Model object or list of Model objects
    :param model_rep: ModelRep object (export settings) or list of ModelRep objects
    :param borders_dict: dictionary of low and upper borders for every variable to vary
    :param option: base budget option / scenario to calculate as dictionary. Optimized values to be inserted
        in this option when optimizing and later export
    :param option: base budget option / scenario to calculate as dictionary. Optimized values to be inserted
        in this option when optimizing and later export
    :param opt_set: OptionSettings object (option setting: target period etc.) or list of OptionSettings objects (one per model)
    :param bdg_size: fixed budget size (we change budget split, but not budget size)
    :param if_exact: apply only to the specific time period, without next years
    :param if_long: optimize long or short KPI. If `None`, `if_exact` is used
    :param if_volume: optimize volume or value KPI
    :param epochs: number of epochs
    :param iters_epoch: number of objective function calculations per epoch
    :param error_score: extremely big value to be used as score if fit_predict returns None (error)
    :param ftol_abs: early stop threshold: minimum absolute objective function return value gain per iteration
    :param xtol_abs: early stop threshold: minimum absolute change of independent variable
    :param verbose: print diagnostic or progress information
    :return: optimal option as dict
    """

    if if_long is None:
        if_long = not if_exact

    option = copy.deepcopy(option)
    borders = pd.DataFrame(borders_dict)

    borders_fixed = borders[list(borders.columns[borders.nunique() == 1].values)]
    borders = borders[list(borders.columns[borders.nunique() != 1].values)]

    bdg_size = bdg_size - borders_fixed.iloc[2].sum()

    if bdg_size <= borders.iloc[2].sum():
        borders.iloc[2] = borders.iloc[2] * bdg_size / borders.iloc[2].sum() * 0.9

    tmp = borders.iloc[2]
    for i in tmp.index:
        if tmp[i] < borders.iloc[0][i]:
            tmp[i] = borders.iloc[0][i]
    borders.iloc[2] = tmp

    option = dict_multifill(option, borders_fixed.columns, borders_fixed.iloc[2])

    for i in range(epochs):

        objective_fun = ObjectiveFunctionBudget(ds=ds
                                               , model=model
                                               , model_rep=model_rep
                                               , option=option
                                               , opt_set=opt_set
                                               , if_exact=if_exact
                                               , if_long=if_long
                                               , if_volume=if_volume
                                               , verbose=verbose
                                               , error_score=error_score
                                               , borders=borders)

        start_values = borders.iloc[
            2].values  # {"bdg_tv" : [0, 30, 10], "bdg_dgt" : [5, 50, 7]}  <- take [10 , 7] from this

        opt_obj = nlopt.opt(nlopt.LN_COBYLA, len(start_values))

        opt_obj.set_min_objective(objective_fun)
        opt_obj.set_lower_bounds(
            borders.iloc[0].values)  # {"bdg_tv" : [0, 30, 10], "bdg_dgt" : [5, 50, 7]} <- take [0 , 5]  from this
        opt_obj.set_upper_bounds(
            borders.iloc[1].values)  # {"bdg_tv" : [0, 30, 10], "bdg_dgt" : [5, 50, 7]} <- take [30 , 50]  from this

        def constraint(params_cur: list | np.ndarray | pd.Series | None = None
                       , solution_idx=None):
            return sum(params_cur) - bdg_size

        # https://nlopt.readthedocs.io/en/latest/NLopt_Python_Reference/#bound-constraints
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
            option = dict_multifill(option, list(borders.keys()), rtrn)
            borders.iloc[2] = rtrn
        else:
            logging.warning("Epoche did not terminate successfully")

    option['bdg'] = sum(v for k, v in option.items() if k.startswith('bdg_') and k != 'bdg')

    return option


def optimize_budget_local_ncma(ds: pd.DataFrame | list
                                , model: "Model | list"
                                , model_rep: "ModelRep | list"
                                , borders_dict: dict
                                , option: dict
                                , opt_set: "OptionSettings | list"
                                , bdg_size: float = 100
                                , if_exact: bool = False
                                , if_long: bool | None = None
                                , if_volume: bool = True
                                , epochs: int = 1
                                , iters_epoch: int = 1000
                                , error_score: float = 1e+12
                                , cores: int = 4
                                , verbose: bool = True):
    """
    Maximize KPI with known budget via CMA-ES algorithm using nevergrad library.
    CMA-ES is a derivative-free global optimizer that maintains search history between iterations.
    It can handle bounds and is robust to noisy objective functions.

    Epochs are required to shuffle params a bit and to help algo get out from local optimum (sometimes)
    and to speed up calculation.

    :param ds: dataset or list of datasets
    :param model: Model object or list of Model objects
    :param model_rep: ModelRep object (export settings) or list of ModelRep objects
    :param borders_dict: dictionary of low and upper borders for every variable to vary
    :param option: base budget option / scenario to calculate as dictionary. Optimized values to be inserted
        in this option when optimizing and later export
    :param opt_set: OptionSettings object (option setting: target period etc.) or list of OptionSettings objects (one per model)
    :param bdg_size: fixed budget size (we change budget split, but not budget size)
    :param if_exact: apply only to the specific time period, without next years
    :param if_long: optimize long or short KPI. If `None`, `if_exact` is used
    :param if_volume: optimize volume or value KPI
    :param epochs: number of epochs
    :param iters_epoch: number of objective function calculations per epoch
    :param error_score: extremely big value to be used as score if fit_predict returns None (error)
    :param cores: processor cores to be used in parallel computing
    :param verbose: print diagnostic or progress information
    :return: optimal option as dict
    """

    if if_long is None:
        if_long = not if_exact

    option = copy.deepcopy(option)
    borders = pd.DataFrame(borders_dict)

    borders_fixed = borders[list(borders.columns[borders.nunique() == 1].values)]
    borders = borders[list(borders.columns[borders.nunique() != 1].values)]

    bdg_size = bdg_size - borders_fixed.iloc[2].sum()

    if bdg_size <= borders.iloc[2].sum():
        borders.iloc[2] = borders.iloc[2] * bdg_size / borders.iloc[2].sum() * 0.9

    tmp = borders.iloc[2]
    for i in tmp.index:
        if tmp[i] < borders.iloc[0][i]:
            tmp[i] = borders.iloc[0][i]
    borders.iloc[2] = tmp

    option = dict_multifill(option, borders_fixed.columns, borders_fixed.iloc[2])

    # Setup for CMA-ES

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

    # destroy callable objects for multiprocessing

    model_iter = copy.deepcopy(model)

    if isinstance(model_iter, list):
        for i, val in enumerate(model_iter):
            model_iter[i].obj.transform_lhs_fn = None
            model_iter[i].obj.custom_predict_fn = None
    else:
        model_iter.obj.transform_lhs_fn = None
        model_iter.obj.custom_predict_fn = None

    with ProcessPoolExecutor(max_workers=cores) as executor:

        for epoch in range(epochs):

            # Create objective function
            objective_fun = ObjectiveFunctionBudget(ds=ds
                                                   , model=model_iter
                                                   , model_rep=model_rep
                                                   , option=option
                                                   , opt_set=opt_set
                                                   , if_exact=if_exact
                                                   , if_long=if_long
                                                   , if_volume=if_volume
                                                   , verbose=False
                                                   , error_score=error_score
                                                   , borders=borders
                                                   )

            # Setup search space
            search_space = ng.p.Instrumentation(**{
                name: ng.p.Scalar(lower=low, upper=high, init=val)
                for name, low, high, val in zip(param_names, lwr, upr, start_values)
            })

            # Create wrapper for nevergrad with budget constraint
            objective_ng = ObjectiveBudgetNG(param_names=param_names
                                            , objective_fun=objective_fun
                                            , bdg_size=bdg_size)

            # Initialize CMA-ES optimizer
            optimizer = ng.optimizers.CMA(
                parametrization=search_space,
                budget=iters_epoch,
                num_workers=cores,
            )

            # Verbose callback
            if verbose:
                verbose_cb = VerboseNG(10)
                optimizer.register_callback('tell', verbose_cb)

            # Run optimization
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

            # Update parameters
            if isinstance(best_params, tuple):
                params = [best_params[1][par] for par in param_names]

                # Normalize to budget
                total = sum(params)
                if total > 0:
                    params = [p * bdg_size / total for p in params]
                
                option = dict_multifill(option, param_names, params)
                borders.iloc[2] = params
                start_values = np.array(params)
            else:
                logging.warning("Epoch did not terminate successfully")

    option['bdg'] = sum(v for k, v in option.items() if k.startswith('bdg_') and k != 'bdg')

    return option


def optimize_target_local_cobyla(ds: pd.DataFrame | list
                                 , model_rep: "ModelRep | list"
                                 , model: "Model | list"
                                 , borders_dict: dict
                                 , opt_set: "OptionSettings | list"
                                 , option: dict
                                 , trg: float = 100
                                 , trg_tol_ratio: float = 0.0001
                                 , if_exact: bool = False
                                 , if_long: bool | None = None
                                 , if_volume: bool = True
                                 , epochs: int = 3
                                 , iters_epoch: int = 300
                                 , error_score: float = 1e+12
                                 , ftol_abs: float = .01
                                 , xtol_abs: float | None = None
                                 , verbose: bool = True):
    """
    Minimize budget size to achieve known KPI via COBYLA constrained optimization by local approximations
    algorithm (local). COBYLA is derivative-free, so no analytical gradient is required.

    Epochs are required to shuffle params a bit and to help algo get out from local optimum (sometimes)
    and to speed up calculation.

    Early stop threshold is defined as minimum absolute score gain per iteration. However, algo doesn't
    respect it directly, so don't be upset to see it working with much lesser gain per iteration.

    For more hints about COBYLA algorithm check docs of `fermatrica.model.optim.optimize_local_cobyla`
    or https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#cobyla-constrained-optimization-by-linear-approximations
    and Powell's articles mentioned via link.

    :param ds: dataset or list of datasets
    :param model: Model object or list of Model objects
    :param model_rep: ModelRep object (export settings) or list of ModelRep objects
    :param borders_dict: dictionary of low and upper borders for every variable to vary
    :param option: base budget option / scenario to calculate as dictionary. Optimized values to be inserted
        in this option when optimizing and later export
    :param option: base budget option / scenario to calculate as dictionary. Optimized values to be inserted
        in this option when optimizing and later export
    :param opt_set: OptionSettings object (option setting: target period etc.) or list of OptionSettings objects (one per model)
    :param trg: target (KPI) value to achieve with minimum budget
    :param trg_tol_ratio: tolerance ratio for target (KPI) value: how small difference consider as nearly equal
    :param if_exact: apply only to the specific time period, without next years
    :param if_long: optimize long or short KPI. If `None`, `if_exact` is used
    :param if_volume: optimize volume or value KPI
    :param epochs: number of epochs
    :param iters_epoch: number of objective function calculations per epoch
    :param error_score: extremely big value to be used as score if fit_predict returns None (error)
    :param ftol_abs: early stop threshold: minimum absolute objective function return value gain per iteration
    :param xtol_abs: early stop threshold: minimum absolute change of independent variable
    :param verbose: print diagnostic or progress information
    :return: optimal option as dict
    """

    if if_long is None:
        if_long = not if_exact

    borders = pd.DataFrame(borders_dict)
    option = copy.deepcopy(option)

    borders_fixed = borders[list(borders.columns[borders.nunique() == 1].values)]
    borders = borders[list(borders.columns[borders.nunique() != 1].values)]

    option = dict_multifill(option, borders_fixed.columns, borders_fixed.iloc[2])

    for i in range(epochs):

        def objective_fun(params_cur: list | np.ndarray | pd.Series | None = None
                          , solution_idx=None):
            return sum(params_cur)

        constraint_fun = ConstraintFunctionTarget(ds=ds
                                                 , model=model
                                                 , model_rep=model_rep
                                                 , option=option
                                                 , opt_set=opt_set
                                                 , if_exact=if_exact
                                                 , if_long=if_long
                                                 , if_volume=if_volume
                                                 , error_score=error_score
                                                 , verbose=verbose
                                                 , borders=borders
                                                 , trg=trg)

        start_values = borders.iloc[
            2].values  # {"bdg_tv" : [0, 30, 10], "bdg_dgt" : [5, 50, 7]}  <- take [10 , 7] from this

        opt_obj = nlopt.opt(nlopt.LN_COBYLA, len(start_values))

        opt_obj.set_min_objective(objective_fun)
        opt_obj.set_lower_bounds(
            borders.iloc[0].values)  # {"bdg_tv" : [0, 30, 10], "bdg_dgt" : [5, 50, 7]} <- take [0 , 5]  from this
        opt_obj.set_upper_bounds(
            borders.iloc[1].values)  # {"bdg_tv" : [0, 30, 10], "bdg_dgt" : [5, 50, 7]} <- take [30 , 50]  from this

        opt_obj.add_inequality_constraint(constraint_fun,
                                          trg_tol_ratio * trg)  # https://nlopt.readthedocs.io/en/latest/NLopt_Python_Reference/#bound-constraints

        opt_obj.set_ftol_abs(ftol_abs)
        if xtol_abs is not None:
            opt_obj.set_xtol_abs(xtol_abs)

        opt_obj.set_maxeval(iters_epoch)
        opt_obj.set_maxtime(0)

        try:
            # with suppress(nlopt.RoundoffLimited):
            rtrn = opt_obj.optimize(start_values)
        except (nlopt.RoundoffLimited, ValueError):
            rtrn = start_values
            logging.warning("Epoche terminated unsuccessfully due to ROUNDOFF error")

        print('Epoche ' + str(i + 1) + ' finished')

        if (isinstance(rtrn, np.ndarray)) or (isinstance(rtrn, list)):
            option = dict_multifill(option, list(borders.keys()), rtrn)
            borders.iloc[2] = rtrn
        else:
            logging.warning("Epoche did not terminate successfully")

    option['bdg'] = sum(v for k, v in option.items() if k.startswith('bdg_') and k != 'bdg')

    return option


def optimize_target_local_ncma(ds: pd.DataFrame | list
                                , model: "Model | list"
                                , model_rep: "ModelRep | list"
                                , borders_dict: dict
                                , option: dict
                                , opt_set: "OptionSettings | list"
                                , trg: float = 100
                                , trg_tol_ratio: float = 0.0001
                                , if_exact: bool = False
                                , if_long: bool | None = None
                                , if_volume: bool = True
                                , epochs: int = 1
                                , iters_epoch: int = 1000
                                , error_score: float = 1e+12
                                , cores: int = 4
                                , verbose: bool = True):
    """
    Minimize budget size to achieve known KPI via CMA-ES algorithm using nevergrad library.
    CMA-ES is a derivative-free global optimizer that maintains search history between iterations.
    It can handle bounds and is robust to noisy objective functions.

    Epochs are required to shuffle params a bit and to help algo get out from local optimum (sometimes)
    and to speed up calculation.

    :param ds: dataset or list of datasets
    :param model: Model object or list of Model objects
    :param model_rep: ModelRep object (export settings) or list of ModelRep objects
    :param borders_dict: dictionary of low and upper borders for every variable to vary
    :param option: base budget option / scenario to calculate as dictionary. Optimized values to be inserted
        in this option when optimizing and later export
    :param opt_set: OptionSettings object (option setting: target period etc.) or list of OptionSettings objects (one per model)
    :param trg: target (KPI) value to achieve with minimum budget
    :param trg_tol_ratio: tolerance ratio for target (KPI) value: how small difference consider as nearly equal
    :param if_exact: apply only to the specific time period, without next years
    :param if_long: optimize long or short KPI. If `None`, `if_exact` is used
    :param if_volume: optimize volume or value KPI
    :param epochs: number of epochs
    :param iters_epoch: number of objective function calculations per epoch
    :param error_score: extremely big value to be used as score if fit_predict returns None (error)
    :param cores: processor cores to be used in parallel computing
    :param verbose: print diagnostic or progress information
    :return: optimal option as dict
    """

    if if_long is None:
        if_long = not if_exact

    option = copy.deepcopy(option)
    borders = pd.DataFrame(borders_dict)

    borders_fixed = borders[list(borders.columns[borders.nunique() == 1].values)]
    borders = borders[list(borders.columns[borders.nunique() != 1].values)]

    option = dict_multifill(option, borders_fixed.columns, borders_fixed.iloc[2])

    # Setup for CMA-ES

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

    # destroy callable objects for multiprocessing

    model_iter = copy.deepcopy(model)

    if isinstance(model_iter, list):
        for i, val in enumerate(model_iter):
            model_iter[i].obj.transform_lhs_fn = None
            model_iter[i].obj.custom_predict_fn = None
    else:
        model_iter.obj.transform_lhs_fn = None
        model_iter.obj.custom_predict_fn = None

    with ProcessPoolExecutor(max_workers=cores) as executor:

        for epoch in range(epochs):

            # Create constraint function
            constraint_fun = ConstraintFunctionTarget(ds=ds
                                                     , model=model_iter
                                                     , model_rep=model_rep
                                                     , option=option
                                                     , opt_set=opt_set
                                                     , if_exact=if_exact
                                                     , if_long=if_long
                                                     , if_volume=if_volume
                                                     , error_score=error_score
                                                     , verbose=False
                                                     , borders=borders
                                                     , trg=trg)

            # Setup search space
            search_space = ng.p.Instrumentation(**{
                name: ng.p.Scalar(lower=low, upper=high, init=val)
                for name, low, high, val in zip(param_names, lwr, upr, start_values)
            })

            # Create wrapper for nevergrad with target constraint
            objective_ng = ObjectiveTargetNG(param_names=param_names
                                            , constraint_fun=constraint_fun
                                            , trg=trg
                                            , trg_tol_ratio=trg_tol_ratio)

            # Initialize CMA-ES optimizer
            optimizer = ng.optimizers.CMA(
                parametrization=search_space,
                budget=iters_epoch,
                num_workers=cores,
            )

            # Verbose callback
            # w/o visibility of constraint data, but of some use anyway to track optimization progress
            if verbose:
                verbose_cb = VerboseNG(10)
                optimizer.register_callback('tell', verbose_cb)

            # Run optimization
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

            # Update parameters
            if isinstance(best_params, tuple):
                params = [best_params[1][par] for par in param_names]
                
                option = dict_multifill(option, param_names, params)
                borders.iloc[2] = params
                start_values = np.array(params)
            else:
                logging.warning("Epoch did not terminate successfully")

    option['bdg'] = sum(v for k, v in option.items() if k.startswith('bdg_') and k != 'bdg')

    return option
