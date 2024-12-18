"""
Algorithmic optimization of the budget split: either maximising KPI with
known budget or minimising budget with known KPI.

As for now COBYLA constrained optimization by local approximations is used
as fairly well suited for optimization with complex and not known in advance
model structure.

Beware! Multiprocessing calculations are used.
"""


import copy
import datetime
import logging
import math
import pandas as pd
import numpy as np

import nlopt

from fermatrica_utils import dict_multifill

from fermatrica_rep.options.define import OptionSettings
import fermatrica_rep.options.calc as calc


def _objective_fun_generate(ds: pd.DataFrame | list
                            , model: "Model | list"
                            , model_rep: "ModelRep | list"
                            , borders: pd.DataFrame
                            , option: dict
                            , opt_set: "OptionSettings"
                            , if_exact: bool = False
                            , if_volume: bool = True
                            , error_score: float = 1e+12
                            , verbose: bool = True):
    """
    Objective function generator / fabric. Variables required by model to run could be assigned to
    the closure one time instead of passing every iteration. Works fine in single-process environment
    only, so to be used for local algorithms mostly.

    :param ds: dataset or list of datasets
    :param model: Model object or list Model objects
    :param model_rep: ModelRep object (export settings) or list of ModelRep objects
    :param borders: dataframe of low and upper borders for every variable to vary
    :param option: base budget option / scenario to calculate as dictionary. Optimized values to be inserted
        in this option when optimizing and later export
    :param opt_set: OptionSettings object (option setting: target period etc.)
    :param if_exact: apply only to the specific time period, without next years
    :param if_volume: optimize volume or value KPI
    :param error_score: extremely big (or small) value to be used as score if fit_predict returns None (error)
    :param verbose: print diagnostic / progress info
    :return: generated objective function
    """

    # set optimising / data environment in closure because some algos cannot read additional arguments

    iter_idx = 1

    def objective_fun(params_cur: list | np.ndarray | pd.Series | None = None
                      , solution_idx=None):
        """
        Objective function with environment variables from closure.

        :param params_cur: current params vector
        :param solution_idx: not used explicitly
        :return: current score
        """

        # use `nonlocal` to access data stored in closure

        nonlocal model
        nonlocal model_rep
        nonlocal option
        nonlocal opt_set
        nonlocal ds
        nonlocal verbose
        nonlocal error_score
        nonlocal iter_idx
        nonlocal borders
        nonlocal if_exact
        nonlocal if_volume

        model = copy.deepcopy(model)

        # apply current values (from algo) to params frame (for option_report function)
        if np.isnan(params_cur).any():
            return error_score

        if isinstance(params_cur, pd.Series):
            params_cur = params_cur.values

        if params_cur is not None:
            option = dict_multifill(option, borders.columns, params_cur)

        ds, ds_pred, opt_sum = calc.option_report(model, ds, model_rep,
                                                  option, opt_set, if_exact=if_exact)

        if isinstance(model_rep, list):
            opt_sum = opt_sum[-1]

        if if_exact:
            if if_volume:
                score = opt_sum["pred_exact_0"]["pred_exact_vol"]
            else:
                score = opt_sum["pred_exact_0"]["pred_exact_val"]
        else:
            if if_volume:
                score = opt_sum["pred_long_vol"]
            else:
                score = opt_sum["pred_long_val"]

        if (score is None) or (math.isinf(score)):
            score = error_score

        # print info

        if verbose:
            if iter_idx % 10 == 0:
                score_print = round(score, 5) if abs(score) < 500 else round(score, 0)
                print('Total budget current : ' + str(sum(params_cur)))
                print('Scoring : ' + str(score_print) + ' ; step : ' + str(
                    iter_idx) + ' : ' + datetime.datetime.now().isoformat(sep=" ", timespec="seconds"))
            iter_idx += 1

        return -score

    return objective_fun


def _constraint_fun_generate_target(ds: pd.DataFrame | list
                                    , model: "Model | list"
                                    , model_rep: "ModelRep | list"
                                    , borders: pd.DataFrame
                                    , option: dict
                                    , opt_set: "OptionSettings"
                                    , trg: float = 100
                                    , if_exact: bool = False
                                    , if_volume: bool = True
                                    , error_score: float = 1e+12
                                    , verbose: bool = True
                                    ):
    """
    Inequality constraints function generator / fabric. Variables required by model to run could be assigned to
    the closure one time instead of passing every iteration. Works fine in single-process environment
    only, so to be used for local algorithms mostly.

    For optimization with known KPI value objective function is rather simple and equals to budget size. All real
    calculations are moved to inequality constraints function.

    :param ds: dataset or list of datasets
    :param model: Model object or list of Model objects
    :param model_rep: ModelRep object (export settings) or list of ModelRep objects
    :param borders: dataframe of low and upper borders for every variable to vary
    :param option: base budget option / scenario to calculate as dictionary. Optimized values to be inserted
        in this option when optimizing and later export
    :param opt_set: OptionSettings object (option setting: target period etc.)
    :param trg: target KPI value
    :param if_exact: apply only to the specific time period, without next years
    :param if_volume: optimize volume or value KPI
    :param error_score: extremely big (or small) value to be used as score if fit_predict returns None (error)
    :param verbose: print diagnostic / progress info
    :return: generated objective function
    :return:
    """

    # set optimising / data environment in closure because some algos cannot read additional arguments

    iter_idx = 1

    def constraint_fun(params_cur: list | np.ndarray | pd.Series | None = None
                       , solution_idx=None):
        """
        Inequality constraints function with environment variables from closure.

        :param params_cur: current params vector
        :param solution_idx: not used explicitly
        :return: current score
        """

        # use `nonlocal` to access data stored in closure

        nonlocal model
        nonlocal model_rep
        nonlocal option
        nonlocal opt_set
        nonlocal ds
        nonlocal error_score
        nonlocal verbose
        nonlocal iter_idx
        nonlocal borders
        nonlocal if_exact
        nonlocal if_volume

        model = copy.deepcopy(model)

        # apply current values (from algo) to params frame (for option_report function)
        if np.isnan(params_cur).any():
            return error_score

        if isinstance(params_cur, pd.Series):
            params_cur = params_cur.values

        # params_cur = np.round(params_cur, 2)

        if params_cur is not None:
            option = dict_multifill(option, borders.columns, params_cur)

        ds, ds_pred, opt_sum = calc.option_report(model, ds, model_rep,
                                                  option, opt_set, if_exact=if_exact)

        if isinstance(model_rep, list):
            opt_sum = opt_sum[-1]

        if if_exact:
            if if_volume:
                score = opt_sum["pred_exact_0"]["pred_exact_vol"]
            else:
                score = opt_sum["pred_exact_0"]["pred_exact_val"]
        else:
            if if_volume:
                score = opt_sum["pred_long_vol"]
            else:
                score = opt_sum["pred_long_val"]

        if (score is None) or (math.isinf(score)):
            score = error_score

        # print info
        if verbose:
            if iter_idx % 10 == 0:
                score_print = round(score, 5) if abs(score) < 500 else round(score, 0)
                print('Total budget current : ' + str(sum(params_cur)))
                print('Scoring : ' + str(score_print) + ' ; step : ' + str(
                    iter_idx) + ' : ' + datetime.datetime.now().isoformat(sep=" ", timespec="seconds"))
            iter_idx += 1

        return abs(abs(score) - trg)

    return constraint_fun


def optimize_budget_local_cobyla(ds: pd.DataFrame | list
                                 , model: "Model | list"
                                 , model_rep: "ModelRep | list"
                                 , borders_dict: dict
                                 , option: dict
                                 , opt_set: "OptionSettings"
                                 , bdg_size: float = 100
                                 , if_exact: bool = False
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
    :param opt_set: OptionSettings object (option setting: target period etc.)
    :param bdg_size: fixed budget size (we change budget split, but not budget size)
    :param if_exact: apply only to the specific time period, without next years
    :param if_volume: optimize volume or value KPI
    :param epochs: number of epochs
    :param iters_epoch: number of objective function calculations per epoch
    :param error_score: extremely big value to be used as score if fit_predict returns None (error)
    :param ftol_abs: early stop threshold: minimum absolute objective function return value gain per iteration
    :param xtol_abs: early stop threshold: minimum absolute change of independent variable
    :param verbose: print diagnostic or progress information
    :return: optimal option as dict
    """

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

        objective_fun = _objective_fun_generate(ds=ds
                                                , model=model
                                                , model_rep=model_rep
                                                , option=option
                                                , opt_set=opt_set
                                                , if_exact=if_exact
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

    return option


def optimize_target_local_cobyla(ds: pd.DataFrame | list
                                 , model_rep: "ModelRep | list"
                                 , model: "Model | list"
                                 , borders_dict: dict
                                 , opt_set
                                 , option: dict
                                 , trg: float = 100
                                 , trg_tol_ratio: float = 0.0001
                                 , if_exact: bool = False
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
    :param opt_set: OptionSettings object (option setting: target period etc.)
    :param trg: target (KPI) value to achieve with minimum budget
    :param trg_tol_ratio: tolerance ratio for target (KPI) value: how small difference consider as nearly equal
    :param if_exact: apply only to the specific time period, without next years
    :param if_volume: optimize volume or value KPI
    :param epochs: number of epochs
    :param iters_epoch: number of objective function calculations per epoch
    :param error_score: extremely big value to be used as score if fit_predict returns None (error)
    :param ftol_abs: early stop threshold: minimum absolute objective function return value gain per iteration
    :param xtol_abs: early stop threshold: minimum absolute change of independent variable
    :param verbose: print diagnostic or progress information
    :return: optimal option as dict
    """

    borders = pd.DataFrame(borders_dict)
    option = copy.deepcopy(option)

    borders_fixed = borders[list(borders.columns[borders.nunique() == 1].values)]
    borders = borders[list(borders.columns[borders.nunique() != 1].values)]

    option = dict_multifill(option, borders_fixed.columns, borders_fixed.iloc[2])

    for i in range(epochs):

        def objective_fun(params_cur: list | np.ndarray | pd.Series | None = None
                          , solution_idx=None):
            return sum(params_cur)

        constraint_fun = _constraint_fun_generate_target(ds=ds
                                                         , model=model
                                                         , model_rep=model_rep
                                                         , option=option
                                                         , opt_set=opt_set
                                                         , if_exact=if_exact
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

    return option
