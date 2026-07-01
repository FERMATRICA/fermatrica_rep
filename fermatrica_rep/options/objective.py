import copy
import datetime
import math
import numpy as np
import pandas as pd

from fermatrica import Model
from fermatrica_utils import dict_multifill

import fermatrica_rep.options.calc as calc
from fermatrica_rep.options.define import OptionSettings


class ObjectiveBudgetNG:
    '''
    Wrapper for nevergrad objective function to apply budget constraint penalty. Variables required by model to run
    '''

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
    '''
    Wrapper for nevergrad objective function to minimize budget while achieving target KPI. 
    Variables required by model to run
    '''

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
    Nevergrad callback to print progress
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
    Objective function callable class for budget optimization. Variables required by model to run
    are assigned to the instance.
    """

    def __init__(self
                 , ds: pd.DataFrame | list
                 , model: "Model | list"
                 , model_rep: "ModelRep | list"
                 , borders: pd.DataFrame
                 , option: dict
                 , opt_set: "OptionSettings | list"
                 , if_exact: bool = False
                 , if_long: bool | None = None
                 , if_volume: bool = True
                 , error_score: float = 1e+12
                 , verbose: bool = True
                 ):
        """
        :param ds: dataset or list of datasets
        :param model: Model object or list Model objects
        :param model_rep: ModelRep object (export settings) or list of ModelRep objects
        :param borders: dataframe of low and upper borders for every variable to vary
        :param option: base budget option / scenario to calculate as dictionary. Optimized values to be inserted
            in this option when optimizing and later export
        :param opt_set: OptionSettings object (option setting: target period etc.) or list of OptionSettings objects (one per model)
        :param if_exact: apply only to the specific time period, without next years
        :param if_long: optimize long or short KPI. If `None`, `if_exact` is used
        :param if_volume: optimize volume or value KPI
        :param error_score: extremely big (or small) value to be used as score if fit_predict returns None (error)
        :param verbose: print diagnostic / progress info
        # :param if_multiprocess: use multiprocessing for calculations
        """
        self.ds = ds
        self.model = model
        self.model_rep = model_rep
        self.borders = borders
        self.option = option
        self.opt_set = opt_set
        self.if_exact = if_exact
        self.if_long = if_long if if_long is not None else not if_exact
        self.if_volume = if_volume
        self.error_score = error_score
        self.verbose = verbose
        self.iter_idx = 1

    def __call__(self
                 , params_cur: list | np.ndarray | pd.Series | None = None
                 , solution_idx=None):
        """
        Objective function with environment variables from instance.

        :param params_cur: current params vector
        :param solution_idx: not used explicitly
        :return: current score
        """

        model = copy.deepcopy(self.model)

        # apply current values (from algo) to params frame (for option_report function)
        if np.isnan(params_cur).any():
            return self.error_score

        if isinstance(params_cur, pd.Series):
            params_cur = params_cur.values

        if params_cur is not None:
            option = dict_multifill(self.option, self.borders.columns, params_cur)
        else:
            option = self.option

        ds, ds_pred, opt_sum = calc.option_report(model, self.ds, self.model_rep,
                                                  option, self.opt_set, if_exact=self.if_exact)

        if isinstance(self.model_rep, list):
            opt_sum = opt_sum[-1]

        if self.if_long:
            if self.if_volume:
                score = opt_sum["pred_long_vol"]
            else:
                score = opt_sum["pred_long_val"]
        else:
            if self.if_volume:
                score = opt_sum["pred_exact_0"]["pred_exact_vol"]
            else:
                score = opt_sum["pred_exact_0"]["pred_exact_val"]

        if (score is None) or (math.isinf(score)):
            score = self.error_score

        # print info

        if self.verbose:
            if self.iter_idx % 10 == 0:
                score_print = round(score, 5) if abs(score) < 500 else round(score, 0)
                print('Total budget current : ' + str(sum(params_cur)))
                print('Scoring : ' + str(score_print) + ' ; step : ' + str(
                    self.iter_idx) + ' : ' + datetime.datetime.now().isoformat(sep=" ", timespec="seconds"))
            self.iter_idx += 1

        return -score


class ConstraintFunctionTarget:
    """
    Inequality constraints function callable class for target optimization. Variables required by model to run
    are assigned to the instance.

    For optimization with known KPI value objective function is rather simple and equals to budget size. All real
    calculations are moved to inequality constraints function.
    """

    def __init__(self
                 , ds: pd.DataFrame | list
                 , model: "Model | list"
                 , model_rep: "ModelRep | list"
                 , borders: pd.DataFrame
                 , option: dict
                 , opt_set: "OptionSettings | list"
                 , trg: float = 100
                 , if_exact: bool = False
                 , if_long: bool | None = None
                 , if_volume: bool = True
                 , error_score: float = 1e+12
                 , verbose: bool = True):
        """
        :param ds: dataset or list of datasets
        :param model: Model object or list of Model objects
        :param model_rep: ModelRep object (export settings) or list of ModelRep objects
        :param borders: dataframe of low and upper borders for every variable to vary
        :param option: base budget option / scenario to calculate as dictionary. Optimized values to be inserted
            in this option when optimizing and later export
        :param opt_set: OptionSettings object (option setting: target period etc.) or list of OptionSettings objects (one per model)
        :param trg: target KPI value
        :param if_exact: apply only to the specific time period, without next years
        :param if_long: optimize long or short KPI. If `None`, `if_exact` is used
        :param if_volume: optimize volume or value KPI
        :param error_score: extremely big (or small) value to be used as score if fit_predict returns None (error)
        :param verbose: print diagnostic / progress info
        """
        self.ds = ds
        self.model = model
        self.model_rep = model_rep
        self.borders = borders
        self.option = option
        self.opt_set = opt_set
        self.trg = trg
        self.if_exact = if_exact
        self.if_long = if_long if if_long is not None else not if_exact
        self.if_volume = if_volume
        self.error_score = error_score
        self.verbose = verbose
        self.iter_idx = 1

    def __call__(self
                 , params_cur: list | np.ndarray | pd.Series | None = None
                 , solution_idx=None):
        """
        Inequality constraints function with environment variables from instance.

        :param params_cur: current params vector
        :param solution_idx: not used explicitly
        :return: current score
        """

        model = copy.deepcopy(self.model)

        # apply current values (from algo) to params frame (for option_report function)
        if np.isnan(params_cur).any():
            return self.error_score

        if isinstance(params_cur, pd.Series):
            params_cur = params_cur.values

        # params_cur = np.round(params_cur, 2)

        if params_cur is not None:
            option = dict_multifill(self.option, self.borders.columns, params_cur)
        else:
            option = self.option

        ds, ds_pred, opt_sum = calc.option_report(model, self.ds, self.model_rep,
                                                  option, self.opt_set, if_exact=self.if_exact)

        if isinstance(self.model_rep, list):
            opt_sum = opt_sum[-1]

        if self.if_long:
            if self.if_volume:
                score = opt_sum["pred_long_vol"]
            else:
                score = opt_sum["pred_long_val"]
        else:
            if self.if_volume:
                score = opt_sum["pred_exact_0"]["pred_exact_vol"]
            else:
                score = opt_sum["pred_exact_0"]["pred_exact_val"]

        if (score is None) or (math.isinf(score)):
            score = self.error_score

        # print info
        if self.verbose:
            if self.iter_idx % 10 == 0:
                score_print = round(score, 5) if abs(score) < 500 else round(score, 0)
                print('Total budget current : ' + str(sum(params_cur)))
                print('Scoring : ' + str(score_print) + ' ; step : ' + str(
                    self.iter_idx) + ' : ' + datetime.datetime.now().isoformat(sep=" ", timespec="seconds"))
            self.iter_idx += 1

        return abs(abs(score) - self.trg)