"""
Calculate option: given known media and non-media specification for some period
(mostly future one), apply it to the dataset, calculate prediction and
get summaries.

Main steps of the calculation process are:
1. Translate option into real data (yearly budget to weekly/monthly OTS etc.)
2. Run transformation and prediction with updated data
3. Summarize results to get quick access to business valuable metrics
    (yearly sales, uplifts etc.)

If there are many options, multiprocessing could be used, so beware to use it
in the environment allowing multiprocessing.
"""


import copy
import inspect
import itertools
import logging
import multiprocessing

from fermatrica_utils import import_module_from_string
from fermatrica.model.model import Model, ModelConf
from fermatrica.model.predict import predict_ext
from fermatrica.model.transform import transform

from fermatrica_rep.options.define import OptionSettings
from fermatrica_rep.options.translators import *


"""
Translation basics
"""


def _option_translate_exact(model_conf: "ModelConf"
                            , ds: pd.DataFrame
                            , model_rep  #: "ModelRep"
                            , option_dict: dict
                            , option_settings: "OptionSettings"
                            , allow_past: bool = False) -> pd.DataFrame:
    """
    Translate specific option to the specific period.

    :param model_conf: ModelConf object (part of Model). Used instead of Model to boost performance
        in situation where user's code is not expected
    :param ds: dataset
    :param model_rep: ModelRep object (export settings)
    :param option_dict: budget option / scenario to calculate as dictionary
    :param option_settings: OptionSettings object (option setting: target period etc.)
    :param allow_past: allow translation to the past (filled with observed data) periods
    :return: dataset `ds` with applied option
    """

    # prepare

    if 'text_flags' in option_dict.keys():
        text_flags = option_dict['text_flags']
    else:
        text_flags = {}

    option_dict = {k: v for k, v in option_dict.items() if k not in ['text_flags', 'bdg']}

    # load adhoc code

    if pd.notna(model_rep.adhoc_code_src) and len(model_rep.adhoc_code_src) > 0:
        for k, v in model_rep.adhoc_code_src.items():
            fr_cur_name = inspect.currentframe().f_globals['__name__']
            import_module_from_string(k, v, fr_cur_name)

    # create mask
    ds_mask = True
    for i, (apply_var, trgt) in enumerate(zip(option_settings.apply_vars, option_settings.target)):
        ds_mask = ds_mask * ((ds['date'] >= option_settings.date_start) & (ds['date'] <= option_settings.date_end) &
                             (ds[apply_var].isin(trgt)))

    if not allow_past:
        ds_mask = ds_mask & (ds['listed'].isin([4]))

    # run

    for k, val in option_dict.items():

        if k not in model_rep.trans_dict['function'].keys():
            continue

        fun_name = model_rep.trans_dict['function'][k]
        var_name = model_rep.trans_dict['variable'][k]

        if fun_name not in locals() and fun_name not in globals() and \
                not (len(fun_name.split('.')) > 1 and getattr(globals()['.'.join(fun_name.split('.')[:-1])],
                                                              fun_name.split('.')[-1], False)):
            continue

        ds = eval(fun_name)(model_conf, ds, model_rep, k, var_name, val, ds_mask)

    return ds


def _option_translate_list(model_conf: "ModelConf"
                           , ds: pd.DataFrame
                           , model_rep  #: "ModelRep"
                           , option_dict_list: list  # list of budget options / scenarios to calculate
                           , option_settings_list: list  # list of option settings: target period etc.
                           , allow_past: bool = False) -> pd.DataFrame:
    """
    Translate the list of options to the  list of periods. It could be the list of identical options for
    different periods (such as used in `option_translate_long`) or truly independent options for
    different periods (say 5M TV in year 1 and 7.5M TV in year 2).

    :param model_conf: ModelConf object (part of Model). Used instead of Model to boost performance
        in situation where user's code is not expected
    :param ds: dataset
    :param model_rep: ModelRep object (export settings)
    :param option_dict_list: list of budget options / scenarios to calculate, each as dictionary
    :param option_settings_list: list of OptionSettings objects (option setting: target period etc.)
    :param allow_past: allow translation to the past (filled with observed data) periods
    :return: dataset `ds` with applied options
    """

    for i, val in enumerate(option_dict_list):
        ds = _option_translate_exact(model_conf, ds, model_rep, option_dict_list[i], option_settings_list[i], allow_past)

    return ds


def option_translate_long(model_conf: "ModelConf"
                          , ds: pd.DataFrame
                          , model_rep  #: "ModelRep"
                          , option_dict: dict
                          , option_settings: "OptionSettings"
                          , allow_past: bool = False) -> pd.DataFrame:
    """
    Translate specific option to the specific period and corresponding periods in next years.

    IMPORTANT. Do not call this function directly, add to ModelRep and use as ModelRep.option_translate_long().
    This approach allows to use adhoc long-translation function if necessary.

    :param model_conf: ModelConf object (part of Model). Used instead of Model to boost performance
        in situation where user's code is not expected
    :param ds: dataset
    :param model_rep: ModelRep object (export settings)
    :param option_dict: budget option / scenario to calculate as dictionary
    :param option_settings: OptionSettings object (option setting: target period etc.)
    :param allow_past: allow translation to the past (filled with observed data) periods
    :return: dataset `ds` with applied option

    """

    option_dict_list = [option_dict, option_dict]

    option_settings_long = copy.deepcopy(option_settings)

    option_settings_long.date_start = option_settings_long.date_end + pd.DateOffset(1)
    option_settings_long.date_end = ds.loc[ds['listed'] == 4, 'date'].max()
    option_settings_long.plan_period = 'year'

    option_settings_list = [option_settings, option_settings_long]

    ds = _option_translate_list(model_conf, ds, model_rep, option_dict_list, option_settings_list, allow_past)

    return ds


"""
Predict & summaries
"""


def _option_report_worker(model: "Model"
                          , ds: pd.DataFrame
                          , model_rep  #: "ModelRep"
                          , option_dict: dict
                          , option_settings: "OptionSettings"
                          , targets_new: dict | None = None
                          , allow_past: bool = False
                          , if_exact: bool = False
                          , if_multi: bool = False) -> tuple:

    """
    Worker function to calculate and report specific option (one model in chain only):

    1. Translate option into real data (yearly budget to weekly/monthly OTS etc.)
    2. Run transformation and prediction with updated data
    3. Summarize results to get quick access to business valuable metrics
        (yearly sales, uplifts etc.)

    :param model: Model object or list of Model objects
    :param ds: dataset or list of datasets
    :param model_rep: ModelRep object (export settings) of list of ModelRep objects
    :param option_dict: budget option / scenario to calculate as dictionary or list of option dictionaries
    :param option_settings: OptionSettings object (option setting: target period etc.) or list of OptionSettings
    :param targets_new: apply option to one entity, summarize another (useful for cross-elasticity). If not None:
        {'targets_new': [], 'apply_vars_new': []}
    :param allow_past: allow translation to the past (filled with observed data) periods
    :param if_exact: apply only to the specific time period, without next years
    :param if_multi: return only summary, without data and prediction (use if a lot of options are calculated
        and only summaries are actually necessary)
    :return: (1, summary) or (dataset, prediction data, summary). 1 is required for technical reasons
    """

    model_conf = model.conf

    # translate budgets into ds

    if isinstance(option_settings.target, str):
        option_settings.target = [[option_settings.target]]

    if isinstance(option_settings.target, list) and isinstance(option_settings.target[0], str):
        option_settings.target = [option_settings.target]

    #

    if if_exact:
        ds = _option_translate_exact(model_conf, ds, model_rep, option_dict, option_settings, allow_past)
    else:
        ds = model_rep.option_translate_long(model_conf, ds, model_rep, option_dict, option_settings, allow_past)

    # run transformations and get prediction

    model, ds = transform(ds=ds
                          , model=model
                          , set_start=False
                          , if_by_ref=False)

    dt_pred = predict_ext(model, ds)

    # get summaries

    option_summary = option_summarize(model_conf, model_rep, option_dict, option_settings, dt_pred, targets_new)

    # return

    return ds, dt_pred, option_summary


def option_report(model: "Model | list"
                  , ds: pd.DataFrame | list
                  , model_rep  #: "ModelRep"
                  , option_dict: dict | list
                  , option_settings: "OptionSettings"
                  , targets_new: dict | None = None
                  , allow_past: bool = False
                  , if_exact: bool = False
                  , if_multi: bool = False) -> tuple:
    """
    Calculate and report specific option:

    1. Translate option into real data (yearly budget to weekly/monthly OTS etc.)
    2. Run transformation and prediction with updated data
    3. Summarize results to get quick access to business valuable metrics
        (yearly sales, uplifts etc.)

    Higher-level function to be used directly in user's pipeline.

    :param model: Model object or list of Model objects
    :param ds: dataset or list of datasets
    :param model_rep: ModelRep object (export settings) of list of ModelRep objects
    :param option_dict: budget option / scenario to calculate as dictionary or list of option dictionaries
    :param option_settings: OptionSettings object (option setting: target period etc.) or list of OptionSettings
    :param targets_new: apply option to one entity, summarize another (useful for cross-elasticity). If not None:
        {'targets_new': [], 'apply_vars_new': []}
    :param allow_past: allow translation to the past (filled with observed data) periods
    :param if_exact: apply only to the specific time period, without next years
    :param if_multi: return only summary, without data and prediction (use if a lot of options are calculated
        and only summaries are actually necessary)
    :return: (1, summary) or (dataset, prediction data, summary). 1 is required for technical reasons
    """

    if isinstance(model_rep, list):

        dt_pred = [None] * len(model_rep)
        option_summary = [None] * len(model_rep)

        for i, val in enumerate(model_rep):

            # fuse output of previous model into the current one

            if i > 0:

                if pd.notna(model_rep[i].adhoc_code_src) and len(model_rep[i].adhoc_code_src) > 0:

                    for k, v in model_rep[i].adhoc_code_src.items():
                        fr_cur_name = inspect.currentframe().f_globals['__name__']
                        import_module_from_string(k, v, fr_cur_name)

                    fun_name = 'code_py.adhoc.reporting.pred_merge_' + str(i)

                    if (fun_name in locals() or fun_name in globals() or
                            (len(fun_name.split('.')) > 1 and getattr(globals()['.'.join(fun_name.split('.')[:-1])],
                                                                      fun_name.split('.')[-1], False))):

                        ds[i] = eval(fun_name)(ds_cur=ds[i]
                                            , dt_pred_prev=dt_pred[i-1]
                                            , allow_past=allow_past)

            # run

            ds[i], dt_pred[i], option_summary[i] = _option_report_worker(model=model[i]
                                                                         , ds=ds[i]
                                                                         , model_rep=model_rep[i]
                                                                         , option_dict=option_dict
                                                                         , option_settings=option_settings
                                                                         , targets_new=targets_new
                                                                         , allow_past=allow_past
                                                                         , if_exact=if_exact
                                                                         , if_multi=if_multi)

    else:
        ds, dt_pred, option_summary = _option_report_worker(model=model
                                     , ds=ds
                                     , model_rep=model_rep
                                     , option_dict=option_dict
                                     , option_settings=option_settings
                                     , targets_new=targets_new
                                     , allow_past=allow_past
                                     , if_exact=if_exact
                                     , if_multi=if_multi)


    if if_multi:
        del ds, dt_pred
        return 1, option_summary
    else:
        return ds, dt_pred, option_summary


def option_summarize_exact(model_conf: "ModelConf"
                           , option_settings
                           , dt_pred: pd.DataFrame):
    """
    Summarize option for the exact period of translation.

    :param model_conf: ModelConf object (part of Model). Used instead of Model to boost performance
        in situation where user's code is not expected
    :param option_settings: OptionSettings object (option setting: target period etc.)
    :param dt_pred: prediction data (table)
    :return: summary dictionary
    """

    summary_type = 'sum'

    if hasattr(model_conf, 'summary_type'):
        summary_type = model_conf.summary_type

    option_summary = {}

    # exact period

    if summary_type == 'fin':

        dt_mask = True
        for i, (apply_var, trgt) in enumerate(zip(option_settings.apply_vars, option_settings.target)):
            dt_mask = dt_mask * ((dt_pred['listed'].isin([2, 3, 4])) & (dt_pred['date'] == option_settings.date_end) & (
                dt_pred[apply_var].isin(trgt)))

        option_summary['pred_exact_vol'] = dt_pred.loc[dt_mask, 'predicted'].sum()

        if hasattr(model_conf, 'price_var') and pd.notna(model_conf.price_var) and model_conf.price_var != '':
            option_summary['pred_exact_val'] = dt_pred.loc[dt_mask, 'predicted'] * dt_pred.loc[
                dt_mask, model_conf.price_var]
            option_summary['pred_exact_val'] = option_summary['pred_exact_val'].sum()

    elif summary_type == 'mean_fin':

        dt_mask = True
        for i, (apply_var, trgt) in enumerate(zip(option_settings.apply_vars, option_settings.target)):
            dt_mask = dt_mask * ((dt_pred['listed'].isin([2, 3, 4])) & (dt_pred['date'] == option_settings.date_end) & (
                dt_pred[apply_var].isin(trgt)))

        option_summary['pred_exact_vol'] = dt_pred.loc[dt_mask, 'predicted'].mean()

        if hasattr(model_conf, 'price_var') and pd.notna(model_conf.price_var) and model_conf.price_var != '':
            option_summary['pred_exact_val'] = dt_pred.loc[dt_mask, 'predicted'] * dt_pred.loc[
                dt_mask, model_conf.price_var]
            option_summary['pred_exact_val'] = option_summary['pred_exact_val'].mean()

    else:
        dt_mask = True
        for i, (apply_var, trgt) in enumerate(zip(option_settings.apply_vars, option_settings.target)):
            dt_mask = dt_mask * (
                    (dt_pred['listed'].isin([2, 3, 4])) & (dt_pred['date'] >= option_settings.date_start) & (
                    dt_pred['date'] <= option_settings.date_end) & (dt_pred[apply_var].isin(trgt)))

        option_summary['pred_exact_vol'] = dt_pred.loc[dt_mask, 'predicted'].sum()

        if hasattr(model_conf, 'price_var') and pd.notna(model_conf.price_var) and model_conf.price_var != '':
            option_summary['pred_exact_val'] = dt_pred.loc[dt_mask, 'predicted'] * dt_pred.loc[
                dt_mask, model_conf.price_var]
            option_summary['pred_exact_val'] = option_summary['pred_exact_val'].sum()

    return option_summary


def option_summarize(model_conf: "ModelConf"
                     , model_rep
                     , option_dict: dict | list
                     , option_settings: "OptionSettings"
                     , dt_pred: pd.DataFrame
                     , targets_new: dict | None = None) -> dict:
    """
    Summarize option for both exact and long-term periods.

    :param model_conf: ModelConf object (part of Model). Used instead of Model to boost performance
        in situation where user's code is not expected
    :param model_rep: ModelRep object (export settings)
    :param option_dict: budget option / scenario to calculate as dictionary or list of option dictionaries
    :param option_settings: OptionSettings object (option setting: target period etc.) or list of OptionSettings
    :param dt_pred: prediction data
    :param targets_new: apply option to one entity, summarize another (useful for cross-elasticity). If not None:
        {'targets_new': [], 'apply_vars_new': []}
    :return: summary dictionary
    """

    option_settings_copy = copy.deepcopy(option_settings)

    if targets_new:
        option_settings_copy.target = copy.deepcopy(targets_new["target_new"])
        option_settings_copy.apply_vars = copy.deepcopy(targets_new["apply_vars_new"])

    date_max = dt_pred['date'].max()
    date_obs_max = dt_pred.loc[dt_pred['listed'].isin([2, 3]), 'date'].max()
    summary_type = 'sum'

    if hasattr(model_conf, 'summary_type'):
        summary_type = model_conf.summary_type

    option_summary = {}

    if type(option_settings_copy) != list:
        option_settings_copy = [option_settings_copy]

    # exact period

    for k, val in enumerate(option_settings_copy):
        option_summary['pred_exact_' + str(k)] = \
            option_summarize_exact(model_conf, val, dt_pred)

    # long period

    if option_settings_copy[0].date_end < date_max:

        if summary_type == 'fin':

            dt_mask = True
            for i, (apply_var, trgt) in enumerate(
                    zip(option_settings_copy[0].apply_vars, option_settings_copy[0].target)):
                dt_mask = dt_mask * (
                        (dt_pred['listed'].isin([2, 3, 4])) & (dt_pred['date'] == option_settings_copy[0].date_max) & (
                    dt_pred[apply_var].isin(trgt)))

            option_summary['pred_long_vol'] = dt_pred.loc[dt_mask, 'predicted'].sum()

            if hasattr(model_conf, 'price_var') and pd.notna(model_conf.price_var) and model_conf.price_var != '':
                option_summary['pred_long_val'] = dt_pred.loc[dt_mask, 'predicted'] * dt_pred.loc[
                    dt_mask, model_conf.price_var]
                option_summary['pred_long_val'] = option_summary['pred_long_val'].sum()

        elif summary_type == 'mean_fin':

            dt_mask = True
            for i, (apply_var, trgt) in enumerate(
                    zip(option_settings_copy[0].apply_vars, option_settings_copy[0].target)):
                dt_mask = dt_mask * (
                        (dt_pred['listed'].isin([2, 3, 4])) & (dt_pred['date'] == date_max) & (
                    dt_pred[apply_var].isin(trgt)))

            option_summary['pred_long_vol'] = dt_pred.loc[dt_mask, 'predicted'].mean()

            if hasattr(model_conf, 'price_var') and pd.notna(model_conf.price_var) and model_conf.price_var != '':
                option_summary['pred_long_val'] = dt_pred.loc[dt_mask, 'predicted'] * dt_pred.loc[
                    dt_mask, model_conf.price_var]
                option_summary['pred_long_val'] = option_summary['pred_long_val'].mean()

        else:
            dt_mask = True
            for i, (apply_var, trgt) in enumerate(
                    zip(option_settings_copy[0].apply_vars, option_settings_copy[0].target)):
                dt_mask = dt_mask * (
                        (dt_pred['listed'].isin([2, 3, 4])) & (
                            dt_pred['date'] >= option_settings_copy[0].date_start) & (
                            dt_pred[apply_var].isin(trgt)))

            option_summary['pred_long_vol'] = dt_pred.loc[dt_mask, 'predicted'].sum()

            if hasattr(model_conf, 'price_var') and pd.notna(model_conf.price_var) and model_conf.price_var != '':
                option_summary['pred_long_val'] = dt_pred.loc[dt_mask, 'predicted'] * dt_pred.loc[
                    dt_mask, model_conf.price_var]
                option_summary['pred_long_val'] = option_summary['pred_long_val'].sum()

    # previous / reference period
    dt_mask = True
    for i, (apply_var, trgt) in enumerate(zip(option_settings_copy[0].apply_vars, option_settings_copy[0].target)):
        dt_mask = dt_mask * (
                (dt_pred['listed'].isin([2, 3, 4])) & (dt_pred['date'] >= option_settings_copy[0].ref_date_start) & (
                dt_pred['date'] <= option_settings_copy[0].ref_date_end) & (dt_pred[apply_var].isin(trgt)))

    option_summary['pred_ref_vol'] = dt_pred.loc[dt_mask, 'predicted'].sum()

    if hasattr(model_conf, 'price_var') and pd.notna(model_conf.price_var) and model_conf.price_var != '':
        option_summary['pred_ref_val'] = dt_pred.loc[dt_mask, 'predicted'] * dt_pred.loc[dt_mask, model_conf.price_var]
        option_summary['pred_ref_val'] = option_summary['pred_ref_val'].sum()

    if option_settings_copy[0].ref_date_end <= date_obs_max:

        option_summary['ref_vol'] = dt_pred.loc[dt_mask, 'observed'].sum()

        if hasattr(model_conf, 'price_var') and pd.notna(model_conf.price_var) and model_conf.price_var != '':
            option_summary['ref_val'] = dt_pred.loc[dt_mask, 'observed'] * dt_pred.loc[dt_mask, model_conf.price_var]
            option_summary['ref_val'] = option_summary['ref_val'].sum()

    else:
        dt_mask_obs = True
        for i, (apply_var, trgt) in enumerate(zip(option_settings_copy[0].apply_vars, option_settings_copy[0].target)):
            dt_mask_obs = dt_mask_obs * (
                    (dt_pred['listed'].isin([2, 3, 4])) & (
                        dt_pred['date'] >= option_settings_copy[0].ref_date_start) & (
                            dt_pred['date'] <= date_obs_max) & (dt_pred[apply_var].isin(trgt)))

        dt_mask_pred = True
        for i, (apply_var, trgt) in enumerate(zip(option_settings_copy[0].apply_vars, option_settings_copy[0].target)):
            dt_mask_pred = dt_mask_pred * ((dt_pred['listed'].isin([2, 3, 4])) & (dt_pred['date'] > date_obs_max) & (
                    dt_pred['date'] <= option_settings_copy[0].ref_date_end) & (dt_pred[apply_var].isin(trgt)))

        option_summary['ref_vol'] = dt_pred.loc[dt_mask_obs, 'observed'].sum() + dt_pred.loc[
            dt_mask_pred, 'observed'].sum()

        if hasattr(model_conf, 'price_var') and pd.notna(model_conf.price_var) and model_conf.price_var != '':
            tmp_1 = dt_pred.loc[dt_mask_obs, 'observed'] * dt_pred.loc[dt_mask_obs, model_conf.price_var]
            tmp_2 = dt_pred.loc[dt_mask_pred, 'observed'] * dt_pred.loc[dt_mask_pred, model_conf.price_var]
            option_summary['ref_val'] = tmp_1.sum() + tmp_2.sum()

    # growth

    option_summary['growth_vol'] = option_summary['pred_exact_0']['pred_exact_vol'] / option_summary['ref_vol']

    if hasattr(model_conf, 'price_var') and pd.notna(model_conf.price_var) and model_conf.price_var != '':
        option_summary['growth_val'] = option_summary['pred_exact_0']['pred_exact_val'] / option_summary['ref_val']

    # adhoc

    if pd.notna(model_rep.option_summary_adhoc):
        option_summary = model_rep.option_summary_adhoc(model_conf, model_rep, option_dict, option_settings_copy, dt_pred)

    return option_summary


def option_report_multi(model: "Model"
                        , ds: pd.DataFrame
                        , model_rep  #: "ModelRep"
                        , option_dict_d: dict
                        , option_settings: "OptionSettings"
                        , allow_past: bool = False
                        , if_exact: bool = False
                        , if_multi: bool = True
                        , cores: int = 1) -> dict:
    """
    Calculate and report multiple options using multiprocessing.
    Pass options and options settings as dictionary of dictionaries and dictionary of OptionsSettings objects
    respectively.

    Version 1, lesser used.

    :param model: Model object
    :param ds: dataset
    :param model_rep: ModelRep object (export settings)
    :param option_dict_d: dictionary of budget options / scenarios to calculate as dictionary
    :param option_settings: OptionSettings object (option setting: target period etc.)
    :param allow_past: allow translation to the past (filled with observed data) periods
    :param if_exact: apply only to the specific time period, without next years
    :param if_multi: return only summary, without data and prediction (use if a lot of options are calculated
        and only summaries are actually necessary)
    :param cores: number of processor cores to use in calculations
    :return: dictionary of summaries
    """

    if cores > (multiprocessing.cpu_count() - 1):
        cores = (multiprocessing.cpu_count() - 1)
        logging.warning('Cores number for parallel computing is set too high for this computer. ' +
                        'Reset to ' + str(cores))

    if cores > 1:

        with multiprocessing.Pool(cores) as pool:

            opt_count = len(option_dict_d)
            opt_keys = option_dict_d.keys()
            opt_list = [option_dict_d[x] for x in opt_keys]

            option_summaries = {}

            for key, rtrn in zip(opt_keys
                    , pool.starmap(option_report
                        , zip(itertools.repeat(model, opt_count)
                            , itertools.repeat(ds, opt_count)
                            , itertools.repeat(model_rep, opt_count)
                            , opt_list
                            , itertools.repeat(option_settings, opt_count)
                            , itertools.repeat(allow_past, opt_count)
                            , itertools.repeat(if_exact, opt_count)
                            , itertools.repeat(if_multi, opt_count)), chunksize=None)):

                option_summaries[key] = rtrn[1]

    return option_summaries


def option_report_multi_var(model: "Model | list"
                            , ds: pd.DataFrame | list
                            , model_rep  #: "ModelRep"
                            , option_dict_d: dict
                            , option_settings: "OptionSettings"
                            , targets_new: dict | None = None
                            , allow_past: bool = False
                            , if_exact: bool = False
                            , if_multi: bool = True
                            , cores: int | None = None) -> dict:
    """
    Calculate and report multiple options using multiprocessing.
    Pass options and options settings as dictionary of dictionaries and dictionary of OptionsSettings objects
    respectively.

    Version 2, actively used.

    :param model: Model object or list of Model objects
    :param ds: dataset or list of datasets
    :param model_rep: ModelRep object (export settings) or list of ModelRep objects
    :param option_dict_d: dictionary of budget options / scenarios to calculate as dictionary
    :param option_settings: OptionSettings object (option setting: target period etc.)
    :param targets_new: apply option to one entity, summarize another (useful for cross-elasticity). If not None:
        {'targets_new': [], 'apply_vars_new': []}
    :param allow_past: allow translation to the past (filled with observed data) periods
    :param if_exact: apply only to the specific time period, without next years
    :param if_multi: return only summary, without data and prediction (use if a lot of options are calculated
        and only summaries are actually necessary)
    :param cores: number of processor cores to use in calculations
    :return: dictionary of summaries
    """

    if cores is None:
        logging.warning('Cores number for parallel computing is not set. ' +
                        'Reset to ' + str(cores))

    elif cores > (multiprocessing.cpu_count() - 1):
        cores = (multiprocessing.cpu_count() - 1)
        logging.warning('Cores number for parallel computing is set too high for this computer. ' +
                        'Reset to ' + str(cores))

    opt_count = len(option_dict_d)
    opt_keys = option_dict_d.keys()
    opt_list = [option_dict_d[x] for x in opt_keys]

    option_summaries = {}

    if cores > 1:

        # destroy callable objects for multiprocessing
        model_iter = copy.deepcopy(model)

        if isinstance(model_iter, list):
            for i, val in enumerate(model_iter):
                model_iter[i].obj.transform_lhs_fn = None
                model_iter[i].obj.custom_predict_fn = None
        else:
            model_iter.obj.transform_lhs_fn = None
            model_iter.obj.custom_predict_fn = None

        with multiprocessing.Pool(cores) as pool:

            rtrn = pool.starmap_async(option_report, zip(itertools.repeat(model_iter, opt_count)
                                                         , itertools.repeat(ds, opt_count)
                                                         , itertools.repeat(model_rep, opt_count)
                                                         , opt_list
                                                         , itertools.repeat(option_settings, opt_count)
                                                         , itertools.repeat(targets_new, opt_count)
                                                         , itertools.repeat(allow_past, opt_count)
                                                         , itertools.repeat(if_exact, opt_count)
                                                         , itertools.repeat(if_multi, opt_count)), chunksize=None)

            for i in range(len(rtrn.get())):
                option_summaries[i] = rtrn.get()[i][1]

    else:
        for i, v in enumerate(opt_list):
            option_summaries[i] = option_report(model, ds, model_rep, v, option_settings, targets_new
                                                    , allow_past, if_exact, if_multi)[1]

    return option_summaries


