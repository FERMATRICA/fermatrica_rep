"""
Calculate and report multiple options with grid of options. Use it to calculate efficiency curves
(full version) and option combinations without optimisation.
"""


import re
import itertools
import numpy as np
import pandas as pd
import copy
from typing import Callable

from fermatrica_utils import step_generator
from fermatrica.model.model import Model

from fermatrica_rep.options.define import OptionSettings
from fermatrica_rep.model_rep import ModelRep
import fermatrica_rep.options.calc as calc


def opt_grid(translation: pd.DataFrame
             , budget_step: int | float = 1
             , fixed_vars: dict | None = None) -> dict:
    """
    Create empty option grid (effectively dataframe) to fill with prediction summaries per option.

    This version gets range and step for a couple of variables and creates grid, where only one
    variable for every option is changed and all others are set to 0.

    Use it to calculate curves, i.e. range of growing budgets for specific marketing tool:
    digital OLV range (1, 10), step 1 means: [1M, 2M, 3M, 4M, 5M, 6M, 7M, 8M, 9M, 10M] in OLV,
    all other tools are fixed.

    :param translation: translation dataframe (from files like `options.xlsx`, `translation` sheet)
    :param budget_step: budget iteration step in millions, defaults to 1 (i.e. 1M)
    :param fixed_vars: translation variables with their values to be fixed across grid
    :return: grid dataframe with multiple options ready to be filled
    """

    translation = translation.copy(deep=True)
    translation = translation[translation['max_budg'] > 0]

    zero = pd.DataFrame(0, index=np.arange(1), columns=["option"] + translation.index.tolist())
    cln_no_budg = zero.columns[~zero.columns.str.match('bdg')]

    for col in cln_no_budg:
        zero[col] = 1

    zero["option"] = "zero"
    grid = copy.copy(zero)

    for opt in zero.columns[zero.columns.str.match('bdg')]:
        tmp_opt = zero.iloc[0, :].replace('zero', opt)

        max_budg = int(translation.loc[translation.index == opt, "max_budg"].iloc[0])

        list_bdg = list(step_generator(budget_step, max_budg, budget_step))

        tmp = pd.DataFrame([tmp_opt] * len(list_bdg))
        tmp[opt] = list_bdg

        tmp["option"] = opt

        grid = pd.concat([grid, tmp], axis=0)

    grid = grid.convert_dtypes(infer_objects=True, convert_string=True, convert_integer=True
                               , convert_boolean=False, convert_floating=False)
    grid['option'] = grid['option'].astype('string')

    # total budget

    cols = [x for x in grid.columns.tolist() if x not in ['option']]

    grid['bdg'] = 0

    for col in cols:
        grid.loc[:, 'bdg'] = grid.loc[:, 'bdg'] + grid.loc[:, col]

    # fixed vars (price, trends etc.)

    if type(fixed_vars) == dict and len(fixed_vars) > 0:
        for k, v in fixed_vars.items():
            grid[k] = v

    return grid


def opt_grid_expand(borders: dict
                    , budget_step: int | float = 1
                    , fixed_vars: dict | None = None) -> dict:
    """
    Create empty option grid (effectively dataframe) to fill with prediction summaries per option.

    This version gets range (borders) and step for a couple of variables and creates grid with all
    combinations available (Cartesian product).

    Use it to find near optimal solutions for large number of budget sizes, if algorithmic optimisation
    is not available or less performance efficient than grid search (e.g. number of combinations
    for every budget size is 100 and number of algo iterations per budget size is 300).

    :param borders: dictionary of low and upper borders for every variable to vary
    :param budget_step: budget iteration step in millions, defaults to 1 (i.e. 1M)
    :param fixed_vars: translation variables with their values to be fixed across grid
    :return: grid dataframe with multiple options ready to be filled
    """

    for k, v in borders.items():
        borders[k] = list(step_generator(v[0], v[1], budget_step))

    borders_gen = (dict(zip(borders, x)) for x in itertools.product(*borders.values()))
    borders_gen = {ind: x for ind, x in enumerate(borders_gen)}

    grid = pd.DataFrame.from_dict(borders_gen, orient='index')
    grid['option'] = grid.index.astype(str)

    # total budget

    cols = [x for x in grid.columns.tolist() if x not in ['option']]

    grid['bdg'] = 0

    for col in cols:
        grid.loc[:, 'bdg'] = grid.loc[:, 'bdg'] + grid.loc[:, col]

    # fixed vars (price, trends etc.)

    if type(fixed_vars) == dict and len(fixed_vars) > 0:
        for k, v in fixed_vars.items():
            grid[k] = v

    return grid


def _opt_grid_summarize(model_rep: "ModelRep"
                        , grid: pd.DataFrame
                        , opt_sum_fr: pd.DataFrame
                        , grid_type: str = 'curves'
                        ) -> pd.DataFrame:
    """
    Fill options grid with options summaries after all options are calculated.

    :param model_rep: ModelRep object (export settings)
    :param grid: empty options grid dataframe to be filled
    :param opt_sum_fr: calculated options summaries as dataframe
    :param grid_type: "curves" or "cartesian"
    :return: options summaries per every option with option split and info, as dataframe
    """

    tmp_exact = pd.DataFrame(opt_sum_fr["pred_exact_0"])
    ddf = pd.DataFrame(columns=['pred_exact_vol', 'pred_exact_val'])

    for (i, r) in tmp_exact.iterrows():
        e = r['pred_exact_0']
        ddf.loc[i] = [e['pred_exact_vol'], e['pred_exact_val']]

    # Replace ds with the output of concat(ds, ddf)
    tmp_exact = pd.concat([tmp_exact, ddf], axis=1)

    del tmp_exact["pred_exact_0"]

    tmp_long = opt_sum_fr[['pred_long_vol', 'pred_long_val']]
    curves_full_df = pd.concat([tmp_exact, tmp_long], axis=1)

    if grid_type == 'curves':
        cols = ['bdg', 'option']
        if 'max_obs_budg' in grid.columns:
            cols = cols + ['max_obs_budg']
        curves_full_df = pd.concat([curves_full_df, grid[cols].reset_index(drop=True)], axis=1)
    else:
        curves_full_df = pd.concat([curves_full_df, grid.reset_index(drop=True)], axis=1)

    # get display names

    pattern = re.compile(r'^[0-9]+_')
    language = model_rep.language

    if not model_rep.vis_dict.empty:

        for name in curves_full_df['option'].unique():

            disp_name = re.sub(r'^bdg_', '', name)

            if pattern.match(disp_name) is None:
                ptrn = ''
            else:
                ptrn = pattern.match(disp_name)[0]

            if pattern.sub('', disp_name) in model_rep.vis_dict.loc[model_rep.vis_dict['section'].isin(['costs_type']), 'variable'].array:

                curves_full_df.loc[curves_full_df['option'] == name, 'option'] = ptrn + \
                    model_rep.vis_dict.loc[(model_rep.vis_dict['variable'] == pattern.sub('', disp_name)) &
                    (model_rep.vis_dict['section'].isin(['costs_type'])), language].iloc[0]

    return curves_full_df


def option_report_multi_post(model: "Model"
                             , ds: pd.DataFrame
                             , model_rep: "ModelRep"
                             , opt_set: "OptionSettings"
                             , translation: pd.DataFrame
                             , adhoc_curves_max_costs: "None | Callable" = None
                             , budget_step: int | float = 1
                             , fixed_vars: dict | None = None
                             , cores: int | None = None
                             , if_exact: bool = False
                             , grid_type: str = 'curves'
                             , borders: dict | None = None
                             , bdg_max: int | None = 301
                             ) -> pd.DataFrame:
    """
    Calculate and report multiple options with grid of options.

    Higher-level function for multiple option calculations. Use it to calculate efficiency curves
    (full version) and option combinations without optimisation.

    :param model: Model object
    :param ds: dataset
    :param model_rep: ModelRep object (export settings)
    :param opt_set: OptionSettings object (option setting: target period etc.)
    :param translation: translation dataframe (from files like `options.xlsx`, `translation` sheet)
    :param adhoc_curves_max_costs: adhoc function to set maximum observed values for every variable (optional)
    :param budget_step: budget iteration step in millions, defaults to 1 (i.e. 1M)
    :param fixed_vars: translation variables with their values to be fixed across grid
    :param cores: number of processor cores to use in calculations; None sets to all computer logical cores - 1
    :param if_exact: apply only to the specific time period, without next years
    :param grid_type: "curves" or "cartesian"
    :param borders: dictionary of low and upper borders for every variable to vary (only if `grid_type` set to
        "cartesian", otherwise ignored)
    :param bdg_max: maximum budget size (all options with larger budgets to be dropped)
    :return: options summaries per every option with option split and info, as dataframe
    """

    if grid_type == 'curves':
        grid = opt_grid(translation, budget_step, fixed_vars)
    elif grid_type == 'cartesian':
        grid = opt_grid_expand(borders, budget_step, fixed_vars)
    else:
        return pd.DataFrame({})

    if isinstance(adhoc_curves_max_costs, Callable):
        grid = adhoc_curves_max_costs(grid, ds)

    if bdg_max is not None:
        grid = grid.loc[grid['bdg'] <= bdg_max, :]

    print('Options to calculate: ' + str(grid.shape[0]))

    curves_full_df = grid.reset_index(drop=True) \
        .drop('option', axis=1) \
        .to_dict(orient='index')

    opt_sum = calc.option_report_multi_var(model, ds, model_rep, curves_full_df, opt_set,
                                           cores=cores, if_exact=if_exact)
    opt_sum_fr = pd.DataFrame.from_dict(opt_sum, orient='index')

    grid_data = _opt_grid_summarize(model_rep, grid, opt_sum_fr, grid_type)

    return grid_data
