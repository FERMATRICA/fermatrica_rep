"""
Calculate price elasticity.

Parallel computing is assuming in called functions.
"""


import copy
import numpy as np
import pandas as pd

from line_profiler_pycharm import profile

from fermatrica_utils import step_generator

from fermatrica.model.model import Model
from fermatrica_rep.model_rep import ModelRep
import fermatrica_rep.options.calc as calc
from fermatrica_rep.options.define import OptionSettings


@profile
def elasticity_data(model: "Model"
                    , model_rep: "ModelRep"
                    , ds: pd.DataFrame
                    , opt_set: "OptionSettings"
                    , option: dict
                    , targets_new: dict | None = None
                    , step: float = .1
                    , price_range: tuple = (-50, 50)
                    , price_name_opt: str = "price"
                    , cores: int = 10
                    , if_exact: bool = False
                    , var: str = "exact"
                    ):
    """
    Get data to calculate price elasticity

    :param model: Model object
    :param model_rep: ModelRep object (export settings)
    :param ds: dataset
    :param opt_set: OptionSetting object containing calculate settings
    :param option: specific option as dictionary
    :param targets_new: apply option to one entity, summarize another (useful for cross-elasticity). If not None:
        {'targets_new': [], 'apply_vars_new': []}
    :param step: price step (in decimals)
    :param price_range: min and max prices to use as tuple
    :param price_name_opt: how price is named in `option` ("price", "price_rel" etc.)
    :param cores: number of processor cores to use in parallel computing (set None for automatic detecting)
    :param if_exact: apply only to specific time period, without next years
    :param var: "exact" or "long" dataset use for summarizing
    :return: ready to plot price elasticity dataset
    """

    grid = price_grid(option=option, step=step, price_range=price_range, price_name_opt=price_name_opt)

    print('Options to calculate: ' + str(len(grid)))

    opt_sum = calc.option_report_multi_var(model, ds, model_rep, grid, opt_set,
                                           targets_new=targets_new,
                                           cores=cores, if_exact=if_exact)

    opt_sum_fr = pd.DataFrame.from_dict(opt_sum, orient='index')

    opt_sum_fr.set_index(np.round([i for i in step_generator(price_range[0] - step, price_range[1] + step, step)],
                                  int(np.log10(1 / step)) + 1), inplace=True)

    tmp_exact = pd.DataFrame(opt_sum_fr["pred_exact_0"])
    ddf = pd.DataFrame(columns=['pred_exact_vol', 'pred_exact_val'])

    for (i, r) in tmp_exact.iterrows():
        e = r['pred_exact_0']
        ddf.loc[i] = [e['pred_exact_vol'], e['pred_exact_val']]

    # Replace ds with the output of concat(ds, ddf)
    tmp_exact = pd.concat([tmp_exact, ddf], axis=1)

    del tmp_exact["pred_exact_0"]

    tmp_long = opt_sum_fr[['pred_long_vol', 'pred_long_val']]
    if var == "exact":
        price_df = tmp_exact.reset_index()
    elif var == "long":
        price_df = tmp_long.reset_index()
    else:
        print('/"var/" must be in ["exact", "long"]')
        return -1

    price_df.rename(columns={'index': price_name_opt}, inplace=True)
    price_df[price_name_opt] = price_df[price_name_opt] + 1

    def moving_diff(a):
        ret = a.values
        return ret[2:] - ret[:-2]

    def moving_diff_price(a):
        ret = a.values
        return ret[2:] / ret[1:-1] - ret[:-2] / ret[1:-1]

    price_df_diff_all = pd.DataFrame(moving_diff(price_df.loc[:, price_df.columns != price_name_opt]))
    price_df_diff_price = pd.DataFrame(moving_diff_price(price_df.loc[:, price_df.columns == price_name_opt]))

    price_df_diff_all.columns = price_df.loc[:, price_df.columns != price_name_opt].columns + "_diff"
    price_df_diff_price.columns = price_df.loc[:, price_df.columns == price_name_opt].columns + "_diff"

    price_df_total = pd.concat(
        [price_df.iloc[1:-1].reset_index(drop=True), price_df_diff_all, price_df_diff_price],
        axis=1)

    elasticity = pd.Series((price_df_total["pred_" + var + "_vol_diff"] / price_df_total["pred_" + var + "_vol"]) / \
                           (price_df_total[price_name_opt + "_diff"]
                            ) * (-1.0),
                           name="elasticity")

    price_df_total = pd.concat([price_df_total, elasticity], axis=1)

    return price_df_total


def price_grid(option: dict
               , step: float
               , price_range: tuple
               , price_name_opt: str):
    """
    Create price grid from option, range and step

    :param option: specific option as dictionary
    :param step: price step (in decimals)
    :param price_range: min and max prices to use as tuple
    :param price_name_opt: how price is named in `option` ("price", "price_rel" etc.)
    :return: price grid as dictionary of dictionaries
    """

    grid = {}

    for i in np.round([i for i in step_generator(price_range[0] - step, price_range[1] + step, step)],
                      int(np.log10(1 / step)) + 1):
        option_cur = copy.deepcopy(option)
        option_cur[price_name_opt] = i
        grid[i] = option_cur

    return grid

