"""
Calculate price elasticity via MetaModel API.

Computes predicted volume and value for a grid of price scenarios
and derives arc elasticity via centered differences.
"""


import copy
import logging
import numpy as np
import pandas as pd

from line_profiler import profile

from fermatrica_utils import step_generator
from fermatrica.basics.basics import fermatrica_error

from fermatrica_rep.meta_model.node import ModelNode
from fermatrica_rep.meta_model.meta_model import MetaModel
from fermatrica_rep.options.scope import OptionScope
from fermatrica_rep.options.option import Option


def _moving_diff(a):
    ret = a.values
    return ret[2:] - ret[:-2]


def _moving_price_diff(a):
    ret = a.values
    return ret[2:] / ret[1:-1] - ret[:-2] / ret[1:-1]


@profile
def elasticity_data(model: "ModelNode | MetaModel",
                     scope: "OptionScope",
                     option: "Option",
                     step: float = .1,
                     price_range: tuple = (-50, 50),
                     price_name_opt: str = "nobdg_price",
                     if_exact: bool = False,
                     var: str = "exact",
                     ):
    """
    Get data to calculate price elasticity.

    Computes predicted volume and value for a grid of price values
    (varying ``price_name_opt`` across ``price_range`` with ``step``
    increments) and derives arc elasticity via centered differences.

    :param model: ModelNode or MetaModel object
    :param scope: OptionScope defining date range, entity filter,
        and reference period
    :param option: Option with base budget allocations (all items
        except ``price_name_opt``; the price value is varied)
    :param step: price step (in decimals)
    :param price_range: min and max price values to use as tuple
    :param price_name_opt: how price is named in ``option``
        ("price", "price_rel" etc.)
    :param if_exact: apply only to specific time period, without
        next years
    :param var: "exact" or "long" dataset used for summarizing
    :return: DataFrame with columns: price, pred_{var}_vol,
        pred_{var}_val, pred_{var}_vol_diff, pred_{var}_val_diff,
        price_diff, elasticity.  When ``model`` is a ``MetaModel``,
        returns ``dict[str, pd.DataFrame]`` keyed by node name.
    """

    if isinstance(model, ModelNode):
        mm = MetaModel(nodes={"model": model})
    else:
        mm = model

    results = {}
    for name, node in mm.nodes.items():
        results[name] = _elasticity_data_worker(
            node=node,
            scope=scope,
            option=option,
            step=step,
            price_range=price_range,
            price_name_opt=price_name_opt,
            if_exact=if_exact,
            var=var,
        )

    if isinstance(model, MetaModel):
        return results
    return list(results.values())[0]


@profile
def _elasticity_data_worker(
    node: "ModelNode",
    scope: "OptionScope",
    option: "Option",
    step: float,
    price_range: tuple,
    price_name_opt: str,
    if_exact: bool,
    var: str,
) -> pd.DataFrame:

    grid = _price_grid(option=option, step=step, price_range=price_range, price_name_opt=price_name_opt)

    logging.debug("Price elasticity options to calculate: %d", len(grid))

    scopes = [scope]

    price_values = sorted(grid.keys())
    options_list = [grid[p] for p in price_values]

    scopes_list = [scopes] * len(options_list)

    raw_opt_sum = _compute_price_options(
        node=node,
        options=options_list,
        scopes_list=scopes_list,
        if_exact=if_exact,
    )

    summaries: dict[float, dict] = {}
    for idx, summary in enumerate(raw_opt_sum):
        if summary is not None:
            summaries[price_values[idx]] = {
                "pred_exact_vol": summary.get("pred_exact_vol"),
                "pred_exact_val": summary.get("pred_exact_val"),
                "pred_long_vol": summary.get("pred_long_vol"),
                "pred_long_val": summary.get("pred_long_val"),
            }
        else:
            logging.warning("Price elasticity step at price=%s returned None; skipping", price_values[idx])

    opt_sum_fr = pd.DataFrame.from_dict(summaries, orient="index")

    if var == "exact":
        price_df = opt_sum_fr[["pred_exact_vol", "pred_exact_val"]].reset_index()
    elif var == "long":
        price_df = opt_sum_fr[["pred_long_vol", "pred_long_val"]].reset_index()
    else:
        fermatrica_error("'var' must be in ['exact', 'long'], got: " + str(var))

    price_df.rename(columns={"index": price_name_opt}, inplace=True)
    price_df[price_name_opt] = price_df[price_name_opt] + 1

    price_df_diff_all = pd.DataFrame(_moving_diff(price_df.loc[:, price_df.columns != price_name_opt]))
    price_df_diff_price = pd.DataFrame(_moving_price_diff(price_df.loc[:, price_df.columns == price_name_opt]))

    price_df_diff_all.columns = price_df.loc[:, price_df.columns != price_name_opt].columns + "_diff"
    price_df_diff_price.columns = price_df.loc[:, price_df.columns == price_name_opt].columns + "_diff"

    price_df_total = pd.concat(
        [price_df.iloc[1:-1].reset_index(drop=True), price_df_diff_all, price_df_diff_price],
        axis=1,
    )

    elasticity = pd.Series(
        (price_df_total["pred_" + var + "_vol_diff"] / price_df_total["pred_" + var + "_vol"])
        / (price_df_total[price_name_opt + "_diff"])
        * (-1.0),
        name="elasticity",
    )

    price_df_total = pd.concat([price_df_total, elasticity], axis=1)

    return price_df_total


def _compute_price_options(
    node: "ModelNode",
    options: list["Option"],
    scopes_list: list[list["OptionScope"]],
    if_exact: bool,
) -> list[dict | None]:
    """Compute price options sequentially (always fresh copy per option).

    Does not use ``compute_multi()`` because it must run each option on a
    fresh copy of the dataset — ModelNode.compute() mutates the internal
    ``ds``, and sequential reuse would corrupt state.
    """
    results = []
    for opt, scopes in zip(options, scopes_list):
        try:
            node_copy = copy.deepcopy(node)
            mm = MetaModel(nodes={"model": node_copy})
            raw = mm.compute(opt, scopes=scopes, if_exact=if_exact)
            scope_name = scopes[0].scope_name
            results.append(raw["model"][scope_name])
        except Exception:
            logging.warning("Price elasticity option failed; appending None", exc_info=True)
            results.append(None)
    return results


def _price_grid(option: "Option"
                , step: float
                , price_range: tuple
                , price_name_opt: str) -> dict[float, "Option"]:
    """
    Create price grid from option, range and step.

    :param option: base Option whose budget_data is cloned per
        step (with ``price_name_opt`` overridden)
    :param step: price step (in decimals)
    :param price_range: min and max price to use as tuple
    :param price_name_opt: how price is named in ``option``
        ("price", "price_rel" etc.)
    :return: price grid as dict mapping price value -> Option
    """
    grid: dict[float, Option] = {}

    for i in np.round(
        [i for i in step_generator(price_range[0] - step, price_range[1] + step, step)],
        int(np.log10(1 / step)) + 1,
    ):
        budget_cur = copy.deepcopy(option.budget_data)
        for scope_items in budget_cur.values():
            scope_items[price_name_opt] = i
        grid[i] = Option(name=option.name, budget_data=budget_cur)

    return grid
