"""
Calculate efficiency curves per marketing tool.

The version defined here is suited for models without LHS transformation only. It calculates curves
using transformation functions directly, so it is fast and (for user) simple.

If your model is more complicated than just RHS, use `fermatrica_rep.curves_full` module instead.
"""


import datetime
import numpy as np
import pandas as pd
import re

import plotly.graph_objects as go

from line_profiler_pycharm import profile

import fermatrica.model.transform as tr
from fermatrica.model.model import Model
from fermatrica_rep.meta_model.model_rep import ModelRep
import fermatrica_rep.basics


"""
Data
"""


@profile
def _curve_simple_data_worker(
        params: pd.DataFrame
        , coef_value: float | int
        , var: str
        , list_funcs: list
        , price_var: int
        , budget_lim: int | float = 100
        , budget_step: int | float = 1
        , if_precise: bool = False
        , adstock_len: int = 1e+3
):
    """
    Get curve data for single variable

    :param params: transformation params from Model.conf.params
    :param coef_value: media / tool price
    :param var: raw variable name
    :param list_funcs: list of transformation function names relevant to `var`
    :param price_var: product price variable name
    :param budget_lim: maximum budget per tool in millions
    :param budget_step: step in millions
    :param if_precise: some transformations could be calculated slow and precisely or fast and with small deviation
    :param adstock_len: number of periods to calculate decay / time effects (not only adstock)
    :return:
    """

    if if_precise:

        crv = {x: np.concatenate(([x / price_var * 1e+6], np.zeros(adstock_len - 1)))
               for x in np.arange(0, budget_lim + 1e-8, budget_step)}

        for func in list_funcs:
            params_tmp = params[(params['variable'] == var) & (params['fun'] == func)][['arg', 'value']]

            try:
                func_tmp = getattr(tr, func)

                crv = {k: func_tmp(pd.DataFrame({var: v,
                                                 'date': datetime.datetime.now(),
                                                 'wrk_index': list(range(0, len(v))),
                                                 'kpi_coef_cor_sm': 1.}), var, params_tmp) for k, v in crv.items()}

            except AttributeError as ae:
                pass

            var += f'_{func}'

        # find predictor coef

        crv = {k: v.sum() * coef_value for k, v in crv.items()}
        crv = pd.Series(crv)

    else:

        crv = {x: np.concatenate(([x / price_var * 1e+6], np.zeros(adstock_len - 1)))
               for x in np.arange(0, budget_lim + 1e-8, budget_step)}

        crv = pd.concat([pd.DataFrame({'bdg__': k, var: v}) for k, v in crv.items()], ignore_index=True)
        crv['date'] = datetime.datetime.now()
        crv['kpi_coef_cor_sm'] = 1.
        crv['wrk_index'] = crv.index

        for func in list_funcs:
            params_tmp = params[(params['variable'] == var) & (params['fun'] == func)][['arg', 'value']]

            try:
                func_tmp = getattr(tr, func)
                crv.loc[:, var] = func_tmp(crv, var, params_tmp)

            except AttributeError as ae:
                pass

            var_new = var + f'_{func}'
            crv.rename(columns={var: var_new}, inplace=True)
            var = var_new

        crv.loc[:, var] = crv.loc[:, var] * coef_value
        crv = crv.groupby('bdg__')[var].sum()

    return crv


@profile
def curves_simple_data(
        model: "Model"
        , ds: pd.DataFrame
        , model_rep: "ModelRep"
        , budget_lim: int | float = 100
        , budget_step: int | float = 1
        , if_precise: bool = False
        , adstock_len: int | float = 1000
):
    """
    Prepare data to plot simple curves
    
    :param model: Model object
    :param ds: dataset
    :param model_rep: ModelRep object (export setting)
    :param budget_lim: maximum budget per tool in millions
    :param budget_step: step in millions
    :param if_precise: some transformations could be calculated slow and precisely or fast and with small deviation
    :param adstock_len: number of periods to calculate decay / time effects (not only adstock)
    :return: table with curves data
    """
    
    budget_curves = pd.DataFrame()

    if type(adstock_len) is float:
        adstock_len = int(adstock_len)

    if ds is None or ds.empty:
        ds = model.obj.models['main'].model.data.frame

    if 'main' not in model.obj.models:
        return '"main" not found in model_objects - no main model info in loaded model'

    if not hasattr(model.obj.models['main'], 'params'):
        return '"main" model doesn`t have any info about predictor coefficients'

    vars_subset = fermatrica_rep.basics.coef_var_align(
        model.conf.trans_path_df[(model.conf.trans_path_df['price'] > 0) & (model.conf.trans_path_df['variable_fin'] != '') &
                                (pd.notna(model.conf.trans_path_df ['variable_fin']))]
        , model.obj.models['main'].params
    )

    if 'display_var' not in vars_subset.columns:
        vars_subset['display_var'] = vars_subset['variable_fin']

    # iterate over variables

    for var in vars_subset[['variable', 'variable_fin', 'price', 'coef_value', 'display_var']].to_numpy():

        if var[0].__contains__('adhoc'):
            pass
        else:
            funcs = var[1].replace(var[0], '').split('_')[1:]
            budget_curve = _curve_simple_data_worker(model.conf.params
                                                     , coef_value=var[3]
                                                     , var=var[0]
                                                     , list_funcs=funcs
                                                     , price_var=var[2]
                                                     , budget_lim=budget_lim
                                                     , budget_step=budget_step
                                                     , if_precise=if_precise
                                                     , adstock_len=adstock_len)

            max_bdgt = ds.get(var[0], default=pd.Series(budget_lim * 1e+6)).max() * var[2]
            max_bdgt = (max_bdgt + (budget_step * 1e+6) / 2) // (budget_step * 1e+6) * (budget_step * 1e+6)

            pattern = re.compile(r'^[0-9]+_')
            disp_name = var[4]

            if pattern.sub('', disp_name) in model_rep.vis_dict.loc[
                model_rep.vis_dict['section'].isin(['factor_decomposition', 'costs_type']), 'variable'].array:
                if pattern.match(disp_name) is None:
                    ptrn = ''
                else:
                    ptrn = pattern.match(disp_name)[0]

                disp_name = ptrn + model_rep.vis_dict.loc[(model_rep.vis_dict['variable'] == pattern.sub('', disp_name)) &
                                              (model_rep.vis_dict['section'].isin(['factor_decomposition', 'costs_type']))
                    , model_rep.language].iloc[0]

            budget_curves[disp_name] = budget_curve.where(budget_curve.index * 1e+6 <= max_bdgt)
            budget_curves[f"{disp_name}_extrapolated"] = budget_curve.where(
                (budget_curve.index * 1e+6 >= max_bdgt) & (budget_curve.index * 1e+6 <= 1.5 * max_bdgt))

    # update palettes

    lst = budget_curves.columns.tolist()
    lst = [x for x in lst if not re.search(r'_extrapolated$', x)]
    lst.sort()

    model_rep.fill_colours_tools(lst)

    for i in lst:
        model_rep.palette_tools[i + '_extrapolated'] = model_rep.palette_tools[i]

    return budget_curves


"""
Visualization
"""


@profile
def _curves_simple_plot_worker(ds: pd.DataFrame,
                               model_rep: "ModelRep",
                               title: str,
                               ):
    """
    Plot single type of the curve: incremental, return or ROI

    each column = subplot
    ds.index = x (budget)
    ds.values = y (response curves)

    :param ds:
    :param model_rep:
    :param title:
    :return:
    """

    layout = {'title': {'text': title}}

    fig = go.Figure(layout=layout)

    for curve in ds.items():

        nm = curve[0]
        ptrn = re.compile(r'^[0-9]+_')
        if ptrn.match(nm):
            nm = ptrn.sub('', nm)

        if curve[0].__contains__('extrapolated'):

            fig.add_trace(go.Scatter(
                x=curve[1].index,
                y=curve[1].values,
                line_color=model_rep.palette_tools[curve[0]],
                name=nm,
                line=dict(dash='dash'),
                showlegend=False
            ))

        else:

            fig.add_trace(go.Scatter(
                x=curve[1].index,
                y=curve[1].values,
                line_color=model_rep.palette_tools[curve[0]],
                name=nm
            ))

    return fig


@profile
def curves_simple_plot(ds: pd.DataFrame,
                       model_rep: "ModelRep",
                       price: float | None = None,
                       conv: float | None = None,
                       ):
    """
    Plot efficiency curves per marketing tool.

    This version is designed for models without LHS transformation only. If your model is complicated,
    use `fermatrica_rep.curves_full` module. That one is much more flexible, but also much slower.

    :param ds: prepared dataset
    :param model_rep: ModelRep object (export settings)
    :param price: product price
    :param conv: conversion rat
    :return:
    """

    if pd.isna(price):
        price = 1
    if pd.isna(conv):
        conv = 1

    # volume (y = vol)
    fig_vol = _curves_simple_plot_worker(ds, model_rep, 'Incremental Volume')

    # value (y = vol * price * conv)
    ds = ds * price * conv  # volume -> values
    fig_val = _curves_simple_plot_worker(ds, model_rep, 'Incremental Value')

    # profit (y = val - bgt)
    fig_incr = _curves_simple_plot_worker(ds.sub(ds.index * 1_000_000, axis=0), model_rep, 'Return / Profit')

    # roi (y = val / bgt)
    fig_roi = _curves_simple_plot_worker(ds.div(ds.index * 1_000_000, axis=0), model_rep, 'ROI')

    return {'Incremental Volume': fig_vol,
            'Incremental Value': fig_val,
            'Return / Profit': fig_incr,
            'ROI': fig_roi}
