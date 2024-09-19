"""
Report fit and predict of the main model.

For the fit and predict of the category model see `fermatrica_rep.category`.

Multiple and single plots are defined separately, because intervals could not be plotted
via plotly.express friendly interface.
"""


import copy
import re

import numpy as np
import pandas as pd

from line_profiler_pycharm import profile

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from fermatrica_utils import date_to_period

from fermatrica.model.model import Model
from fermatrica_rep.model_rep import ModelRep


"""
Main / basic functions
"""


@profile
def fit_main_data(model: "Model"
                  , dt_pred: pd.DataFrame
                  , period: str = 'day'
                  , err_int: float = .1
                  , show_future: bool = False) -> (pd.DataFrame, pd.DataFrame):
    """
    Prepare data describing main model to plot or use as is. Data is grouped by superbrand
    (umbrella brand), no other option is allowed by now.

    :param model: Model object
    :param dt_pred: prediction data
    :param period: time period / interval to group by: 'day', 'week', 'month', 'quarter', 'year'
    :param err_int: error interval in decimal (typically .1, .05, .2 etc.)
    :param show_future: show future periods or not
    :return: tuple of wide and long dataframes
    """

    model_conf = model.conf

    if_conversion_fun = hasattr(model_conf, "conversion_fun") and \
                        isinstance(model_conf.conversion_fun, (list, tuple)) and len(model_conf.conversion_fun) > 0

    # filter listed period

    if show_future:
        ds = dt_pred[dt_pred['listed'].isin([2, 3, 4])]
    else:
        ds = dt_pred[dt_pred['listed'].isin([2, 3])]

    ds = copy.deepcopy(ds)

    # get values

    ds['observed_value'] = ds['observed'] * ds[model_conf.price_var]
    ds['predicted_value'] = ds['predicted'] * ds[model_conf.price_var]

    if hasattr(model_conf, 'conversion_var') and model_conf.conversion_var is not None:
        ds['observed_value'] = ds['observed_value'] * ds[model_conf.conversion_var]
        ds['predicted_value'] = ds['predicted_value'] * ds[model_conf.conversion_var]

    # reduce

    ds['date'] = date_to_period(ds['date'], period)

    group_id = ['date', 'listed']
    if 'superbrand' in ds.columns:
        group_id.append('superbrand')

    cols = copy.deepcopy(group_id)
    cols.extend([x for x in ds.columns if re.match(r'(observed|predicted)', x)])

    ds = ds[cols].groupby(group_id).sum().reset_index()

    # get error intervals

    ds['observed_lower'] = ds['observed'] * (1 - err_int)
    ds['observed_upper'] = ds['observed'] * (1 + err_int)

    ds['observed_value_lower'] = ds['observed_value'] * (1 - err_int)
    ds['observed_value_upper'] = ds['observed_value'] * (1 + err_int)

    if if_conversion_fun:
        cols = [x for x in ds.columns if re.match(r'observed_', x) and not re.search(r'_(lower|upper)', x)]

        if len(cols) > 0:
            for col in cols:
                ds[col + '_lower'] = ds[col] * (1 - err_int)
                ds[col + '_upper'] = ds[col] * (1 + err_int)

    # melt

    ds_melt = pd.melt(ds, id_vars=group_id)

    return ds, ds_melt


@profile
def _fit_main_plot_main_inner(ds: pd.DataFrame
                              , model_rep: "ModelRep"
                              , fig: go.Figure
                              , row_n: int = 1
                              , col_n: int = 1
                              , plot_type: str = 'vol'
                              , conv_step: int = 0):
    """
    Worker plotting fit and predict of the main model (volume or value depending on arguments).
    This function plots only one iteration of conversions if conversion chain is known,
    without faceting.

    :param ds: prepared dataset (wide format, not `ds_melt`)
    :param model_rep: ModelRep object (export settings)
    :param fig: empty figure (effectively canvas)
    :param row_n: number of "rows" in canvas to be used
    :param col_n: number of "columns" in canvas to be used
    :param plot_type: 'vol' or 'val' (volume or value)
    :param conv_step: conversion step index (integer)
    :return: filled figure (graphic object)
    """

    language = model_rep.language
    obs_name = model_rep.vis_dict.loc[(model_rep.vis_dict['section'] == "predict_table") & (
                model_rep.vis_dict['variable'] == "observed"), language].iloc[0]
    pred_name = model_rep.vis_dict.loc[(model_rep.vis_dict['section'] == "predict_table") & (
                model_rep.vis_dict['variable'] == "predicted"), language].iloc[0]

    if plot_type == 'vol':
        var_pred = 'predicted'
        var_obs = 'observed'
        var_upper = 'observed_upper'
        var_lower = 'observed_lower'
    else:
        var_pred = 'predicted_value'
        var_obs = 'observed_value'
        var_upper = 'observed_value_upper'
        var_lower = 'observed_value_lower'

    if conv_step > 0:
        var_pred = re.sub('^predicted', 'predicted_' + str(conv_step), var_pred)
        var_obs = re.sub('^observed', 'observed_' + str(conv_step), var_obs)
        var_upper = re.sub('^observed', 'observed_' + str(conv_step), var_upper)
        var_lower = re.sub('^observed', 'observed_' + str(conv_step), var_lower)

    if var_upper in ds.columns and var_pred in ds.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([ds['date'], ds['date'][::-1]], axis=0, ignore_index=True)
            , y=pd.concat([ds[var_upper], ds[var_lower][::-1]], axis=0, ignore_index=True)
            , fill='toself'
            , fillcolor='rgba(50,50,50,0.1)'
            , line_color='rgba(255,255,255,0)'
            , showlegend=False
            , name=obs_name
        ), row=row_n, col=col_n)

    if var_obs in ds.columns:
        fig.add_trace(go.Scatter(
            x=ds['date']
            , y=ds[var_obs]
            , line_color='rgb(0,100,80)'
            , name=obs_name
        ), row=row_n, col=col_n)

    if var_pred in ds.columns:
        fig.add_trace(go.Scatter(
            x=ds['date']
            , y=ds[var_pred]
            , line_color='#d62728'
            , name=pred_name
        ), row=row_n, col=col_n)

    # last train and test date

    train_end = ds[ds['listed'] == 2]['date'].max()
    test_end = ds[ds['listed'].isin([2, 3])]['date'].max()

    fig.add_vline(x=train_end, line_width=3, line_dash="dash", line_color="darkgrey", row=row_n, col=col_n)
    fig.add_vline(x=test_end, line_width=3, line_dash="dash", line_color="darkgrey", row=row_n, col=col_n)

    # format

    fig.update_traces(mode='lines')

    return fig


@profile
def _fit_main_plot_err_inner(ds: pd.DataFrame
                             , model_rep: "ModelRep"
                             , fig: go.Figure
                             , err_int: float = .1
                             , row_n: int = 1
                             , col_n: int = 1
                             , plot_type: str = 'vol'
                             , conv_step: int = 0):
    """
    Worker plotting errors of the main model. This function
    plots only one iteration of conversions if conversion chain is known,
    without faceting.

    :param ds: prepared dataset (wide format, not `ds_melt`)
    :param model_rep: ModelRep object (export settings)
    :param fig: empty figure (effectively canvas)
    :param err_int: error interval in decimals (typically .1, .05, .2 etc.)
    :param row_n: number of "rows" in canvas to be used
    :param col_n: number of "columns" in canvas to be used
    :param plot_type: 'vol' or 'val' (volume or value)
    :param conv_step: conversion step index (integer)
    :return: filled figure (graphic object)
    """

    language = model_rep.language
    err_name = model_rep.vis_dict.loc[(model_rep.vis_dict['section'] == "predict_table") & (model_rep.vis_dict['variable'] == "err"), language].iloc[0]

    if plot_type == 'vol':
        var_pred = 'predicted'
        var_obs = 'observed'
    else:
        var_pred = 'predicted_value'
        var_obs = 'observed_value'

    if conv_step > 0:
        var_pred = re.sub('^predicted', 'predicted_' + str(conv_step), var_pred)
        var_obs = re.sub('^observed', 'observed_' + str(conv_step), var_obs)

    # load data

    ds = ds.copy()
    ds['error'] = (ds[var_obs] - ds[var_pred]) / ds[var_obs]

    # create plot

    fig.add_trace(go.Scatter(
        x=ds['date']
        , y=ds['error']
        , line_color='slateblue'
        , name=err_name
    ), row=row_n, col=col_n)

    fig.add_hline(y=err_int, line_width=3, line_dash="dash", line_color="slategrey", row=row_n, col=col_n)
    fig.add_hline(y=-err_int, line_width=3, line_dash="dash", line_color="slategrey", row=row_n, col=col_n)

    # last train and test date

    train_end = ds[ds['listed'] == 2]['date'].max()
    test_end = ds[ds['listed'].isin([2, 3])]['date'].max()

    fig.add_vline(x=train_end, line_width=3, line_dash="dash", line_color="darkgrey", row=row_n, col=col_n)
    fig.add_vline(x=test_end, line_width=3, line_dash="dash", line_color="darkgrey", row=row_n, col=col_n)

    # format

    fig.update_traces(mode='lines')

    return fig


"""
Plot single entity
"""


@profile
def fit_main_plot_vol(model: "Model"
                      , dt_pred: pd.DataFrame
                      , model_rep: "ModelRep"
                      , period: str = 'day'  # 'day', 'week', 'month', 'quarter', 'year'
                      , err_int: float = .1
                      , show_future: bool = False
                      , target_superbrand: str | None = None):
    """
    Plot extended fit of target superbrand's volume KPI: fit, interval, error. Faceting is
    not allowed.

    :param model: Model object
    :param dt_pred: prediction data
    :param model_rep: ModelRep object (export settings)
    :param period: time period / interval to group by: 'day', 'week', 'month', 'quarter', 'year'
    :param err_int: error interval in decimal (typically .1, .05, .2 etc.)
    :param show_future: show future periods or not
    :param target_superbrand: umbrella brand to be plotted
    :return:
    """

    # load data

    ds, ds_melt = fit_main_data(model, dt_pred, period, err_int, show_future)

    if target_superbrand is not None and 'superbrand' in ds.columns:
        ds = ds[ds['superbrand'] == target_superbrand]

    # create plot

    vars_to_plot = [x for x in ds_melt.variable.unique() if
              re.match(r'(observed|predicted)', x) and not re.search(r'lower|upper|value', x)]

    vars_to_plot_main = [x for x in vars_to_plot if re.match('predicted', x)]
    vars_to_plot_err = [x for x in vars_to_plot if re.match('observed', x)]

    # define heights and make subplots

    height_full = len(vars_to_plot_main) * 7 + len(vars_to_plot_err) * 3

    row_heights = []

    for conv_step, v in enumerate(vars_to_plot_main):
        row_heights.append(7 / height_full)
        if len(vars_to_plot_err) > conv_step:
            row_heights.append(3 / height_full)

    subplot_titles = ['basic', 'residuals']
    if hasattr(model.conf, "conversion_fun"):
        for i in model.conf.conversion_fun:
            i = i.replace('code_py.adhoc.model.', '')
            subplot_titles = subplot_titles + [i]

    subplot_titles = [i.capitalize() for i in subplot_titles]

    fig = make_subplots(rows=len(vars_to_plot)
                        , cols=1
                        , row_heights=row_heights
                        , subplot_titles=tuple(subplot_titles))

    # create plots

    row_ind = 1

    for conv_step, v in enumerate(vars_to_plot_main):

        fig = _fit_main_plot_main_inner(ds, model_rep, fig, row_n=1, col_n=1, plot_type='vol', conv_step=conv_step)

        row_ind += 1

        if len(vars_to_plot_err) > conv_step:
            fig = _fit_main_plot_err_inner(ds, model_rep, fig, err_int=err_int, row_n=row_ind, col_n=1, plot_type='vol'
                                           , conv_step=conv_step)

    fig.update_layout(
        height=height_full / 10 * 400
        , margin={'l': 20, 'r': 20, 't': 50, 'b': 20}
    )

    return fig


@profile
def fit_main_plot_val(model: "Model"
                      , dt_pred: pd.DataFrame
                      , model_rep: "ModelRep"
                      , period: str = 'day'  # 'day', 'week', 'month', 'quarter', 'year'
                      , err_int: float = .1
                      , show_future: bool = False
                      , target_superbrand: str | None = None):
    """
    Plot extended fit of target superbrand's value KPI: fit, interval, error. Faceting is
    not allowed.

    :param model: Model object
    :param dt_pred: prediction data
    :param model_rep: ModelRep object (export settings)
    :param period: time period / interval to group by: 'day', 'week', 'month', 'quarter', 'year'
    :param err_int: error interval in decimal (typically .1, .05, .2 etc.)
    :param show_future: show future periods or not
    :param target_superbrand: umbrella brand to be plotted
    :return:
    """

    # load data

    ds, ds_melt = fit_main_data(model, dt_pred, period, err_int, show_future)

    if target_superbrand is not None and 'superbrand' in ds.columns:
        ds = ds[ds['superbrand'] == target_superbrand]

    # create plot

    fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3])

    fig = _fit_main_plot_main_inner(ds, model_rep, fig, row_n=1, col_n=1, plot_type='val')
    fig = _fit_main_plot_err_inner(ds, model_rep, fig, err_int, row_n=2, col_n=1, plot_type='val')

    fig.update_layout(
        margin={'l': 20, 'r': 20, 't': 50, 'b': 20}
    )

    return fig


"""
Plot multiple entities
"""


@profile
def fit_mult_data(model: "Model"
                  , dt_pred: pd.DataFrame
                  , period: str = 'day'  # 'day', 'week', 'month', 'quarter', 'year'
                  , err_int: float = .1
                  , show_future: bool = False
                  , group_var: list | tuple = ('superbrand',)
                  , bs_key_filter: list | tuple | None = None) -> (pd.DataFrame, pd.DataFrame):
    """
    Get data to plot fit and predict of the model with faceting. This function prepares data
    allowing plot multiple entities, but without error intervals. If error intervals are required
    and single entity (superbrand) is enough, see `fermatrica_rep.fit.fit_main_data`.

    :param model: Model object
    :param dt_pred: prediction data
    :param period: time period / interval to group by: 'day', 'week', 'month', 'quarter', 'year'
    :param err_int: error interval in decimal (typically .1, .05, .2 etc.)
    :param show_future: show future periods or not
    :param group_var: group entities by variables (list or tuple of strings)
    :param bs_key_filter: list or tuple of 'bs_key' values to preserve
    :return:
    """

    model_conf = model.conf

    if_conversion_fun = hasattr(model_conf, "conversion_fun") and \
                        isinstance(model_conf.conversion_fun, (list, tuple)) and len(model_conf.conversion_fun) > 0

    # filter

    if show_future:
        ds = dt_pred[dt_pred['listed'].isin([2, 3, 4])]
    else:
        ds = dt_pred[dt_pred['listed'].isin([2, 3])]

    if bs_key_filter is not None:
        ds = ds[ds['bs_key'].isin(bs_key_filter)]

    ds = copy.deepcopy(ds)

    # get values

    ds['observed_value'] = ds['observed'] * ds[model_conf.price_var]
    ds['predicted_value'] = ds['predicted'] * ds[model_conf.price_var]

    if hasattr(model_conf, 'conversion_var') and model_conf.conversion_var is not None:
        ds['observed_value'] = ds['observed_value'] * ds[model_conf.conversion_var]
        ds['predicted_value'] = ds['predicted_value'] * ds[model_conf.conversion_var]

    # reduce

    ds['date'] = date_to_period(ds['date'], period)

    if type(group_var) == tuple:
        group_var = list(group_var)

    group_var_ext = copy.deepcopy(group_var)

    group_var_ext.extend(['date', 'listed'])

    cols = copy.deepcopy(group_var_ext)
    cols.extend([x for x in ds.columns if re.match(r'(observed|predicted)', x)])

    ds = ds[cols].groupby(group_var_ext).sum().reset_index()

    # get error intervals

    ds['observed_lower'] = ds['observed'] * (1 - err_int)
    ds['observed_upper'] = ds['observed'] * (1 + err_int)

    ds['observed_value_lower'] = ds['observed_value'] * (1 - err_int)
    ds['observed_value_upper'] = ds['observed_value'] * (1 + err_int)

    if if_conversion_fun:
        cols = [x for x in ds.columns if re.match(r'observed_', x) and not re.search(r'_(lower|upper)', x)]

        if len(cols) > 0:
            for col in cols:
                ds[col + '_lower'] = ds[col] * (1 - err_int)
                ds[col + '_upper'] = ds[col] * (1 + err_int)

    # melt

    ds_melt = pd.melt(ds, id_vars=group_var_ext)

    ds['group_id'] = ''
    ds_melt['group_id'] = ''

    for i in group_var:
        ds['group_id'] = ds['group_id'] + ' ' + ds[i].astype('str')
        ds_melt['group_id'] = ds_melt['group_id'] + ' ' + ds_melt[i].astype('str')

    return ds, ds_melt


@profile
def _fit_mult_plot_inner(model: "Model"
                         , dt_pred: pd.DataFrame
                         , model_rep: "ModelRep"
                         , period: str = 'day'  # 'day', 'week', 'month', 'quarter', 'year'
                         , show_future: bool = False
                         , group_var: list | tuple = ('superbrand',)
                         , bs_key_filter: list | tuple | None = None
                         , plot_type: str = 'vol'
                         , conv_step: int = 0
                         ):
    """
    Worker plotting fit and predict of the main model (volume or value depending on arguments).

    This function plots multiple entities, but without error intervals. If error intervals are required
    and single entity (superbrand) is enough, see `fermatrica_rep.fit._fit_main_plot_main_inner`.
    As for now conversion chain plotting is not supported (could be changed later).

    :param model: Model object
    :param dt_pred: prediction data
    :param period: time period / interval to group by: 'day', 'week', 'month', 'quarter', 'year'
    :param show_future: show future periods or not
    :param group_var: group entities by variables (list or tuple of strings)
    :param bs_key_filter: list or tuple of 'bs_key' values to preserve
    :param plot_type: 'vol' or 'val' (volume or value)
    :param conv_step: conversion step index (integer): not implemented
    :return: filled figure (graphic object)
    :return:
    """

    language = model_rep.language
    obs_name = model_rep.vis_dict.loc[(model_rep.vis_dict['section'] == "predict_table") & (model_rep.vis_dict['variable'] == "observed"), language].iloc[0]
    pred_name = model_rep.vis_dict.loc[(model_rep.vis_dict['section'] == "predict_table") & (model_rep.vis_dict['variable'] == "predicted"), language].iloc[0]

    if plot_type == 'vol':
        var_pred = 'predicted'
        var_obs = 'observed'
    else:
        var_pred = 'predicted_value'
        var_obs = 'observed_value'

    # load data

    ds, ds_melt = fit_mult_data(model, dt_pred, period, show_future=show_future, group_var=group_var, bs_key_filter=bs_key_filter)

    ds_melt = ds_melt.loc[ds_melt['variable'].isin([var_pred, var_obs]), :]

    ds_melt.loc[ds_melt['variable'] == var_obs, 'variable'] = obs_name
    ds_melt.loc[ds_melt['variable'] == var_pred, 'variable'] = pred_name

    ds_melt.loc[:, 'group_id'] = ds_melt.loc[:, 'group_id'].str.title()

    # create plot

    row_sp = np.floor(len(ds_melt['group_id'].unique()) * 3)
    if row_sp != 0.0:
        row_sp = 1 / row_sp

    hght = np.ceil(len(ds_melt['group_id'].unique()) / 3) * 400
    if hght > 8e3:
        hght = 8e3

    fig = px.line(ds_melt
                  , x='date'
                  , y='value'
                  , color='variable'
                  , color_discrete_map={obs_name: 'rgb(0, 100, 80)', pred_name: '#d62728'}
                  , facet_col='group_id'
                  , facet_col_wrap=3
                  , facet_col_spacing=0.06
                  , facet_row_spacing=row_sp
                  # , text_auto='.2s'
                  , height=hght
                  )

    for i in fig.data:
        if len(i.name) > 40:
            i.name = i.name[0:40] + '...'

    train_end = ds[ds['listed'] == 2]['date'].max()
    test_end = ds[ds['listed'].isin([2, 3])]['date'].max()

    fig.add_vline(x=train_end, line_width=3, line_dash="dash", line_color="darkgrey")
    fig.add_vline(x=test_end, line_width=3, line_dash="dash", line_color="darkgrey")

    fig.for_each_annotation(lambda a: a.update(text=a.text.replace("group_id=", "")))

    fig.update_yaxes(matches=None, showticklabels=True)

    fig.update_layout(
        margin={'l': 20, 'r': 20, 't': 50, 'b': 20},
        legend_title='',
        xaxis_title='',
        yaxis_title=''
    )

    return fig


@profile
def fit_mult_plot_vol(model: "Model"
                      , dt_pred: pd.DataFrame
                      , model_rep: "ModelRep"
                      , period: str = 'day'  # 'day', 'week', 'month', 'quarter', 'year'
                      , show_future: bool = False
                      , group_var: list | tuple = ('superbrand',)
                      , bs_key_filter: list | tuple | None = None):
    """
    Plot extended fit of target superbrand's volume KPI: fit, interval, error.

    This function plots multiple entities, but without error intervals. If error intervals are required
    and single entity (superbrand) is enough, see `fermatrica_rep.fit.fit_main_plot_vol`.
    As for now conversion chain plotting is not supported (could be changed later).

    :param model: Model object
    :param dt_pred: prediction data
    :param model_rep: ModelRep object (export settings)
    :param period: time period / interval to group by: 'day', 'week', 'month', 'quarter', 'year'
    :param show_future: show future periods or not
    :param group_var: group entities by variables (list or tuple of strings)
    :param bs_key_filter: list or tuple of 'bs_key' values to preserve
    :return: filled figure (graphic object)
    :return:
    """

    fig = _fit_mult_plot_inner(model, dt_pred, model_rep, period, show_future, group_var, bs_key_filter, 'vol')

    return fig


@profile
def fit_mult_plot_val(model: "Model"
                      , dt_pred: pd.DataFrame
                      , model_rep: "ModelRep"
                      , period: str = 'day'  # 'day', 'week', 'month', 'quarter', 'year'
                      , show_future: bool = False
                      , group_var: list | tuple = ('superbrand',)
                      , bs_key_filter: list | tuple | None = None):
    """
    Plot extended fit of target superbrand's volume KPI: fit, interval, error.

    This function plots multiple entities, but without error intervals. If error intervals are required
    and single entity (superbrand) is enough, see `fermatrica_rep.fit.fit_main_plot_val`.
    As for now conversion chain plotting is not supported (could be changed later).

    :param model: Model object
    :param dt_pred: prediction data
    :param model_rep: ModelRep object (export settings)
    :param period: time period / interval to group by: 'day', 'week', 'month', 'quarter', 'year'
    :param show_future: show future periods or not
    :param group_var: group entities by variables (list or tuple of strings)
    :param bs_key_filter: list or tuple of 'bs_key' values to preserve
    :return: filled figure (graphic object)
    """

    fig = _fit_mult_plot_inner(model, dt_pred, model_rep, period, show_future, group_var, bs_key_filter, 'val')

    return fig
