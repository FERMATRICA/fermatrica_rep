"""
Report fit and predict of the category model (additional model describing category or categories).

For the fit and predict of the category model see `fermatrica_rep.fit`
"""


import numpy as np
import pandas as pd

from line_profiler import profile

import plotly.io as pio
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from fermatrica_utils import groupby_eff, select_eff
from fermatrica import Model
from fermatrica_rep.meta_model.node import ModelNode
from fermatrica_rep.meta_model.meta_model import _convert_to_metamodel
from fermatrica_rep.meta_model.meta_model import MetaModel

pio.templates.default = 'ggplot2'


@profile
def category_data(model: "Model | ModelNode | MetaModel"
                  , ds: pd.DataFrame | None = None
                  , period: str = 'day'  # 'day', 'week', 'month', 'quarter', 'year'
                  , err_int: float = .1
                  , show_future=False
                  , show_pretrain: bool = False
                  ):
    """
    Prepare data describing category model to plot or use as is.

    :param model: Model, ModelNode, or MetaModel object
    :param ds: dataset (optional if ModelNode passed)
    :param period: time period / interval to group by: 'day', 'week', 'month', 'quarter', 'year'
    :param err_int: error interval in decimal (typically .1, .05, .2 etc.)
    :param show_future: show future periods or not
    :param show_pretrain: show pretrain periods or not
    :return: dataframe to be plotted, or dict keyed by node name
    """

    mm = _convert_to_metamodel(model, ds=ds)

    results = {}
    for name, node in mm.nodes.items():
        results[name] = _category_data_worker(node.model, node.ds,
                                              period=period, err_int=err_int,
                                              show_future=show_future, show_pretrain=show_pretrain)
    if isinstance(model, MetaModel):
        return results
    if isinstance(model, list):
        return list(results.values())
    return list(results.values())[0]


@profile
def _category_data_worker(model
                          , ds: pd.DataFrame
                          , period: str = 'day'  # 'day', 'week', 'month', 'quarter', 'year'
                          , err_int: float = .1
                          , show_future=False
                          , show_pretrain: bool = False
                          ):

    tmp = select_eff(ds, ['date', 'pred_mrk', model.conf.Y_var, 'listed'])
    tmp['observed'] = tmp[model.conf.Y_var]
    tmp['predicted'] = tmp['pred_mrk']

    tmp_mask = tmp['listed'].isin([1, 2, 3, 4])

    if not show_future:
        tmp_mask = tmp_mask & ~tmp['listed'].isin([4])
    if not show_pretrain:
        tmp_mask = tmp_mask & ~tmp['listed'].isin([1])

    tmp = tmp[tmp_mask]

    if period in ['day', 'd', 'D']:
        tmp['date'] = tmp['date'].dt.floor(freq='D')
    elif period in ['week', 'w', 'W']:
        tmp['date'] = tmp['date'].dt.to_period('W').dt.start_time
    elif period in ['month', 'm', 'M']:
        tmp['date'] = tmp['date'].dt.to_period('M').dt.start_time
    elif period in ['quarter', 'q', 'Q']:
        tmp['date'] = tmp['date'].dt.to_period('Q').dt.start_time
    elif period in ['year', 'y', 'Y']:
        tmp['date'] = tmp['date'].dt.to_period('Y').dt.start_time

    tmp['observed_lower'] = tmp['observed'] * (1 - err_int)
    tmp['observed_upper'] = tmp['observed'] * (1 + err_int)

    tmp_pred = tmp.groupby(['date'], as_index=False, sort=True)
    tmp_pred = tmp_pred['predicted'].max()

    tmp_obs = tmp.groupby(['date'], as_index=False, sort=True)
    tmp_obs = tmp_obs['observed'].sum()

    tmp = pd.concat([tmp_pred, tmp_obs.drop('date', axis=1)], axis=1)

    tmp['observed_lower'] = tmp['observed'] * (1 - err_int)
    tmp['observed_upper'] = tmp['observed'] * (1 + err_int)

    return tmp


@profile
def category_main_plot(ds: pd.DataFrame
                       , fig: go.Figure
                       , row_n: int = 1
                       , col_n: int = 1):
    """
    Plot fit and predict of the category model.

    :param ds: prepared dataset (after `category_data()`)
    :param fig: empty figure (effectively canvas)
    :param row_n: number of "rows" in canvas to be used
    :param col_n: number of "columns" in canvas to be used
    :return: filled figure (graphic object)
    """

    fig.add_trace(go.Scatter(
        x=pd.concat([ds['date'], ds['date'][::-1]], axis=0, ignore_index=True)
        , y=pd.concat([ds['observed_upper'], ds['observed_lower'][::-1]], axis=0, ignore_index=True)
        , fill='toself'
        , fillcolor='rgba(50,50,50,0.1)'
        , line_color='rgba(255,255,255,0)'
        , showlegend=False
        , name='Observed'
    ), row=row_n, col=col_n)

    fig.add_trace(go.Scatter(
        x=ds['date']
        , y=ds['observed']
        , line_color='rgb(0,100,80)'
        , name='Observed'
    ), row=row_n, col=col_n)

    fig.add_trace(go.Scatter(
        x=ds['date']
        , y=ds['predicted']
        , line_color='#d62728'
        , name='Predicted'
    ), row=row_n, col=col_n)

    # format

    fig.update_traces(mode='lines')

    return fig


@profile
def category_err_plot(ds: pd.DataFrame
                      , fig: go.Figure
                      , err_int: float = .1
                      , row_n: int = 1
                      , col_n: int = 1):
    """
    Plot errors of the category model.

    :param ds: prepared dataset (after `category_data()`)
    :param fig: empty figure (effectively canvas)
    :param err_int: error interval in decimals (typically .1, .05, .2 etc.)
    :param row_n: number of "rows" in canvas to be used
    :param col_n: number of "columns" in canvas to be used
    :return:
    """

    ds['error'] = (ds['observed'] - ds['predicted']) / ds['observed']

    # horizontal lines - much faster, than add_hline

    y_line = np.full(ds['date'].shape, err_int)
    fig.add_trace(go.Scatter(
        x=ds['date']
        , y=y_line
        , line_color='lightgrey'
        , line_width=3, line_dash="dash"
        , name=''
        , showlegend=False
        , hoverinfo='skip'
    ), row=row_n, col=col_n)

    y_line = np.full(ds['date'].shape, -err_int)
    fig.add_trace(go.Scatter(
        x=ds['date']
        , y=y_line
        , line_color='lightgrey'
        , line_width=3, line_dash="dash"
        , name=''
        , showlegend=False
        , hoverinfo='skip'
    ), row=row_n, col=col_n)

    # main plot

    fig.add_trace(go.Scatter(
        x=ds['date']
        , y=ds['error']
        , line_color='slateblue'
        , name='Prediction Error'
    ), row=row_n, col=col_n)

    # format

    fig.update_traces(mode='lines')

    return fig


@profile
def category_plot(model: "Model | ModelNode | MetaModel"
                  , ds: pd.DataFrame | None = None
                  , period: str = "day"
                  , err_int=0.1
                  , show_future: bool = False
                  , show_pretrain: bool = False
                  ):
    """
    Main plot of the category model: fit, predict and errors

    :param model: Model, ModelNode, or MetaModel object
    :param ds: dataset (optional if ModelNode passed)
    :param period: time period / interval to group by
    :param err_int: error interval in decimal (typically .1, .05, .2 etc.)
    :param show_future: show future periods or not
    :param show_pretrain: show pretrain periods or not
    :return:
    """

    mm = _convert_to_metamodel(model, ds=ds)

    results = {}
    for name, node in mm.nodes.items():
        fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3])
        data = _category_data_worker(model=node.model, ds=node.ds, period=period,
                                     show_future=show_future, show_pretrain=show_pretrain)
        fig = category_main_plot(ds=data, fig=fig, row_n=1, col_n=1)
        fig = category_err_plot(ds=data, fig=fig, row_n=2, col_n=1, err_int=err_int)
        fig.update_layout(
            margin={'l': 20, 'r': 20, 't': 50, 'b': 20}
        )
        results[name] = fig

    if isinstance(model, MetaModel):
        return results
    if isinstance(model, list):
        return list(results.values())
    return list(results.values())[0]

