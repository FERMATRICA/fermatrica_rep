"""
Report fit and predict of the category model (additional model describing category or categories).

For the fit and predict of the category model see `fermatrica_rep.fit`
"""


import pandas as pd

import plotly.io as pio
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from fermatrica_utils import groupby_eff, select_eff
from fermatrica.model.model import Model

pio.templates.default = 'ggplot2'


def category_data(model: "Model"
                  , ds: pd.DataFrame
                  , period: str = 'day'  # 'day', 'week', 'month', 'quarter', 'year'
                  , err_int: float = .1
                  , show_future=False
                  ):
    """
    Prepare data describing category model to plot or use as is.

    :param model: Model object
    :param ds: dataset
    :param period: time period / interval to group by: 'day', 'week', 'month', 'quarter', 'year'
    :param err_int: error interval in decimal (typically .1, .05, .2 etc.)
    :param show_future: show future periods or not
    :return: dataframe to be plotted
    """

    tmp = select_eff(ds, ['date', 'pred_mrk', model.conf.Y_var, 'listed'])
    tmp['observed'] = tmp[model.conf.Y_var]
    tmp['predicted'] = tmp['pred_mrk']

    if show_future:
        tmp = tmp[tmp['listed'].isin([2, 3, 4])]
    else:
        tmp = tmp[tmp['listed'].isin([2, 3])]

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

    tmp_pred = tmp[tmp['listed'].isin([2, 3, 4])].groupby(['date'], as_index=False, sort=True)
    tmp_pred = tmp_pred['predicted'].max()

    tmp_obs = tmp[tmp['listed'].isin([2, 3, 4])].groupby(['date'], as_index=False, sort=True)
    tmp_obs = tmp_obs['observed'].sum()

    tmp = pd.concat([tmp_pred, tmp_obs.drop('date', axis=1)], axis=1)

    tmp['observed_lower'] = tmp['observed'] * (1 - err_int)
    tmp['observed_upper'] = tmp['observed'] * (1 + err_int)

    return tmp


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

    fig.add_trace(go.Scatter(
        x=ds['date']
        , y=ds['error']
        , line_color='slateblue'
        , name='Prediction Error'
    ), row=row_n, col=col_n)

    fig.add_hline(y=err_int, line_width=3, line_dash="dash", line_color="slategrey", row=row_n, col=col_n)
    fig.add_hline(y=-err_int, line_width=3, line_dash="dash", line_color="slategrey", row=row_n, col=col_n)

    # format

    fig.update_traces(mode='lines')

    return fig


def category_plot(model: "Model"
                  , ds: pd.DataFrame
                  , period: str
                  , err_int=0.1
                  , show_future: bool = False
                  ):
    """
    Main plot of the category model: fit, predict and errors

    :param model: Model object
    :param ds: dataset
    :param period: time period / interval to group by
    :param err_int: error interval in decimal (typically .1, .05, .2 etc.)
    :param show_future: show future periods or not
    :return:
    """

    fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3])

    data = category_data(model=model, ds=ds, period=period, show_future=show_future)
    fig = category_main_plot(ds=data, fig=fig, row_n=1, col_n=1)
    fig = category_err_plot(ds=data, fig=fig, row_n=2, col_n=1, err_int=err_int)

    fig.update_layout(
        margin={'l': 20, 'r': 20, 't': 50, 'b': 20}
    )
    return fig

