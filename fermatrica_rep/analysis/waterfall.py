"""
Decompose model into effects (impacts).

This module describes period / interval decomposition, for dynamic decomposition see `fermatrica_rep.decomposition`.
Data preparation (`split_m_m`) is also defined in `fermatrica_rep.decomposition`.
"""


import copy
import pandas as pd

from line_profiler import profile

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from fermatrica_utils import groupby_eff

from fermatrica_rep.basics import _filter_df
from fermatrica_rep.meta_model.model_rep import ModelRep
from fermatrica_rep.meta_model.node import ModelNode
from fermatrica_rep.meta_model.meta_model import _convert_to_metamodel
from fermatrica_rep.meta_model.meta_model import MetaModel

pio.templates.default = 'ggplot2'


@profile
def waterfall_plot(split_m_m: pd.DataFrame | list | dict
                   , brands: list | None = None
                   , model_rep: ModelRep | ModelNode | MetaModel | list | None = None
                   , *, brands_var: str = 'superbrand'
                   , filters: dict | None = None
                   , date_start: str = '2020-01-01'
                   , date_end: str = '2021-01-01'
                   , absolute_sort: bool | str = False
                   , pos_neg: bool = False
                   , show_pretrain: bool = False):
    """
        Wrapper
        Plot decomposition for specific period from `date_start` to `date_end` as waterfall.

        :param split_m_m: prepared dataset (see `fermatrica_rep.extract_effect()`) or list/dict of prepared datasets
        :param brands: list of umbrella brands to preserve, or None for all brands
        :param model_rep: ModelRep, ModelNode, MetaModel, or list of ModelRep objects (deprecated)
        :param brands_var: column the simple filter values apply to
        :param filters: complex filter ``{column: [values]}``
        :param date_start: start of the period
        :param date_end: end of the period
        :param absolute_sort: sort by absolute or signed values
        :param pos_neg: colorize by brand or by positive/negative impacts
        :param show_pretrain: show pretrain periods or not
        :return:
        """

    mm = _convert_to_metamodel(model_rep)

    results = []
    names = []
    for i, (name, node) in enumerate(mm.nodes.items()):
        splits_i = split_m_m[name] if isinstance(split_m_m, dict) else (split_m_m[i] if isinstance(split_m_m, list) else split_m_m)
        fig = _waterfall_plot_worker(split_m_m=splits_i, brands=brands, model_rep=node.model_rep,
                                     date_start=date_start, date_end=date_end,
                                     absolute_sort=absolute_sort, pos_neg=pos_neg,
                                     show_pretrain=show_pretrain,
                                     brands_var=brands_var, filters=filters)
        if fig is not None:
            results.append(fig)
            names.append(name)

    if isinstance(model_rep, MetaModel):
        return dict(zip(names, results))
    if isinstance(model_rep, list):
        return results
    if results:
        return results[0]
    return None


@profile
def _waterfall_plot_worker(split_m_m: pd.DataFrame
                           , brands: list | None = None
                           , model_rep: ModelRep = None
                           , date_start: str = '2020-01-01'
                           , date_end: str = '2021-01-01'
                           , absolute_sort: bool | str = False
                           , pos_neg: bool = False
                           , show_pretrain: bool = False
                           , brands_var: str = 'superbrand'
                           , filters: dict | None = None):
    """
    Worker
    Plot decomposition for specific period from `date_start` to `date_end` as waterfall.

    :param split_m_m: prepared dataset (see `fermatrica_rep.extract_effect()`)
    :param brands: list of umbrella brands to preserve, or None for all brands
    :param model_rep: ModelRep object (export settings)
    :param date_start: start of the period
    :param date_end: end of the period
    :param absolute_sort: sort by absolute or signed values
    :param pos_neg: colorize by brand or by positive/negative impacts
    :param show_pretrain: show pretrain periods or not
    :param brands_var: column the simple filter values apply to
    :param filters: complex filter ``{column: [values]}``
    :return:
    """

    if brands_var in split_m_m.columns:
        if brands is not None and not split_m_m[brands_var].isin(brands).any():
            return None
    elif 'superbrand' not in split_m_m.columns:
        return None

    if pos_neg:
        fig = waterfall_plot_native(split_m_m
                                    , brands
                                    , model_rep
                                    , date_start
                                    , date_end
                                    , absolute_sort
                                    , show_pretrain
                                    , brands_var
                                    , filters)
    else:
        fig = waterfall_plot_bar(split_m_m
                                 , brands
                                 , model_rep
                                 , date_start
                                 , date_end
                                 , absolute_sort
                                 , show_pretrain
                                 , brands_var
                                 , filters)

    return fig


@profile
def waterfall_data(split_m_m: pd.DataFrame | list
                   , brands: list | None = None
                   , *, brands_var: str = 'superbrand'
                   , filters: dict | None = None
                   , date_start: str = '2020-01-01'
                   , date_end: str = '2021-01-01'
                   , absolute_sort: bool | str = False
                   , show_pretrain: bool = False) -> pd.DataFrame | list:
    """
        Wrapper
        Prepare data of the ratio of the influence of the factor. Including for waterfall plots.

        :param split_m_m: prepared dataset (see `fermatrica_rep.extract_effect()`)
        :param brands: list of umbrella brands to preserve, or None for all brands
        :param brands_var: column the simple filter values apply to
        :param filters: complex filter ``{column: [values]}``
        :param date_start: start of the period
        :param date_end: end of the period
        :param absolute_sort: sort by absolute or signed values
        :param show_pretrain: show pretrain periods or not
        :return:
        """

    if isinstance(split_m_m, list):
        split = [None] * len(split_m_m)
        for i in range(len(split_m_m)):
            split[i] = _waterfall_data_worker(split_m_m=split_m_m[i],
                                              brands=brands,
                                              date_start=date_start,
                                              date_end=date_end,
                                              absolute_sort=absolute_sort,
                                              show_pretrain=show_pretrain,
                                              brands_var=brands_var,
                                              filters=filters)
    else:
        split = _waterfall_data_worker(split_m_m=split_m_m,
                                       brands=brands,
                                       date_start=date_start,
                                       date_end=date_end,
                                       absolute_sort=absolute_sort,
                                       show_pretrain=show_pretrain,
                                       brands_var=brands_var,
                                       filters=filters)

    return split


@profile
def _waterfall_data_worker(split_m_m: pd.DataFrame
                           , brands: list | None = None
                           , date_start: str = '2020-01-01'
                           , date_end: str = '2021-01-01'
                           , absolute_sort: bool | str = False
                           , show_pretrain: bool = False
                           , brands_var: str = 'superbrand'
                           , filters: dict | None = None) -> pd.DataFrame:
    """
    Worker
    Prepare data of the ratio of the influence of the factor. Including for waterfall plots.

    :param split_m_m: prepared dataset (see `fermatrica_rep.extract_effect()`)
    :param brands: list of umbrella brands to preserve, or None for all brands
    :param date_start: start of the period
    :param date_end: end of the period
    :param absolute_sort: sort by absolute or signed values
    :param show_pretrain: show pretrain periods or not
    :param brands_var: column the simple filter values apply to
    :param filters: complex filter ``{column: [values]}``
    :return:
    """

    split = _filter_df(split_m_m, brands=brands, brands_var=brands_var, filters=filters)

    mask = (split['date'] >= pd.to_datetime(date_start)) & (split['date'] <= pd.to_datetime(date_end))

    if not show_pretrain:
        mask = mask & ~split['listed'].isin([1])

    split = groupby_eff(split, ['variable'], ['value'], mask, sort=False)['value'] \
        .sum() \
        .reset_index()

    split['abs'] = split['value'].abs()

    if absolute_sort:
        split.sort_values(by='abs', ascending=False, inplace=True)
    else:
        split.sort_values(by='value', ascending=False, inplace=True)

    split['ratio'] = (split['value'] / split['value'].sum()) * 100
    split['ratio_lbl'] = split['ratio'].round(2).astype(str) + '%'

    split['value_cumsum'] = split['value'].cumsum()
    split['ratio_cumsum'] = (split['value_cumsum'] / split['value'].sum()) * 100

    return split


@profile
def waterfall_plot_native(split_m_m: pd.DataFrame
                          , brands: list
                          , model_rep: ModelRep
                          , date_start: str = '2020-01-01'
                          , date_end: str = '2021-01-01'
                          , absolute_sort: bool | str = False
                          , show_pretrain: bool = False
                          , brands_var: str = 'superbrand'
                          , filters: dict | None = None) -> go.Figure:
    """
    Plot native waterfall - with positive and negative coloration.

    :param split_m_m: prepared dataset (see `fermatrica_rep.extract_effect()`)
    :param brands: list of umbrella brands to preserve
    :param model_rep: ModelRep object (export settings)
    :param date_start: start of the period
    :param date_end: end of the period
    :param absolute_sort: sort by absolute or signed values
    :param show_pretrain: show pretrain periods or not
    :param brands_var: column the simple filter values apply to
    :param filters: complex filter ``{column: [values]}``
    :return:
    """

    language = model_rep.language

    if absolute_sort == 'True':
        absolute_sort = True
    elif absolute_sort == 'False':
        absolute_sort = False

    split = waterfall_data(split_m_m=split_m_m,
                           brands=brands,
                           date_start=date_start,
                           date_end=date_end,
                           absolute_sort=absolute_sort,
                           show_pretrain=show_pretrain,
                           brands_var=brands_var,
                           filters=filters)

    split['measure'] = 'relative'

    fig = go.Figure(go.Waterfall(
        x=split['variable']
        , y=split['ratio']
        , measure=split['measure']
        , text=split['ratio'].round(2).astype(str) + '%'
        , textposition="outside"
        , textangle=0
    ))

    for i in range(len(fig.data[0].x)):
        if len(fig.data[0].x[i]) > 40:
            fig.data[0].x[i] = fig.data[0].x[i][0:30] + '...' + fig.data[0].x[i][-10:-1]

    fig.update_layout(
        xaxis={'side': 'top'}
        , xaxis_title=model_rep.vis_dict.loc[(model_rep.vis_dict['section'] == "plot_efficiency") &
                                             (model_rep.vis_dict['variable'] == "factor"), language].iloc[0]
        , yaxis_title='Factors contribution to forecast, %'
    )

    fig.add_hline(y=100, line_width=2, line_dash="dash", line_color="red", opacity=0.2)

    return fig


@profile
def waterfall_plot_bar(split_m_m: pd.DataFrame
                       , brands: list
                       , model_rep: ModelRep
                       , date_start: str = '2020-01-01'
                       , date_end: str = '2021-01-01'
                       , absolute_sort: bool | str = False
                       , show_pretrain: bool = False
                       , brands_var: str = 'superbrand'
                       , filters: dict | None = None) -> go.Figure:
    """
    Plot waterfall based on barplot to get by brand coloration.

    :param split_m_m: prepared dataset (see `fermatrica_rep.extract_effect()`)
    :param brands: list of umbrella brands to preserve
    :param model_rep: ModelRep object (export settings)
    :param date_start: start of the period
    :param date_end: end of the period
    :param absolute_sort: sort by absolute or signed values
    :param show_pretrain: show pretrain periods or not
    :param brands_var: column the simple filter values apply to
    :param filters: complex filter ``{column: [values]}``
    :return:
    """

    language = model_rep.language

    if absolute_sort == 'True':
        absolute_sort = True
    elif absolute_sort == 'False':
        absolute_sort = False

    split = waterfall_data(split_m_m=split_m_m,
                           brands=brands,
                           date_start=date_start,
                           date_end=date_end,
                           absolute_sort=absolute_sort,
                           show_pretrain=show_pretrain,
                           brands_var=brands_var,
                           filters=filters)

    fig = px.bar(x=split['variable']
                 , y=split['ratio']
                 , base=split['ratio_cumsum'] - split['ratio']
                 , color=split['variable']
                 , color_discrete_map=model_rep.palette_tools
                 , text=split['ratio_lbl']
                 , labels=
                 {"color": model_rep.vis_dict.loc[(model_rep.vis_dict['section'] == "plot_efficiency") &
                                                  (model_rep.vis_dict['variable'] == "factor"), language].iloc[0]}
                 , height=700
                 )

    for i in fig.data:
        if len(i.name) > 40:
            i.name = i.name[0:30] + '...' + i.name[-10:-1]
        if len(i['x'][0]) > 40:
            i['x'][0] = i['x'][0][0:30] + '...' + i['x'][0][-10:-1]

    fig.update_layout(
        xaxis={'side': 'top'}
        , xaxis_title=model_rep.vis_dict.loc[(model_rep.vis_dict['section'] == "plot_efficiency") &
                                             (model_rep.vis_dict['variable'] == "factor"), language].iloc[0]
        , yaxis_title='Factors contribution to forecast, %'
        , showlegend=False

    )

    fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False,
                      marker_line_color='black', marker_line_width=0.8, opacity=1)

    fig.add_hline(y=100, line_width=2, line_dash="dash", line_color="red", opacity=0.4)

    return fig
