"""
Generate slide with decomposition plot. Could be used both in retro analysis and in option reporting,
for superbrand as a whole and for specific number of variables known as "SKU"
"""


import pandas as pd
import copy

from pptx.presentation import Presentation
from pptx.chart.data import ChartData
from pptx.util import Pt
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_TICK_LABEL_POSITION
from pptx.dml.color import RGBColor

from fermatrica_utils import groupby_eff, pandas_force_materialize

from fermatrica.model.model import Model

from fermatrica_rep.basics import _filter_df
from fermatrica_rep.export import wording
from fermatrica_rep.export.basics import _resolve_slide_layout, set_chart_colors_fill
from fermatrica_rep.meta_model.model_rep import ModelRep
from fermatrica_rep.meta_model.node import ModelNode


def create(prs: Presentation,
           node: ModelNode,
           *,
           split_m_m: pd.DataFrame,
           option_name: str,
           brands: list | None = None,
           period: str = 'day',
           group_var: list | tuple = ('superbrand', 'market'),
           plot_type: str = 'brand',
           brands_var: str = 'superbrand',
           filters: dict | None = None,
           if_volume: bool = True,
           show_future: bool | str = True,
           show_pretrain: bool = False,
           contour_line: bool = True,
           ):
    """
    Create (dynamic) decomposition slide and add to `prs` PPTX presentation.

    :param prs: Presentation object from python_pptx package
    :param node: ModelNode bundling (model, ds, model_rep)
    :param split_m_m: prepared dataset (see `fermatrica_rep.extract_effect()`)
    :param option_name: name of the option to be reported
    :param brands: superbrand string names as list, None to keep all
    :param period: time period / interval to group by
    :param group_var: group entities by variables
    :param plot_type: 'retro', 'brand' or 'sku'. Used for naming only
    :param brands_var: column the simple filter values apply to
    :param filters: complex filter ``{column: [values]}``
    :param if_volume: optimize volume or value KPI
    :param show_future: show future periods or not
    :param show_pretrain: show pretrain periods or not
    :param contour_line: add contours or not
    :return: Presentation object from python_pptx package
    """

    model = node.model
    model_rep = node.model_rep

    language = model_rep.language

    units_label = wording.units_label(model_rep, model.conf)

    split_m_m = pandas_force_materialize(split_m_m, if_hard=False)

    # prepare slide in presentation

    slide_layout = _resolve_slide_layout(prs, model_rep.pptx_cnf['Blank_slide'])

    slide_height = model_rep.pptx_cnf['slide_height']
    slide_width = model_rep.pptx_cnf['slide_width']

    slide = prs.slides.add_slide(slide_layout)

    if ('retro' in plot_type) & (if_volume is True):
        slide_title = model_rep.vis_dict.loc[(model_rep.vis_dict['section'] == 'factor_decomposition') &
                                             (model_rep.vis_dict['variable'] == 'title_vol'), language].iloc[0]
    elif ('retro' in plot_type) & (if_volume is False):
        slide_title = model_rep.vis_dict.loc[(model_rep.vis_dict['section'] == 'factor_decomposition') &
                                             (model_rep.vis_dict['variable'] == 'title_val'), language].iloc[0]
    else:
        match (plot_type, if_volume):
            case ('brand', True):
                slide_title = model_rep.vis_dict.loc[(model_rep.vis_dict['section'] == 'option.detail.effects.superbrand') &
                                                     (model_rep.vis_dict['variable'] == 'title'), language].iloc[0]
            case ('brand', False):
                slide_title = model_rep.vis_dict.loc[(model_rep.vis_dict['section'] == 'option.detail.effects.superbrand') &
                                                     (model_rep.vis_dict['variable'] == 'title_value'), language].iloc[0]
            case ('sku', True):
                slide_title = model_rep.vis_dict.loc[(model_rep.vis_dict['section'] == 'option.detail.effects.sku') &
                                                     (model_rep.vis_dict['variable'] == 'title'), language].iloc[0]
            case ('sku', False):
                slide_title = model_rep.vis_dict.loc[(model_rep.vis_dict['section'] == 'option.detail.effects.sku') &
                                                     (model_rep.vis_dict['variable'] == 'title_value'), language].iloc[0]
            case _:
                slide_title = model_rep.vis_dict.loc[(model_rep.vis_dict['section'] == 'option.detail.effects.superbrand') &
                                                     (model_rep.vis_dict['variable'] == 'title'), language].iloc[0]

    slide_title = slide_title.replace('***id', option_name)
    slide_title = slide_title.replace('***units', units_label)

    slide.shapes[0].text_frame.text = slide_title
    slide.shapes[0].text_frame.paragraphs[0].font.size = model_rep.pptx_cnf['font_size_gross']
    slide.shapes[0].text_frame.paragraphs[0].font.name = model_rep.pptx_cnf['font_family_header']

    # data prep for decompose
    split = copy.deepcopy(split_m_m)

    df_decompose, sku = _decompose_chart_data(split,
                                              brands=brands,
                                              brands_var=brands_var,
                                              filters=filters,
                                              show_future=show_future,
                                              show_pretrain=show_pretrain,
                                              period=period,
                                              group_var=group_var,
                                              if_volume=if_volume)

    # constructing graphic
    chart_data = ChartData()
    chart_data.categories = df_decompose.index
    for col_name, col in df_decompose.items():
        chart_data.add_series(col_name, col)

    x, y = (0.5 - 0.45) * slide_width, slide.shapes[0].top + slide.shapes[0].height
    cx, cy = 0.9 * slide_width, 0.7 * slide_height

    x, y, cx, cy = int(round(x)), int(round(y)), int(round(cx)), int(round(cy))

    graphic_frame = slide.shapes.add_chart(
        XL_CHART_TYPE.COLUMN_STACKED, x, y, cx, cy, chart_data
    )
    chart = graphic_frame.chart
    chart = set_chart_colors_fill(chart, model_rep)

    chart.has_legend = True
    chart.legend.position = XL_LEGEND_POSITION.LEFT
    chart.legend.include_in_layout = False
    chart.legend.font.name = model_rep.pptx_cnf['font_family_body']
    chart.legend.font.size = model_rep.pptx_cnf['font_size_small']

    chart.plots[0].gap_width = 0

    chart.font.size = model_rep.pptx_cnf['font_size_small']
    chart.font.name = model_rep.pptx_cnf['font_family_body']

    for x in chart.plots[0].series:
        x.invert_if_negative = False
        if contour_line:
            x.format.line.width = Pt(0.5)
            x.format.line.color.rgb = RGBColor(200, 200, 200)

    category_axis = chart.category_axis
    category_axis.tick_label_position = XL_TICK_LABEL_POSITION.LOW
    category_axis.has_minor_gridlines = True
    category_axis.has_major_gridlines = False

    if period == 'month':
        chart._chartSpace.xpath("//c:dateAx")[0].xpath("//c:baseTimeUnit")[0].attrib['val'] = 'months'
    elif period == 'year':
        chart._chartSpace.xpath("//c:dateAx")[0].xpath("//c:baseTimeUnit")[0].attrib['val'] = 'years'

    value_axis = chart.value_axis
    value_axis.axis_title.text_frame.paragraphs[0].font.size = model_rep.pptx_cnf['font_size_main']

    chart.has_title = True
    text_frame = chart.chart_title.text_frame
    caption_var = "factors_title" if if_volume else "factors_title_val"
    caption = wording.text(model_rep, model.conf, section="plot_efficiency", variable=caption_var)
    caption = caption.replace("***units", units_label)
    text_frame.text = f"{sku}: {caption}"
    text_frame.paragraphs[0].font.size = model_rep.pptx_cnf['font_size_main']

    return prs


def create_legacy(prs: Presentation,
                  model,
                  model_rep,
                  *,
                  split_m_m: pd.DataFrame,
                  option_name: str,
                  brands: list,
                  period: str = 'day',
                  group_var: list | tuple = ('superbrand', 'market'),
                  plot_type: str = 'brand',
                  bs_key_filter: list | tuple | None = None,
                  if_volume: bool = True,
                  show_future: bool | str = True,
                  show_pretrain: bool = False,
                  contour_line: bool = True,
                  ):
    import warnings
    warnings.warn(
        "Passing Model and ModelRep to slides.decomposition is deprecated. "
        "Use ModelNode instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    node = ModelNode(model=model, ds=pd.DataFrame(), model_rep=model_rep)
    return create(prs, node,
                  split_m_m=split_m_m,
                  option_name=option_name,
                  brands=brands,
                  period=period,
                  group_var=group_var,
                  plot_type=plot_type,
                  filters={'bs_key': list(bs_key_filter)} if bs_key_filter is not None else None,
                  if_volume=if_volume,
                  show_future=show_future,
                  show_pretrain=show_pretrain,
                  contour_line=contour_line)


def _decompose_chart_data(split: pd.DataFrame,
                          brands: list | None = None,
                          brands_var: str = 'superbrand',
                          filters: dict | None = None,
                          show_future: bool | str = True,
                          show_pretrain: bool = False,
                          period: str = 'day',
                          group_var: list | tuple = ('superbrand', 'market'),
                          if_volume: bool = True) -> tuple:
    """
    Convert 'common' decomposed dataset to dataset prepared to be plotted.

    :param split: prepared dataset (see `fermatrica_rep.extract_effect()`)
    :param brands: simple filter values (e.g. umbrella brands or bs_keys), applied to `brands_var`
    :param brands_var: column the simple filter values apply to
    :param filters: complex filter ``{column: [values]}``
    :param period: time period / interval to group by: 'day', 'week', 'month', 'quarter', 'year'
    :param group_var: group entities by variables (list or tuple of strings)
    :param if_volume: optimize volume or value KPI
    :param show_future: show future periods or not
    :param show_pretrain: show pretrain periods or not
    :return: tuple of pandas DataFrame and sku name as string
    """

    if show_future == 'True':
        show_future = True
    elif show_future == 'False':
        show_future = False

    split = _filter_df(split, brands=brands, brands_var=brands_var, filters=filters)

    if period in ['day', 'd', 'D']:
        split['date'] = split['date'].dt.floor(freq='D')
    elif period in ['week', 'w', 'W']:
        split['date'] = split['date'].dt.to_period('W').dt.start_time
    elif period in ['month', 'm', 'M']:
        split['date'] = split['date'].dt.to_period('M').dt.start_time
    elif period in ['quarter', 'q', 'Q']:
        split['date'] = split['date'].dt.to_period('Q').dt.start_time
    elif period in ['year', 'y', 'Y']:
        split['date'] = split['date'].dt.to_period('Y').dt.start_time

    mask = split['listed'].isin([1, 2, 3, 4])

    if not show_future:
        mask = mask & ~split['listed'].isin([4])
    if not show_pretrain:
        mask = mask & ~split['listed'].isin([1])

    if if_volume is False:
        split['value'] = split['value_rub']

    tmp = groupby_eff(split, list(set(['date'] + list(group_var) + ['variable'])), ['value'], mask, sort=False)
    tmp = tmp.value.sum().reset_index()
    tmp['sku'] = ''

    for i in group_var:
        tmp['sku'] = tmp['sku'] + ' ' + tmp[i].astype('str').str.title()
    tmp['sku'] = tmp['sku'].str.lstrip()

    # if several distinct group_var levels remain per (date, variable) — e.g. per-region
    # scopes on a multi-region dataset — aggregate them into a single total series
    if tmp[['date', 'variable']].duplicated(keep=False).any():
        agg = tmp.groupby(['date', 'variable'], as_index=False)['value'].sum()
        if 'superbrand' in group_var:
            agg['sku'] = str(tmp['superbrand'].astype('str').unique()[0]).title()
        else:
            agg['sku'] = tmp['sku'].iloc[0]
        tmp = agg

    sku = tmp['sku'].unique()[0]

    tmp_wide = pd.pivot(tmp, index='date', columns='variable', values='value')

    # delete all columns with only 0 values
    tmp_wide = tmp_wide.loc[:, (tmp_wide != 0).any(axis=0)]

    return tmp_wide, sku
