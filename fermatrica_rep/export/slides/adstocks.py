"""
Generate slide with adstock plot: adstock (decay, carry-over) effect for number of variables
(marketing tools).
"""


import pandas as pd

import lxml.etree as etree
from pptx.presentation import Presentation
from pptx.chart.data import ChartData
from pptx.util import Pt
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.dml.color import RGBColor

from fermatrica_rep.export.basics import _resolve_slide_layout, set_chart_colors_fill
from fermatrica_rep.analysis.adstocks import adstocks_data as adstocks_data
from fermatrica_rep.meta_model.node import ModelNode


def create(prs: Presentation,
           node: ModelNode,
           *,
           ds: pd.DataFrame,
           cln_meas: list | tuple,
           cln_dim: list | tuple = ('superbrand', 'master', 'bs_key', 'date', 'listed', 'kpi_coef'),
           n: int = 50,
           ):
    """
    Create adstock slide and add to `prs` PPTX presentation.

    :param prs: Presentation object from python_pptx package
    :param node: ModelNode bundling (model, ds, model_rep)
    :param ds: dataset (differs from node.ds after compute)
    :param cln_meas: column names to be used as measurements
    :param cln_dim: column names to be used as dimensions
    :param n: number of observations per column
    :return: Presentation object from python_pptx package
    """

    model = node.model
    model_rep = node.model_rep
    language = model_rep.language
    superbrand = model.conf.target_superbrand

    # data prep
    adstocks_dataframe = adstocks_data(model=node
                                       , ds=ds
                                       , filters={'superbrand': [superbrand]}
                                       , cln_dim=cln_dim
                                       , cln_meas=cln_meas
                                       , n=n
                                       )

    superbrand = superbrand.title()

    # prepare slide in presentation

    slide_layout = _resolve_slide_layout(prs, model_rep.pptx_cnf['Blank_slide'])

    slide_height = model_rep.pptx_cnf['slide_height']
    slide_width = model_rep.pptx_cnf['slide_width']

    slide = prs.slides.add_slide(slide_layout)

    slide_title = model_rep.vis_dict.loc[(model_rep.vis_dict['section'] == 'adstocks') &
                                         (model_rep.vis_dict['variable'] == 'slide_title'), language].iloc[0]

    slide_title = slide_title.replace('***id', superbrand)

    slide.shapes[0].text_frame.text = slide_title
    slide.shapes[0].text_frame.paragraphs[0].font.size = model_rep.pptx_cnf['font_size_gross']
    slide.shapes[0].text_frame.paragraphs[0].font.name = model_rep.pptx_cnf['font_family_header']

    # mock text
    x, y = 0.55 * slide_width, slide.shapes[0].height + slide.shapes[0].top * 1.7
    cx, cy = 0.35 * slide_width, 0.55 * slide_height

    x, y, cx, cy = int(round(x)), int(round(y)), int(round(cx)), int(round(cy))

    textbox = slide.shapes.add_textbox(x, y, cx, cy)
    textbox.text = model_rep.pptx_cnf['mock_text']

    paragraph = textbox.text_frame.paragraphs[0]
    paragraph.font.size = model_rep.pptx_cnf["font_size_main"]
    paragraph.font.name = model_rep.pptx_cnf["font_family_body"]
    textbox.text_frame.word_wrap = True

    if len(adstocks_dataframe) == 0:
        return prs

    adstocks_dataframe = adstocks_dataframe * 100
    mask = adstocks_dataframe < 1e-4
    adstocks_dataframe = adstocks_dataframe.astype(object)
    adstocks_dataframe[mask] = "#N/A"

    # series names come from model_rhs.display_var, so a column is plottable only
    # when an active model-formula token references it
    active_tokens = model.conf.model_rhs.loc[model.conf.model_rhs['if_active'] == 1, 'token']
    adstocks_dataframe = adstocks_dataframe.loc[:, [active_tokens.str.contains(i).any() for i in adstocks_dataframe.columns]]

    if adstocks_dataframe.shape[1] == 0:
        return prs

    for i in adstocks_dataframe.columns:
        disp_var = model.conf.model_rhs.loc[(model.conf.model_rhs.if_active == 1) & (model.conf.model_rhs.token.str.contains(i)), "display_var"].iloc[0]
        if disp_var in model_rep.palette_tools:
            adstocks_dataframe.rename(columns={i: disp_var}, inplace=True)

    # building chart
    chart_data_val = ChartData()
    chart_data_val.categories = pd.Series(adstocks_dataframe.index)

    for i, j in adstocks_dataframe.items():
        chart_data_val.add_series(name=i,
                                  values=j)

    x, y = 0.05 * slide_width, slide.shapes[0].height + slide.shapes[0].top
    cx, cy = 0.45 * slide_width, 0.65 * slide_height

    x, y, cx, cy = int(round(x)), int(round(y)), int(round(cx)), int(round(cy))

    graphic_frame = slide.shapes.add_chart(
        XL_CHART_TYPE.LINE, x, y, cx, cy, chart_data_val
    )

    chart = graphic_frame.chart
    chart = set_chart_colors_fill(chart, model_rep, RGBColor(170, 170, 170))

    for sr in chart.plots[0].series:
        sr.format.line.width = Pt(2.5)
        sr.smooth = True

    etree.SubElement(chart._chartSpace.xpath("//c:valAx")[0], '{http://schemas.openxmlformats.org/drawingml/2006/chart}crossBetween').attrib['val'] = 'midCat'

    skip = str(round(n / 5))

    if int(skip) < 1:
        skip = "1"

    etree.SubElement(chart._chartSpace.xpath("//c:catAx")[0], '{http://schemas.openxmlformats.org/drawingml/2006/chart}tickLblSkip').attrib['val'] = skip
    etree.SubElement(chart._chartSpace.xpath("//c:catAx")[0], '{http://schemas.openxmlformats.org/drawingml/2006/chart}tickMarkSkip').attrib['val'] = skip

    chart.legend.position = XL_LEGEND_POSITION.BOTTOM
    chart.chart_title.text_frame.paragraphs[0].font.size = model_rep.pptx_cnf['font_size_main']

    category_axis = chart.category_axis
    category_axis.has_title = True
    category_axis_title = model_rep.vis_dict.loc[(model_rep.vis_dict['section'] == "adstocks") & (
                model_rep.vis_dict['variable'] == "axis_x"), model_rep.language].iloc[0]

    category_axis_title = category_axis_title.replace('***period', model_rep.vis_dict.loc[(model_rep.vis_dict['section'] == "adstocks") & (
                model_rep.vis_dict['variable'] == model.conf.period_var), model_rep.language].iloc[0])

    category_axis.axis_title.text_frame.text = category_axis_title
    category_axis.axis_title.text_frame.paragraphs[0].font.size = model_rep.pptx_cnf['font_size_small']

    value_axis = chart.value_axis
    value_axis.has_title = True
    value_axis_title = model_rep.vis_dict.loc[(model_rep.vis_dict['section'] == "adstocks") & (
                model_rep.vis_dict['variable'] == "axis_y"), model_rep.language].iloc[0]

    value_axis.axis_title.text_frame.text = value_axis_title
    value_axis.axis_title.text_frame.paragraphs[0].font.size = model_rep.pptx_cnf['font_size_small']

    value_axis.minimum_scale = 0
    value_axis.maximum_scale = 100
    value_axis.major_unit = 20

    value_axis.tick_labels.number_format = r"0\%"

    value_axis.has_minor_gridlines = False
    value_axis.has_major_gridlines = True

    chart.font.size = model_rep.pptx_cnf['font_size_small']
    chart.font.name = model_rep.pptx_cnf['font_family_body']

    chart.has_legend = True
    chart.legend.include_in_layout = False

    text_frame = chart.chart_title.text_frame
    text_frame.text = model_rep.vis_dict.loc[(model_rep.vis_dict['section'] == 'adstocks') &
                                             (model_rep.vis_dict['variable'] == 'chart_title'), language].iloc[0]
    text_frame.text = text_frame.text.replace('***id', superbrand)

    text_frame.paragraphs[0].font.size = model_rep.pptx_cnf['font_size_main']

    return prs


def create_legacy(prs: Presentation,
                  model,
                  ds,
                  model_rep,
                  *,
                  cln_meas: list | tuple,
                  cln_dim: list | tuple = ('superbrand', 'master', 'bs_key', 'date', 'listed', 'kpi_coef'),
                  n: int = 50,
                  ):
    import warnings
    warnings.warn(
        "Passing Model, ds, and ModelRep to slides.adstocks is deprecated. "
        "Use ModelNode instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    node = ModelNode(model=model, ds=ds, model_rep=model_rep)
    return create(prs, node,
                  ds=ds,
                  cln_meas=cln_meas,
                  cln_dim=cln_dim,
                  n=n)

