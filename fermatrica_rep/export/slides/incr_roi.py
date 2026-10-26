"""
Generate 2 slides with efficiency curves (i.e. incremental KPI, profit, ROI) via full curves approach
(row of options to be calculated): short and long term effects.

To get stable results run `fermatrica.options.calc.option_report()`
with some fixed option (as 'zero') and `exact=False` before calling `fermatrica_rep.export.export_pptx.export_curves()`.
Period to take into account could be set via OptionSettings object.

Beware! Multiprocessing calculation to be used by functions of this file.
"""
import pandas as pd

from pptx.presentation import Presentation
from pptx.chart.data import ChartData
from pptx.util import Pt
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.dml.color import RGBColor
import lxml.etree as etree

from fermatrica import ModelConf

from fermatrica_rep.export import wording
from fermatrica_rep.export.basics import _resolve_slide_layout, set_chart_colors_line, set_chart_dashes
from fermatrica_rep.meta_model.model_rep import ModelRep
from fermatrica_rep.meta_model.node import ModelNode
from fermatrica_rep.meta_model.meta_model import MetaModel, _last_node_name
from fermatrica_rep.options.scope import OptionScope
from fermatrica_rep.options.option import Option
from fermatrica_rep.options.parallel import compute_grid
from fermatrica_rep.options.define import _enrich_trans_dict_from_model


def create(prs: Presentation,
            meta_model: MetaModel,
            *,
            scopes: list,
            translation: dict | pd.DataFrame | None = None,
            budget_step: int | float = 5,
            bdg_max: int | float | None = 301,
            fixed_vars: dict | None = {'price': 1},
            if_exact: bool = True,
            if_volume: bool = False,
            cores: int = 4,
            background_option: dict | Option | None = None,
            ):
    """
    Generate efficiency curves slides via full curves approach.

    :param prs: Presentation object from python_pptx package
    :param meta_model: MetaModel DAG
    :param scopes: list of OptionScope objects
    :param translation: translation dict or DataFrame
    :param budget_step: budget iteration step in millions
    :param bdg_max: maximum budget size; None = no cap (axis tick skip derived from grid max)
    :param fixed_vars: translation variables with their values to be fixed across grid
    :param if_exact: apply only to the specific time period
    :param if_volume: show volume or value KPI
    :param cores: number of processor cores
    :param background_option: background budget additions for compute_grid: None (default),
        a raw dict {scope_name: {item: value}}, or an Option (base budget scenario)
        whose per-scope cross-scope diff supplies the additions. See compute_grid().

    :return: Presentation object from python_pptx package
    """

    summary_node = _last_node_name(meta_model)
    model_rep_main = meta_model.nodes[summary_node].model_rep
    model_conf = meta_model.nodes[summary_node].model.conf

    date_max = meta_model.nodes[summary_node].ds['date'].dt.year.max()

    if isinstance(translation, dict):
        translation = pd.DataFrame(translation)

    language = model_rep_main.language
    slide_width = model_rep_main.pptx_cnf['slide_width']
    slide_height = model_rep_main.pptx_cnf['slide_height']

    curves_full_data = compute_grid(
        meta_model=meta_model,
        scopes=scopes,
        translation=translation,
        budget_step=budget_step,
        bdg_max=bdg_max,
        fixed_vars=fixed_vars,
        cores=cores,
        if_exact=if_exact,
        background_option=background_option,
    )[summary_node]

    slide_layout = _resolve_slide_layout(prs, model_rep_main.pptx_cnf['Blank_slide'])

    tmp = curves_full_data.groupby('option').agg({'pred_long_val': 'nunique'})
    lst = tmp.loc[tmp["pred_long_val"] > 1].reset_index().option.tolist()
    lst.sort()

    model_rep_main.fill_colours_tools(lst)

    for i in lst:
        model_rep_main.palette_tools[i + "_extrapolated"] = model_rep_main.palette_tools[i]

    scope = scopes[0] if scopes else None

    # short effect
    slide = prs.slides.add_slide(slide_layout)
    slide = _slide_vis_worker(slide, model_rep=model_rep_main,
                              model_conf=model_conf,
                              scope=scope,
                              curves_full_data=curves_full_data,
                              plot_type='short', year_end=date_max,
                              budget_step=budget_step, bdg_max=bdg_max,
                              if_volume=if_volume)

    # long effect
    slide = prs.slides.add_slide(slide_layout)
    slide = _slide_vis_worker(slide, model_rep=model_rep_main,
                              model_conf=model_conf,
                              scope=scope,
                              curves_full_data=curves_full_data,
                              plot_type='long', year_end=date_max,
                              budget_step=budget_step, bdg_max=bdg_max,
                              if_exact=if_exact, if_volume=if_volume)

    return prs


def create_legacy(prs: Presentation,
                  model = None,
                  model_rep = None,
                  ds = None,
                  opt_set_crv = None,
                  translation: dict | pd.DataFrame | None = None,
                  *,
                  budget_step: int | float = 5,
                  bdg_max: int | float | None = 301,
                  fixed_vars: dict | None = {'price': 1},
                  adhoc_curves_max_costs = None,
                  if_exact: bool = True,
                  if_volume: bool = False,
                  cores: int = 4,
                  scopes = None,
                  background_option: dict | Option | None = None,
                  ):
    import warnings
    warnings.warn(
        "Passing Model, ModelRep, ds, opt_set_crv to slides.incr_roi is deprecated. "
        "Use MetaModel and OptionScope instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    if scopes is not None:
        if not isinstance(model, MetaModel):
            node = ModelNode(model=model, ds=ds, model_rep=model_rep)
            meta_model = MetaModel(nodes=node)
        else:
            meta_model = model
        return create(prs, meta_model,
                      scopes=scopes,
                      translation=translation,
                      budget_step=budget_step,
                      bdg_max=bdg_max,
                      fixed_vars=fixed_vars,
                      if_exact=if_exact,
                      if_volume=if_volume,
                      cores=cores,
                      background_option=background_option)

    if opt_set_crv is not None or model is not None:
        return _create_old(prs, model, model_rep, ds, opt_set_crv, translation,
                           budget_step, bdg_max, fixed_vars, adhoc_curves_max_costs,
                           if_exact, if_volume, cores, background_option)

    from fermatrica.basics.basics import fermatrica_error
    fermatrica_error("Either (model, scopes) for new API or (model, opt_set_crv) for old API must be provided.")


def _create_old(prs, model, model_rep, ds, opt_set_crv, translation,
                budget_step, bdg_max, fixed_vars, adhoc_curves_max_costs,
                if_exact, if_volume, cores, background_option=None):
    """Old API implementation (deprecated, kept for backward compatibility)."""

    if isinstance(model_rep, list):
        model_rep_main = model_rep[-1]
        date_max = ds[-1]['date'].dt.year.max()
        model_conf = model[-1].conf
    else:
        model_rep_main = model_rep
        date_max = ds['date'].dt.year.max()
        model_conf = model.conf

    if isinstance(translation, dict):
        translation = pd.DataFrame(translation)

    language = model_rep_main.language
    slide_width = model_rep_main.pptx_cnf['slide_width']
    slide_height = model_rep_main.pptx_cnf['slide_height']

    # prepare slide in presentation

    if isinstance(model_rep, list):
        node = ModelNode(model=model[-1], ds=ds[-1], model_rep=model_rep[-1])
        _enrich_trans_dict_from_model(model[-1].conf, node.model_rep.trans_dict)
    else:
        node = ModelNode(model=model, ds=ds, model_rep=model_rep)
        _enrich_trans_dict_from_model(model.conf, node.model_rep.trans_dict)

    meta_model = MetaModel(nodes=node)

    opt_set_crv_scope = opt_set_crv[0] if isinstance(opt_set_crv, list) else opt_set_crv
    scope = OptionScope(
        scope_name="crv_scope",
        date_start=opt_set_crv_scope.date_start,
        date_end=opt_set_crv_scope.date_end,
        ref_date_start=opt_set_crv_scope.ref_date_start,
        ref_date_end=opt_set_crv_scope.ref_date_end,
        filters={k: [v] if not isinstance(v, list) else v
                 for k, v in zip(opt_set_crv_scope.apply_vars, opt_set_crv_scope.target)},
        plan_period=opt_set_crv_scope.plan_period,
    )

    curves_full_data = compute_grid(
        meta_model=meta_model,
        scopes=[scope],
        translation=translation,
        budget_step=budget_step,
        bdg_max=bdg_max,
        fixed_vars=fixed_vars,
        cores=cores,
        if_exact=if_exact,
        background_option=background_option,
    )["model"]

    opt_set_crv_vis = opt_set_crv[0] if isinstance(opt_set_crv, list) else opt_set_crv

    slide_layout = _resolve_slide_layout(prs, model_rep_main.pptx_cnf['Blank_slide'])

    # prepare colors
    tmp = curves_full_data.groupby('option').agg({'pred_long_val': 'nunique'})
    lst = tmp.loc[tmp["pred_long_val"] > 1].reset_index().option.tolist()
    lst.sort()

    model_rep_main.fill_colours_tools(lst)

    for i in lst:
        model_rep_main.palette_tools[i + "_extrapolated"] = model_rep_main.palette_tools[i]

    # ----------------- short effect ------------------

    slide = prs.slides.add_slide(slide_layout)

    slide = _slide_vis_worker(slide,
                              model_rep=model_rep_main,
                              model_conf=model_conf,
                              scope=scope,
                              curves_full_data=curves_full_data,
                              plot_type='short',
                              year_end=date_max,
                              budget_step=budget_step,
                              bdg_max=bdg_max,
                              if_volume=if_volume,
                              )

    # ------------------ long effect ------------------

    slide = prs.slides.add_slide(slide_layout)

    slide = _slide_vis_worker(slide,
                              model_rep=model_rep_main,
                              model_conf=model_conf,
                              scope=scope,
                              curves_full_data=curves_full_data,
                              plot_type='long',
                              year_end=date_max,
                              budget_step=budget_step,
                              bdg_max=bdg_max,
                              if_exact=if_exact,
                              if_volume=if_volume,
                              )

    return prs


def _slide_vis_worker(slide,
                      model_rep: "ModelRep",
                      scope: "OptionScope",
                      curves_full_data: pd.DataFrame,
                      plot_type: str,
                      year_end: int,
                      budget_step: int | float = 5,
                      bdg_max: int | float | None = 301,
                      if_exact: bool = True,
                      if_volume: bool = False,
                      model_conf: "ModelConf | None" = None
                      ):
    """
    Render the incremental curves chart on `slide`, plus the profit and ROI charts
    when the model conf declares value output (`if_value_applicable`). Calculations
    are already done, so this function works as visualiser only.
    
    This function calls minor workers to fill current file.

    :param slide: Slide object (from python_pptx package) to be filled
    :param model_rep: ModelRep object (reporting settings)
    :param scope: OptionScope object with date_start and date_end
    :param curves_full_data:
    :param plot_type: 'short' or 'long'
    :param year_end: final year of the calculations; to be used in naming only
    :param budget_step: budget iteration step in millions, defaults to 1 (i.e. 1M)
    :param bdg_max: maximum budget size (all options with larger budgets to be dropped)
    :param if_exact: apply only to the specific time period, without next years
    :param if_volume: show volume or value KPI
    :param model_conf: model conf (KPI/currency facts) for wording resolution

    :return:
    """

    bdg_max = bdg_max if bdg_max is not None else float(curves_full_data["bdg"].max())

    f = wording.facts(model_conf)

    # Mean-aggregated KPIs report an average per period, not a cumulative total,
    # so the long-term axis must not be labelled as a sum even on the exact path.
    summary_type = getattr(model_conf, "summary_type", "sum")
    cumulative = if_exact and plot_type == 'long' and summary_type not in ("mean", "mean_fin")

    if plot_type == 'long':
        n_years = year_end - scope.date_start.year + 1
        axis_years = str(scope.date_start.year) + '-' + str(year_end)
        title_section = 'efficiency.3.year'
    else:
        n_years = 1
        axis_years = str(scope.date_start.year)
        title_section = 'efficiency.1.year'

    slide_title = wording.text(model_rep, model_conf, section=title_section, variable='title')

    slide_height = model_rep.pptx_cnf['slide_height']
    slide_width = model_rep.pptx_cnf['slide_width']

    slide_title = slide_title.replace("***", axis_years)
    slide.shapes[0].text = slide_title
    paragraph = slide.shapes[0].text_frame.paragraphs[0]
    paragraph.font.size = model_rep.pptx_cnf["font_size_gross"]
    paragraph.font.name = model_rep.pptx_cnf["font_family_header"]
    slide.shapes[0].text_frame.word_wrap = True

    x, y = slide.shapes[0].left, slide.shapes[0].height + slide.shapes[0].top + 0.05 * slide_height
    cx, cy = slide.shapes[0].width, 0.2 * slide_height

    x, y, cx, cy = int(round(x)), int(round(y)), int(round(cx)), int(round(cy))

    textbox = slide.shapes.add_textbox(x, y, cx, cy)
    textbox.text = model_rep.pptx_cnf['mock_text']

    paragraph = textbox.text_frame.paragraphs[0]
    paragraph.font.size = model_rep.pptx_cnf["font_size_main"]
    paragraph.font.name = model_rep.pptx_cnf["font_family_body"]
    textbox.text_frame.word_wrap = True

    # Incremental sales

    if if_volume:
        plot_type_cur = 'increment_volume_' + plot_type
    else:
        plot_type_cur = 'increment_value_' + plot_type

    chart_data = _chart_data_worker(curves_full_data, plot_type_cur, if_exact, n_years)

    if f["if_value_applicable"]:
        x, y = (0.5 - 0.45) * slide_width, slide.shapes[1].top + slide.shapes[1].height
        cx, cy = 0.25 * slide_width, 0.5 * slide_height
    else:
        x, y = 0.05 * slide_width, slide.shapes[1].top + slide.shapes[1].height
        cx, cy = 0.9 * slide_width, 0.5 * slide_height

    x, y, cx, cy = int(round(x)), int(round(y)), int(round(cx)), int(round(cy))

    graphic_frame = slide.shapes.add_chart(
        XL_CHART_TYPE.LINE, x, y, cx, cy, chart_data
    )
    chart = graphic_frame.chart
    chart = set_chart_dashes(chart, '_extrapolated')

    var_type = 'incr_sales_cum_axis_y' if cumulative else 'incr_sales_axis_y'

    # When value output is unavailable, no profit/ROI charts follow, so the
    # incremental chart carries the legend itself; otherwise the ROI chart does.
    chart.has_legend = not f["if_value_applicable"]
    if chart.has_legend:
        chart.legend.position = XL_LEGEND_POSITION.RIGHT
        chart.legend.include_in_layout = False
        chart.legend.font.name = model_rep.pptx_cnf['font_family_body']
        chart.legend.font.size = model_rep.pptx_cnf['font_size_small']

    chart = _set_chart_axis(chart, model_rep,
                            model_conf=model_conf,
                            var=var_type,
                            year=axis_years,
                            plot_type=plot_type,
                            bdg_max=bdg_max,
                            if_exact=if_exact,
                            budget_step=budget_step)

    chart = set_chart_colors_line(chart, model_rep, RGBColor(170, 170, 170))

    chart.font.size = model_rep.pptx_cnf['font_size_small']
    chart.font.name = model_rep.pptx_cnf['font_family_body']

    for sr in chart.plots[0].series:
        sr.format.line.width = Pt(2.5)
        sr.smooth = True

    if f["if_value_applicable"] and curves_full_data["pred_exact_val"][1] > 1e6:
        xml_node = etree.Element("{http://schemas.openxmlformats.org/drawingml/2006/chart}dispUnits")
        etree.SubElement(xml_node, '{http://schemas.openxmlformats.org/drawingml/2006/chart}builtInUnit').attrib[
            'val'] = 'millions'
        dispUnitsLbl = etree.SubElement(xml_node,
                                        '{http://schemas.openxmlformats.org/drawingml/2006/chart}dispUnitsLbl')
        layout = etree.SubElement(dispUnitsLbl, '{http://schemas.openxmlformats.org/drawingml/2006/chart}layout')

        manual = etree.SubElement(layout, '{http://schemas.openxmlformats.org/drawingml/2006/chart}manualLayout')
        etree.SubElement(manual, '{http://schemas.openxmlformats.org/drawingml/2006/chart}xMode').attrib['val'] = 'edge'
        etree.SubElement(manual, '{http://schemas.openxmlformats.org/drawingml/2006/chart}yMode').attrib['val'] = 'edge'
        etree.SubElement(manual, '{http://schemas.openxmlformats.org/drawingml/2006/chart}x').attrib['val'] = '0.13'
        etree.SubElement(manual, '{http://schemas.openxmlformats.org/drawingml/2006/chart}y').attrib['val'] = '0.03'

        if model_rep.language == 'russian':
            tx = etree.SubElement(dispUnitsLbl, '{http://schemas.openxmlformats.org/drawingml/2006/chart}tx')
            rich = etree.SubElement(tx, '{http://schemas.openxmlformats.org/drawingml/2006/chart}rich')
            etree.SubElement(rich, '{http://schemas.openxmlformats.org/drawingml/2006/main}bodyPr')
            etree.SubElement(rich, '{http://schemas.openxmlformats.org/drawingml/2006/main}lstStyle')

            p = etree.SubElement(rich, '{http://schemas.openxmlformats.org/drawingml/2006/main}p')

            pPr = etree.SubElement(p, '{http://schemas.openxmlformats.org/drawingml/2006/main}pPr')
            etree.SubElement(pPr, '{http://schemas.openxmlformats.org/drawingml/2006/main}defRPr')

            r = etree.SubElement(p, '{http://schemas.openxmlformats.org/drawingml/2006/main}r')
            rpr = etree.SubElement(r, '{http://schemas.openxmlformats.org/drawingml/2006/main}rPr')

            rpr.attrib['lang'] = 'ru-RU'
            rpr.attrib['dirty'] = '0'

            etree.SubElement(r, '{http://schemas.openxmlformats.org/drawingml/2006/main}t').text = 'млн'

            etree.SubElement(p, '{http://schemas.openxmlformats.org/drawingml/2006/main}endParaRPr').attrib['lang'] = 'en-US'
            # etree.SubElement(p, '{http://schemas.openxmlformats.org/drawingml/2006/main}endParaRPr').attrib['dirty'] = '0'

        chart._chartSpace.xpath("//c:valAx")[0].insert(-1, xml_node)

    if not f["if_value_applicable"]:
        return slide

    # profit
    chart_data = _chart_data_worker(curves_full_data, f'profit_{plot_type}', if_exact, n_years)

    x, y = (0.5 - 0.45 + 0.25) * slide_width, slide.shapes[1].top + slide.shapes[1].height
    cx, cy = 0.25 * slide_width, 0.5 * slide_height

    x, y, cx, cy = int(round(x)), int(round(y)), int(round(cx)), int(round(cy))

    graphic_frame = slide.shapes.add_chart(
        XL_CHART_TYPE.LINE, x, y, cx, cy, chart_data
    )
    chart = graphic_frame.chart
    chart = set_chart_dashes(chart, '_extrapolated')

    chart.has_legend = False

    var_type = 'profit_cum_axis_y' if cumulative else 'profit_axis_y'

    chart = _set_chart_axis(chart
                            , model_rep,
                            model_conf=model_conf,
                            var=var_type,
                            year=axis_years,
                            if_exact=if_exact,
                            plot_type=plot_type,
                            bdg_max=bdg_max,
                            budget_step=budget_step)

    chart = set_chart_colors_line(chart, model_rep, RGBColor(170, 170, 170))

    chart.font.size = model_rep.pptx_cnf['font_size_small']
    chart.font.name = model_rep.pptx_cnf['font_family_body']

    for sr in chart.plots[0].series:
        sr.format.line.width = Pt(2.5)
        sr.smooth = True
    if curves_full_data["pred_exact_val"][1] > 1e6:
        xml_node = etree.Element("{http://schemas.openxmlformats.org/drawingml/2006/chart}dispUnits")
        etree.SubElement(xml_node, '{http://schemas.openxmlformats.org/drawingml/2006/chart}builtInUnit').attrib[
            'val'] = 'millions'
        dispUnitsLbl = etree.SubElement(xml_node,
                                        '{http://schemas.openxmlformats.org/drawingml/2006/chart}dispUnitsLbl')
        layout = etree.SubElement(dispUnitsLbl, '{http://schemas.openxmlformats.org/drawingml/2006/chart}layout')

        manual = etree.SubElement(layout, '{http://schemas.openxmlformats.org/drawingml/2006/chart}manualLayout')
        etree.SubElement(manual, '{http://schemas.openxmlformats.org/drawingml/2006/chart}xMode').attrib['val'] = 'edge'
        etree.SubElement(manual, '{http://schemas.openxmlformats.org/drawingml/2006/chart}yMode').attrib['val'] = 'edge'
        etree.SubElement(manual, '{http://schemas.openxmlformats.org/drawingml/2006/chart}x').attrib['val'] = '0.13'
        etree.SubElement(manual, '{http://schemas.openxmlformats.org/drawingml/2006/chart}y').attrib['val'] = '0.03'

        if model_rep.language == 'russian':
            tx = etree.SubElement(dispUnitsLbl, '{http://schemas.openxmlformats.org/drawingml/2006/chart}tx')
            rich = etree.SubElement(tx, '{http://schemas.openxmlformats.org/drawingml/2006/chart}rich')
            etree.SubElement(rich, '{http://schemas.openxmlformats.org/drawingml/2006/main}bodyPr')
            etree.SubElement(rich, '{http://schemas.openxmlformats.org/drawingml/2006/main}lstStyle')

            p = etree.SubElement(rich, '{http://schemas.openxmlformats.org/drawingml/2006/main}p')

            pPr = etree.SubElement(p, '{http://schemas.openxmlformats.org/drawingml/2006/main}pPr')
            etree.SubElement(pPr, '{http://schemas.openxmlformats.org/drawingml/2006/main}defRPr')

            r = etree.SubElement(p, '{http://schemas.openxmlformats.org/drawingml/2006/main}r')
            rpr = etree.SubElement(r, '{http://schemas.openxmlformats.org/drawingml/2006/main}rPr')

            rpr.attrib['lang'] = 'ru-RU'
            rpr.attrib['dirty'] = '0'

            etree.SubElement(r, '{http://schemas.openxmlformats.org/drawingml/2006/main}t').text = 'млн'

            etree.SubElement(p, '{http://schemas.openxmlformats.org/drawingml/2006/main}endParaRPr').attrib['lang'] = 'en-US'

        chart._chartSpace.xpath("//c:valAx")[0].insert(-1, xml_node)

    # ROI
    chart_data = _chart_data_worker(curves_full_data, f'ROI_{plot_type}', if_exact, n_years)

    x, y = (0.5 - 0.45 + 0.25 + 0.25) * slide_width, slide.shapes[1].top + slide.shapes[1].height
    cx, cy = 0.4 * slide_width, 0.5 * slide_height

    x, y, cx, cy = int(round(x)), int(round(y)), int(round(cx)), int(round(cy))

    graphic_frame = slide.shapes.add_chart(
        XL_CHART_TYPE.LINE, x, y, cx, cy, chart_data
    )
    chart = graphic_frame.chart
    chart = set_chart_dashes(chart, '_extrapolated')
    chart.series.smooth = True
    chart.has_legend = True
    chart.legend.position = XL_LEGEND_POSITION.RIGHT
    chart.legend.include_in_layout = False
    chart.legend.font.name = model_rep.pptx_cnf['font_family_body']
    chart.legend.font.size = model_rep.pptx_cnf['font_size_small']

    chart = _set_chart_axis(chart, model_rep,
                            model_conf=model_conf,
                            if_exact=if_exact,
                            var='roi_axis_y',
                            year=axis_years,
                            plot_type=plot_type,
                            bdg_max=bdg_max,
                            budget_step=budget_step)

    chart = set_chart_colors_line(chart, model_rep, RGBColor(170, 170, 170))

    chart.font.size = model_rep.pptx_cnf['font_size_small']
    chart.font.name = model_rep.pptx_cnf['font_family_body']

    graphic_frame.left = round((0.5 - 0.45 + 0.25 + 0.25) * slide_width)

    for sr in chart.plots[0].series:
        sr.format.line.width = Pt(2.5)
        sr.smooth = True

    return slide


def _chart_data_worker(curves_full_df: pd.DataFrame,
                       curve_type: str,
                       if_exact: bool,
                       n_years: int = 1):
    """
    Prepares already calculated efficiency curves data for specific curve. 

    :param curves_full_df: calculated efficiency curves data
    :param curve_type: 'increment_volume_short', 'increment_value_short', 'profit_short', 'ROI_short'
        , 'increment_volume_long', 'increment_value_long', 'profit_long', 'ROI_long'
    :param if_exact: apply only to the specific time period, without next years
    :param n_years: number of next years accounted in 'long' curves
    :return: ChartData object from python_pptx package
    """

    chart_data = ChartData()
    chart_data.categories = curves_full_df['bdg'].sort_values().unique()

    match curve_type:
        case 'increment_volume_short':
            zero = curves_full_df.loc[curves_full_df['option'] == 'zero', 'pred_exact_vol'].values
            curves_full_df['value'] = curves_full_df['pred_exact_vol'] - zero
        case 'increment_value_short':
            zero = curves_full_df.loc[curves_full_df['option'] == 'zero', 'pred_exact_val'].values
            curves_full_df['value'] = curves_full_df['pred_exact_val'] - zero
        case 'profit_short':
            zero = curves_full_df.loc[curves_full_df['option'] == 'zero', 'pred_exact_val'].values
            curves_full_df['value'] = curves_full_df['pred_exact_val'] - zero - curves_full_df['bdg'] * 1e6
        case 'ROI_short':
            zero = curves_full_df.loc[curves_full_df['option'] == 'zero', 'pred_exact_val'].values
            curves_full_df['value'] = (curves_full_df['pred_exact_val'] - zero) / (curves_full_df['bdg'] * 1e6)
        case 'increment_volume_long':
            zero = curves_full_df.loc[curves_full_df['option'] == 'zero', 'pred_long_vol'].values
            curves_full_df['value'] = curves_full_df['pred_long_vol'] - zero
        case 'increment_value_long':
            zero = curves_full_df.loc[curves_full_df['option'] == 'zero', 'pred_long_val'].values
            curves_full_df['value'] = curves_full_df['pred_long_val'] - zero
        case 'profit_long':
            zero = curves_full_df.loc[curves_full_df['option'] == 'zero', 'pred_long_val'].values
            # fix for correct calcs with if_exact = False
            if if_exact is False:
                curves_full_df['value'] = (curves_full_df['pred_long_val'] - zero) - curves_full_df['bdg'] * 1e6 * n_years
            else:
                curves_full_df['value'] = (curves_full_df['pred_long_val'] - zero) - curves_full_df['bdg'] * 1e6
        case 'ROI_long':
            zero = curves_full_df.loc[curves_full_df['option'] == 'zero', 'pred_long_val'].values
            curves_full_df['value'] = (curves_full_df['pred_long_val'] - zero) / (curves_full_df['bdg'] * 1e6)

    if ("long" in curve_type) & (if_exact is False):
        curves_full_df['value'] = curves_full_df['value'] / n_years

    tmp = curves_full_df.groupby('option').agg({'value': 'nunique'})
    opts = tmp.loc[tmp['value'] > 1].reset_index().option.tolist()
    
    for opt in opts:
        
        tmp = curves_full_df.loc[(curves_full_df['option'] == opt)].copy()

        cols = [x for x in tmp.columns.tolist() if x not in ['option']]
        tmp.loc[-1, cols] = 0

        if "ROI" in curve_type:
            cols = [x for x in tmp.columns.tolist() if x not in ['option', 'max_obs_budg', 'bdg']]
            tmp[cols] = tmp[cols].astype(object)
            tmp.loc[-1, cols] = "#N/A"

        tmp.index = tmp.index + 1
        tmp.sort_index(inplace=True)
        if 'max_obs_budg' in curves_full_df.columns.tolist():
            tmp_obs = tmp.loc[(tmp['bdg'] <= tmp['max_obs_budg'])].copy()
            if len(tmp_obs) > 0:
                chart_data.add_series(opt, tmp_obs['value'])
                last_obs = tmp_obs.iloc[-1]

            tmp_extra = tmp.loc[(tmp['bdg'] > tmp['max_obs_budg'])].copy()
            if len(tmp_extra) > 0:
                if len(tmp_obs) > 0:
                    tmp_extra.loc[-1] = last_obs
                    tmp_extra.index = tmp_extra.index + 1
                    tmp_extra.sort_index(inplace=True)
                chart_data.add_series(opt + '_extrapolated',
                                      pd.concat([pd.Series([''] * (len(tmp_obs) - 1)), tmp_extra['value']], ignore_index=True))

        else:
            chart_data.add_series(opt, tmp['value'])

    return chart_data


def _set_chart_axis(chart,
                    model_rep: "ModelRep",
                    var: str,
                    year: str,
                    budget_step: int | float = 5,
                    bdg_max: int | float = 301,
                    plot_type: str = 'short',
                    if_exact: bool = False,
                    model_conf: "ModelConf | None" = None):
    """
    Set axis attributes for efficiency curves chart: axis titles, formatting etc.

    :param chart: PPTX chart
    :param model_rep: ModelRep object (reporting settings)
    :param var: kind of plot type; cursed
    :param year: year or years (as string with dash) to be added to titles
    :param budget_step: budget iteration step in millions, defaults to 1 (i.e. 1M)
    :param bdg_max: maximum budget size (all options with larger budgets to be dropped)
    :param plot_type: 'short' or 'long'
    :param if_exact: apply only to the specific time period, without next years
    :param model_conf: model conf (KPI/currency facts) for wording resolution
    :return:
    """

    value_axis = chart.value_axis
    value_axis_title = wording.text(model_rep, model_conf,
                                    section=f'efficiency_{plot_type}', variable=var)
    value_axis_title = value_axis_title.replace('***', year)
    value_axis.axis_title.text_frame.text = value_axis_title
    value_axis.axis_title.text_frame.paragraphs[0].font.size = model_rep.pptx_cnf['font_size_small']

    category_axis = chart.category_axis
    category_axis.has_title = True
    if if_exact & (plot_type == 'long'):
        category_axis_title = wording.text(model_rep, model_conf,
                                           section=f'efficiency_{plot_type}',
                                           variable='axis_x_if_exact_true')

        year = year[-4:]
    elif (not if_exact) & (plot_type == 'long'):
        category_axis_title = wording.text(model_rep, model_conf,
                                           section=f'efficiency_{plot_type}',
                                           variable='axis_x_if_exact_false')
    else:
        category_axis_title = wording.text(model_rep, model_conf,
                                           section=f'efficiency_{plot_type}',
                                           variable='axis_x')

    category_axis_title = category_axis_title.replace('***', year)
    category_axis.axis_title.text_frame.text = category_axis_title
    category_axis.axis_title.text_frame.paragraphs[0].font.size = model_rep.pptx_cnf['font_size_small']

    etree.SubElement(chart._chartSpace.xpath("//c:valAx")[0], '{http://schemas.openxmlformats.org/drawingml/2006/chart}crossBetween').attrib['val'] = 'midCat'

    skip = str(round((bdg_max / budget_step) / 5))

    if int(skip) < 1:
        skip = "1"

    etree.SubElement(chart._chartSpace.xpath("//c:catAx")[0], '{http://schemas.openxmlformats.org/drawingml/2006/chart}tickLblSkip').attrib['val'] = skip
    etree.SubElement(chart._chartSpace.xpath("//c:catAx")[0], '{http://schemas.openxmlformats.org/drawingml/2006/chart}tickMarkSkip').attrib['val'] = skip

    return chart


