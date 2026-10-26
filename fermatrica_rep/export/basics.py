"""
Basic utilities for PPTX export of FERMATRICA_REP modelling reporting.
"""


import copy
import re
import pandas as pd

from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.text import PP_PARAGRAPH_ALIGNMENT
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Cm, Pt

from fermatrica_utils import hex_to_rgb

from fermatrica_rep import ModelRep
from fermatrica_rep.options.define import OptionSettings
from fermatrica_rep.options.scope import OptionScope


def set_chart_colors_line(chart
                          , model_rep  #: "ModelRep"
                          , gridlines_color: None | RGBColor = None
                          ):
    """
    Set chart lines from marketing tools palette (`ModelRep.palette_tools`).

    :param chart: PPTX chart
    :param model_rep: ModelRep object
    :param gridlines_color: None or something like RGBColor(170, 170, 170)
    :return:
    """

    for x in chart.series:
        if x.name in model_rep.palette_tools:
            x.format.line.color.rgb = RGBColor(*hex_to_rgb(model_rep.palette_tools[x.name]))

    if gridlines_color is not None:
        chart.value_axis.major_gridlines.format.line.color.rgb = gridlines_color

    return chart


def set_chart_colors_fill(chart
                          , model_rep  #: "ModelRep"
                          , gridlines_color: None | RGBColor = None):
    """
    Set chart fill colours from marketing tools palette (`ModelRep.palette_tools`) + grid color

    :param chart: PPTX chart
    :param model_rep: ModelRep object
    :param gridlines_color: None or something like RGBColor(170, 170, 170)
    :return:
    """
    for x in chart.series:
        if x.name in model_rep.palette_tools:
            x.format.fill.solid()
            x.format.fill.fore_color.rgb = RGBColor(*hex_to_rgb(model_rep.palette_tools[x.name]))

    if gridlines_color is not None:
        chart.value_axis.major_gridlines.format.line.color.rgb = gridlines_color

    return chart


def _sub_element(parent
                 , tagname
                 , **kwargs):
    """
    Worker function to set simple sub-element with attributes to the parent element

    :param parent: element to bind the child
    :param tagname: name of the child element
    :param kwargs: attributes of the child
    :return: element (as part of the larger XML / PPTX object)
    """

    element = OxmlElement(tagname)
    element.attrib.update(kwargs)
    parent.append(element)

    return element


def set_cell_border(cell,
                    border_color: str = "FFFFFF",
                    border_width: str | int = '12700',
                    borders_positions: tuple | str = ('left', 'right', 'top', 'bottom')):
    """
    Hack function to set border width and border color:
        - left border
        - right border
        - top border
        - bottom border

    Assigned is performed by reference, so not to be upset with 'not used' variables (actually used)

    :param cell: table cell object to add border to
    :param border_color: border HEX color
    :param border_width: border with in ...
    :param borders_positions: any combination of 'left', 'right', 'top', 'bottom'
    :return:
    """

    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()

    if isinstance(border_width, int):
        border_width = str(border_width)

    if isinstance(borders_positions, str):
        borders_positions = [borders_positions]

    # Left Cell Border
    if 'left' in borders_positions:
        lnL = _sub_element(tcPr, 'a:lnL', w=border_width, cap='flat', cmpd='sng', algn='ctr')
        lnL_solidFill = _sub_element(lnL, 'a:solidFill')
        lnL_srgbClr = _sub_element(lnL_solidFill, 'a:srgbClr', val=border_color)
        lnL_prstDash = _sub_element(lnL, 'a:prstDash', val='solid')
        lnL_round_ = _sub_element(lnL, 'a:round')
        lnL_headEnd = _sub_element(lnL, 'a:headEnd', type='none', w='med', len='med')
        lnL_tailEnd = _sub_element(lnL, 'a:tailEnd', type='none', w='med', len='med')

    # Right Cell Border
    if 'right' in borders_positions:
        lnR = _sub_element(tcPr, 'a:lnR', w=border_width, cap='flat', cmpd='sng', algn='ctr')
        lnR_solidFill = _sub_element(lnR, 'a:solidFill')
        lnR_srgbClr = _sub_element(lnR_solidFill, 'a:srgbClr', val=border_color)
        lnR_prstDash = _sub_element(lnR, 'a:prstDash', val='solid')
        lnR_round_ = _sub_element(lnR, 'a:round')
        lnR_headEnd = _sub_element(lnR, 'a:headEnd', type='none', w='med', len='med')
        lnR_tailEnd = _sub_element(lnR, 'a:tailEnd', type='none', w='med', len='med')

    # Top Cell Border
    if 'top' in borders_positions:
        lnT = _sub_element(tcPr, 'a:lnT', w=border_width, cap='flat', cmpd='sng', algn='ctr')
        lnT_solidFill = _sub_element(lnT, 'a:solidFill')
        lnT_srgbClr = _sub_element(lnT_solidFill, 'a:srgbClr', val=border_color)
        lnT_prstDash = _sub_element(lnT, 'a:prstDash', val='solid')
        lnT_round_ = _sub_element(lnT, 'a:round')
        lnT_headEnd = _sub_element(lnT, 'a:headEnd', type='none', w='med', len='med')
        lnT_tailEnd = _sub_element(lnT, 'a:tailEnd', type='none', w='med', len='med')

    # Bottom Cell Border
    if 'bottom' in borders_positions:
        lnB = _sub_element(tcPr, 'a:lnB', w=border_width, cap='flat', cmpd='sng', algn='ctr')
        lnB_solidFill = _sub_element(lnB, 'a:solidFill')
        lnB_srgbClr = _sub_element(lnB_solidFill, 'a:srgbClr', val=border_color)
        lnB_prstDash = _sub_element(lnB, 'a:prstDash', val='solid')
        lnB_round_ = _sub_element(lnB, 'a:round')
        lnB_headEnd = _sub_element(lnB, 'a:headEnd', type='none', w='med', len='med')
        lnB_tailEnd = _sub_element(lnB, 'a:tailEnd', type='none', w='med', len='med')


def set_table_border(table,
                     border_color: str = '595959',
                     border_width: str = '12700'):
    """
    Hack function to set border width and border color through entire table, zebra coloration.

    :param table: Table object from python_pptx package
    :param border_color: border HEX color
    :param border_width: border with in ...
    :return: void
    """

    for n in range(0, len(table.columns)):
        set_cell_border(cell=table.cell(0, n),
                        border_color=border_color,
                        border_width=border_width,
                        borders_positions=('top', 'bottom'))

    for n in range(0, len(table.columns)):
        set_cell_border(cell=table.cell(len(table.rows) - 1, n),
                        border_color=border_color,
                        border_width=border_width,
                        borders_positions=('bottom',))


def fill_table(table
               , ds: pd.DataFrame):
    """
    Fill table (as Table from python_pptx package) with data from `ds` dataset.
    Use it when generating PPTX slides.

    :param table: empty table (Table from python_pptx package)
    :param ds: dataset to be placed into `table`
    :return: void
    """

    for n, x in enumerate(ds.columns):

        table.cell(0, n).text = x
        table.cell(0, n).margin_bottom = Cm(0.1)
        table.cell(0, n).margin_top = Cm(0.1)

    for index, row in ds.iterrows():

        for n, item in enumerate(row):
            if isinstance(item, str):
                table.cell(index + 1, n).text = item
            elif isinstance(item, float):
                table.cell(index + 1, n).text = str(round(item, 2))
            else:
                table.cell(index + 1, n).text = str(int(item))
            table.cell(index + 1, n).margin_bottom = Pt(0)
            table.cell(index + 1, n).margin_top = Pt(0)


def set_chart_dashes(chart
                     , ptrn: str = '_extrapolated'):
    """
    Filters series by `ptrn` regex pattern and set corresponding line type to dashed.

    :param chart: Chart object from python_pptx package
    :param ptrn: pattern to select series to make dashed
    :return: Chart object from python_pptx package
    """

    for x in chart.series:
        if re.search(ptrn, x.name):
            x.format.line.dash_style = MSO_LINE_DASH_STYLE.DASH

    return chart


def adjust_table_width(table,
                       slide_width: int | float):
    """
    Adjust table width to match slide width in PPTX.

    :param table: Table object from python_pptx package
    :param slide_width: slide width (should be saved in ModelRep object whith `config_set()`)
    :return: Table object from python_pptx package
    """

    # 0.9 of slide width

    # adjust width
    for column in table.columns:
        column.width = round(1/10 * 0.9 * slide_width)
    table.columns[0].width = round(1/5 * 0.9 * slide_width)

    return table


def table_text_format(table
                      , model_rep: "ModelRep"):
    """
    Format text in table in PPTX.

    :param table: Table object from python_pptx package
    :param model_rep: ModelRep object (reporting settings)
    :return: Table object from python_pptx package
    """

    for ind, cell in enumerate(table.iter_cells()):
        for paragraph in cell.text_frame.paragraphs:
            paragraph.font.size = model_rep.pptx_cnf['font_size_small']
            paragraph.font.name = model_rep.pptx_cnf['font_family_body']
            paragraph.alignment = PP_PARAGRAPH_ALIGNMENT.CENTER

    return table


def _extract_options_data(options_m: dict
                          , option_key: str
                          , opt_set: "OptionSettings | list[OptionSettings]"
                          , is_list_mode: bool = False):
    """
    Extract options data for a given option key, handling both list and single opt_set.

    :param options_m: dictionary of dictionaries defining options
    :param option_key: key to extract from options_m
    :param opt_set: OptionSetting object or list of OptionSettings
    :param is_list_mode: whether opt_set is a chronological list of OptionSettings
    :return: list of dicts (if opt_set is list) or single dict
    """
    if isinstance(opt_set, list) and is_list_mode:
        return [copy.deepcopy(x) for k, x in options_m[option_key].items()
                if k in range(opt_set[0].date_start.year, opt_set[-1].date_end.year + 1)]
    else:
        return options_m[option_key]


def _resolve_slide_layout(prs, layout_name: str = 'Blank_slide'):
    """
    Resolve a slide layout by name from a Presentation object.

    Avoids storing a live ``SlideLayout`` object in configuration dicts that may be
    deep-copied. Storing the layout object leads to duplicate Parts in the output ZIP
    because ``Part.__hash__`` is identity-based and cannot deduplicate deep-copied
    layout instances with the same partname.

    :param prs: ``Presentation`` object from python-pptx
    :param layout_name: name of the slide layout to look up (default ``'Blank_slide'``)
    :return: ``SlideLayout`` object
    """
    for master in prs.slide_masters:
        for layout in master.slide_layouts:
            if layout.name == layout_name:
                return layout
    return prs.slide_layouts[0]


def resolve_scopes_for_node(node_result: dict, scopes: list[OptionScope]) -> list[tuple[OptionScope, str]]:
    """
    Return all ``(OptionScope, scope_name)`` present in this node's compute result.

    Finds every scope in ``scopes`` whose ``scope_name`` is a key
    in ``node_result`` **and** whose summary has a non-zero prediction.
    Skips scopes with all-zero (or ``None``) results. Scopes are
    returned in ``scopes`` order. An empty list means no scope of this
    node produced a non-zero prediction.

    ``node_result`` values may be ``(ds, dt_pred, summary)`` tuples
    (``return_full=True``) or plain summary dicts.

    :param node_result: dict ``{scope_name: summary_or_tuple}`` for one node
    :param scopes: list of ``OptionScope`` objects, in priority order
    :return: list of ``(OptionScope, scope_name)`` tuples
    """
    resolved = []
    for scope in scopes:
        sresult = node_result.get(scope.scope_name)
        if sresult is None:
            continue
        if isinstance(sresult, tuple):
            sresult = sresult[2]
        if not isinstance(sresult, dict):
            continue
        vol = sresult.get("pred_exact_vol")
        if vol and vol > 0:
            resolved.append((scope, scope.scope_name))
    return resolved


def resolve_scope_for_node(node_result: dict, scopes: list[OptionScope]) -> tuple[OptionScope, str]:
    """
    Return the ``(OptionScope, scope_name)`` present in this node's compute result.

    Thin wrapper over :func:`resolve_scopes_for_node` returning the **first**
    non-zero scope (or ``scopes[0]`` when no scope matches), kept for backward
    compatibility. New code should use the plural version to obtain every
    scope an option actually covers.

    :param node_result: dict ``{scope_name: summary_or_tuple}`` for one node
    :param scopes: list of ``OptionScope`` objects, in priority order
    :return: ``(OptionScope, scope_name)``
    """
    resolved = resolve_scopes_for_node(node_result, scopes)
    if resolved:
        return resolved[0]
    return scopes[0], scopes[0].scope_name
