"""
Basic utilities for PPTX export of FERMATRICA_REP modelling reporting.
"""


import re

import pandas as pd
from fermatrica_utils import hex_to_rgb
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.text import PP_PARAGRAPH_ALIGNMENT
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Cm, Pt

from fermatrica_rep import ModelRep


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
