"""
Generate PowerPoint presentation with standard slides describing model. Slides / components
available to export now are:

- Retro analysis
    - Fit and predict by superbrand
    - Fit and predict by SKU / arbitrary set of variables
    - Dynamic decomposition by superbrand
    - Dynamic decomposition by SKU / arbitrary set of variables
    - Waterfall decomposition for arbitrary historical period
- Marketing tools efficiency
    - Short-term curves
        - Incremental KPI volume
        - Incremental KPI value
        - ROI
    - Long-term curves
        - Incremental KPI volume
        - Incremental KPI value
        - ROI
- Summary table for set of predefined budget options for future periods
- Detailed report for every predefined budget option for future periods
    - Fit and predict by superbrand
    - Fit and predict by SKU / arbitrary set of variables
    - Dynamic decomposition by superbrand
    - Dynamic decomposition by SKU / arbitrary set of variables
    - Waterfall decomposition for defined future period

There are two ways to generate presentation:
1. `create_presentation()` generates presentation at once, could look shorter, but is not flexible
2. number of functions to generate presentation slide by slide (or block of slide after another).
    More verbose, but also more flexible
"""


import os
import tempfile
import shutil
import pandas as pd
import io
import pkgutil
import sys
from typing_extensions import Callable

from pptx import Presentation
from pptx.util import Pt
from lxml import etree

from fermatrica import Model

from fermatrica_rep.model_rep import ModelRep
from fermatrica_rep.decomposition import extract_effect
from fermatrica_rep.options.define import OptionSettings
import fermatrica_rep.export.slides as slides
import fermatrica_rep.options.calc as calc


"""
Generate presentation at once
"""


def create_presentation(prs,
                        model: "Model",
                        model_rep: "ModelRep",
                        dt_pred: pd.DataFrame,
                        ds: pd.DataFrame,
                        opt_set: "OptionSettings",
                        opt_set_crv: "OptionSettings",
                        trans_dict: dict,
                        options_m: dict,
                        target: list,
                        apply_vars: list,
                        sku_var: list,
                        budget_step: int | float = 5,
                        cores: int = 4,
                        if_volume: bool = True,
                        if_exact: bool = True,
                        if_decompose: bool = True,
                        if_waterfall: bool = True,
                        if_options_table: bool = True,
                        if_incr_roi: bool = True,
                        if_options_slides: bool = True,
                        if_sku_fit_predict: bool = True,
                        if_sku_decompose: bool = True,
                        if_sku_waterfall: bool = True,
                        custom_extract_effect: "Callable | None" = None):
    """
    Generate reporting presentation in one step. Could look shorter, less flexible

    :param prs: Presentation object from python_pptx package
    :param model: Model object
    :param model_rep: ModelRep object (reporting settings)
    :param dt_pred: prediction data
    :param ds: main dataset
    :param opt_set: OptionSetting object containing calculate settings
    :param opt_set_crv: OptionSetting object containing calculate settings for curve calculation
    :param trans_dict: translation dictionary (how to "translate" options into independent variables)
    :param options_m: dictionary of dictionaries defining options to calculate in detailed report and option table
    :param target: values in `apply_vars` columns to filter target entities to apply option
    :param apply_vars: columns (variables) to filter target entities to apply option
    :param sku_var: columns (variables) to report as "SKU" in detailed reporting
    :param budget_step: budget iteration step in millions, defaults to 1 (i.e. 1M) (to calculate curves)
    :param cores: number of processor cores to use in multiprocessing calculations.
        None sets to all computer logical cores - 1
    :param if_volume: optimize volume or value KPI
    :param if_exact: apply only to specific time period, without next years
    :param if_decompose: include decomposition plot in retro reporting
    :param if_waterfall: include waterfall plot in retro reporting
    :param if_options_table: include options table
    :param if_incr_roi: include efficiency curves
    :param if_options_slides: include detailed reporting for every option
    :param if_sku_fit_predict: include fit-predict per SKU in options slides
    :param if_sku_decompose: include decomposition per SKU in options slides
    :param if_sku_waterfall: include waterfall per SKU in options slides
    :param custom_extract_effect: custom function to extract effect for decomposition and waterfall
    :return: Presentation object from python_pptx package
    """

    superbrand = model.conf.target_superbrand

    if custom_extract_effect is None:
        split_m_m = extract_effect(model, ds, model_rep)
    else:
        split_m_m = custom_extract_effect(model, ds, model_rep)

    df_bs_keys = dt_pred[dt_pred['superbrand'] == superbrand][sku_var].drop_duplicates()

    if if_decompose:
        prs = slides.decomposition(prs=prs,
                                   model=model,
                                   model_rep=model_rep,
                                   split_m_m=split_m_m,
                                   option_name=' '.join(sum(target, [])).title(),
                                   brands=[superbrand],
                                   period='month',
                                   show_future=True,
                                   group_var=apply_vars,
                                   if_volume=if_volume,
                                   bs_key_filter=split_m_m.loc[
                                       (split_m_m['superbrand'] == superbrand), 'bs_key'].unique())
    if if_waterfall:
        prs = slides.waterfall(prs=prs,
                               model=model,
                               model_rep=model_rep,
                               split_m_m=split_m_m,
                               date_start=str(opt_set.date_start),
                               date_end=str(opt_set.date_end),
                               option_name=' '.join(sum(target, [])).title(),
                               brands=[superbrand],
                               absolute_sort=False,
                               if_volume=if_volume,
                               bs_key_filter=split_m_m.loc[
                                   (split_m_m['superbrand'] == superbrand), 'bs_key'].unique())
    if if_incr_roi:
        prs = slides.incr_roi(prs=prs,
                              model=model,
                              model_rep=model_rep,
                              ds=ds,
                              opt_set_crv=opt_set_crv,
                              translation=pd.DataFrame(trans_dict),
                              budget_step=budget_step,
                              if_exact=if_exact,
                              cores=cores)
    if if_options_table:
        prs = slides.options(prs=prs,
                             model=model,
                             model_rep=model_rep,
                             dt_pred=dt_pred,
                             ds=ds,
                             opt_set=opt_set,
                             options_m=options_m,
                             if_exact=if_exact,
                             bs_key_filter=split_m_m.loc[
                                 (split_m_m['superbrand'] == superbrand), 'bs_key'].unique())

    if if_options_slides:
        for option_name, option in options_m.items():
            ds, dt_pred, opt_sum = calc.option_report(model, ds, model_rep,
                                                        option, opt_set, if_exact=if_exact)
            split_m_m = extract_effect(model, ds, model_rep)

            prs = slides.fit_predict(prs=prs,
                                     model=model,
                                     model_rep=model_rep,
                                     dt_pred=dt_pred,
                                     option=option,
                                     option_name=option_name + '. ' + ' '.join(sum(target, [])).title(),
                                     opt_set=opt_set,
                                     period='month',
                                     show_future=True,
                                     group_var=apply_vars,
                                     bs_key_filter=dt_pred.loc[
                                         (dt_pred['superbrand'] == superbrand), 'bs_key'].unique())

            prs = slides.decomposition(prs=prs,
                                       model=model,
                                       model_rep=model_rep,
                                       split_m_m=split_m_m,
                                       option_name=option_name + '. ' + ' '.join(sum(target, [])).title(),
                                       brands=[superbrand],
                                       period='month',
                                       show_future=True,
                                       group_var=apply_vars,
                                       plot_type='brand',
                                       if_volume=if_volume,
                                       bs_key_filter=split_m_m.loc[
                                           (split_m_m['superbrand'] == superbrand), 'bs_key'].unique())

            prs = slides.waterfall(prs=prs,
                                   model=model,
                                   model_rep=model_rep,
                                   date_start=str(opt_set.date_start),
                                   date_end=str(opt_set.date_end),
                                   option_name=option_name + '. ' + ' '.join(sum(target, [])).title(),
                                   brands=[superbrand],
                                   absolute_sort=False,
                                   split_m_m=split_m_m,
                                   plot_type='brand',
                                   if_volume=if_volume,
                                   bs_key_filter=split_m_m.loc[
                                       (split_m_m['superbrand'] == superbrand), 'bs_key'].unique())

            # if if_options_sku:
            if any([if_sku_fit_predict, if_sku_decompose, if_sku_waterfall]):
                for index, key_row in df_bs_keys.iterrows():
                    bs_keys = key_row.to_frame().transpose().merge(dt_pred, on=sku_var, how='left')['bs_key'].unique()

                    if if_sku_fit_predict:
                        prs = slides.fit_predict(prs=prs,
                                                 model=model,
                                                 model_rep=model_rep,
                                                 dt_pred=dt_pred,
                                                 option=option,
                                                 option_name=option_name + '. ' + ' '.join(
                                                     key_row.astype(str)).title(),
                                                 opt_set=opt_set,
                                                 period='month',
                                                 show_future=True,
                                                 group_var=sku_var,
                                                 bs_key_filter=bs_keys)

                    if if_sku_decompose:
                        prs = slides.decomposition(prs=prs,
                                                   model=model,
                                                   model_rep=model_rep,
                                                   split_m_m=split_m_m,
                                                   option_name=option_name + '. ' + ' '.join(
                                                       key_row.astype(str)).title(),
                                                   brands=[superbrand],
                                                   period='month',
                                                   show_future=True,
                                                   group_var=sku_var,
                                                   plot_type='sku',
                                                   if_volume=if_volume,
                                                   bs_key_filter=bs_keys)

                    if if_sku_waterfall:
                        prs = slides.waterfall(prs=prs,
                                               model=model,
                                               model_rep=model_rep,
                                               split_m_m=split_m_m,
                                               date_start=str(opt_set.date_start),
                                               date_end=str(opt_set.date_end),
                                               option_name=option_name + '. ' + ' '.join(
                                                   key_row.astype(str)).title(),
                                               brands=[superbrand],
                                               absolute_sort=False,
                                               plot_type='sku',
                                               if_volume=if_volume,
                                               bs_key_filter=bs_keys)

    return prs


"""
Generate presentation step by step
"""


def export_option_table(prs,
                        model: "Model | list",
                        model_rep: "ModelRep | list",
                        dt_pred: pd.DataFrame | list,
                        ds: pd.DataFrame | list,
                        opt_set: "OptionSettings",
                        options_m: dict,
                        if_exact: bool = True,
                        ):
    """
    Export options summary as a table.

    :param prs: Presentation object from python_pptx package
    :param model: Model object or list of Model objects
    :param model_rep: ModelRep object (reporting settings) or list ModelRep objects
    :param dt_pred: prediction data or list of prediction datas
    :param ds: main dataset or list of main datasets
    :param opt_set: OptionSetting object containing calculate settings
    :param options_m: dictionary of dictionaries defining options to calculate in detailed report and option table
    :param if_exact: apply only to specific time period, without next years
    :return: Presentation object from python_pptx package
    """

    if isinstance(model, list):
        superbrand = model[-1].conf.target_superbrand
        bs_key_filter = dt_pred[-1].loc[(dt_pred[-1]['superbrand'] == superbrand), 'bs_key'].unique()
    else:
        superbrand = model.conf.target_superbrand
        bs_key_filter = dt_pred.loc[(dt_pred['superbrand'] == superbrand), 'bs_key'].unique()

    prs = slides.options(prs=prs,
                         model=model,
                         model_rep=model_rep,
                         dt_pred=dt_pred,
                         ds=ds,
                         options_m=options_m,
                         opt_set=opt_set,
                         if_exact=if_exact,
                         bs_key_filter=bs_key_filter)

    return prs


def export_option_detail(prs,
                         model: "Model | list",
                         model_rep: "ModelRep | list",
                         dt_pred: pd.DataFrame | list,
                         ds: pd.DataFrame | list,
                         opt_set: "OptionSettings",
                         sku_var: list | None = None,
                         options_m: dict | None = None,
                         period: str = 'month',  # 'day', 'week', 'month', 'quarter', 'year'
                         if_volume: bool = True,
                         if_exact: bool = True,
                         if_fit: bool = True,
                         if_decompose: bool = True,
                         if_waterfall: bool = True,
                         if_sku_fit_predict: bool = True,
                         if_sku_decompose: bool = True,
                         if_sku_waterfall: bool = True,
                         custom_extract_effect=None
                         ):
    """
    Export detailing report per every option in `options_m` dictionary.
    Set `None` to `options_m` param to get retro analysis (w/o future period).

    :param prs: Presentation object from python_pptx package
    :param model: Model object or list of Model objects
    :param model_rep: ModelRep object (reporting settings) or list ModelRep objects
    :param dt_pred: prediction data or list of prediction datas
    :param ds: main dataset or list of main datasets
    :param opt_set: OptionSetting object containing calculate settings
    :param sku_var: columns (variables) to report as "SKU" in detailed reporting
    :param options_m: dictionary of dictionaries defining options to calculate in detailed report and option table
    :param period: time period / interval to group by: 'day', 'week', 'month', 'quarter', 'year'
    :param if_volume: optimize volume or value KPI
    :param if_exact: apply only to specific time period, without next years
    :param if_fit: include fit-predict plot
    :param if_decompose: include decomposition plot
    :param if_waterfall: include waterfall plot
    :param if_sku_fit_predict: include fit-predict per SKU
    :param if_sku_decompose: include decomposition per SKU
    :param if_sku_waterfall: include waterfall per SKU
    :param custom_extract_effect: custom function to extract effect for decomposition and waterfall
    :return: Presentation object from python_pptx package
    """

    # prepare

    if isinstance(model, list):
        superbrand = model[-1].conf.target_superbrand
        if any([if_sku_fit_predict, if_sku_decompose, if_sku_waterfall]):
            df_bs_keys = dt_pred[-1][dt_pred[-1]['superbrand'] == superbrand][sku_var].drop_duplicates()
    else:
        superbrand = model.conf.target_superbrand
        if any([if_sku_fit_predict, if_sku_decompose, if_sku_waterfall]):
            df_bs_keys = dt_pred[dt_pred['superbrand'] == superbrand][sku_var].drop_duplicates()

    if options_m is None:
        options_m = {'': None}

    target_var = opt_set.target
    group_var = opt_set.apply_vars

    # iterate over options

    for option_name, option in options_m.items():

        # if option is None = no options or retroanalysis

        if option is not None:
            ds, dt_pred, opt_sum = calc.option_report(model, ds, model_rep, option, opt_set,
                                                        if_exact=if_exact)
            opt_title = option_name + '. ' + ' '.join(sum(target_var, [])).title()
            if_retro = False
        else:
            opt_title = ' '.join(sum(target_var, [])).title()
            if_retro = True

        if isinstance(model, list):
            model = model[-1]
            ds = ds[-1]
            dt_pred = dt_pred[-1]
            model_rep = model_rep[-1]

        if custom_extract_effect is None:
            split_m_m = extract_effect(model, ds, model_rep)
        else:
            split_m_m = custom_extract_effect(model, ds, model_rep)

        # main (superbrand detailing)

        if if_fit:
            prs = slides.fit_predict(prs=prs,
                                     model=model,
                                     model_rep=model_rep,
                                     dt_pred=dt_pred,
                                     option=option,
                                     option_name=opt_title,
                                     opt_set=opt_set,
                                     period=period,
                                     show_future=True,
                                     group_var=group_var,
                                     plot_type='brand' + if_retro * '_retro',
                                     bs_key_filter=dt_pred.loc[
                                         (dt_pred['superbrand'] == superbrand), 'bs_key'].unique())

        if if_decompose:
            prs = slides.decomposition(prs=prs,
                                       model=model,
                                       model_rep=model_rep,
                                       option_name=opt_title,
                                       brands=[superbrand],
                                       split_m_m=split_m_m,
                                       period=period,
                                       show_future=True,
                                       group_var=group_var,
                                       plot_type='brand' + if_retro * '_retro',
                                       if_volume=if_volume,
                                       bs_key_filter=split_m_m.loc[
                                           (split_m_m['superbrand'] == superbrand), 'bs_key'].unique())

        if if_waterfall:
            prs = slides.waterfall(prs=prs,
                                   model=model,
                                   model_rep=model_rep,
                                   date_start=str(opt_set.date_start),
                                   date_end=str(opt_set.date_end),
                                   option_name=opt_title,
                                   brands=[superbrand],
                                   absolute_sort=False,
                                   split_m_m=split_m_m,
                                   plot_type='brand' + if_retro * '_retro',
                                   if_volume=if_volume,
                                   bs_key_filter=split_m_m.loc[
                                       (split_m_m['superbrand'] == superbrand), 'bs_key'].unique())

        # sku (or other deep dive) detailing

        if any([if_sku_fit_predict, if_sku_decompose, if_sku_waterfall]):

            for index, key_row in df_bs_keys.iterrows():
                bs_keys = key_row.to_frame().transpose().merge(dt_pred, on=sku_var, how='left')['bs_key'].unique()

                if option is not None:
                    opt_title = option_name + '. ' + ' '.join(key_row.astype(str)).title()
                else:
                    opt_title = ' '.join(key_row.astype(str)).title()

                if if_sku_fit_predict:
                    prs = slides.fit_predict(prs=prs,
                                             model=model,
                                             dt_pred=dt_pred,
                                             model_rep=model_rep,
                                             option=option,
                                             option_name=opt_title,
                                             opt_set=opt_set,
                                             period=period,
                                             show_future=True,
                                             group_var=sku_var,
                                             plot_type='sku' + if_retro * '_retro',
                                             bs_key_filter=bs_keys)

                if if_sku_decompose:
                    prs = slides.decomposition(prs=prs,
                                               model=model,
                                               model_rep=model_rep,
                                               option_name=opt_title,
                                               brands=[superbrand],
                                               split_m_m=split_m_m,
                                               period=period,
                                               show_future=True,
                                               group_var=sku_var,
                                               plot_type='sku' + if_retro * '_retro',
                                               if_volume=if_volume,
                                               bs_key_filter=bs_keys)

                if if_sku_waterfall:
                    prs = slides.waterfall(prs=prs,
                                           model=model,
                                           model_rep=model_rep,
                                           date_start=str(opt_set.date_start),
                                           date_end=str(opt_set.date_end),
                                           option_name=opt_title,
                                           brands=[superbrand],
                                           split_m_m=split_m_m,
                                           absolute_sort=False,
                                           plot_type='sku' + if_retro * '_retro',
                                           if_volume=if_volume,
                                           bs_key_filter=bs_keys)

    return prs


def export_curves(prs,
                  model: "Model | list",
                  model_rep: "ModelRep | list",
                  ds: pd.DataFrame | list,
                  opt_set_crv: "OptionSettings",
                  trans_dict: dict | None = None,
                  budget_step: int | float = 5,
                  bdg_max: int | float = 301,
                  fixed_vars: dict | None = {'price': 1},
                  cores: int = 10,
                  if_exact: bool = False,
                  adhoc_curves_max_costs=None
                  ):
    """
    Export efficiency curves (i.e. incremental KPI, profit, ROI) via full curves approach
    (row of options to be calculated). To get stable results run `fermatrica.options.calc.option_report()`
    with some fixed option (as 'zero') and `exact=False` before.

    Right now just a wrapper, maybe to be extended later.

    :param prs: Presentation object from python_pptx package
    :param model: Model object or list of Model objects
    :param model_rep: ModelRep object (export settings) or list of ModelRep objects
    :param ds: dataset or list of datasets
    :param opt_set_crv: OptionSetting object containing calculate settings
    :param trans_dict: translation dictionary (how to "translate" options into independent variables) or None,
        (i.e. defaults to trans_dict attribute of `model_rep` argument)
    :param budget_step: budget iteration step in millions, defaults to 1 (i.e. 1M) (to calculate curves)
    :param bdg_max: maximum budget size (all options with larger budgets to be dropped)
    :param fixed_vars: translation variables with their values to be fixed across grid
    :param cores: number of processor cores to use in multiprocessing calculations.
        None sets to all computer logical cores - 1
    :param if_exact: apply only to the specific time period, without next years
    :param adhoc_curves_max_costs: adhoc function to set maximum observed values for every variable (optional)
    :return: Presentation object from python_pptx package
    """

    prs = slides.incr_roi(prs=prs,
                          model=model,
                          model_rep=model_rep,
                          ds=ds,
                          opt_set_crv=opt_set_crv,
                          translation=trans_dict,
                          budget_step=budget_step,
                          bdg_max=bdg_max,
                          fixed_vars=fixed_vars,
                          adhoc_curves_max_costs=adhoc_curves_max_costs,
                          if_exact=if_exact,
                          cores=cores)

    return prs


def export_adstocks(prs,
                    model: "Model | list",
                    ds: pd.DataFrame | list,
                    model_rep,  #: "ModelRep"
                    cln_meas: list | tuple,
                    option_dict: dict | list = None,
                    option_settings: "OptionSettings" = None,
                    cln_dim: list | tuple = ('superbrand', 'master', 'bs_key', 'date', 'listed', 'kpi_coef'),
                    n: int = 50
                    ):
    """
    Export adstock (decay, carry-on) curves.

    Right now just a wrapper, maybe to be extended later.

    :param prs: Presentation object from python_pptx package
    :param model: Model object or list of Model objects
    :param ds: dataset or list of datasets
    :param model_rep: ModelRep object (export settings) of list of ModelRep objects
    :param option_dict: budget option / scenario to calculate as dictionary or list of option dictionaries
    :param option_settings: OptionSettings object (option setting: target period etc.) or list of OptionSettings
    :param cln_meas: column names to be used as measurements
    :param cln_dim: column names to be used as dimensions
    :param n: number of observations per column
    :return: Presentation object from python_pptx package
    """

    prs = slides.adstocks(prs=prs,
                          model=model,
                          model_rep=model_rep,
                          ds=ds,
                          option_dict=option_dict,
                          option_settings=option_settings,
                          cln_dim=cln_dim,
                          cln_meas=cln_meas,
                          n=n
                          )

    return prs


"""
Preparation and settings
"""


def set_config(model_rep: "ModelRep",
               template_name: str = "blanc") -> tuple:
    """
    Sets export configuration to ModelRep reporting setting object and loads empty PPTX presentation.

    :param model_rep: ModelRep object (reporting settings)
    :param template_name: name of the predefined template or path to the custom template
    :return: update ModelRep model_rep object and Presentation object
    """

    # preparing environment
    model_rep.pptx_cnf = {}

    try:
        rsrc = pkgutil.get_data(f"fermatrica_rep", f"/res/templates/{template_name}.pptx")
        pptxio = io.BytesIO(rsrc)
        template_path = os.path.join(os.path.dirname(sys.modules['fermatrica_rep'].__file__),
                                     f"res\\templates\\{template_name}.pptx")
    except:
        pptxio = template_name

    rsrc = pkgutil.get_data("fermatrica_rep", f"/res/ppt_objects/mock_text_{model_rep.language}.txt")
    txtio = io.BytesIO(rsrc)

    with txtio as f:
        model_rep.pptx_cnf['mock_text'] = f.read()

    dir_path = tempfile.mkdtemp()
    shutil.unpack_archive(template_path, dir_path, format="zip")
    theme_xml = etree.parse(dir_path + r"/ppt/theme/theme1.xml")

    a_nmsp = {'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'}
    p_nmsp = {'p': r'http://schemas.openxmlformats.org/presentationml/2006/main'}

    tmpl = Presentation(pptx=pptxio)

    # set sizes, fonts, colors
    model_rep.pptx_cnf['slide_height'] = tmpl.slide_height
    model_rep.pptx_cnf['slide_width'] = tmpl.slide_width

    model_rep = _set_config_fonts(model_rep, theme_xml, a_nmsp)
    model_rep = _set_config_colors(model_rep, theme_xml, a_nmsp)

    # mode

    tmpl_layouts = []
    for master in tmpl.slide_masters:
        for layout in master.slide_layouts:
            tmpl_layouts.append(layout.name)
            if layout.name == 'Blank_slide':
                model_rep.pptx_cnf['Blank_slide'] = layout

    return model_rep, tmpl


def _set_config_fonts(model_rep: "ModelRep"
                      , theme_xml
                      , a_nmsp: dict):
    """
    Extract fonts from presentation template XML and set to ModelRep reporting settings.

    :param model_rep: ModelRep (reporting settings)
    :param theme_xml: ElementTree XML object form `libxml` package containing theme info
    :param a_nmsp: namespace dictionary (required by xpath)
    :return: ModelRep reporting settings object
    """

    model_rep.pptx_cnf["font_family_header"] = theme_xml.xpath('//a:majorFont/a:latin/@typeface', namespaces=a_nmsp)[0]
    model_rep.pptx_cnf["font_family_body"] = theme_xml.xpath('//a:minorFont/a:latin/@typeface', namespaces=a_nmsp)[0]

    model_rep.pptx_cnf["font_size_main"] = round(model_rep.pptx_cnf["slide_height"] / 5.5e+5)
    model_rep.pptx_cnf["font_size_small"] = Pt(round(model_rep.pptx_cnf["font_size_main"] / 1.2))
    model_rep.pptx_cnf["font_size_footnote"] = Pt(round(model_rep.pptx_cnf["font_size_main"] / 1.5))
    model_rep.pptx_cnf["font_size_large"] = Pt(round(model_rep.pptx_cnf["font_size_main"] * 1.2))
    model_rep.pptx_cnf["font_size_gross"] = Pt(round(model_rep.pptx_cnf["font_size_main"] * 1.5))
    model_rep.pptx_cnf["font_size_main"] = Pt(model_rep.pptx_cnf["font_size_main"])

    return model_rep


def _set_config_colors(model_rep: "ModelRep"
                       , theme_xml
                       , a_nmsp):
    """
    Extract colors from presentation template XML and set to ModelRep reporting settings.

    :param model_rep: ModelRep (reporting settings)
    :param theme_xml: ElementTree XML object form `libxml` package containing theme info
    :param a_nmsp: namespace dictionary (required by xpath)
    :return: ModelRep reporting settings object
    """

    model_rep.pptx_cnf["dk1"] = theme_xml.xpath("//a:dk1/a:srgbClr/@val|//a:dk1/a:sysClr/@lastClr", namespaces=a_nmsp)[0]
    model_rep.pptx_cnf["dk2"] = theme_xml.xpath("//a:dk2/a:srgbClr/@val|//a:dk2/a:sysClr/@lastClr", namespaces=a_nmsp)[0]
    model_rep.pptx_cnf["lt1"] = theme_xml.xpath("//a:lt1/a:srgbClr/@val|//a:lt1/a:sysClr/@lastClr", namespaces=a_nmsp)[0]
    model_rep.pptx_cnf["lt2"] = theme_xml.xpath("//a:lt2/a:srgbClr/@val|//a:lt2/a:sysClr/@lastClr", namespaces=a_nmsp)[0]

    model_rep.pptx_cnf["accent1"] = \
        theme_xml.xpath("//a:accent1/a:srgbClr/@val|//a:accent1/a:sysClr/@lastClr", namespaces=a_nmsp)[0]
    model_rep.pptx_cnf["accent2"] = \
        theme_xml.xpath("//a:accent2/a:srgbClr/@val|//a:accent2/a:sysClr/@lastClr", namespaces=a_nmsp)[0]
    model_rep.pptx_cnf["accent3"] = \
        theme_xml.xpath("//a:accent3/a:srgbClr/@val|//a:accent3/a:sysClr/@lastClr", namespaces=a_nmsp)[0]
    model_rep.pptx_cnf["accent4"] = \
        theme_xml.xpath("//a:accent4/a:srgbClr/@val|//a:accent4/a:sysClr/@lastClr", namespaces=a_nmsp)[0]
    model_rep.pptx_cnf["accent5"] = \
        theme_xml.xpath("//a:accent5/a:srgbClr/@val|//a:accent5/a:sysClr/@lastClr", namespaces=a_nmsp)[0]
    model_rep.pptx_cnf["accent6"] = \
        theme_xml.xpath("//a:accent6/a:srgbClr/@val|//a:accent6/a:sysClr/@lastClr", namespaces=a_nmsp)[0]

    model_rep.pptx_cnf["hlink"] = \
        theme_xml.xpath("//a:hlink/a:srgbClr/@val|//a:hlink/a:sysClr/@lastClr", namespaces=a_nmsp)[0]
    model_rep.pptx_cnf["folHlink"] = \
        theme_xml.xpath("//a:folHlink/a:srgbClr/@val|//a:folHlink/a:sysClr/@lastClr", namespaces=a_nmsp)[0]

    return model_rep
