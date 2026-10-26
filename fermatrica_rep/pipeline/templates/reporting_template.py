"""
Canonical reporting script template.

Used by ``fermatrica_rep.pipeline.generator`` (``generate_pipeline``) to
produce per-node ``reporting_<node>.py`` scripts. Placeholders written as
double-braced tokens are substituted with ``str.replace`` per generated
script: ADHOC_IMPORTS (adhoc module imports, reporting only when edges
exist), OPTIONS_PATH (path to the generated options XLSX), NODE_LOAD_BLOCK
(per-node latest model_* auto-detection plus ``ModelNode(model_path=...,
adhoc_code=..., language='english')``), NODES_DICT (nodes dict or
``node`` for single-node), EDGES_LIST (merge-function edges), and
LAST_NODE (the script's sink node key in the MetaModel; ``"model"`` for
single-node scripts).

Every placeholder appears inside a string literal so the template itself
is syntactically valid Python before substitution. Per-region loops
(p08-specific) are intentionally not generated — left as TODO comments.
"""

import os
import re
import sys
import time
import warnings
from datetime import datetime

import fermatrica_rep as fmr
import fermatrica_rep.export.export_pptx as export_pptx
import pandas as pd
import plotly.io as pio

"{{ ADHOC_IMPORTS }}"

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..'))
os.chdir(project_root)
sys.path.insert(0, project_root)

warnings.simplefilter(action='ignore', category=FutureWarning)

pio.renderers.default = "browser"

if __name__ == '__main__':

    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 2000)

    os.makedirs("output/data", exist_ok=True)

    # script settings

    if_visual = False
    if_curves = False
    if_bdg_optim_budget = False
    if_bdg_optim_budget_multi = False
    if_bdg_optim_target = False
    if_bdg_optim_target_multi = False
    if_bdg_optim_nodes = False
    if_elasticity = False
    if_export = True
    if_summary = False
    if_save = False
    if_load_meta = False
    if_debug = True

    # -------------------- load options ------------------------ #

    options_path = os.path.join('{{ OPTIONS_PATH }}')
    ctx = fmr.load_context(options_path)

    # ------------------ load nodes ---------------------- #

    "{{ NODE_LOAD_BLOCK }}"

    # ------------------ create MetaModel DAG -------------------- #

    mm = fmr.MetaModel(nodes="{{ NODES_DICT }}", edges="{{ EDGES_LIST }}", compute_context=ctx)

    # ------------------- diagnostics --------------------- #

    d = fmr.MetaModelDiagnostics(mm)
    report = d.run(options=ctx.options, scopes=ctx.scopes, raise_on_first=False)
    if not report.passed:
        print(report.summary)
    else:
        print("Diagnostics: all checks passed")
        if report.warnings:
            print(report.warnings_summary)

    # ------------------- run options ---------------------- #

    opt_sum = mm.compute(ctx.options['predefined'], allow_past=True, scopes=ctx.scopes)

    # --------------------- output ----------------------- #

    last_node = '{{ LAST_NODE }}'
    plan_year = max(s.start_year for s in ctx.scopes)

    if if_visual:

        print(fmr.metrics_group_table(mm, group_vars=['superbrand']))

        for nname in mm.nodes:
            fig = fmr.fit_main_plot_vol(mm.nodes[nname], period='month', show_future=True)
            fig.show()

        split_m_m = fmr.extract_effect(mm.nodes[last_node])
        split_m_m = split_m_m.sort_values(['superbrand', 'bs_key', 'date'])

        fig = fmr.decompose_main_plot(split_m_m=split_m_m, model_rep=mm.nodes[last_node]
                                      , period='m', show_future=True, contour_line=True)
        fig.show()

        fig = fmr.waterfall_plot(split_m_m=split_m_m, model_rep=mm.nodes[last_node]
                                 , date_start=str(plan_year - 1) + '-01-01'
                                 , date_end=str(plan_year - 1) + '-12-31')
        fig.show()

    if if_curves:

        # TODO: per-region loops (e.g. p08) are intentionally not generated.
        #       Adapt `scopes_crv` and `background_option` to your project.
        scopes_crv = fmr.filter_scopes(ctx.scopes, start_year_min=plan_year)

        print(datetime.now())
        curves_full_data = fmr.compute_grid(
            mm,
            scopes_crv,
            budget_step=5,  # TODO: budget iteration step (millions)
            bdg_max=300,    # TODO: maximum budget (millions)
            cores=14,       # TODO: number of CPU cores
            if_exact=True,
            allow_past=True,
            background_option=None,  # TODO: pass a base Option to hold other scopes' budgets fixed (see p08)
        )
        print(datetime.now())

        curves_full_figs = fmr.curves_full_plot_short(curves_full_data[last_node], model_rep=mm.nodes[last_node])
        for fig in curves_full_figs.values():
            fig.show()
            break

        time.sleep(5)

        curves_full_figs = fmr.curves_full_plot_long(curves_full_data[last_node], model_rep=mm.nodes[last_node])
        for fig in curves_full_figs.values():
            fig.show()
            break

    if if_bdg_optim_budget:

        # TODO: set borders_dict for each budget item
        borders_dict = {
            'bdg_tv': [0, 300, 100],
        }
        bdg_size = 300

        opt_option_budget = fmr.optimize_budget_local_cobyla(
            meta_model=mm,
            scopes=ctx.scopes,
            budget_scope=ctx.scopes[0].scope_name,  # TODO: pick the budget scope
            borders_dict=borders_dict,
            option=ctx.options['zero'],
            epochs=1,
            iters_epoch=300,
            error_score=1e12,
            bdg_size=bdg_size,
            ftol_abs=.1,
            xtol_abs=None,
            if_exact=False,
            if_volume=True,
            warn_nan_seasonality=False
        )

        print(opt_option_budget)

        budget_items = opt_option_budget.budget_for_scope(ctx.scopes[0].scope_name)
        pd.DataFrame({'value': budget_items}) \
            .to_excel("output/data/bdg_optim_budget_" + str(bdg_size) + "_" + pd.Timestamp('today').strftime('%Y%m%d') + ".xlsx",
                      index=True, freeze_panes=(1, 0))

    if if_bdg_optim_budget_multi:

        # TODO: set borders_dict and bdg_sizes
        borders_dict = {
            'bdg_tv': [0, 300, 100],
        }
        bdg_sizes = [100, 200, 300]

        opt_options_budget_multi = fmr.optimize_budget_local_cobyla_multi(
            meta_model=mm,
            scopes=ctx.scopes,
            budget_scope=ctx.scopes[0].scope_name,  # TODO: pick the budget scope
            borders_dict=borders_dict,
            option=ctx.options['zero'],
            bdg_sizes=bdg_sizes,
            epochs=1,
            iters_epoch=300,
            error_score=1e12,
            ftol_abs=.1,
            xtol_abs=None,
            if_exact=False,
            if_volume=True,
            warn_nan_seasonality=False,
            cores=4,
        )

        print(opt_options_budget_multi)

        results = []
        for bdg_size, opt_option in opt_options_budget_multi.items():
            budget_items = opt_option.budget_for_scope(ctx.scopes[0].scope_name)
            budget_items['bdg_size'] = bdg_size
            results.append(budget_items)

        pd.DataFrame(results).transpose() \
            .to_excel("output/data/bdg_optim_budget_multi_" + pd.Timestamp('today').strftime('%Y%m%d') + ".xlsx",
                      index=True, freeze_panes=(1, 0))

    if if_bdg_optim_target:

        # TODO: set borders_dict and trg
        borders_dict = {
            'bdg_tv': [0, 300, 100],
        }
        trg = 1000000

        opt_option_kpi = fmr.optimize_target_local_cobyla(
            meta_model=mm,
            scopes=ctx.scopes,
            budget_scope=ctx.scopes[0].scope_name,  # TODO: pick the target scope
            borders_dict=borders_dict,
            option=ctx.options['zero'],
            epochs=1,
            iters_epoch=300,
            error_score=1e12,
            trg=trg,
            trg_tol_ratio=0.001,
            ftol_abs=.1,
            xtol_abs=None,
            if_exact=True,
            if_volume=True,
            warn_nan_seasonality=False
        )

        opt_sum = mm.compute(opt_option_kpi, scopes=ctx.scopes)
        print(opt_sum)

        budget_items = opt_option_kpi.budget_for_scope(ctx.scopes[0].scope_name)
        pd.DataFrame({'value': budget_items}) \
            .to_excel("output/data/bdg_optim_target_" + pd.Timestamp('today').strftime('%Y%m%d') + ".xlsx",
                      index=True, freeze_panes=(1, 0))

    if if_bdg_optim_target_multi:

        # TODO: set borders_dict and trg_list
        borders_dict = {
            'bdg_tv': [0, 300, 100],
        }
        trg_list = [500000, 1000000, 1500000]

        opt_options_kpi_multi = fmr.optimize_target_local_cobyla_multi(
            meta_model=mm,
            scopes=ctx.scopes,
            budget_scope=ctx.scopes[0].scope_name,  # TODO: pick the target scope
            borders_dict=borders_dict,
            option=ctx.options['zero'],
            trg_list=trg_list,
            epochs=1,
            iters_epoch=300,
            error_score=1e12,
            ftol_abs=.1,
            xtol_abs=None,
            if_exact=True,
            if_volume=True,
            warn_nan_seasonality=False,
            cores=4,
        )

        print(opt_options_kpi_multi)

        results = []
        for trg, opt_option in opt_options_kpi_multi.items():
            opt_sum = mm.compute(opt_option, scopes=ctx.scopes)
            budget_items = opt_option.budget_for_scope(ctx.scopes[0].scope_name)
            budget_items['trg'] = trg
            results.append(budget_items)

        pd.DataFrame(results).transpose() \
            .to_excel("output/data/bdg_optim_target_multi_" + pd.Timestamp('today').strftime('%Y%m%d') + ".xlsx",
                      index=True, freeze_panes=(1, 0))

    if if_bdg_optim_nodes:

        # TODO: multi-node optimization against a weighted combination of nodes.
        #       Set summary_nodes = {'<node>': (weight, scope_name), ...}
        borders_dict = {
            'bdg_tv': [0, 300, 100],
        }

        opt_option_nodes = fmr.optimize_budget_local_cobyla(
            meta_model=mm,
            scopes=ctx.scopes,
            budget_scope=ctx.scopes[0].scope_name,  # TODO: pick the budget scope
            borders_dict=borders_dict,
            option=ctx.options['zero'],
            epochs=1,
            iters_epoch=300,
            error_score=1e12,
            bdg_size=300,
            ftol_abs=.1,
            xtol_abs=None,
            if_exact=False,
            if_volume=True,
            warn_nan_seasonality=False,
            summary_nodes={'{{ LAST_NODE }}': (1.0, ctx.scopes[0].scope_name)},  # TODO: weights per node
        )
        print(opt_option_nodes)

    if if_elasticity:

        # TODO: set the elasticity scope and the price translation item
        scope = ctx.scopes[0]

        mm.compute(ctx.options['predefined'], scopes=[scope], if_exact=True)

        price_df_total = fmr.analysis.elasticity_price.elasticity_data(
            model=mm.nodes['{{ LAST_NODE }}'],
            scope=scope,
            option=ctx.options['predefined'],
            step=.01,
            price_range=(-0.25, 0.26),
            price_name_opt="nobdg_price",  # TODO: price translation item
            var="exact",
            if_exact=True,
        )

        price_df_total.to_excel("output/data/price_total_" + pd.Timestamp('today').strftime('%Y%m%d') + ".xlsx")

    if if_export:

        mm.nodes[last_node].model_rep, prs = export_pptx.set_config(mm.nodes[last_node].model_rep, template_name='blanc')

        scopes_retro = [s.replace_dates(date_start=str(plan_year - 1) + '-01-01', date_end=str(plan_year - 1) + '-12-31')
                        for s in fmr.filter_scopes(ctx.scopes, start_year=plan_year)]

        scopes_plan = fmr.filter_scopes(ctx.scopes, start_year_min=plan_year)

        # retro

        prs = export_pptx.options_detail(prs
                                         , meta_model=mm
                                         , scopes=scopes_retro
                                         , options=None
                                         , sku_var=['superbrand']  # TODO: SKU grouping columns
                                         , period='month'
                                         , listed_filter=(2, 3)
                                         , if_volume=True
                                         , if_sku_fit_predict=False
                                         , if_sku_decompose=False
                                         , if_sku_waterfall=False
                                         , summary_scopes=None  # TODO: per-region/subscope summaries
                                         )

        # options table

        prs = export_pptx.options_table(prs
                                        , meta_model=mm
                                        , scopes=scopes_plan
                                        , options=ctx.options
                                        , if_exact=True
                                        )

        # options

        prs = export_pptx.options_detail(prs
                                         , meta_model=mm
                                         , scopes=scopes_plan
                                         , options=ctx.options
                                         , sku_var=['superbrand']  # TODO: SKU grouping columns
                                         , period='month'
                                         , if_exact=True
                                         , if_volume=True
                                         , if_sku_fit_predict=False
                                         , if_sku_decompose=False
                                         , if_sku_waterfall=False
                                         , summary_scopes=None  # TODO: per-region/subscope summaries
                                         )

        # adstock

        cln_dim = ['listed', 'date', 'superbrand', 'bs_key', 'master', 'kpi_coef', 'region']
        cln_meas = [re.sub(r'_adstockp*d*$', '', x) for x in mm.nodes['{{ LAST_NODE }}'].ds.columns
                    if re.search(r'_adstockp*d*$', x)
                      and sum(mm.nodes['{{ LAST_NODE }}'].model.conf.model_rhs['token'].str.contains(x))]

        prs = export_pptx.adstocks(prs
                                    , meta_model=mm
                                    , cln_meas=cln_meas
                                    , cln_dim=cln_dim
                                    , n=52
                                    )

        prs.save(
            "output/_Econometrics_" + mm.nodes[last_node].model.conf.target_superbrand.capitalize() + "_"
            + pd.Timestamp('today').strftime('%Y%m%d') + ".pptx")

    if if_summary:

        # per-brand XLSX summaries (model setup, curves, decomposition, fit, adstocks)
        written = fmr.export_summaries(
            mm, ctx,
            output_dir='output/summary',
            budget_step=5, bdg_max=300, cores=14,  # TODO: budget step / max / CPU cores
            if_exact=True, allow_past=True,
        )
        print(written)

    if if_save:

        mm.save(path=os.path.join(os.getcwd(), 'code_py', 'model_data')
                , save_tag=model_tag_last + '_dash'
                , include_context_path=ctx.path)

    if if_load_meta:

        meta_dir = os.path.join(os.getcwd(), 'code_py', 'model_data'
                                , 'meta_' + model_tag_last + '_dash')
        if not os.path.isdir(meta_dir):
            meta_dir = os.path.join(os.getcwd(), 'code_py', 'model_data'
                                    , 'model_' + model_tag_last + '_dash')
        meta_model = fmr.MetaModel.load(meta_dir)
        opt_sum_loaded = meta_model.compute(ctx.options['predefined'], allow_past=True, scopes=ctx.scopes)
        print(opt_sum_loaded)

    if if_debug:
        breakpoint()
        1
