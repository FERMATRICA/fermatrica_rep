"""
Generate slide(s) with comparative table containing summaries by number of options,
defined in `options_m` dictionary.
"""


import warnings

import pandas as pd
import numpy as np

from pptx.presentation import Presentation
from pptx.dml.color import RGBColor

from fermatrica_rep.basics import _filter_df, _auto_brand_bs_keys
from fermatrica_rep.export import wording
from fermatrica_rep.export.basics import _resolve_slide_layout, set_table_border, adjust_table_width, table_text_format
from fermatrica_rep.meta_model.node import ModelNode
from fermatrica_rep.meta_model.meta_model import MetaModel, _last_node_name, _convert_to_metamodel
from fermatrica_rep.options.scope import OptionScope
from fermatrica_rep.options.option import Option
from fermatrica_rep.options.parallel import compute_multi
from fermatrica_rep.options.define import _enrich_trans_dict_from_model, _convert_old_options


warnings.simplefilter(action='ignore', category=UserWarning)


def _normalize_options(options, scopes):
    """Normalize dict or list-of-Option to dict[str, Option]."""
    if isinstance(options, list):
        return {opt.name: opt for opt in options}
    return {
        name: data if isinstance(data, Option) else Option(name, budget_data=data)
        for name, data in options.items()
    }


def create(prs: Presentation,
           meta_model: MetaModel,
           scopes: list,
           options: dict | list,
           *,
           brands: list | None = None,
           brands_var: str = 'superbrand',
           filters: dict | None = None,
           if_exact: bool = True,
           summary_node: str | None = None,
           cores: int = 1
           ):
    """
    Generate slide(s) with comparative table containing summaries by number of options.

    :param prs: Presentation object from python_pptx package
    :param meta_model: MetaModel DAG
    :param scopes: list of OptionScope objects
    :param options: dict of option name to budget data ({scope_name: {item: value}}), or list of Option objects
    :param brands: simple filter values, applied to `brands_var`. When `brands`, `brands_var`
        and `filters` are all None, defaults to ``{'bs_key': target-brand bs_keys}``
    :param brands_var: column the simple filter values apply to
    :param filters: complex filter ``{column: [values]}``
    :param if_exact: apply only to the specific time period, without next years
    :param summary_node: name of the DAG node providing KPI metrics (default: last sink node)
    :param cores: number of processor cores for compute_multi (default 1 for safety)
    :return: Presentation object from python_pptx package
    """
    import logging
    options = _normalize_options(options, scopes)
    if 'zero' not in options:
        options['zero'] = Option('zero', budget_data={})

    summary_node = summary_node or _last_node_name(meta_model)
    model_rep_main = meta_model.nodes[summary_node].model_rep
    dt_pred = meta_model.nodes[summary_node].dt_pred
    model = meta_model.nodes[summary_node].model

    language = model_rep_main.language
    slide_width = model_rep_main.pptx_cnf['slide_width']
    slide_height = model_rep_main.pptx_cnf['slide_height']
    slide_layout = _resolve_slide_layout(prs, model_rep_main.pptx_cnf['Blank_slide'])

    price_var = model.conf.price_var
    model_conf = model.conf

    summary_type = getattr(model.conf, "summary_type", "sum")
    if summary_type in ("mean", "mean_fin"):
        agg_fn = "mean"
    else:
        agg_fn = "sum"

    if brands is None and filters is None:
        node_cur = meta_model.nodes[summary_node]
        model_cur = node_cur.model
        src = node_cur.dt_pred if node_cur.dt_pred is not None else node_cur.ds
        if src is not None and model_cur is not None and 'superbrand' in src.columns:
            filters = {'bs_key': _auto_brand_bs_keys(src, model_cur)}

    dt_pred_prep = _filter_df(dt_pred, brands=brands, brands_var=brands_var, filters=filters). \
        assign(year=dt_pred['date'].dt.year,
               observed_val=dt_pred['observed'] * dt_pred[price_var],
               predicted_val=dt_pred['predicted'] * dt_pred[price_var]). \
        loc[dt_pred['listed'].isin([2, 3, 4])]. \
        groupby('year')[['observed', 'predicted', 'observed_val', 'predicted_val']].agg(agg_fn).reset_index()

    pred_year_end = dt_pred_prep['year'].max()

    all_option_names = [k for k in options if k != 'zero']
    all_option_names = ['zero'] + all_option_names
    all_options_for_compute = [options[name] for name in all_option_names]
    results = compute_multi(meta_model, all_options_for_compute, scopes, cores=cores, if_exact=if_exact)
    name_to_idx = {name: idx for idx, name in enumerate(all_option_names)}

    relevant_scopes = _get_nonzero_scopes(results, scopes, summary_node, name_to_idx)

    options_names = [k for k in options if k != 'zero']
    options_names = [options_names[n:n + 3] for n in range(0, len(options_names), 3)]

    if language == 'english':
        opt_str = 'Option'
    else:
        opt_str = 'Опция'

    for scope, _scope_name, zero_available in relevant_scopes:
        pred_year_start = scope.date_start.year

        for options_names_subset in options_names:
            opt_table_data, calc_table_data = _tables_data_worker(
                meta_model=meta_model,
                scopes=scopes,
                options=options,
                options_names_subset=options_names_subset,
                pred_year_start=pred_year_start,
                pred_year_end=pred_year_end,
                target_scope_name=_scope_name,
                if_exact=if_exact,
                hide_rows=['text_flags', 'google', 'price'],
                summary_node=summary_node,
                cores=cores,
                precomputed_results=results,
                name_to_idx=name_to_idx,
                zero_available=zero_available,
                model_conf=model_conf,
            )

            if opt_table_data.empty or calc_table_data.empty:
                logging.warning(
                    f"options_table: scope '{_scope_name}' skipped — empty tables"
                )
                continue

            slide = prs.slides.add_slide(slide_layout)

            slide_title = model_rep_main.vis_dict.loc[(model_rep_main.vis_dict['section'] == 'option_compare') & (
                    model_rep_main.vis_dict['variable'] == 'title'), language].iloc[0]
            slide.shapes[0].text_frame.text = slide_title
            slide.shapes[0].text_frame.paragraphs[0].font.size = model_rep_main.pptx_cnf['font_size_gross']
            slide.shapes[0].text_frame.paragraphs[0].font.name = model_rep_main.pptx_cnf['font_family_header']

            y_label_base = slide.shapes[0].top + slide.shapes[0].height + 0.02 * slide_height

            scope_box = slide.shapes.add_textbox(
                slide.shapes[0].left,
                int(round(y_label_base)),
                slide.shapes[0].width,
                int(round(0.04 * slide_height)))
            scope_box.text_frame.paragraphs[0].text = f"Scope: {_scope_name}"
            scope_box.text_frame.paragraphs[0].font.size = model_rep_main.pptx_cnf["font_size_main"]
            scope_box.text_frame.paragraphs[0].font.name = model_rep_main.pptx_cnf["font_family_body"]

            y_label_base = scope_box.top + scope_box.height + 0.03 * slide_height

            label_cy = int(round(0.05 * slide_height))

            x = (0.05 + 0.9 * 2 / 10) * slide_width
            cx = 0.9 * 2 / 10 * slide_width
            x = int(round(x))
            y_label_x = int(round(y_label_base))

            textbox = slide.shapes.add_textbox(x, y_label_x, cx, label_cy)
            paragraph = textbox.text_frame.paragraphs[0]
            paragraph.text = f"{opt_str} 1. zero"
            paragraph.font.bold = True
            paragraph.font.size = model_rep_main.pptx_cnf["font_size_main"]
            paragraph.font.name = model_rep_main.pptx_cnf["font_family_body"]
            textbox.text_frame.word_wrap = True

            for n, option_name in enumerate(options_names_subset):
                x = (0.05 + 0.9 * 2 * (n + 2) / 10) * slide_width
                y = y_label_x
                cx = 0.9 * 2 / 10 * slide_width
                cy = label_cy
                x, y, cx, cy = int(round(x)), int(round(y)), int(round(cx)), int(round(cy))

                textbox = slide.shapes.add_textbox(x, y, cx, cy)
                paragraph = textbox.text_frame.paragraphs[0]
                idx = name_to_idx[option_name]
                paragraph.text = f"{opt_str} {idx}. {option_name}"
                paragraph.font.bold = True
                paragraph.font.size = model_rep_main.pptx_cnf["font_size_main"]
                paragraph.font.name = model_rep_main.pptx_cnf["font_family_body"]
                textbox.text_frame.word_wrap = True

            n_rows = len(opt_table_data.iloc[:, 0]) + 1
            n_cols = 3 + 2 * len(options_names_subset)

            cell_width = round(1 / 10 * slide_width)
            cell_height = round(1 / 28 * slide_height)

            x, y = (0.5 - 0.45) * slide_width, y_label_x + label_cy
            cx, cy = n_cols * cell_width, n_rows * cell_height
            x, y, cx, cy = int(round(x)), int(round(y)), int(round(cx)), int(round(cy))

            opt_shape = slide.shapes.add_table(n_rows, n_cols, x, y, cx, cy)
            opt_table = opt_shape.table

            set_table_border(opt_table, border_color='595959', border_width='25000')

            opt_table = _fill_table(opt_table, ds=opt_table_data,
                                    pred_year_start=pred_year_start, pred_year_end=pred_year_end,
                                    last_row_bold=True)

            opt_table = table_text_format(opt_table, model_rep_main)
            opt_table = adjust_table_width(table=opt_table, slide_width=slide_width)

            n_rows = len(calc_table_data) + 1
            n_cols = 3 + 2 * len(options_names_subset)

            x, y = (0.5 - 0.45) * slide_width, opt_shape.top + opt_shape.height + 0.05 * slide_height
            cx, cy = n_cols * cell_width, n_rows * cell_height
            x, y, cx, cy = int(round(x)), int(round(y)), int(round(cx)), int(round(cy))

            calc_shape = slide.shapes.add_table(n_rows, n_cols, x, y, cx, cy)
            calc_table = calc_shape.table

            set_table_border(calc_table, border_color='595959', border_width='25000')

            calc_table = _fill_table(calc_table, ds=calc_table_data,
                                     pred_year_start=pred_year_start, pred_year_end=pred_year_end)

            calc_table = table_text_format(calc_table, model_rep_main)
            calc_table = adjust_table_width(table=calc_table, slide_width=slide_width)

    return prs


def create_legacy(prs: Presentation,
                  model = None,
                  model_rep = None,
                  ds = None,
                  options_m = None,
                  opt_set = None,
                  dt_pred = None,
                  *,
                  bs_key_filter: list | tuple | None = None,
                  if_exact: bool = True,
                  summary_node: str | None = None,
                  cores: int = 1,
                  ):
    import warnings
    warnings.warn(
        "Passing Model, ModelRep, ds, options_m, opt_set to slides.options is deprecated. "
        "Use MetaModel, OptionScope, and Option instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    meta_model = _convert_to_metamodel(
        model, ds=ds, model_rep=model_rep, dt_pred=dt_pred,
        merge_fn_pattern="code_py.adhoc.reporting.pred_merge_{i}",
    )
    scopes, options = _convert_old_options(opt_set, options_m)
    for node in meta_model.nodes.values():
        if node.model is not None:
            _enrich_trans_dict_from_model(node.model.conf, node.model_rep.trans_dict)
    return create(prs, meta_model, scopes, options,
                  filters={'bs_key': list(bs_key_filter)} if bs_key_filter is not None else {},
                  if_exact=if_exact,
                  summary_node=summary_node, cores=cores)


def _get_nonzero_scopes(results, scopes, summary_node, name_to_idx):
    """Return [(OptionScope, scope_name, zero_available), ...] for relevant scopes.

    A scope is relevant when at least one option (the zero baseline included —
    per-option scopes may be bound only to the zero option, e.g. p09's
    ``zero_brand_1_*``) has a non-zero result for it (a summary dict with
    ``pred_exact_vol`` > 0). ``zero_available`` is True when the zero option's
    result also covers the scope; scopes the zero option is not bound to are
    still reported, with the zero-dependent comparison cells rendered as ``'-'``.

    :param results: list of ``meta_model.compute()`` outputs, one per option
        (indexed by ``name_to_idx``); ``None`` entries mean a failed option
    :param scopes: list of OptionScope objects
    :param summary_node: name of the DAG node providing KPI metrics
    :param name_to_idx: mapping from option name to index into ``results``
    """
    zero_idx = name_to_idx.get('zero')
    nonzero = []
    for scope in scopes:
        sn = scope.scope_name
        relevant = False
        for name, idx in name_to_idx.items():
            result = results[idx]
            if result is None:
                continue
            sr = result.get(summary_node, {}).get(sn)
            if isinstance(sr, tuple):
                sr = sr[2]
            if isinstance(sr, dict):
                vol = sr.get("pred_exact_vol")
                if vol and vol > 0:
                    relevant = True
                    break
        if not relevant:
            continue
        zero_available = False
        if zero_idx is not None and results[zero_idx] is not None:
            sr = results[zero_idx].get(summary_node, {}).get(sn)
            if isinstance(sr, tuple):
                sr = sr[2]
            zero_available = isinstance(sr, dict)
        nonzero.append((scope, sn, zero_available))
    if not nonzero:
        zero_available = False
        if zero_idx is not None and results[zero_idx] is not None:
            sr = results[zero_idx].get(summary_node, {}).get(scopes[0].scope_name)
            if isinstance(sr, tuple):
                sr = sr[2]
            zero_available = isinstance(sr, dict)
        nonzero.append((scopes[0], scopes[0].scope_name, zero_available))
    return nonzero


def _tables_data_worker(meta_model, scopes, options,
                        options_names_subset, pred_year_start, pred_year_end,
                        target_scope_name,
                        if_exact=True, hide_rows=('text_flags',), summary_node=None, cores=1,
                        precomputed_results=None, name_to_idx=None, zero_available=True,
                        model_conf=None):
    """
    Calculate subset of options and fill the table with option data and summaries for one slide only.

    :param target_scope_name: scope name to extract budget/results for (required)
    :param precomputed_results: precomputed results from a prior compute_multi() call (optional)
    :param name_to_idx: mapping from option name to index in precomputed_results (required if precomputed)
    :param zero_available: whether the zero option's result covers ``target_scope_name``
        (default True — classic shared-scope behavior). When False, the zero rows
        and all zero-dependent comparison cells (incremental sales, profit, ROMI,
        long-term growth) are rendered as ``'-'``; the options' absolute KPIs are
        still shown. Missing per-option results render as ``'-'`` too.
    :param model_conf: model configuration carrying KPI/currency facts (default
        None — sales/RUB wording and the full metric set). When
        ``if_value_applicable`` is False, value/profit/ROMI rows are dropped
        from the calc table.
    """

    summary_node = summary_node or _last_node_name(meta_model)
    model_rep_main = meta_model.nodes[summary_node].model_rep
    language = model_rep_main.language

    zero_opt = options.get('zero')
    if zero_opt is None:
        zero_opt = Option('zero', budget_data={})
        options['zero'] = zero_opt
    budgets = {name: opt.budget_data for name, opt in options.items()}
    zero_all = budgets['zero']

    if precomputed_results is not None:
        results = [precomputed_results[name_to_idx['zero']]]
        for name in options_names_subset:
            results.append(precomputed_results[name_to_idx[name]])
    else:
        all_options = [zero_opt] + [options[name] for name in options_names_subset]
        results = compute_multi(meta_model, all_options, scopes, cores=cores, if_exact=if_exact)

    # ---------- options table ----------

    invest_col = wording.text(model_rep_main, model_conf, section='option_compare', variable='costs_title')

    n_years = pred_year_end - pred_year_start + 1 if not if_exact else 1

    summary_type = getattr(meta_model.nodes[summary_node].model.conf, "summary_type", "sum")
    long_growth_div = 1 if summary_type in ("mean", "mean_fin") else n_years

    zero_budget = zero_all.get(target_scope_name, {})
    zero_items = {k: v for k, v in zero_budget.items() if k not in hide_rows}
    zero_total = sum(v for k, v in zero_budget.items() if k.startswith('bdg_'))

    if not zero_items:
        seed_items = set()
        for option_name in options_names_subset:
            for item in budgets[option_name].get(target_scope_name, {}):
                if item not in hide_rows:
                    seed_items.add(item)
        zero_items = {item: 0.0 for item in seed_items}

    zero_rows = []
    for item, val in zero_items.items():
        zero_rows.append({invest_col: item,
                          'zero_short': val,
                          'zero_long': val * n_years})
    zero_rows.append({invest_col: 'bdg',
                      'zero_short': zero_total,
                      'zero_long': zero_total * n_years})

    options_df = pd.DataFrame(zero_rows)

    for option_name in options_names_subset:
        opt_budget = budgets[option_name].get(target_scope_name, {})
        opt_items = {k: v for k, v in opt_budget.items() if k not in hide_rows}
        opt_short_col = f'{option_name}_short'
        opt_long_col = f'{option_name}_long'

        opt_rows = []
        for item, val in opt_items.items():
            opt_rows.append({'tmp': item, opt_short_col: val, opt_long_col: val * n_years})

        tmp_df = pd.DataFrame(opt_rows)
        if tmp_df.empty:
            options_df[opt_short_col] = 0.0
            options_df[opt_long_col] = 0.0
        else:
            options_df = options_df.merge(tmp_df, how='left', left_on=invest_col, right_on='tmp')
            options_df.drop('tmp', axis=1, inplace=True)

    for option_name in options_names_subset:
        opt_budget = budgets[option_name].get(target_scope_name, {})
        opt_total = sum(v for k, v in opt_budget.items() if k.startswith('bdg_'))
        options_df.loc[options_df[invest_col] == 'bdg', f'{option_name}_short'] = opt_total
        options_df.loc[options_df[invest_col] == 'bdg', f'{option_name}_long'] = opt_total * n_years

    num_cols = [c for c in options_df.columns if c.endswith('_short') or c.endswith('_long')]
    options_df[num_cols] = options_df[num_cols].fillna(0.0)

    group_cols = list(options_df)

    options_df['tmp'] = options_df[invest_col].str.replace('bdg_', '')
    options_df = options_df.merge(
        model_rep_main.vis_dict.loc[(model_rep_main.vis_dict['section'] == 'costs_type'), ['variable', language]],
        how='left', left_on='tmp', right_on='variable')

    options_df = options_df.groupby(group_cols).first().reset_index()
    options_df.loc[options_df[language].notna(), invest_col] = options_df.loc[options_df[language].notna(), language]

    options_df.drop(language, axis=1, inplace=True)
    options_df.drop('variable', axis=1, inplace=True)
    options_df.drop('tmp', axis=1, inplace=True)

    short_cols = [c for c in options_df.columns if c.endswith('_short')]
    numeric_mask = options_df[short_cols].map(lambda x: pd.to_numeric(x, errors='coerce')).notnull().all(axis=1)
    options_df = options_df.loc[numeric_mask]
    options_df = options_df.loc[(options_df[short_cols] != '0').any(axis=1)]

    options_df = options_df.loc[options_df[invest_col] != '0']

    if options_df.empty:
        return options_df, pd.DataFrame()

    row_to_move = options_df.iloc[0]
    options_df = options_df.drop(options_df.index[0])
    options_df = pd.concat([options_df, row_to_move.to_frame().T])

    options_df = options_df.reset_index().drop('index', axis=1)

    # ---------- calc table ----------

    sales_col_name = wording.text(model_rep_main, model_conf, section='option_compare', variable='sales_title')

    calc_spec = [
        ('res', 'vol'),
        ('res_rub_distr', 'val'),
        ('res_rub_progress_distr', 'incr'),
        ('profit', 'profit'),
        ('romi', 'romi'),
        ('grw', 'grw'),
    ]
    facts = wording.facts(model_conf)
    if not facts['if_value_applicable']:
        calc_spec = [s for s in calc_spec if s[1] in ('vol', 'grw')]

    labels_by_var = {}
    metric_cols = []
    for var, _kind in calc_spec:
        label = wording.text(model_rep_main, model_conf, section='option_compare_sales', variable=var)
        if var == 'grw':
            label = label.replace('***', str(pred_year_start - 1))
        labels_by_var[var] = label
        metric_cols.append(label)

    calc_df = pd.DataFrame(columns=[sales_col_name] + metric_cols)
    calc_df[sales_col_name] = list(options_df)[1:]

    def fill_row(row_label, values_by_var):
        calc_df.loc[calc_df[sales_col_name] == row_label, metric_cols] = [
            values_by_var.get(var, '-') for var, _kind in calc_spec]

    zero_result = None if results[0] is None else results[0].get(summary_node, {}).get(target_scope_name)

    if not zero_available or zero_result is None:
        fill_row('zero_short', {})
        fill_row('zero_long', {})
        zero_val_short = None
        zero_val_long = None
    else:
        zero_val_short = zero_result.get('pred_exact_val', 0)
        zero_val_long = zero_result.get('pred_long_val', 0)
        fill_row('zero_short', {
            'res': zero_result.get('pred_exact_vol', 0),
            'res_rub_distr': zero_val_short,
            'grw': zero_result.get('growth_vol', 0) - 1,
        })
        fill_row('zero_long', {
            'res': zero_result.get('pred_long_vol', 0),
            'res_rub_distr': zero_val_long,
            'grw': zero_result.get('pred_long_vol', 0) / zero_result.get('ref_vol', 1) / long_growth_div - 1,
        })

    for i, option_name in enumerate(options_names_subset):
        opt_result = None if results[i + 1] is None else results[i + 1].get(summary_node, {}).get(target_scope_name)

        opt_budget = budgets[option_name].get(target_scope_name, {})
        bdg_cur = sum(v for k, v in opt_budget.items() if k.startswith('bdg_'))
        bdg_long = bdg_cur * n_years

        if opt_result is None:
            fill_row(f'{option_name}_short', {})
            fill_row(f'{option_name}_long', {})
            continue

        short_row = {
            'res': opt_result.get('pred_exact_vol', 0),
            'res_rub_distr': opt_result.get('pred_exact_val', 0),
            'grw': opt_result.get('growth_vol', 0) - 1,
        }
        long_row = {
            'res': opt_result.get('pred_long_vol', 0),
            'res_rub_distr': opt_result.get('pred_long_val', 0),
        }

        if not zero_available or zero_val_short is None:
            fill_row(f'{option_name}_short', short_row)
            fill_row(f'{option_name}_long', long_row)
            continue

        incr_sales_short = opt_result.get('pred_exact_val', 0) - zero_val_short
        profit_short = incr_sales_short - 1e6 * bdg_cur
        romi_short = incr_sales_short / (1e6 * bdg_cur) if bdg_cur > 0 else 0
        fill_row(f'{option_name}_short', {
            **short_row,
            'res_rub_progress_distr': incr_sales_short,
            'profit': profit_short,
            'romi': romi_short,
        })

        incr_sales_long = opt_result.get('pred_long_val', 0) - zero_val_long
        profit_long = incr_sales_long - 1e6 * bdg_long
        romi_long = incr_sales_long / (1e6 * bdg_long) if bdg_long > 0 else 0
        fill_row(f'{option_name}_long', {
            **long_row,
            'res_rub_progress_distr': incr_sales_long,
            'profit': profit_long,
            'romi': romi_long,
            'grw': opt_result.get('pred_long_vol', 0) / zero_result.get('ref_vol', 1) / long_growth_div - 1,
        })

    calc_df = calc_df.T
    new_header = calc_df.iloc[0]
    calc_df = calc_df[1:]
    calc_df.columns = new_header

    calc_df = calc_df.reset_index().rename(columns={'index': sales_col_name})

    calc_df[calc_df[sales_col_name] == labels_by_var['grw']] = [f"{x:.2%}" if isinstance(x, (float, np.float64)) else x
                                                                for x in calc_df[calc_df[sales_col_name] == labels_by_var['grw']].iloc[0]]

    if 'romi' in labels_by_var:
        calc_df[calc_df[sales_col_name] == labels_by_var['romi']] = [f"{round(x, 2)}" if isinstance(x, (float, np.float64)) else x
                                                                     for x in calc_df[calc_df[sales_col_name] == labels_by_var['romi']].iloc[0]]

    apply_cols = list(calc_df)[1:]
    for col in calc_df[apply_cols]:
        calc_df[col] = ['{:,.0f}'.format(x).replace(',', ' ') if isinstance(x, (float, np.float64)) and abs(x) > 1000
                        else '{:,.2f}'.format(x).replace(',', ' ') if isinstance(x, (float, np.float64))
                        else x for x in calc_df[col]]

    return options_df, calc_df


def _fill_table(table, ds, pred_year_start, pred_year_end, last_row_bold=False):
    """Fill table for options PPTX export with zebra coloration."""

    for n, x in enumerate(ds.columns):
        if n == 0:
            table.cell(0, n).text = x
        elif n % 2 == 1:
            table.cell(0, n).text = str(pred_year_start)
        else:
            table.cell(0, n).text = f"{str(pred_year_start)}-{str(pred_year_end)}"

        if n % 4 in [1, 2]:
            table.cell(0, n).fill.solid()
            table.cell(0, n).fill.fore_color.rgb = RGBColor(127, 127, 127)

    if last_row_bold:
        for index, row in ds.iterrows():
            for n, item in enumerate(row):
                if isinstance(item, str):
                    table.cell(index + 1, n).text = item
                elif isinstance(item, float):
                    table.cell(index + 1, n).text = str(round(item, 2))
                else:
                    table.cell(index + 1, n).text = str(int(item))
                if index == len(ds) - 1:
                    table.cell(index + 1, n).text_frame.paragraphs[0].font.bold = True
    else:
        for index, row in ds.iterrows():
            for n, item in enumerate(row):
                if isinstance(item, str):
                    table.cell(index + 1, n).text = item
                elif isinstance(item, float):
                    table.cell(index + 1, n).text = str(round(item, 2))
                else:
                    table.cell(index + 1, n).text = str(int(item))
    return table
