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

Functions to generate presentation slide by slide (or block of slide after another)
"""


import os
import tempfile
import shutil
import warnings

import pandas as pd
import io
import pkgutil
import sys
import copy

from pptx import Presentation
from lxml import etree

from fermatrica import Model
from fermatrica_utils import pandas_force_materialize

from fermatrica_rep.basics import _merge_filters, _auto_brand_bs_keys
from fermatrica_rep.meta_model.model_rep import ModelRep
from fermatrica_rep.meta_model.node import ModelNode
from fermatrica_rep.meta_model.meta_model import MetaModel, _last_node_name, _convert_to_metamodel
from fermatrica_rep.analysis.decomposition import extract_effect
from fermatrica_rep.export.basics import resolve_scopes_for_node
from fermatrica_rep.options.define import OptionSettings, _enrich_trans_dict_from_model, _convert_old_options, _option_settings_to_scope
from fermatrica_rep.options.scope import OptionScope
from fermatrica_rep.options.option import Option
import fermatrica_rep.export.slides as slides



"""
Generate presentation step by step (new clean API)
"""


def _auto_bs_key_filters(meta_model: MetaModel, summary_node: str) -> dict | None:
    """``{'bs_key': target-brand bs_keys}`` for the summary node, derived from its
    ``dt_pred`` (falling back to its ``ds`` when ``dt_pred`` is not populated, e.g.
    a freshly loaded MetaModel). Returns ``None`` when derivation is not possible."""
    node = meta_model.nodes[summary_node]
    model = node.model
    src = node.dt_pred if node.dt_pred is not None else node.ds
    if src is not None and model is not None and 'superbrand' in src.columns:
        return {'bs_key': _auto_brand_bs_keys(src, model)}
    return None


def options_table(prs,
                  meta_model: MetaModel,
                  scopes: list,
                   options: dict | list,
                  *,
                  brands: list | None = None,
                  brands_var: str = 'superbrand',
                  filters: dict | None = None,
                  if_exact: bool = True,
                  summary_node: str | None = None,
                  cores: int = 1,
                  ):
    """
    Export options summary as a table (new API only).

    Per scope, when the zero option has no budget/result bound to it (p09
    per-option scopes), the slide is still generated with the option's absolute
    KPIs but the zero-dependent comparison cells (incremental sales, profit,
    ROMI, long-term growth) are rendered as ``'-'``; scopes bound only to the
    zero option are reported the same way (zero baseline KPIs, unbound options
    as ``'-'``).

    Filtering (user-filters-replace-default semantics): when `brands` and
    `filters` are both None, the filter defaults to the target brand's bs_keys
    ``{'bs_key': ...}`` (current behavior); passing any user filter parameter
    uses only the user's filters.

    :param prs: Presentation object from python_pptx package
    :param meta_model: MetaModel DAG
    :param scopes: list of OptionScope objects
    :param options: dict of option name to budget data ({scope_name: {item: value}}), or list of Option objects
    :param brands: simple filter values, applied to `brands_var`
    :param brands_var: column the simple filter values apply to
    :param filters: complex filter ``{column: [values]}``
    :param if_exact: apply only to specific time period, without next years
    :param summary_node: name of the DAG node providing KPI metrics
    :param cores: number of processor cores for compute_multi (default 1 for safety)
    :return: Presentation object from python_pptx package
    """
    summary_node = summary_node or _last_node_name(meta_model)

    if brands is None and filters is None:
        filters = _auto_bs_key_filters(meta_model, summary_node)

    prs = slides.options.create(prs=prs,
                         meta_model=meta_model,
                         scopes=scopes,
                         options=options,
                         if_exact=if_exact,
                         brands=brands,
                         brands_var=brands_var,
                         filters=filters,
                         summary_node=summary_node,
                         cores=cores)

    return prs


def options_detail(prs,
                   meta_model: MetaModel,
                   scopes: list,
                   options: dict | list | None = None,
                   *,
                   brands: list | None = None,
                   brands_var: str = 'superbrand',
                   filters: dict | None = None,
                   sku_var: list | None = None,
                   period: str = 'month',
                   if_volume: bool = True,
                   if_exact: bool = True,
                   if_fit: bool = True,
                   if_decompose: bool = True,
                   if_waterfall: bool = True,
                   if_sku_fit_predict: bool = True,
                   if_sku_decompose: bool = True,
                   if_sku_waterfall: bool = True,
                   custom_extract_effect=None,
                   show_pretrain: bool = False,
                   if_debug: bool = False,
                   summary_node: str | None = None,
                   node_name: str | list[str] | None = None,
                   report_all_nodes: bool = False,
                   listed_filter: tuple = (2, 3, 4),
                   summary_scopes: list | None = None,
                   if_sku_hierarchy: bool = False,
                   whole_graph: bool | str = False,
                   ):
    """
    Export detailing report per every option. Set options=None for retro analysis (new API only).

    :param prs: Presentation object from python_pptx package
    :param meta_model: MetaModel DAG
    :param scopes: list of OptionScope objects
    :param options: dict of option name → Option, list of Option objects, or None for retro-analysis
    :param brands: simple filter values, applied to `brands_var`. When `brands` and
        `filters` are both None, defaults to the target brand's bs_keys
        ``{'bs_key': ...}`` (current behavior); passing any user filter parameter
        uses only the user's filters.
    :param brands_var: column the simple filter values apply to
    :param filters: complex filter ``{column: [values]}``
    :param sku_var: columns (variables) to report as "SKU" in detailed reporting
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
    :param show_pretrain: show pretrain periods or not
    :param if_debug: if True, debug breakpoints enabled
    :param summary_node: name of the DAG node providing KPI metrics
    :param node_name: specific node(s) to generate reports for. None uses summary_node.
    :param report_all_nodes: if True, generate reports for all nodes in the DAG
    :param listed_filter: listed values to include (default: 2=train, 3=test, 4=future).
        Pass e.g. (2, 3) to exclude future period.
    :param summary_scopes: list of OptionScope objects to aggregate into a single
        top-level slide set per Option (simultaneous scopes, e.g. per-region scopes
        of one brand). When provided, one slide set covers the union of these scopes
        instead of one slide set per resolved scope, and its waterfall window covers
        the union of scope dates. Defaults to None (one slide set per resolved scope,
        legacy behavior).
    :param if_sku_hierarchy: if True, the per-SKU section reports every prefix level of
        sku_var (coarse -> fine), e.g. with sku_var=['superbrand','master','form','region']
        it reports ['superbrand','master'], ['superbrand','master','form'],
        ['superbrand','master','form','region']. A level is skipped when its columns are
        already covered by the brand-level group_var (e.g. ['superbrand'] when group_var
        is ['superbrand']). Defaults to False (only the bottom full combination, legacy
        behavior).
    :param whole_graph: expand the decomposition link factors through the DAG (see
        ``extract_effect(whole_graph=...)``). ``False`` (default) — per-node splits.
        ``True`` or a node name expands the reported node's decomposition (and the
        per-SKU decomposition/waterfall slides) transitively. Requires the MetaModel
        to be computed (option or retro branch).
    :return: Presentation object from python_pptx package
    """
    summary_node = summary_node or _last_node_name(meta_model)

    if show_pretrain and 1 not in listed_filter:
        listed_filter = (1,) + tuple(listed_filter)

    if isinstance(options, list):
        options = {opt.name: opt for opt in options}

    if brands is None and filters is None:
        filters = _auto_bs_key_filters(meta_model, summary_node)
    else:
        filters = _merge_filters(brands=brands, brands_var=brands_var, filters=filters)

    options_iter = options if options is not None else {'mock': None}

    if options is not None and len(options) > 10:
        warnings.warn(
            f"options_detail: computing {len(options)} options with deep-copy per option. "
            "Consider using compute_multi() for batch computation if slide generation overhead is acceptable.",
            stacklevel=2,
        )

    def _resolve_nodes_to_report(mm):
        if report_all_nodes:
            return list(mm.nodes.keys())
        if node_name is not None:
            if isinstance(node_name, str):
                return [node_name]
            return list(node_name)
        return [summary_node]

    # Pre-compute fallback scopes + local state for retro branch (no compute results)
    if summary_scopes is not None:
        fallback_scopes = list(summary_scopes)
    else:
        fallback_scopes = [scopes[0]]
    fallback_meta = _scope_group_meta(fallback_scopes)

    for key, option in options_iter.items():
        if option is not None:
            mm_copy = copy.deepcopy(meta_model)
            result = mm_copy.compute(option, scopes=scopes, return_full=True, if_exact=if_exact)
            nodes_to_report = _resolve_nodes_to_report(mm_copy)

            for node_name_cur in nodes_to_report:
                try:
                    current_node = mm_copy.nodes[node_name_cur]
                    node_result = result[node_name_cur]
                    if summary_scopes is not None:
                        node_scopes = [s for s in summary_scopes if s.scope_name in node_result]
                        if not node_scopes:
                            import logging
                            logging.warning(f"Node '{node_name_cur}': none of summary_scopes present "
                                            "in compute result — skipping")
                            continue
                    else:
                        resolved = resolve_scopes_for_node(node_result, scopes)
                        node_scopes = [scope for scope, _ in resolved]
                        if not node_scopes:
                            import logging
                            logging.warning(f"Node '{node_name_cur}': none of the scopes "
                                            "in the compute result produced a non-zero "
                                            "prediction — skipping")
                            continue
                    meta = _scope_group_meta(node_scopes)
                    ds_cur = node_result[node_scopes[0].scope_name][0]
                    dt_pred_cur = node_result[node_scopes[0].scope_name][1]
                    opt_title = option.name + '. ' + ' '.join(meta['target_vals']).title()
                    if_retro = False
                    _generate_node_slides(prs, current_node, ds_cur, dt_pred_cur,
                                          option, opt_title, if_retro, node_scopes,
                                          sku_var, period, if_volume, if_fit,
                                          if_decompose, if_waterfall,
                                          if_sku_fit_predict, if_sku_decompose,
                                          if_sku_waterfall, custom_extract_effect,
                                          show_pretrain, if_debug, listed_filter,
                                          meta['group_var'], if_sku_hierarchy,
                                          filters,
                                          meta_model=mm_copy,
                                          node_name=node_name_cur,
                                          whole_graph=whole_graph)
                except Exception as e:
                    import logging
                    logging.warning(f"Failed to generate slides for node '{node_name_cur}': {e}")
        else:
            nodes_to_report = _resolve_nodes_to_report(meta_model)

            for node_name_cur in nodes_to_report:
                try:
                    current_node = meta_model.nodes[node_name_cur]
                    ds_cur = current_node.ds
                    dt_pred_cur = current_node.dt_pred
                    opt_title = ' '.join(fallback_meta['target_vals']).title()
                    if_retro = True
                    _generate_node_slides(prs, current_node, ds_cur, dt_pred_cur,
                                          None, opt_title, if_retro, fallback_scopes,
                                          sku_var, period, if_volume, if_fit,
                                          if_decompose, if_waterfall,
                                          if_sku_fit_predict, if_sku_decompose,
                                          if_sku_waterfall, custom_extract_effect,
                                          show_pretrain, if_debug, listed_filter,
                                          fallback_meta['group_var'], if_sku_hierarchy,
                                          filters,
                                          meta_model=meta_model,
                                          node_name=node_name_cur,
                                          whole_graph=whole_graph)
                except Exception as e:
                    import logging
                    logging.warning(f"Failed to generate retro slides for node '{node_name_cur}': {e}")

    return prs


def _scope_group_meta(scopes_eff):
    """date_start/date_end (union), group_var, target_vals (common filter values)."""
    date_start = min(s.date_start for s in scopes_eff)
    date_end = max(s.date_end for s in scopes_eff)
    keys_sets = [set(s.filters.keys()) for s in scopes_eff]
    common_keys = set.intersection(*keys_sets) if keys_sets else set()
    const_keys = [k for k in scopes_eff[0].filters
                  if k in common_keys
                  and len({tuple(sorted(s.filters[k])) for s in scopes_eff}) == 1]
    if const_keys:
        group_var = const_keys
        target_vals = [v for k in const_keys
                       for v in sorted(set().union(*(s.filters[k] for s in scopes_eff)))]
    else:
        group_var = ['superbrand']
        target_vals = []
    return {'date_start': date_start, 'date_end': date_end,
            'group_var': group_var, 'target_vals': target_vals}


def _sum_budgets(option, scopes_eff):
    """Sum budget items across scopes. Non-Option (dict) passed through unchanged."""
    if not hasattr(option, 'budget_for_scope'):
        return option
    items = set()
    for s in scopes_eff:
        items |= set(option.budget_for_scope(s.scope_name).keys())
    return {item: sum(option.budget_for_scope(s.scope_name).get(item, 0.0)
                      for s in scopes_eff) for item in items}


def _union_scope_filters(scopes_eff):
    """{col: sorted unique union of values} across the scope set."""
    union = {}
    for s in scopes_eff:
        for k, vals in s.filters.items():
            union.setdefault(k, set()).update(vals)
    return {k: sorted(v) for k, v in union.items()}


def _merge_filter_dicts(base: dict | None, extra: dict | None) -> dict:
    """Merge two ``{col: [values]}`` filter dicts; same-key values are
    intersected (AND semantics)."""
    merged = dict(base) if base else {}
    for col, values in (extra or {}).items():
        if col in merged:
            merged[col] = sorted(set(merged[col]) & set(values))
        else:
            merged[col] = list(values)
    return merged


def _find_scope_for_row(scopes_eff, row, sku_var):
    """Scope matching the most filter keys whose values equal row's sku_var values."""
    best, best_score = None, -1
    for s in scopes_eff:
        score = sum(1 for k, vals in s.filters.items()
                    if k in sku_var and row.get(k) in vals)
        if score > best_score:
            best, best_score = s, score
    return best


def _sku_levels_expand(sku_var, group_var, if_sku_hierarchy):
    """Expand sku_var into the prefix levels to report.

    :param sku_var: columns to report as "SKU" (coarse -> fine order)
    :param group_var: brand-level group columns already reported on a top-level slide
    :param if_sku_hierarchy: report every prefix level of sku_var (True) or only the
        bottom (full) combination (False, legacy behavior)
    :return: list[list[str]] of column levels, in coarse -> fine order
    """
    if not sku_var:
        return []
    if not if_sku_hierarchy:
        return [list(sku_var)]
    group_set = set(group_var)
    levels = [list(sku_var[:k]) for k in range(1, len(sku_var) + 1)]
    levels = [level for level in levels if not set(level) <= group_set]
    if not levels:
        levels = [list(sku_var)]
    return levels


def _iter_sku_level_keys(dt_pred_cur, superbrand, level):
    """Yield (key_row, bs_keys) for every deduped combination of `level` columns
    belonging to `superbrand` in dt_pred_cur."""
    sub = dt_pred_cur[dt_pred_cur['superbrand'] == superbrand]
    df = sub[level].drop_duplicates()
    for _, key_row in df.iterrows():
        bs_keys = key_row.to_frame().transpose().merge(sub, on=level, how='left')['bs_key'].unique()
        yield key_row, bs_keys


def _generate_node_slides(prs, node, ds_cur, dt_pred_cur,
                          option, opt_title, if_retro, node_scopes,
                          sku_var, period, if_volume, if_fit,
                          if_decompose, if_waterfall,
                          if_sku_fit_predict, if_sku_decompose,
                          if_sku_waterfall, custom_extract_effect,
                          show_pretrain, if_debug, listed_filter,
                          group_var, if_sku_hierarchy=False,
                          filters=None,
                          meta_model=None,
                          node_name=None,
                          whole_graph=False):

    superbrand = node.model.conf.target_superbrand

    ds_cur = ds_cur[ds_cur['listed'].isin(listed_filter)]
    dt_pred_cur = dt_pred_cur[dt_pred_cur['listed'].isin(listed_filter)]
    ds_cur = pandas_force_materialize(ds_cur, if_hard=False, if_soft=True)

    if sku_var:
        existing_sku_vars = [v for v in sku_var if v in dt_pred_cur.columns]
        if len(existing_sku_vars) != len(sku_var):
            import logging
            logging.warning(f"Skipping SKU slides for node: dt_pred_cur missing columns {set(sku_var) - set(existing_sku_vars)}")
            existing_sku_vars = None
    else:
        existing_sku_vars = None

    if custom_extract_effect is None:
        if whole_graph and node_name:
            split_m_m = extract_effect(meta_model, whole_graph=node_name,
                                       listed_filter=listed_filter, if_debug=if_debug)
        else:
            split_m_m = extract_effect(node, ds=ds_cur,
                                       listed_filter=listed_filter, if_debug=if_debug)
    else:
        split_m_m = custom_extract_effect(node, ds=ds_cur)

    if if_fit:
        option_cur = _sum_budgets(option, node_scopes) if len(node_scopes) > 1 else option
        prs = slides.fit_predict.create(prs=prs,
                                        node=node,
                                        dt_pred=dt_pred_cur,
                                        option_name=opt_title,
                                        opt_set=node_scopes[0],
                                        option=option_cur,
                                        period=period,
                                        show_future=True,
                                        show_pretrain=show_pretrain,
                                        group_var=group_var,
                                        plot_type='brand' + '_retro' * if_retro,
                                        filters=filters)

    if if_decompose:
        prs = slides.decomposition.create(prs=prs,
                                          node=node,
                                          split_m_m=split_m_m,
                                          option_name=opt_title,
                                          brands=[superbrand],
                                          period=period,
                                          show_future=True,
                                          show_pretrain=show_pretrain,
                                          group_var=group_var,
                                          plot_type='brand' + '_retro' * if_retro,
                                          if_volume=if_volume,
                                          filters=filters)

    if if_waterfall:
        date_start = str(min(s.date_start for s in node_scopes).date())
        date_end = str(max(s.date_end for s in node_scopes).date())

        prs = slides.waterfall.create(prs=prs,
                                      node=node,
                                      split_m_m=split_m_m,
                                      date_start=date_start,
                                      date_end=date_end,
                                      option_name=opt_title,
                                      brands=[superbrand],
                                      absolute_sort=False,
                                      plot_type='brand' + '_retro' * if_retro,
                                      if_volume=if_volume,
                                      show_pretrain=show_pretrain,
                                      filters=_merge_filter_dicts(filters,
                                                                  _union_scope_filters(node_scopes)))

    if existing_sku_vars and any([if_sku_fit_predict, if_sku_decompose, if_sku_waterfall]):
        levels = _sku_levels_expand(existing_sku_vars, group_var, if_sku_hierarchy)
        level_keys = {tuple(level): list(_iter_sku_level_keys(dt_pred_cur, superbrand, level))
                      for level in levels}
        n_levels = sum(len(items) for items in level_keys.values())
        if n_levels > 50:
            warnings.warn(
                f"_generate_node_slides: {n_levels} SKU slide sets across "
                f"{len(levels)} hierarchy levels exceed 50. Consider narrowing sku_var "
                "or disabling if_sku_hierarchy.",
                stacklevel=2,
            )

        for level in levels:
            for key_row, bs_keys in level_keys[tuple(level)]:
                try:
                    if option is not None:
                        opt_title_sku = option.name + '. ' + ' '.join(key_row.astype(str)).title()
                    else:
                        opt_title_sku = ' '.join(key_row.astype(str)).title()

                    opt_set_sku = _find_scope_for_row(node_scopes, key_row, level) or node_scopes[0]

                    filters_sku = dict(filters) if filters else {}
                    filters_sku['bs_key'] = list(bs_keys)

                    if if_sku_fit_predict:
                        prs = slides.fit_predict.create(prs=prs,
                                                        node=node,
                                                        dt_pred=dt_pred_cur,
                                                        option_name=opt_title_sku,
                                                        opt_set=opt_set_sku,
                                                        option=option,
                                                        period=period,
                                                        show_future=True,
                                                        show_pretrain=show_pretrain,
                                                        group_var=level,
                                                        plot_type='sku' + '_retro' * if_retro,
                                                        filters=filters_sku)

                    if if_sku_decompose:
                        prs = slides.decomposition.create(prs=prs,
                                                          node=node,
                                                          split_m_m=split_m_m,
                                                          option_name=opt_title_sku,
                                                          brands=[superbrand],
                                                          period=period,
                                                          show_future=True,
                                                          show_pretrain=show_pretrain,
                                                          group_var=level,
                                                          plot_type='sku' + '_retro' * if_retro,
                                                          if_volume=if_volume,
                                                          filters=filters_sku)

                    if if_sku_waterfall:
                        prs = slides.waterfall.create(prs=prs,
                                                      node=node,
                                                      split_m_m=split_m_m,
                                                      date_start=date_start,
                                                      date_end=date_end,
                                                      option_name=opt_title_sku,
                                                      brands=[superbrand],
                                                      absolute_sort=False,
                                                      plot_type='sku' + '_retro' * if_retro,
                                                      if_volume=if_volume,
                                                      show_pretrain=show_pretrain,
                                                      filters=filters_sku)
                except Exception as e:
                    import logging
                    logging.warning(f"Failed to generate SKU slides for key_row: {e}")


def curves(prs,
           meta_model: MetaModel,
           scopes: list,
           *,
           translation: dict | pd.DataFrame | None = None,
           budget_step: int | float = 5,
           bdg_max: int | float | None = 301,
            fixed_vars: dict | None = {'price': 1},
            if_exact: bool = False,
           if_volume: bool = False,
           cores: int = 4,
           background_option: dict | Option | None = None,
           ):
    """
    Export efficiency curves (incremental KPI, profit, ROI) via full curves approach (new API only).

    :param prs: Presentation object from python_pptx package
    :param meta_model: MetaModel DAG
    :param scopes: list of OptionScope objects
    :param translation: translation dict or DataFrame
    :param budget_step: budget iteration step in millions
    :param bdg_max: maximum budget size
    :param fixed_vars: translation variables with their values to be fixed across grid
    :param if_exact: apply only to the specific time period
    :param if_volume: show volume or value KPI
    :param cores: number of processor cores
    :param background_option: background budget additions for compute_grid: None (default),
        a raw dict {scope_name: {item: value}}, or an Option (base budget scenario)
        whose per-scope cross-scope diff supplies the additions. See compute_grid().
    :return: Presentation object from python_pptx package
    """
    prs = slides.incr_roi.create(prs=prs,
                               meta_model=meta_model,
                               scopes=scopes,
                               translation=translation,
                               budget_step=budget_step,
                               bdg_max=bdg_max,
                                fixed_vars=fixed_vars,
                                if_exact=if_exact,
                               if_volume=if_volume,
                               cores=cores,
                               background_option=background_option)

    return prs


def adstocks(prs,
             meta_model: MetaModel,
             cln_meas: list | tuple,
             *,
             cln_dim: list | tuple = ('superbrand', 'master', 'bs_key', 'date', 'listed', 'kpi_coef'),
             n: int = 50,
             summary_node: str | None = None,
             ):
    """
    Export adstock (decay, carry-on) curves (new API only).

    :param prs: Presentation object from python_pptx package
    :param meta_model: MetaModel DAG
    :param cln_meas: column names to be used as measurements
    :param cln_dim: column names to be used as dimensions
    :param n: number of observations per column
    :param summary_node: name of the DAG node providing the dataset for adstock calculation
    :return: Presentation object from python_pptx package
    """
    summary_node = summary_node or _last_node_name(meta_model)
    node = meta_model.nodes[summary_node]
    prs = slides.adstocks.create(prs=prs,
                                 node=node,
                                 ds=node.ds,
                                 cln_meas=cln_meas,
                                 cln_dim=cln_dim,
                                 n=n)

    return prs


"""
Generate presentation step by step (deprecated wrappers)
"""


def export_option_table(prs,
                        model: "Model | list" = None,
                        model_rep: "ModelRep | list" = None,
                        dt_pred: pd.DataFrame | list = None,
                        ds: pd.DataFrame | list = None,
                        opt_set: "OptionSettings" = None,
                        options_m: dict = None,
                        if_exact: bool = True,
                        ):
    """
    Export options summary as a table (deprecated wrapper).

    Use options_table(prs, meta_model, scopes, options, ...) instead.

    :param prs: Presentation object from python_pptx package
    :param model: Model object or list of Models
    :param model_rep: ModelRep object or list
    :param dt_pred: prediction data or list
    :param ds: main dataset or list
    :param opt_set: OptionSetting object
    :param options_m: dictionary of dictionaries defining options
    :param if_exact: apply only to specific time period, without next years
    :return: Presentation object from python_pptx package
    """
    warnings.warn(
        "export_option_table() is deprecated. Use options_table() with MetaModel, OptionScope, and Option instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    meta_model = _convert_to_metamodel(
        model, ds=ds, model_rep=model_rep, dt_pred=dt_pred,
        merge_fn_pattern="code_py.adhoc.reporting.pred_merge_{i}",
    )
    for node in meta_model.nodes.values():
        if node.model is not None:
            _enrich_trans_dict_from_model(node.model.conf, node.model_rep.trans_dict)
    scopes, options = _convert_old_options(opt_set, options_m)

    summary_node = _last_node_name(meta_model)

    if isinstance(model, list):
        superbrand = model[-1].conf.target_superbrand
        bs_key_filter = dt_pred[-1].loc[(dt_pred[-1]['superbrand'] == superbrand), 'bs_key'].unique()
    else:
        superbrand = model.conf.target_superbrand
        bs_key_filter = dt_pred.loc[(dt_pred['superbrand'] == superbrand), 'bs_key'].unique()

    return options_table(prs=prs,
                         meta_model=meta_model,
                         scopes=scopes,
                         options=options,
                         filters={'bs_key': bs_key_filter},
                         if_exact=if_exact,
                         summary_node=summary_node)


def export_option_detail(prs,
                         model: "Model | list" = None,
                         model_rep: "ModelRep | list" = None,
                         dt_pred: pd.DataFrame | list = None,
                         ds: pd.DataFrame | list = None,
                         opt_set: "OptionSettings" = None,
                         sku_var: list | None = None,
                         options_m: dict | None = None,
                         period: str = 'month',
                         if_volume: bool = True,
                         if_exact: bool = True,
                         if_fit: bool = True,
                         if_decompose: bool = True,
                         if_waterfall: bool = True,
                         if_sku_fit_predict: bool = True,
                         if_sku_decompose: bool = True,
                         if_sku_waterfall: bool = True,
                         custom_extract_effect=None,
                         show_pretrain: bool = False,
                         if_debug: bool = False,
                         listed_filter: tuple = (2, 3, 4),
                         summary_scopes: list | None = None,
                         if_sku_hierarchy: bool = False,
                         whole_graph: bool | str = False,
                         ):
    """
    Export detailing report per every option (deprecated wrapper).

    Use options_detail(prs, meta_model, scopes, options, ...) instead.

    :param prs: Presentation object from python_pptx package
    :param model: Model object or list of Models
    :param model_rep: ModelRep object or list
    :param dt_pred: prediction data or list
    :param ds: main dataset or list
    :param opt_set: OptionSetting object
    :param sku_var: columns (variables) to report as "SKU" in detailed reporting
    :param options_m: dictionary of dictionaries defining options (None for retro-analysis)
    :param period: time period / interval to group by
    :param if_volume: optimize volume or value KPI
    :param if_fit: include fit-predict plot
    :param if_decompose: include decomposition plot
    :param if_waterfall: include waterfall plot
    :param if_sku_fit_predict: include fit-predict per SKU
    :param if_sku_decompose: include decomposition per SKU
    :param if_sku_waterfall: include waterfall per SKU
    :param custom_extract_effect: custom function to extract effect for decomposition and waterfall
    :param show_pretrain: show pretrain periods or not
    :param if_debug: if True, debug breakpoints enabled
    :param listed_filter: listed values to include (default: 2=train, 3=test, 4=future)
    :param summary_scopes: list of OptionScope objects to aggregate into a single
        top-level slide set per Option (simultaneous scopes). See options_detail().
    :param if_sku_hierarchy: if True, report every prefix level of sku_var in the
        per-SKU section (coarse -> fine), skipping levels already covered by the
        brand-level group_var. Defaults to False (bottom full combination only).
    :param whole_graph: forward to options_detail() — expand the reported node's
        decomposition link factors through the DAG (deprecated shim).
    :return: Presentation object from python_pptx package
    """
    warnings.warn(
        "export_option_detail() is deprecated. Use options_detail() with MetaModel, OptionScope, and Option instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    if opt_set is not None:
        if not isinstance(opt_set, list):
            opt_set = [opt_set]
        scopes = [_option_settings_to_scope(os, i) for i, os in enumerate(opt_set)]
    else:
        from fermatrica.basics.basics import fermatrica_error
        fermatrica_error("opt_set must be provided for old API.")

    if options_m is None:
        if isinstance(model, list):
            node = ModelNode(model=model[-1], ds=ds[-1], model_rep=model_rep[-1])
            if dt_pred is not None and isinstance(dt_pred, list):
                node.dt_pred = dt_pred[-1]
        else:
            node = ModelNode(model=model, ds=ds, model_rep=model_rep)
            if dt_pred is not None:
                node.dt_pred = dt_pred
        meta_model = MetaModel(nodes=node)
        return options_detail(prs=prs,
                             meta_model=meta_model,
                             scopes=scopes,
                             options=None,
                             sku_var=sku_var,
                             period=period,
                             if_volume=if_volume,
                             if_exact=if_exact,
                             if_fit=if_fit,
                             if_decompose=if_decompose,
                             if_waterfall=if_waterfall,
                             if_sku_fit_predict=if_sku_fit_predict,
                             if_sku_decompose=if_sku_decompose,
                             if_sku_waterfall=if_sku_waterfall,
                             custom_extract_effect=custom_extract_effect,
                             show_pretrain=show_pretrain,
                             if_debug=if_debug,
                             listed_filter=listed_filter,
                             summary_scopes=summary_scopes,
                             if_sku_hierarchy=if_sku_hierarchy,
                             whole_graph=whole_graph)

    meta_model = _convert_to_metamodel(
        model, ds=ds, model_rep=model_rep, dt_pred=dt_pred,
        merge_fn_pattern="code_py.adhoc.reporting.pred_merge_{i}",
    )
    for node in meta_model.nodes.values():
        if node.model is not None:
            _enrich_trans_dict_from_model(node.model.conf, node.model_rep.trans_dict)
    scopes, options = _convert_old_options(opt_set, options_m)

    return options_detail(prs=prs,
                         meta_model=meta_model,
                         scopes=scopes,
                         options=options,
                         sku_var=sku_var,
                         period=period,
                         if_volume=if_volume,
                         if_exact=if_exact,
                         if_fit=if_fit,
                         if_decompose=if_decompose,
                         if_waterfall=if_waterfall,
                         if_sku_fit_predict=if_sku_fit_predict,
                         if_sku_decompose=if_sku_decompose,
                         if_sku_waterfall=if_sku_waterfall,
                         custom_extract_effect=custom_extract_effect,
                         show_pretrain=show_pretrain,
                         if_debug=if_debug,
                         listed_filter=listed_filter,
                         summary_scopes=summary_scopes,
                         if_sku_hierarchy=if_sku_hierarchy,
                         whole_graph=whole_graph)


"""
Deprecated wrappers (use the new clean functions above instead)
"""


def export_curves(prs,
                  model: "Model | list" = None,
                  model_rep: "ModelRep | list" = None,
                  ds: pd.DataFrame | list = None,
                  opt_set_crv: "OptionSettings | list" = None,
                  trans_dict: dict | pd.DataFrame | None = None,
                  budget_step: int | float = 5,
                  bdg_max: int | float | None = 301,
                  fixed_vars: dict | None = {'price': 1},
                  cores: int = 10,
                  if_exact: bool = False,
                  adhoc_curves_max_costs=None,
                  if_volume: bool = False,
                  ):
    """
    Export efficiency curves (incremental KPI, profit, ROI) via full curves approach (deprecated wrapper).

    Use curves(prs, meta_model, scopes, translation=..., ...) instead.

    :param prs: Presentation object from python_pptx package
    :param model: Model object or list of Models
    :param model_rep: ModelRep object or list
    :param ds: main dataset or list
    :param opt_set_crv: OptionSetting object for curves calculation
    :param trans_dict: translation dictionary
    :param budget_step: budget iteration step in millions
    :param bdg_max: maximum budget size
    :param fixed_vars: translation variables with their values to be fixed across grid
    :param cores: number of processor cores
    :param if_exact: apply only to the specific time period
    :param adhoc_curves_max_costs: adhoc function to set maximum observed values
    :param if_volume: show volume or value KPI
    :return: Presentation object from python_pptx package
    """
    warnings.warn(
        "export_curves() is deprecated. Use curves() with MetaModel, OptionScope, and translation instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    if isinstance(model, list):
        nodes = {}
        for i in range(len(model)):
            d = ds[i] if isinstance(ds, list) else ds
            mr = model_rep[i] if isinstance(model_rep, list) else model_rep
            node = ModelNode(model=model[i], ds=d, model_rep=mr)
            _enrich_trans_dict_from_model(model[i].conf, node.model_rep.trans_dict)
            nodes[f"model_{i}"] = node
        edges = []
        for i in range(len(model) - 1):
            edges.append((f"model_{i}", f"model_{i + 1}", f"code_py.adhoc.reporting.pred_merge_{i + 1}"))
        meta_model = MetaModel(nodes=nodes, edges=edges)
    else:
        node = ModelNode(model=model, ds=ds, model_rep=model_rep)
        _enrich_trans_dict_from_model(model.conf, node.model_rep.trans_dict)
        meta_model = MetaModel(nodes=node)

    if isinstance(opt_set_crv, list):
        scopes = [_option_settings_to_scope(os, i) for i, os in enumerate(opt_set_crv)]
    else:
        scopes = [_option_settings_to_scope(opt_set_crv, 0)]

    return curves(prs=prs,
                  meta_model=meta_model,
                  scopes=scopes,
                  translation=trans_dict,
                  budget_step=budget_step,
                  bdg_max=bdg_max,
                  fixed_vars=fixed_vars,
                   if_exact=if_exact,
                  if_volume=if_volume,
                  cores=cores)


def export_adstocks(prs,
                    model: "Model | list" = None,
                    ds: pd.DataFrame | list = None,
                    model_rep=None,
                    cln_meas: list | tuple = None,
                    option_dict: dict | list = None,
                    option_settings: "OptionSettings" = None,
                    cln_dim: list | tuple = ('superbrand', 'master', 'bs_key', 'date', 'listed', 'kpi_coef'),
                    n: int = 50,
                    ):
    """
    Export adstock (decay, carry-on) curves (deprecated wrapper).

    Use adstocks(prs, meta_model, cln_meas, ...) instead.

    :param prs: Presentation object from python_pptx package
    :param model: Model object or list of Models
    :param ds: main dataset or list
    :param model_rep: ModelRep object or list
    :param cln_meas: column names to be used as measurements
    :param option_dict: budget option / scenario to calculate
    :param option_settings: OptionSettings object
    :param cln_dim: column names to be used as dimensions
    :param n: number of observations per column
    :return: Presentation object from python_pptx package
    """
    warnings.warn(
        "export_adstocks() is deprecated. Use adstocks() with MetaModel instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    if isinstance(model, list):
        node = ModelNode(model=model[-1], ds=ds[-1], model_rep=model_rep[-1])
        _enrich_trans_dict_from_model(model[-1].conf, node.model_rep.trans_dict)
    else:
        node = ModelNode(model=model, ds=ds, model_rep=model_rep)
        _enrich_trans_dict_from_model(model.conf, node.model_rep.trans_dict)
    meta_model = MetaModel(nodes=node)

    return adstocks(prs=prs,
                    meta_model=meta_model,
                    cln_meas=cln_meas,
                    cln_dim=cln_dim,
                    n=n)


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
                                     "res", "templates", template_name + ".pptx")
    except:
        pptxio = template_name

    rsrc = pkgutil.get_data("fermatrica_rep", f"/res/ppt_objects/mock_text_{model_rep.language}.txt")
    txtio = io.BytesIO(rsrc)
    txtio = io.TextIOWrapper(txtio, encoding='UTF-8', errors='replace')

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
                model_rep.pptx_cnf['Blank_slide'] = layout.name

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

    # Font sizes are stored as plain int (EMU) rather than Pt objects.
    # Pt.__reduce_ex__ from python-pptx is broken: it serialises the
    # internal EMU value as the constructor argument, causing deepcopy
    # and pickle to multiply by 12700 a second time on reconstruction.
    # Computing in point-space and multiplying by the EMU factor directly
    # avoids Pt and preserves correct values through copy / pickle.
    _pt_main = round(model_rep.pptx_cnf["slide_height"] / 5.5e+5)
    model_rep.pptx_cnf["font_size_main"] = _pt_main * 12700
    model_rep.pptx_cnf["font_size_small"] = round(_pt_main / 1.2) * 12700
    model_rep.pptx_cnf["font_size_footnote"] = round(_pt_main / 1.5) * 12700
    model_rep.pptx_cnf["font_size_large"] = round(_pt_main * 1.2) * 12700
    model_rep.pptx_cnf["font_size_gross"] = round(_pt_main * 1.5) * 12700

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
