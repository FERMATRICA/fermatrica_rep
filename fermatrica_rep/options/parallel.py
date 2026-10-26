"""
Parallel dispatcher for multi-option computation using MetaModel.compute().

Replaces deprecated option_report_multi_var / option_report_multi_post
multiprocessing path with the new MetaModel / ModelNode API.
"""


import copy
import logging
import multiprocessing
import pickle
import re
from typing import Callable

import pandas as pd

from fermatrica import fermatrica_error
from fermatrica_utils import StableClass, fermatrica_hard_warner
from fermatrica_rep.meta_model.meta_model import _last_node_name

from fermatrica_rep.meta_model.meta_model import MetaModel
from fermatrica_rep.meta_model.model_rep import ModelRep
from fermatrica_rep.options.option import Option
from fermatrica_rep.options.scope import OptionScope, _filters_strict_subset
from fermatrica_rep.options.calc_multi import opt_grid, opt_grid_expand


def _strip_meta_model_callables(meta_model: MetaModel) -> MetaModel:
    """Deep-copy MetaModel and nullify non-picklable callables on all models.

    Multiprocessing via ``multiprocessing.Pool`` requires pickling every
    object passed to worker processes. Model objects may hold bound methods
    (``transform_lhs_fn``, ``custom_predict_fn``) that reference module-level
    code the child process cannot unpickle.

    This function deep-copies the entire MetaModel including all nodes,
    datasets, and model_reps, then sets ``transform_lhs_fn = None`` and
    ``custom_predict_fn = None`` on every ``model.obj`` in every node.

    Does NOT touch model_rep attributes (``pptx_cnf``, etc.) — those are
    the caller's responsibility if they also contain unpicklable objects.

    Args:
        meta_model: Original MetaModel (not mutated — a copy is made).

    Returns:
        A deep copy of ``meta_model`` with all model callables nullified.
        The copy is safe to pickle and send to worker processes.
    """
    stripped = copy.deepcopy(meta_model)

    for node in stripped.nodes.values():
        if hasattr(node.model, "obj"):
            obj = node.model.obj
            if hasattr(obj, "transform_lhs_fn"):
                obj.transform_lhs_fn = None
            if hasattr(obj, "custom_predict_fn"):
                obj.custom_predict_fn = None

    return stripped


def _compute_option_worker(
    meta_model: MetaModel,
    option: Option,
    scopes: list[OptionScope],
    allow_past: bool,
    if_exact: bool,
) -> dict:
    """Picklable worker for ProcessPoolExecutor / Pool.starmap."""
    return meta_model.compute(option, scopes=scopes, allow_past=allow_past, if_exact=if_exact)


def compute_multi(
    meta_model: MetaModel,
    options: dict[str, Option] | list[Option],
    scopes: list[OptionScope],
    cores: int | None = None,
    *,
    allow_past: bool = False,
    if_exact: bool = False,
) -> list[dict[str, dict[str, tuple]] | None] | dict[str, dict[str, dict[str, tuple]] | None]:
    """Compute multiple budget scenarios in parallel against the same scopes.

    Each Option in `options` is a different budget spending plan (e.g.
    "option_1": 50M on TV, 30M on digital; "option_2": 20M on TV, 80M on
    digital). Every Option is computed against the **same** `scopes` —
    the scopes define the target periods and entities, not the budgets.

    The return type mirrors the input type: ``dict[str, Option]`` in →
    ``dict[str, result_dict]`` out; ``list[Option]`` in → ``list[result_dict]``
    out. Dict key order equals list iteration order (Python 3.7+ insertion order).

    Args:
        meta_model: The MetaModel DAG holding all ModelNodes. Deep-copied
            and callable-stripped before multiprocessing; the original is
            never mutated.
        options: Budget scenarios to compute, as a ``list[Option]`` or
            ``dict[str, Option]``. Each Option carries its own
            ``budget_data`` dict keyed by scope name. There is no
            requirement that ``len(options)`` equals ``len(scopes)`` —
            every option is computed against every scope.
        scopes: Target periods and entity filters where each option is
            applied. Passed unchanged to ``meta_model.compute(option,
            scopes=scopes)`` for every option. Same scopes for all.
        cores: Number of worker processes. ``None`` (default) uses
            ``cpu_count() - 1``. Set to ``1`` to force sequential
            execution (useful for debugging, no pickling required).
        allow_past: If True, budget translation may overwrite historical
            (``listed`` ≤ 3) rows. Forwarded directly to MetaModel.compute().
        if_exact: If True, budgets apply only to the exact ``scopes``
            period (no long-term extension into following years).
            Forwarded directly to MetaModel.compute().

    Returns:
        List or dict matching the input type. Element ``i`` of the list
        (or dict value for key ``i``) is ``meta_model.compute(options[i],
        scopes=scopes, ...)`` — a dict ``{node_name: {scope_name: summary}}``.
        If a particular option fails, its slot is ``None``. The caller
        should handle ``None`` entries (``_summarize_grid_results``
        skips them automatically).
    """

    input_was_dict = isinstance(options, dict)
    if input_was_dict:
        option_names = list(options.keys())
        options = list(options.values())
    else:
        option_names = None

    if cores is None:
        cores = max(1, multiprocessing.cpu_count() - 1)

    results = None

    if cores > 1 and len(options) > 1:
        try:
            stripped = _strip_meta_model_callables(meta_model)
            args = [
                (stripped, opt, scopes, allow_past, if_exact)
                for opt in options
            ]
            with multiprocessing.Pool(cores) as pool:
                raw = pool.starmap(_compute_option_worker, args)
            results = list(raw)
        except (pickle.PicklingError, TypeError, AttributeError) as e:
            logging.warning(
                f"Multiprocessing pickling failed ({e}). Falling back to sequential."
            )

    if results is None:
        results = []
        for opt in options:
            try:
                results.append(
                    meta_model.compute(
                        opt, scopes=scopes, allow_past=allow_past, if_exact=if_exact
                    )
                )
            except Exception as e:
                logging.warning(
                    f"Option '{opt.name}' failed: {e}. Appending None."
                )
                results.append(None)

    if input_was_dict:
        return dict(zip(option_names, results))
    return results


def _summarize_grid_results(
    results: list[dict[str, dict[str, tuple]] | None],
    grid: pd.DataFrame,
    meta_model: MetaModel,
    grid_type: str = "curves",
) -> dict[str, pd.DataFrame]:
    """Convert raw MetaModel.compute() results into curves DataFrames, one per node.

    Each element of ``results`` is the nested dict returned by
    ``MetaModel.compute()``: ``{node_name: {scope_name: summary}}``
    (when ``return_full=False``, the default). This function:

    1. Discovers all node names from the first non-None result entry.
    2. For each node, auto-detects the first scope with data.
    3. Flattens summary dict keys (``pred_exact_vol``, ``pred_long_vol``,
       etc.) into DataFrame rows.
    4. Joins metadata columns from the grid (``bdg``, ``option``,
       ``max_obs_budg``) to produce DataFrames matching the shape
       expected by ``curves_full_plot_short`` / ``curves_full_plot_long``.
    5. Replaces ``bdg_<name>`` option labels with display names from
       the node's ``model_rep.vis_dict`` (costs_type section).

    Args:
        results: Raw output from ``compute_multi()`` (or any list shaped
            ``[{node: {scope: summary}}, ...]``).
            ``None`` entries (failed options) are silently skipped.
        grid: The budget grid DataFrame returned by ``opt_grid()`` or
            ``opt_grid_expand()``. Must have the same number of rows as
            ``results`` (rows and results are position-aligned).
        meta_model: The MetaModel instance used to produce ``results``.
            Needed for ``model_rep`` (``vis_dict`` display names).
        grid_type: ``"curves"`` — attach only ``bdg``, ``option`` and
            ``max_obs_budg`` columns from the grid. ``"cartesian"`` —
            attach all grid columns.

    Returns:
        Dict ``{node_name: DataFrame}`` where each DataFrame has columns
        ``pred_exact_vol``, ``pred_exact_val``, ``pred_long_vol``,
        ``pred_long_val`` plus grid metadata columns (``bdg``, ``option``,
        etc.). Ready for plotting functions.
    """

    def _build_node_df(node_name: str, node_results: list[dict | None]) -> pd.DataFrame:
        scope_name = None
        for r in node_results:
            if r is None:
                continue
            for sname, sresult in r.items():
                vol = sresult.get("pred_exact_vol")
                if vol and vol > 0:
                    scope_name = sname
                    break
            if scope_name is not None:
                break

        if scope_name is None:
            return pd.DataFrame()

        rows = []
        for node_result in node_results:
            if node_result is None:
                continue

            scope_result = node_result.get(scope_name)
            if scope_result is None:
                continue

            row = {
                "pred_exact_vol": scope_result.get("pred_exact_vol"),
                "pred_exact_val": scope_result.get("pred_exact_val"),
                "pred_long_vol": scope_result.get("pred_long_vol"),
                "pred_long_val": scope_result.get("pred_long_val"),
            }
            rows.append(row)

        curves_full_df = pd.DataFrame(rows)

        pred_cols = [c for c in curves_full_df.columns if c.startswith("pred_")]
        if curves_full_df.empty or curves_full_df[pred_cols].isna().all(axis=None) or (
            not curves_full_df.empty and (curves_full_df[pred_cols] == 0).all(axis=None)
        ):
            fermatrica_hard_warner.warning(
                f"Grid summary for node '{node_name}', scope '{scope_name}' is empty. "
                "This usually means the scope does not match "
                "the model's data filters or has no budget items allocated. "
                "If you have multiple scopes, check their order — the grid "
                "keys every Option to scopes[0].scope_name, and only that "
                "scope's summary is used for the output."
            )

        if grid_type == "curves":
            cols = ["bdg", "option"]
            if "max_obs_budg" in grid.columns:
                cols = cols + ["max_obs_budg"]
            curves_full_df = pd.concat(
                [curves_full_df, grid[cols].reset_index(drop=True).iloc[: len(curves_full_df)]],
                axis=1,
            )
        else:
            curves_full_df = pd.concat(
                [curves_full_df, grid.reset_index(drop=True).iloc[: len(curves_full_df)]],
                axis=1,
            )

        model_rep = meta_model.nodes[node_name].model_rep
        pattern = re.compile(r"^[0-9]+_")
        language = model_rep.language

        if not model_rep.vis_dict.empty and "option" in curves_full_df.columns:
            for name in curves_full_df["option"].unique():
                disp_name = re.sub(r"^bdg_", "", name)

                if pattern.match(disp_name) is None:
                    ptrn = ""
                else:
                    ptrn = pattern.match(disp_name)[0]

                if (
                    pattern.sub("", disp_name)
                    in model_rep.vis_dict.loc[
                        model_rep.vis_dict["section"].isin(["costs_type"]), "variable"
                    ].array
                ):
                    curves_full_df.loc[
                        curves_full_df["option"] == name, "option"
                    ] = (
                        ptrn
                        + model_rep.vis_dict.loc[
                            (model_rep.vis_dict["variable"] == pattern.sub("", disp_name))
                            & (model_rep.vis_dict["section"].isin(["costs_type"])),
                            language,
                        ].iloc[0]
                    )

        return curves_full_df

    if not results:
        return {}

    node_names = []
    for r in results:
        if r is not None:
            node_names = list(r.keys())
            break

    result_dict: dict[str, pd.DataFrame] = {}
    for node_name in node_names:
        node_results = [
            r[node_name] if (r is not None and node_name in r) else None
            for r in results
        ]
        result_dict[node_name] = _build_node_df(node_name, node_results)

    return result_dict


def _scope_filter_subset(base_filters: dict, cand_filters: dict) -> bool:
    """Return True when ``cand_filters`` strictly contains ``base_filters``.

    ``cand_filters`` strictly contains ``base_filters`` when every key of
    ``base_filters`` is present in ``cand_filters`` with the same allowed
    values, and ``cand_filters`` has at least one extra key. An empty
    ``base_filters`` is strictly contained by any non-empty
    ``cand_filters``.

    Delegates to the shared :func:`_filters_strict_subset` so the
    background-option and entity parent-scope decisions cannot drift.

    Args:
        base_filters: Filter dict of the (wider) parent scope.
        cand_filters: Filter dict of a candidate (narrower) leaf scope.

    Returns:
        True when ``cand_filters`` is a strict superset of
        ``base_filters``.
    """
    return _filters_strict_subset(base_filters, cand_filters)


def _cross_scope_background_option(
    base: Option,
    scopes: list[OptionScope],
) -> dict[str, dict[str, float]]:
    """Derive per-scope background budget additions from a base budget Option.

    For every scope ``s`` in ``scopes`` that appears in ``base.budget_data``,
    find the candidate leaf scope ``t`` whose ``filters`` strictly contain
    ``s.filters`` (i.e. ``s`` is the national/parent scope and ``t`` a
    regional sub-scope). The background for ``s`` is the cross-scope diff
    ``base[s] - base[leaf]`` — what ``s`` spends on top of the leaf scope
    (p08: ``brand_1_2026`` minus ``brand_1_cairo_2026`` = other regions'
    ``bdg_reg_*``; shared national items cancel to 0 and are dropped).

    Scopes without a candidate leaf (leaf / standalone scopes) get no
    background. Candidate leaves are preferred when their ``date_start``
    year matches the parent scope's year; ambiguity raises
    ``FermatricaError`` asking the caller to pass an explicit
    ``background_option`` dict. Items where the leaf spends more than the
    parent are excluded with a warning. Base scopes not present in
    ``scopes`` are ignored.

    Args:
        base: Base budget scenario (e.g. ``options['predefined']``) whose
            per-scope budget_data is the reference for the diff.
        scopes: Scopes passed to ``compute_grid``.

    Returns:
        Dict ``{scope_name: {item: value}}`` of additions to merge into
        the grid items of each (non-leaf) scope.
    """
    resolved: dict[str, dict[str, float]] = {}

    for s in scopes:
        if s.scope_name not in base.budget_data:
            continue

        base_items = base.budget_data[s.scope_name]
        leaves = [
            t
            for t in scopes
            if t.scope_name in base.budget_data
            and _scope_filter_subset(s.filters, t.filters)
        ]
        if not leaves:
            continue

        matched = [t for t in leaves if t.date_start.year == s.date_start.year]
        if matched:
            if len(matched) > 1:
                fermatrica_error(
                    f"compute_grid background_option: scope '{s.scope_name}' has multiple "
                    f"candidate leaf scopes for year {s.date_start.year} "
                    f"({[t.scope_name for t in matched]}). Pass an explicit background_option dict instead."
                )
            leaf = matched[0]
        else:
            if len(leaves) > 1:
                fermatrica_error(
                    f"compute_grid background_option: scope '{s.scope_name}' has multiple "
                    f"candidate leaf scopes ({[t.scope_name for t in leaves]}). "
                    "Pass an explicit background_option dict instead."
                )
            leaf = leaves[0]

        leaf_items = base.budget_data[leaf.scope_name]
        diff: dict[str, float] = {}
        for item, val in base_items.items():
            leaf_val = leaf_items.get(item, 0.0)
            if val > leaf_val:
                diff[item] = val - leaf_val
            elif val < leaf_val:
                logging.warning(
                    f"compute_grid background_option: scope '{s.scope_name}', item '{item}' "
                    f"spends less ({val}) than the leaf scope '{leaf.scope_name}' "
                    f"({leaf_val}); excluding the item."
                )

        if diff:
            resolved[s.scope_name] = diff

    return resolved


def _resolve_background_option(
    background_option: dict | Option | None,
    scopes: list[OptionScope],
) -> dict[str, dict[str, float]] | None:
    """Normalize the ``background_option`` argument of :func:`compute_grid`.

    - ``None`` → ``None`` (no background; current behavior).
    - ``Option`` → per-scope cross-scope diff via
      :func:`_cross_scope_background_option`.
    - ``dict`` → shallow-copied raw additions
      ``{scope_name: {item: value}}``; every scope name must be present in
      ``scopes`` or ``FermatricaError`` is raised.
    - Any other type → ``FermatricaError``.

    Every resolved item value must be numeric (``int``/``float``, not
    ``bool``); non-numeric values raise ``FermatricaError`` naming the
    scope and item before any merging happens.

    Args:
        background_option: Background budget additions as a raw dict, an
            ``Option``, or ``None``.
        scopes: Scopes passed to ``compute_grid`` (validated against for
            the raw dict form).

    Returns:
        ``{scope_name: {item: value}}`` additions or ``None``.
    """
    if background_option is None:
        return None
    if isinstance(background_option, Option):
        resolved = _cross_scope_background_option(background_option, scopes)
    elif isinstance(background_option, dict):
        resolved = dict(background_option)
        scopes_names = {s.scope_name for s in scopes}
        for scope_name in resolved:
            if scope_name not in scopes_names:
                fermatrica_error(
                    f"compute_grid background_option scope '{scope_name}' is not among "
                    "the scopes passed to compute_grid."
                )
    else:
        fermatrica_error(
            "background_option must be None, a dict of {scope_name: {item: value}}, or an Option."
        )

    for scope_name, items in resolved.items():
        for item, val in items.items():
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                fermatrica_error(
                    f"compute_grid background_option item '{item}' for scope '{scope_name}' "
                    "must be numeric."
                )

    return resolved


def compute_grid(
    meta_model: MetaModel,
    scopes: list[OptionScope],
    translation: pd.DataFrame | None = None,
    budget_step: float = 1,
    fixed_vars: dict | None = None,
    grid_type: str = "curves",
    borders: dict | None = None,
    bdg_max: int | None = 301,
    cores: int | None = None,
    *,
    allow_past: bool = False,
    if_exact: bool = False,
    background_option: dict | Option | None = None,
) -> dict[str, pd.DataFrame]:
    """Calculate efficiency curves by computing a grid of budget options.

    Convenience function that builds a budget grid from ``translation``,
    converts each grid row into an ``Option``, dispatches all options to
    ``compute_multi()``, and summarizes the results into one DataFrame per
    DAG node, ready for ``curves_full_plot_short`` / ``curves_full_plot_long``.

    The same budget grid is applied to every scope in ``scopes`` — each
    Option carries ``{scope_name: budget_items}`` for all scopes.
    All scopes are passed through to ``MetaModel.compute()`` unchanged,
    so a multi-scope MetaModel sees all its scopes.

    Steps:
        1. Build budget grid via ``opt_grid()`` or ``opt_grid_expand()``.
        2. Filter by ``bdg_max``.
        3. Resolve ``background_option`` additions (see below) once.
        4. For each grid row, extract only ``bdg_*`` columns into an
           ``Option(budget_data={scope_name: {...}})`` and merge the
           background_option items into each scope. Non-``bdg_`` columns
           (``option`` label, ``bdg`` total, ``max_obs_budg``,
           fixed vars) are metadata — they are never passed as budget items.
        5. Call ``compute_multi(meta_model, options, scopes, ...)``.
        6. Call ``_summarize_grid_results()`` to flatten summaries into
           one DataFrame per DAG node.

    Background additions (per-region curves with national-model fixed
    costs): when one node's scope is a regional sub-scope (e.g. Cairo)
    and another node's scope is the national parent, the national node
    must receive the varying regional grid items **plus** the other
    regions' costs held fixed as a background. ``background_option``
    accepts:

    - ``None`` (default) — no background; every scope receives only the
      grid items.
    - ``dict[str, dict[str, float]]`` — raw additions
      ``{scope_name: {item: value}}`` merged (added) into that scope's
      budget items in every grid Option. Every scope name must be among
      ``scopes``.
    - ``Option`` — a base budget scenario (e.g. ``options['predefined']``).
      The background for each parent scope ``n`` is the cross-scope diff
      ``base[n] - base[leaf]``, where ``leaf`` is the grid's regional scope
      whose ``filters`` strictly contain ``n.filters`` (p08:
      ``brand_1_2026`` minus ``brand_1_cairo_2026`` = other regions'
      ``bdg_reg_*``; shared national items cancel to 0). Leaf scopes (with
      a ``region`` filter) get no background.

    The merged national scope therefore sees ``grid_items + other_regions``,
    so the national model varies with the region while holding other
    regions fixed. Nodes that need different budget compositions must use
    disjoint scope names (the existing ``scope_names`` pattern; a single
    ``Option`` is shared by all nodes — this limitation is unchanged).
    The output already returns one DataFrame per node, so the national
    node's curve becomes a natural extra key (e.g. ``curves_full_data['uba']``),
    non-breaking for existing callers.

    Args:
        meta_model: MetaModel DAG. Deep-copied and callable-stripped
            inside ``compute_multi()``; the original is never mutated.
        scopes: Target periods and entity filters. Every scope receives
            the same budget grid. Typically a single-element list, e.g.
            ``[scope]``, but multiple scopes (e.g. per-brand) are supported.
        translation: Translation DataFrame (as read from ``options.xlsx``
            ``translation`` sheet). Must have columns ``max_budg``,
            ``function``, ``variable``, ``base_price``. If ``None``
            (default), falls back to ``model_rep.trans_dict`` of the
            last (sink) node in the DAG, converted to DataFrame.
        budget_step: Increment between grid rows, in millions. E.g.
            ``5`` means [0M, 5M, 10M, ...] per variable.
        fixed_vars: Variables held constant across every grid row,
            e.g. ``{"price": 1}``. Merged into the grid as regular columns
            (not ``bdg_`` items, so not passed to the Option).
        grid_type: ``"curves"`` — vary one variable at a time, others
            zeroed. ``"cartesian"`` — vary all combinations using
            ``opt_grid_expand()`` with ``borders``.
        borders: Required only when ``grid_type="cartesian"``. Dict
            ``{var: [low, high]}`` defining the range for each variable.
        bdg_max: Drop any grid row whose total budget exceeds this.
            ``None`` means no filtering.
        cores: Passed to ``compute_multi()``. ``None`` → ``cpu_count()-1``.
        allow_past: Forwarded to ``MetaModel.compute()`` via ``compute_multi()``.
        if_exact: Forwarded to ``MetaModel.compute()`` via ``compute_multi()``.
        background_option: Background budget additions merged into the grid
            items of every Option, per scope. ``None`` (default), a raw
            dict ``{scope_name: {item: value}}``, or an ``Option`` whose
            per-scope cross-scope diff supplies the additions (see above).

    Returns:
        Dict ``{node_name: DataFrame}`` where each DataFrame has columns
        ``pred_exact_vol``, ``pred_exact_val``, ``pred_long_vol``,
        ``pred_long_val``, ``bdg``, ``option`` (with display names) and
        optionally ``max_obs_budg``. Same shape as the old
        ``option_report_multi_post()`` output, but keyed per node.
    """

    if translation is None:
        last_node = _last_node_name(meta_model)
        trans_dict = meta_model.nodes[last_node].model_rep.trans_dict
        if trans_dict is None:
            fermatrica_error(
                "translation is None and the sink node's model_rep.trans_dict is also None. "
                "Provide a translation DataFrame or set trans_dict on the ModelRep."
            )
        translation = pd.DataFrame(trans_dict)

    if grid_type == "curves":
        grid = opt_grid(translation, budget_step, fixed_vars)
    elif grid_type == "cartesian":
        grid = opt_grid_expand(borders, budget_step, fixed_vars)
    else:
        return {}

    if bdg_max is not None:
        grid = grid.loc[grid["bdg"] <= bdg_max, :]

    grid.reset_index(drop=True, inplace=True)  # otherwise all zeroes

    background_option_items = _resolve_background_option(background_option, scopes)

    options: dict[str, Option] = {}
    for _idx, row in grid.iterrows():
        budget_items = {
            col: float(row[col])
            for col in grid.columns
            if col.startswith("bdg_")
        }
        budget_data = {}
        for s in scopes:
            items = dict(budget_items)
            if background_option_items and s.scope_name in background_option_items:
                for k, v in background_option_items[s.scope_name].items():
                    items[k] = items.get(k, 0.0) + v
            budget_data[s.scope_name] = items
        opt = Option(
            name=str(row["option"]),
            budget_data=budget_data,
        )
        options[str(_idx)] = opt

    print(f"Options to calculate: {len(options)}")

    results_dict = compute_multi(
        meta_model,
        options,
        scopes,
        cores=cores,
        allow_past=allow_past,
        if_exact=if_exact,
    )
    results = list(results_dict.values())

    return _summarize_grid_results(
        results,
        grid,
        meta_model,
        grid_type=grid_type,
    )
