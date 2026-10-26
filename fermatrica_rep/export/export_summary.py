"""
Brand summary XLSX export.

For every brand in a MetaModel DAG, exports one XLSX file containing:

- basic model info (model type, prices, distribution / conf tables, per-brand stats)
- full efficiency curves (short- and long-term: incremental volume/value, profit, ROI)
- dynamic decomposition for older (train+test) periods
- adstocks (decay traces)

Revives the old R tool ``curves.saver.r`` (``phrm.curves.save``) that wrote
``output/summary/<brand>.xlsx`` with sheets ``dt.resp``, ``dt.prof``,
``dt.roi``, ``dt.resp.long``, ``dt.prof.long``, ``dt.roi.long``, ``setup``.
"""


import copy
import logging
import os

import pandas as pd

from fermatrica_utils import date_to_period

from fermatrica import fermatrica_error
from fermatrica_rep.analysis.adstocks import adstocks_data
from fermatrica_rep.analysis.decomposition import extract_effect
from fermatrica_rep.analysis.fit import fit_main_data
from fermatrica_rep.basics import _filter_df
from fermatrica_rep.meta_model.meta_model import MetaModel, _last_node_name
from fermatrica_rep.meta_model.node import _get_price_var
from fermatrica_rep.options.context import ComputeContext
from fermatrica_rep.options.parallel import compute_grid
from fermatrica_rep.options.scope import OptionScope, _filters_strict_subset, filter_scopes


def _derive_plan_scope_template(
    meta_model: MetaModel,
    ctx: ComputeContext,
    summary_node: str | None,
    plan_year: int | None,
) -> OptionScope:
    """
    Pick the target-brand plan scope used as the template for brand cloning.

    The target brand is ``meta_model.nodes[summary_node].model.conf.target_superbrand``.
    Candidates are the context scopes matching that brand (``match=('superbrand',
    target)`` — a scope with an explicit superbrand filter). The candidate with the
    max ``start_year`` is chosen; when several share the max year (multi-region
    layout), ``plan_year`` is required. No candidate at all raises FermatricaError.

    :param meta_model: MetaModel DAG
    :param ctx: ComputeContext holding the scopes
    :param summary_node: node providing the target brand; ``None`` resolves to the
        single sink node via ``_last_node_name``
    :param plan_year: explicit plan year to disambiguate multiple max-year candidates
    :return: the template OptionScope
    :raises FermatricaError: on missing target brand, no matching scope, or an
        unresolved ambiguity
    """
    summary_node = summary_node or _last_node_name(meta_model)
    target = getattr(meta_model.nodes[summary_node].model.conf, "target_superbrand", None)
    if not target:
        fermatrica_error(
            f"Node '{summary_node}' has no target_superbrand; cannot derive "
            "the plan scope template."
        )

    candidates = filter_scopes(ctx.scopes, match=("superbrand", target))
    if not candidates:
        fermatrica_error(
            f"No scope in the context matches superbrand '{target}'. "
            "Cannot derive the plan scope template for brand summaries."
        )

    if plan_year is not None:
        matched = [s for s in candidates if s.start_year == plan_year]
        if not matched:
            fermatrica_error(
                f"No target-brand scope for plan_year={plan_year} among "
                f"{[s.scope_name for s in candidates]}."
            )
        return matched[0]

    max_year = max(s.start_year for s in candidates)
    at_max = [s for s in candidates if s.start_year == max_year]
    if len(at_max) > 1:
        fermatrica_error(
            f"Multiple target-brand plan scopes share the max year {max_year}: "
            f"{[s.scope_name for s in at_max]}. Pass plan_year to disambiguate."
        )
    return at_max[0]


def _derive_brand_scopes(
    template: OptionScope,
    brand: str,
    seasonality_mode: str | None,
) -> list[OptionScope]:
    """
    Clone the template scope for one brand.

    The derived scope keeps all template attributes (dates, plan period, inflation,
    reference period) but gets a renamed ``scope_name`` (target brand replaced in the
    name, or ``<name>_<brand>`` appended when absent) and the ``superbrand`` filter
    swapped to ``[brand]`` (other filter columns are kept as-is).

    ``seasonality_mode='inherit'`` (default) reuses the template's seasonality
    distribution: when the template has no explicit ``seasonality`` key,
    ``_apply_media_ssn`` falls back to the scope name, so the derived scope points at
    the template's scope name. ``None`` disables seasonality explicitly.

    :param template: target-brand plan scope to clone
    :param brand: brand to clone the scope for
    :param seasonality_mode: ``'inherit'`` (default) or ``None``
    :return: single-element list with the derived scope
    :raises FermatricaError: on an unknown ``seasonality_mode``
    """
    target_vals = template.filters.get("superbrand", [])
    target = str(target_vals[0]) if target_vals else ""

    s = template.copy(deep=True)
    if target and target in s.scope_name:
        s.scope_name = s.scope_name.replace(target, brand)
    else:
        s.scope_name = f"{s.scope_name}_{brand}"
    s.filters = {
        col: ([brand] if col == "superbrand" else vals)
        for col, vals in template.filters.items()
    }

    if seasonality_mode == "inherit":
        s.seasonality = template.seasonality or template.scope_name
    elif seasonality_mode is None:
        s.seasonality = None
    else:
        fermatrica_error(
            f"Unknown seasonality_mode '{seasonality_mode}'. Use 'inherit' or None."
        )

    return [s]


def _non_superbrand_filters(filters: dict[str, list[str]]) -> dict[str, str]:
    """The scope's entity filter minus the ``superbrand`` dimension.

    Returns ``{column: single_value}`` (one value per non-superbrand column).
    """
    return {
        col: str(vals[0])
        for col, vals in filters.items()
        if col != "superbrand" and vals
    }


def _derive_clone_brand(
    meta_model: MetaModel,
    ctx: ComputeContext,
    summary_node: str,
    brand: str,
    plan_year: int | None,
    seasonality_mode: str | None,
) -> tuple[dict[str, str], str, list[OptionScope]]:
    """Fallback clone path for brands without context plan scopes.

    Returns ``(entity, template_scope_name, derived_scopes)`` where the
    entity is the derived scope's non-superbrand filter combo (the clone
    keeps the template's non-superbrand filter columns, so a regional
    template yields a regional entity). Shared by ``summary_data`` and
    ``export_summaries`` so the two call sites agree.
    """
    template = _derive_plan_scope_template(meta_model, ctx, summary_node, plan_year)
    derived = _derive_brand_scopes(template, brand, seasonality_mode)
    return _non_superbrand_filters(derived[0].filters), template.scope_name, derived


def _brand_plan_scopes(
    ctx: ComputeContext,
    brand: str,
    plan_year: int | None,
) -> list[OptionScope]:
    """Plan scopes of one brand: scopes matching the brand on the
    ``superbrand`` dimension for the plan year.

    ``plan_year=None`` resolves to the max start year among the matching
    scopes (ties across regions are fine — they become distinct entities).
    """
    candidates = filter_scopes(ctx.scopes, match=("superbrand", brand))
    if not candidates:
        return []
    year = plan_year if plan_year is not None else max(s.start_year for s in candidates)
    return filter_scopes(candidates, start_year=year)


def _brand_entities(brand_plan_scopes: list[OptionScope]) -> list[dict[str, str]]:
    """Distinct non-superbrand filter combos among the brand's plan scopes.

    Each entity is ``{column: value}``; the combos are sorted for
    deterministic file naming.
    """
    entities = {frozenset(_non_superbrand_filters(s.filters).items()) for s in brand_plan_scopes}
    return [dict(e) for e in sorted(entities, key=lambda e: sorted(e))]


def _entity_plan_scopes(
    brand_plan_scopes: list[OptionScope],
    entity: dict[str, str],
) -> list[OptionScope]:
    """Plan scopes whose non-superbrand filter dict *equals* ``entity``.

    Equality (not ``s.match``) — a national scope must not double as a
    regional entity's scope.
    """
    return [s for s in brand_plan_scopes if _non_superbrand_filters(s.filters) == entity]


def _entity_label(entity: dict[str, str]) -> str:
    """File-suffix / sheet label for one entity: lowercased values joined by
    ``_`` (``{}`` → ``""``)."""
    return "_".join(str(val).lower() for _, val in sorted(entity.items()))


def _entity_filters_for_node(node, entity: dict[str, str] | None) -> dict | None:
    """Entity filters ``{column: [value]}`` restricted to columns present in
    the node's dataset. ``None``/empty → ``None`` (no entity filter)."""
    if not entity:
        return None
    out = {col: [val] for col, val in entity.items() if col in node.ds.columns}
    return out or None


def _sanitize_filename(value: str) -> str:
    """Replace filesystem-invalid characters in a file name stem."""
    for ch in '\\/:*?"<>|':
        value = value.replace(ch, "_")
    return value


def _parent_scopes(
    ctx: ComputeContext,
    meta_model: MetaModel,
    entity: dict[str, str],
    entity_scopes: list[OptionScope],
    plan_year: int | None,
) -> list[OptionScope]:
    """Parent (wider) grid scopes for a partly-regional DAG.

    For every node that has ``scope_names`` but none of the entity's plan
    scopes, the node's own context scopes (plan year) whose non-superbrand
    filter is a **strict subset** of the entity's are added, so the node
    still gets a grid scope (p08 ``uba``: ``brand_1_2026``, ``{}`` ⊂
    ``{region: Cairo}``). Deduplicated, input order preserved.
    """
    year = plan_year if plan_year is not None else max(s.start_year for s in entity_scopes)
    entity_names = {s.scope_name for s in entity_scopes}

    parents: list[OptionScope] = []
    seen: set[str] = set()
    for node in meta_model.nodes.values():
        if node.scope_names is None or entity_names & node.scope_names:
            continue
        for s in ctx.scopes:
            if (
                s.scope_name not in node.scope_names
                or s.start_year != year
                or s.scope_name in seen
            ):
                continue
            if _filters_strict_subset(_non_superbrand_filters(s.filters), entity):
                parents.append(s)
                seen.add(s.scope_name)
    return parents


def _brand_stats(node, brand: str, filters: dict | None = None) -> pd.DataFrame | None:
    """
    Per-brand statistics derived from the node dataset (long variable|value frame).

    Rows: ``sku_count`` (unique ``bs_key``), ``date_min``/``date_max``, ``n_rows``,
    ``price_mean``/``price_median`` (when a price var exists), and
    ``observed_sum``/``predicted_sum`` on ``listed.isin([2, 3])`` (when present).
    Returns ``None`` when the brand has no rows in ``node.ds``.
    """
    ds = _filter_df(node.ds, brands=[brand], filters=filters)
    if ds.empty:
        logging.warning(f"brand_summary: brand '{brand}' has no rows in node ds — brand_stats omitted")
        return None

    stats: dict[str, object] = {}
    if "bs_key" in ds.columns:
        stats["sku_count"] = int(ds["bs_key"].nunique())
    if "date" in ds.columns:
        stats["date_min"] = pd.Timestamp(ds["date"].min())
        stats["date_max"] = pd.Timestamp(ds["date"].max())
    stats["n_rows"] = len(ds)
    price_var = _get_price_var(node.model.conf)
    if price_var and price_var in ds.columns:
        stats["price_mean"] = ds[price_var].mean()
        stats["price_median"] = ds[price_var].median()
    if "listed" in ds.columns:
        mask = ds["listed"].isin([2, 3])
        for col in ("observed", "predicted"):
            if col in ds.columns:
                stats[f"{col}_sum"] = ds.loc[mask, col].sum()

    return pd.DataFrame({"variable": list(stats.keys()), "value": list(stats.values())})


def _fit_table(
    node,
    brand: str,
    period: str,
    filters: dict | None = None,
) -> pd.DataFrame | None:
    """
    Fit (observed vs predicted) wide table for one brand, grouped by period.

    Requires ``node.dt_pred`` (returns ``None`` with a warning otherwise). Error
    interval columns (``*_lower`` / ``*_upper``) are dropped.
    """
    if node.dt_pred is None:
        logging.warning(
            f"brand_summary: node '{node.model.conf.Y_var}' has no dt_pred — fit sheet omitted"
        )
        return None
    wide = fit_main_data(
        node,
        period=period,
        show_future=False,
        show_pretrain=False,
        brands=[brand],
        filters=filters,
    )[0]
    drop = [c for c in wide.columns if c.endswith("_lower") or c.endswith("_upper")]
    if drop:
        wide = wide.drop(columns=drop)
    return wide


def _decomposition_table(
    node,
    brand: str,
    period: str,
    listed_filter: tuple | list,
    col: str = "value",
    filters: dict | None = None,
    meta_model: MetaModel | None = None,
    node_name: str | None = None,
    whole_graph: bool = False,
) -> pd.DataFrame | None:
    """
    Dynamic decomposition for one brand, wide by variable, aggregated by period.

    ``extract_effect`` output is floored to the period start, summed per
    ``(date, variable)`` on ``col`` (``'value'`` for volume, ``'value_rub'`` for
    value), and pivoted wide. Returns ``None`` when the brand has no effects.
    With ``whole_graph=True`` the link factors are expanded through the DAG
    (see ``extract_effect(whole_graph=...)``).
    """
    if whole_graph:
        split = extract_effect(
            meta_model, whole_graph=node_name, listed_filter=listed_filter,
            brands=[brand], filters=filters,
        )
    else:
        split = extract_effect(
            node, listed_filter=listed_filter, brands=[brand], filters=filters
        )
    if split is None or split.empty:
        logging.warning(
            f"brand_summary: empty decomposition for node '{node.model.conf.Y_var}', brand '{brand}' — omitted"
        )
        return None
    split = split.copy()
    split["date"] = date_to_period(split["date"], period)
    agg = split.groupby(["date", "variable"], as_index=False)[col].sum()
    wide = agg.pivot(index="date", columns="variable", values=col).reset_index()
    wide.columns.name = None
    wide = wide.sort_values("date").reset_index(drop=True)
    return wide


def _curves_tables(curves_node_df: pd.DataFrame | None):
    """
    Incremental curves (short- and long-term) from one node's grid DataFrame.

    The ``"zero"`` grid row is the baseline; per non-zero option row:
    ``incremental_volume``/``incremental_value`` vs the baseline,
    ``profit = incremental_value - bdg * 1e6``, ``roi = incremental_value / (bdg * 1e6)``.
    Returns ``(short, long)`` frames with columns ``[bdg, option,
    incremental_volume, incremental_value, profit, roi]``; ``"zero"`` rows are
    dropped. Either frame is ``None`` when it cannot be computed (missing baseline
    or empty grid).
    """
    if (
        curves_node_df is None
        or curves_node_df.empty
        or "option" not in curves_node_df.columns
        or "bdg" not in curves_node_df.columns
    ):
        return None, None

    base_mask = curves_node_df["option"] == "zero"
    if not base_mask.any():
        return None, None

    out = {}
    for key, vol_col, val_col in (
        ("short", "pred_exact_vol", "pred_exact_val"),
        ("long", "pred_long_vol", "pred_long_val"),
    ):
        if vol_col not in curves_node_df.columns or val_col not in curves_node_df.columns:
            out[key] = None
            continue
        base_vol = curves_node_df.loc[base_mask, vol_col]
        base_val = curves_node_df.loc[base_mask, val_col]
        if base_vol.empty or base_val.empty or pd.isna(base_vol.iloc[0]) or pd.isna(base_val.iloc[0]):
            out[key] = None
            continue
        base_vol = float(base_vol.iloc[0])
        base_val = float(base_val.iloc[0])

        rows = []
        for _, row in curves_node_df.loc[~base_mask].iterrows():
            bdg = float(row["bdg"])
            inc_vol = float(row[vol_col]) - base_vol
            inc_val = float(row[val_col]) - base_val
            rows.append({
                "bdg": bdg,
                "option": row["option"],
                "incremental_volume": inc_vol,
                "incremental_value": inc_val,
                "profit": inc_val - bdg * 1e6,
                "roi": inc_val / (bdg * 1e6) if bdg > 0 else float("nan"),
            })
        out[key] = pd.DataFrame(rows, columns=[
            "bdg", "option", "incremental_volume", "incremental_value", "profit", "roi",
        ])

    return out["short"], out["long"]


def _adstocks_table(
    node,
    brand: str,
    n: int,
    scaled: bool,
    filters: dict | None = None,
) -> pd.DataFrame | None:
    """
    Adstock decay traces for one brand, long ``step | variable | value`` frame.

    Measurement columns are auto-derived from ``model.conf.params`` rows whose
    ``fun`` contains ``adstock|dwblp``. Returns ``None`` (with a warning) when the
    node has no adstock/dwblp params or produces no data.
    """
    conf = node.model.conf
    if not hasattr(conf, "params") or conf.params is None or conf.params.empty:
        return None
    meas = conf.params.loc[
        conf.params["fun"].str.contains("adstock|dwblp", na=False), "variable"
    ].unique()
    if len(meas) == 0:
        logging.warning(
            f"brand_summary: node '{conf.Y_var}' has no adstock/dwblp params — adstocks sheet omitted"
        )
        return None

    df = adstocks_data(
        node, brands=[brand], filters=filters, cln_meas=list(meas), n=n, if_scaled=scaled
    )
    if df is None or df.empty:
        logging.warning(
            f"brand_summary: empty adstock trace for node '{conf.Y_var}', brand '{brand}' — omitted"
        )
        return None

    # keep only columns referenced by an active model-formula token (same rule as
    # the adstock slide): variables present in ds but absent from the formula have
    # no display name in the model and are excluded from the sheet
    active_tokens = conf.model_rhs.loc[conf.model_rhs['if_active'] == 1, 'token']
    df = df.loc[:, [active_tokens.str.contains(c).any() for c in df.columns]]

    if df.shape[1] == 0:
        logging.warning(
            f"brand_summary: no model-formula adstock variables for node '{conf.Y_var}', brand '{brand}' — omitted"
        )
        return None

    df = df.reset_index(drop=True)
    df = df.reset_index(drop=False)
    df = df.rename(columns={df.columns[0]: "step"})
    df = df.melt(id_vars="step", var_name="variable", value_name="value")
    return df.sort_values(["step", "variable"]).reset_index(drop=True)


def _sheet_prefixes(meta_model: MetaModel) -> list[str]:
    """Node names in DAG topological order — used for sheet ordering and prefixes."""
    return meta_model._topological_order()


def _description_sheet() -> pd.DataFrame:
    """Sheet glossary (``Sheet | Description | Detailed``), mirroring ModelConfExp._info."""
    descr = {
        "brands": ["Brand identification: brand, target brand, entity (regional segment), scope name(s) (+ legacy template/derived scope rows)", ""],
        "description": ["Sheet glossary", ""],
        "setup": ["Model configuration key-value pairs (raw setup sheet of model_conf)", ""],
        "brand_stats": ["Per-brand statistics derived from the node dataset", ""],
        "params": ["Model params", ""],
        "rhs": ["RHS", ""],
        "lhs": ["LHS", ""],
        "trans_path": ["Transformation path table (prices etc.)", ""],
        "fit": ["Observed and predicted value and volume by brand by period", ""],
        "decomposition": ["Decomposition of the impact of predictors by period (volume)", ""],
        "decomposition_rub": ["Decomposition of the impact of predictors by period (value)", ""],
        "curves_short": ["Short-term efficiency curves: incremental volume/value, profit, ROI vs zero budget", ""],
        "curves_long": ["Long-term efficiency curves: incremental volume/value, profit, ROI vs zero budget", ""],
        "adstocks": ["Adstock decay traces per media variable", ""],
    }
    df = pd.DataFrame(descr).T.reset_index()
    df.columns = ["Sheet", "Description", "Detailed"]
    return df


def summary_data(
    meta_model: MetaModel,
    ctx: ComputeContext,
    brand: str,
    *,
    entity: dict[str, str] | None = None,
    scopes: list[OptionScope] | None = None,
    budget_step: float = 5,
    bdg_max: int = 300,
    cores: int | None = None,
    if_exact: bool = True,
    allow_past: bool = False,
    decomp_period: str = "month",
    decomp_listed_filter: tuple = (2, 3),
    if_fit: bool = True,
    adstock_n: int = 50,
    adstock_scaled: bool = True,
    seasonality_mode: str | None = "inherit",
    plan_year: int | None = None,
    summary_node: str | None = None,
    whole_graph: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Build the full sheet dict for ONE (brand, entity).

    Scopes are derived from the brand's context plan scopes (``scopes=None``,
    default) or taken from the ``scopes`` override as-is (verbatim, no entity
    logic). The MetaModel is **deep-copied** before ``compute_grid`` (its
    sequential branch mutates the original), so the caller's ``meta_model``
    (used for the decomposition / fit / adstocks / setup sheets, and for
    subsequent brands/entities) is never corrupted.

    Sheets that cannot be produced (no adstock variables, no ``dt_pred``, empty
    brand filter, empty grid) are omitted with a ``logging.warning``.

    :param meta_model: MetaModel DAG (in-memory; ``node.ds`` / ``node.dt_pred``
        reflect the reporting script's ``mm.compute(option, scopes)`` state)
    :param ctx: ComputeContext providing scopes / seasonality / media prices
    :param brand: brand (``superbrand`` value) to export
    :param entity: entity (non-superbrand filter combo, ``{column: value}``) to
        build the sheets for; ``None`` derives the single entity (clone path or
        an error when the brand has multiple entities)
    :param scopes: override — use these scopes as-is instead of deriving them
        from the plan scopes (no entity splitting, no entity filters)
    :param budget_step: curves grid step in millions
    :param bdg_max: curves grid maximum total budget
    :param cores: worker processes for the grid (``None`` → ``cpu_count() - 1``)
    :param if_exact: apply budgets only to the exact scope period (no long-term extension)
    :param allow_past: allow budget translation over historical rows (forwarded to compute_grid)
    :param decomp_period: decomposition / fit aggregation period
    :param decomp_listed_filter: listed values for the decomposition sheets (train+test by default)
    :param if_fit: include ``{n}_fit`` sheets (requires ``node.dt_pred``)
    :param adstock_n: number of adstock decay trace steps
    :param adstock_scaled: scale adstock traces by their maximum value
    :param seasonality_mode: ``'inherit'`` (default) or ``None``; see ``_derive_brand_scopes``
    :param plan_year: plan year to pick a non-max plan year explicitly
    :param summary_node: node providing the target brand (default: the DAG sink)
    :param whole_graph: expand the decomposition sheets' link factors through the
        DAG (see ``extract_effect(whole_graph=...)``). ``False`` (default) — per-node
        splits. Applies to the ``decomposition`` and ``decomposition_rub`` sheets.
    :return: ``{sheet_name: DataFrame}``
    """
    summary_node = summary_node or _last_node_name(meta_model)
    target = getattr(meta_model.nodes[summary_node].model.conf, "target_superbrand", "")

    if scopes is not None:
        entity = None
        entity_label = ""
        grid_scopes = list(scopes)
        scope_names = ", ".join(s.scope_name for s in scopes)
        background = None
        template_scope = ""
    else:
        brand_plan_scopes = _brand_plan_scopes(ctx, brand, plan_year)
        if not brand_plan_scopes:
            # fallback clone path (brands without context scopes, e.g. p06
            # brand_2/brand_3): one derived scope; its non-superbrand filters
            # become the (single) entity
            entity, template_scope, derived = _derive_clone_brand(
                meta_model, ctx, summary_node, brand, plan_year, seasonality_mode
            )
            grid_scopes = derived
            scope_names = ", ".join(s.scope_name for s in derived)
            background = None
        else:
            entities = _brand_entities(brand_plan_scopes)
            if entity is None:
                if len(entities) > 1:
                    fermatrica_error(
                        f"Brand '{brand}' has multiple entities "
                        f"({entities}). Pass entity=... to pick one."
                    )
                entity = entities[0]
            entity_scopes = _entity_plan_scopes(brand_plan_scopes, entity)
            if not entity_scopes:
                fermatrica_error(
                    f"No plan scope for brand '{brand}' matches entity {entity}."
                )
            parents = _parent_scopes(ctx, meta_model, entity, entity_scopes, plan_year)
            grid_scopes = entity_scopes + parents
            scope_names = ", ".join(s.scope_name for s in entity_scopes)
            background = ctx.options.get("predefined") if parents else None
            template_scope = ""
            if parents and background is None:
                logging.warning(
                    f"brand_summary: entity {entity} of brand '{brand}' added parent "
                    "scopes but the context has no 'predefined' option — running "
                    "the grid without a background."
                )
        entity_label = _entity_label(entity)

    mm_copy = copy.deepcopy(meta_model)
    grid = compute_grid(
        mm_copy,
        grid_scopes,
        budget_step=budget_step,
        bdg_max=bdg_max,
        cores=cores,
        if_exact=if_exact,
        allow_past=allow_past,
        background_option=background,
    )

    sheets: dict[str, pd.DataFrame] = {
        "brands": pd.DataFrame({
            "variable": ["brand", "target_brand", "entity", "scopes", "template_scope", "derived_scopes"],
            "value": [
                brand,
                target,
                entity_label,
                scope_names,
                template_scope,
                scope_names,
            ],
        }),
        "description": _description_sheet(),
    }

    for name in _sheet_prefixes(meta_model):
        node = meta_model.nodes[name]
        conf = node.model.conf
        p = f"{name}_"
        node_filters = _entity_filters_for_node(node, entity)

        setup = getattr(conf, "_model_setup", None)
        if setup is not None and not setup.empty:
            sheets[p + "setup"] = setup.copy()

        stats = _brand_stats(node, brand, filters=node_filters)
        if stats is not None:
            sheets[p + "brand_stats"] = stats

        sheets[p + "params"] = conf.params.copy()
        sheets[p + "rhs"] = conf.model_rhs.copy()
        if conf.model_lhs is not None:
            sheets[p + "lhs"] = conf.model_lhs.copy()
        if conf.trans_path_df is not None:
            sheets[p + "trans_path"] = conf.trans_path_df.copy()

        if if_fit:
            fit = _fit_table(node, brand, decomp_period, filters=node_filters)
            if fit is not None:
                sheets[p + "fit"] = fit

        decomp = _decomposition_table(
            node, brand, decomp_period, decomp_listed_filter,
            col="value", filters=node_filters,
            meta_model=meta_model, node_name=name, whole_graph=whole_graph,
        )
        if decomp is not None:
            sheets[p + "decomposition"] = decomp
        decomp_rub = _decomposition_table(
            node, brand, decomp_period, decomp_listed_filter,
            col="value_rub", filters=node_filters,
            meta_model=meta_model, node_name=name, whole_graph=whole_graph,
        )
        if decomp_rub is not None:
            sheets[p + "decomposition_rub"] = decomp_rub

        short, long = _curves_tables(grid.get(name))
        if short is not None:
            sheets[p + "curves_short"] = short
        if long is not None:
            sheets[p + "curves_long"] = long

        adst = _adstocks_table(node, brand, adstock_n, adstock_scaled, filters=node_filters)
        if adst is not None:
            sheets[p + "adstocks"] = adst

    return sheets


def _write_workbook(path: str, sheets: dict[str, pd.DataFrame]) -> None:
    """Write the sheet dict to one XLSX workbook (xlsxwriter engine)."""
    with pd.ExcelWriter(path, engine="xlsxwriter", datetime_format="YYYY-MM-DD") as writer:
        for sheet_name, df in sheets.items():
            if df is None or df.empty:
                continue
            safe = sheet_name[:31]
            df = df.copy()
            df = df.where(pd.notna(df), None)
            df.to_excel(writer, sheet_name=safe, index=False)
            writer.sheets[safe].freeze_panes(1, 0)


def export_summaries(
    meta_model: MetaModel,
    ctx: ComputeContext,
    *,
    brands: list[str] | None = None,
    entities: list[dict[str, str]] | None = None,
    scopes: list[OptionScope] | None = None,
    output_dir: str = "output/summary",
    budget_step: float = 5,
    bdg_max: int = 300,
    cores: int | None = None,
    if_exact: bool = True,
    allow_past: bool = False,
    decomp_period: str = "month",
    if_fit: bool = True,
    adstock_n: int = 50,
    adstock_scaled: bool = True,
    seasonality_mode: str | None = "inherit",
    plan_year: int | None = None,
    summary_node: str | None = None,
    whole_graph: bool = False,
) -> list[str]:
    """
    Export one XLSX file per (brand, entity) into ``output_dir``.

    A brand whose plan scopes carry non-superbrand filters (region/master) is
    split into one file per entity (non-superbrand filter combo) by default,
    e.g. p04 → ``brand_1_moscow.xlsx`` … ``brand_1_cairo.xlsx``; a national
    entity ``{}`` yields ``brand.xlsx``. Pure-brand models (p06) keep today's
    behavior (one file per brand, derived scopes).

    Brands default to the sorted unique ``superbrand`` values of the sink node's
    dataset. Each (brand, entity) is isolated in its own try/except (a failing
    one is skipped with a warning, mirroring ``options_detail``); a file is
    written only when its sheet dict is non-empty.

    :param meta_model: MetaModel DAG
    :param ctx: ComputeContext
    :param brands: brands to export; ``None`` auto-derives from the sink node's ds
    :param entities: restrict to these entities (e.g. ``[{"region": "Cairo"}]``
        or ``[{}]`` for the national file); ``None`` auto-derives all entities
        per brand. Ignored when ``scopes`` is given.
    :param scopes: override — use these scopes as-is instead of deriving them
        from the plan scopes (verbatim grid scopes, no entity splitting)
    :param output_dir: output directory (created when missing)
    :param budget_step: curves grid step in millions
    :param bdg_max: curves grid maximum total budget
    :param cores: worker processes for the grid (``None`` → ``cpu_count() - 1``)
    :param if_exact: apply budgets only to the exact scope period
    :param allow_past: allow budget translation over historical rows
    :param decomp_period: decomposition / fit aggregation period
    :param if_fit: include ``{n}_fit`` sheets
    :param adstock_n: number of adstock decay trace steps
    :param adstock_scaled: scale adstock traces by their maximum value
    :param seasonality_mode: ``'inherit'`` (default) or ``None``
    :param plan_year: plan year to pick a non-max plan year explicitly
    :param summary_node: node providing the target brand (default: the DAG sink)
    :param whole_graph: expand the decomposition sheets' link factors through the
        DAG (see ``extract_effect(whole_graph=...)``). Forwarded to ``summary_data``.
    :return: list of written file paths
    :raises FermatricaError: when no brands can be derived (no ``superbrand`` column)
    """
    summary_node = summary_node or _last_node_name(meta_model)

    if brands is None:
        ds = meta_model.nodes[summary_node].ds
        if "superbrand" not in ds.columns:
            fermatrica_error(
                f"MetaModel sink node '{summary_node}' dataset has no 'superbrand' "
                "column. Pass brands explicitly."
            )
        brands = sorted(ds["superbrand"].dropna().unique().tolist())

    entities_set = None
    if entities is not None:
        entities_set = {frozenset(e.items()) for e in entities}

    os.makedirs(output_dir, exist_ok=True)
    written: list[str] = []

    for brand in brands:
        try:
            if scopes is not None:
                brand_entities: list[dict[str, str] | None] = [None]
            else:
                brand_plan_scopes = _brand_plan_scopes(ctx, brand, plan_year)
                if not brand_plan_scopes:
                    # clone path — one derived scope; its non-superbrand filters
                    # are the (single) entity (kept from the template, so it may
                    # be regional)
                    clone_entity, _, _ = _derive_clone_brand(
                        meta_model, ctx, summary_node, brand, plan_year, seasonality_mode
                    )
                    if entities_set is not None and frozenset(clone_entity.items()) not in entities_set:
                        brand_entities = []
                    else:
                        brand_entities = [clone_entity]
                else:
                    brand_entities = _brand_entities(brand_plan_scopes)
                    if entities_set is not None:
                        brand_entities = [
                            e for e in brand_entities if frozenset(e.items()) in entities_set
                        ]
        except Exception as e:
            logging.warning(f"brand_summary: export failed for brand '{brand}': {e}")
            continue

        for entity in brand_entities:
            try:
                sheets = summary_data(
                    meta_model,
                    ctx,
                    brand,
                    entity=entity,
                    scopes=scopes,
                    budget_step=budget_step,
                    bdg_max=bdg_max,
                    cores=cores,
                    if_exact=if_exact,
                    allow_past=allow_past,
                    decomp_period=decomp_period,
                    if_fit=if_fit,
                    adstock_n=adstock_n,
                    adstock_scaled=adstock_scaled,
                    seasonality_mode=seasonality_mode,
                    plan_year=plan_year,
                    summary_node=summary_node,
                    whole_graph=whole_graph,
                )
            except Exception as e:
                logging.warning(
                    f"brand_summary: export failed for brand '{brand}', "
                    f"entity '{entity}': {e}"
                )
                continue

            if not sheets:
                logging.warning(
                    f"brand_summary: no sheets produced for brand '{brand}', "
                    f"entity '{entity}' — file skipped"
                )
                continue

            label = _entity_label(entity) if entity else ""
            stem = _sanitize_filename(brand)
            if label:
                stem += "_" + _sanitize_filename(label)
            path = os.path.join(output_dir, f"{stem}.xlsx")
            _write_workbook(path, sheets)
            written.append(path)

    return written
