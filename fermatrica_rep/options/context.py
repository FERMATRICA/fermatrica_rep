"""
Context — a :class:`ComputeContext` and everything that produces one.

A ``ComputeContext`` is the single holder for all options-related settings
scopes / options / seasonality / media prices / translation sheets — that a
reporting pipeline needs, consumed by ``MetaModel(compute_context=...)``.

Two producers live here:

- :func:`load_context` — read an existing ``context.xlsx`` (scopes +
  allocations + optional seasonality/media_prices/translation sheets).
- :func:`campaigns_to_options` — derive scopes / allocations / seasonality
  from a simple campaign table (``option | item | date_start | date_end |
  val``, optionally with a ``target`` column), optionally writing a
  ready-to-load ``context.xlsx``.

The low-level XLSX sheet readers live in :mod:`fermatrica_rep.options.define`.
"""


import dataclasses
import os
import shutil

import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.utils import get_column_letter

from fermatrica_utils import date_range_to_periods, fermatrica_warner
from fermatrica.basics.basics import fermatrica_error

from fermatrica_rep.options.define import (
    scopes_read,
    seasonality_read,
    media_prices_read,
    trans_dict_create,
)
from fermatrica_rep.options.option import Option
from fermatrica_rep.options.scope import OptionScope


@dataclasses.dataclass(frozen=True)
class ComputeContext:
    """
    Single holder for everything loaded from a context XLSX file.

    Produced by :func:`load_context` and :func:`campaigns_to_options`,
    consumed by :class:`MetaModel` (``compute_context=``) to wire per-node
    ``trans_dict`` / ``ssn_dict`` / ``media_prices``. Scopes are not stored
    on the ``MetaModel`` — they must be passed explicitly to every
    computation entry point (``compute()``, optimizers, ``compute_grid``).

    ``trans_dicts`` is keyed by node name (suffix after
    ``translation_`` in the sheet name); a plain ``translation`` sheet
    (no suffix) is keyed under ``""`` so the single-sheet fallback in
    :class:`MetaModel` can distribute it to every node.
    """

    path: str
    scopes: list[OptionScope]
    options: dict[str, Option]
    ssn_dict: dict[str, pd.DataFrame]
    media_prices: dict[str, dict[str, float]]
    trans_dicts: dict[str, dict]

    def __repr__(self) -> str:
        parts = [f"path='{self.path}'"]
        if self.scopes:
            parts.append(f"scopes={len(self.scopes)}: {[s.scope_name for s in self.scopes]}")
        if self.options:
            parts.append(f"options={list(self.options.keys())}")
        if self.ssn_dict:
            parts.append(f"ssn={list(self.ssn_dict.keys())}")
        if self.media_prices:
            parts.append(f"media_prices={list(self.media_prices.keys())}")
        if self.trans_dicts:
            parts.append(f"trans_dicts={list(self.trans_dicts.keys())}")
        return f"ComputeContext({', '.join(parts)})"

    def __str__(self) -> str:
        lines = [f"ComputeContext: {self.path}"]
        if not self.scopes:
            lines.append("  Scopes: (none)")
        else:
            lines.append(f"  Scopes: {len(self.scopes)}")
            for s in self.scopes:
                lines.append(f"    {s.scope_name}: {s.date_start.strftime('%Y-%m-%d')} -> "
                             f"{s.date_end.strftime('%Y-%m-%d')}")
        lines.append("  Options: " + (", ".join(self.options.keys()) if self.options else "(none)"))
        lines.append("  Seasonality: " + (", ".join(self.ssn_dict.keys()) if self.ssn_dict else "(none)"))
        lines.append("  Media prices: " + (", ".join(self.media_prices.keys()) if self.media_prices else "(none)"))
        lines.append("  Translation sheets: " + (", ".join(self.trans_dicts.keys()) if self.trans_dicts else "(none)"))
        return "\n".join(lines)


def load_context(path: str) -> ComputeContext:
    """
    Load everything the reporting pipeline needs from a context XLSX file.

    Wraps ``scopes_read``, ``Option.from_excel_all``, ``seasonality_read``
    and ``media_prices_read`` into a single :class:`ComputeContext`.
    Translation sheets are discovered from the workbook: every
    ``translation_<node>`` sheet is built via ``trans_dict_create`` and
    keyed by node name, while a plain ``translation`` sheet (no suffix)
    is keyed under ``""``.

    The ``scopes`` and ``allocations`` sheets are mandatory — options are
    unusable without them. ``seasonality`` and ``media_prices`` are
    optional (empty dicts when absent).

    :param path: path to the context XLSX file
    :return: :class:`ComputeContext` with all options-related settings
    :raises FermatricaError: if the ``scopes`` or ``allocations`` sheet
        is missing
    """
    xl_file = pd.ExcelFile(path, engine="openpyxl")
    sheet_names = xl_file.sheet_names
    xl_file.close()

    if "scopes" not in sheet_names:
        fermatrica_error(f"Error reading XLSX file: {path}. Sheet 'scopes' not found")
    if "allocations" not in sheet_names:
        fermatrica_error(f"Error reading XLSX file: {path}. Sheet 'allocations' not found")

    trans_dicts: dict[str, dict] = {}
    for s in sheet_names:
        if s.startswith("translation_"):
            trans_dicts[s[len("translation_"):]] = trans_dict_create(path=path, sheet=s)
        elif s == "translation":
            trans_dicts[""] = trans_dict_create(path=path, sheet=s)

    return ComputeContext(
        path=path,
        scopes=scopes_read(path),
        options=Option.from_excel_all(path),
        ssn_dict=seasonality_read(path),
        media_prices=media_prices_read(path),
        trans_dicts=trans_dicts,
    )


def campaigns_to_options(
    dt_campaigns: pd.DataFrame | str,
    *,
    sheet: str = "campaigns",
    granularity: str = "month",
    path: str | None = None,
    template_path: str | None = None,
    apply_vars: str = "superbrand",
    plan_period: str = "exact",
    inflation_multiplier: float = 1.0,
) -> ComputeContext:
    """
    Derive scopes / allocations / seasonality from a campaign table.

    Input schema (required columns in any order): ``option``, ``item``,
    ``date_start``, ``date_end``, ``val``; optional ``target`` (entity value).
    When ``target`` is present, each scope gets ``filters={apply_vars: [target]}``
    and is named ``{option}_{target}_{year}``; without it, scopes are named
    ``{option}_{year}`` and carry no filter. ``val`` is in millions (consistent
    with the allocations sheet and ``base_price``).

    Scopes are fully derived: one per (option × target × year) with full
    calendar-year dates (``date_start={y}-01-01``, ``date_end={y}-12-31``,
    reference period = previous year), ``plan_period``, ``inflation_multiplier``
    and ``seasonality=scope_name``.

    Allocations sum campaign budgets per (option, scope, item). Items are kept
    even when the summed value is ``0.0``, so a fully zero-budget option (e.g. a
    "zero" baseline) carries explicit zeros instead of an empty per-scope dict —
    an empty dict would break PPTX export, which renders a per-tool costs table
    from the budget dict. Seasonality holds per-period budget shares summing to
    1.0 per scope-year (all periods of the scope year are emitted, ``0.0`` where
    no budget falls).

    When ``path`` is given, a ready-to-load context XLSX is written. When
    ``template_path`` is given, the template is copied first and only the
    ``scopes`` / ``allocations`` / ``seasonality`` sheets are overwritten
    (``translation_*`` and ``media_prices`` sheets are preserved); a
    ``campaigns`` sheet with the input table is added for provenance.

    :param dt_campaigns: campaign DataFrame, or path to an XLSX file with a
        ``campaigns`` sheet
    :param sheet: sheet name when ``dt_campaigns`` is a path (default
        ``'campaigns'``)
    :param granularity: 'month' (default), 'week' or 'day'
    :param path: optional output context.xlsx path
    :param template_path: optional template XLSX with ``translation_*`` sheets
        (its sheets are preserved, campaign-derived ones overwritten)
    :param apply_vars: filter column for the optional ``target`` value
        (default ``'superbrand'``)
    :param plan_period: scope planning period (default ``'exact'``)
    :param inflation_multiplier: default inflation for all scopes (default 1.0)
    :return: :class:`ComputeContext` with derived scopes, options, seasonality,
        empty media_prices and (when ``template_path`` given) trans_dicts
    :raises FermatricaError: on missing required columns, unparseable dates,
        inverted date ranges, or negative / non-numeric val
    """

    if granularity not in ("month", "week", "day"):
        fermatrica_error(
            f"campaigns_to_options: unknown granularity '{granularity}'. "
            "Use 'month', 'week' or 'day'"
        )

    if isinstance(dt_campaigns, str):
        if not os.path.exists(dt_campaigns):
            fermatrica_error(
                f"campaigns_to_options: file not found: {dt_campaigns}"
            )
        dt_campaigns = pd.read_excel(dt_campaigns, sheet_name=sheet)

    if not isinstance(dt_campaigns, pd.DataFrame):
        fermatrica_error(
            "campaigns_to_options: dt_campaigns must be a DataFrame or a path "
            "to an XLSX file with a campaigns sheet"
        )

    required_cols = {"option", "item", "date_start", "date_end", "val"}
    missing = required_cols - set(dt_campaigns.columns)
    if missing:
        fermatrica_error(
            f"campaigns table missing required columns: {missing}"
        )

    has_target = "target" in dt_campaigns.columns

    # ---- parse & validate -------------------------------------------------

    dt_campaigns = dt_campaigns.copy()
    dt_campaigns["date_start"] = pd.to_datetime(dt_campaigns["date_start"])
    dt_campaigns["date_end"] = pd.to_datetime(dt_campaigns["date_end"])
    try:
        dt_campaigns["val"] = pd.to_numeric(dt_campaigns["val"])
    except (ValueError, TypeError) as e:
        fermatrica_error(f"campaigns_to_options: val must be numeric: {e}")

    if (dt_campaigns["date_end"] < dt_campaigns["date_start"]).any():
        bad = dt_campaigns.loc[dt_campaigns["date_end"] < dt_campaigns["date_start"]]
        fermatrica_error(
            "campaigns_to_options: date_end precedes date_start for row(s): "
            + ", ".join(str(i) for i in bad.index)
        )
    if (dt_campaigns["val"] < 0).any():
        bad = dt_campaigns.loc[dt_campaigns["val"] < 0]
        fermatrica_error(
            "campaigns_to_options: negative val for row(s): "
            + ", ".join(str(i) for i in bad.index)
        )

    if has_target:
        dt_campaigns["target"] = dt_campaigns["target"].astype(object)
        dt_campaigns.loc[dt_campaigns["target"].isna(), "target"] = ""

    # ---- derive scopes / allocations / seasonality ------------------------

    scopes: dict[tuple[str, str, int], OptionScope] = {}
    alloc: dict[tuple[str, str, int, str], float] = {}
    ssn_num: dict[tuple[str, str, int, str, int], float] = {}
    scope_items: dict[tuple[str, str, int], set[str]] = {}

    def _target_val(row) -> str | None:
        if not has_target:
            return None
        tgt = str(row["target"]).strip()
        return tgt if tgt else None

    def _scope_key_for(option: str, target: str | None, year: int):
        key = (option, target, year)
        if key not in scopes:
            scopes[key] = _scope_build(
                _scope_name(option, target, year), year, apply_vars, target,
                plan_period, inflation_multiplier,
            )
        return key

    # pass 1 — calendar-year splits: scopes + per-scope-year allocation totals
    for _, row in dt_campaigns.iterrows():
        option = str(row["option"]).strip()
        item = str(row["item"]).strip()
        target = _target_val(row)
        val = float(row["val"])
        date_start = row["date_start"]
        date_end = row["date_end"]

        total_days = (date_end - date_start).days + 1

        for year in range(date_start.year, date_end.year + 1):
            y_start = pd.Timestamp(year, 1, 1)
            y_end = pd.Timestamp(year, 12, 31)
            ov_start = date_start if date_start > y_start else y_start
            ov_end = date_end if date_end < y_end else y_end
            days_year = (ov_end - ov_start).days + 1

            scope_key = _scope_key_for(option, target, year)
            scope_items.setdefault(scope_key, set()).add(item)
            alloc[(*scope_key, item)] = alloc.get((*scope_key, item), 0.0) \
                + val * days_year / total_days

    # pass 2 — per-period budget shares (only for scopes derived in pass 1)
    for _, row in dt_campaigns.iterrows():
        option = str(row["option"]).strip()
        item = str(row["item"]).strip()
        target = _target_val(row)
        val = float(row["val"])
        date_start = row["date_start"]
        date_end = row["date_end"]

        periods = date_range_to_periods(date_start, date_end, period=granularity)

        for _, pr in periods.iterrows():
            anchor = pr["date"]
            scope_key = (option, target, anchor.year)
            if scope_key not in scopes:
                continue
            ssn_num[(*scope_key, item, _period_id(anchor, granularity))] = \
                ssn_num.get((*scope_key, item, _period_id(anchor, granularity)), 0.0) \
                + val * pr["share"]

    if not scopes:
        fermatrica_error("campaigns_to_options: no scopes derived (empty campaigns table)")

    # ---- trans_dicts from template -----------------------------------------

    trans_dicts: dict[str, dict] = {}
    if template_path is not None:
        if not os.path.exists(template_path):
            fermatrica_error(
                f"campaigns_to_options: template file not found: {template_path}"
            )
        trans_dicts = _read_translation_sheets(template_path)

    _warn_unknown_items(dt_campaigns, trans_dicts)

    # ---- assemble ----------------------------------------------------------

    scope_list = sorted(scopes.values(), key=lambda s: (s.date_start, s.scope_name))

    budget_data: dict[str, dict[str, dict[str, float]]] = {}
    for (option, target, year), scope in scopes.items():
        items = {}
        for item in sorted(scope_items[(option, target, year)]):
            # Keep zero-valued items too: an option whose campaigns all have
            # val == 0 is a legitimate "no spend" baseline and must carry
            # explicit zeros (an empty per-scope dict breaks slide export,
            # which renders the per-tool costs table from this dict).
            items[item] = alloc.get((option, target, year, item), 0.0)
        budget_data.setdefault(option, {})[scope.scope_name] = items

    options = {name: Option(name=name, budget_data=data)
               for name, data in budget_data.items()}

    ssn_dict: dict[str, pd.DataFrame] = {}
    for (option, target, year), scope in scopes.items():
        period_col = _period_col(granularity)
        period_ids = _year_period_ids(year, granularity)
        items = sorted(scope_items[(option, target, year)])
        rows = []
        for pid in period_ids:
            row = {period_col: pid}
            for item in items:
                denom = alloc.get((option, target, year, item), 0.0)
                num = ssn_num.get((option, target, year, item, pid), 0.0)
                row[_bare_item(item)] = num / denom if denom > 0 else 0.0
            rows.append(row)
        ssn_dict[scope.scope_name] = pd.DataFrame(rows, columns=[period_col] + [_bare_item(i) for i in items])

    if path is not None:
        _write_context_xlsx(
            path=path,
            template_path=template_path,
            dt_campaigns=dt_campaigns,
            scopes=scope_list,
            options=options,
            ssn_dict=ssn_dict,
        )

    return ComputeContext(
        path=path,
        scopes=scope_list,
        options=options,
        ssn_dict=ssn_dict,
        media_prices={},
        trans_dicts=trans_dicts,
    )


def _scope_name(option: str, target: str | None, year: int) -> str:
    if target:
        return f"{option}_{target}_{year}"
    return f"{option}_{year}"


def _scope_build(scope_name: str, year: int, apply_vars: str,
                 target: str | None, plan_period: str,
                 inflation_multiplier: float) -> OptionScope:
    return OptionScope(
        scope_name=scope_name,
        date_start=f"{year}-01-01",
        date_end=f"{year}-12-31",
        ref_date_start=f"{year - 1}-01-01",
        ref_date_end=f"{year - 1}-12-31",
        filters={apply_vars: [target]} if target else {},
        plan_period=plan_period,
        inflation_multiplier=inflation_multiplier,
        seasonality=scope_name,
    )


def _period_id(anchor: pd.Timestamp, granularity: str) -> int:
    if granularity == "month":
        return int(anchor.month)
    if granularity == "week":
        return int(anchor.strftime("%W"))
    return int(anchor.dayofyear)


def _period_col(granularity: str) -> str:
    if granularity == "month":
        return "month"
    if granularity == "week":
        return "week_number"
    return "day"


def _year_period_ids(year: int, granularity: str) -> list[int]:
    if granularity == "month":
        return list(range(1, 13))
    if granularity == "week":
        mondays = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="W-MON")
        return [int(m.strftime("%W")) for m in mondays]
    days = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    return [int(d.dayofyear) for d in days]


def _bare_item(item: str) -> str:
    return item[4:] if item.startswith("bdg_") else item


def _read_translation_sheets(path: str) -> dict[str, dict]:
    xl_file = pd.ExcelFile(path, engine="openpyxl")
    sheet_names = xl_file.sheet_names
    xl_file.close()

    trans_dicts: dict[str, dict] = {}
    for s in sheet_names:
        if s.startswith("translation_"):
            trans_dicts[s[len("translation_"):]] = trans_dict_create(path=path, sheet=s)
        elif s == "translation":
            trans_dicts[""] = trans_dict_create(path=path, sheet=s)
    return trans_dicts


def _warn_unknown_items(dt_campaigns: pd.DataFrame, trans_dicts: dict[str, dict]) -> None:
    if not trans_dicts:
        return
    known: set[str] = set()
    for trans_dict in trans_dicts.values():
        known |= set(trans_dict.get("function", {}).keys())
    unknown = sorted({str(x).strip() for x in dt_campaigns["item"].unique()} - known)
    for item in unknown:
        fermatrica_warner.warning(
            f"campaigns_to_options: item '{item}' is not present in any "
            "translation sheet of the template — its budget will not be translated"
        )


def _write_context_xlsx(path: str,
                        template_path: str | None,
                        dt_campaigns: pd.DataFrame,
                        scopes: list[OptionScope],
                        options: dict[str, Option],
                        ssn_dict: dict[str, pd.DataFrame]) -> None:
    if template_path is not None:
        shutil.copy2(template_path, path)
        wb = load_workbook(path)
        for sheet_name in ("scopes", "allocations", "seasonality", "campaigns"):
            if sheet_name in wb.sheetnames:
                del wb[sheet_name]
    else:
        wb = Workbook()
        wb.remove(wb.active)

    def _write_df_sheet(ws, df: pd.DataFrame) -> None:
        df = df.where(pd.notna(df), None)
        for col_idx, col_name in enumerate(df.columns, 1):
            ws.cell(row=1, column=col_idx, value=col_name)
        for row_idx, row in enumerate(df.itertuples(index=False), 2):
            for col_idx, value in enumerate(row, 1):
                ws.cell(row=row_idx, column=col_idx, value=value)
        for col_idx in range(1, len(df.columns) + 1):
            max_width = max(
                len(str(df.columns[col_idx - 1])),
                max((len(str(row[col_idx - 1])) for row in df.itertuples(index=False)), default=0),
            ) + 2
            ws.column_dimensions[get_column_letter(col_idx)].width = min(max_width, 50)

    scope_rows = []
    for s in scopes:
        scope_rows.append({
            "scope_name": s.scope_name,
            "date_start": s.date_start.strftime("%Y-%m-%d"),
            "date_end": s.date_end.strftime("%Y-%m-%d"),
            "ref_date_start": s.ref_date_start.strftime("%Y-%m-%d"),
            "ref_date_end": s.ref_date_end.strftime("%Y-%m-%d"),
            "apply_vars": next(iter(s.filters), ""),
            "target": _single_filter_value(s) or "",
            "plan_period": s.plan_period,
            "inflation_multiplier": s.inflation_multiplier,
            "seasonality": s.seasonality or "",
        })
    dt_scopes = pd.DataFrame(scope_rows, columns=[
        "scope_name", "date_start", "date_end",
        "ref_date_start", "ref_date_end", "apply_vars", "target",
        "plan_period", "inflation_multiplier", "seasonality",
    ])

    alloc_rows = []
    for option in options.values():
        for scope_name, items in option.budget_data.items():
            for item in sorted(items):
                alloc_rows.append({
                    "option": option.name,
                    "scope_name": scope_name,
                    "item": item,
                    "val": items[item],
                })
    dt_alloc = pd.DataFrame(alloc_rows, columns=["option", "scope_name", "item", "val"])

    frames = []
    for s in scopes:
        dt = ssn_dict[s.scope_name].copy()
        dt.insert(0, "scope_name", s.scope_name)
        frames.append(dt)
    dt_ssn_all = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["scope_name"])

    ws_scopes = wb.create_sheet("scopes", 0)
    _write_df_sheet(ws_scopes, dt_scopes)
    ws_alloc = wb.create_sheet("allocations")
    _write_df_sheet(ws_alloc, dt_alloc)
    ws_ssn = wb.create_sheet("seasonality")
    _write_df_sheet(ws_ssn, dt_ssn_all)
    ws_cmp = wb.create_sheet("campaigns")
    _write_df_sheet(ws_cmp, dt_campaigns)

    wb.save(path)


def _single_filter_value(scope: OptionScope) -> str | None:
    for values in scope.filters.values():
        if values:
            return str(values[0])
    return None
