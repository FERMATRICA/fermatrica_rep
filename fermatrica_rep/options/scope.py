"""
OptionScope — named application scope for budget scenario calculation.

An OptionScope answers the question "where and when should this budget be
applied?" It encapsulates the target date range, the entity filter (which
brands, markets, or segments), a reference period for growth computation,
planning period resolution, and an inflation multiplier for media prices.

Scopes are **atomic** — they do not inherit from or compose with each other.
Multiple scopes can target different subsets of the same dataset (e.g.
different brands or regions) and compute independently.
"""


import datetime
import pandas as pd

from fermatrica_utils import StableClass
from fermatrica.basics.basics import fermatrica_error


class OptionScope(StableClass):
    """
    Named scope defining **where** and **when** to apply a budget scenario.

    Each scope specifies:

    - A target date window (``date_start`` -> ``date_end``) during which
      budgets are translated into media variables.
    - An entity filter (``filters``) — a dict mapping column names to lists
      of allowed values. Only rows matching all filter criteria are affected.
      For example ``{"superbrand": ["BrandX"], "region": ["Moscow"]}``.
    - A reference period (``ref_date_start`` -> ``ref_date_end``) for
      computing growth vs prior year.
    - A planning period resolution (``plan_period``: ``"year"``, ``"exact"``,
      ``"hy"``, or ``"quarter"``) controlling how summaries are aggregated.
    - An inflation multiplier (``inflation_multiplier``) applied to all
      media prices for this scope, overridable per-tool via
      ``media_prices``.
    - An optional seasonality key (``seasonality``) referencing an entry in
      the ModelRep seasonality dictionary. Falls back to ``scope_name``
      when not set.

    Scopes extend ``StableClass`` — all attributes must be set during
    ``__init__``; no new attributes can be added afterward.

    Parameters
    ----------
    scope_name : str
        Unique name for this scope. Referenced by budget rows in the
        allocations sheet and matched during ``Option.iter_scopes()``.
    date_start : str or datetime.date
        First date of the forecast period (inclusive). Parsed via
        ``pd.to_datetime``.
    date_end : str or datetime.date
        Last date of the forecast period (inclusive). Parsed via
        ``pd.to_datetime``.
    ref_date_start : str or datetime.date
        First date of the reference period (inclusive). Used to
        calculate growth (forecast / reference).
    ref_date_end : str or datetime.date
        Last date of the reference period (inclusive). If read from
        the scopes sheet, defaults to ``date_start - 1 day`` when
        omitted.
    filters : dict[str, list[str]], optional
        Entity filter mapping column names to allowed values. Rows not
        matching are excluded from budget translation and summaries.
        Defaults to ``{}`` (all rows match).
    plan_period : str, optional
        Summary aggregation granularity: ``"exact"``, ``"year"``
        (default), ``"hy"``, or ``"quarter"``.
    inflation_multiplier : float, optional
        Default multiplier applied to all media prices for this scope.
        Individual tools can override this via ``media_prices``.
        Defaults to ``1.0``.
    seasonality : str or None, optional
        Key into the seasonality dictionary in ``ModelRep.ssn_dict``.
        When ``None``, ``scope_name`` is used as the lookup key.
        Defaults to ``None``.
    """

    scope_name: str
    date_start: pd.Timestamp
    date_end: pd.Timestamp
    ref_date_start: pd.Timestamp
    ref_date_end: pd.Timestamp
    filters: dict[str, list[str]]
    plan_period: str
    inflation_multiplier: float
    seasonality: str | None

    def __init__(
        self,
        scope_name: str,
        date_start: str | datetime.date,
        date_end: str | datetime.date,
        ref_date_start: str | datetime.date,
        ref_date_end: str | datetime.date,
        filters: dict[str, list[str]] | None = None,
        plan_period: str = "year",
        inflation_multiplier: float = 1.0,
        seasonality: str | None = None,
    ):
        if filters is None:
            filters = {}

        if not scope_name:
            fermatrica_error("OptionScope requires a non-empty scope_name")

        self.scope_name = scope_name
        self.date_start = pd.to_datetime(date_start)
        self.date_end = pd.to_datetime(date_end)
        self.ref_date_start = pd.to_datetime(ref_date_start)
        self.ref_date_end = pd.to_datetime(ref_date_end)
        self.filters = dict(filters)
        self.plan_period = plan_period
        self.inflation_multiplier = float(inflation_multiplier)
        self.seasonality = seasonality

    def __repr__(self) -> str:
        parts = [
            f"'{self.scope_name}'",
            f"{self.date_start.strftime('%Y-%m-%d')} -> {self.date_end.strftime('%Y-%m-%d')}",
        ]
        if self.ref_date_start != self.date_start or self.ref_date_end != self.date_end:
            parts.append(f"ref={self.ref_date_start.strftime('%Y-%m-%d')} -> {self.ref_date_end.strftime('%Y-%m-%d')}")
        if self.filters:
            parts.append(f"filters={self.filters}")
        if self.plan_period != "year":
            parts.append(f"plan={self.plan_period}")
        if self.inflation_multiplier != 1.0:
            parts.append(f"infl={self.inflation_multiplier}")
        if self.seasonality:
            parts.append(f"ssn={self.seasonality}")
        return f"OptionScope({', '.join(parts)})"

    def __str__(self) -> str:
        lines = [f"OptionScope: {self.scope_name}"]
        lines.append(f"  Period:  {self.date_start.strftime('%Y-%m-%d')} -> {self.date_end.strftime('%Y-%m-%d')}")
        lines.append(f"  Ref:     {self.ref_date_start.strftime('%Y-%m-%d')} -> {self.ref_date_end.strftime('%Y-%m-%d')}")
        if self.filters:
            lines.append(f"  Filters: {self.filters}")
        lines.append(f"  Plan:    {self.plan_period}")
        lines.append(f"  Infl:    {self.inflation_multiplier}")
        if self.seasonality:
            lines.append(f"  Season:  {self.seasonality}")
        return "\n".join(lines)

    def build_mask(self, ds: pd.DataFrame, allow_past: bool = False) -> pd.Series:
        """
        Build a boolean mask selecting rows that fall within this scope.

        Combines three conditions:
        1. Date range: ``ds["date"]`` is between ``date_start`` and
           ``date_end`` (inclusive).
        2. Entity filter: for each ``(col, values)`` in ``self.filters``,
           ``ds[col]`` must be in the allowed ``values`` list.
        3. Future-only: ``ds["listed"] == 4``, unless ``allow_past`` is
           ``True``, in which case historical rows (listed <= 3) are also
           included. Budget translation typically uses ``allow_past=False``
           to avoid overwriting observed data.

        Parameters
        ----------
        ds : pd.DataFrame
            Dataset with ``"date"``, ``"listed"`` columns and any columns
            referenced in ``self.filters``.
        allow_past : bool, optional
            If ``True``, include historical rows (listed 2, 3). Default
            ``False`` limits the mask to future rows (listed 4).

        Returns
        -------
        pd.Series
            Boolean mask with the same index as ``ds``. ``True`` for rows
            that belong to this scope.
        """
        mask = (
            (ds["date"] >= self.date_start)
            & (ds["date"] <= self.date_end)
        )
        for col, values in self.filters.items():
            if col in ds.columns:
                mask = mask & ds[col].isin(values)
        if not allow_past:
            mask = mask & ds["listed"].isin([4])
        return mask

    def build_entity_mask(self, ds: pd.DataFrame) -> pd.Series:
        """
        Build a boolean mask matching only the entity filter, without any
        date range restriction.

        Used in summary computation to slice the prediction DataFrame by
        entity (e.g. a specific brand or region) before applying date-based
        masks for exact, long, and reference periods.

        Parameters
        ----------
        ds : pd.DataFrame
            Dataset with columns referenced in ``self.filters``.

        Returns
        -------
        pd.Series
            Boolean mask with the same index as ``ds``. ``True`` for rows
            matching all entity filter criteria. If no filters are defined,
            returns all-``True``.
        """
        mask = pd.Series(True, index=ds.index)
        for col, values in self.filters.items():
            if col in ds.columns:
                mask = mask & ds[col].isin(values)
        return mask

    @property
    def start_year(self) -> int:
        """The scope's start year (``date_start.year``). Replaces the
        ``s.date_start.year == Y`` and ``s.date_start.year >= Y`` idioms."""
        return self.date_start.year

    def has_filter(self, col: str) -> bool:
        """True when the scope's entity filter includes ``col``."""
        return col in self.filters

    def filter(self, col: str, value: str) -> bool:
        """True when the scope's filter on ``col`` has ``value`` as its
        first (only) value.

        False when the scope has no filter on ``col``, on empty filter
        lists, and when the first value differs. Single-value assumption:
        compares ``vals[0]``, not the whole list (``== [value]``).
        Replaces ``list(s.filters['master'])[0] == rgn`` and the
        ``('region' in s.filters and s.filters['region'] == [rgn])``
        idiom.
        """
        values = self.filters.get(col)
        if not values:
            return False
        return values[0] == value

    def match(self, col: str, value: str) -> bool:
        """True when the scope's entity mask includes ``value`` on ``col``:
        either via a filter on ``col`` whose first value is ``value``, or
        when the scope has no filter on ``col`` at all (a national scope
        covers every value).

        Replaces the ``s.filter(col, value) or not s.has_filter(col)``
        OR-composite idiom.
        """
        return self.filter(col, value) or not self.has_filter(col)

    def copy(self, deep: bool = True) -> "OptionScope":
        """New independent scope (``deep=True`` copies the filters dict;
        ``deep=False`` shares it). Original untouched."""
        if deep:
            filters = {k: list(v) for k, v in self.filters.items()}
        else:
            filters = dict(self.filters)
        return OptionScope(
            scope_name=self.scope_name,
            date_start=self.date_start,
            date_end=self.date_end,
            ref_date_start=self.ref_date_start,
            ref_date_end=self.ref_date_end,
            filters=filters,
            plan_period=self.plan_period,
            inflation_multiplier=self.inflation_multiplier,
            seasonality=self.seasonality,
        )

    def replace_dates(self, date_start=None, date_end=None,
                      ref_date_start=None, ref_date_end=None) -> "OptionScope":
        """New scope with the given dates overridden; original untouched.
        Only provided dates change (matches the sample retro idiom
        exactly)."""
        new = self.copy()
        if date_start is not None:
            new.date_start = pd.to_datetime(date_start)
        if date_end is not None:
            new.date_end = pd.to_datetime(date_end)
        if ref_date_start is not None:
            new.ref_date_start = pd.to_datetime(ref_date_start)
        if ref_date_end is not None:
            new.ref_date_end = pd.to_datetime(ref_date_end)
        return new

    def shift_years(self, delta: int = 1) -> "OptionScope":
        """New scope with all four dates rolled by ``delta`` calendar
        years (``pd.DateOffset(years=delta)``; safe on Feb 29).
        ``scope_name`` unchanged."""
        new = self.copy()
        new.date_start = self.date_start + pd.DateOffset(years=delta)
        new.date_end = self.date_end + pd.DateOffset(years=delta)
        new.ref_date_start = self.ref_date_start + pd.DateOffset(years=delta)
        new.ref_date_end = self.ref_date_end + pd.DateOffset(years=delta)
        return new


def _as_value_set(value):
    """Normalize a filter value (scalar or iterable) to a set for comparison."""
    if isinstance(value, (list, tuple, set)):
        return set(value)
    return {value}


def _filters_strict_subset(inner: dict, outer: dict) -> bool:
    """True when filter dict ``outer`` strictly contains ``inner``.

    Every ``(key, value)`` pair of ``inner`` is present in ``outer`` and
    ``outer`` has at least one extra key. Values may be scalars or
    lists/tuples — both are compared as value sets (``x`` == ``[x]``).
    An empty ``inner`` is strictly contained by any non-empty ``outer``;
    nothing is a strict subset of ``{}``.

    Shared by the background-option resolution in ``options/parallel`` and
    the entity parent-scope derivation in ``export/export_summary`` so the
    two strict-subset decisions can never drift apart.
    """
    if not inner:
        return bool(outer)
    if len(outer) <= len(inner):
        return False
    for key, values in inner.items():
        if key not in outer:
            return False
        if _as_value_set(values) != _as_value_set(outer[key]):
            return False
    return True


def find_scope(scopes: list[OptionScope] | None, scope_name: str) -> OptionScope | None:
    """First scope in ``scopes`` whose ``scope_name`` matches, else ``None``."""
    if scopes is None:
        return None
    for s in scopes:
        if s.scope_name == scope_name:
            return s
    return None


def filter_scopes(
    scopes: list["OptionScope"] | None,
    *,
    scope_name: str | list[str] | None = None,   # exact match; list = any-of (p07 dual-scope)
    start_year: int | None = None,               # s.start_year == y
    start_year_min: int | None = None,           # s.start_year >= y  (plan/retro split)
    start_year_max: int | None = None,           # s.start_year <= y
    has_filter: str | None = None,               # col in s.filters
    without_filter: str | None = None,           # col not in s.filters  (national scopes)
    match: tuple[str, str] | None = None,        # (col, value) — s.match(col, value) (national-or-explicit)
) -> list["OptionScope"]:
    """Filter a list of scopes by AND-composed criteria; input order preserved.
    Returns a new list; the input is never mutated. Only provided criteria apply.
    None/absent scopes → []."""
    if scopes is None:
        return []
    names = None
    if scope_name is not None:
        names = {scope_name} if isinstance(scope_name, str) else set(scope_name)
    result = []
    for s in scopes:
        if names is not None and s.scope_name not in names:
            continue
        if start_year is not None and s.start_year != start_year:
            continue
        if start_year_min is not None and s.start_year < start_year_min:
            continue
        if start_year_max is not None and s.start_year > start_year_max:
            continue
        if has_filter is not None and not s.has_filter(has_filter):
            continue
        if without_filter is not None and s.has_filter(without_filter):
            continue
        if match is not None and not s.match(*match):
            continue
        result.append(s)
    return result
