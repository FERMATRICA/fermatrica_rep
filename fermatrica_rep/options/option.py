"""
Option — named budget scenario that aggregates budget rows from an
allocations sheet or a programmatic dict.

An Option specifies *what* budget to spend (e.g. "50M TV, 30M digital"),
while an :class:`OptionScope` specifies *where*/*when* to apply it.
Multiple Options can be computed against the same scopes for scenario
comparison.
"""


import pandas as pd

from fermatrica_utils import StableClass
from fermatrica.basics.basics import fermatrica_error


class Option(StableClass):
    """
    Named budget scenario aggregating budget rows keyed by scope.

    Budget data is stored as ``{scope_name: {item: value}}`` — a two-level
    dict where the outer key is a scope name (matching
    :attr:`OptionScope.scope_name`) and the inner dict maps budget items
    (e.g. ``"bdg_tv"``) to float values in millions.

    Construction is flexible:
    - Direct: ``Option(name="opt1", budget_data={"scope_a": {"bdg_tv": 50}})``
    - From Excel (single): ``Option.from_excel("options.xlsx", name="opt1")`` reads
      the ``allocations`` sheet and returns one option by name.
    - From Excel (all): ``Option.from_excel_all("options.xlsx")`` reads
      the ``allocations`` sheet once and returns ``{name: Option}`` for all options.

    The special item ``"text_flags"`` is handled separately — its value is
    stored as a string in :attr:`text_flags` rather than as a budget item.
    It is consumed by the deprecated ``option_report`` path to select
    a seasonality profile (``name_<key>``); the live path reads
    :attr:`OptionScope.seasonality` instead.

    Extends ``StableClass`` — all attributes must be set during ``__init__``.

    Parameters
    ----------
    name : str
        Unique name for this option. Referenced in the ``option`` column
        of the allocations sheet.
    budget_data : dict, optional
        Nested dict ``{scope_name: {item: value}}``. Each ``value`` is
        a budget amount in millions. Defaults to ``{}``.
    default_scope : str or None, optional
        Fallback scope name used when an allocations row has no (or empty)
        ``scope_name``. Only relevant when reading from Excel.
    text_flags : str or None, optional
        Space-separated flags parsed from the ``"text_flags"`` row in the
        allocations sheet. Only used by the deprecated ``option_report``
        path (e.g. ``"name_2025"`` to select a seasonality profile); the
        live path reads :attr:`OptionScope.seasonality`. Defaults to
        ``None``.
    """

    name: str
    default_scope: str | None
    budget_data: dict[str, dict[str, float]]
    text_flags: str | None

    def __init__(
        self,
        name: str,
        budget_data: dict | None = None,
        default_scope: str | None = None,
        text_flags: str | None = None,
    ):
        if not name:
            fermatrica_error("Option requires a non-empty name")

        self.name = name
        self.default_scope = default_scope
        self.budget_data = budget_data if budget_data is not None else {}
        self.text_flags = text_flags

    def __repr__(self) -> str:
        parts = [f"'{self.name}'"]
        if self.text_flags:
            parts.append(f"flags='{self.text_flags}'")
        if self.default_scope:
            parts.append(f"default_scope='{self.default_scope}'")
        for scope_name, items in self.budget_data.items():
            items_str = ", ".join(f"{k}: {v}" for k, v in items.items())
            parts.append(f"{scope_name}={{{items_str}}}")
        return f"Option({', '.join(parts)})"

    def __str__(self) -> str:
        lines = [f"Option: {self.name}"]
        if self.text_flags:
            lines.append(f"  Flags: {self.text_flags}")
        if self.default_scope:
            lines.append(f"  Default scope: {self.default_scope}")
        if not self.budget_data:
            lines.append("  (no budget data)")
        else:
            for scope_name, items in self.budget_data.items():
                lines.append(f"  Scope {scope_name}:")
                for k, v in items.items():
                    lines.append(f"    {k}:  {v:,.2f}")
        return "\n".join(lines)

    @classmethod
    def _parse_rows(
        cls,
        dt_budget: pd.DataFrame,
        name: str,
        default_scope: str | None = None,
    ):
        budget_data: dict[str, dict[str, float]] = {}
        text_flags: str | None = None
        for _, row in dt_budget.iterrows():
            item = str(row["item"]).strip()
            if item == "text_flags":
                text_flags = str(row["val"]).strip()
                continue
            scope = row.get("scope_name")
            if pd.isna(scope) or not str(scope).strip():
                scope = default_scope
            scope = str(scope)
            val = float(row["val"])
            budget_data.setdefault(scope, {})[item] = val

        return cls(name=name, budget_data=budget_data, default_scope=default_scope, text_flags=text_flags)

    @classmethod
    def from_excel(
        cls,
        path: str,
        name: str,
        sheet: str = "allocations",
        default_scope: str | None = None,
    ):
        xl_file = pd.ExcelFile(path, engine="openpyxl")

        if sheet not in xl_file.sheet_names:
            fermatrica_error(
                f"Error reading XLSX file: {path}. Sheet {sheet} not found"
            )

        dt_budget = xl_file.parse(sheet_name=sheet)
        xl_file.close()

        required_cols = {"option", "scope_name", "item", "val"}
        missing = required_cols - set(dt_budget.columns)
        if missing:
            fermatrica_error(
                f"Allocations sheet missing required columns: {missing}"
            )

        dt_budget = dt_budget.loc[dt_budget["option"] == name, :].copy()

        if dt_budget.empty:
            fermatrica_error(
                f"No budget rows found for option '{name}' in {path}"
            )

        return cls._parse_rows(dt_budget, name, default_scope)

    @classmethod
    def from_excel_all(
        cls,
        path: str,
        sheet: str = "allocations",
        default_scope: str | None = None,
    ) -> dict[str, "Option"]:
        xl_file = pd.ExcelFile(path, engine="openpyxl")

        if sheet not in xl_file.sheet_names:
            fermatrica_error(
                f"Error reading XLSX file: {path}. Sheet {sheet} not found"
            )

        dt_budget = xl_file.parse(sheet_name=sheet)
        xl_file.close()

        required_cols = {"option", "scope_name", "item", "val"}
        missing = required_cols - set(dt_budget.columns)
        if missing:
            fermatrica_error(
                f"Allocations sheet missing required columns: {missing}"
            )

        if dt_budget.empty:
            fermatrica_error(
                f"No budget rows found in {path}"
            )

        options: dict[str, "Option"] = {}
        for name, group in dt_budget.groupby("option", sort=False):
            options[str(name)] = cls._parse_rows(group, str(name), default_scope)

        return options

    def budget_for_scope(self, scope_name: str) -> dict[str, float]:
        """
        Return the budget items dict for a specific scope.

        Parameters
        ----------
        scope_name : str
            Scope name to look up in ``budget_data``.

        Returns
        -------
        dict[str, float]
            Budget items for the scope, e.g.
            ``{"bdg_tv": 50, "bdg_digital": 30}``.
            Returns an empty dict if the scope is not found.
        """
        return self.budget_data.get(scope_name, {})

    def all_scopes(self) -> list[str]:
        """
        Return all scope names present in this option.

        Returns
        -------
        list[str]
            Scope name strings in insertion order (Python 3.7+ dict order).
        """
        return list(self.budget_data.keys())

    def iter_scopes(self):
        """
        Iterate over ``(scope_name, budget_items)`` tuples.

        This is the primary interface used by ``ModelNode.translate`` and
        ``MetaModel.compute`` to enumerate scopes during translation and
        compute phases.

        Yields
        ------
        scope_name : str
            Scope name string matching :attr:`OptionScope.scope_name`.
        budget_items : dict[str, float]
            Budget items for the scope, e.g. ``{"bdg_tv": 50}``.
        """
        for scope_name, items in self.budget_data.items():
            yield scope_name, items

    def copy(self, deep: bool = True) -> "Option":
        """New Option with the same ``name``/``default_scope``/
        ``text_flags`` and an independent ``budget_data`` (inner dicts
        copied; no shared references). Budget values are floats, so the
        ``deep`` flag is accepted for symmetry with
        :meth:`OptionScope.copy`."""
        budget_data = {k: dict(v) for k, v in self.budget_data.items()}
        return Option(
            name=self.name,
            budget_data=budget_data,
            default_scope=self.default_scope,
            text_flags=self.text_flags,
        )

    def replace_budget(self, scope_name: str, budget_items: dict[str, float]) -> "Option":
        """New Option with ``scope_name``'s budget items replaced by a
        copy of ``budget_items``; all other scopes' inner dicts are
        copied (no aliasing). Preserves ``name``/``default_scope``/
        ``text_flags``. Original untouched."""
        new = self.copy()
        new.budget_data[scope_name] = dict(budget_items)
        return new
