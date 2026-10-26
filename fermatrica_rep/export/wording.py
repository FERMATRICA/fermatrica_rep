"""KPI/currency-aware wording resolution for PPTX export.

Facts live on ``model.conf`` only; ModelRep stays a presentation dictionary.
All dictionary lookups that may depend on KPI or currency should go through
:func:`text` / :func:`units_label` so ``vis_dict`` overrides and the ``***cur``
token resolve consistently.

Usage::

    wording.text(model_rep, model.conf, section="option_compare", variable="costs_title")

Resolution precedence: for a non-sales KPI the KPI-specific override section
``f"{section}.{kpi_name}"`` (e.g. ``option.detail.sales.brand_awareness``) is
used when it carries the requested row, otherwise the base ``section`` row is
used. A ``***cur`` token in the resolved text is substituted with the currency
label looked up in ``shared.<currency>_short``, falling back to the ISO currency
code when the row is absent.

Error contract: a missing dictionary row raises ``FermatricaError`` via
:func:`fermatrica_error`; units resolution first falls back to the neutral
``shared.units`` row.
"""

from fermatrica import fermatrica_error

DEFAULT_KPI = "sales"
DEFAULT_CURRENCY = "RUB"
DEFAULT_IF_VALUE = True
DEFAULT_UNITS = "packs"

_TRUE_STR = {"1", "true", "yes", "on", "y"}


def facts(model_conf=None) -> dict:
    """Canonical reporting facts from a model conf (with defaults).

    ``None`` conf (ModelRep-only call sites) yields pure defaults.
    """
    if model_conf is None:
        model_conf = object()
    kpi = getattr(model_conf, "kpi_name", None) or DEFAULT_KPI
    currency = getattr(model_conf, "currency", None) or DEFAULT_CURRENCY
    if_value = getattr(model_conf, "if_value_applicable", None)
    units = getattr(model_conf, "display_units", None) or DEFAULT_UNITS
    return {
        "kpi_name": str(kpi).strip().lower() or DEFAULT_KPI,
        "currency": str(currency).strip().upper() or DEFAULT_CURRENCY,
        "if_value_applicable": _as_bool(if_value, DEFAULT_IF_VALUE),
        "units": str(units).strip() or DEFAULT_UNITS,
    }


def _as_bool(value, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in _TRUE_STR


def _lookup(vis_dict, section: str, variable: str, language: str) -> str | None:
    sub = vis_dict.loc[(vis_dict["section"] == section) & (vis_dict["variable"] == variable)]
    if sub.empty:
        return None
    return sub[language].iloc[0]


def text(model_rep, model_conf=None, *, section: str, variable: str, language: str | None = None) -> str:
    """Localized dictionary text with KPI-section override and ``***cur`` substitution.

    ``model_conf=None`` (no model available) uses sales/RUB defaults.
    """
    lang = language or model_rep.language
    f = facts(model_conf)
    base = _lookup(model_rep.vis_dict, section, variable, lang)
    if f["kpi_name"] != DEFAULT_KPI:
        override = _lookup(model_rep.vis_dict, f"{section}.{f['kpi_name']}", variable, lang)
        if override is not None:
            base = override
    if base is None:
        fermatrica_error(f"vis_dict: missing row section='{section}', variable='{variable}'")
    if "***cur" in base:
        base = base.replace("***cur", currency_label(model_rep, f["currency"], lang))
    return base


def currency_label(model_rep, currency: str, language: str | None = None) -> str:
    lang = language or model_rep.language
    label = _lookup(model_rep.vis_dict, "shared", f"{currency.lower()}_short", lang)
    return currency if label is None else label


def units_label(model_rep, model_conf=None, language: str | None = None) -> str:
    """Localized units label; unknown units fall back to the neutral ``units`` row."""
    lang = language or model_rep.language
    f = facts(model_conf)
    label = _lookup(model_rep.vis_dict, "shared", f["units"], lang)
    if label is None:
        label = _lookup(model_rep.vis_dict, "shared", "units", lang)
    if label is None:
        fermatrica_error(f"vis_dict: missing units row for '{f['units']}'")
    return label
