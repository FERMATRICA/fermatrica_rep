# tests/test_export_wording.py
import copy
import types

import pytest

from fermatrica_rep.export import wording


def _conf(**kw):
    base = dict(kpi_name="sales", currency="RUB", if_value_applicable=True, display_units="packs")
    base.update(kw)
    return types.SimpleNamespace(**base)


def _tokenized_costs(model_rep, text_value):
    """Inject a '***cur' token into option_compare/costs_title for a test."""
    mr = copy.copy(model_rep)
    df = model_rep.vis_dict.copy()
    mask = (df["section"] == "option_compare") & (df["variable"] == "costs_title")
    df.loc[mask, ["english", "russian"]] = [text_value, text_value]
    mr.vis_dict = df
    return mr


class TestFacts:

    def test_defaults(self):
        assert wording.facts(_conf()) == {"kpi_name": "sales", "currency": "RUB",
                                          "if_value_applicable": True, "units": "packs"}

    def test_missing_attrs_default(self):
        f = wording.facts(types.SimpleNamespace())
        assert f["kpi_name"] == "sales" and f["currency"] == "RUB"
        assert f["if_value_applicable"] is True and f["units"] == "packs"

    def test_none_conf_defaults(self):
        f = wording.facts(None)
        assert f == {"kpi_name": "sales", "currency": "RUB",
                     "if_value_applicable": True, "units": "packs"}

    def test_string_bool_parsed(self):
        assert wording.facts(_conf(if_value_applicable="False"))["if_value_applicable"] is False


class TestText:

    def test_sales_english_title_default(self, basic_model_rep):
        out = wording.text(basic_model_rep, _conf(),
                           section="option.detail.sales", variable="title",
                           language="english")
        assert "expected sales" in out
        assert out.endswith("RUB")

    def test_currency_token_replaced(self, basic_model_rep):
        mr = _tokenized_costs(basic_model_rep, "Costs, Mln. ***cur")
        out = wording.text(mr, _conf(currency="EUR"), section="option_compare",
                           variable="costs_title", language="english")
        assert "Mln. EUR" in out

    def test_unknown_currency_falls_back_to_iso(self, basic_model_rep):
        mr = _tokenized_costs(basic_model_rep, "Costs, Mln. ***cur")
        out = wording.text(mr, _conf(currency="GBP"), section="option_compare",
                           variable="costs_title", language="english")
        assert out.endswith("GBP")

    def test_kpi_override_section(self, basic_model_rep):
        out = wording.text(basic_model_rep, _conf(kpi_name="brand_awareness"),
                           section="option.detail.sales", variable="title", language="english")
        assert "Brand Awareness" in out

    def test_unknown_kpi_falls_back_to_sales(self, basic_model_rep):
        out = wording.text(basic_model_rep, _conf(kpi_name="unknown_kpi"),
                           section="option.detail.sales", variable="title", language="english")
        assert "sales" in out.lower()


class TestTokenizedTitleRows:

    def test_tokenized_titles_keep_russian_text(self, basic_model_rep):
        df = basic_model_rep.vis_dict
        rows = [
            ("predict_table", "title"),
            ("option.detail.sales", "title"),
            ("efficiency_short", "axis_x"),
            ("efficiency_short", "incr_sales_axis_y"),
            ("efficiency_long", "axis_x"),
            ("efficiency_long", "incr_sales_axis_y"),
            ("efficiency_long", "incr_sales_cum_axis_y"),
        ]
        for section, variable in rows:
            sub = df.loc[(df["section"] == section) & (df["variable"] == variable)].iloc[0]
            assert "***cur" in sub["english"], f"{section}/{variable} english"
            assert sub["russian"] != sub["english"], f"{section}/{variable} differs"
            assert any("а" <= ch.lower() <= "я" for ch in sub["russian"]), f"{section}/{variable} cyrillic"
            assert "***cur" in sub["russian"], f"{section}/{variable} russian token"


class TestUnitsLabel:

    def test_pct(self, basic_model_rep):
        assert wording.units_label(basic_model_rep, _conf(display_units="pct"),
                                   language="english") == "%"

    def test_unknown_units_neutral_fallback(self, basic_model_rep):
        assert wording.units_label(basic_model_rep, _conf(display_units="nope"),
                                   language="english") == "Units"

    def test_unknown_units_neutral_fallback_russian(self, basic_model_rep):
        basic_model_rep.language = "russian"
        assert wording.units_label(basic_model_rep, _conf(display_units="nope")) == "ед."


class TestCurrencyLabel:

    def test_known_currency_row_russian(self, basic_model_rep):
        assert wording.currency_label(basic_model_rep, "RUB", language="russian") == "руб."


class TestMeanCurveAxisWording:

    def test_mean_long_axis_russian_uses_generic_average(self, basic_model_rep):
        # Mean-aggregated KPIs are per-period averages, not annual ones, so the
        # long-term axis must avoid the year-specific "Среднегодов" wording.
        title = wording.text(basic_model_rep, _conf(kpi_name="brand_awareness",
                                                    if_value_applicable=False),
                             section="efficiency_long", variable="incr_sales_axis_y",
                             language="russian")
        assert "Средн" in title
        assert "Среднегодов" not in title

    def test_mean_long_profit_axis_russian_uses_generic_average(self, basic_model_rep):
        title = wording.text(basic_model_rep, _conf(),
                             section="efficiency_long", variable="profit_axis_y",
                             language="russian")
        assert "Средн" in title
        assert "Среднегодов" not in title
