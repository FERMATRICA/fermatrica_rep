"""Integration: injected ABA conf facts on the real p03 sample drive export wording."""
import pytest

import fermatrica_rep as fmr
from fermatrica_rep.export import wording


@pytest.fixture(scope="module")
def aba_node(p03_model_ds):
    model, ds = p03_model_ds
    conf = model.conf
    saved = {k: getattr(conf, k) for k in ("kpi_name", "currency",
                                           "if_value_applicable", "display_units")}
    try:
        conf.kpi_name = "brand_awareness"
        conf.currency = "RUB"
        conf.if_value_applicable = False
        conf.display_units = "pct"
        node = fmr.ModelNode(model=model, ds=ds, language="english")
        node.compute()  # no-arg refresh: fills node.dt_pred for the fit-title probe
        yield node
    finally:
        for k, v in saved.items():
            setattr(conf, k, v)


def _fit_data(node, opt_set=None):
    return fmr.export.slides.fit_predict._prepare_fit_data(
        model=node.model, model_rep=node.model_rep,
        dt_pred=node.dt_pred,
        opt_set=opt_set, option_name="aba", option=None,
        plot_type="brand")


class TestAbaWordingIntegration:

    def test_facts_and_units(self, aba_node):
        f = wording.facts(aba_node.model.conf)
        assert f["kpi_name"] == "brand_awareness"
        assert f["if_value_applicable"] is False
        assert wording.units_label(aba_node.model_rep, aba_node.model.conf,
                                   language="english") == "%"

    def test_fit_predict_title(self, aba_node):
        title = _fit_data(aba_node)["slide_title"]
        assert "Brand Awareness" in title
        assert "sales" not in title.lower()

    def test_fit_predict_title_kpi_not_zero(self, aba_node):
        # Scope year present in node.dt_pred so the KPI branch runs; 2023 is a
        # historical year. The assertions are tied to the resolved value, so a
        # regenerated sample cannot make them flaky on correct formatting.
        scope = fmr.OptionScope(
            scope_name="aba_2023",
            date_start="2023-01-01", date_end="2023-12-31",
            ref_date_start="2022-01-01", ref_date_end="2022-12-31",
            filters={"superbrand": [aba_node.model.conf.target_superbrand]},
        )
        result = _fit_data(aba_node, opt_set=scope)
        title = result["slide_title"]
        vol_kpi = result["vol_kpi"]
        assert vol_kpi != "0"
        assert f"{vol_kpi} %" in title
        assert ": 0 %" not in title

    def test_value_axis_currency_label(self, aba_node):
        assert _fit_data(aba_node)["axis_title_val"] == "RUB"
        assert wording.currency_label(aba_node.model_rep, "RUB",
                                      language="russian") == "руб."
