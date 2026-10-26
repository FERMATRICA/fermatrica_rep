"""Export threading of whole-DAG decomposition.

``options_detail(whole_graph=...)`` must thread the flag through to the
decomposition / waterfall slides of the reported node, and the deprecated
``export_option_detail`` shim must forward it.
"""

import pytest


@pytest.mark.integration
class TestWholeGraphExportPptx:
    """PPTX slide generation with whole-DAG decomposition on p10."""

    @pytest.fixture
    def p10_computed(self, p10_models_ds, samples_path):
        from tests.test_integration.conftest import _activate_adhoc_path

        _activate_adhoc_path(samples_path / "p10_sample")

        import code_py.adhoc.reporting
        import code_py.adhoc.translators
        from code_py.adhoc.reporting import pred_merge_1
        import fermatrica_rep as fmr

        category_model, category_ds, sales_model, sales_ds = p10_models_ds
        options_path = str(
            samples_path
            / "p10_sample"
            / "code_py"
            / "model_data"
            / "options"
            / "context.xlsx"
        )
        ctx = fmr.load_context(options_path)

        category = fmr.ModelNode(
            model=category_model,
            ds=category_ds,
            adhoc_code=[code_py.adhoc.translators, code_py.adhoc.reporting],
            language="english",
            static=True,
        )
        sales = fmr.ModelNode(
            model=sales_model,
            ds=sales_ds,
            adhoc_code=[code_py.adhoc.translators, code_py.adhoc.reporting],
            language="english",
        )
        category.scope_names = {
            s.scope_name for s in ctx.scopes
            if all(category.ds[col].isin(values).any() for col, values in s.filters.items())
        }
        sales.scope_names = {
            s.scope_name for s in ctx.scopes
            if all(s.filters.get(col) and sales.ds[col].isin(values).any()
                   for col, values in s.filters.items())
        }
        mm = fmr.MetaModel(
            nodes={"category": category, "sales": sales},
            edges=[("category", "sales", pred_merge_1)],
            compute_context=ctx,
        )
        mm.compute(ctx.options["predefined"], scopes=ctx.scopes, allow_past=True)
        return mm, ctx

    def test_options_detail_whole_graph_no_crash_and_default_unchanged(self, p10_computed):
        """Both whole_graph=True and the default False produce slides."""
        import fermatrica_rep as fmr

        mm, ctx = p10_computed
        kwargs = dict(
            scopes=ctx.scopes,
            options=None,
            if_fit=False,
            if_decompose=True,
            if_waterfall=True,
            if_sku_fit_predict=False,
            if_sku_decompose=False,
            if_sku_waterfall=False,
        )

        sales_rep, prs = fmr.export_pptx.set_config(
            mm.nodes["sales"].model_rep, template_name="blanc"
        )
        prs = fmr.export_pptx.options_detail(
            prs, mm, whole_graph="sales", **kwargs
        )
        assert len(prs.slides) > 0

        sales_rep2, prs2 = fmr.export_pptx.set_config(
            mm.nodes["sales"].model_rep, template_name="blanc"
        )
        prs2 = fmr.export_pptx.options_detail(
            prs2, mm, whole_graph=False, **kwargs
        )
        assert len(prs2.slides) == len(prs.slides)

    def test_options_detail_whole_graph_decomposition_split_dotted(self, p10_computed):
        """The decomposition split consumed by the slides carries dotted leaves."""
        import fermatrica_rep as fmr

        mm, _ = p10_computed
        split = fmr.extract_effect(mm, whole_graph="sales")
        variables = split["variable"].unique()
        assert not any(v == "Category" for v in variables)
        assert any(isinstance(v, str) and v.startswith("Category.") for v in variables)

    def test_deprecated_shim_forwards_whole_graph(self, p10_computed, monkeypatch):
        """export_option_detail forwards whole_graph to options_detail."""
        import fermatrica_rep.export.export_pptx as ep

        mm, _ = p10_computed
        sales_node = mm.nodes["sales"]

        captured = {}

        def fake_options_detail(prs, meta_model, scopes, options=None, **kwargs):
            captured["whole_graph"] = kwargs.get("whole_graph")
            captured["meta_model"] = meta_model
            return prs

        monkeypatch.setattr(ep, "options_detail", fake_options_detail)

        import warnings
        from fermatrica_rep.options.define import OptionSettings

        opt_set = OptionSettings(
            target="brand_1",
            date_start="2026-01-01",
            date_end="2026-12-31",
            ref_date_start="2025-01-01",
            ref_date_end="2025-12-31",
            apply_vars=["superbrand"],
            plan_period="year",
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ep.export_option_detail(
                None,
                model=sales_node.model,
                model_rep=sales_node.model_rep,
                ds=sales_node.ds,
                dt_pred=sales_node.dt_pred,
                opt_set=opt_set,
                options_m=None,
                whole_graph="sales",
            )

        assert captured["whole_graph"] == "sales"
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)
