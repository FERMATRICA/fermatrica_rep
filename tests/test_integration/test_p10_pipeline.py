"""Integration tests for the p10 static-node reporting DAG (category static -> sales).

p10 is a copy of p07 with the Category node marked static: its ``translate()``
is a no-op, no ``trans_dict`` is required, and its output is option-invariant
while the Sales node still translates budgets and varies across options.
"""

import numpy as np
import pytest


@pytest.mark.integration
class TestP10StaticNode:
    """Exercise the static-node feature on a real DAG (p10 sample)."""

    @pytest.fixture
    def p10_setup(self, p10_models_ds, samples_path):
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

        return fmr.MetaModel(
            nodes={"category": category, "sales": sales},
            edges=[("category", "sales", pred_merge_1)],
            compute_context=ctx,
        ), ctx.scopes, ctx.options, category, sales

    def test_category_is_static_and_exempt_from_trans_dict(self, p10_setup):
        mm, scopes, options, category, sales = p10_setup

        assert category.static is True
        assert category.model_rep.trans_dict is None
        assert sales.static is False
        assert sales.model_rep.trans_dict is not None

    def test_category_output_option_invariant(self, p10_setup):
        mm, scopes, options, category, sales = p10_setup

        saved = {name: (node.ds.copy(), node.dt_pred) for name, node in mm.nodes.items()}
        r1 = mm.compute(options["predefined"], scopes=scopes, allow_past=True)
        for name, node in mm.nodes.items():
            node.ds, node.dt_pred = saved[name]
        r2 = mm.compute(options["zero"], scopes=scopes, allow_past=True)

        cat_scope = next(iter(r1["category"]))
        cat_1 = r1["category"][cat_scope]
        cat_2 = r2["category"][cat_scope]
        assert np.isfinite(cat_1["pred_exact_vol"]) and cat_1["pred_exact_vol"] > 0
        assert np.isclose(cat_1["pred_exact_vol"], cat_2["pred_exact_vol"], rtol=1e-9)
        assert np.isclose(cat_1["pred_long_vol"], cat_2["pred_long_vol"], rtol=1e-9)

        sales_scope = next(iter(r1["sales"]))
        sales_1 = r1["sales"][sales_scope]
        sales_2 = r2["sales"][sales_scope]
        assert np.isfinite(sales_1["pred_exact_vol"]) and sales_1["pred_exact_vol"] > 0
        assert not np.isclose(sales_1["pred_exact_vol"], sales_2["pred_exact_vol"], rtol=1e-9)

    def test_sales_only_context_reports_no_category(self, p10_models_ds, samples_path):
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
            / "context_sales_only.xlsx"
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

        assert not category.scope_names
        assert sales.scope_names

        r = mm.compute(ctx.options["predefined"], scopes=ctx.scopes, allow_past=True)
        assert r["category"] == {}
        assert r["sales"]
        assert all(np.isfinite(s["pred_exact_vol"]) for s in r["sales"].values())

    def test_diagnostics_passes_with_static_node(self, p10_setup):
        import fermatrica_rep as fmr

        mm, scopes, options, category, sales = p10_setup

        diag = fmr.MetaModelDiagnostics(mm)
        report = diag.run(options=options, scopes=scopes)

        assert report.passed
        cat_merge_results = [r for r in report.results
                             if r.check_name.startswith("merge_")
                             and r.details.get("node_name") == "category"]
        assert all(r.passed for r in cat_merge_results)
