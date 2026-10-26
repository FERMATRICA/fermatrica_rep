"""Integration tests for the p07 category-to-sales reporting DAG."""

import numpy as np
import pytest


@pytest.mark.integration
class TestP07NewApi:
    """Exercise p07 with the current ModelNode/MetaModel reporting API."""

    @pytest.fixture
    def p07_new_setup(self, p07_models_ds, samples_path):
        import code_py.adhoc.reporting
        import code_py.adhoc.translators
        from code_py.adhoc.reporting import pred_merge_1
        import fermatrica_rep as fmr

        category_model, category_ds, sales_model, sales_ds = p07_models_ds
        options_path = str(
            samples_path
            / "p07_sample"
            / "code_py"
            / "model_data"
            / "options"
            / "context.xlsx"
        )
        scopes = fmr.scopes_read(options_path)
        options = fmr.Option.from_excel_all(options_path)
        ssn_dict = fmr.seasonality_read(options_path)

        category = fmr.ModelNode(
            model=category_model,
            ds=category_ds,
            options_path=options_path,
            trans_dict=fmr.trans_dict_create(
                path=options_path, sheet="translation_category"
            ),
            adhoc_code=[code_py.adhoc.translators, code_py.adhoc.reporting],
            language="english",
            ssn_dict=ssn_dict,
        )
        sales = fmr.ModelNode(
            model=sales_model,
            ds=sales_ds,
            options_path=options_path,
            trans_dict=fmr.trans_dict_create(
                path=options_path, sheet="translation_sales"
            ),
            adhoc_code=[code_py.adhoc.translators, code_py.adhoc.reporting],
            language="english",
            ssn_dict=ssn_dict,
        )
        category.scope_names = {
            s.scope_name for s in scopes
            if all(category.ds[col].isin(values).any() for col, values in s.filters.items())
        }
        sales.scope_names = {
            s.scope_name for s in scopes
            if all(s.filters.get(col) and sales.ds[col].isin(values).any()
                   for col, values in s.filters.items())
        }

        return fmr.MetaModel(
            nodes={"category": category, "sales": sales},
            edges=[("category", "sales", pred_merge_1)],
        ), scopes, options, category, sales

    def test_scope_routing_matches_model_grain(self, p07_new_setup):
        _, _, _, category, sales = p07_new_setup

        assert all(name.startswith("category_") for name in category.scope_names)
        assert all(name.startswith("brand_1_") for name in sales.scope_names)

    def test_new_api_compute_returns_finite_node_summaries(self, p07_new_setup):
        mm, scopes, options, category, sales = p07_new_setup

        result = mm.compute(options["predefined"], scopes=scopes, allow_past=True)

        for node_name, node in (("category", category), ("sales", sales)):
            expected_scopes = set(options["predefined"].budget_data) & node.scope_names
            assert set(result[node_name]) == expected_scopes
            assert all(
                np.isfinite(summary["pred_exact_vol"])
                and summary["pred_exact_vol"] > 0
                for summary in result[node_name].values()
            )
