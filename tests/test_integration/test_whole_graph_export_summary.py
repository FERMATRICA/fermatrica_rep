"""XLSX brand-summary threading of whole-DAG decomposition on p06.

p06 is a 3-node star DAG (aba -> sales, consideration -> sales). With
``whole_graph=True`` the sink node's ``decomposition`` / ``decomposition_rub``
sheets must pivot the expanded ``<link>.<factor>`` variables and sum to the
same per-period totals as the unexpanded sheets.
"""

import pytest


@pytest.mark.integration
class TestWholeGraphExportSummary:
    """summary_data / export_summaries with whole_graph=True on p06."""

    @pytest.fixture(scope="module")
    def p06_meta(self, samples_path):
        from tests.test_integration.conftest import (
            _activate_adhoc_path,
            _deactivate_adhoc_path,
            _find_latest_model,
        )

        root = samples_path / "p06_sample"
        _activate_adhoc_path(root)

        try:
            import code_py.adhoc.reporting
            import code_py.adhoc.translators
            from code_py.adhoc.reporting import pred_merge_1, pred_merge_2

            import fermatrica_rep as fmr

            options_path = str(root / "code_py" / "model_data" / "options" / "context.xlsx")
            ctx = fmr.load_context(options_path)

            node_aba = fmr.ModelNode(
                model_path=_find_latest_model(root / "code_py" / "model_data_aba"),
                trans_dict=fmr.trans_dict_create(path=options_path, sheet="translation_aba"),
                language="english",
                adhoc_code=[code_py.adhoc.translators, code_py.adhoc.reporting],
                ssn_dict=fmr.seasonality_read(path=options_path),
                missed_stop=True,
            )
            node_cons = fmr.ModelNode(
                model_path=_find_latest_model(root / "code_py" / "model_data_consideration"),
                trans_dict=fmr.trans_dict_create(path=options_path, sheet="translation_consideration"),
                language="english",
                adhoc_code=[code_py.adhoc.translators, code_py.adhoc.reporting],
                ssn_dict=fmr.seasonality_read(path=options_path),
                missed_stop=True,
            )
            node_sales = fmr.ModelNode(
                model_path=_find_latest_model(root / "code_py" / "model_data_sales"),
                trans_dict=fmr.trans_dict_create(path=options_path, sheet="translation_sales"),
                language="english",
                adhoc_code=[code_py.adhoc.translators, code_py.adhoc.reporting],
                ssn_dict=fmr.seasonality_read(path=options_path),
                missed_stop=True,
            )

            mm = fmr.MetaModel(
                nodes={"aba": node_aba, "consideration": node_cons, "sales": node_sales},
                edges=[("aba", "sales", pred_merge_1), ("consideration", "sales", pred_merge_2)],
            )
            mm.compute(ctx.options["predefined"], scopes=ctx.scopes, allow_past=True)
            yield {"mm": mm, "ctx": ctx}
        finally:
            _deactivate_adhoc_path(root)

    def test_summary_data_whole_graph_pivots_expanded(self, p06_meta):
        """Sink decomposition sheets pivot expanded variables, summing like the default."""
        import fermatrica_rep as fmr

        mm = p06_meta["mm"]
        ctx = p06_meta["ctx"]

        grid_kwargs = dict(budget_step=25, bdg_max=100, cores=1, if_exact=True)

        wg = fmr.summary_data(mm, ctx, "brand_1", whole_graph=True, **grid_kwargs)
        default = fmr.summary_data(mm, ctx, "brand_1", whole_graph=False, **grid_kwargs)

        wg_decomp = wg["sales_decomposition"]
        def_decomp = default["sales_decomposition"]

        wg_cols = [c for c in wg_decomp.columns if c != "date"]
        def_cols = [c for c in def_decomp.columns if c != "date"]

        # bare link columns replaced by dotted leaves
        assert "ABA" in def_cols and "Consideration" in def_cols
        assert "ABA" not in wg_cols and "Consideration" not in wg_cols
        dotted = [c for c in wg_cols if "." in c]
        assert any(c.startswith("ABA.") for c in dotted)
        assert any(c.startswith("Consideration.") for c in dotted)

        # per-period totals identical to the unexpanded sheets
        tot_wg = wg_decomp.set_index("date")[wg_cols].sum(axis=1)
        tot_def = def_decomp.set_index("date")[def_cols].sum(axis=1)
        assert (tot_wg - tot_def).abs().max() < 1e-6

        # value sheets expanded the same way
        wg_rub = wg["sales_decomposition_rub"]
        rub_cols = [c for c in wg_rub.columns if c != "date"]
        assert any(c.startswith("ABA.") for c in rub_cols)
        assert "ABA" not in rub_cols

        # non-sink node (aba, no incoming links) unchanged
        assert wg["aba_decomposition"].equals(default["aba_decomposition"])
        assert wg["aba_decomposition_rub"].equals(default["aba_decomposition_rub"])
