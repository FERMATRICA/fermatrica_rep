"""Integration test — campaign-based options generator on p09_sample.

p09 derives its context.xlsx (scopes / allocations / seasonality) from a
campaign table via `campaigns_to_options()` (per-option scopes). The test
checks that the in-memory derivation agrees with `load_context` of the saved
file, that a 3-node MetaModel computes per-option-scope summaries for
`predefined`, and that `options_detail` emits one slide set per option.
"""
import os

import pandas as pd
import pytest

from tests.test_integration.conftest import (
    _activate_adhoc_path,
    _deactivate_adhoc_path,
)


def _latest_model(model_dir):
    dirs = [d for d in os.listdir(model_dir)
            if d.startswith("model_") and not d.endswith("_dash")
            and os.path.isdir(os.path.join(model_dir, d))]
    dirs.sort(reverse=True)
    return os.path.join(model_dir, dirs[0])


@pytest.mark.integration
class TestP09CampaignPipeline:

    @pytest.fixture(scope="module")
    def p09_setup(self, samples_path):
        """Load p09 context (generated + in-memory) and build the MetaModel."""
        root = samples_path / "p09_sample"
        _activate_adhoc_path(root)

        try:
            import code_py.adhoc.reporting
            import code_py.adhoc.translators
            from code_py.adhoc.reporting import pred_merge_1, pred_merge_2

            import fermatrica_rep as fmr

            options_path = str(root / "code_py" / "model_data" / "options" / "context.xlsx")
            campaigns_path = str(root / "code_py" / "model_data" / "options" / "campaigns.xlsx")

            dt_campaigns = pd.read_excel(campaigns_path, sheet_name="campaigns")
            ctx_mem = fmr.campaigns_to_options(dt_campaigns, granularity="month")
            ctx_loaded = fmr.load_context(options_path)

            trans_dict_aba = fmr.trans_dict_create(path=options_path, sheet="translation_aba")
            trans_dict_cons = fmr.trans_dict_create(path=options_path, sheet="translation_consideration")
            trans_dict_sales = fmr.trans_dict_create(path=options_path, sheet="translation_sales")

            node_aba = fmr.ModelNode(
                model_path=_latest_model(str(root / "code_py" / "model_data_aba")),
                trans_dict=trans_dict_aba, language="english",
                adhoc_code=[code_py.adhoc.translators, code_py.adhoc.reporting],
                missed_stop=True,
            )
            node_cons = fmr.ModelNode(
                model_path=_latest_model(str(root / "code_py" / "model_data_consideration")),
                trans_dict=trans_dict_cons, language="english",
                adhoc_code=[code_py.adhoc.translators, code_py.adhoc.reporting],
                missed_stop=True,
            )
            node_sales = fmr.ModelNode(
                model_path=_latest_model(str(root / "code_py" / "model_data_sales")),
                trans_dict=trans_dict_sales, language="english",
                adhoc_code=[code_py.adhoc.translators, code_py.adhoc.reporting],
                missed_stop=True,
            )

            mm = fmr.MetaModel(
                nodes={"aba": node_aba, "consideration": node_cons, "sales": node_sales},
                edges=[("aba", "sales", pred_merge_1), ("consideration", "sales", pred_merge_2)],
                compute_context=ctx_loaded,
            )

            yield {
                "root": root,
                "mm": mm,
                "ctx_mem": ctx_mem,
                "ctx_loaded": ctx_loaded,
                "options_path": options_path,
            }
        finally:
            _deactivate_adhoc_path(root)

    def test_memory_derivation_agrees_with_saved_context(self, p09_setup):
        ctx_mem = p09_setup["ctx_mem"]
        ctx_loaded = p09_setup["ctx_loaded"]

        assert [s.scope_name for s in ctx_mem.scopes] == [
            "predefined_brand_1_2025", "zero_brand_1_2025",
            "predefined_brand_1_2026", "zero_brand_1_2026",
            "predefined_brand_1_2027", "zero_brand_1_2027",
        ]
        assert [s.scope_name for s in ctx_loaded.scopes] == [
            s.scope_name for s in ctx_mem.scopes
        ]

        # per-option scopes: full calendar years, previous-year reference
        for s in ctx_loaded.scopes:
            assert s.date_start.day == 1 and s.date_start.month == 1
            assert s.date_end.month == 12 and s.date_end.day == 31
            assert s.filters == {"superbrand": ["brand_1"]}
            assert s.seasonality == s.scope_name

        # allocations agree (day-weighted sums); the saved file carries the
        # zero option too (all-zero val rows)
        assert set(ctx_mem.options.keys()) == {"predefined", "zero"}
        assert set(ctx_loaded.options.keys()) == {"predefined", "zero"}
        for scope_name in ("predefined_brand_1_2025", "predefined_brand_1_2026",
                           "predefined_brand_1_2027"):
            mem = ctx_mem.options["predefined"].budget_for_scope(scope_name)
            loaded = ctx_loaded.options["predefined"].budget_for_scope(scope_name)
            assert set(mem.keys()) == set(loaded.keys())
            for item in mem:
                assert mem[item] == pytest.approx(loaded[item])

        # seasonality identical and sums to 1.0 per scope-year
        # (zero scopes carry all-zero coefficients — no budget)
        for scope_name in ctx_mem.ssn_dict:
            pd.testing.assert_frame_equal(
                ctx_mem.ssn_dict[scope_name], ctx_loaded.ssn_dict[scope_name]
            )
            cols = [c for c in ctx_loaded.ssn_dict[scope_name].columns if c != "month"]
            expected_sum = 1.0 if scope_name.startswith("predefined_") else 0.0
            for c in cols:
                assert ctx_loaded.ssn_dict[scope_name][c].sum() == pytest.approx(expected_sum), (
                    f"{scope_name}/{c} does not sum to {expected_sum}"
                )

        # 2025 has no H1 campaigns — earliest activity starts 2025-07-01
        ssn_2025 = ctx_loaded.ssn_dict["predefined_brand_1_2025"]
        active_months = ssn_2025.loc[ssn_2025["tv"] > 0, "month"].tolist()
        assert active_months
        assert min(active_months) >= 7 and max(active_months) <= 10

    def test_compute_predefined_covers_all_scopes(self, p09_setup):
        mm = p09_setup["mm"]
        ctx_loaded = p09_setup["ctx_loaded"]

        opt_sum = mm.compute(ctx_loaded.options["predefined"], allow_past=True,
                             scopes=ctx_loaded.scopes)
        for node_name in ("aba", "consideration", "sales"):
            assert set(opt_sum[node_name].keys()) == {
                "predefined_brand_1_2025", "predefined_brand_1_2026",
                "predefined_brand_1_2027",
            }
            for scope_name in opt_sum[node_name]:
                assert pd.notna(opt_sum[node_name][scope_name]["pred_exact_vol"])
                assert pd.notna(opt_sum[node_name][scope_name]["pred_long_vol"])

    def test_options_detail_one_slide_set_per_option(self, p09_setup):
        import fermatrica_rep as fmr
        import fermatrica_rep.export.export_pptx as export_pptx

        mm = p09_setup["mm"]
        ctx_loaded = p09_setup["ctx_loaded"]

        scopes_plan = fmr.filter_scopes(ctx_loaded.scopes, start_year_min=2026)

        node = mm.nodes["sales"]
        node.model_rep, prs = export_pptx.set_config(node.model_rep, template_name="blanc")

        prs = export_pptx.options_detail(
            prs,
            meta_model=mm,
            scopes=scopes_plan,
            options=ctx_loaded.options,
            sku_var=["superbrand", "master", "form"],
            period="month",
            if_exact=True,
            if_volume=True,
            if_sku_fit_predict=False,
            if_sku_decompose=False,
            if_sku_waterfall=False,
        )

        # one slide set for the single predefined option (zero has no slides)
        assert len(prs.slides._sldIdLst) >= 1

    def test_options_table_works_with_per_option_scopes(self, p09_setup):
        import fermatrica_rep.export.export_pptx as export_pptx

        mm = p09_setup["mm"]
        ctx_loaded = p09_setup["ctx_loaded"]

        # populate dt_pred (needed by bs_key_filter in the wrapper)
        mm.compute(ctx_loaded.options["predefined"], allow_past=True, scopes=ctx_loaded.scopes)

        node = mm.nodes["sales"]
        node.model_rep, prs = export_pptx.set_config(node.model_rep, template_name="blanc")
        n_before = len(prs.slides._sldIdLst)

        prs = export_pptx.options_table(
            prs,
            meta_model=mm,
            scopes=ctx_loaded.scopes,
            options=ctx_loaded.options,
            if_exact=True,
        )

        # one slide per scope — predefined (zero unavailable) and zero-only
        # scopes alike, i.e. 3 + 3 slides
        assert len(prs.slides._sldIdLst) - n_before == 6
