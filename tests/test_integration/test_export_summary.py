"""Integration test — brand summary XLSX export on p06_sample.

p06 is a 3-node star-topology DAG (aba → sales, consideration → sales).
The test builds the MetaModel + ComputeContext exactly like the reporting
scripts do, runs ``mm.compute(predefined, scopes)`` so ``dt_pred`` is set on
every node, then exports one XLSX per brand via ``export_summaries``
and verifies the sheet layout, the derived scope names, and that the curves
sheets match a direct ``compute_grid`` cross-check.
"""
import pytest

from tests.test_integration.conftest import (
    _activate_adhoc_path,
    _deactivate_adhoc_path,
    _find_latest_model,
)


@pytest.mark.integration
class TestBrandSummaryExport:

    @pytest.fixture(scope="module")
    def p06_meta(self, samples_path):
        """Build a real 3-node MetaModel + ComputeContext from p06 saved models."""
        root = samples_path / "p06_sample"
        _activate_adhoc_path(root)

        try:
            import code_py.adhoc.reporting
            import code_py.adhoc.translators
            from code_py.adhoc.reporting import pred_merge_1, pred_merge_2

            import fermatrica_rep as fmr

            options_path = str(root / "code_py" / "model_data" / "options" / "context.xlsx")
            ctx = fmr.load_context(options_path)

            trans_dict_aba = fmr.trans_dict_create(path=options_path, sheet="translation_aba")
            trans_dict_cons = fmr.trans_dict_create(path=options_path, sheet="translation_consideration")
            trans_dict_sales = fmr.trans_dict_create(path=options_path, sheet="translation_sales")

            ssn_dict = fmr.seasonality_read(path=options_path)

            node_aba = fmr.ModelNode(
                model_path=_find_latest_model(root / "code_py" / "model_data_aba"),
                trans_dict=trans_dict_aba, language="english",
                adhoc_code=[code_py.adhoc.translators, code_py.adhoc.reporting],
                ssn_dict=ssn_dict, missed_stop=True,
            )
            node_cons = fmr.ModelNode(
                model_path=_find_latest_model(root / "code_py" / "model_data_consideration"),
                trans_dict=trans_dict_cons, language="english",
                adhoc_code=[code_py.adhoc.translators, code_py.adhoc.reporting],
                ssn_dict=ssn_dict, missed_stop=True,
            )
            node_sales = fmr.ModelNode(
                model_path=_find_latest_model(root / "code_py" / "model_data_sales"),
                trans_dict=trans_dict_sales, language="english",
                adhoc_code=[code_py.adhoc.translators, code_py.adhoc.reporting],
                ssn_dict=ssn_dict, missed_stop=True,
            )

            mm = fmr.MetaModel(
                nodes={"aba": node_aba, "consideration": node_cons, "sales": node_sales},
                edges=[("aba", "sales", pred_merge_1), ("consideration", "sales", pred_merge_2)],
            )
            # reporting-script state: option path sets dt_pred on every node
            mm.compute(ctx.options["predefined"], scopes=ctx.scopes, allow_past=True)

            yield {"mm": mm, "ctx": ctx}
        finally:
            _deactivate_adhoc_path(root)

    def test_export_per_brand(self, p06_meta, tmp_path):
        """Three brand files are written with the expected sheet set, and
        brand_1's curves match a direct compute_grid cross-check."""
        import copy

        import fermatrica_rep as fmr
        import pandas as pd
        from fermatrica_rep.export.export_summary import _curves_tables
        from fermatrica_rep.options.scope import find_scope

        mm = p06_meta["mm"]
        ctx = p06_meta["ctx"]

        files = fmr.export_summaries(
            mm, ctx,
            brands=["brand_1", "brand_2", "brand_3"],
            output_dir=str(tmp_path),
            budget_step=25, bdg_max=100, cores=1, if_exact=True,
        )

        assert len(files) == 3
        assert all(brand in path for brand, path in zip(["brand_1", "brand_2", "brand_3"], files))

        for brand, path in zip(["brand_1", "brand_2", "brand_3"], files):
            xl = pd.ExcelFile(path)
            sheets = set(xl.sheet_names)
            for node in ("aba", "consideration", "sales"):
                assert f"{node}_setup" in sheets
                assert f"{node}_brand_stats" in sheets
                assert f"{node}_params" in sheets
                assert f"{node}_rhs" in sheets
                assert f"{node}_decomposition" in sheets
                assert f"{node}_decomposition_rub" in sheets
                assert f"{node}_curves_short" in sheets
                assert f"{node}_curves_long" in sheets
                assert f"{node}_adstocks" in sheets
                assert f"{node}_fit" in sheets

            curves_short = pd.read_excel(path, sheet_name="sales_curves_short")
            assert not curves_short.empty
            assert not (curves_short["option"] == "zero").any()
            assert set(curves_short.columns) == {
                "bdg", "option", "incremental_volume", "incremental_value", "profit", "roi",
            }

            decomp = pd.read_excel(path, sheet_name="sales_decomposition")
            assert not decomp.empty
            assert "date" in decomp.columns

            brands_sheet = pd.read_excel(path, sheet_name="brands")
            scopes_row = brands_sheet.loc[
                brands_sheet["variable"] == "scopes", "value"
            ].iloc[0]
            if brand != "brand_1":
                assert f"{brand}_2026" in scopes_row
            xl.close()

        # cross-check: brand_1 sales_curves_short matches a direct compute_grid
        # at the same (coarse) resolution the export used. Step 25 bounds the
        # grid to a few points per tool (runtime scales linearly with grid
        # rows) while keeping the curve shapes recognizable.
        scope = find_scope(ctx.scopes, "brand_1_2026")
        assert scope is not None
        expected_short, _ = _curves_tables(
            fmr.compute_grid(
                copy.deepcopy(mm), [scope],
                budget_step=25, bdg_max=100, cores=1, if_exact=True,
            )["sales"]
        )
        actual_short = pd.read_excel(files[0], sheet_name="sales_curves_short")
        pd.testing.assert_frame_equal(
            actual_short, expected_short, check_exact=False, check_dtype=False
        )

    def test_export_with_explicit_scopes(self, p06_meta, tmp_path):
        """An explicit scopes override skips the plan-scope template derivation
        and is used verbatim (regional layouts: one brand summary over all scopes)."""
        import fermatrica_rep as fmr
        import pandas as pd

        mm = p06_meta["mm"]
        ctx = p06_meta["ctx"]

        files = fmr.export_summaries(
            mm, ctx,
            brands=["brand_1"],
            scopes=fmr.filter_scopes(ctx.scopes, scope_name="brand_1_2026"),
            output_dir=str(tmp_path),
            budget_step=25, bdg_max=100, cores=1, if_exact=True,
        )

        assert len(files) == 1
        brands_sheet = pd.read_excel(files[0], sheet_name="brands")
        scopes_row = brands_sheet.loc[
            brands_sheet["variable"] == "scopes", "value"
        ].iloc[0]
        assert scopes_row == "brand_1_2026"

    def test_adstocks_sheets_contain_only_formula_variables(self, p06_meta, tmp_path):
        """Adstock sheets exclude variables absent from the active model formula.

        Every node's adstocks sheet must contain exactly the adstock/dwblp
        columns referenced by an active formula token. Variables present in the
        dataset but not used in the model (inactive tokens, no-token transforms)
        must not appear.
        """
        import fermatrica_rep as fmr
        import pandas as pd

        mm = p06_meta["mm"]
        ctx = p06_meta["ctx"]

        files = fmr.export_summaries(
            mm, ctx,
            brands=["brand_1"],
            output_dir=str(tmp_path),
            budget_step=25, bdg_max=100, cores=1, if_exact=True,
        )
        assert len(files) == 1

        xl = pd.ExcelFile(files[0])
        try:
            for node_name in ("aba", "consideration", "sales"):
                sheet = pd.read_excel(files[0], sheet_name=f"{node_name}_adstocks")
                assert not sheet.empty

                conf = mm.nodes[node_name].model.conf
                active_tokens = conf.model_rhs.loc[
                    conf.model_rhs["if_active"] == 1, "token"
                ]
                actual = set(sheet["variable"].unique())

                # nothing outside the active formula may appear on the sheet
                for var in actual:
                    assert active_tokens.str.contains(var).any(), (
                        f"{node_name} adstocks sheet contains non-formula variable {var}"
                    )

                # every adstock/dwblp column of the active formula must appear
                params = conf.params
                meas = params.loc[
                    params["fun"].str.contains("adstock|dwblp", na=False), "variable"
                ].unique()
                expected = set()
                for v in meas:
                    for fun in params.loc[params["variable"] == v, "fun"].unique():
                        col = f"{v}_{fun}"
                        if active_tokens.str.contains(col).any():
                            expected.add(col)
                assert actual == expected, (
                    f"{node_name} adstocks variables mismatch: {actual} != {expected}"
                )
        finally:
            xl.close()
