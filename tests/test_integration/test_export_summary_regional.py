"""Integration test — regional (entity-based) brand summary export.

p04 (single brand, one scope per ``master`` × year): ``export_summaries``
auto-splits into one XLSX per region — ``brand_1_moscow.xlsx`` …
``brand_1_cairo.xlsx`` — without requiring ``plan_year``.

p08 (uba=national, sales=regional DAG): each region file carries the regional
``sales_curves_*`` plus the national ``uba_curves_*`` (with the other regions'
budgets held fixed via ``background_option``); the national ``brand_1.xlsx``
carries ``uba_curves_*`` only.

The export itself is run once per class in a module fixture (it is the slow
part — a grid of budget options computed per entity); the tests only read back
the written workbooks. Grid sizes are kept small to bound runtime.
"""
import copy
from pathlib import Path

import pandas as pd
import pytest

from tests.test_integration.conftest import (
    _activate_adhoc_path,
    _deactivate_adhoc_path,
)

GRID_STEP = 50
BDG_MAX = 50


def _latest_model(model_dir: Path) -> str:
    """Newest committed model_* directory (excluding *_dash saves)."""
    candidates = sorted(
        path
        for path in model_dir.iterdir()
        if path.is_dir()
        and path.name.startswith("model_")
        and not path.name.endswith("_dash")
    )
    return str(candidates[-1])


def _export_params():
    return dict(budget_step=GRID_STEP, bdg_max=BDG_MAX, cores=1, if_exact=True)


@pytest.mark.integration
class TestRegionalExportP04:
    """p04: one brand, scopes = superbrand,master (4 regions × 2 years)."""

    BRAND = "brand_1"
    MASTERS = ["Cairo", "Moscow", "Paris", "Warsaw"]

    @pytest.fixture(scope="module")
    def p04_summary_setup(self, p04_model_ds, samples_path):
        """Single-node MetaModel from p04 context.xlsx, computed to set dt_pred."""
        import fermatrica_rep as fmr

        model, ds = p04_model_ds
        ds = ds.copy()

        root = samples_path / "p04_sample"
        _activate_adhoc_path(root)
        try:
            import code_py.adhoc.reporting
            import code_py.adhoc.translators
            from code_py.adhoc.reporting import pre_reporting_new

            opt_path = str(root / "code_py" / "model_data" / "options" / "context.xlsx")
            ctx = fmr.load_context(opt_path)

            node = fmr.ModelNode(
                model=model,
                ds=ds,
                trans_dict=ctx.trans_dicts.get(""),
                adhoc_code=[code_py.adhoc.translators, code_py.adhoc.reporting],
                language="english",
                ssn_dict=ctx.ssn_dict,
                scope_names={s.scope_name for s in ctx.scopes},
            )
            pre_reporting_new(node)
            node.model_rep.media_prices = ctx.media_prices

            mm = fmr.MetaModel(nodes=node)
            mm.compute(ctx.options["predefined"], scopes=ctx.scopes, allow_past=True)

            yield {"mm": mm, "ctx": ctx}
        finally:
            _deactivate_adhoc_path(root)

    @pytest.fixture(scope="module")
    def p04_files(self, p04_summary_setup, tmp_path_factory):
        """Run the full per-entity export once and return the written paths."""
        import fermatrica_rep as fmr

        mm = p04_summary_setup["mm"]
        ctx = p04_summary_setup["ctx"]
        out = tmp_path_factory.mktemp("p04_summary")

        return fmr.export_summaries(
            mm, ctx,
            output_dir=str(out),
            **_export_params(),
        )

    def test_export_splits_into_entity_files(self, p04_files):
        """No plan_year needed: one file per region (master entity)."""
        assert len(p04_files) == len(self.MASTERS)
        expected = {f"{self.BRAND}_{m.lower()}.xlsx" for m in self.MASTERS}
        actual = {Path(f).name for f in p04_files}
        assert actual == expected

    def test_moscow_file_contents(self, p04_summary_setup, p04_files):
        """Moscow file: curves equal a direct compute_grid on the Moscow-only
        plan scope; decomposition is entity-filtered; brands sheet lists the
        entity and its scopes."""
        import fermatrica_rep as fmr
        from fermatrica_rep.export.export_summary import (
            _curves_tables,
            _decomposition_table,
        )
        from fermatrica_rep.options.scope import find_scope

        mm = p04_summary_setup["mm"]
        ctx = p04_summary_setup["ctx"]
        path = next(p for p in p04_files if p.endswith("brand_1_moscow.xlsx"))

        scope = find_scope(ctx.scopes, "brand_1_moscow_2026")
        assert scope is not None
        expected_short, _ = _curves_tables(
            fmr.compute_grid(
                copy.deepcopy(mm), [scope], **_export_params(),
            )["model"]
        )
        actual_short = pd.read_excel(path, sheet_name="model_curves_short")
        pd.testing.assert_frame_equal(
            actual_short, expected_short, check_exact=False, check_dtype=False
        )

        node = mm.nodes["model"]
        expected_decomp = _decomposition_table(
            node, self.BRAND, "month", (2, 3), col="value",
            filters={"master": ["Moscow"]},
        )
        actual_decomp = pd.read_excel(path, sheet_name="model_decomposition")
        pd.testing.assert_frame_equal(
            actual_decomp, expected_decomp, check_exact=False, check_dtype=False
        )

        brands_sheet = pd.read_excel(path, sheet_name="brands")
        values = dict(zip(brands_sheet["variable"], brands_sheet["value"]))
        assert values["brand"] == self.BRAND
        assert values["entity"] == "moscow"
        assert values["scopes"] == "brand_1_moscow_2026"
        # legacy rows kept alongside entity/scopes for backward compatibility
        assert pd.isna(values["template_scope"])
        assert values["derived_scopes"] == "brand_1_moscow_2026"

    def test_entities_restriction(self, p04_summary_setup, tmp_path_factory):
        """entities=[...] restricts the exported files."""
        import fermatrica_rep as fmr

        mm = p04_summary_setup["mm"]
        ctx = p04_summary_setup["ctx"]
        out = tmp_path_factory.mktemp("p04_summary_cairo")

        files = fmr.export_summaries(
            mm, ctx,
            entities=[{"master": "Cairo"}],
            output_dir=str(out),
            **_export_params(),
        )
        assert len(files) == 1
        assert files[0].endswith("brand_1_cairo.xlsx")


@pytest.mark.integration
class TestRegionalExportP08:
    """p08: national uba → regional sales DAG."""

    BRAND = "brand_1"
    REGIONS = ["Cairo", "Moscow", "Paris", "Warsaw"]

    @pytest.fixture(scope="module")
    def p08_summary_setup(self, samples_path):
        """Two-node MetaModel from p08 context.xlsx, computed to set dt_pred."""
        import fermatrica as fm
        import fermatrica_rep as fmr

        root = samples_path / "p08_sample"
        _activate_adhoc_path(root)
        try:
            import code_py.adhoc.reporting
            import code_py.adhoc.translators
            from code_py.adhoc.reporting import pred_merge_1

            options_path = str(root / "code_py" / "model_data" / "options" / "context.xlsx")

            uba_model, uba_ds, _ = fm.model_load_ext(
                _latest_model(root / "code_py" / "model_data_uba"), missed_stop=True
            )
            sales_model, sales_ds, _ = fm.model_load_ext(
                _latest_model(root / "code_py" / "model_data_sales"), missed_stop=True
            )

            ctx = fmr.load_context(options_path)

            uba = fmr.ModelNode(
                model=uba_model,
                ds=uba_ds,
                options_path=options_path,
                trans_dict=ctx.trans_dicts["uba"],
                adhoc_code=[code_py.adhoc.translators, code_py.adhoc.reporting],
                language="english",
                ssn_dict=ctx.ssn_dict,
                scope_names={s.scope_name for s in ctx.scopes if "region" not in s.filters},
            )
            sales = fmr.ModelNode(
                model=sales_model,
                ds=sales_ds,
                options_path=options_path,
                trans_dict=ctx.trans_dicts["sales"],
                adhoc_code=[code_py.adhoc.translators, code_py.adhoc.reporting],
                language="english",
                ssn_dict=ctx.ssn_dict,
                scope_names={s.scope_name for s in ctx.scopes if "region" in s.filters},
            )

            mm = fmr.MetaModel(
                nodes={"uba": uba, "sales": sales},
                edges=[("uba", "sales", pred_merge_1)],
            )
            mm.compute(ctx.options["predefined"], scopes=ctx.scopes, allow_past=True)

            yield {"mm": mm, "ctx": ctx}
        finally:
            _deactivate_adhoc_path(root)

    @pytest.fixture(scope="module")
    def p08_files(self, p08_summary_setup, tmp_path_factory):
        """Run the full per-entity export once and return the written paths."""
        import fermatrica_rep as fmr

        mm = p08_summary_setup["mm"]
        ctx = p08_summary_setup["ctx"]
        out = tmp_path_factory.mktemp("p08_summary")

        return fmr.export_summaries(
            mm, ctx,
            output_dir=str(out),
            **_export_params(),
        )

    def test_export_splits_into_entity_files(self, p08_files):
        """Four regional files + the national brand_1.xlsx (5 files)."""
        assert len(p08_files) == len(self.REGIONS) + 1
        expected = {f"{self.BRAND}_{r.lower()}.xlsx" for r in self.REGIONS} | {
            f"{self.BRAND}.xlsx"
        }
        actual = {Path(f).name for f in p08_files}
        assert actual == expected

    def test_cairo_file_has_both_node_curves(self, p08_summary_setup, p08_files):
        """brand_1_cairo.xlsx: sales curves from the Cairo scope and uba
        curves from the national scope with the other regions held fixed."""
        import fermatrica_rep as fmr
        from fermatrica_rep.export.export_summary import _curves_tables
        from fermatrica_rep.options.scope import find_scope

        mm = p08_summary_setup["mm"]
        ctx = p08_summary_setup["ctx"]
        path = next(p for p in p08_files if p.endswith("brand_1_cairo.xlsx"))
        xl = pd.ExcelFile(path)
        sheet_names = set(xl.sheet_names)
        xl.close()

        assert "sales_curves_short" in sheet_names
        assert "sales_curves_long" in sheet_names
        assert "uba_curves_short" in sheet_names
        assert "uba_curves_long" in sheet_names

        cairo = find_scope(ctx.scopes, "brand_1_cairo_2026")
        national = find_scope(ctx.scopes, "brand_1_2026")
        assert cairo is not None and national is not None

        # sales: the export grid runs [cairo, national] — the national scope
        # drives the uba node whose prediction merges into sales, so the
        # cross-check must use the same scope set and background
        expected_sales_short, _ = _curves_tables(
            fmr.compute_grid(
                copy.deepcopy(mm), [cairo, national],
                background_option=ctx.options["predefined"],
                **_export_params(),
            )["sales"]
        )
        actual_sales_short = pd.read_excel(path, sheet_name="sales_curves_short")
        pd.testing.assert_frame_equal(
            actual_sales_short, expected_sales_short,
            check_exact=False, check_dtype=False,
        )

        # uba: national scope with the predefined-minus-Cairo background
        expected_uba_short, _ = _curves_tables(
            fmr.compute_grid(
                copy.deepcopy(mm), [cairo, national],
                background_option=ctx.options["predefined"],
                **_export_params(),
            )["uba"]
        )
        actual_uba_short = pd.read_excel(path, sheet_name="uba_curves_short")
        pd.testing.assert_frame_equal(
            actual_uba_short, expected_uba_short,
            check_exact=False, check_dtype=False,
        )

    def test_national_file_uba_only(self, p08_files):
        """brand_1.xlsx has uba curves and no sales curves (sales has no
        national scope)."""
        path = next(p for p in p08_files if p.endswith("brand_1.xlsx"))
        xl = pd.ExcelFile(path)
        sheet_names = set(xl.sheet_names)
        xl.close()

        assert "uba_curves_short" in sheet_names
        assert "uba_curves_long" in sheet_names
        assert "sales_curves_short" not in sheet_names
        assert "sales_curves_long" not in sheet_names

    def test_brands_sheet_lists_entity_and_scopes(self, p08_files):
        """The Cairo file's brands sheet reports the entity and its plan scopes."""
        path = next(p for p in p08_files if p.endswith("brand_1_cairo.xlsx"))

        brands_sheet = pd.read_excel(path, sheet_name="brands")
        values = dict(zip(brands_sheet["variable"], brands_sheet["value"]))
        assert values["brand"] == self.BRAND
        assert values["entity"] == "cairo"
        assert values["scopes"] == "brand_1_cairo_2026"
        # legacy rows kept alongside entity/scopes for backward compatibility
        assert pd.isna(values["template_scope"])
        assert values["derived_scopes"] == "brand_1_cairo_2026"
