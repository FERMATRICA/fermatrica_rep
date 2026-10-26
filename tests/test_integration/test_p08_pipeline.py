"""Integration tests for the p08 national-to-regional reporting DAG."""

from pathlib import Path

import numpy as np
import pytest

from tests.test_integration.conftest import _activate_adhoc_path, _deactivate_adhoc_path


@pytest.mark.integration
class TestP08NewApi:
    """Exercise automatic scope routing for p08 UBA and Sales nodes."""

    @pytest.fixture
    def p08_new_setup(self, samples_path):
        import fermatrica as fm
        import fermatrica_rep as fmr

        root = samples_path / "p08_sample"
        _activate_adhoc_path(root)
        try:
            import code_py.adhoc.reporting
            import code_py.adhoc.translators
            from code_py.adhoc.reporting import pred_merge_1
            options_path = str(
                root / "code_py" / "model_data" / "options" / "context.xlsx"
            )
            def latest_model(model_dir: Path) -> str:
                candidates = sorted(
                    path
                    for path in model_dir.iterdir()
                    if path.is_dir()
                    and path.name.startswith("model_")
                    and not path.name.endswith("_dash")
                )
                return str(candidates[-1])

            uba_model, uba_ds, _ = fm.model_load_ext(
                latest_model(root / "code_py" / "model_data_uba"), missed_stop=True
            )
            sales_model, sales_ds, _ = fm.model_load_ext(
                latest_model(root / "code_py" / "model_data_sales"), missed_stop=True
            )
            scopes = fmr.scopes_read(options_path)
            options = fmr.Option.from_excel_all(options_path)
            ssn_dict = fmr.seasonality_read(options_path)

            uba = fmr.ModelNode(
                model=uba_model,
                ds=uba_ds,
                options_path=options_path,
                trans_dict=fmr.trans_dict_create(
                    path=options_path, sheet="translation_uba"
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
            uba.scope_names = {s.scope_name for s in scopes if "region" not in s.filters}
            sales.scope_names = {s.scope_name for s in scopes if "region" in s.filters}

            mm = fmr.MetaModel(
                nodes={"uba": uba, "sales": sales},
                edges=[("uba", "sales", pred_merge_1)],
            )
            yield mm, scopes, options, uba, sales
        finally:
            _deactivate_adhoc_path(root)

    def test_scope_routing_separates_national_and_regional_nodes(self, p08_new_setup):
        mm, scopes, options, uba, sales = p08_new_setup

        assert uba.scope_names == {"brand_1_2025", "brand_1_2026"}
        assert sales.scope_names
        sales_scopes = {s.scope_name for s in scopes if "region" in s.filters}
        assert sales.scope_names == sales_scopes

    def test_new_api_compute_returns_finite_summaries(self, p08_new_setup):
        mm, scopes, options, _, sales = p08_new_setup

        result = mm.compute(options["predefined"], scopes=scopes, allow_past=True)
        expected_scopes = set(options["predefined"].budget_data) & sales.scope_names

        assert set(result["sales"]) == expected_scopes
        assert all(
            np.isfinite(summary["pred_exact_vol"])
            and summary["pred_exact_vol"] > 0
            for summary in result["sales"].values()
        )
