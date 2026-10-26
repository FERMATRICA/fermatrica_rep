"""Integration test — MetaModel.save/load gold path on p06_sample.

p06 is a 3-node star-topology DAG (aba → sales, consideration → sales)
built from modelling-stage model dirs that lack ``model_rep.pkl.lzma``.
The test saves the fully configured MetaModel with
``MetaModel.save()``, loads it back with ``MetaModel.load()``, and checks
that node keys, edges (merge functions), and option summaries round-trip.
"""
import os

import pytest

from tests.test_integration.conftest import (
    _activate_adhoc_path,
    _deactivate_adhoc_path,
    _find_latest_model,
)


@pytest.mark.integration
class TestP06MetaModelSaveLoad:

    @pytest.fixture(scope="module")
    def p06_setup(self, samples_path):
        """Build a real 3-node MetaModel from p06 saved models."""
        root = samples_path / "p06_sample"
        _activate_adhoc_path(root)

        try:
            import code_py.adhoc.reporting
            import code_py.adhoc.translators
            from code_py.adhoc.reporting import pred_merge_1, pred_merge_2

            import fermatrica_rep as fmr

            options_path = str(root / "code_py" / "model_data" / "options" / "context.xlsx")

            trans_dict_aba = fmr.trans_dict_create(path=options_path, sheet="translation_aba")
            trans_dict_cons = fmr.trans_dict_create(path=options_path, sheet="translation_consideration")
            trans_dict_sales = fmr.trans_dict_create(path=options_path, sheet="translation_sales")

            scopes = fmr.scopes_read(options_path)
            options = fmr.Option.from_excel_all(options_path)
            ssn_dict = fmr.seasonality_read(path=options_path)

            pth_aba = _find_latest_model(root / "code_py" / "model_data_aba")
            pth_cons = _find_latest_model(root / "code_py" / "model_data_consideration")
            pth_sales = _find_latest_model(root / "code_py" / "model_data_sales")

            node_aba = fmr.ModelNode(
                model_path=pth_aba, trans_dict=trans_dict_aba, language="english",
                adhoc_code=[code_py.adhoc.translators, code_py.adhoc.reporting],
                ssn_dict=ssn_dict, missed_stop=True,
            )
            node_cons = fmr.ModelNode(
                model_path=pth_cons, trans_dict=trans_dict_cons, language="english",
                adhoc_code=[code_py.adhoc.translators, code_py.adhoc.reporting],
                ssn_dict=ssn_dict, missed_stop=True,
            )
            node_sales = fmr.ModelNode(
                model_path=pth_sales, trans_dict=trans_dict_sales, language="english",
                adhoc_code=[code_py.adhoc.translators, code_py.adhoc.reporting],
                ssn_dict=ssn_dict, missed_stop=True,
            )

            mm = fmr.MetaModel(
                nodes={"aba": node_aba, "consideration": node_cons, "sales": node_sales},
                edges=[("aba", "sales", pred_merge_1), ("consideration", "sales", pred_merge_2)],
            )

            yield {
                "mm": mm,
                "scopes": scopes,
                "options": options,
                "paths": {"aba": pth_aba, "consideration": pth_cons, "sales": pth_sales},
            }
        finally:
            _deactivate_adhoc_path(root)

    @pytest.fixture(scope="module")
    def p06_summaries(self, p06_setup):
        """Compute once with the pre-save MetaModel."""
        mm = p06_setup["mm"]
        scopes = p06_setup["scopes"]
        options = p06_setup["options"]
        return mm.compute(options["predefined"], scopes=scopes, allow_past=True)

    def test_auto_rep_for_modelling_stage_dir(self, p06_setup):
        """p06 model dirs lack model_rep.pkl.lzma → ModelNode.load defaults the rep."""
        import fermatrica_rep as fmr

        node = fmr.ModelNode.load(p06_setup["paths"]["aba"])
        assert node.model_rep is not None
        assert node.model_rep.language == "english"

    def test_round_trip_layout_and_structure(self, p06_setup, p06_summaries, samples_path, tmp_path):
        """Save + load: folder layout, clean node keys, string merge fns."""
        import fermatrica_rep as fmr

        mm = p06_setup["mm"]
        options_path = str(samples_path / "p06_sample" / "code_py" / "model_data"
                           / "options" / "context.xlsx")

        meta_dir = mm.save(path=str(tmp_path), save_tag="test",
                           include_context_path=options_path)

        assert os.path.basename(meta_dir) == "meta_test"
        assert os.path.isfile(os.path.join(meta_dir, "meta_conf.toml"))
        assert os.path.isfile(os.path.join(meta_dir, "context.xlsx"))
        for node_name in ("aba", "consideration", "sales"):
            node_dir = os.path.join(meta_dir, f"model_{node_name}")
            assert os.path.isfile(os.path.join(node_dir, "model_rep.pkl.lzma"))
            assert os.path.isfile(os.path.join(node_dir, "dt_p.pkl.zstd"))

        mm2 = fmr.MetaModel.load(meta_dir)

        assert set(mm2.nodes.keys()) == {"aba", "consideration", "sales"}
        assert len(mm2.edges) == 2
        assert {e[0] for e in mm2.edges} == {"aba", "consideration"}
        assert all(e[1] == "sales" for e in mm2.edges)
        merge_names = {e[2] for e in mm2.edges}
        assert merge_names == {
            "code_py.adhoc.reporting.pred_merge_1",
            "code_py.adhoc.reporting.pred_merge_2",
        }

        # reporting settings survive the pickle round trip
        for node_name in ("aba", "consideration", "sales"):
            rep = mm2.nodes[node_name].model_rep
            assert rep.trans_dict is not None
            assert rep.ssn_dict is not None

    def test_round_trip_summaries_match(self, p06_setup, p06_summaries, tmp_path):
        """Re-running compute on the loaded MetaModel reproduces summaries."""
        import fermatrica_rep as fmr

        mm = p06_setup["mm"]
        scopes = p06_setup["scopes"]
        options = p06_setup["options"]

        meta_dir = mm.save(path=str(tmp_path), save_tag="test")
        mm2 = fmr.MetaModel.load(meta_dir)
        summaries_after = mm2.compute(options["predefined"], scopes=scopes, allow_past=True)

        for node_name in ("aba", "consideration", "sales"):
            assert set(summaries_after[node_name].keys()) == set(p06_summaries[node_name].keys())
            for scope_name in summaries_after[node_name]:
                before = p06_summaries[node_name][scope_name]
                after = summaries_after[node_name][scope_name]
                for key in ("pred_exact_vol", "pred_long_vol"):
                    assert before[key] == pytest.approx(after[key]), (
                        f"{node_name}/{scope_name}: {key} changed after save/load "
                        f"({before[key]} vs {after[key]})"
                    )
