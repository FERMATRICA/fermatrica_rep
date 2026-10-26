"""Unit tests for MetaModelDiagnostics — validation on demand.

Tests cover CheckResult, DiagnosticsReport, and all four diagnostic checks:
  1. Node types
  2. Node attributes (model, ds, model_rep.*, dt_pred, adhoc_code)
  3. Edge resolution (merge_fn lookup + signature)
  4. Merge effects (ds/dt_pred comparison between compute runs)
"""

import numpy as np
import pandas as pd
import pytest

from fermatrica.basics.basics import FermatricaError
from fermatrica_rep.meta_model.node import ModelNode
from fermatrica_rep.meta_model.meta_model import MetaModel
from fermatrica_rep.meta_model.model_rep import ModelRep
from fermatrica_rep.meta_model.diagnostics import (
    MetaModelDiagnostics,
    DiagnosticsReport,
    CheckResult,
)
from fermatrica_rep.options.scope import OptionScope
from fermatrica_rep.options.option import Option


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _MockModel:
    def __init__(self, name=""):
        self.conf = _MockConf(name)

    @property
    def brands(self):
        return getattr(self.conf, "brands", [])


class _MockConf:
    def __init__(self, name=""):
        self.brands = []
        self.Y_var = "kpi"
        self.name = name


def _basic_ds(n=10):
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "date": dates,
        "superbrand": ["x"] * n,
        "bs_key": [1] * n,
        "listed": [4] * n,
        "predicted": np.random.uniform(90, 110, n),
        "observed": np.random.uniform(85, 115, n),
    })


def _basic_model_rep(ds=None):
    if ds is None:
        ds = _basic_ds()
    return ModelRep(ds, language="english", trans_dict={"bdg_tv": {"variable": "tv", "base_price": 1.0}})


def _basic_node(name=""):
    ds = _basic_ds()
    return ModelNode(model=_MockModel(name), ds=ds, model_rep=_basic_model_rep(ds))


def _basic_meta_model(n_nodes=1):
    if n_nodes == 1:
        return MetaModel(nodes=_basic_node())
    nodes = {}
    for i in range(n_nodes):
        name = f"node_{i}"
        nodes[name] = _basic_node(name)
    edges = [(f"node_{i}", f"node_{i+1}", None) for i in range(n_nodes - 1)]
    return MetaModel(nodes=nodes, edges=edges)


# ---------------------------------------------------------------------------
# TestCheckResult
# ---------------------------------------------------------------------------

class TestCheckResult:
    def test_basic_construction(self):
        r = CheckResult(check_name="node_type", level="error", passed=False, message="fail")
        assert r.check_name == "node_type"
        assert r.level == "error"
        assert not r.passed
        assert r.message == "fail"
        assert r.details is None

    def test_with_details(self):
        r = CheckResult(
            check_name="node_attrs", level="warning", passed=True,
            message="ok", details={"node_name": "x", "attr": "trans_dict"},
        )
        assert r.details == {"node_name": "x", "attr": "trans_dict"}

    def test_levels(self):
        for level in ("error", "warning", "info"):
            r = CheckResult(check_name="test", level=level, passed=True, message="")
            assert r.level == level


# ---------------------------------------------------------------------------
# TestDiagnosticsReport
# ---------------------------------------------------------------------------

class TestDiagnosticsReport:
    def test_passed_all_errors_pass(self):
        r = DiagnosticsReport(
            results=[
                CheckResult("a", "error", True, ""),
                CheckResult("b", "error", True, ""),
                CheckResult("c", "warning", False, "ignored"),
            ],
            checks_ran=["a", "b"],
            checks_skipped=[],
        )
        assert r.passed

    def test_passed_one_error_fails(self):
        r = DiagnosticsReport(
            results=[
                CheckResult("a", "error", True, ""),
                CheckResult("b", "error", False, "fail"),
            ],
            checks_ran=["a", "b"],
            checks_skipped=[],
        )
        assert not r.passed

    def test_passed_warnings_ignored(self):
        r = DiagnosticsReport(
            results=[
                CheckResult("a", "warning", False, "warn"),
                CheckResult("b", "info", False, "info"),
            ],
            checks_ran=["a"],
            checks_skipped=[],
        )
        assert r.passed

    def test_passed_empty_results(self):
        r = DiagnosticsReport(results=[], checks_ran=[], checks_skipped=[])
        assert r.passed

    def test_summary_includes_errors_warnings_info(self):
        r = DiagnosticsReport(
            results=[
                CheckResult("chk", "error", False, "error msg"),
                CheckResult("chk", "warning", False, "warn msg"),
                CheckResult("chk", "info", True, "info msg"),
            ],
            checks_ran=[],
            checks_skipped=["skip reason"],
        )
        s = r.summary
        assert "[FAIL]" in s
        assert "[WARN]" in s
        assert "[PASS]" in s
        assert "Errors: 1" in s
        assert "Warnings: 1" in s
        assert "Info: 1" in s
        assert "Skipped: skip reason" in s

    def test_summary_no_skipped(self):
        r = DiagnosticsReport(results=[], checks_ran=[], checks_skipped=[])
        assert "Skipped: none" in r.summary

    def test_errors_property(self):
        r = DiagnosticsReport(
            results=[
                CheckResult("a", "error", False, "err1"),
                CheckResult("b", "error", True, "err2"),
                CheckResult("c", "warning", False, "warn"),
                CheckResult("d", "info", True, "info"),
            ],
            checks_ran=[],
            checks_skipped=[],
        )
        assert len(r.errors) == 2
        assert all(e.level == "error" for e in r.errors)

    def test_warnings_property(self):
        r = DiagnosticsReport(
            results=[
                CheckResult("a", "error", False, "err"),
                CheckResult("b", "warning", False, "warn"),
                CheckResult("c", "info", True, "info"),
            ],
            checks_ran=[],
            checks_skipped=[],
        )
        assert len(r.warnings) == 1
        assert r.warnings[0].level == "warning"

    def test_infos_property(self):
        r = DiagnosticsReport(
            results=[
                CheckResult("a", "info", True, "info1"),
                CheckResult("b", "info", True, "info2"),
                CheckResult("c", "error", False, "err"),
            ],
            checks_ran=[],
            checks_skipped=[],
        )
        assert len(r.infos) == 2
        assert all(i.level == "info" for i in r.infos)

    def test_errors_summary(self):
        r = DiagnosticsReport(
            results=[
                CheckResult("chk", "error", False, "error msg"),
                CheckResult("chk", "warning", False, "warn msg"),
                CheckResult("chk", "info", True, "info msg"),
            ],
            checks_ran=[],
            checks_skipped=[],
        )
        s = r.errors_summary
        assert "[FAIL]" in s
        assert "[WARN]" not in s
        assert "[PASS]" not in s
        assert "Errors: 1" in s

    def test_errors_summary_empty(self):
        r = DiagnosticsReport(
            results=[CheckResult("chk", "info", True, "all good")],
            checks_ran=[],
            checks_skipped=[],
        )
        assert r.errors_summary == ""

    def test_warnings_summary(self):
        r = DiagnosticsReport(
            results=[
                CheckResult("chk", "error", False, "error msg"),
                CheckResult("chk", "warning", False, "warn msg"),
                CheckResult("chk", "info", True, "info msg"),
            ],
            checks_ran=[],
            checks_skipped=[],
        )
        s = r.warnings_summary
        assert "[WARN]" in s
        assert "[FAIL]" not in s
        assert "[PASS]" not in s
        assert "Warnings: 1" in s

    def test_warnings_summary_empty(self):
        r = DiagnosticsReport(
            results=[CheckResult("chk", "info", True, "all good")],
            checks_ran=[],
            checks_skipped=[],
        )
        assert r.warnings_summary == ""


# ---------------------------------------------------------------------------
# TestMetaModelDiagnostics — integration
# ---------------------------------------------------------------------------

class TestMetaModelDiagnosticsIntegration:

    def test_run_returns_report(self):
        mm = _basic_meta_model()
        diag = MetaModelDiagnostics(mm)
        report = diag.run()
        assert isinstance(report, DiagnosticsReport)
        assert len(report.results) > 0
        assert "node_type" in report.checks_ran
        assert "node_attrs" in report.checks_ran
        assert "edge_resolution" in report.checks_ran

    def test_run_skips_merge_when_no_options(self):
        mm = _basic_meta_model()
        diag = MetaModelDiagnostics(mm)
        report = diag.run(options=None, scopes=None)
        assert any("missing options, scopes" in s for s in report.checks_skipped)

    def test_run_skips_merge_with_few_options(self):
        mm = _basic_meta_model()
        diag = MetaModelDiagnostics(mm)
        scope = OptionScope(
            scope_name="s1", date_start="2025-01-03", date_end="2025-01-08",
            ref_date_start="2025-01-03", ref_date_end="2025-01-08",
        )
        option = Option(name="opt1", budget_data={"s1": {"bdg_tv": 10}})
        report = diag.run(options=[option], scopes=[scope])
        assert any("need at least 2 options" in s for s in report.checks_skipped)


# ---------------------------------------------------------------------------
# TestCheckNodeTypes
# ---------------------------------------------------------------------------

class TestCheckNodeTypes:

    def test_model_node_passes(self):
        mm = _basic_meta_model()
        diag = MetaModelDiagnostics(mm)
        report = diag.run()
        node_results = [r for r in report.results if r.check_name == "node_type"]
        assert all(r.passed for r in node_results)
        assert all(r.level == "info" for r in node_results)

    def test_non_model_node_fails(self):
        ds = _basic_ds()
        mr = _basic_model_rep(ds)
        node = ModelNode(model=_MockModel(), ds=ds, model_rep=mr)
        mm = MetaModel(
            nodes={"good": node, "bad": "not_a_node"},
            edges=[("good", "bad", None)],
        )
        diag = MetaModelDiagnostics(mm)
        report = diag.run()
        type_results = [r for r in report.results if r.check_name == "node_type"]
        good = [r for r in type_results if r.details["node_name"] == "good"]
        bad = [r for r in type_results if r.details["node_name"] == "bad"]
        assert good[0].passed
        assert not bad[0].passed
        assert bad[0].level == "error"

    def test_raise_on_first_error(self):
        ds = _basic_ds()
        mr = _basic_model_rep(ds)
        node = ModelNode(model=_MockModel(), ds=ds, model_rep=mr)
        mm = MetaModel(
            nodes={"good": node, "bad": 123},
            edges=[("good", "bad", None)],
        )
        diag = MetaModelDiagnostics(mm)
        with pytest.raises(FermatricaError):
            diag.run(raise_on_first=True)

    def test_raise_on_first_does_not_trigger_on_info(self):
        import fermatrica_rep.meta_model.diagnostics as diag_mod
        real_model_ref = diag_mod.Model
        diag_mod.Model = (real_model_ref, _MockModel)
        try:
            mm = _basic_meta_model()
            diag = MetaModelDiagnostics(mm)
            report = diag.run(raise_on_first=True)
            assert report.passed
        finally:
            diag_mod.Model = real_model_ref


# ---------------------------------------------------------------------------
# TestCheckNodeAttrs
# ---------------------------------------------------------------------------

class TestCheckNodeAttrs:

    def test_valid_node_passes_all_attrs(self):
        import fermatrica_rep.meta_model.diagnostics as diag_mod
        real_model_ref = diag_mod.Model
        diag_mod.Model = (real_model_ref, _MockModel)
        try:
            mm = _basic_meta_model()
            diag = MetaModelDiagnostics(mm)
            report = diag.run()
            attr_results = [r for r in report.results if r.check_name == "node_attrs"]
            error_results = [r for r in attr_results if r.level == "error"]
            assert len(error_results) == 0
        finally:
            diag_mod.Model = real_model_ref

    def test_model_none_error(self):
        ds = _basic_ds()
        node = ModelNode(model=None, ds=ds, model_rep=_basic_model_rep(ds), _display_only=True)
        mm = MetaModel(nodes=node)
        diag = MetaModelDiagnostics(mm)
        report = diag.run()
        model_results = [r for r in report.results
                         if r.check_name == "node_attrs" and r.details.get("attr") == "model"]
        assert not model_results[0].passed
        assert model_results[0].level == "error"

    def test_ds_missing_error(self):
        node = ModelNode(model=_MockModel(), ds=pd.DataFrame(), model_rep=None)
        mm = MetaModel(nodes=node)
        diag = MetaModelDiagnostics(mm)
        report = diag.run()
        ds_results = [r for r in report.results
                      if r.check_name == "node_attrs" and r.details.get("attr") == "ds"]
        assert not ds_results[0].passed
        assert ds_results[0].level == "error"

    def test_model_rep_missing_error(self):
        node = _basic_node()
        node.model_rep = None
        mm = MetaModel(nodes=node)
        diag = MetaModelDiagnostics(mm)
        report = diag.run()
        mr_results = [r for r in report.results
                      if r.check_name == "node_attrs" and r.details.get("attr") == "model_rep"]
        assert not mr_results[0].passed
        assert mr_results[0].level == "error"

    def test_vis_dict_missing_error(self):
        ds = _basic_ds()
        mr = ModelRep(ds, language="english")
        mr.vis_dict = pd.DataFrame()
        node = ModelNode(model=_MockModel(), ds=ds, model_rep=mr)
        mm = MetaModel(nodes=node)
        diag = MetaModelDiagnostics(mm)
        report = diag.run()
        vis_results = [r for r in report.results
                       if r.check_name == "node_attrs" and r.details.get("attr") == "model_rep.vis_dict"]
        assert any(not vr.passed for vr in vis_results)

    def test_trans_dict_none_error(self):
        ds = _basic_ds()
        mr = ModelRep(ds, language="english")
        mr.trans_dict = None
        node = ModelNode(model=_MockModel(), ds=ds, model_rep=mr)
        mm = MetaModel(nodes=node)
        diag = MetaModelDiagnostics(mm)
        report = diag.run()
        trans_results = [r for r in report.results
                         if r.check_name == "node_attrs" and r.details.get("attr") == "model_rep.trans_dict"]
        assert not trans_results[0].passed
        assert trans_results[0].level == "error"

    def test_static_node_trans_dict_none_is_info(self):
        ds = _basic_ds()
        mr = ModelRep(ds, language="english")
        mr.trans_dict = None
        node = ModelNode(model=_MockModel(), ds=ds, model_rep=mr, static=True)
        mm = MetaModel(nodes=node)
        diag = MetaModelDiagnostics(mm)
        report = diag.run()
        trans_results = [r for r in report.results
                         if r.check_name == "node_attrs" and r.details.get("attr") == "model_rep.trans_dict"]
        assert trans_results[0].passed
        assert trans_results[0].level == "info"

    def test_invalid_language_error(self):
        node = _basic_node()
        node.model_rep.language = "french"
        mm = MetaModel(nodes=node)
        diag = MetaModelDiagnostics(mm)
        report = diag.run()
        lang_results = [r for r in report.results
                        if r.check_name == "node_attrs" and r.details.get("attr") == "model_rep.language"]
        assert not lang_results[0].passed
        assert lang_results[0].level == "error"

    def test_dt_pred_none_info(self):
        node = _basic_node()
        node.dt_pred = None
        mm = MetaModel(nodes=node)
        diag = MetaModelDiagnostics(mm)
        report = diag.run()
        pred_results = [r for r in report.results
                        if r.check_name == "node_attrs" and r.details.get("attr") == "dt_pred"]
        assert any(r.level == "info" and r.passed for r in pred_results)

    def test_dt_pred_present_info(self):
        node = _basic_node()
        node.dt_pred = pd.DataFrame({"date": [pd.Timestamp("2025-01-01")], "predicted": [1.0]})
        mm = MetaModel(nodes=node)
        diag = MetaModelDiagnostics(mm)
        report = diag.run()
        pred_results = [r for r in report.results
                        if r.check_name == "node_attrs" and r.details.get("attr") == "dt_pred"]
        assert any("present" in r.message.lower() for r in pred_results)

    def test_dt_pred_wrong_type_warning(self):
        node = _basic_node()
        node.dt_pred = "not_a_dataframe"
        mm = MetaModel(nodes=node)
        diag = MetaModelDiagnostics(mm)
        report = diag.run()
        pred_results = [r for r in report.results
                        if r.check_name == "node_attrs" and r.details.get("attr") == "dt_pred"]
        assert any(r.level == "warning" for r in pred_results)

    def test_adhoc_code_dir_not_exists_skip(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        node = _basic_node()
        mm = MetaModel(nodes=node)
        diag = MetaModelDiagnostics(mm)
        report = diag.run()
        adhoc_results = [r for r in report.results
                         if r.check_name == "node_attrs" and r.details.get("attr") == "model_rep.adhoc_code_src"]
        assert any("no code_py/adhoc" in r.message.lower() for r in adhoc_results)

    def test_adhoc_code_present_no_src_warning(self, tmp_path, monkeypatch):
        adhoc_dir = tmp_path / "code_py" / "adhoc"
        adhoc_dir.mkdir(parents=True)
        (adhoc_dir / "translator_custom.py").write_text("def adhoc_test(): pass\nprint('hello')")
        monkeypatch.chdir(tmp_path)
        node = _basic_node()
        mm = MetaModel(nodes=node)
        diag = MetaModelDiagnostics(mm)
        report = diag.run()
        adhoc_results = [r for r in report.results
                         if r.check_name == "node_attrs" and r.details.get("attr") == "model_rep.adhoc_code_src"]
        assert any(r.level == "warning" for r in adhoc_results)
        assert any("adhoc_code_src is empty" in r.message.lower() for r in adhoc_results)

    def test_adhoc_code_present_with_src_passes(self, tmp_path, monkeypatch):
        adhoc_dir = tmp_path / "code_py" / "adhoc"
        adhoc_dir.mkdir(parents=True)
        (adhoc_dir / "translator_custom.py").write_text("def adhoc_test(): pass\nprint('hello')")
        monkeypatch.chdir(tmp_path)

        def _dummy_fn():
            pass

        ds = _basic_ds()
        mr = ModelRep(ds, language="english", adhoc_code=[_dummy_fn])
        node = ModelNode(model=_MockModel(), ds=ds, model_rep=mr)
        mm = MetaModel(nodes=node)
        diag = MetaModelDiagnostics(mm)
        report = diag.run()
        adhoc_results = [r for r in report.results
                         if r.check_name == "node_attrs" and r.details.get("attr") == "model_rep.adhoc_code_src"]
        assert any("populated" in r.message.lower() for r in adhoc_results)

    def test_adhoc_only_comment_skip(self, tmp_path, monkeypatch):
        adhoc_dir = tmp_path / "code_py" / "adhoc"
        adhoc_dir.mkdir(parents=True)
        (adhoc_dir / "reporting_adhoc.py").write_text("# just a comment\n# nothing here")
        monkeypatch.chdir(tmp_path)
        node = _basic_node()
        mm = MetaModel(nodes=node)
        diag = MetaModelDiagnostics(mm)
        report = diag.run()
        adhoc_results = [r for r in report.results
                         if r.check_name == "node_attrs" and r.details.get("attr") == "model_rep.adhoc_code_src"]
        assert any("no non-comment code" in r.message.lower() for r in adhoc_results)


# ---------------------------------------------------------------------------
# TestCheckEdgeResolution
# ---------------------------------------------------------------------------

class TestCheckEdgeResolution:

    def test_single_node_no_edges_info(self):
        mm = _basic_meta_model(n_nodes=1)
        diag = MetaModelDiagnostics(mm)
        report = diag.run()
        edge_results = [r for r in report.results if r.check_name == "edge_resolution"]
        assert all(r.passed for r in edge_results)
        assert any("no edges" in r.message.lower() for r in edge_results)

    def test_none_merge_fn_info(self):
        ds = _basic_ds()
        mr = _basic_model_rep(ds)
        nodes = {
            "a": ModelNode(model=_MockModel("a"), ds=ds, model_rep=mr),
            "b": ModelNode(model=_MockModel("b"), ds=ds, model_rep=mr),
        }
        mm = MetaModel(nodes=nodes, edges=[("a", "b", None)])
        diag = MetaModelDiagnostics(mm)
        report = diag.run()
        edge_results = [r for r in report.results if r.check_name == "edge_resolution"]
        assert all(r.passed for r in edge_results)
        assert any("parallel branch" in r.message.lower() for r in edge_results)

    def test_callable_merge_fn_info(self):
        ds = _basic_ds()
        mr = _basic_model_rep(ds)

        def my_merge(ds_cur, dt_pred_prev):
            return ds_cur

        nodes = {
            "a": ModelNode(model=_MockModel("a"), ds=ds, model_rep=mr),
            "b": ModelNode(model=_MockModel("b"), ds=ds, model_rep=mr),
        }
        mm = MetaModel(nodes=nodes, edges=[("a", "b", my_merge)])
        diag = MetaModelDiagnostics(mm)
        report = diag.run()
        edge_results = [r for r in report.results if r.check_name == "edge_resolution"]
        assert all(r.passed for r in edge_results)

    def test_string_resolves_via_eval(self, monkeypatch):
        import fermatrica_rep.meta_model.diagnostics as diag_mod

        def _test_merge(ds_cur, dt_pred_prev):
            return ds_cur

        monkeypatch.setattr(diag_mod, "_test_merge", _test_merge, raising=False)

        ds = _basic_ds()
        mr = _basic_model_rep(ds)
        nodes = {
            "a": ModelNode(model=_MockModel("a"), ds=ds, model_rep=mr),
            "b": ModelNode(model=_MockModel("b"), ds=ds, model_rep=mr),
        }
        mm = MetaModel(nodes=nodes, edges=[("a", "b", "_test_merge")])
        diag = MetaModelDiagnostics(mm)
        report = diag.run()
        edge_results = [r for r in report.results if r.check_name == "edge_resolution"]
        assert any("resolves OK" in r.message for r in edge_results)

    def test_string_unresolvable_error(self):
        ds = _basic_ds()
        mr = _basic_model_rep(ds)
        nodes = {
            "a": ModelNode(model=_MockModel("a"), ds=ds, model_rep=mr),
            "b": ModelNode(model=_MockModel("b"), ds=ds, model_rep=mr),
        }
        mm = MetaModel(nodes=nodes, edges=[("a", "b", "nonexistent_merge_fn_xyzzy")])
        diag = MetaModelDiagnostics(mm)
        report = diag.run()
        edge_results = [r for r in report.results if r.check_name == "edge_resolution"]
        unresolved = [r for r in edge_results if not r.passed]
        assert len(unresolved) > 0
        assert any("could not be resolved" in r.message.lower() for r in unresolved)

    def test_non_callable_non_string_error(self):
        ds = _basic_ds()
        mr = _basic_model_rep(ds)
        nodes = {
            "a": ModelNode(model=_MockModel("a"), ds=ds, model_rep=mr),
            "b": ModelNode(model=_MockModel("b"), ds=ds, model_rep=mr),
        }
        mm = MetaModel(nodes=nodes, edges=[("a", "b", 123)])
        diag = MetaModelDiagnostics(mm)
        report = diag.run()
        edge_results = [r for r in report.results if r.check_name == "edge_resolution"]
        assert any("not callable or str" in r.message for r in edge_results)

    def test_cannot_inspect_signature_warning(self, monkeypatch):
        import fermatrica_rep.meta_model.diagnostics as diag_mod

        class _BadSig:
            pass

        monkeypatch.setattr(diag_mod, "_bad_sig", _BadSig(), raising=False)

        ds = _basic_ds()
        mr = _basic_model_rep(ds)
        nodes = {
            "a": ModelNode(model=_MockModel("a"), ds=ds, model_rep=mr),
            "b": ModelNode(model=_MockModel("b"), ds=ds, model_rep=mr),
        }

        def _bad_fn(ds_cur, dt_pred_prev):
            return ds_cur

        original_sig = __import__("inspect").signature
        call_count = [0]

        def _fake_signature(fn):
            call_count[0] += 1
            if call_count[0] == 1:
                raise ValueError("cannot inspect")
            return original_sig(fn)

        monkeypatch.setattr("fermatrica_rep.meta_model.diagnostics.inspect.signature", _fake_signature)

        mm = MetaModel(nodes=nodes, edges=[("a", "b", _bad_fn)])
        diag = MetaModelDiagnostics(mm)
        report = diag.run()
        edge_results = [r for r in report.results if r.check_name == "edge_resolution"]
        warnings = [r for r in edge_results if r.level == "warning"]
        assert any("cannot inspect" in r.message for r in warnings)


# ---------------------------------------------------------------------------
# TestCheckMergeEffects
# ---------------------------------------------------------------------------

class TestCheckMergeEffects:

    def _make_diag_nodes(self):
        ds = _basic_ds()
        mr = _basic_model_rep(ds)
        return {
            "awareness": ModelNode(model=_MockModel("awa"), ds=ds.copy(), model_rep=mr),
            "sales": ModelNode(model=_MockModel("sal"), ds=ds.copy(), model_rep=mr),
        }

    def _make_scopes(self):
        return [OptionScope(
            scope_name="s1", date_start="2025-01-03", date_end="2025-01-06",
            ref_date_start="2025-01-03", ref_date_end="2025-01-06",
        )]

    def _make_options(self):
        return [
            Option(name="opt_a", budget_data={"s1": {"bdg_tv": 10}}),
            Option(name="opt_b", budget_data={"s1": {"bdg_tv": 20}}),
        ]

    def test_identical_options_skipped(self):
        scopes = self._make_scopes()
        opt_a = Option(name="opt_a", budget_data={"s1": {"bdg_tv": 10}})
        opt_b = Option(name="opt_b", budget_data={"s1": {"bdg_tv": 10}})
        nodes = self._make_diag_nodes()
        mm = MetaModel(nodes=nodes, edges=[("awareness", "sales", None)])
        diag = MetaModelDiagnostics(mm)
        report = diag.run(options=[opt_a, opt_b], scopes=scopes)
        assert any("identical" in s for s in report.checks_skipped)

    def test_options_dict_behaves_like_list(self, monkeypatch):
        scopes = self._make_scopes()
        options = self._make_options()
        nodes = self._make_diag_nodes()
        mm = MetaModel(nodes=nodes, edges=[("awareness", "sales", None)])

        compute_calls = []

        def fake_compute(self, option, scopes, **kwargs):
            compute_calls.append(option.name)
            for name, node in self.nodes.items():
                node.ds.loc[node.ds["listed"] == 4, "predicted"] += (
                    10 if option.name == "opt_a" else 20
                )
                node.dt_pred = node.ds.copy()

        monkeypatch.setattr(
            "fermatrica_rep.meta_model.meta_model.MetaModel.compute", fake_compute
        )

        diag = MetaModelDiagnostics(mm)
        report = diag.run(
            options={"a": options[0], "b": options[1]}, scopes=scopes
        )

        assert compute_calls == ["opt_a", "opt_b"]
        assert any(
            "differs" in r.message.lower()
            for r in report.results
            if r.check_name == "merge_dt_pred" and r.level == "info"
        )

    def test_options_dict_skipped_when_few(self):
        scopes = self._make_scopes()
        nodes = self._make_diag_nodes()
        mm = MetaModel(nodes=nodes, edges=[("awareness", "sales", None)])
        diag = MetaModelDiagnostics(mm)
        report = diag.run(options={"only": Option(name="opt_a", budget_data={"s1": {"bdg_tv": 10}})},
                          scopes=scopes)
        assert any("need at least 2 options" in s for s in report.checks_skipped)

    def test_merge_effects_detects_propagation(self, monkeypatch):
        scopes = self._make_scopes()
        options = self._make_options()
        nodes = self._make_diag_nodes()
        mm = MetaModel(nodes=nodes, edges=[("awareness", "sales", None)])

        compute_calls = []

        def fake_compute(self, option, scopes, **kwargs):
            compute_calls.append(option.name)
            for name, node in self.nodes.items():
                node.ds.loc[node.ds["listed"] == 4, "predicted"] += (
                    10 if option.name == "opt_a" else 20
                )
                node.dt_pred = node.ds.copy()

        monkeypatch.setattr(
            "fermatrica_rep.meta_model.meta_model.MetaModel.compute", fake_compute
        )

        diag = MetaModelDiagnostics(mm)
        report = diag.run(options=options, scopes=scopes)

        assert len(compute_calls) == 2
        assert compute_calls == ["opt_a", "opt_b"]

        ds_merge_results = [r for r in report.results
                            if r.check_name == "merge_ds" and r.level == "info"
                            and "differs" in r.message.lower()]
        pred_merge_results = [r for r in report.results
                              if r.check_name == "merge_dt_pred" and r.level == "info"
                              and "differs" in r.message.lower()]
        assert len(ds_merge_results) > 0
        assert len(pred_merge_results) > 0

    def test_merge_effects_restores_state_on_exception(self, monkeypatch):
        scopes = self._make_scopes()
        options = self._make_options()
        nodes = self._make_diag_nodes()
        mm = MetaModel(nodes=nodes, edges=[("awareness", "sales", None)])

        original_ds = {name: node.ds.copy() for name, node in mm.nodes.items()}
        original_dt_pred = {name: node.dt_pred for name, node in mm.nodes.items()}

        def failing_compute(self, option, scopes, **kwargs):
            for name, node in self.nodes.items():
                node.ds.loc[0, "predicted"] = -999
                node.dt_pred = node.ds.copy()
            raise RuntimeError("simulated crash")

        monkeypatch.setattr(
            "fermatrica_rep.meta_model.meta_model.MetaModel.compute", failing_compute
        )

        diag = MetaModelDiagnostics(mm)

        with pytest.raises(RuntimeError, match="simulated crash"):
            diag.run(options=options, scopes=scopes)

        for name, node in mm.nodes.items():
            pd.testing.assert_frame_equal(node.ds, original_ds[name])
            assert node.dt_pred is original_dt_pred[name]

    def test_merge_effects_no_comparison_when_prediction_none(self, monkeypatch):
        scopes = self._make_scopes()
        options = self._make_options()
        nodes = self._make_diag_nodes()

        for node in nodes.values():
            node.dt_pred = None

        mm = MetaModel(nodes=nodes, edges=[("awareness", "sales", None)])

        def fake_compute(self, option, scopes, **kwargs):
            for name, node in self.nodes.items():
                node.dt_pred = None

        monkeypatch.setattr(
            "fermatrica_rep.meta_model.meta_model.MetaModel.compute", fake_compute
        )

        diag = MetaModelDiagnostics(mm)
        report = diag.run(options=options, scopes=scopes)

        none_results = [r for r in report.results
                        if ("merge_ds" in r.check_name or "merge_dt_pred" in r.check_name)
                        and r.level == "info"]
        assert any("one or both DataFrames are None" in r.message for r in none_results)

    def test_merge_all_columns_identical_error(self, monkeypatch):
        scopes = self._make_scopes()
        options = self._make_options()
        nodes = self._make_diag_nodes()
        mm = MetaModel(nodes=nodes, edges=[("awareness", "sales", None)])

        def fake_compute(self, option, scopes, **kwargs):
            for name, node in self.nodes.items():
                node.dt_pred = node.ds.copy()

        monkeypatch.setattr(
            "fermatrica_rep.meta_model.meta_model.MetaModel.compute", fake_compute
        )

        diag = MetaModelDiagnostics(mm)
        report = diag.run(options=options, scopes=scopes)

        err_results = [r for r in report.results
                       if r.level == "error" and "unchanged" in r.message.lower()]
        assert len(err_results) > 0

    def test_static_node_unchanged_is_info_not_error(self, monkeypatch):
        scopes = self._make_scopes()
        options = self._make_options()
        nodes = self._make_diag_nodes()
        nodes["awareness"].static = True
        mm = MetaModel(nodes=nodes, edges=[("awareness", "sales", None)])

        def fake_compute(self, option, scopes, **kwargs):
            for name, node in self.nodes.items():
                if node.static:
                    node.dt_pred = node.ds.copy()
                else:
                    delta = 10 if option.name == "opt_a" else 20
                    node.ds.loc[node.ds["listed"] == 4, "predicted"] += delta
                    node.dt_pred = node.ds.copy()

        monkeypatch.setattr(
            "fermatrica_rep.meta_model.meta_model.MetaModel.compute", fake_compute
        )

        diag = MetaModelDiagnostics(mm)
        report = diag.run(options=options, scopes=scopes)

        err_results = [r for r in report.results
                       if r.level == "error" and "unchanged" in r.message.lower()]
        assert len(err_results) == 0
        info_results = [r for r in report.results
                        if r.level == "info" and "unchanged between options" in r.message.lower()
                        and r.details.get("node_name") == "awareness"]
        assert len(info_results) == 2

    def test_static_node_with_option_influenced_upstream_keeps_error(self, monkeypatch):
        scopes = self._make_scopes()
        options = self._make_options()
        nodes = self._make_diag_nodes()
        nodes["sales"].static = True
        mm = MetaModel(nodes=nodes, edges=[("awareness", "sales", None)])

        def fake_compute(self, option, scopes, **kwargs):
            for name, node in self.nodes.items():
                if not node.static:
                    delta = 10 if option.name == "opt_a" else 20
                    node.ds.loc[node.ds["listed"] == 4, "predicted"] += delta
                    node.dt_pred = node.ds.copy()
                else:
                    node.dt_pred = node.ds.copy()

        monkeypatch.setattr(
            "fermatrica_rep.meta_model.meta_model.MetaModel.compute", fake_compute
        )

        diag = MetaModelDiagnostics(mm)
        report = diag.run(options=options, scopes=scopes)

        err_results = [r for r in report.results
                       if r.level == "error" and "unchanged" in r.message.lower()
                       and r.details.get("node_name") == "sales"]
        assert len(err_results) > 0

    def test_float_columns_diff_warning(self, monkeypatch):
        scopes = self._make_scopes()
        options = self._make_options()
        nodes = self._make_diag_nodes()
        mm = MetaModel(nodes=nodes, edges=[("awareness", "sales", None)])

        call_count = [0]

        def fake_compute(self, option, scopes, **kwargs):
            for name, node in self.nodes.items():
                node.dt_pred = node.ds.copy()
                if call_count[0] == 1:
                    node.dt_pred["predicted"] = node.dt_pred["predicted"] + 100.0
            call_count[0] += 1

        monkeypatch.setattr(
            "fermatrica_rep.meta_model.meta_model.MetaModel.compute", fake_compute
        )

        diag = MetaModelDiagnostics(mm)
        report = diag.run(options=options, scopes=scopes)

        float_diff_results = [r for r in report.results
                              if r.level == "info" and "float columns differ" in r.message.lower()]
        assert len(float_diff_results) > 0

    def test_non_float_columns_diff_warning(self, monkeypatch):
        scopes = self._make_scopes()
        options = self._make_options()
        nodes = self._make_diag_nodes()
        mm = MetaModel(nodes=nodes, edges=[("awareness", "sales", None)])

        call_count = [0]

        def fake_compute(self, option, scopes, **kwargs):
            for name, node in self.nodes.items():
                node.dt_pred = node.ds.copy()
                if call_count[0] == 1:
                    node.dt_pred["superbrand"] = "changed"
            call_count[0] += 1

        monkeypatch.setattr(
            "fermatrica_rep.meta_model.meta_model.MetaModel.compute", fake_compute
        )

        diag = MetaModelDiagnostics(mm)
        report = diag.run(options=options, scopes=scopes)

        non_float_results = [r for r in report.results
                             if r.level == "info" and "non-float columns differ" in r.message.lower()]
        assert len(non_float_results) > 0

    def test_single_node_merge_effects(self, monkeypatch):
        scopes = self._make_scopes()
        options = self._make_options()
        mm = _basic_meta_model(n_nodes=1)

        call_count = [0]

        def fake_compute(self, option, scopes, **kwargs):
            for name, node in self.nodes.items():
                node.dt_pred = node.ds.copy()
                node.dt_pred["predicted"] = node.dt_pred["predicted"] + (call_count[0] * 50.0)
            call_count[0] += 1

        monkeypatch.setattr(
            "fermatrica_rep.meta_model.meta_model.MetaModel.compute", fake_compute
        )

        diag = MetaModelDiagnostics(mm)
        report = diag.run(options=options, scopes=scopes)

        pred_diff = [r for r in report.results
                     if r.check_name == "merge_dt_pred" and r.level == "info"
                     and "differs" in r.message.lower()]
        assert len(pred_diff) > 0


# ---------------------------------------------------------------------------
# TestCheckMergeEdgePropagation — per-edge signal attribution
# ---------------------------------------------------------------------------

class TestCheckMergeEdgePropagation:

    def _make_nodes(self, n=2):
        ds = _basic_ds()
        mr = _basic_model_rep(ds)
        return {
            f"n{i}": ModelNode(model=_MockModel(f"n{i}"), ds=ds.copy(), model_rep=mr)
            for i in range(n)
        }

    def _make_scopes(self):
        return [OptionScope(
            scope_name="s1", date_start="2025-01-03", date_end="2025-01-06",
            ref_date_start="2025-01-03", ref_date_end="2025-01-06",
        )]

    def _make_options(self):
        return [
            Option(name="opt_a", budget_data={"s1": {"bdg_tv": 10}}),
            Option(name="opt_b", budget_data={"s1": {"bdg_tv": 20}}),
        ]

    def _patch_compute(self, monkeypatch, offset_a=10.0, offset_b=20.0):
        def fake_compute(self, option, scopes, **kwargs):
            offset = offset_a if option.name == "opt_a" else offset_b
            for name, node in self.nodes.items():
                node.ds = node.ds.copy()
                node.ds["predicted"] = node.ds["predicted"] + offset
                node.dt_pred = node.ds.copy()
        monkeypatch.setattr(
            "fermatrica_rep.meta_model.meta_model.MetaModel.compute", fake_compute
        )

    def _working_merge(self, ds_cur, dt_pred_prev, allow_past=False):
        tmp = dt_pred_prev.loc[:, ["date", "predicted"]].copy()
        return pd.merge(ds_cur, tmp, how="left", on="date", sort=False, suffixes=("", "_prev"))

    def _edge_results(self, report, left=None, right=None):
        out = [r for r in report.results if r.check_name == "merge_edge"]
        if left is not None:
            out = [r for r in out if r.details.get("left") == left]
        if right is not None:
            out = [r for r in out if r.details.get("right") == right]
        return out

    def test_propagating_merge_reports_info(self, monkeypatch):
        nodes = self._make_nodes(2)
        mm = MetaModel(nodes=nodes, edges=[("n0", "n1", self._working_merge)])
        self._patch_compute(monkeypatch)
        diag = MetaModelDiagnostics(mm)
        report = diag.run(options=self._make_options(), scopes=self._make_scopes())
        results = self._edge_results(report, "n0", "n1")
        assert any(r.level == "info" and "propagates" in r.message for r in results)

    def test_noop_merge_reports_error(self, monkeypatch):
        nodes = self._make_nodes(2)

        def noop_merge(ds_cur, dt_pred_prev, allow_past=False):
            return ds_cur

        mm = MetaModel(nodes=nodes, edges=[("n0", "n1", noop_merge)])
        self._patch_compute(monkeypatch)
        diag = MetaModelDiagnostics(mm)
        report = diag.run(options=self._make_options(), scopes=self._make_scopes())
        results = self._edge_results(report, "n0", "n1")
        assert any(r.level == "error" and not r.passed and "no output change" in r.message
                   for r in results)
        assert not report.passed

    def test_constant_merge_reports_warning(self, monkeypatch):
        nodes = self._make_nodes(2)

        def constant_merge(ds_cur, dt_pred_prev, allow_past=False):
            out = ds_cur.copy()
            out["merged_const"] = 5.0
            return out

        mm = MetaModel(nodes=nodes, edges=[("n0", "n1", constant_merge)])
        self._patch_compute(monkeypatch)
        diag = MetaModelDiagnostics(mm)
        report = diag.run(options=self._make_options(), scopes=self._make_scopes())
        results = self._edge_results(report, "n0", "n1")
        assert any(r.level == "warning" and "writes columns" in r.message for r in results)

    def test_no_upstream_signal_info(self, monkeypatch):
        nodes = self._make_nodes(2)
        mm = MetaModel(nodes=nodes, edges=[("n0", "n1", self._working_merge)])
        self._patch_compute(monkeypatch, offset_a=10.0, offset_b=10.0)
        diag = MetaModelDiagnostics(mm)
        report = diag.run(options=self._make_options(), scopes=self._make_scopes())
        results = self._edge_results(report, "n0", "n1")
        assert any("not exercisable" in r.message for r in results)

    def test_left_dt_pred_none_info(self, monkeypatch):
        nodes = self._make_nodes(2)
        mm = MetaModel(nodes=nodes, edges=[("n0", "n1", self._working_merge)])

        def fake_compute(self, option, scopes, **kwargs):
            for name, node in self.nodes.items():
                if name == "n0":
                    node.dt_pred = None
                else:
                    node.dt_pred = node.ds.copy()

        monkeypatch.setattr(
            "fermatrica_rep.meta_model.meta_model.MetaModel.compute", fake_compute
        )
        diag = MetaModelDiagnostics(mm)
        report = diag.run(options=self._make_options(), scopes=self._make_scopes())
        results = self._edge_results(report, "n0", "n1")
        assert any("cannot test" in r.message for r in results)

    def test_parallel_edge_no_results(self, monkeypatch):
        nodes = self._make_nodes(2)
        mm = MetaModel(nodes=nodes, edges=[("n0", "n1", None)])
        self._patch_compute(monkeypatch)
        diag = MetaModelDiagnostics(mm)
        report = diag.run(options=self._make_options(), scopes=self._make_scopes())
        assert self._edge_results(report) == []

    def test_unresolvable_string_merge_fn_info(self, monkeypatch):
        nodes = self._make_nodes(2)
        mm = MetaModel(nodes=nodes, edges=[("n0", "n1", "nonexistent_probe_fn_xyzzy")])
        self._patch_compute(monkeypatch)
        diag = MetaModelDiagnostics(mm)
        report = diag.run(options=self._make_options(), scopes=self._make_scopes())
        results = self._edge_results(report, "n0", "n1")
        assert any("not resolvable" in r.message for r in results)

    def test_string_merge_fn_resolved(self, monkeypatch):
        import fermatrica_rep.meta_model.diagnostics as diag_mod

        def _test_merge_probe(ds_cur, dt_pred_prev, allow_past=False):
            tmp = dt_pred_prev.loc[:, ["date", "predicted"]].copy()
            return pd.merge(ds_cur, tmp, how="left", on="date", sort=False, suffixes=("", "_prev"))

        monkeypatch.setattr(diag_mod, "_test_merge_probe", _test_merge_probe, raising=False)

        nodes = self._make_nodes(2)
        mm = MetaModel(nodes=nodes, edges=[("n0", "n1", "_test_merge_probe")])
        self._patch_compute(monkeypatch)
        diag = MetaModelDiagnostics(mm)
        report = diag.run(options=self._make_options(), scopes=self._make_scopes())
        results = self._edge_results(report, "n0", "n1")
        assert any("propagates" in r.message for r in results)

    def test_fan_in_broken_edge_detected(self, monkeypatch):
        nodes = self._make_nodes(3)
        edges = [
            ("n0", "n2", self._working_merge),
            ("n1", "n2", lambda ds_cur, dt_pred_prev, allow_past=False: ds_cur),
        ]
        mm = MetaModel(nodes=nodes, edges=edges)
        self._patch_compute(monkeypatch)
        diag = MetaModelDiagnostics(mm)
        report = diag.run(options=self._make_options(), scopes=self._make_scopes())

        good = self._edge_results(report, "n0", "n2")
        bad = self._edge_results(report, "n1", "n2")
        assert any(r.level == "info" and "propagates" in r.message for r in good)
        assert any(r.level == "error" and not r.passed and "no output change" in r.message
                   for r in bad)
        assert not report.passed

        node_level = [r for r in report.results
                      if r.check_name == "merge_ds" and r.details.get("node_name") == "n2"]
        assert any(r.level == "info" and "differs" in r.message.lower() for r in node_level)

    def test_dotted_adhoc_merge_string_resolved(self, monkeypatch):
        import types
        import fermatrica_rep.meta_model.meta_model as mm_mod

        def _adhoc_merge(ds_cur, dt_pred_prev, allow_past=False):
            tmp = dt_pred_prev.loc[:, ["date", "predicted"]].copy()
            return pd.merge(ds_cur, tmp, how="left", on="date", sort=False, suffixes=("", "_prev"))

        fake_mod = types.ModuleType("code_py.adhoc.reporting")
        fake_mod.pred_merge_probe = _adhoc_merge
        monkeypatch.setitem(mm_mod.__dict__, "code_py.adhoc.reporting", fake_mod)

        nodes = self._make_nodes(2)
        mm = MetaModel(
            nodes=nodes,
            edges=[("n0", "n1", "code_py.adhoc.reporting.pred_merge_probe")],
        )
        self._patch_compute(monkeypatch)
        diag = MetaModelDiagnostics(mm)
        report = diag.run(options=self._make_options(), scopes=self._make_scopes())

        edge_res = [r for r in report.results if r.check_name == "edge_resolution"]
        assert any("resolves OK" in r.message for r in edge_res)

        results = self._edge_results(report, "n0", "n1")
        assert any("propagates" in r.message for r in results)
