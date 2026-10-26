"""
MetaModelDiagnostics — validates a MetaModel object on demand.

Checks:
  1. Node types — all nodes in the DAG are ModelNode instances
  2. Node attributes — critical attributes are present and valid
  3. Edge resolution — merge functions resolve correctly
  4. Merge effects — ds/dt_pred differ between two compute runs;
     per-edge propagation — each merge carries the upstream signal
"""


import dataclasses
import inspect
import os
import sys
import glob as glob_module

import numpy as np
import pandas as pd

from fermatrica import Model, fermatrica_error
from fermatrica_rep.meta_model.model_rep import ModelRep
from fermatrica_rep.meta_model.node import ModelNode
from fermatrica_rep.meta_model.meta_model import MetaModel
from fermatrica_rep.options.scope import OptionScope


@dataclasses.dataclass
class CheckResult:
    check_name: str
    level: str
    passed: bool
    message: str
    details: dict | None = None


@dataclasses.dataclass
class DiagnosticsReport:
    results: list[CheckResult]
    checks_ran: list[str]
    checks_skipped: list[str]

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results if r.level == "error")

    @property
    def errors(self) -> list[CheckResult]:
        return [r for r in self.results if r.level == "error"]

    @property
    def warnings(self) -> list[CheckResult]:
        return [r for r in self.results if r.level == "warning"]

    @property
    def infos(self) -> list[CheckResult]:
        return [r for r in self.results if r.level == "info"]

    @property
    def errors_summary(self) -> str:
        error_list = self.errors
        if not error_list:
            return ""

        lines = [f"  [FAIL] {r.check_name}: {r.message}" for r in error_list]
        lines.append("")
        lines.append(f"  Errors: {len(error_list)}")
        return "\n".join(lines)

    @property
    def warnings_summary(self) -> str:
        warn_list = self.warnings
        if not warn_list:
            return ""

        lines = [f"  [WARN] {r.check_name}: {r.message}" for r in warn_list]
        lines.append("")
        lines.append(f"  Warnings: {len(warn_list)}")
        return "\n".join(lines)

    @property
    def summary(self) -> str:
        errors = [r for r in self.results if r.level == "error"]
        warnings = [r for r in self.results if r.level == "warning"]
        infos = [r for r in self.results if r.level == "info"]

        lines = []
        for r in self.results:
            if r.passed:
                lines.append(f"  [PASS] {r.check_name}: {r.message}")
            elif r.level == "warning":
                lines.append(f"  [WARN] {r.check_name}: {r.message}")
            else:
                lines.append(f"  [FAIL] {r.check_name}: {r.message}")

        lines.append("")
        skipped_str = ", ".join(self.checks_skipped) if self.checks_skipped else "none"
        lines.append(f"  Errors: {len(errors)}, Warnings: {len(warnings)}, Info: {len(infos)}")
        lines.append(f"  Skipped: {skipped_str}")

        return "\n".join(lines)


class MetaModelDiagnostics:

    def __init__(self, meta_model: MetaModel):
        self.meta_model = meta_model

    def run(
        self,
        options: list | dict | None = None,
        scopes: list[OptionScope] | None = None,
        raise_on_first: bool = False,
    ) -> DiagnosticsReport:
        """
        Run all diagnostic checks and collect the results.

        ``options`` accepts either a ``list`` of ``Option`` objects or a
        ``dict`` (e.g. ``ctx.options``) — dict values are used in
        insertion order.
        """
        if isinstance(options, dict):
            options = list(options.values())

        report = DiagnosticsReport(results=[], checks_ran=[], checks_skipped=[])

        self._check_node_types(report, raise_on_first)
        self._check_node_attrs(report, raise_on_first)
        self._check_edge_resolution(report, raise_on_first)

        if options is not None and scopes is not None:
            if len(options) >= 2:
                self._check_merge_effects(report, options, scopes, raise_on_first)
            else:
                report.checks_skipped.append(
                    "merge_effect: need at least 2 options (got %d)" % len(options)
                )
        else:
            missing = []
            if options is None:
                missing.append("options")
            if scopes is None:
                missing.append("scopes")
            report.checks_skipped.append("merge_effect: missing %s" % ", ".join(missing))

        return report

    def _add_result(self, report: DiagnosticsReport, result: CheckResult, raise_on_first: bool):
        report.results.append(result)
        if not result.passed and result.level == "error" and raise_on_first:
            fermatrica_error(result.message)

    def _check_node_types(self, report: DiagnosticsReport, raise_on_first: bool):
        report.checks_ran.append("node_type")

        for name, node in self.meta_model.nodes.items():
            if isinstance(node, ModelNode):
                self._add_result(
                    report,
                    CheckResult(
                        check_name="node_type",
                        level="info",
                        passed=True,
                        message=f"node '{name}' is a valid ModelNode",
                        details={"node_name": name},
                    ),
                    raise_on_first,
                )
            else:
                self._add_result(
                    report,
                    CheckResult(
                        check_name="node_type",
                        level="error",
                        passed=False,
                        message=f"node '{name}' is not a ModelNode — found {type(node).__name__}",
                        details={"node_name": name, "type_found": type(node).__name__},
                    ),
                    raise_on_first,
                )

    def _check_node_attrs(self, report: DiagnosticsReport, raise_on_first: bool):
        report.checks_ran.append("node_attrs")

        for name, node in self.meta_model.nodes.items():
            if not isinstance(node, ModelNode):
                self._add_result(
                    report,
                    CheckResult(
                        check_name="node_attrs",
                        level="error",
                        passed=False,
                        message=f"node '{name}' is not a ModelNode — cannot check attributes",
                        details={"node_name": name, "type_found": type(node).__name__},
                    ),
                    raise_on_first,
                )
                continue

            if node.model is not None and isinstance(node.model, Model):
                self._add_result(
                    report,
                    CheckResult(
                        check_name="node_attrs",
                        level="info",
                        passed=True,
                        message=f"node '{name}' model is valid",
                        details={"node_name": name, "attr": "model"},
                    ),
                    raise_on_first,
                )
            else:
                self._add_result(
                    report,
                    CheckResult(
                        check_name="node_attrs",
                        level="error",
                        passed=False,
                        message=f"node '{name}' model is None or not a Model instance",
                        details={
                            "node_name": name,
                            "attr": "model",
                            "type_found": type(node.model).__name__ if node.model is not None else "None",
                        },
                    ),
                    raise_on_first,
                )

            if isinstance(node.ds, pd.DataFrame) and len(node.ds) > 0:
                self._add_result(
                    report,
                    CheckResult(
                        check_name="node_attrs",
                        level="info",
                        passed=True,
                        message=f"node '{name}' ds is valid ({len(node.ds)} rows)",
                        details={"node_name": name, "attr": "ds"},
                    ),
                    raise_on_first,
                )
            else:
                self._add_result(
                    report,
                    CheckResult(
                        check_name="node_attrs",
                        level="error",
                        passed=False,
                        message=f"node '{name}' ds is missing, empty, or not a DataFrame",
                        details={"node_name": name, "attr": "ds"},
                    ),
                    raise_on_first,
                )

            if isinstance(node.model_rep, ModelRep):
                self._add_result(
                    report,
                    CheckResult(
                        check_name="node_attrs",
                        level="info",
                        passed=True,
                        message=f"node '{name}' model_rep is valid",
                        details={"node_name": name, "attr": "model_rep"},
                    ),
                    raise_on_first,
                )
            else:
                self._add_result(
                    report,
                    CheckResult(
                        check_name="node_attrs",
                        level="error",
                        passed=False,
                        message=f"node '{name}' model_rep is missing or not a ModelRep",
                        details={"node_name": name, "attr": "model_rep"},
                    ),
                    raise_on_first,
                )
                return

            mr = node.model_rep

            if isinstance(mr.vis_dict, pd.DataFrame) and len(mr.vis_dict) > 0:
                self._add_result(
                    report,
                    CheckResult(
                        check_name="node_attrs",
                        level="info",
                        passed=True,
                        message=f"node '{name}' vis_dict is valid",
                        details={"node_name": name, "attr": "model_rep.vis_dict"},
                    ),
                    raise_on_first,
                )
            else:
                self._add_result(
                    report,
                    CheckResult(
                        check_name="node_attrs",
                        level="error",
                        passed=False,
                        message=f"node '{name}' vis_dict is missing, empty, or not a DataFrame",
                        details={"node_name": name, "attr": "model_rep.vis_dict"},
                    ),
                    raise_on_first,
                )

            if mr.trans_dict is None:
                if node.static:
                    self._add_result(
                        report,
                        CheckResult(
                            check_name="node_attrs",
                            level="info",
                            passed=True,
                            message=f"node '{name}' is static — trans_dict not required (translation skipped)",
                            details={"node_name": name, "attr": "model_rep.trans_dict"},
                        ),
                        raise_on_first,
                    )
                else:
                    self._add_result(
                        report,
                        CheckResult(
                            check_name="node_attrs",
                            level="error",
                            passed=False,
                            message=f"node '{name}' trans_dict is None — options/forecasts won't work",
                            details={"node_name": name, "attr": "model_rep.trans_dict"},
                        ),
                        raise_on_first,
                    )
            else:
                self._add_result(
                    report,
                    CheckResult(
                        check_name="node_attrs",
                        level="info",
                        passed=True,
                        message=f"node '{name}' trans_dict is present",
                        details={"node_name": name, "attr": "model_rep.trans_dict"},
                    ),
                    raise_on_first,
                )

            if mr.language in ("english", "russian"):
                self._add_result(
                    report,
                    CheckResult(
                        check_name="node_attrs",
                        level="info",
                        passed=True,
                        message=f"node '{name}' language is '{mr.language}'",
                        details={"node_name": name, "attr": "model_rep.language"},
                    ),
                    raise_on_first,
                )
            else:
                self._add_result(
                    report,
                    CheckResult(
                        check_name="node_attrs",
                        level="error",
                        passed=False,
                        message=f"node '{name}' language is '{mr.language}' — expected 'english' or 'russian'",
                        details={"node_name": name, "attr": "model_rep.language"},
                    ),
                    raise_on_first,
                )

            self._check_adhoc_code(report, name, mr, raise_on_first)

            if node.dt_pred is None:
                self._add_result(
                    report,
                    CheckResult(
                        check_name="node_attrs",
                        level="info",
                        passed=True,
                        message=f"node '{name}' dt_pred is None — prediction not yet computed",
                        details={"node_name": name, "attr": "dt_pred"},
                    ),
                    raise_on_first,
                )
            elif isinstance(node.dt_pred, pd.DataFrame):
                self._add_result(
                    report,
                    CheckResult(
                        check_name="node_attrs",
                        level="info",
                        passed=True,
                        message=f"node '{name}' dt_pred is present ({len(node.dt_pred)} rows)",
                        details={"node_name": name, "attr": "dt_pred"},
                    ),
                    raise_on_first,
                )
            else:
                self._add_result(
                    report,
                    CheckResult(
                        check_name="node_attrs",
                        level="warning",
                        passed=True,
                        message=f"node '{name}' dt_pred is set but not a DataFrame — found {type(node.dt_pred).__name__}",
                        details={"node_name": name, "attr": "dt_pred", "type_found": type(node.dt_pred).__name__},
                    ),
                    raise_on_first,
                )

    def _check_adhoc_code(self, report: DiagnosticsReport, node_name: str, mr: ModelRep, raise_on_first: bool):
        cwd = os.getcwd()
        adhoc_dir = os.path.join(cwd, "code_py", "adhoc")

        if not os.path.isdir(adhoc_dir):
            self._add_result(
                report,
                CheckResult(
                    check_name="node_attrs",
                    level="info",
                    passed=True,
                    message=f"node '{node_name}' no code_py/adhoc/ directory at CWD — adhoc code check skipped",
                    details={"node_name": node_name, "attr": "model_rep.adhoc_code_src", "adhoc_dir": adhoc_dir},
                ),
                raise_on_first,
            )
            return

        matching_files = []
        for pattern in ["reporting*.py", "translator*.py"]:
            matching_files.extend(glob_module.glob(os.path.join(adhoc_dir, pattern)))

        has_adhoc_code = False
        for fpath in matching_files:
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f.readlines()]
                non_comment_lines = [line for line in lines if line and not line.startswith("#")]
                if non_comment_lines:
                    has_adhoc_code = True
                    break
            except OSError:
                pass

        if has_adhoc_code:
            if mr.adhoc_code_src is not None and len(mr.adhoc_code_src) > 0:
                self._add_result(
                    report,
                    CheckResult(
                        check_name="node_attrs",
                        level="info",
                        passed=True,
                        message=f"node '{node_name}' adhoc_code_src is populated (project has adhoc code)",
                        details={"node_name": node_name, "attr": "model_rep.adhoc_code_src"},
                    ),
                    raise_on_first,
                )
            else:
                self._add_result(
                    report,
                    CheckResult(
                        check_name="node_attrs",
                        level="warning",
                        passed=True,
                        message=f"node '{node_name}' project has adhoc code at code_py/adhoc/ but model_rep.adhoc_code_src is empty/missing",
                        details={"node_name": node_name, "attr": "model_rep.adhoc_code_src", "adhoc_dir": adhoc_dir},
                    ),
                    raise_on_first,
                )
        else:
            self._add_result(
                report,
                CheckResult(
                    check_name="node_attrs",
                    level="info",
                    passed=True,
                    message=f"node '{node_name}' no non-comment code in adhoc reporting/translator files — adhoc_code_src not needed",
                    details={"node_name": node_name, "attr": "model_rep.adhoc_code_src"},
                ),
                raise_on_first,
            )

    def _check_edge_resolution(self, report: DiagnosticsReport, raise_on_first: bool):
        report.checks_ran.append("edge_resolution")

        if not self.meta_model.edges:
            self._add_result(
                report,
                CheckResult(
                    check_name="edge_resolution",
                    level="info",
                    passed=True,
                    message="no edges to check (single node or no edges)",
                    details={},
                ),
                raise_on_first,
            )
            return

        for left, right, merge_fn in self.meta_model.edges:
            if merge_fn is None:
                self._add_result(
                    report,
                    CheckResult(
                        check_name="edge_resolution",
                        level="info",
                        passed=True,
                        message=f"edge {left} -> {right}: parallel branch, no merge function needed",
                        details={"left": left, "right": right, "merge_fn": None},
                    ),
                    raise_on_first,
                )
                continue

            if callable(merge_fn):
                self._add_result(
                    report,
                    CheckResult(
                        check_name="edge_resolution",
                        level="info",
                        passed=True,
                        message=f"edge {left} -> {right}: merge fn '{getattr(merge_fn, '__name__', merge_fn)}' is callable",
                        details={"left": left, "right": right, "merge_fn": getattr(merge_fn, "__name__", str(merge_fn))},
                    ),
                    raise_on_first,
                )
                self._check_merge_fn_signature(report, left, right, merge_fn, raise_on_first)
                continue

            if not isinstance(merge_fn, str):
                self._add_result(
                    report,
                    CheckResult(
                        check_name="edge_resolution",
                        level="error",
                        passed=False,
                        message=f"edge {left} -> {right}: merge_fn is not callable or str — found {type(merge_fn).__name__}",
                        details={"left": left, "right": right, "merge_fn": str(merge_fn), "type_found": type(merge_fn).__name__},
                    ),
                    raise_on_first,
                )
                continue

            resolved = self._resolve_merge_fn(merge_fn)

            if resolved is None:
                self._add_result(
                    report,
                    CheckResult(
                        check_name="edge_resolution",
                        level="error",
                        passed=False,
                        message=f"edge {left} -> {right}: merge fn '{merge_fn}' could not be resolved",
                        details={"left": left, "right": right, "merge_fn": merge_fn, "reason": "not found in locals, globals, dotted modules, or module attribute scan"},
                    ),
                    raise_on_first,
                )
                continue

            self._add_result(
                report,
                CheckResult(
                    check_name="edge_resolution",
                    level="info",
                    passed=True,
                    message=f"edge {left} -> {right}: merge fn '{merge_fn}' resolves OK",
                    details={"left": left, "right": right, "merge_fn": merge_fn},
                ),
                raise_on_first,
            )
            self._check_merge_fn_signature(report, left, right, resolved, raise_on_first)

    def _resolve_merge_fn(self, merge_fn: str):
        caller_locals = locals()
        caller_globals = globals()

        if merge_fn in caller_locals or merge_fn in caller_globals:
            try:
                return eval(merge_fn)
            except Exception:
                pass

        if len(merge_fn.split(".")) > 1:
            mod_name = ".".join(merge_fn.split(".")[:-1])
            attr_name = merge_fn.split(".")[-1]
            mod = caller_globals.get(mod_name, None)
            if mod is not None and hasattr(mod, attr_name) and callable(getattr(mod, attr_name)):
                try:
                    return eval(merge_fn)
                except Exception:
                    pass

        for v in caller_globals.values():
            if (
                inspect.ismodule(v)
                and hasattr(v, merge_fn)
                and callable(getattr(v, merge_fn))
            ):
                return getattr(v, merge_fn)

        # Adhoc merge functions are attached to the meta_model module namespace
        # by MetaModel._load_adhoc_from_node during compute — search it so
        # string merges loaded from meta_conf.toml resolve like they do at
        # compute time (edge_resolution and the per-edge probe agree).
        mm_module = sys.modules.get("fermatrica_rep.meta_model.meta_model")
        if mm_module is not None:
            resolved = self._resolve_merge_fn_in_namespace(mm_module.__dict__, merge_fn)
            if resolved is not None:
                return resolved

        return None

    @staticmethod
    def _resolve_merge_fn_in_namespace(namespace: dict, merge_fn: str):
        """
        Resolve ``merge_fn`` inside a module namespace dict.

        Supports bare names, dotted module attributes (``module.fn``), and a
        scan over module values for an attribute matching the name. Returns
        ``None`` when not found.
        """
        if merge_fn in namespace and callable(namespace[merge_fn]):
            return namespace[merge_fn]

        if len(merge_fn.split(".")) > 1:
            mod_name = ".".join(merge_fn.split(".")[:-1])
            attr_name = merge_fn.split(".")[-1]
            mod = namespace.get(mod_name, None)
            if mod is not None and callable(getattr(mod, attr_name, None)):
                return getattr(mod, attr_name)

        for v in namespace.values():
            if inspect.ismodule(v) and callable(getattr(v, merge_fn, None)):
                return getattr(v, merge_fn)

        return None

    def _check_merge_fn_signature(
        self,
        report: DiagnosticsReport,
        left: str,
        right: str,
        fn,
        raise_on_first: bool,
    ):
        try:
            sig = inspect.signature(fn)
        except ValueError:
            self._add_result(
                report,
                CheckResult(
                    check_name="edge_resolution",
                    level="warning",
                    passed=True,
                    message=f"edge {left} -> {right}: cannot inspect signature of merge fn '{getattr(fn, '__name__', fn)}' (C extension?)",
                    details={"left": left, "right": right, "merge_fn": getattr(fn, "__name__", str(fn))},
                ),
                raise_on_first,
            )
            return

        params = sig.parameters
        required = {"ds_cur", "dt_pred_prev"}
        missing = required - set(params.keys())

        if missing:
            self._add_result(
                report,
                CheckResult(
                    check_name="edge_resolution",
                    level="warning",
                    passed=True,
                    message=f"edge {left} -> {right}: merge fn '{getattr(fn, '__name__', fn)}' does not accept required kwargs: {', '.join(sorted(missing))}",
                    details={"left": left, "right": right, "merge_fn": getattr(fn, "__name__", str(fn)), "missing_params": sorted(missing)},
                ),
                raise_on_first,
            )

    def _check_merge_effects(
        self,
        report: DiagnosticsReport,
        options: list,
        scopes: list[OptionScope],
        raise_on_first: bool,
    ):
        if options[0].budget_data == options[1].budget_data:
            report.checks_skipped.append(
                "merge_effect: options '%s' and '%s' are identical"
                % (options[0].name, options[1].name)
            )
            return

        saved = {}
        for name, node in self.meta_model.nodes.items():
            saved[name] = (node.ds.copy(), node.dt_pred)

        try:
            self.meta_model.compute(options[0], scopes)
            ds_A = {name: node.ds.copy() for name, node in self.meta_model.nodes.items()}
            dt_pred_A = {
                name: node.dt_pred.copy() if node.dt_pred is not None else None
                for name, node in self.meta_model.nodes.items()
            }

            for name, node in self.meta_model.nodes.items():
                node.ds, node.dt_pred = saved[name]

            self.meta_model.compute(options[1], scopes)
            ds_B = {name: node.ds.copy() for name, node in self.meta_model.nodes.items()}
            dt_pred_B = {
                name: node.dt_pred.copy() if node.dt_pred is not None else None
                for name, node in self.meta_model.nodes.items()
            }

            for name, node in self.meta_model.nodes.items():
                node.ds, node.dt_pred = saved[name]

            option_influenced = self._option_influenced_nodes()
            for name in self.meta_model.nodes:
                self._compare_dataframes(
                    report, name, "ds", ds_A[name], ds_B[name], scopes, raise_on_first,
                    expected_invariant=name not in option_influenced,
                )
                self._compare_dataframes(
                    report, name, "dt_pred", dt_pred_A[name], dt_pred_B[name], scopes, raise_on_first,
                    expected_invariant=name not in option_influenced,
                )

            self._check_edge_propagation(
                report, ds_A, dt_pred_A, dt_pred_B, raise_on_first
            )

        except Exception:
            for name, node in self.meta_model.nodes.items():
                node.ds, node.dt_pred = saved[name]
            raise

    def _option_influenced_nodes(self) -> set[str]:
        """Nodes whose output may vary between options.

        A node is option-influenced when it is non-static (it translates
        budgets) or is reachable from a non-static node via edges (it
        consumes dynamic upstream predictions). Static nodes outside this
        set are expected to be option-invariant.
        """
        influenced = {
            name for name, node in self.meta_model.nodes.items() if not node.static
        }
        changed = True
        while changed:
            changed = False
            for left, right, _ in self.meta_model.edges:
                if left in influenced and right not in influenced:
                    influenced.add(right)
                    changed = True
        return influenced

    def _compare_dataframes(
        self,
        report: DiagnosticsReport,
        node_name: str,
        df_label: str,
        df_A: pd.DataFrame | None,
        df_B: pd.DataFrame | None,
        scopes: list[OptionScope],
        raise_on_first: bool,
        expected_invariant: bool = False,
    ):
        report.checks_ran.append(f"merge_{df_label}")

        if df_A is None or df_B is None:
            self._add_result(
                report,
                CheckResult(
                    check_name=f"merge_{df_label}",
                    level="info",
                    passed=True,
                    message=f"'{node_name}' {df_label}: one or both DataFrames are None — cannot compare (prediction not computed?)",
                    details={"node_name": node_name, "df_label": df_label},
                ),
                raise_on_first,
            )
            return

        rows_A, cols_A = df_A.shape
        rows_B, cols_B = df_B.shape

        float_cols = []
        other_cols = []
        for c in df_A.columns:
            if c not in df_B.columns:
                continue
            try:
                if np.issubdtype(df_A[c].dtype, np.floating):
                    float_cols.append(c)
                else:
                    other_cols.append(c)
            except TypeError:
                other_cols.append(c)

        all_identical = True
        any_differ = False

        if float_cols:
            if not np.allclose(
                df_A[float_cols].to_numpy(dtype=float, na_value=np.nan),
                df_B[float_cols].to_numpy(dtype=float, na_value=np.nan),
                rtol=1e-7,
                atol=1e-12,
                equal_nan=True,
            ):
                all_identical = False
                any_differ = True
                diff_cols = []
                for c in float_cols:
                    if not np.allclose(
                        df_A[c].to_numpy(dtype=float, na_value=np.nan),
                        df_B[c].to_numpy(dtype=float, na_value=np.nan),
                        rtol=1e-7,
                        atol=1e-12,
                        equal_nan=True,
                    ):
                        diff_cols.append(c)
                self._add_result(
                    report,
                    CheckResult(
                        check_name=f"merge_{df_label}",
                        level="info",
                        passed=True,
                        message=f"'{node_name}' {df_label}: {len(diff_cols)} float columns differ between options: {diff_cols[:10]}",
                        details={"node_name": node_name, "df_label": df_label, "diff_float_cols": diff_cols[:10]},
                    ),
                    raise_on_first,
                )

        if other_cols:
            diff_other = []
            for c in other_cols:
                if not df_A[c].equals(df_B[c]):
                    diff_other.append(c)

            if diff_other:
                all_identical = False
                any_differ = True
                self._add_result(
                    report,
                    CheckResult(
                        check_name=f"merge_{df_label}",
                        level="info",
                        passed=True,
                        message=f"'{node_name}' {df_label}: {len(diff_other)} non-float columns differ between options: {diff_other[:10]}",
                        details={"node_name": node_name, "df_label": df_label, "diff_other_cols": diff_other[:10]},
                    ),
                    raise_on_first,
                )

        if all_identical and not any_differ:
            if expected_invariant:
                self._add_result(
                    report,
                    CheckResult(
                        check_name=f"merge_{df_label}",
                        level="info",
                        passed=True,
                        message=f"'{node_name}' {df_label}: unchanged between options — static node with no option-influenced upstream (expected invariant)",
                        details={"node_name": node_name, "df_label": df_label, "rows": (rows_A, rows_B), "cols": (cols_A, cols_B)},
                    ),
                    raise_on_first,
                )
            else:
                self._add_result(
                    report,
                    CheckResult(
                        check_name=f"merge_{df_label}",
                        level="error",
                        passed=False,
                        message=f"'{node_name}' {df_label}: unchanged between options — merge may not be propagating or options may not affect this node",
                        details={"node_name": node_name, "df_label": df_label, "rows": (rows_A, rows_B), "cols": (cols_A, cols_B)},
                    ),
                    raise_on_first,
                )
        elif any_differ:
            self._add_result(
                report,
                CheckResult(
                    check_name=f"merge_{df_label}",
                    level="info",
                    passed=True,
                    message=f"'{node_name}' {df_label}: differs between options — merge effects propagate",
                    details={"node_name": node_name, "df_label": df_label, "rows": (rows_A, rows_B), "cols": (cols_A, cols_B)},
                ),
                raise_on_first,
            )
        else:
            self._add_result(
                report,
                CheckResult(
                    check_name=f"merge_{df_label}",
                    level="info",
                    passed=True,
                    message=f"'{node_name}' {df_label}: all columns identical between options",
                    details={"node_name": node_name, "df_label": df_label, "rows": (rows_A, rows_B), "cols": (cols_A, cols_B)},
                ),
                raise_on_first,
            )

    def _check_edge_propagation(
        self,
        report: DiagnosticsReport,
        ds_A: dict,
        dt_pred_A: dict,
        dt_pred_B: dict,
        raise_on_first: bool,
    ):
        """
        Per-edge attribution of the option-driven signal.

        The node-level comparison above proves that a node's output *somewhere*
        changed between options. In a fan-in DAG (a node with 2+ predecessors)
        that can hide a broken edge: one merge propagates the signal while
        another silently no-ops or writes constant values, and the node still
        differs overall.

        For each edge ``left -> right`` with a merge function, this probe
        reuses the snapshots from the two option runs: when ``left.dt_pred``
        differs between options (upstream signal present), the merge is invoked
        directly on a copy of ``right.ds`` with each option's predecessor
        prediction. Identical merged output means the edge itself does not
        propagate the signal:

        - merge writes **nothing** at all → error (silent no-op — the edge is
          definitively dead for the active options and the node-level check
          cannot see it in a fan-in DAG);
        - merge writes columns but produces identical values → warning (a
          constant-value merge may be intentional).

        ``allow_past=False`` matches the option-path semantics used by the two
        :meth:`MetaModel.compute` runs above.
        """
        for left, right, merge_fn in self.meta_model.edges:
            if merge_fn is None:
                continue

            report.checks_ran.append("merge_edge")

            left_dt_A = dt_pred_A.get(left)
            left_dt_B = dt_pred_B.get(left)
            base = ds_A.get(right)

            if left_dt_A is None or left_dt_B is None:
                self._add_result(
                    report,
                    CheckResult(
                        check_name="merge_edge",
                        level="info",
                        passed=True,
                        message=f"edge {left} -> {right}: cannot test — left node dt_pred is None in one or both runs",
                        details={"left": left, "right": right, "reason": "left dt_pred missing"},
                    ),
                    raise_on_first,
                )
                continue

            if self._frames_identical(left_dt_A, left_dt_B):
                self._add_result(
                    report,
                    CheckResult(
                        check_name="merge_edge",
                        level="info",
                        passed=True,
                        message=f"edge {left} -> {right}: no upstream signal between these options — edge not exercisable",
                        details={"left": left, "right": right, "reason": "left dt_pred identical across options"},
                    ),
                    raise_on_first,
                )
                continue

            fn = self._resolve_merge_fn_for_probe(merge_fn)
            if fn is None:
                self._add_result(
                    report,
                    CheckResult(
                        check_name="merge_edge",
                        level="info",
                        passed=True,
                        message=f"edge {left} -> {right}: cannot test — merge fn '{merge_fn}' not resolvable (see edge_resolution)",
                        details={"left": left, "right": right, "reason": "unresolvable merge fn"},
                    ),
                    raise_on_first,
                )
                continue

            try:
                merged_A = self.meta_model._apply_merge_fn(
                    fn, base.copy(), left_dt_A.copy(), allow_past=False
                )
                merged_B = self.meta_model._apply_merge_fn(
                    fn, base.copy(), left_dt_B.copy(), allow_past=False
                )
            except Exception as exc:
                self._add_result(
                    report,
                    CheckResult(
                        check_name="merge_edge",
                        level="warning",
                        passed=True,
                        message=f"edge {left} -> {right}: merge fn raised during propagation probe: {exc}",
                        details={"left": left, "right": right, "exception": str(exc)},
                    ),
                    raise_on_first,
                )
                continue

            diff_left_cols = self._diff_columns(left_dt_A, left_dt_B)
            propagates = not self._frames_identical(merged_A, merged_B)
            wrote_any = not self._frames_identical(merged_A, base)

            if propagates:
                self._add_result(
                    report,
                    CheckResult(
                        check_name="merge_edge",
                        level="info",
                        passed=True,
                        message=(
                            f"edge {left} -> {right}: merge propagates upstream signal "
                            f"(left dt_pred differs in {len(diff_left_cols)} col(s): {diff_left_cols[:5]})"
                        ),
                        details={"left": left, "right": right, "diff_left_cols": diff_left_cols[:10]},
                    ),
                    raise_on_first,
                )
            elif wrote_any:
                self._add_result(
                    report,
                    CheckResult(
                        check_name="merge_edge",
                        level="warning",
                        passed=True,
                        message=(
                            f"edge {left} -> {right}: merge writes columns but output is identical across options — "
                            f"does not propagate upstream signal "
                            f"({len(diff_left_cols)} col(s) differ in left dt_pred: {diff_left_cols[:5]})"
                        ),
                        details={
                            "left": left,
                            "right": right,
                            "diff_left_cols": diff_left_cols[:10],
                            "reason": "constant output despite upstream signal",
                        },
                    ),
                    raise_on_first,
                )
            else:
                self._add_result(
                    report,
                    CheckResult(
                        check_name="merge_edge",
                        level="error",
                        passed=False,
                        message=(
                            f"edge {left} -> {right}: merge produced no output change on either option — "
                            f"silent no-op; upstream signal "
                            f"({len(diff_left_cols)} col(s): {diff_left_cols[:5]}) not propagated"
                        ),
                        details={
                            "left": left,
                            "right": right,
                            "diff_left_cols": diff_left_cols[:10],
                            "reason": "no columns written",
                        },
                    ),
                    raise_on_first,
                )

    def _resolve_merge_fn_for_probe(self, merge_fn) -> "callable | None":
        """
        Resolve a merge function for the per-edge propagation probe.

        Delegates to :meth:`_resolve_merge_fn`, which also searches the
        ``fermatrica_rep.meta_model.meta_model`` module namespace where adhoc
        merge functions are attached by ``MetaModel._load_adhoc_from_node``
        during :meth:`MetaModel.compute`.
        """
        if callable(merge_fn):
            return merge_fn
        if not isinstance(merge_fn, str):
            return None
        return self._resolve_merge_fn(merge_fn)

    @staticmethod
    def _frames_identical(
        df_A: pd.DataFrame | None, df_B: pd.DataFrame | None
    ) -> bool:
        """Float-aware equality for two DataFrames (same tolerance as the node-level check)."""
        if df_A is None or df_B is None:
            return df_A is df_B
        if df_A.shape != df_B.shape or list(df_A.columns) != list(df_B.columns):
            return False

        float_cols = []
        other_cols = []
        for c in df_A.columns:
            try:
                if np.issubdtype(df_A[c].dtype, np.floating):
                    float_cols.append(c)
                else:
                    other_cols.append(c)
            except TypeError:
                other_cols.append(c)

        if float_cols and not np.allclose(
            df_A[float_cols].to_numpy(dtype=float, na_value=np.nan),
            df_B[float_cols].to_numpy(dtype=float, na_value=np.nan),
            rtol=1e-7,
            atol=1e-12,
            equal_nan=True,
        ):
            return False

        return all(df_A[c].equals(df_B[c]) for c in other_cols)

    @staticmethod
    def _diff_columns(df_A: pd.DataFrame, df_B: pd.DataFrame) -> list:
        """Return the columns where two DataFrames differ (float-aware)."""
        if df_A is None or df_B is None:
            return []
        diff = []
        for c in df_A.columns:
            if c not in df_B.columns:
                diff.append(c)
            elif not MetaModelDiagnostics._frames_identical(df_A[[c]], df_B[[c]]):
                diff.append(c)
        return diff
