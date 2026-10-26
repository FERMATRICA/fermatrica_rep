"""Unit tests for pipeline/generator.py helpers and generate_pipeline().

Uses monkeypatched read_pickle_compat and synthetic XLSX files.
"""


import os
import pandas as pd
import pytest
from fermatrica.basics.basics import FermatricaError
from fermatrica_rep.pipeline.generator import (
    _read_translation_sheets,
    _validate_inputs,
    _output_xlsx_path,
    _edge_index_map,
    _compute_predecessors,
)


class TestReadTranslationSheets:

    def test_detects_translation_sheets(self, tmp_path):
        xlsx_path = tmp_path / "options.xlsx"
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            pd.DataFrame({"item": ["bdg_tv"], "function": ["basic"], "variable": ["x1"]}).to_excel(
                writer, sheet_name="translation_a", index=False)
            pd.DataFrame({"item": ["bdg_digital"], "function": ["basic"], "variable": ["x2"]}).to_excel(
                writer, sheet_name="translation_b", index=False)
            pd.DataFrame({"x": [1]}).to_excel(writer, sheet_name="data", index=False)

        node_names, trans_data = _read_translation_sheets(str(xlsx_path))
        assert sorted(node_names) == ["a", "b"]
        assert "a" in trans_data
        assert "b" in trans_data

    def test_raises_on_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            _read_translation_sheets("nonexistent.xlsx")

    def test_raises_on_no_translation_sheets(self, tmp_path):
        xlsx_path = tmp_path / "options.xlsx"
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            pd.DataFrame({"x": [1]}).to_excel(writer, sheet_name="data", index=False)

        node_names, trans_data = _read_translation_sheets(str(xlsx_path))
        assert node_names == []
        assert trans_data == {}

    def test_raises_on_missing_required_columns(self, tmp_path):
        xlsx_path = tmp_path / "options.xlsx"
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            pd.DataFrame({"item": ["bdg_tv"], "function": ["basic"]}).to_excel(
                writer, sheet_name="translation_x", index=False)

        with pytest.raises(FermatricaError, match="missing"):
            _read_translation_sheets(str(xlsx_path))


class TestValidateInputs:

    def test_raises_on_nonexistent_file(self, tmp_path):
        with pytest.raises(FermatricaError, match="not found"):
            _validate_inputs(
                options_path=str(tmp_path / "nonexistent.xlsx"),
                model_paths={},
                targets={},
                edges=None,
                project_root=str(tmp_path),
                force=True,
                node_names=["a"],
                predecessors={"a": {"nodes": ["a"], "edges": []}},
            )


class TestHelpers:

    def test_output_xlsx_path_strips_template(self):
        result = _output_xlsx_path(
            "D:/some/project/context_template.xlsx",
            "D:/some/project",
        )
        expected_tail = os.path.join("model_data", "options", "context.xlsx")
        assert result.endswith(expected_tail)

    def test_output_xlsx_path_strips_templ(self):
        result = _output_xlsx_path(
            "D:/some/project/context_templ.xlsx",
            "D:/some/project",
        )
        expected_tail = os.path.join("model_data", "options", "context.xlsx")
        assert result.endswith(expected_tail)

    def test_edge_index_map(self):
        edges = [("a", "b"), ("b", "c")]
        result = _edge_index_map(edges)
        assert result[("a", "b")] == 1
        assert result[("b", "c")] == 2

    def test_compute_predecessors_chain(self):
        preds = _compute_predecessors(
            node_names=["a", "b", "c"],
            edges=[("a", "b"), ("b", "c")],
            edge_index=_edge_index_map([("a", "b"), ("b", "c")]),
        )
        assert set(preds["a"]["nodes"]) == {"a"}
        assert set(preds["b"]["nodes"]) == {"a", "b"}
        assert set(preds["c"]["nodes"]) == {"a", "b", "c"}

    def test_compute_predecessors_no_edges(self):
        preds = _compute_predecessors(
            node_names=["a", "b"],
            edges=[],
            edge_index={},
        )
        assert set(preds["a"]["nodes"]) == {"a"}
        assert set(preds["b"]["nodes"]) == {"b"}


class TestGeneratePipelineE2E:

    def _make_template_xlsx(self, tmp_path, node_names=("a", "b")):
        xlsx_path = tmp_path / "context_template.xlsx"
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            for n in node_names:
                pd.DataFrame({
                    "item": ["bdg_tv", "bdg_digital"],
                    "function": ["basic", "basic"],
                    "variable": [f"tv_var_{n}", f"dig_var_{n}"],
                }).to_excel(writer, sheet_name=f"translation_{n}", index=False)
        return xlsx_path

    def _make_model_dirs(self, tmp_path, node_names=("a", "b")):
        for n in node_names:
            pth = tmp_path / "models" / n / "model_001"
            pth.mkdir(parents=True, exist_ok=True)
            (pth / "dt_p.pkl.zstd").touch()
        return tmp_path

    def _make_synthetic_dt_p(self):
        return pd.DataFrame({
            "date": pd.date_range("2025-01-01", periods=365, freq="D"),
            "listed": [4] * 365,
            "superbrand": ["BrandX"] * 365,
        })

    def test_produces_output_xlsx_with_all_sheets(self, tmp_path, monkeypatch):
        from fermatrica_rep.pipeline.generator import generate_pipeline

        xlsx_path = self._make_template_xlsx(tmp_path)
        project_root = self._make_model_dirs(tmp_path)

        synthetic_dt_p = self._make_synthetic_dt_p()
        monkeypatch.setattr(
            "fermatrica_rep.pipeline.generator.read_pickle_compat",
            lambda path: synthetic_dt_p,
        )

        model_paths = {
            "a": str(project_root / "models" / "a"),
            "b": str(project_root / "models" / "b"),
        }
        targets = {
            "a": [("superbrand", "BrandX")],
            "b": [("superbrand", "BrandX")],
        }

        generate_pipeline(
            options_path=str(xlsx_path),
            model_paths=model_paths,
            targets=targets,
            project_root=str(project_root),
            force=True,
        )

        output_path = project_root / "model_data" / "options" / "context.xlsx"
        assert output_path.exists()

        output_sheets = pd.ExcelFile(str(output_path)).sheet_names
        assert "scopes" in output_sheets
        assert "allocations" in output_sheets
        assert "media_prices" in output_sheets
        assert "seasonality" in output_sheets

    def test_strips_template_suffix(self, tmp_path, monkeypatch):
        from fermatrica_rep.pipeline.generator import generate_pipeline

        xlsx_path = self._make_template_xlsx(tmp_path)
        project_root = self._make_model_dirs(tmp_path)

        synthetic_dt_p = self._make_synthetic_dt_p()
        monkeypatch.setattr(
            "fermatrica_rep.pipeline.generator.read_pickle_compat",
            lambda path: synthetic_dt_p,
        )

        model_paths = {
            "a": str(project_root / "models" / "a"),
            "b": str(project_root / "models" / "b"),
        }
        targets = {
            "a": [("superbrand", "BrandX")],
            "b": [("superbrand", "BrandX")],
        }

        generate_pipeline(
            options_path=str(xlsx_path),
            model_paths=model_paths,
            targets=targets,
            project_root=str(project_root),
            force=True,
        )

        output_path = project_root / "model_data" / "options" / "context.xlsx"
        assert output_path.exists()
        assert "context.xlsx" == output_path.name

    def test_creates_adhoc_stubs(self, tmp_path, monkeypatch):
        from fermatrica_rep.pipeline.generator import generate_pipeline

        xlsx_path = self._make_template_xlsx(tmp_path)
        project_root = self._make_model_dirs(tmp_path)

        synthetic_dt_p = self._make_synthetic_dt_p()
        monkeypatch.setattr(
            "fermatrica_rep.pipeline.generator.read_pickle_compat",
            lambda path: synthetic_dt_p,
        )

        model_paths = {
            "a": str(project_root / "models" / "a"),
            "b": str(project_root / "models" / "b"),
        }
        targets = {
            "a": [("superbrand", "BrandX")],
            "b": [("superbrand", "BrandX")],
        }

        generate_pipeline(
            options_path=str(xlsx_path),
            model_paths=model_paths,
            targets=targets,
            project_root=str(project_root),
            force=True,
        )

        assert (project_root / "adhoc" / "__init__.py").exists()
        assert (project_root / "adhoc" / "translators.py").exists()
        assert (project_root / "adhoc" / "reporting.py").exists()

    def test_creates_reporting_scripts(self, tmp_path, monkeypatch):
        from fermatrica_rep.pipeline.generator import generate_pipeline

        xlsx_path = self._make_template_xlsx(tmp_path)
        project_root = self._make_model_dirs(tmp_path)

        synthetic_dt_p = self._make_synthetic_dt_p()
        monkeypatch.setattr(
            "fermatrica_rep.pipeline.generator.read_pickle_compat",
            lambda path: synthetic_dt_p,
        )

        model_paths = {
            "a": str(project_root / "models" / "a"),
            "b": str(project_root / "models" / "b"),
        }
        targets = {
            "a": [("superbrand", "BrandX")],
            "b": [("superbrand", "BrandX")],
        }

        generate_pipeline(
            options_path=str(xlsx_path),
            model_paths=model_paths,
            targets=targets,
            project_root=str(project_root),
            force=True,
        )

        assert (project_root / "scripts" / "reporting_a.py").exists()
        assert (project_root / "scripts" / "reporting_b.py").exists()
        reporting_a = (project_root / "scripts" / "reporting_a.py").read_text()
        assert "fmr.load_context" in reporting_a
        assert "compute_context=ctx" in reporting_a
        assert "MetaModel(nodes=node, compute_context=ctx)" in reporting_a
        assert "start_year" in reporting_a
        assert "replace_dates" in reporting_a
        assert "filter_scopes" in reporting_a
        assert "budget_for_scope" in reporting_a
        assert "import copy" not in reporting_a
        compile(reporting_a, "<gen>", "exec")

    def test_creates_multi_node_script_with_edges(self, tmp_path, monkeypatch):
        from fermatrica_rep.pipeline.generator import generate_pipeline

        xlsx_path = self._make_template_xlsx(tmp_path)
        project_root = self._make_model_dirs(tmp_path)

        synthetic_dt_p = self._make_synthetic_dt_p()
        monkeypatch.setattr(
            "fermatrica_rep.pipeline.generator.read_pickle_compat",
            lambda path: synthetic_dt_p,
        )

        model_paths = {
            "a": str(project_root / "models" / "a"),
            "b": str(project_root / "models" / "b"),
        }
        targets = {
            "a": [("superbrand", "BrandX")],
            "b": [("superbrand", "BrandX")],
        }

        generate_pipeline(
            options_path=str(xlsx_path),
            model_paths=model_paths,
            targets=targets,
            edges=[("a", "b")],
            project_root=str(project_root),
            force=True,
        )

        reporting_a = (project_root / "scripts" / "reporting_a.py").read_text()
        reporting_b = (project_root / "scripts" / "reporting_b.py").read_text()

        # single-node predecessor script
        assert "MetaModel(nodes=node, compute_context=ctx)" in reporting_a
        assert "pred_merge_1" not in reporting_a

        # multi-node script includes the full chain + edges + merge import
        assert "MetaModel(nodes={'a': node_a, 'b': node_b}" in reporting_b
        assert '("a", "b", reporting.pred_merge_1)' in reporting_b
        assert "compute_context=ctx" in reporting_b
        assert "from code_py.adhoc import reporting" in reporting_b
        assert "fmr.load_context" in reporting_b
        assert "start_year" in reporting_b
        assert "replace_dates" in reporting_b
        assert "filter_scopes" in reporting_b
        assert "budget_for_scope" in reporting_b
        assert "import copy" not in reporting_b
        compile(reporting_b, "<gen>", "exec")

    def test_force_false_raises_when_exists(self, tmp_path, monkeypatch):
        from fermatrica_rep.pipeline.generator import generate_pipeline

        xlsx_path = self._make_template_xlsx(tmp_path)
        project_root = self._make_model_dirs(tmp_path)

        synthetic_dt_p = self._make_synthetic_dt_p()
        monkeypatch.setattr(
            "fermatrica_rep.pipeline.generator.read_pickle_compat",
            lambda path: synthetic_dt_p,
        )

        model_paths = {
            "a": str(project_root / "models" / "a"),
            "b": str(project_root / "models" / "b"),
        }
        targets = {
            "a": [("superbrand", "BrandX")],
            "b": [("superbrand", "BrandX")],
        }

        generate_pipeline(
            options_path=str(xlsx_path),
            model_paths=model_paths,
            targets=targets,
            project_root=str(project_root),
            force=True,
        )

        with pytest.raises(FermatricaError, match="already exist"):
            generate_pipeline(
                options_path=str(xlsx_path),
                model_paths=model_paths,
                targets=targets,
                project_root=str(project_root),
                force=False,
            )
