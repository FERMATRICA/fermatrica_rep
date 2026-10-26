"""Unit tests for Option.from_excel and Option.from_excel_all."""


import pandas as pd
import pytest
from fermatrica.basics.basics import FermatricaError
from fermatrica_rep.options.option import Option


class TestOptionFromExcel:

    def test_loads_option_by_name(self, tmp_path):
        xlsx_path = tmp_path / "options.xlsx"
        dt = pd.DataFrame({
            "option": ["scenario_a", "scenario_a", "scenario_b"],
            "scope_name": ["brandx_2025", "brandx_2025", "brandy_2025"],
            "item": ["bdg_tv", "bdg_digital", "bdg_tv"],
            "val": [50, 30, 20],
        })
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            dt.to_excel(writer, sheet_name="allocations", index=False)

        opt = Option.from_excel(str(xlsx_path), "scenario_a")
        assert isinstance(opt, Option)
        assert opt.name == "scenario_a"
        assert "brandx_2025" in opt.budget_data
        assert opt.budget_data["brandx_2025"]["bdg_tv"] == 50
        assert opt.budget_data["brandx_2025"]["bdg_digital"] == 30

    def test_from_excel_all_returns_dict(self, tmp_path):
        xlsx_path = tmp_path / "options.xlsx"
        dt = pd.DataFrame({
            "option": ["scenario_a", "scenario_b", "scenario_c"],
            "scope_name": ["brandx_2025", "brandy_2025", "brandz_2025"],
            "item": ["bdg_tv", "bdg_tv", "bdg_tv"],
            "val": [50, 30, 20],
        })
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            dt.to_excel(writer, sheet_name="allocations", index=False)

        result = Option.from_excel_all(str(xlsx_path))
        assert isinstance(result, dict)
        assert set(result.keys()) == {"scenario_a", "scenario_b", "scenario_c"}
        assert all(isinstance(o, Option) for o in result.values())

    def test_raises_when_option_not_found(self, tmp_path):
        xlsx_path = tmp_path / "options.xlsx"
        dt = pd.DataFrame({
            "option": ["scenario_a", "scenario_a"],
            "scope_name": ["brandx_2025", "brandx_2025"],
            "item": ["bdg_tv", "bdg_digital"],
            "val": [50, 30],
        })
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            dt.to_excel(writer, sheet_name="allocations", index=False)

        with pytest.raises(FermatricaError, match="budget"):
            Option.from_excel(str(xlsx_path), "nonexistent")

    def test_respects_default_scope(self, tmp_path):
        xlsx_path = tmp_path / "options.xlsx"
        dt = pd.DataFrame({
            "option": ["scenario_a", "scenario_a"],
            "scope_name": ["brandx_2025", None],
            "item": ["bdg_tv", "bdg_digital"],
            "val": [50, 30],
        })
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            dt.to_excel(writer, sheet_name="allocations", index=False)

        opt = Option.from_excel(str(xlsx_path), "scenario_a", default_scope="default_scope")
        assert "default_scope" in opt.budget_data
        assert opt.budget_data["default_scope"]["bdg_digital"] == 30
