"""Unit tests for untested functions in options/define.py.

Covers scopes_read, seasonality_read, media_prices_read, and
_enrich_trans_dict_from_model.
"""


import pandas as pd
import pytest
from fermatrica.basics.basics import FermatricaError
from fermatrica_rep.options.define import (
    scopes_read,
    seasonality_read,
    media_prices_read,
    _enrich_trans_dict_from_model,
)
from fermatrica_rep.options.scope import OptionScope


class TestScopesRead:

    def test_reads_scopes_sheet(self, tmp_path):
        xlsx_path = tmp_path / "options.xlsx"
        dt_scopes = pd.DataFrame({
            "scope_name": ["scope_a", "scope_b"],
            "date_start": ["2025-01-01", "2025-01-01"],
            "date_end": ["2025-12-31", "2025-12-31"],
            "ref_date_start": ["2024-01-01", "2024-01-01"],
            "ref_date_end": ["2024-12-31", "2024-12-31"],
            "apply_vars": ["superbrand", "superbrand"],
            "target": ["BrandA", "BrandB"],
            "plan_period": ["exact", "year"],
            "inflation_multiplier": [1.04, 1.0],
            "seasonality": ["", ""],
        })
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            dt_scopes.to_excel(writer, sheet_name="scopes", index=False)

        result = scopes_read(str(xlsx_path))
        assert len(result) == 2
        assert all(isinstance(s, OptionScope) for s in result)

    def test_each_scope_has_expected_fields(self, tmp_path):
        xlsx_path = tmp_path / "options.xlsx"
        dt_scopes = pd.DataFrame({
            "scope_name": ["scope_a"],
            "date_start": ["2025-01-01"],
            "date_end": ["2025-12-31"],
            "ref_date_start": ["2024-01-01"],
            "ref_date_end": ["2024-12-31"],
            "apply_vars": pd.Series([None], dtype=object),
            "target": pd.Series([None], dtype=object),
            "plan_period": ["year"],
            "inflation_multiplier": [1.0],
            "seasonality": [""],
        })
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            dt_scopes.to_excel(writer, sheet_name="scopes", index=False)

        result = scopes_read(str(xlsx_path))
        s = result[0]
        assert s.scope_name == "scope_a"
        assert pd.notna(s.date_start)
        assert pd.notna(s.date_end)
        assert pd.notna(s.ref_date_start)
        assert pd.notna(s.ref_date_end)

    def test_apply_vars_target_become_filters(self, tmp_path):
        xlsx_path = tmp_path / "options.xlsx"
        dt_scopes = pd.DataFrame({
            "scope_name": ["scope_a"],
            "date_start": ["2025-01-01"],
            "date_end": ["2025-12-31"],
            "ref_date_start": ["2024-01-01"],
            "ref_date_end": ["2024-12-31"],
            "apply_vars": ["superbrand,region"],
            "target": ["BrandX,NY"],
            "plan_period": ["year"],
            "inflation_multiplier": [1.0],
            "seasonality": [""],
        })
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            dt_scopes.to_excel(writer, sheet_name="scopes", index=False)

        result = scopes_read(str(xlsx_path))
        s = result[0]
        assert s.filters == {"superbrand": ["BrandX"], "region": ["NY"]}

    def test_ref_date_end_defaults_when_empty(self, tmp_path):
        xlsx_path = tmp_path / "options.xlsx"
        dt_scopes = pd.DataFrame({
            "scope_name": ["scope_a"],
            "date_start": ["2025-01-01"],
            "date_end": ["2025-12-31"],
            "ref_date_start": ["2024-01-01"],
            "ref_date_end": [None],
            "apply_vars": pd.Series([None], dtype=object),
            "target": pd.Series([None], dtype=object),
            "plan_period": ["year"],
            "inflation_multiplier": [1.0],
            "seasonality": [""],
        })
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            dt_scopes.to_excel(writer, sheet_name="scopes", index=False)

        result = scopes_read(str(xlsx_path))
        s = result[0]
        expected = pd.Timestamp("2024-12-31")
        assert s.ref_date_end == expected

    def test_raises_on_missing_sheet(self, tmp_path):
        xlsx_path = tmp_path / "options.xlsx"
        dt = pd.DataFrame({"col": [1, 2]})
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            dt.to_excel(writer, sheet_name="other", index=False)

        with pytest.raises(FermatricaError, match="scopes"):
            scopes_read(str(xlsx_path), sheet="scopes")

    def test_raises_on_missing_scope_name_column(self, tmp_path):
        xlsx_path = tmp_path / "options.xlsx"
        dt_scopes = pd.DataFrame({
            "date_start": ["2025-01-01"],
            "date_end": ["2025-12-31"],
            "ref_date_start": ["2024-01-01"],
            "ref_date_end": ["2024-12-31"],
        })
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            dt_scopes.to_excel(writer, sheet_name="scopes", index=False)

        with pytest.raises(FermatricaError, match="scope_name"):
            scopes_read(str(xlsx_path))


class TestSeasonalityRead:

    def test_reads_seasonality_sheet(self, tmp_path):
        xlsx_path = tmp_path / "options.xlsx"
        dt_ssn = pd.DataFrame({
            "scope_name": ["scope_a"] * 12 + ["scope_b"] * 12,
            "period": list(range(1, 13)) * 2,
            "bdg_tv": [0.08] * 12 + [0.1] * 12,
            "bdg_digital": [0.05] * 24,
        })
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            dt_ssn.to_excel(writer, sheet_name="seasonality", index=False)

        result = seasonality_read(str(xlsx_path))
        assert isinstance(result, dict)
        assert len(result) == 2
        assert "scope_a" in result
        assert "scope_b" in result

    def test_each_df_has_period_no_scope_name(self, tmp_path):
        xlsx_path = tmp_path / "options.xlsx"
        dt_ssn = pd.DataFrame({
            "scope_name": ["scope_a"] * 12,
            "period": list(range(1, 13)),
            "bdg_tv": [0.08] * 12,
        })
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            dt_ssn.to_excel(writer, sheet_name="seasonality", index=False)

        result = seasonality_read(str(xlsx_path))
        df_a = result["scope_a"]
        assert "period" in df_a.columns
        assert "scope_name" not in df_a.columns

    def test_empty_dict_when_sheet_absent(self, tmp_path):
        xlsx_path = tmp_path / "options.xlsx"
        dt = pd.DataFrame({"col": [1]})
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            dt.to_excel(writer, sheet_name="other", index=False)

        result = seasonality_read(str(xlsx_path))
        assert result == {}

    def test_raises_when_no_scope_name_column(self, tmp_path):
        xlsx_path = tmp_path / "options.xlsx"
        dt_ssn = pd.DataFrame({
            "period": [1, 2, 3],
            "bdg_tv": [0.08, 0.09, 0.1],
        })
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            dt_ssn.to_excel(writer, sheet_name="seasonality", index=False)

        with pytest.raises(FermatricaError, match="scope_name"):
            seasonality_read(str(xlsx_path))


class TestMediaPricesRead:

    def test_reads_media_prices(self, tmp_path):
        xlsx_path = tmp_path / "options.xlsx"
        dt_mp = pd.DataFrame({
            "scope_name": ["scope_a", "scope_a", "scope_b"],
            "item": ["bdg_tv", "bdg_digital", "bdg_tv"],
            "multiplier": [1.5, 2.0, 1.3],
        })
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            dt_mp.to_excel(writer, sheet_name="media_prices", index=False)

        result = media_prices_read(str(xlsx_path))
        assert isinstance(result, dict)
        assert "scope_a" in result
        assert result["scope_a"]["bdg_tv"] == 1.5

    def test_empty_dict_when_sheet_absent(self, tmp_path):
        xlsx_path = tmp_path / "options.xlsx"
        dt = pd.DataFrame({"col": [1]})
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            dt.to_excel(writer, sheet_name="other", index=False)

        result = media_prices_read(str(xlsx_path))
        assert result == {}

    def test_correct_multiplier_values(self, tmp_path):
        xlsx_path = tmp_path / "options.xlsx"
        dt_mp = pd.DataFrame({
            "scope_name": ["scope_a"],
            "item": ["bdg_tv"],
            "multiplier": [2.5],
        })
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            dt_mp.to_excel(writer, sheet_name="media_prices", index=False)

        result = media_prices_read(str(xlsx_path))
        assert result["scope_a"]["bdg_tv"] == 2.5


class TestEnrichTransDictFromModel:

    def test_copies_price_from_trans_path_df(self):
        class MockConf:
            trans_path_df = pd.DataFrame({
                "variable": ["brnd_tv_ots_clip"],
                "price": [0.05],
            })

        trans_dict = {
            "function": {"bdg_tv": "basic"},
            "variable": {"bdg_tv": "brnd_tv_ots_clip"},
        }

        result = _enrich_trans_dict_from_model(MockConf(), trans_dict)
        assert result["base_price"]["bdg_tv"] == 0.05

    def test_noop_when_trans_dict_none(self):
        class MockConf:
            trans_path_df = pd.DataFrame({
                "variable": ["brnd_tv_ots_clip"],
                "price": [0.05],
            })

        result = _enrich_trans_dict_from_model(MockConf(), None)
        assert result is None

    def test_noop_when_trans_path_df_missing(self):
        class MockConf:
            pass

        trans_dict = {
            "function": {"bdg_tv": "basic"},
            "variable": {"bdg_tv": "brnd_tv_ots_clip"},
        }
        trans_before = dict(trans_dict)

        result = _enrich_trans_dict_from_model(MockConf(), trans_dict)
        assert result == trans_before
        assert "base_price" not in result or result.get("base_price", {}) == {}

    def test_noop_when_required_columns_absent(self):
        class MockConf:
            trans_path_df = pd.DataFrame({
                "not_variable": ["brnd_tv_ots_clip"],
                "not_price": [0.05],
            })

        trans_dict = {
            "function": {"bdg_tv": "basic"},
            "variable": {"bdg_tv": "brnd_tv_ots_clip"},
        }

        result = _enrich_trans_dict_from_model(MockConf(), trans_dict)
        assert "base_price" not in result

    def test_multiple_items(self):
        class MockConf:
            trans_path_df = pd.DataFrame({
                "variable": ["brnd_tv_ots_clip", "digital_imp"],
                "price": [0.05, 0.03],
            })

        trans_dict = {
            "function": {"bdg_tv": "basic", "bdg_digital": "basic"},
            "variable": {"bdg_tv": "brnd_tv_ots_clip", "bdg_digital": "digital_imp"},
        }

        result = _enrich_trans_dict_from_model(MockConf(), trans_dict)
        assert result["base_price"]["bdg_tv"] == 0.05
        assert result["base_price"]["bdg_digital"] == 0.03
