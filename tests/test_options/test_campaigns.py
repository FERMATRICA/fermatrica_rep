"""Unit tests for campaigns_to_options (options/context.py).

Pure-data tests — no sample models needed. Covers scope derivation, day-weighted
allocations, per-scope-year seasonality, save/load round-trip, and validation.
"""
import pandas as pd
import pytest

import fermatrica_rep as fmr
from fermatrica.basics.basics import FermatricaError
from fermatrica_rep.options.define import _seasonality_to_dates


def _campaigns(rows):
    return pd.DataFrame(rows, columns=["option", "item", "date_start", "date_end", "val", "target"])


class TestMonthSplit:

    def test_share_split_and_allocation(self):
        camps = _campaigns([["predefined", "bdg_tv", "2026-03-15", "2026-04-10", 27.0, "brand_1"]])
        ctx = fmr.campaigns_to_options(camps)
        assert [s.scope_name for s in ctx.scopes] == ["predefined_brand_1_2026"]

        alloc = ctx.options["predefined"].budget_for_scope("predefined_brand_1_2026")
        assert alloc["bdg_tv"] == pytest.approx(27.0)

        ssn = ctx.ssn_dict["predefined_brand_1_2026"]
        assert ssn.loc[ssn["month"] == 3, "tv"].iloc[0] == pytest.approx(17 / 27)
        assert ssn.loc[ssn["month"] == 4, "tv"].iloc[0] == pytest.approx(10 / 27)
        assert ssn["tv"].sum() == pytest.approx(1.0)

    def test_leap_year_february(self):
        camps = _campaigns([["predefined", "bdg_tv", "2024-02-01", "2024-02-29", 29.0, "brand_1"]])
        ctx = fmr.campaigns_to_options(camps)
        ssn = ctx.ssn_dict["predefined_brand_1_2024"]
        assert ssn.loc[ssn["month"] == 2, "tv"].iloc[0] == pytest.approx(1.0)
        assert ssn["tv"].sum() == pytest.approx(1.0)
        alloc = ctx.options["predefined"].budget_for_scope("predefined_brand_1_2024")
        assert alloc["bdg_tv"] == pytest.approx(29.0)


class TestMultiYear:

    def test_two_scopes_day_weighted(self):
        camps = _campaigns([["predefined", "bdg_tv", "2025-12-15", "2026-02-10", 58.0, "brand_1"]])
        ctx = fmr.campaigns_to_options(camps)
        names = [s.scope_name for s in ctx.scopes]
        assert names == ["predefined_brand_1_2025", "predefined_brand_1_2026"]

        alloc_25 = ctx.options["predefined"].budget_for_scope("predefined_brand_1_2025")
        alloc_26 = ctx.options["predefined"].budget_for_scope("predefined_brand_1_2026")
        assert alloc_25["bdg_tv"] == pytest.approx(58.0 * 17 / 58)
        assert alloc_26["bdg_tv"] == pytest.approx(58.0 * 41 / 58)
        assert alloc_25["bdg_tv"] + alloc_26["bdg_tv"] == pytest.approx(58.0)

        for scope_name in ("predefined_brand_1_2025", "predefined_brand_1_2026"):
            assert ctx.ssn_dict[scope_name]["tv"].sum() == pytest.approx(1.0)


class TestMultipleOptions:

    def test_two_options_same_year_target(self):
        camps = _campaigns([
            ["predefined", "bdg_tv", "2026-03-15", "2026-04-10", 27.0, "brand_1"],
            ["zero", "bdg_tv", "2026-03-15", "2026-04-10", 0.0, "brand_1"],
        ])
        ctx = fmr.campaigns_to_options(camps)
        names = [s.scope_name for s in ctx.scopes]
        assert names == ["predefined_brand_1_2026", "zero_brand_1_2026"]
        assert set(ctx.options.keys()) == {"predefined", "zero"}

        assert ctx.ssn_dict["predefined_brand_1_2026"]["tv"].sum() == pytest.approx(1.0)
        assert ctx.ssn_dict["zero_brand_1_2026"]["tv"].sum() == pytest.approx(0.0)
        assert ctx.ssn_dict["zero_brand_1_2026"]["tv"].eq(0.0).all()

        # a zero-budget option must carry explicit zeros (not an empty dict),
        # otherwise PPTX export crashes on the per-tool costs table
        assert ctx.options["zero"].budget_for_scope("zero_brand_1_2026") == {"bdg_tv": 0.0}


class TestNoTarget:

    def test_scope_without_target_no_filters(self):
        camps = pd.DataFrame({
            "option": ["predefined"], "item": ["bdg_tv"],
            "date_start": ["2026-03-01"], "date_end": ["2026-03-31"], "val": [10.0],
        })
        ctx = fmr.campaigns_to_options(camps)
        assert [s.scope_name for s in ctx.scopes] == ["predefined_2026"]
        assert ctx.scopes[0].filters == {}


class TestScopeAttributes:

    def test_scope_dates_and_seasonality_key(self):
        camps = _campaigns([["predefined", "bdg_tv", "2026-03-01", "2026-03-31", 10.0, "brand_1"]])
        ctx = fmr.campaigns_to_options(camps)
        s = ctx.scopes[0]
        assert s.date_start == pd.Timestamp("2026-01-01")
        assert s.date_end == pd.Timestamp("2026-12-31")
        assert s.ref_date_start == pd.Timestamp("2025-01-01")
        assert s.ref_date_end == pd.Timestamp("2025-12-31")
        assert s.filters == {"superbrand": ["brand_1"]}
        assert s.plan_period == "exact"
        assert s.inflation_multiplier == 1.0
        assert s.seasonality == "predefined_brand_1_2026"


class TestSaveRoundTrip:

    def test_round_trip_with_template(self, tmp_path):
        template = str(tmp_path / "context_template.xlsx")
        with pd.ExcelWriter(template, engine="openpyxl") as writer:
            pd.DataFrame({
                "item": ["bdg_tv", "bdg_dgt_perform"],
                "function": ["basic", "basic"],
                "variable": ["brnd_tv_ots_clip", "brnd_cl_dgtind_perform"],
                "base_price": [100.0, 10.0],
            }).to_excel(writer, sheet_name="translation_aba", index=False)

        camps = _campaigns([
            ["predefined", "bdg_tv", "2026-03-15", "2026-04-10", 27.0, "brand_1"],
            ["predefined", "bdg_dgt_perform", "2026-09-01", "2026-10-31", 30.0, "brand_1"],
        ])
        out_path = str(tmp_path / "context.xlsx")
        ctx = fmr.campaigns_to_options(
            camps, template_path=template, path=out_path,
        )

        assert list(ctx.trans_dicts.keys()) == ["aba"]
        assert set(ctx.trans_dicts["aba"]["function"].keys()) == {"bdg_tv", "bdg_dgt_perform"}

        ctx2 = fmr.load_context(out_path)
        assert [s.scope_name for s in ctx2.scopes] == [s.scope_name for s in ctx.scopes]
        assert set(ctx2.options.keys()) == set(ctx.options.keys())
        for scope_name, df in ctx.ssn_dict.items():
            assert scope_name in ctx2.ssn_dict
            pd.testing.assert_frame_equal(df, ctx2.ssn_dict[scope_name])

    def test_sheets_written(self, tmp_path):
        camps = _campaigns([["predefined", "bdg_tv", "2026-03-01", "2026-03-31", 10.0, "brand_1"]])
        out_path = str(tmp_path / "context.xlsx")
        fmr.campaigns_to_options(camps, path=out_path)
        xl = pd.ExcelFile(out_path)
        assert set(xl.sheet_names) == {"scopes", "allocations", "seasonality", "campaigns"}


class TestWeekGranularity:

    def test_week_split_and_percent_w_round_trip(self):
        camps = _campaigns([["predefined", "bdg_tv", "2026-01-05", "2026-02-08", 35.0, "brand_1"]])
        ctx = fmr.campaigns_to_options(camps, granularity="week")
        ssn = ctx.ssn_dict["predefined_brand_1_2026"]
        assert list(ssn["week_number"]) == list(range(1, 53))
        assert ssn["tv"].sum() == pytest.approx(1.0)
        nonzero = ssn.loc[ssn["tv"] > 0]
        assert len(nonzero) == 5
        assert nonzero["tv"].tolist() == pytest.approx([0.2] * 5)

        ds = pd.DataFrame({"date": pd.date_range("2026-01-01", "2026-12-31", freq="W-MON")})
        converted = _seasonality_to_dates(ssn.copy(), "week_number", ds,
                                          "2026-01-01", "2026-12-31")
        for _, row in converted.iterrows():
            w = int(row["date"].strftime("%W"))
            assert w in set(ssn["week_number"])
            assert ssn.loc[ssn["week_number"] == w, "tv"].iloc[0] == row["tv"]


class TestValidation:

    def test_missing_column(self):
        camps = pd.DataFrame({"option": ["x"], "item": ["bdg_tv"],
                              "date_start": ["2026-01-01"], "val": [1.0]})
        with pytest.raises(FermatricaError):
            fmr.campaigns_to_options(camps)

    def test_inverted_dates(self):
        camps = _campaigns([["x", "bdg_tv", "2026-02-01", "2026-01-01", 1.0, "brand_1"]])
        with pytest.raises(FermatricaError):
            fmr.campaigns_to_options(camps)

    def test_negative_val(self):
        camps = _campaigns([["x", "bdg_tv", "2026-01-01", "2026-01-31", -1.0, "brand_1"]])
        with pytest.raises(FermatricaError):
            fmr.campaigns_to_options(camps)

    def test_non_numeric_val(self):
        camps = pd.DataFrame({
            "option": ["x"], "item": ["bdg_tv"],
            "date_start": ["2026-01-01"], "date_end": ["2026-01-31"],
            "val": ["abc"], "target": ["brand_1"],
        })
        with pytest.raises(FermatricaError):
            fmr.campaigns_to_options(camps)

    def test_empty_table(self):
        camps = _campaigns([])
        with pytest.raises(FermatricaError):
            fmr.campaigns_to_options(camps)

    def test_unknown_granularity(self):
        camps = _campaigns([["x", "bdg_tv", "2026-01-01", "2026-01-31", 1.0, "brand_1"]])
        with pytest.raises(FermatricaError):
            fmr.campaigns_to_options(camps, granularity="quarter")


class TestFromPath:

    def test_read_from_xlsx(self, tmp_path):
        camps = _campaigns([["predefined", "bdg_tv", "2026-03-15", "2026-04-10", 27.0, "brand_1"]])
        camps_path = str(tmp_path / "campaigns.xlsx")
        camps.to_excel(camps_path, sheet_name="campaigns", index=False)

        ctx = fmr.campaigns_to_options(camps_path)
        alloc = ctx.options["predefined"].budget_for_scope("predefined_brand_1_2026")
        assert alloc["bdg_tv"] == pytest.approx(27.0)

    def test_missing_file(self, tmp_path):
        with pytest.raises(FermatricaError):
            fmr.campaigns_to_options(str(tmp_path / "nope.xlsx"))
