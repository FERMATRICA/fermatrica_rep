"""Unit tests for the OptionScope / Option manipulation API."""


from fermatrica_rep.options.option import Option
from fermatrica_rep.options.scope import OptionScope, find_scope, filter_scopes


def _make_scope(**kwargs) -> OptionScope:
    params = dict(
        scope_name="brandx_2025",
        date_start="2025-01-01",
        date_end="2025-12-31",
        ref_date_start="2024-01-01",
        ref_date_end="2024-12-31",
        filters={"superbrand": ["BrandX"], "region": ["Moscow"]},
        plan_period="year",
        inflation_multiplier=1.04,
        seasonality="brandx_default",
    )
    params.update(kwargs)
    return OptionScope(**params)


class TestStartYear:

    def test_equals_date_start_year(self):
        s = _make_scope(date_start="2026-06-01")
        assert s.start_year == 2026

    def test_works_with_eq_and_ge_comprehensions(self):
        scopes = [
            _make_scope(scope_name="a", date_start="2025-01-01"),
            _make_scope(scope_name="b", date_start="2026-01-01"),
            _make_scope(scope_name="c", date_start="2027-01-01"),
        ]
        assert [s.scope_name for s in scopes if s.start_year == 2026] == ["b"]
        assert [s.scope_name for s in scopes if s.start_year >= 2026] == ["b", "c"]


class TestHasFilter:

    def test_present_column(self):
        s = _make_scope()
        assert s.has_filter("region") is True

    def test_absent_column(self):
        s = _make_scope()
        assert s.has_filter("master") is False


class TestFilter:

    def test_matching_value(self):
        s = _make_scope(filters={"region": ["Moscow"]})
        assert s.filter("region", "Moscow") is True

    def test_differing_value(self):
        s = _make_scope(filters={"region": ["Moscow"]})
        assert s.filter("region", "Paris") is False

    def test_absent_column(self):
        s = _make_scope()
        assert s.filter("master", "x") is False

    def test_empty_filter_list(self):
        s = _make_scope(filters={"region": []})
        assert s.filter("region", "Moscow") is False

    def test_first_of_multi_value_matches(self):
        s = _make_scope(filters={"region": ["Moscow", "SPb"]})
        assert s.filter("region", "Moscow") is True


class TestScopeMatch:

    def test_explicit_value_matches(self):
        s = _make_scope(filters={"region": ["Moscow"]})
        assert s.match("region", "Moscow") is True

    def test_national_scope_matches_any_value(self):
        s = _make_scope(filters={"superbrand": ["BrandX"]})
        assert s.match("region", "Paris") is True

    def test_differing_explicit_value_does_not_match(self):
        s = _make_scope(filters={"region": ["Moscow"]})
        assert s.match("region", "Paris") is False

    def test_absent_column_without_filter_matches(self):
        s = _make_scope(filters={"superbrand": ["BrandX"]})
        assert s.match("master", "x") is True

    def test_empty_filter_list_does_not_match(self):
        s = _make_scope(filters={"region": []})
        assert s.match("region", "Moscow") is False

    def test_first_of_multi_value_matches(self):
        s = _make_scope(filters={"region": ["Moscow", "SPb"]})
        assert s.match("region", "Moscow") is True


class TestScopeCopy:

    def test_deep_copy_filters_independent(self):
        s = _make_scope()
        s2 = s.copy()
        s2.filters["region"].append("SPb")
        assert s.filters["region"] == ["Moscow"]

    def test_shallow_copy_shares_filters(self):
        s = _make_scope()
        s2 = s.copy(deep=False)
        s2.filters["region"].append("SPb")
        assert s.filters["region"] == ["Moscow", "SPb"]

    def test_fields_preserved(self):
        s = _make_scope()
        s2 = s.copy()
        assert s2.scope_name == s.scope_name
        assert s2.plan_period == s.plan_period
        assert s2.inflation_multiplier == s.inflation_multiplier
        assert s2.seasonality == s.seasonality
        assert s2.date_start == s.date_start
        assert s2.ref_date_start == s.ref_date_start


class TestReplaceDates:

    def test_only_provided_dates_change(self):
        s = _make_scope()
        s2 = s.replace_dates(date_start="2026-01-01", date_end="2026-12-31")
        assert s2.date_start.year == 2026
        assert s2.date_end.year == 2026
        assert s2.ref_date_start == s.ref_date_start
        assert s2.ref_date_end == s.ref_date_end

    def test_returns_new_object_original_untouched(self):
        s = _make_scope()
        s2 = s.replace_dates(date_start="2026-01-01")
        assert s is not s2
        assert s.date_start.year == 2025

    def test_other_fields_preserved(self):
        s = _make_scope()
        s2 = s.replace_dates(date_start="2026-01-01")
        assert s2.scope_name == s.scope_name
        assert s2.filters == s.filters
        assert s2.seasonality == s.seasonality


class TestShiftYears:

    def test_all_dates_rolled(self):
        s = _make_scope()
        s2 = s.shift_years()
        assert s2.date_start.year == 2026
        assert s2.date_end.year == 2026
        assert s2.ref_date_start.year == 2025
        assert s2.ref_date_end.year == 2025

    def test_scope_name_unchanged(self):
        s = _make_scope()
        assert s.shift_years().scope_name == s.scope_name

    def test_feb_29_does_not_raise(self):
        s = _make_scope(date_start="2020-02-29", date_end="2020-02-29",
                        ref_date_start="2020-02-29", ref_date_end="2020-02-29")
        s2 = s.shift_years()
        assert s2.date_start.year == 2021

    def test_delta_two(self):
        s = _make_scope()
        s2 = s.shift_years(delta=2)
        assert s2.date_start.year == 2027
        assert s2.ref_date_start.year == 2026

    def test_original_untouched(self):
        s = _make_scope()
        s.shift_years()
        assert s.date_start.year == 2025
        assert s.ref_date_start.year == 2024


class TestOptionCopy:

    def test_independent_budget_data(self):
        opt = Option(name="opt1", budget_data={"a": {"bdg_tv": 50}, "b": {"bdg_tv": 30}})
        opt2 = opt.copy()
        opt2.budget_data["a"]["bdg_tv"] = 999
        assert opt.budget_data["a"]["bdg_tv"] == 50
        opt2.budget_data["c"] = {"bdg_x": 1}
        assert "c" not in opt.budget_data

    def test_preserves_metadata(self):
        opt = Option(name="opt1", budget_data={"a": {"bdg_tv": 50}},
                     default_scope="a", text_flags="name_2025")
        opt2 = opt.copy()
        assert opt2.name == "opt1"
        assert opt2.default_scope == "a"
        assert opt2.text_flags == "name_2025"

    def test_deep_false_also_independent_inner_dicts(self):
        opt = Option(name="opt1", budget_data={"a": {"bdg_tv": 50}})
        opt2 = opt.copy(deep=False)
        opt2.budget_data["a"]["bdg_tv"] = 999
        assert opt.budget_data["a"]["bdg_tv"] == 50


class TestReplaceBudget:

    def test_replaces_scope_items(self):
        opt = Option(name="opt1", budget_data={
            "a": {"bdg_tv": 50, "bdg_digital": 30},
            "b": {"bdg_pr": 10},
        })
        opt2 = opt.replace_budget("a", {"bdg_tv": 80})
        assert opt2.budget_data["a"] == {"bdg_tv": 80}
        assert opt2.budget_data["b"] == {"bdg_pr": 10}

    def test_other_scopes_independent_of_original(self):
        opt = Option(name="opt1", budget_data={"a": {"bdg_tv": 50}, "b": {"bdg_pr": 10}})
        opt2 = opt.replace_budget("a", {"bdg_tv": 80})
        opt2.budget_data["b"]["bdg_pr"] = 999
        assert opt.budget_data["b"]["bdg_pr"] == 10

    def test_preserves_metadata(self):
        opt = Option(name="opt1", budget_data={"a": {"bdg_tv": 50}},
                     default_scope="a", text_flags="name_2025")
        opt2 = opt.replace_budget("a", {"bdg_tv": 80})
        assert opt2.name == "opt1"
        assert opt2.default_scope == "a"
        assert opt2.text_flags == "name_2025"

    def test_original_untouched(self):
        opt = Option(name="opt1", budget_data={"a": {"bdg_tv": 50, "bdg_digital": 30}})
        opt.replace_budget("a", {"bdg_tv": 80})
        assert opt.budget_data["a"] == {"bdg_tv": 50, "bdg_digital": 30}


class TestFindScope:

    def test_found(self):
        scopes = [_make_scope(), _make_scope(scope_name="brandy_2025")]
        assert find_scope(scopes, "brandy_2025").scope_name == "brandy_2025"

    def test_not_found(self):
        scopes = [_make_scope()]
        assert find_scope(scopes, "missing") is None

    def test_none_scopes(self):
        assert find_scope(None, "any") is None

    def test_exported_from_package(self):
        import fermatrica_rep as fmr
        assert fmr.find_scope is find_scope


class TestFilterScopes:

    def _make_scopes(self):
        return [
            _make_scope(scope_name="brandx_2025", date_start="2025-01-01",
                        filters={"superbrand": ["BrandX"], "region": ["Moscow"]}),
            _make_scope(scope_name="brandx_2026", date_start="2026-01-01",
                        filters={"superbrand": ["BrandX"], "region": ["Moscow"]}),
            _make_scope(scope_name="brandy_2026", date_start="2026-01-01",
                        filters={"superbrand": ["BrandY"]}),
            _make_scope(scope_name="brandy_2027", date_start="2027-01-01",
                        filters={"superbrand": ["BrandY"]}),
        ]

    def test_scope_name_str(self):
        out = filter_scopes(self._make_scopes(), scope_name="brandx_2026")
        assert [s.scope_name for s in out] == ["brandx_2026"]

    def test_scope_name_list_any_of(self):
        out = filter_scopes(self._make_scopes(), scope_name=["brandx_2025", "brandy_2027"])
        assert [s.scope_name for s in out] == ["brandx_2025", "brandy_2027"]

    def test_start_year(self):
        out = filter_scopes(self._make_scopes(), start_year=2026)
        assert [s.scope_name for s in out] == ["brandx_2026", "brandy_2026"]

    def test_start_year_min(self):
        out = filter_scopes(self._make_scopes(), start_year_min=2026)
        assert [s.scope_name for s in out] == ["brandx_2026", "brandy_2026", "brandy_2027"]

    def test_start_year_max(self):
        out = filter_scopes(self._make_scopes(), start_year_max=2026)
        assert [s.scope_name for s in out] == ["brandx_2025", "brandx_2026", "brandy_2026"]

    def test_has_filter(self):
        out = filter_scopes(self._make_scopes(), has_filter="region")
        assert [s.scope_name for s in out] == ["brandx_2025", "brandx_2026"]

    def test_without_filter(self):
        out = filter_scopes(self._make_scopes(), without_filter="region")
        assert [s.scope_name for s in out] == ["brandy_2026", "brandy_2027"]

    def test_match_explicit_value(self):
        out = filter_scopes(self._make_scopes(), match=("region", "Moscow"))
        assert [s.scope_name for s in out] == ["brandx_2025", "brandx_2026", "brandy_2026", "brandy_2027"]

    def test_match_national_scopes_only(self):
        out = filter_scopes(self._make_scopes(), match=("region", "Paris"))
        assert [s.scope_name for s in out] == ["brandy_2026", "brandy_2027"]

    def test_match_composed_with_start_year(self):
        out = filter_scopes(self._make_scopes(), start_year=2026, match=("region", "Moscow"))
        assert [s.scope_name for s in out] == ["brandx_2026", "brandy_2026"]

    def test_and_composition(self):
        out = filter_scopes(self._make_scopes(), start_year_min=2026, without_filter="region")
        assert [s.scope_name for s in out] == ["brandy_2026", "brandy_2027"]

    def test_order_preserved(self):
        scopes = list(reversed(self._make_scopes()))
        out = filter_scopes(scopes, start_year=2026)
        assert [s.scope_name for s in out] == ["brandy_2026", "brandx_2026"]

    def test_none_scopes(self):
        assert filter_scopes(None, start_year=2026) == []

    def test_no_matches_returns_empty(self):
        assert filter_scopes(self._make_scopes(), start_year=2030) == []

    def test_no_criteria_returns_all(self):
        scopes = self._make_scopes()
        out = filter_scopes(scopes)
        assert out == scopes

    def test_input_list_not_mutated(self):
        scopes = self._make_scopes()
        original = list(scopes)
        filter_scopes(scopes, start_year=2026)
        assert scopes == original
        assert all(s.filters is o.filters for s, o in zip(scopes, original))

    def test_exported_from_package(self):
        import fermatrica_rep as fmr
        assert fmr.filter_scopes is filter_scopes
