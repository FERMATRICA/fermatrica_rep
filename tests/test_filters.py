"""Tests for the unified filtering interface (brands / brands_var / filters).

Covers the shared helpers in `fermatrica_rep.basics`, the analysis-layer
filtering (waterfall_data / decompose / adstocks), the options_table
replace-default semantics, and the waterfall slide scope-union merge.
"""
import logging

import numpy as np
import pandas as pd
import pytest

from fermatrica_rep.basics import _merge_filters, _filter_df
from fermatrica_rep.analysis.waterfall import waterfall_data


@pytest.fixture
def split():
    np.random.seed(7)
    n = 12
    return pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=n, freq='MS'),
        'superbrand': ['brand_1'] * 6 + ['brand_2'] * 6,
        'bs_key': [1, 2] * 6,
        'region': ['Moscow', 'Moscow', 'SPb', 'SPb'] * 3,
        'listed': [2] * n,
        'variable': ['tv'] * n,
        'value': list(range(1, n + 1)),
    })


class TestMergeFilters:

    def test_brands_overwrite_same_key(self):
        result = _merge_filters(brands=['brand_1'], filters={'superbrand': ['brand_2']})
        assert result == {'superbrand': ['brand_1']}

    def test_distinct_keys_combined(self):
        result = _merge_filters(brands=['brand_1'], filters={'region': ['Moscow']})
        assert result == {'superbrand': ['brand_1'], 'region': ['Moscow']}

    def test_none_brands_keeps_user_filters(self):
        assert _merge_filters(filters={'region': ['Moscow']}) == {'region': ['Moscow']}

    def test_brands_var_applied(self):
        result = _merge_filters(brands=[1, 2], brands_var='bs_key')
        assert result == {'bs_key': [1, 2]}


class TestFilterDf:

    def test_simple_and_complex_are_and(self, split):
        result = _filter_df(split, brands=['brand_1'], filters={'region': ['Moscow']})
        expected = split[(split['superbrand'] == 'brand_1') & (split['region'] == 'Moscow')]
        assert result.equals(expected)

    def test_missing_column_warns_and_skips(self, split, caplog):
        with caplog.at_level(logging.WARNING):
            result = _filter_df(split, brands=['brand_1'], filters={'ghost': ['x']})
        assert any('ghost' in r.getMessage() for r in caplog.records)
        assert result.equals(split[split['superbrand'] == 'brand_1'])

    def test_no_filters_returns_unchanged(self, split):
        result = _filter_df(split)
        assert len(result) == len(split)


class TestWaterfallFiltering:

    def test_simple_filter_equivalence(self, split):
        r_brands = waterfall_data(split, brands=['brand_1'],
                                  date_start='2024-01-01', date_end='2024-12-31')
        r_brands_var = waterfall_data(split, brands=['brand_1'], brands_var='superbrand',
                                      date_start='2024-01-01', date_end='2024-12-31')
        r_filters = waterfall_data(split, filters={'superbrand': ['brand_1']},
                                   date_start='2024-01-01', date_end='2024-12-31')
        assert r_brands.equals(r_filters)
        assert r_brands.equals(r_brands_var)

    def test_brands_var_bs_key_reproduces_bs_key_filter(self, split):
        r_var = waterfall_data(split, brands=[1], brands_var='bs_key',
                               date_start='2024-01-01', date_end='2024-12-31')
        r_filters = waterfall_data(split, filters={'bs_key': [1]},
                                   date_start='2024-01-01', date_end='2024-12-31')
        assert r_var.equals(r_filters)
        expected = split[split['bs_key'] == 1]['value'].sum()
        assert r_var['value'].sum() == expected

    def test_complex_filter_combined_with_brands(self, split):
        result = waterfall_data(split, brands=['brand_1'], filters={'region': ['Moscow']},
                                date_start='2024-01-01', date_end='2024-12-31')
        expected = split[(split['superbrand'] == 'brand_1') & (split['region'] == 'Moscow')]
        assert result['value'].sum() == expected['value'].sum()

    def test_missing_filter_column_no_crash(self, split, caplog):
        with caplog.at_level(logging.WARNING):
            result = waterfall_data(split, brands=['brand_1'], filters={'ghost': ['x']},
                                    date_start='2024-01-01', date_end='2024-12-31')
        assert len(result) > 0
        assert result['value'].sum() == split[split['superbrand'] == 'brand_1']['value'].sum()

    def test_brands_optional_all_data(self, split):
        result = waterfall_data(split, date_start='2024-01-01', date_end='2024-12-31')
        assert result['value'].sum() == split['value'].sum()


class TestOptionsTableReplaceSemantics:

    @pytest.fixture
    def meta_model(self):
        from types import SimpleNamespace

        from fermatrica_rep.meta_model.meta_model import MetaModel

        dt_pred = pd.DataFrame({
            'date': pd.date_range('2026-01-01', periods=3, freq='MS'),
            'superbrand': ['brand_1', 'brand_1', 'brand_2'],
            'bs_key': [1, 2, 3],
            'listed': [4, 4, 4],
        })
        node = SimpleNamespace(
            dt_pred=dt_pred,
            model=SimpleNamespace(conf=SimpleNamespace(target_superbrand='brand_1')),
        )
        return MetaModel(nodes={'model': node})

    def _capture_create(self, monkeypatch):
        from fermatrica_rep.export import export_pptx

        captured = {}

        def fake_create(prs, meta_model, scopes, options, **kwargs):
            captured.update(kwargs)
            return prs

        monkeypatch.setattr(export_pptx.slides.options, 'create', fake_create)
        return captured

    def test_no_filters_uses_target_brand_bs_keys(self, meta_model, monkeypatch):
        from fermatrica_rep.export.export_pptx import options_table

        captured = self._capture_create(monkeypatch)
        options_table(object(), meta_model, [object()], {'opt': {}})
        assert list(captured['filters']['bs_key']) == [1, 2]

    def test_any_user_filter_uses_only_user(self, meta_model, monkeypatch):
        from fermatrica_rep.export.export_pptx import options_table

        captured = self._capture_create(monkeypatch)
        options_table(object(), meta_model, [object()], {'opt': {}},
                      filters={'region': ['Moscow']})
        assert captured['filters'] == {'region': ['Moscow']}
        assert 'bs_key' not in captured['filters']

    def test_user_brands_replace_default(self, meta_model, monkeypatch):
        from fermatrica_rep.export.export_pptx import options_table

        captured = self._capture_create(monkeypatch)
        options_table(object(), meta_model, [object()], {'opt': {}},
                      brands=['brand_2'])
        # the user's simple filter is passed through; the auto bs_key default is dropped
        assert captured['brands'] == ['brand_2']
        assert captured['filters'] is None


class TestAutoBsKeyFilters:
    """_auto_bs_key_filters derives target-brand bs_keys from dt_pred, falling
    back to the node's ds when dt_pred is not populated (freshly loaded models)."""

    @pytest.fixture
    def model(self):
        from types import SimpleNamespace

        return SimpleNamespace(conf=SimpleNamespace(target_superbrand='brand_1'))

    def _meta_model(self, dt_pred, ds, model):
        from types import SimpleNamespace

        from fermatrica_rep.meta_model.meta_model import MetaModel

        node = SimpleNamespace(dt_pred=dt_pred, ds=ds, model=model)
        return MetaModel(nodes={'model': node})

    def _data(self):
        return pd.DataFrame({
            'superbrand': ['brand_1', 'brand_1', 'brand_2'],
            'bs_key': [1, 2, 3],
            'listed': [4, 4, 4],
        })

    def test_derives_from_dt_pred(self, model):
        from fermatrica_rep.export.export_pptx import _auto_bs_key_filters

        mm = self._meta_model(self._data(), self._data(), model)
        assert list(_auto_bs_key_filters(mm, 'model')['bs_key']) == [1, 2]

    def test_falls_back_to_ds_when_dt_pred_none(self, model):
        from fermatrica_rep.export.export_pptx import _auto_bs_key_filters

        # freshly loaded MetaModel: dt_pred is not persisted, ds is populated
        mm = self._meta_model(None, self._data(), model)
        assert list(_auto_bs_key_filters(mm, 'model')['bs_key']) == [1, 2]

    def test_returns_none_when_no_source(self, model):
        from fermatrica_rep.export.export_pptx import _auto_bs_key_filters

        mm = self._meta_model(None, None, model)
        assert _auto_bs_key_filters(mm, 'model') is None

    def test_returns_none_without_superbrand_column(self, model):
        from fermatrica_rep.export.export_pptx import _auto_bs_key_filters

        df = pd.DataFrame({'bs_key': [1, 2], 'listed': [4, 4]})
        mm = self._meta_model(df, None, model)
        assert _auto_bs_key_filters(mm, 'model') is None


class TestDerivedBrandsDoNotOverwriteUserFilters:
    """Auto-derived `brands` (when None) must not override a user-supplied
    same-column `filters` entry on the plot paths."""

    @pytest.fixture
    def split(self):
        n = 8
        return pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=n, freq='MS'),
            'superbrand': ['brand_1'] * 4 + ['brand_2'] * 4,
            'bs_key': [1, 1, 2, 2, 3, 3, 4, 4],
            'listed': [2] * n,
            'variable': ['tv'] * 4 + ['digital'] * 4,
            'value': list(range(1, n + 1)),
        })

    @pytest.fixture
    def model_rep(self, split):
        from fermatrica_rep.meta_model.model_rep import ModelRep

        return ModelRep(split, language='english')

    def _bar_values(self, fig):
        values = []
        for tr in fig.data:
            if hasattr(tr, 'y') and tr.y is not None:
                values.append(sum(v for v in tr.y if v is not None))
        return values

    def test_waterfall_plot_filters_apply(self, split, model_rep):
        from fermatrica_rep.analysis.waterfall import waterfall_plot

        fig_filters = waterfall_plot(split, None, model_rep,
                                     filters={'superbrand': ['brand_1']},
                                     date_start='2024-01-01', date_end='2024-12-31')
        fig_brands = waterfall_plot(split, ['brand_1'], model_rep,
                                    date_start='2024-01-01', date_end='2024-12-31')
        assert self._bar_values(fig_filters) == self._bar_values(fig_brands)
        # brand_1 only -> single variable (tv); brand_2's digital must not appear
        vars_in = set()
        for tr in fig_filters.data:
            if hasattr(tr, 'x') and tr.x is not None:
                vars_in.update(str(v) for v in tr.x)
        assert vars_in == {'tv'}

    def test_decompose_main_plot_filters_apply(self, split, model_rep):
        from fermatrica_rep.analysis.decomposition import decompose_main_plot

        fig_filters = decompose_main_plot(split, brands=None, model_rep=model_rep,
                                          filters={'superbrand': ['brand_1']},
                                          period='m', show_future=True, show_pretrain=True)
        fig_brands = decompose_main_plot(split, brands=['brand_1'], model_rep=model_rep,
                                         period='m', show_future=True, show_pretrain=True)
        assert fig_filters.layout.title.text == fig_brands.layout.title.text
        assert 'brand_1' in fig_filters.layout.title.text
        assert 'brand_2' not in fig_filters.layout.title.text
        assert self._bar_values(fig_filters) == self._bar_values(fig_brands)

    def test_decompose_chart_data_brands_var(self, split):
        from fermatrica_rep.export.slides.decomposition import _decompose_chart_data

        result, sku = _decompose_chart_data(split, brands=[1], brands_var='bs_key',
                                            show_future=True, period='m',
                                            group_var=['superbrand'], if_volume=True)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        expected = split[split['bs_key'] == 1]
        assert result['tv'].sum() == expected[expected['variable'] == 'tv']['value'].sum()


class TestWaterfallSlideScopeUnion:
    def _scope(self, name, region):
        from fermatrica_rep.options.scope import OptionScope

        return OptionScope(
            scope_name=name,
            date_start='2026-01-01',
            date_end='2026-12-31',
            ref_date_start='2025-01-01',
            ref_date_end='2025-12-31',
            filters={'superbrand': ['brand_1'], 'region': [region]},
        )

    def test_user_filters_anded_with_scope_union(self):
        from fermatrica_rep.export.export_pptx import _merge_filter_dicts, _union_scope_filters
        from fermatrica_rep.export.slides.waterfall import _waterfall_chart_data

        scopes = [self._scope('moscow', 'Moscow'), self._scope('spb', 'SPb')]
        user_filters = {'region': ['Moscow', 'SPb', 'Kazan']}

        eff = _merge_filter_dicts(user_filters, _union_scope_filters(scopes))
        assert eff == {'superbrand': ['brand_1'], 'region': ['Moscow', 'SPb']}

        n = 12
        split = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=n, freq='MS'),
            'superbrand': ['brand_1'] * n,
            'region': ['Moscow'] * 4 + ['SPb'] * 4 + ['Kazan'] * 4,
            'bs_key': [1] * n,
            'listed': [2] * n,
            'variable': ['tv'] * 8 + ['digital'] * 4,
            'value': list(range(1, n + 1)),
        })
        result = _waterfall_chart_data(split=split, brands=['brand_1'],
                                       date_start='2024-01-01', date_end='2024-12-31',
                                       if_volume=True,
                                       filters=eff)
        expected = _waterfall_chart_data(split=split[split['region'].isin(['Moscow', 'SPb'])],
                                         brands=['brand_1'],
                                         date_start='2024-01-01', date_end='2024-12-31',
                                         if_volume=True)
        assert result.equals(expected)
        # Kazan rows (with the digital variable) are excluded by the ANDed filters
        assert len(result) == 1
        unfiltered = _waterfall_chart_data(split=split, brands=['brand_1'],
                                           date_start='2024-01-01', date_end='2024-12-31',
                                           if_volume=True)
        assert len(unfiltered) == 2

    def test_scope_union_no_user_filters(self):
        from fermatrica_rep.export.export_pptx import _merge_filter_dicts, _union_scope_filters

        scopes = [self._scope('moscow', 'Moscow'), self._scope('spb', 'SPb')]
        eff = _merge_filter_dicts(None, _union_scope_filters(scopes))
        assert eff == {'superbrand': ['brand_1'], 'region': ['Moscow', 'SPb']}


class TestAdstocksDefault:

    def test_defaults_to_target_superbrand(self, monkeypatch):
        from fermatrica_rep.analysis.adstocks import adstocks_data
        from fermatrica_rep.meta_model.model_rep import ModelRep
        from fermatrica_rep.meta_model.node import ModelNode
        from types import SimpleNamespace

        ds = pd.DataFrame({
            'superbrand': ['brand_1', 'brand_1'],
            'bs_key': [1, 1],
            'date': pd.date_range('2024-01-01', periods=2, freq='MS'),
        })
        mock_model = SimpleNamespace(conf=SimpleNamespace(target_superbrand='brand_1'))
        model_rep = ModelRep(ds, language='english')
        node = ModelNode(model=mock_model, ds=ds, model_rep=model_rep)

        captured = {}

        def fake_worker(model, ds, brands, brands_var, filters, cln_meas, cln_dim, n, if_scaled):
            captured['brands'] = brands
            captured['brands_var'] = brands_var
            captured['filters'] = filters
            return pd.DataFrame()

        monkeypatch.setattr('fermatrica_rep.analysis.adstocks._adstocks_data_worker_inner',
                            fake_worker)
        adstocks_data(node)
        assert captured['brands'] is None
        assert captured['brands_var'] == 'superbrand'
        assert captured['filters'] == {'superbrand': ['brand_1']}

    def test_explicit_filters_used(self, monkeypatch):
        from fermatrica_rep.analysis.adstocks import adstocks_data
        from fermatrica_rep.meta_model.model_rep import ModelRep
        from fermatrica_rep.meta_model.node import ModelNode
        from types import SimpleNamespace

        ds = pd.DataFrame({
            'superbrand': ['brand_1', 'brand_1'],
            'bs_key': [1, 1],
            'date': pd.date_range('2024-01-01', periods=2, freq='MS'),
        })
        mock_model = SimpleNamespace(conf=SimpleNamespace(target_superbrand='brand_1'))
        model_rep = ModelRep(ds, language='english')
        node = ModelNode(model=mock_model, ds=ds, model_rep=model_rep)

        captured = {}

        def fake_worker(model, ds, brands, brands_var, filters, cln_meas, cln_dim, n, if_scaled):
            captured['filters'] = filters
            captured['brands'] = brands
            return pd.DataFrame()

        monkeypatch.setattr('fermatrica_rep.analysis.adstocks._adstocks_data_worker_inner',
                            fake_worker)
        adstocks_data(node, brands=['brand_2'])
        assert captured['brands'] == ['brand_2']
        assert captured['filters'] is None
