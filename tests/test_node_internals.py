"""Unit tests for ModelNode internals.

Covers _build_bdg_ssn_cols, _apply_media_ssn, _translate_budget, translate,
and compute() with real translation (budget > 0).
"""


import logging
import numpy as np
import pandas as pd
import pytest

from fermatrica.basics.basics import FermatricaError
from fermatrica_rep.options.scope import OptionScope
from fermatrica_rep.options.option import Option
from fermatrica_rep.meta_model.node import ModelNode
from fermatrica_rep.meta_model.model_rep import ModelRep


class MockConf:
    price_var = None
    summary_type = "sum"
    Y_var = "y"
    model_type = "OLS"
    params: dict = {}


class MockModel:
    def __init__(self):
        self.conf = MockConf()
        self.obj = type("Obj", (), {"models": {"main": None}})()


class UnfittedMockModel:
    def __init__(self):
        self.conf = MockConf()
        self.obj = type("Obj", (), {"models": {}})()


class PicklableMockModel:
    """Module-level MockModel whose instance survives pickle.dumps (spawn workers)."""

    def __init__(self):
        self.conf = MockConf()
        self.obj = PicklableMockObj()


class PicklableMockObj:
    models = {"main": None}
    transform_lhs_fn = None
    custom_predict_fn = None


def _basic_ds(n=30):
    return pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=n, freq="D"),
        "listed": [4] * n,
        "predicted": [100.0] * n,
        "observed": [90.0] * n,
    })


def _basic_scope(**overrides):
    kwargs = {
        "scope_name": "test_scope",
        "date_start": "2025-01-01",
        "date_end": "2025-12-31",
        "ref_date_start": "2024-01-01",
        "ref_date_end": "2024-12-31",
    }
    kwargs.update(overrides)
    return OptionScope(**kwargs)


def _basic_ds_with_ssn(n=365):
    ds = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=n, freq="D"),
        "listed": [4] * n,
        "predicted": [100.0] * n,
        "observed": [90.0] * n,
        "brnd_tv_ots_clip": [0.0] * n,
        "opt_media_ssn_tv": [0.5] * n,
    })
    ds["period"] = "day"
    return ds


class TestBuildBdgSsnCols:

    def test_extracts_bdg_prefix_keys(self):
        trans_dict = {
            "function": {"bdg_tv": "basic", "bdg_digital": "basic", "nobdg_price": "basic"},
            "variable": {"bdg_tv": "x1", "bdg_digital": "x2", "nobdg_price": "x3"},
        }
        model_rep = ModelRep(pd.DataFrame({"date": [pd.Timestamp("2025-01-01")], "superbrand": ["x"]}))
        model_rep.trans_dict = trans_dict
        node = ModelNode(MockModel(), _basic_ds(), model_rep=model_rep)

        result = node._build_bdg_ssn_cols()
        assert result == {"tv", "digital"}

    def test_returns_empty_when_trans_dict_none(self):
        model_rep = ModelRep(pd.DataFrame({"date": [pd.Timestamp("2025-01-01")], "superbrand": ["x"]}))
        model_rep.trans_dict = None
        node = ModelNode(MockModel(), _basic_ds(), model_rep=model_rep)

        result = node._build_bdg_ssn_cols()
        assert result == set()

    def test_ignores_non_bdg_keys(self):
        trans_dict = {
            "function": {"nobdg_price": "basic", "other_key": "basic"},
            "variable": {"nobdg_price": "x1", "other_key": "x2"},
        }
        model_rep = ModelRep(pd.DataFrame({"date": [pd.Timestamp("2025-01-01")], "superbrand": ["x"]}))
        model_rep.trans_dict = trans_dict
        node = ModelNode(MockModel(), _basic_ds(), model_rep=model_rep)

        result = node._build_bdg_ssn_cols()
        assert result == set()


class TestApplyMediaSsn:
    """Tests for _apply_media_ssn using monkeypatching."""

    def test_noop_when_ssn_dict_none(self):
        ds = _basic_ds()
        model_rep = ModelRep(pd.DataFrame({"date": [pd.Timestamp("2025-01-01")], "superbrand": ["x"]}))
        model_rep.ssn_dict = None
        node = ModelNode(MockModel(), ds, model_rep=model_rep)
        scope = _basic_scope()

        result = node._apply_media_ssn(ds, scope, set())
        assert result.equals(ds)

    def test_noop_when_key_not_found(self, monkeypatch):
        ds = _basic_ds()
        model_rep = ModelRep(pd.DataFrame({"date": [pd.Timestamp("2025-01-01")], "superbrand": ["x"]}))
        model_rep.ssn_dict = {"other": pd.DataFrame({"period": [1, 2], "val": [0.5, 0.5]})}
        node = ModelNode(MockModel(), ds, model_rep=model_rep)
        scope = _basic_scope(scope_name="test_scope")

        called = []

        def fake_media_ssn_apply(**kwargs):
            called.append(kwargs)
            return kwargs["ds"]

        monkeypatch.setattr("fermatrica_rep.options.define.media_ssn_apply", fake_media_ssn_apply)

        result = node._apply_media_ssn(ds, scope, set())
        assert not called
        assert result.equals(ds)

    def test_calls_media_ssn_apply_when_key_found(self, monkeypatch):
        ds = _basic_ds()
        model_rep = ModelRep(pd.DataFrame({"date": [pd.Timestamp("2025-01-01")], "superbrand": ["x"]}))
        ssn_df = pd.DataFrame({"period": [1, 2], "val": [0.5, 0.5]})
        model_rep.ssn_dict = {"test_scope": ssn_df}
        node = ModelNode(MockModel(), ds, model_rep=model_rep)
        scope = _basic_scope(scope_name="test_scope")

        called = []

        def fake_media_ssn_apply(**kwargs):
            called.append(kwargs)
            return kwargs["ds"]

        monkeypatch.setattr("fermatrica_rep.options.define.media_ssn_apply", fake_media_ssn_apply)

        node._apply_media_ssn(ds, scope, set())
        assert len(called) == 1
        assert called[0]["ds"] is ds
        assert called[0]["dt_ssn"] is ssn_df

    def test_fallback_to_scope_name_when_seasonality_none(self, monkeypatch):
        ds = _basic_ds()
        model_rep = ModelRep(pd.DataFrame({"date": [pd.Timestamp("2025-01-01")], "superbrand": ["x"]}))
        ssn_df = pd.DataFrame({"period": [1, 2], "val": [0.5, 0.5]})
        model_rep.ssn_dict = {"test_scope": ssn_df}
        node = ModelNode(MockModel(), ds, model_rep=model_rep)
        scope = _basic_scope(scope_name="test_scope", seasonality=None)

        called = []

        def fake_media_ssn_apply(**kwargs):
            called.append(kwargs)
            return kwargs["ds"]

        monkeypatch.setattr("fermatrica_rep.options.define.media_ssn_apply", fake_media_ssn_apply)

        node._apply_media_ssn(ds, scope, set())
        assert len(called) == 1
        assert called[0]["dt_ssn"] is ssn_df

    def test_uses_seasonality_attr_when_set(self, monkeypatch):
        ds = _basic_ds()
        model_rep = ModelRep(pd.DataFrame({"date": [pd.Timestamp("2025-01-01")], "superbrand": ["x"]}))
        ssn_df_custom = pd.DataFrame({"period": [1, 2], "val": [0.7, 0.3]})
        model_rep.ssn_dict = {"custom_ssn": ssn_df_custom}
        node = ModelNode(MockModel(), ds, model_rep=model_rep)
        scope = _basic_scope(scope_name="test_scope", seasonality="custom_ssn")

        called = []

        def fake_media_ssn_apply(**kwargs):
            called.append(kwargs)
            return kwargs["ds"]

        monkeypatch.setattr("fermatrica_rep.options.define.media_ssn_apply", fake_media_ssn_apply)

        node._apply_media_ssn(ds, scope, set())
        assert len(called) == 1
        assert called[0]["dt_ssn"] is ssn_df_custom


class TestTranslateBudget:
    """Tests for _translate_budget dispatch mechanism."""

    def test_translates_budget_via_basic(self):
        ds = _basic_ds_with_ssn()
        trans_dict = {
            "function": {"bdg_tv": "basic"},
            "variable": {"bdg_tv": "brnd_tv_ots_clip"},
            "base_price": {"bdg_tv": 10.0},
        }
        model_rep = ModelRep(pd.DataFrame({"date": [pd.Timestamp("2025-01-01")], "superbrand": ["x"]}))
        model_rep.trans_dict = trans_dict
        model_rep.media_prices = None
        node = ModelNode(MockModel(), ds.copy(), model_rep=model_rep)
        scope = _basic_scope()

        mask = pd.Series([True] * 10 + [False] * 355, index=ds.index)
        budget_items = {"bdg_tv": 50}

        result = node._translate_budget(ds, node.model.conf, budget_items, mask, scope)

        expected = 0.5 * 50 * 1e6 / 10.0
        assert result.loc[0, "brnd_tv_ots_clip"] == pytest.approx(expected)

    def test_skips_unknown_budget_key(self):
        ds = _basic_ds()
        trans_dict = {
            "function": {"bdg_known": "basic"},
            "variable": {"bdg_known": "some_var"},
        }
        model_rep = ModelRep(pd.DataFrame({"date": [pd.Timestamp("2025-01-01")], "superbrand": ["x"]}))
        model_rep.trans_dict = trans_dict
        node = ModelNode(MockModel(), ds.copy(), model_rep=model_rep)
        scope = _basic_scope()

        ds_before = ds.copy()
        mask = pd.Series([True] * 10 + [False] * 20, index=ds.index)
        budget_items = {"bdg_unknown": 50}

        result = node._translate_budget(ds, node.model.conf, budget_items, mask, scope)
        assert result.equals(ds_before)

    def test_skips_unresolvable_function(self):
        ds = _basic_ds()
        trans_dict = {
            "function": {"bdg_tv": "no_such_function_xyz"},
            "variable": {"bdg_tv": "brnd_tv_ots_clip"},
        }
        model_rep = ModelRep(pd.DataFrame({"date": [pd.Timestamp("2025-01-01")], "superbrand": ["x"]}))
        model_rep.trans_dict = trans_dict
        node = ModelNode(MockModel(), ds.copy(), model_rep=model_rep)
        scope = _basic_scope()

        ds_before = ds.copy()
        mask = pd.Series([True] * 10 + [False] * 20, index=ds.index)
        budget_items = {"bdg_tv": 50}

        result = node._translate_budget(ds, node.model.conf, budget_items, mask, scope)
        assert result.equals(ds_before)

    def test_noop_when_trans_dict_none(self):
        ds = _basic_ds()
        model_rep = ModelRep(pd.DataFrame({"date": [pd.Timestamp("2025-01-01")], "superbrand": ["x"]}))
        model_rep.trans_dict = None
        node = ModelNode(MockModel(), ds.copy(), model_rep=model_rep)

        ds_before = ds.copy()
        mask = pd.Series([True] * 10 + [False] * 20, index=ds.index)
        result = node._translate_budget(ds, node.model.conf, {"bdg_tv": 50}, mask)
        assert result.equals(ds_before)

    def test_scope_inflation_multiplier(self):
        ds = _basic_ds_with_ssn()
        trans_dict = {
            "function": {"bdg_tv": "basic"},
            "variable": {"bdg_tv": "brnd_tv_ots_clip"},
            "base_price": {"bdg_tv": 10.0},
        }
        model_rep = ModelRep(pd.DataFrame({"date": [pd.Timestamp("2025-01-01")], "superbrand": ["x"]}))
        model_rep.trans_dict = trans_dict
        model_rep.media_prices = {}
        node = ModelNode(MockModel(), ds.copy(), model_rep=model_rep)
        scope = _basic_scope(inflation_multiplier=2.0)

        mask = pd.Series([True] * 10 + [False] * 355, index=ds.index)
        budget_items = {"bdg_tv": 50}

        result = node._translate_budget(ds, node.model.conf, budget_items, mask, scope)

        expected = 0.5 * 50 * 1e6 / (10.0 * 2.0)
        assert result.loc[0, "brnd_tv_ots_clip"] == pytest.approx(expected)

    def test_per_tool_media_prices_override(self):
        ds = _basic_ds_with_ssn()
        trans_dict = {
            "function": {"bdg_tv": "basic"},
            "variable": {"bdg_tv": "brnd_tv_ots_clip"},
            "base_price": {"bdg_tv": 10.0},
        }
        model_rep = ModelRep(pd.DataFrame({"date": [pd.Timestamp("2025-01-01")], "superbrand": ["x"]}))
        model_rep.trans_dict = trans_dict
        model_rep.media_prices = {"test_scope": {"bdg_tv": 3.0}}
        node = ModelNode(MockModel(), ds.copy(), model_rep=model_rep)
        scope = _basic_scope(scope_name="test_scope", inflation_multiplier=2.0)

        mask = pd.Series([True] * 10 + [False] * 355, index=ds.index)
        budget_items = {"bdg_tv": 50}

        result = node._translate_budget(ds, node.model.conf, budget_items, mask, scope)

        expected = 0.5 * 50 * 1e6 / (10.0 * 3.0)
        assert result.loc[0, "brnd_tv_ots_clip"] == pytest.approx(expected)

    def test_empty_budget_items_is_noop(self):
        ds = _basic_ds()
        trans_dict = {
            "function": {"bdg_tv": "basic"},
            "variable": {"bdg_tv": "brnd_tv_ots_clip"},
        }
        model_rep = ModelRep(pd.DataFrame({"date": [pd.Timestamp("2025-01-01")], "superbrand": ["x"]}))
        model_rep.trans_dict = trans_dict
        node = ModelNode(MockModel(), ds.copy(), model_rep=model_rep)
        scope = _basic_scope()

        ds_before = ds.copy()
        mask = pd.Series([True] * 10 + [False] * 20, index=ds.index)
        result = node._translate_budget(ds, node.model.conf, {}, mask, scope)
        assert result.equals(ds_before)

    def test_all_false_ds_mask_is_noop(self):
        ds = _basic_ds_with_ssn()
        trans_dict = {
            "function": {"bdg_tv": "basic"},
            "variable": {"bdg_tv": "brnd_tv_ots_clip"},
            "base_price": {"bdg_tv": 1.0},
        }
        model_rep = ModelRep(pd.DataFrame({"date": [pd.Timestamp("2025-01-01")], "superbrand": ["x"]}))
        model_rep.trans_dict = trans_dict
        node = ModelNode(MockModel(), ds.copy(), model_rep=model_rep)
        scope = _basic_scope()

        ds_before = ds.copy()
        mask = pd.Series([False] * 365, index=ds.index)
        result = node._translate_budget(ds, node.model.conf, {"bdg_tv": 50}, mask, scope)
        assert result["brnd_tv_ots_clip"].equals(ds_before["brnd_tv_ots_clip"])

    def test_zero_val_sets_media_to_zero(self):
        ds = _basic_ds_with_ssn()
        ds["brnd_tv_ots_clip"] = [1.0] * 365
        trans_dict = {
            "function": {"bdg_tv": "basic"},
            "variable": {"bdg_tv": "brnd_tv_ots_clip"},
            "base_price": {"bdg_tv": 1.0},
        }
        model_rep = ModelRep(pd.DataFrame({"date": [pd.Timestamp("2025-01-01")], "superbrand": ["x"]}))
        model_rep.trans_dict = trans_dict
        node = ModelNode(MockModel(), ds.copy(), model_rep=model_rep)
        scope = _basic_scope()

        mask = pd.Series([True] * 10 + [False] * 355, index=ds.index)
        result = node._translate_budget(ds, node.model.conf, {"bdg_tv": 0}, mask, scope)
        assert (result.loc[mask, "brnd_tv_ots_clip"] == 0.0).all()

    def test_nan_seasonality_triggers_warning(self, caplog):
        ds = _basic_ds_with_ssn()
        ds.loc[0:4, "opt_media_ssn_tv"] = np.nan
        trans_dict = {
            "function": {"bdg_tv": "basic"},
            "variable": {"bdg_tv": "brnd_tv_ots_clip"},
            "base_price": {"bdg_tv": 1.0},
        }
        model_rep = ModelRep(pd.DataFrame({"date": [pd.Timestamp("2025-01-01")], "superbrand": ["x"]}))
        model_rep.trans_dict = trans_dict
        node = ModelNode(MockModel(), ds.copy(), model_rep=model_rep)
        scope = _basic_scope()

        mask = pd.Series([True] * 10 + [False] * 355, index=ds.index)

        with caplog.at_level(logging.WARNING):
            node._translate_budget(ds, node.model.conf, {"bdg_tv": 50}, mask, scope)

        warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("NaN" in msg for msg in warning_messages)


class TestTranslateNewColumnWarning:
    """Translation must overwrite existing ds columns.

    A translator that writes to a missing column silently *appends* a new
    column (pandas .loc enlargement) whose values are never consumed by the
    model. Such a write must raise a one-time warning naming the budget item,
    the translator, and the new column.
    """

    def _make_node(self, var_name="brnd_tv_ots_clip"):
        ds = _basic_ds()
        trans_dict = {
            "function": {"bdg_tv": "basic_fix"},
            "variable": {"bdg_tv": var_name},
        }
        model_rep = ModelRep(
            pd.DataFrame({"date": [pd.Timestamp("2025-01-01")], "superbrand": ["x"]})
        )
        model_rep.trans_dict = trans_dict
        node = ModelNode(MockModel(), ds.copy(), model_rep=model_rep)
        return node, ds

    def test_existing_target_no_warning(self, caplog):
        node, ds = self._make_node(var_name="predicted")
        mask = pd.Series([True] * 10 + [False] * 20, index=ds.index)

        with caplog.at_level(logging.WARNING):
            node._translate_budget(ds, node.model.conf, {"bdg_tv": 50}, mask)

        messages = [r.message for r in caplog.records if "created new column" in r.message]
        assert messages == []

    def test_new_column_warns_once(self, caplog):
        import fermatrica_rep.meta_model.node as node_mod

        node_mod._warned_new_translation_cols.discard(("bdg_tv", "brand_new_col"))

        node, ds = self._make_node(var_name="brand_new_col")
        mask = pd.Series([True] * 10 + [False] * 20, index=ds.index)

        with caplog.at_level(logging.WARNING):
            node._translate_budget(ds, node.model.conf, {"bdg_tv": 50}, mask)
            node._translate_budget(ds, node.model.conf, {"bdg_tv": 30}, mask)

        messages = [r.message for r in caplog.records if "created new column" in r.message]
        assert len(messages) == 1, f"expected one warning, got {len(messages)}: {messages}"
        assert "bdg_tv" in messages[0]
        assert "basic_fix" in messages[0]
        assert "brand_new_col" in messages[0]

    def test_new_column_is_written(self, caplog):
        node, ds = self._make_node(var_name="brand_new_col")
        mask = pd.Series([True] * 10 + [False] * 20, index=ds.index)

        with caplog.at_level(logging.WARNING):
            result = node._translate_budget(ds, node.model.conf, {"bdg_tv": 5.0}, mask)

        assert "brand_new_col" in result.columns
        assert (result.loc[mask, "brand_new_col"] == 5.0).all()


class TestAdhocLoadAcrossPickle:
    """_load_adhoc_code idempotency must be keyed on process-local globals.

    Regression for the spawn-Pool bug: the old per-node ``_loaded_adhoc`` set
    was pickled into worker processes that start with a fresh
    ``fermatrica_rep.meta_model.node`` namespace. The memo said "already
    imported", so adhoc modules were never re-imported in the worker and
    ``_translate_budget`` silently skipped every dotted adhoc translator
    (budget grid came back flat). The gate must be ``k in globals()``.
    """

    ADHOC_SRC = {
        "code_py.adhoc.translators": (
            "def adhoc_basic(md, ds, model_rep, var_key, var_name, val, ds_mask, media_price=None):\n"
            "    ds.loc[ds_mask, var_name] = ds.loc[ds_mask, var_name] + val * 1000.0\n"
            "    return ds\n"
        ),
    }

    def _make_node(self):
        ds = _basic_ds_with_ssn()
        trans_dict = {
            "function": {"bdg_tv": "code_py.adhoc.translators.adhoc_basic"},
            "variable": {"bdg_tv": "brnd_tv_ots_clip"},
            "base_price": {"bdg_tv": 1.0},
        }
        model_rep = ModelRep(
            pd.DataFrame({"date": [pd.Timestamp("2025-01-01")], "superbrand": ["x"]})
        )
        model_rep.trans_dict = trans_dict
        model_rep.adhoc_code_src = dict(self.ADHOC_SRC)
        return ModelNode(PicklableMockModel(), ds.copy(), model_rep=model_rep)

    def test_pickled_node_reimports_adhoc_in_fresh_namespace(self):
        import pickle

        import fermatrica_rep.meta_model.node as node_mod

        node = self._make_node()

        # parent side: import once, then verify the module is registered
        node._load_adhoc_code()
        assert "code_py.adhoc.translators" in node_mod.__dict__

        # simulate a spawn worker: the pickle carries the node state, but the
        # worker's node.py namespace is fresh (no code_py* module globals)
        removed = [k for k in list(node_mod.__dict__) if k.startswith("code_py")]
        for k in removed:
            del node_mod.__dict__[k]
        try:
            node_w = pickle.loads(pickle.dumps(node))
            node_w._load_adhoc_code()

            assert "code_py.adhoc.translators" in node_mod.__dict__, (
                "adhoc module was not re-imported after crossing the pickle boundary"
            )

            ds = node_w.ds.copy()
            mask = pd.Series(True, index=ds.index)
            node_w._translate_budget(ds, node_w.model.conf, {"bdg_tv": 1.0}, mask)
            assert ds["brnd_tv_ots_clip"].sum() > 0, (
                "dotted adhoc translator silently skipped in the worker namespace"
            )
        finally:
            # restore globals so other tests in this process are unaffected
            for k in list(node_mod.__dict__):
                if k.startswith("code_py"):
                    del node_mod.__dict__[k]

    def test_missing_translator_warns_once(self, caplog):
        """An unresolvable translator is warned once per unique name, not per item."""
        import fermatrica_rep.meta_model.node as node_mod

        # module-level warn-once state is process-global; reset for this test
        node_mod._warned_missing_translators.discard(
            "code_py.adhoc.translators.does_not_exist"
        )

        ds = _basic_ds_with_ssn()
        trans_dict = {
            "function": {
                "bdg_tv": "code_py.adhoc.translators.does_not_exist",
                "bdg_radio": "code_py.adhoc.translators.does_not_exist",
            },
            "variable": {"bdg_tv": "brnd_tv_ots_clip", "bdg_radio": "brnd_tv_ots_clip"},
            "base_price": {"bdg_tv": 1.0, "bdg_radio": 1.0},
        }
        model_rep = ModelRep(
            pd.DataFrame({"date": [pd.Timestamp("2025-01-01")], "superbrand": ["x"]})
        )
        model_rep.trans_dict = trans_dict
        model_rep.adhoc_code_src = dict(self.ADHOC_SRC)
        node = ModelNode(PicklableMockModel(), ds.copy(), model_rep=model_rep)

        mask = pd.Series(True, index=ds.index)
        with caplog.at_level(logging.WARNING):
            # two different budget items referencing the same missing function
            node._translate_budget(ds, node.model.conf, {"bdg_tv": 1.0}, mask)
            node._translate_budget(ds, node.model.conf, {"bdg_radio": 1.0}, mask)

        missing = [
            r.message for r in caplog.records
            if "not found" in r.message and "does_not_exist" in r.message
        ]
        assert len(missing) == 1, f"expected one warning, got {len(missing)}: {missing}"


class TestTranslate:
    """Tests for translate() full flow."""

    def test_single_scope_exact(self):
        ds = _basic_ds_with_ssn()
        ds["period"] = "day"
        trans_dict = {
            "function": {"bdg_tv": "basic"},
            "variable": {"bdg_tv": "brnd_tv_ots_clip"},
            "base_price": {"bdg_tv": 1.0},
        }
        model_rep = ModelRep(pd.DataFrame({"date": [pd.Timestamp("2025-01-01")], "superbrand": ["x"]}))
        model_rep.trans_dict = trans_dict
        model_rep.ssn_dict = None
        node = ModelNode(MockModel(), ds.copy(), model_rep=model_rep)
        scope = _basic_scope(date_start="2025-01-01", date_end="2025-01-10")
        option = Option(name="test", budget_data={"test_scope": {"bdg_tv": 50}})

        node.translate(option, [scope], if_exact=True)

        mask_first_10 = node.ds["date"] <= pd.Timestamp("2025-01-10")
        assert (node.ds.loc[mask_first_10, "brnd_tv_ots_clip"] != 0.0).any()
        mask_last = node.ds["date"] > pd.Timestamp("2025-01-10")
        assert (node.ds.loc[mask_last, "brnd_tv_ots_clip"] == 0.0).all()

    def test_single_scope_with_long(self):
        ds = _basic_ds_with_ssn()
        ds["period"] = "day"
        trans_dict = {
            "function": {"bdg_tv": "basic"},
            "variable": {"bdg_tv": "brnd_tv_ots_clip"},
            "base_price": {"bdg_tv": 1.0},
        }
        model_rep = ModelRep(pd.DataFrame({"date": [pd.Timestamp("2025-01-01")], "superbrand": ["x"]}))
        model_rep.trans_dict = trans_dict
        model_rep.ssn_dict = None
        node = ModelNode(MockModel(), ds.copy(), model_rep=model_rep)
        scope = _basic_scope(date_start="2025-01-01", date_end="2025-01-10")
        option = Option(name="test", budget_data={"test_scope": {"bdg_tv": 50}})

        node.translate(option, [scope], if_exact=False)

        mask_first_10 = node.ds["date"] <= pd.Timestamp("2025-01-10")
        assert (node.ds.loc[mask_first_10, "brnd_tv_ots_clip"] != 0.0).any()
        mask_after = node.ds["date"] > pd.Timestamp("2025-01-10")
        assert (node.ds.loc[mask_after, "brnd_tv_ots_clip"] != 0.0).any()

    def test_cumulative_scopes(self):
        ds = _basic_ds_with_ssn()
        ds["period"] = "day"
        trans_dict = {
            "function": {"bdg_tv": "basic", "bdg_digital": "basic"},
            "variable": {"bdg_tv": "brnd_tv_ots_clip", "bdg_digital": "brnd_tv_ots_clip"},
            "base_price": {"bdg_tv": 1.0, "bdg_digital": 1.0},
        }
        model_rep = ModelRep(pd.DataFrame({"date": [pd.Timestamp("2025-01-01")], "superbrand": ["x"]}))
        model_rep.trans_dict = trans_dict
        model_rep.ssn_dict = None
        node = ModelNode(MockModel(), ds.copy(), model_rep=model_rep)
        scope = _basic_scope(date_start="2025-01-01", date_end="2025-01-10")
        option = Option(name="test", budget_data={
            "test_scope": {"bdg_tv": 50, "bdg_digital": 30},
        })

        node.translate(option, [scope], if_exact=True)

        mask = node.ds["date"] <= pd.Timestamp("2025-01-10")
        vals = node.ds.loc[mask, "brnd_tv_ots_clip"].unique()
        assert len(vals) > 0
        assert (vals > 0).all()

    def test_skips_missing_scope(self, caplog):
        ds = _basic_ds_with_ssn()
        ds["period"] = "day"
        trans_dict = {
            "function": {"bdg_tv": "basic"},
            "variable": {"bdg_tv": "brnd_tv_ots_clip"},
            "base_price": {"bdg_tv": 1.0},
        }
        model_rep = ModelRep(pd.DataFrame({"date": [pd.Timestamp("2025-01-01")], "superbrand": ["x"]}))
        model_rep.trans_dict = trans_dict
        node = ModelNode(MockModel(), ds.copy(), model_rep=model_rep)
        option = Option(name="test", budget_data={"missing_scope": {"bdg_tv": 50}})

        with caplog.at_level(logging.WARNING):
            node.translate(option, [], if_exact=True)

        warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("missing_scope" in msg and "skipping" in msg.lower() for msg in warning_messages)

    def test_scope_filters_column_not_in_ds(self):
        ds = _basic_ds_with_ssn()
        ds["period"] = "day"
        trans_dict = {
            "function": {"bdg_tv": "basic"},
            "variable": {"bdg_tv": "brnd_tv_ots_clip"},
            "base_price": {"bdg_tv": 1.0},
        }
        model_rep = ModelRep(pd.DataFrame({"date": [pd.Timestamp("2025-01-01")], "superbrand": ["x"]}))
        model_rep.trans_dict = trans_dict
        model_rep.ssn_dict = None
        node = ModelNode(MockModel(), ds.copy(), model_rep=model_rep)
        scope = _basic_scope(
            date_start="2025-01-01",
            date_end="2025-01-10",
            filters={"nonexistent": ["X"]},
        )
        option = Option(name="test", budget_data={"test_scope": {"bdg_tv": 50}})

        node.translate(option, [scope], if_exact=False)
        # Should not crash

        mask = node.ds["date"] <= pd.Timestamp("2025-01-10")
        assert (node.ds.loc[mask, "brnd_tv_ots_clip"] != 0.0).any()

    def test_raises_when_trans_dict_missing_and_budgets_present(self):
        ds = _basic_ds_with_ssn()
        ds["period"] = "day"
        model_rep = ModelRep(pd.DataFrame({"date": [pd.Timestamp("2025-01-01")], "superbrand": ["x"]}))
        model_rep.trans_dict = None
        node = ModelNode(MockModel(), ds.copy(), model_rep=model_rep)
        scope = _basic_scope(date_start="2025-01-01", date_end="2025-01-10")
        option = Option(name="test", budget_data={"test_scope": {"bdg_tv": 50}})

        with pytest.raises(FermatricaError, match="trans_dict"):
            node.translate(option, [scope], if_exact=True)

    def test_no_error_when_trans_dict_missing_and_budgets_empty(self):
        ds = _basic_ds_with_ssn()
        ds["period"] = "day"
        model_rep = ModelRep(pd.DataFrame({"date": [pd.Timestamp("2025-01-01")], "superbrand": ["x"]}))
        model_rep.trans_dict = None
        model_rep.ssn_dict = None
        node = ModelNode(MockModel(), ds.copy(), model_rep=model_rep)
        scope = _basic_scope(date_start="2025-01-01", date_end="2025-01-10")
        option = Option(name="test", budget_data={"test_scope": {}})

        node.translate(option, [scope], if_exact=True)
        assert (node.ds["brnd_tv_ots_clip"] == 0.0).all()

    def test_raises_when_trans_dict_missing_on_compute(self):
        ds = _basic_ds_with_ssn()
        ds["period"] = "day"
        model_rep = ModelRep(pd.DataFrame({"date": [pd.Timestamp("2025-01-01")], "superbrand": ["x"]}))
        model_rep.trans_dict = None
        node = ModelNode(MockModel(), ds.copy(), model_rep=model_rep)
        scope = _basic_scope(date_start="2025-01-01", date_end="2025-01-10")
        option = Option(name="test", budget_data={"test_scope": {"bdg_tv": 50}})

        with pytest.raises(FermatricaError, match="trans_dict"):
            node.compute(option, [scope], if_exact=True)


class TestComputeWithTranslation:
    """Tests for compute() with actual translation through to summary."""

    def test_compute_returns_summary(self, monkeypatch):
        ds = _basic_ds_with_ssn(365)
        ds["period"] = "day"
        ds["predicted"] = [100.0] * 365
        ds["observed"] = [90.0] * 365

        trans_dict = {
            "function": {"bdg_tv": "basic"},
            "variable": {"bdg_tv": "brnd_tv_ots_clip"},
            "base_price": {"bdg_tv": 1.0},
        }
        model_rep = ModelRep(ds)
        model_rep.trans_dict = trans_dict
        model_rep.ssn_dict = None
        node = ModelNode(MockModel(), ds.copy(), model_rep=model_rep)

        scope = _basic_scope(date_start="2025-01-01", date_end="2025-12-31")
        option = Option(name="test", budget_data={"test_scope": {"bdg_tv": 50}})

        def fake_transform(**kw):
            return (node.model, ds)

        monkeypatch.setattr("fermatrica_rep.meta_model.node.transform", fake_transform)

        dt_pred = pd.DataFrame({
            "date": ds["date"],
            "listed": [4] * 365,
            "predicted": [200.0] * 365,
            "observed": [90.0] * 365,
        })
        monkeypatch.setattr("fermatrica_rep.meta_model.node.predict_ext", lambda model, ds: dt_pred)

        result = node.compute(option, [scope])
        assert "test_scope" in result
        assert "pred_exact_vol" in result["test_scope"]
        assert result["test_scope"]["pred_exact_vol"] > 0

    def test_budget_increases_kpi(self, monkeypatch):
        ds = _basic_ds_with_ssn(365)
        ds["period"] = "day"
        ds["predicted"] = [100.0] * 365
        ds["observed"] = [90.0] * 365

        trans_dict = {
            "function": {"bdg_tv": "basic"},
            "variable": {"bdg_tv": "brnd_tv_ots_clip"},
            "base_price": {"bdg_tv": 1.0},
        }
        model_rep = ModelRep(ds)
        model_rep.trans_dict = trans_dict
        model_rep.ssn_dict = None

        scope = _basic_scope(date_start="2025-01-01", date_end="2025-12-31")

        def make_dt_pred(base_ds):
            return pd.DataFrame({
                "date": base_ds["date"],
                "listed": [4] * 365,
                "predicted": [200.0] * 365,
                "observed": [90.0] * 365,
            })

        def fake_transform(**kw):
            return (kw["model"], kw["ds"])

        monkeypatch.setattr("fermatrica_rep.meta_model.node.transform", fake_transform)

        # Zero budget
        node_zero = ModelNode(MockModel(), ds.copy(), model_rep=model_rep)
        option_zero = Option(name="test_zero", budget_data={"test_scope": {"bdg_tv": 0}})
        dt_pred_zero = make_dt_pred(ds)
        monkeypatch.setattr("fermatrica_rep.meta_model.node.predict_ext", lambda model, ds: dt_pred_zero)
        result_zero = node_zero.compute(option_zero, [scope])
        kpi_zero = result_zero["test_scope"]["pred_exact_vol"]

        # Positive budget
        dt_pred_pos = make_dt_pred(ds)
        dt_pred_pos["predicted"] = [300.0] * 365
        monkeypatch.setattr("fermatrica_rep.meta_model.node.predict_ext", lambda model, ds: dt_pred_pos)
        node_pos = ModelNode(MockModel(), ds.copy(), model_rep=model_rep)
        option_pos = Option(name="test_pos", budget_data={"test_scope": {"bdg_tv": 50}})
        result_pos = node_pos.compute(option_pos, [scope])
        kpi_pos = result_pos["test_scope"]["pred_exact_vol"]

        assert kpi_pos >= kpi_zero

    def test_return_full(self, monkeypatch):
        ds = _basic_ds_with_ssn(30)
        ds["period"] = "day"

        trans_dict = {
            "function": {"bdg_tv": "basic"},
            "variable": {"bdg_tv": "brnd_tv_ots_clip"},
            "base_price": {"bdg_tv": 1.0},
        }
        model_rep = ModelRep(ds)
        model_rep.trans_dict = trans_dict
        model_rep.ssn_dict = None
        node = ModelNode(MockModel(), ds.copy(), model_rep=model_rep)

        scope = _basic_scope(date_start="2025-01-01", date_end="2025-12-31")
        option = Option(name="test", budget_data={"test_scope": {"bdg_tv": 50}})

        def fake_transform(**kw):
            return (node.model, ds)

        monkeypatch.setattr("fermatrica_rep.meta_model.node.transform", fake_transform)

        dt_pred = pd.DataFrame({
            "date": ds["date"],
            "listed": [4] * 30,
            "predicted": [200.0] * 30,
            "observed": [90.0] * 30,
        })
        monkeypatch.setattr("fermatrica_rep.meta_model.node.predict_ext", lambda model, ds: dt_pred)

        result = node.compute(option, [scope], return_full=True)
        assert "test_scope" in result
        ds_out, dt_pred_out, summary = result["test_scope"]
        assert isinstance(ds_out, pd.DataFrame)
        assert isinstance(dt_pred_out, pd.DataFrame)
        assert isinstance(summary, dict)
        assert "pred_exact_vol" in summary

    def test_pred_long_when_scope_ends_before_horizon(self, monkeypatch):
        ds = _basic_ds_with_ssn(365)
        ds["period"] = "day"
        ds["predicted"] = [100.0] * 365
        ds["observed"] = [90.0] * 365

        trans_dict = {
            "function": {"bdg_tv": "basic"},
            "variable": {"bdg_tv": "brnd_tv_ots_clip"},
            "base_price": {"bdg_tv": 1.0},
        }
        model_rep = ModelRep(ds)
        model_rep.trans_dict = trans_dict
        model_rep.ssn_dict = None
        node = ModelNode(MockModel(), ds.copy(), model_rep=model_rep)

        scope = _basic_scope(date_start="2025-06-01", date_end="2025-06-30")
        option = Option(name="test", budget_data={"test_scope": {"bdg_tv": 50}})

        def fake_transform(**kw):
            return (node.model, ds)

        monkeypatch.setattr("fermatrica_rep.meta_model.node.transform", fake_transform)

        dt_pred = pd.DataFrame({
            "date": ds["date"],
            "listed": [4] * 365,
            "predicted": [200.0] * 365,
            "observed": [90.0] * 365,
        })
        monkeypatch.setattr("fermatrica_rep.meta_model.node.predict_ext", lambda model, ds: dt_pred)

        result = node.compute(option, [scope])
        assert "pred_long_vol" in result["test_scope"]
        assert result["test_scope"]["pred_long_vol"] >= result["test_scope"]["pred_exact_vol"]

    def test_mean_summary_type(self, monkeypatch):
        ds = _basic_ds_with_ssn(365)
        ds["period"] = "day"
        ds["predicted"] = list(range(365))
        ds["observed"] = list(range(365))

        trans_dict = {
            "function": {"bdg_tv": "basic"},
            "variable": {"bdg_tv": "brnd_tv_ots_clip"},
            "base_price": {"bdg_tv": 1.0},
        }
        model_rep = ModelRep(ds)
        model_rep.trans_dict = trans_dict
        model_rep.ssn_dict = None

        m = MockModel()
        m.conf = MockConf()
        m.conf.summary_type = "mean"

        node = ModelNode(m, ds.copy(), model_rep=model_rep)
        scope = _basic_scope(date_start="2025-01-01", date_end="2025-12-31")
        option = Option(name="test", budget_data={"test_scope": {}})

        def fake_transform(**kw):
            return (node.model, ds)

        monkeypatch.setattr("fermatrica_rep.meta_model.node.transform", fake_transform)

        dt_pred = pd.DataFrame({
            "date": ds["date"],
            "listed": [4] * 365,
            "predicted": list(range(365)),
            "observed": list(range(365)),
        })
        monkeypatch.setattr("fermatrica_rep.meta_model.node.predict_ext", lambda model, ds: dt_pred)

        result = node.compute(option, [scope])
        expected_mean = sum(range(365)) / 365
        assert result["test_scope"]["pred_exact_vol"] == pytest.approx(expected_mean)

    def test_mean_summary_type_reference_uses_mean(self, monkeypatch):
        dates = pd.date_range("2024-01-01", periods=731, freq="D")
        listed = [2] * 366 + [4] * 365
        ds = pd.DataFrame({
            "date": dates,
            "listed": listed,
            "predicted": [100.0] * 731,
            "observed": [90.0] * 731,
            "brnd_tv_ots_clip": [0.0] * 731,
            "opt_media_ssn_tv": [0.5] * 731,
        })
        ds["period"] = "day"

        trans_dict = {
            "function": {"bdg_tv": "basic"},
            "variable": {"bdg_tv": "brnd_tv_ots_clip"},
            "base_price": {"bdg_tv": 1.0},
        }
        model_rep = ModelRep(ds)
        model_rep.trans_dict = trans_dict
        model_rep.ssn_dict = None

        m = MockModel()
        m.conf = MockConf()
        m.conf.summary_type = "mean"

        node = ModelNode(m, ds.copy(), model_rep=model_rep)
        scope = _basic_scope(date_start="2025-01-01", date_end="2025-12-31")
        option = Option(name="test", budget_data={"test_scope": {}})

        def fake_transform(**kw):
            return (node.model, ds)

        monkeypatch.setattr("fermatrica_rep.meta_model.node.transform", fake_transform)

        dt_pred = ds[["date", "listed", "predicted", "observed"]].copy()
        monkeypatch.setattr("fermatrica_rep.meta_model.node.predict_ext", lambda model, ds: dt_pred)

        result = node.compute(option, [scope])
        assert result["test_scope"]["ref_vol"] == pytest.approx(90.0)
        assert result["test_scope"]["pred_exact_vol"] == pytest.approx(100.0)
        assert result["test_scope"]["growth_vol"] == pytest.approx(100.0 / 90.0)

    def test_mean_reference_split_uses_combined_mean(self, monkeypatch):
        # Reference period straddles the observed/future boundary: the observed
        # part (listed 2) and the predicted part (listed 4) must be aggregated
        # once as one series — not summed as two partial means (double counting).
        dates = pd.date_range("2025-01-01", "2026-12-31", freq="D")
        ds = pd.DataFrame({
            "date": dates,
            "listed": [2 if d <= pd.Timestamp("2025-06-30") else 4 for d in dates],
            "predicted": [200.0 if d.year == 2025 else 100.0 for d in dates],
            "observed": [90.0] * len(dates),
            "brnd_tv_ots_clip": [0.0] * len(dates),
            "opt_media_ssn_tv": [0.5] * len(dates),
        })
        ds["period"] = "day"

        trans_dict = {
            "function": {"bdg_tv": "basic"},
            "variable": {"bdg_tv": "brnd_tv_ots_clip"},
            "base_price": {"bdg_tv": 1.0},
        }
        model_rep = ModelRep(ds)
        model_rep.trans_dict = trans_dict
        model_rep.ssn_dict = None

        m = MockModel()
        m.conf = MockConf()
        m.conf.summary_type = "mean"

        node = ModelNode(m, ds.copy(), model_rep=model_rep)
        scope = _basic_scope(date_start="2026-01-01", date_end="2026-12-31",
                             ref_date_start="2025-01-01", ref_date_end="2025-12-31")
        option = Option(name="test", budget_data={"test_scope": {}})

        def fake_transform(**kw):
            return (node.model, ds)

        monkeypatch.setattr("fermatrica_rep.meta_model.node.transform", fake_transform)

        dt_pred = ds[["date", "listed", "predicted", "observed"]].copy()
        monkeypatch.setattr("fermatrica_rep.meta_model.node.predict_ext", lambda model, ds: dt_pred)

        result = node.compute(option, [scope])

        n_obs = int(((dt_pred["date"] <= pd.Timestamp("2025-06-30")) & (dt_pred["date"] <= pd.Timestamp("2025-12-31"))).sum())
        n_pred = int(((dt_pred["date"] > pd.Timestamp("2025-06-30")) & (dt_pred["date"] <= pd.Timestamp("2025-12-31"))).sum())
        expected_ref = (90.0 * n_obs + 200.0 * n_pred) / (n_obs + n_pred)

        assert result["test_scope"]["ref_vol"] == pytest.approx(expected_ref)
        assert result["test_scope"]["pred_exact_vol"] == pytest.approx(100.0)
        assert result["test_scope"]["growth_vol"] == pytest.approx(100.0 / expected_ref)

    def test_mean_reference_entirely_future_is_not_nan(self, monkeypatch):
        # A scope whose whole reference period lies beyond the observed horizon
        # must aggregate the predicted reference — the empty observed part must
        # not poison the mean into NaN (which surfaced as ``nan%`` growth).
        dates = pd.date_range("2025-01-01", "2028-12-31", freq="D")
        ds = pd.DataFrame({
            "date": dates,
            "listed": [2 if d.year == 2025 else 4 for d in dates],
            "predicted": [150.0 if d.year == 2026 else (120.0 if d.year == 2027 else 100.0) for d in dates],
            "observed": [90.0] * len(dates),
            "brnd_tv_ots_clip": [0.0] * len(dates),
            "opt_media_ssn_tv": [0.5] * len(dates),
        })
        ds["period"] = "day"

        trans_dict = {
            "function": {"bdg_tv": "basic"},
            "variable": {"bdg_tv": "brnd_tv_ots_clip"},
            "base_price": {"bdg_tv": 1.0},
        }
        model_rep = ModelRep(ds)
        model_rep.trans_dict = trans_dict
        model_rep.ssn_dict = None

        m = MockModel()
        m.conf = MockConf()
        m.conf.summary_type = "mean"

        node = ModelNode(m, ds.copy(), model_rep=model_rep)
        scope = _basic_scope(date_start="2027-01-01", date_end="2027-12-31",
                             ref_date_start="2026-01-01", ref_date_end="2026-12-31")
        option = Option(name="test", budget_data={"test_scope": {}})

        def fake_transform(**kw):
            return (node.model, ds)

        monkeypatch.setattr("fermatrica_rep.meta_model.node.transform", fake_transform)

        dt_pred = ds[["date", "listed", "predicted", "observed"]].copy()
        monkeypatch.setattr("fermatrica_rep.meta_model.node.predict_ext", lambda model, ds: dt_pred)

        result = node.compute(option, [scope])

        assert result["test_scope"]["ref_vol"] == pytest.approx(150.0)
        assert result["test_scope"]["pred_exact_vol"] == pytest.approx(120.0)
        assert result["test_scope"]["growth_vol"] == pytest.approx(120.0 / 150.0)

    def test_fin_summary_type(self, monkeypatch):
        ds = _basic_ds_with_ssn(365)
        ds["period"] = "day"

        trans_dict = {
            "function": {"bdg_tv": "basic"},
            "variable": {"bdg_tv": "brnd_tv_ots_clip"},
            "base_price": {"bdg_tv": 1.0},
        }
        model_rep = ModelRep(ds)
        model_rep.trans_dict = trans_dict
        model_rep.ssn_dict = None

        m = MockModel()
        m.conf = MockConf()
        m.conf.summary_type = "fin"

        node = ModelNode(m, ds.copy(), model_rep=model_rep)
        scope = _basic_scope(date_start="2025-01-01", date_end="2025-12-31")
        option = Option(name="test", budget_data={"test_scope": {}})

        def fake_transform(**kw):
            return (node.model, ds)

        monkeypatch.setattr("fermatrica_rep.meta_model.node.transform", fake_transform)

        dt_pred = pd.DataFrame({
            "date": ds["date"],
            "listed": [4] * 365,
            "predicted": list(range(365)),
            "observed": list(range(365)),
        })
        monkeypatch.setattr("fermatrica_rep.meta_model.node.predict_ext", lambda model, ds: dt_pred)

        result = node.compute(option, [scope])
        assert "pred_exact_vol" in result["test_scope"]


class TestScopeNames:
    """Tests for ModelNode.scope_names filtering."""

    def _make_node(self, scope_names=None, ds=None):
        if ds is None:
            ds = _basic_ds_with_ssn(365)
        ds["period"] = "day"
        trans_dict = {
            "function": {"bdg_tv": "basic"},
            "variable": {"bdg_tv": "brnd_tv_ots_clip"},
            "base_price": {"bdg_tv": 1.0},
        }
        model_rep = ModelRep(pd.DataFrame({"date": [pd.Timestamp("2025-01-01")], "superbrand": ["x"]}))
        model_rep.trans_dict = trans_dict
        model_rep.ssn_dict = None
        return ModelNode(MockModel(), ds.copy(), model_rep=model_rep, scope_names=scope_names)

    def _scopes(self):
        scope_a = _basic_scope(scope_name="scope_a", date_start="2025-01-01", date_end="2025-01-10")
        scope_b = _basic_scope(scope_name="scope_b", date_start="2025-02-01", date_end="2025-02-10")
        return scope_a, scope_b

    def test_translate_filters_scopes(self):
        node = self._make_node(scope_names={"scope_a"})
        option = Option(name="test", budget_data={
            "scope_a": {"bdg_tv": 50},
            "scope_b": {"bdg_tv": 30},
        })
        scope_a, scope_b = self._scopes()

        node.translate(option, [scope_a, scope_b], if_exact=True)

        mask_a = node.ds["date"] <= pd.Timestamp("2025-01-10")
        mask_b = (node.ds["date"] >= pd.Timestamp("2025-02-01")) & (node.ds["date"] <= pd.Timestamp("2025-02-10"))
        assert (node.ds.loc[mask_a, "brnd_tv_ots_clip"] != 0.0).any()
        assert (node.ds.loc[mask_b, "brnd_tv_ots_clip"] == 0.0).all()

    def test_translate_keeps_all_when_none(self):
        node = self._make_node(scope_names=None)
        option = Option(name="test", budget_data={
            "scope_a": {"bdg_tv": 50},
            "scope_b": {"bdg_tv": 30},
        })
        scope_a, scope_b = self._scopes()

        node.translate(option, [scope_a, scope_b], if_exact=True)

        mask_a = node.ds["date"] <= pd.Timestamp("2025-01-10")
        mask_b = (node.ds["date"] >= pd.Timestamp("2025-02-01")) & (node.ds["date"] <= pd.Timestamp("2025-02-10"))
        assert (node.ds.loc[mask_a, "brnd_tv_ots_clip"] != 0.0).any()
        assert (node.ds.loc[mask_b, "brnd_tv_ots_clip"] != 0.0).any()

    def test_compute_filters_summaries(self, monkeypatch):
        node = self._make_node(scope_names={"scope_a"})
        option = Option(name="test", budget_data={
            "scope_a": {"bdg_tv": 50},
            "scope_b": {"bdg_tv": 30},
        })
        scope_a, scope_b = self._scopes()

        def fake_transform(**kw):
            return (node.model, node.ds)

        monkeypatch.setattr("fermatrica_rep.meta_model.node.transform", fake_transform)

        dt_pred = pd.DataFrame({
            "date": node.ds["date"],
            "listed": [4] * 365,
            "predicted": [200.0] * 365,
            "observed": [90.0] * 365,
        })
        monkeypatch.setattr("fermatrica_rep.meta_model.node.predict_ext", lambda model, ds: dt_pred)

        result = node.compute(option, [scope_a, scope_b])
        assert "scope_a" in result
        assert "scope_b" not in result

    def test_compute_keeps_all_when_none(self, monkeypatch):
        node = self._make_node(scope_names=None)
        option = Option(name="test", budget_data={
            "scope_a": {"bdg_tv": 50},
            "scope_b": {"bdg_tv": 30},
        })
        scope_a, scope_b = self._scopes()

        def fake_transform(**kw):
            return (node.model, node.ds)

        monkeypatch.setattr("fermatrica_rep.meta_model.node.transform", fake_transform)

        dt_pred = pd.DataFrame({
            "date": node.ds["date"],
            "listed": [4] * 365,
            "predicted": [200.0] * 365,
            "observed": [90.0] * 365,
        })
        monkeypatch.setattr("fermatrica_rep.meta_model.node.predict_ext", lambda model, ds: dt_pred)

        result = node.compute(option, [scope_a, scope_b])
        assert "scope_a" in result
        assert "scope_b" in result


class TestStaticNode:
    """Tests for the static (no-translation) node flag."""

    def _make_static_node(self):
        ds = _basic_ds_with_ssn(365)
        ds["period"] = "day"
        model_rep = ModelRep(pd.DataFrame({"date": [pd.Timestamp("2025-01-01")], "superbrand": ["x"]}))
        return ModelNode(MockModel(), ds.copy(), model_rep=model_rep, static=True)

    def test_translate_is_noop_without_trans_dict(self):
        node = self._make_static_node()
        option = Option(name="test", budget_data={"test_scope": {"bdg_tv": 50}})

        node.translate(option, [_basic_scope()], if_exact=True)

        assert (node.ds["brnd_tv_ots_clip"] == 0.0).all()

    def test_static_node_still_computes(self, monkeypatch):
        node = self._make_static_node()
        option = Option(name="test", budget_data={"test_scope": {"bdg_tv": 50}})

        def fake_transform(**kw):
            return (node.model, node.ds)

        monkeypatch.setattr("fermatrica_rep.meta_model.node.transform", fake_transform)

        dt_pred = pd.DataFrame({
            "date": node.ds["date"],
            "listed": [4] * 365,
            "predicted": [200.0] * 365,
            "observed": [90.0] * 365,
        })
        monkeypatch.setattr("fermatrica_rep.meta_model.node.predict_ext", lambda model, ds: dt_pred)

        result = node.compute(option, [_basic_scope()])

        assert "pred_exact_vol" in result["test_scope"]

    def test_display_only_allows_model_none(self):
        node = ModelNode(
            model=None, ds=_basic_ds(),
            model_rep=ModelRep(pd.DataFrame({"date": [pd.Timestamp("2025-01-01")], "superbrand": ["x"]})),
            _display_only=True,
        )
        assert node.model is None


class TestFittedGuard:
    """Tests for the fitted-model guard in _refresh()."""

    def _node(self, model=None):
        ds = _basic_ds()
        model_rep = ModelRep(pd.DataFrame({"date": [pd.Timestamp("2025-01-01")], "superbrand": ["x"]}))
        return ModelNode(model or MockModel(), ds.copy(), model_rep=model_rep)

    def test_fitted_model_refresh_ok(self, monkeypatch):
        node = self._node()

        def fake_transform(ds, model, set_start=False, if_by_ref=False):
            return model, ds

        monkeypatch.setattr("fermatrica_rep.meta_model.node.transform", fake_transform)
        monkeypatch.setattr(
            "fermatrica_rep.meta_model.node.predict_ext",
            lambda model, ds: pd.DataFrame({
                "date": ds["date"], "listed": ds["listed"],
                "predicted": [100.0] * len(ds), "observed": [90.0] * len(ds),
            }),
        )

        result = node.compute()
        assert node.dt_pred is result

    def test_unfitted_model_raises(self):
        node = self._node(UnfittedMockModel())
        with pytest.raises(FermatricaError, match="fit"):
            node.compute()

    def test_unfitted_model_raises_on_option_path(self):
        node = self._node(UnfittedMockModel())
        node.model_rep.trans_dict = {
            "function": {"bdg_tv": "basic"},
            "variable": {"bdg_tv": "brnd_tv_ots_clip"},
            "base_price": {"bdg_tv": 1.0},
        }
        option = Option(name="test", budget_data={"test_scope": {"bdg_tv": 50}})
        with pytest.raises(FermatricaError, match="fit"):
            node.compute(option, [_basic_scope()])
