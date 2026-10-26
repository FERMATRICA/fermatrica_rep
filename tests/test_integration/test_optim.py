"""Integration tests for budget optimization against real p00 model.

Uses p00_sample (OLS, 1 brand, no adhoc code) for fast iteration.
"""
import numpy as np
import pandas as pd
import pytest
from datetime import date


@pytest.mark.integration
class TestP00Optim:
    """Budget optimization tests against p00_sample (time series OLS)."""

    BRAND = "brand_x"

    @pytest.fixture
    def setup(self, p00_model_ds, samples_path):
        """Build reporting config from model and dataset (function-scoped)."""
        import fermatrica_rep as fmr

        model, ds = p00_model_ds
        ds = ds.copy()

        root = samples_path / "p00_sample"
        opt_path = str(root / "code_py" / "model_data" / "options" / "options.xlsx")
        ssn_path = str(root / "code_py" / "model_data" / "options" / "seasonality.xlsx")

        trans_dict = fmr.trans_dict_create(path=opt_path, sheet="translation")
        model_rep = fmr.ModelRep(ds, trans_dict=trans_dict, language="english")
        ds = fmr.media_ssn_apply(ds, path=ssn_path, sheet="data")

        return {
            "model": model, "ds": ds, "model_rep": model_rep,
            "trans_dict": trans_dict,
        }

    def test_optimize_budget_cobyla_converges(self, setup):
        """optimize_budget_local_cobyla returns positive budgets within borders."""
        import fermatrica_rep as fmr
        from fermatrica_rep.options.optim import optimize_budget_local_cobyla

        model = setup["model"]
        ds = setup["ds"]
        model_rep = setup["model_rep"]

        scope = fmr.OptionScope(
            scope_name="p00_2018",
            date_start=date(2018, 1, 1),
            date_end=date(2018, 12, 31),
            ref_date_start=date(2017, 1, 1),
            ref_date_end=date(2017, 12, 31),
            filters={"superbrand": [self.BRAND]},
            plan_period="exact",
        )

        budget_items = [k for k in setup["trans_dict"]["function"] if k.startswith("bdg_")]
        if not budget_items:
            pytest.skip("No budget items in trans_dict")

        borders_dict = {k: [0.0, 20.0, 10.0] for k in budget_items}
        option = fmr.Option(
            name="test_opt",
            budget_data={"p00_2018": {k: 10.0 for k in budget_items}},
        )

        node = fmr.ModelNode(model=model, ds=ds, model_rep=model_rep)
        meta_model = fmr.MetaModel(nodes=node)

        result = optimize_budget_local_cobyla(
            meta_model=meta_model, scopes=[scope], budget_scope="p00_2018",
            borders_dict=borders_dict, option=option,
            bdg_size=100.0, if_exact=True, epochs=1, iters_epoch=30,
            verbose=False,
        )

        assert isinstance(result, fmr.Option)
        bdg = result.budget_data["p00_2018"]
        for k in budget_items:
            assert bdg[k] >= 0.0, f"Budget {k} is negative ({bdg[k]})"
            assert bdg[k] <= borders_dict[k][1], (
                f"Budget {k} = {bdg[k]} exceeds upper bound {borders_dict[k][1]}"
            )

    def test_optimize_target_cobyla_converges(self, setup):
        """optimize_target_local_cobyla returns non-negative budgets."""
        import fermatrica_rep as fmr
        from fermatrica_rep.options.optim import optimize_target_local_cobyla

        model = setup["model"]
        ds = setup["ds"]
        model_rep = setup["model_rep"]

        scope = fmr.OptionScope(
            scope_name="p00_2018",
            date_start=date(2018, 1, 1),
            date_end=date(2018, 12, 31),
            ref_date_start=date(2017, 1, 1),
            ref_date_end=date(2017, 12, 31),
            filters={"superbrand": [self.BRAND]},
            plan_period="exact",
        )

        budget_items = [k for k in setup["trans_dict"]["function"] if k.startswith("bdg_")]
        if not budget_items:
            pytest.skip("No budget items in trans_dict")

        borders_dict = {k: [0.0, 20.0, 10.0] for k in budget_items}

        node = fmr.ModelNode(model=model, ds=ds, model_rep=model_rep)
        meta_model = fmr.MetaModel(nodes=node)

        zero_option = fmr.Option(
            name="zero",
            budget_data={"p00_2018": {k: 0.0 for k in budget_items}},
        )
        results_zero = meta_model.compute(zero_option, scopes=[scope], if_exact=True)
        zero_kpi = results_zero["model"]["p00_2018"]["pred_exact_vol"]
        target_kpi = zero_kpi * 1.01

        option = fmr.Option(
            name="test_opt",
            budget_data={"p00_2018": {k: 0.0 for k in budget_items}},
        )

        result = optimize_target_local_cobyla(
            meta_model=meta_model, scopes=[scope], budget_scope="p00_2018",
            borders_dict=borders_dict, option=option,
            trg=target_kpi, trg_tol_ratio=0.01,
            if_exact=True, epochs=1, iters_epoch=30,
            verbose=False,
        )

        assert isinstance(result, fmr.Option)
        bdg = result.budget_data["p00_2018"]
        for k in budget_items:
            assert bdg[k] >= 0.0, f"Budget {k} is negative ({bdg[k]})"

    def test_optimize_budget_cobyla_multi_returns_dict(self, setup):
        """optimize_budget_local_cobyla_multi returns dict of Option keyed by budget size."""
        import fermatrica_rep as fmr
        from fermatrica_rep.options.optim import optimize_budget_local_cobyla_multi

        model = setup["model"]
        ds = setup["ds"]
        model_rep = setup["model_rep"]

        scope = fmr.OptionScope(
            scope_name="p00_2018",
            date_start=date(2018, 1, 1),
            date_end=date(2018, 12, 31),
            ref_date_start=date(2017, 1, 1),
            ref_date_end=date(2017, 12, 31),
            filters={"superbrand": [self.BRAND]},
            plan_period="exact",
        )

        budget_items = [k for k in setup["trans_dict"]["function"] if k.startswith("bdg_")]
        if not budget_items:
            pytest.skip("No budget items in trans_dict")

        borders_dict = {k: [0.0, 20.0, 10.0] for k in budget_items}
        option = fmr.Option(
            name="test_opt",
            budget_data={"p00_2018": {k: 10.0 for k in budget_items}},
        )

        node = fmr.ModelNode(model=model, ds=ds, model_rep=model_rep)
        meta_model = fmr.MetaModel(nodes=node)

        result = optimize_budget_local_cobyla_multi(
            meta_model=meta_model, scopes=[scope], budget_scope="p00_2018",
            borders_dict=borders_dict, option=option, bdg_sizes=[50.0, 100.0, 150.0],
            if_exact=True, epochs=1, iters_epoch=30, verbose=False, cores=1,
        )

        assert isinstance(result, dict)
        assert len(result) == 3
        for bdg_size, opt_result in result.items():
            assert isinstance(opt_result, fmr.Option)
            bdg = opt_result.budget_data["p00_2018"]
            for k in budget_items:
                assert bdg[k] >= 0.0, f"Budget {k} is negative ({bdg[k]})"
                assert bdg[k] <= borders_dict[k][1], \
                    f"Budget {k} = {bdg[k]} exceeds upper bound {borders_dict[k][1]}"

    def test_optimize_budget_cobyla_multi_sequential(self, setup):
        """optimize_budget_local_cobyla_multi with cores=1 is sequential and returns dict."""
        import fermatrica_rep as fmr
        from fermatrica_rep.options.optim import optimize_budget_local_cobyla_multi

        model = setup["model"]
        ds = setup["ds"]
        model_rep = setup["model_rep"]

        scope = fmr.OptionScope(
            scope_name="p00_2018",
            date_start=date(2018, 1, 1),
            date_end=date(2018, 12, 31),
            ref_date_start=date(2017, 1, 1),
            ref_date_end=date(2017, 12, 31),
            filters={"superbrand": [self.BRAND]},
            plan_period="exact",
        )

        budget_items = [k for k in setup["trans_dict"]["function"] if k.startswith("bdg_")]
        if not budget_items:
            pytest.skip("No budget items in trans_dict")

        borders_dict = {k: [0.0, 20.0, 10.0] for k in budget_items}
        option = fmr.Option(
            name="test_opt",
            budget_data={"p00_2018": {k: 10.0 for k in budget_items}},
        )

        node = fmr.ModelNode(model=model, ds=ds, model_rep=model_rep)
        meta_model = fmr.MetaModel(nodes=node)

        result = optimize_budget_local_cobyla_multi(
            meta_model=meta_model, scopes=[scope], budget_scope="p00_2018",
            borders_dict=borders_dict, option=option, bdg_sizes=[50.0, 100.0],
            if_exact=True, epochs=1, iters_epoch=30, verbose=False, cores=1,
        )

        assert isinstance(result, dict)
        assert len(result) == 2
        for bdg_size, opt_result in result.items():
            bdg = opt_result.budget_data["p00_2018"]
            for k in budget_items:
                assert bdg[k] >= 0.0

    def test_optimize_budget_cobyla_multi_different_sizes(self, setup):
        """Different bdg_sizes produce different optimized splits."""
        import fermatrica_rep as fmr
        from fermatrica_rep.options.optim import optimize_budget_local_cobyla_multi

        model = setup["model"]
        ds = setup["ds"]
        model_rep = setup["model_rep"]

        scope = fmr.OptionScope(
            scope_name="p00_2018",
            date_start=date(2018, 1, 1),
            date_end=date(2018, 12, 31),
            ref_date_start=date(2017, 1, 1),
            ref_date_end=date(2017, 12, 31),
            filters={"superbrand": [self.BRAND]},
            plan_period="exact",
        )

        budget_items = [k for k in setup["trans_dict"]["function"] if k.startswith("bdg_")]
        if not budget_items:
            pytest.skip("No budget items in trans_dict")

        borders_dict = {k: [0.0, 20.0, 10.0] for k in budget_items}
        option = fmr.Option(
            name="test_opt",
            budget_data={"p00_2018": {k: 10.0 for k in budget_items}},
        )

        node = fmr.ModelNode(model=model, ds=ds, model_rep=model_rep)
        meta_model = fmr.MetaModel(nodes=node)

        result = optimize_budget_local_cobyla_multi(
            meta_model=meta_model, scopes=[scope], budget_scope="p00_2018",
            borders_dict=borders_dict, option=option, bdg_sizes=[50.0, 200.0],
            if_exact=True, epochs=1, iters_epoch=30, verbose=False, cores=1,
        )

        bdg_50 = result[50.0].budget_data["p00_2018"]
        bdg_200 = result[200.0].budget_data["p00_2018"]
        sum_50 = sum(bdg_50.values())
        sum_200 = sum(bdg_200.values())
        assert sum_200 > sum_50, "Larger budget should have larger total spend"

    def test_optimize_target_cobyla_multi_returns_dict(self, setup):
        """optimize_target_local_cobyla_multi returns dict of Option keyed by target."""
        import fermatrica_rep as fmr
        from fermatrica_rep.options.optim import optimize_target_local_cobyla_multi

        model = setup["model"]
        ds = setup["ds"]
        model_rep = setup["model_rep"]

        scope = fmr.OptionScope(
            scope_name="p00_2018",
            date_start=date(2018, 1, 1),
            date_end=date(2018, 12, 31),
            ref_date_start=date(2017, 1, 1),
            ref_date_end=date(2017, 12, 31),
            filters={"superbrand": [self.BRAND]},
            plan_period="exact",
        )

        budget_items = [k for k in setup["trans_dict"]["function"] if k.startswith("bdg_")]
        if not budget_items:
            pytest.skip("No budget items in trans_dict")

        borders_dict = {k: [0.0, 20.0, 10.0] for k in budget_items}

        node = fmr.ModelNode(model=model, ds=ds, model_rep=model_rep)
        meta_model = fmr.MetaModel(nodes=node)

        zero_option = fmr.Option(
            name="zero",
            budget_data={"p00_2018": {k: 0.0 for k in budget_items}},
        )
        results_zero = meta_model.compute(zero_option, scopes=[scope], if_exact=True)
        zero_kpi = results_zero["model"]["p00_2018"]["pred_exact_vol"]
        target_1 = zero_kpi * 1.01
        target_2 = zero_kpi * 1.02

        option = fmr.Option(
            name="test_opt",
            budget_data={"p00_2018": {k: 0.0 for k in budget_items}},
        )

        result = optimize_target_local_cobyla_multi(
            meta_model=meta_model, scopes=[scope], budget_scope="p00_2018",
            borders_dict=borders_dict, option=option,
            trg_list=[target_1, target_2], trg_tol_ratio=0.01,
            if_exact=True, epochs=1, iters_epoch=30, verbose=False, cores=1,
        )

        assert isinstance(result, dict)
        assert len(result) == 2
        for trg, opt_result in result.items():
            assert isinstance(opt_result, fmr.Option)
            bdg = opt_result.budget_data["p00_2018"]
            for k in budget_items:
                assert bdg[k] >= 0.0, f"Budget {k} is negative ({bdg[k]})"

    def test_optimize_budget_cobyla_if_volume_false(self, setup):
        """optimize_budget_local_cobyla with if_volume=False returns Option."""
        import fermatrica_rep as fmr
        from fermatrica_rep.options.optim import optimize_budget_local_cobyla

        model = setup["model"]
        ds = setup["ds"]
        model_rep = setup["model_rep"]

        scope = fmr.OptionScope(
            scope_name="p00_2018",
            date_start=date(2018, 1, 1),
            date_end=date(2018, 12, 31),
            ref_date_start=date(2017, 1, 1),
            ref_date_end=date(2017, 12, 31),
            filters={"superbrand": [self.BRAND]},
            plan_period="exact",
        )

        budget_items = [k for k in setup["trans_dict"]["function"] if k.startswith("bdg_")]
        if not budget_items:
            pytest.skip("No budget items in trans_dict")

        borders_dict = {k: [0.0, 20.0, 10.0] for k in budget_items}
        option = fmr.Option(
            name="test_opt",
            budget_data={"p00_2018": {k: 10.0 for k in budget_items}},
        )

        node = fmr.ModelNode(model=model, ds=ds, model_rep=model_rep)
        meta_model = fmr.MetaModel(nodes=node)

        result = optimize_budget_local_cobyla(
            meta_model=meta_model, scopes=[scope], budget_scope="p00_2018",
            borders_dict=borders_dict, option=option,
            bdg_size=100.0, if_exact=True, if_volume=False,
            epochs=1, iters_epoch=30, verbose=False,
        )

        assert isinstance(result, fmr.Option)
        bdg = result.budget_data["p00_2018"]
        for k in budget_items:
            assert bdg[k] >= 0.0, f"Budget {k} is negative ({bdg[k]})"
            assert bdg[k] <= borders_dict[k][1], \
                f"Budget {k} = {bdg[k]} exceeds upper bound {borders_dict[k][1]}"

    def test_optimize_budget_cobyla_if_long_true(self, setup):
        """optimize_budget_local_cobyla with if_long=True, if_exact=False returns Option."""
        import fermatrica_rep as fmr
        from fermatrica_rep.options.optim import optimize_budget_local_cobyla

        model = setup["model"]
        ds = setup["ds"]
        model_rep = setup["model_rep"]

        scope = fmr.OptionScope(
            scope_name="p00_2018",
            date_start=date(2018, 1, 1),
            date_end=date(2018, 12, 31),
            ref_date_start=date(2017, 1, 1),
            ref_date_end=date(2017, 12, 31),
            filters={"superbrand": [self.BRAND]},
            plan_period="exact",
        )

        budget_items = [k for k in setup["trans_dict"]["function"] if k.startswith("bdg_")]
        if not budget_items:
            pytest.skip("No budget items in trans_dict")

        borders_dict = {k: [0.0, 20.0, 10.0] for k in budget_items}
        option = fmr.Option(
            name="test_opt",
            budget_data={"p00_2018": {k: 10.0 for k in budget_items}},
        )

        node = fmr.ModelNode(model=model, ds=ds, model_rep=model_rep)
        meta_model = fmr.MetaModel(nodes=node)

        result = optimize_budget_local_cobyla(
            meta_model=meta_model, scopes=[scope], budget_scope="p00_2018",
            borders_dict=borders_dict, option=option,
            bdg_size=100.0, if_exact=False, if_long=True,
            epochs=1, iters_epoch=30, verbose=False,
        )

        assert isinstance(result, fmr.Option)
        bdg = result.budget_data["p00_2018"]
        for k in budget_items:
            assert bdg[k] >= 0.0

    def test_optimize_budget_ncma_returns_option(self, setup):
        """optimize_budget_local_ncma returns Option with single-node model."""
        import fermatrica_rep as fmr
        from fermatrica_rep.options.optim import optimize_budget_local_ncma

        model = setup["model"]
        ds = setup["ds"]
        model_rep = setup["model_rep"]

        scope = fmr.OptionScope(
            scope_name="p00_2018",
            date_start=date(2018, 1, 1),
            date_end=date(2018, 12, 31),
            ref_date_start=date(2017, 1, 1),
            ref_date_end=date(2017, 12, 31),
            filters={"superbrand": [self.BRAND]},
            plan_period="exact",
        )

        budget_items = [k for k in setup["trans_dict"]["function"] if k.startswith("bdg_")]
        if not budget_items:
            pytest.skip("No budget items in trans_dict")

        borders_dict = {k: [0.0, 20.0, 10.0] for k in budget_items}
        option = fmr.Option(
            name="test_opt",
            budget_data={"p00_2018": {k: 10.0 for k in budget_items}},
        )

        node = fmr.ModelNode(model=model, ds=ds, model_rep=model_rep)
        meta_model = fmr.MetaModel(nodes=node)

        result = optimize_budget_local_ncma(
            meta_model=meta_model, scopes=[scope], budget_scope="p00_2018",
            borders_dict=borders_dict, option=option,
            bdg_size=100.0, if_exact=True,
            epochs=1, iters_epoch=20, cores=1, verbose=False,
        )

        assert isinstance(result, fmr.Option)
        assert "p00_2018" in result.budget_data


# ---------------------------------------------------------------------------
# TestP07Optim — Multi-node budget optimization (Category -> Sales DAG)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestP07Optim:
    """Budget optimization tests against p07_sample (multi-node Category -> Sales DAG)."""

    BRAND = "brand_1"
    CAT_BRAND = "mock_brand"
    SCOPE_NAME = "brand_1_2026"
    CAT_SCOPE_NAME = "category_2026"

    BUDGET_ITEMS = ["bdg_tv", "bdg_ooh", "bdg_dgt_display", "bdg_dgt_video", "bdg_pos"]
    # 5 items for speed; real usage has ~10

    @pytest.fixture
    def setup(self, p07_models_ds, samples_path):
        """Build MetaModel with Category -> Sales DAG (function-scoped)."""
        import fermatrica_rep as fmr
        from code_py.adhoc.reporting import pred_merge_1

        cat_model, cat_ds, sales_model, sales_ds = p07_models_ds
        cat_ds = cat_ds.copy()
        sales_ds = sales_ds.copy()

        root = samples_path / "p07_sample"
        opt_path = str(root / "code_py" / "model_data" / "options" / "context.xlsx")

        trans_dict_cat = fmr.trans_dict_create(path=opt_path, sheet="translation_category")
        trans_dict_sales = fmr.trans_dict_create(path=opt_path, sheet="translation_sales")
        ssn_dict = fmr.seasonality_read(opt_path)

        node_cat = fmr.ModelNode(
            model=cat_model, ds=cat_ds,
            language="english", trans_dict=trans_dict_cat, ssn_dict=ssn_dict,
        )
        node_sales = fmr.ModelNode(
            model=sales_model, ds=sales_ds,
            language="english", trans_dict=trans_dict_sales, ssn_dict=ssn_dict,
        )
        meta_model = fmr.MetaModel(
            nodes={"category": node_cat, "sales": node_sales},
            edges=[("category", "sales", pred_merge_1)],
        )

        scopes = fmr.scopes_read(opt_path)

        borders_dict = {k: [0.0, 100.0, 30.0] for k in self.BUDGET_ITEMS}

        return {
            "meta_model": meta_model, "scopes": scopes,
            "borders_dict": borders_dict,
        }

    def _make_option(self, sales_budget=None):
        """Create option with budget data for both category and sales scopes."""
        import fermatrica_rep as fmr
        if sales_budget is None:
            sales_budget = {k: 30.0 for k in self.BUDGET_ITEMS}
        return fmr.Option(
            name="test_opt",
            budget_data={
                self.SCOPE_NAME: sales_budget,
                self.CAT_SCOPE_NAME: {},
            },
        )

    def test_dag_computes(self, setup):
        """MetaModel.compute() with a predefined option returns results for both nodes."""
        import fermatrica_rep as fmr

        meta_model = setup["meta_model"]
        scopes = setup["scopes"]

        option = self._make_option()

        results = meta_model.compute(option, scopes=scopes, allow_past=True)

        assert "category" in results
        assert "sales" in results
        assert self.CAT_SCOPE_NAME in results["category"]
        assert self.SCOPE_NAME in results["sales"]

        cat_summary = results["category"][self.CAT_SCOPE_NAME]
        sales_summary = results["sales"][self.SCOPE_NAME]
        assert "pred_exact_vol" in cat_summary
        assert "pred_exact_vol" in sales_summary

    def test_optimize_budget_cobyla_multi_node(self, setup):
        """optimize_budget_local_cobyla with summary_nodes returns Option within borders."""
        import fermatrica_rep as fmr
        from fermatrica_rep.options.optim import optimize_budget_local_cobyla

        meta_model = setup["meta_model"]
        scopes = setup["scopes"]
        borders_dict = setup["borders_dict"]

        summary_nodes = {
            "category": (0.3, self.CAT_SCOPE_NAME),
            "sales": (0.7, self.SCOPE_NAME),
        }

        option = self._make_option()

        result = optimize_budget_local_cobyla(
            meta_model=meta_model, scopes=scopes, budget_scope=self.SCOPE_NAME,
            borders_dict=borders_dict, option=option,
            bdg_size=150.0, if_exact=True,
            epochs=1, iters_epoch=30, verbose=False,
            summary_nodes=summary_nodes,
            warn_nan_seasonality=False,
        )

        assert isinstance(result, fmr.Option)
        bdg = result.budget_data[self.SCOPE_NAME]
        for k in self.BUDGET_ITEMS:
            assert bdg[k] >= 0.0, f"Budget {k} is negative ({bdg[k]})"
            assert bdg[k] <= borders_dict[k][1], \
                f"Budget {k} = {bdg[k]} exceeds upper bound {borders_dict[k][1]}"

    def test_optimize_target_cobyla_multi_node(self, setup):
        """optimize_target_local_cobyla with summary_nodes returns non-negative budgets."""
        import fermatrica_rep as fmr
        from fermatrica_rep.options.optim import optimize_target_local_cobyla

        meta_model = setup["meta_model"]
        scopes = setup["scopes"]
        borders_dict = setup["borders_dict"]

        summary_nodes = {
            "category": (0.3, self.CAT_SCOPE_NAME),
            "sales": (0.7, self.SCOPE_NAME),
        }

        zero_option = self._make_option({k: 0.0 for k in self.BUDGET_ITEMS})
        results_zero = meta_model.compute(zero_option, scopes=scopes,
                                          allow_past=True, if_exact=True)
        zero_kpi = 0.0
        for node_name, (weight, node_scope) in summary_nodes.items():
            zero_kpi += weight * results_zero[node_name][node_scope].get("pred_exact_vol", 0)
        target_kpi = zero_kpi * 1.01 if zero_kpi > 0 else 1.0

        option = self._make_option({k: 0.0 for k in self.BUDGET_ITEMS})

        trg_dict = {
            node_name: target_val
            for node_name, target_val in zip(summary_nodes.keys(), [target_kpi * 0.5, target_kpi * 0.5])
        }

        result = optimize_target_local_cobyla(
            meta_model=meta_model, scopes=scopes, budget_scope=self.SCOPE_NAME,
            borders_dict=borders_dict, option=option,
            trg=trg_dict, trg_tol_ratio=0.01,
            if_exact=True, epochs=1, iters_epoch=30, verbose=False,
            summary_nodes=summary_nodes,
            warn_nan_seasonality=False,
        )

        assert isinstance(result, fmr.Option)
        bdg = result.budget_data[self.SCOPE_NAME]
        for k in self.BUDGET_ITEMS:
            assert bdg[k] >= 0.0, f"Budget {k} is negative ({bdg[k]})"

    def test_optimize_budget_cobyla_multi_node_multi(self, setup):
        """optimize_budget_local_cobyla_multi with summary_nodes returns dict of Options."""
        import fermatrica_rep as fmr
        from fermatrica_rep.options.optim import optimize_budget_local_cobyla_multi

        meta_model = setup["meta_model"]
        scopes = setup["scopes"]
        borders_dict = setup["borders_dict"]

        summary_nodes = {
            "category": (0.3, self.CAT_SCOPE_NAME),
            "sales": (0.7, self.SCOPE_NAME),
        }

        option = self._make_option()

        result = optimize_budget_local_cobyla_multi(
            meta_model=meta_model, scopes=scopes, budget_scope=self.SCOPE_NAME,
            borders_dict=borders_dict, option=option,
            bdg_sizes=[100.0, 150.0],
            if_exact=True, epochs=1, iters_epoch=30, verbose=False, cores=1,
            summary_nodes=summary_nodes,
            warn_nan_seasonality=False,
        )

        assert isinstance(result, dict)
        assert len(result) == 2
        for bdg_size, opt_result in result.items():
            assert isinstance(opt_result, fmr.Option)
            bdg = opt_result.budget_data[self.SCOPE_NAME]
            for k in self.BUDGET_ITEMS:
                assert bdg[k] >= 0.0

    def test_optimize_target_cobyla_multi_node_multi(self, setup):
        """optimize_target_local_cobyla_multi with summary_nodes returns dict of Options."""
        import fermatrica_rep as fmr
        from fermatrica_rep.options.optim import optimize_target_local_cobyla_multi

        meta_model = setup["meta_model"]
        scopes = setup["scopes"]
        borders_dict = setup["borders_dict"]

        summary_nodes = {
            "category": (0.3, self.CAT_SCOPE_NAME),
            "sales": (0.7, self.SCOPE_NAME),
        }

        zero_option = self._make_option({k: 0.0 for k in self.BUDGET_ITEMS})
        results_zero = meta_model.compute(zero_option, scopes=scopes,
                                          allow_past=True, if_exact=True)
        zero_kpi = 0.0
        for node_name, (weight, node_scope) in summary_nodes.items():
            zero_kpi += weight * results_zero[node_name][node_scope].get("pred_exact_vol", 0)
        target_1 = zero_kpi * 1.01 if zero_kpi > 0 else 1.0
        target_2 = zero_kpi * 1.02 if zero_kpi > 0 else 2.0

        option = self._make_option({k: 0.0 for k in self.BUDGET_ITEMS})

        trg_dict_1 = {node_name: target_1 * 0.5 for node_name in summary_nodes}
        trg_dict_2 = {node_name: target_2 * 0.5 for node_name in summary_nodes}

        result = optimize_target_local_cobyla_multi(
            meta_model=meta_model, scopes=scopes, budget_scope=self.SCOPE_NAME,
            borders_dict=borders_dict, option=option,
            trg_list=[trg_dict_1, trg_dict_2], trg_tol_ratio=0.01,
            if_exact=True, epochs=1, iters_epoch=30, verbose=False, cores=1,
            summary_nodes=summary_nodes,
            warn_nan_seasonality=False,
        )

        assert isinstance(result, dict)
        assert len(result) == 2
        for trg, opt_result in result.items():
            assert isinstance(opt_result, fmr.Option)
            bdg = opt_result.budget_data[self.SCOPE_NAME]
            for k in self.BUDGET_ITEMS:
                assert bdg[k] >= 0.0

