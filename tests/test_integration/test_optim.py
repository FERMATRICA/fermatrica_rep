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
        from fermatrica_rep.options.optim import optimize_budget_local_cobyla
        from fermatrica_rep.options.define import OptionSettings

        model = setup["model"]
        ds = setup["ds"]
        model_rep = setup["model_rep"]

        opt_set = OptionSettings(
            target=[[self.BRAND]],
            date_start=date(2018, 1, 1),
            date_end=date(2018, 12, 31),
            ref_date_start=date(2017, 1, 1),
            ref_date_end=date(2017, 12, 31),
            apply_vars=["superbrand"],
            plan_period="exact",
        )

        budget_items = [k for k in setup["trans_dict"]["function"] if k.startswith("bdg_")]
        if not budget_items:
            pytest.skip("No budget items in trans_dict")

        borders_dict = {k: (0.0, 20.0, 10.0) for k in budget_items}
        option = {k: 10.0 for k in budget_items}

        result = optimize_budget_local_cobyla(
            ds=ds, model=model, model_rep=model_rep,
            borders_dict=borders_dict, option=option, opt_set=opt_set,
            bdg_size=100.0, if_exact=True, epochs=1, iters_epoch=30,
            verbose=False,
        )

        for k in budget_items:
            assert result[k] >= 0.0, f"Budget {k} is negative ({result[k]})"
            assert result[k] <= borders_dict[k][1], (
                f"Budget {k} = {result[k]} exceeds upper bound {borders_dict[k][1]}"
            )

    def test_optimize_target_cobyla_converges(self, setup):
        """optimize_target_local_cobyla returns non-negative budgets."""
        from fermatrica_rep.options.optim import optimize_target_local_cobyla
        from fermatrica_rep.options.define import OptionSettings
        from fermatrica_rep.options.calc import option_report

        model = setup["model"]
        ds = setup["ds"]
        model_rep = setup["model_rep"]

        opt_set = OptionSettings(
            target=[[self.BRAND]],
            date_start=date(2018, 1, 1),
            date_end=date(2018, 12, 31),
            ref_date_start=date(2017, 1, 1),
            ref_date_end=date(2017, 12, 31),
            apply_vars=["superbrand"],
            plan_period="exact",
        )

        budget_items = [k for k in setup["trans_dict"]["function"] if k.startswith("bdg_")]
        if not budget_items:
            pytest.skip("No budget items in trans_dict")

        borders_dict = {k: (0.0, 20.0, 10.0) for k in budget_items}
        option_zero = {k: 0.0 for k in budget_items}

        _, _, opt_sum_zero = option_report(
            model, ds, model_rep, option_zero, opt_set, if_exact=True,
        )
        zero_kpi = opt_sum_zero["pred_exact_0"]["pred_exact_vol"]
        target_kpi = zero_kpi * 1.01

        result = optimize_target_local_cobyla(
            ds=ds, model=model, model_rep=model_rep,
            borders_dict=borders_dict, option=option_zero, opt_set=opt_set,
            trg=target_kpi, trg_tol_ratio=0.01,
            if_exact=True, epochs=1, iters_epoch=30,
            verbose=False,
        )

        for k in budget_items:
            assert result[k] >= 0.0, f"Budget {k} is negative ({result[k]})"
