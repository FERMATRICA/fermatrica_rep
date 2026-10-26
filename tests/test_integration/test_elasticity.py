"""Integration tests for elasticity_price.py: elasticity_data.

Tests that price elasticity data generation works against real model data.
p00 is skipped — it has no price variables.
"""
import numpy as np
import pytest
from datetime import date

import fermatrica_rep as fmr


@pytest.mark.integration
class TestP01Elasticity:
    """Price elasticity tests against p01_sample (panel LME, 3 brands)."""

    BRAND = "brand_1"

    @pytest.fixture
    def setup(self, p01_model_ds, samples_path):
        """Build ModelNode, scope, and option for elasticity calculation."""
        model, ds = p01_model_ds
        ds = ds.copy()

        root = samples_path / "p01_sample"
        opt_path = str(root / "code_py" / "model_data" / "options" / "options.xlsx")
        ssn_path = str(root / "code_py" / "model_data" / "options" / "seasonality.xlsx")

        trans_dict = fmr.trans_dict_create(path=opt_path, sheet="translation")
        model_rep = fmr.ModelRep(ds, trans_dict=trans_dict, language="english")
        ds = fmr.media_ssn_apply(ds, path=ssn_path, sheet="data")

        node = fmr.ModelNode(model=model, ds=ds, model_rep=model_rep)

        scope = fmr.OptionScope(
            scope_name=self.BRAND,
            date_start=date(2025, 6, 30),
            date_end=date(2025, 12, 31),
            ref_date_start=date(2024, 1, 1),
            ref_date_end=date(2024, 12, 31),
            filters={"superbrand": [self.BRAND]},
            plan_period="exact",
        )

        budget_data = {
            scope.scope_name: {"bdg_tv": 50, "bdg_digital": 30, "bdg_pr": 20, "nobdg_price": 1.0}
        }
        option = fmr.Option(name="base", budget_data=budget_data)

        return {
            "node": node,
            "scope": scope,
            "option": option,
        }

    def test_elasticity_data(self, setup):
        """elasticity_data returns a non-empty DataFrame."""
        result = fmr.elasticity_data(
            model=setup["node"],
            scope=setup["scope"],
            option=setup["option"],
            step=0.1,
            price_range=(-0.5, 0.5),
            if_exact=True,
        )

        assert result is not None
        assert len(result) > 0

        num_cols = result.select_dtypes(include=[np.number]).columns
        assert np.all(np.isfinite(result[num_cols].values)), (
            "elasticity_data produced NaN or Inf values"
        )

    def test_elasticity_has_expected_columns(self, setup):
        """Result has expected column names."""
        result = fmr.elasticity_data(
            model=setup["node"],
            scope=setup["scope"],
            option=setup["option"],
            step=0.1,
            price_range=(-0.5, 0.5),
            if_exact=True,
        )

        expected = {
            "nobdg_price", "pred_exact_vol", "pred_exact_val",
            "pred_exact_vol_diff", "pred_exact_val_diff",
            "nobdg_price_diff", "elasticity",
        }
        assert expected.issubset(set(result.columns)), (
            f"Missing columns: {expected - set(result.columns)}"
        )
