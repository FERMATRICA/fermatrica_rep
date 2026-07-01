"""Integration tests for elasticity_price.py: elasticity_data.

Tests that price elasticity data generation works against real model data.
p00 is skipped — it has no price variables.
"""
import numpy as np
import pytest
from datetime import date


@pytest.mark.integration
class TestP01Elasticity:
    """Price elasticity tests against p01_sample (panel LME, 3 brands)."""

    BRAND = "brand_1"

    @pytest.fixture
    def setup(self, p01_model_ds, samples_path):
        """Build model_rep, opt_set, and option dict for elasticity calculation."""
        import fermatrica_rep as fmr

        model, ds = p01_model_ds
        ds = ds.copy()

        root = samples_path / "p01_sample"
        opt_path = str(root / "code_py" / "model_data" / "options" / "options.xlsx")
        ssn_path = str(root / "code_py" / "model_data" / "options" / "seasonality.xlsx")

        trans_dict = fmr.trans_dict_create(path=opt_path, sheet="translation")
        model_rep = fmr.ModelRep(ds, trans_dict=trans_dict, language="english")
        ds = fmr.media_ssn_apply(ds, path=ssn_path, sheet="data")

        opt_set = fmr.OptionSettings(
            target=[[self.BRAND]],
            date_start=date(2025, 6, 30),
            date_end=date(2025, 12, 31),
            ref_date_start=date(2024, 1, 1),
            ref_date_end=date(2024, 12, 31),
            apply_vars=["superbrand"],
            plan_period="exact",
        )

        option = {"bdg_tv": 50, "bdg_digital": 30, "bdg_pr": 20, "price": 1.0}

        return {"model": model, "ds": ds, "model_rep": model_rep,
                "opt_set": opt_set, "option": option}

    def test_elasticity_data(self, setup):
        """elasticity_data returns a non-empty DataFrame."""
        from fermatrica_rep.elasticity_price import elasticity_data

        result = elasticity_data(
            model=setup["model"],
            model_rep=setup["model_rep"],
            ds=setup["ds"],
            opt_set=setup["opt_set"],
            option=setup["option"],
            step=0.1,
            price_range=(-0.5, 0.5),
            cores=1,
            if_exact=True,
        )

        assert result is not None
        assert len(result) > 0

        # Verify all numeric columns are finite
        num_cols = result.select_dtypes(include=[np.number]).columns
        assert np.all(np.isfinite(result[num_cols].values)), \
            "elasticity_data produced NaN or Inf values"

    def test_elasticity_has_expected_columns(self, setup):
        """Result has expected column names."""
        from fermatrica_rep.elasticity_price import elasticity_data

        result = elasticity_data(
            model=setup["model"],
            model_rep=setup["model_rep"],
            ds=setup["ds"],
            opt_set=setup["opt_set"],
            option=setup["option"],
            step=0.1,
            price_range=(-0.5, 0.5),
            cores=1,
            if_exact=True,
        )

        expected = {"price", "pred_exact_vol", "pred_exact_val",
                    "pred_exact_vol_diff", "pred_exact_val_diff",
                    "price_diff", "elasticity"}
        assert expected.issubset(set(result.columns)), \
            f"Missing columns: {expected - set(result.columns)}"
