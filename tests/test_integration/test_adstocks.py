"""Integration tests for adstocks.py: adstocks_data.

Tests that adstock decay data generation works against real model data.
p00 is skipped — it has no adstock transformations.
"""
import numpy as np
import pytest


@pytest.mark.integration
class TestP01Adstocks:
    """Adstock data tests against p01_sample (panel LME, 3 brands)."""

    BRAND = "brand_1"

    @pytest.fixture
    def setup(self, p01_model_ds, samples_path):
        """Build ModelRep and discover media variables for adstock calculation."""
        import fermatrica_rep as fmr

        model, ds = p01_model_ds
        ds = ds.copy()

        root = samples_path / "p01_sample"
        opt_path = str(root / "code_py" / "model_data" / "options" / "options.xlsx")

        trans_dict = fmr.trans_dict_create(path=opt_path, sheet="translation")
        model_rep = fmr.ModelRep(ds, trans_dict=trans_dict, language="english")

        all_vars = ds.columns.tolist()
        media_exclude = ("date", "listed", "Intercept", "base", "price",
                         "season", "superbrand", "bs_key", "kpi_coef",
                         "units", "aba", "master", "opt_media")
        cln_meas = tuple(
            v for v in all_vars
            if not any(x in v for x in media_exclude)
            and any(x in v for x in ("tv", "ots", "ooh", "pr_", "digital"))
        )[:3]

        return {"model": model, "ds": ds, "model_rep": model_rep,
                "cln_meas": cln_meas}

    def test_adstocks_data(self, setup):
        """adstocks_data returns a non-empty DataFrame."""
        from fermatrica_rep.analysis.adstocks import adstocks_data

        result = adstocks_data(
            model=setup["model"],
            ds=setup["ds"],
            model_rep=setup["model_rep"],
            filters={"superbrand": [self.BRAND]},
            cln_meas=setup["cln_meas"],
            cln_dim=("superbrand", "bs_key", "date", "listed"),
            n=10,
        )

        assert result is not None
        assert len(result) > 0

        # Verify decay shape: weights should be non-increasing over time
        for col in result.columns:
            if col != "step":
                values = result[col].values
                diffs = np.diff(values)
                assert np.all(diffs <= 1e-10), f"Column '{col}' is not non-increasing (not decaying)"

    def test_adstock_weights_sum_to_one(self, setup):
        """Each adstock column's weights sum to approximately 1.0."""
        from fermatrica_rep.analysis.adstocks import adstocks_data

        result = adstocks_data(
            model=setup["model"],
            ds=setup["ds"],
            model_rep=setup["model_rep"],
            filters={"superbrand": [self.BRAND]},
            cln_meas=setup["cln_meas"],
            cln_dim=("superbrand", "bs_key", "date", "listed"),
            n=10,
        )

        for col in result.columns:
            if col != "step":
                col_sum = result[col].sum()
                assert abs(col_sum - 1.0) < 1e-6, \
                    f"Column '{col}' weights sum to {col_sum}, expected ~1.0"

    def test_adstock_weights_first_is_one(self, setup):
        """First weight (lag=0, first row) should be 1.0."""
        from fermatrica_rep.analysis.adstocks import adstocks_data

        result = adstocks_data(
            model=setup["model"],
            ds=setup["ds"],
            model_rep=setup["model_rep"],
            filters={"superbrand": [self.BRAND]},
            cln_meas=setup["cln_meas"],
            cln_dim=("superbrand", "bs_key", "date", "listed"),
            n=10,
        )

        for col in result.columns:
            if col != "step":
                assert abs(result[col].iloc[0] - 1.0) < 1e-10, \
                    f"Column '{col}' first weight = {result[col].iloc[0]}, expected 1.0"
