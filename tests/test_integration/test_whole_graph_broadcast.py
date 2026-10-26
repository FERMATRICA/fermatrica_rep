"""Whole-DAG decomposition broadcast tests on p05 (BHT -> Sales).

p05 is a two-level panel: the brand-month BHT model's prediction is merged
into the SKU-week Sales dataset on ``(superbrand, year, month)`` and written
to ``aba_predicted``. The whole-DAG expansion must broadcast the BHT factor
shares onto the finer-grained Sales rows, preserving sums.
"""

import numpy as np
import pytest


@pytest.mark.integration
class TestWholeGraphBroadcast:
    """Broadcast alignment of whole-DAG expansion on the p05 DAG."""

    @pytest.fixture
    def p05_setup(self, p05_models_ds, samples_path):
        from tests.test_integration.conftest import _activate_adhoc_path

        _activate_adhoc_path(samples_path / "p05_sample")

        import code_py.adhoc.reporting
        import code_py.adhoc.translators
        from code_py.adhoc.reporting import pred_merge_1
        import fermatrica_rep as fmr

        bht_model, bht_ds, sales_model, sales_ds = p05_models_ds
        options_path = str(
            samples_path
            / "p05_sample"
            / "code_py"
            / "model_data"
            / "options"
            / "context.xlsx"
        )
        ctx = fmr.load_context(options_path)

        node_bht = fmr.ModelNode(
            model=bht_model,
            ds=bht_ds,
            adhoc_code=[code_py.adhoc.translators, code_py.adhoc.reporting],
            language="english",
        )
        node_sales = fmr.ModelNode(
            model=sales_model,
            ds=sales_ds,
            adhoc_code=[code_py.adhoc.translators, code_py.adhoc.reporting],
            language="english",
        )
        node_bht.scope_names = {s.scope_name for s in ctx.scopes}
        node_sales.scope_names = {s.scope_name for s in ctx.scopes}
        mm = fmr.MetaModel(
            nodes={"bht": node_bht, "sales": node_sales},
            edges=[("bht", "sales", pred_merge_1)],
            compute_context=ctx,
        )
        mm.compute(ctx.options["predefined"], scopes=ctx.scopes, allow_past=True)
        return mm

    def test_aba_link_expanded_and_broadcast_sums_preserved(self, p05_setup):
        """ABA is split into ABA.<leaf> rows summing to the single link value."""
        import fermatrica_rep as fmr

        mm = p05_setup
        old = fmr.extract_effect(mm)["sales"]
        new = fmr.extract_effect(mm, whole_graph=True)["sales"]

        variables = new["variable"].unique()
        assert not any(v == "ABA" for v in variables)
        leaves = [v for v in variables if isinstance(v, str) and v.startswith("ABA.")]
        assert leaves, "No ABA.<factor> leaves produced"

        gcols = ["superbrand", "bs_key", "date", "listed"]
        old_aba = old.loc[old["variable"] == "ABA", gcols + ["value"]].rename(
            columns={"value": "old_aba"}
        )
        new_aba = (
            new.loc[new["variable"].astype(str).str.startswith("ABA."), gcols + ["value"]]
            .groupby(gcols, as_index=False)["value"]
            .sum()
            .rename(columns={"value": "new_aba"})
        )
        merged = old_aba.merge(new_aba, on=gcols, how="outer")
        assert not merged.empty
        diff = (merged["new_aba"].fillna(0) - merged["old_aba"].fillna(0)).abs()
        assert diff.max() < 1e-6, f"max |expanded sum - single link| = {diff.max()}"

    def test_total_decomposition_unchanged(self, p05_setup):
        """Whole-DAG total per (superbrand, date) equals the per-node total."""
        import fermatrica_rep as fmr

        mm = p05_setup
        old_tot = fmr.extract_effect(mm)["sales"].groupby(
            ["superbrand", "date"], as_index=False
        )["value"].sum()
        new_tot = fmr.extract_effect(mm, whole_graph=True)["sales"].groupby(
            ["superbrand", "date"], as_index=False
        )["value"].sum()
        merged = old_tot.merge(new_tot, on=["superbrand", "date"],
                               suffixes=("_old", "_new"))
        assert np.allclose(merged["value_old"], merged["value_new"], rtol=1e-6, atol=1e-6)

    def test_leaves_match_predecessor_factors(self, p05_setup):
        """ABA.<leaf> names equal the BHT node's own factor names."""
        import fermatrica_rep as fmr

        mm = p05_setup
        bht_factors = set(fmr.extract_effect(mm)["bht"]["variable"].unique())
        leaves = set(
            v[4:] for v in fmr.extract_effect(mm, whole_graph=True)["sales"]["variable"]
            if isinstance(v, str) and v.startswith("ABA.")
        )
        assert leaves == bht_factors
