"""Whole-DAG decomposition tests on p10 (category static -> sales).

p10 is a two-level OLS -> LME DAG where ``pred_merge_1`` broadcasts the
category prediction on ``date`` and the sales model consumes the non-linear
``category_predicted_softmax`` transform. ``extract_effect(whole_graph=True)``
must replace the single ``Category`` factor with ``Category.<factor>`` leaves
that sum back to the original link contribution.
"""

import numpy as np
import pytest


@pytest.mark.integration
class TestWholeGraphP10:
    """Exercise whole-DAG decomposition on the p10 static-node DAG."""

    BRAND = "brand_1"

    @pytest.fixture
    def p10_setup(self, p10_models_ds, samples_path):
        from tests.test_integration.conftest import _activate_adhoc_path

        _activate_adhoc_path(samples_path / "p10_sample")

        import code_py.adhoc.reporting
        import code_py.adhoc.translators
        from code_py.adhoc.reporting import pred_merge_1
        import fermatrica_rep as fmr

        category_model, category_ds, sales_model, sales_ds = p10_models_ds
        options_path = str(
            samples_path
            / "p10_sample"
            / "code_py"
            / "model_data"
            / "options"
            / "context.xlsx"
        )
        ctx = fmr.load_context(options_path)

        category = fmr.ModelNode(
            model=category_model,
            ds=category_ds,
            adhoc_code=[code_py.adhoc.translators, code_py.adhoc.reporting],
            language="english",
            static=True,
        )
        sales = fmr.ModelNode(
            model=sales_model,
            ds=sales_ds,
            adhoc_code=[code_py.adhoc.translators, code_py.adhoc.reporting],
            language="english",
        )
        category.scope_names = {
            s.scope_name for s in ctx.scopes
            if all(category.ds[col].isin(values).any() for col, values in s.filters.items())
        }
        sales.scope_names = {
            s.scope_name for s in ctx.scopes
            if all(s.filters.get(col) and sales.ds[col].isin(values).any()
                   for col, values in s.filters.items())
        }
        mm = fmr.MetaModel(
            nodes={"category": category, "sales": sales},
            edges=[("category", "sales", pred_merge_1)],
            compute_context=ctx,
        )
        mm.compute(ctx.options["predefined"], scopes=ctx.scopes, allow_past=True)
        return mm, ctx

    def test_expanded_sales_split_sums_equal_prediction(self, p10_setup):
        """Expanded sales decomposition sums per (superbrand, date) equal prediction."""
        import fermatrica_rep as fmr

        mm, ctx = p10_setup
        split = fmr.extract_effect(mm, whole_graph=True)["sales"]

        summed = split.groupby(["superbrand", "date"])["value"].sum().reset_index()
        pred = (
            mm.nodes["sales"]
            .dt_pred[["superbrand", "date", "predicted"]]
            .groupby(["superbrand", "date"], as_index=False)["predicted"]
            .sum()
        )
        merged = summed.merge(pred, on=["superbrand", "date"], how="inner")
        assert not merged.empty
        for _, row in merged.iterrows():
            if abs(row["predicted"]) > 1e-10:
                rel = abs(row["value"] - row["predicted"]) / abs(row["predicted"])
                assert rel < 1e-6, (
                    f"Expanded decomposition [{row['superbrand']}, {row['date']}]: "
                    f"sum={row['value']:.4f}, predicted={row['predicted']:.4f}, rel={rel:.4%}"
                )

    def test_no_bare_category_factor(self, p10_setup):
        """The single Category factor is replaced by Category.<leaf> rows."""
        import fermatrica_rep as fmr

        mm, _ = p10_setup
        split = fmr.extract_effect(mm, whole_graph=True)["sales"]
        variables = split["variable"].unique()

        assert not any(v == "Category" for v in variables)
        leaves = [v for v in variables if isinstance(v, str) and v.startswith("Category.")]
        assert leaves, "No Category.<factor> leaves produced"
        assert any(v.endswith("TV") for v in leaves)

    def test_category_leaves_sum_to_single_category(self, p10_setup):
        """Category.<factor> rows sum to the old single Category value per group."""
        import fermatrica_rep as fmr

        mm, _ = p10_setup
        old = fmr.extract_effect(mm)["sales"]
        new = fmr.extract_effect(mm, whole_graph=True)["sales"]

        gcols = ["superbrand", "bs_key", "date", "listed"]
        old_cat = old.loc[old["variable"] == "Category", gcols + ["value"]].rename(
            columns={"value": "old_cat"}
        )
        new_cat = (
            new.loc[new["variable"].astype(str).str.startswith("Category."), gcols + ["value"]]
            .groupby(gcols, as_index=False)["value"]
            .sum()
            .rename(columns={"value": "new_cat"})
        )
        merged = old_cat.merge(new_cat, on=gcols, how="outer")
        assert not merged.empty
        diff = (merged["new_cat"].fillna(0) - merged["old_cat"].fillna(0)).abs()
        assert diff.max() < 1e-6, f"max |expanded sum - single link| = {diff.max()}"

    def test_individual_leaves_finite_and_signed(self, p10_setup):
        """Each Category.<leaf> row is finite; leaves take both signs."""
        import fermatrica_rep as fmr

        mm, _ = p10_setup
        split = fmr.extract_effect(mm, whole_graph=True)["sales"]
        leaves = split.loc[split["variable"].astype(str).str.startswith("Category.")]
        assert np.isfinite(leaves["value"]).all()
        assert (leaves["value"] > 0).any() and (leaves["value"] < 0).any()

    def test_target_name_returns_single_frame(self, p10_setup):
        """whole_graph='sales' returns a single DataFrame equal to the dict entry."""
        import pandas as pd
        import fermatrica_rep as fmr

        mm, _ = p10_setup
        by_name = fmr.extract_effect(mm, whole_graph="sales")
        by_flag = fmr.extract_effect(mm, whole_graph=True)
        assert isinstance(by_name, pd.DataFrame)
        assert by_name.equals(by_flag["sales"])

    def test_predecessor_node_split_unchanged(self, p10_setup):
        """The static predecessor (no incoming links) matches the per-node split."""
        import fermatrica_rep as fmr

        mm, _ = p10_setup
        old = fmr.extract_effect(mm)["category"]
        new = fmr.extract_effect(mm, whole_graph=True)["category"]
        assert old.equals(new)
