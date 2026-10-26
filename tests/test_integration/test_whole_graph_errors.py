"""Whole-DAG decomposition error handling and single-node identity.

``extract_effect(whole_graph=...)`` requires a computed MetaModel (every
node's ``dt_pred`` carrying ``predicted``), rejects unknown target node
names, and leaves single-node MetaModels byte-identical to the per-node
path.
"""

import pytest
from fermatrica import FermatricaError


@pytest.mark.integration
class TestWholeGraphErrors:
    """Error cases for whole-DAG decomposition on p10."""

    @pytest.fixture
    def p10_models(self, p10_models_ds, samples_path):
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
        return mm, ctx

    def test_raises_before_compute(self, p10_models):
        """whole_graph=True before mm.compute() raises FermatricaError."""
        import fermatrica_rep as fmr

        mm, _ = p10_models
        with pytest.raises(FermatricaError):
            fmr.extract_effect(mm, whole_graph=True)

    def test_unknown_target_name(self, p10_models):
        """Unknown whole_graph node name raises FermatricaError."""
        import fermatrica_rep as fmr

        mm, ctx = p10_models
        mm.compute(ctx.options["predefined"], scopes=ctx.scopes, allow_past=True)
        with pytest.raises(FermatricaError):
            fmr.extract_effect(mm, whole_graph="nope")

    def test_single_node_identical_to_default(self, p10_models_ds, samples_path):
        """A single-node MetaModel with whole_graph=True matches whole_graph=False."""
        from tests.test_integration.conftest import _activate_adhoc_path

        _activate_adhoc_path(samples_path / "p10_sample")

        import code_py.adhoc.reporting
        import code_py.adhoc.translators
        import fermatrica_rep as fmr

        _, _, sales_model, sales_ds = p10_models_ds
        options_path = str(
            samples_path
            / "p10_sample"
            / "code_py"
            / "model_data"
            / "options"
            / "context.xlsx"
        )
        ctx = fmr.load_context(options_path)

        sales = fmr.ModelNode(
            model=sales_model,
            ds=sales_ds,
            adhoc_code=[code_py.adhoc.translators, code_py.adhoc.reporting],
            language="english",
        )
        sales.scope_names = {
            s.scope_name for s in ctx.scopes
            if all(s.filters.get(col) and sales.ds[col].isin(values).any()
                   for col, values in s.filters.items())
        }
        mm = fmr.MetaModel(nodes=sales, compute_context=ctx)
        mm.compute(ctx.options["predefined"], scopes=ctx.scopes, allow_past=True)

        default = fmr.extract_effect(mm, whole_graph=False)
        whole = fmr.extract_effect(mm, whole_graph=True)
        assert list(whole.keys()) == ["model"]
        assert default["model"].equals(whole["model"])
