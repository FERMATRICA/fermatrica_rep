"""Integration tests for decomposition.py: decompose_main_plot.

Tests that dynamic decomposition plots return valid Plotly figures.
"""
import pytest
import pandas as pd
import numpy as np
import plotly.graph_objects as go



def _check_figure_traces(fig, min_traces=1):
    """Verify all traces have non-empty x/y data."""
    assert len(fig.data) >= min_traces
    for t in fig.data:
        assert len(t.x) > 0, f"Trace '{t.name}' has empty x"
        assert len(t.y) > 0, f"Trace '{t.name}' has empty y"

@pytest.mark.integration
class TestP00Decomp:
    """Decomposition plot tests against p00_sample (OLS, 1 brand)."""

    BRAND = "brand_x"

    @pytest.fixture
    def setup(self, p00_model_ds, samples_path):
        """Run extract_effect to produce split_m_m for decomposition plots."""
        import fermatrica_rep as fmr

        model, ds = p00_model_ds
        ds = ds.copy()

        root = samples_path / "p00_sample"
        opt_path = str(root / "code_py" / "model_data" / "options" / "options.xlsx")

        trans_dict = fmr.trans_dict_create(path=opt_path, sheet="translation")
        model_rep = fmr.ModelRep(ds, trans_dict=trans_dict, language="english")

        split_m_m = fmr.extract_effect(model, ds, model_rep)
        split_m_m = split_m_m.sort_values(["superbrand", "bs_key", "date"])

        return {"split_m_m": split_m_m, "model_rep": model_rep}

    def test_decompose_main_plot(self, setup):
        """decompose_main_plot returns go.Figure with stacked bar traces."""
        from fermatrica_rep.decomposition import decompose_main_plot

        fig = decompose_main_plot(
            split_m_m=setup["split_m_m"],
            brands=[self.BRAND],
            model_rep=setup["model_rep"],
            period="m",
            show_future=True,
            contour_line=True,
        )

        _check_figure_traces(fig)

    def test_decompose_main_plot_contour_line(self, setup):
        """decompose_main_plot with contour_line=True returns go.Figure."""
        from fermatrica_rep.decomposition import decompose_main_plot

        fig = decompose_main_plot(
            split_m_m=setup["split_m_m"],
            brands=[self.BRAND],
            model_rep=setup["model_rep"],
            period="m",
            show_future=True,
            contour_line=True,
        )

        assert isinstance(fig, go.Figure)

    def test_edge_case_show_future_false(self, setup):
        """decompose_main_plot with show_future=False returns go.Figure."""
        from fermatrica_rep.decomposition import decompose_main_plot

        fig = decompose_main_plot(
            split_m_m=setup["split_m_m"],
            brands=[self.BRAND],
            model_rep=setup["model_rep"],
            period="m",
            show_future=False,
            contour_line=True,
        )

        assert isinstance(fig, go.Figure)

    def test_decomp_golden(self, setup, if_debug):
        """decompose_main_plot matches golden expected trace values (p00 OLS)."""
        import os
        import yaml
        from fermatrica_rep.decomposition import decompose_main_plot
        from tests.test_integration.helpers import compare_plot_golden

        pth = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', 'test_cases', 'decomp_golden_p00.yaml')
        if not os.path.exists(pth):
            pytest.skip("Golden YAML file not found")

        with open(pth, mode='rt', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f)

        fig = decompose_main_plot(
            split_m_m=setup["split_m_m"],
            brands=[self.BRAND],
            model_rep=setup["model_rep"],
            period="m",
            show_future=True,
            contour_line=True,
        )

        for item in yaml_data:
            if item.get('sample') == 'p00':
                expected = item.get('expected', {})
                try:
                    compare_plot_golden(fig, expected)
                except AssertionError:
                    if if_debug:
                        import traceback
                        traceback.print_exc()
                        breakpoint()
                        for t in fig.data:
                            print(f"Trace '{t.name}': y_sum={sum(t.y):.4f}, y={list(t.y)}")
                    else:
                        raise
                else:
                    break

    def test_decomp_value_sum_matches_prediction(self, p00_model_ds, setup):
        """extract_effect decomposition values sum equals model prediction per period."""
        import fermatrica_rep as fmr
        import fermatrica as fm

        model, ds = p00_model_ds
        ds = ds.copy()
        model_rep = setup["model_rep"]

        split_m_m = fmr.extract_effect(model, ds, model_rep)

        sum_per_period = split_m_m.groupby(["superbrand", "bs_key", "date"])["value"].sum().reset_index()
        sum_per_period.columns = ["superbrand", "bs_key", "date", "decomposed_total"]

        pred_values, _, _ = fm.fit_predict(ds, model, if_fit=False, return_full=True)
        ds_pred = ds[["superbrand", "bs_key", "date"]].copy()
        ds_pred["predicted"] = pred_values.values
        ds_pred = ds_pred.dropna(subset=["predicted"])
        pred_agg = ds_pred.groupby(["superbrand", "bs_key", "date"], as_index=False)["predicted"].sum()

        merged = sum_per_period.merge(pred_agg, on=["superbrand", "bs_key", "date"], how="inner")

        for _, row in merged.iterrows():
            if abs(row["predicted"]) > 1e-10:
                rel_diff = abs(row["decomposed_total"] - row["predicted"]) / abs(row["predicted"])
                assert rel_diff < 1e-6, (
                    f"Decomposition [{row['superbrand']}, {row['date']}]: "
                    f"decomposed={row['decomposed_total']:.4f}, "
                    f"predicted={row['predicted']:.4f}, rel_diff={rel_diff:.4%}"
                )


@pytest.mark.integration
class TestP01Decomp:
    """Decomposition plot tests against p01_sample (panel LME, 3 brands)."""

    BRAND = "brand_1"

    @pytest.fixture
    def setup(self, p01_model_ds, samples_path):
        """Run extract_effect to produce split_m_m for decomposition plots."""
        import fermatrica_rep as fmr

        model, ds = p01_model_ds
        ds = ds.copy()

        root = samples_path / "p01_sample"
        opt_path = str(root / "code_py" / "model_data" / "options" / "options.xlsx")

        trans_dict = fmr.trans_dict_create(path=opt_path, sheet="translation")
        model_rep = fmr.ModelRep(ds, trans_dict=trans_dict, language="english")

        split_m_m = fmr.extract_effect(model, ds, model_rep)
        split_m_m = split_m_m.sort_values(["superbrand", "bs_key", "date"])

        return {"split_m_m": split_m_m, "model_rep": model_rep}

    def test_decompose_main_plot(self, setup):
        """decompose_main_plot returns go.Figure with stacked bar traces."""
        from fermatrica_rep.decomposition import decompose_main_plot

        fig = decompose_main_plot(
            split_m_m=setup["split_m_m"],
            brands=[self.BRAND],
            model_rep=setup["model_rep"],
            period="m",
            show_future=True,
            contour_line=True,
        )

        _check_figure_traces(fig)

    def test_decompose_main_plot_multi_brand(self, setup):
        """decompose_main_plot can accept multiple brands."""
        from fermatrica_rep.decomposition import decompose_main_plot

        fig = decompose_main_plot(
            split_m_m=setup["split_m_m"],
            brands=["brand_1", "brand_2"],
            model_rep=setup["model_rep"],
            period="m",
            show_future=False,
            contour_line=False,
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_edge_case_weekly_period(self, setup):
        """decompose_main_plot with period='w' returns go.Figure."""
        from fermatrica_rep.decomposition import decompose_main_plot

        fig = decompose_main_plot(
            split_m_m=setup["split_m_m"],
            brands=[self.BRAND],
            model_rep=setup["model_rep"],
            period="w",
            show_future=True,
            contour_line=True,
        )

        assert isinstance(fig, go.Figure)

    def test_edge_case_brands_none(self, setup):
        """decompose_main_plot with brands=None returns go.Figure for all brands."""
        from fermatrica_rep.decomposition import decompose_main_plot

        fig = decompose_main_plot(
            split_m_m=setup["split_m_m"],
            brands=None,
            model_rep=setup["model_rep"],
            period="m",
            show_future=True,
            contour_line=True,
        )

        assert isinstance(fig, go.Figure)

    def test_decomp_golden(self, setup, if_debug):
        """decompose_main_plot matches golden expected trace values (p01 LME)."""
        import os
        import yaml
        from fermatrica_rep.decomposition import decompose_main_plot
        from tests.test_integration.helpers import compare_plot_golden

        pth = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', 'test_cases', 'decomp_golden_p01.yaml')
        if not os.path.exists(pth):
            pytest.skip("Golden YAML file not found")

        with open(pth, mode='rt', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f)

        fig = decompose_main_plot(
            split_m_m=setup["split_m_m"],
            brands=[self.BRAND],
            model_rep=setup["model_rep"],
            period="m",
            show_future=True,
            contour_line=True,
        )

        for item in yaml_data:
            if item.get('sample') == 'p01':
                expected = item.get('expected', {})
                try:
                    compare_plot_golden(fig, expected)
                except AssertionError:
                    if if_debug:
                        import traceback
                        traceback.print_exc()
                        breakpoint()
                        for t in fig.data:
                            print(f"Trace '{t.name}': y_sum={sum(t.y):.4f}, y={list(t.y)}")
                    else:
                        raise
                else:
                    break

    def test_decomp_value_sum_matches_prediction(self, p01_model_ds, setup):
        """extract_effect decomposition values sum equals model prediction per period."""
        import fermatrica_rep as fmr
        import fermatrica as fm

        model, ds = p01_model_ds
        ds = ds.copy()
        model_rep = setup["model_rep"]

        split_m_m = fmr.extract_effect(model, ds, model_rep)

        sum_per_period = split_m_m.groupby(["superbrand", "bs_key", "date"])["value"].sum().reset_index()
        sum_per_period.columns = ["superbrand", "bs_key", "date", "decomposed_total"]

        pred_values, _, _ = fm.fit_predict(ds, model, if_fit=False, return_full=True)
        ds_pred = ds[["superbrand", "bs_key", "date"]].copy()
        ds_pred["predicted"] = pred_values.values
        ds_pred = ds_pred.dropna(subset=["predicted"])
        pred_agg = ds_pred.groupby(["superbrand", "bs_key", "date"], as_index=False)["predicted"].sum()

        merged = sum_per_period.merge(pred_agg, on=["superbrand", "bs_key", "date"], how="inner")

        for _, row in merged.iterrows():
            if abs(row["predicted"]) > 1e-10:
                rel_diff = abs(row["decomposed_total"] - row["predicted"]) / abs(row["predicted"])
                assert rel_diff < 1e-6, (
                    f"Decomposition [{row['superbrand']}, {row['date']}]: "
                    f"decomposed={row['decomposed_total']:.4f}, "
                    f"predicted={row['predicted']:.4f}, rel_diff={rel_diff:.4%}"
                )


@pytest.mark.integration
class TestP02Decomp:
    """Decomposition plot tests against p02_sample (SKU-level panel)."""

    BRAND = "brand_1"

    @pytest.fixture
    def setup(self, p02_model_ds, samples_path):
        """Run extract_effect to produce split_m_m for decomposition plots."""
        import fermatrica_rep as fmr

        model, ds = p02_model_ds
        ds = ds.copy()

        root = samples_path / "p02_sample"
        opt_path = str(root / "code_py" / "model_data" / "options" / "options.xlsx")

        trans_dict = fmr.trans_dict_create(path=opt_path, sheet="translation")
        model_rep = fmr.ModelRep(ds, trans_dict=trans_dict, language="english")

        split_m_m = fmr.extract_effect(model, ds, model_rep)
        split_m_m = split_m_m.sort_values(["superbrand", "bs_key", "date"])

        return {"split_m_m": split_m_m, "model_rep": model_rep}

    def test_decompose_main_plot(self, setup):
        """decompose_main_plot returns go.Figure for single brand."""
        from fermatrica_rep.decomposition import decompose_main_plot

        fig = decompose_main_plot(
            split_m_m=setup["split_m_m"],
            brands=[self.BRAND],
            model_rep=setup["model_rep"],
            period="m",
            show_future=True,
            contour_line=True,
        )

        _check_figure_traces(fig)

    def test_decompose_main_plot_multi_brand(self, setup):
        """decompose_main_plot can accept two brands."""
        from fermatrica_rep.decomposition import decompose_main_plot

        fig = decompose_main_plot(
            split_m_m=setup["split_m_m"],
            brands=["brand_1", "brand_2"],
            model_rep=setup["model_rep"],
            period="m",
            show_future=False,
            contour_line=False,
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_decomp_value_sum_matches_prediction(self, p02_model_ds, setup):
        """extract_effect decomposition values sum equals model prediction per period."""
        import fermatrica_rep as fmr
        import fermatrica as fm

        model, ds = p02_model_ds
        ds = ds.copy()
        model_rep = setup["model_rep"]

        split_m_m = fmr.extract_effect(model, ds, model_rep)

        sum_per_period = split_m_m.groupby(["superbrand", "bs_key", "date"])["value"].sum().reset_index()
        sum_per_period.columns = ["superbrand", "bs_key", "date", "decomposed_total"]

        pred_values, _, _ = fm.fit_predict(ds, model, if_fit=False, return_full=True)
        ds_pred = ds[["superbrand", "bs_key", "date"]].copy()
        ds_pred["predicted"] = pred_values.values
        ds_pred = ds_pred.dropna(subset=["predicted"])
        pred_agg = ds_pred.groupby(["superbrand", "bs_key", "date"], as_index=False)["predicted"].sum()

        merged = sum_per_period.merge(pred_agg, on=["superbrand", "bs_key", "date"], how="inner")

        for _, row in merged.iterrows():
            if abs(row["predicted"]) > 1e-10:
                rel_diff = abs(row["decomposed_total"] - row["predicted"]) / abs(row["predicted"])
                assert rel_diff < 1e-6, (
                    f"Decomposition [{row['superbrand']}, {row['date']}]: "
                    f"decomposed={row['decomposed_total']:.4f}, "
                    f"predicted={row['predicted']:.4f}, rel_diff={rel_diff:.4%}"
                )


@pytest.mark.integration
class TestP03Decomp:
    """Decomposition plot tests against p03_sample (ABA monthly panel)."""

    BRAND = "brand_1"

    @pytest.fixture
    def setup(self, p03_model_ds, samples_path):
        """Run extract_effect to produce split_m_m for ABA decomposition."""
        import fermatrica_rep as fmr

        model, ds = p03_model_ds
        ds = ds.copy()

        root = samples_path / "p03_sample"
        opt_path = str(root / "code_py" / "model_data" / "options" / "options.xlsx")

        trans_dict = fmr.trans_dict_create(path=opt_path, sheet="translation")
        model_rep = fmr.ModelRep(ds, trans_dict=trans_dict, language="english")

        split_m_m = fmr.extract_effect(model, ds, model_rep)
        split_m_m = split_m_m.sort_values(["superbrand", "bs_key", "date"])

        return {"split_m_m": split_m_m, "model_rep": model_rep}

    def test_decompose_main_plot(self, setup):
        """decompose_main_plot returns go.Figure for ABA model."""
        from fermatrica_rep.decomposition import decompose_main_plot

        fig = decompose_main_plot(
            split_m_m=setup["split_m_m"],
            brands=[self.BRAND],
            model_rep=setup["model_rep"],
            period="m",
            show_future=True,
            contour_line=True,
        )

        _check_figure_traces(fig)

    def test_decomp_value_sum_matches_prediction(self, p03_model_ds, setup):
        """extract_effect decomposition values sum equals model prediction per period."""
        import fermatrica_rep as fmr
        import fermatrica as fm

        model, ds = p03_model_ds
        ds = ds.copy()
        model_rep = setup["model_rep"]

        split_m_m = fmr.extract_effect(model, ds, model_rep)

        sum_per_period = split_m_m.groupby(["superbrand", "bs_key", "date"])["value"].sum().reset_index()
        sum_per_period.columns = ["superbrand", "bs_key", "date", "decomposed_total"]

        pred_values, _, _ = fm.fit_predict(ds, model, if_fit=False, return_full=True)
        ds_pred = ds[["superbrand", "bs_key", "date"]].copy()
        ds_pred["predicted"] = pred_values.values
        ds_pred = ds_pred.dropna(subset=["predicted"])
        pred_agg = ds_pred.groupby(["superbrand", "bs_key", "date"], as_index=False)["predicted"].sum()

        merged = sum_per_period.merge(pred_agg, on=["superbrand", "bs_key", "date"], how="inner")

        for _, row in merged.iterrows():
            if abs(row["predicted"]) > 1e-10:
                rel_diff = abs(row["decomposed_total"] - row["predicted"]) / abs(row["predicted"])
                assert rel_diff < 1e-6, (
                    f"Decomposition [{row['superbrand']}, {row['date']}]: "
                    f"decomposed={row['decomposed_total']:.4f}, "
                    f"predicted={row['predicted']:.4f}, rel_diff={rel_diff:.4%}"
                )


@pytest.mark.integration
class TestP04Decomp:
    """Decomposition plot tests against p04_sample (regional panel)."""

    BRAND = "brand_1"

    @pytest.fixture
    def setup(self, p04_model_ds, samples_path):
        """Run extract_effect to produce split_m_m for regional decomposition."""
        import fermatrica_rep as fmr

        model, ds = p04_model_ds
        ds = ds.copy()

        root = samples_path / "p04_sample"
        opt_path = str(root / "code_py" / "model_data" / "options" / "options.xlsx")

        trans_dict = fmr.trans_dict_create(path=opt_path, sheet="translation")
        model_rep = fmr.ModelRep(ds, trans_dict=trans_dict, language="english")

        split_m_m = fmr.extract_effect(model, ds, model_rep)
        split_m_m = split_m_m.sort_values(["superbrand", "bs_key", "date", "master"])

        return {"split_m_m": split_m_m, "model_rep": model_rep}

    def test_decompose_main_plot(self, setup):
        """decompose_main_plot returns go.Figure for single region."""
        from fermatrica_rep.decomposition import decompose_main_plot

        fig = decompose_main_plot(
            split_m_m=setup["split_m_m"],
            brands=[self.BRAND],
            model_rep=setup["model_rep"],
            period="m",
            show_future=True,
            contour_line=True,
        )

        _check_figure_traces(fig)

    def test_decompose_main_plot_multi_region(self, setup):
        """decompose_main_plot can accept multiple regions via brands=None."""
        from fermatrica_rep.decomposition import decompose_main_plot

        fig = decompose_main_plot(
            split_m_m=setup["split_m_m"],
            brands=None,
            model_rep=setup["model_rep"],
            period="m",
            show_future=False,
            contour_line=False,
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_decomp_value_sum_matches_prediction(self, p04_model_ds, setup):
        """extract_effect decomposition values sum equals model prediction per period (regional)."""
        import fermatrica_rep as fmr
        import fermatrica as fm

        model, ds = p04_model_ds
        ds = ds.copy()
        model_rep = setup["model_rep"]

        split_m_m = fmr.extract_effect(model, ds, model_rep)

        sum_per_period = split_m_m.groupby(["superbrand", "bs_key", "date", "master"])["value"].sum().reset_index()
        sum_per_period.columns = ["superbrand", "bs_key", "date", "master", "decomposed_total"]

        pred_values, _, _ = fm.fit_predict(ds, model, if_fit=False, return_full=True)
        ds_pred = ds[["superbrand", "bs_key", "date", "master"]].copy()
        ds_pred["predicted"] = pred_values.values
        ds_pred = ds_pred.dropna(subset=["predicted"])
        pred_agg = ds_pred.groupby(["superbrand", "bs_key", "date", "master"], as_index=False)["predicted"].sum()

        merged = sum_per_period.merge(pred_agg, on=["superbrand", "bs_key", "date", "master"], how="inner")

        for _, row in merged.iterrows():
            if abs(row["predicted"]) > 1e-10:
                rel_diff = abs(row["decomposed_total"] - row["predicted"]) / abs(row["predicted"])
                assert rel_diff < 1e-6, (
                    f"Decomposition [{row['superbrand']}, {row['master']}, {row['date']}]: "
                    f"decomposed={row['decomposed_total']:.4f}, "
                    f"predicted={row['predicted']:.4f}, rel_diff={rel_diff:.4%}"
                )
