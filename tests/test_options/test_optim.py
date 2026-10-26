import numpy as np
import pandas as pd
import pytest

from fermatrica.basics.basics import fermatrica_error

from fermatrica_rep.options.option import Option
from fermatrica_rep.options.scope import OptionScope
from fermatrica_rep.meta_model.meta_model import MetaModel
from fermatrica_rep.meta_model.node import ModelNode

from tests.test_options.mock_helpers import (
    _make_mock_compute, _make_mock_meta_model, _make_ncma_meta_model,
    _MockNcmaModel, _MockComputeMulti, _make_mock_meta_model_multi,
)






class TestOptimizeBudgetLocalCobyla:

    SCOPE_NAME = "test_scope"

    def test_returns_option_with_budget_data(self, monkeypatch):
        monkeypatch.setattr(
            'fermatrica_rep.options.objective._last_node_name',
            lambda mm: "model",
        )
        meta_model = _make_mock_meta_model(self.SCOPE_NAME, kpi_multiplier=2.0)
        scopes = [OptionScope(
            scope_name=self.SCOPE_NAME,
            date_start="2024-01-01", date_end="2024-12-01",
            ref_date_start="2023-01-01", ref_date_end="2023-12-01",
        )]
        borders_dict = {'bdg_x': [0, 100, 50], 'bdg_y': [0, 100, 50]}
        option = Option(name="test_opt", budget_data={self.SCOPE_NAME: {'bdg_x': 50, 'bdg_y': 50}})

        from fermatrica_rep.options.optim import optimize_budget_local_cobyla

        result = optimize_budget_local_cobyla(
            meta_model=meta_model, scopes=scopes, budget_scope=self.SCOPE_NAME,
            borders_dict=borders_dict, option=option, bdg_size=100,
            epochs=1, iters_epoch=50, ftol_abs=1e-4, verbose=False,
        )

        assert isinstance(result, Option)
        assert self.SCOPE_NAME in result.budget_data
        bdg = result.budget_data[self.SCOPE_NAME]
        assert sum(v for k, v in bdg.items() if k.startswith('bdg_')) > 0

    def test_bdg_size_zero_does_not_crash(self, monkeypatch):
        monkeypatch.setattr(
            'fermatrica_rep.options.objective._last_node_name',
            lambda mm: "model",
        )
        meta_model = _make_mock_meta_model(self.SCOPE_NAME)
        scopes = [OptionScope(
            scope_name=self.SCOPE_NAME,
            date_start="2024-01-01", date_end="2024-03-01",
            ref_date_start="2024-01-01", ref_date_end="2024-03-01",
        )]
        borders_dict = {'bdg_x': [0, 100, 0], 'bdg_y': [0, 100, 0]}
        option = Option(name="test_opt", budget_data={self.SCOPE_NAME: {'bdg_x': 0, 'bdg_y': 0}})

        from fermatrica_rep.options.optim import optimize_budget_local_cobyla

        result = optimize_budget_local_cobyla(
            meta_model=meta_model, scopes=scopes, budget_scope=self.SCOPE_NAME,
            borders_dict=borders_dict, option=option, bdg_size=0,
            epochs=1, iters_epoch=10, ftol_abs=1e-2, verbose=False,
        )
        assert isinstance(result, Option)

    def test_prefers_higher_coef_variable(self, monkeypatch):
        monkeypatch.setattr(
            'fermatrica_rep.options.objective._last_node_name',
            lambda mm: "model",
        )
        meta_model = _make_mock_meta_model(
            self.SCOPE_NAME, coefs={'bdg_x': 3.0, 'bdg_y': 1.0},
        )
        scopes = [OptionScope(
            scope_name=self.SCOPE_NAME,
            date_start="2024-01-01", date_end="2024-12-01",
            ref_date_start="2023-01-01", ref_date_end="2023-12-01",
        )]
        borders_dict = {'bdg_x': [0, 100, 50], 'bdg_y': [0, 100, 50]}
        option = Option(name="test_opt", budget_data={self.SCOPE_NAME: {'bdg_x': 50, 'bdg_y': 50}})

        from fermatrica_rep.options.optim import optimize_budget_local_cobyla

        result = optimize_budget_local_cobyla(
            meta_model=meta_model, scopes=scopes, budget_scope=self.SCOPE_NAME,
            borders_dict=borders_dict, option=option, bdg_size=100,
            epochs=1, iters_epoch=100, ftol_abs=1e-4, verbose=False,
        )
        bdg = result.budget_data[self.SCOPE_NAME]
        assert bdg.get('bdg_x', 0) >= bdg.get('bdg_y', 0)

    def test_raises_on_invalid_borders(self):
        from fermatrica_rep.options.optim import optimize_budget_local_cobyla

        meta_model = _make_mock_meta_model(self.SCOPE_NAME)
        scopes = [OptionScope(
            scope_name=self.SCOPE_NAME,
            date_start="2024-01-01", date_end="2024-05-01",
            ref_date_start="2023-01-01", ref_date_end="2023-12-01",
        )]
        option = Option(name="test_opt", budget_data={self.SCOPE_NAME: {}})

        with pytest.raises(Exception):
            optimize_budget_local_cobyla(
                meta_model=meta_model, scopes=scopes, budget_scope=self.SCOPE_NAME,
                borders_dict={'bdg_x': [0]}, option=option,
                bdg_size=100, epochs=1, iters_epoch=10, verbose=False,
            )


# ---------------------------------------------------------------------------
# TestOptimizerDoesNotMutateInput
# ---------------------------------------------------------------------------

class TestOptimizerDoesNotMutateInput:

    SCOPE_NAME = "test_scope"

    def test_does_not_mutate_input_option(self, monkeypatch):
        monkeypatch.setattr(
            'fermatrica_rep.options.objective._last_node_name',
            lambda mm: "model",
        )
        meta_model = _make_mock_meta_model(self.SCOPE_NAME, kpi_multiplier=2.0)
        scopes = [OptionScope(
            scope_name=self.SCOPE_NAME,
            date_start="2024-01-01", date_end="2024-12-01",
            ref_date_start="2023-01-01", ref_date_end="2023-12-01",
        )]
        borders_dict = {'bdg_x': [0, 100, 50], 'bdg_y': [0, 100, 50]}
        option = Option(name="test_opt", budget_data={
            self.SCOPE_NAME: {'bdg_x': 50, 'bdg_y': 50},
        })

        from fermatrica_rep.options.optim import optimize_budget_local_cobyla

        snapshot = {k: dict(v) for k, v in option.budget_data.items()}

        optimize_budget_local_cobyla(
            meta_model=meta_model, scopes=scopes, budget_scope=self.SCOPE_NAME,
            borders_dict=borders_dict, option=option, bdg_size=100,
            epochs=2, iters_epoch=50, ftol_abs=1e-4, verbose=False,
        )

        assert option.budget_data == snapshot

    def test_returned_option_shares_no_inner_dict_with_input(self, monkeypatch):
        monkeypatch.setattr(
            'fermatrica_rep.options.objective._last_node_name',
            lambda mm: "model",
        )
        meta_model = _make_mock_meta_model(self.SCOPE_NAME, kpi_multiplier=2.0)
        scopes = [OptionScope(
            scope_name=self.SCOPE_NAME,
            date_start="2024-01-01", date_end="2024-12-01",
            ref_date_start="2023-01-01", ref_date_end="2023-12-01",
        )]
        borders_dict = {'bdg_x': [0, 100, 50], 'bdg_y': [0, 100, 50]}
        option = Option(name="test_opt", budget_data={
            self.SCOPE_NAME: {'bdg_x': 50, 'bdg_y': 50},
        })

        from fermatrica_rep.options.optim import optimize_budget_local_cobyla

        result = optimize_budget_local_cobyla(
            meta_model=meta_model, scopes=scopes, budget_scope=self.SCOPE_NAME,
            borders_dict=borders_dict, option=option, bdg_size=100,
            epochs=2, iters_epoch=50, ftol_abs=1e-4, verbose=False,
        )

        assert result.budget_data[self.SCOPE_NAME] is not option.budget_data[self.SCOPE_NAME]
        result.budget_data[self.SCOPE_NAME]['bdg_x'] = 999
        assert option.budget_data[self.SCOPE_NAME]['bdg_x'] == 50


# ---------------------------------------------------------------------------
# TestOptimizeBudgetLocalNcma
# ---------------------------------------------------------------------------

class TestOptimizeBudgetLocalNcma:

    SCOPE_NAME = "test_scope"
    _mock_model = _MockNcmaModel()

    def test_returns_option_with_budget_data(self, monkeypatch):
        monkeypatch.setattr(MetaModel, 'compute', _make_mock_compute(self.SCOPE_NAME))
        meta_model = _make_ncma_meta_model(self.SCOPE_NAME, self._mock_model)
        scopes = [OptionScope(
            scope_name=self.SCOPE_NAME,
            date_start="2024-01-01", date_end="2024-12-01",
            ref_date_start="2023-01-01", ref_date_end="2023-12-01",
        )]
        borders_dict = {'bdg_x': [0, 100, 50], 'bdg_y': [0, 100, 50]}
        option = Option(name="test_opt", budget_data={self.SCOPE_NAME: {'bdg_x': 50, 'bdg_y': 50}})

        from fermatrica_rep.options.optim import optimize_budget_local_ncma

        result = optimize_budget_local_ncma(
            meta_model=meta_model, scopes=scopes, budget_scope=self.SCOPE_NAME,
            borders_dict=borders_dict, option=option, bdg_size=100,
            epochs=1, iters_epoch=10, cores=1, verbose=False,
        )

        assert isinstance(result, Option)
        assert self.SCOPE_NAME in result.budget_data

    def test_runs_without_crash(self, monkeypatch):
        monkeypatch.setattr(MetaModel, 'compute', _make_mock_compute(self.SCOPE_NAME))
        meta_model = _make_ncma_meta_model(self.SCOPE_NAME, self._mock_model)
        scopes = [OptionScope(
            scope_name=self.SCOPE_NAME,
            date_start="2024-01-01", date_end="2024-12-01",
            ref_date_start="2023-01-01", ref_date_end="2023-12-01",
        )]
        borders_dict = {'bdg_x': [0, 100, 50], 'bdg_y': [0, 100, 50]}
        option = Option(name="test_opt", budget_data={self.SCOPE_NAME: {'bdg_x': 50, 'bdg_y': 50}})

        from fermatrica_rep.options.optim import optimize_budget_local_ncma

        result = optimize_budget_local_ncma(
            meta_model=meta_model, scopes=scopes, budget_scope=self.SCOPE_NAME,
            borders_dict=borders_dict, option=option, bdg_size=100,
            epochs=1, iters_epoch=10, cores=1, verbose=False,
        )
        assert isinstance(result, Option)
        assert self.SCOPE_NAME in result.budget_data


# ---------------------------------------------------------------------------
# TestOptimizeTargetLocalCobyla
# ---------------------------------------------------------------------------

class TestOptimizeTargetLocalCobyla:

    SCOPE_NAME = "test_scope"

    def test_returns_option_with_budget_data(self, monkeypatch):
        monkeypatch.setattr(
            'fermatrica_rep.options.objective._last_node_name',
            lambda mm: "model",
        )
        meta_model = _make_mock_meta_model(self.SCOPE_NAME)
        scopes = [OptionScope(
            scope_name=self.SCOPE_NAME,
            date_start="2024-01-01", date_end="2024-12-01",
            ref_date_start="2023-01-01", ref_date_end="2023-12-01",
        )]
        borders_dict = {'bdg_x': [0, 100, 50], 'bdg_y': [0, 100, 50]}
        option = Option(name="test_opt", budget_data={self.SCOPE_NAME: {'bdg_x': 50, 'bdg_y': 50}})

        from fermatrica_rep.options.optim import optimize_target_local_cobyla

        result = optimize_target_local_cobyla(
            meta_model=meta_model, scopes=scopes, budget_scope=self.SCOPE_NAME,
            borders_dict=borders_dict, option=option,
            trg=100, epochs=1, iters_epoch=50, verbose=False,
        )
        assert isinstance(result, Option)
        assert self.SCOPE_NAME in result.budget_data
        bdg = result.budget_data[self.SCOPE_NAME]
        assert sum(v for k, v in bdg.items() if k.startswith('bdg_')) >= 0

    def test_achieves_target_kpi(self, monkeypatch):
        monkeypatch.setattr(
            'fermatrica_rep.options.objective._last_node_name',
            lambda mm: "model",
        )
        meta_model = _make_mock_meta_model(self.SCOPE_NAME)
        scopes = [OptionScope(
            scope_name=self.SCOPE_NAME,
            date_start="2024-01-01", date_end="2024-12-01",
            ref_date_start="2023-01-01", ref_date_end="2023-12-01",
        )]
        borders_dict = {'bdg_x': [0, 100, 50], 'bdg_y': [0, 100, 50]}
        option = Option(name="test_opt", budget_data={self.SCOPE_NAME: {'bdg_x': 50, 'bdg_y': 50}})

        from fermatrica_rep.options.optim import optimize_target_local_cobyla

        result = optimize_target_local_cobyla(
            meta_model=meta_model, scopes=scopes, budget_scope=self.SCOPE_NAME,
            borders_dict=borders_dict, option=option,
            trg=100, epochs=1, iters_epoch=50, verbose=False,
        )
        bdg = result.budget_data[self.SCOPE_NAME]
        assert sum(v for k, v in bdg.items() if k.startswith('bdg_')) >= 0

    def test_budget_within_borders(self, monkeypatch):
        monkeypatch.setattr(
            'fermatrica_rep.options.objective._last_node_name',
            lambda mm: "model",
        )
        meta_model = _make_mock_meta_model(self.SCOPE_NAME)
        scopes = [OptionScope(
            scope_name=self.SCOPE_NAME,
            date_start="2024-01-01", date_end="2024-12-01",
            ref_date_start="2023-01-01", ref_date_end="2023-12-01",
        )]
        borders_dict = {'bdg_x': [0, 100, 50], 'bdg_y': [0, 100, 50]}
        option = Option(name="test_opt", budget_data={self.SCOPE_NAME: {'bdg_x': 50, 'bdg_y': 50}})

        from fermatrica_rep.options.optim import optimize_target_local_cobyla

        result = optimize_target_local_cobyla(
            meta_model=meta_model, scopes=scopes, budget_scope=self.SCOPE_NAME,
            borders_dict=borders_dict, option=option,
            trg=100, epochs=1, iters_epoch=50, verbose=False,
        )
        bdg = result.budget_data[self.SCOPE_NAME]
        for k, border in borders_dict.items():
            if k in bdg:
                assert border[0] <= bdg[k] <= border[1], \
                    f"{k}={bdg[k]} outside [{border[0]}, {border[1]}]"


# ---------------------------------------------------------------------------
# TestOptimizeBudgetLocalCobylaMulti
# ---------------------------------------------------------------------------

class TestOptimizeBudgetLocalCobylaMulti:

    SCOPE_NAME = "test_scope"

    def test_returns_dict_keyed_by_bdg_size(self, monkeypatch):
        monkeypatch.setattr(
            'fermatrica_rep.options.objective._last_node_name',
            lambda mm: "model",
        )
        meta_model = _make_mock_meta_model(self.SCOPE_NAME, kpi_multiplier=2.0)
        scopes = [OptionScope(
            scope_name=self.SCOPE_NAME,
            date_start="2024-01-01", date_end="2024-12-01",
            ref_date_start="2023-01-01", ref_date_end="2023-12-01",
        )]
        borders_dict = {'bdg_x': [0, 100, 50], 'bdg_y': [0, 100, 50]}
        option = Option(name="test_opt", budget_data={self.SCOPE_NAME: {'bdg_x': 50, 'bdg_y': 50}})

        from fermatrica_rep.options.optim import optimize_budget_local_cobyla_multi

        result = optimize_budget_local_cobyla_multi(
            meta_model=meta_model, scopes=scopes, budget_scope=self.SCOPE_NAME,
            borders_dict=borders_dict, option=option, bdg_sizes=[100, 200, 300],
            epochs=1, iters_epoch=50, ftol_abs=1e-4, verbose=False, cores=1,
        )

        assert isinstance(result, dict)
        assert set(result.keys()) == {100.0, 200.0, 300.0}
        for k, v in result.items():
            assert isinstance(v, Option)

    def test_single_budget_size_returns_dict_of_one(self, monkeypatch):
        monkeypatch.setattr(
            'fermatrica_rep.options.objective._last_node_name',
            lambda mm: "model",
        )
        meta_model = _make_mock_meta_model(self.SCOPE_NAME, kpi_multiplier=2.0)
        scopes = [OptionScope(
            scope_name=self.SCOPE_NAME,
            date_start="2024-01-01", date_end="2024-12-01",
            ref_date_start="2023-01-01", ref_date_end="2023-12-01",
        )]
        borders_dict = {'bdg_x': [0, 100, 50], 'bdg_y': [0, 100, 50]}
        option = Option(name="test_opt", budget_data={self.SCOPE_NAME: {'bdg_x': 50, 'bdg_y': 50}})

        from fermatrica_rep.options.optim import optimize_budget_local_cobyla_multi

        result = optimize_budget_local_cobyla_multi(
            meta_model=meta_model, scopes=scopes, budget_scope=self.SCOPE_NAME,
            borders_dict=borders_dict, option=option, bdg_sizes=[100],
            epochs=1, iters_epoch=50, ftol_abs=1e-4, verbose=False, cores=1,
        )

        assert isinstance(result, dict)
        assert len(result) == 1
        assert 100.0 in result
        assert isinstance(result[100.0], Option)

    def test_cores_one_sequential(self, monkeypatch):
        monkeypatch.setattr(
            'fermatrica_rep.options.objective._last_node_name',
            lambda mm: "model",
        )
        meta_model = _make_mock_meta_model(self.SCOPE_NAME)
        scopes = [OptionScope(
            scope_name=self.SCOPE_NAME,
            date_start="2024-01-01", date_end="2024-12-01",
            ref_date_start="2023-01-01", ref_date_end="2023-12-01",
        )]
        borders_dict = {'bdg_x': [0, 100, 50], 'bdg_y': [0, 100, 50]}
        option = Option(name="test_opt", budget_data={self.SCOPE_NAME: {'bdg_x': 50, 'bdg_y': 50}})

        from fermatrica_rep.options.optim import optimize_budget_local_cobyla_multi

        result = optimize_budget_local_cobyla_multi(
            meta_model=meta_model, scopes=scopes, budget_scope=self.SCOPE_NAME,
            borders_dict=borders_dict, option=option, bdg_sizes=[100, 200],
            epochs=1, iters_epoch=50, ftol_abs=1e-4, verbose=False, cores=1,
        )

        assert isinstance(result, dict)
        assert len(result) == 2

    def test_error_on_empty_bdg_sizes(self):
        from fermatrica_rep.options.optim import optimize_budget_local_cobyla_multi
        import pytest
        from fermatrica.basics.basics import fermatrica_error

        with pytest.raises(Exception):
            optimize_budget_local_cobyla_multi(
                meta_model=None, scopes=None, budget_scope="x",
                borders_dict={}, option=Option(name="x", budget_data={}),
                bdg_sizes=[], epochs=1, iters_epoch=10, verbose=False,
            )

    def test_error_on_duplicate_bdg_sizes(self):
        from fermatrica_rep.options.optim import optimize_budget_local_cobyla_multi
        import pytest

        with pytest.raises(Exception):
            optimize_budget_local_cobyla_multi(
                meta_model=None, scopes=None, budget_scope="x",
                borders_dict={}, option=Option(name="x", budget_data={}),
                bdg_sizes=[100, 100], epochs=1, iters_epoch=10, verbose=False,
            )

    def test_error_on_dict_option(self):
        from fermatrica_rep.options.optim import optimize_budget_local_cobyla_multi
        import pytest

        with pytest.raises(Exception):
            optimize_budget_local_cobyla_multi(
                meta_model=None, scopes=None, budget_scope="x",
                borders_dict={}, option={}, bdg_sizes=[100],
                epochs=1, iters_epoch=10, verbose=False,
            )

    def test_positive_budgets_within_borders(self, monkeypatch):
        monkeypatch.setattr(
            'fermatrica_rep.options.objective._last_node_name',
            lambda mm: "model",
        )
        meta_model = _make_mock_meta_model(self.SCOPE_NAME, kpi_multiplier=2.0)
        scopes = [OptionScope(
            scope_name=self.SCOPE_NAME,
            date_start="2024-01-01", date_end="2024-12-01",
            ref_date_start="2023-01-01", ref_date_end="2023-12-01",
        )]
        borders_dict = {'bdg_x': [0, 100, 50], 'bdg_y': [0, 100, 50]}
        option = Option(name="test_opt", budget_data={self.SCOPE_NAME: {'bdg_x': 50, 'bdg_y': 50}})

        from fermatrica_rep.options.optim import optimize_budget_local_cobyla_multi

        result = optimize_budget_local_cobyla_multi(
            meta_model=meta_model, scopes=scopes, budget_scope=self.SCOPE_NAME,
            borders_dict=borders_dict, option=option, bdg_sizes=[100, 200, 300],
            epochs=1, iters_epoch=50, ftol_abs=1e-4, verbose=False, cores=2,
        )

        for bdg_size, opt_result in result.items():
            bdg = opt_result.budget_data[self.SCOPE_NAME]
            for k, border in borders_dict.items():
                if k in bdg:
                    assert border[0] <= bdg[k] <= border[1], \
                        f"{k}={bdg[k]} outside [{border[0]}, {border[1]}]"

    def test_budget_varying_produces_different_results(self, monkeypatch):
        monkeypatch.setattr(
            'fermatrica_rep.options.objective._last_node_name',
            lambda mm: "model",
        )
        meta_model = _make_mock_meta_model(self.SCOPE_NAME, kpi_multiplier=2.0)
        scopes = [OptionScope(
            scope_name=self.SCOPE_NAME,
            date_start="2024-01-01", date_end="2024-12-01",
            ref_date_start="2023-01-01", ref_date_end="2023-12-01",
        )]
        borders_dict = {'bdg_x': [0, 100, 50], 'bdg_y': [0, 100, 50]}
        option = Option(name="test_opt", budget_data={self.SCOPE_NAME: {'bdg_x': 50, 'bdg_y': 50}})

        from fermatrica_rep.options.optim import optimize_budget_local_cobyla_multi

        result = optimize_budget_local_cobyla_multi(
            meta_model=meta_model, scopes=scopes, budget_scope=self.SCOPE_NAME,
            borders_dict=borders_dict, option=option, bdg_sizes=[50, 200],
            epochs=1, iters_epoch=50, ftol_abs=1e-4, verbose=False, cores=1,
        )

        bdg_50 = result[50.0].budget_data[self.SCOPE_NAME]
        bdg_200 = result[200.0].budget_data[self.SCOPE_NAME]
        sums_50 = sum(v for k, v in bdg_50.items() if k.startswith('bdg_'))
        sums_200 = sum(v for k, v in bdg_200.items() if k.startswith('bdg_'))
        assert sums_200 > sums_50, "Larger budget should have larger total spend"


# ---------------------------------------------------------------------------
# TestOptimizeTargetLocalCobylaMulti
# ---------------------------------------------------------------------------

class TestOptimizeTargetLocalCobylaMulti:

    SCOPE_NAME = "test_scope"

    def test_returns_dict_keyed_by_trg(self, monkeypatch):
        monkeypatch.setattr(
            'fermatrica_rep.options.objective._last_node_name',
            lambda mm: "model",
        )
        meta_model = _make_mock_meta_model(self.SCOPE_NAME, kpi_multiplier=2.0)
        scopes = [OptionScope(
            scope_name=self.SCOPE_NAME,
            date_start="2024-01-01", date_end="2024-12-01",
            ref_date_start="2023-01-01", ref_date_end="2023-12-01",
        )]
        borders_dict = {'bdg_x': [0, 100, 50], 'bdg_y': [0, 100, 50]}
        option = Option(name="test_opt", budget_data={self.SCOPE_NAME: {'bdg_x': 50, 'bdg_y': 50}})

        from fermatrica_rep.options.optim import optimize_target_local_cobyla_multi

        result = optimize_target_local_cobyla_multi(
            meta_model=meta_model, scopes=scopes, budget_scope=self.SCOPE_NAME,
            borders_dict=borders_dict, option=option, trg_list=[50, 100, 200],
            epochs=1, iters_epoch=50, verbose=False, cores=1,
        )

        assert isinstance(result, dict)
        assert set(result.keys()) == {50.0, 100.0, 200.0}
        for k, v in result.items():
            assert isinstance(v, Option)

    def test_single_target_returns_dict_of_one(self, monkeypatch):
        monkeypatch.setattr(
            'fermatrica_rep.options.objective._last_node_name',
            lambda mm: "model",
        )
        meta_model = _make_mock_meta_model(self.SCOPE_NAME)
        scopes = [OptionScope(
            scope_name=self.SCOPE_NAME,
            date_start="2024-01-01", date_end="2024-12-01",
            ref_date_start="2023-01-01", ref_date_end="2023-12-01",
        )]
        borders_dict = {'bdg_x': [0, 100, 50], 'bdg_y': [0, 100, 50]}
        option = Option(name="test_opt", budget_data={self.SCOPE_NAME: {'bdg_x': 50, 'bdg_y': 50}})

        from fermatrica_rep.options.optim import optimize_target_local_cobyla_multi

        result = optimize_target_local_cobyla_multi(
            meta_model=meta_model, scopes=scopes, budget_scope=self.SCOPE_NAME,
            borders_dict=borders_dict, option=option, trg_list=[100],
            epochs=1, iters_epoch=50, verbose=False, cores=1,
        )

        assert isinstance(result, dict)
        assert len(result) == 1
        assert 100.0 in result

    def test_cores_one_sequential(self, monkeypatch):
        monkeypatch.setattr(
            'fermatrica_rep.options.objective._last_node_name',
            lambda mm: "model",
        )
        meta_model = _make_mock_meta_model(self.SCOPE_NAME)
        scopes = [OptionScope(
            scope_name=self.SCOPE_NAME,
            date_start="2024-01-01", date_end="2024-12-01",
            ref_date_start="2023-01-01", ref_date_end="2023-12-01",
        )]
        borders_dict = {'bdg_x': [0, 100, 50], 'bdg_y': [0, 100, 50]}
        option = Option(name="test_opt", budget_data={self.SCOPE_NAME: {'bdg_x': 50, 'bdg_y': 50}})

        from fermatrica_rep.options.optim import optimize_target_local_cobyla_multi

        result = optimize_target_local_cobyla_multi(
            meta_model=meta_model, scopes=scopes, budget_scope=self.SCOPE_NAME,
            borders_dict=borders_dict, option=option, trg_list=[50, 100],
            epochs=1, iters_epoch=50, verbose=False, cores=1,
        )

        assert isinstance(result, dict)
        assert len(result) == 2

    def test_error_on_empty_trg_list(self):
        from fermatrica_rep.options.optim import optimize_target_local_cobyla_multi
        import pytest

        with pytest.raises(Exception):
            optimize_target_local_cobyla_multi(
                meta_model=None, scopes=None, budget_scope="x",
                borders_dict={}, option=Option(name="x", budget_data={}),
                trg_list=[], epochs=1, iters_epoch=10, verbose=False,
            )

    def test_error_on_duplicate_trg_list(self):
        from fermatrica_rep.options.optim import optimize_target_local_cobyla_multi
        import pytest

        with pytest.raises(Exception):
            optimize_target_local_cobyla_multi(
                meta_model=None, scopes=None, budget_scope="x",
                borders_dict={}, option=Option(name="x", budget_data={}),
                trg_list=[100, 100], epochs=1, iters_epoch=10, verbose=False,
            )

    def test_error_on_dict_option(self):
        from fermatrica_rep.options.optim import optimize_target_local_cobyla_multi
        import pytest

        with pytest.raises(Exception):
            optimize_target_local_cobyla_multi(
                meta_model=None, scopes=None, budget_scope="x",
                borders_dict={}, option={}, trg_list=[100],
                epochs=1, iters_epoch=10, verbose=False,
            )

    def test_budgets_within_borders(self, monkeypatch):
        monkeypatch.setattr(
            'fermatrica_rep.options.objective._last_node_name',
            lambda mm: "model",
        )
        meta_model = _make_mock_meta_model(self.SCOPE_NAME, kpi_multiplier=2.0)
        scopes = [OptionScope(
            scope_name=self.SCOPE_NAME,
            date_start="2024-01-01", date_end="2024-12-01",
            ref_date_start="2023-01-01", ref_date_end="2023-12-01",
        )]
        borders_dict = {'bdg_x': [0, 100, 50], 'bdg_y': [0, 100, 50]}
        option = Option(name="test_opt", budget_data={self.SCOPE_NAME: {'bdg_x': 50, 'bdg_y': 50}})

        from fermatrica_rep.options.optim import optimize_target_local_cobyla_multi

        result = optimize_target_local_cobyla_multi(
            meta_model=meta_model, scopes=scopes, budget_scope=self.SCOPE_NAME,
            borders_dict=borders_dict, option=option, trg_list=[50, 100],
            epochs=1, iters_epoch=50, verbose=False, cores=2,
        )

        for trg, opt_result in result.items():
            bdg = opt_result.budget_data[self.SCOPE_NAME]
            for k, border in borders_dict.items():
                if k in bdg:
                    assert border[0] <= bdg[k] <= border[1], \
                        f"{k}={bdg[k]} outside [{border[0]}, {border[1]}]"


# ---------------------------------------------------------------------------
# TestOptimizeTargetLocalNcma
# ---------------------------------------------------------------------------

class TestOptimizeTargetLocalNcma:

    SCOPE_NAME = "test_scope"
    _mock_model = _MockNcmaModel()

    def test_returns_option_with_budget_data(self, monkeypatch):
        monkeypatch.setattr(MetaModel, 'compute', _make_mock_compute(self.SCOPE_NAME))
        meta_model = _make_ncma_meta_model(self.SCOPE_NAME, self._mock_model)
        scopes = [OptionScope(
            scope_name=self.SCOPE_NAME,
            date_start="2024-01-01", date_end="2024-12-01",
            ref_date_start="2023-01-01", ref_date_end="2023-12-01",
        )]
        borders_dict = {'bdg_x': [0, 100, 50], 'bdg_y': [0, 100, 50]}
        option = Option(name="test_opt", budget_data={self.SCOPE_NAME: {'bdg_x': 50, 'bdg_y': 50}})

        from fermatrica_rep.options.optim import optimize_target_local_ncma

        result = optimize_target_local_ncma(
            meta_model=meta_model, scopes=scopes, budget_scope=self.SCOPE_NAME,
            borders_dict=borders_dict, option=option,
            trg=100, epochs=1, iters_epoch=10, cores=1, verbose=False,
        )
        assert isinstance(result, Option)
        assert self.SCOPE_NAME in result.budget_data

    def test_bdg_size_zero_does_not_crash(self, monkeypatch):
        monkeypatch.setattr(MetaModel, 'compute', _make_mock_compute(self.SCOPE_NAME))
        meta_model = _make_ncma_meta_model(self.SCOPE_NAME, self._mock_model)
        scopes = [OptionScope(
            scope_name=self.SCOPE_NAME,
            date_start="2024-01-01", date_end="2024-03-01",
            ref_date_start="2024-01-01", ref_date_end="2024-03-01",
        )]
        borders_dict = {'bdg_x': [0, 100, 0], 'bdg_y': [0, 100, 0]}
        option = Option(name="test_opt", budget_data={self.SCOPE_NAME: {'bdg_x': 0, 'bdg_y': 0}})

        from fermatrica_rep.options.optim import optimize_target_local_ncma

        result = optimize_target_local_ncma(
            meta_model=meta_model, scopes=scopes, budget_scope=self.SCOPE_NAME,
            borders_dict=borders_dict, option=option,
            trg=100, epochs=1, iters_epoch=10, cores=1, verbose=False,
        )
        assert isinstance(result, Option)


# ---------------------------------------------------------------------------
# TestPrepareBorders
# ---------------------------------------------------------------------------

class TestPrepareBorders:

    def test_basic_split(self):
        from fermatrica_rep.options.optim import _prepare_borders
        borders_dict = {'bdg_x': [0, 100, 50], 'bdg_y': [0, 100, 50], 'bdg_fixed': [30, 30, 30]}
        borders, borders_fixed, adj_bdg = _prepare_borders(borders_dict, bdg_size=200)
        assert list(borders.columns) == ['bdg_x', 'bdg_y']
        assert list(borders_fixed.columns) == ['bdg_fixed']
        assert adj_bdg == 170.0

    def test_all_variable_no_fixed(self):
        from fermatrica_rep.options.optim import _prepare_borders
        borders_dict = {'bdg_x': [0, 100, 50], 'bdg_y': [10, 200, 60]}
        borders, borders_fixed, adj_bdg = _prepare_borders(borders_dict, bdg_size=200)
        assert list(borders.columns) == ['bdg_x', 'bdg_y']
        assert borders_fixed.empty
        assert adj_bdg == 200.0

    def test_all_fixed_no_variable(self):
        from fermatrica_rep.options.optim import _prepare_borders
        borders_dict = {'bdg_fixed': [30, 30, 30]}
        borders, borders_fixed, adj_bdg = _prepare_borders(borders_dict, bdg_size=200)
        assert borders.empty or len(borders.columns) == 0
        assert list(borders_fixed.columns) == ['bdg_fixed']
        assert adj_bdg == 170.0

    def test_bdg_size_larger_than_start_sum(self):
        from fermatrica_rep.options.optim import _prepare_borders
        borders_dict = {'bdg_x': [0, 100, 50], 'bdg_y': [0, 100, 50]}
        borders, borders_fixed, adj_bdg = _prepare_borders(borders_dict, bdg_size=200)
        assert list(borders.columns) == ['bdg_x', 'bdg_y']
        assert np.isclose(borders.iloc[2]['bdg_x'], 50)
        assert np.isclose(borders.iloc[2]['bdg_y'], 50)
        assert adj_bdg == 200.0

    def test_bdg_size_smaller_than_start_sum(self):
        from fermatrica_rep.options.optim import _prepare_borders
        borders_dict = {'bdg_x': [0, 100, 50], 'bdg_y': [0, 100, 50]}
        borders, borders_fixed, adj_bdg = _prepare_borders(borders_dict, bdg_size=50)
        assert np.isclose(borders.iloc[2]['bdg_x'], 50 * 50 / 100 * 0.9)
        assert np.isclose(borders.iloc[2]['bdg_y'], 50 * 50 / 100 * 0.9)
        assert adj_bdg == 50.0

    def test_start_below_lower_bound_clamped(self):
        from fermatrica_rep.options.optim import _prepare_borders
        borders_dict = {'bdg_x': [10, 100, 5], 'bdg_y': [0, 100, 50]}
        borders, borders_fixed, adj_bdg = _prepare_borders(borders_dict, bdg_size=50)
        assert borders.iloc[2]['bdg_x'] >= 10

    def test_bdg_size_none_target_path(self):
        from fermatrica_rep.options.optim import _prepare_borders
        borders_dict = {'bdg_x': [0, 100, 50], 'bdg_y': [0, 100, 50]}
        borders, borders_fixed, adj_bdg = _prepare_borders(borders_dict, bdg_size=None)
        assert list(borders.columns) == ['bdg_x', 'bdg_y']
        assert adj_bdg is None


# ---------------------------------------------------------------------------
# TestComputeBaseline
# ---------------------------------------------------------------------------

class TestComputeBaseline:

    CAT_SCOPE = "cat_2025"
    SALES_SCOPE = "sales_2025"

    def test_computes_baseline_for_multi_node(self):
        from fermatrica_rep.options.optim import _compute_baseline

        meta_model = _make_mock_meta_model_multi(self.CAT_SCOPE, self.SALES_SCOPE,
                                                  cat_kpi=1.0, sales_kpi=2.0)
        meta_model.nodes["category"].ds = pd.DataFrame({'x': [1]})
        meta_model.nodes["sales"].ds = pd.DataFrame({'y': [2]})
        option_budget_data = {
            self.CAT_SCOPE: {'bdg_x': 50},
            self.SALES_SCOPE: {'bdg_y': 75},
        }
        summary_nodes = {
            "category": (0.3, self.CAT_SCOPE),
            "sales": (0.7, self.SALES_SCOPE),
        }
        scopes = [OptionScope(
            scope_name=self.CAT_SCOPE,
            date_start="2024-01-01", date_end="2024-03-01",
            ref_date_start="2024-01-01", ref_date_end="2024-03-01",
        )]
        baseline = _compute_baseline(
            meta_model, option_budget_data, self.CAT_SCOPE, scopes,
            if_exact=True, if_long=False, if_volume=True, summary_nodes=summary_nodes,
        )
        assert "category" in baseline
        assert "sales" in baseline
        assert np.isclose(baseline["category"], 50 * 1.0)
        assert np.isclose(baseline["sales"], 75 * 2.0)

    def test_restores_ds_after_compute(self):
        from fermatrica_rep.options.optim import _compute_baseline
        from unittest.mock import MagicMock

        mm = _make_mock_meta_model_multi(self.CAT_SCOPE, self.SALES_SCOPE,
                                          cat_kpi=1.0, sales_kpi=2.0)
        orig_ds_cat = pd.DataFrame({'x': [1]})
        orig_ds_sales = pd.DataFrame({'y': [2]})
        mm.nodes["category"].ds = orig_ds_cat
        mm.nodes["sales"].ds = orig_ds_sales
        option_budget_data = {
            self.CAT_SCOPE: {'bdg_x': 50},
            self.SALES_SCOPE: {'bdg_y': 75},
        }
        summary_nodes = {
            "category": (0.3, self.CAT_SCOPE),
            "sales": (0.7, self.SALES_SCOPE),
        }
        scopes = [OptionScope(
            scope_name=self.CAT_SCOPE,
            date_start="2024-01-01", date_end="2024-03-01",
            ref_date_start="2024-01-01", ref_date_end="2024-03-01",
        )]
        _compute_baseline(
            mm, option_budget_data, self.CAT_SCOPE, scopes,
            if_exact=True, if_long=False, if_volume=True, summary_nodes=summary_nodes,
        )
        assert mm.nodes["category"].ds.equals(orig_ds_cat)

    def test_raises_on_zero_baseline(self):
        from fermatrica_rep.options.optim import _compute_baseline

        mm = _make_mock_meta_model_multi(self.CAT_SCOPE, self.SALES_SCOPE,
                                          cat_kpi=0.0, sales_kpi=2.0)

        class _MockComputeZeroCat:
            def __call__(self_, option, scopes=None, **kwargs):
                return {
                    "category": {self.CAT_SCOPE: {"pred_exact_vol": 0.0}},
                    "sales": {self.SALES_SCOPE: {"pred_exact_vol": 200.0}},
                }
        mm.compute = _MockComputeZeroCat()

        option_budget_data = {
            self.CAT_SCOPE: {'bdg_x': 50},
            self.SALES_SCOPE: {'bdg_y': 75},
        }
        summary_nodes = {
            "category": (0.3, self.CAT_SCOPE),
            "sales": (0.7, self.SALES_SCOPE),
        }
        scopes = [OptionScope(
            scope_name=self.CAT_SCOPE,
            date_start="2024-01-01", date_end="2024-03-01",
            ref_date_start="2024-01-01", ref_date_end="2024-03-01",
        )]
        with pytest.raises(Exception):
            _compute_baseline(
                mm, option_budget_data, self.CAT_SCOPE, scopes,
                if_exact=True, if_long=False, if_volume=True, summary_nodes=summary_nodes,
            )


# ---------------------------------------------------------------------------
# TestImplEdgeCases
# ---------------------------------------------------------------------------

class TestImplEdgeCases:

    SCOPE_NAME = "test_scope"

    def test_missing_budget_scope_raises(self):
        from fermatrica_rep.options.optim import _optimize_budget_local_cobyla_impl

        meta_model = _make_mock_meta_model(self.SCOPE_NAME)
        scopes = [OptionScope(
            scope_name=self.SCOPE_NAME,
            date_start="2024-01-01", date_end="2024-12-01",
            ref_date_start="2023-01-01", ref_date_end="2023-12-01",
        )]
        option = Option(name="test_opt", budget_data={"other_scope": {'bdg_x': 50}})
        borders_dict = {'bdg_x': [0, 100, 50]}

        with pytest.raises(Exception):
            _optimize_budget_local_cobyla_impl(
                meta_model=meta_model, scopes=scopes, budget_scope="nonexistent",
                borders_dict=borders_dict, option=option, bdg_size=100,
                epochs=1, iters_epoch=10, ftol_abs=1e-2, verbose=False,
            )

    def test_if_long_defaults_when_none(self, monkeypatch):
        from fermatrica_rep.options.optim import _optimize_budget_local_cobyla_impl
        from fermatrica_rep.options.objective import ObjectiveFunctionBudget

        meta_model = _make_mock_meta_model(self.SCOPE_NAME)
        scopes = [OptionScope(
            scope_name=self.SCOPE_NAME,
            date_start="2024-01-01", date_end="2024-12-01",
            ref_date_start="2023-01-01", ref_date_end="2023-12-01",
        )]
        option = Option(name="test_opt", budget_data={self.SCOPE_NAME: {'bdg_x': 50}})
        borders_dict = {'bdg_x': [0, 100, 50]}

        monkeypatch.setattr(
            'fermatrica_rep.options.objective._last_node_name',
            lambda mm: "model",
        )
        monkeypatch.setattr(
            'fermatrica_rep.options.optim._compute_baseline',
            lambda *a, **kw: None,
        )

        try:
            _optimize_budget_local_cobyla_impl(
                meta_model=meta_model, scopes=scopes, budget_scope=self.SCOPE_NAME,
                borders_dict=borders_dict, option=option, bdg_size=100,
                if_exact=False, if_long=None, if_volume=True,
                epochs=1, iters_epoch=10, ftol_abs=1e-2, verbose=False,
            )
        except Exception:
            pass

    def test_returns_option_with_text_flags(self, monkeypatch):
        from fermatrica_rep.options.optim import _optimize_budget_local_cobyla_impl

        monkeypatch.setattr(
            'fermatrica_rep.options.objective._last_node_name',
            lambda mm: "model",
        )
        meta_model = _make_mock_meta_model(self.SCOPE_NAME)
        scopes = [OptionScope(
            scope_name=self.SCOPE_NAME,
            date_start="2024-01-01", date_end="2024-12-01",
            ref_date_start="2023-01-01", ref_date_end="2023-12-01",
        )]
        text_flags = {"note": "test note"}
        option = Option(name="test_opt", budget_data={self.SCOPE_NAME: {'bdg_x': 50}},
                        text_flags=text_flags)
        borders_dict = {'bdg_x': [0, 100, 50]}

        result = _optimize_budget_local_cobyla_impl(
            meta_model=meta_model, scopes=scopes, budget_scope=self.SCOPE_NAME,
            borders_dict=borders_dict, option=option, bdg_size=100,
            epochs=1, iters_epoch=10, ftol_abs=1e-2, verbose=False,
        )
        assert result.text_flags == text_flags


# ---------------------------------------------------------------------------
# TestCobylaWorkers
# ---------------------------------------------------------------------------

class TestCobylaWorkers:

    SCOPE_NAME = "test_scope"

    def test_budget_worker_toggles_warn_nan_seasonality(self, monkeypatch):
        from fermatrica_rep.options.optim import _cobyla_budget_worker
        import fermatrica_rep.options.translators as _translators_mod

        original = _translators_mod.WARN_NAN_SEASONALITY

        def mock_impl(**kw):
            return Option(name="test", budget_data={self.SCOPE_NAME: {'bdg_x': 50}})

        monkeypatch.setattr(
            'fermatrica_rep.options.optim._optimize_budget_local_cobyla_impl',
            lambda **kw: mock_impl(**kw),
        )
        option = Option(name="test_opt", budget_data={self.SCOPE_NAME: {'bdg_x': 50}})
        borders_dict = {'bdg_x': [0, 100, 50]}

        _cobyla_budget_worker(
            meta_model=None, scopes=None, budget_scope=self.SCOPE_NAME,
            borders_dict=borders_dict, option=option, bdg_size=100,
            if_exact=True, if_long=False, if_volume=True,
            epochs=1, iters_epoch=10, error_score=1e12, ftol_abs=0.01,
            xtol_abs=None, verbose=False, warn_nan_seasonality=False,
        )
        assert _translators_mod.WARN_NAN_SEASONALITY == original

    def test_target_worker_toggles_warn_nan_seasonality(self, monkeypatch):
        from fermatrica_rep.options.optim import _cobyla_target_worker
        import fermatrica_rep.options.translators as _translators_mod

        original = _translators_mod.WARN_NAN_SEASONALITY

        def mock_impl(**kw):
            return Option(name="test", budget_data={self.SCOPE_NAME: {'bdg_x': 50}})

        monkeypatch.setattr(
            'fermatrica_rep.options.optim._optimize_target_local_cobyla_impl',
            lambda **kw: mock_impl(**kw),
        )
        option = Option(name="test_opt", budget_data={self.SCOPE_NAME: {'bdg_x': 50}})
        borders_dict = {'bdg_x': [0, 100, 50]}

        _cobyla_target_worker(
            meta_model=None, scopes=None, budget_scope=self.SCOPE_NAME,
            borders_dict=borders_dict, option=option, trg=100, trg_tol_ratio=0.01,
            if_exact=True, if_long=False, if_volume=True,
            epochs=1, iters_epoch=10, error_score=1e12, ftol_abs=0.01,
            xtol_abs=None, verbose=False, warn_nan_seasonality=False,
        )
        assert _translators_mod.WARN_NAN_SEASONALITY == original


# ---------------------------------------------------------------------------
# Deprecated API tests
# ---------------------------------------------------------------------------

class TestDeprecatedApiBudgetCobyla:

    def test_old_args_still_work(self, monkeypatch):
        from fermatrica_rep.options.define import OptionSettings

        meta_model = _make_mock_meta_model("brand_x")
        scopes = [OptionScope(
            scope_name="brand_x",
            date_start="2024-01-01", date_end="2024-12-01",
            ref_date_start="2023-01-01", ref_date_end="2023-12-01",
        )]
        ds = pd.DataFrame({'date': pd.date_range('2024-01-01', periods=12, freq='MS'),
                           'superbrand': ['brand_x'] * 12, 'listed': [2] * 12,
                           'y': range(12)})
        opt_set = OptionSettings(
            target='brand_x', date_start='2024-01-01', date_end='2024-12-01',
            ref_date_start='2023-01-01', ref_date_end='2023-12-01',
        )
        borders_dict = {'bdg_x': [0, 100, 50], 'bdg_y': [0, 100, 50]}
        option = {'bdg_x': 50, 'bdg_y': 50}

        from fermatrica_rep.options.optim import optimize_budget_local_cobyla

        monkeypatch.setattr(
            'fermatrica_rep.options.optim._optimize_budget_local_cobyla_impl',
            lambda meta_model, scopes, budget_scope, borders_dict, option,
                   bdg_size, if_exact, if_long, if_volume, epochs,
                   iters_epoch, error_score, ftol_abs, xtol_abs, verbose,
                   **kwargs:
            Option(name=option.name, budget_data=option.budget_data)
        )

        with pytest.warns(DeprecationWarning):
            result = optimize_budget_local_cobyla(
                ds=ds, model=_MockNcmaModel(), model_rep=None, borders_dict=borders_dict,
                option=option, opt_set=opt_set, bdg_size=100,
                epochs=1, iters_epoch=10, ftol_abs=1e-2, verbose=False,
            )
        assert isinstance(result, dict)
        assert 'bdg' in result


class TestDeprecatedApiBudgetNcma:

    def test_old_args_still_work(self, monkeypatch):
        from fermatrica_rep.options.define import OptionSettings

        ds = pd.DataFrame({'date': pd.date_range('2024-01-01', periods=12, freq='MS'),
                           'superbrand': ['brand_x'] * 12, 'listed': [2] * 12,
                           'y': range(12)})
        opt_set = OptionSettings(
            target='brand_x', date_start='2024-01-01', date_end='2024-12-01',
            ref_date_start='2023-01-01', ref_date_end='2023-12-01',
        )
        borders_dict = {'bdg_x': [0, 100, 50], 'bdg_y': [0, 100, 50]}
        option = {'bdg_x': 50, 'bdg_y': 50}

        from fermatrica_rep.options.optim import optimize_budget_local_ncma

        monkeypatch.setattr(
            'fermatrica_rep.options.optim._optimize_budget_local_ncma_impl',
            lambda meta_model, scopes, budget_scope, borders_dict, option,
                   bdg_size, if_exact, if_long, if_volume, epochs,
                   iters_epoch, error_score, cores, verbose,
                   **kwargs:
            Option(name=option.name, budget_data=option.budget_data)
        )

        with pytest.warns(DeprecationWarning):
            result = optimize_budget_local_ncma(
                ds=ds, model=_MockNcmaModel(), model_rep=None,
                borders_dict=borders_dict, option=option, opt_set=opt_set,
                bdg_size=100, epochs=1, iters_epoch=10, cores=1, verbose=False,
            )
        assert isinstance(result, dict)
        assert 'bdg' in result


class TestDeprecatedApiTargetCobyla:

    def test_old_args_still_work(self, monkeypatch):
        from fermatrica_rep.options.define import OptionSettings

        ds = pd.DataFrame({'date': pd.date_range('2024-01-01', periods=12, freq='MS'),
                           'superbrand': ['brand_x'] * 12, 'listed': [2] * 12,
                           'y': range(12)})
        opt_set = OptionSettings(
            target='brand_x', date_start='2024-01-01', date_end='2024-12-01',
            ref_date_start='2023-01-01', ref_date_end='2023-12-01',
        )
        borders_dict = {'bdg_x': [0, 100, 50], 'bdg_y': [0, 100, 50]}
        option = {'bdg_x': 50, 'bdg_y': 50}

        from fermatrica_rep.options.optim import optimize_target_local_cobyla

        monkeypatch.setattr(
            'fermatrica_rep.options.optim._optimize_target_local_cobyla_impl',
            lambda meta_model, scopes, budget_scope, borders_dict, option,
                   trg, trg_tol_ratio, if_exact, if_long, if_volume,
                   epochs, iters_epoch, error_score, ftol_abs,
                   xtol_abs, verbose,
                   **kwargs:
            Option(name=option.name, budget_data=option.budget_data)
        )

        with pytest.warns(DeprecationWarning):
            result = optimize_target_local_cobyla(
                ds=ds, model=_MockNcmaModel(), model_rep=None, borders_dict=borders_dict,
                option=option, opt_set=opt_set,
                trg=100, epochs=1, iters_epoch=10, verbose=False,
            )
        assert isinstance(result, dict)
        assert 'bdg' in result


class TestDeprecatedApiTargetNcma:

    def test_old_args_still_work(self, monkeypatch):
        from fermatrica_rep.options.define import OptionSettings

        ds = pd.DataFrame({'date': pd.date_range('2024-01-01', periods=12, freq='MS'),
                           'superbrand': ['brand_x'] * 12, 'listed': [2] * 12,
                           'y': range(12)})
        opt_set = OptionSettings(
            target='brand_x', date_start='2024-01-01', date_end='2024-12-01',
            ref_date_start='2023-01-01', ref_date_end='2023-12-01',
        )
        borders_dict = {'bdg_x': [0, 100, 50], 'bdg_y': [0, 100, 50]}
        option = {'bdg_x': 50, 'bdg_y': 50}

        from fermatrica_rep.options.optim import optimize_target_local_ncma

        monkeypatch.setattr(
            'fermatrica_rep.options.optim._optimize_target_local_ncma_impl',
            lambda meta_model, scopes, budget_scope, borders_dict, option,
                   trg, trg_tol_ratio, if_exact, if_long, if_volume,
                   epochs, iters_epoch, error_score, cores, verbose,
                   **kwargs:
            Option(name=option.name, budget_data=option.budget_data)
        )

        with pytest.warns(DeprecationWarning):
            result = optimize_target_local_ncma(
                ds=ds, model=_MockNcmaModel(), model_rep=None,
                borders_dict=borders_dict, option=option, opt_set=opt_set,
                trg=100, epochs=1, iters_epoch=10, cores=1, verbose=False,
            )
        assert isinstance(result, dict)
        assert 'bdg' in result
