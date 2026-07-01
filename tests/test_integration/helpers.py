"""Helper functions for golden-value assertions in integration tests.

Provides compare_plot_golden() for verifying plot trace data against
recorded expected values, and parametrize_plot_golden() for loading
golden-value YAML files.

This is NOT a model-fit assessment — it compares predicted traces against
pre-recorded expected values (regression testing for plot stability).
"""
import os
import yaml
import numpy as np
import plotly.graph_objects as go


def compare_plot_golden(fig, expected, rtol=1e-5, atol=1e-8):
    """Assert plotly Figure matches golden expected values.

    Parameters
    ----------
    fig : go.Figure
        Actual plotly figure from the function under test.
    expected : dict
        Expected values with keys: n_traces, plus trace-level
        assertions (y array or y_sum per named trace).
    rtol : float
        Relative tolerance for numerical comparisons (default 1e-5).
    atol : float
        Absolute tolerance for numerical comparisons (default 1e-8).
    """
    assert isinstance(fig, go.Figure), "Expected go.Figure"

    if 'n_traces' in expected:
        assert len(fig.data) == expected['n_traces'], (
            f"Expected {expected['n_traces']} traces, got {len(fig.data)}"
        )

    if 'n_periods_min' in expected:
        periods = set()
        for t in fig.data:
            if hasattr(t, 'x') and t.x is not None:
                periods.update(t.x)
        assert len(periods) >= expected['n_periods_min'], (
            f"Expected >= {expected['n_periods_min']} periods, got {len(periods)}"
        )

    if 'traces' in expected:
        for exp_trace in expected['traces']:
            name = exp_trace.get('name')
            matching = [t for t in fig.data if t.name == name]
            if not matching:
                raise AssertionError(f"No trace found with name '{name}'")
            t = matching[0]

            if 'y' in exp_trace:
                actual_y = np.array(t.y)
                expected_y = np.array(exp_trace['y'])
                np.testing.assert_allclose(actual_y, expected_y, rtol=rtol, atol=atol,
                                           err_msg=f"y values for trace '{name}'")
            if 'y_sum' in exp_trace:
                np.testing.assert_allclose(sum(t.y), exp_trace['y_sum'], rtol=rtol,
                                           err_msg=f"y sum for trace '{name}'")


def parametrize_plot_golden(file_name):
    """Load golden-value YAML and return pytest parametrization tuples.

    Mirrors parametrize_stability_params from test_budget_ssn_create.py.

    Returns list of (description, model_tag, sample, function, params, expected)
    """
    pth = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', 'test_cases', file_name)
    if not os.path.exists(pth):
        return []

    params = []
    with open(pth, mode='rt', encoding='utf-8') as yaml_io:
        yaml_data = yaml.safe_load(yaml_io)
        for item in yaml_data:
            params.append((
                item.get('description', ''),
                item.get('model_tag', ''),
                item.get('sample', ''),
                item.get('function', ''),
                item.get('params', {}),
                item.get('expected', {}),
            ))
    return params
