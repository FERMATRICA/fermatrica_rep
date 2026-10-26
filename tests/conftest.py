import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fermatrica_rep.meta_model.model_rep import ModelRep


def _read_config_samples_path() -> str | None:
    """Read ``[paths].samples`` from repo-root ``config.toml`` if present."""
    import tomllib

    cfg_path = Path(__file__).resolve().parent.parent / "config.toml"
    if not cfg_path.exists():
        return None
    try:
        with cfg_path.open("rb") as fh:
            cfg = tomllib.load(fh)
        return cfg.get("paths", {}).get("samples")
    except (tomllib.TOMLDecodeError, OSError) as exc:
        logging.warning("Ignoring malformed config.toml (%s)", exc)
        return None


def pytest_addoption(parser):
    parser.addoption(
        "--samples-path", action="store", default=None,
        help="Path to a directory containing sample model projects (e.g. C:/path/to/samples)"
    )
    parser.addoption(
        "--if-debug", action="store_true", default=False,
        help="Enable debug mode: enter breakpoint on assertion failures"
    )


@pytest.fixture(scope="session")
def samples_path(request):
    """Path to a directory containing sample model projects.

    Resolved from (in order of priority):
    1. ``--samples-path`` CLI flag (e.g. ``pytest --samples-path=...``)
    2. ``tests/config.toml`` ``[paths].samples``
    3. Skipped if none of the above provide a valid path
    """
    path = request.config.getoption("--samples-path")
    if path is None:
        path = _read_config_samples_path()
    if path is None:
        pytest.skip("--samples-path not provided; "
                     "create config.toml with [paths].samples")
    p = Path(path)
    if not p.exists():
        pytest.skip(f"Samples path does not exist: {path}")
    if not (p / "p00_sample" / "SPEC.md").exists():
        pytest.exit(f"{path} does not appear to be the fermatrica samples directory "
                     f"(missing p00_sample/SPEC.md)")
    return p


@pytest.fixture
def if_debug(request):
    # set True to explore specific case (specific test class)
    # set False to run all tests
    return request.config.getoption("--if-debug", default=False)


@pytest.fixture(scope="session")
def fixtures_path():
    """Path to static test fixtures shipped with the repository."""
    return Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="session")
def options_xlsx_path(fixtures_path):
    """Path to the options.xlsx fixture file."""
    return str(fixtures_path / "options.xlsx")


@pytest.fixture(scope="session")
def seasonality_xlsx_path(fixtures_path):
    """Path to the seasonality.xlsx fixture file."""
    return str(fixtures_path / "seasonality.xlsx")


@pytest.fixture
def split_m_m():
    """Synthetic decomposition data (12 months, one brand, 4 variables).

    Useful for waterfall and decomposition tests.
    """
    np.random.seed(42)
    n = 12
    return pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=n, freq='MS'),
        'superbrand': ['brand_1'] * n,
        'bs_key': [1] * n,
        'listed': [2] * n,
        'variable': (['tv'] * 3 + ['digital'] * 3 + ['pr'] * 3 + ['base'] * 3),
        'value': np.random.uniform(-10, 20, n)
    })


@pytest.fixture
def basic_model_rep(split_m_m):
    """ModelRep built on the shared split_m_m fixture."""
    return ModelRep(split_m_m, language='english')