import sys
from pathlib import Path

import pytest


def _find_latest_model(model_dir: Path) -> str:
    """Find the latest model_* directory and return its path as string."""
    dirs = [d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith("model_")]
    if not dirs:
        pytest.exit(f"No model_* directories found in {model_dir}")
    dirs.sort(key=lambda d: d.name)
    return str(dirs[-1])


def _activate_adhoc_path(root: Path):
    """Evict cached adhoc modules and insert sample root onto sys.path."""
    keys = [k for k in list(sys.modules) if k.startswith("code_py")]
    for k in keys:
        del sys.modules[k]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import code_py.adhoc.model  # noqa: F401


def _deactivate_adhoc_path(root: Path):
    """Remove sample root from sys.path."""
    if str(root) in sys.path:
        sys.path.remove(str(root))


P00_MODEL_TAG = "2024-11-19_174649"
P01_MODEL_TAG = "2026-06-18_124832"
P02_MODEL_TAG = "2026-06-24_161124"
P03_MODEL_TAG = "2026-06-18_180047"
P04_MODEL_TAG = "2026-06-24_161827"
P05_BHT_MODEL_TAG = "2026-06-24_170328"
P05_SALES_MODEL_TAG = "2026-06-24_170351"

EXPECTED_MODEL_TYPES = {
    "p00": "OLS",
    "p01": "LME",
    "p02": "LME",
    "p03": "LME",
    "p04": "LME",
}


def _validate_model_type(sample_name, model, if_debug=False):
    """Assert model.conf.model_type matches expected value for the sample."""
    expected = EXPECTED_MODEL_TYPES.get(sample_name)
    if expected is None:
        return
    actual = model.conf.model_type
    assert actual == expected, (
        f"{sample_name}: expected model_type={expected}, got {actual}"
    )


@pytest.fixture(scope="session")
def p00_model_ds(samples_path):
    """Load p00 model + dataset once per session. No adhoc code needed."""
    import fermatrica as fm

    root = samples_path / "p00_sample"
    model_path = str(root / "code_py" / "model_data" / f"model_{P00_MODEL_TAG}")

    model, ds, _ = fm.model_load_ext(model_path, missed_stop=True)
    assert model is not None, "model_load_ext returned None"
    assert ds is not None, "model_load_ext returned None dataset"
    _validate_model_type("p00", model)
    return model, ds


@pytest.fixture(scope="session")
def p01_model_ds(samples_path):
    """Load p01 model + dataset once per session, with adhoc code on sys.path."""

    root = samples_path / "p01_sample"

    sys.path.insert(0, str(root))
    try:
        import code_py.adhoc.model  # noqa: F401
    except ImportError as e:
        pytest.exit(f"Failed to import adhoc code from {root}: {e}")

    import fermatrica as fm

    model_path = str(root / "code_py" / "model_data" / f"model_{P01_MODEL_TAG}")

    model, ds, _ = fm.model_load_ext(model_path, missed_stop=True)
    assert model is not None, "model_load_ext returned None"
    assert ds is not None, "model_load_ext returned None dataset"
    _validate_model_type("p01", model)

    yield model, ds

    sys.path.remove(str(root))


@pytest.fixture(scope="session")
def p02_model_ds(samples_path):
    """Load p02 model + dataset (panel per SKU, LME, 13 SKUs)."""
    root = samples_path / "p02_sample"
    _activate_adhoc_path(root)
    import fermatrica as fm

    model_path = str(root / "code_py" / "model_data" / f"model_{P02_MODEL_TAG}")
    model, ds, _ = fm.model_load_ext(model_path, missed_stop=True)
    assert model is not None and ds is not None
    _validate_model_type("p02", model)
    yield model, ds
    _deactivate_adhoc_path(root)


@pytest.fixture(scope="session")
def p03_model_ds(samples_path):
    """Load p03 model + dataset (ABA target, monthly, LME, 3 brands)."""
    root = samples_path / "p03_sample"
    _activate_adhoc_path(root)
    import fermatrica as fm

    model_path = str(root / "code_py" / "model_data" / f"model_{P03_MODEL_TAG}")
    model, ds, _ = fm.model_load_ext(model_path, missed_stop=True)
    assert model is not None and ds is not None
    _validate_model_type("p03", model)
    yield model, ds
    _deactivate_adhoc_path(root)


@pytest.fixture(scope="session")
def p04_model_ds(samples_path):
    """Load p04 model + dataset (panel per region, LME, 4 regions)."""
    root = samples_path / "p04_sample"
    _activate_adhoc_path(root)
    import fermatrica as fm

    model_path = str(root / "code_py" / "model_data" / f"model_{P04_MODEL_TAG}")
    model, ds, _ = fm.model_load_ext(model_path, missed_stop=True)
    assert model is not None and ds is not None
    _validate_model_type("p04", model)
    yield model, ds
    _deactivate_adhoc_path(root)


@pytest.fixture(scope="session")
def p05_models_ds(samples_path):
    """Load p05 BHT + Sales models with their datasets (chained two-level panel)."""
    root = samples_path / "p05_sample"
    _activate_adhoc_path(root)
    import fermatrica as fm

    bht_path = str(root / "code_py" / "model_data_bht" / f"model_{P05_BHT_MODEL_TAG}")
    sales_path = str(root / "code_py" / "model_data_sales" / f"model_{P05_SALES_MODEL_TAG}")

    bht_model, bht_ds, _ = fm.model_load_ext(bht_path, missed_stop=True)
    sales_model, sales_ds, _ = fm.model_load_ext(sales_path, missed_stop=True)
    assert all(x is not None for x in [bht_model, bht_ds, sales_model, sales_ds])
    yield bht_model, bht_ds, sales_model, sales_ds
    _deactivate_adhoc_path(root)
