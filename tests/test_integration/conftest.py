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
    """Evict cached adhoc modules and insert sample root at the front of sys.path."""
    keys = [k for k in list(sys.modules) if k.startswith("code_py")]
    for k in keys:
        del sys.modules[k]
    root_str = str(root)
    while root_str in sys.path:
        sys.path.remove(root_str)
    sys.path.insert(0, root_str)
    import code_py.adhoc.model  # noqa: F401


def _deactivate_adhoc_path(root: Path):
    """Remove sample root from sys.path."""
    if str(root) in sys.path:
        sys.path.remove(str(root))


P00_MODEL_TAG = "2026-08-04_222911"
P01_MODEL_TAG = "2026-08-04_204155"
P02_MODEL_TAG = "2026-08-04_202726"
P03_MODEL_TAG = "2026-08-04_154614"
P04_MODEL_TAG = "2026-08-04_112441"
P05_BHT_MODEL_TAG = "2026-07-31_164017"
P05_SALES_MODEL_TAG = "2026-07-31_164247"

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
    model_dir = root / "code_py" / "model_data"
    model_path = model_dir / f"model_{P00_MODEL_TAG}"
    if not model_path.is_dir():
        model_path = Path(_find_latest_model(model_dir))
    model, ds, _ = fm.model_load_ext(str(model_path), missed_stop=True)
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

    model_dir = root / "code_py" / "model_data"
    model_path = model_dir / f"model_{P01_MODEL_TAG}"
    if not model_path.is_dir():
        model_path = Path(_find_latest_model(model_dir))
    model, ds, _ = fm.model_load_ext(str(model_path), missed_stop=True)
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

    model_dir = root / "code_py" / "model_data"
    model_path = model_dir / f"model_{P02_MODEL_TAG}"
    if not model_path.is_dir():
        model_path = Path(_find_latest_model(model_dir))
    model, ds, _ = fm.model_load_ext(str(model_path), missed_stop=True)
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

    model_dir = root / "code_py" / "model_data"
    model_path = model_dir / f"model_{P03_MODEL_TAG}"
    if not model_path.is_dir():
        model_path = Path(_find_latest_model(model_dir))
    model, ds, _ = fm.model_load_ext(str(model_path), missed_stop=True)
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

    model_dir = root / "code_py" / "model_data"
    model_path = model_dir / f"model_{P04_MODEL_TAG}"
    if not model_path.is_dir():
        # refreshed model dir not yet committed under the pinned tag:
        # fall back to the newest model_* directory
        model_path = Path(_find_latest_model(model_dir))
    model, ds, _ = fm.model_load_ext(str(model_path), missed_stop=True)
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

    bht_dir = root / "code_py" / "model_data_bht"
    bht_path = bht_dir / f"model_{P05_BHT_MODEL_TAG}"
    if not bht_path.is_dir():
        bht_path = Path(_find_latest_model(bht_dir))
    sales_dir = root / "code_py" / "model_data_sales"
    sales_path = sales_dir / f"model_{P05_SALES_MODEL_TAG}"
    if not sales_path.is_dir():
        sales_path = Path(_find_latest_model(sales_dir))

    bht_model, bht_ds, _ = fm.model_load_ext(str(bht_path), missed_stop=True)
    sales_model, sales_ds, _ = fm.model_load_ext(str(sales_path), missed_stop=True)
    assert all(x is not None for x in [bht_model, bht_ds, sales_model, sales_ds])
    yield bht_model, bht_ds, sales_model, sales_ds
    _deactivate_adhoc_path(root)


P07_CATEGORY_MODEL_TAG = "2026-07-17_191154"
P07_SALES_MODEL_TAG = "2026-07-17_191630"


@pytest.fixture(scope="session")
def p07_models_ds(samples_path):
    """Load p07 Category + Sales models (chained OLS -> LME DAG).

    p07 is a two-level model: Category (weekly OLS) -> Sales (weekly LME, 13 SKUs).
    Merge function pred_merge_1 broadcasts category predictions to SKU-level data.
    """
    import fermatrica as fm

    root = samples_path / "p07_sample"
    _activate_adhoc_path(root)

    cat_path = str(root / "code_py" / "model_data_category" / f"model_{P07_CATEGORY_MODEL_TAG}")
    sales_path = str(root / "code_py" / "model_data_sales" / f"model_{P07_SALES_MODEL_TAG}")

    cat_model, cat_ds, _ = fm.model_load_ext(cat_path, missed_stop=True)
    sales_model, sales_ds, _ = fm.model_load_ext(sales_path, missed_stop=True)
    assert all(x is not None for x in [cat_model, cat_ds, sales_model, sales_ds])

    yield cat_model, cat_ds, sales_model, sales_ds
    _deactivate_adhoc_path(root)


P10_CATEGORY_MODEL_TAG = "2026-07-17_191154"
P10_SALES_MODEL_TAG = "2026-07-17_191630"


@pytest.fixture(scope="session")
def p10_models_ds(samples_path):
    """Load p10 Category (static) + Sales models (chained OLS -> LME DAG).

    p10 is a copy of p07 with the Category node marked static: its
    translate() is a no-op, no trans_dict is required, and its output is
    option-invariant while Sales still translates budgets.
    """
    import fermatrica as fm

    root = samples_path / "p10_sample"
    _activate_adhoc_path(root)

    cat_path = str(root / "code_py" / "model_data_category" / f"model_{P10_CATEGORY_MODEL_TAG}")
    sales_path = str(root / "code_py" / "model_data_sales" / f"model_{P10_SALES_MODEL_TAG}")

    cat_model, cat_ds, _ = fm.model_load_ext(cat_path, missed_stop=True)
    sales_model, sales_ds, _ = fm.model_load_ext(sales_path, missed_stop=True)
    assert all(x is not None for x in [cat_model, cat_ds, sales_model, sales_ds])

    yield cat_model, cat_ds, sales_model, sales_ds
    _deactivate_adhoc_path(root)
