"""Fidelity guards for the sample model conf generators.

The ``p0*/code_py/_generate_scripts/create_model_conf*.py`` scripts are the
committed source of truth for each sample's base ``model_conf.xlsx``. They must
keep reproducing their committed base configs: any accidental change to a
generator (or to the data it reads) would silently desync the sample from its
documented configuration.

p03 is special: its generator intentionally keeps ``fixed=0`` optimizer rows as
its own starting seeds, so regenerating it yields the *base* conf, while the
committed *fitted* conf (inside the pinned ``model_*`` dir) differs from it only
on those ``fixed=0`` seed cells. ``test_p03_residual_vs_fitted_is_generator_seeds_only``
pins that residual, which is the permanent guard against silently pinning a
formerly-free optimizer parameter (``fixed=1`` / ``set_start=1`` rows must never
diverge).
"""

import re
from pathlib import Path

import openpyxl
from openpyxl.utils import get_column_letter
import pytest

try:
    # Reuse the pinned fitted-model tag from the integration conftest so the
    # guard follows a re-save/rename there instead of drifting.
    from tests.test_integration.conftest import P03_MODEL_TAG
except ImportError:  # pragma: no cover - defensive outside pytest's rootdir
    P03_MODEL_TAG = "2026-08-04_154614"


# (sample, script relative to samples_path, committed base conf relative to samples_path)
GENERATORS = [
    pytest.param(
        "p03",
        "p03_sample/code_py/_generate_scripts/create_model_conf.py",
        "p03_sample/code_py/model_data/model_conf.xlsx",
        id="p03",
    ),
    pytest.param(
        "p05",
        "p05_sample/code_py/_generate_scripts/create_model_conf_bht.py",
        "p05_sample/code_py/model_data_bht/model_conf.xlsx",
        id="p05",
    ),
    pytest.param(
        "p06",
        "p06_sample/code_py/_generate_scripts/create_model_conf_aba.py",
        "p06_sample/code_py/model_data_aba/model_conf.xlsx",
        id="p06",
    ),
    pytest.param(
        "p09",
        "p09_sample/code_py/_generate_scripts/create_model_conf_aba.py",
        "p09_sample/code_py/model_data_aba/model_conf.xlsx",
        id="p09",
    ),
]

P03_SCRIPT = "p03_sample/code_py/_generate_scripts/create_model_conf.py"
P03_BASE = "p03_sample/code_py/model_data/model_conf.xlsx"
P03_MODEL_DIR = "p03_sample/code_py/model_data"
P03_FITTED = f"{P03_MODEL_DIR}/model_{P03_MODEL_TAG}/model_conf.xlsx"

NON_PARAMS_SHEETS = ["RHS", "LHS", "setup", "scoring", "trans_path_df"]


def _run_generator_to_temp(script_path: Path, out_dir: Path) -> Path:
    """Run a sample conf generator with only its ``out_path`` redirected.

    The script source is read, the single ``^out_path = ...$`` line is replaced
    with a temp-file target, and the result is ``exec``-ed with the *real*
    ``__file__`` so the script's ``_SCRIPT_DIR`` / ``pXX_root`` derivation keeps
    pointing at the sample tree. Only the generated workbook is written, and it
    lands in ``out_dir`` (a pytest ``tmp_path``). No sample repo file is touched.
    """
    src = Path(script_path).read_text(encoding="utf-8")
    pattern = re.compile(r"^out_path\s*=.*$", re.MULTILINE)
    matches = pattern.findall(src)
    assert len(matches) == 1, (
        f"Expected exactly one '^out_path = ...' line in {script_path}, "
        f"found {len(matches)}"
    )

    out_file = Path(out_dir) / "generated_model_conf.xlsx"
    replacement = 'out_path = r"' + str(out_file) + '"'
    new_src = pattern.sub(lambda _m: replacement, src, count=1)

    exec(
        compile(new_src, str(script_path), "exec"),
        {"__name__": "__main__", "__file__": str(script_path)},
    )
    assert out_file.exists(), f"generator did not write {out_file}"
    return out_file


def _load_sheet_rows(path: Path) -> dict[str, list[list]]:
    wb = openpyxl.load_workbook(str(path), data_only=False)
    sheets: dict[str, list[list]] = {}
    for ws in wb.worksheets:
        sheets[ws.title] = [list(row) for row in ws.iter_rows(values_only=True)]
    return sheets


def _cell_diffs(sheet: str, rows_a: list[list], rows_b: list[list]):
    cell_diffs = []
    n_rows = max(len(rows_a), len(rows_b))
    for ri in range(n_rows):
        row_a = rows_a[ri] if ri < len(rows_a) else []
        row_b = rows_b[ri] if ri < len(rows_b) else []
        n_cols = max(len(row_a), len(row_b))
        for ci in range(n_cols):
            val_a = row_a[ci] if ci < len(row_a) else None
            val_b = row_b[ci] if ci < len(row_b) else None
            if val_a != val_b:
                coord = f"{get_column_letter(ci + 1)}{ri + 1}"
                cell_diffs.append((sheet, coord, val_a, val_b))
    return cell_diffs


def _cfgs_equal(path_a: Path, path_b: Path, sheets: list[str] | None = None):
    """Compare two conf workbooks cell by cell.

    Returns a list of ``(sheet, cell, a, b)`` differences. When ``sheets`` is
    given, only those sheets are compared; otherwise every sheet is compared
    and the sheet-name lists must match.
    """
    conf_a = _load_sheet_rows(path_a)
    conf_b = _load_sheet_rows(path_b)

    diffs = []
    if sheets is None:
        if list(conf_a) != list(conf_b):
            diffs.append(("<workbook>", "sheetnames", list(conf_a), list(conf_b)))
        sheets = list(conf_a)

    for sheet in sheets:
        if sheet not in conf_a or sheet not in conf_b:
            diffs.append((sheet, "<missing>", sheet in conf_a, sheet in conf_b))
            continue
        diffs.extend(_cell_diffs(sheet, conf_a[sheet], conf_b[sheet]))
    return diffs


def _format_diffs(diffs, limit: int = 20) -> str:
    shown = "\n".join(
        f"  - {sheet}!{cell}: generated={val_a!r} committed={val_b!r}"
        for sheet, cell, val_a, val_b in diffs[:limit]
    )
    extra = "" if len(diffs) <= limit else f"\n  ... and {len(diffs) - limit} more"
    return f"{len(diffs)} difference(s):\n{shown}{extra}"


def _params_by_key(path: Path):
    """Return ``(header, header_index, {key: row})`` for the ``params`` sheet.

    Key is ``(variable, fun, arg)`` — the row identity used by the conf loader.
    """
    wb = openpyxl.load_workbook(str(path), data_only=False)
    rows = [list(row) for row in wb["params"].iter_rows(values_only=True)]
    header = rows[0]
    index = {name: i for i, name in enumerate(header)}
    table = {}
    for row in rows[1:]:
        key = (
            row[index["variable"]],
            row[index["fun"]],
            row[index["arg"]],
        )
        table[key] = row
    return header, index, table


def _as_flag(value):
    return int(value) if value is not None else 0


def _find_latest_fitted_conf(model_dir: Path) -> Path | None:
    """Newest ``model_*`` sub-directory holding a ``model_conf.xlsx``.

    Mirrors the name-sort convention of ``conftest._find_latest_model``, adapted
    for a must-fail guard: it returns ``None`` instead of aborting the whole
    session via ``pytest.exit``, requires an actual ``model_conf.xlsx``, and
    skips ``*_dash`` reporting saves (the p03 ``_dash`` conf rounds
    ``scoring!D5`` differently from the fitted conf and would spuriously trip
    the structural check).
    """
    if not model_dir.is_dir():
        return None
    candidates = [
        d
        for d in model_dir.iterdir()
        if d.is_dir()
        and d.name.startswith("model_")
        and not d.name.endswith("_dash")
        and (d / "model_conf.xlsx").is_file()
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda d: d.name)
    return candidates[-1] / "model_conf.xlsx"


@pytest.mark.integration
@pytest.mark.parametrize("sample, script_rel, base_rel", GENERATORS)
def test_generator_reproduces_base_conf(
    samples_path, tmp_path, sample, script_rel, base_rel
):
    """Each generator must reproduce its committed base conf with 0 cell diffs."""
    script = samples_path / script_rel
    base = samples_path / base_rel

    if not script.exists():
        pytest.skip(f"{sample}: generator script not found: {script}")
    if not base.exists():
        pytest.skip(f"{sample}: committed base conf not found: {base}")

    generated = _run_generator_to_temp(script, tmp_path)
    diffs = _cfgs_equal(generated, base)
    assert not diffs, (
        f"{sample}: generator no longer reproduces the committed base conf — "
        f"{_format_diffs(diffs)}"
    )


@pytest.mark.integration
def test_p03_residual_vs_fitted_is_generator_seeds_only(samples_path, tmp_path):
    """The p03 generated base vs committed fitted conf may differ only on
    ``fixed=0`` optimizer seed cells (columns ``value`` / ``upper``)."""
    script = samples_path / P03_SCRIPT
    base = samples_path / P03_BASE

    if not script.exists():
        pytest.skip(f"p03: generator script not found: {script}")
    if not base.exists():
        pytest.skip(f"p03: committed base conf not found: {base}")

    pinned_fitted = samples_path / P03_FITTED
    fitted = (
        pinned_fitted
        if pinned_fitted.is_file()
        else _find_latest_fitted_conf(samples_path / P03_MODEL_DIR)
    )
    if fitted is None:
        pytest.fail(
            "p03: script and base conf are present but no fitted conf could be "
            f"found — looked for the pinned {pinned_fitted} and any "
            "non-'*_dash' 'model_*' directory holding model_conf.xlsx under "
            f"{samples_path / P03_MODEL_DIR}. The residual guard must not go "
            "dormant; restore the fitted p03 model dir (or update "
            "P03_MODEL_TAG)."
        )

    generated = _run_generator_to_temp(script, tmp_path)

    # Non-params sheets must match the fitted conf exactly: the generator
    # reproduces the fitted structure, only optimizer cells may residualize.
    structural_diffs = _cfgs_equal(generated, fitted, sheets=NON_PARAMS_SHEETS)
    assert not structural_diffs, (
        "p03: generated base diverges from the fitted conf on non-params "
        f"sheets — {_format_diffs(structural_diffs)}"
    )

    header, index, gen_params = _params_by_key(generated)
    _, _, fit_params = _params_by_key(fitted)

    missing = set(fit_params) - set(gen_params)
    extra = set(gen_params) - set(fit_params)
    assert not missing and not extra, (
        "p03: params row keys differ between generated base and fitted conf — "
        f"missing={sorted(missing, key=str)} extra={sorted(extra, key=str)}"
    )

    allowed_columns = ("value", "upper")
    residual = []
    offending = []
    pinned_diffs = []

    for key in gen_params:
        row_gen = gen_params[key]
        row_fit = fit_params[key]
        fit_fixed = _as_flag(row_fit[index["fixed"]])
        fit_set_start = _as_flag(row_fit[index["set_start"]])
        for column in header:
            ci = index[column]
            val_gen = row_gen[ci] if ci < len(row_gen) else None
            val_fit = row_fit[ci] if ci < len(row_fit) else None
            if val_gen == val_fit:
                continue
            residual.append((key, column, val_gen, val_fit))
            if fit_fixed == 0 and column in allowed_columns:
                continue
            offending.append((key, column, val_gen, val_fit))
            if fit_fixed == 1 or fit_set_start == 1:
                pinned_diffs.append((key, column, fit_fixed, fit_set_start))

    assert not pinned_diffs, (
        "p03: a silent pin changed a fixed=1 / set_start=1 row — "
        + "; ".join(
            f"{key} col={col} (fitted fixed={fx}, set_start={ss})"
            for key, col, fx, ss in pinned_diffs[:20]
        )
    )

    assert not offending, (
        "p03: generated/fitted residual escapes the fixed=0 seed set — only "
        "fixed=0 rows may differ and only on 'value'/'upper' — "
        + "; ".join(
            f"{key} col={col} generated={vg!r} fitted={vf!r}"
            for key, col, vg, vf in offending[:20]
        )
    )

    assert residual, (
        "p03: expected a non-empty residual (fixed=0 optimizer seed cells), "
        "but generated base equals the fitted conf cell for cell — the "
        "seed-residual guarantee is no longer being exercised"
    )
