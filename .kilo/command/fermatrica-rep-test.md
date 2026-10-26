---
description: Run FERMATRICA_REP tests with proper configuration
---

# Run FERMATRICA_REP Tests

You are a FERMATRICA_REP testing expert. Run the appropriate tests for the changes made and analyze failures.

## Test Commands

```bash
# All tests
pytest tests/

# Options module tests
pytest tests/test_options/ -v

# Single test file
pytest tests/test_options/test_budget_ssn_create.py -v

# With debug mode (drops into breakpoint() on assertion failure)
pytest tests/ --if-debug

# Verbose with shortened traceback
pytest tests/test_options/ -v --tb=short

# Run and stop at first failure
pytest tests/test_options/test_budget_ssn_create.py -x
```

## What to test when changing code

| Change location | Tests to run |
|-----------------|-------------|
| `fermatrica_rep/options/define.py` | `tests/test_options/test_budget_ssn_create.py` |
| `fermatrica_rep/options/translators.py` | `tests/test_options/` (if test cases exist) |
| `fermatrica_rep/options/calc.py` | `tests/test_options/` (if test cases exist) |
| `fermatrica_rep/options/optim.py` | `tests/test_options/` (if test cases exist) |
| `fermatrica_rep/basics.py` | Manual inspection (no dedicated tests yet) |
| `fermatrica_rep/analysis/stats.py` | Manual inspection |
| `fermatrica_rep/analysis/decomposition.py` | Manual inspection |
| `fermatrica_rep/export/` | Manual inspection (requires MS Office) |
| Any new module | Add test cases + YAML files in `tests/` |

## Test Architecture

### YAML-driven parametrization
Test cases are defined in `tests/**/test_cases/*.yaml` with custom YAML constructors (`!pd.DataFrame`, `!pd.Series`, `!np.array`). Each test case entry has:
- `description`: test name
- `params`: input values
- `expected_result`: expected output
- `exp_err`: `pytest.raises(FermatricaError)` or `nullcontext()`

Helper functions in `tests/basics/basics.py`:
- `parametrize_fun(file_name)` — for testing standalone functions
- `parametrize_class(file_name)` — for testing class methods

### YAML constructors used
```yaml
expected_result: !pd.DataFrame
  data: [[1, 2], [3, 4]]
  columns: [a, b]
```

```yaml
expected_result: !pd.Series
  data: [1, 2, 3]
  name: col_a
```

```yaml
expected_result: !np.array
  object: [1, 2, 3]
```

### Debug flag
Root `conftest.py` provides `if_debug` fixture. When `True`, drops into `breakpoint()` on failure:
```bash
pytest tests/ --if-debug
```

## When tests fail

1. **Read the assertion error carefully** — note the expected vs actual values
2. **Check if the test expectations need updating** (new behavior is correct) or code needs fixing
3. **Options test failures** usually indicate a change in OptionSettings constructor or budget_ssn_create logic
4. **YAML test failures** — check if `expected_result` in the YAML matches the new code behavior
5. **Use `--if-debug`** to get an interactive debugger on assertion failures
6. **No dedicated tests yet** — for modules without tests (basics, stats, decomposition, export), manual verification is required
