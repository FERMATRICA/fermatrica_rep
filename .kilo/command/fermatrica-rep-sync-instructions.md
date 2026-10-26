---
description: Review and sync AGENTS.md with current project structure
---

# Sync FERMATRICA_REP Instructions

Review AGENTS.md and `.kilo/command/*.md` against the actual project structure and code conventions. Detect and fix any drift.

## Checklist for this review

1. **Directory structure** — read `fermatrica_rep/` recursively. Does the module tree description in AGENTS.md match reality? Are there new modules or deleted modules?

2. **Public API** — read `fermatrica_rep/__init__.py`. Are all public symbols still present, or were any added/removed?

3. **ModelRep class** — read `fermatrica_rep/meta_model/model_rep.py`. Are all constructor parameters and fields still correct?

4. **OptionSettings** — read `fermatrica_rep/options/define.py`. Are the OptionSettings fields and methods still accurate?

5. **Translators** — read `fermatrica_rep/options/translators.py`. Are all translator functions (`basic`, `basic_fix`, `price`, `tv_complex`, etc.) still present with the correct signature?

6. **Option calculation** — read `fermatrica_rep/options/calc.py` and `calc_multi.py`. Are the function signatures and return types still correct?

7. **Budget optimization** — read `fermatrica_rep/options/optim.py`. Are the optimizer function signatures still correct?

8. **PPTX export** — read `fermatrica_rep/export/export_pptx.py`. Are the main export functions (`create_presentation`, per-component functions) still correct?

9. **Tests** — read `tests/` listing. Are the test module descriptions and what-to-test tables up to date?

10. **Conventions** — read sample code or recent commits. Are the import ordering, error handling, type annotation, and performance profiling descriptions still correct?

11. **Command files** — read `.kilo/command/` files. Do they still reference correct paths, function signatures, and workflow steps?

For each discrepancy found, update the relevant file. Do not delete information about deprecated features — instead, add a note marking them as deprecated and noting what replaced them.
