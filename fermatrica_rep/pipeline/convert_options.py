#!/usr/bin/env python
"""
Migrate old options.xlsx (flat budget sheet) to new multi-sheet format.

Usage:
    python -m fermatrica_rep.pipeline.convert_options <old_options.xlsx> [--output new_options.xlsx] [--scope-name scope1] [--date-start 2025-01-01] [--date-end 2025-12-31]

Creates:
  - scopes sheet with a single default scope
  - budget sheet in long format with scope_name
  - translation sheet with price renamed to base_price
"""


import argparse
import sys
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Convert old options.xlsx to new format")
    parser.add_argument("input", help="Path to old options.xlsx")
    parser.add_argument("--output", default=None, help="Path to new options.xlsx (default: input name + '_new')")
    parser.add_argument("--scope-name", default="default", help="Scope name (default: 'default')")
    parser.add_argument("--date-start", default="2025-01-01", help="Scope date_start")
    parser.add_argument("--date-end", default="2025-12-31", help="Scope date_end")
    parser.add_argument("--ref-date-start", default=None, help="Scope ref_date_start (default: date_start)")
    parser.add_argument("--ref-date-end", default=None, help="Scope ref_date_end (default: date_end)")
    parser.add_argument("--apply-vars", default="superbrand", help="Comma-separated apply_vars")
    parser.add_argument("--target", default="", help="Comma-separated target values")
    parser.add_argument("--plan-period", default="year", help="plan_period (default: year)")

    args = parser.parse_args()

    if args.output is None:
        args.output = args.input.replace(".xlsx", "_new.xlsx")

    ref_start = args.ref_date_start or args.date_start
    ref_end = args.ref_date_end or args.date_end

    input_file = args.input

    try:
        xl = pd.ExcelFile(input_file)
    except Exception as e:
        print(f"Error opening {input_file}: {e}")
        sys.exit(1)

    sheet_names = xl.sheet_names
    print(f"Found sheets: {sheet_names}")

    scope_data = pd.DataFrame(
        [
            {
                "scope_name": args.scope_name,
                "date_start": args.date_start,
                "date_end": args.date_end,
                "ref_date_start": ref_start,
                "ref_date_end": ref_end,
                "apply_vars": args.apply_vars,
                "target": args.target,
                "plan_period": args.plan_period,
                "inflation_multiplier": 1.0,
            }
        ]
    )

    budget_data = pd.DataFrame()
    translation_data = pd.DataFrame()
    seasonality_data = pd.DataFrame()

    if "budget" in sheet_names:
        old_budget = pd.read_excel(input_file, sheet_name="budget")
        budget_records = []
        option_col = None

        for col in old_budget.columns:
            if col.lower() == "option":
                option_col = col
                break

        if option_col is not None:
            for _, row in old_budget.iterrows():
                option_name = str(row[option_col])
                for col in old_budget.columns:
                    if col == option_col or col.lower() in ("bdg", "text_flags"):
                        continue
                    val = row[col]
                    if pd.notna(val) and val != 0:
                        budget_records.append(
                            {
                                "option": option_name,
                                "scope_name": args.scope_name,
                                "item": col,
                                "val": float(val),
                            }
                        )
            budget_data = pd.DataFrame(budget_records)
            print(f"Converted {len(budget_records)} budget rows")
        else:
            print("Warning: no 'option' column found in budget sheet — skipping")

    if "translation" in sheet_names:
        translation_data = pd.read_excel(input_file, sheet_name="translation")
        if "price" in translation_data.columns and "base_price" not in translation_data.columns:
            translation_data["base_price"] = translation_data["price"]
            translation_data.drop(columns=["price"], inplace=True)
        print(f"Converted translation sheet ({len(translation_data)} rows)")

    if "seasonality" in sheet_names:
        old_ssn = pd.read_excel(input_file, sheet_name="seasonality")
        old_ssn["scope_name"] = args.scope_name
        cols = ["scope_name"] + [c for c in old_ssn.columns if c != "scope_name"]
        seasonality_data = old_ssn[cols]
        print(f"Converted seasonality sheet ({len(seasonality_data)} rows)")

    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        scope_data.to_excel(writer, sheet_name="scopes", index=False)
        if not budget_data.empty:
            budget_data.to_excel(writer, sheet_name="allocations", index=False)
        if not translation_data.empty:
            translation_data.to_excel(writer, sheet_name="translation", index=False)
        if not seasonality_data.empty:
            seasonality_data.to_excel(writer, sheet_name="seasonality", index=False)

    print(f"Written to {args.output}")
    print("Review the output file and add any additional scopes manually.")


if __name__ == "__main__":
    main()
