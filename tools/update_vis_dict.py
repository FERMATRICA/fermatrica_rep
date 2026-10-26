"""Maintain fermatrica_rep/res/dict/vis_dict.xlsx.

Re-runnable updater that:
1. adds/keeps a 'status' column ('active'/'legacy', default 'active'),
2. marks known-unused legacy rows/sections,
3. seeds additive rows (section/variable/en/ru) when absent,
4. applies keyed in-place updates.

Run from the repo root:  python tools/update_vis_dict.py
"""
import pandas as pd

XLSX = r"fermatrica_rep/res/dict/vis_dict.xlsx"

LEGACY_SECTIONS = {
    "sales_type", "export_shared", "replace",
    "budget.split.options.short", "budget.split.options.long",
    "budget_options_plot", "budget_options_table",
    "option.detail.sales.sku", "option.detail.effects.superbrand.year",
}
LEGACY_ROWS = {("plot_efficiency", "factors_title_dash")}

ROWS = [  # (section, variable, english, russian) — new rows inserted with status "active"
    ("shared", "pct", "%", "%"),
    ("option.detail.sales.brand_awareness", "title",
     "Option ***id. ***year expected Brand Awareness: ***volume_sales ***units",
     "Опция ***id. Знание в ***year г.: ***volume_sales ***units"),
    ("predict_table.brand_awareness", "title",
     "***id. ***year expected Brand Awareness: ***volume_sales ***units",
     "***id. Знание в ***year г.: ***volume_sales ***units"),
    ("option_compare.brand_awareness", "sales_title", "Brand Awareness", "Знание"),
    ("option_compare_sales.brand_awareness", "res", "Brand Awareness, %", "Знание, %"),
    ("option_compare_sales.brand_awareness", "grw",
     "Brand Awareness growth to ***", "Прирост знания к *** г."),
    ("plot_efficiency.brand_awareness", "factors_title",
     "***units Brand Awareness Factors", "декомпозиция знания в ***units"),
    ("efficiency_short.brand_awareness", "incr_sales_axis_y",
     "*** Incr. Brand Awareness, %", "Инкр. знание, %, *** г."),
    ("efficiency_long.brand_awareness", "incr_sales_axis_y",
     "*** Avg. Incr. Brand Awareness, %", "Средний инкр. знание, %, *** г."),
    ("efficiency_long.brand_awareness", "incr_sales_cum_axis_y",
     "*** Sum. Incr. Brand Awareness, %", "Суммарный инкр. знание, %, *** г."),
]

UPDATES = {  # (section, variable) -> {column: value} — in-place updates of existing rows
    ("shared", "units"): {"russian": "ед."},
    ("predict_costs_table", "value"): {"english": "Mln. ***cur, per year", "russian": "Млн.***cur, в год"},
    ("option_compare", "costs_title"): {"english": "Costs, Mln. ***cur", "russian": "Инвестиции, млн.***cur"},
    ("option_compare_sales", "res_rub_distr"): {"english": "Value Sales, ***cur", "russian": "Продажи, ***cur"},
    ("option_compare_sales", "res_rub_progress_distr"): {"english": "Incremental Sales, ***cur", "russian": "Инкр. продажи, ***cur"},
    ("option_compare_sales", "profit"): {"english": "Profit, ***cur", "russian": "Окупаемость, ***cur"},
    ("option.detail.sales", "title"): {"english": "Option ***id. ***year expected sales: ***volume_sales ***units, ***value_sales ***cur",
                                       "russian": "Опция ***id. Продажи в ***year г.: ***volume_sales ***units, ***value_sales ***cur"},
    ("predict_table", "title"): {"english": "***id. ***year expected sales: ***volume_sales ***units, ***value_sales ***cur",
                                 "russian": "***id. Продажи в ***year г.: ***volume_sales ***units, ***value_sales ***cur"},
    ("efficiency_short", "axis_x"): {"english": "*** Costs, Mln. ***cur", "russian": "Инвестиции, млн.***cur, *** г."},
    ("efficiency_short", "incr_sales_axis_y"): {"english": "*** Incr. Sales, Mln. ***cur", "russian": "Инкр. продажи, млн.***cur, *** г."},
    ("efficiency_short", "profit_axis_y"): {"english": "Profit ***, ***cur", "russian": "Окупаемость, ***cur, *** г."},
    ("efficiency_long", "axis_x"): {"english": "*** Yearly Costs, Mln. ***cur", "russian": "Годовые инвестиции, млн.***cur, *** г."},
    ("efficiency_long", "axis_x_if_exact_true"): {"english": "Annual costs only in *** year, Mln. ***cur", "russian": "Годовые инвестиции только в *** г., млн.***cur"},
    ("efficiency_long", "axis_x_if_exact_false"): {"english": "Costs every year ***, Mln. ***cur", "russian": "Инвестиции каждый год *** г., млн.***cur"},
    ("efficiency_long", "incr_sales_axis_y"): {"english": "*** Avg. Incr. Sales, Mln. ***cur", "russian": "Средние инкр. продажи, млн.***cur, *** г."},
    ("efficiency_long", "profit_axis_y"): {"english": "Avg. Profit ***, ***cur", "russian": "Средняя окупаемость, ***cur, *** г."},
    ("efficiency_long", "incr_sales_cum_axis_y"): {"english": "*** Sum. Incr. Sales, Mln. ***cur", "russian": "Суммарные инкр. продажи, млн.***cur, *** г."},
    ("efficiency_long", "profit_cum_axis_y"): {"english": "Sum. Profit ***, ***cur", "russian": "Суммарная окупаемость, ***cur, *** г."},
}


def _apply_rows(df, rows, updates):
    for section, variable, en, ru in rows:
        mask = (df["section"] == section) & (df["variable"] == variable)
        if mask.any():
            df.loc[mask, ["english", "russian"]] = [en, ru]
        else:
            df = pd.concat([df, pd.DataFrame([{"section": section, "variable": variable,
                                               "english": en, "russian": ru, "status": "active"}])],
                           ignore_index=True)
    for key, cols in updates.items():
        mask = (df["section"] == key[0]) & (df["variable"] == key[1])
        assert mask.any(), f"row not found: {key}"
        for col, val in cols.items():
            df.loc[mask, col] = val
    return df


def main():
    df = pd.read_excel(XLSX)
    if "status" not in df.columns:
        df["status"] = "active"
    mask_sec = df["section"].isin(LEGACY_SECTIONS)
    mask_row = df.apply(lambda r: (r["section"], r["variable"]) in LEGACY_ROWS, axis=1)
    df.loc[mask_sec | mask_row, "status"] = "legacy"
    df = _apply_rows(df, ROWS, UPDATES)
    df["status"] = df["status"].fillna("active")
    df.to_excel(XLSX, index=False, sheet_name="data")


if __name__ == "__main__":
    main()
