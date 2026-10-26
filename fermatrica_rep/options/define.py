"""
Define option and translation settings and prepare dataset for option calculation.
"""


import copy
import datetime
import warnings
import pandas as pd
import re
from openpyxl import load_workbook
from pandas.api.types import is_string_dtype, is_object_dtype

from fermatrica_utils import StableClass, period_align
from fermatrica.basics.basics import fermatrica_error

from fermatrica_rep.options.scope import OptionScope
from fermatrica_rep.options.option import Option


def trans_dict_create(dt_trans: pd.DataFrame | None = None
                      , path: str | None = None
                      , sheet: str | None = 'translation') -> dict:
    """
    Create translation rules dictionary from XLSX path or y from DataFrame. Set either `path` or
    `dt_trans` respectively. It binds tool name with translation function, data variable and max
    budget if curves to be calculated.

    Translation dictionary is vital for option calculation and hence for the export of future
    periods.

    :param dt_trans: translation dataframe (as `translation` sheet in "options.xlsx" file)
    :param path: path to XLSX file like "options.xlsx" with `sheet` sheet
    :param sheet: name of the sheet with translation data (only if `path` is set), defaults to
        "translation"
    :return: translation dictionary
    """

    # load

    if not pd.isna(path) and path != "":

        wb = load_workbook(path)
        shts = wb.sheetnames
        wb.close()

        if sheet not in shts:
            fermatrica_error('Error reading XLSX file with translation rules: ' + path + '. Sheet ' + sheet + ' not found')

        try:
            dt_trans = pd.read_excel(path, sheet_name=sheet)
        except Exception as e:
            fermatrica_error('Error reading XLSX file with translation rules: ' + path + '. File not found')

    elif pd.isna(dt_trans):
        fermatrica_error('Error creating translation rules dictionary. Not dict nor path to XLSX file is provided')

    # convert to dictionary

    if "item_group" not in dt_trans.columns:
        dt_trans["item_group"] = dt_trans["item"].str.replace("bdg_", "")

    if "base_price" not in dt_trans.columns and "price" in dt_trans.columns:
        dt_trans["base_price"] = dt_trans["price"]
    if "base_price" not in dt_trans.columns:
        dt_trans["base_price"] = 1.0

    dt_trans = dt_trans.loc[(dt_trans["function"] != "") & (pd.notna(dt_trans["function"])), :]

    dt_trans.set_index("item", inplace=True)
    dct = dt_trans.to_dict(orient="dict")

    return dct


def _enrich_trans_dict_from_model(model_conf, trans_dict: dict) -> dict:
    """
    Enrich ``trans_dict["base_price"]`` with media prices from
    ``model.conf.trans_path_df``.

    Matches ``trans_path_df["variable"]`` to ``trans_dict["variable"][item]``
    and copies the ``price`` column into ``trans_dict["base_price"][item]``.
    Does nothing if ``trans_dict`` is ``None``, ``trans_path_df`` is missing
    or empty, or if required columns are absent.

    :param model_conf: ModelConf object (``model.conf``)
    :param trans_dict: translation dictionary (mutated in place)
    :return: the same ``trans_dict`` (mutated in place, also returned for chaining)
    """
    if trans_dict is None:
        return trans_dict
    if not hasattr(model_conf, 'trans_path_df') or model_conf.trans_path_df is None:
        return trans_dict

    tdf = model_conf.trans_path_df
    if tdf.empty or 'variable' not in tdf.columns or 'price' not in tdf.columns:
        return trans_dict

    price_map = tdf.set_index('variable')['price'].to_dict()
    trans_dict.setdefault('base_price', {})

    for item, var_name in trans_dict.get('variable', {}).items():
        if var_name in price_map:
            trans_dict['base_price'][item] = float(price_map[var_name])

    return trans_dict


def budget_dict_create(dct: dict | pd.DataFrame | None = None
                       , path: str | None = None
                       , sheet: str | None = 'main'
                       , vat_rate: float = 0.) -> dict:
    """
    Creates budget dictionary from raw dictionary or from XLSX path. Budget dictionary is dictionary
    containing named options prepared by user by hand with yearly (or similar) costs.

    Use the function to load manually prepared options to the environment.
    Set either `dct` to create from existing dictionary or `dct` to load from disc.

    :param dct: raw dictionary (in a form of a dictionary or a dataframe)
    :param path: path to XLSX file like "options.xlsx" with `sheet` sheet
    :param sheet: name of the sheet with option data (only if `path` is set), defaults to
        "main"
    :param vat_rate: VAT rate, defaults to 0. Set it different if costs used in the model include VAT
    :return: dictionary with named options as separate items
    """

    warnings.warn(
        "budget_dict_create() is deprecated. Use Option.from_excel() with an 'allocations' sheet instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    # load

    if not pd.isna(path) and path != "":

        wb = load_workbook(path)
        shts = wb.sheetnames
        wb.close()

        if sheet not in shts:
            fermatrica_error('Error reading XLSX file with budget options: ' + path + '. Sheet ' + sheet + ' not found')

        try:
            tmp = pd.read_excel(path, sheet_name=sheet)
        except Exception as e:
            fermatrica_error('Error reading XLSX file with budget options: ' + path + '. File not found')

        tmp = tmp.convert_dtypes(infer_objects=True, convert_string=True, convert_integer=True
                                 , convert_boolean=False, convert_floating=True)
        tmp['option'] = tmp['option'].astype('string')

        tmp.set_index('option', inplace=True)
        dct = tmp.to_dict(orient='index')

    elif isinstance(dct, pd.DataFrame):

        dct = dct.convert_dtypes(infer_objects=True, convert_string=True, convert_integer=True
                                 , convert_boolean=False, convert_floating=True)
        dct['option'] = dct['option'].astype('string')

        dct.set_index('option', inplace=True)
        dct = dct.to_dict(orient='index')

    elif pd.isna(dct):
        fermatrica_error('Error creating budget options dictionary. Not dict nor path to XLSX file is provided')

    # apply VAT & calculate total budget

    for k, v in dct.items():
        bdg = 0
        for k1 in v:
            if re.match(r'bdg_', k1):
                if vat_rate > 0:
                    v[k1] *= (1 + vat_rate)
                bdg += v[k1]
        v['bdg'] = bdg

    return dct


def budget_ssn_create(dct: dict | None = None
                      , path: str | None = None
                      , sheets: list | tuple | str | None = None
                      , vat_rate: float = 0.) -> tuple[dict, dict]:
    """
    Loads periodical (monthly, weekly, daily) distribution of budget for specific media from XLSX file
    or from existing dictionary of dataframes and creates yearly budget options and seasonality data from it.
    
    Unlike `budget_dict_create` which takes yearly aggregates, this function takes period-level data
    and calculates both yearly totals and seasonality distributions.
    
    Multi-year data is detected either by presence of 'year' column or by 'date' column spanning 
    multiple calendar years. In multi-year case, yearly totals and seasonalities are calculated 
    separately for each year.
    
    Supports two types of measure columns:
    - 'bdg_*' columns: aggregated with sum, VAT applied, normalized in seasonality
    - 'nobdg_*' columns: aggregated with max, no VAT, copied as-is in seasonality
    
    :param dct: dictionary of DataFrames where keys are option names and values are DataFrames with 
        period-level budget data (optional, use either this or path)
    :param path: path to XLSX file containing multiple sheets with period-level budget data (optional)
    :param sheets: optional list/tuple of specific sheet names or regex pattern to filter sheets
        (only used when path is provided). If None, all sheets are used
    :param vat_rate: VAT rate, defaults to 0. Set it different if costs used in the model include VAT (applies only to bdg_ columns)
    :return: tuple of (budget_dict, seasonality_dict) where budget_dict always contains yearly totals by year
        (structure: {option: {year: {bdg_cols...}}}), even for single-year data. Seasonality_dict contains 
        normalized period distributions with 'year' column
    """
    
    # Load data
    if dct is not None:
        # Use provided dictionary
        if not isinstance(dct, dict):
            fermatrica_error('Error creating budget and seasonality. Provided dct is not a dictionary')
        data_dict = dct
    
    elif not pd.isna(path) and path != "":
        # Load from XLSX file
        try:
            wb = load_workbook(path)
            shts = wb.sheetnames
            wb.close()
        except Exception as e:
            fermatrica_error('Error reading XLSX file with budget data: ' + path + '. File not found or cannot be opened')
        
        if not shts:
            fermatrica_error('Error reading XLSX file with budget data: ' + path + '. No sheets found')
        
        # Filter sheets based on the sheets argument
        if sheets is not None:
            if isinstance(sheets, (list, tuple)):
                shts = [s for s in shts if s in sheets]
            elif isinstance(sheets, str):
                pattern = re.compile(sheets)
                shts = [s for s in shts if pattern.search(s)]
            
            if not shts:
                fermatrica_error('Error filtering sheets in XLSX file: ' + path + '. No sheets match the provided filter')
        
        # Load data from sheets
        data_dict = {}
        for sheet in shts:
            try:
                data_dict[sheet] = pd.read_excel(path, sheet_name=sheet)
            except Exception as e:
                fermatrica_error('Error reading sheet ' + sheet + ' from XLSX file: ' + path + '. ' + str(e))
    
    else:
        fermatrica_error('Error creating budget and seasonality. Either dct or path must be provided')
    
    # Process each option
    budget_dict = {}
    seasonality_dict = {}
    
    for option_name, df in data_dict.items():
        
        df = df.copy()
        
        # Identify period column (first column) and budget columns
        period_var = df.columns.tolist()[0]
        bdg_cols = [x for x in df.columns if re.match(r'^bdg_', x)]
        nobdg_cols = [x for x in df.columns if re.match(r'^nobdg_', x)]
        
        if not bdg_cols and not nobdg_cols:
            fermatrica_error(f'Error processing option {option_name}. No budget columns (starting with "bdg_" or "nobdg_") found')
        
        # Remove check/total rows if present
        if is_string_dtype(df[period_var]) | is_object_dtype(df[period_var]):
            df = df[df[period_var].str.lower() != 'total']
        
        # Ensure year column exists
        if 'year' not in df.columns:
            if 'date' in df.columns:
                # Extract year from date column
                df['date'] = pd.to_datetime(df['date'])
                df['year'] = df['date'].dt.year
            else:
                # No year or date column - raise error
                fermatrica_error(f'Error processing option {option_name}. Neither "year" nor "date" column found in data')
        else:
            # Convert existing year column to numeric
            df['year'] = pd.to_numeric(df['year'])
        
        # Unified processing: always use groupby approach
        # Calculate yearly totals with different aggregations: sum for bdg_, max for nobdg_
        agg_dict = {col: 'sum' for col in bdg_cols}
        agg_dict.update({col: 'max' for col in nobdg_cols})

        if 'text_flags' in df.columns:
            agg_dict['text_flags'] = 'first'

        yearly_totals_df = df.groupby('year').agg(agg_dict)
        
        # Initialize nested dictionary for this option
        budget_dict[option_name] = {}
        
        # Process budget dictionary
        for year in yearly_totals_df.index:
            yearly_totals = {}
            
            # Process bdg_ columns with VAT
            for col in bdg_cols:
                yearly_total = yearly_totals_df.loc[year, col]
                if vat_rate > 0:
                    yearly_total *= (1 + vat_rate)
                yearly_totals[col] = yearly_total
            
            # Process nobdg_ columns (no VAT applied)
            for col in nobdg_cols:
                yearly_totals[col] = yearly_totals_df.loc[year, col]
            
            # Calculate total budget across all bdg_ items
            if bdg_cols:
                total_bdg = sum(yearly_totals[col] for col in bdg_cols)
                yearly_totals['bdg'] = total_bdg

            # Update text_flags with option_name if not already present
            if 'text_flags' not in yearly_totals:
                yearly_totals['text_flags'] = ''

            if not 'name_' + str(option_name) in yearly_totals['text_flags']:
                yearly_totals['text_flags'] = yearly_totals['text_flags'] + 'name_' + str(option_name)

            budget_dict[option_name][year] = yearly_totals
        
        # Calculate seasonality (vectorized): divide each row by its year's total
        # Merge yearly totals back to original dataframe (only for bdg_ columns)
        if bdg_cols:
            bdg_totals_df = yearly_totals_df[bdg_cols]
            df_with_totals = df.merge(bdg_totals_df, on='year', suffixes=('', '_total'))
        
        # Prepare seasonality dataframe
        all_measure_cols = bdg_cols + nobdg_cols
        df_ssn = df[[period_var, 'year'] + [x for x in df.columns if x not in all_measure_cols and x not in [period_var, 'year']]].copy()
        
        # Vectorized division for bdg_ columns (normalized seasonality)
        if bdg_cols:
            for col in bdg_cols:
                total_col = col + '_total'
                # Avoid division by zero
                mask = df_with_totals[total_col] > 0
                df_ssn[col] = 0.0
                df_ssn.loc[mask, col] = df_with_totals.loc[mask, col] / df_with_totals.loc[mask, total_col]
        
        # Copy nobdg_ columns with +1 shift
        for col in nobdg_cols:
            df_ssn[col] = df[col] + 1.0
        
        # Rename columns: remove bdg_ and nobdg_ prefixes
        rename_dict = {col: col.replace('bdg_', '', 1) for col in bdg_cols}
        rename_dict.update({col: col.replace('nobdg_', '', 1) for col in nobdg_cols})

        df_ssn.rename(columns=rename_dict, inplace=True)

        if 'year' in df_ssn.columns and 'date' in df_ssn.columns:
            # If both year and date are present, keep only date
            df_ssn.drop(columns=['year'], inplace=True)
        
        # Add to seasonality dictionary
        seasonality_dict[option_name] = df_ssn
    
    return budget_dict, seasonality_dict


def media_ssn_create(path: str
                     , sheets: list | tuple | str | None = None) -> dict:
    """
    Loads periodical (monthly, weekly, daily) distribution of media seasonality for specific media from XLSX file
    for different options as a dict with option name as key and seasonality dataframe as value.
    Seasonality data is used for distribution of yearly budget by specific periods (months, weeks) within year.
    
    :param path: path to XLSX file containing multiple sheets with seasonality data
    :param sheets: optional list/tuple of specific sheet names or regex pattern to filter sheets. 
        If None, all sheets are used. If list/tuple, only specified sheets are used. 
        If string, treated as regex pattern to match sheet names
    :return: dictionary with option names (sheet names) as keys and seasonality dataframes as values
    """
    
    if pd.isna(path) or path == "":
        fermatrica_error('Error creating media seasonality dictionary. Path to XLSX file is not provided')
    
    # Load workbook and get sheet names
    try:
        wb = load_workbook(path)
        shts = wb.sheetnames
        wb.close()
    except Exception as e:
        fermatrica_error('Error reading XLSX file with media seasonality options: ' + path + '. File not found or cannot be opened')
    
    if not shts:
        fermatrica_error('Error reading XLSX file with media seasonality options: ' + path + '. No sheets found')
    
    # Filter sheets based on the sheets argument
    if sheets is not None:
        if isinstance(sheets, (list, tuple)):
            # Use explicit list of sheet names
            shts = [s for s in shts if s in sheets]
        elif isinstance(sheets, str):
            # Use regex pattern to filter
            pattern = re.compile(sheets)
            shts = [s for s in shts if pattern.search(s)]
        
        if not shts:
            fermatrica_error('Error filtering sheets in XLSX file: ' + path + '. No sheets match the provided filter')
    
    # Create dictionary of seasonality dataframes
    ssn_dict = {}
    
    for sheet in shts:
        try:
            dt_ssn = pd.read_excel(path, sheet_name=sheet)
            ssn_dict[sheet] = dt_ssn
        except Exception as e:
            fermatrica_error('Error reading sheet ' + sheet + ' from XLSX file: ' + path + '. ' + str(e))
    
    return ssn_dict


def _seasonality_to_dates(
    dt_ssn: pd.DataFrame,
    period_var: str,
    ds: pd.DataFrame,
    date_start: datetime.date | str | None,
    date_end: datetime.date | str | None,
) -> pd.DataFrame:
    """
    Convert seasonality DataFrame to date-based format with a ``period`` column.

    Handles period_var values: ``month``, ``week_number``, ``day``, ``date``,
    and ``period`` (from ``seasonality_read``, inferred from value range).
    Both with-year and without-year cases are supported.
    """

    dt_ssn = dt_ssn.copy()
    has_year = "year" in dt_ssn.columns

    if date_start is not None:
        ds_start = pd.to_datetime(date_start)
    else:
        ds_start = ds["date"].min()
    if date_end is not None:
        ds_end = pd.to_datetime(date_end)
    else:
        ds_end = ds["date"].max()

    years = sorted(set(range(ds_start.year, ds_end.year + 1)))
    if not has_year:
        # Yearless annual pattern: expand from the scope start year through the
        # end of the dataset horizon so that seasonality covers the exact scope
        # period and the long-term forecast, while historical years before the
        # scope start are left untouched (NaN). Restores the legacy behavior of
        # the old `seasonality.xlsx` flow and keeps `_translate_long` finite for
        # years past the plan year.
        years = sorted(set(range(ds_start.year, int(ds["date"].dt.year.max()) + 1)))

    if period_var == "date":
        dt_ssn["date"] = pd.to_datetime(dt_ssn["date"])
        dt_ssn["period"] = "day"
        dt_ssn.drop(columns=["year"], inplace=True, errors="ignore")
        return dt_ssn

    src_period: str
    if "month" in period_var.lower() or (
        period_var == "period" and dt_ssn[period_var].max() <= 12
    ):
        src_period = "month"
    elif "week" in period_var.lower() or (
        period_var == "period" and dt_ssn[period_var].max() > 12
    ):
        src_period = "week"
    elif "day" in period_var.lower():
        src_period = "day"
    else:
        fermatrica_error(
            f"Cannot determine seasonality granularity from first column: '{period_var}'"
        )

    dt_ssn[period_var] = pd.to_numeric(dt_ssn[period_var])

    if not has_year:
        expanded = []
        for _, row in dt_ssn.iterrows():
            for y in years:
                r = row.to_dict()
                r["_year"] = y
                expanded.append(r)
        dt_ssn = pd.DataFrame(expanded)
        has_year = True
        year_col = "_year"
    else:
        year_col = "year"

    year_int = dt_ssn[year_col].astype(int)

    if src_period == "month":
        dt_ssn["date"] = pd.to_datetime(
            year_int.astype(str)
            + "-"
            + dt_ssn[period_var].astype(int).astype(str)
            + "-01"
        )
    elif src_period == "week":
        dt_ssn["date"] = pd.to_datetime(
            year_int.astype(str)
            + "-W"
            + dt_ssn[period_var].astype(int).astype(str)
            + "-1",
            format="%Y-W%W-%w",
        )
        # Years with a 53rd Monday (e.g. 2029-12-31) fall outside the %Y-W%W
        # calendar and are left uncovered, leaving NaN seasonality. Reuse the
        # last pattern week for every ds Monday still missing so that the whole
        # dataset receives a seasonality value.
        covered = set(pd.DatetimeIndex(dt_ssn["date"]))
        extras = []
        for d in pd.DatetimeIndex(ds["date"].unique()):
            if d in covered or d.year not in set(years):
                continue
            sel = dt_ssn.loc[dt_ssn["date"].dt.year == d.year]
            if sel.empty:
                continue
            row = sel.iloc[-1].copy()
            row["date"] = d
            extras.append(row)
        if extras:
            dt_ssn = pd.concat([dt_ssn, pd.DataFrame(extras)], ignore_index=True)
    elif src_period == "day":
        dt_ssn["date"] = pd.to_datetime(
            year_int.astype(str) + "-01-01"
        ) + pd.to_timedelta(dt_ssn[period_var].astype(int) - 1, unit="D")

    dt_ssn["period"] = src_period
    dt_ssn.drop(columns=[period_var, "year", "_year"], inplace=True, errors="ignore")

    return dt_ssn


def _normalize_seasonality(
    dt_ssn: pd.DataFrame, bdg_opt_cols: set[str]
) -> pd.DataFrame:
    """
    Normalize budget seasonality columns to sum to 1.0 per year.

    For already-normalized data this is a no-op (division by 1.0).
    Non-budget columns are left untouched.
    """

    if not bdg_opt_cols:
        return dt_ssn

    dt_ssn = dt_ssn.copy()
    dt_ssn["_year"] = dt_ssn["date"].dt.year

    for col in bdg_opt_cols:
        if col not in dt_ssn.columns:
            continue
        totals = dt_ssn.groupby("_year")[col].transform("sum")
        mask = totals > 0
        if mask.any():
            dt_ssn.loc[mask, col] = dt_ssn.loc[mask, col] / totals.loc[mask]

    dt_ssn.drop(columns=["_year"], inplace=True)
    return dt_ssn


def media_ssn_apply(ds: pd.DataFrame
                    , dt_ssn: pd.DataFrame | None = None
                    , path: str | None = None
                    , sheet: str | None = 'data'
                    , fix_53w: bool = False
                    , date_start: str | datetime.date | None = None
                    , date_end: str | datetime.date | None = None
                    , bdg_ssn_cols: set[str] | None = None) -> pd.DataFrame:
    """
    Applies media seasonality to the main dataset. Media seasonality is here for distribution of
    yearly budget by specific periods (months, weeks) within year.

    Media seasonality data to be passed either via ``path`` and ``sheet`` args (to load from
    disc) or as already existing dataframe as ``dt_ssn`` argument.

    :param ds: main dataset
    :param dt_ssn: media seasonal distribution data (optional)
    :param path: path to XLSX file like "seasonality.xlsx" with ``sheet`` sheet
    :param sheet: name of the sheet with option data (only if ``path`` is set), defaults to
        ``"data"``
    :param fix_53w: (deprecated) no longer used — kept for backward compatibility
    :param date_start: start date for applying seasonality (optional)
    :param date_end: end date for applying seasonality (optional)
    :param bdg_ssn_cols: set of bare seasonality column names (without ``opt_media_ssn_`` prefix)
        that represent budget items. Columns in this set are normalized to sum to 1 per year
        and use ``"sum"`` aggregation during cross-granularity conversion. Non-budget columns
        use ``"first"`` aggregation and are not normalized.
    :return: ``ds`` main dataset
    """

    # load

    if not pd.isna(path) and path != "":

        wb = load_workbook(path)
        shts = wb.sheetnames
        wb.close()

        if sheet not in shts:
            fermatrica_error('Error reading XLSX file with media seasonality: ' + path + '. Sheet ' + sheet + ' not found')

        try:
            dt_ssn = pd.read_excel(path, sheet_name=sheet)
        except Exception as e:
            fermatrica_error('Error reading XLSX file with media seasonality: ' + path + '. File not found')

    elif dt_ssn is None:
        fermatrica_error('Error creating media seasonality frame. Not frame nor path to XLSX file is provided')

    else:
        dt_ssn = dt_ssn.copy()

    # get period data

    period_var = dt_ssn.columns.tolist()[0]

    if 'period' not in ds.columns:
         # rather dirty hack for some older datasets without period column, to avoid errors in the next steps. It will be removed later, so no harm is done
        if period_var == 'month':
            ds['period'] = 'month'
        elif 'week' in period_var.lower():
            ds['period'] = 'week'
        elif 'quarter' in period_var.lower():
            ds['period'] = 'quarter'
        elif 'day' in period_var.lower():
            ds['period'] = 'day'

    # remove check row

    if is_string_dtype(dt_ssn[period_var]) | is_object_dtype(dt_ssn[period_var]):
        dt_ssn = dt_ssn[dt_ssn[period_var].str.lower() != 'total']

    # rename seasonality columns and cleanse main dataset

    dt_ssn.columns = ['opt_media_ssn_' + x if x not in [period_var, 'year', 'date', 'period']
                      else x for x in dt_ssn.columns.tolist()]

    cols_to_rm = [x for x in ds.columns if re.match(r'^opt_media_ssn_', x)]

    # save existing seasonality values if date range is specified (only rows outside range)
    saved_ssn_values = None
    outside_mask = None

    if (date_start is not None or date_end is not None) and cols_to_rm:
        # Create mask for rows outside the date range
        outside_mask = pd.Series(False, index=ds.index)
        
        if date_start is not None:
            date_start = pd.to_datetime(date_start)
            outside_mask = outside_mask | (ds['date'] < date_start)
        
        if date_end is not None:
            date_end = pd.to_datetime(date_end)
            outside_mask = outside_mask | (ds['date'] > date_end)
        
        # Save only values outside the date range (vectorized)
        if outside_mask.any():
            saved_ssn_values = ds.loc[outside_mask, cols_to_rm].copy()

    for col in cols_to_rm:
        if col in ds.columns:
            del ds[col]

    # convert seasonality to date-based format and normalize

    dt_ssn = _seasonality_to_dates(dt_ssn, period_var, ds, date_start, date_end)

    bdg_opt_cols: set[str] = set()
    if bdg_ssn_cols:
        bdg_opt_cols = {"opt_media_ssn_" + c for c in bdg_ssn_cols}

    dt_ssn = _normalize_seasonality(dt_ssn, bdg_opt_cols)

    # cross-granularity conversion via period_align

    dt_ssn_period = dt_ssn["period"].iloc[0]
    period_var_trg = ds["period"].iloc[0]

    if dt_ssn_period != period_var_trg:
        opt_cols = [c for c in dt_ssn.columns if c.startswith("opt_media_ssn_")]
        vars_agg_dict = {}
        for col in opt_cols:
            vars_agg_dict[col] = "sum" if col in bdg_opt_cols else "first"

        dt_ssn = period_align(dt_ssn, period_var_trg, vars_agg_dict, None)

    # rescale budget cols to sum to 1 per year within the scope date range

    if (
        date_start is not None
        and date_end is not None
        and bdg_opt_cols
        and any(c in dt_ssn.columns for c in bdg_opt_cols)
    ):
        ssn_start = pd.to_datetime(date_start)
        ssn_end = pd.to_datetime(date_end)
        scope_mask = (dt_ssn["date"] >= ssn_start) & (dt_ssn["date"] <= ssn_end)
        if scope_mask.any():
            dt_ssn["_scope_year"] = dt_ssn["date"].dt.year
            for col in bdg_opt_cols:
                if col not in dt_ssn.columns:
                    continue
                totals = dt_ssn.loc[scope_mask].groupby("_scope_year")[col].sum()
                dt_ssn["_scope_total"] = dt_ssn["_scope_year"].map(totals)
                rescale_mask = scope_mask & (dt_ssn["_scope_total"] > 0)
                if rescale_mask.any():
                    dt_ssn.loc[rescale_mask, col] = (
                        dt_ssn.loc[rescale_mask, col]
                        / dt_ssn.loc[rescale_mask, "_scope_total"]
                    )
            dt_ssn.drop(columns=["_scope_year", "_scope_total"], inplace=True, errors="ignore")

    # merge with data on date

    opt_cols = [c for c in dt_ssn.columns if c.startswith("opt_media_ssn_")]
    dt_ssn["date"] = dt_ssn["date"].astype("datetime64[us]")
    ds = ds.merge(
        dt_ssn[["date"] + opt_cols], how="left", on="date", sort=False, copy=False
    )

    # restore previous values outside the date range (vectorized)
    if saved_ssn_values is not None and outside_mask is not None:
        ssn_cols = [x for x in saved_ssn_values.columns if x in ds.columns]
        if ssn_cols:
            ds.loc[outside_mask, ssn_cols] = saved_ssn_values[ssn_cols]

    return ds


class OptionSettings(StableClass):
    """
    Option settings / controls. Effectively structure rather than class.
    """

    target: list
    date_start: str | datetime.date
    date_end: str | datetime.date
    ref_date_start: str | datetime.date
    ref_date_end: str | datetime.date
    plan_period: str  # 'exact', 'year', 'hy', 'quarter'
    apply_vars: list

    def __init__(
            self
            , target: list | tuple | str
            , date_start: str | datetime.date
            , date_end: str | datetime.date
            , ref_date_start: str | datetime.date
            , ref_date_end: str | datetime.date
            , apply_vars: list | tuple | str = ('superbrand',)
            , plan_period: str = 'year'
    ):
        """
        Initialize class.

        :param target: values of variables listed in `apply_vars` to apply option. Could be string
            "my_brand" or tuple / list of strings or of tuples / lists [["my_brand"], ["market_1", "market_2"]]
        :param date_start: start of the period to apply option
        :param date_end: end of the period to apply option
        :param ref_date_start: start of the reference period (to calculate growth)
        :param ref_date_end: end of the reference period (to calculate growth)
        :param apply_vars: variable to apply option. Could be string
            "superbrand" or tuple / list of strings ["superbrand", "market"]
        :param plan_period: planning period: 'exact', 'year', 'hy', 'quarter'
        """

        if isinstance(target, str):
            target = [target]
        if isinstance(apply_vars, str):
            apply_vars = [apply_vars]

        warnings.warn(
            "OptionSettings is deprecated. Use OptionScope + Option instead. "
            "See MetaModel.compute() and ModelNode.compute() for the new API.",
            DeprecationWarning,
            stacklevel=2,
        )

        self.target = target
        self.date_start = pd.to_datetime(date_start)
        self.date_end = pd.to_datetime(date_end)
        self.ref_date_start = pd.to_datetime(ref_date_start)
        self.ref_date_end = pd.to_datetime(ref_date_end)
        self.plan_period = plan_period
        self.apply_vars = apply_vars
        self.zip_mask = None


def option_settings_list_create(target: list | tuple | str
                                , date_start: str | datetime.date
                                , date_end: str | datetime.date
                                , ref_date_start: str | datetime.date
                                , ref_date_end: str | datetime.date
                                , apply_vars: list | tuple | str = ('superbrand',)
                                , plan_period: str = 'year'
                                , split_period: str = 'year'
                                , options_dict: dict | None = None) -> list[OptionSettings]:
    """
    Create a list of OptionSettings objects.

    :param target: values of variables listed in `apply_vars` to apply option
    :param date_start: start of the period to apply option
    :param date_end: end of the period to apply option
    :param ref_date_start: start of the reference period (to calculate growth)
    :param ref_date_end: end of the reference period (to calculate growth)
    :param apply_vars: variable to apply option
    :param plan_period: planning period: 'exact', 'year', 'hy', 'quarter'
    :param split_period: period to split option into if date range is longer than one period. Could be 'year' as for now
    :param options_dict: dictionary of options to apply
    :return: list of OptionSettings objects or None
    """

    # Convert dates to datetime.date
    if isinstance(date_start, str):
        date_start = pd.to_datetime(date_start).date()
    if isinstance(date_end, str):
        date_end = pd.to_datetime(date_end).date()
    if isinstance(ref_date_start, str):
        ref_date_start = pd.to_datetime(ref_date_start).date()
    if isinstance(ref_date_end, str):
        ref_date_end = pd.to_datetime(ref_date_end).date()
    
    option_settings_list = []

    start_year = date_start.year
    end_year = date_end.year

    if split_period == 'year':
        
        # Split period into yearly intervals (calendar years)
        
        for year in range(start_year, end_year + 1):
            # Determine year boundaries for plan period
            year_start = datetime.date(year, 1, 1) if year > start_year else date_start
            year_end = datetime.date(year, 12, 31) if year < end_year else date_end
            
            # Create OptionSettings object for this year
            option_settings = OptionSettings(
                target=target,
                date_start=year_start,
                date_end=year_end,
                ref_date_start=ref_date_start,
                ref_date_end=ref_date_end,
                apply_vars=apply_vars,
                plan_period=plan_period
            )
            option_settings_list.append(option_settings)
    else:
        # For non-year split_period, create a single OptionSettings
        # (mock in fact, as split_period other than 'year' is not implemented yet)
        option_settings = OptionSettings(
            target=target,
            date_start=date_start,
            date_end=date_end,
            ref_date_start=ref_date_start,
            ref_date_end=ref_date_end,
            apply_vars=apply_vars,
            plan_period=plan_period
        )
        option_settings_list.append(option_settings)

    if options_dict is not None:
        options_list = [copy.deepcopy(x) for k, x in options_dict.items() if k in range(start_year, end_year + 1)]
        return option_settings_list, options_list
    
    else:
        return option_settings_list


def scopes_read(path: str, sheet: str = "scopes"):
    """
    Read scopes sheet from options.xlsx and return a list of OptionScope
    objects.

    Parses the scopes sheet which defines named application scopes — where
    and when budgets should be applied. Handles CSV-formatted
    ``apply_vars`` and ``target`` columns by splitting on commas into the
    new ``filters`` dict format. The ``ref_date_end`` column is optional
    and defaults to ``date_start - 1 day`` when omitted or blank.

    Parameters
    ----------
    path : str
        Path to the options XLSX file (e.g. ``"options.xlsx"``).
    sheet : str, optional
        Sheet name containing scopes data. Defaults to ``"scopes"``.

    Returns
    -------
    list[OptionScope]
        List of OptionScope objects, one per row in the scopes sheet.

    Raises
    ------
    FermatricaError
        If the sheet is missing or required columns (``scope_name``,
        ``date_start``, ``date_end``, ``ref_date_start``) are absent.
    """
    from fermatrica_rep.options.scope import OptionScope

    wb = load_workbook(path)
    shts = wb.sheetnames
    wb.close()

    if sheet not in shts:
        fermatrica_error(
            f"Error reading XLSX file: {path}. Sheet {sheet} not found"
        )

    dt_scopes = pd.read_excel(path, sheet_name=sheet)

    required_cols = {"scope_name", "date_start", "date_end", "ref_date_start", "ref_date_end"}
    missing = required_cols - set(dt_scopes.columns)
    if missing:
        fermatrica_error(
            f"Scopes sheet missing required columns: {missing}"
        )

    scopes = []
    for _, row in dt_scopes.iterrows():
        filters: dict[str, list[str]] = {}
        apply_vars_raw = row.get("apply_vars", None)
        target_raw = row.get("target", None)
        if pd.notna(apply_vars_raw) and pd.notna(target_raw):
            apply_vars_list = [x.strip() for x in str(apply_vars_raw).split(",")]
            target_list = [x.strip() for x in str(target_raw).split(",")]
            for col, val in zip(apply_vars_list, target_list):
                filters.setdefault(col, []).append(val)

        ref_date_end_raw = row.get("ref_date_end", None)
        if pd.isna(ref_date_end_raw) or str(ref_date_end_raw).strip() == "":
            ref_date_end = str(pd.to_datetime(row["date_start"]) - pd.DateOffset(1))
        else:
            ref_date_end = str(ref_date_end_raw)

        seasonality_raw = row.get("seasonality", None)
        seasonality = str(seasonality_raw) if pd.notna(seasonality_raw) and str(seasonality_raw).strip() else None

        plan_period_raw = row.get("plan_period", None)
        plan_period = str(plan_period_raw).strip() if pd.notna(plan_period_raw) and str(plan_period_raw).strip() else "year"

        inflation_multiplier_raw = row.get("inflation_multiplier", None)
        if pd.notna(inflation_multiplier_raw) and str(inflation_multiplier_raw).strip() != "":
            inflation_multiplier = float(inflation_multiplier_raw)
        else:
            inflation_multiplier = 1.0

        scopes.append(
            OptionScope(
                scope_name=str(row["scope_name"]),
                date_start=str(row["date_start"]),
                date_end=str(row["date_end"]),
                ref_date_start=str(row["ref_date_start"]),
                ref_date_end=ref_date_end,
                filters=filters,
                plan_period=plan_period,
                inflation_multiplier=inflation_multiplier,
                seasonality=seasonality,
            )
        )
    return scopes


def media_prices_read(
    path: str, sheet: str = "media_prices"
) -> dict[str, dict[str, float]]:
    """
    Read the media_prices sheet from options.xlsx.

    Returns per-tool inflation multipliers organized as a nested dict
    ``{scope_name: {item: multiplier}}``. These multipliers override the
    scope-level ``inflation_multiplier`` for individual budget items
    within a scope.

    If the sheet is absent from the file, returns an empty dict silently
    (the media_prices sheet is optional).

    Parameters
    ----------
    path : str
        Path to the options XLSX file.
    sheet : str, optional
        Sheet name. Defaults to ``"media_prices"``.

    Returns
    -------
    dict[str, dict[str, float]]
        Nested dict mapping ``{scope_name: {item: multiplier}}``.
        Empty dict if the sheet is not present.
    """
    wb = load_workbook(path)
    shts = wb.sheetnames
    wb.close()

    if sheet not in shts:
        return {}

    dt_prices = pd.read_excel(path, sheet_name=sheet)

    result: dict[str, dict[str, float]] = {}
    for _, row in dt_prices.iterrows():
        scope_name = str(row["scope_name"])
        item = str(row["item"])
        multiplier = float(row["multiplier"])
        result.setdefault(scope_name, {})[item] = multiplier

    return result


def seasonality_read(
    path: str, sheet: str = "seasonality"
) -> dict[str, pd.DataFrame]:
    """
    Read the seasonality sheet from options.xlsx, keyed by scope_name.

    Seasonality data contains weekly or monthly coefficients that adjust
    media variables to reflect seasonal patterns. The sheet is grouped by
    ``scope_name`` — each group becomes a DataFrame with a ``period``
    column and one column per media tool (e.g. ``bdg_tv``,
    ``bdg_digital``).

    If the sheet is absent from the file, returns an empty dict silently
    (the seasonality sheet is optional).

    Parameters
    ----------
    path : str
        Path to the options XLSX file.
    sheet : str, optional
        Sheet name. Defaults to ``"seasonality"``.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dict mapping ``{scope_name: DataFrame}`` where each DataFrame
        has a ``period`` column and columns for each media tool's
        seasonal coefficients. Empty dict if the sheet is not present.

    Raises
    ------
    FermatricaError
        If the sheet exists but has no ``scope_name`` column.
    """
    wb = load_workbook(path)
    shts = wb.sheetnames
    wb.close()

    if sheet not in shts:
        return {}

    dt_ssn = pd.read_excel(path, sheet_name=sheet)

    if "scope_name" not in dt_ssn.columns:
        fermatrica_error(
            "Seasonality sheet must have a 'scope_name' column"
        )

    result: dict[str, pd.DataFrame] = {}
    for scope_name, group in dt_ssn.groupby("scope_name"):
        result[str(scope_name)] = group.drop(columns=["scope_name"]).reset_index(
            drop=True
        )

    return result


def _option_settings_to_scope(opt_set, idx=0):
    """Convert a single OptionSettings to an OptionScope."""
    return OptionScope(
        scope_name=getattr(opt_set, 'prefix', f"scope_{idx}"),
        date_start=opt_set.date_start,
        date_end=opt_set.date_end,
        ref_date_start=getattr(opt_set, 'ref_date_start', opt_set.date_start),
        ref_date_end=getattr(opt_set, 'ref_date_end', opt_set.date_end),
        filters={k: [v] if not isinstance(v, list) else v
                 for k, v in zip(opt_set.apply_vars, opt_set.target)},
        plan_period=getattr(opt_set, 'plan_period', 'year'),
    )


def _convert_old_options(opt_set, options_m):
    """Convert old-style OptionSettings + raw dict to new OptionScope + Option lists.

    Parameters
    ----------
    opt_set : OptionSettings | list[OptionSettings]
        Old OptionSettings object(s).
    options_m : dict
        Dict of option name to raw budget data dict.

    Returns
    -------
    tuple[list[OptionScope], list[Option]]
    """
    if isinstance(opt_set, list):
        scopes = [_option_settings_to_scope(os, i) for i, os in enumerate(opt_set)]
    else:
        scopes = [_option_settings_to_scope(opt_set)]

    first_scope_name = scopes[0].scope_name

    options = []
    for option_name, option_data in options_m.items():
        if isinstance(option_data, list):
            budget_items = {}
            text_flags = None
            for sub_dict in option_data:
                if isinstance(sub_dict, dict):
                    for k, v in sub_dict.items():
                        if k.startswith('bdg_') and k != 'bdg' and not isinstance(v, dict):
                            budget_items[k] = float(v)
                        elif k == 'text_flags':
                            text_flags = str(v)
            budget_data = {s.scope_name: dict(budget_items) for s in scopes}
            options.append(Option(name=option_name, budget_data=budget_data,
                                  text_flags=str(text_flags) if text_flags is not None else None))
        elif isinstance(option_data, dict) and any(isinstance(v, dict) for v in option_data.values()):
            budget_data = {}
            text_flags = None
            for key, items in option_data.items():
                if isinstance(key, int):
                    matching_scopes = [s for s in scopes if s.date_start.year == key]
                    sn = matching_scopes[0].scope_name if matching_scopes else first_scope_name
                    budget_items = {k: float(v) for k, v in items.items()
                                    if k.startswith('bdg_') and k != 'bdg' and not isinstance(v, dict)}
                    budget_data[sn] = budget_items
                elif key == 'text_flags':
                    text_flags = str(items)
            options.append(Option(name=option_name, budget_data=budget_data, text_flags=text_flags))
        else:
            budget_items = {k: float(v) for k, v in option_data.items()
                            if k.startswith('bdg_') and k != 'bdg' and not isinstance(v, dict)}
            text_flags = option_data.get('text_flags')
            budget_data = {s.scope_name: budget_items for s in scopes}
            options.append(Option(name=option_name, budget_data=budget_data,
                                  text_flags=str(text_flags) if text_flags is not None else None))

    return scopes, options

