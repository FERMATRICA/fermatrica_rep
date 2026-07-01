"""
Define option and translation settings and prepare dataset for option calculation.
"""


import copy
import datetime
import pandas as pd
import re
from openpyxl import load_workbook
from pandas.api.types import is_string_dtype, is_object_dtype

from fermatrica_utils import StableClass, ds_merge, date_to_period
from fermatrica.basics.basics import fermatrica_error


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

    if 'item_group' not in dt_trans.columns:
        dt_trans['item_group'] = dt_trans['item'].str.replace('bdg_', '')

    dt_trans = dt_trans.loc[(dt_trans['function'] != '') & (pd.notna(dt_trans['function'])), :]

    dt_trans.set_index('item', inplace=True)
    dct = dt_trans.to_dict(orient='dict')

    return dct


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


def media_ssn_apply(ds: pd.DataFrame
                    , dt_ssn: pd.DataFrame | None = None
                    , path: str | None = None
                    , sheet: str | None = 'data'
                    , fix_53w: bool = False
                    , date_start: str | datetime.date | None = None
                    , date_end: str | datetime.date | None = None) -> pd.DataFrame:
    """
    Applies media seasonality to the main dataset. Media seasonality is here for distribution of
    yearly budget by specific periods (months, weeks) within year.

    Media seasonality data to be passed either via `path` and `sheet` args (to load from
    disc) or as already existing dataframe as `dt_ssn` argument.

    :param ds: main dataset
    :param dt_ssn: media seasonal distribution data (optional)
    :param path: path to XLSX file like "seasonality.xlsx" with `sheet` sheet
    :param sheet: name of the sheet with option data (only if `path` is set), defaults to
        "data"
    :param fix_53w: if False, last week in the year starting 29-31 Dec is counted as 1st week of the next year
    :param date_start: start date for applying seasonality (optional). If provided, seasonality before this date remains unchanged
    :param date_end: end date for applying seasonality (optional). If provided, seasonality after this date remains unchanged
    :return: `ds` main dataset
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

    if period_var not in ds.columns:
        period_var_trg = ds['period'].iloc[0]
    else:
        period_var_trg = period_var

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

    # cast types and prepare merge

    if period_var == 'month':
        dt_ssn[period_var] = pd.to_numeric(dt_ssn.loc[:, period_var])
        ds[period_var] = ds.loc[:, 'date'].dt.month

    elif period_var == 'week_number':
        dt_ssn[period_var] = pd.to_numeric(dt_ssn.loc[:, period_var])

        date_start_full = ds['date'].min()
        date_end_full = ds['date'].max()

        clnd = pd.date_range(date_start_full, date_end_full)
        clnd = pd.DataFrame({'date': clnd})
        clnd["date"] = date_to_period(clnd["date"], period="week")
        clnd = clnd.drop_duplicates()
        
        clnd['week_number'] = clnd['date'].dt.isocalendar().week
        if fix_53w:
            clnd.loc[(clnd['week_number'] == 1) & (clnd['date'].dt.year != clnd['date'].dt.isocalendar().year), 'week_number'] = 53

        if 'year' in dt_ssn.columns:
            clnd['year'] = clnd['date'].dt.year
            dt_ssn = dt_ssn.merge(clnd, how='inner', on=['year', 'week_number'], sort=False, copy=True)
        else:
            dt_ssn = dt_ssn.merge(clnd, how='inner', on=['week_number'], sort=False, copy=True)

        dt_ssn.drop(columns=['week_number', period_var, 'year'], inplace=True, errors='ignore')
        dt_ssn['period'] = 'week'
        period_var = 'date'

    elif period_var == 'day' and period_var_trg in ['day']:
        dt_ssn[period_var] = pd.to_numeric(dt_ssn.loc[:, period_var])
        ds[period_var] = ds.loc[:, 'date'].dt.dayofyear

    # merge with data

    if 'year' in dt_ssn.columns:
        dt_ssn['year'] = pd.to_numeric(dt_ssn.loc[:, 'year'])
        ds['year'] = ds.loc[:, 'date'].dt.year
        ds = ds.merge(dt_ssn, how='left', on=[period_var, 'year'], sort=False, copy=False)

    elif (period_var == 'date') and ('period' in dt_ssn.columns):
        dt_ssn['date'] = pd.to_datetime(dt_ssn.loc[:, 'date'])
        
        cln = dt_ssn.columns.to_list()
        cln = [x for x in cln if re.match(r'^opt_media_ssn_', x)]

        vars_agg_dict = dict.fromkeys(cln, 'sum')

        dim_cols = None
        ds = ds_merge(ds, dt_ssn, ds['period'].iloc[0], vars_agg_dict, dim_cols)

    else:
        ds = ds.merge(dt_ssn, how='left', on=period_var, sort=False, copy=False)

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

