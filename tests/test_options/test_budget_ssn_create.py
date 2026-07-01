"""
Tests for budget_ssn_create function from fermatrica_rep.options.define module.
"""

import pytest
import pandas as pd
import numpy as np
import os
import yaml
from contextlib import nullcontext

from fermatrica_rep.options.define import budget_ssn_create
from fermatrica.basics.basics import FermatricaError


class TestBudgetSsnCreateFromPath:
    """Tests for budget_ssn_create loading data from Excel file path."""
    
    @pytest.fixture
    def test_data_path(self):
        """Path to test data Excel file."""
        return os.path.join(
            os.path.dirname(__file__),
            'test_cases',
            'budget_ssn_test_data.xlsx'
        )
    
    def test_load_all_sheets_no_vat(self, test_data_path):
        """Load all sheets from Excel file without VAT."""
        budget_dict, seasonality_dict = budget_ssn_create(path=test_data_path, vat_rate=0.0)
        
        # Check that both options are loaded
        assert 'option_alpha' in budget_dict
        assert 'option_beta' in budget_dict
        assert 'option_gamma' in budget_dict
        
        assert 'option_alpha' in seasonality_dict
        assert 'option_beta' in seasonality_dict
        assert 'option_gamma' in seasonality_dict
        
        # Check budget_dict structure for multi-year data (option_alpha)
        assert isinstance(budget_dict['option_alpha'], dict)
        assert 2024 in budget_dict['option_alpha']
        assert 2025 in budget_dict['option_alpha']
        
        # Check that bdg columns exist and total budget is calculated
        alpha_2024 = budget_dict['option_alpha'][2024]
        assert 'bdg' in alpha_2024
        assert 'bdg_channel_A' in alpha_2024
        assert 'bdg_channel_B' in alpha_2024
        
        # Check that nobdg columns exist
        assert 'nobdg_metric_alpha' in alpha_2024
        assert 'nobdg_metric_beta' in alpha_2024
        
        # Verify total budget is sum of bdg_ columns
        bdg_cols = [k for k in alpha_2024.keys() if k.startswith('bdg_')]
        expected_total = sum(alpha_2024[col] for col in bdg_cols)
        assert np.isclose(alpha_2024['bdg'], expected_total)
        
        # Check seasonality_dict structure
        assert isinstance(seasonality_dict['option_alpha'], pd.DataFrame)
        ssn_alpha = seasonality_dict['option_alpha']
        assert 'week_number' in ssn_alpha.columns
        assert 'year' in ssn_alpha.columns
        
        # Check column names are renamed (bdg_ prefix removed)
        assert 'channel_A' in ssn_alpha.columns
        assert 'channel_B' in ssn_alpha.columns
        assert 'metric_alpha' in ssn_alpha.columns  # from nobdg_metric_alpha
        
        # Verify seasonality columns sum to 1 for each year (for bdg columns)
        for year in [2024, 2025]:
            year_data = ssn_alpha[ssn_alpha['year'] == year]
            for col in ['channel_A', 'channel_B', 'channel_C']:
                col_sum = year_data[col].sum()
                assert np.isclose(col_sum, 1.0, atol=1e-5), f"Seasonality for {col} in {year} should sum to 1"
    
    def test_load_specific_sheets_list(self, test_data_path):
        """Load only specific sheets by providing list of names."""
        budget_dict, seasonality_dict = budget_ssn_create(
            path=test_data_path,
            sheets=['option_alpha', 'option_beta'],
            vat_rate=0.0
        )
        
        # Check only specified sheets are loaded
        assert 'option_alpha' in budget_dict
        assert 'option_beta' in budget_dict
        assert 'option_gamma' not in budget_dict
        
        assert 'option_alpha' in seasonality_dict
        assert 'option_beta' in seasonality_dict
        assert 'option_gamma' not in seasonality_dict
    
    def test_load_single_sheet_list(self, test_data_path):
        """Load single sheet by providing list with one name."""
        budget_dict, seasonality_dict = budget_ssn_create(
            path=test_data_path,
            sheets=['option_beta'],
            vat_rate=0.0
        )
        
        # Check only specified sheet is loaded
        assert 'option_beta' in budget_dict
        assert 'option_alpha' not in budget_dict
        assert 'option_gamma' not in budget_dict
    
    def test_load_sheets_regex_pattern(self, test_data_path):
        """Load sheets matching regex pattern."""
        # Load sheets containing 'alpha' or 'beta'
        budget_dict, seasonality_dict = budget_ssn_create(
            path=test_data_path,
            sheets='alpha|beta',
            vat_rate=0.0
        )
        
        assert 'option_alpha' in budget_dict
        assert 'option_beta' in budget_dict
        assert 'option_gamma' not in budget_dict
        
        # Load sheets starting with 'option_'
        budget_dict2, seasonality_dict2 = budget_ssn_create(
            path=test_data_path,
            sheets='^option_',
            vat_rate=0.0
        )
        
        assert len(budget_dict2) == 3
        assert 'option_alpha' in budget_dict2
        assert 'option_beta' in budget_dict2
        assert 'option_gamma' in budget_dict2
    
    def test_vat_rate_application(self, test_data_path):
        """Test that VAT rate is correctly applied to bdg_ columns only."""
        vat_rate = 0.20
        
        # Load without VAT
        budget_no_vat, _ = budget_ssn_create(path=test_data_path, sheets=['option_alpha'], vat_rate=0.0)
        
        # Load with VAT
        budget_with_vat, _ = budget_ssn_create(path=test_data_path, sheets=['option_alpha'], vat_rate=vat_rate)
        
        # Check VAT applied to bdg_ columns
        alpha_no_vat = budget_no_vat['option_alpha'][2024]
        alpha_with_vat = budget_with_vat['option_alpha'][2024]
        
        for col in ['bdg_channel_A', 'bdg_channel_B', 'bdg_channel_C']:
            expected_with_vat = alpha_no_vat[col] * (1 + vat_rate)
            assert np.isclose(alpha_with_vat[col], expected_with_vat), \
                f"VAT not correctly applied to {col}"
        
        # Check VAT NOT applied to nobdg_ columns
        for col in ['nobdg_metric_alpha', 'nobdg_metric_beta']:
            assert np.isclose(alpha_no_vat[col], alpha_with_vat[col]), \
                f"VAT should not be applied to {col}"
        
        # Check total budget includes VAT
        expected_total_with_vat = alpha_no_vat['bdg'] * (1 + vat_rate)
        assert np.isclose(alpha_with_vat['bdg'], expected_total_with_vat), \
            "Total budget should include VAT"
    
    def test_single_year_with_date_column(self, test_data_path):
        """Test handling of single-year data with date column."""
        budget_dict, seasonality_dict = budget_ssn_create(
            path=test_data_path,
            sheets=['option_gamma'],
            vat_rate=0.0
        )
        
        gamma_budget = budget_dict['option_gamma']
        
        # Check year is extracted from date column
        assert 2025 in gamma_budget
        
        # Check budget values exist
        gamma_2025 = gamma_budget[2025]
        assert 'bdg_promo_1' in gamma_2025
        assert 'bdg_promo_2' in gamma_2025
        assert 'bdg' in gamma_2025
        
        # Check seasonality
        gamma_ssn = seasonality_dict['option_gamma']
        assert 'date' in gamma_ssn.columns
        assert 'year' not in gamma_ssn.columns  # Year should be removed when date is present
        assert 'promo_1' in gamma_ssn.columns
        
        # Verify seasonality sums to 1
        col_sum = gamma_ssn['promo_1'].sum()
        assert np.isclose(col_sum, 1.0, atol=1e-5)


class TestBudgetSsnCreateFromDict:
    """Tests for budget_ssn_create loading data from dictionary."""
    
    @pytest.fixture
    def test_data_dict(self):
        """Create test data as dictionary of DataFrames."""
        np.random.seed(123)
        
        # Create option with 2 years of data
        data1 = []
        for year in [2023, 2024]:
            for week in range(1, 53):
                row = {
                    'week_number': week,
                    'year': year,
                    'nobdg_param_x': np.random.uniform(0.9, 1.1),
                    'bdg_activity_A': np.random.uniform(100, 300),
                    'bdg_activity_B': np.random.uniform(50, 150),
                }
                data1.append(row)
        
        df1 = pd.DataFrame(data1)
        
        # Create option with single year (no year column)
        data2 = []
        for month in range(1, 13):
            row = {
                'month': month,
                'nobdg_rate': np.random.uniform(1.0, 1.5),
                'bdg_campaign': np.random.uniform(1000, 5000),
            }
            data2.append(row)
        
        df2 = pd.DataFrame(data2)
        df2['date'] = pd.date_range('2024-01-01', periods=12, freq='MS')
        
        return {
            'test_option_1': df1,
            'test_option_2': df2
        }
    
    def test_load_from_dict(self, test_data_dict):
        """Load data from dictionary of DataFrames."""
        budget_dict, seasonality_dict = budget_ssn_create(dct=test_data_dict, vat_rate=0.0)
        
        # Check both options loaded
        assert 'test_option_1' in budget_dict
        assert 'test_option_2' in budget_dict
        
        # Check multi-year structure
        assert 2023 in budget_dict['test_option_1']
        assert 2024 in budget_dict['test_option_1']
        
        # Check single year structure
        assert 2024 in budget_dict['test_option_2']
        
        # Verify data integrity
        opt1_2023 = budget_dict['test_option_1'][2023]
        assert 'bdg_activity_A' in opt1_2023
        assert 'bdg_activity_B' in opt1_2023
        assert 'nobdg_param_x' in opt1_2023
        assert 'bdg' in opt1_2023
        
        # Check seasonality
        ssn1 = seasonality_dict['test_option_1']
        assert 'week_number' in ssn1.columns
        assert 'activity_A' in ssn1.columns
        assert 'activity_B' in ssn1.columns
        assert 'param_x' in ssn1.columns
    
    def test_dict_with_vat(self, test_data_dict):
        """Test VAT application when loading from dict."""
        vat_rate = 0.15
        
        budget_no_vat, _ = budget_ssn_create(dct=test_data_dict, vat_rate=0.0)
        budget_with_vat, _ = budget_ssn_create(dct=test_data_dict, vat_rate=vat_rate)
        
        opt1_no_vat = budget_no_vat['test_option_1'][2023]
        opt1_with_vat = budget_with_vat['test_option_1'][2023]
        
        # Check VAT applied correctly
        expected = opt1_no_vat['bdg_activity_A'] * (1 + vat_rate)
        assert np.isclose(opt1_with_vat['bdg_activity_A'], expected)


class TestBudgetSsnCreateEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_error_no_input(self):
        """Test error when neither dct nor path is provided."""
        with pytest.raises(FermatricaError) as exc_info:
            budget_ssn_create()
        
        assert 'Either dct or path must be provided' in str(exc_info.value)
    
    def test_error_invalid_path(self):
        """Test error when path doesn't exist."""
        with pytest.raises(FermatricaError) as exc_info:
            budget_ssn_create(path='nonexistent_file.xlsx')
        
        assert 'File not found' in str(exc_info.value)
    
    def test_error_no_budget_columns(self):
        """Test error when no bdg_ or nobdg_ columns found."""
        # Create data without budget columns
        df = pd.DataFrame({
            'week_number': range(1, 11),
            'year': [2024] * 10,
            'some_column': range(10, 20)
        })
        
        with pytest.raises(FermatricaError) as exc_info:
            budget_ssn_create(dct={'bad_option': df})
        
        assert 'No budget columns' in str(exc_info.value)
    
    def test_error_no_year_or_date(self):
        """Test error when neither year nor date column exists."""
        df = pd.DataFrame({
            'week_number': range(1, 11),
            'bdg_item': np.random.uniform(100, 200, 10)
        })
        
        with pytest.raises(FermatricaError) as exc_info:
            budget_ssn_create(dct={'bad_option': df})
        
        assert 'Neither "year" nor "date" column found' in str(exc_info.value)
    
    def test_error_no_sheets_match_filter(self):
        """Test error when sheet filter doesn't match any sheets."""
        test_data_path = os.path.join(
            os.path.dirname(__file__),
            'test_cases',
            'budget_ssn_test_data.xlsx'
        )
        
        with pytest.raises(FermatricaError) as exc_info:
            budget_ssn_create(path=test_data_path, sheets=['nonexistent_sheet'])
        
        assert 'No sheets match' in str(exc_info.value)
    
    def test_empty_data_handling(self):
        """Test handling of options with minimal data."""
        df = pd.DataFrame({
            'week_number': [1, 2],
            'year': [2024, 2024],
            'bdg_single': [100.0, 200.0],
            'nobdg_metric': [1.0, 1.1]
        })
        
        budget_dict, seasonality_dict = budget_ssn_create(
            dct={'minimal_option': df},
            vat_rate=0.0
        )
        
        # Should work with minimal data
        assert 'minimal_option' in budget_dict
        assert 2024 in budget_dict['minimal_option']
        assert budget_dict['minimal_option'][2024]['bdg'] == 300.0
        
        # Check seasonality
        ssn = seasonality_dict['minimal_option']
        assert len(ssn) == 2
        assert np.isclose(ssn['single'].sum(), 1.0)


class TestBudgetSsnCreateSeasonality:
    """Tests for seasonality calculation logic."""
    
    def test_seasonality_normalization(self):
        """Test that seasonality values are properly normalized."""
        # Create data with known values
        df = pd.DataFrame({
            'week_number': [1, 2, 3, 4],
            'year': [2024] * 4,
            'bdg_test': [100.0, 200.0, 300.0, 400.0],  # Total = 1000
            'nobdg_value': [10, 20, 30, 40]  # Should not be normalized
        })
        
        budget_dict, seasonality_dict = budget_ssn_create(
            dct={'test_opt': df},
            vat_rate=0.0
        )
        
        ssn = seasonality_dict['test_opt']
        
        # Check bdg column is normalized
        expected_ssn = [0.1, 0.2, 0.3, 0.4]
        np.testing.assert_array_almost_equal(ssn['test'].values, expected_ssn)
        
        # Check nobdg column has +1 shift applied
        np.testing.assert_array_equal(ssn['value'].values, [11, 21, 31, 41])
    
    def test_multi_year_seasonality_separate(self):
        """Test that seasonality is calculated separately for each year."""
        df = pd.DataFrame({
            'week_number': [1, 2, 1, 2],
            'year': [2023, 2023, 2024, 2024],
            'bdg_item': [100.0, 200.0, 300.0, 600.0]  # Different totals per year
        })
        
        budget_dict, seasonality_dict = budget_ssn_create(
            dct={'multi_year_opt': df},
            vat_rate=0.0
        )
        
        ssn = seasonality_dict['multi_year_opt']
        
        # For 2023: total=300, so seasonality should be [1/3, 2/3]
        ssn_2023 = ssn[ssn['year'] == 2023]['item'].values
        np.testing.assert_array_almost_equal(ssn_2023, [1/3, 2/3], decimal=6)
        
        # For 2024: total=900, so seasonality should be [1/3, 2/3]
        ssn_2024 = ssn[ssn['year'] == 2024]['item'].values
        np.testing.assert_array_almost_equal(ssn_2024, [1/3, 2/3], decimal=6)
    
    def test_zero_budget_handling(self):
        """Test handling of zero or very small budget values."""
        df = pd.DataFrame({
            'week_number': [1, 2, 3],
            'year': [2024] * 3,
            'bdg_item': [0.0, 100.0, 200.0]  # One week with zero
        })
        
        budget_dict, seasonality_dict = budget_ssn_create(
            dct={'zero_test': df},
            vat_rate=0.0
        )
        
        ssn = seasonality_dict['zero_test']
        
        # Should handle zeros gracefully
        expected = [0.0, 1/3, 2/3]
        np.testing.assert_array_almost_equal(ssn['item'].values, expected, decimal=6)
        
        # Total budget should be 300
        assert budget_dict['zero_test'][2024]['bdg'] == 300.0


class TestBudgetSsnCreateIntegration:
    """Integration tests combining multiple features."""
    
    def test_full_workflow_multi_year_multi_option(self):
        """Complete workflow with multiple options and years."""
        # Create comprehensive test data
        data_dict = {}
        
        for opt_num in [1, 2, 3]:
            data = []
            for year in [2022, 2023, 2024]:
                for period in range(1, 13):  # Monthly data
                    row = {
                        'month': period,
                        'year': year,
                        'nobdg_kpi': np.random.uniform(0.8, 1.2),
                    }
                    
                    # Different number of budget items per option
                    for i in range(1, opt_num + 2):
                        row[f'bdg_item_{i}'] = np.random.uniform(100, 500)
                    
                    data.append(row)
            
            data_dict[f'option_{opt_num}'] = pd.DataFrame(data)
        
        # Process with VAT
        budget_dict, seasonality_dict = budget_ssn_create(
            dct=data_dict,
            vat_rate=0.19
        )
        
        # Comprehensive checks
        for opt_num in [1, 2, 3]:
            opt_name = f'option_{opt_num}'
            
            # Check all options loaded
            assert opt_name in budget_dict
            assert opt_name in seasonality_dict
            
            # Check all years present
            for year in [2022, 2023, 2024]:
                assert year in budget_dict[opt_name]
                
                # Check budget integrity
                year_budget = budget_dict[opt_name][year]
                assert 'bdg' in year_budget
                
                # Verify total calculation
                bdg_cols = [k for k in year_budget.keys() if k.startswith('bdg_')]
                expected_total = sum(year_budget[col] for col in bdg_cols)
                assert np.isclose(year_budget['bdg'], expected_total)
            
            # Check seasonality structure
            ssn = seasonality_dict[opt_name]
            assert len(ssn) == 36  # 3 years * 12 months
            assert 'month' in ssn.columns
            assert 'year' in ssn.columns
            
            # Verify seasonality normalization per year
            for year in [2022, 2023, 2024]:
                year_ssn = ssn[ssn['year'] == year]
                for i in range(1, opt_num + 2):
                    col_name = f'item_{i}'
                    col_sum = year_ssn[col_name].sum()
                    assert np.isclose(col_sum, 1.0, atol=1e-5), \
                        f"Seasonality for {col_name} in {year} should sum to 1"

# ============================================================================
# Helper functions for stability tests
# ============================================================================

def parametrize_stability_params(file_name):
    '''Load test parameters from YAML file for stability testing.'''
    pth = os.path.join(
        os.path.dirname(__file__),
        'test_cases',
        file_name
    )
    params = []
    
    with open(pth, mode='rt', encoding='utf-8') as yaml_io:
        yaml_data = yaml.full_load(yaml_io)
        
        for item in yaml_data:
            if 'testing case' in item['description']:
                # Extract test parameters
                test_data = item['test_data']
                vat_rate = item['vat_rate']
                use_path = item['use_path']
                expected_budget = item['expected_budget']
                expected_seasonality = item['expected_seasonality']
                
                # Handle expected error
                if item['exp_err']:
                    exp_err = pytest.raises(FermatricaError)
                else:
                    exp_err = nullcontext()
                
                params.append((
                    test_data,
                    vat_rate,
                    use_path,
                    expected_budget,
                    expected_seasonality,
                    exp_err
                ))
    
    return params


def compare_budget_dicts(actual, expected, atol=1e-5):
    '''
    Compare two budget dictionaries with nested structure.
    
    :param actual: Actual budget dict from function
    :param expected: Expected budget dict
    :param atol: Absolute tolerance for float comparison (default 1e-5 for 5 decimal places)
    '''
    assert actual.keys() == expected.keys(), \
        f'Budget dict keys mismatch. Actual: {actual.keys()}, Expected: {expected.keys()}'
    
    for option_name in expected.keys():
        actual_option = actual[option_name]
        expected_option = expected[option_name]
        
        assert actual_option.keys() == expected_option.keys(), \
            f'Year keys mismatch for {option_name}. Actual: {actual_option.keys()}, Expected: {expected_option.keys()}'
        
        for year in expected_option.keys():
            actual_year = actual_option[year]
            expected_year = expected_option[year]
            
            assert actual_year.keys() == expected_year.keys(), \
                f'Budget keys mismatch for {option_name}[{year}]. Actual: {actual_year.keys()}, Expected: {expected_year.keys()}'
            
            for key in expected_year.keys():
                if isinstance(actual_year[key], str):
                    assert actual_year[key] == expected_year[key], \
                        f'String mismatch for {option_name}[{year}][{key}]. Actual: {actual_year[key]}, Expected: {expected_year[key]}'
                else:
                    assert np.isclose(actual_year[key], expected_year[key], atol=atol), \
                        f'Value mismatch for {option_name}[{year}][{key}]. Actual: {actual_year[key]}, Expected: {expected_year[key]}'


def compare_seasonality_dicts(actual, expected, atol=1e-5):
    '''
    Compare two seasonality dictionaries containing DataFrames.
    
    :param actual: Actual seasonality dict from function
    :param expected: Expected seasonality dict
    :param atol: Absolute tolerance for float comparison (default 1e-5 for 5 decimal places)
    '''
    assert actual.keys() == expected.keys(), \
        f'Seasonality dict keys mismatch. Actual: {actual.keys()}, Expected: {expected.keys()}'
    
    for option_name in expected.keys():
        actual_df = actual[option_name]
        expected_spec = expected[option_name]
        
        # Construct expected DataFrame from spec
        expected_df = pd.DataFrame(expected_spec['data'])
        
        # Check columns match
        assert set(actual_df.columns) == set(expected_df.columns), \
            f'Column mismatch for {option_name}. Actual: {actual_df.columns.tolist()}, Expected: {expected_df.columns.tolist()}'
        
        # Sort both by all columns to ensure same order
        sort_cols = expected_df.columns.tolist()
        actual_sorted = actual_df.sort_values(sort_cols).reset_index(drop=True)
        expected_sorted = expected_df.sort_values(sort_cols).reset_index(drop=True)
        
        # Compare each column
        for col in expected_df.columns:
            if pd.api.types.is_numeric_dtype(expected_sorted[col]):
                np.testing.assert_allclose(
                    actual_sorted[col].values,
                    expected_sorted[col].values,
                    atol=atol,
                    err_msg=f'Column {col} mismatch for {option_name}'
                )
            else:
                pd.testing.assert_series_equal(
                    actual_sorted[col],
                    expected_sorted[col],
                    check_names=False,
                    obj=f'Column {col} for {option_name}'
                )


class TestBudgetSsnCreateStability:
    '''
    Mathematical stability tests comparing actual results with expected outputs.
    Tests verify that function produces exact, reproducible results for known inputs.
    '''
    
    @pytest.fixture
    def stability_data_path(self, tmp_path):
        '''Create temporary Excel file with test data.'''
        return tmp_path / 'stability_test.xlsx'
    
    def create_test_excel(self, test_data, path):
        '''Helper to create Excel file from test data dictionary.'''
        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            for sheet_name, data_dict in test_data.items():
                df = pd.DataFrame(data_dict)
                df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    @pytest.mark.parametrize(
        "test_data,vat_rate,use_path,expected_budget,expected_seasonality,exp_err",
        parametrize_stability_params('budget_ssn_create_stability.yaml')
    )
    def test_budget_ssn_stability(
        self,
        test_data,
        vat_rate,
        use_path,
        expected_budget,
        expected_seasonality,
        exp_err,
        stability_data_path,
        if_debug
    ):
        '''Test mathematical stability of budget_ssn_create against expected results.'''
        
        with exp_err:
            if use_path:
                # Create temporary Excel file
                self.create_test_excel(test_data, stability_data_path)
                
                # Call function with path
                actual_budget, actual_seasonality = budget_ssn_create(
                    path=str(stability_data_path),
                    vat_rate=vat_rate
                )
            else:
                # Call function with dict
                data_dict = {k: pd.DataFrame(v) for k, v in test_data.items()}
                actual_budget, actual_seasonality = budget_ssn_create(
                    dct=data_dict,
                    vat_rate=vat_rate
                )
            
            if if_debug:
                try:
                    # Compare budget dictionaries
                    compare_budget_dicts(actual_budget, expected_budget)
                    
                    # Compare seasonality dictionaries
                    compare_seasonality_dicts(actual_seasonality, expected_seasonality)
                except Exception as e:
                    print(f'\n=== Debug Info ===')
                    print(f'Actual Budget: {actual_budget}')
                    print(f'Expected Budget: {expected_budget}')
                    print(f'\nActual Seasonality:')
                    for k, v in actual_seasonality.items():
                        print(f'{k}:\n{v}')
                    print(f'\nExpected Seasonality: {expected_seasonality}')
                    breakpoint()
            else:
                # Compare budget dictionaries
                compare_budget_dicts(actual_budget, expected_budget)
                
                # Compare seasonality dictionaries
                compare_seasonality_dicts(actual_seasonality, expected_seasonality)
