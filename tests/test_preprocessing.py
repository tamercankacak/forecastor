
import os
import sys
import pandas as pd
import pytest

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from preprocessing import DataPreprocessor

@pytest.fixture
def preprocessor_instance():
    """Provides a DataPreprocessor instance initialized with sample data."""
    sample_data_path = os.path.join(os.path.dirname(__file__), 'sample_data.csv')
    return DataPreprocessor(file_path=sample_data_path)

def test_preprocess_output_shapes(preprocessor_instance):
    """
    Tests the output shapes of the preprocess method.
    The sample data has 20 rows. With lags/rolling means, we lose some rows.
    With a lag of 30, all rows would be dropped. Let's adjust the preprocessor for a realistic test.
    For this test, we can manually create features to avoid dropping all rows.
    """
    # Since the default lag is 30, our small 20-row sample will result in an empty dataframe.
    # This test should confirm that behavior.
    with pytest.raises(ValueError, match="resulting train set will be empty"):
         preprocessor_instance.preprocess()

def test_feature_engineering(preprocessor_instance):
    """Tests the feature engineering logic on a small dataframe."""
    df = preprocessor_instance._load_and_clean_data()
    df = preprocessor_instance._feature_engineer(df)

    # Check if new columns are created
    expected_new_cols = ['Month', 'Week_of_Year', 'Day_of_Week', 'Sales_Lag_7', 'Sales_Lag_30', 'Rolling_Mean_7']
    for col in expected_new_cols:
        assert col in df.columns

    # After engineering on our tiny dataset, the first 7 rows for Product_001 should have NaN in Sales_Lag_7
    assert df[df['Product_ID'] == 'Product_001']['Sales_Lag_7'].iloc[:7].isnull().all()
    # The 8th row (index 7) should have the value from the 1st row (index 0)
    assert df[df['Product_ID'] == 'Product_001']['Sales_Lag_7'].iloc[7] == df[df['Product_ID'] == 'Product_001']['Sales_Quantity'].iloc[0]

def test_full_preprocessing_with_sufficient_data():
    """
    Tests the preprocessor with a slightly larger, self-contained dataframe
    that won't be entirely dropped after creating lag features.
    """
    # Create a dataframe with 40 days of data for one product
    dates = pd.date_range(start="2024-01-01", periods=40, freq='D')
    data = {
        'Date': dates,
        'Product_ID': ['Product_001'] * 40,
        'Region': ['Region_1'] * 40,
        'Sales_Quantity': [100 + i for i in range(40)],
        'Inventory_Level': [500 - i for i in range(40)]
    }
    df = pd.DataFrame(data)
    
    # Save to a temporary csv to use with the preprocessor
    temp_csv_path = 'tests/temp_test_data.csv'
    df.to_csv(temp_csv_path, index=False)
    
    preprocessor = DataPreprocessor(file_path=temp_csv_path)
    X_train, X_test, y_train, y_test, test_dates = preprocessor.preprocess()
    
    # After creating lag features up to 30 days, we should have 40 - 30 = 10 rows left.
    # Train/test split is 80/20. 8 rows for train, 2 for test.
    assert X_train.shape[0] == 8
    assert X_test.shape[0] == 2
    assert y_train.shape[0] == 8
    assert y_test.shape[0] == 2
    assert test_dates.shape[0] == 2 # There should be a date for each test sample

    # Clean up the temporary file
    os.remove(temp_csv_path)
