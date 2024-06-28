import pytest
import pandas as pd
import requests
from data_utils import prepared_data, get_raw_data, convert_to_dateTimeObject, check_duplicates


@pytest.fixture
def sample_data():
    data = pd.read_csv('../daily_IBM.csv')
    df = pd.DataFrame(data)
    return df

# Mock response data for the API
@pytest.fixture
def mock_api_response():
    return {
        "Time Series (Daily)": {
            "2023-04-10": {"Open": "100.00", "High": "105.00", "Low": "95.00", "Close": "102.50", "Volume": "100000"},
            "2023-04-09": {"Open": "98.00", "High": "103.00", "Low": "96.00", "Close": "101.00", "Volume": "95000"},
        }
    }

# Test case for valid response from the API
def test_valid_response(mock_api_response, mocker):
    # Mock the requests.get function
    mock_get = mocker.patch('requests.get')
    mock_get.return_value.json.return_value = mock_api_response

    # Call the function with mock symbol and timeframe
    symbol = "AAPL"
    timeframe = "TIME_SERIES_DAILY"
    df = get_raw_data(symbol, timeframe)
    
    # Check if the DataFrame is not None and has the expected columns
    assert df is not None
    assert "Open" in df.columns
    assert "High" in df.columns
    assert "Low" in df.columns
    assert "Close" in df.columns
    assert "Volume" in df.columns

    # Check DataFrame's expected number of rows
    expected_num_rows = len(mock_api_response["Time Series (Daily)"])
    assert len(df) == expected_num_rows

# Test case to capture invalid response from the API
def test_invalid_response(mocker):
    # dummy empty response from the API
    mocker.patch('requests.get', return_value={})

    # function call  with mock symbol and timeframe
    symbol = "InvalidSymbol"
    timeframe = "TIME_SERIES_DAILY"
    df = get_raw_data(symbol, timeframe)

    #checks if the DataFrame is None
    assert df is None


# Test case for check_duplicates function
def test_check_duplicates():
    # Create a sample DataFrame with duplicate and non-duplicate values
    data = {
        'A': [1, 2, 3, 3, 4],
        'B': [5, 6, 7, 8, 9]
    }
    df = pd.DataFrame(data)

    # Test the function with a column containing duplicates
    duplicates_count = check_duplicates(df, 'A')

    # Check if the result is as expected
    assert duplicates_count[True] == 1  # There is 1 duplicate value

    # Test the function with a column containing no duplicates
    duplicates_count = check_duplicates(df, 'B')

    # Check if the result is as expected
    assert duplicates_count.get(True) is None  # No duplicate values

# Test case for convert_to_dateTimeObject function
def test_convert_to_dateTimeObject():
    # Create a sample DataFrame with date strings
    data = {
        'Date': ['2023-04-10', '2023-04-11', '2023-04-12']
    }
    df = pd.DataFrame(data)

    # Test the function to convert the 'Date' column to datetime object
    df['Date'] = convert_to_dateTimeObject(df, 'Date')

    # Check if the column is converted to datetime object
    assert isinstance(df['Date'][0], pd.Timestamp)  # Check if the first element is a Timestamp object
    assert df['Date'][0].strftime('%Y-%m-%d') == '2023-04-10'  # Check if the date is converted correctly

    # Test the function with an empty DataFrame
    empty_df = pd.DataFrame()

    # Convert an empty DataFrame, should return None
    result_df = convert_to_dateTimeObject(empty_df, 'Date')

    # Check if the result is None
    assert result_df is None