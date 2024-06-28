import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

import requests
import csv
import pandas as pd
import copy as cp

################## Data Preprocessing ####################
def create_dataFrame(file_path: str)-> pd.DataFrame:
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print("Error occurred while reading the CSV file:", e)
        return None


#parse row json to arrays
#Helper function for get raw data
def fetch_data(url):
    '''
    This funciton will fetch the data from URL

    params:
    url: link to API with api parameters
    '''
    try:
        data  = requests.get(url)
        return data.json()
    except Exception as e:
        print("Error occurred while fetching data from URL:", e)
        return None

def get_raw_data(symbol, timeframe = "TIME_SERIES_DAILY" ):
    '''
    This function prepares raw dataframe of data fetched from api and make it suitable for data preprocessing by
    transfering it to required format
    params:
    symbol: selected stock ticker by the user
    timeframe: specifies time period of stock data (Defualt: TIME_SERIES_DAILY) (Dynamic inputs from user: future enhancements)
    '''
    try:
        symbol = symbol
        function_timeframe = timeframe
        API_KEY = "EJP6DLJ28OUOXIXV" 
        data_type = "json"
        #API endpoint with variable values
        url = "https://www.alphavantage.co/query?function="+function_timeframe+"&symbol="+symbol+"&outputsize=full&apikey="+API_KEY+"&datatype="+data_type
        data = fetch_data(url)
        if data is not None and "Time Series (Daily)" in data:
            df = pd.DataFrame.from_dict(data["Time Series (Daily)"])
            df = df.transpose()
            return df
        else:
            print("Error: Invalid or empty data returned from the API.")
            return None
    except Exception as e:
        print("Error occurred while fetching raw data:", e)
        return None

# Optional method to check the size and shape of data
def getDataframeSize(stocks_data: pd.DataFrame) -> list[int]:
    '''
    Returns the shape of data as a list

    parameters:
    stocks_data: dataframe containing raw data
    '''
    try:
        return [stocks_data.shape[0],stocks_data.shape[1]]
    except Exception as e:
        print("Error occurred while getting dataframe size:", e)
        return None

# Following function checks for duplicate values if any it will return 
def check_duplicates(df, column: pd.DataFrame) -> pd.DataFrame:
    '''
    Check for duplicate values in the data

    parameters:
    df: raw dataframe
    column: specific colun that neeeds to be checked for duplicates
    '''
    try:
        return df[column].duplicated().value_counts()
    except Exception as e:
        print("Error occurred while checking for duplicates:", e)
        return None

#converting string dates to date time object
def convert_to_dateTimeObject(df: pd.DataFrame, column: str) -> pd.DataFrame:
    '''
    Creates pandas datetime object on specified column

    parameters:
    df: raw dataframe
    column: specific column that needs to be transformed
    '''
    try:
        return pd.to_datetime(df[column])
    except Exception as e:
        print("Error occurred while converting to datetime object:", e)
        return None

#dataframe sorting
def sort_data_frame(df: pd.DataFrame, column: str, ascending: bool = True) -> pd.DataFrame:
    '''
    Sorts dataframe into order of specific column (Default: ascending)

    parameters:
    df: raw dataframe
    column: specific column on which sorting is required
    ascending: specifies sorting order either ascending or descending (Default: ascending)
    '''
    try:
        return df.sort_values(by=column, ascending=ascending) #removed ignore_index = True(11:27, Monday, 8th Apr)
    except Exception as e:
        print("Error occurred while sorting dataframe:", e)
        return None

#renamin columns
def rename_columns(df: pd.DataFrame, columns_mapping: dict) -> pd.DataFrame:
    '''
    Renames dataframe columns acording to input

    parameters:
    df: raw dataframe
    columns_mapping: dict containing old name as a key and new name as a value
    '''
    try:
        return df.rename(columns=columns_mapping)
    except Exception as e:
        print("Error occurred while renaming columns:", e)
        return None

def visualizeClosingPrice(df: pd.DataFrame):
    try:
        plt.figure(figsize=(10, 6))
        sns.lineplot(x='Date', y='Close', data=df)
        plt.title('Closing Stock Price Over Time')
        plt.xlabel('Date')
        plt.ylabel('Closing Stock Price')
        plt.show()
    except Exception as e:
        print("Error occurred while visualizing closing price:", e)
    

################## Feature Engineering ####################

#normalization
def scalingData(data, scaler) -> np.ndarray:
    '''
    Normalize data for smooth model convergence and enhances stability

    parameters:
    data: numpy array with raw data
    '''
    try:
        data = data.values.reshape(-1, 1)
        return scaler.fit_transform(data)
    except Exception as e:
        print("Error occurred while scaling data:", e)
        return None
#lag features
def dataPreparationForModel(closing_prices_df: pd.DataFrame, n_steps: int) -> pd.DataFrame:
    '''
    Generates lag features to feed the LSTM with lagged data points

    parameters:
    closing_prices_df: dataframe 
    n_stpes: steps in the lag (5 in this case)
    '''
    try:
        df = closing_prices_df.copy()
        for i in range(1, n_steps + 1):
            df[f'Close(t-{i})'] = df['Close'].shift(i) #watch first letter of column name
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print("Error occurred while preparing data for the model:", e)
        return None

class StockDataset(Dataset):
    '''
    Creates custom dataset class inherited from Dataset class of pyTorch to utilize utilities of PyTorch

    Attributes:
    X: Raw data
    y: target

    methods:
    __init__(): Constructor for the dataset (Entry point)
    __len__(): Returns the number of samples in a dataset
    __getitem__(): Loads and returns sample on specific index
    '''
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

def createDataLoaders(train_dataset: Dataset, test_dataset: Dataset, batch_size: int) -> tuple:
    try:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader
    except Exception as e:
        print("Error occurred while creating data loaders:", e)
        return None, None

def descaleValues(array: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    try:
        
        return scaler.inverse_transform(array)
    except Exception as e:
        print("Error occurred while descaling values:", e)
        return None

# Main Block
def prepared_data(scaler, data = None):        
    # Load data
    '''
    if data.empty:
        stocks_data = data
    else:
        stocks_data = create_dataFrame(file_path)
    '''
    # Preprocess data
    try:
        scaler = scaler
        #data = pd.read_csv('../daily_IBM.csv')
        #df = pd.DataFrame(data)
        df = data
        df_size = getDataframeSize(df)
        column_names = df.columns.values.tolist()
        df = df.reset_index()
        df = rename_columns(df, {'index': 'Date', column_names[0]: 'Open', column_names[1]: 'High', column_names[2]: 'Low', column_names[3]: 'Close', column_names[4]: 'Volume'})
        duplicated_datetime = check_duplicates(df, 'Date')
        type_of_timestamp = convert_to_dateTimeObject(df, 'Date')
        df = sort_data_frame(df, 'Date', ascending=True)
        # Visualize closing prices

        # Normalize data
        normalized_numpy_array_close = scalingData(df['Close'], scaler)
        df['Close'] = pd.DataFrame(normalized_numpy_array_close, index=df.index)
        df = df[['Date', 'Close']]

        # Prepare data for model
        df.set_index('Date', inplace = True)
        target_df = df['Close'].tail(30)
        targets = target_df.to_numpy().reshape(-1)

        df['Close'] = scalingData(df['Close'], scaler=scaler)
        
        featured_df = dataPreparationForModel(df, 5) #5 represent trading days in a week
        data_for_model = featured_df.iloc[:, 1:].to_numpy()
        data_for_model = cp.deepcopy(np.flip(data_for_model, axis=1))
        
        #extra
        model_data = torch.tensor(data_for_model).float().reshape((-1, 5, 1))

        #calculating batch size
        batch_size = round(len(model_data) ** 0.5)
        return scaler, batch_size, model_data, targets
    except Exception as e:
        print("Error occurred during data preparation:", e)
        return None, None, None

if __name__ == "__main__":
    main()