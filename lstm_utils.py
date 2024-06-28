# general imports
import data_utils
import requests
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import MinMaxScaler
import copy as cp

#import for pytorch
import torch
import torch.nn as nn

#import local functions
import data_utils
from data_utils import get_raw_data, DataLoader, descaleValues, getDataframeSize, rename_columns, check_duplicates, convert_to_dateTimeObject, sort_data_frame, scalingData, dataPreparationForModel

#scaler object
scaler = MinMaxScaler(feature_range=(-1, 1))

#LSTM Model Definition
class LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_stacked_layers: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, device):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


#Trainmodel function
def trainModel(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, loss_function: nn.Module, optimizer: torch.optim.Optimizer, num_epochs: int, device: str):
    avg_train_loss = 0.0
    avg_validation_loss = 0.0
    for epoch in range(num_epochs):
        train_loss = trainOneEpoch(model, train_loader, loss_function, optimizer, device)
        avg_train_loss += train_loss
        val_loss = validateOneEpoch(model, test_loader, loss_function, device)
        avg_validation_loss += val_loss
        print(f'Epoch: {epoch + 1}, Train Loss: {train_loss:.8f}, Validation Loss: {val_loss:.8f}')
    return train_loss, val_loss


def trainOneEpoch(model: nn.Module, train_loader: DataLoader, loss_function: nn.Module, optimizer: torch.optim.Optimizer, device):
    model.train(True)
    running_loss = 0.0
    for batch_index, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        output = model(x_batch, device)
        loss = loss_function(output, y_batch)
        running_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return running_loss / len(train_loader)

def validateOneEpoch(model: nn.Module, test_loader: DataLoader, loss_function: nn.Module, device):
    model.train(False)
    running_loss = 0.0
    for batch_index, batch in enumerate(test_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        with torch.no_grad():
            output = model(x_batch, device)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()
    return running_loss / len(test_loader)

def infer(model: nn.Module, X: torch.Tensor, device: str):
    model.eval()
    with torch.no_grad():
        predicted = model(X.to(device), device).to('cpu')
    return predicted

#accuracy measurement
def calculate_accuracy(predictions, targets, threshold=2):
    correct = 0
    for pred, target in zip(predictions, targets):
        if abs(pred - target) <= threshold:
            correct += 1
    accuracy = correct / len(predictions) * 100
    return accuracy


def prepared_data(data = None):        
    # Load data
    '''
    if data.empty:
        stocks_data = data
    else:
        stocks_data = create_dataFrame(file_path)
    '''
    # Preprocess data
    try:
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
        df.tail(5)

        target_df = df['Close'].tail(30)
        targets = target_df.to_numpy().reshape(-1)
        # Normalize data
        normalized_numpy_array_close = scaler.fit_transform(df['Close'].values.reshape(-1,1))
        df['Close'] = pd.DataFrame(normalized_numpy_array_close, index=df.index)
        df = df[['Date', 'Close']]

        # Prepare data for model
        df.set_index('Date', inplace = True)
        featured_df = dataPreparationForModel(df, 5) #5 represent trading days in a week
        data_for_model = featured_df.iloc[:, 1:].to_numpy()
        data_for_model = cp.deepcopy(np.flip(data_for_model, axis=1))
        
        #extra
        model_data = torch.tensor(data_for_model).float().reshape((-1, 5, 1))

        #calculating batch size
        batch_size = round(len(model_data) ** 0.5)
        return batch_size, model_data, targets
    except Exception as e:
        print("Error occurred during data preparation:", e)
        return None, None, None

#predict using lstm
def get_predictions(symbol, n):
    #unpacking args
    symbol = symbol

    #loads pretrained model
    model = torch.load('./best_trained model.pth')
    device = 'cpu'
    model.to(device)
    #evaluation mode enabled
    model.eval()

    #data prep
    stock_data = get_raw_data(symbol)
    #stock_data = pd.read_csv('../daily_IBM.csv')
    stock_df = pd.DataFrame(stock_data)
    batch_size, data, targets = prepared_data(data=stock_df)
    #scaler = MinMaxScaler(feature_range=(-1, 1))
    n = n

    # Make predictions
    with torch.no_grad():
        predictions =  infer(model, data, 'cpu').flatten().numpy()
        last_row_with_5_col = []
        last_30_predictions = [predictions[-1]]
        for i in range(5):
            last_row_with_5_col.append( predictions[-i])
        predictions = scaler.inverse_transform(predictions[-n:].reshape(-1,1)).reshape(-1)
        targets = scaler.inverse_transform(targets.reshape(-1,1)).reshape(-1)
        accuracy = calculate_accuracy(predictions, targets)
        print(f"accuracy of model is for last 30 samples is {accuracy:.2f}")
        next_n_predictions = next_n_days(model, n, last_30_predictions, last_row_with_5_col, scaler)
    return next_n_predictions, stock_df, accuracy

#predictions for future n days
def next_n_days(model, n, last_30_predictions, last_row_with_5_col, scaler):
    last_row_with_5_col_arr = np.array(last_row_with_5_col)
    last_row_with_5_col_arr = last_row_with_5_col_arr.reshape(-1, 5, 1)
    new_data = torch.tensor(last_row_with_5_col_arr).float()
    for i in range(n):
        out = infer(model, new_data, 'cpu').flatten().numpy()
        last_30_predictions.append(out[-1])
        last_row_with_5_col.append(out[-1])
        last_row_with_5_col.pop(0)
        last_row_with_5_col_arr = np.array(last_row_with_5_col)
        last_row_with_5_col_arr = last_row_with_5_col_arr.reshape(-1, 5, 1)
        new_data = torch.tensor(last_row_with_5_col_arr).float()
    pred_array = np.array(last_30_predictions).reshape(-1, 1)
    predictions  = scaler.inverse_transform(pred_array).reshape(-1)
    return predictions

def main():
    next_n_predictions, raw_data, accuracy = get_predictions('AAPL')
    #print(f"{next_n_predictions}, raw data and accuracy {accuracy}")

if __name__ == '__main__':
    main()