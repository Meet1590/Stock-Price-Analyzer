import pandas as pd
import numpy as np
import time
import copy as cp

import torch
import torch.nn as nn

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import  train_test_split, KFold
from torch.utils.data import Dataset, DataLoader
from lstm_utils import trainModel, trainOneEpoch, validateOneEpoch, LSTM, calculate_accuracy

scaler = MinMaxScaler(feature_range=(-1,1))

from data_utils import convert_to_dateTimeObject, sort_data_frame, scalingData,dataPreparationForModel,createDataLoaders, StockDataset, descaleValues, DataLoader

df = pd.read_csv("../daily_TSCO.csv", nrows=None)

df  = df.rename(columns = {"timestamp":"Date"})

df = df[["Date", "close"]]
convert_to_dateTimeObject(df, 'Date')
df.set_index('Date', inplace = True)

df = sort_data_frame(df, column = 'Date', ascending=True)
df['close'] = scalingData(df['close'], scaler=scaler)
target_df = df.tail(30)
df = df.drop(df.tail(30).index)
targets_x = target_df.to_numpy().reshape((-1,1,1))
targets = target_df.to_numpy().reshape(-1)

data = dataPreparationForModel(df, 5)#5 represent trading days in a week

featured_data  = pd.DataFrame(data)
X_data = featured_data.iloc[:, 1:].to_numpy().reshape((-1,5,1))
X_data = cp.deepcopy(np.flip(X_data, axis=1))
y_targets = featured_data.iloc[:, 0].to_numpy().reshape((-1,1))

#hyper parameters
input_size = 1
hidden_size = 1
num_stacked_layers= 3
device = 'cpu'#'cuda:0' if torch.cuda.is_available() else 'cpu' for dynamic checking of gpu or cpu
learning_rate = 0.01
num_of_epochs = 10
loss_function = nn.MSELoss()



#Hold-out validation
def hold_out_split(X_data, y_targets, train_test_split):
    X = X_data
    y = y_targets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)
    return X_train, X_test, y_train, y_test

def convert_to_tensors(X_train, X_test, y_train, y_test, torch):
    X_train = torch.tensor(X_train).float()
    X_test = torch.tensor(X_test).float()
    y_train = torch.tensor(y_train).float()
    y_test = torch.tensor(y_test).float()
    return X_train, X_test, y_train, y_test


def hold_out_method(X_train, X_test, y_train, y_test, StockDataset, DataLoader, input_size, hidden_size, num_stacked_layers, targets_x, targets, num_of_epochs, loss_function):

    batch_size = round(X_train.shape[0] ** 0.5)

    # prepare dataset and loaders
    train_dataset = StockDataset(X_train, y_train) #initializing train_dataset with tensors and dataset utilities
    test_dataset = StockDataset(X_test, y_test) ##initializing test_dataset with tensors and dataset utilities

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = LSTM(input_size, hidden_size, num_stacked_layers)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    start_time = time.time()
    train_loss, val_loss = trainModel(model, train_loader, test_loader, loss_function, optimizer, num_of_epochs, device)
    print(f"model's training for {num_of_epochs} is resulted in the model with train loss of {train_loss:.10f} and validation loss of {val_loss:.10f}")
    end_time = time.time()
    model.eval()
    with torch.no_grad():
        targets_x = torch.tensor(targets_x).float()
        targets_x = targets_x.to(device)
        #test_dataset = StockDataset(targets_x, targets_x)
        #test_loader = DataLoader(test_dataset, batch_size=30, shuffle=False)
        test_pred= model(targets_x, device)
        test_pred = test_pred.cpu().numpy()
        targets = scaler.inverse_transform(targets.reshape(-1,1)).reshape(-1)
        predictions = scaler.inverse_transform(test_pred.reshape(-1,1))
        predictions = predictions.reshape(-1)  # Convert predictions to numpy array
        print(targets[-5:])
        print(predictions[-5:])
        accuracy = calculate_accuracy(predictions, targets)
    print("accuracy of the model is ", accuracy)
    print(f"time taken in hold-out execution for {num_of_epochs} is {end_time-start_time}")

X_train, X_test, y_train, y_test = hold_out_split(X_data, y_targets, train_test_split)
X_train, X_test, y_train, y_test = convert_to_tensors(X_train, X_test, y_train, y_test, torch) # avoid input clashing
#hold_out_method( X_train, X_test, y_train, y_test, StockDataset, DataLoader, input_size, hidden_size, num_stacked_layers,targets_x, targets, num_of_epochs, loss_function)

'''
def kfold_cross_validation(X, y, StockDataset, DataLoader, loss_function, num_epochs, k_folds=10, device='cpu'):
    """
    Perform k-fold cross-validation for a PyTorch model.

     Args:
        model (torch.nn.Module): PyTorch model to be trained and evaluated.
        X (torch.Tensor): Input features tensor.
        y (torch.Tensor): Target tensor.
        loss_function: Loss function for training.
        optimizer: Optimizer for training.
        num_epochs (int): Number of training epochs.
        k_folds (int): Number of folds for cross-validation. Default is 5.
        device (str): Device to run the computations ('cpu' or 'cuda'). Default is 'cpu'.

    Returns:
        List of floats: Mean validation loss for each fold.
    """
    # Initialize k-fold cross-validation
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    device = device
    validation_losses = []

    #model and other imitialization 
    model = LSTM(input_size, hidden_size, num_stacked_layers)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Iterate over each fold
    start_time = time.time()
    for fold, (train_index, val_index) in enumerate(kf.split(X)):
        fold_start_time = time.time()
        print(f'Fold {fold + 1}/{k_folds}')
        
        # Split data into training and validation sets for this fold
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        # Convert data to PyTorch tensors
        X_train, X_val, y_train, y_val = convert_to_tensors(X_train, X_val, y_train, y_val, torch)

        #treain and test loaders
        # prepare dataset and loaders
        train_dataset = StockDataset(X_train, y_train) #initializing train_dataset with tensors and dataset utilities
        val_dataset = StockDataset(X_val, y_val) ##initializing test_dataset with tensors and dataset utilities

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Train the model
        epoch_start_time = time.time()
        avg_train_loss, avg_val_loss = trainModel(model, train_loader, test_loader, loss_function, optimizer, num_epochs, device)
        print(f"model's training for {num_of_epochs} is resulted as avg training loss of {avg_train_loss:.10f} and avg validation loss of {avg_val_loss:.10f}")

        for epoch in range(num_epochs):
            train_loss = trainOneEpoch(model, train_loader, loss_function, optimizer, device)
            val_loss= validateOneEpoch(model, test_loader, loss_function, device)
            f"val loss is {val_loss}, for fold {fold+1}"
        epoch_end_time = time.time()
        print(f"time taken for 1000 epoch in {fold+1} fold is {epoch_end_time- epoch_start_time}")
        # Evaluate the model on validation data

        model.eval()
        with torch.no_grad():
            y_pred_val = model(X_val_tensor, device)
            val_loss = loss_function(y_pred_val, y_val_tensor)
            print(f"val loss is {val_loss}, for fold {fold+1}")
            validation_losses.append(val_loss.item())
        fold_end_time = time.time()
        print(f"fold {fold+1} took {fold_end_time-fold_end_time} time")
        avg_evaluaiton_loss = (validation_losses)/10
    end_time = time.time()
    print("time taken in execution of k fold", (end_time-start_time) )
    print(f"evaluation loss for model with k-fold cross validation is {avg_evaluaiton_loss} ")
    return 

kfold_cross_validation(X_data, y_targets, Dataset, DataLoader, loss_function, num_epochs=10, k_folds=10, device='cpu')
'''


def kfold_cross_validation(X, y, StockDataset, DataLoader, trainModel, loss_function, input_size, hidden_size, num_stacked_layers, learning_rate, targets_x, targets, num_epochs, k_folds=10, device='cpu'):
    """
    Perform k-fold cross-validation for a PyTorch model.

    Args:
        X (torch.Tensor): Input features tensor.
        y (torch.Tensor): Target tensor.
        StockDataset (class): Custom dataset class inherited from Dataset class of PyTorch.
        DataLoader (class): DataLoader class from PyTorch.
        trainModel (function): Function to train the model.
        loss_function: Loss function for training.
        input_size (int): Input size for the LSTM model.
        hidden_size (int): Hidden size for the LSTM model.
        num_stacked_layers (int): Number of stacked LSTM layers.
        learning_rate (float): Learning rate for optimizer.
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size for DataLoader.
        k_folds (int): Number of folds for cross-validation. Default is 10.
        device (str): Device to run the computations ('cpu' or 'cuda'). Default is 'cpu'.

    Returns:
        List of floats: Mean validation loss for each fold.
    """
    # Initialize k-fold cross-validation
    #data splitting using sklearns predefined KFold method
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    device = device
    validation_losses = []
    best_model = None
    best_validation_loss = float('inf')

    # Iterate over each fold
    start_time = time.time()
    for fold, (train_index, val_index) in enumerate(kf.split(X)):
        print(f'Fold {fold + 1}/{k_folds}')
        
        # Split data into training and validation sets for this fold
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        # Convert data to PyTorch tensors
        X_train, X_val, y_train, y_val = X_train, X_test, y_train, y_test = convert_to_tensors(X_train, X_val, y_train, y_val, torch) 
        X_train, X_val, y_train, y_val = X_train.to(device), X_val.to(device), y_train.to(device), y_val.to(device)

        batch_size = round(X_train.shape[0] ** 0.5)
        # Initialize the model
        model = LSTM(input_size, hidden_size, num_stacked_layers)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # Initialize train and validation datasets
        train_dataset = StockDataset(X_train, y_train)
        val_dataset = StockDataset(X_val, y_val)

        # Initialize train and validation data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Train the model
        # Train the model
        epoch_start_time = time.time()
        for epoch in range(num_epochs):
            train_loss = trainOneEpoch(model, train_loader, loss_function, optimizer, device)
            val_loss= validateOneEpoch(model, val_loader, loss_function, device)
        #print(f"train loss  for fold {fold+1} is {val_loss}")
        epoch_end_time = time.time()
        #print(f"time taken for {num_epochs} epoch in {fold+1} fold is {epoch_end_time- epoch_start_time}")
        # Evaluate the model on validation data
        model.eval()
        with torch.no_grad():
            y_pred_val = model(X_val, device)
            val_loss = loss_function(y_pred_val, y_val)
            #print(f"for fold {fold+1} val loss is {val_loss}")
            validation_losses.append(val_loss.item())
        if val_loss < best_validation_loss:
            best_validation_loss = val_loss
            best_model = model
    model.eval()
    with torch.no_grad():
        targets_x = torch.tensor(targets_x).float()
        targets_x = targets_x.to(device)
        #test_dataset = StockDataset(targets_x, targets_x)
        #test_loader = DataLoader(test_dataset, batch_size=30, shuffle=False)
        test_pred= best_model(targets_x, device)
        test_pred = test_pred.cpu().numpy()
        targets = scaler.inverse_transform(targets.reshape(-1,1)).reshape(-1)
        predictions = scaler.inverse_transform(test_pred.reshape(-1,1))
        predictions = predictions.reshape(-1)  # Convert predictions to numpy array
        print(targets)
        print(predictions)
        accuracy = calculate_accuracy(predictions, targets)
    end_time = time.time()
    print(f"Time taken for execution of k-fold: {end_time - start_time:.2f} seconds")
    print(f"Mean validation loss for each fold: {validation_losses}")
    print(f"End validation mode is {min(validation_losses)}")
    print(f"accuracy of the model is {accuracy}")
    return best_model

# Example usage:
best_model = kfold_cross_validation(X_data, y_targets, StockDataset, DataLoader, trainModel, loss_function, input_size, hidden_size, num_stacked_layers, learning_rate, targets_x, targets, num_epochs = num_of_epochs,   k_folds=10, device='cpu')

# In another file or script
import torch

# save the model
model = torch.save(best_model, 'best_trained model.pth')