import pytest
import pandas as pd
import numpy as np
import torch
import copy as cp
from lstm_utils import prepared_data, get_predictions

@pytest.fixture
def sample_data():
    data = pd.read_csv('../daily_IBM.csv')
    df = pd.DataFrame(data)
    return df

def test_prepared_data(sample_data):
    batch_size, model_data, targets = prepared_data(sample_data)
    assert isinstance(batch_size, int)
    assert isinstance(model_data, torch.Tensor)
    assert isinstance(targets, np.ndarray)
    assert len(model_data.shape) == 3  # Check if the shape of model_data is (batch_size, sequence_length, input_size)
    assert model_data.shape[1] == 5  # Check if the sequence length is correct
    assert model_data.shape[2] == 1  # Check if the input size is correct
    assert len(targets) == 30  # Check if the number of targets is correct

@pytest.fixture
def n():
    return 10
@pytest.mark.parametrize("symbol", ["AAPL", "GOOGL", "MSFT"])
def test_get_predictions(symbol, n):
    # Call the function and capture the output
    predictions, data, accuracy = get_predictions(symbol, n)

    # Assertions
    assert isinstance(predictions, np.ndarray), "Predictions should be a numpy array"
    assert isinstance(data, pd.DataFrame), "Data should be a PyTorch tensor"
    assert isinstance(accuracy, float), "Accuracy should be a float"
    assert 0 <= accuracy <= 100, "Accuracy should be a percentage between 0 and 100"
