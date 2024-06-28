'''
import pytest
from lstm_training import kfold_cross_validation
from test_lstm_utils import sample_data

@pytest.fixture
def synthetic_data():
    return X_train, y_train, X_test, y_test

def test_kfold_cross_validation():
    #unpack synthetic data
    X_train, y_train, X_test, y_test = synthetic_data
    
    #define hyperparameters and other required inputs
    input_size = 10
    hidden_size = 64
    num_stacked_layers = 2
    learning_rate = 0.001
    num_epochs = 10
    k_folds = 5
    
    #perform k-fold cross-validation
    best_model = kfold_cross_validation(
        X_train, y_train, StockDataset, DataLoader,
        trainOneEpoch, validateOneEpoch, calculate_accuracy,
        input_size, hidden_size, num_stacked_layers, learning_rate,
        X_test, y_test, num_epochs, k_folds
    )

    #assert that the best_model is not None
    assert best_model is not None
'''