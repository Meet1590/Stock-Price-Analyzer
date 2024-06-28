import torch
import pytest
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from finbert_utils import estimate_sentiment


tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
device = "cpu"
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)


# Mocking the tokenizer and model
class MockTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, news, return_tensors, padding):
        return self.tokenizer(news, return_tensors=return_tensors, padding=padding)

class MockModel:
    def __init__(self, model):
        self.model = model

    def to(self, device):
        return self.model.to(device)

    def __call__(self, input_ids, attention_mask):
        return {"logits": torch.tensor([[0.1, 0.2, 0.7]])}  # Mock logits

@pytest.fixture
def mock_tokenizer():
    return MockTokenizer(tokenizer)

@pytest.fixture
def mock_model():
    return MockModel(model)

# Test case for estimate_sentiment function
def test_estimate_sentiment_with_news(mock_tokenizer, mock_model):
    # Define input news
    news = "This is a positive news."

    # Call the estimate_sentiment function
    probability, sentiment = estimate_sentiment(news)

    # Check if the probability and sentiment are correct
    assert probability > 0.5
    assert sentiment == "positive"

# Test case for estimate_sentiment function with empty news
def test_estimate_sentiment_with_empty_news(mock_tokenizer, mock_model):
    # Define empty input news
    news = ""

    # Call the estimate_sentiment function
    probability, sentiment = estimate_sentiment(news)

    # Check if the probability and sentiment are correct for empty news
    assert probability == 0
    assert sentiment == "neutral"