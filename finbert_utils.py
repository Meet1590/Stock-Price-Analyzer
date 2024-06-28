from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Tuple
from alpaca_trade_api import REST
from timedelta import Timedelta
from datetime import datetime, timedelta, date

import random
from stocknews import StockNews

device = "cuda:0" if torch.cuda.is_available() else "cpu"

#tokenizer integration for pre training data
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
# model integration to infer it with pre trained data
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)
labels = ["positive", "negative", "neutral"]

def estimate_sentiment(news):
    if news:
        tokens = tokenizer(news, return_tensors="pt", padding=True).to(device)
        result = model(tokens["input_ids"], attention_mask=tokens["attention_mask"])[
            "logits"
        ]
        result = torch.nn.functional.softmax(torch.sum(result, 0), dim=-1)
        probability = result[torch.argmax(result)]
        sentiment = labels[torch.argmax(result)]
        return probability, sentiment
    else:
        return 0, labels[-1]



############### ------------- !!!!!! ------------- ###############
API_KEY = "PK791PTJ0AW8LN16WJT0"
API_SECRET = "6Fgy47b1JAB04Vynjqflw9TMwvZjRVsk2feAsL0W"
BASE_URL = "https://data.alpaca.markets/v1beta1/news?symbol="
api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=API_SECRET)

def get_dates():
    today = date.today()
    a_week_prior = today - Timedelta(days = 30)
    return today.strftime('%Y-%m-%d'), a_week_prior.strftime('%Y-%m-%d')

def get_sentiment():
    symbols = ( "AAPL", "MSFT", "HSBC", "TSCO", "GOOG", 'BIDU','GOOGL','META','NFLX','RBLX','SNAP','T','TKO','TTD' )
    symbol = random.choice(symbols)
    print(symbol)
    today, a_week_prior = get_dates()
    news = api.get_news(symbol, start = a_week_prior, end = today)
    news = [ev.__dict__["_raw"]["headline"] for ev in news]
    probability, sentiment = estimate_sentiment(news)
    print(probability, sentiment)
    return probability, sentiment, news

def main():
    get_sentiment()
    #sentiments = [estimate_sentiment(article['title']) for article in news_articles]
    #sentiment_counts = {'positive': sentiments.count('positive'),
    #                'negative': sentiments.count('negative'),
    #                'neutral': sentiments.count('neutral')}

    #print(sentiment_counts)

if __name__ == "__main__":
    main()
    #tensor, sentiment = estimate_sentiment(['markets responded Positively to the news!','traders were pleased!'])
    #print(tensor, sentiment)
    #print(torch.cuda.is_available())