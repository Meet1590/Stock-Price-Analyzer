# general imports
import streamlit as st #python webapplication framework
import pandas as pd
import altair as alt
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import random

#data sources
import yfinance as yf
import yahoo_fin.stock_info as si
from stocknews import StockNews

#personal module imports
from lstm_utils import get_predictions
import finbert_utils

st.set_page_config(
    page_title="Stock Analyzer and Predictor Dashboard",
    page_icon="Analyzer",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")


class Stock:
    def __init__(self, stock_ticker: str):
        self.info = yf.Ticker(selected_stock).info
        self.ticker = stock_ticker
        self.currentPrice = self.info['currentPrice']
        self.last_close = self.info['regularMarketPreviousClose']

    def show_stock_data(self, st, selected_stock):
        pricing_data, fundamental_data, news = st.tabs(["pricing_data", "fundamental_data", "news"])
        
        with pricing_data:
            self.show_pricing_data(selected_stock)
        
        with fundamental_data:
            self.show_fundamental_data(selected_stock)
        
        with news:
            self.show_news(selected_stock)
            st.subheader("Some good readings")
            news = yf.Ticker(selected_stock).news
            for article in news:
                heading = f"{article['title']} ({article['publisher']})"
                st.subheader(heading)
                st.markdown(f"<sub>{article['type']} published at {article['providerPublishTime']}</sub>", unsafe_allow_html=True)
                st.markdown(f"<a>{article['link']}</a>", unsafe_allow_html=True)


    def show_pricing_data(self, selected_stock):
        st.header(f"Pricing Data for {selected_stock}")
        #pricing_data = si.get_quote_table(selected_stock)
        #for key, value in pricing_data.items():
        #    st.write(f"{key}: {value}")
    
    def download_fundamental_data(self, st, symbol):
        # Download fundamental data for the specified stock symbol
        stock = yf.Ticker(symbol)
        
        # Get information about the company
        info = stock.info
        print("Company Information:")
        for key, value in info.items():
            st.write(f"{key}: {value}")
        
        # Get the balance sheet
        balance_sheet = stock.balance_sheet
        st.write("\nBalance Sheet:")
        st.write(balance_sheet)
        
        # Get the income statement
        income_statement = stock.financials
        st.write("\nIncome Statement:")
        st.write(income_statement)
        
        # Get the cash flow statement
        cash_flow = stock.cashflow
        st.write("\nCash Flow Statement:")
        st.write(cash_flow)

    def show_fundamental_data(self, selected_stock):
        st.header(f"Fundamental Data for {selected_stock}")
        self.download_fundamental_data(st, selected_stock)
    
    def show_news(self, selected_stock):
        st.header(f"Latest news for {selected_stock}")
        news = StockNews(selected_stock, save_news = False)
        news_df = news.read_rss()
        for i in range(5):
            st.subheader(f"{i+1}")
            st.write(news_df["published"][i]) 
            st.write(news_df["title"][i]) 
            st.write(news_df["summary"][i])
            sentiment_title = news_df["sentiment_title"][i]
            if sentiment_title > 0: 
                st.write(f"Title sentiment <p style ='color: Green'>{sentiment_title}</p>", unsafe_allow_html = True)
            else: 
                st.write(f"Title sentiment <p style ='color: Red'>{sentiment_title}</p>", unsafe_allow_html = True)
            sentiment_news = news_df["sentiment_summary"][i]
            if sentiment_news > 0: 
                st.write(f"News sentiment <p style ='color: Green'> {sentiment_news}</p>", unsafe_allow_html = True)
            else:
                st.write(f"News sentiment <p style ='color: Red'> {sentiment_news}</p>", unsafe_allow_html = True)

def prepare_df(stock_data):
    df = stock_data
    df.reset_index(inplace = True)
    df = df.rename(columns = {'index':'timestamp', '4. close':'close', '5. volume':'volume'})
    column_types = {'timestamp': str, 'close': np.float64, 'volume': np.float64}

    # Convert each element in the specified columns to the desired types
    df = df.astype(column_types)
    return df

def calculate_moving_avg(df):
    windows = [10,20,50]
    for window in windows:
        df[f'MA_{window}'] = df['close'].rolling(window=window).mean()
    return df

def app_logic(st, selected_stock, n_days):
    stock = Stock(selected_stock)
    st.title(f"Ai generated Predictions with sentiment analysis based on news for {selected_stock}")
    col1, col2 = st.columns((1,1))

    with col1:
        predictions, stock_data, accuracy = get_predictions(selected_stock, n_days)
        st.write(f"Stock price predictions forecasted by model for next {n_days} days for {selected_stock}")
        fig = px.line(y=predictions, labels={'x':'Number of Days','y': 'Price'}, title='Predicted Price Over Time')
        st.plotly_chart(fig, use_container_width=True, width=500, height=500)
        #st.write(f"Model predicted last 30 values on this data with {accuracy} % of accuracy!")
        stock_data = prepare_df(stock_data)

    with col2:
        st.write(f'Sentiment analysis with the help of news for {selected_stock}')
        news = StockNews(selected_stock, save_news = False)
        with st.spinner("fetching news...."):
            news_articles = news.read_rss()
        # Perform sentiment analysis on news articles
        analyzed_title_sentiments = []
        analyzed_summary_sentiments = []
        analyzed_title_probability = []
        analyzed_summary_probability = []
        title_sentiments = []
        print(" news, please wait...")
        with st.spinner( "Reading and Analyzing news...." ):
            for i in range(n_days):
                analyzed_title_sentiments.append(finbert_utils.estimate_sentiment(news_articles['title'][i])[1])
                analyzed_title_probability.append(finbert_utils.estimate_sentiment(news_articles['title'][i])[0])
                analyzed_summary_sentiments.append(finbert_utils.estimate_sentiment(news_articles['summary'][i])[1]) 
                analyzed_summary_probability.append(finbert_utils.estimate_sentiment(news_articles['summary'][i])[0]) 
                title_sentiments.append(news_articles["sentiment_title"][i])
        # Count the number of positive, negative, and neutral sentiments
        sentiment_counts = {'positive': analyzed_title_sentiments.count('positive'),
                            'negative': analyzed_title_sentiments.count('negative'),
                            'neutral': analyzed_title_sentiments.count('neutral')}
        # Perform sentiment analysis on news articles
        fig = px.pie(values=list(sentiment_counts.values()), names=list(sentiment_counts.keys()), hole=0.3)
        fig.update_traces(textposition='inside', textinfo='percent+label', title='News sentiment in current market')
        st.plotly_chart(fig, use_container_width=True, width=500, height=500)
    
    
    st.title(f"Visualization of {selected_stock}'s technical data")

    col3, col4 = st.columns([1, 1])
    with col3:
        st.write(f" ")
        st.write(f" ")
        st.subheader(f"5 aggregate targets based on predictions with approximate time")
        st.write(f" ")
        st.write(f" ")
        n_step = max(1, len(predictions) // 5)
        targets = [f"Target - {i}" for i in range(1, len(predictions) + 1, n_step) ]
        pred_data = [predictions[i] for i in range(0, len(predictions), n_step)]
        data = {"Targets": targets, "Price": pred_data}
        df = pd.DataFrame(data)
        st.table(df)
        st.write(f" ")
        st.write(f" ")
        st.write(f" ")
        st.write(f" ")
        st.write(f" ")
        st.write(f" ")
        st.write(f"Historical chart for {selected_stock}")
        fig = px.line(stock_data.head(365*2), y ='close')
        st.plotly_chart(fig, use_container_width=True, width=500, height=500)
        
    with col4:
        st.write(f"10, 20, 50 day's moving averages of {selected_stock}")
        stock_df = calculate_moving_avg(stock_data).head(365*2)

        #plt.plot(stock_df['timestamp'], stock_df[['close', 'MA_10', 'MA_20', 'MA_50']])
        #plt.xlabel('price')
        #plt.ylabel('Time period')
        #plt.title('Moving averages')
        #plt.legend(['close', 'MA_10', 'MA_20', 'MA_50'])
        #plt.show()
        fig = px.line(stock_df, x='timestamp', y=['close', 'MA_10', 'MA_20', 'MA_50'])
        st.plotly_chart(fig, use_container_width=True, width=500, height=500)
        st.write(f"volume traded for {selected_stock} over time")
        fig = px.bar(x=stock_df['timestamp'], y=stock_df['volume'].head(365*2))
        st.plotly_chart(fig, use_container_width=True, width=500, height=500)
    
    stock.show_stock_data(st, selected_stock)
with st.sidebar:
    st.title('Stock Analyzer and Predictor Dashboard')
    #following line will representall stock tickers
    stocks = ( "AAPL", "MSFT", "HSBC", "TSCO", "GOOG" 'BIDU','GOOGL','META','NFLX','RBLX','SNAP','T','TKO','TTD' )
    selected_stock = st.selectbox( "Please select stock ticker to get detailed insights" , stocks, index = None)
    n_days = st.number_input("How long you want to visualize the future?", step=1)


if (selected_stock is not None) and (n_days > 0):
    app_logic(st, selected_stock, n_days)
else:
    st.subheader("Please select a stock from sidebar!")

