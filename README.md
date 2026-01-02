# 📈 Stock Price Analyzer & Forecasting System
#### End-to-End Time Series & Sentiment-Aware ML Application

## 📌 Project Overview
This project demonstrates the development of an end-to-end machine learning pipeline for stock price forecasting using historical financial data.
The objective is to predict future stock price movements by applying statistical analysis, feature engineering, and supervised learning techniques, while ensuring reproducibility and interpretability. The project reflects real-world ML workflows commonly used in UK FinTech and data-driven teams, from data ingestion to model evaluation.

## 🎯 Problem Statement: 
Financial markets generate large volumes of time-series data, yet accurately predicting price movements remains challenging due to volatility and noise.

## Goal:
Build a predictive system that:
 - Learns patterns from historical stock data
 - Produces reliable forecasts
 - Evaluates performance using quantitative metrics

## 🧠 Key Features

 - 📊 Time Series Forecasting using PyTorch LSTM networks
 - 📰 Financial Sentiment Analysis with FinBERT
 - 🧩 Modular data preprocessing & model utilities
 - 📈 Interactive Streamlit dashboard for prediction visualisation
 - ♻️ Reproducible training & inference pipelines

## 🏗️ Project Structure
- app.py – Streamlit application (visualisation & inference)
- lstm_training.py – Model training & experimentation
- lstm_utils.py – LSTM architecture & helpers
- finbert_utils.py – Financial sentiment processing
- data_utils.py – Data preprocessing & feature engineering
- best_trained_model.pth – Persisted best-performing model

## 🧠 Solution Approach
### 1. Data Ingestion
 - Historical stock data loaded from CSV
 - Time-series indexing and cleaning
 - Handling missing values and anomalies

### 2. Feature Engineering
 - Lagged price features
 - Rolling statistics (moving averages, volatility)
 - Trend-based indicators

### 3. Model Development
 - Supervised ML models trained on engineered features
 - Train / validation split respecting time-series ordering
 - Hyperparameter tuning for performance stability

### 4. Evaluation
 - Quantitative performance metrics (e.g. accuracy / error metrics)
 - Visual comparison of predicted vs actual prices
 - Error analysis to assess model robustness

## 📊 Results
1. Successfully trained a predictive model on historical data
2. Demonstrated the ability to capture short-term price trends

### Results indicate strong potential for further enhancement using:
 - Advanced deep learning models
 - Sentiment or macro-economic features
(Exact metrics can be extended as the project evolves)

## 🛠️ Technologies Used
Programming Language: Python

## ▶️ How to Run the Project

### Clone the repository:
git clone https://github.com/Meet1590/<repository-name>.git
cd <repository-name>

### Install dependencies:
pip install -r requirements.txt

### 🚀 How to Run the application:
pip install -r requirements.txt
streamlit run app.py

### 🧪 Model Training
python lstm_training.py

**Note**: Hyperparameters can be adjusted to experiment with different sequence lengths, architectures, and learning rates.

## 💼 Why This Project Matters

**This project demonstrates:**
 - Practical machine learning applied to financial data
 - Strong Python and data-handling skills
- Understanding of time-series modelling challenges
- Ability to structure ML projects in a production-oriented manner

## 👤 Author
**Meetkumar Patel**
- **Machine Learning / Data Science**
- **GitHub**: https://github.com/Meet1590
- **LinkedIn**: https://www.linkedin.com/in/meet-07-patel/

## 🔍 Notes for Recruiters
 - Code is written for clarity and extensibility
 - Dataset included for reproducibility
 - Designed to be extended into a production-grade ML system
