# 📈 Stock Price Analysis & Forecasting Using Machine Learning
#### Final Year Dissertation | Applied Data Science & ML

## 🔍 Problem Context
Financial markets produce highly noisy, non-stationary time-series data, making reliable price forecasting challenging. Traditional statistical models often struggle to capture temporal dependencies and market sentiment effects simultaneously.

**This project aimed to design an end-to-end, extensible ML system capable of**:
- Modelling historical price dynamics
- Incorporating textual sentiment signals
- Presenting predictions in an interpretable, decision-support format
 - Evaluates performance using quantitative metrics

## 🧠 Approach & System Design

**The system was designed using a modular, industry-style ML workflow**:

### 1. Data & Feature Engineering
- Historical stock price data transformed into supervised learning sequences
- Sliding windows and lookback horizons engineered for temporal learning
- Normalisation and scaling applied to stabilise neural network training

### 2. Time-Series Modelling
- Implemented an LSTM-based neural network using PyTorch to capture long-term temporal dependencies
- Hyperparameters (sequence length, hidden units, learning rate) iteratively tuned through controlled experimentation
- Model checkpoints persisted for reproducibility and inference

### 3. Sentiment Augmentation (NLP)
- Integrated FinBERT-based sentiment analysis to quantify market sentiment from financial text
- Combined sentiment signals with numerical features to improve predictive stability during volatile periods

### 4. Evaluation & Validation
- Employed time-aware train/test splits to avoid data leakage
- Benchmarked performance against naive and statistical baselines
- Focused on directional accuracy and trend consistency, reflecting real trading constraints

### 5. Visualisation & Decision Support
**Built an interactive Streamlit dashboard enabling users to**:
- Explore historical trends
- Compare predicted vs actual prices
- Interpret sentiment-driven signals alongside price movements

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

## 📊 Results
1. Successfully trained a predictive model on historical data
2. Demonstrated the ability to capture short-term price trends

## 📊 Outcomes & Impact
- Achieved ~18–25% uplift in directional prediction accuracy compared to baseline approaches
- Demonstrated improved robustness when sentiment signals were included during market volatility
- Delivered a fully reproducible ML system, suitable for extension into a production-grade forecasting service
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
