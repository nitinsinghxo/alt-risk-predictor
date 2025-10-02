# Alt-Risk Predictor: Stock Market Volatility with Alternative Data

An end-to-end machine learning pipeline that predicts stock market volatility risk by integrating traditional financial data with alternative data sources (social sentiment, Google Trends). Demonstrates skills in data engineering, ML modeling, and financial analytics.

## 🎯 Project Overview

This project combines multiple data sources to predict stock volatility:
- **Financial Data**: Stock prices, returns, volumes via Yahoo Finance
- **Alternative Data**: Reddit posts, Twitter sentiment, Google Trends search interest
- **ML Models**: Random Forest & Gradient Boosting for classification and regression
- **Evaluation**: Sharpe ratio backtesting, confusion matrices, feature importance
- **Dashboard**: Interactive Streamlit app with real-time visualizations

## 📊 Results Summary

- **Classification Accuracy**: ~44% (3-class: low/medium/high risk)
- **Regression RMSE**: ~0.10 (volatility prediction)
- **Backtest Sharpe Ratio**: 0.93
- **Features**: 35+ technical indicators, sentiment scores, trends data
- **Time Period**: 2020-2025 (5 years of data)

## 🚀 Quick Start

```bash
# Clone and setup
git clone <repo-url>
cd alt-risk-predictor
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run end-to-end pipeline
PYTHONPATH=. python scripts/run_collectors.py --tickers AAPL,MSFT,NVDA,SPY
PYTHONPATH=. python scripts/build_dataset.py
PYTHONPATH=. python scripts/train.py --task regression
PYTHONPATH=. python scripts/train.py --task classification
PYTHONPATH=. python scripts/predict.py
PYTHONPATH=. python scripts/backtest.py

# Launch dashboard
streamlit run dashboard/App.py
```

## 📁 Project Structure

```
alt-risk-predictor/
├── src/
│   ├── data/           # Data collectors (yfinance, pytrends, reddit, twitter)
│   ├── features/       # Feature engineering (technical, sentiment, volatility)
│   ├── models/         # Training and evaluation
│   └── utils/          # Configuration and helpers
├── scripts/            # Pipeline entrypoints
├── notebooks/          # Jupyter notebooks for analysis
├── dashboard/          # Streamlit app
├── data/              # Raw and processed data (gitignored)
└── requirements.txt   # Dependencies
```

## 🔧 Data Sources

### Financial Data
- **Yahoo Finance**: Stock prices, volumes, returns
- **Technical Indicators**: SMA, EMA, RSI, MACD, Bollinger Bands
- **Volatility Measures**: Realized volatility, rolling std, GARCH(1,1)

### Alternative Data
- **Reddit**: Posts from r/stocks, r/investing, r/wallstreetbets
- **Twitter**: Tweets mentioning stock tickers (via snscrape)
- **Google Trends**: Search interest for stock symbols
- **Sentiment Analysis**: VADER and FinBERT models

## 🤖 Machine Learning Pipeline

### Feature Engineering
- Technical indicators (20+ features per ticker)
- Daily sentiment aggregation (VADER + FinBERT)
- Google Trends normalization
- Volatility target creation (5-day forward)

### Models
- **Classification**: 3-class risk buckets (low/medium/high)
- **Regression**: Direct volatility prediction
- **Algorithms**: Random Forest, Gradient Boosting
- **Evaluation**: F1-score, RMSE, Sharpe ratio backtesting

### Backtesting Strategy
- **Allocation**: 100% low-risk, 50% medium-risk, 0% high-risk
- **Rebalancing**: Daily based on predictions
- **Metrics**: Sharpe ratio, max drawdown, total return

## 📈 Dashboard Features

The Streamlit dashboard provides:
- **Price Charts**: Interactive time series of stock prices
- **Sentiment Trends**: Daily sentiment scores over time
- **Predictions**: Latest risk bucket predictions with confidence
- **Backtest Results**: Cumulative returns and performance metrics
- **Feature Importance**: Top predictive features

## 📚 Jupyter Notebooks

- `01_data_prep.ipynb`: Data collection and quality assessment
- `02_feature_engineering.ipynb`: Technical indicators and sentiment analysis
- `03_modeling.ipynb`: Model training and evaluation
- `04_evaluation.ipynb`: Backtesting and performance analysis

## ⚙️ Configuration

### Environment Variables
Copy `.env.example` to `.env` and configure:
```bash
# Reddit API (optional - for enhanced Reddit data)
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=alt-risk-predictor/0.1 by your_username
```

### Dependencies
Key packages: pandas, scikit-learn, yfinance, pytrends, praw, snscrape, streamlit, plotly

## 🎯 Key Insights

1. **Alternative Data Value**: Social sentiment and search trends provide predictive signals beyond price data
2. **Risk Management**: Classification approach enables systematic risk-based allocation
3. **Feature Engineering**: Technical indicators combined with sentiment create robust feature set
4. **Model Performance**: Ensemble methods (RF + GB) outperform individual algorithms
5. **Backtesting**: Strategy shows positive Sharpe ratio with controlled drawdowns

## 🔮 Future Enhancements

- **Real-time Data**: Live data feeds and real-time predictions
- **Deep Learning**: LSTM/Transformer models for sequence prediction
- **Ensemble Methods**: Voting and stacking classifiers
- **Risk Management**: Dynamic position sizing and stop-losses
- **Alternative Assets**: Extend to crypto, forex, commodities

## 📊 Sample Results

### Model Performance
```
Regression Models:
  Random Forest: RMSE=0.1095, R²=0.2341
  Gradient Boosting: RMSE=0.0997, R²=0.3456

Classification Models:
  Random Forest: F1=0.4219
  Gradient Boosting: F1=0.3669
```

### Backtest Performance
```
Total Return: 15.2%
Annualized Volatility: 12.8%
Sharpe Ratio: 0.93
Maximum Drawdown: -8.4%
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- Yahoo Finance for market data
- Reddit API for social sentiment
- Google Trends for search interest
- Hugging Face for FinBERT model
- Streamlit for dashboard framework