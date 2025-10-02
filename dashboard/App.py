import os
import pandas as pd
import numpy as np
from typing import Optional
import plotly.express as px
import streamlit as st

from src.utils.config import load_config

st.set_page_config(page_title="Alt-Risk Predictor", layout="wide")

cfg = load_config()
raw_dir = cfg.raw_dir

st.title("Alt-Risk Predictor: Volatility with Alternative Data")

prices_path = os.path.join(raw_dir, 'prices.csv')
trends_path = os.path.join(raw_dir, 'trends.csv')
tweets_sent_path = os.path.join(raw_dir, 'tweets_daily_sentiment.csv')
reddit_sent_path = os.path.join(raw_dir, 'reddit_daily_sentiment.csv')
dataset_path = os.path.join(cfg.processed_dir, 'dataset.csv')
preds_path = os.path.join(cfg.processed_dir, 'predictions_classification.csv')
bt_path = os.path.join(cfg.processed_dir, 'backtest.csv')

@st.cache_data(show_spinner=False)
def load_prices(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=['Date']).set_index('Date').sort_index()
    return df

@st.cache_data(show_spinner=False)
def load_df(path: str, date_col: Optional[str] = None) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    if date_col:
        return pd.read_csv(path, parse_dates=[date_col])
    return pd.read_csv(path)

# Sidebar filters
prices = load_prices(prices_path)
all_tickers = sorted({c.split('_')[0] for c in prices.columns if c.endswith('_close')}) if not prices.empty else []

st.sidebar.header("Filters")
selected_tickers = st.sidebar.multiselect("Tickers", options=all_tickers, default=all_tickers)

min_date = prices.index.min().date() if not prices.empty else pd.Timestamp("2020-01-01").date()
max_date = prices.index.max().date() if not prices.empty else pd.Timestamp.today().date()
date_range = st.sidebar.date_input("Date range", value=(min_date, max_date))
start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])

tab_overview, tab_predictions, tab_backtest, tab_dataset = st.tabs(["Overview", "Predictions", "Backtest", "Dataset"])

with tab_overview:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Prices overview")
        if prices.empty:
            st.info("Run scripts/run_collectors.py to populate data.")
        else:
            subcols = [f"{t}_close" for t in selected_tickers if f"{t}_close" in prices.columns]
            view = prices.loc[(prices.index >= start_date) & (prices.index <= end_date), subcols]
            if not view.empty:
                st.line_chart(view)
            else:
                st.info("No data for selected filters.")
    with col2:
        st.subheader("Search interest (Google Trends)")
        trends = load_df(trends_path, 'date')
        if trends.empty:
            st.info("No trends data yet.")
        else:
            view = trends[(trends['date'] >= start_date) & (trends['date'] <= end_date)]
            st.line_chart(view.set_index('date'))

    st.subheader("Daily Sentiment")
    ts = load_df(tweets_sent_path, 'Unnamed: 0').rename(columns={'Unnamed: 0':'date'}) if os.path.exists(tweets_sent_path) else pd.DataFrame()
    rs = load_df(reddit_sent_path, 'Unnamed: 0').rename(columns={'Unnamed: 0':'date'}) if os.path.exists(reddit_sent_path) else pd.DataFrame()
    if not ts.empty:
        st.line_chart(ts.set_index('date'))
    if not rs.empty:
        st.line_chart(rs.set_index('date'))

with tab_predictions:
    st.subheader("Predictions")
    pr = load_df(preds_path, 'date')
    if pr.empty:
        st.info("No predictions yet. Run scripts/predict.py")
    else:
        mask = (pr['date'] >= start_date) & (pr['date'] <= end_date)
        if selected_tickers:
            mask &= pr['ticker'].isin(selected_tickers)
        view = pr.loc[mask].copy()
        if view.empty:
            st.info("No predictions for selected filters.")
        else:
            colp1, colp2 = st.columns([2, 1])
            with colp1:
                st.write("Recent predictions")
                st.dataframe(view.sort_values('date').tail(1000), use_container_width=True)
            with colp2:
                by_bucket = view['pred_bucket'].value_counts(normalize=True).sort_index()
                if not by_bucket.empty:
                    st.write("Bucket distribution")
                    st.bar_chart(by_bucket)

with tab_backtest:
    st.subheader("Backtest")
    bt = load_df(bt_path, 'date')
    if bt.empty:
        st.info("No backtest yet. Run scripts/backtest.py")
    else:
        view = bt[(bt['date'] >= start_date) & (bt['date'] <= end_date)]
        colb1, colb2, colb3 = st.columns(3)
        if not view.empty:
            total_return = (view['cum_return'].iloc[-1] - view['cum_return'].iloc[0]) if len(view) > 1 else view['cum_return'].iloc[0]
            drawdown = (view['cum_return'] - view['cum_return'].cummax()).min()
            ret_series = view['strategy_return'].dropna() if 'strategy_return' in view.columns else pd.Series(dtype=float)
            sharpe = np.sqrt(252) * (ret_series.mean() / ret_series.std()) if not ret_series.empty and ret_series.std() != 0 else np.nan
            colb1.metric("Total Return", f"{total_return:.2%}" if pd.notna(total_return) else "-")
            colb2.metric("Max Drawdown", f"{drawdown:.2%}" if pd.notna(drawdown) else "-")
            colb3.metric("Sharpe (est)", f"{sharpe:.2f}" if pd.notna(sharpe) else "-")
            fig = px.line(view, x='date', y='cum_return', title='Cumulative Return (Strategy)')
            st.plotly_chart(fig, use_container_width=True)

with tab_dataset:
    st.subheader("Processed Dataset")
    ds = load_df(dataset_path, 'date')
    if ds.empty:
        st.info("Dataset missing. Run scripts/build_dataset.py")
    else:
        mask = (ds['date'] >= start_date) & (ds['date'] <= end_date)
        if selected_tickers:
            mask &= ds['ticker'].isin(selected_tickers)
        view = ds.loc[mask]
        st.write(f"Rows: {len(view):,}  |  Columns: {len(view.columns):,}")
        st.dataframe(view, use_container_width=True, height=500)
        @st.cache_data
        def _to_csv(df: pd.DataFrame) -> bytes:
            return df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", data=_to_csv(view), file_name="dataset_filtered.csv", mime="text/csv")
