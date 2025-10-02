from __future__ import annotations

import os
import argparse
import pandas as pd
import numpy as np

from src.utils.config import load_config
from src.features.technical import add_technical_indicators
from src.features.volatility import realized_volatility, rolling_std, garch_volatility


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--tickers', type=str, default='AAPL,MSFT,NVDA,SPY')
    p.add_argument('--horizon', type=int, default=5, help='Forward days for volatility target')
    return p.parse_args()


def make_targets(df: pd.DataFrame, close_col: str, horizon: int) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out['rv_future'] = realized_volatility(df[close_col], window=horizon).shift(-horizon)
    # Buckets via quantiles (drop nan first)
    q = out['rv_future'].quantile([0.33, 0.66])
    lo, hi = float(q.loc[0.33]), float(q.loc[0.66])
    def bucket(x: float) -> str:
        if np.isnan(x):
            return np.nan
        if x < lo:
            return 'low'
        if x < hi:
            return 'medium'
        return 'high'
    out['vol_bucket'] = out['rv_future'].apply(bucket)
    return out


def main():
    args = parse_args()
    cfg = load_config()

    raw = cfg.raw_dir
    processed = cfg.processed_dir
    os.makedirs(processed, exist_ok=True)

    prices_path = os.path.join(raw, 'prices.csv')
    trends_path = os.path.join(raw, 'trends.csv')
    tweets_daily_path = os.path.join(raw, 'tweets_daily_sentiment.csv')
    reddit_daily_path = os.path.join(raw, 'reddit_daily_sentiment.csv')

    if not os.path.exists(prices_path):
        raise SystemExit('Missing prices.csv. Run scripts/run_collectors.py first.')

    prices = pd.read_csv(prices_path, parse_dates=['Date']).set_index('Date').sort_index()

    # Build per-ticker frames
    frames = []
    tickers = sorted({c.split('_')[0] for c in prices.columns if c.endswith('_close')})
    for t in tickers:
        close = f'{t}_close'
        high = f'{t}_high'
        low = f'{t}_low'
        vol = f'{t}_volume'
        sub = prices[[close, high, low, vol]].dropna().copy()
        sub = add_technical_indicators(sub, close_col=close, high_col=high, low_col=low, vol_col=vol)

        # Targets
        tgt = make_targets(sub, close_col=close, horizon=args.horizon)
        feat = sub.join(tgt, how='left')
        feat['ticker'] = t
        frames.append(feat)

    df = pd.concat(frames).reset_index()
    df = df.rename(columns={'index': 'date', 'Date': 'date'})
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    # Join trends
    if os.path.exists(trends_path):
        trends = pd.read_csv(trends_path, parse_dates=['date'])
        for t in tickers:
            if t in trends.columns:
                tmp = trends[['date', t]].rename(columns={t: f'trends_{t}'})
                tmp['ticker'] = t
                df = df.merge(tmp, on=['date', 'ticker'], how='left')

    # Join daily sentiment (tweets and reddit)
    for path, prefix in [(tweets_daily_path, 'tw'), (reddit_daily_path, 'rd')]:
        if os.path.exists(path):
            s = pd.read_csv(path, parse_dates=['Unnamed: 0']).rename(columns={'Unnamed: 0':'date'})
            s.columns = [f'{prefix}_{c}' if c != 'date' else 'date' for c in s.columns]
            df = df.merge(s, on='date', how='left')

    out_path = os.path.join(processed, 'dataset.csv')
    df.to_csv(out_path, index=False)
    print(f'Wrote {out_path} with shape {df.shape}')


if __name__ == '__main__':
    main()
