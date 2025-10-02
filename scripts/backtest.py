from __future__ import annotations

import os
import numpy as np
import pandas as pd

from src.utils.config import load_config


RISK_TO_WEIGHT = {
    'low': 1.0,
    'medium': 0.5,
    'high': 0.0,
}


def compute_sharpe(returns: pd.Series, risk_free: float = 0.0, annualization: int = 252) -> float:
    excess = returns - risk_free / annualization
    mu = excess.mean() * annualization
    sigma = excess.std() * np.sqrt(annualization)
    if sigma == 0 or np.isnan(sigma):
        return np.nan
    return float(mu / sigma)


def main():
    cfg = load_config()
    preds_path = os.path.join(cfg.processed_dir, 'predictions_classification.csv')
    prices_path = os.path.join(cfg.raw_dir, 'prices.csv')
    if not (os.path.exists(preds_path) and os.path.exists(prices_path)):
        raise SystemExit('Missing predictions or prices. Run predict.py and collectors first.')

    preds = pd.read_csv(preds_path, parse_dates=['date'])
    prices = pd.read_csv(prices_path, parse_dates=['Date']).set_index('Date').sort_index()

    # Compute next-day returns per ticker
    frames = []
    for t in sorted({c.split('_')[0] for c in prices.columns if c.endswith('_close')}):
        ret = prices[f'{t}_close'].pct_change().shift(-1).rename('return')
        ret = ret.reset_index().rename(columns={'Date': 'date'})
        ret['ticker'] = t
        frames.append(ret)
    fwd_ret = pd.concat(frames, axis=0, ignore_index=True)

    df = preds.merge(fwd_ret, on=['date','ticker'], how='left')

    # Weight positions by predicted bucket
    df['weight'] = df['pred_bucket'].map(RISK_TO_WEIGHT).fillna(0.0)
    df['strategy_return'] = df['weight'] * df['return']

    # Aggregate equally across tickers each day
    daily = df.groupby('date')['strategy_return'].mean().dropna()

    sharpe = compute_sharpe(daily)
    cumret = (1.0 + daily).cumprod()

    out = pd.DataFrame({'date': daily.index, 'strategy_return': daily.values, 'cum_return': cumret.values})
    out_path = os.path.join(cfg.processed_dir, 'backtest.csv')
    out.to_csv(out_path, index=False)

    print(f'Sharpe: {sharpe:.2f} | wrote {out_path} with {len(out)} rows')


if __name__ == '__main__':
    main()
