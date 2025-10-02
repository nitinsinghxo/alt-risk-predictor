from __future__ import annotations

import pandas as pd
import ta


def add_technical_indicators(price_df: pd.DataFrame, close_col: str, high_col: str, low_col: str, vol_col: str) -> pd.DataFrame:
    df = price_df.copy()
    df['sma_10'] = df[close_col].rolling(10).mean()
    df['sma_20'] = df[close_col].rolling(20).mean()
    df['ema_12'] = df[close_col].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df[close_col].ewm(span=26, adjust=False).mean()

    rsi = ta.momentum.rsi(df[close_col], window=14)
    macd = ta.trend.MACD(df[close_col])
    bb = ta.volatility.BollingerBands(df[close_col], window=20, window_dev=2)

    df['rsi_14'] = rsi
    if macd is not None:
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
    if bb is not None:
        df['bb_hband'] = bb.bollinger_hband()
        df['bb_lband'] = bb.bollinger_lband()
        df['bb_mavg'] = bb.bollinger_mavg()

    return df
