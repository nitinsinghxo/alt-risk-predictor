from __future__ import annotations

import numpy as np
import pandas as pd
from arch import arch_model


def compute_returns(close: pd.Series) -> pd.Series:
    return close.pct_change()


def realized_volatility(close: pd.Series, window: int = 5, annualization: int = 252) -> pd.Series:
    r = compute_returns(close)
    rv = r.rolling(window).std() * np.sqrt(annualization)
    return rv.rename(f"rv_{window}")


def rolling_std(close: pd.Series, window: int = 20, annualization: int = 252) -> pd.Series:
    r = compute_returns(close)
    vol = r.rolling(window).std() * np.sqrt(annualization)
    return vol.rename(f"rollstd_{window}")


def garch_volatility(close: pd.Series, p: int = 1, q: int = 1, annualization: int = 252) -> pd.Series:
    r = 100 * compute_returns(close).dropna()
    if r.empty:
        return pd.Series(index=close.index, dtype=float, name="garch_vol")
    am = arch_model(r, vol='Garch', p=p, q=q, mean='Zero', dist='normal')
    res = am.fit(disp='off')
    cond_vol = res.conditional_volatility / 100.0 * np.sqrt(annualization)
    cond_vol.name = 'garch_vol'
    # Reindex to original
    cond_vol = cond_vol.reindex(close.index)
    return cond_vol
