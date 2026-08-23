"""
Technical indicator calculations: EMAs, SMAs, golden cross, RSI, ATR, RVOL.
Operates on a single ticker's OHLCV DataFrame (as returned by yfinance),
with columns: Open, High, Low, Close, Volume.

Uses only pandas/numpy — no external TA libraries needed.
"""

import numpy as np
import pandas as pd

from config import EMA_PERIODS, SMA_PERIODS, RSI_PERIOD, ATR_PERIOD


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds EMA20/50/200, SMA50/200, RSI14, ATR14, and a 20-day avg volume
    column to a single-ticker OHLCV DataFrame. Returns the same DataFrame
    with new columns appended. Does not mutate the input in place.
    """
    out = df.copy()

    for p in EMA_PERIODS:
        out[f"EMA{p}"] = _ema(out["Close"], p)

    for p in SMA_PERIODS:
        out[f"SMA{p}"] = _sma(out["Close"], p)

    out["RSI"] = _rsi(out["Close"], RSI_PERIOD)
    out["ATR"] = _atr(out["High"], out["Low"], out["Close"], ATR_PERIOD)
    out["AvgVol20"] = out["Volume"].rolling(window=20).mean()

    return out


def golden_cross_flag(df: pd.DataFrame, fast_col: str = "SMA50", slow_col: str = "SMA200") -> bool:
    """
    True if fast MA crossed above slow MA on the most recent bar
    (i.e. fast was below slow yesterday, at/above slow today).
    Requires df to already have the indicator columns from add_indicators().
    """
    if len(df) < 2 or fast_col not in df.columns or slow_col not in df.columns:
        return False
    prev = df.iloc[-2]
    last = df.iloc[-1]
    if pd.isna(prev[fast_col]) or pd.isna(prev[slow_col]) or pd.isna(last[fast_col]) or pd.isna(last[slow_col]):
        return False
    return prev[fast_col] < prev[slow_col] and last[fast_col] >= last[slow_col]


def trend_alignment_score(last_row: pd.Series) -> int:
    """
    Simple 0-4 score for bullish trend alignment on the most recent bar:
    +1 if Close > EMA20, +1 if EMA20 > EMA50, +1 if EMA50 > EMA200, +1 if Close > SMA200.
    Missing values are treated as failing that condition (score 0 for it).
    """
    score = 0
    try:
        if last_row["Close"] > last_row["EMA20"]:
            score += 1
        if last_row["EMA20"] > last_row["EMA50"]:
            score += 1
        if last_row["EMA50"] > last_row["EMA200"]:
            score += 1
        if last_row["Close"] > last_row["SMA200"]:
            score += 1
    except KeyError:
        pass
    return score


def relative_volume(last_row: pd.Series) -> float:
    """Today's volume divided by the 20-day average volume. NaN-safe -> returns 0.0 if unavailable."""
    avg = last_row.get("AvgVol20")
    vol = last_row.get("Volume")
    if pd.isna(avg) or avg == 0 or pd.isna(vol):
        return 0.0
    return round(vol / avg, 2)
