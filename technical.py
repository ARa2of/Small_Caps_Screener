"""
Technical indicator calculations: EMAs, SMAs, golden cross, RSI, ATR, RVOL.
Operates on a single ticker's OHLCV DataFrame (as returned by yfinance),
with columns: Open, High, Low, Close, Volume.
"""

import pandas as pd
import pandas_ta as ta

from config import EMA_PERIODS, SMA_PERIODS, RSI_PERIOD, ATR_PERIOD


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds EMA20/50/200, SMA50/200, RSI14, ATR14, and a 20-day avg volume
    column to a single-ticker OHLCV DataFrame. Returns the same DataFrame
    with new columns appended. Does not mutate the input in place.
    """
    out = df.copy()

    for p in EMA_PERIODS:
        out[f"EMA{p}"] = ta.ema(out["Close"], length=p)

    for p in SMA_PERIODS:
        out[f"SMA{p}"] = ta.sma(out["Close"], length=p)

    out["RSI"] = ta.rsi(out["Close"], length=RSI_PERIOD)
    out["ATR"] = ta.atr(out["High"], out["Low"], out["Close"], length=ATR_PERIOD)
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
