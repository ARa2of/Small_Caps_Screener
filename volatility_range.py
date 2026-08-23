"""
Volatility-based 3-month price range estimator.

Computes a statistical expected range (not a point prediction) using:
1. Historical daily log-return volatility → annualized → projected to 3 months
2. ATR14-based range as a secondary reference

Framed as a probability band: "Based on historical vol, ~68% of 3-month
outcomes fall within $X–$Y" (1 standard deviation).
"""

import logging

import numpy as np
import pandas as pd

from config import VOLATILITY_RANGE_TRADING_DAYS, VOLATILITY_RANGE_CONFIDENCE_SD

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def compute_volatility_range(symbol: str, price_data: pd.DataFrame) -> dict:
    """
    Given a single ticker's OHLCV DataFrame (from yfinance, 1 year daily),
    computes the 3-month expected price range based on historical volatility.

    Returns a dict with:
      vol_3mo_lower: lower bound of 1-SD range
      vol_3mo_upper: upper bound of 1-SD range
      annualized_vol: annualized historical volatility (%)
      atr_3mo_lower: ATR-based lower bound (secondary)
      atr_3mo_upper: ATR-based upper bound (secondary)
      vol_range_note: human-readable summary
    """
    result = {
        "vol_3mo_lower": None,
        "vol_3mo_upper": None,
        "annualized_vol": None,
        "atr_3mo_lower": None,
        "atr_3mo_upper": None,
        "vol_range_note": "N/A — insufficient data",
    }

    if price_data is None or price_data.empty or len(price_data) < 60:
        return result

    try:
        closes = price_data["Close"].dropna()
        if len(closes) < 60:
            return result

        current_price = float(closes.iloc[-1])
        if current_price <= 0:
            return result

        # --- Method 1: Historical return volatility ---
        log_returns = np.log(closes / closes.shift(1)).dropna()
        daily_vol = float(log_returns.std())
        annualized_vol = daily_vol * np.sqrt(252)

        # Project to 3 months (63 trading days)
        move_1sd = current_price * daily_vol * np.sqrt(VOLATILITY_RANGE_TRADING_DAYS) * VOLATILITY_RANGE_CONFIDENCE_SD
        lower = current_price - move_1sd
        upper = current_price + move_1sd

        result["vol_3mo_lower"] = round(lower, 2)
        result["vol_3mo_upper"] = round(upper, 2)
        result["annualized_vol"] = round(annualized_vol * 100, 1)

        # --- Method 2: ATR-based range (secondary reference) ---
        highs = price_data["High"].dropna()
        lows = price_data["Low"].dropna()
        closes_for_atr = price_data["Close"].dropna()

        if len(highs) >= 14 and len(lows) >= 14 and len(closes_for_atr) >= 14:
            # Manual ATR calculation (14-period)
            tr1 = highs - lows
            tr2 = abs(highs - closes_for_atr.shift(1))
            tr3 = abs(lows - closes_for_atr.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr14 = float(tr.rolling(14).mean().iloc[-1])

            if not np.isnan(atr14) and atr14 > 0:
                atr_move = atr14 * np.sqrt(VOLATILITY_RANGE_TRADING_DAYS) * VOLATILITY_RANGE_CONFIDENCE_SD
                result["atr_3mo_lower"] = round(current_price - atr_move, 2)
                result["atr_3mo_upper"] = round(current_price + atr_move, 2)

        # --- Build human-readable note ---
        sd_label = f"{VOLATILITY_RANGE_CONFIDENCE_SD}-SD" if VOLATILITY_RANGE_CONFIDENCE_SD != 1 else "1-SD"
        result["vol_range_note"] = (
            f"{sd_label} 3mo range: ${result['vol_3mo_lower']:.2f}–${result['vol_3mo_upper']:.2f} "
            f"(ann. vol {result['annualized_vol']:.0f}%)"
        )
        if result["atr_3mo_lower"] is not None:
            result["vol_range_note"] += (
                f" | ATR range: ${result['atr_3mo_lower']:.2f}–${result['atr_3mo_upper']:.2f}"
            )

    except Exception as e:
        log.debug(f"Volatility range computation failed for {symbol}: {e}")

    return result


def compute_all_volatility_ranges(shortlist_df: pd.DataFrame, price_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Enriches the shortlist DataFrame with volatility range columns.
    Returns the input DataFrame with new columns added.
    """
    if shortlist_df.empty:
        return shortlist_df

    vol_rows = []
    for _, row in shortlist_df.iterrows():
        symbol = row["symbol"]
        data = price_data.get(symbol)
        vol_data = compute_volatility_range(symbol, data)
        vol_data["symbol"] = symbol
        vol_rows.append(vol_data)

    vol_df = pd.DataFrame(vol_rows)
    result = shortlist_df.merge(vol_df, on="symbol", how="left")
    log.info(f"Volatility ranges computed for {len(result)} tickers.")
    return result
