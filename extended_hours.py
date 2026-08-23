"""
Extended Hours (Pre-Market / After-Hours) Data Fetcher

For shortlisted tickers, fetches intraday data with prepost=True to capture
pre-market and after-hours trading. Extracts per-ticker AH/PM metrics that
feed into the scoring table and AI prompt.

yfinance's prepost flag only works with intraday intervals (not daily),
so this runs as a separate step after the daily bulk download + shortlist.
"""

import logging
import time

import numpy as np
import pandas as pd
import yfinance as yf

from config import (
    EXTENDED_HOURS_ENABLED,
    EXTENDED_HOURS_INTERVAL,
    EXTENDED_HOURS_PERIOD,
    EXTENDED_HOURS_BATCH_SIZE,
    EXTENDED_HOURS_PAUSE_SEC,
)

log = logging.getLogger(__name__)

# US market session boundaries (Eastern Time, approximate)
# Regular hours: 09:30-16:00 ET
# Pre-market: 04:00-09:30 ET
# After-hours: 16:00-20:00 ET
_PREMARKET_START = pd.Timestamp("04:00").time()
_PREMARKET_END = pd.Timestamp("09:30").time()
_REGULAR_START = pd.Timestamp("09:30").time()
_REGULAR_END = pd.Timestamp("16:00").time()
_AFTERHOURS_START = pd.Timestamp("16:00").time()
_AFTERHOURS_END = pd.Timestamp("20:00").time()


def _classify_session(ts: pd.Timestamp) -> str:
    """Classify a timestamp into regular/pre-market/after-hours based on ET hour."""
    t = ts.time()
    if _PREMARKET_START <= t < _PREMARKET_END:
        return "pre_market"
    elif _REGULAR_START <= t < _REGULAR_END:
        return "regular"
    elif _AFTERHOURS_START <= t < _AFTERHOURS_END:
        return "after_hours"
    else:
        return "overnight"


def _chunk(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def fetch_extended_hours_batch(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """
    Fetch intraday data with prepost=True for a list of tickers.
    Returns {ticker: DataFrame} with an added 'session' column
    (regular/pre_market/after_hours/overnight).
    """
    results: dict[str, pd.DataFrame] = {}
    batches = list(_chunk(tickers, EXTENDED_HOURS_BATCH_SIZE))

    for i, batch in enumerate(batches, 1):
        log.info(f"Extended hours batch {i}/{len(batches)} ({len(batch)} tickers)...")
        try:
            data = yf.download(
                tickers=batch,
                period=EXTENDED_HOURS_PERIOD,
                interval=EXTENDED_HOURS_INTERVAL,
                group_by="ticker",
                auto_adjust=True,
                prepost=True,
                threads=True,
                progress=False,
            )
        except Exception as e:
            log.warning(f"Extended hours batch {i} failed: {e}")
            time.sleep(EXTENDED_HOURS_PAUSE_SEC)
            continue

        for symbol in batch:
            try:
                if len(batch) == 1:
                    df = data.copy()
                else:
                    if symbol not in data.columns.get_level_values(0):
                        continue
                    df = data[symbol].copy()

                df = df.dropna(how="all")
                if df.empty:
                    continue

                # Ensure timezone-aware index for session classification
                if df.index.tz is not None:
                    df.index = df.index.tz_convert("US/Eastern")
                else:
                    df.index = df.index.tz_localize("UTC").tz_convert("US/Eastern")

                df["session"] = df.index.map(_classify_session)
                results[symbol] = df

            except Exception as e:
                log.debug(f"Skipping {symbol} in extended hours: {e}")
                continue

        time.sleep(EXTENDED_HOURS_PAUSE_SEC)

    log.info(f"Extended hours download complete: {len(results)}/{len(tickers)} tickers.")
    return results


def compute_extended_hours_metrics(
    shortlist_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each shortlisted ticker, fetch intraday AH/PM data and compute
    extended hours metrics. Returns shortlist_df with new columns:

      - ah_close: last after-hours close price (regular day close + AH session)
      - ah_volume: total volume traded during after-hours sessions (last 5 days)
      - ah_pct_move: % change from last regular close to last AH close
      - pm_open: last pre-market open price
      - pm_volume: total volume during pre-market sessions (last 5 days)
      - pm_pct_move: % change from previous regular close to PM open
      - last_ah_high: highest price in after-hours sessions (last 5 days)
      - last_ah_low: lowest price in after-hours sessions (last 5 days)
      - has_extended_data: True if AH/PM data was available for this ticker
    """
    if not EXTENDED_HOURS_ENABLED:
        log.info("Extended hours data disabled in config. Skipping.")
        shortlist_df["has_extended_data"] = False
        return shortlist_df

    tickers = shortlist_df["symbol"].tolist()
    if not tickers:
        shortlist_df["has_extended_data"] = False
        return shortlist_df

    eh_data = fetch_extended_hours_batch(tickers)

    metrics = []
    for symbol in tickers:
        row = {}
        df = eh_data.get(symbol)

        if df is None or df.empty:
            row.update({
                "ah_close": None, "ah_volume": None, "ah_pct_move": None,
                "pm_open": None, "pm_volume": None, "pm_pct_move": None,
                "last_ah_high": None, "last_ah_low": None,
                "has_extended_data": False,
            })
        else:
            # Split by session
            ah = df[df["session"] == "after_hours"]
            pm = df[df["session"] == "pre_market"]
            regular = df[df["session"] == "regular"]

            # After-hours: last session's close and volume
            if not ah.empty:
                last_ah_close = float(ah["Close"].iloc[-1])
                ah_vol = float(ah["Volume"].sum())
                ah_high = float(ah["High"].max())
                ah_low = float(ah["Low"].min())

                # % move from last regular close to AH close
                if not regular.empty:
                    last_reg_close = float(regular["Close"].iloc[-1])
                    ah_pct = ((last_ah_close - last_reg_close) / last_reg_close * 100
                              if last_reg_close > 0 else None)
                else:
                    ah_pct = None

                row.update({
                    "ah_close": round(last_ah_close, 4),
                    "ah_volume": round(ah_vol, 0),
                    "ah_pct_move": round(ah_pct, 2) if ah_pct is not None else None,
                    "last_ah_high": round(ah_high, 4),
                    "last_ah_low": round(ah_low, 4),
                })
            else:
                row.update({
                    "ah_close": None, "ah_volume": None, "ah_pct_move": None,
                    "last_ah_high": None, "last_ah_low": None,
                })

            # Pre-market: last session's open and volume
            if not pm.empty:
                last_pm_open = float(pm["Open"].iloc[-1])
                pm_vol = float(pm["Volume"].sum())

                # % move from previous regular close to PM open
                if not regular.empty:
                    last_reg_close = float(regular["Close"].iloc[-1])
                    pm_pct = ((last_pm_open - last_reg_close) / last_reg_close * 100
                              if last_reg_close > 0 else None)
                else:
                    pm_pct = None

                row.update({
                    "pm_open": round(last_pm_open, 4),
                    "pm_volume": round(pm_vol, 0),
                    "pm_pct_move": round(pm_pct, 2) if pm_pct is not None else None,
                })
            else:
                row.update({
                    "pm_open": None, "pm_volume": None, "pm_pct_move": None,
                })

            row["has_extended_data"] = True

        row["symbol"] = symbol
        metrics.append(row)

    metrics_df = pd.DataFrame(metrics)
    result = shortlist_df.merge(metrics_df, on="symbol", how="left")

    # Fill NaN for missing data
    for col in ["ah_close", "ah_volume", "ah_pct_move", "pm_open", "pm_volume",
                "pm_pct_move", "last_ah_high", "last_ah_low"]:
        if col in result.columns:
            result[col] = result[col].where(result[col].notna(), None)

    if "has_extended_data" not in result.columns:
        result["has_extended_data"] = False

    has_data_count = result["has_extended_data"].sum() if "has_extended_data" in result.columns else 0
    log.info(f"Extended hours metrics computed: {has_data_count}/{len(tickers)} tickers have AH/PM data.")

    return result
