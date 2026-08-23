"""
Stage 1b: Bulk-download daily OHLCV for the ticker universe via yfinance
(free, no API key), filter down to a shortlist by price range, liquidity,
and a basic technical trigger (elevated relative volume and/or bullish
trend alignment / fresh golden cross).

Output of this stage feeds Stage 2 (catalyst/news lookup), which only
runs against this much smaller shortlist to stay within free-tier
rate limits on news APIs.
"""

import logging
import time

import pandas as pd
import yfinance as yf

from config import (
    MIN_PRICE, MAX_PRICE, MIN_AVG_DOLLAR_VOL, MIN_RVOL,
    HISTORY_PERIOD, HISTORY_INTERVAL, BATCH_SIZE, REQUEST_PAUSE_SEC,
)
from technical import add_indicators, golden_cross_flag, trend_alignment_score, relative_volume

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def _chunk(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def bulk_download(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """
    Downloads daily OHLCV for a list of tickers in batches.
    Returns a dict of {ticker: DataFrame}. Tickers with no usable data
    (delisted, insufficient history, etc.) are omitted.
    """
    results: dict[str, pd.DataFrame] = {}
    batches = list(_chunk(tickers, BATCH_SIZE))

    for i, batch in enumerate(batches, 1):
        log.info(f"Downloading batch {i}/{len(batches)} ({len(batch)} tickers)...")
        try:
            data = yf.download(
                tickers=batch,
                period=HISTORY_PERIOD,
                interval=HISTORY_INTERVAL,
                group_by="ticker",
                auto_adjust=True,
                threads=True,
                progress=False,
            )
        except Exception as e:
            log.warning(f"Batch {i} download failed entirely: {e}")
            time.sleep(REQUEST_PAUSE_SEC)
            continue

        for symbol in batch:
            try:
                if len(batch) == 1:
                    df = data
                else:
                    if symbol not in data.columns.get_level_values(0):
                        continue
                    df = data[symbol]
                df = df.dropna(how="all")
                if df.empty or len(df) < 210:  # need ~200+ bars for SMA200/EMA200
                    continue
                results[symbol] = df
            except Exception:
                continue

        time.sleep(REQUEST_PAUSE_SEC)

    log.info(f"Bulk download complete: {len(results)}/{len(tickers)} tickers returned usable data.")
    return results


def build_shortlist(price_volume_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Applies price range, liquidity, and technical trigger filters.
    Returns a DataFrame ranked by relative volume (highest = most "active" today).
    """
    rows = []

    for symbol, df in price_volume_data.items():
        try:
            df_ind = add_indicators(df)
            last = df_ind.iloc[-1]

            close = last["Close"]
            avg_vol20 = last["AvgVol20"]
            if pd.isna(close) or pd.isna(avg_vol20):
                continue

            avg_dollar_vol = close * avg_vol20
            if not (MIN_PRICE <= close <= MAX_PRICE):
                continue
            if avg_dollar_vol < MIN_AVG_DOLLAR_VOL:
                continue

            rvol = relative_volume(last)
            trend_score = trend_alignment_score(last)
            golden_cross = golden_cross_flag(df_ind)

            # Trigger: something has to make this name interesting today.
            # Either unusually high volume, OR a fresh golden cross event.
            if rvol < MIN_RVOL and not golden_cross:
                continue

            rows.append({
                "symbol": symbol,
                "close": round(close, 2),
                "rvol": rvol,
                "avg_dollar_vol_20d": round(avg_dollar_vol, 0),
                "rsi": round(last["RSI"], 1) if pd.notna(last["RSI"]) else None,
                "ema20": round(last["EMA20"], 2) if pd.notna(last["EMA20"]) else None,
                "ema50": round(last["EMA50"], 2) if pd.notna(last["EMA50"]) else None,
                "ema200": round(last["EMA200"], 2) if pd.notna(last["EMA200"]) else None,
                "sma50": round(last["SMA50"], 2) if pd.notna(last["SMA50"]) else None,
                "sma200": round(last["SMA200"], 2) if pd.notna(last["SMA200"]) else None,
                "atr": round(last["ATR"], 2) if pd.notna(last["ATR"]) else None,
                "trend_alignment_score": trend_score,
                "golden_cross_today": golden_cross,
            })
        except Exception as e:
            log.debug(f"Skipping {symbol} due to error: {e}")
            continue

    shortlist = pd.DataFrame(rows)
    if not shortlist.empty:
        shortlist = shortlist.sort_values(by=["golden_cross_today", "rvol"], ascending=[False, False])
        shortlist = shortlist.reset_index(drop=True)

    log.info(f"Shortlist built: {len(shortlist)} tickers passed Stage 1 filters "
              f"(price ${MIN_PRICE}-${MAX_PRICE}, min RVOL {MIN_RVOL}x or golden cross).")
    return shortlist


if __name__ == "__main__":
    from universe import get_universe

    uni = get_universe()
    test_tickers = uni["symbol"].tolist()  # in practice: run full list; slice for a quick test
    data = bulk_download(test_tickers)
    shortlist = build_shortlist(data)
    print(shortlist.head(30))
    print(f"\nShortlist size: {len(shortlist)}")
