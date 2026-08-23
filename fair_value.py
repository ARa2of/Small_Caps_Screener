"""
Fair value contextual reference: fetches yfinance fundamentals for each
shortlisted ticker and computes relative valuation metrics (P/S, P/B,
EV/EBITDA vs sector medians), analyst target price, and Graham number.

This is informational only — these fields are NOT used in scoring.
Micro-caps frequently have incomplete data, so each field gracefully
falls back to None / "N/A — insufficient data" when unavailable.
"""

import logging

import pandas as pd
import yfinance as yf

from config import FAIR_VALUE_MIN_SECTOR_SAMPLE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def _fetch_info(symbol: str) -> dict:
    """Fetch yfinance Ticker.info for a single symbol. Returns empty dict on failure."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
        return info
    except Exception as e:
        log.debug(f"Failed to fetch info for {symbol}: {e}")
        return {}


def _compute_sector_medians(infos: dict[str, dict]) -> dict[str, dict]:
    """
    Given a dict of {symbol: info_dict}, compute per-sector medians for
    P/S, P/B, and EV/EBITDA. Only includes sectors with >= FAIR_VALUE_MIN_SECTOR_SAMPLE
    tickers to avoid misleading comparisons.
    """
    sector_data: dict[str, list[dict]] = {}
    for symbol, info in infos.items():
        sector = info.get("sector")
        if not sector:
            continue
        ps = info.get("priceToSalesTrailing12Months")
        pb = info.get("priceToBook")
        ev_ebitda = info.get("enterpriseToEbitda")
        if ps is not None or pb is not None or ev_ebitda is not None:
            sector_data.setdefault(sector, []).append({
                "ps": ps, "pb": pb, "ev_ebitda": ev_ebitda
            })

    medians = {}
    for sector, values in sector_data.items():
        if len(values) < FAIR_VALUE_MIN_SECTOR_SAMPLE:
            continue
        df = pd.DataFrame(values)
        medians[sector] = {
            "ps": df["ps"].median() if df["ps"].notna().any() else None,
            "pb": df["pb"].median() if df["pb"].notna().any() else None,
            "ev_ebitda": df["ev_ebitda"].median() if df["ev_ebitda"].notna().any() else None,
            "count": len(values),
        }
    return medians


def _graham_number(info: dict) -> float | None:
    """
    Graham number = sqrt(22.5 * EPS * Book Value per share).
    Only valid when both trailing EPS and book value are positive.
    """
    eps = info.get("trailingEps")
    bv = info.get("bookValue")
    if eps is None or bv is None:
        return None
    if eps <= 0 or bv <= 0:
        return None
    try:
        return round((22.5 * eps * bv) ** 0.5, 2)
    except (TypeError, ValueError):
        return None


def _relative_label(ratio: float | None, median: float | None) -> str | None:
    """Returns a human-readable comparison label, or None if either value is missing."""
    if ratio is None or median is None or median == 0:
        return None
    multiple = ratio / median
    if multiple > 1.2:
        return f"{ratio:.1f}x vs {median:.1f}x median — trading at premium"
    elif multiple < 0.8:
        return f"{ratio:.1f}x vs {median:.1f}x median — trading at discount"
    else:
        return f"{ratio:.1f}x vs {median:.1f}x median — in-line"


def compute_fair_value(shortlist_df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches the shortlist DataFrame with fair value reference columns.
    Fetches yfinance .info for each ticker (slow for large lists — only
    call on the already-filtered shortlist, not the full universe).

    Returns the input DataFrame with new columns added:
      ps_ratio, ps_vs_sector, pb_ratio, pb_vs_sector,
      ev_ebitda, analyst_target, graham_number, fair_value_note
    """
    if shortlist_df.empty:
        return shortlist_df

    symbols = shortlist_df["symbol"].tolist()
    log.info(f"Fetching fair value data for {len(symbols)} tickers...")

    infos = {}
    for i, symbol in enumerate(symbols, 1):
        infos[symbol] = _fetch_info(symbol)
        if i % 20 == 0:
            log.info(f"Fair value fetch progress: {i}/{len(symbols)}")

    sector_medians = _compute_sector_medians(infos)
    log.info(f"Computed sector medians for {len(sector_medians)} sectors.")

    rows = []
    for _, row in shortlist_df.iterrows():
        symbol = row["symbol"]
        info = infos.get(symbol, {})
        sector = info.get("sector")
        medians = sector_medians.get(sector, {})

        ps = info.get("priceToSalesTrailing12Months")
        pb = info.get("priceToBook")
        ev_ebitda = info.get("enterpriseToEbitda")
        target = info.get("targetMeanPrice")
        graham = _graham_number(info)

        ps_vs = _relative_label(ps, medians.get("ps"))
        pb_vs = _relative_label(pb, medians.get("pb"))
        ev_vs = _relative_label(ev_ebitda, medians.get("ev_ebitda"))

        # Build human-readable note
        notes = []
        if ps_vs:
            notes.append(f"P/S {ps_vs}")
        if pb_vs:
            notes.append(f"P/B {pb_vs}")
        if ev_vs:
            notes.append(f"EV/EBITDA {ev_vs}")
        if target is not None:
            notes.append(f"Analyst target ${target:.2f}")
        if graham is not None:
            notes.append(f"Graham # ${graham:.2f}")

        if not notes:
            note = "N/A — insufficient data"
        else:
            note = " | ".join(notes)

        rows.append({
            "symbol": symbol,
            "ps_ratio": round(ps, 2) if ps is not None else None,
            "ps_vs_sector": ps_vs or ("N/A — insufficient data" if ps is None else "N/A — sector too small"),
            "pb_ratio": round(pb, 2) if pb is not None else None,
            "pb_vs_sector": pb_vs or ("N/A — insufficient data" if pb is None else "N/A — sector too small"),
            "ev_ebitda": round(ev_ebitda, 2) if ev_ebitda is not None else None,
            "analyst_target": round(target, 2) if target is not None else None,
            "graham_number": graham,
            "fair_value_note": note,
        })

    fair_value_df = pd.DataFrame(rows)
    result = shortlist_df.merge(fair_value_df, on="symbol", how="left")
    log.info(f"Fair value data computed for {len(result)} tickers.")
    return result
