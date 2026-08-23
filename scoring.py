"""
Stage 3: composite scoring. Combines Stage 1 technicals with Stage 2 SEC
filing catalysts into a transparent, inspectable point score and a
Strong/Moderate/Watch/Avoid recommendation tier per ticker.

Design intent: every score is a sum of visible, labeled components (not
a black-box number), so each row in the final Excel output can show
WHY a ticker scored the way it did.
"""

import logging

import numpy as np
import pandas as pd

from config import (
    CATALYST_POINTS, RECENCY_DECAY_DAYS,
    RVOL_MULTIPLIER_FLOOR, RVOL_MULTIPLIER_CEILING,
    GOLDEN_CROSS_BONUS,
    RSI_SWEET_SPOT_RANGE, RSI_SWEET_SPOT_POINTS,
    RSI_MILD_HIGH_RANGE, RSI_MILD_HIGH_POINTS,
    RSI_OVERBOUGHT_THRESHOLD, RSI_OVERBOUGHT_POINTS,
    RSI_WEAK_THRESHOLD, RSI_WEAK_POINTS,
    TIER_THRESHOLDS, MIN_RVOL,
    CATALYST_TIMING_NO_PENALTY_MAX, CATALYST_TIMING_MILD_WARNING_MIN,
    CATALYST_TIMING_LATE_THRESHOLD, CATALYST_TIMING_LATE_PENALTY,
    CATALYST_TIMING_VERY_LATE_THRESHOLD, CATALYST_TIMING_VERY_LATE_PENALTY,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def _event_category_key(row: pd.Series) -> str:
    """
    Maps a filings_df row to the CATALYST_POINTS key to use. Form 4 rows
    use their resolved insider_direction (buy/sell/other); everything
    else uses its filing category directly.
    """
    if row["category"] == "insider_transaction":
        direction = row.get("insider_direction")
        if direction in ("insider_buy", "insider_sell", "insider_other"):
            return direction
        return "insider_other"  # unknown/unparseable treated as neutral, not penalized
    return row["category"]


def _recency_multiplier(filed_date: str) -> float:
    try:
        filed = pd.to_datetime(filed_date)
    except Exception:
        return 0.0
    days_ago = (pd.Timestamp.today().normalize() - filed).days
    return max(0.0, 1.0 - (days_ago / RECENCY_DECAY_DAYS))


def compute_catalyst_score(symbol_filings: pd.DataFrame) -> tuple[float, list[str], list[str]]:
    """
    Given a single ticker's filings (subset of filings_df), returns:
      (raw_catalyst_score, risk_flags, event_labels)
    risk_flags: list like ["dilution_risk", "delisting_risk"] for filings present.
    event_labels: human-readable one-line summaries of each contributing event,
                  for display in the output Excel so the score is inspectable.
    """
    score = 0.0
    risk_flags = []
    event_labels = []

    for _, row in symbol_filings.iterrows():
        key = _event_category_key(row)
        base_points = CATALYST_POINTS.get(key, 0)
        multiplier = _recency_multiplier(row["filed_date"])
        weighted = base_points * multiplier
        score += weighted

        if row["category"] in ("dilution_risk", "delisting_risk"):
            risk_flags.append(row["category"])

        event_labels.append(f"{row['filed_date']} {row['form']} ({key}, {weighted:+.1f}pts)")

    return score, sorted(set(risk_flags)), event_labels


def compute_technical_score(shortlist_row: pd.Series) -> tuple[float, list[str]]:
    """
    Returns (technical_score, component_labels) from a Stage 1 shortlist row.
    """
    score = 0.0
    labels = []

    trend = shortlist_row.get("trend_alignment_score")
    if pd.notna(trend):
        score += trend
        labels.append(f"trend_alignment {trend:+.0f}")

    if shortlist_row.get("golden_cross_today"):
        score += GOLDEN_CROSS_BONUS
        labels.append(f"golden_cross {GOLDEN_CROSS_BONUS:+.0f}")

    rsi = shortlist_row.get("rsi")
    if pd.notna(rsi):
        lo, hi = RSI_SWEET_SPOT_RANGE
        mlo, mhi = RSI_MILD_HIGH_RANGE
        if lo <= rsi <= hi:
            score += RSI_SWEET_SPOT_POINTS
            labels.append(f"rsi_sweet_spot {RSI_SWEET_SPOT_POINTS:+.0f}")
        elif mlo < rsi <= mhi:
            score += RSI_MILD_HIGH_POINTS
            labels.append(f"rsi_mild_high {RSI_MILD_HIGH_POINTS:+.0f}")
        elif rsi > RSI_OVERBOUGHT_THRESHOLD:
            score += RSI_OVERBOUGHT_POINTS
            labels.append(f"rsi_overbought {RSI_OVERBOUGHT_POINTS:+.0f}")
        elif rsi < RSI_WEAK_THRESHOLD:
            score += RSI_WEAK_POINTS
            labels.append(f"rsi_weak {RSI_WEAK_POINTS:+.0f}")

    return score, labels


def _find_nearest_close_after_date(price_df: pd.DataFrame, target_date: str) -> float | None:
    """
    Given a ticker's OHLCV DataFrame and a target date string, finds the
    Close price on or the nearest trading day AFTER that date.
    Returns None if no suitable price is found.
    """
    try:
        target = pd.to_datetime(target_date).normalize()
    except Exception:
        return None

    if price_df is None or price_df.empty:
        return None

    dates = price_df.index
    # Find first date >= target_date
    mask = dates >= target
    if not mask.any():
        # All dates are before target — stock may have been halted or delisted
        # Use the last available price as fallback
        return float(price_df["Close"].iloc[-1])

    idx = dates[mask][0]
    close = price_df.loc[idx, "Close"]
    return float(close) if pd.notna(close) else None


def compute_catalyst_timing(
    symbol: str,
    symbol_filings: pd.DataFrame,
    price_data: dict[str, pd.DataFrame],
) -> tuple[float, str, float | None]:
    """
    Checks if the stock has already moved significantly since its catalyst.

    Returns:
      (timing_penalty, timing_tag, pct_since_catalyst)
      - timing_penalty: points to subtract from composite score
      - timing_tag: "Early", "Moved", "Late Entry", "Late Entry — Priced In", or "N/A"
      - pct_since_catalyst: percentage change since catalyst date (None if unavailable)
    """
    if symbol_filings.empty:
        return 0.0, "N/A", None

    ticker_data = price_data.get(symbol)
    if ticker_data is None or ticker_data.empty:
        return 0.0, "N/A", None

    # Use the earliest catalyst filing for timing check
    try:
        earliest = symbol_filings["filed_date"].min()
    except Exception:
        return 0.0, "N/A", None

    catalyst_close = _find_nearest_close_after_date(ticker_data, earliest)
    if catalyst_close is None or catalyst_close <= 0:
        return 0.0, "N/A", None

    current_close = float(ticker_data["Close"].iloc[-1])
    if current_close <= 0:
        return 0.0, "N/A", None

    pct_change = (current_close - catalyst_close) / catalyst_close

    # Determine penalty and tag
    if pct_change >= CATALYST_TIMING_VERY_LATE_THRESHOLD:
        return CATALYST_TIMING_VERY_LATE_PENALTY, "Late Entry — Priced In", pct_change
    elif pct_change >= CATALYST_TIMING_LATE_THRESHOLD:
        return CATALYST_TIMING_LATE_PENALTY, "Late Entry", pct_change
    elif pct_change >= CATALYST_TIMING_MILD_WARNING_MIN:
        return 0.0, "Moved", pct_change
    elif pct_change >= CATALYST_TIMING_NO_PENALTY_MAX:
        return 0.0, "Early", pct_change
    else:
        # Negative or very small positive change — catalyst hasn't moved yet
        return 0.0, "Early", pct_change


def _rvol_multiplier(rvol: float) -> float:
    if pd.isna(rvol) or rvol <= 0:
        return RVOL_MULTIPLIER_FLOOR
    raw = rvol / MIN_RVOL
    return max(RVOL_MULTIPLIER_FLOOR, min(RVOL_MULTIPLIER_CEILING, raw))


def classify_tier(composite_score: float, risk_flags: list[str]) -> str:
    if "delisting_risk" in risk_flags:
        return "Avoid"

    if composite_score >= TIER_THRESHOLDS["Strong"]:
        tier = "Strong"
    elif composite_score >= TIER_THRESHOLDS["Moderate"]:
        tier = "Moderate"
    elif composite_score >= TIER_THRESHOLDS["Watch"]:
        tier = "Watch"
    else:
        tier = "Avoid"

    if "dilution_risk" in risk_flags and tier == "Strong":
        tier = "Watch"  # dilution risk caps the tier even if the score looks strong

    return tier


def build_scored_table(
    shortlist_df: pd.DataFrame,
    filings_df: pd.DataFrame,
    price_data: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """
    Merges Stage 1 technicals + Stage 2 filings into one scored,
    tiered recommendation table -- one row per shortlisted ticker.
    If price_data is provided, catalyst timing check is included.
    """
    rows = []

    for _, srow in shortlist_df.iterrows():
        symbol = srow["symbol"]
        symbol_filings = filings_df[filings_df["symbol"] == symbol] if not filings_df.empty else pd.DataFrame()

        catalyst_score, risk_flags, event_labels = compute_catalyst_score(symbol_filings)
        technical_score, technical_labels = compute_technical_score(srow)
        rvol_mult = _rvol_multiplier(srow.get("rvol"))

        # Catalyst timing check (requires price_data)
        timing_penalty = 0.0
        timing_tag = "N/A"
        pct_since_catalyst = None
        if price_data is not None:
            timing_penalty, timing_tag, pct_since_catalyst = compute_catalyst_timing(
                symbol, symbol_filings, price_data
            )

        composite = (catalyst_score * rvol_mult) + technical_score + timing_penalty
        tier = classify_tier(composite, risk_flags)

        rows.append({
            "symbol": symbol,
            "recommendation": tier,
            "composite_score": round(composite, 2),
            "catalyst_score_raw": round(catalyst_score, 2),
            "rvol_multiplier": round(rvol_mult, 2),
            "technical_score": round(technical_score, 2),
            "timing_penalty": round(timing_penalty, 2),
            "timing_tag": timing_tag,
            "pct_since_catalyst": round(pct_since_catalyst * 100, 1) if pct_since_catalyst is not None else None,
            "risk_flags": ", ".join(risk_flags) if risk_flags else "",
            "catalyst_events": " | ".join(event_labels) if event_labels else "none",
            "technical_notes": ", ".join(technical_labels) if technical_labels else "",
            "close": srow.get("close"),
            "rvol": srow.get("rvol"),
            "rsi": srow.get("rsi"),
            "ah_close": srow.get("ah_close"),
            "ah_volume": srow.get("ah_volume"),
            "ah_pct_move": srow.get("ah_pct_move"),
            "pm_open": srow.get("pm_open"),
            "pm_volume": srow.get("pm_volume"),
            "pm_pct_move": srow.get("pm_pct_move"),
            "last_ah_high": srow.get("last_ah_high"),
            "last_ah_low": srow.get("last_ah_low"),
            "has_extended_data": srow.get("has_extended_data", False),
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        tier_order = {"Strong": 0, "Moderate": 1, "Watch": 2, "Avoid": 3}
        result["_tier_sort"] = result["recommendation"].map(tier_order)
        result = result.sort_values(by=["_tier_sort", "composite_score"], ascending=[True, False])
        result = result.drop(columns=["_tier_sort"]).reset_index(drop=True)

    log.info(f"Scoring complete: {len(result)} tickers scored. "
             f"Tier counts: {result['recommendation'].value_counts().to_dict() if not result.empty else {}}")
    return result


if __name__ == "__main__":
    # smoke test with synthetic data
    shortlist = pd.DataFrame([
        {"symbol": "AAA", "close": 5.0, "rvol": 4.0, "rsi": 55, "trend_alignment_score": 4, "golden_cross_today": True},
        {"symbol": "BBB", "close": 12.0, "rvol": 1.0, "rsi": 80, "trend_alignment_score": 1, "golden_cross_today": False},
    ])
    filings = pd.DataFrame([
        {"symbol": "AAA", "form": "8-K", "filed_date": pd.Timestamp.today().date().isoformat(),
         "category": "material_event", "insider_direction": None},
        {"symbol": "AAA", "form": "4", "filed_date": pd.Timestamp.today().date().isoformat(),
         "category": "insider_transaction", "insider_direction": "insider_buy"},
        {"symbol": "BBB", "form": "S-1", "filed_date": pd.Timestamp.today().date().isoformat(),
         "category": "dilution_risk", "insider_direction": None},
    ])
    scored = build_scored_table(shortlist, filings, price_data=None)
    print(scored.to_string())
