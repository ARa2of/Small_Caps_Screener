"""
Run-to-run churn tracking: persists ticker state between pipeline runs
to detect names that appear across multiple consecutive runs with rising
scores — these are the names with the strongest tradable edge.

State file: output/screener_state.json
  {ticker: {first_seen, last_seen, consecutive_runs, last_score, scores_history}}
"""

import json
import logging
import os
from datetime import date

import pandas as pd

from config import OUTPUT_DIR

log = logging.getLogger(__name__)

STATE_FILENAME = "screener_state.json"


def _state_path() -> str:
    return os.path.join(OUTPUT_DIR, STATE_FILENAME)


def load_state() -> dict:
    """Load previous run state from JSON. Returns empty dict on first run."""
    path = _state_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log.warning(f"Failed to load state file {path}: {e}")
        return {}


def save_state(state: dict) -> None:
    """Persist state to JSON for next run."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = _state_path()
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        log.info(f"State saved to {path} ({len(state)} tickers tracked).")
    except Exception as e:
        log.warning(f"Failed to save state file {path}: {e}")


def update_churn_tracking(scored_df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches the scored DataFrame with churn-tracking columns:
      - first_seen_date: date this ticker first appeared in any run
      - consecutive_runs: how many consecutive runs this ticker has been present
      - score_trend: 'rising', 'falling', 'stable', or 'new'
      - prev_score: composite score from the previous run (if available)

    Persists state for the next run.
    """
    if scored_df.empty:
        return scored_df

    today = date.today().isoformat()
    prev_state = load_state()

    new_state = {}
    rows = []

    for _, row in scored_df.iterrows():
        symbol = row["symbol"]
        current_score = row.get("composite_score", 0)

        prev = prev_state.get(symbol)
        if prev:
            # Ticker was seen before
            first_seen = prev["first_seen"]
            last_seen_prev = prev.get("last_seen", "")
            prev_score = prev.get("last_score", 0)
            was_yesterday = _is_consecutive_day(last_seen_prev, today)
            consecutive = prev.get("consecutive_runs", 1)

            if was_yesterday:
                consecutive += 1
                score_trend = "rising" if current_score > prev_score else (
                    "falling" if current_score < prev_score else "stable"
                )
            else:
                # Gap in runs — reset consecutive count
                consecutive = 1
                score_trend = "new"

            new_state[symbol] = {
                "first_seen": first_seen,
                "last_seen": today,
                "consecutive_runs": consecutive,
                "last_score": current_score,
            }
        else:
            # New ticker
            first_seen = today
            consecutive = 1
            prev_score = None
            score_trend = "new"

            new_state[symbol] = {
                "first_seen": today,
                "last_seen": today,
                "consecutive_runs": 1,
                "last_score": current_score,
            }

        rows.append({
            "first_seen_date": first_seen,
            "consecutive_runs": consecutive,
            "score_trend": score_trend,
            "prev_score": prev_score,
        })

    # Merge churn columns into scored_df
    churn_df = pd.DataFrame(rows, index=scored_df.index)
    result = pd.concat([scored_df, churn_df], axis=1)

    # Save state for next run
    save_state(new_state)

    # Log stats
    new_count = sum(1 for r in rows if r["score_trend"] == "new")
    rising_count = sum(1 for r in rows if r["score_trend"] == "rising")
    consecutive_count = sum(1 for r in rows if r["consecutive_runs"] >= 2)
    log.info(f"Churn tracking: {new_count} new, {rising_count} rising, "
             f"{consecutive_count} appearing 2+ consecutive runs.")

    return result


def _is_consecutive_day(date_str: str, today: str) -> bool:
    """Check if date_str is yesterday or today (allowing for weekends/holidays)."""
    if not date_str:
        return False
    try:
        prev = pd.to_datetime(date_str).date()
        curr = pd.to_datetime(today).date()
        delta = (curr - prev).days
        # Allow up to 4 days gap (covers weekends + 1 holiday)
        return 0 <= delta <= 4
    except Exception:
        return False
