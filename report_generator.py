"""
AI Prompt / Analysis Report Generator

After the screener pipeline runs, this module generates a comprehensive
text file that can be pasted into any free AI agent (ChatGPT, Claude,
Gemini web, etc.) for qualitative analysis.

The prompt includes:
  1. Methodology explanation (how the Excel was created, what each column means)
  2. Screener output data (condensed for the AI to reason over)
  3. Specific analysis instructions (catalyst verification, news, fair value)
  4. Suggested follow-up questions for the user to ask

Output: output/ai_analysis_prompt.txt
"""

import os
from datetime import datetime

import pandas as pd

from config import OUTPUT_DIR, AI_PROMPT_FILENAME


def _fmt(val, decimals=2):
    if val is None or pd.isna(val):
        return "N/A"
    try:
        return f"{float(val):,.{decimals}f}"
    except (TypeError, ValueError):
        return str(val)


def _build_methodology_section() -> str:
    return """
================================================================================
  METHODOLOGY: HOW THIS SCREENER WORKS
================================================================================

This screener scans US-listed stocks priced $1-$20 with market cap under $500M
(micro/small-cap territory). It runs in 3 stages:

STAGE 1 — UNIVERSE & TECHNICAL FILTER
  Source: Nasdaq screener API + yfinance (free, no API key)
  - Fetches all US-listed stocks, filters by price ($1-$20), market cap (<$500M),
    and minimum volume ($500K avg dollar volume over 20 days)
  - Applies technical trigger: either Relative Volume (RVOL) >= 1.5x today's
    volume vs 20-day average, OR a fresh golden cross (SMA50 crossing above SMA200)
  - Computes indicators: EMA(20/50/200), SMA(50/200), RSI(14), ATR(14), trend
    alignment score (0-4 based on price vs moving averages)

STAGE 2 — SEC EDGAR CATALYST DETECTION
  Source: SEC EDGAR (free, public)
  - Scans each shortlisted ticker's recent SEC filings (last 10 days)
  - Classifies forms into catalyst categories:
      8-K / 8-K/A        → material_event (+2 pts base, adjusted by item)
      Form 4 insider buy  → insider_buy (+3 pts)
      Form 4 insider sell → insider_sell (-1 pt)
      S-1/S-3/424B/F-1    → dilution_risk (-4 pts, caps tier to Watch)
      Form 15/25          → delisting_risk (-6 pts, hard Avoid)
      SC 13D/13G          → ownership_change (+1 pt)
  - 8-K items are parsed for significance:
      Item 1.01 (Material Agreement)  → +3 pts bonus
      Item 2.01 (Acquisition)         → +4 pts bonus
      Item 1.03 (Bankruptcy)          → -4 pts
      Item 3.01 (Delisting)           → -6 pts
      Item 3.02 (Unregistered Equity) → -3 pts
      + keyword signals in title (merger, termination, offering, etc.)
  - Form 4 XML is parsed for transaction codes: P/A = buy, S = sell
  - Points decay over 10 days (same-day = full weight, 10 days = ~0)

STAGE 3 — COMPOSITE SCORING & TIERING
  Composite = (Catalyst Score × RVOL Multiplier) + Technical Score + Timing Penalty

  RVOL Multiplier: clip(RVOL / 1.5, floor=0.5, ceiling=2.0)
    → Low-volume names get penalized, high-volume names get boosted

  Technical Score:
    + trend_alignment (0-4 pts)
    + golden_cross_today (+2 pts)
    + RSI sweet spot 40-65 (+2 pts) / mild high 65-75 (+1) / overbought >75 (-2) / weak <30 (-1)

  Catalyst Timing Penalty:
    Compares current price vs price on/near the catalyst filing date:
      0-10% up  → "Early" (no penalty — move hasn't happened yet)
      10-15% up → "Moved" (tag only)
      15-25% up → "Late Entry" (-2 pts — you may be chasing)
      >25% up   → "Late Entry — Priced In" (-4 pts — move likely done)
      Down/neg  → "Early" (catalyst hasn't worked yet — potential opportunity)

  Tier Classification:
    composite >= 8  → Strong
    composite >= 4  → Moderate
    composite >= 0  → Watch
    composite < 0   → Avoid
    (dilution_risk caps tier to Watch; delisting_risk = hard Avoid)

STAGE 4 — RUN-TO-RUN CHURN TRACKING
  Persists ticker state between runs to detect:
  - first_seen_date: when the ticker first appeared in any run
  - consecutive_runs: how many consecutive runs (up to 4-day gap for weekends)
  - score_trend: "rising" (score improving), "falling", "stable", or "new"
  - Names appearing 2-3 consecutive runs with rising scores have the best edge

FAIR VALUE (contextual reference, NOT used in scoring):
  - P/S, P/B, EV/EBITDA ratios vs sector medians (where available)
  - Analyst mean target price (from yfinance, many micro-caps have none)
  - Graham Number = sqrt(22.5 × EPS × Book Value) for asset-heavy names
  - Labeled "N/A — insufficient data" where data is missing

VOLATILITY RANGE (statistical, NOT a prediction):
  - 1-SD 3-month price range based on historical daily log-return volatility
  - "Based on this stock's volatility, ~68% of 3-month outcomes fall within $X-$Y"
  - Also shows ATR-based range as a secondary reference

EXTENDED HOURS DATA (after-hours / pre-market):
  - Intraday data (5-min bars) with prepost=True captures AH (4PM-8PM ET) and PM (4AM-9:30AM ET) sessions
  - AH Close: last after-hours close price (regular close + AH session activity)
  - AH % Move: % change from last regular close to AH close
  - AH Volume: total volume across all AH sessions in last 5 days
  - AH Range: highest/lowest prices during AH sessions
  - PM Open: last pre-market open price
  - PM % Move: % change from previous regular close to PM open
  - WHY THIS MATTERS: SEC filings (8-K, earnings) are released after market close. AH volume/price action
    shows institutional/smart money reacting to catalysts before the next regular session opens.
    A large AH move on high volume = the catalyst is being actively traded, not just filed.
"""


def _build_data_section(scored_df: pd.DataFrame) -> str:
    """Build the scored data table in a format the AI can reason over."""
    lines = [
        "================================================================================",
        "  SCREENER OUTPUT: TODAY'S SCORED STOCKS",
        "================================================================================",
        "",
        "Columns: SYMBOL | TIER | SCORE | CATALYST_RAW | RVOL_MULT | TECH_SCORE |",
        "         TIMING_PENALTY | TIMING_TAG | %_SINCE_CATALYST | RISK_FLAGS |",
        "         CLOSE | RVOL | RSI | CATALYST_EVENTS | TECHNICAL_NOTES |",
        "         PS_RATIO | PB_RATIO | EV_EBITDA | ANALYST_TARGET | GRAHAM_NUMBER |",
        "         FAIR_VALUE_NOTE | VOL_3MO_LOWER | VOL_3MO_UPPER | ANN_VOL% |",
        "         AH_CLOSE | AH_VOLUME | AH_%_MOVE | PM_OPEN | PM_VOLUME | PM_%_MOVE |",
        "         AH_HIGH | AH_LOW",
        "",
    ]

    if scored_df.empty:
        lines.append("  (No stocks passed filters today)")
        return "\n".join(lines)

    # Sort by tier then score
    tier_order = {"Strong": 0, "Moderate": 1, "Watch": 2, "Avoid": 3}
    df = scored_df.copy()
    df["_sort"] = df["recommendation"].map(tier_order).fillna(4)
    df = df.sort_values(["_sort", "composite_score"], ascending=[True, False])

    for _, row in df.iterrows():
        lines.append(f"--- {row['symbol']} ---")
        lines.append(f"  Tier: {row.get('recommendation', 'N/A')}")
        lines.append(f"  Composite Score: {_fmt(row.get('composite_score'))}")
        lines.append(f"  Catalyst Score (raw): {_fmt(row.get('catalyst_score_raw'))}")
        lines.append(f"  RVOL Multiplier: {_fmt(row.get('rvol_multiplier'))}x")
        lines.append(f"  Technical Score: {_fmt(row.get('technical_score'))}")
        lines.append(f"  Timing Penalty: {_fmt(row.get('timing_penalty'))} pts")
        lines.append(f"  Timing Tag: {row.get('timing_tag', 'N/A')}")
        lines.append(f"  % Since Catalyst: {_fmt(row.get('pct_since_catalyst'), 1)}%")
        lines.append(f"  Risk Flags: {row.get('risk_flags', 'none')}")
        lines.append(f"  Close: ${_fmt(row.get('close'))}")
        lines.append(f"  RVOL: {_fmt(row.get('rvol'))}x")
        lines.append(f"  RSI: {_fmt(row.get('rsi'), 1)}")

        # Churn tracking
        first_seen = row.get("first_seen_date")
        consec = row.get("consecutive_runs")
        trend = row.get("score_trend")
        prev_sc = row.get("prev_score")
        if first_seen and trend:
            churn_info = f"  First Seen: {first_seen}"
            if consec and consec > 1:
                churn_info += f" | Consecutive Runs: {consec}"
            if trend == "rising":
                churn_info += " | Trend: RISING (score improving)"
            elif trend == "falling":
                churn_info += " | Trend: FALLING"
            elif trend == "stable":
                churn_info += " | Trend: STABLE"
            if prev_sc is not None:
                churn_info += f" | Prev Score: {_fmt(prev_sc)}"
            lines.append(churn_info)

        events = row.get("catalyst_events", "none")
        if events and events != "none":
            lines.append(f"  Catalyst Events: {events[:200]}")

        tech_notes = row.get("technical_notes", "")
        if tech_notes:
            lines.append(f"  Technical Notes: {tech_notes[:150]}")

        # Fair value
        ps = row.get("ps_ratio")
        pb = row.get("pb_ratio")
        ev = row.get("ev_ebitda")
        target = row.get("analyst_target")
        graham = row.get("graham_number")
        fv_note = row.get("fair_value_note", "")

        has_fv = any(v is not None and not pd.isna(v) for v in [ps, pb, ev, target, graham])
        if has_fv or (fv_note and fv_note != "N/A — insufficient data"):
            lines.append(f"  Fair Value: P/S={_fmt(ps)} | P/B={_fmt(pb)} | EV/EBITDA={_fmt(ev)}")
            lines.append(f"    Analyst Target: ${_fmt(target)} | Graham #: ${_fmt(graham)}")
            lines.append(f"    Note: {fv_note[:120]}")

        # Volatility range
        vol_lo = row.get("vol_3mo_lower")
        vol_hi = row.get("vol_3mo_upper")
        ann_vol = row.get("annualized_vol")
        if vol_lo is not None and not pd.isna(vol_lo):
            lines.append(f"  3-Mo Vol Range: ${_fmt(vol_lo)} - ${_fmt(vol_hi)} (ann. {_fmt(ann_vol, 0)}%)")

        # Extended hours (AH/PM) data
        if row.get("has_extended_data"):
            ah_close = row.get("ah_close")
            ah_vol = row.get("ah_volume")
            ah_pct = row.get("ah_pct_move")
            pm_open = row.get("pm_open")
            pm_vol = row.get("pm_volume")
            pm_pct = row.get("pm_pct_move")
            ah_hi = row.get("last_ah_high")
            ah_lo = row.get("last_ah_low")

            has_ah = ah_close is not None
            has_pm = pm_open is not None

            if has_ah or has_pm:
                lines.append(f"  Extended Hours:")
                if has_ah:
                    lines.append(f"    AH Close: ${_fmt(ah_close)} | AH Vol: {_fmt(ah_vol, 0)} | AH Move: {_fmt(ah_pct, 2)}%")
                    if ah_hi is not None and ah_lo is not None:
                        lines.append(f"    AH Range: ${_fmt(ah_lo)} - ${_fmt(ah_hi)}")
                if has_pm:
                    lines.append(f"    PM Open: ${_fmt(pm_open)} | PM Vol: {_fmt(pm_vol, 0)} | PM Move: {_fmt(pm_pct, 2)}%")
                if has_ah and ah_pct is not None and not pd.isna(ah_pct):
                    if float(ah_pct) >= 5:
                        lines.append(f"    >> SIGNIFICANT AH MOVE: +{_fmt(ah_pct, 1)}% — catalyst may be driving after-hours interest")
                    elif float(ah_pct) <= -5:
                        lines.append(f"    >> AH SELLOFF: {_fmt(ah_pct, 1)}% — caution, possible negative catalyst reaction")

        lines.append("")

    return "\n".join(lines)


def _build_analysis_instructions() -> str:
    return """
================================================================================
  ANALYSIS INSTRUCTIONS FOR THE AI
================================================================================

You are a micro-cap/small-cap catalyst swing trading analyst. Based on the
screener data above, perform the following analysis:

1. CATALYST VERIFICATION & NEWS CHECK
   For each stock rated "Strong" or "Moderate":
   - Search for recent news about the company (last 7-14 days)
   - Check what the SEC filing (8-K, Form 4, etc.) was actually about
   - Determine if the catalyst impact is POSITIVE, NEGATIVE, or NEUTRAL
   - Note if the stock has already reacted to the news (price move since filing)
   - Check AH data: did the stock move in after-hours after the filing?
   - Flag any upcoming events (earnings, FDA decisions, etc.)

2. CATALYST FRESHNESS ASSESSMENT
   - Is the timing_tag "Early"? → Catalyst is fresh, move may still be developing
   - Is the timing_tag "Late Entry"? → Stock already moved 15-25%, chasing risk
   - Is the timing_tag "Late Entry — Priced In"? → Stock moved 25%+, likely done
   - For "Early" stocks: is there a reason the catalyst HASN'T moved the stock yet?
     (low awareness, market-wide weakness, waiting for confirmation?)
   - AH data adds context: a stock may look "Early" on regular hours but have a
     large AH move — that's the catalyst working in real-time

3. EXTENDED HOURS SIGNALS
   - Check AH % Move for each stock with extended data
   - AH move on HIGH AH volume = institutional reaction (reliable signal)
   - AH move on LOW AH volume = low-liquidity noise (less reliable)
   - If AH % is significantly different from regular day %, the market is repricing overnight

3. FAIR VALUE CHECK (your judgment, not just the numbers)
   - The screener provides P/S, P/B, EV/EBITDA, analyst target, Graham number
   - Many micro-caps have incomplete data — that's expected
   - For stocks with data: are they trading at a discount or premium to fair value?
   - For stocks without data: what's your assessment based on the business?

4. TECHNICAL SETUP QUALITY
   - RSI in 40-65 sweet spot = room to run
   - Golden cross = bullish momentum confirmation
   - High RVOL = market is paying attention
   - Trend alignment 3-4 = strong uptrend

5. RISK ASSESSMENT
   - dilution_risk flag = company may be issuing shares (dilutive)
   - delisting_risk flag = avoid at all costs
   - High % since catalyst = chasing, giving back edge
   - Low RVOL = nobody's watching, exit may be hard

6. EXTENDED HOURS ANALYSIS (AH/PM data)
   - AH % Move >= +5% on high volume = market is actively trading the catalyst
   - AH % Move >= +10% = strong institutional reaction, likely to gap up at open
   - AH % Move <= -5% = possible negative reaction to filing, caution
   - PM % Move shows pre-market sentiment for today's session
   - Compare AH volume to regular volume: high AH/regular ratio = smart money active
   - AH data helps distinguish "filed but nobody noticed" from "filed and being traded"

7. CHURN TRACKING & MOMENTUM
   - "rising" trend + 2-3 consecutive runs = name is building momentum — strongest edge
   - "new" ticker = fresh appearance, needs monitoring on next run
   - "falling" trend = losing steam, catalyst may be fading
   - Look for tickers that appeared in previous runs AND have rising scores — these
     are names the market keeps revisiting, not one-day flukes

8. RANKED SHORTLIST
   Provide a ranked list of the top 5-8 stocks with:
   - 1-line thesis for each (why it's interesting)
   - Key risk for each
   - Suggested action: DEEP DIVE / WATCH / SKIP
   - For DEEP DIVE stocks: what specifically should the user research further?

9. OVERALL MARKET ASSESSMENT
   - Is the current environment favorable for micro-cap catalyst plays?
   - Any sector themes or patterns you notice?
   - Any macro risks to be aware of?
"""


def _build_follow_up_questions() -> str:
    return """
### CONVERSATION STARTER ###

Now that you have the data, here's how to start:

FIRST QUESTION (copy this):
  "Based on the screener data above, give me your top 5 picks with a 1-line
   thesis for each. For each, search for the latest news and tell me:
   (a) what the catalyst actually is, (b) whether the news is positive or
   negative, (c) if the move has already happened or is still developing.
   Flag any stocks I should avoid."

THEN ASK FOLLOW-UPS LIKE:
  - "Deep dive on [TICKER] — what does the 8-K say? Any recent news?"
  - "Compare [A] vs [B] — which has better risk/reward?"
  - "Is the insider buying on [TICKER] significant? What's the context?"
  - "What's the float/short interest on [TICKER]?"
  - "Any upcoming earnings or catalysts for [TICKER]?"
  - "Give me a trade plan for [TICKER] — entry, stop, target"
  - "What's the sector outlook for [TICKER's sector]?"
  - "Which stocks should I absolutely avoid and why?"
  - "Is the market environment good for small-cap catalyst plays right now?"
  - "Re-run with a focus on [specific criteria]"
  - "Which stocks had the biggest after-hours moves? Are those moves justified?"
  - "For stocks with AH data, is the AH volume significant relative to regular volume?"
  - "Is the AH move likely to hold at the open, or is it a low-volume AH spike?"
  - "Which stocks have rising scores across multiple runs? Are they building momentum?"
  - "For stocks with consecutive runs, is the trend driven by the same catalyst or new ones?"

You can also just paste the Excel file (output/stage1_shortlist.xlsx)
if you want the AI to see the full formatted data.
"""


def generate_ai_prompt(scored_df: pd.DataFrame) -> str:
    """
    Generate the full AI analysis prompt text.
    Designed to be pasted ONCE into any AI agent — after that, the user
    asks questions in natural conversation.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    header = f"""
### SYSTEM CONTEXT — US SMALL CAPS SCREENER ###

You are acting as a micro-cap/small-cap catalyst swing trading analyst.
A Python screener pipeline has just run and produced the data below.
Your job is to help me (the trader) make sense of it.

The data is from: {timestamp}

WHAT YOU HAVE:
- A scored list of US stocks ($1-$20, micro/small-cap) that passed
  technical and catalyst filters
- Each stock has a composite score, tier rating, catalyst timing assessment,
  fair value reference data, and a 3-month volatility range
- The methodology is explained below the data

WHAT YOU CAN DO (use web search when needed):
- Verify what SEC filings (8-K, Form 4, etc.) actually say
- Check recent news and whether it's positive/negative for each stock
- Look up insider trading context, sector trends, upcoming events
- Assess if a catalyst is fresh or already priced in
- Check fair value, float, short interest, any public info

HOW THIS WORKS:
1. I'll paste this context block once (you already have it)
2. Then I'll ask questions in plain English
3. You answer using the data below + your web search + general knowledge
4. We go back and forth until I'm satisfied

RULES:
- Be direct and opinionated. Don't hedge with "it depends."
- Use the data for facts. Use web search for news/context.
- When I ask about a specific stock, search for its latest news and filings.
- Telegram/markdown formatting is fine. No tables needed.
- Max 300 words per answer unless I ask for detail.

### SCREENER DATA ###

"""

    methodology = _build_methodology_section()
    data = _build_data_section(scored_df)
    instructions = _build_analysis_instructions()
    follow_ups = _build_follow_up_questions()

    summary_stats = _build_summary_stats(scored_df)

    return header + summary_stats + methodology + data + instructions + follow_ups


def _build_summary_stats(scored_df: pd.DataFrame) -> str:
    """Quick stats summary at the top of the prompt."""
    lines = [
        "================================================================================",
        "  QUICK STATS",
        "================================================================================",
        "",
    ]

    if scored_df.empty:
        lines.append("  No stocks passed filters today.")
        return "\n".join(lines)

    total = len(scored_df)
    tier_counts = scored_df["recommendation"].value_counts()
    lines.append(f"  Total stocks scanned: {total}")
    for rec in ["Strong", "Moderate", "Watch", "Avoid"]:
        c = tier_counts.get(rec, 0)
        lines.append(f"    {rec}: {c}")

    # Timing breakdown
    timing_counts = scored_df["timing_tag"].value_counts()
    lines.append(f"  Catalyst Timing:")
    for tag in ["Early", "Moved", "Late Entry", "Late Entry — Priced In", "N/A"]:
        c = timing_counts.get(tag, 0)
        if c > 0:
            lines.append(f"    {tag}: {c}")

    # Risk flags
    risk_stocks = scored_df[scored_df["risk_flags"].astype(str).str.strip().astype(bool)]
    if not risk_stocks.empty:
        lines.append(f"  Stocks with risk flags: {len(risk_stocks)}")
        for _, row in risk_stocks.iterrows():
            lines.append(f"    {row['symbol']}: {row['risk_flags']}")

    lines.append("")
    return "\n".join(lines)


def export_ai_prompt(scored_df: pd.DataFrame, filename: str = AI_PROMPT_FILENAME) -> str:
    """
    Write the AI prompt to a text file in OUTPUT_DIR.
    Returns the full output path.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, filename)

    prompt_text = generate_ai_prompt(scored_df)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(prompt_text)

    return out_path
