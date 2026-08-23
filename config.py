"""
Central configuration for the small-cap catalyst screener.
Adjust thresholds here rather than scattering magic numbers through the codebase.
"""

# --- Universe filter (Stage 1) ---
MIN_PRICE = 1.00        # filters out pure sub-penny junk; set to 0 to disable
MAX_PRICE = 20.00       # widened scope per latest request (was $10)
MAX_MARKET_CAP = 500_000_000    # ceiling to keep this in micro/small-cap territory
MIN_LAST_DAY_VOLUME = 300_000   # rough first-pass liquidity gate, applied at the bulk screener stage
MIN_AVG_DOLLAR_VOL = 500_000    # 20-day avg $ volume floor, applied later in the yfinance technical pass
MIN_RVOL = 1.5          # today's volume vs 20-day avg volume, flags "something is happening"

# --- Technical indicator windows ---
EMA_PERIODS = [20, 50, 200]
SMA_PERIODS = [50, 200]
RSI_PERIOD = 14
ATR_PERIOD = 14

# --- Data lookback ---
HISTORY_PERIOD = "1y"   # need >=200 trading days for SMA200/EMA200
HISTORY_INTERVAL = "1d"

# --- Batch settings for yfinance bulk download ---
BATCH_SIZE = 200        # tickers per yf.download() call, keeps requests well-behaved
REQUEST_PAUSE_SEC = 1.0 # polite pause between batches

# --- Country pre-filter (applied before any price/volume download) ---
# The free Nasdaq screener endpoint leaves "country" BLANK for US-domiciled
# companies and only populates it for foreign private issuers, so blank/NaN
# must be treated as United States, not excluded.
ALLOWED_COUNTRIES = {
    "United States", "",  # "" / NaN handled as US in code
    "Canada",
    # Europe
    "United Kingdom", "Ireland", "Germany", "France", "Netherlands",
    "Switzerland", "Sweden", "Norway", "Denmark", "Finland", "Belgium",
    "Spain", "Italy", "Portugal", "Austria",
    "Luxembourg", "Iceland", "Monaco", "Liechtenstein",
}

NASDAQ_SCREENER_URL = "https://api.nasdaq.com/api/screener/stocks"
# api.nasdaq.com blocks the default python-requests User-Agent; needs to
# look like a real browser request.
NASDAQ_SCREENER_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"),
    "Accept": "application/json",
}

# --- Free data source URLs (fallback universe source, no country field) ---
# Note: the old "old.nasdaqtrader.com" host has been retired and no longer
# resolves. Use www.nasdaqtrader.com instead.
NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

# --- Output ---
# Relative to this config.py file's own location, so it works regardless
# of what folder you run the scripts from -- creates an "output" subfolder
# right next to the scripts themselves.
import os as _os
OUTPUT_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "output")
SHORTLIST_FILENAME = "stage1_shortlist.xlsx"

# --- Stage 2: SEC EDGAR filing-based catalysts ---
# SEC's data.sec.gov actively enforces a descriptive User-Agent (org/name +
# contact email) and will block/429 generic agents. REPLACE the email below
# with your own before running -- this is required by SEC's fair-access policy.
SEC_USER_AGENT = "ipademailam@gmail.com"

SEC_TICKER_CIK_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL_TEMPLATE = "https://data.sec.gov/submissions/CIK{cik10}.json"

FILING_LOOKBACK_DAYS = 10  # how far back to scan each ticker's recent filings

# Form-type -> catalyst category mapping used to classify recent filings.
# Positive/neutral signal:
MATERIAL_EVENT_FORMS = {"8-K", "8-K/A"}
INSIDER_TRANSACTION_FORMS = {"4", "4/A"}
# Negative / risk-flag signal (dilution, going-concern, delisting):
DILUTION_FORMS = {"S-1", "S-1/A", "S-3", "S-3/A", "424B1", "424B2", "424B3",
                   "424B4", "424B5", "F-1", "F-1/A", "F-3"}
DELISTING_FORMS = {"25", "25-NSE", "15", "15-12B", "15-12G"}
OWNERSHIP_CHANGE_FORMS = {"SC 13D", "SC 13D/A", "SC 13G", "SC 13G/A"}

# --- Stage 3: composite scoring ---
# Points per catalyst event category (before recency decay is applied).
CATALYST_POINTS = {
    "material_event": 2,       # 8-K -- neutral-positive, needs volume confirmation
    "insider_buy": 3,          # Form 4, transaction code P (open-market purchase)
    "insider_sell": -1,        # Form 4, transaction code S (open-market sale)
    "insider_other": 0,        # Form 4, other codes (grants, exercises, gifts, tax withholding)
    "dilution_risk": -4,       # S-1/S-3/424B family
    "ownership_change": 1,     # 13D/13G -- context-dependent, mild positive by default
    "delisting_risk": -6,      # Form 15/25 -- treated as a hard risk flag, not just points
}

# Recency decay: an event's points are multiplied by max(0, 1 - days_ago/RECENCY_DECAY_DAYS).
# A same-day filing gets full weight; by RECENCY_DECAY_DAYS old it contributes ~0.
RECENCY_DECAY_DAYS = 10

# RVOL confirmation multiplier applied to the catalyst score: a catalyst with no
# volume reaction shouldn't score as high as one the market is actually trading.
# multiplier = clip(rvol / MIN_RVOL, RVOL_MULTIPLIER_FLOOR, RVOL_MULTIPLIER_CEILING)
RVOL_MULTIPLIER_FLOOR = 0.5
RVOL_MULTIPLIER_CEILING = 2.0

# Technical score components
GOLDEN_CROSS_BONUS = 2
RSI_SWEET_SPOT_RANGE = (40, 65)    # room to run without being overbought
RSI_SWEET_SPOT_POINTS = 2
RSI_MILD_HIGH_RANGE = (65, 75)
RSI_MILD_HIGH_POINTS = 1
RSI_OVERBOUGHT_THRESHOLD = 75
RSI_OVERBOUGHT_POINTS = -2
RSI_WEAK_THRESHOLD = 30
RSI_WEAK_POINTS = -1

# Composite score -> recommendation tier thresholds
TIER_THRESHOLDS = {
    "Strong": 8,
    "Moderate": 4,
    "Watch": 0,
    # anything below "Watch" threshold = "Avoid"
}

# --- Catalyst timing: has the move already happened? ---
CATALYST_TIMING_NO_PENALTY_MAX = 0.10    # up to +10% since catalyst: no penalty
CATALYST_TIMING_MILD_WARNING_MIN = 0.10  # +10% to +15%: tag only, no point penalty
CATALYST_TIMING_LATE_THRESHOLD = 0.15    # +15%+: late entry penalty kicks in
CATALYST_TIMING_LATE_PENALTY = -2        # points for late entry (+15% to +25%)
CATALYST_TIMING_VERY_LATE_THRESHOLD = 0.25  # +25%+: move likely priced in
CATALYST_TIMING_VERY_LATE_PENALTY = -4   # points for very late entry

# --- Fair value reference (contextual, NOT used in scoring) ---
FAIR_VALUE_MIN_SECTOR_SAMPLE = 3  # only show "vs sector" when ≥3 tickers share the sector

# --- Volatility-based 3-month price range ---
VOLATILITY_RANGE_TRADING_DAYS = 63  # ~3 months of trading days
VOLATILITY_RANGE_CONFIDENCE_SD = 1  # standard deviations (1 = ~68% probability band)

# --- Extended hours (pre-market / after-hours) data ---
EXTENDED_HOURS_ENABLED = True    # fetch AH/PM data for shortlisted tickers
EXTENDED_HOURS_INTERVAL = "5m"   # intraday resolution for AH/PM extraction
EXTENDED_HOURS_PERIOD = "5d"     # lookback for intraday (covers multiple AH sessions)
EXTENDED_HOURS_BATCH_SIZE = 10   # yfinance intraday is slower, use smaller batches
EXTENDED_HOURS_PAUSE_SEC = 2.0   # polite pause between intraday batches

# --- AI prompt / report output ---
AI_PROMPT_FILENAME = "ai_analysis_prompt.txt"
