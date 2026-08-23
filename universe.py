"""
Stage 1a: Build the tradable US equity ticker universe, pre-filtered by
country BEFORE any price/volume data is downloaded. This is the key
time-saver: instead of pulling yfinance history for 6,000+ tickers and
then discarding foreign issuers, we discard them first using a single
free bulk call.

Primary source: api.nasdaq.com's screener endpoint (undocumented but
widely used, free, no key). Returns symbol, name, country, sector,
market cap, last price for the full NASDAQ+NYSE+AMEX universe in one
call.

Fallback source: NASDAQ Trader's plain symbol directory files (no
country field -- used only if the screener endpoint is unreachable,
in which case country filtering falls back to a per-symbol check
later, at extra cost).
"""

import io
import logging

import pandas as pd
import requests

from config import (
    NASDAQ_SCREENER_URL, NASDAQ_SCREENER_HEADERS, ALLOWED_COUNTRIES,
    NASDAQ_LISTED_URL, OTHER_LISTED_URL,
    MIN_PRICE, MAX_PRICE, MAX_MARKET_CAP, MIN_LAST_DAY_VOLUME,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def _is_allowed_country(country: str) -> bool:
    """Blank/NaN country in the Nasdaq screener data means US-domiciled -- allowed."""
    if pd.isna(country) or str(country).strip() == "":
        return True
    return str(country).strip() in ALLOWED_COUNTRIES


def get_universe_via_screener() -> pd.DataFrame:
    """
    Primary path: pulls the full listed-stock universe (symbol, name,
    country, sector, market cap, last sale) from the Nasdaq screener
    API in one call, and filters to US/Canada/Europe-domiciled
    companies before returning. This avoids ever downloading OHLCV
    history for excluded countries.

    Returns columns: symbol, security_name, country, sector, market_cap, last_sale
    """
    params = {"tableonly": "true", "limit": "10000", "offset": "0", "download": "true"}
    resp = requests.get(NASDAQ_SCREENER_URL, headers=NASDAQ_SCREENER_HEADERS, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    rows = payload.get("data", {}).get("rows")
    if not rows:
        raise ValueError("Nasdaq screener API returned no rows -- response shape may have changed.")

    df = pd.DataFrame(rows)
    df = df.rename(columns={
        "symbol": "symbol",
        "name": "security_name",
        "country": "country",
        "sector": "sector",
        "marketCap": "market_cap",
        "lastsale": "last_sale",
        "volume": "last_day_volume",
    })

    keep_cols = [c for c in ["symbol", "security_name", "country", "sector",
                              "market_cap", "last_sale", "last_day_volume"]
                 if c in df.columns]
    df = df[keep_cols]

    # The screener API returns these as formatted strings (e.g. "$4.52",
    # "1,234,567", "$123,456,789.00") -- strip to numeric before filtering.
    for col in ["market_cap", "last_sale", "last_day_volume"]:
        if col in df.columns:
            df[col] = (df[col].astype(str)
                       .str.replace(r"[$,]", "", regex=True)
                       .str.strip())
            df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    if "last_sale" in df.columns:
        df = df[(df["last_sale"] >= MIN_PRICE) & (df["last_sale"] <= MAX_PRICE)]
    if "market_cap" in df.columns:
        # market_cap of 0/NaN usually means the field wasn't populated (common
        # for some small OTC-adjacent names) -- don't silently drop those on
        # cap alone, only drop names we KNOW exceed the ceiling.
        df = df[df["market_cap"].isna() | (df["market_cap"] <= MAX_MARKET_CAP)]
    if "last_day_volume" in df.columns:
        df = df[df["last_day_volume"].isna() | (df["last_day_volume"] >= MIN_LAST_DAY_VOLUME)]
    log.info(f"Price/market-cap/volume pre-filter: {before} -> {len(df)} tickers "
              f"(price ${MIN_PRICE}-${MAX_PRICE}, market cap <= ${MAX_MARKET_CAP:,.0f}, "
              f"last-day volume >= {MIN_LAST_DAY_VOLUME:,}).")

    before = len(df)
    df = df[df["country"].apply(_is_allowed_country)]
    log.info(f"Country pre-filter: {before} -> {len(df)} tickers "
              f"(kept US/Canada/Europe, dropped foreign private issuers).")

    # Basic ticker hygiene: drop symbols with odd suffixes (warrants/units/rights
    # typically show up as SYM.W, SYM.U, SYM/WS etc. in this feed too).
    df = df.dropna(subset=["symbol"])
    df["symbol"] = df["symbol"].astype(str).str.strip()
    df = df[df["symbol"] != ""]
    df = df[~df["symbol"].str.contains(r"[.\-/^]", regex=True, na=False)]
    df = df.drop_duplicates(subset=["symbol"]).reset_index(drop=True)

    return df


def get_universe_via_symbol_directory() -> pd.DataFrame:
    """
    Fallback path (no country field): NASDAQ Trader's plain symbol
    directory files. Use this only if the screener API is unreachable.
    Country filtering will need to happen later via a per-symbol
    lookup, which is slower and defeats the purpose of pre-filtering --
    treat this as a degraded-mode fallback, not the default.
    """
    def _fetch(url):
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        lines = resp.text.strip().split("\n")
        if lines and lines[-1].lower().startswith("file creation time"):
            lines = lines[:-1]
        return pd.read_csv(io.StringIO("\n".join(lines)), sep="|")

    log.warning("Falling back to NASDAQ Trader symbol directory -- no country data available. "
                "Country filtering will not be applied at this stage.")

    nasdaq = _fetch(NASDAQ_LISTED_URL)
    nasdaq = nasdaq.rename(columns={"Symbol": "symbol", "Security Name": "security_name"})
    nasdaq = nasdaq[(nasdaq["Test Issue"] == "N") & (nasdaq["ETF"] == "N")]
    nasdaq["exchange"] = "NASDAQ"

    other = _fetch(OTHER_LISTED_URL)
    other = other.rename(columns={"ACT Symbol": "symbol", "Security Name": "security_name",
                                   "Exchange": "exchange"})
    other = other[(other["Test Issue"] == "N") & (other["ETF"] == "N")]

    universe = pd.concat([nasdaq[["symbol", "security_name", "exchange"]],
                           other[["symbol", "security_name", "exchange"]]], ignore_index=True)
    universe = universe.dropna(subset=["symbol"]).drop_duplicates(subset=["symbol"])
    universe["symbol"] = universe["symbol"].astype(str).str.strip()
    universe = universe[~universe["symbol"].str.contains(r"[.\-]", regex=True, na=False)]
    return universe.reset_index(drop=True)


def get_universe() -> pd.DataFrame:
    """
    Entry point: tries the country-aware screener API first, falls back
    to the plain symbol directory (unfiltered by country) if that fails.
    """
    try:
        return get_universe_via_screener()
    except Exception as e:
        log.warning(f"Nasdaq screener API failed ({e}); using symbol-directory fallback.")
        return get_universe_via_symbol_directory()


if __name__ == "__main__":
    df = get_universe()
    print(df.head(20))
    print(f"\nTotal tickers after country pre-filter: {len(df)}")
