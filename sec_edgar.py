"""
Stage 2a: SEC EDGAR filing-based catalyst detection (free, no API key).

Two free SEC data sources, used together:
1. company_tickers.json  -- maps ticker symbol -> CIK (needed once per run,
   cached in memory, NOT per-ticker).
2. data.sec.gov/submissions/CIK##########.json -- one call per shortlisted
   ticker, returns that company's recent filings (form type + date), which
   we classify into catalyst categories.

This intentionally only runs against the Stage 1 shortlist (tens/hundreds
of tickers), not the full market, to stay well within SEC's fair-access
rate expectations (~10 req/s, descriptive User-Agent required).
"""

import logging
import time

import pandas as pd
import requests

from config import (
    SEC_USER_AGENT, SEC_TICKER_CIK_MAP_URL, SEC_SUBMISSIONS_URL_TEMPLATE,
    FILING_LOOKBACK_DAYS, MATERIAL_EVENT_FORMS, INSIDER_TRANSACTION_FORMS,
    DILUTION_FORMS, DELISTING_FORMS, OWNERSHIP_CHANGE_FORMS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

HEADERS = {"User-Agent": SEC_USER_AGENT, "Accept": "application/json"}

_TICKER_CIK_CACHE: dict[str, str] | None = None


def load_ticker_cik_map() -> dict[str, str]:
    """
    Fetches SEC's free ticker->CIK mapping once and caches it in memory
    for the life of the process. Returns {TICKER: 10-digit zero-padded CIK}.
    """
    global _TICKER_CIK_CACHE
    if _TICKER_CIK_CACHE is not None:
        return _TICKER_CIK_CACHE

    log.info("Fetching SEC ticker -> CIK mapping...")
    resp = requests.get(SEC_TICKER_CIK_MAP_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    raw = resp.json()  # dict of {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}

    mapping = {}
    for entry in raw.values():
        ticker = str(entry.get("ticker", "")).upper().strip()
        cik = entry.get("cik_str")
        if ticker and cik is not None:
            mapping[ticker] = str(cik).zfill(10)

    _TICKER_CIK_CACHE = mapping
    log.info(f"Loaded {len(mapping)} ticker -> CIK mappings.")
    return mapping


def _classify_form(form: str) -> str | None:
    form = (form or "").strip()
    if form in MATERIAL_EVENT_FORMS:
        return "material_event"
    if form in INSIDER_TRANSACTION_FORMS:
        return "insider_transaction"
    if form in DILUTION_FORMS:
        return "dilution_risk"
    if form in DELISTING_FORMS:
        return "delisting_risk"
    if form in OWNERSHIP_CHANGE_FORMS:
        return "ownership_change"
    return None


def _build_document_url(cik10: str, accession_number: str, primary_document: str) -> str:
    """
    Constructs the SEC Archives URL for a specific filing document.
    e.g. https://www.sec.gov/Archives/edgar/data/{cik}/{accession-no-dashes}/{doc}
    """
    cik_no_leading_zeros = str(int(cik10))
    accession_no_dashes = accession_number.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{cik_no_leading_zeros}/{accession_no_dashes}/{primary_document}"


def classify_form4_direction(cik10: str, accession_number: str, primary_document: str) -> str:
    """
    Fetches a Form 4 filing's XML and classifies it as 'insider_buy',
    'insider_sell', 'insider_other', or 'insider_unknown' (fetch/parse
    failure) based on transaction codes:
      P = open-market or private purchase  -> insider_buy
      S = open-market or private sale      -> insider_sell
      anything else (A/M/G/F/etc.)         -> insider_other
    A filing can contain multiple transactions; if it contains ANY 'P'
    code we call it insider_buy (a purchase alongside routine tax-
    withholding sales is still a meaningful buy signal), else if it
    contains any 'S' we call it insider_sell, else insider_other.
    """
    import xml.etree.ElementTree as ET

    url = _build_document_url(cik10, accession_number, primary_document)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
    except Exception as e:
        log.debug(f"Form 4 fetch/parse failed for {url}: {e}")
        return "insider_unknown"

    codes = set()
    for tx in root.iter():
        if tx.tag.endswith("transactionCode"):
            if tx.text:
                codes.add(tx.text.strip().upper())

    if "P" in codes:
        return "insider_buy"
    if "S" in codes:
        return "insider_sell"
    if codes:
        return "insider_other"
    return "insider_unknown"


def get_recent_filings(cik10: str, lookback_days: int = FILING_LOOKBACK_DAYS) -> pd.DataFrame:
    """
    Fetches a company's recent filings from the SEC submissions API and
    returns those within the lookback window, classified by catalyst
    category. Returns an empty DataFrame if the company has no matching
    recent filings or the lookup fails.
    """
    url = SEC_SUBMISSIONS_URL_TEMPLATE.format(cik10=cik10)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log.debug(f"Submissions lookup failed for CIK {cik10}: {e}")
        return pd.DataFrame()

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])

    if not forms:
        return pd.DataFrame()

    cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=lookback_days)

    rows = []
    for i, form in enumerate(forms):
        try:
            filed = pd.to_datetime(dates[i])
        except Exception:
            continue
        if filed < cutoff:
            continue
        category = _classify_form(form)
        if category is None:
            continue  # not a catalyst-relevant form type, skip

        accession = accessions[i] if i < len(accessions) else None
        primary_doc = primary_docs[i] if i < len(primary_docs) else None

        # Form 4 filings are only tagged "insider_transaction" by form type --
        # resolve buy vs. sell vs. other by parsing the actual XML transaction
        # codes, since that's the part that actually carries signal.
        insider_direction = None
        if category == "insider_transaction" and accession and primary_doc:
            insider_direction = classify_form4_direction(cik10, accession, primary_doc)

        rows.append({
            "form": form,
            "filed_date": filed.date().isoformat(),
            "category": category,
            "insider_direction": insider_direction,
            "accession_number": accession,
            "primary_document": primary_doc,
        })

    return pd.DataFrame(rows)


def scan_shortlist_for_filings(tickers: list[str], pause_sec: float = 0.15) -> pd.DataFrame:
    """
    Runs get_recent_filings() for each ticker in the shortlist and returns
    a combined DataFrame with a 'symbol' column added, one row per
    catalyst-relevant filing found. Tickers with no CIK match or no
    recent relevant filings simply contribute no rows.
    """
    cik_map = load_ticker_cik_map()
    all_rows = []

    for i, ticker in enumerate(tickers, 1):
        cik10 = cik_map.get(ticker.upper())
        if cik10 is None:
            log.debug(f"No CIK found for {ticker}, skipping.")
            continue

        filings = get_recent_filings(cik10)
        if not filings.empty:
            filings.insert(0, "symbol", ticker)
            all_rows.append(filings)

        if i % 20 == 0:
            log.info(f"SEC filing scan progress: {i}/{len(tickers)} tickers checked...")
        time.sleep(pause_sec)  # stay well under SEC's fair-access rate expectations

    if not all_rows:
        return pd.DataFrame(columns=["symbol", "form", "filed_date", "category",
                                      "accession_number", "primary_document"])

    result = pd.concat(all_rows, ignore_index=True)
    log.info(f"SEC filing scan complete: {len(result)} catalyst-relevant filings "
              f"found across {result['symbol'].nunique()} tickers.")
    return result


if __name__ == "__main__":
    # quick smoke test against a couple of well-known tickers
    test_tickers = ["AAPL", "TSLA", "GME"]
    df = scan_shortlist_for_filings(test_tickers)
    print(df)
