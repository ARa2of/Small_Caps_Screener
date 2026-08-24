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

# 8-K Item significance tiers for scoring
# High-impact items get a catalyst boost; routine items get base score only
ITEM_HIGH_IMPACT = {
    "1.01": ("entry_agreement", 3),       # Entry into Material Definitive Agreement
    "1.02": ("termination_agreement", 1), # Termination of Material Definitive Agreement
    "1.03": ("bankruptcy", -4),            # Bankruptcy or Receivership
    "2.01": ("acquisition", 4),            # Completion of Acquisition or Disposition of Assets
    "2.03": ("direct_obligation", -1),     # Creation of Direct Financial Obligation
    "2.04": ("price_trigger", 2),          # Triggering Events That Increase/Decrease Price
    "2.06": ("material_impairment", -2),   # Material Impairments
    "3.01": ("delisting", -6),             # Delisting Notice
    "3.02": ("unregistered_equity", -3),   # Unregistered Sales of Equity Securities
    "3.03": ("rights_modification", -1),   # Material Modification to Rights
    "5.01": ("control_change", 2),         # Changes in Control
    "5.02": ("officer_departure", -1),     # Departure of Directors or Officers
}

# Keywords in 8-K titles that signal significance
KEYWORD_SIGNALS = {
    "merger": 3, "acquisition": 3, "acquire": 3, "terminated": -2,
    "termination": -2, "offering": -3, "registered": -3, "dilutive": -3,
    "bankruptcy": -4, "receivership": -4, "delisting": -6,
    "material definitive agreement": 2, "definitive agreement": 2,
    "convertible": -2, "warrant": -1, "purchase agreement": 1,
    "underwriting": -2, "shelf": -2, "atm": -3,
}


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


def _parse_8k_items_and_title(cik10: str, accession_number: str, primary_document: str) -> tuple[list[str], str, int]:
    """
    Fetches an 8-K filing and extracts:
      1. List of item numbers (e.g. ["1.01", "2.01"])
      2. Filing title/description (first meaningful text)
      3. Bonus score from keyword signals in the title
    Returns ([], "", 0) on failure.
    """
    import re

    cik_int = str(int(cik10))
    acc_nodash = accession_number.replace("-", "")
    # The 8-K primary document is usually the .htm file; try it directly
    url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{primary_document}"

    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        text = resp.text
    except Exception as e:
        log.debug(f"8-K fetch failed for {url}: {e}")
        return [], "", 0

    # Extract item numbers: patterns like "Item 1.01", "Item 2.01", etc.
    items = re.findall(r"Item\s+(\d+\.\d{2})", text, re.IGNORECASE)
    items = list(dict.fromkeys(items))  # dedupe, preserve order

    # Extract title from the filing header or first significant text
    title = ""
    # Try to find the "Description" or title near the top
    title_match = re.search(
        r"(?:Description|Subject|Title)[:\s]*([^\n<]{10,200})",
        text[:5000], re.IGNORECASE
    )
    if title_match:
        title = title_match.group(1).strip()
    else:
        # Try to find text between Item 1.01 and its body
        first_item_match = re.search(r"Item\s+1\.0[1-3]\s*[-–.]?\s*(.{10,200})", text, re.IGNORECASE)
        if first_item_match:
            title = first_item_match.group(1).strip()
            # Clean HTML tags
            title = re.sub(r"<[^>]+>", "", title).strip()[:200]

    # Score keywords in the title
    keyword_score = 0
    title_lower = title.lower()
    for keyword, pts in KEYWORD_SIGNALS.items():
        if keyword in title_lower:
            keyword_score += pts

    return items, title, keyword_score


def _8k_item_score(items: list[str], title: str) -> tuple[int, str]:
    """
    Given the list of8-K items and title, returns:
      (bonus_score, item_summary)
    The bonus_score is ADDED to the base material_event score (2 pts).
    """
    if not items:
        return 0, "no items parsed"

    total_bonus = 0
    summaries = []

    for item_num in items:
        if item_num in ITEM_HIGH_IMPACT:
            tag, bonus = ITEM_HIGH_IMPACT[item_num]
            total_bonus += bonus
            summaries.append(f"Item {item_num} ({tag}, {bonus:+d})")

    if not summaries:
        return 0, f"Items: {', '.join(items)} (routine)"

    return total_bonus, " | ".join(summaries)


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
      A = direct purchase on open market    -> insider_buy
      S = open-market or private sale      -> insider_sell
      anything else (M/G/F/etc.)           -> insider_other
    """
    import re
    import xml.etree.ElementTree as ET

    cik_int = str(int(cik10))
    acc_nodash = accession_number.replace("-", "")
    dir_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/"

    xml_url = None
    try:
        resp = requests.get(dir_url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        # Find XML files in the directory listing
        xml_files = re.findall(r'href="([^"]*\.xml)"', resp.text)
        # Prefer the raw XML (wk-form4*.xml or accession*.xml), not the XSL-rendered version
        for f in xml_files:
            fname = f.split("/")[-1]
            if fname.startswith("wk-form4") or fname.endswith(f"{acc_nodash}.xml"):
                xml_url = f"https://www.sec.gov{f}" if f.startswith("/") else f"{dir_url}{f}"
                break
        if not xml_url and xml_files:
            xml_url = f"https://www.sec.gov{xml_files[0]}" if xml_files[0].startswith("/") else f"{dir_url}{xml_files[0]}"
    except Exception as e:
        log.debug(f"Form 4 directory listing failed for {dir_url}: {e}")

    if not xml_url:
        log.debug(f"No XML file found in directory for {accession_number}")
        return "insider_unknown"

    try:
        resp = requests.get(xml_url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
    except ET.ParseError as e:
        log.debug(f"Form 4 XML parse failed for {xml_url}: {e}")
        return "insider_unknown"
    except Exception as e:
        log.debug(f"Form 4 fetch failed for {xml_url}: {e}")
        return "insider_unknown"

    codes = set()
    for tx in root.iter():
        tag = tx.tag.split("}")[-1] if "}" in tx.tag else tx.tag
        if tag == "transactionCode" and tx.text:
            codes.add(tx.text.strip().upper())

    if "P" in codes or "A" in codes:
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

        # For 8-K filings, parse items and title for scoring nuance
        item_bonus = 0
        item_summary = ""
        item_title = ""
        if category == "material_event" and accession and primary_doc:
            items, item_title, keyword_bonus = _parse_8k_items_and_title(cik10, accession, primary_doc)
            item_bonus, item_summary = _8k_item_score(items, item_title)
            item_bonus += keyword_bonus  # add keyword signal on top of item score

        rows.append({
            "form": form,
            "filed_date": filed.date().isoformat(),
            "category": category,
            "insider_direction": insider_direction,
            "item_bonus": item_bonus,
            "item_summary": item_summary,
            "item_title": item_title[:200] if item_title else "",
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
                                      "insider_direction", "item_bonus", "item_summary",
                                      "item_title", "accession_number", "primary_document"])

    result = pd.concat(all_rows, ignore_index=True)
    log.info(f"SEC filing scan complete: {len(result)} catalyst-relevant filings "
              f"found across {result['symbol'].nunique()} tickers.")
    return result


if __name__ == "__main__":
    # quick smoke test against a couple of well-known tickers
    test_tickers = ["AAPL", "TSLA", "GME"]
    df = scan_shortlist_for_filings(test_tickers)
    print(df)
