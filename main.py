"""
Pipeline orchestrator: runs Stage 1 (universe + price/technical filter) and
Stage 2 (SEC EDGAR filing catalysts) against the same ticker list, then
exports both to one Excel workbook.

Run this file directly -- it's the single entry point for the whole
pipeline so far. Individual modules (universe.py, screener.py,
sec_edgar.py) can still be run standalone for testing/debugging one
piece at a time, but they don't talk to each other unless you run this.
"""

import logging
import os
import sys

# Some IDE run/debug configurations execute this file with a working
# directory or sys.path that doesn't include this script's own folder,
# which breaks the sibling imports below (universe, screener, sec_edgar,
# excel_export) even though those files are sitting right next to this
# one. Explicitly adding this file's directory to sys.path fixes that
# regardless of how the IDE invokes the run.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from universe import get_universe
from screener import bulk_download, build_shortlist
from sec_edgar import scan_shortlist_for_filings
from scoring import build_scored_table
from fair_value import compute_fair_value
from volatility_range import compute_all_volatility_ranges
from excel_export import export_to_excel
from report_generator import export_ai_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def run_pipeline():
    log.info("=== Stage 1: building universe ===")
    universe_df = get_universe()
    tickers = universe_df["symbol"].tolist()
    log.info(f"Universe size after country/price/cap/volume pre-filters: {len(tickers)}")

    log.info("=== Stage 1: bulk price/volume download + technical filter ===")
    price_data = bulk_download(tickers)
    shortlist_df = build_shortlist(price_data)

    if shortlist_df.empty:
        log.warning("Stage 1 shortlist is empty -- nothing to scan in Stage 2. "
                    "Exporting an empty workbook so the run still produces a file.")
        export_to_excel(shortlist_df, filings_df=None)
        return

    log.info(f"=== Stage 2: SEC EDGAR filing scan on {len(shortlist_df)} shortlisted tickers ===")
    shortlist_tickers = shortlist_df["symbol"].tolist()
    filings_df = scan_shortlist_for_filings(shortlist_tickers)

    log.info("=== Stage 3: fair value reference data ===")
    shortlist_df = compute_fair_value(shortlist_df)

    log.info("=== Stage 3: volatility-based 3-month price range ===")
    shortlist_df = compute_all_volatility_ranges(shortlist_df, price_data)

    log.info("=== Stage 3: composite scoring + recommendation tiers (with catalyst timing) ===")
    scored_df = build_scored_table(shortlist_df, filings_df, price_data=price_data)

    log.info("=== Generating AI analysis prompt ===")
    prompt_path = export_ai_prompt(scored_df)
    log.info(f"AI prompt generated: {prompt_path}")

    log.info("=== Exporting to Excel ===")
    out_path = export_to_excel(shortlist_df, filings_df, scored_df)
    log.info(f"Pipeline complete. Output: {out_path}")


if __name__ == "__main__":
    run_pipeline()
