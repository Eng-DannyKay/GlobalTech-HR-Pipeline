"""
pipeline.py — GlobalTech HR Integration Pipeline Orchestrator
==============================================================
Main entry point. Run this file to execute the full pipeline.

Usage:
    python pipeline.py

Completed steps are run; incomplete steps are skipped with a notice.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DATA_RAW    = BASE_DIR / "data" / "raw"
DATA_OUT    = BASE_DIR / "data" / "processed"
LOGS_DIR    = BASE_DIR / "logs"
SRC_DIR     = BASE_DIR / "src"

DATA_OUT.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SRC_DIR))

# ── Logging ───────────────────────────────────────────────────────────────────
log_file = LOGS_DIR / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def main():
    start = datetime.now()
    logger.info("=" * 60)
    logger.info("GLOBALTECH HR INTEGRATION PIPELINE")
    logger.info(f"Run started: {start.isoformat()}")
    logger.info("=" * 60)

    # ── Step 1: Ingestion ──────────────────────────────────────────────────
    from ingest import ingest_all_sources, get_dead_letters
    sources = ingest_all_sources(DATA_RAW)

    dead_letters = get_dead_letters()
    if not dead_letters.empty:
        dl_path = DATA_OUT / "dead_letters.csv"
        dead_letters.to_csv(dl_path, index=False)
        logger.warning(f"Dead-letter records saved: {dl_path}")

    # ── Step 2: Cleaning ───────────────────────────────────────────────────
    from clean import clean_all
    cleaned = clean_all(sources)

    # ── Step 3: Deduplication ─────────────────────────────────────────────
    logger.info("Step 3 (dedup.py) — not yet built, skipping.")
    golden = None

    # ── Step 4: Validation ────────────────────────────────────────────────
    logger.info("Step 4 (validate.py) — not yet built, skipping.")

    # ── Step 5: Visualization ─────────────────────────────────────────────
    logger.info("Step 5 (visualize.py) — not yet built, skipping.")

    # ── Step 6: Export ────────────────────────────────────────────────────
    logger.info("Step 6 (export) — waiting for dedup output, skipping.")

    # ── Summary ───────────────────────────────────────────────────────────
    duration = (datetime.now() - start).total_seconds()
    logger.info("=" * 60)
    logger.info("PIPELINE RUN COMPLETE (partial — steps 3-6 pending)")
    logger.info(f"  GlobalTech HRIS : {len(cleaned['hris']):>6,} records cleaned")
    logger.info(f"  AcquiredCo HRIS : {len(cleaned['acquiredco']):>6,} records cleaned")
    logger.info(f"  Payroll         : {len(cleaned['payroll']):>6,} records cleaned")
    logger.info(f"  Benefits        : {len(cleaned['benefits']):>6,} records cleaned")
    logger.info(f"  Duration        : {duration:.1f}s")
    logger.info(f"  Log file        : {log_file}")
    logger.info("=" * 60)

    return cleaned


if __name__ == "__main__":
    main()
