import logging
import sys
from datetime import datetime
from pathlib import Path
from ingest import ingest_all_sources, get_dead_letters
from clean import clean_all
from dedup import deduplicate
from validate import validate
from visualize import visualize

BASE_DIR    = Path(__file__).parent
DATA_RAW    = BASE_DIR / "data" / "raw"
DATA_OUT    = BASE_DIR / "data" / "processed"
CHARTS_DIR  = DATA_OUT / "charts"
LOGS_DIR    = BASE_DIR / "logs"
SRC_DIR     = BASE_DIR / "src"

DATA_OUT.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SRC_DIR))

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
    logger.info(f"Run started : {start.isoformat()}")
    logger.info(f"Raw data    : {DATA_RAW}")
    logger.info(f"Output      : {DATA_OUT}")
    logger.info("=" * 60)

    
    sources = ingest_all_sources(DATA_RAW)

    dead_letters = get_dead_letters()
    if not dead_letters.empty:
        dl_path = DATA_OUT / "dead_letters.csv"
        dead_letters.to_csv(dl_path, index=False)
        logger.warning(f"  Dead-letter records saved: {dl_path}")

    
    cleaned = clean_all(sources)

  
    dedup_result = deduplicate(cleaned)

    golden   = dedup_result["golden"]
    ghosts   = dedup_result["ghosts"]
    probable = dedup_result["probable_matches"]

    
    report = validate(golden, output_dir=DATA_OUT)

   
    charts = visualize(golden, report, output_dir=CHARTS_DIR)

    logger.info("=" * 60)
    logger.info("EXPORT LAYER")
    logger.info("=" * 60)

    parquet_path = DATA_OUT / "golden_hr_dataset.parquet"
    golden.to_parquet(
        parquet_path,
        index=False,
        partition_cols=["company_origin"],
        engine="pyarrow",
    )
    logger.info(f"  Golden Parquet  -> {parquet_path}  ({len(golden):,} records)")

    if not ghosts.empty:
        ghost_path = DATA_OUT / "ghost_employees.csv"
        ghost_out = ghosts.copy()
        ghost_out.insert(0, "payroll_employee_id", ghost_out.pop("employee_id") if "employee_id" in ghost_out.columns else "")
        ghost_out["name"] = "N/A - No matching HRIS record"
        ghost_out.to_csv(ghost_path, index=False)
        logger.info(f"  Ghost CSV       -> {ghost_path}  ({len(ghosts):,} records)")

    if not probable.empty:
        prob_path = DATA_OUT / "probable_matches.csv"
        probable.to_csv(prob_path, index=False)
        logger.info(f"  Probable matches-> {prob_path}  ({len(probable):,} pairs)")

    duration = (datetime.now() - start).total_seconds()
    n_pass = (report["status"] == "PASS").sum()
    n_warn = (report["status"] == "WARN").sum()
    n_fail = (report["status"] == "FAIL").sum()

    logger.info("=" * 60)
    logger.info("PIPELINE RUN COMPLETE")
    logger.info(f"  Golden records  : {len(golden):,}")
    logger.info(f"  Ghost employees : {len(ghosts):,}")
    logger.info(f"  Probable matches: {len(probable):,} pairs")
    logger.info(f"  Quality checks  : {n_pass} PASS | {n_warn} WARN | {n_fail} FAIL")
    logger.info(f"  Charts saved    : {len(charts)} / 6")
    logger.info(f"  Duration        : {duration:.1f}s")
    logger.info(f"  Log file        : {log_file}")
    logger.info("=" * 60)

    return {
        "golden":   golden,
        "ghosts":   ghosts,
        "probable": probable,
        "report":   report,
        "charts":   charts,
    }


if __name__ == "__main__":
    main()
