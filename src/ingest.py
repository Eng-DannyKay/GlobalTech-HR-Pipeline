import json
import logging
from pathlib import Path

import pandas as pd
from lxml import etree

logger = logging.getLogger(__name__)

STANDARD_SCHEMA = [
    "employee_id",
    "first_name",
    "last_name",
    "email",
    "department",
    "job_title",
    "hire_date",
    "country",
    "employment_type",
    "manager_id",
    "source",
]

_dead_letters: list[dict] = []


def get_dead_letters() -> pd.DataFrame:
    return pd.DataFrame(_dead_letters)


def _log_dead_letter(source: str, record: dict, reason: str) -> None:
    _dead_letters.append({"source": source, "reason": reason, "record": str(record)})
    logger.warning(f"  [DEAD-LETTER] {source}: {reason}")



def ingest_globaltech_hris(filepath: Path) -> pd.DataFrame:
    logger.info(f"Ingesting GlobalTech HRIS CSV: {filepath}")

    if not filepath.exists():
        logger.error(f"  File not found: {filepath}")
        return pd.DataFrame(columns=STANDARD_SCHEMA)

    try:
        df = pd.read_csv(
            filepath,
            encoding="utf-8",
            dtype=str,                          # Read everything as string first
            na_values=["", "N/A", "NULL", "null", "None", "NaN", "#N/A"],
        )
    except Exception as exc:
        logger.error(f"  Failed to read {filepath}: {exc}")
        return pd.DataFrame(columns=STANDARD_SCHEMA)

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^\w]", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )

    df["source"] = "globaltech_hris"

    # Dead-letter: rows with no employee_id
    bad_mask = df["employee_id"].isna()
    if bad_mask.any():
        for _, row in df[bad_mask].iterrows():
            _log_dead_letter("globaltech_hris", row.to_dict(), "Missing employee_id")
        df = df[~bad_mask].copy()

    logger.info(f"  Ingested {len(df):,} records from GlobalTech HRIS")
    return df



def _parse_acquiredco_record(raw: dict) -> dict | None:
    try:
        return {
            "employee_id":     raw.get("employee_identifier"),
            "first_name":      raw.get("name", {}).get("first"),
            "last_name":       raw.get("name", {}).get("last"),
            "email":           raw.get("contact", {}).get("email"),
            "department":      raw.get("assignment", {}).get("department"),
            "job_title":       raw.get("assignment", {}).get("role"),
            "hire_date":       raw.get("assignment", {}).get("hire_timestamp"),
            "country":         raw.get("assignment", {}).get("location"),
            "employment_type": raw.get("employment", {}).get("type"),
            "manager_id":      raw.get("manager_employee_id"),
            "source":          "acquiredco_hris",
        }
    except Exception as exc:
        logger.warning(f"  Could not parse AcquiredCo record: {exc} | raw={raw}")
        return None


def ingest_acquiredco_json(filepath: Path, page_size: int = 500) -> pd.DataFrame:
    logger.info(f"Ingesting AcquiredCo JSON (paginated simulation): {filepath}")

    if not filepath.exists():
        logger.error(f"  File not found: {filepath}")
        return pd.DataFrame(columns=STANDARD_SCHEMA)

    try:
        raw_data = json.loads(filepath.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.error(f"  Failed to parse {filepath}: {exc}")
        return pd.DataFrame(columns=STANDARD_SCHEMA)

    all_employees: list[dict] = raw_data.get("employees", [])
    total = raw_data.get("total_records", len(all_employees))
    total_pages = (total + page_size - 1) // page_size

    logger.info(f"  Total records reported: {total:,} | Simulating {total_pages} pages @ {page_size}/page")

    records = []
    for page in range(1, total_pages + 1):
        start = (page - 1) * page_size
        end = start + page_size
        batch = all_employees[start:end]
        logger.info(f"  Page {page}/{total_pages}: {len(batch)} records")

        for raw_rec in batch:
            parsed = _parse_acquiredco_record(raw_rec)
            if parsed is None or not parsed.get("employee_id"):
                _log_dead_letter("acquiredco_hris", raw_rec, "Missing employee_identifier or parse failure")
                continue
            records.append(parsed)

    df = pd.DataFrame(records)
    logger.info(f"  Ingested {len(df):,} records from AcquiredCo HRIS")
    return df


# ── Source 3: Combined Payroll Excel ──────────────────────────────────────────

def ingest_payroll_excel(filepath: Path) -> pd.DataFrame:
    logger.info(f"Ingesting Payroll Excel: {filepath}")

    if not filepath.exists():
        logger.error(f"  File not found: {filepath}")
        return pd.DataFrame()

    try:
        df = pd.read_excel(
            filepath,
            dtype=str,                          # Read everything as string first
            na_values=["", "N/A", "NULL", "null", "None", "NaN", "#N/A"],
        )
    except Exception as exc:
        logger.error(f"  Failed to read {filepath}: {exc}")
        return pd.DataFrame()

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^\w]", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )

    df["ingest_source"] = "payroll"

    bad_mask = df["employee_id"].isna()
    if bad_mask.any():
        for _, row in df[bad_mask].iterrows():
            _log_dead_letter("payroll", row.to_dict(), "Missing employee_id")
        df = df[~bad_mask].copy()

    logger.info(f"  Ingested {len(df):,} records from Payroll Excel")
    return df



def ingest_benefits_xml(filepath: Path) -> pd.DataFrame:
    logger.info(f"Ingesting Benefits XML: {filepath}")

    if not filepath.exists():
        logger.error(f"  File not found: {filepath}")
        return pd.DataFrame()

    try:
        tree = etree.parse(str(filepath))
        root = tree.getroot()
    except etree.XMLSyntaxError as exc:
        logger.error(f"  Malformed XML in {filepath}: {exc}")
        return pd.DataFrame()

    FIELDS = [
        "employee_id", "plan_type", "coverage_level",
        "enrollment_date", "premium_employee", "premium_employer",
    ]

    records = []
    for enrollment in root.findall("enrollment"):
        record: dict = {}
        try:
            for field in FIELDS:
                node = enrollment.find(field)
                record[field] = node.text.strip() if node is not None and node.text else None

            if not record.get("employee_id"):
                _log_dead_letter("benefits", record, "Missing employee_id in XML node")
                continue

            record["source"] = "benefits"
            records.append(record)

        except Exception as exc:
            _log_dead_letter("benefits", record, f"Parse error: {exc}")
            continue

    df = pd.DataFrame(records)
    logger.info(f"  Ingested {len(df):,} records from Benefits XML")
    return df


def align_schema(df: pd.DataFrame) -> pd.DataFrame:
    for col in STANDARD_SCHEMA:
        if col not in df.columns:
            df[col] = None
    return df


def ingest_all_sources(raw_dir: Path) -> dict[str, pd.DataFrame]:
    logger.info("=" * 60)
    logger.info("INGESTION LAYER — Loading all 4 source systems")
    logger.info("=" * 60)

    hris_df      = ingest_globaltech_hris(raw_dir / "globaltech_hris.csv")
    acquiredco_df = ingest_acquiredco_json(raw_dir / "acquiredco_api.json")
    payroll_df   = ingest_payroll_excel(raw_dir / "payroll_data.xlsx")
    benefits_df  = ingest_benefits_xml(raw_dir / "benefits_enrollment.xml")

    # Align HRIS and AcquiredCo to standard schema
    hris_df       = align_schema(hris_df)
    acquiredco_df = align_schema(acquiredco_df)

    logger.info("=" * 60)
    logger.info("INGESTION SUMMARY")
    logger.info(f"  GlobalTech HRIS  : {len(hris_df):>6,} records")
    logger.info(f"  AcquiredCo HRIS  : {len(acquiredco_df):>6,} records")
    logger.info(f"  Payroll          : {len(payroll_df):>6,} records")
    logger.info(f"  Benefits         : {len(benefits_df):>6,} records")
    logger.info(f"  Dead-letter total: {len(_dead_letters):>6,} records")
    logger.info("=" * 60)

    return {
        "hris":       hris_df,
        "acquiredco": acquiredco_df,
        "payroll":    payroll_df,
        "benefits":   benefits_df,
    }
