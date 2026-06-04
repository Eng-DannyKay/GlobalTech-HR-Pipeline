"""
ingest.py — Multi-Source HR Data Ingestion Module
===================================================
Loads all 4 source systems into standardized Pandas DataFrames.

Sources:
    1. GlobalTech HRIS  — CSV (UTF-8), ~15,000 records
    2. AcquiredCo HRIS  — JSON (paginated API simulation), ~3,200 records
    3. Combined Payroll — Excel (.xlsx), ~18,500 records
    4. Benefits Provider— XML, ~12,000 records

Standard Employee Schema (output of align_schema):
    employee_id       : str   — Namespaced ID set by clean.py (GT-XXXXXX / AC-XXXXXX)
    first_name        : str
    last_name         : str
    email             : str
    department        : str   — Raw dept code or name; unified by clean.py
    job_title         : str
    hire_date         : str   — Raw string; parsed by clean.py
    country           : str
    employment_type   : str   — Raw value; normalized by clean.py
    manager_id        : str   — Raw manager ID; namespaced by clean.py
    source            : str   — Source system tag

Author: GlobalTech Data Engineering
Run via: pipeline.py
"""

import json
import logging
from pathlib import Path

import pandas as pd
from lxml import etree

# ── Logging ───────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Standard output schema for all sources ────────────────────────────────────
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

# ── Dead-letter store (malformed records accumulate here during ingestion) ─────
_dead_letters: list[dict] = []


def get_dead_letters() -> pd.DataFrame:
    """Return all dead-letter records collected during ingestion."""
    return pd.DataFrame(_dead_letters)


def _log_dead_letter(source: str, record: dict, reason: str) -> None:
    """Append a malformed record to the dead-letter store instead of crashing."""
    _dead_letters.append({"source": source, "reason": reason, "record": str(record)})
    logger.warning(f"  [DEAD-LETTER] {source}: {reason}")


# ── Source 1: GlobalTech HRIS CSV ─────────────────────────────────────────────

def ingest_globaltech_hris(filepath: Path) -> pd.DataFrame:
    """
    Ingest the GlobalTech Workday HRIS CSV export.

    Args:
        filepath: Path to globaltech_hris.csv (UTF-8 encoded).

    Returns:
        Raw DataFrame with source tag. Column names match the standard schema
        directly — no renaming needed.

    Schema (as exported):
        employee_id, first_name, last_name, email, department, job_title,
        hire_date, country, employment_type, manager_id

    Notes:
        - employee_id values are plain integers (e.g. 1042); namespacing to
          GT-XXXXXX is handled in clean.py.
        - department uses codes like ENG-01, MKT-03; mapped in clean.py.
        - hire_date format: YYYY-MM-DD.
    """
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

    # Standardize column names to snake_case
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


# ── Source 2: AcquiredCo HRIS JSON (simulated paginated API) ──────────────────

def _parse_acquiredco_record(raw: dict) -> dict | None:
    """
    Flatten one nested AcquiredCo JSON employee record into the standard schema.

    Raw structure:
        {
          "employee_identifier": "ACQ_00001",
          "name": {"first": "...", "last": "...", "full": "..."},
          "contact": {"email": "..."},
          "assignment": {
              "department": "...", "role": "...",
              "location": "...", "hire_timestamp": "2024-06-27T00:00:00"
          },
          "employment": {"type": "PT", "status": "Active"},
          "manager_employee_id": "ACQ_02436"
        }

    Returns:
        Flattened dict aligned to standard schema, or None if critically malformed.
    """
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
    """
    Ingest the AcquiredCo BambooHR JSON export, simulating paginated API access.

    In production this would call:
        GET /api/v1/employees?page=N&per_page=500
        Authorization: Bearer <token>

    Here we read from file and iterate in page_size chunks to mirror that pattern.

    Args:
        filepath:  Path to acquiredco_api.json.
        page_size: Simulated records-per-page (default 500, matching API spec).

    Returns:
        Flattened DataFrame with source tag.

    Notes:
        - employee_id values are prefixed ACQ_XXXXX; renamed to AC-XXXXXX in clean.py.
        - hire_date is an ISO-8601 timestamp; truncated to date in clean.py.
        - employment_type abbreviations (PT, FT, CT) expanded in clean.py.
        - department values are names (Engineering, Marketing); mapped in clean.py.
    """
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
    """
    Ingest the ADP combined payroll Excel export.

    Args:
        filepath: Path to payroll_data.xlsx.

    Returns:
        DataFrame with payroll columns and source tag.

    Schema (as exported):
        employee_id      — plain integer or string ID; source column indicates origin
        source           — 'globaltech' or 'acquiredco' (tells us which namespace)
        base_salary      — may be string like "$85,000" or float; cleaned in clean.py
        currency         — USD, EUR, GBP
        pay_frequency    — Annual, Monthly, Bi-Weekly
        bonus_target_pct — float percentage
        effective_date   — date of payroll record

    Notes:
        - This file covers both GlobalTech and AcquiredCo employees.
        - Some records are duplicated; removed in dedup.py.
        - Currency conversion to USD happens in clean.py.
        - Payroll-only records (no HRIS match) are ghost employees, flagged in dedup.py.
    """
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

    # Standardize column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^\w]", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )

    df["ingest_source"] = "payroll"

    # Dead-letter: rows with no employee_id
    bad_mask = df["employee_id"].isna()
    if bad_mask.any():
        for _, row in df[bad_mask].iterrows():
            _log_dead_letter("payroll", row.to_dict(), "Missing employee_id")
        df = df[~bad_mask].copy()

    logger.info(f"  Ingested {len(df):,} records from Payroll Excel")
    return df


# ── Source 4: Benefits Provider XML ───────────────────────────────────────────

def ingest_benefits_xml(filepath: Path) -> pd.DataFrame:
    """
    Ingest the MedShield benefits enrollment XML export.

    Args:
        filepath: Path to benefits_enrollment.xml.

    Returns:
        DataFrame with benefits columns and source tag.

    XML Structure:
        <benefits_enrollments>
          <enrollment>
            <employee_id>...</employee_id>
            <plan_type>...</plan_type>
            <coverage_level>...</coverage_level>
            <enrollment_date>...</enrollment_date>
            <premium_employee>...</premium_employee>
            <premium_employer>...</premium_employer>
          </enrollment>
          ...
        </benefits_enrollments>

    Notes:
        - Covers GlobalTech employees only; not all employees are enrolled.
        - enrollment_date format: DD-Mon-YYYY (e.g. 15-Jan-2022); parsed in clean.py.
        - employee_id here matches GlobalTech HRIS IDs (plain integers).
    """
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


# ── Schema Alignment ──────────────────────────────────────────────────────────

def align_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Align a source DataFrame to the STANDARD_SCHEMA.

    - Columns in STANDARD_SCHEMA but missing from df → added as NaN.
    - Columns not in STANDARD_SCHEMA → retained (payroll/benefits have extra cols).

    Args:
        df: Source DataFrame after ingestion.

    Returns:
        DataFrame with at minimum all STANDARD_SCHEMA columns present.
    """
    for col in STANDARD_SCHEMA:
        if col not in df.columns:
            df[col] = None
    return df


# ── Combined Ingestion Entry Point ────────────────────────────────────────────

def ingest_all_sources(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """
    Ingest all 4 source systems and return a dict of DataFrames.

    Args:
        raw_dir: Directory containing all 4 raw source files.

    Returns:
        Dict with keys: 'hris', 'acquiredco', 'payroll', 'benefits'
        Each value is the ingested (not yet cleaned) DataFrame.
    """
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
