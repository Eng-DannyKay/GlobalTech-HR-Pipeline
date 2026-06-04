"""
clean.py — Data Cleaning & Transformation Module
==================================================
Applies all cleaning and standardization transformations to the ingested
DataFrames before deduplication.

Transformations applied:
    1. Employee ID namespacing       GT-XXXXXX / AC-XXXXXX
    2. Name normalization            Unicode, title case, whitespace
    3. Employment type expansion     PT/FT/CONTRACTOR → standard labels
    4. Department taxonomy mapping   Codes & names → standard taxonomy
    5. Date standardization          All formats → datetime64[ns]
    6. Currency normalization        EUR/GBP → USD; pay frequency → annual
    7. Payroll employee ID mapping   Namespace by source column

Author: GlobalTech Data Engineering
"""

import logging
import unicodedata
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Exchange rates (fixed; update for production runs) ────────────────────────
# Source: approximate mid-market rates as of 2026-06-04
EXCHANGE_RATES_TO_USD: dict[str, float] = {
    "USD": 1.0,
    "EUR": 1.08,
    "GBP": 1.27,
}

# ── Pay frequency multipliers → annual ────────────────────────────────────────
PAY_FREQUENCY_MULTIPLIER: dict[str, float] = {
    "Annual":    1.0,
    "Monthly":   12.0,
    "Bi-Weekly": 26.0,
}

# ── Employment type normalisation ─────────────────────────────────────────────
# AcquiredCo uses abbreviations; GlobalTech uses full labels
EMPLOYMENT_TYPE_MAP: dict[str, str] = {
    # AcquiredCo abbreviations
    "ft":         "Full-Time",
    "pt":         "Part-Time",
    "contractor": "Contractor",
    "ct":         "Contractor",
    # GlobalTech full labels (already correct — included for safety)
    "full-time":  "Full-Time",
    "part-time":  "Part-Time",
}

# ── Department taxonomy ───────────────────────────────────────────────────────
# Both GlobalTech and AcquiredCo happen to use the same department names in this
# dataset. The mapping table is retained for forward-compatibility and to handle
# any stray codes (e.g. ENG-01) that may appear in edge records.
DEPARTMENT_MAP: dict[str, str] = {
    # Standard names (pass-through)
    "engineering":          "Engineering",
    "marketing":            "Marketing",
    "sales":                "Sales",
    "finance":              "Finance",
    "human resources":      "Human Resources",
    "hr":                   "Human Resources",
    "legal":                "Legal",
    "operations":           "Operations",
    "product":              "Product",
    "data science":         "Data Science",
    "devops":               "DevOps",
    "it":                   "Information Technology",
    "information technology": "Information Technology",
    "customer success":     "Customer Success",
    "quality assurance":    "Quality Assurance",
    "qa":                   "Quality Assurance",
    "supply chain":         "Supply Chain",
    "manufacturing":        "Manufacturing",
    "strategy":             "Strategy",
    "communications":       "Communications",
    "business development": "Business Development",
    # Legacy GlobalTech department codes
    "eng-01": "Engineering",
    "eng-02": "Engineering",
    "mkt-01": "Marketing",
    "mkt-02": "Marketing",
    "mkt-03": "Marketing",
    "sal-01": "Sales",
    "sal-02": "Sales",
    "fin-01": "Finance",
    "fin-02": "Finance",
    "hr-01":  "Human Resources",
    "leg-01": "Legal",
    "ops-01": "Operations",
    "ops-02": "Operations",
    "prd-01": "Product",
    "prd-02": "Product",
    "ds-01":  "Data Science",
    "dvo-01": "DevOps",
    "it-01":  "Information Technology",
    "it-02":  "Information Technology",
    "cs-01":  "Customer Success",
    "qa-01":  "Quality Assurance",
    "sc-01":  "Supply Chain",
    "mfg-01": "Manufacturing",
    "str-01": "Strategy",
    "com-01": "Communications",
    "bd-01":  "Business Development",
}

# ── 1. Employee ID Namespacing ────────────────────────────────────────────────

def namespace_globaltech_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert plain integer GlobalTech employee IDs to GT-XXXXXX format.

    Before: '1042'  →  After: 'GT-001042'
    manager_id follows the same rule.

    Args:
        df: GlobalTech HRIS DataFrame (source == 'globaltech_hris').

    Returns:
        DataFrame with namespaced employee_id and manager_id.
    """
    def _format_gt(val: str | None) -> str | None:
        if pd.isna(val) or str(val).strip() == "":
            return None
        try:
            return f"GT-{int(float(str(val).strip())):06d}"
        except (ValueError, OverflowError):
            return val  # Return as-is if not parseable

    df = df.copy()
    df["employee_id"] = df["employee_id"].apply(_format_gt)
    df["manager_id"]  = df["manager_id"].apply(_format_gt)
    logger.info("  [IDs] GlobalTech IDs namespaced to GT-XXXXXX")
    return df


def namespace_acquiredco_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert AcquiredCo IDs from ACQ_XXXXX format to AC-XXXXXX format.

    Before: 'ACQ_00001'  →  After: 'AC-000001'
    manager_id follows the same rule.

    Args:
        df: AcquiredCo HRIS DataFrame (source == 'acquiredco_hris').

    Returns:
        DataFrame with namespaced employee_id and manager_id.
    """
    def _format_ac(val: str | None) -> str | None:
        if pd.isna(val) or str(val).strip() == "":
            return None
        val = str(val).strip()
        # Strip ACQ_ or ACQ prefix, then zero-pad to 6 digits
        numeric = val.replace("ACQ_", "").replace("ACQ", "").lstrip("0") or "0"
        try:
            return f"AC-{int(numeric):06d}"
        except ValueError:
            return val  # Return as-is if not parseable

    df = df.copy()
    df["employee_id"] = df["employee_id"].apply(_format_ac)
    df["manager_id"]  = df["manager_id"].apply(_format_ac)
    logger.info("  [IDs] AcquiredCo IDs namespaced to AC-XXXXXX")
    return df


def namespace_payroll_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Namespace payroll employee IDs using the 'source' column.

    Payroll 'source' column values: 'GlobalTech' | 'AcquiredCo'
    Plain integer IDs become GT-XXXXXX or AC-XXXXXX accordingly.

    Args:
        df: Payroll DataFrame with 'employee_id' and 'source' columns.

    Returns:
        DataFrame with namespaced employee_id.
    """
    df = df.copy()

    def _apply_namespace(row: pd.Series) -> str | None:
        val = row["employee_id"]
        src = str(row.get("source", "")).strip().lower()
        if pd.isna(val):
            return None
        raw = str(val).strip()
        # Strip known prefixes (ACQ_, ACQ, plain integer)
        numeric_str = raw.replace("ACQ_", "").replace("ACQ", "").lstrip("0") or "0"
        try:
            num = int(float(numeric_str))
        except (ValueError, TypeError):
            # Fall back to plain integer parse of original value
            try:
                num = int(float(raw))
            except (ValueError, TypeError):
                return raw  # Return as-is if completely unparseable
        if src == "globaltech":
            return f"GT-{num:06d}"
        elif src == "acquiredco":
            return f"AC-{num:06d}"
        return raw

    df["employee_id"] = df.apply(_apply_namespace, axis=1)
    logger.info("  [IDs] Payroll IDs namespaced using source column")
    return df


def namespace_benefits_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Namespace benefits employee IDs to GT-XXXXXX.
    Benefits cover GlobalTech employees only.

    Args:
        df: Benefits DataFrame with plain integer employee_id.

    Returns:
        DataFrame with namespaced employee_id.
    """
    df = df.copy()

    def _format_gt(val: str | None) -> str | None:
        if pd.isna(val) or str(val).strip() == "":
            return None
        try:
            return f"GT-{int(float(str(val).strip())):06d}"
        except (ValueError, OverflowError):
            return val

    df["employee_id"] = df["employee_id"].apply(_format_gt)
    logger.info("  [IDs] Benefits IDs namespaced to GT-XXXXXX")
    return df


# ── 2. Name Normalisation ─────────────────────────────────────────────────────

def _normalize_unicode(text: str) -> str:
    """
    Normalize Unicode to NFC form, preserving accented characters.
    NFC ensures composed form (é as single codepoint, not e + combining accent).
    """
    return unicodedata.normalize("NFC", text)


def _title_case_name(text: str) -> str:
    """
    Apply title case while correctly handling:
    - Hyphenated names:   'van-der-berg' → 'Van-Der-Berg'
    - Apostrophe names:   "o'brien"      → "O'Brien"
    - Multi-word names:   'van der berg' → 'Van Der Berg'
    """
    # Split on spaces, capitalize each word; then handle hyphens within words
    words = text.split(" ")
    result = []
    for word in words:
        # Handle hyphenated parts
        parts = word.split("-")
        result.append("-".join(p.capitalize() for p in parts))
    cased = " ".join(result)
    # Handle apostrophes: O'Brien → O'Brien (capitalize after apostrophe)
    if "'" in cased:
        segments = cased.split("'")
        cased = "'".join(s.capitalize() for s in segments)
    return cased


def normalize_names(series: pd.Series) -> pd.Series:
    """
    Full name normalization pipeline:
    1. Strip whitespace
    2. Collapse internal whitespace
    3. Unicode NFC normalization
    4. Title case (handles hyphens, apostrophes, multi-word)

    Args:
        series: Raw first_name or last_name Series.

    Returns:
        Cleaned Series.
    """
    def _clean(val):
        if pd.isna(val):
            return np.nan
        val = str(val).strip()
        val = " ".join(val.split())       # Collapse multiple spaces
        val = _normalize_unicode(val)
        val = _title_case_name(val)
        return val if val else np.nan

    return series.apply(_clean)


# ── 3. Employment Type Normalization ──────────────────────────────────────────

def normalize_employment_type(series: pd.Series) -> pd.Series:
    """
    Standardize employment type to: Full-Time | Part-Time | Contractor

    AcquiredCo abbreviations:
        FT → Full-Time
        PT → Part-Time
        CONTRACTOR → Contractor

    Args:
        series: Raw employment_type Series.

    Returns:
        Normalized Series. Unmapped values become NaN.
    """
    def _map(val):
        if pd.isna(val):
            return np.nan
        key = str(val).strip().lower()
        mapped = EMPLOYMENT_TYPE_MAP.get(key)
        if mapped is None:
            logger.warning(f"  [employment_type] Unmapped value: '{val}'")
        return mapped

    return series.apply(_map)


# ── 4. Department Taxonomy Mapping ────────────────────────────────────────────

def normalize_departments(series: pd.Series, log_unmapped: bool = True) -> pd.Series:
    """
    Map raw department codes and names to the standard taxonomy.

    Logs any unmapped values for manual review.

    Args:
        series: Raw department Series.
        log_unmapped: Whether to emit warnings for unmapped values.

    Returns:
        Standardized Series. Unmapped values are returned as-is (not NaN)
        so they can be reviewed rather than silently lost.
    """
    unmapped: set[str] = set()

    def _map(val):
        if pd.isna(val):
            return np.nan
        key = str(val).strip().lower()
        mapped = DEPARTMENT_MAP.get(key)
        if mapped is None:
            unmapped.add(str(val))
            return str(val)  # Retain original for review
        return mapped

    result = series.apply(_map)

    if log_unmapped and unmapped:
        logger.warning(f"  [department] {len(unmapped)} unmapped values (retained as-is): {sorted(unmapped)}")
    else:
        logger.info("  [department] All department values mapped successfully")

    return result


# ── 5. Date Standardization ───────────────────────────────────────────────────

DATE_FORMATS = [
    "%Y-%m-%d",           # GlobalTech HRIS: 2022-03-15
    "%Y-%m-%dT%H:%M:%S",  # AcquiredCo JSON: 2024-06-27T00:00:00
    "%m/%d/%Y",           # Alternative: 06/27/2024
    "%d-%b-%Y",           # Benefits (if present): 15-Jan-2022
    "%Y-%m-%d %H:%M:%S",  # Timestamp variant
]

_TODAY = pd.Timestamp(datetime.now().date())
_MIN_DATE = pd.Timestamp("1970-01-01")


def parse_dates(series: pd.Series, column_name: str = "date") -> pd.Series:
    """
    Parse dates from any of the known source formats into datetime64[ns].

    Tries each format in DATE_FORMATS in order. Falls back to pandas
    inference if none match. Dates outside [1970-01-01, today] are flagged
    with a warning but retained — range validation is done in validate.py.

    Args:
        series: Raw date string Series.
        column_name: Name used in log messages.

    Returns:
        datetime64[ns] Series. Unparseable values become NaT.
    """
    def _parse_single(val):
        if pd.isna(val):
            return pd.NaT
        val = str(val).strip()
        for fmt in DATE_FORMATS:
            try:
                ts = pd.Timestamp(datetime.strptime(val, fmt))
                # Always return tz-naive
                return ts.tz_localize(None) if ts.tzinfo is not None else ts
            except (ValueError, OverflowError):
                continue
        # Last resort: let pandas infer
        try:
            ts = pd.Timestamp(val)
            return ts.tz_localize(None) if ts.tzinfo is not None else ts
        except Exception:
            return pd.NaT

    parsed = series.apply(_parse_single)

    unparseable = parsed.isna().sum() - series.isna().sum()
    if unparseable > 0:
        logger.warning(f"  [date:{column_name}] {unparseable} values could not be parsed → NaT")

    # Strip timezone info before range comparison (belt-and-suspenders)
    parsed_naive = parsed.dt.tz_localize(None) if (hasattr(parsed, "dt") and getattr(parsed.dt, "tz", None) is not None) else parsed
    out_of_range = ((parsed_naive < _MIN_DATE) | (parsed_naive > _TODAY)).sum()
    if out_of_range > 0:
        logger.warning(f"  [date:{column_name}] {out_of_range} dates outside plausible range (pre-1970 or future)")

    return parsed


# ── 6. Currency & Salary Normalization ────────────────────────────────────────

def _parse_salary_string(val) -> float | None:
    """
    Parse salary strings to float.

    Handles:
        '77935'      → 77935.0
        '$85,000'    → 85000.0
        '£72,000.50' → 72000.5
        85000.0      → 85000.0
        None / NaN   → None
    """
    if pd.isna(val):
        return None
    val = str(val).strip()
    # Remove currency symbols and thousands separators
    val = val.replace("$", "").replace("£", "").replace("€", "").replace(",", "")
    try:
        return float(val)
    except ValueError:
        return None


def normalize_payroll(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and normalize payroll salary data.

    Steps:
        1. Parse base_salary strings → float
        2. Normalize pay frequency → annual equivalent
        3. Convert currency → USD using EXCHANGE_RATES_TO_USD
        4. Add salary_usd_annual column; retain originals

    Args:
        df: Payroll DataFrame with columns:
            base_salary, currency, pay_frequency

    Returns:
        DataFrame with additional columns:
            base_salary_parsed   — float, original currency
            salary_usd_annual    — float, USD annual equivalent
    """
    df = df.copy()

    # Step 1: Parse salary strings
    df["base_salary_parsed"] = df["base_salary"].apply(_parse_salary_string)

    unparseable = df["base_salary_parsed"].isna().sum() - df["base_salary"].isna().sum()
    if unparseable > 0:
        logger.warning(f"  [salary] {unparseable} salary values could not be parsed")

    # Step 2: Normalize pay frequency to annual
    df["pay_frequency"] = df["pay_frequency"].str.strip()
    df["frequency_multiplier"] = df["pay_frequency"].map(PAY_FREQUENCY_MULTIPLIER)

    unmapped_freq = df["frequency_multiplier"].isna().sum()
    if unmapped_freq > 0:
        unmapped_vals = df[df["frequency_multiplier"].isna()]["pay_frequency"].unique()
        logger.warning(f"  [pay_frequency] {unmapped_freq} unmapped values: {unmapped_vals}")
        df["frequency_multiplier"] = df["frequency_multiplier"].fillna(1.0)

    df["base_salary_annual"] = df["base_salary_parsed"] * df["frequency_multiplier"]

    # Step 3: Currency → USD
    df["currency"] = df["currency"].str.strip().str.upper()
    df["fx_rate"] = df["currency"].map(EXCHANGE_RATES_TO_USD)

    unmapped_curr = df["fx_rate"].isna().sum()
    if unmapped_curr > 0:
        unmapped_vals = df[df["fx_rate"].isna()]["currency"].unique()
        logger.warning(f"  [currency] {unmapped_curr} records with unknown currency: {unmapped_vals}")
        df["fx_rate"] = df["fx_rate"].fillna(1.0)

    df["salary_usd_annual"] = (df["base_salary_annual"] * df["fx_rate"]).round(2)

    # Drop working columns
    df = df.drop(columns=["frequency_multiplier", "base_salary_annual", "fx_rate"])

    logger.info(
        f"  [payroll] Salary normalization complete. "
        f"USD annual range: ${df['salary_usd_annual'].min():,.0f} – ${df['salary_usd_annual'].max():,.0f}"
    )
    return df


# ── 7. Benefits Date Normalization ────────────────────────────────────────────

def normalize_benefits(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean benefits enrollment dates and numeric premium columns.

    Args:
        df: Benefits DataFrame.

    Returns:
        Cleaned DataFrame with enrollment_date as datetime64[ns]
        and premium columns as float.
    """
    df = df.copy()
    df["enrollment_date"]    = parse_dates(df["enrollment_date"], "enrollment_date")
    df["premium_employee"]   = pd.to_numeric(df["premium_employee"], errors="coerce")
    df["premium_employer"]   = pd.to_numeric(df["premium_employer"], errors="coerce")
    return df


# ── Master Cleaning Functions ─────────────────────────────────────────────────

def clean_hris(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning transformations to the GlobalTech HRIS DataFrame.

    Args:
        df: Raw GlobalTech HRIS DataFrame from ingest.py.

    Returns:
        Cleaned DataFrame.
    """
    logger.info("Cleaning GlobalTech HRIS...")
    df = namespace_globaltech_ids(df)
    df["first_name"]      = normalize_names(df["first_name"])
    df["last_name"]       = normalize_names(df["last_name"])
    df["employment_type"] = normalize_employment_type(df["employment_type"])
    df["department"]      = normalize_departments(df["department"])
    df["hire_date"]       = parse_dates(df["hire_date"], "hire_date")
    logger.info(f"  GlobalTech HRIS clean complete: {len(df):,} records")
    return df


def clean_acquiredco(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning transformations to the AcquiredCo HRIS DataFrame.

    Args:
        df: Raw AcquiredCo HRIS DataFrame from ingest.py.

    Returns:
        Cleaned DataFrame.
    """
    logger.info("Cleaning AcquiredCo HRIS...")
    df = namespace_acquiredco_ids(df)
    df["first_name"]      = normalize_names(df["first_name"])
    df["last_name"]       = normalize_names(df["last_name"])
    df["employment_type"] = normalize_employment_type(df["employment_type"])
    df["department"]      = normalize_departments(df["department"])
    df["hire_date"]       = parse_dates(df["hire_date"], "hire_date")
    logger.info(f"  AcquiredCo HRIS clean complete: {len(df):,} records")
    return df


def clean_payroll(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning transformations to the Payroll DataFrame.

    Args:
        df: Raw Payroll DataFrame from ingest.py.

    Returns:
        Cleaned DataFrame with salary_usd_annual added.
    """
    logger.info("Cleaning Payroll data...")
    df = namespace_payroll_ids(df)
    df["effective_date"] = parse_dates(df["effective_date"], "effective_date")
    df = normalize_payroll(df)
    logger.info(f"  Payroll clean complete: {len(df):,} records")
    return df


def clean_benefits(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning transformations to the Benefits DataFrame.

    Args:
        df: Raw Benefits DataFrame from ingest.py.

    Returns:
        Cleaned DataFrame.
    """
    logger.info("Cleaning Benefits data...")
    df = namespace_benefits_ids(df)
    df = normalize_benefits(df)
    logger.info(f"  Benefits clean complete: {len(df):,} records")
    return df


def clean_all(sources: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Run all cleaning transformations across all 4 source DataFrames.

    Args:
        sources: Dict from ingest.ingest_all_sources()
                 Keys: 'hris', 'acquiredco', 'payroll', 'benefits'

    Returns:
        Dict with same keys, each value fully cleaned.
    """
    logger.info("=" * 60)
    logger.info("CLEANING LAYER — Transforming all 4 sources")
    logger.info("=" * 60)

    cleaned = {
        "hris":       clean_hris(sources["hris"]),
        "acquiredco": clean_acquiredco(sources["acquiredco"]),
        "payroll":    clean_payroll(sources["payroll"]),
        "benefits":   clean_benefits(sources["benefits"]),
    }

    logger.info("=" * 60)
    logger.info("CLEANING SUMMARY")
    for name, df in cleaned.items():
        logger.info(f"  {name:<12}: {len(df):>6,} records")
    logger.info("=" * 60)

    return cleaned
