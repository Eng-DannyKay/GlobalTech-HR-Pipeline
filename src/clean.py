import logging
import unicodedata
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


EXCHANGE_RATES_TO_USD: dict[str, float] = {
    "USD": 1.0,
    "EUR": 1.08,
    "GBP": 1.27,
}


PAY_FREQUENCY_MULTIPLIER: dict[str, float] = {
    "Annual":    1.0,
    "Monthly":   12.0,
    "Bi-Weekly": 26.0,
}

EMPLOYMENT_TYPE_MAP: dict[str, str] = {
    "ft":         "Full-Time",
    "pt":         "Part-Time",
    "contractor": "Contractor",
    "ct":         "Contractor",
    "full-time":  "Full-Time",
    "part-time":  "Part-Time",
}


DEPARTMENT_MAP: dict[str, str] = {
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

def namespace_globaltech_ids(df: pd.DataFrame) -> pd.DataFrame:
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
    def _format_ac(val: str | None) -> str | None:
        if pd.isna(val) or str(val).strip() == "":
            return None
        val = str(val).strip().upper()
        # Intentional duplicate records — flag for removal downstream
        if "DUP" in val:
            return "__DROP__"
        # Strip known prefixes, then zero-pad to 6 digits
        for prefix in ("ACQ_", "ACQ", "AC-"):
            if val.startswith(prefix):
                val = val[len(prefix):]
                break
        try:
            return f"AC-{int(val):06d}"
        except ValueError:
            return None  # malformed — treat as missing

    df = df.copy()
    df["employee_id"] = df["employee_id"].apply(_format_ac)
    df["manager_id"]  = df["manager_id"].apply(_format_ac)

    # Drop intentional duplicate records (ACQ_DUP_* IDs)
    n_dup = (df["employee_id"] == "__DROP__").sum()
    if n_dup:
        df = df[df["employee_id"] != "__DROP__"].reset_index(drop=True)
        logger.info(f"  [dedup] Removed {n_dup} intentional duplicate source records (ACQ_DUP_*)")

    logger.info("  [IDs] AcquiredCo IDs namespaced to AC-XXXXXX")
    return df


def namespace_payroll_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def _apply_namespace(row: pd.Series) -> str | None:
        val = row["employee_id"]
        src = str(row.get("source", "")).strip().lower()
        if pd.isna(val):
            return None
        raw = str(val).strip()
        numeric_str = raw.replace("ACQ_", "").replace("ACQ", "").lstrip("0") or "0"
        try:
            num = int(float(numeric_str))
        except (ValueError, TypeError):
            try:
                num = int(float(raw))
            except (ValueError, TypeError):
                return raw
        if src == "globaltech":
            return f"GT-{num:06d}"
        elif src == "acquiredco":
            return f"AC-{num:06d}"
        return raw

    df["employee_id"] = df.apply(_apply_namespace, axis=1)
    logger.info("  [IDs] Payroll IDs namespaced using source column")
    return df


def namespace_benefits_ids(df: pd.DataFrame) -> pd.DataFrame:
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


def _normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _title_case_name(text: str) -> str:
    # Split on spaces, capitalize each word; then handle hyphens within words
    words = text.split(" ")
    result = []
    for word in words:
        # Handle hyphenated parts
        parts = word.split("-")
        result.append("-".join(p.capitalize() for p in parts))
    cased = " ".join(result)
    if "'" in cased:
        segments = cased.split("'")
        cased = "'".join(s.capitalize() for s in segments)
    return cased


def normalize_names(series: pd.Series) -> pd.Series:
    def _clean(val):
        if pd.isna(val):
            return np.nan
        val = str(val).strip()
        val = " ".join(val.split())
        val = _normalize_unicode(val)
        val = _title_case_name(val)
        return val if val else np.nan

    return series.apply(_clean)


def normalize_employment_type(series: pd.Series) -> pd.Series:
    def _map(val):
        if pd.isna(val):
            return np.nan
        key = str(val).strip().lower()
        mapped = EMPLOYMENT_TYPE_MAP.get(key)
        if mapped is None:
            logger.warning(f"  [employment_type] Unmapped value: '{val}'")
        return mapped

    return series.apply(_map)


def normalize_departments(series: pd.Series, log_unmapped: bool = True) -> pd.Series:
    unmapped: set[str] = set()

    def _map(val):
        if pd.isna(val):
            return np.nan
        key = str(val).strip().lower()
        mapped = DEPARTMENT_MAP.get(key)
        if mapped is None:
            unmapped.add(str(val))
            return str(val)
        return mapped

    result = series.apply(_map)

    if log_unmapped and unmapped:
        logger.warning(f"  [department] {len(unmapped)} unmapped values (retained as-is): {sorted(unmapped)}")
    else:
        logger.info("  [department] All department values mapped successfully")

    return result


DATE_FORMATS = [
    "%Y-%m-%d",
    "%Y-%m-%dT%H:%M:%S",
    "%m/%d/%Y",
    "%d-%b-%Y",
    "%Y-%m-%d %H:%M:%S",
]

_TODAY = pd.Timestamp(datetime.now().date())
_MIN_DATE = pd.Timestamp("1970-01-01")


def parse_dates(series: pd.Series, column_name: str = "date") -> pd.Series:
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
    df = df.copy()

    df["base_salary_parsed"] = df["base_salary"].apply(_parse_salary_string)

    unparseable = df["base_salary_parsed"].isna().sum() - df["base_salary"].isna().sum()
    if unparseable > 0:
        logger.warning(f"  [salary] {unparseable} salary values could not be parsed")

    df["pay_frequency"] = df["pay_frequency"].str.strip()
    df["frequency_multiplier"] = df["pay_frequency"].map(PAY_FREQUENCY_MULTIPLIER)

    unmapped_freq = df["frequency_multiplier"].isna().sum()
    if unmapped_freq > 0:
        unmapped_vals = df[df["frequency_multiplier"].isna()]["pay_frequency"].unique()
        logger.warning(f"  [pay_frequency] {unmapped_freq} unmapped values: {unmapped_vals}")
        df["frequency_multiplier"] = df["frequency_multiplier"].fillna(1.0)

    df["base_salary_annual"] = df["base_salary_parsed"] * df["frequency_multiplier"]

    df["currency"] = df["currency"].str.strip().str.upper()
    df["fx_rate"] = df["currency"].map(EXCHANGE_RATES_TO_USD)

    unmapped_curr = df["fx_rate"].isna().sum()
    if unmapped_curr > 0:
        unmapped_vals = df[df["fx_rate"].isna()]["currency"].unique()
        logger.warning(f"  [currency] {unmapped_curr} records with unknown currency: {unmapped_vals}")
        df["fx_rate"] = df["fx_rate"].fillna(1.0)

    df["salary_usd_annual"] = (df["base_salary_annual"] * df["fx_rate"]).round(2)

    df = df.drop(columns=["frequency_multiplier", "base_salary_annual", "fx_rate"])

    logger.info(
        f"  [payroll] Salary normalization complete. "
        f"USD annual range: ${df['salary_usd_annual'].min():,.0f} – ${df['salary_usd_annual'].max():,.0f}"
    )
    return df


def normalize_benefits(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["enrollment_date"]    = parse_dates(df["enrollment_date"], "enrollment_date")
    df["premium_employee"]   = pd.to_numeric(df["premium_employee"], errors="coerce")
    df["premium_employer"]   = pd.to_numeric(df["premium_employer"], errors="coerce")
    return df


def clean_hris(df: pd.DataFrame) -> pd.DataFrame:
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
    logger.info("Cleaning Payroll data...")
    df = namespace_payroll_ids(df)
    df["effective_date"] = parse_dates(df["effective_date"], "effective_date")
    df = normalize_payroll(df)
    logger.info(f"  Payroll clean complete: {len(df):,} records")
    return df


def clean_benefits(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Cleaning Benefits data...")
    df = namespace_benefits_ids(df)
    df = normalize_benefits(df)
    logger.info(f"  Benefits clean complete: {len(df):,} records")
    return df


def clean_all(sources: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
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
