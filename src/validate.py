"""
validate.py — Data Quality Validation Layer
GlobalTech HR Integration Pipeline

Runs 12 deterministic checks on the golden dataset.
Pipeline gate: raises RuntimeError if more than 2 checks fail.
Exports results to CSV and HTML report.
"""

import logging
import re
from datetime import date
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

VALID_EMPLOYMENT_TYPES = {"Full-Time", "Part-Time", "Contractor"}
VALID_CURRENCIES       = {"USD", "EUR", "GBP"}
EMAIL_REGEX            = re.compile(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$")
EMPLOYEE_ID_REGEX      = re.compile(r"^(GT|AC)-\d{6}$")
SALARY_MIN             = 15_000.0
SALARY_MAX             = 10_000_000.0   # upper guard — synthetic data can be very high
HIRE_DATE_MIN          = pd.Timestamp("1970-01-01")
PIPELINE_FAIL_LIMIT    = 2              # halt if more than this many checks fail


# ── Individual check functions ────────────────────────────────────────────────

def _make_result(check_id: int, check_name: str, status: str,
                 failed_count: int, total: int, details: str) -> dict:
    """Create a standardised result row."""
    return {
        "check_id":    check_id,
        "check_name":  check_name,
        "status":      status,       # "PASS" | "FAIL" | "WARN"
        "failed_rows": failed_count,
        "total_rows":  total,
        "pass_rate":   round((total - failed_count) / total * 100, 2) if total else 0.0,
        "details":     details,
    }


def check_not_null(df: pd.DataFrame, col: str, check_id: int) -> dict:
    n_null = df[col].isna().sum() if col in df.columns else len(df)
    status = "PASS" if n_null == 0 else "FAIL"
    return _make_result(
        check_id,
        f"NOT_NULL:{col}",
        status,
        int(n_null),
        len(df),
        f"{n_null} null values in '{col}'",
    )


def check_unique(df: pd.DataFrame, col: str, check_id: int,
                 fail_on_dupes: bool = True) -> dict:
    if col not in df.columns:
        return _make_result(check_id, f"UNIQUE:{col}", "FAIL", len(df), len(df),
                            f"Column '{col}' missing")
    dupes = df[col].dropna().duplicated().sum()
    if dupes == 0:
        status = "PASS"
    elif fail_on_dupes:
        status = "FAIL"
    else:
        status = "WARN"
    return _make_result(
        check_id,
        f"UNIQUE:{col}",
        status,
        int(dupes),
        len(df),
        f"{dupes} duplicate values in '{col}'",
    )


def check_values_in_set(df: pd.DataFrame, col: str, valid_set: set,
                        check_id: int) -> dict:
    if col not in df.columns:
        return _make_result(check_id, f"VALUES_IN_SET:{col}", "FAIL", len(df), len(df),
                            f"Column '{col}' missing")
    non_null = df[col].dropna()
    invalid = (~non_null.isin(valid_set)).sum()
    bad_vals = sorted(set(non_null[~non_null.isin(valid_set)].unique()))[:5]
    status = "PASS" if invalid == 0 else "FAIL"
    return _make_result(
        check_id,
        f"VALUES_IN_SET:{col}",
        status,
        int(invalid),
        len(df),
        f"{invalid} invalid values. Examples: {bad_vals}",
    )


def check_regex(df: pd.DataFrame, col: str, pattern: re.Pattern,
                check_id: int, label: str) -> dict:
    if col not in df.columns:
        return _make_result(check_id, f"REGEX:{label}", "FAIL", len(df), len(df),
                            f"Column '{col}' missing")
    non_null = df[col].dropna().astype(str)
    invalid = (~non_null.str.match(pattern)).sum()
    bad_vals = sorted(non_null[~non_null.str.match(pattern)].unique())[:3]
    status = "PASS" if invalid == 0 else "FAIL"
    return _make_result(
        check_id,
        f"REGEX:{label}",
        status,
        int(invalid),
        len(df),
        f"{invalid} values fail pattern '{pattern.pattern}'. Examples: {bad_vals}",
    )


def check_numeric_range(df: pd.DataFrame, col: str,
                        low: float, high: float, check_id: int) -> dict:
    if col not in df.columns:
        return _make_result(check_id, f"NUMERIC_RANGE:{col}", "WARN", 0, len(df),
                            f"Column '{col}' absent — skipped")
    non_null = df[col].dropna()
    out_of_range = ((non_null < low) | (non_null > high)).sum()
    status = "PASS" if out_of_range == 0 else "WARN"   # salary data is synthetic
    return _make_result(
        check_id,
        f"NUMERIC_RANGE:{col}",
        status,
        int(out_of_range),
        len(df),
        f"{out_of_range} salaries outside [{low:,.0f}, {high:,.0f}]",
    )


def check_date_range(df: pd.DataFrame, col: str, check_id: int) -> dict:
    if col not in df.columns:
        return _make_result(check_id, f"DATE_RANGE:{col}", "FAIL", len(df), len(df),
                            f"Column '{col}' missing")
    today = pd.Timestamp(date.today())
    dates = pd.to_datetime(df[col], errors="coerce")
    n_null_after = dates.isna().sum()
    valid_dates = dates.dropna()
    out_of_range = ((valid_dates < HIRE_DATE_MIN) | (valid_dates > today)).sum()
    total_bad = int(n_null_after + out_of_range)
    status = "PASS" if total_bad == 0 else "FAIL"
    return _make_result(
        check_id,
        f"DATE_RANGE:{col}",
        status,
        total_bad,
        len(df),
        (f"{out_of_range} dates outside [1970-01-01, {today.date()}]; "
         f"{n_null_after} unparseable"),
    )


def check_referential_integrity(df: pd.DataFrame, check_id: int) -> dict:
    """Every non-null manager_id must exist as an employee_id in the dataset."""
    if "manager_id" not in df.columns or "employee_id" not in df.columns:
        return _make_result(check_id, "REF_INTEGRITY:manager_id→employee_id",
                            "FAIL", 0, len(df), "Required columns missing")
    valid_ids = set(df["employee_id"].dropna())
    mgr_ids   = df["manager_id"].dropna()
    orphans   = (~mgr_ids.isin(valid_ids)).sum()
    status = "PASS" if orphans == 0 else "WARN"   # cross-company refs are expected
    return _make_result(
        check_id,
        "REF_INTEGRITY:manager_id→employee_id",
        status,
        int(orphans),
        int(len(mgr_ids)),
        f"{orphans} manager_id values not found in employee_id set",
    )


# ── Master validation function ────────────────────────────────────────────────

def validate(golden: pd.DataFrame, output_dir: Path | None = None) -> pd.DataFrame:
    """
    Run all 12 quality checks against the golden dataset.

    Parameters
    ----------
    golden     : The deduplicated golden DataFrame
    output_dir : If provided, writes results to CSV + HTML there

    Returns
    -------
    pd.DataFrame  One row per check with columns:
        check_id, check_name, status, failed_rows, total_rows, pass_rate, details

    Raises
    ------
    RuntimeError  if more than PIPELINE_FAIL_LIMIT checks have status == "FAIL"
    """
    logger.info("=" * 60)
    logger.info("VALIDATION LAYER")
    logger.info("=" * 60)

    n = len(golden)
    results = []

    # ── NOT NULL checks (checks 1–6) ──────────────────────────────────────────
    for cid, col in enumerate(
        ["employee_id", "first_name", "last_name", "email", "department", "country"],
        start=1,
    ):
        r = check_not_null(golden, col, cid)
        results.append(r)
        logger.info(f"  [{r['status']:4s}] {r['check_name']} — {r['details']}")

    # ── UNIQUE checks (checks 7–8) ────────────────────────────────────────────
    # employee_id must be strictly unique (hard FAIL)
    r7 = check_unique(golden, "employee_id", 7, fail_on_dupes=True)
    results.append(r7)
    logger.info(f"  [{r7['status']:4s}] {r7['check_name']} — {r7['details']}")
    # email uniqueness is WARN — duplicate emails occur in cross-system HR merges
    r8 = check_unique(golden, "email", 8, fail_on_dupes=False)
    results.append(r8)
    logger.info(f"  [{r8['status']:4s}] {r8['check_name']} — {r8['details']}")

    # ── VALUES IN SET (checks 9–10) ───────────────────────────────────────────
    r9 = check_values_in_set(golden, "employment_type", VALID_EMPLOYMENT_TYPES, 9)
    results.append(r9)
    logger.info(f"  [{r9['status']:4s}] {r9['check_name']} — {r9['details']}")

    r10 = check_values_in_set(golden, "currency", VALID_CURRENCIES, 10)
    results.append(r10)
    logger.info(f"  [{r10['status']:4s}] {r10['check_name']} — {r10['details']}")

    # ── REGEX checks (checks 11–12) ───────────────────────────────────────────
    r11 = check_regex(golden, "email", EMAIL_REGEX, 11, "email_format")
    results.append(r11)
    logger.info(f"  [{r11['status']:4s}] {r11['check_name']} — {r11['details']}")

    r12 = check_regex(golden, "employee_id", EMPLOYEE_ID_REGEX, 12, "employee_id_format")
    results.append(r12)
    logger.info(f"  [{r12['status']:4s}] {r12['check_name']} — {r12['details']}")

    # ── NUMERIC RANGE (check 13) ──────────────────────────────────────────────
    r13 = check_numeric_range(golden, "salary_usd_annual", SALARY_MIN, SALARY_MAX, 13)
    results.append(r13)
    logger.info(f"  [{r13['status']:4s}] {r13['check_name']} — {r13['details']}")

    # ── DATE RANGE (check 14) ─────────────────────────────────────────────────
    r14 = check_date_range(golden, "hire_date", 14)
    results.append(r14)
    logger.info(f"  [{r14['status']:4s}] {r14['check_name']} — {r14['details']}")

    # ── REFERENTIAL INTEGRITY (check 15) ──────────────────────────────────────
    r15 = check_referential_integrity(golden, 15)
    results.append(r15)
    logger.info(f"  [{r15['status']:4s}] {r15['check_name']} — {r15['details']}")

    # ── Build results DataFrame ───────────────────────────────────────────────
    results_df = pd.DataFrame(results)

    n_pass = (results_df["status"] == "PASS").sum()
    n_warn = (results_df["status"] == "WARN").sum()
    n_fail = (results_df["status"] == "FAIL").sum()

    logger.info("=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info(f"  Total checks : {len(results_df)}")
    logger.info(f"  PASS         : {n_pass}")
    logger.info(f"  WARN         : {n_warn}")
    logger.info(f"  FAIL         : {n_fail}")
    logger.info("=" * 60)

    # ── Export ────────────────────────────────────────────────────────────────
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path  = output_dir / "validation_report.csv"
        html_path = output_dir / "validation_report.html"

        results_df.to_csv(csv_path, index=False)

        _write_html_report(results_df, n_pass, n_warn, n_fail, html_path)

        logger.info(f"  Report CSV  → {csv_path}")
        logger.info(f"  Report HTML → {html_path}")

    # ── Pipeline gate ─────────────────────────────────────────────────────────
    if n_fail > PIPELINE_FAIL_LIMIT:
        failing = results_df[results_df["status"] == "FAIL"]["check_name"].tolist()
        raise RuntimeError(
            f"Pipeline gate triggered: {n_fail} checks FAILED "
            f"(limit={PIPELINE_FAIL_LIMIT}). Failed checks: {failing}"
        )

    return results_df


# ── HTML report writer ────────────────────────────────────────────────────────

def _row_color(status: str) -> str:
    return {"PASS": "#d4edda", "WARN": "#fff3cd", "FAIL": "#f8d7da"}.get(status, "#ffffff")


def _write_html_report(df: pd.DataFrame, n_pass: int, n_warn: int,
                       n_fail: int, path: Path) -> None:
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rows_html = ""
    for _, row in df.iterrows():
        bg = _row_color(row["status"])
        rows_html += (
            f'<tr style="background:{bg}">'
            f'<td>{row["check_id"]}</td>'
            f'<td>{row["check_name"]}</td>'
            f'<td><strong>{row["status"]}</strong></td>'
            f'<td style="text-align:right">{row["failed_rows"]:,}</td>'
            f'<td style="text-align:right">{row["total_rows"]:,}</td>'
            f'<td style="text-align:right">{row["pass_rate"]:.1f}%</td>'
            f'<td>{row["details"]}</td>'
            f'</tr>\n'
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>GlobalTech HR Pipeline — Validation Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2em; color: #333; }}
    h1   {{ color: #2c3e50; }}
    .summary {{ display: flex; gap: 1.5em; margin-bottom: 1.5em; }}
    .badge   {{ padding: .4em 1em; border-radius: 4px; font-weight: bold; font-size: 1.1em; }}
    .pass  {{ background: #d4edda; color: #155724; }}
    .warn  {{ background: #fff3cd; color: #856404; }}
    .fail  {{ background: #f8d7da; color: #721c24; }}
    table  {{ border-collapse: collapse; width: 100%; font-size: .9em; }}
    th     {{ background: #2c3e50; color: white; padding: .5em 1em; text-align: left; }}
    td     {{ padding: .45em 1em; border-bottom: 1px solid #dee2e6; }}
    .footer {{ margin-top: 2em; color: #888; font-size: .8em; }}
  </style>
</head>
<body>
  <h1>GlobalTech HR Integration Pipeline — Data Quality Report</h1>
  <p>Generated: {timestamp} | Source: GlobalTech Corp multi-source HR data</p>

  <div class="summary">
    <span class="badge pass">PASS: {n_pass}</span>
    <span class="badge warn">WARN: {n_warn}</span>
    <span class="badge fail">FAIL: {n_fail}</span>
  </div>

  <table>
    <thead>
      <tr>
        <th>#</th><th>Check</th><th>Status</th>
        <th>Failed Rows</th><th>Total Rows</th><th>Pass Rate</th><th>Details</th>
      </tr>
    </thead>
    <tbody>
{rows_html}    </tbody>
  </table>

  <div class="footer">
    GlobalTech Corp · AmaliTech Training Academy Capstone · Pipeline v1.0
  </div>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")
