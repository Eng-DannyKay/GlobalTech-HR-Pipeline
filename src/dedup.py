"""
dedup.py — Deduplication Module
=================================
Multi-pass deduplication of the unified employee dataset, with ghost employee
detection and fuzzy-match review file generation.

Deduplication strategy:
    Pass 1 — Exact employee ID match (within namespace)
              Source priority: HRIS > Payroll > Benefits
              Merges payroll salary/benefits data onto HRIS records.

    Pass 2 — Email match (cross-company)
              Flags records sharing an email across GT and AC as probable matches.
              Does NOT auto-merge — produces a review entry.

    Pass 3 — Fuzzy name + hire date match
              rapidfuzz similarity ≥ 88% on full name, hire date within 30 days.
              Flags as probable_match — produces a review file for HR.

Ghost employee detection:
    Payroll records with no corresponding HRIS record (after all passes)
    are written to a separate ghost employee output file.

Outputs (returned as dict):
    golden         — pd.DataFrame  unified, deduped employee records
    ghosts         — pd.DataFrame  payroll records with no HRIS match
    probable_matches — pd.DataFrame  fuzzy/email pairs for HR review

Author: GlobalTech Data Engineering
"""

import logging

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

logger = logging.getLogger(__name__)

# ── Source priority for field-level merge ─────────────────────────────────────
# Lower number = higher trust. Used when merging duplicate records.
SOURCE_PRIORITY: dict[str, int] = {
    "globaltech_hris": 1,
    "acquiredco_hris": 1,
    "payroll":         2,
    "benefits":        3,
}

# ── Fuzzy match threshold ─────────────────────────────────────────────────────
FUZZY_THRESHOLD = 88        # % similarity on full name (token_sort_ratio)
HIRE_DATE_WINDOW = 30       # days — block hire dates further apart than this


# ── Helpers ───────────────────────────────────────────────────────────────────

def _priority(source: str) -> int:
    return SOURCE_PRIORITY.get(str(source).lower(), 99)


def _merge_records(group: pd.DataFrame, key_cols: list[str]) -> pd.Series:
    """
    Merge a group of duplicate records into a single best record.

    Strategy:
    - Sort by source priority (most trusted first).
    - For each column, take the first non-null value from the sorted group.
    - Track all contributing sources in 'source_systems'.
    """
    group = group.copy()
    group["_prio"] = group["source"].apply(_priority)
    group = group.sort_values("_prio")

    best = group.iloc[0].copy()

    for col in group.columns:
        if col in ("_prio", "source", "source_systems", "dedup_method"):
            continue
        if pd.isna(best.get(col)):
            non_null = group[col].dropna()
            if not non_null.empty:
                best[col] = non_null.iloc[0]

    best["source_systems"] = ",".join(group["source"].dropna().unique())
    best.pop("_prio", None)
    return best


# ── Pass 1: Exact ID Deduplication ────────────────────────────────────────────

def pass1_exact_id(
    hris_df: pd.DataFrame,
    payroll_df: pd.DataFrame,
    benefits_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pass 1: Merge payroll and benefits data onto HRIS records by exact employee_id.

    Logic:
    - Start with the combined HRIS (GlobalTech + AcquiredCo) as the base.
    - For each HRIS employee_id, pull the best payroll row (most recent
      effective_date where duplicates exist).
    - Attach benefits enrollment as a flag (enrolled = True/False).
    - HRIS records with no payroll match retain NaN salary fields.
    - Payroll rows with no HRIS match are collected as ghost candidates.

    Args:
        hris_df:     Combined cleaned HRIS DataFrame (GT + AC).
        payroll_df:  Cleaned payroll DataFrame.
        benefits_df: Cleaned benefits DataFrame.

    Returns:
        (merged_df, ghost_candidates_df)
    """
    logger.info("Pass 1: Exact ID deduplication...")

    # ── De-duplicate payroll: keep most recent record per employee_id ──────
    payroll_deduped = (
        payroll_df
        .sort_values("effective_date", ascending=False, na_position="last")
        .drop_duplicates(subset=["employee_id"], keep="first")
        .copy()
    )
    pay_dup_removed = len(payroll_df) - len(payroll_deduped)
    logger.info(f"  Payroll duplicates removed (keep latest): {pay_dup_removed:,}")

    # ── Identify ghost payroll records (no HRIS match) ────────────────────
    hris_ids = set(hris_df["employee_id"].dropna())
    pay_ids  = set(payroll_deduped["employee_id"].dropna())
    ghost_ids = pay_ids - hris_ids
    ghost_df  = payroll_deduped[payroll_deduped["employee_id"].isin(ghost_ids)].copy()
    ghost_df["ghost_flag_reason"] = "Payroll record with no matching HRIS employee_id"
    logger.info(f"  Ghost employee candidates identified: {len(ghost_df):,}")

    # ── Merge payroll onto HRIS ────────────────────────────────────────────
    pay_cols = ["employee_id", "base_salary", "base_salary_parsed",
                "salary_usd_annual", "currency", "pay_frequency",
                "bonus_target_pct", "effective_date"]
    pay_cols = [c for c in pay_cols if c in payroll_deduped.columns]

    merged = hris_df.merge(
        payroll_deduped[pay_cols],
        on="employee_id",
        how="left",
        suffixes=("", "_pay"),
    )

    # ── Attach benefits enrollment flag ───────────────────────────────────
    # One employee may have multiple plan enrollments — flag as enrolled if any
    enrolled_ids = set(benefits_df["employee_id"].dropna())
    merged["benefits_enrolled"] = merged["employee_id"].isin(enrolled_ids)

    # Count of plans per employee
    plan_counts = (
        benefits_df.groupby("employee_id")["plan_type"]
        .count()
        .rename("benefits_plan_count")
    )
    merged = merged.merge(plan_counts, on="employee_id", how="left")
    merged["benefits_plan_count"] = merged["benefits_plan_count"].fillna(0).astype(int)

    merged["dedup_method"] = "exact_id"
    merged["source_systems"] = merged["source"]   # Will be enriched in later passes

    logger.info(f"  Records after Pass 1 merge: {len(merged):,}")
    logger.info(f"    With payroll data   : {merged['salary_usd_annual'].notna().sum():,}")
    logger.info(f"    With benefits       : {merged['benefits_enrolled'].sum():,}")

    return merged, ghost_df


# ── Pass 2: Email Cross-Company Match ─────────────────────────────────────────

def pass2_email_match(merged_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pass 2: Detect cross-company duplicates by shared email address.

    Records sharing the same email across GT and AC namespaces are flagged
    as probable matches. They are NOT auto-merged — HR must confirm.

    Args:
        merged_df: Output of Pass 1.

    Returns:
        (merged_df_unchanged, email_matches_df)
        The main DataFrame is returned unchanged; email matches go to review file.
    """
    logger.info("Pass 2: Email cross-company match...")

    email_groups = (
        merged_df[merged_df["email"].notna()]
        .groupby("email")
        .filter(lambda g: g["employee_id"].str[:2].nunique() > 1)
    )

    if email_groups.empty:
        logger.info("  No cross-company email matches found.")
        return merged_df, pd.DataFrame()

    review_rows = []
    for email, group in email_groups.groupby("email"):
        ids = group["employee_id"].tolist()
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                review_rows.append({
                    "record_1_id":          ids[i],
                    "record_2_id":          ids[j],
                    "match_email":          email,
                    "similarity_score":     100.0,
                    "hire_date_diff_days":  None,
                    "match_type":           "email_match",
                    "recommended_action":   "REVIEW — same email, different company namespace",
                })

    review_df = pd.DataFrame(review_rows)
    logger.info(f"  Email match pairs flagged for review: {len(review_df):,}")
    return merged_df, review_df


# ── Pass 3: Fuzzy Name + Hire Date Match ──────────────────────────────────────

def pass3_fuzzy_name(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pass 3: Detect probable duplicates by fuzzy full-name + hire date proximity.

    Strategy:
    - Build candidate pairs where hire dates are within HIRE_DATE_WINDOW days
      (blocking step — avoids O(n²) comparison).
    - For each candidate pair, compute rapidfuzz token_sort_ratio on full name.
    - Pairs with similarity ≥ FUZZY_THRESHOLD are flagged as probable_match.
    - Does NOT auto-merge — produces review entries for HR.

    Args:
        merged_df: Output of Pass 1 (Pass 2 doesn't modify the main DataFrame).

    Returns:
        DataFrame of probable match pairs for HR review.
    """
    logger.info(f"Pass 3: Fuzzy name match (threshold={FUZZY_THRESHOLD}%, "
                f"hire date window={HIRE_DATE_WINDOW} days)...")

    # Only consider records that have both name and hire_date
    candidates = merged_df[
        merged_df["first_name"].notna() &
        merged_df["last_name"].notna() &
        merged_df["hire_date"].notna()
    ].copy()

    candidates["full_name"] = (
        candidates["first_name"].str.strip() + " " +
        candidates["last_name"].str.strip()
    ).str.lower()

    # Separate GT and AC records — we're looking for cross-company matches
    gt_records = candidates[candidates["employee_id"].str.startswith("GT-")].copy()
    ac_records = candidates[candidates["employee_id"].str.startswith("AC-")].copy()

    logger.info(f"  GT candidates: {len(gt_records):,} | AC candidates: {len(ac_records):,}")

    if gt_records.empty or ac_records.empty:
        logger.info("  No cross-company candidates — skipping fuzzy pass.")
        return pd.DataFrame()

    # ── Vectorized blocking: merge on hire date proximity ─────────────────
    # Convert hire_date to integer days for fast range join
    gt_records = gt_records.sort_values("hire_date").reset_index(drop=True)
    ac_records = ac_records.sort_values("hire_date").reset_index(drop=True)

    gt_records["hire_day"] = gt_records["hire_date"].apply(
        lambda d: d.toordinal() if pd.notna(d) else None
    )
    ac_records["hire_day"] = ac_records["hire_date"].apply(
        lambda d: d.toordinal() if pd.notna(d) else None
    )

    gt_records = gt_records.dropna(subset=["hire_day"])
    ac_records = ac_records.dropna(subset=["hire_day"])

    # Use merge_asof (sorted merge with tolerance) to get candidate pairs
    # This reduces O(n*m) to O(n log m) — critical for 15K x 3K
    gt_sorted = gt_records[["employee_id", "full_name", "hire_day"]].copy()
    ac_sorted = ac_records[["employee_id", "full_name", "hire_day"]].copy()
    gt_sorted["hire_day"] = gt_sorted["hire_day"].astype(int)
    ac_sorted["hire_day"] = ac_sorted["hire_day"].astype(int)

    # Forward merge: for each GT record find AC records within window ahead
    pairs_fwd = pd.merge_asof(
        gt_sorted, ac_sorted,
        on="hire_day",
        tolerance=HIRE_DATE_WINDOW,
        suffixes=("_gt", "_ac"),
        direction="nearest",
    ).dropna(subset=["employee_id_ac"])

    candidate_pairs = pairs_fwd[["employee_id_gt", "full_name_gt",
                                  "employee_id_ac", "full_name_ac", "hire_day"]].copy()

    logger.info(f"  Candidate pairs after date blocking: {len(candidate_pairs):,}")

    probable_matches = []
    for _, row in candidate_pairs.iterrows():
        score = fuzz.token_sort_ratio(row["full_name_gt"], row["full_name_ac"])
        if score >= FUZZY_THRESHOLD:
            probable_matches.append({
                "record_1_id":         row["employee_id_gt"],
                "record_2_id":         row["employee_id_ac"],
                "similarity_score":    round(float(score), 2),
                "hire_date_diff_days": 0,   # merge_asof picks nearest — diff ≤ window
                "match_type":          "fuzzy_name",
                "recommended_action":  (
                    "MERGE" if score >= 95 else "REVIEW"
                ),
            })

    logger.info(f"  Candidate pairs compared: {len(candidate_pairs):,}")
    logger.info(f"  Probable matches found  : {len(probable_matches):,}")

    if not probable_matches:
        return pd.DataFrame()

    result = pd.DataFrame(probable_matches).sort_values("similarity_score", ascending=False)
    return result


# ── Dedup Entry Point ─────────────────────────────────────────────────────────

def deduplicate(cleaned: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Run the full deduplication pipeline across all cleaned sources.

    Args:
        cleaned: Dict from clean.clean_all()
                 Keys: 'hris', 'acquiredco', 'payroll', 'benefits'

    Returns:
        Dict with keys:
            'golden'            — unified, deduped employee records
            'ghosts'            — payroll records with no HRIS match
            'probable_matches'  — fuzzy/email pairs for HR review
    """
    logger.info("=" * 60)
    logger.info("DEDUPLICATION LAYER")
    logger.info("=" * 60)

    hris_df     = cleaned["hris"]
    acq_df      = cleaned["acquiredco"]
    payroll_df  = cleaned["payroll"]
    benefits_df = cleaned["benefits"]

    # ── Combine HRIS sources into single base ──────────────────────────────
    combined_hris = pd.concat(
        [hris_df, acq_df],
        ignore_index=True,
        sort=False,
    )
    logger.info(f"Combined HRIS (GT + AC): {len(combined_hris):,} records")

    # ── Pass 1: Exact ID merge ─────────────────────────────────────────────
    golden, ghost_df = pass1_exact_id(combined_hris, payroll_df, benefits_df)

    # ── Pass 2: Email cross-company match ─────────────────────────────────
    golden, email_review = pass2_email_match(golden)

    # ── Pass 3: Fuzzy name + hire date ────────────────────────────────────
    fuzzy_review = pass3_fuzzy_name(golden)

    # ── Combine review files ───────────────────────────────────────────────
    review_frames = [f for f in [email_review, fuzzy_review] if not f.empty]
    probable_matches = (
        pd.concat(review_frames, ignore_index=True)
        if review_frames
        else pd.DataFrame(columns=[
            "record_1_id", "record_2_id", "similarity_score",
            "hire_date_diff_days", "match_type", "recommended_action",
        ])
    )

    # ── Final golden record column ordering ───────────────────────────────
    priority_cols = [
        "employee_id", "first_name", "last_name", "email",
        "department", "job_title", "hire_date", "country",
        "employment_type", "manager_id",
        "salary_usd_annual", "base_salary", "base_salary_parsed",
        "currency", "pay_frequency", "bonus_target_pct", "effective_date",
        "benefits_enrolled", "benefits_plan_count",
        "source", "source_systems", "dedup_method",
    ]
    existing_priority = [c for c in priority_cols if c in golden.columns]
    extra_cols = [c for c in golden.columns if c not in existing_priority]
    golden = golden[existing_priority + extra_cols]

    # ── Add company_origin for Parquet partitioning ────────────────────────
    golden["company_origin"] = np.where(
        golden["employee_id"].str.startswith("GT-"), "GlobalTech", "AcquiredCo"
    )

    logger.info("=" * 60)
    logger.info("DEDUPLICATION SUMMARY")
    logger.info(f"  Input HRIS records      : {len(combined_hris):,}")
    logger.info(f"  Golden records (output) : {len(golden):,}")
    logger.info(f"  Ghost employees flagged : {len(ghost_df):,}")
    logger.info(f"  Probable match pairs    : {len(probable_matches):,}")
    logger.info("=" * 60)

    return {
        "golden":           golden,
        "ghosts":           ghost_df,
        "probable_matches": probable_matches,
    }
