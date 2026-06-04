import logging

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

logger = logging.getLogger(__name__)


SOURCE_PRIORITY: dict[str, int] = {
    "globaltech_hris": 1,
    "acquiredco_hris": 1,
    "payroll":         2,
    "benefits":        3,
}

FUZZY_THRESHOLD = 88
HIRE_DATE_WINDOW = 30


def _priority(source: str) -> int:
    return SOURCE_PRIORITY.get(str(source).lower(), 99)


def _merge_records(group: pd.DataFrame, key_cols: list[str]) -> pd.Series:
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
    logger.info("Pass 1: Exact ID deduplication...")


    payroll_deduped = (
        payroll_df
        .sort_values("effective_date", ascending=False, na_position="last")
        .drop_duplicates(subset=["employee_id"], keep="first")
        .copy()
    )
    pay_dup_removed = len(payroll_df) - len(payroll_deduped)
    logger.info(f"  Payroll duplicates removed (keep latest): {pay_dup_removed:,}")


    hris_ids = set(hris_df["employee_id"].dropna())
    pay_ids  = set(payroll_deduped["employee_id"].dropna())
    ghost_ids = pay_ids - hris_ids
    ghost_df  = payroll_deduped[payroll_deduped["employee_id"].isin(ghost_ids)].copy()
    ghost_df["ghost_flag_reason"] = "Payroll record with no matching HRIS employee_id"
    logger.info(f"  Ghost employee candidates identified: {len(ghost_df):,}")

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

    enrolled_ids = set(benefits_df["employee_id"].dropna())
    merged["benefits_enrolled"] = merged["employee_id"].isin(enrolled_ids)

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


def pass2_email_match(merged_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    logger.info(f"Pass 3: Fuzzy name match (threshold={FUZZY_THRESHOLD}%, "
                f"hire date window={HIRE_DATE_WINDOW} days)...")

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

    gt_sorted = gt_records[["employee_id", "full_name", "hire_day"]].copy()
    ac_sorted = ac_records[["employee_id", "full_name", "hire_day"]].copy()
    gt_sorted["hire_day"] = gt_sorted["hire_day"].astype(int)
    ac_sorted["hire_day"] = ac_sorted["hire_day"].astype(int)

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
    logger.info(f"  Probable matches found  : {len(probable_matches):,}")

    if not probable_matches:
        return pd.DataFrame()

    result = pd.DataFrame(probable_matches).sort_values("similarity_score", ascending=False)
    return result


def deduplicate(cleaned: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    logger.info("=" * 60)
    logger.info("DEDUPLICATION LAYER")
    logger.info("=" * 60)

    hris_df     = cleaned["hris"]
    acq_df      = cleaned["acquiredco"]
    payroll_df  = cleaned["payroll"]
    benefits_df = cleaned["benefits"]


    combined_hris = pd.concat(
        [hris_df, acq_df],
        ignore_index=True,
        sort=False,
    )
    logger.info(f"Combined HRIS (GT + AC): {len(combined_hris):,} records")

    golden, ghost_df = pass1_exact_id(combined_hris, payroll_df, benefits_df)

    golden, email_review = pass2_email_match(golden)

    fuzzy_review = pass3_fuzzy_name(golden)


    review_frames = [f for f in [email_review, fuzzy_review] if not f.empty]
    probable_matches = (
        pd.concat(review_frames, ignore_index=True)
        if review_frames
        else pd.DataFrame(columns=[
            "record_1_id", "record_2_id", "similarity_score",
            "hire_date_diff_days", "match_type", "recommended_action",
        ])
    )


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
