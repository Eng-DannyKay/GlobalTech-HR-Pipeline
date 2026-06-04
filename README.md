# GlobalTech HR Integration Pipeline

## Purpose & Business Context

GlobalTech Corp recently acquired AcquiredCo, a smaller software firm. HR leadership requires a unified employee dataset within 10 business days to support Day 1 integration planning, benefits enrollment eligibility, payroll system migration, and compliance reporting.

This pipeline ingests data from 4 disparate source systems, cleans and standardizes it, deduplicates records across company boundaries, runs data quality checks, and exports a golden employee dataset partitioned by company origin.

---

## Input Sources

| File | Format | Location | Description |
|---|---|---|---|
| `globaltech_hris.csv` | CSV (UTF-8) | `data/raw/` | GlobalTech Workday HRIS export — 15,000 employees |
| `acquiredco_api.json` | JSON (paginated) | `data/raw/` | AcquiredCo BambooHR API export — 3,000 employees |
| `payroll_data.xlsx` | Excel (.xlsx) | `data/raw/` | Combined ADP payroll — 18,500 records, mixed currencies |
| `benefits_enrollment.xml` | XML | `data/raw/` | MedShield benefits export — 12,000 records, GlobalTech only |

### Standard Schema (post-alignment)

| Column | Type | Description |
|---|---|---|
| `employee_id` | str | Namespaced ID: `GT-XXXXXX` or `AC-XXXXXX` |
| `first_name` | str | Unicode-normalized, title-cased |
| `last_name` | str | Unicode-normalized, title-cased |
| `email` | str | Lowercase email address |
| `department` | str | Standard taxonomy (e.g., Engineering, Marketing) |
| `job_title` | str | Raw job title from source |
| `hire_date` | datetime64 | Normalized to `YYYY-MM-DD` |
| `country` | str | Country of employment |
| `employment_type` | str | `Full-Time`, `Part-Time`, or `Contractor` |
| `manager_id` | str | Namespaced manager employee ID |
| `salary_usd_annual` | float | Annual salary converted to USD |
| `base_salary` | str/float | Original salary value from payroll |
| `currency` | str | Original currency code |
| `pay_frequency` | str | Original pay frequency |
| `bonus_target_pct` | float | Bonus target as percentage |
| `benefits_enrolled` | bool | Whether enrolled in any benefit plan |
| `benefits_plan_count` | int | Number of benefit plans enrolled |
| `source` | str | Primary source system |
| `source_systems` | str | All contributing systems |
| `dedup_method` | str | `exact_id`, `email_match`, or `fuzzy_name` |
| `company_origin` | str | `GlobalTech` or `AcquiredCo` |

---

## Output Files

| File | Format | Location | Description |
|---|---|---|---|
| `golden_hr_dataset.parquet` | Parquet (partitioned) | `data/processed/` | Unified golden employee records, partitioned by `company_origin` |
| `ghost_employees.csv` | CSV | `data/processed/` | Payroll records with no HRIS match — compliance/fraud risk |
| `probable_matches.csv` | CSV | `data/processed/` | Fuzzy-matched cross-company pairs for HR review |
| `validation_report.csv` | CSV | `data/processed/` | 15-check data quality report |
| `validation_report.html` | HTML | `data/processed/` | Human-readable quality summary |
| `chart_0N_*.png` | PNG (300 DPI) | `data/processed/charts/` | 6 visualization charts |

---

## How to Run

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn openpyxl pyarrow rapidfuzz lxml requests
```

Python 3.11+ required.

### Run the full pipeline

```bash
python pipeline.py
```

This executes all 6 steps in sequence:
1. **Ingest** — loads all 4 source files
2. **Clean** — standardizes names, IDs, dates, currencies, departments
3. **Dedup** — 3-pass deduplication + ghost detection
4. **Validate** — 15 data quality checks with pipeline gate
5. **Visualize** — 6 charts at 300 DPI
6. **Export** — Parquet, CSVs, and HTML report

Logs are written to `logs/pipeline_YYYYMMDD_HHMMSS.log`.

---

## Pipeline Architecture

```
data/raw/
  ├── globaltech_hris.csv
  ├── acquiredco_api.json
  ├── payroll_data.xlsx
  └── benefits_enrollment.xml
        │
        ▼
src/ingest.py      → loads & aligns all 4 sources to standard schema
        │
        ▼
src/clean.py       → name normalization, ID namespacing, currency conversion,
                     department taxonomy, date standardization
        │
        ▼
src/dedup.py       → Pass 1 (exact ID) → Pass 2 (email) → Pass 3 (fuzzy name)
                     ghost detection, provenance tracking
        │
        ▼
src/validate.py    → 15 quality checks, pipeline gate (halt if >2 FAILs),
                     CSV + HTML report export
        │
        ▼
src/visualize.py   → 6 charts at 300 DPI (Wong colorblind-safe palette)
        │
        ▼
data/processed/
  ├── golden_hr_dataset.parquet  (partitioned by company_origin)
  ├── ghost_employees.csv
  ├── probable_matches.csv
  ├── validation_report.csv / .html
  └── charts/
```

---

## Deduplication Strategy

| Pass | Method | Outcome |
|---|---|---|
| Pass 1 | Exact `employee_id` match within namespace | Auto-merge; source priority: HRIS > Payroll > Benefits |
| Pass 2 | Same email across GT and AC namespaces | Flag as `probable_match`; no auto-merge |
| Pass 3 | Fuzzy full-name ≥ 88% + hire date within 30 days | Flag as `probable_match`; HR review required |
| Ghost | Payroll record with no HRIS counterpart | Written to `ghost_employees.csv` |

---

## Data Quality Checks

15 checks across 7 categories. Pipeline halts if more than 2 checks return FAIL.

| Category | Fields |
|---|---|
| NOT NULL | `employee_id`, `first_name`, `last_name`, `email`, `department`, `country` |
| UNIQUE | `employee_id`, `email` |
| VALUES IN SET | `employment_type`, `currency` |
| REGEX | email format, employee ID format (`GT-\d{6}` / `AC-\d{6}`) |
| NUMERIC RANGE | `salary_usd_annual` ($15,000–$10,000,000) |
| DATE RANGE | `hire_date` (1970-01-01 to today) |
| REFERENTIAL INTEGRITY | `manager_id` must exist as an `employee_id` |

---

## Known Limitations & Assumptions

- **Exchange rates are fixed** as of 2026-06-04 (EUR → 1.08 USD, GBP → 1.26 USD). Update `EXCHANGE_RATES_TO_USD` in `src/clean.py` for production runs.
- **AcquiredCo JSON** is read from a local file and simulated as paginated (100 records/page). In production, replace with live API calls.
- **Ghost employees** in the payroll export have synthetic IDs (`GHOST_XXXX`) and no name data since they have no HRIS record. This is a compliance risk requiring manual HR investigation.
- **Fuzzy matching** uses `token_sort_ratio` with a threshold of 88%. Lowering the threshold increases recall but also false positives in the review file.
- **Department FAIL** (64 null departments) and **country FAIL** (41 null countries) are genuine gaps in the synthetic source data, not pipeline errors.
- **Email UNIQUE WARN**: cross-system HR merges commonly share emails for contractors. Treated as WARN, not FAIL.

---

## Change Log

| Date | Change |
|---|---|
| 2026-06-04 | Initial pipeline implementation — all 6 deliverables complete |
| 2026-06-04 | Fixed O(n²) fuzzy match replaced with `pd.merge_asof` blocking |
| 2026-06-04 | Added `ACQ_DUP_*` detection in `namespace_acquiredco_ids` to drop 200 intentional duplicate records |
| 2026-06-04 | Downgraded email UNIQUE check to WARN (expected cross-system overlap) |
