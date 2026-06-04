"""
visualize.py — Data Visualization Layer
GlobalTech HR Integration Pipeline

Produces 6 publication-quality charts at 300 DPI using a colorblind-safe palette.
All charts include titles, axis labels, data source annotation, and a generation timestamp.
"""

import logging
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for pipeline runs
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Design constants ──────────────────────────────────────────────────────────
# Wong (2011) colorblind-safe 8-color palette
PALETTE = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#999999",  # grey
]

DPI         = 300
FONT_FAMILY = "DejaVu Sans"
SOURCE_NOTE = "Source: GlobalTech Corp Multi-Source HR Integration Pipeline"
TIMESTAMP   = datetime.now().strftime("%Y-%m-%d %H:%M")


def _apply_style() -> None:
    """Apply consistent matplotlib style."""
    plt.rcParams.update({
        "font.family":          FONT_FAMILY,
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "axes.titlesize":       13,
        "axes.titleweight":     "bold",
        "axes.labelsize":       10,
        "xtick.labelsize":      9,
        "ytick.labelsize":      9,
        "figure.dpi":           DPI,
        "savefig.dpi":          DPI,
        "figure.facecolor":     "white",
        "axes.facecolor":       "white",
    })


def _add_footnote(fig: plt.Figure) -> None:
    """Add source note and timestamp to the bottom of a figure."""
    fig.text(
        0.01, 0.005,
        f"{SOURCE_NOTE}  |  Generated: {TIMESTAMP}",
        fontsize=6, color="#666666", ha="left", va="bottom",
    )


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"  Saved: {path.name}")


# ── Chart 1: Headcount by Department ─────────────────────────────────────────

def chart_headcount_by_department(golden: pd.DataFrame, out_dir: Path) -> Path:
    _apply_style()
    counts = (
        golden["department"].dropna()
        .value_counts()
        .sort_values(ascending=True)
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(counts.index, counts.values, color=PALETTE[1], edgecolor="white")

    # Value labels
    for bar, val in zip(bars, counts.values):
        ax.text(
            val + 20, bar.get_y() + bar.get_height() / 2,
            f"{val:,}", va="center", ha="left", fontsize=8,
        )

    ax.set_xlabel("Number of Employees")
    ax.set_title("Employee Headcount by Department")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xlim(0, counts.max() * 1.12)

    _add_footnote(fig)
    fig.tight_layout(rect=[0, 0.02, 1, 1])

    path = out_dir / "chart_01_headcount_by_department.png"
    _save(fig, path)
    return path


# ── Chart 2: Headcount by Country ────────────────────────────────────────────

def chart_headcount_by_country(golden: pd.DataFrame, out_dir: Path) -> Path:
    _apply_style()
    counts = (
        golden["country"].dropna()
        .value_counts()
        .head(20)           # top 20 countries for readability
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(11, 5))
    x = range(len(counts))
    bars = ax.bar(x, counts.values, color=PALETTE[0], edgecolor="white", width=0.65)

    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, val + 15,
            f"{val:,}", ha="center", va="bottom", fontsize=7.5,
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(counts.index, rotation=40, ha="right")
    ax.set_ylabel("Number of Employees")
    ax.set_title("Top 20 Countries by Employee Headcount")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.set_ylim(0, counts.max() * 1.15)

    _add_footnote(fig)
    fig.tight_layout(rect=[0, 0.02, 1, 1])

    path = out_dir / "chart_02_headcount_by_country.png"
    _save(fig, path)
    return path


# ── Chart 3: Salary Distribution by Employment Type (violin) ─────────────────

def chart_salary_distribution(golden: pd.DataFrame, out_dir: Path) -> Path:
    _apply_style()

    salary_col = "salary_usd_annual"
    if salary_col not in golden.columns:
        logger.warning("  salary_usd_annual column missing — skipping chart 3")
        return None

    data = golden[[salary_col, "employment_type"]].dropna()
    emp_types = sorted(data["employment_type"].unique())

    fig, ax = plt.subplots(figsize=(9, 6))

    parts = ax.violinplot(
        [data.loc[data["employment_type"] == et, salary_col].values for et in emp_types],
        positions=range(len(emp_types)),
        showmedians=True,
        showextrema=True,
    )

    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(PALETTE[i % len(PALETTE)])
        pc.set_alpha(0.75)
    parts["cmedians"].set_color("#333333")
    parts["cbars"].set_color("#888888")
    parts["cmins"].set_color("#888888")
    parts["cmaxes"].set_color("#888888")

    # Overlay median labels
    for i, et in enumerate(emp_types):
        med = data.loc[data["employment_type"] == et, salary_col].median()
        ax.text(
            i, med,
            f"  ${med/1000:.0f}K",
            va="center", fontsize=8, color="#333333",
        )

    ax.set_xticks(range(len(emp_types)))
    ax.set_xticklabels(emp_types)
    ax.set_ylabel("Annual Salary (USD)")
    ax.set_title("Salary Distribution by Employment Type")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"${v/1_000_000:.1f}M" if v >= 1_000_000 else f"${v/1_000:.0f}K")
    )

    _add_footnote(fig)
    fig.tight_layout(rect=[0, 0.02, 1, 1])

    path = out_dir / "chart_03_salary_distribution.png"
    _save(fig, path)
    return path


# ── Chart 4: Tenure Distribution (histogram) ─────────────────────────────────

def chart_tenure_distribution(golden: pd.DataFrame, out_dir: Path) -> Path:
    _apply_style()

    today = pd.Timestamp(datetime.today().date())
    hire_dates = pd.to_datetime(golden["hire_date"], errors="coerce").dropna()
    tenure_years = ((today - hire_dates).dt.days / 365.25).round(2)
    tenure_years = tenure_years[tenure_years >= 0]

    fig, ax = plt.subplots(figsize=(9, 5))

    n_bins = 20
    ax.hist(
        tenure_years, bins=n_bins,
        color=PALETTE[2], edgecolor="white", linewidth=0.6,
    )

    median_t = tenure_years.median()
    ax.axvline(median_t, color=PALETTE[5], linewidth=1.5, linestyle="--",
               label=f"Median: {median_t:.1f} yrs")
    ax.legend(fontsize=9)

    ax.set_xlabel("Tenure (years)")
    ax.set_ylabel("Number of Employees")
    ax.set_title("Employee Tenure Distribution")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

    _add_footnote(fig)
    fig.tight_layout(rect=[0, 0.02, 1, 1])

    path = out_dir / "chart_04_tenure_distribution.png"
    _save(fig, path)
    return path


# ── Chart 5: Benefits Enrollment Rate by Department ──────────────────────────

def chart_benefits_enrollment(golden: pd.DataFrame, out_dir: Path) -> Path:
    _apply_style()

    if "benefits_enrolled" not in golden.columns:
        logger.warning("  benefits_enrolled column missing — skipping chart 5")
        return None

    dept_data = (
        golden.dropna(subset=["department"])
        .groupby("department")["benefits_enrolled"]
        .agg(enrolled="sum", total="count")
    )
    dept_data["rate"] = dept_data["enrolled"] / dept_data["total"] * 100
    dept_data = dept_data.sort_values("rate", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(
        dept_data.index, dept_data["rate"],
        color=[PALETTE[2] if r >= 50 else PALETTE[5] for r in dept_data["rate"]],
        edgecolor="white",
    )

    for bar, (_, row) in zip(bars, dept_data.iterrows()):
        ax.text(
            row["rate"] + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{row['rate']:.1f}%  ({int(row['enrolled'])}/{int(row['total'])})",
            va="center", ha="left", fontsize=8,
        )

    ax.set_xlabel("Enrollment Rate (%)")
    ax.set_title("Benefits Enrollment Rate by Department")
    ax.set_xlim(0, 115)
    ax.axvline(50, color="#aaaaaa", linewidth=0.8, linestyle="--")

    _add_footnote(fig)
    fig.tight_layout(rect=[0, 0.02, 1, 1])

    path = out_dir / "chart_05_benefits_enrollment.png"
    _save(fig, path)
    return path


# ── Chart 6: Data Quality Summary ────────────────────────────────────────────

def chart_data_quality_summary(report: pd.DataFrame, out_dir: Path) -> Path:
    _apply_style()

    labels     = report["check_name"].str.replace(r"^[^:]+:", "", regex=True)
    passed     = report["total_rows"] - report["failed_rows"]
    failed     = report["failed_rows"]
    x          = np.arange(len(labels))
    bar_width  = 0.4

    # Map status to colours
    status_color = {"PASS": PALETTE[2], "WARN": PALETTE[0], "FAIL": PALETTE[5]}
    bar_colors = [status_color.get(s, PALETTE[7]) for s in report["status"]]

    fig, ax = plt.subplots(figsize=(14, 5))

    bars_pass = ax.bar(x - bar_width / 2, passed, bar_width,
                       label="Passed", color=PALETTE[2], alpha=0.85)
    bars_fail = ax.bar(x + bar_width / 2, failed, bar_width,
                       label="Failed / Flagged", color=PALETTE[5], alpha=0.85)

    # Status badges above each pass bar
    for i, (_, row) in enumerate(report.iterrows()):
        color = status_color.get(row["status"], PALETTE[7])
        ax.text(
            x[i] - bar_width / 2,
            passed.iloc[i] + report["total_rows"].max() * 0.01,
            row["status"],
            ha="center", va="bottom", fontsize=6.5,
            color=color, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Row Count")
    ax.set_title("Data Quality Check Results — Passed vs. Failed Rows")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.legend(fontsize=9)

    _add_footnote(fig)
    fig.tight_layout(rect=[0, 0.02, 1, 1])

    path = out_dir / "chart_06_data_quality_summary.png"
    _save(fig, path)
    return path


# ── Master function ───────────────────────────────────────────────────────────

def visualize(golden: pd.DataFrame, report: pd.DataFrame,
              output_dir: Path) -> list[Path]:
    """
    Generate all 6 charts and save to output_dir.

    Parameters
    ----------
    golden     : Deduplicated golden DataFrame
    report     : Validation report DataFrame (from validate.validate())
    output_dir : Directory to save PNG files

    Returns
    -------
    List of Path objects for each saved chart
    """
    logger.info("=" * 60)
    logger.info("VISUALIZATION LAYER")
    logger.info("=" * 60)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    charts = []

    logger.info("  Chart 1: Headcount by Department")
    charts.append(chart_headcount_by_department(golden, output_dir))

    logger.info("  Chart 2: Headcount by Country")
    charts.append(chart_headcount_by_country(golden, output_dir))

    logger.info("  Chart 3: Salary Distribution by Employment Type")
    charts.append(chart_salary_distribution(golden, output_dir))

    logger.info("  Chart 4: Tenure Distribution")
    charts.append(chart_tenure_distribution(golden, output_dir))

    logger.info("  Chart 5: Benefits Enrollment Rate by Department")
    charts.append(chart_benefits_enrollment(golden, output_dir))

    logger.info("  Chart 6: Data Quality Summary")
    charts.append(chart_data_quality_summary(report, output_dir))

    saved = [c for c in charts if c is not None]

    logger.info("=" * 60)
    logger.info("VISUALIZATION SUMMARY")
    logger.info(f"  Charts generated: {len(saved)} / 6")
    logger.info(f"  Output directory: {output_dir}")
    logger.info("=" * 60)

    return saved
