# app.py
# ============================================================
# Medicare Provider Benchmarking Engine (Portfolio Showcase App)
# Python Shiny app that tells a coherent story:
#   Overview -> Benchmark Explorer -> Anomaly Surfacing -> Tiering -> Explainer -> Documentation
#
# Design goals:
#   - Narrative-first, fewer controls
#   - Reproducible: loads frozen artifacts + params manifests
#   - Drill-downs are lazy: eval_scored_DG_V3 is only filtered when needed
#
# To run (use your active env python):
#   python -m shiny run --reload app.py
# ============================================================

from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from shiny import App, ui, render, reactive

# shinywidgets is optional. Only import if installed.
try:
    from shinywidgets import output_widget, render_widget  # noqa: F401
except Exception:
    output_widget = None
    render_widget = None

import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)

# ============================================================
# CONFIG) Artifact auto-discovery (always load the latest frozen runs)
# ============================================================

REPO_ROOT = Path(__file__).resolve().parent
ARTIFACTS = REPO_ROOT / "artifacts"


def _latest_run_dir(parent: Path, prefix: str = "run_") -> Path:
    if not parent.exists():
        raise FileNotFoundError(f"Missing artifacts directory: {parent}")
    runs = [p for p in parent.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not runs:
        raise FileNotFoundError(f"No {prefix}* folders found under: {parent}")
    # folder names are run_YYYYMMDD_HHMMSS so lexical sort works
    return sorted(runs, key=lambda p: p.name)[-1]


def _pick_one(run_dir: Path, pattern: str) -> Path:
    hits = sorted(run_dir.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"No files matching {pattern} in {run_dir}")
    return hits[-1]


# ---- Eval universe (single source of truth) ----
EVAL_DIR = ARTIFACTS / "eval_universe"
EVAL_SCORED_PATH = EVAL_DIR / "eval_scored_DG_V3.parquet"
if not EVAL_SCORED_PATH.exists():
    raise FileNotFoundError(f"Missing eval universe parquet: {EVAL_SCORED_PATH}")

# Small read for year choices (fast, one column)
YEARS = sorted(
    pd.read_parquet(EVAL_SCORED_PATH, columns=["Year"])["Year"]
    .dropna()
    .astype(int)
    .unique()
    .tolist()
)

# ---- Anomaly surfacing (latest frozen run) ----
ANOM_DIR = ARTIFACTS / "anomaly_surfaces"
ANOM_RUN = _latest_run_dir(ANOM_DIR)

ROWS_PRIMARY_PATH = _pick_one(ANOM_RUN, "rows_primary__ANOM_2_a_1__*.parquet")
ROWS_ALT_PATH = _pick_one(ANOM_RUN, "rows_alt__ANOM_2_b_1__*.parquet")
PROV_PRIMARY_PATH = _pick_one(ANOM_RUN, "providers_primary__ANOM_3_a_1__*.parquet")
PROV_ALT_PATH = _pick_one(ANOM_RUN, "providers_alt__ANOM_3_b_1__*.parquet")
ANOM_PARAMS_JSON = _pick_one(ANOM_RUN, "run_params__*.json")

# ---- Provider tiering (latest frozen run) ----
TIER_DIR = ARTIFACTS / "provider_tiering"
TIER_RUN = _latest_run_dir(TIER_DIR)

PROVIDER_SCORECARD_PATH = _pick_one(TIER_RUN, "provider_scorecard_v1_clustered__*.parquet")
CLUSTER_PROFILES_PATH = _pick_one(TIER_RUN, "cluster_profiles_v1_compact__*.parquet")
TIER_PARAMS_JSON = _pick_one(TIER_RUN, "params__*.json")

# ---- Provider classification (latest frozen run) ----
CLF_DIR = ARTIFACTS / "provider_classification"
CLF_RUN = _latest_run_dir(CLF_DIR)

FINAL_EXPLAINER_JOBLIB = _pick_one(CLF_RUN, "final_explainer_logistic__*.joblib")
CLF_PARAMS_JSON = _pick_one(CLF_RUN, "params__*.json")

print("\n[Artifact discovery]")
print("EVAL_SCORED_PATH:", EVAL_SCORED_PATH)
print("ANOM_RUN:", ANOM_RUN)
print("TIER_RUN:", TIER_RUN)
print("CLF_RUN:", CLF_RUN)
print("PROVIDER_SCORECARD_PATH:", PROVIDER_SCORECARD_PATH)
print("FINAL_EXPLAINER_JOBLIB:", FINAL_EXPLAINER_JOBLIB)




# ---- UI choice lists (read minimal columns) ----
POS_CHOICES = sorted(
    pd.read_parquet(EVAL_SCORED_PATH, columns=["Place_Of_Srvc"])["Place_Of_Srvc"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

STATE_CHOICES = sorted(
    pd.read_parquet(EVAL_SCORED_PATH, columns=["state"])["state"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)





# ============================================================
# 1) LOADERS (fast artifacts are loaded at startup)
# ============================================================

def _require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")

def _safe_read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())

def _safe_read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)

# Validate minimum set so app fails early with a clear message
_require(EVAL_SCORED_PATH)
_require(PROVIDER_SCORECARD_PATH)
_require(ROWS_PRIMARY_PATH)
_require(ROWS_ALT_PATH)
_require(PROV_PRIMARY_PATH)
_require(PROV_ALT_PATH)
_require(ANOM_PARAMS_JSON)

scorecard_all = pd.read_parquet(PROVIDER_SCORECARD_PATH)
tiering_params = _safe_read_json(TIER_PARAMS_JSON)

anom_rows_primary = _safe_read_parquet(ROWS_PRIMARY_PATH)
anom_rows_alt = _safe_read_parquet(ROWS_ALT_PATH)
anom_prov_primary = _safe_read_parquet(PROV_PRIMARY_PATH)
anom_prov_alt = _safe_read_parquet(PROV_ALT_PATH)
anom_params = _safe_read_json(ANOM_PARAMS_JSON)

# Optional: cluster profiles artifact (not required, but nice)
cluster_profiles_compact = _safe_read_parquet(CLUSTER_PROFILES_PATH)

# Optional explainer model
clf_pipe = None
clf_params = _safe_read_json(CLF_PARAMS_JSON)
if FINAL_EXPLAINER_JOBLIB.exists():
    try:
        clf_pipe = joblib.load(FINAL_EXPLAINER_JOBLIB)
    except Exception:
        clf_pipe = None


# ============================================================
# 2) SHARED HELPERS
# ============================================================

KEY = ["Rndrng_NPI", "provider_type", "state"]





SLICE_SUMMARY_COL_DICT = {
    "Metric (counts)": (
        "A descriptive label for a count-style metric (units are typically providers, HCPCS codes, or anomaly rows). "
        "Example: 'Providers (provider grain)' or 'Unique HCPCS (row grain)'."
    ),
    "Value (counts)": (
        "The value for the count-style metric shown in 'Metric (counts)'. "
        "Example: 23,366 providers, or 1,449 unique HCPCS codes."
    ),
    "Metric (volume/events)": (
        "A descriptive label for a volume or event-style metric (units are services, benes, or anomaly rows). "
        "Example: 'Total services (provider scorecard)' or 'Robust events (ANOM.2.a.1 rows)'."
    ),
    "Value (volume/events)": (
        "The value for the volume/event metric shown in 'Metric (volume/events)'. "
        "Example: 1,261,109,154 services, or 200 robust anomaly rows."
    ),
}








CLUSTER_CARD_COL_DICT = {
    "cluster_label_v1": (
        "The cluster identifier assigned by KMeans for eligible providers. "
        "Values: 'cluster_0' (Typical cost behavior) and 'cluster_1' (Elevated anomaly burden). "
        "Non-eligible providers are excluded from this table."
    ),
    "cluster_definition_v1": (
        "Human-readable meaning of the cluster label. "
        "This is your narrative label and should be used in writeups and the UI."
    ),
    "n_providers": (
        "Number of providers in this cluster under the current filters. "
        "Unit: providers at provider grain (Rndrng_NPI, provider_type, state)."
    ),
    "median_n_anom_rows_robust": (
        "Median number of robust tail-event rows per provider in this cluster. "
        "Robust events come from ANOM.2.a.1-style logic (repeat-offender signal). "
        "Example: median=1 means the typical provider in this cluster has 1 robust flagged row."
    ),
    "median_anom_rate_pct_robust": (
        "Median provider-level robust anomaly rate (percent). "
        "Computed per provider as (robust anomalous rows / provider rows) * 100, then the median is taken within the cluster. "
        "Example: 1.013 means ~1.013% of that provider’s rows are robust anomalies, for the median provider in the cluster."
    ),
    "median_p90_log_oe": (
        "Median of the provider-level p90 log(O/E) within the cluster. "
        "For each provider we compute p90(log_oe) across their rows, then we take the median of that statistic across providers. "
        "Interpretation: a typical provider’s upper-tail over-expected behavior in this cluster."
    ),
    "pct_with_any_shock_event": (
        "Percent of providers in this cluster with at least one shock event. "
        "Shock event definition uses the magnitude-aware rule (ANOM.2.b.1-style), often requiring log_oe above a global severity cutoff plus slice and confidence gates."
    ),
    "median_total_services_rob": (
        "Median of total services per provider (summed across rows) within the cluster. "
        "This is a scale indicator, useful context for why some providers accumulate more events."
    ),
}









BENCHMARK_TOP_ROWS_COL_DICT = {
    "row_id": (
        "Unique identifier for the modeled row. "
        "Row grain: (Rndrng_NPI, HCPCS_Cd, Place_Of_Srvc, Year). "
        "Use this as the stable key when cross-referencing anomalies and slices."
    ),
    "Rndrng_NPI": (
        "Provider identifier (NPI). This is the billing provider at the row grain."
    ),
    "HCPCS_Cd": (
        "HCPCS code (procedure, service, or drug/supply code) for the row."
    ),
    "Place_Of_Srvc": (
        "CMS place-of-service code (where the service occurred). "
        "Example: 'O' often reflects outpatient settings (depends on your source coding)."
    ),
    "Year": (
        "Calendar year for the row."
    ),
    "observed_cost": (
        "Observed standardized cost per unit for this row (what was actually billed/paid in the dataset). "
        "Unit matches your modeling target scale (the same unit used for expected_cost)."
    ),
    "expected_cost": (
        "Model-predicted expected standardized cost per unit for this row (D/G+V3 output). "
        "This is the benchmark baseline used to compute residual and O/E."
    ),
    "residual": (
        "Residual cost difference for this row: observed_cost - expected_cost. "
        "Positive means over-expected. Negative means under-expected."
    ),
    "log_oe": (
        "Log over-expected ratio: log(observed_cost / expected_cost). "
        "Interpretation: 0 means observed equals expected. Positive means over-expected. "
        "Example: log_oe=0.69 corresponds to ~2.0x O/E (since exp(0.69)≈2)."
    ),
    "oe_ratio": (
        "Over-expected ratio: observed_cost / expected_cost. "
        "Interpretation: 1.0 means observed equals expected. 10 means 10x expected."
    ),
    "services": (
        "Service volume (count) for this row. "
        "Higher values mean more volume supporting the row estimate and totals."
    ),
    "benes": (
        "Beneficiary count for this row. "
        "Higher values mean more unique patients contributing to the observed utilization/cost."
    ),
}








ANOM2A1_COL_DICT = {
    # Identity / grain
    "row_id": (
        "Unique identifier for a row-grain record. "
        "Row grain is (Rndrng_NPI, HCPCS_Cd, Place_Of_Srvc, Year)."
    ),
    "Rndrng_NPI": "Provider identifier (NPI).",
    "provider_type": "Provider specialty/type used for grouping and reporting (e.g., Medical Oncology).",
    "state": "Provider state.",
    "zip5": "Provider ZIP5 (reporting context only, not used in event logic).",

    # Code + taxonomy descriptors
    "HCPCS_Cd": "HCPCS code defining the billed service/procedure/supply.",
    "hcpcs_desc": "Human-readable HCPCS description (context only).",
    "rbcs_family_desc": (
        "RBCS family descriptor if available. Useful for grouping related services for interpretation."
    ),

    # Row-grain coordinates
    "Place_Of_Srvc": (
        "Place of service code (POS). Part of the row grain. "
        "Example: 'O' typically indicates outpatient."
    ),
    "Year": "Claim/service year. Part of the row grain.",

    # Slice features used for “size-aware” gating
    "slice_n": (
        "Slice size for the peer group used to compute within-slice percentiles. "
        "In this notebook, slice is typically (HCPCS_Cd, Year). Larger slice_n implies more stable percentiles."
    ),
    "slice_key": (
        "Convenience identifier for the slice. Typically formatted like '{HCPCS}_{Year}'."
    ),

    # Volume context
    "services": "Service volume for the row (count-like measure). Used as a stability proxy.",
    "benes": "Beneficiary count for the row. Used as a stability proxy.",

    # Support / routing context from modeling
    "expected_cost_support_tier": (
        "Support tier for the expected_cost estimate (e.g., high, medium_high, medium, low). "
        "Derived from whether the row has lag support or which cold-start anchor source was used."
    ),
    "has_lag": (
        "Whether the row had lag-based historical support available (True = hot-start; False = cold-start)."
    ),
    "route": (
        "Modeling route context (e.g., hot_start vs cold_start). "
        "Useful for interpreting stability and why expected_cost is well-supported or not."
    ),

    # Core scored columns (what the benchmark engine produces)
    "observed_cost": "Observed standardized cost per service at the row grain (dollars).",
    "expected_cost": "Expected (predicted) cost per service at the same row grain (dollars).",
    "residual": (
        "Residual = observed_cost - expected_cost (dollars). "
        "Positive residual means higher-than-expected cost."
    ),
    "oe_ratio": (
        "Observed-to-Expected ratio = observed_cost / expected_cost. "
        "Multiplicative deviation from benchmark."
    ),
    "log_oe": (
        "log(O/E) where O/E = oe_ratio. "
        "Interpretation: exp(log_oe) gives the O/E multiplier."
    ),

    # Percentile rank features (within the slice)
    "log_oe_pct_in_slice": (
        "Percentile rank of log_oe within the peer slice (HCPCS_Cd, Year). "
        "Example: 0.99 means top 1% within that slice."
    ),
    "resid_pct_in_slice": (
        "Percentile rank of residual within the peer slice (HCPCS_Cd, Year). "
        "Used as supporting context for tail behavior."
    ),

    # Confidence gates
    "is_high_conf": (
        "Strict confidence gate used in anomaly surfacing. "
        "Typically requires expected_cost_support_tier in {high, medium_high} and minimum services/benes."
    ),
    "high_confidence_anomaly_candidate": (
        "Legacy candidate flag for high-confidence anomalies from modeling outputs. "
        "Often reflects high support + high positive residual + high O/E type signals."
    ),

    # Guardrail / post-processing context
    "any_guardrail_changed": (
        "Whether any guardrail adjusted the original expected/residual-derived fields for this row."
    ),
    "guardrail_name": (
        "Name of the guardrail policy applied (if any). Useful for auditability."
    ),

    # Robust anomaly scoring + explanation
    "anom_score_robust": (
        "Row-level robust anomaly score used for ranking within the ANOM.2.a.1 list. "
        "Designed to prioritize high-confidence, positive-residual, tail log_oe behavior in sufficiently large slices."
    ),
    "anom_reason": (
        "Human-readable reason string summarizing which rule components fired. "
        "Example: 'top_1pct_log_oe_in_slice + high_conf + positive_residual'."
    ),
}





# Start from the robust dictionary and override only the mag-specific fields
ANOM2B1_COL_DICT = dict(ANOM2A1_COL_DICT)

ANOM2B1_COL_DICT.update({
    "anom_score_mag": (
        "Row-level magnitude anomaly score used for ranking within the ANOM.2.b.1 list. "
        "This variant emphasizes severity. It requires the same basic trust gates (positive residual, tail rank in-slice, "
        "minimum slice_n, and confidence gate), plus an additional severity condition (log_oe above the global severity cut "
        "computed from the positive-residual universe)."
    ),
    "anom_reason": (
        "Human-readable reason string summarizing which rule components fired for the magnitude-aware definition. "
        "Same idea as primary, but the underlying rule includes the global severity cutoff."
    ),
})







ANOM3A1_COL_DICT = {
    "Rndrng_NPI": (
        "Provider identifier (NPI). Provider grain key."
    ),
    "provider_type": (
        "Provider specialty/type used for benchmarking slices and rollups (e.g., Medical Oncology, Hematology-Oncology)."
    ),
    "state": (
        "Provider state (2-letter). Provider grain key."
    ),

    "n_rows": (
        "Number of row-grain records for this provider in the scored universe. "
        "Each row corresponds to a (NPI, HCPCS, Place_Of_Srvc, Year) combination."
    ),

    "n_anom_rows_robust": (
        "Count of row-grain records for this provider that were flagged as robust anomalies under ANOM.3.a.1. "
        "This is the core 'repeat-offender' signal. "
        "Example: n_anom_rows_robust=7 means 7 distinct (HCPCS, POS, Year) rows for this provider met the robust anomaly rule."
    ),

    "anom_rate_pct_robust": (
        "Percent of this provider’s rows that are robust anomalies. "
        "Computed as 100 * (n_anom_rows_robust / n_rows). "
        "Example: 0.69 means ~0.69% of the provider’s rows were flagged."
    ),

    "n_unique_codes": (
        "Breadth metric. Number of unique HCPCS codes present in this provider’s rows (across all years and places of service). "
        "Higher values typically mean broader service mix."
    ),

    "n_unique_years": (
        "Breadth metric. Number of distinct years in which this provider appears in the scored universe."
    ),

    "median_oe": (
        "Provider-level median observed-to-expected ratio across all their rows. "
        "oe_ratio = observed_cost / expected_cost. "
        "Interpretation: ~1.00 is 'on benchmark' at the median row. >1.00 is over-expected."
    ),

    "median_log_oe": (
        "Provider-level median log(O/E) across rows, where log_oe = log(oe_ratio). "
        "Interpretation: 0 is neutral. Positive means over-expected. Negative means under-expected."
    ),

    "median_residual": (
        "Provider-level median residual across rows, where residual = observed_cost - expected_cost. "
        "Interpretation: near 0 means typical at the median row. Positive means observed > expected."
    ),

    "total_services": (
        "Total submitted service count summed across this provider’s rows in the scored universe. "
        "This is a scale/volume context metric (not an anomaly score by itself)."
    ),

    "total_benes": (
        "Total unique beneficiaries (or beneficiary counts) summed across this provider’s rows in the scored universe. "
        "Also a scale/volume context metric."
    ),

    "provider_anom_score_robust": (
        "Composite provider-level robust score used for ranking in the repeat-offender list. "
        "Definition (v1): n_anom_rows_robust + 0.25*n_unique_codes + 0.25*n_unique_years. "
        "Interpretation: rewards both repeat anomalies and breadth (more codes/years)."
    ),
}








ANOM3B1_COL_DICT = {
    "Rndrng_NPI": (
        "Provider identifier (NPI). Provider grain key."
    ),
    "provider_type": (
        "Provider specialty/type. Used for peer comparisons and rollups."
    ),
    "state": (
        "Provider state (2-letter). Provider grain key."
    ),

    "n_rows": (
        "Number of row-grain records for this provider in the scored universe. "
        "Each row corresponds to a (NPI, HCPCS, Place_Of_Srvc, Year) combination."
    ),

    "n_anom_rows_mag": (
        "Count of row-grain records flagged as 'shock' (magnitude-aware) anomalies under ANOM.3.b.1. "
        "This is the provider-level count of severe tail events, not the total row count."
    ),

    "anom_rate_pct_mag": (
        "Percent of this provider’s rows that are shock anomalies. "
        "Computed as 100 * (n_anom_rows_mag / n_rows). "
        "Example: 0.26 means ~0.26% of rows for this provider were shock events."
    ),

    "n_unique_codes": (
        "Breadth metric. Number of unique HCPCS codes present for this provider across all rows."
    ),

    "n_unique_years": (
        "Breadth metric. Number of distinct years present for this provider across all rows."
    ),

    "max_log_oe_mag": (
        "Maximum log(O/E) among THIS provider’s shock-flagged rows only. "
        "Interpretation: the provider’s single most extreme shock event in log space. "
        "Convert to an O/E multiplier via exp(log_oe). "
        "Example: max_log_oe_mag=3.75 implies O/E≈exp(3.75)≈42.6x for that row."
    ),

    "p95_log_oe_mag": (
        "95th percentile of log(O/E) among THIS provider’s shock-flagged rows only. "
        "Interpretation: a more robust severity statistic than max (less sensitive to one spike). "
        "Important nuance: if a provider has only 1 shock row, then p95_log_oe_mag == max_log_oe_mag."
    ),

    "median_log_oe": (
        "Provider-level median log(O/E) across ALL rows (not just shock rows). "
        "Interpretation: typical row behavior for the provider stays near 0 even if they have shocks."
    ),

    "total_services": (
        "Total service count summed across this provider’s rows in the scored universe. "
        "Scale context metric (not a severity measure by itself)."
    ),

    "total_benes": (
        "Total beneficiary counts summed across this provider’s rows in the scored universe. "
        "Scale context metric."
    ),

    "provider_anom_score_mag": (
        "Composite provider-level magnitude (shock) score used for ranking. "
        "Definition (v1): n_anom_rows_mag + 0.25*n_unique_codes + 0.25*n_unique_years + 0.10*p95_log_oe_mag. "
        "Interpretation: rewards shock frequency + breadth + severity."
    ),
}







TIER_ATYPICAL_COL_DICT = {
    "Rndrng_NPI": (
        "Provider identifier (NPI). Provider grain key."
    ),
    "provider_type": (
        "Provider specialty/type. Used for peer context and segmentation."
    ),
    "state": (
        "Provider state (2-letter). Provider grain key."
    ),
    "cluster_label_v1": (
        "Cluster assignment from tiering (eligible providers only). "
        "Values: cluster_0 (typical cost behavior) or cluster_1 (elevated anomaly burden). "
        "Ineligible providers have no cluster assignment and do not appear in this table."
    ),
    "cluster_definition_v1": (
        "Human-readable definition of the cluster label. "
        "These are descriptive labels, not quality ratings."
    ),
    "dist_to_centroid_v1": (
        "Euclidean distance from the provider’s point to its assigned cluster centroid, computed in the robust-scaled feature space used for clustering. "
        "Feature space used: median_log_oe_rob_rs, p90_log_oe_rs, n_anom_rows_robust_rs, anom_rate_pct_robust_rs, p95_log_oe_mag_rs, pct_high_conf_rows_rs. "
        "Interpretation: larger distance means the provider is less typical relative to its own cluster. "
        "Use this for within-cluster 'most atypical' surfacing, not as a severity score by itself."
    ),
    "n_anom_rows_robust": (
        "Count of provider rows flagged as robust extreme events (repeat-offender definition). "
        "This is the primary signal that separates cluster_1 from cluster_0."
    ),
    "anom_rate_pct_robust": (
        "Percent of this provider’s rows that are robust extreme events. "
        "Computed as 100 * (n_anom_rows_robust / n_rows_rob). "
        "Example: 5.13 means about 5.13% of the provider’s rows were flagged as robust events."
    ),
    "p90_log_oe": (
        "Provider-level 90th percentile of log(O/E) across that provider’s rows. "
        "Interpretation: how elevated the provider’s upper tail looks in general. "
        "Convert to an approximate O/E multiplier with exp(p90_log_oe). "
        "Example: p90_log_oe=0.50 implies O/E≈exp(0.50)≈1.65x at the provider’s 90th percentile."
    ),
    "total_services_rob": (
        "Total services aggregated for the provider in the tiering scorecard. "
        "Scale context metric. Bigger providers tend to have more 'opportunity' to accumulate events."
    ),
}






CLUSTER_PROFILE_TABLE_COL_DICT = {
    "cluster_label_v1": (
        "Cluster identifier assigned during tiering (KMeans on eligible providers). "
        "cluster_0 and cluster_1 are the two discovered groups. "
        "ineligible providers are not shown in this table."
    ),
    "cluster_definition_v1": (
        "Human-readable meaning of the cluster label. "
        "This is a descriptive summary, not a quality grade."
    ),

    # ----- log(O/E) features -----
    "median_log_oe_rob__median": (
        "Within this cluster, the median (across providers) of each provider’s median log(O/E) across its rows. "
        "Interpretation: a typical provider’s central tendency vs expected. "
        "log(O/E) near 0 means O/E near 1. exp(log_oe) gives the O/E multiplier."
    ),
    "median_log_oe_rob___p95": (
        "Within this cluster, the 95th percentile (across providers) of each provider’s median log(O/E). "
        "Interpretation: among providers in this cluster, a high-end (but not max) median over-expected level."
    ),

    "p90_log_oe__median": (
        "Within this cluster, the median (across providers) of each provider’s 90th percentile log(O/E) across its rows. "
        "Interpretation: for a typical provider in this cluster, how elevated their upper tail looks. "
        "exp(p90_log_oe) converts to an approximate upper-tail O/E multiplier."
    ),
    "p90_log_oe___p95": (
        "Within this cluster, the 95th percentile (across providers) of each provider’s p90_log_oe. "
        "Interpretation: the upper-end of upper-tail elevation among providers in this cluster."
    ),

    # ----- robust repeat-offender features -----
    "n_anom_rows_robust__median": (
        "Within this cluster, the median (across providers) count of robust anomalous rows per provider "
        "(ANOM.3.a.1 repeat-offender definition). "
        "Interpretation: how many robust tail events a typical provider has."
    ),
    "n_anom_rows_robust___p95": (
        "Within this cluster, the 95th percentile (across providers) of robust anomalous-row counts. "
        "Interpretation: high but not extreme repeat-offender burden inside the cluster."
    ),

    "anom_rate_pct_robust__median": (
        "Within this cluster, the median (across providers) robust anomaly rate per provider "
        "(100 × mean of the robust-event flag across that provider’s rows). "
        "Interpretation: typical provider’s share of rows that are robust anomalies."
    ),
    "anom_rate_pct_robust___p95": (
        "Within this cluster, the 95th percentile (across providers) of robust anomaly rates. "
        "Interpretation: high-end robust anomaly rate among providers in this cluster."
    ),

    # ----- shock severity feature (masked) -----
    "p95_log_oe_mag__median": (
        "Within this cluster, the median (across providers) of each provider’s p95 log(O/E) computed on shock-flagged rows only "
        "(ANOM.3.b.1 severity definition). "
        "Note: providers with zero shock events have this value set to 0 by construction."
    ),
    "p95_log_oe_mag___p95": (
        "Within this cluster, the 95th percentile (across providers) of provider-level p95_log_oe_mag. "
        "Interpretation: among providers in this cluster, how severe shock behavior can get (for those who have shock events). "
        "exp(value) converts to an O/E multiplier for shock intensity."
    ),

    # ----- confidence mix -----
    "pct_high_conf_rows__median": (
        "Within this cluster, the median (across providers) percentage of rows that meet the strict high-confidence rule "
        "(expected_cost_support_tier in {high, medium_high} AND services>=MIN_SERVICES AND benes>=MIN_BENES)."
    ),
    "pct_high_conf_rows___p95": (
        "Within this cluster, the 95th percentile (across providers) of pct_high_conf_rows. "
        "Interpretation: clusters with higher values concentrate more of their rows in strong-support regions."
    ),

    # ----- scale / exposure -----
    "total_services_rob__median": (
        "Within this cluster, the median (across providers) total services summed across the provider’s rows. "
        "Interpretation: typical service volume (exposure) for providers in the cluster."
    ),
    "total_services_rob___p95": (
        "Within this cluster, the 95th percentile (across providers) of total services. "
        "Interpretation: high-volume providers within this cluster."
    ),

    "total_benes_rob__median": (
        "Within this cluster, the median (across providers) total beneficiaries summed across the provider’s rows. "
        "Interpretation: typical beneficiary volume (exposure) in the cluster."
    ),
    "total_benes_rob___p95": (
        "Within this cluster, the 95th percentile (across providers) of total beneficiaries. "
        "Interpretation: high-bene providers within this cluster."
    ),

    "n_rows_rob__median": (
        "Within this cluster, the median (across providers) number of row-grain records per provider "
        "(rows are NPI × HCPCS × Place_Of_Srvc × Year). "
        "Interpretation: typical evidence depth behind each provider’s metrics."
    ),
    "n_rows_rob___p95": (
        "Within this cluster, the 95th percentile (across providers) of row counts. "
        "Interpretation: providers with broad code/year coverage inside the cluster."
    ),
}












EXPLAINER_ARTIFACT_COL_DICT = {
    "item": "Name of the frozen artifact or config file being referenced. These are provenance pointers, not model outputs.",
    "value": "Resolved filesystem path to the frozen artifact used by the Explainer Model tab.",
}

EXPLAINER_CLASS_REPORT_COL_DICT = {
    "label": "Target class being evaluated. cluster_0 = typical cost behavior; cluster_1 = elevated anomaly burden.",
    "precision": "Of all rows predicted to be this class, the share that were correct.",
    "recall": "Of all true rows in this class, the share correctly recovered by the explainer model.",
    "f1-score": "Harmonic mean of precision and recall. Useful when balancing false positives and false negatives.",
    "support": "Number of test-set rows belonging to this class.",
}

EXPLAINER_DRIVER_COL_DICT = {
    "feature": "Model input feature or one-hot encoded category used by the frozen logistic explainer.",
    "coef_log_odds": "Log-odds coefficient from logistic regression. Positive values push predictions toward cluster_1; negative values push away.",
    "odds_ratio": "exp(coef_log_odds). Multiplicative change in odds associated with a one-unit increase in the feature, holding others fixed. For scaled numeric features, this is per +1 SD.",
}

EXPLAINER_BASELINE_COL_DICT = {
    "baseline": "Reference category dropped by OneHotEncoder for each categorical variable. All categorical coefficients are interpreted relative to these baselines.",
}

















PP_TIERING_ANOM_COL_DICT = {
    "n_rows_rob": (
        "Provider-level row count in the tiering scorecard. "
        "Rows are the model’s row grain: NPI × HCPCS × Place_Of_Srvc × Year. "
        "Interpretation: evidence depth behind this provider’s metrics."
    ),
    "total_services_rob": (
        "Total services aggregated for this provider across the row-grain records used in tiering. "
        "Exposure/scale context. Larger providers have more opportunity to accumulate events."
    ),
    "total_benes_rob": (
        "Total beneficiaries aggregated for this provider across the row-grain records used in tiering. "
        "Exposure/scale context, complementary to total services."
    ),
    "pct_high_conf_rows": (
        "Percent of this provider’s rows that meet the strict high-confidence gate "
        "(strong support tier plus minimum evidence rules). "
        "Higher means more of the provider’s data sits in strong-support regions."
    ),
    "n_anom_rows_robust": (
        "Count of this provider’s row-grain records flagged as robust extreme events "
        "(repeat-offender definition, aligned with ANOM.3.a.1). "
        "Primary signal separating cluster_1 from cluster_0."
    ),
    "anom_rate_pct_robust": (
        "Percent of this provider’s rows flagged as robust events. "
        "Computed as 100 × (n_anom_rows_robust / n_rows_rob). "
        "Example: 0.69 means ~0.69% of rows were robust anomalies."
    ),
    "p90_log_oe": (
        "Provider-level 90th percentile of log(O/E) across rows. "
        "Interpretation: how elevated the provider’s upper tail looks. "
        "exp(p90_log_oe) approximates an upper-tail O/E multiplier."
    ),
    "n_anom_rows_mag": (
        "Count of this provider’s row-grain records flagged as magnitude (shock) events "
        "(severity-first definition, aligned with ANOM.3.b.1)."
    ),
    "p95_log_oe_mag": (
        "Provider-level p95 log(O/E) on shock-flagged rows only. "
        "If the provider has zero shock events, this is 0 by construction."
    ),
    "cluster_label_v1": (
        "Cluster assignment from tiering (eligible providers only). "
        "Values: cluster_0 (typical) or cluster_1 (elevated anomaly burden)."
    ),
    "cluster_definition_v1": (
        "Human-readable meaning of the cluster label. Descriptive, not a quality grade."
    ),
    "dist_to_centroid_v1": (
        "Euclidean distance from the provider to its assigned cluster centroid, computed in the scaled feature space used for clustering. "
        "Interpretation: larger distance means the provider is less typical relative to others in the same cluster. "
        "Use for within-cluster ‘most atypical’ surfacing, not as a severity score by itself. "
        "Scaled clustering features: median_log_oe_rob_rs, p90_log_oe_rs, n_anom_rows_robust_rs, anom_rate_pct_robust_rs, "
        "p95_log_oe_mag_rs, pct_high_conf_rows_rs."
    ),
}








PP_EXPLAINER_CONTRIB_COL_DICT = {
    "feature": (
        "Feature name in the explainer pipeline’s transformed space. "
        "Prefix meanings: "
        "`num__` = numeric feature after StandardScaler; "
        "`cat__` = one-hot encoded category (if present). "
        "Example: `num__log_total_services_base` is the standardized service-volume feature."
    ),
    "value_xformed": (
        "The transformed feature value that the logistic model actually sees. "
        "For numeric features, this is the standardized value (z-score): "
        "(raw_value - train_mean) / train_std. "
        "Interpretation: +1 means 1 standard deviation above average in the training set."
    ),
    "coef_log_odds": (
        "Logistic regression coefficient for this transformed feature. "
        "It is a change in log-odds per +1 unit of the transformed feature. "
        "For standardized numeric features, it is per +1 SD."
    ),
    "contrib_log_odds": (
        "Per-feature contribution to the provider’s log-odds under the linear model: "
        "`contrib_log_odds = value_xformed × coef_log_odds`. "
        "Interpretation: positive pushes toward cluster_1, negative pushes toward cluster_0. "
        "This is an explanation aid, not a causal statement."
    ),
}






PP_ANOM_ROWS_COL_DICT = {
    "source": (
        "Which frozen anomaly worklist this row came from. "
        "`primary` = robust + size-aware (ANOM.2.a.1). "
        "`alternative` = magnitude-aware + size-aware (ANOM.2.b.1)."
    ),
    "row_id": (
        "Unique identifier for the row-grain record in `eval_scored_DG_V3.parquet`. "
        "This lets you trace the exact row back to the single source of truth."
    ),
    "HCPCS_Cd": (
        "HCPCS procedure/drug/supply code for this row."
    ),
    "Place_Of_Srvc": (
        "Medicare place of service code for this row (site-of-care context). "
        "Example: `O` often indicates an outpatient setting in your dataset."
    ),
    "Year": (
        "Calendar year for this row."
    ),
    "observed_cost": (
        "Observed allowed/standardized cost for the row (what was billed/paid, depending on your definition)."
    ),
    "expected_cost": (
        "Model-predicted expected cost for the same row, from the D/G+V3 expected-cost model."
    ),
    "residual": (
        "Cost deviation vs model expectation. "
        "`residual = observed_cost - expected_cost`. "
        "Positive residual means higher-than-expected cost."
    ),
    "log_oe": (
        "Log over-expected ratio. "
        "`log_oe = log(observed_cost / expected_cost)` when expected_cost > 0. "
        "0 means exactly expected. Positive means over-expected."
    ),
    "oe_ratio": (
        "Over-expected ratio. "
        "`oe_ratio = observed_cost / expected_cost`. "
        "Example: oe_ratio=1.27 means ~27% above expected."
    ),
    "slice_n": (
        "Peer-group size used to compute within-slice percentiles (the slice is typically HCPCS × Year). "
        "Larger slice_n means more stable percentile comparisons."
    ),
    "log_oe_pct_in_slice": (
        "Percentile rank of this row’s log_oe within its peer slice. "
        "Example: 1.0 means it is at (or extremely near) the top of the slice distribution."
    ),
}















def _fmt_int(x) -> str:
    try:
        return f"{int(x):,}"
    except Exception:
        return str(x)

def _fmt_pct(x) -> str:
    try:
        return f"{float(x):.2f}%"
    except Exception:
        return str(x)

def _kpi_card(title: str, value: ui.Tag, subtitle: str | None = None):
    return ui.card(
        ui.card_header(ui.h6(title)),
        ui.h3(value),
        ui.p(subtitle) if subtitle else None,
        class_="kpi-card",
    )

def _read_eval_filtered(
    year: int | None,
    provider_type: str | None,
    state: str | None,
    hcpcs: str | None,
    pos: str | None,
    limit: int = 250_000,
) -> pd.DataFrame:
    """
    Lazy load a filtered slice from eval_scored_DG_V3.
    For now: read parquet then filter, with a row cap.
    Later: swap to DuckDB for true query-on-demand.
    """
    df = pd.read_parquet(
        EVAL_SCORED_PATH,
        columns=[
            "row_id","Rndrng_NPI","provider_type","state","HCPCS_Cd","Place_Of_Srvc","Year",
            "observed_cost","expected_cost","residual","log_oe","oe_ratio","services","benes",
            "high_confidence_anomaly_candidate"
        ],
    )
    if year is not None:
        df = df[df["Year"] == year]
    if provider_type and provider_type != "All":
        df = df[df["provider_type"].astype(str) == provider_type]
    if state and state != "All":
        df = df[df["state"].astype(str) == state]
    if hcpcs and hcpcs != "All":
        df = df[df["HCPCS_Cd"].astype(str) == hcpcs]
    if pos and pos != "All":
        df = df[df["Place_Of_Srvc"].astype(str) == pos]

    if len(df) > limit:
        df = df.sample(n=limit, random_state=7)
    return df

def _selected_row_from_grid(input_obj, grid_id: str, df: pd.DataFrame) -> pd.Series | None:
    """
    Shiny DataGrid selection input naming can vary by version.
    This helper tries common patterns.
    """
    candidates = [
        f"{grid_id}_selected_rows",
        f"{grid_id}_rows_selected",
        f"{grid_id}_selected",
    ]
    sel = None
    for key in candidates:
        try:
            if hasattr(input_obj, key):
                sel = getattr(input_obj, key)()
                break
        except Exception:
            continue

    if sel is None:
        return None

    try:
        if isinstance(sel, (list, tuple)) and len(sel) > 0:
            i = int(sel[0])
            if 0 <= i < len(df):
                return df.iloc[i]
    except Exception:
        return None
    return None

def _rules_text_for(which: str) -> str:
    """
    which in {"row_primary","row_alt","prov_primary","prov_alt"}
    If your run_params.json has different nesting, we fall back to dumping whole file.
    """
    if not anom_params:
        return "No params.json loaded."

    # Best-effort extraction
    try:
        row = anom_params.get("row_level", {})
        prov = anom_params.get("provider_level", {})
        if which == "row_primary":
            return json.dumps(row.get("ANOM_2_a_1", {}), indent=2)
        if which == "row_alt":
            return json.dumps(row.get("ANOM_2_b_1", {}), indent=2)
        if which == "prov_primary":
            return json.dumps(prov.get("ANOM_3_a_1", {}), indent=2)
        if which == "prov_alt":
            return json.dumps(prov.get("ANOM_3_b_1", {}), indent=2)
    except Exception:
        pass

    return json.dumps(anom_params, indent=2)


def _provider_covariates_from_eval(npi: str) -> dict:
    """
    Build no-leak covariates used in classification, for ONE provider on demand.
    Weighted means use benes. ruca_bucket uses mode.
    """
    cols = [
        "Rndrng_NPI","provider_type","state",
        "p_cancer6","p_diabetes","p_ckd","p_copd","p_htn",
        "bene_avg_risk_score","years_since_enumeration",
        "ruca_bucket","benes","services"
    ]
    df = pd.read_parquet(EVAL_SCORED_PATH, columns=cols)
    df = df[df["Rndrng_NPI"].astype(str) == str(npi)]
    if len(df) == 0:
        return {}

    num_cols = ["p_cancer6","p_diabetes","p_ckd","p_copd","p_htn","bene_avg_risk_score","years_since_enumeration","benes","services"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    def _wmean(x: pd.Series, w: pd.Series) -> float:
        x = pd.to_numeric(x, errors="coerce")
        w = pd.to_numeric(w, errors="coerce").fillna(0)
        mask = x.notna() & np.isfinite(x) & np.isfinite(w)
        if mask.sum() == 0:
            return float("nan")
        x = x[mask]
        w = w[mask]
        sw = float(w.sum())
        if sw <= 0:
            return float(x.mean())
        return float((x * w).sum() / sw)

    def _mode_str(x: pd.Series) -> str:
        x = x.astype(str)
        vc = x.value_counts(dropna=False)
        return str(vc.index[0]) if len(vc) else "unknown"

    out = {
        "p_cancer6": _wmean(df["p_cancer6"], df["benes"]),
        "p_diabetes": _wmean(df["p_diabetes"], df["benes"]),
        "p_ckd": _wmean(df["p_ckd"], df["benes"]),
        "p_copd": _wmean(df["p_copd"], df["benes"]),
        "p_htn": _wmean(df["p_htn"], df["benes"]),
        "bene_avg_risk_score": _wmean(df["bene_avg_risk_score"], df["benes"]),
        "years_since_enumeration": float(np.nanmedian(df["years_since_enumeration"].to_numpy())),
        "ruca_bucket": _mode_str(df["ruca_bucket"]),
        "log_total_services_base": float(np.log1p(np.nansum(df["services"].to_numpy()))),
        "provider_type": str(df["provider_type"].iloc[0]),
        "state": str(df["state"].iloc[0]),
    }
    return out


def _explainer_predict_and_contrib(npi: str) -> tuple[float | None, pd.DataFrame]:
    """
    Uses the frozen logistic pipeline (joblib) if available.
    Returns:
      - predicted probability of cluster_1 (None if unavailable)
      - contribution table (top |coef*value| in transformed feature space)
    """
    if clf_pipe is None:
        return None, pd.DataFrame()

    cov = _provider_covariates_from_eval(npi)
    if not cov:
        return None, pd.DataFrame()

    Xrow = pd.DataFrame([{
        "p_cancer6": cov.get("p_cancer6"),
        "p_diabetes": cov.get("p_diabetes"),
        "p_ckd": cov.get("p_ckd"),
        "p_copd": cov.get("p_copd"),
        "p_htn": cov.get("p_htn"),
        "bene_avg_risk_score": cov.get("bene_avg_risk_score"),
        "years_since_enumeration": cov.get("years_since_enumeration"),
        "log_total_services_base": cov.get("log_total_services_base"),
        "ruca_bucket": cov.get("ruca_bucket", "unknown"),
        "provider_type": cov.get("provider_type", "unknown"),
    }])

    try:
        p = float(clf_pipe.predict_proba(Xrow)[:, 1][0])
    except Exception:
        p = None

    try:
        preprocess = clf_pipe.named_steps["preprocess"]
        model = clf_pipe.named_steps["model"]
        Xtr = preprocess.transform(Xrow)
        coefs = model.coef_.ravel()

        x_vec = Xtr.toarray().ravel() if hasattr(Xtr, "toarray") else np.asarray(Xtr).ravel()
        contrib = x_vec * coefs

        feat_names = list(preprocess.get_feature_names_out())
        dfc = pd.DataFrame({
            "feature": feat_names,
            "value_xformed": x_vec,
            "coef_log_odds": coefs,
            "contrib_log_odds": contrib,
        })
        dfc["abs_contrib"] = dfc["contrib_log_odds"].abs()
        dfc = dfc.sort_values("abs_contrib", ascending=False).head(15).drop(columns="abs_contrib")
    except Exception:
        dfc = pd.DataFrame()

    return p, dfc













def _build_explainer_tab_diagnostics() -> dict:
    """
    Rebuilds the classification diagnostics shown in the slide:
      - classification report table
      - confusion matrix
      - ROC curve
      - PR curve
      - baselines used by OHE
      - top positive / negative coefficient tables

    Uses:
      - latest frozen provider scorecard
      - latest eval_scored_DG_V3 parquet
      - frozen final logistic explainer pipeline
    """
    out = {
        "artifact_df": pd.DataFrame([
            {"item": "model_joblib", "value": str(FINAL_EXPLAINER_JOBLIB)},
            {"item": "classification_params_json", "value": str(CLF_PARAMS_JSON)},
        ]),
        "report_df": pd.DataFrame(),
        "cm": np.array([[0, 0], [0, 0]], dtype=int),
        "roc_auc": None,
        "pr_auc": None,
        "fpr": np.array([]),
        "tpr": np.array([]),
        "precision_curve": np.array([]),
        "recall_curve": np.array([]),
        "baseline_text": "Baselines unavailable",
        "coef_pos": pd.DataFrame(),
        "coef_neg": pd.DataFrame(),
        "note": "Explainer diagnostics unavailable.",
    }

    if clf_pipe is None:
        out["note"] = "No frozen explainer model could be loaded (joblib)."
        return out

    TARGET = "is_cluster_1"
    NUM_FEATS = [
        "p_cancer6", "p_diabetes", "p_ckd", "p_copd", "p_htn",
        "bene_avg_risk_score", "years_since_enumeration",
        "log_total_services_base",
    ]
    CAT_FEATS = ["ruca_bucket", "provider_type"]

    eligible = scorecard_all.copy()
    eligible = eligible[
        eligible["cluster_label_v1"].astype(str).isin(["cluster_0", "cluster_1"])
    ].copy()

    if eligible.empty:
        out["note"] = "No eligible providers found in provider_scorecard."
        return out

    keep_cols = [
        "Rndrng_NPI", "provider_type", "state", "cluster_label_v1", "cluster_definition_v1"
    ]
    eligible = eligible[keep_cols].drop_duplicates()

    eval_cols = [
        "Rndrng_NPI", "provider_type", "state",
        "p_cancer6", "p_diabetes", "p_ckd", "p_copd", "p_htn",
        "bene_avg_risk_score", "years_since_enumeration",
        "ruca_bucket", "benes", "services",
    ]
    ev = pd.read_parquet(EVAL_SCORED_PATH, columns=eval_cols).copy()

    # Normalize join keys
    for c in ["Rndrng_NPI", "provider_type", "state"]:
        eligible[c] = eligible[c].astype(str)
        ev[c] = ev[c].astype(str)

    ev = ev.merge(
        eligible[["Rndrng_NPI", "provider_type", "state"]],
        on=["Rndrng_NPI", "provider_type", "state"],
        how="inner",
    )

    if ev.empty:
        out["note"] = "No matching provider covariates found in eval_scored_DG_V3."
        return out

    num_cols = [
        "p_cancer6", "p_diabetes", "p_ckd", "p_copd", "p_htn",
        "bene_avg_risk_score", "years_since_enumeration", "benes", "services",
    ]
    for c in num_cols:
        ev[c] = pd.to_numeric(ev[c], errors="coerce")

    def _wmean(x: pd.Series, w: pd.Series) -> float:
        x = pd.to_numeric(x, errors="coerce")
        w = pd.to_numeric(w, errors="coerce").fillna(0)
        mask = x.notna() & np.isfinite(x) & np.isfinite(w)
        if mask.sum() == 0:
            return float("nan")
        x = x[mask]
        w = w[mask]
        sw = float(w.sum())
        if sw <= 0:
            return float(x.mean())
        return float((x * w).sum() / sw)

    def _mode_str(x: pd.Series) -> str:
        x = x.dropna().astype(str)
        if len(x) == 0:
            return "Unknown"
        mode = x.mode(dropna=True)
        return str(mode.iloc[0]) if len(mode) else "Unknown"

    rows = []
    for (npi, ptype, state), g in ev.groupby(["Rndrng_NPI", "provider_type", "state"], dropna=False):
        rows.append({
            "Rndrng_NPI": str(npi),
            "provider_type": str(ptype),
            "state": str(state),
            "p_cancer6": _wmean(g["p_cancer6"], g["benes"]),
            "p_diabetes": _wmean(g["p_diabetes"], g["benes"]),
            "p_ckd": _wmean(g["p_ckd"], g["benes"]),
            "p_copd": _wmean(g["p_copd"], g["benes"]),
            "p_htn": _wmean(g["p_htn"], g["benes"]),
            "bene_avg_risk_score": _wmean(g["bene_avg_risk_score"], g["benes"]),
            "years_since_enumeration": (
                float(pd.to_numeric(g["years_since_enumeration"], errors="coerce").dropna().median())
                if pd.to_numeric(g["years_since_enumeration"], errors="coerce").dropna().shape[0] > 0
                else np.nan
            ),
            "ruca_bucket": _mode_str(g["ruca_bucket"]),
            "log_total_services_base": float(np.log1p(np.nansum(pd.to_numeric(g["services"], errors="coerce").to_numpy()))),
        })

    cov_df = pd.DataFrame(rows)

    clf_df_app = eligible.merge(
        cov_df,
        on=["Rndrng_NPI", "provider_type", "state"],
        how="inner",
    ).copy()

    if clf_df_app.empty:
        out["note"] = "Could not assemble explainer diagnostics dataset."
        return out






    clf_df_app[TARGET] = (clf_df_app["cluster_label_v1"].astype(str) == "cluster_1").astype(int)

    # Defensive cleanup: this frozen logistic pipeline does NOT have an imputer
    clf_df_app = clf_df_app.copy()
    clf_df_app[NUM_FEATS] = clf_df_app[NUM_FEATS].apply(pd.to_numeric, errors="coerce")
    clf_df_app = clf_df_app.replace([np.inf, -np.inf], np.nan)

    model_cols = NUM_FEATS + CAT_FEATS + [TARGET]
    n_before = len(clf_df_app)
    clf_df_app = clf_df_app.dropna(subset=model_cols).copy()
    n_after = len(clf_df_app)

    if clf_df_app.empty:
        out["note"] = "Explainer diagnostics unavailable after dropping rows with missing model inputs."
        return out

    # Need both classes present after cleanup
    class_counts = clf_df_app[TARGET].value_counts(dropna=False)
    if set(class_counts.index.tolist()) != {0, 1}:
        out["note"] = "Explainer diagnostics unavailable because both classes are not present after input cleanup."
        return out

    X = clf_df_app[NUM_FEATS + CAT_FEATS].copy()
    y = clf_df_app[TARGET].astype(int).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=7, stratify=y
    )

    p_test = clf_pipe.predict_proba(X_test)[:, 1]






    pred_test = (p_test >= 0.50).astype(int)

    roc = float(roc_auc_score(y_test, p_test))
    pr = float(average_precision_score(y_test, p_test))
    cm = confusion_matrix(y_test, pred_test)

    rep = classification_report(
        y_test, pred_test, digits=3, output_dict=True, zero_division=0
    )
    report_df = (
        pd.DataFrame(rep)
        .T
        .reset_index()
        .rename(columns={"index": "label"})
    )
    report_df = report_df[report_df["label"].isin(["0", "1"])].copy()
    report_df["label"] = report_df["label"].map({"0": "cluster_0", "1": "cluster_1"})
    report_df = report_df[["label", "precision", "recall", "f1-score", "support"]]

    fpr, tpr, _ = roc_curve(y_test, p_test)
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, p_test)

    preprocess = clf_pipe.named_steps["preprocess"]
    model = clf_pipe.named_steps["model"]
    ohe = preprocess.named_transformers_["cat"]

    baselines = []
    for feat, cats, dropped_idx in zip(CAT_FEATS, ohe.categories_, ohe.drop_idx_):
        base = cats[dropped_idx] if dropped_idx is not None else None
        baselines.append(f"{feat}: {base}")
    baseline_text = " | ".join(baselines)

    cat_names = list(ohe.get_feature_names_out(CAT_FEATS))
    feat_names = NUM_FEATS + cat_names
    coefs = model.coef_.ravel()

    coef_df = (
        pd.DataFrame({"feature": feat_names, "coef_log_odds": coefs})
        .assign(odds_ratio=lambda d: np.exp(d["coef_log_odds"]))
        .sort_values("coef_log_odds", ascending=False)
    )

    coef_pos = coef_df.head(15).copy()
    coef_neg = coef_df.tail(15).iloc[::-1].copy()

    out.update({
        "report_df": report_df,
        "cm": cm,
        "roc_auc": roc,
        "pr_auc": pr,
        "fpr": fpr,
        "tpr": tpr,
        "precision_curve": precision_curve,
        "recall_curve": recall_curve,
        "baseline_text": baseline_text,
        "coef_pos": coef_pos,
        "coef_neg": coef_neg,
        "note": f"Explainer model loaded from frozen artifact. Classification diagnostics below are rebuilt from the frozen model and latest eligible provider data. Rows used after input cleanup: {n_after:,} of {n_before:,}.",
    })
    return out


EXPLAINER_TAB_DIAG = _build_explainer_tab_diagnostics()

















def _apply_filters_scorecard(df: pd.DataFrame, provider_type: str, state: str, eligible_only: bool) -> pd.DataFrame:
    out = df.copy()
    if provider_type and provider_type != "All":
        out = out[out["provider_type"].astype(str) == provider_type]
    if state and state != "All":
        out = out[out["state"].astype(str) == state]
    if eligible_only:
        out = out[out["cluster_label_v1"].astype(str).str.startswith("cluster_")]
    return out


# ============================================================
# 3) UI (narrative-first)
# ============================================================

def _choices_from(series: pd.Series) -> list[str]:
    return ["All"] + sorted(series.dropna().astype(str).unique().tolist())

PROVIDER_TYPES = _choices_from(scorecard_all["provider_type"].astype(str))
STATES = _choices_from(scorecard_all["state"].astype(str))

app_ui = ui.page_fluid(
    ui.tags.style("""
    .kpi-card { min-height: 120px; }
    .muted { color: #666; }
    .btn-row { margin-right: 8px; }
    """),

    ui.h2("Medicare Provider Benchmarking Engine"),
    ui.p(
        "A reproducible workflow for expected-cost benchmarking, anomaly surfacing, provider tiering, and explainability.",
        class_="muted",
    ),

    ui.navset_pill(
        # ----------------------------------------------------
        # 1) OVERVIEW
        # ----------------------------------------------------
        ui.nav_panel(
            "Overview",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.h5("Narrative filters"),
                    ui.input_select("ov_year", "Year (row-grain plots)", choices=["All"] + [str(y) for y in YEARS], selected="All"),
                    ui.input_select("ov_provider_type", "Provider type", choices=PROVIDER_TYPES, selected="All"),
                    ui.input_select("ov_state", "State", choices=STATES, selected="All"),
                    ui.input_checkbox("ov_eligible_only", "Eligible only (clustered providers)", False),
                    ui.hr(),
                    ui.p("Counts update from frozen provider-tiering artifacts.", class_="muted"),
                    width=320,
                ),
                ui.layout_columns(
                    _kpi_card("Providers (total)", ui.output_text("kpi_total_providers")),
                    _kpi_card("Eligible (clustered)", ui.output_text("kpi_eligible_providers")),
                    _kpi_card("Cluster 0", ui.output_text("kpi_c0")),
                    _kpi_card("Cluster 1", ui.output_text("kpi_c1")),
                    col_widths=(3, 3, 3, 3),
                ),
                ui.layout_columns(
                    _kpi_card("Robust events (providers)", ui.output_text("kpi_robust_prov")),
                    _kpi_card("Shock events (providers)", ui.output_text("kpi_shock_prov")),
                    _kpi_card("Watchlist size", ui.output_text("kpi_watchlist")),
                    _kpi_card("Cluster_1 prevalence", ui.output_text("kpi_prev")),
                    col_widths=(3, 3, 3, 3),
                ),
                ui.hr(),

                # Two descriptive plots side-by-side
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Distribution: log(O/E) on provider rows (filtered sample)"),
                        ui.output_plot("plot_overview_logoe"),
                        ui.p("Row-grain plot. Filtered. Sampled if needed for responsiveness.", class_="muted"),
                    ),
                    ui.card(
                        ui.card_header("Cluster mix (providers) under current filters"),
                        ui.output_plot("plot_overview_cluster_mix"),
                        ui.p("Provider-grain from frozen tiering scorecard (fast, exact).", class_="muted"),
                    ),
                    col_widths=(7, 5),
                ),

                ui.hr(),

                # Slice summary table
                ui.card(
                    ui.card_header("Selected slice summary (fast, defensible)"),
                    ui.output_data_frame("tbl_overview_slice_summary"),
                    ui.output_ui("tbl_overview_slice_summary_dict"),
                    ui.p("Provider counts come from the frozen provider scorecard. Row-level stats come from eval_base and frozen anomaly surfaces.", class_="muted"),
                ),

                ui.hr(),

                # Cluster label card moved here (keep as-is)
                ui.card(
                    ui.card_header("Cluster label card"),
                    ui.output_data_frame("tbl_cluster_card"),
                    ui.output_ui("tbl_cluster_card_dict"),
                    ui.p("One-slide definition of the segmentation.", class_="muted"),
                ),
            ),
        ),

        # ----------------------------------------------------
        # 2) BENCHMARK EXPLORER
        # ----------------------------------------------------
        ui.nav_panel(
            "Benchmark Explorer",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.h5("Pick a view"),
                    ui.input_radio_buttons(
                        "bx_mode",
                        "Explorer mode",
                        choices=["Slice explorer (HCPCS x Year)", "Provider explorer (NPI)"],
                        selected="Slice explorer (HCPCS x Year)",
                    ),
                    ui.hr(),
                    ui.h6("Slice explorer inputs"),
                    ui.input_text("bx_hcpcs", "HCPCS (exact)", value=""),
                    ui.input_numeric("bx_year", "Year", value=YEARS[-1] if YEARS else 2022, min=2000, max=2100),
                    # NEW: optional refinements for slice view
                    ui.input_select("bx_pos", "Place of service (optional)", choices=["All"] + POS_CHOICES, selected="All"),
                    ui.input_select("bx_state", "State (optional)", choices=["All"] + STATE_CHOICES, selected="All"),
                    ui.input_text("bx_npi_slice", "NPI (optional)", value=""),
                    ui.hr(),
                    ui.h6("Provider explorer inputs"),
                    ui.input_text("bx_npi", "NPI (exact)", value=""),
                    ui.input_checkbox("bx_log_scale", "Log scale for observed/expected", True),
                    width=320,
                ),
                ui.card(
                    ui.card_header(ui.output_text("bx_title")),
                    ui.output_ui("bx_grain_note"),
                    ui.layout_columns(
                        ui.card(ui.card_header("Observed vs Expected"), ui.output_plot("bx_scatter_obs_exp")),
                        ui.card(ui.card_header("Residual distribution"), ui.output_plot("bx_resid_hist")),
                        col_widths=(6, 6),
                    ),
                    ui.hr(),
                    ui.card(
                        ui.card_header("Top over-expected rows in this view"),
                        ui.output_data_frame("bx_top_rows"),
                        ui.output_ui("bx_top_rows_dict"),
                        ui.p("Trust builder. Shows signal at the same row grain used downstream.", class_="muted"),
                    ),
                ),
            ),
        ),

        # ----------------------------------------------------
        # 3) ANOMALY SURFACING
        # ----------------------------------------------------
        ui.nav_panel(
            "Anomaly Surfacing",
            ui.navset_tab(
                ui.nav_panel(
                    "Row-level",
                    ui.p("Two complementary row-level worklists, frozen from your anomaly surfacing run.", class_="muted"),
                    ui.div(
                        ui.input_action_button("btn_row_primary_detail", "Show selected Primary row details", class_="btn-row"),
                        ui.input_action_button("btn_row_alt_detail", "Show selected Alt row details", class_="btn-row"),
                    ),




                    # Stack the two row-level worklists vertically (Primary above Alternative)
                    ui.card(
                        ui.card_header("Primary (ANOM.2.a.1) robust + size-aware"),
                        ui.output_data_frame("tbl_rows_primary"),
                        ui.output_ui("tbl_rows_primary_dict"),
                    ),
                    ui.hr(),
                    ui.card(
                        ui.card_header("Alternative (ANOM.2.b.1) magnitude-aware + size-aware"),
                        ui.output_data_frame("tbl_rows_alt"),
                        ui.output_ui("tbl_rows_alt_dict"),
                    ),




                    ui.hr(),
                    ui.card(
                        ui.card_header("Rules and parameters (from freeze params.json)"),
                        ui.output_ui("rules_anom"),
                    ),
                ),
                ui.nav_panel(
                    "Provider-level",
                    ui.p("Two complementary provider-level worklists, frozen from anomaly surfacing.", class_="muted"),
                    ui.div(
                        ui.input_action_button("btn_prov_primary_detail", "Show selected Repeat-offender provider details", class_="btn-row"),
                        ui.input_action_button("btn_prov_alt_detail", "Show selected Shock provider details", class_="btn-row"),
                    ),




                    # Stack the two provider-level worklists vertically (Repeat offenders above Shock)
                    ui.card(
                        ui.card_header("Repeat offenders (ANOM.3.a.1)"),
                        ui.output_data_frame("tbl_prov_primary"),
                        ui.output_ui("tbl_prov_primary_dict"),
                    ),
                    ui.hr(),
                    ui.card(
                        ui.card_header("Shock providers (ANOM.3.b.1)"),
                        ui.output_data_frame("tbl_prov_alt"),
                        ui.output_ui("tbl_prov_alt_dict"),
                    ),




                ),
            ),
        ),

        # ----------------------------------------------------
        # 4) TIERING
        # ----------------------------------------------------
        ui.nav_panel(
            "Tiering",
            ui.navset_tab(
                ui.nav_panel(
                    "Eligibility + clusters",
                    ui.p("Tiering applies only to eligible providers (minimum evidence gate).", class_="muted"),
                    ui.layout_columns(
                        ui.card(
                            ui.card_header("Eligibility gate summary"),
                            ui.output_ui("tier_gate_note"),
                            ui.output_plot("tier_gate_hist"),
                        ),
                        ui.card(
                            ui.card_header("Cluster overlap proof"),
                            ui.output_plot("tier_ecdf_n_anom"),
                        ),
                        col_widths=(6, 6),
                    ),
                    ui.hr(),
                    ui.card(
                        ui.card_header("Most atypical providers (largest distance-to-centroid)"),
                        ui.output_data_frame("tbl_atypical"),
                        ui.output_ui("tbl_atypical_dict"),
                    ),
                ),
                ui.nav_panel(
                    "Cluster profiling",
                    ui.p("Cluster personalities based on raw features used to interpret clusters.", class_="muted"),
                    ui.output_data_frame("tbl_cluster_profile_compact"),
                    ui.hr(),
                    ui.output_ui("tbl_cluster_profile_compact_dict"),
                ),
            ),
        ),











        # ----------------------------------------------------
        # 5) EXPLAINER MODEL
        # ----------------------------------------------------
        ui.nav_panel(
            "Explainer Model",
            ui.p(
                "Interpretability module. These diagnostics summarize how the frozen logistic explainer separates cluster_1 from cluster_0 among eligible providers.",
                class_="muted",
            ),

            ui.card(
                ui.card_header("Explainer artifact and config"),
                ui.output_ui("explainer_note"),
                ui.output_data_frame("tbl_explainer_artifact"),
                ui.output_ui("tbl_explainer_artifact_dict"),
            ),

            ui.hr(),

            ui.card(
                ui.card_header("Classification report"),
                ui.output_data_frame("tbl_explainer_classification_report"),
                ui.output_ui("tbl_explainer_classification_report_dict"),
            ),

            ui.hr(),

            ui.layout_columns(
                ui.card(
                    ui.card_header("Confusion Matrix @ 0.50 Threshold"),
                    ui.output_plot("plt_explainer_cm"),
                ),
                ui.card(
                    ui.card_header("ROC Curve"),
                    ui.output_plot("plt_explainer_roc"),
                ),
                ui.card(
                    ui.card_header("Precision-Recall Curve"),
                    ui.output_plot("plt_explainer_pr"),
                ),
                col_widths=[4, 4, 4],
            ),

            ui.hr(),

            ui.card(
                ui.card_header("Baselines used by the explainer"),
                ui.output_ui("txt_explainer_baselines"),
                ui.output_ui("tbl_explainer_baselines_dict"),
            ),

            ui.hr(),

            ui.layout_columns(
                ui.card(
                    ui.card_header("Top positive drivers (higher => more likely cluster_1)"),
                    ui.output_data_frame("tbl_explainer_pos_drivers"),
                    ui.output_ui("tbl_explainer_pos_driver_dict"),
                ),
                ui.card(
                    ui.card_header("Top negative drivers (lower => less likely cluster_1)"),
                    ui.output_data_frame("tbl_explainer_neg_drivers"),
                    ui.output_ui("tbl_explainer_neg_driver_dict"),
                ),
                col_widths=[6, 6],
            ),
        ),












        # ----------------------------------------------------
        # 5.5) PROVIDER PROFILE (stitched narrative)
        # ----------------------------------------------------
        ui.nav_panel(
            "Provider Profile",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.h5("Provider lookup"),
                    ui.input_text("pp_npi", "NPI (exact)", value=""),
                    ui.input_action_button("pp_refresh", "Refresh profile"),
                    ui.hr(),
                    ui.p("Stitches anomalies, tiering, and explainer outputs.", class_="muted"),
                    width=320,
                ),
                ui.card(
                    ui.card_header("Provider summary"),
                    ui.output_ui("pp_header"),





                    # Stack Tiering+anomalies above Explainer model (instead of side-by-side)
                    ui.card(
                        ui.card_header("Tiering + anomalies"),
                        ui.output_data_frame("pp_metrics"),
                        ui.output_ui("pp_metrics_dict"),
                    ),
                    ui.hr(),
                    ui.card(
                        ui.card_header("Explainer model"),
                        ui.output_ui("pp_explainer"),
                        ui.output_data_frame("pp_contrib"),
                        ui.output_ui("pp_contrib_dict"),
                    ),




                    ui.hr(),
                    ui.card(
                        ui.card_header("Provider’s anomalous rows (from frozen worklists, if present)"),
                        ui.output_data_frame("pp_anom_rows"),
                        ui.output_ui("pp_anom_rows_dict"),
                    ),
                ),
            ),
        ),

        # ----------------------------------------------------
        # 6) DOCUMENTATION
        # ----------------------------------------------------
        ui.nav_panel(
            "Documentation",
            ui.card(
                ui.card_header("Run provenance"),
                ui.output_ui("doc_provenance"),
            ),
            ui.card(
                ui.card_header("Definitions"),
                ui.markdown("""
                **Grains**
                - Row grain: `(Rndrng_NPI, HCPCS_Cd, Place_Of_Srvc, Year)`
                - Provider grain: `(Rndrng_NPI, provider_type, state)`

                **Core signals**
                - `residual = observed_cost - expected_cost`
                - `oe_ratio = observed_cost / expected_cost`
                - `log_oe = log(oe_ratio)` (when expected_cost > 0)

                **Tiering**
                - We cluster only eligible providers (minimum evidence gate).
                - Cluster labels are descriptive, not “Gold/Silver”.
                """),
            ),
            ui.card(
                ui.card_header("Params dumps (as loaded)"),
                ui.h6("Anomaly surfacing params"),
                ui.output_ui("doc_anom_params"),
                ui.hr(),
                ui.h6("Tiering params"),
                ui.output_ui("doc_tier_params"),
                ui.hr(),
                ui.h6("Classification params"),
                ui.output_ui("doc_clf_params"),
            ),
        ),
    ),
)


# ============================================================
# 4) SERVER
# ============================================================

def server(input, output, session):

    # -----------------------------
    # OVERVIEW KPIs
    # -----------------------------
    @output
    @render.text
    def kpi_total_providers():
        return _fmt_int(len(scorecard_all))

    @output
    @render.text
    def kpi_eligible_providers():
        return _fmt_int(int(scorecard_all["cluster_label_v1"].astype(str).str.startswith("cluster_").sum()))

    @output
    @render.text
    def kpi_c0():
        return _fmt_int(int((scorecard_all["cluster_label_v1"] == "cluster_0").sum()))

    @output
    @render.text
    def kpi_c1():
        return _fmt_int(int((scorecard_all["cluster_label_v1"] == "cluster_1").sum()))

    @output
    @render.text
    def kpi_robust_prov():
        if "has_any_robust_event" in scorecard_all.columns:
            return _fmt_int(int(scorecard_all["has_any_robust_event"].sum()))
        return "n/a"

    @output
    @render.text
    def kpi_shock_prov():
        if "has_any_shock_event" in scorecard_all.columns:
            return _fmt_int(int(scorecard_all["has_any_shock_event"].sum()))
        return "n/a"

    @output
    @render.text
    def kpi_watchlist():
        return _fmt_int(int((scorecard_all["cluster_label_v1"] == "cluster_1").sum()))

    @output
    @render.text
    def kpi_prev():
        elig = scorecard_all[scorecard_all["cluster_label_v1"].astype(str).str.startswith("cluster_")]
        if len(elig) == 0:
            return "n/a"
        return _fmt_pct(float((elig["cluster_label_v1"] == "cluster_1").mean() * 100))

    @output
    @render.plot
    def plot_overview_logoe():
        year = None if input.ov_year() == "All" else int(input.ov_year())
        df = _read_eval_filtered(
            year=year,
            provider_type=None if input.ov_provider_type() == "All" else input.ov_provider_type(),
            state=None if input.ov_state() == "All" else input.ov_state(),
            hcpcs=None,
            pos=None,
            limit=200_000,
        )

        if bool(input.ov_eligible_only()):
            # eligible NPIs under current provider-grain filters
            sc = _apply_filters_scorecard(scorecard_all, input.ov_provider_type(), input.ov_state(), eligible_only=True)
            elig_npis = set(sc["Rndrng_NPI"].astype(str).unique().tolist())
            df = df[df["Rndrng_NPI"].astype(str).isin(elig_npis)]


        x = pd.to_numeric(df["log_oe"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(111)
        ax.hist(x, bins=60)
        ax.set_xlabel("log_oe")
        ax.set_ylabel("row count")
        ax.set_title("log(O/E) distribution (filtered sample)")
        plt.tight_layout()
        return fig

    @output
    @render.data_frame
    def tbl_cluster_card():
        elig = scorecard_all[scorecard_all["cluster_label_v1"].astype(str).str.startswith("cluster_")].copy()
        elig["n_anom_rows_mag"] = pd.to_numeric(elig.get("n_anom_rows_mag", 0), errors="coerce").fillna(0)

        card = (
            elig.assign(has_any_shock=(elig["n_anom_rows_mag"] > 0))
            .groupby(["cluster_label_v1", "cluster_definition_v1"], dropna=False)
            .agg(
                n_providers=("Rndrng_NPI", "size"),
                median_n_anom_rows_robust=("n_anom_rows_robust", "median"),
                median_anom_rate_pct_robust=("anom_rate_pct_robust", "median"),
                median_p90_log_oe=("p90_log_oe", "median"),
                pct_with_any_shock_event=("has_any_shock", lambda s: float(s.mean() * 100)),
                median_total_services_rob=("total_services_rob", "median"),
            )
            .reset_index()
        )

        order = {"cluster_0": 0, "cluster_1": 1}
        card["__o"] = card["cluster_label_v1"].map(order).fillna(99)
        card = card.sort_values("__o").drop(columns="__o")

        card["pct_with_any_shock_event"] = card["pct_with_any_shock_event"].round(2)
        card["median_anom_rate_pct_robust"] = pd.to_numeric(card["median_anom_rate_pct_robust"], errors="coerce").round(3)
        card["median_p90_log_oe"] = pd.to_numeric(card["median_p90_log_oe"], errors="coerce").round(3)

        return render.DataGrid(card, height="260px")




    @output
    @render.ui
    def tbl_cluster_card_dict():
        lines = ["**Column dictionary (Cluster label card):**\n"]
        for k, v in CLUSTER_CARD_COL_DICT.items():
            lines.append(f"- **{k}**: {v}")
        return ui.markdown("\n".join(lines))




    # -----------------------------
    # OVERVIEW: filtered provider scorecard (fast, provider-grain)
    # -----------------------------
    @reactive.calc
    def ov_score():
        df = scorecard_all.copy()

        if input.ov_provider_type() != "All":
            df = df[df["provider_type"].astype(str) == input.ov_provider_type()]

        if input.ov_state() != "All":
            df = df[df["state"].astype(str) == input.ov_state()]

        if input.ov_eligible_only():
            df = df[df["cluster_label_v1"].astype(str).str.startswith("cluster_")]

        return df





    @output
    @render.plot
    def plot_overview_cluster_mix():
        # Use filtered provider-grain view
        df = ov_score()

        # Normalize labels so anything non-cluster_* becomes "ineligible"
        lab = df["cluster_label_v1"].astype(str).fillna("ineligible")
        lab = lab.where(lab.str.startswith("cluster_"), other="ineligible")

        # Fixed ordering
        order = ["cluster_0", "cluster_1", "ineligible"]
        counts = (
            lab.value_counts()
            .reindex(order)
            .fillna(0)
            .astype(int)
        )

        total = int(counts.sum()) if int(counts.sum()) > 0 else 1
        pct = counts / total * 100.0

        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)

        bars = ax.bar(counts.index.tolist(), counts.values.tolist())

        ax.set_ylabel("providers")
        ax.set_title("Cluster mix (providers) under current filters")

        # Add labels: "N (p%)"
        ymax = max(counts.values.tolist() + [1])
        ax.set_ylim(0, ymax * 1.15)

        for i, b in enumerate(bars):
            n = int(counts.iloc[i])
            p = float(pct.iloc[i])
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height(),
                f"{n:,}\n({p:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.tight_layout()
        return fig





    @output
    @render.data_frame
    def tbl_overview_slice_summary():
        # UI selections
        year = None if input.ov_year() == "All" else int(input.ov_year())
        pt = input.ov_provider_type()
        st = input.ov_state()
        eligible_only = bool(input.ov_eligible_only())

        # Provider-grain slice from scorecard (fast, exact)
        sc = _apply_filters_scorecard(scorecard_all, pt, st, eligible_only)

        n_providers = int(len(sc))
        total_services_provider = float(pd.to_numeric(sc.get("total_services_rob", 0), errors="coerce").fillna(0).sum())
        total_benes_provider = float(pd.to_numeric(sc.get("total_benes_rob", 0), errors="coerce").fillna(0).sum())

        # Row-grain slice from eval_base (may be large, so cap if needed)
        eval_cols = ["Rndrng_NPI", "HCPCS_Cd", "Year", "provider_type", "state", "services", "benes"]
        df = pd.read_parquet(EVAL_SCORED_PATH, columns=eval_cols)

        # Apply filters
        if year is not None:
            df = df[df["Year"] == year]
        if pt != "All":
            df = df[df["provider_type"].astype(str) == pt]
        if st != "All":
            df = df[df["state"].astype(str) == st]

        # Cap for responsiveness
        sampled = False
        MAX_ROWS_SUMMARY = 400_000
        if len(df) > MAX_ROWS_SUMMARY:
            df = df.sample(n=MAX_ROWS_SUMMARY, random_state=7)
            sampled = True

        n_hcpcs = int(df["HCPCS_Cd"].astype(str).nunique()) if "HCPCS_Cd" in df.columns else 0
        total_services_row = float(pd.to_numeric(df.get("services", 0), errors="coerce").fillna(0).sum())
        total_benes_row = float(pd.to_numeric(df.get("benes", 0), errors="coerce").fillna(0).sum())

        # Anomaly counts from frozen anomaly surfaces (row-level)
        def _anom_count(anom_df: pd.DataFrame) -> int:
            if anom_df is None or len(anom_df) == 0:
                return 0
            d = anom_df.copy()
            if year is not None and "Year" in d.columns:
                d = d[d["Year"] == year]
            if pt != "All" and "provider_type" in d.columns:
                d = d[d["provider_type"].astype(str) == pt]
            if st != "All" and "state" in d.columns:
                d = d[d["state"].astype(str) == st]
            return int(len(d))

        n_robust_rows = _anom_count(anom_rows_primary)  # ANOM.2.a.1
        n_severe_rows = _anom_count(anom_rows_alt)      # ANOM.2.b.1

        # Two-column “card style” table
        left = [
            ("Filters", f"Year={year if year is not None else 'All'} | ProviderType={pt} | State={st} | EligibleOnly={eligible_only}"),
            ("Providers (provider grain)", f"{n_providers:,}"),
            ("Unique HCPCS (row grain)", f"{n_hcpcs:,}" + (" (sampled)" if sampled else "")),
        ]
        right = [
            ("Robust events (ANOM.2.a.1 rows)", f"{n_robust_rows:,}"),
            ("Severe events (ANOM.2.b.1 rows)", f"{n_severe_rows:,}"),
            ("Total services (provider scorecard)", f"{total_services_provider:,.0f}"),
            ("Total benes (provider scorecard)", f"{total_benes_provider:,.0f}"),
            ("Total services (row slice)", f"{total_services_row:,.0f}" + (" (sampled)" if sampled else "")),
            ("Total benes (row slice)", f"{total_benes_row:,.0f}" + (" (sampled)" if sampled else "")),
        ]

        # Interleave into two columns
        n_rows = max(len(left), len(right))
        left = left + [("", "")] * (n_rows - len(left))
        right = right + [("", "")] * (n_rows - len(right))

        out_rows = []
        for i in range(n_rows):
            out_rows.append({
                "Metric (counts)": left[i][0],
                "Value": left[i][1],
                "Metric (volume/events)": right[i][0],
                "Value ": right[i][1],  # trailing space to avoid duplicate column name
            })

        return render.DataGrid(pd.DataFrame(out_rows), height="260px")


    @output
    @render.ui
    def tbl_overview_slice_summary_dict():
        lines = ["**Column dictionary (Selected slice summary):**\n"]
        for k, v in SLICE_SUMMARY_COL_DICT.items():
            lines.append(f"- **{k}**: {v}")
        return ui.markdown("\n".join(lines))


    # -----------------------------
    # BENCHMARK EXPLORER
    # -----------------------------
    @output
    @render.text
    def bx_title():
        if input.bx_mode().startswith("Slice"):
            return f"Slice explorer. HCPCS={input.bx_hcpcs()} | Year={int(input.bx_year())}"
        npi = input.bx_npi().strip()
        return f"Provider explorer. NPI={npi if npi else '(enter an NPI)'}"

    @output
    @render.ui
    def bx_grain_note():
        return ui.p(
            "Grain is row-level: (NPI, HCPCS, Place_Of_Srvc, Year). This is a trust builder, not fraud detection.",
            class_="muted",
        )




    @reactive.calc
    def bx_df():
        if input.bx_mode().startswith("Slice"):
            df = _read_eval_filtered(
                year=int(input.bx_year()),
                provider_type=None,
                state=None,
                hcpcs=input.bx_hcpcs().strip(),
                pos=None,
                limit=250_000,
            )

            # NEW: optional slice refinements (only in slice mode)
            pos = input.bx_pos()
            st = input.bx_state()
            npi_slice = input.bx_npi_slice().strip()

            if pos and pos != "All":
                df = df[df["Place_Of_Srvc"].astype(str) == str(pos)]
            if st and st != "All":
                df = df[df["state"].astype(str) == str(st)]
            if npi_slice:
                df = df[df["Rndrng_NPI"].astype(str) == str(npi_slice)]

            return df

        # Provider explorer (unchanged)
        npi = input.bx_npi().strip()
        if not npi:
            return pd.DataFrame()
        df = _read_eval_filtered(
            year=None,
            provider_type=None,
            state=None,
            hcpcs=None,
            pos=None,
            limit=300_000,
        )
        return df[df["Rndrng_NPI"].astype(str) == npi].copy()







    @output
    @render.plot
    def bx_scatter_obs_exp():
        df = bx_df()
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        if df is None or len(df) == 0:
            ax.text(0.5, 0.5, "No data for this selection.", ha="center", va="center")
            ax.axis("off")
            return fig

        x = pd.to_numeric(df["expected_cost"], errors="coerce")
        y = pd.to_numeric(df["observed_cost"], errors="coerce")
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]

        if input.bx_log_scale():
            ax.scatter(np.log1p(x), np.log1p(y), s=8, alpha=0.4)
            ax.set_xlabel("log1p(expected_cost)")
            ax.set_ylabel("log1p(observed_cost)")
        else:
            ax.scatter(x, y, s=8, alpha=0.4)
            ax.set_xlabel("expected_cost")
            ax.set_ylabel("observed_cost")

        ax.set_title("Observed vs Expected")
        plt.tight_layout()
        return fig

    @output
    @render.plot
    def bx_resid_hist():
        df = bx_df()
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        if df is None or len(df) == 0:
            ax.text(0.5, 0.5, "No data for this selection.", ha="center", va="center")
            ax.axis("off")
            return fig
        r = pd.to_numeric(df["residual"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        ax.hist(r, bins=60)
        ax.set_xlabel("residual")
        ax.set_ylabel("row count")
        ax.set_title("Residual distribution")
        plt.tight_layout()
        return fig

    @output
    @render.data_frame
    def bx_top_rows():
        df = bx_df()
        if df is None or len(df) == 0:
            return render.DataGrid(pd.DataFrame(), height="240px")
        out = df.copy()
        out["log_oe"] = pd.to_numeric(out["log_oe"], errors="coerce")
        out = out.sort_values("log_oe", ascending=False).head(30)
        cols = ["row_id","Rndrng_NPI","HCPCS_Cd","Place_Of_Srvc","Year","observed_cost","expected_cost","residual","log_oe","oe_ratio","services","benes"]
        cols = [c for c in cols if c in out.columns]
        return render.DataGrid(out[cols], height="240px")






    @output
    @render.ui
    def bx_top_rows_dict():
        lines = ["**Column dictionary (Benchmark Explorer: Top over-expected rows):**\n"]
        for k, v in BENCHMARK_TOP_ROWS_COL_DICT.items():
            lines.append(f"- **{k}**: {v}")
        return ui.markdown("\n".join(lines))






    # -----------------------------
    # ANOMALY SURFACING TABLES
    # -----------------------------
    @output
    @render.data_frame
    def tbl_rows_primary():
        df = anom_rows_primary.copy()
        return render.DataGrid(df.head(200), height="320px", selection_mode="row")




    @output
    @render.ui
    def tbl_rows_primary_dict():
        lines = ["**Column dictionary (ANOM.2.a.1 Primary: robust + size-aware):**\n"]
        for k, v in ANOM2A1_COL_DICT.items():
            lines.append(f"- **{k}**: {v}")
        return ui.markdown("\n".join(lines))




    @output
    @render.data_frame
    def tbl_rows_alt():
        df = anom_rows_alt.copy()
        return render.DataGrid(df.head(200), height="320px", selection_mode="row")




    @output
    @render.ui
    def tbl_rows_alt_dict():
        lines = ["**Column dictionary (ANOM.2.b.1 Alternative: magnitude-aware + size-aware):**\n"]
        for k, v in ANOM2B1_COL_DICT.items():
            lines.append(f"- **{k}**: {v}")
        return ui.markdown("\n".join(lines))






    @output
    @render.data_frame
    def tbl_prov_primary():
        df = anom_prov_primary.copy()
        return render.DataGrid(df.head(200), height="320px", selection_mode="row")





    @output
    @render.ui
    def tbl_prov_primary_dict():
        lines = ["**Column dictionary (ANOM.3.a.1 Provider-level: repeat offenders, robust + size-aware):**\n"]
        for k, v in ANOM3A1_COL_DICT.items():
            lines.append(f"- **{k}**: {v}")
        return ui.markdown("\n".join(lines))







    @output
    @render.data_frame
    def tbl_prov_alt():
        df = anom_prov_alt.copy()
        return render.DataGrid(df.head(200), height="320px", selection_mode="row")




    @output
    @render.ui
    def tbl_prov_alt_dict():
        lines = ["**Column dictionary (ANOM.3.b.1 Provider-level: shock providers, magnitude-aware):**\n"]
        for k, v in ANOM3B1_COL_DICT.items():
            lines.append(f"- **{k}**: {v}")
        return ui.markdown("\n".join(lines))






    @output
    @render.ui
    def rules_anom():
        if not anom_params:
            return ui.p("No anomaly params.json found.", class_="muted")
        txt = json.dumps(anom_params, indent=2)
        return ui.accordion(
            ui.accordion_panel("Show anomaly surfacing rules and knobs (params.json)", ui.tags.pre(txt)),
            open=False,
        )

    # ---- Detail "drawers" implemented as modals (reliable) ----
    @reactive.effect
    @reactive.event(input.btn_row_primary_detail)
    def _show_row_primary_detail():
        df = anom_rows_primary.head(200).copy()
        row = _selected_row_from_grid(input, "tbl_rows_primary", df)
        if row is None:
            ui.modal_show(ui.modal("Select a row in the Primary table first.", title="No row selected"))
            return

        fields = [c for c in [
            "row_id","Rndrng_NPI","HCPCS_Cd","Place_Of_Srvc","Year",
            "observed_cost","expected_cost","residual","log_oe","oe_ratio",
            "slice_n","log_oe_pct_in_slice"
        ] if c in row.index]

        body = ui.TagList(
            ui.h6("Row details"),
            ui.tags.pre(row[fields].to_string()),
            ui.hr(),
            ui.h6("Why flagged"),
            ui.tags.pre(_rules_text_for("row_primary")),
        )
        ui.modal_show(ui.modal(body, title="Primary row detail", size="l"))

    @reactive.effect
    @reactive.event(input.btn_row_alt_detail)
    def _show_row_alt_detail():
        df = anom_rows_alt.head(200).copy()
        row = _selected_row_from_grid(input, "tbl_rows_alt", df)
        if row is None:
            ui.modal_show(ui.modal("Select a row in the Alternative table first.", title="No row selected"))
            return

        fields = [c for c in [
            "row_id","Rndrng_NPI","HCPCS_Cd","Place_Of_Srvc","Year",
            "observed_cost","expected_cost","residual","log_oe","oe_ratio",
            "slice_n","log_oe_pct_in_slice"
        ] if c in row.index]

        body = ui.TagList(
            ui.h6("Row details"),
            ui.tags.pre(row[fields].to_string()),
            ui.hr(),
            ui.h6("Why flagged"),
            ui.tags.pre(_rules_text_for("row_alt")),
        )
        ui.modal_show(ui.modal(body, title="Alternative row detail", size="l"))

    @reactive.effect
    @reactive.event(input.btn_prov_primary_detail)
    def _show_prov_primary_detail():
        df = anom_prov_primary.head(200).copy()
        row = _selected_row_from_grid(input, "tbl_prov_primary", df)
        if row is None:
            ui.modal_show(ui.modal("Select a provider row first.", title="No provider selected"))
            return

        body = ui.TagList(
            ui.h6("Provider summary (repeat offenders)"),
            ui.tags.pre(row.to_string()),
            ui.hr(),
            ui.h6("Why flagged"),
            ui.tags.pre(_rules_text_for("prov_primary")),
        )
        ui.modal_show(ui.modal(body, title="Repeat-offender provider detail", size="l"))

    @reactive.effect
    @reactive.event(input.btn_prov_alt_detail)
    def _show_prov_alt_detail():
        df = anom_prov_alt.head(200).copy()
        row = _selected_row_from_grid(input, "tbl_prov_alt", df)
        if row is None:
            ui.modal_show(ui.modal("Select a provider row first.", title="No provider selected"))
            return

        body = ui.TagList(
            ui.h6("Provider summary (shock providers)"),
            ui.tags.pre(row.to_string()),
            ui.hr(),
            ui.h6("Why flagged"),
            ui.tags.pre(_rules_text_for("prov_alt")),
        )
        ui.modal_show(ui.modal(body, title="Shock provider detail", size="l"))

    # -----------------------------
    # TIERING
    # -----------------------------
    @output
    @render.ui
    def tier_gate_note():
        return ui.p("Eligibility gate: clustered only providers with sufficient evidence (volume and row count).", class_="muted")

    @output
    @render.plot
    def tier_gate_hist():
        df = scorecard_all.copy()
        fig = plt.figure(figsize=(7, 4))
        ax = fig.add_subplot(111)
        x = pd.to_numeric(df.get("n_rows_rob", pd.Series(dtype=float)), errors="coerce").dropna()
        ax.hist(x, bins=60)
        ax.set_xlabel("n_rows_rob")
        ax.set_ylabel("providers")
        ax.set_title("Eligibility evidence proxy: n_rows_rob distribution")
        plt.tight_layout()
        return fig

    @output
    @render.plot
    def tier_ecdf_n_anom():
        df = scorecard_all.copy()
        elig = df[df["cluster_label_v1"].astype(str).str.startswith("cluster_")].copy()
        fig = plt.figure(figsize=(7, 4))
        ax = fig.add_subplot(111)
        for lab in ["cluster_0", "cluster_1"]:
            sub = elig[elig["cluster_label_v1"] == lab]
            x = pd.to_numeric(sub["n_anom_rows_robust"], errors="coerce").fillna(0).to_numpy()
            x = np.sort(x)
            y = np.arange(1, len(x) + 1) / max(len(x), 1)
            ax.plot(x, y, label=lab)
        ax.set_xlabel("n_anom_rows_robust")
        ax.set_ylabel("ECDF")
        ax.set_title("Distribution overlap: robust repeat-offender count")
        ax.legend(frameon=False)
        plt.tight_layout()
        return fig

    @output
    @render.data_frame
    def tbl_atypical():
        df = scorecard_all.copy()
        elig = df[df["cluster_label_v1"].astype(str).str.startswith("cluster_")].copy()
        elig["dist_to_centroid_v1"] = pd.to_numeric(elig.get("dist_to_centroid_v1", np.nan), errors="coerce")
        elig = elig.sort_values("dist_to_centroid_v1", ascending=False).head(30)
        cols = ["Rndrng_NPI","provider_type","state","cluster_label_v1","cluster_definition_v1","dist_to_centroid_v1",
                "n_anom_rows_robust","anom_rate_pct_robust","p90_log_oe","total_services_rob"]
        cols = [c for c in cols if c in elig.columns]
        return render.DataGrid(elig[cols], height="260px")





    @output
    @render.ui
    def tbl_atypical_dict():
        lines = ["**Column dictionary (Tiering: most atypical providers by distance-to-centroid):**\n"]
        for k, v in TIER_ATYPICAL_COL_DICT.items():
            lines.append(f"- **{k}**: {v}")
        return ui.markdown("\n".join(lines))










    @output
    @render.data_frame
    def tbl_cluster_profile_compact():
        df = scorecard_all.copy()
        elig = df[df["cluster_label_v1"].astype(str).str.startswith("cluster_")].copy()

        prof_cols = [
            "median_log_oe_rob","p90_log_oe","n_anom_rows_robust","anom_rate_pct_robust",
            "p95_log_oe_mag","pct_high_conf_rows","total_services_rob","total_benes_rob","n_rows_rob"
        ]
        prof_cols = [c for c in prof_cols if c in elig.columns]

        def _p95(s): return float(pd.to_numeric(s, errors="coerce").quantile(0.95))

        out = (
            elig.groupby(["cluster_label_v1","cluster_definition_v1"], dropna=False)[prof_cols]
            .agg(["median", _p95])
        )
        out.columns = [f"{a}__{b if isinstance(b, str) else 'p95'}" for (a, b) in out.columns]
        out = out.reset_index()
        return render.DataGrid(out, height="320px")





    @output
    @render.ui
    def tbl_cluster_profile_compact_dict():
        # Keep ordering consistent with the dict definition (Python 3.7+ preserves insertion order)
        lines = ["### Column dictionary (Tiering: cluster profiling)\n"]
        for col, desc in CLUSTER_PROFILE_TABLE_COL_DICT.items():
            lines.append(f"- **{col}**: {desc}")
        return ui.markdown("\n".join(lines))













    # -----------------------------
    # EXPLAINER MODEL
    # -----------------------------
    @output
    @render.ui
    def explainer_note():
        return ui.p(EXPLAINER_TAB_DIAG["note"], class_="muted")


    @output
    @render.data_frame
    def tbl_explainer_artifact():
        return render.DataGrid(EXPLAINER_TAB_DIAG["artifact_df"], height="220px")





    @output
    @render.ui
    def tbl_explainer_artifact_dict():
        lines = ["### Column dictionary (Explainer Model: explainer artifact and config)\n"]
        for col, desc in EXPLAINER_ARTIFACT_COL_DICT.items():
            lines.append(f"- **{col}**: {desc}")
        return ui.markdown("\n".join(lines))






    @output
    @render.data_frame
    def tbl_explainer_classification_report():
        df = EXPLAINER_TAB_DIAG["report_df"].copy()
        if df.empty:
            return render.DataGrid(pd.DataFrame(), height="220px")
        return render.DataGrid(df.round(3), height="220px")






    @output
    @render.ui
    def tbl_explainer_classification_report_dict():
        lines = ["### Column dictionary (Explainer Model: classification report)\n"]
        for col, desc in EXPLAINER_CLASS_REPORT_COL_DICT.items():
            lines.append(f"- **{col}**: {desc}")
        return ui.markdown("\n".join(lines))







    @output
    @render.plot(alt="Explainer confusion matrix")
    def plt_explainer_cm():
        cm = EXPLAINER_TAB_DIAG["cm"]

        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, interpolation="nearest", aspect="auto")
        fig.colorbar(im, ax=ax)

        ax.set_title("Confusion Matrix @ 0.50 Threshold")
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["cluster_0", "cluster_1"])
        ax.set_yticklabels(["cluster_0", "cluster_1"])

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center")

        fig.tight_layout()
        return fig


    @output
    @render.plot(alt="Explainer ROC curve")
    def plt_explainer_roc():
        fpr = EXPLAINER_TAB_DIAG["fpr"]
        tpr = EXPLAINER_TAB_DIAG["tpr"]
        roc_auc = EXPLAINER_TAB_DIAG["roc_auc"]

        fig, ax = plt.subplots(figsize=(5, 4))
        if len(fpr) > 0:
            ax.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
            ax.plot([0, 1], [0, 1], linestyle="--")
            ax.legend(loc="lower right", frameon=False)
        else:
            ax.text(0.5, 0.5, "ROC unavailable", ha="center", va="center")

        ax.set_title("ROC Curve")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        fig.tight_layout()
        return fig


    @output
    @render.plot(alt="Explainer precision-recall curve")
    def plt_explainer_pr():
        precision_curve = EXPLAINER_TAB_DIAG["precision_curve"]
        recall_curve = EXPLAINER_TAB_DIAG["recall_curve"]
        pr_auc = EXPLAINER_TAB_DIAG["pr_auc"]

        fig, ax = plt.subplots(figsize=(5, 4))
        if len(precision_curve) > 0 and len(recall_curve) > 0:
            ax.plot(recall_curve, precision_curve, label=f"PR AUC = {pr_auc:.3f}")
            ax.legend(loc="lower left", frameon=False)
        else:
            ax.text(0.5, 0.5, "PR curve unavailable", ha="center", va="center")

        ax.set_title("Precision-Recall Curve")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        fig.tight_layout()
        return fig


    @output
    @render.ui
    def txt_explainer_baselines():
        txt = EXPLAINER_TAB_DIAG["baseline_text"]
        return ui.markdown(f"**Baselines** | `{txt}`")






    @output
    @render.ui
    def tbl_explainer_baselines_dict():
        lines = ["### Column dictionary (Explainer Model: baselines)\n"]
        for col, desc in EXPLAINER_BASELINE_COL_DICT.items():
            lines.append(f"- **{col}**: {desc}")
        return ui.markdown("\n".join(lines))







    @output
    @render.data_frame
    def tbl_explainer_pos_drivers():
        df = EXPLAINER_TAB_DIAG["coef_pos"].copy()
        if df.empty:
            return render.DataGrid(pd.DataFrame(), height="320px")
        return render.DataGrid(df.round(6), height="320px")





    @output
    @render.ui
    def tbl_explainer_pos_driver_dict():
        lines = ["### Column dictionary (Explainer Model: driver tables)\n"]
        for col, desc in EXPLAINER_DRIVER_COL_DICT.items():
            lines.append(f"- **{col}**: {desc}")
        return ui.markdown("\n".join(lines))







    @output
    @render.data_frame
    def tbl_explainer_neg_drivers():
        df = EXPLAINER_TAB_DIAG["coef_neg"].copy()
        if df.empty:
            return render.DataGrid(pd.DataFrame(), height="320px")
        return render.DataGrid(df.round(6), height="320px")



    @output
    @render.ui
    def tbl_explainer_neg_driver_dict():
        lines = ["### Column dictionary (Explainer Model: driver tables)\n"]
        for col, desc in EXPLAINER_DRIVER_COL_DICT.items():
            lines.append(f"- **{col}**: {desc}")
        return ui.markdown("\n".join(lines))





















    # -----------------------------
    # PROVIDER PROFILE (stitched narrative)
    # -----------------------------
    @reactive.calc
    def pp_npi():
        input.pp_refresh()
        return input.pp_npi().strip()

    @output
    @render.ui
    def pp_header():
        npi = pp_npi()
        if not npi:
            return ui.p("Enter an NPI and click Refresh.", class_="muted")
        hit = scorecard_all[scorecard_all["Rndrng_NPI"].astype(str) == npi]
        if len(hit) == 0:
            return ui.p("NPI not found in provider_scorecard.", class_="muted")

        r = hit.iloc[0]
        lines = [
            f"NPI: {npi}",
            f"provider_type: {r.get('provider_type','')}",
            f"state: {r.get('state','')}",
            f"cluster: {r.get('cluster_label_v1','')} ({r.get('cluster_definition_v1','')})",
            f"dist_to_centroid: {r.get('dist_to_centroid_v1','')}",
        ]
        return ui.tags.ul(*[ui.tags.li(x) for x in lines])

    @output
    @render.data_frame
    def pp_metrics():
        npi = pp_npi()
        if not npi:
            return render.DataGrid(pd.DataFrame(), height="220px")
        hit = scorecard_all[scorecard_all["Rndrng_NPI"].astype(str) == npi]
        if len(hit) == 0:
            return render.DataGrid(pd.DataFrame(), height="220px")

        cols = [
            "n_rows_rob","total_services_rob","total_benes_rob","pct_high_conf_rows",
            "n_anom_rows_robust","anom_rate_pct_robust","p90_log_oe",
            "n_anom_rows_mag","p95_log_oe_mag",
            "cluster_label_v1","cluster_definition_v1","dist_to_centroid_v1"
        ]
        cols = [c for c in cols if c in hit.columns]
        return render.DataGrid(hit[cols].iloc[0:1], height="220px")





    @output
    @render.ui
    def pp_metrics_dict():
        lines = ["### Column dictionary (Provider Profile: tiering + anomalies)\n"]
        for col, desc in PP_TIERING_ANOM_COL_DICT.items():
            lines.append(f"- **{col}**: {desc}")
        return ui.markdown("\n".join(lines))






    @output
    @render.ui
    def pp_explainer():
        npi = pp_npi()
        if not npi:
            return ui.p("Enter an NPI and click Refresh.", class_="muted")
        p, _ = _explainer_predict_and_contrib(npi)
        if p is None:
            return ui.p("Explainer model unavailable, or provider covariates not found in eval_universe.", class_="muted")
        return ui.TagList(
            ui.p(f"Predicted P(cluster_1) among eligible providers: {p:.3f}", class_="muted"),
            ui.p("Explanation aid only. Not decision logic.", class_="muted"),
        )





    @output
    @render.ui
    def pp_contrib_dict():
        lines = ["### Column dictionary (Provider Profile: explainer contributions)\n"]
        for col, desc in PP_EXPLAINER_CONTRIB_COL_DICT.items():
            lines.append(f"- **{col}**: {desc}")
        return ui.markdown("\n".join(lines))








    @output
    @render.data_frame
    def pp_contrib():
        npi = pp_npi()
        if not npi:
            return render.DataGrid(pd.DataFrame(), height="260px")
        _, dfc = _explainer_predict_and_contrib(npi)
        return render.DataGrid(dfc, height="260px")

    @output
    @render.data_frame
    def pp_anom_rows():
        npi = pp_npi()
        if not npi:
            return render.DataGrid(pd.DataFrame(), height="220px")

        a = anom_rows_primary[anom_rows_primary["Rndrng_NPI"].astype(str) == npi].copy() if "Rndrng_NPI" in anom_rows_primary.columns else pd.DataFrame()
        b = anom_rows_alt[anom_rows_alt["Rndrng_NPI"].astype(str) == npi].copy() if "Rndrng_NPI" in anom_rows_alt.columns else pd.DataFrame()
        out = pd.concat([a.assign(source="primary"), b.assign(source="alt")], axis=0, ignore_index=True)

        if len(out) == 0:
            return render.DataGrid(pd.DataFrame({"note": ["No rows for this NPI in the frozen top-N anomaly worklists."]}), height="140px")

        cols = [c for c in [
            "source","row_id","HCPCS_Cd","Place_Of_Srvc","Year",
            "observed_cost","expected_cost","residual","log_oe","oe_ratio",
            "slice_n","log_oe_pct_in_slice"
        ] if c in out.columns]

        return render.DataGrid(out[cols].head(200), height="260px")






    @output
    @render.ui
    def pp_anom_rows_dict():
        lines = ["### Column dictionary (Provider Profile: anomalous rows)\n"]
        for col, desc in PP_ANOM_ROWS_COL_DICT.items():
            lines.append(f"- **{col}**: {desc}")
        return ui.markdown("\n".join(lines))








    # -----------------------------
    # DOCUMENTATION
    # -----------------------------
    @output
    @render.ui
    def doc_provenance():
        items = [
            ("EVAL_SCORED_PATH", str(EVAL_SCORED_PATH)),
            ("ANOM_RUN", str(ANOM_RUN)),
            ("TIER_RUN", str(TIER_RUN)),
            ("CLF_RUN", str(CLF_RUN)),
            ("ROWS_PRIMARY_PATH", str(ROWS_PRIMARY_PATH)),
            ("ROWS_ALT_PATH", str(ROWS_ALT_PATH)),
            ("PROV_PRIMARY_PATH", str(PROV_PRIMARY_PATH)),
            ("PROV_ALT_PATH", str(PROV_ALT_PATH)),
            ("ANOM_PARAMS_JSON", str(ANOM_PARAMS_JSON)),
            ("PROVIDER_SCORECARD_PATH", str(PROVIDER_SCORECARD_PATH)),
            ("TIER_PARAMS_JSON", str(TIER_PARAMS_JSON)),
            ("FINAL_EXPLAINER_JOBLIB", str(FINAL_EXPLAINER_JOBLIB)),
            ("CLF_PARAMS_JSON", str(CLF_PARAMS_JSON)),
        ]
        return ui.tags.ul(*[ui.tags.li(f"{k}: {v}") for k, v in items])

    @output
    @render.ui
    def doc_anom_params():
        if not anom_params:
            return ui.p("No anomaly params loaded.", class_="muted")
        return ui.tags.pre(json.dumps(anom_params, indent=2))

    @output
    @render.ui
    def doc_tier_params():
        if not tiering_params:
            return ui.p("No tiering params loaded.", class_="muted")
        return ui.tags.pre(json.dumps(tiering_params, indent=2))

    @output
    @render.ui
    def doc_clf_params():
        if not clf_params:
            return ui.p("No classification params loaded.", class_="muted")
        return ui.tags.pre(json.dumps(clf_params, indent=2))


app = App(app_ui, server)