# src/bench/eval_pack.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from datetime import datetime, timezone


DEFAULT_CAT_BUCKETS: Tuple[str, ...] = (
    "cat_abs_residual_q99",
    "cat_oe_q99",
    "cat_large_positive_residual",
    "cat_large_negative_residual",
    "cat_large_error_high_support",
    "cat_large_error_tail_row",
    "cat_underprediction",
    "cat_overprediction",
    "cat_high_conf_anomaly",
    "cat_cold_low_support_failure",
    "cat_hot_failure",
    "cat_cold_failure",
)

DEFAULT_GROUP_COLS: Tuple[str, ...] = ("Year", "hcpcs_stability_group")


def make_exec_summary_tables(
    *,
    eval_src: pd.DataFrame,
    eval_scored_dg: pd.DataFrame,
    eval_scored_dg_v3: pd.DataFrame,
    thresholds: Dict,
    guardrail_c5v2_policy: Dict,
    apply_catastrophic_flags: Callable[[pd.DataFrame, Dict, str], pd.DataFrame],
    cat_buckets: Sequence[str] = DEFAULT_CAT_BUCKETS,
    group_cols: Sequence[str] = DEFAULT_GROUP_COLS,
    protected_slice: str = "Year=2023 & 2023_only",
) -> Dict[str, pd.DataFrame]:
    """
    Build the EVAL.2b exec summary pack tables (D/G vs D/G+V3 only).

    Inputs
    ------
    eval_src:
        The evaluation universe frame (must contain scope columns used in checks).
    eval_scored_dg:
        Same row universe as eval_src, with expected_cost set to D/G predictions.
    eval_scored_dg_v3:
        Same row universe as eval_src, with expected_cost after applying V3.
    thresholds:
        thresholds_v2 dict (must include epsilon or we'll default to 1e-9).
    guardrail_c5v2_policy:
        Policy dict used for the stable_all_years leak check (C5v2 scope).
    apply_catastrophic_flags:
        Function that adds catastrophic booleans given thresholds and prefix.

    Returns
    -------
    dict[str, pd.DataFrame] with keys:
      - overall_summary
      - year_group_summary
      - bucket_snapshot_overall
      - no_new_cat_table
      - leak_table
    """
    # -----------------------------
    # Preconditions / alignment
    # -----------------------------
    if len(eval_src) != len(eval_scored_dg) or len(eval_src) != len(eval_scored_dg_v3):
        raise ValueError("eval_src, eval_scored_dg, eval_scored_dg_v3 must have same number of rows")

    for col in ["Year", "hcpcs_stability_group", "HCPCS_Cd", "expected_cost_support_tier", "route", "has_lag"]:
        if col not in eval_src.columns:
            raise KeyError(f"eval_src missing required column for checks: {col}")

    if "expected_cost" not in eval_scored_dg.columns:
        raise KeyError("eval_scored_dg missing expected_cost")
    if "expected_cost" not in eval_scored_dg_v3.columns:
        raise KeyError("eval_scored_dg_v3 missing expected_cost")

    # -----------------------------
    # Helpers
    # -----------------------------
    def _score_flags(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        out = apply_catastrophic_flags(df.copy(), thresholds, prefix=prefix)
        cols = [f"{prefix}is_any_catastrophic"] + [f"{prefix}{b}" for b in cat_buckets]
        for c in cols:
            out[c] = out[c].astype(bool)
        return out

    def _overall_any(df_flags: pd.DataFrame, prefix: str) -> float:
        return float(df_flags[f"{prefix}is_any_catastrophic"].mean() * 100)

    # -----------------------------
    # 0) Apply catastrophic flags
    # -----------------------------
    dg = _score_flags(eval_scored_dg, "dg__")
    v3 = _score_flags(eval_scored_dg_v3, "v3__")

    # -----------------------------
    # 1) Overall summary
    # -----------------------------
    overall_summary = pd.DataFrame(
        [
            {"model": "D/G", "is_any_catastrophic_rate_%": _overall_any(dg, "dg__")},
            {"model": "D/G+V3", "is_any_catastrophic_rate_%": _overall_any(v3, "v3__")},
        ]
    )
    dg_rate = float(overall_summary.loc[overall_summary["model"] == "D/G", "is_any_catastrophic_rate_%"].iloc[0])
    overall_summary["delta_vs_DG_pp"] = overall_summary["is_any_catastrophic_rate_%"] - dg_rate
    overall_summary.loc[overall_summary["model"] == "D/G", "delta_vs_DG_pp"] = 0.0

    # -----------------------------
    # 2) Year × stability group summary (is_any only)
    # -----------------------------
    cols_dg = ["dg__is_any_catastrophic"]
    cols_v3 = ["v3__is_any_catastrophic"]

    year_group_dg = (
        dg.groupby(list(group_cols), dropna=False)[cols_dg]
        .mean()
        .reset_index()
        .rename(columns={"dg__is_any_catastrophic": "is_any_catastrophic_rate_%_dg"})
    )
    year_group_dg["is_any_catastrophic_rate_%_dg"] *= 100
    year_group_dg["n_rows"] = dg.groupby(list(group_cols), dropna=False).size().to_numpy()

    year_group_v3 = (
        v3.groupby(list(group_cols), dropna=False)[cols_v3]
        .mean()
        .reset_index()
        .rename(columns={"v3__is_any_catastrophic": "is_any_catastrophic_rate_%_v3"})
    )
    year_group_v3["is_any_catastrophic_rate_%_v3"] *= 100
    year_group_v3["n_rows"] = v3.groupby(list(group_cols), dropna=False).size().to_numpy()

    year_group_summary = year_group_dg.merge(year_group_v3, on=list(group_cols) + ["n_rows"], how="inner")
    year_group_summary["delta_V3_vs_DG_pp"] = (
        year_group_summary["is_any_catastrophic_rate_%_v3"] - year_group_summary["is_any_catastrophic_rate_%_dg"]
    )

    year_group_summary = year_group_summary[
        list(group_cols)
        + ["n_rows", "is_any_catastrophic_rate_%_dg", "is_any_catastrophic_rate_%_v3", "delta_V3_vs_DG_pp"]
    ].sort_values(list(group_cols))

    # -----------------------------
    # 3) Bucket snapshot (overall)
    # -----------------------------
    rows = []
    for b in ("is_any_catastrophic",) + tuple(cat_buckets):
        b_dg = float(dg[f"dg__{b}"].mean() * 100)
        b_v3 = float(v3[f"v3__{b}"].mean() * 100)
        rows.append(
            {
                "bucket": b,
                "n_rows": len(dg),
                "dg_rate_%": b_dg,
                "v3_rate_%": b_v3,
                "v3_minus_dg_pp": b_v3 - b_dg,
            }
        )
    bucket_snapshot_overall = pd.DataFrame(rows).sort_values("v3_minus_dg_pp")

    # -----------------------------
    # 4) No-new-cat check (protected slice)
    #    Default is the slice you care about: Year=2023 & 2023_only
    # -----------------------------
    if protected_slice != "Year=2023 & 2023_only":
        raise ValueError("Only supported protected_slice right now: 'Year=2023 & 2023_only'")

    mask_2023_only = ((eval_src["Year"] == 2023) & (eval_src["hcpcs_stability_group"] == "2023_only")).to_numpy()
    dg_any = dg.loc[mask_2023_only, "dg__is_any_catastrophic"].to_numpy(dtype=bool)
    v3_any = v3.loc[mask_2023_only, "v3__is_any_catastrophic"].to_numpy(dtype=bool)

    new_cat = v3_any & (~dg_any)
    no_new_cat_table = pd.DataFrame(
        [
            {
                "slice": "Year=2023 & 2023_only",
                "n_rows": int(mask_2023_only.sum()),
                "new_cat_rows": int(new_cat.sum()),
                "new_cat_%": float(new_cat.mean() * 100),
            }
        ]
    )

    # -----------------------------
    # 5) Leak check: stable_all_years expected can change ONLY inside C5v2 allowed scope
    # -----------------------------
    stable_mask = (eval_src["hcpcs_stability_group"] == "stable_all_years").to_numpy()

    codes = set(str(x) for x in guardrail_c5v2_policy["target_codes"])
    tier = str(guardrail_c5v2_policy["expected_cost_support_tier"])
    route = str(guardrail_c5v2_policy["route"])

    allowed = (
        stable_mask
        & (eval_src["HCPCS_Cd"].astype(str).isin(codes)).to_numpy()
        & (eval_src["expected_cost_support_tier"].astype(str) == tier).to_numpy()
        & (eval_src["route"].astype(str) == route).to_numpy()
        & (~eval_src["has_lag"].astype(bool)).to_numpy()
    )

    exp_dg = pd.to_numeric(eval_scored_dg["expected_cost"], errors="coerce").to_numpy(dtype="float64")
    exp_v3 = pd.to_numeric(eval_scored_dg_v3["expected_cost"], errors="coerce").to_numpy(dtype="float64")

    changed = ~np.isclose(exp_dg, exp_v3, atol=0, rtol=0, equal_nan=True)
    leak = stable_mask & changed & (~allowed)

    leak_table = pd.DataFrame(
        [
            {
                "stable_all_years_rows": int(stable_mask.sum()),
                "changed_rows_within_stable_all_years": int((stable_mask & changed).sum()),
                "allowed_scope_rows": int(allowed.sum()),
                "changed_rows_within_allowed_scope": int((allowed & changed).sum()),
                "leak_rows": int(leak.sum()),
                "leak_%_of_stable_all_years": float(leak.sum() / max(1, stable_mask.sum()) * 100),
            }
        ]
    )

    # Keep module behavior strict (same as notebook assert)
    if int(leak.sum()) != 0:
        raise AssertionError("LEAK FAIL: expected_cost changed inside stable_all_years outside C5v2 allowed scope.")

    return {
        "overall_summary": overall_summary,
        "year_group_summary": year_group_summary,
        "bucket_snapshot_overall": bucket_snapshot_overall,
        "no_new_cat_table": no_new_cat_table,
        "leak_table": leak_table,
    }


def make_eval_pack_run_dir(root: Path = Path("eval_pack_runs")) -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = root / run_id
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir

def export_eval_pack(
    tables: Dict[str, pd.DataFrame],
    *,
    out_dir: Path,
    formats: Iterable[str] = ("parquet", "csv"),
    index: bool = False,
) -> Dict[str, Dict[str, str]]:
    """
    Write each table to out_dir in the requested formats.

    Returns dict:
      {table_name: {fmt: path_str}}
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fmts = tuple(formats)

    out_paths: Dict[str, Dict[str, str]] = {}
    for name, df in tables.items():
        out_paths[name] = {}
        for fmt in fmts:
            fmt_l = fmt.lower().strip()
            if fmt_l == "parquet":
                p = out_dir / f"{name}.parquet"
                df.to_parquet(p, index=index)
                out_paths[name]["parquet"] = str(p)
            elif fmt_l == "csv":
                p = out_dir / f"{name}.csv"
                df.to_csv(p, index=index)
                out_paths[name]["csv"] = str(p)
            else:
                raise ValueError(f"Unsupported format: {fmt}")
    return out_paths