# src/bench/holdout_sanity.py
from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from pathlib import Path


CAT_BUCKETS_DEFAULT: List[str] = [
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
]


def _require_cols(df: pd.DataFrame, cols: Iterable[str], *, where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{where} is missing required columns: {missing}")


def _score_flags(
    df: pd.DataFrame,
    *,
    thresholds: Dict,
    apply_catastrophic_flags: Callable,
    prefix: str,
    cat_buckets: List[str],
) -> pd.DataFrame:
    out = apply_catastrophic_flags(df.copy(), thresholds, prefix=prefix)
    flag_cols = [f"{prefix}is_any_catastrophic"] + [f"{prefix}{b}" for b in cat_buckets]
    for c in flag_cols:
        out[c] = out[c].astype(bool)
    return out


def _rate(s: pd.Series) -> float:
    return float(np.mean(s.astype(bool).to_numpy()) * 100.0)


def year_sliced_effect_tables(
    *,
    eval_src: pd.DataFrame,
    eval_scored_dg: pd.DataFrame,
    eval_scored_dg_v3: pd.DataFrame,
    thresholds: Dict,
    apply_catastrophic_flags: Callable,
    cat_buckets: Optional[List[str]] = None,
    group_cols: Tuple[str, ...] = ("Year", "hcpcs_stability_group"),
) -> Dict[str, pd.DataFrame]:
    """
    Holdout-style effect tables:
      - overall is_any (D/G vs D/G+V3)
      - by year
      - by year x stability group (default)
      - bucket snapshot overall
    """
    cat_buckets = cat_buckets or CAT_BUCKETS_DEFAULT

    _require_cols(eval_src, ["Year", "hcpcs_stability_group"], where="eval_src")
    _require_cols(
        eval_scored_dg,
        ["expected_cost", "observed_cost", "residual", "abs_residual", "oe_ratio"],
        where="eval_scored_dg",
    )
    _require_cols(
        eval_scored_dg_v3,
        ["expected_cost", "observed_cost", "residual", "abs_residual", "oe_ratio"],
        where="eval_scored_dg_v3",
    )

    dg = _score_flags(
        eval_scored_dg,
        thresholds=thresholds,
        apply_catastrophic_flags=apply_catastrophic_flags,
        prefix="dg__",
        cat_buckets=cat_buckets,
    )
    v3 = _score_flags(
        eval_scored_dg_v3,
        thresholds=thresholds,
        apply_catastrophic_flags=apply_catastrophic_flags,
        prefix="v3__",
        cat_buckets=cat_buckets,
    )

    # 1) Overall
    overall = pd.DataFrame(
        [
            {"model": "D/G", "is_any_catastrophic_rate_%": _rate(dg["dg__is_any_catastrophic"])},
            {"model": "D/G+V3", "is_any_catastrophic_rate_%": _rate(v3["v3__is_any_catastrophic"])},
        ]
    )
    dg_rate = float(overall.loc[overall["model"] == "D/G", "is_any_catastrophic_rate_%"].iloc[0])
    overall["delta_vs_DG_pp"] = overall["is_any_catastrophic_rate_%"] - dg_rate
    overall.loc[overall["model"] == "D/G", "delta_vs_DG_pp"] = 0.0

    # 2) By Year
    by_year_idx = (
        pd.DataFrame({"Year": eval_src["Year"].to_numpy()})
        .assign(_ix=np.arange(len(eval_src)))
        .groupby("Year", dropna=False)["_ix"]
        .apply(list)
        .reset_index(name="_rows")
    )

    rows = []
    for _, r in by_year_idx.iterrows():
        idx = np.array(r["_rows"], dtype=int)
        rows.append(
            {
                "Year": int(r["Year"]),
                "n_rows": int(len(idx)),
                "is_any_catastrophic_rate_%_dg": _rate(dg.loc[idx, "dg__is_any_catastrophic"]),
                "is_any_catastrophic_rate_%_v3": _rate(v3.loc[idx, "v3__is_any_catastrophic"]),
            }
        )
    by_year_tbl = pd.DataFrame(rows)
    by_year_tbl["delta_V3_vs_DG_pp"] = (
        by_year_tbl["is_any_catastrophic_rate_%_v3"] - by_year_tbl["is_any_catastrophic_rate_%_dg"]
    )
    by_year_tbl = by_year_tbl.sort_values("Year")

    # 3) By Year x stability group
    dg_any = (
        dg.groupby(list(group_cols), dropna=False)["dg__is_any_catastrophic"]
        .mean()
        .mul(100)
        .reset_index(name="is_any_catastrophic_rate_%_dg")
    )
    v3_any = (
        v3.groupby(list(group_cols), dropna=False)["v3__is_any_catastrophic"]
        .mean()
        .mul(100)
        .reset_index(name="is_any_catastrophic_rate_%_v3")
    )
    n_rows = eval_src.groupby(list(group_cols), dropna=False).size().reset_index(name="n_rows")

    year_group = (
        n_rows.merge(dg_any, on=list(group_cols), how="inner")
        .merge(v3_any, on=list(group_cols), how="inner")
    )
    year_group["delta_V3_vs_DG_pp"] = (
        year_group["is_any_catastrophic_rate_%_v3"] - year_group["is_any_catastrophic_rate_%_dg"]
    )
    year_group = year_group.sort_values(list(group_cols))

    # 4) Bucket snapshot overall
    snap_rows = []
    for b in ["is_any_catastrophic"] + cat_buckets:
        snap_rows.append(
            {
                "bucket": b,
                "n_rows": len(eval_src),
                "dg_rate_%": _rate(dg[f"dg__{b}"]),
                "v3_rate_%": _rate(v3[f"v3__{b}"]),
            }
        )
    bucket_overall = pd.DataFrame(snap_rows)
    bucket_overall["v3_minus_dg_pp"] = bucket_overall["v3_rate_%"] - bucket_overall["dg_rate_%"]
    bucket_overall = bucket_overall.sort_values("v3_minus_dg_pp")

    return {
        "overall_summary": overall,
        "by_year_summary": by_year_tbl,
        "year_group_summary": year_group,
        "bucket_snapshot_overall": bucket_overall,
    }


def epsilon_sensitivity(
    *,
    eval_scored_dg: pd.DataFrame,
    eval_scored_dg_v3: pd.DataFrame,
    thresholds: Dict,
    apply_catastrophic_flags: Callable,
    epsilons: Iterable[float] = (1e-12, 1e-10, 1e-9, 1e-8, 1e-6),
    cat_buckets: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Sensitivity check for epsilon only.
    We re-run catastrophic flagging with thresholds['epsilon'] overwritten.
    Output is overall is_any and oe bucket rates under each epsilon.
    """
    cat_buckets = cat_buckets or CAT_BUCKETS_DEFAULT
    out_rows = []

    for eps in epsilons:
        thr = dict(thresholds)
        thr["epsilon"] = float(eps)

        dg = _score_flags(
            eval_scored_dg,
            thresholds=thr,
            apply_catastrophic_flags=apply_catastrophic_flags,
            prefix="dg__",
            cat_buckets=cat_buckets,
        )
        v3 = _score_flags(
            eval_scored_dg_v3,
            thresholds=thr,
            apply_catastrophic_flags=apply_catastrophic_flags,
            prefix="v3__",
            cat_buckets=cat_buckets,
        )

        dg_any = _rate(dg["dg__is_any_catastrophic"])
        v3_any = _rate(v3["v3__is_any_catastrophic"])

        row = {
            "epsilon": float(eps),
            "dg_is_any_%": dg_any,
            "v3_is_any_%": v3_any,
            "delta_V3_vs_DG_pp": v3_any - dg_any,
        }

        if "cat_oe_q99" in cat_buckets:
            dg_oe = _rate(dg["dg__cat_oe_q99"])
            v3_oe = _rate(v3["v3__cat_oe_q99"])
            row["dg_cat_oe_q99_%"] = dg_oe
            row["v3_cat_oe_q99_%"] = v3_oe
            row["delta_cat_oe_q99_pp"] = v3_oe - dg_oe

        out_rows.append(row)

    return pd.DataFrame(out_rows).sort_values("epsilon")


def make_holdout_sanity_pack(
    *,
    eval_src: pd.DataFrame,
    eval_scored_dg: pd.DataFrame,
    eval_scored_dg_v3: pd.DataFrame,
    thresholds: Dict,
    apply_catastrophic_flags: Callable,
    cat_buckets: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    One-call helper that returns:
      - year_sliced_effect_tables (4 tables)
      - epsilon_sensitivity (1 table)
    """
    tables = year_sliced_effect_tables(
        eval_src=eval_src,
        eval_scored_dg=eval_scored_dg,
        eval_scored_dg_v3=eval_scored_dg_v3,
        thresholds=thresholds,
        apply_catastrophic_flags=apply_catastrophic_flags,
        cat_buckets=cat_buckets,
    )
    tables["epsilon_sensitivity"] = epsilon_sensitivity(
        eval_scored_dg=eval_scored_dg,
        eval_scored_dg_v3=eval_scored_dg_v3,
        thresholds=thresholds,
        apply_catastrophic_flags=apply_catastrophic_flags,
        cat_buckets=cat_buckets,
    )
    return tables


def export_holdout_sanity_pack(
    tables: Dict[str, pd.DataFrame],
    *,
    out_dir: Path,
    formats: Tuple[str, ...] = ("parquet", "csv"),
) -> Dict[str, Dict[str, str]]:
    """
    Export holdout sanity tables to disk.

    Parameters
    ----------
    tables
        Output of make_holdout_sanity_pack(...)
    out_dir
        Folder to write files into.
    formats
        Any of ("parquet", "csv"). Defaults to both.

    Returns
    -------
    dict
        Mapping: table_name -> {format: path}
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    fmts = tuple(f.lower() for f in formats)
    allowed = {"parquet", "csv"}
    bad = [f for f in fmts if f not in allowed]
    if bad:
        raise ValueError(f"Unsupported formats: {bad}. Allowed: {sorted(allowed)}")

    out_paths: Dict[str, Dict[str, str]] = {}

    for name, df in tables.items():
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"tables['{name}'] is {type(df)}; expected pd.DataFrame")

        out_paths[name] = {}

        if "parquet" in fmts:
            p = out_dir / f"{name}.parquet"
            df.to_parquet(p, index=False)
            out_paths[name]["parquet"] = str(p)

        if "csv" in fmts:
            p = out_dir / f"{name}.csv"
            df.to_csv(p, index=False)
            out_paths[name]["csv"] = str(p)

    return out_paths