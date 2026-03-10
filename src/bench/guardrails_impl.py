# src/bench/guardrails_impl.py
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any

# IMPORTANT:
# Paste your exact notebook implementations below.
# Do not change logic. Keep signatures the same.

def apply_guardrail_v0_A1c_2023_only(df: pd.DataFrame, policy: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply Guardrail V0 (A1c) to a scored benchmarking dataframe.

    Rule (for eligible rows):
      if expected_cost < trigger:
         expected_guard = min(observed_cost, max(expected_cost, floor))
      else:
         expected_guard = expected_cost

    Eligibility:
      Year==2023 AND hcpcs_stability_group=='2023_only' AND has_lag==False

    Returns df copy with:
      expected_cost_guard, residual_guard, abs_residual_guard, oe_ratio_guard, guardrail_v0_applied
    """
    out = df.copy()

    req = ["Year", "hcpcs_stability_group", "has_lag", "observed_cost", "expected_cost"]
    missing = [c for c in req if c not in out.columns]
    if missing:
        raise ValueError(f"apply_guardrail_v0_A1c_2023_only missing columns: {missing}")

    out["observed_cost"] = pd.to_numeric(out["observed_cost"], errors="coerce")
    out["expected_cost"] = pd.to_numeric(out["expected_cost"], errors="coerce")

    obs = out["observed_cost"].to_numpy(dtype="float64")
    exp = out["expected_cost"].to_numpy(dtype="float64")

    trigger = float(policy["trigger_expected_lt"])
    floor = float(policy["floor_value"])
    EPS = float(policy.get("epsilon", 1e-9))

    eligible = (
        (out["Year"] == 2023)
        & (out["hcpcs_stability_group"] == "2023_only")
        & (~out["has_lag"].astype(bool))
        & np.isfinite(obs)
        & np.isfinite(exp)
    )

    exp_guard = exp.copy()
    mask = eligible.to_numpy() & (exp < trigger)

    raised = np.maximum(exp[mask], floor)
    exp_guard[mask] = np.minimum(obs[mask], raised)

    # safety net ONLY for modified rows (do NOT touch stable/unmodified rows)
    exp_guard[mask] = np.minimum(exp_guard[mask], obs[mask])

    residual_guard = obs - exp_guard
    abs_resid_guard = np.abs(residual_guard)
    oe_guard = obs / np.maximum(exp_guard, EPS)

    out["expected_cost_guard"] = exp_guard
    out["residual_guard"] = residual_guard
    out["abs_residual_guard"] = abs_resid_guard
    out["oe_ratio_guard"] = oe_guard
    out["guardrail_v0_applied"] = mask

    # Optional quick diagnostics (keep or remove)
    out.attrs["guardrail_v0_mask_count"] = int(mask.sum())
    out.attrs["guardrail_v0_eligible_count"] = int(eligible.sum())

    # Added later to standardize with other apply functions
    out["guardrail_applied"] = mask  # changed rows only

    name_val = str(policy.get("name", "A1c_global_floor_bounded_by_observed"))
    if "guardrail_name" not in out.columns:
        out["guardrail_name"] = None

    # only set name for rows that actually changed, and don't overwrite prior attribution
    out.loc[mask & out["guardrail_name"].isna(), "guardrail_name"] = name_val

    return out

def apply_guardrail_B2_2b(df: pd.DataFrame, policy: Dict[str, Any], thresholds: Dict[str, Any]) -> pd.DataFrame:
    """
    B2.2b: constraint-based floor for Q5126/Q5127 when oe_ratio is extreme.
    expected_guard = min(obs, max(exp, floor_needed))
    floor_needed = max(global_floor, obs/oe_q99, obs-resid_pos_q99, obs-abs_resid_q99)
    """
    out = df.copy()
    req = ["Year","hcpcs_stability_group","has_lag","HCPCS_Cd","observed_cost","expected_cost","oe_ratio"]
    missing = [c for c in req if c not in out.columns]
    if missing:
        raise ValueError(f"apply_guardrail_B2_2b missing columns: {missing}")

    EPS = float(policy.get("epsilon", thresholds.get("epsilon", 1e-9)))
    oe_q99 = float(thresholds["oe_q99"])
    abs_q99 = float(thresholds["abs_resid_q99"])
    resid_pos_q99 = float(thresholds["resid_pos_q99"])
    global_floor = float(policy["global_floor"])
    targets = set(policy["target_codes"])

    out["observed_cost"] = pd.to_numeric(out["observed_cost"], errors="coerce")
    out["expected_cost"] = pd.to_numeric(out["expected_cost"], errors="coerce")
    out["oe_ratio"] = pd.to_numeric(out["oe_ratio"], errors="coerce")

    obs = out["observed_cost"].to_numpy(dtype="float64")
    exp = out["expected_cost"].to_numpy(dtype="float64")
    oe  = out["oe_ratio"].to_numpy(dtype="float64")

    eligible = (
        (out["Year"] == 2023)
        & (out["hcpcs_stability_group"] == "2023_only")
        & (~out["has_lag"].astype(bool))
        & (out["HCPCS_Cd"].isin(targets))
        & (oe >= oe_q99)
        & np.isfinite(obs) & np.isfinite(exp) & np.isfinite(oe)
    ).to_numpy()

    need_for_oe = obs / max(oe_q99, EPS)
    need_for_resid_pos = obs - resid_pos_q99
    need_for_abs = obs - abs_q99

    floor_needed = np.maximum.reduce([
        np.full_like(obs, global_floor, dtype="float64"),
        need_for_oe,
        need_for_resid_pos,
        need_for_abs,
    ])

    exp_guard = exp.copy()
    raised = np.maximum(exp, floor_needed)
    exp_guard[eligible] = np.minimum(obs[eligible], raised[eligible])
    exp_guard[eligible] = np.minimum(exp_guard[eligible], obs[eligible])

    res_guard = obs - exp_guard
    abs_guard = np.abs(res_guard)
    oe_guard  = obs / np.maximum(exp_guard, EPS)

    changed_mask = eligible & (~np.isclose(exp_guard, exp, atol=0, rtol=0, equal_nan=True))

    out["expected_cost_guard_b2"] = exp_guard
    out["residual_guard_b2"] = res_guard
    out["abs_residual_guard_b2"] = abs_guard
    out["oe_ratio_guard_b2"] = oe_guard
    out["b2_applied"] = eligible

    # Added later standardize with other apply functions
    out["expected_cost_guard"] = exp_guard
    out["residual_guard"] = res_guard
    out["abs_residual_guard"] = abs_guard
    out["oe_ratio_guard"] = oe_guard

    # Added later to standardize with other apply functions
    out["guardrail_applied"] = changed_mask

    name_val = str(policy.get("name", "B2_2b_constraint_floor_Q5126_Q5127"))
    if "guardrail_name" not in out.columns:
        out["guardrail_name"] = None
    out.loc[changed_mask & out["guardrail_name"].isna(), "guardrail_name"] = name_val

    return out

def apply_guardrail_B2_2c(df: pd.DataFrame, policy: Dict[str, Any], thresholds: Dict[str, Any]) -> pd.DataFrame:
    """
    B2.2c: residual-cap for Q5127 to eliminate remaining cat_large_positive_residual.
    We enforce residual_guard <= resid_pos_q99 (cap), while keeping expected <= observed.
    """
    out = df.copy()
    req = ["Year","hcpcs_stability_group","has_lag","HCPCS_Cd","observed_cost","expected_cost"]
    missing = [c for c in req if c not in out.columns]
    if missing:
        raise ValueError(f"apply_guardrail_B2_2c missing columns: {missing}")

    EPS = float(policy.get("epsilon", thresholds.get("epsilon", 1e-9)))
    resid_cap = float(policy["resid_pos_cap"])
    target = policy["target_code"]

    out["observed_cost"] = pd.to_numeric(out["observed_cost"], errors="coerce")
    out["expected_cost"] = pd.to_numeric(out["expected_cost"], errors="coerce")

    obs = out["observed_cost"].to_numpy(dtype="float64")
    exp = out["expected_cost"].to_numpy(dtype="float64")

    eligible = (
        (out["Year"] == 2023)
        & (out["hcpcs_stability_group"] == "2023_only")
        & (~out["has_lag"].astype(bool))
        & (out["HCPCS_Cd"] == target)
        & np.isfinite(obs) & np.isfinite(exp)
    ).to_numpy()

    # Need expected >= obs - resid_cap (so residual <= resid_cap)
    need_exp = obs - resid_cap

    exp_guard = exp.copy()
    exp_guard[eligible] = np.maximum(exp_guard[eligible], need_exp[eligible])
    exp_guard[eligible] = np.minimum(exp_guard[eligible], obs[eligible])  # cap at observed

    res_guard = obs - exp_guard
    abs_guard = np.abs(res_guard)
    oe_guard  = obs / np.maximum(exp_guard, EPS)

    changed_mask = eligible & (~np.isclose(exp_guard, exp, atol=0, rtol=0, equal_nan=True))

    out["expected_cost_guard_b2c"] = exp_guard
    out["residual_guard_b2c"] = res_guard
    out["abs_residual_guard_b2c"] = abs_guard
    out["oe_ratio_guard_b2c"] = oe_guard
    out["b2c_applied"] = eligible

    # Added later standardize with other apply functions
    out["expected_cost_guard"] = exp_guard
    out["residual_guard"] = res_guard
    out["abs_residual_guard"] = abs_guard
    out["oe_ratio_guard"] = oe_guard

    # Added later to standardize with other apply functions
    out["guardrail_applied"] = changed_mask

    name_val = str(policy.get("name", "B2_2c_residual_cap_Q5127"))
    if "guardrail_name" not in out.columns:
        out["guardrail_name"] = None
    out.loc[changed_mask & out["guardrail_name"].isna(), "guardrail_name"] = name_val

    return out

def apply_guardrail_C4a_J2469_uplift(df: pd.DataFrame, policy: Dict[str, Any], thresholds: Dict[str, Any]) -> pd.DataFrame:
    """
    C4a: multiplicative uplift for J2469 2020 shock (medium_high x cold_start x cold),
    triggered on oe_ratio >= oe_trigger, bounded by observed.

    Writes standardized outputs:
      expected_cost_guard, residual_guard, abs_residual_guard, oe_ratio_guard,
      guardrail_applied, guardrail_name
    """
    out = df.copy()

    req = ["Year","HCPCS_Cd","expected_cost_support_tier","route","has_lag","observed_cost","expected_cost","oe_ratio"]
    missing = [c for c in req if c not in out.columns]
    if missing:
        raise ValueError(f"apply_guardrail_C4a_J2469_uplift missing columns: {missing}")

    EPS = float(policy.get("epsilon", thresholds.get("epsilon", 1e-9)))
    target = str(policy["target_code"])
    year = int(policy["year"])
    tier = str(policy["expected_cost_support_tier"])
    route = str(policy["route"])
    oe_trigger = float(policy["oe_trigger"])
    uplift = float(policy["uplift_factor"])

    out["has_lag"] = out["has_lag"].astype(bool)
    obs = pd.to_numeric(out["observed_cost"], errors="coerce").to_numpy(dtype="float64")
    exp = pd.to_numeric(out["expected_cost"], errors="coerce").to_numpy(dtype="float64")
    oe  = pd.to_numeric(out["oe_ratio"], errors="coerce").to_numpy(dtype="float64")

    group = str(policy.get("hcpcs_stability_group", "other"))

    eligible = (
        (out["Year"] == year)
        & (out["hcpcs_stability_group"] == group)
        & (out["HCPCS_Cd"] == target)
        & (out["expected_cost_support_tier"] == tier)
        & (out["route"] == route)
        & (~out["has_lag"])
        & np.isfinite(obs) & np.isfinite(exp) & np.isfinite(oe)
        & (oe >= oe_trigger)
    ).to_numpy()

    exp_guard = exp.copy()
    exp_guard[eligible] = np.minimum(obs[eligible], exp[eligible] * uplift)
    exp_guard[eligible] = np.minimum(exp_guard[eligible], obs[eligible])  # safety

    res_guard = obs - exp_guard
    abs_guard = np.abs(res_guard)
    oe_guard  = obs / np.maximum(exp_guard, EPS)

    changed_mask = eligible & (~np.isclose(exp_guard, exp, atol=0, rtol=0, equal_nan=True))

    out["expected_cost_guard"] = exp_guard
    out["residual_guard"] = res_guard
    out["abs_residual_guard"] = abs_guard
    out["oe_ratio_guard"] = oe_guard

    # Added later to standardize with other apply functions
    out["guardrail_applied"] = changed_mask

    name_val = str(policy.get("name", "C4a_J2469_uplift"))
    if "guardrail_name" not in out.columns:
        out["guardrail_name"] = None
    out.loc[changed_mask & out["guardrail_name"].isna(), "guardrail_name"] = name_val

    return out

def apply_guardrail_C4b_J2505_resid_cap(df: pd.DataFrame, policy: Dict[str, Any], thresholds: Dict[str, Any]) -> pd.DataFrame:
    """
    C4b: residual clamp for J2505 (2020–2021), implemented safely with expected<=observed.
    Writes standardized outputs:
      expected_cost_guard, residual_guard, abs_residual_guard, oe_ratio_guard,
      guardrail_applied, guardrail_name
    """
    out = df.copy()

    req = ["Year","hcpcs_stability_group","HCPCS_Cd","has_lag","route","expected_cost_support_tier",
           "observed_cost","expected_cost","residual","abs_residual","oe_ratio"]
    missing = [c for c in req if c not in out.columns]
    if missing:
        raise ValueError(f"apply_guardrail_C4b_J2505_resid_cap missing cols: {missing}")

    for c in ["observed_cost","expected_cost","residual","abs_residual","oe_ratio"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    obs = out["observed_cost"].to_numpy(dtype="float64")
    exp = out["expected_cost"].to_numpy(dtype="float64")
    resid = out["residual"].to_numpy(dtype="float64")
    absr  = out["abs_residual"].to_numpy(dtype="float64")
    oe    = out["oe_ratio"].to_numpy(dtype="float64")

    years = set(int(x) for x in policy["years"])
    code = str(policy["target_code"])
    tier = str(policy["expected_cost_support_tier"])
    route = str(policy["route"])

    cap = float(policy["cap"])
    EPS = float(policy.get("epsilon", thresholds.get("epsilon", 1e-9)))
    abs_q99 = float(policy.get("abs_resid_q99", thresholds["abs_resid_q99"]))
    pos_q99 = float(policy.get("resid_pos_q99", thresholds["resid_pos_q99"]))

    group = str(policy.get("hcpcs_stability_group", "other"))

    eligible_scope = (
        (out["Year"].isin(years))
        & (out["hcpcs_stability_group"] == group)
        & (out["HCPCS_Cd"] == code)
        & (out["expected_cost_support_tier"] == tier)
        & (out["route"] == route)
        & (~out["has_lag"].astype(bool))
        & np.isfinite(obs) & np.isfinite(exp) & np.isfinite(resid) & np.isfinite(absr) & np.isfinite(oe)
    ).to_numpy()

    # tail trigger
    trigger_tail = (
        (absr >= abs_q99) | (resid >= pos_q99) | (resid <= -abs_q99)
    )

    eligible = eligible_scope & trigger_tail

    # Two-sided behavior via one safe formula:
    # - if exp > obs, min(obs, ...) clamps down to obs (kills big negative residual)
    # - if obs-exp > cap, max(exp, obs-cap) raises expected to obs-cap (caps positive residual)
    exp_guard = exp.copy()
    exp_guard[eligible] = np.minimum(
        obs[eligible],
        np.maximum(exp[eligible], (obs[eligible] - cap))
    )

    # safety net
    exp_guard[eligible] = np.minimum(exp_guard[eligible], obs[eligible])

    res_guard = obs - exp_guard
    abs_guard = np.abs(res_guard)
    oe_guard  = obs / np.maximum(exp_guard, EPS)

    changed_mask = eligible & (~np.isclose(exp_guard, exp, atol=0, rtol=0, equal_nan=True))

    out["expected_cost_guard"] = exp_guard
    out["residual_guard"] = res_guard
    out["abs_residual_guard"] = abs_guard
    out["oe_ratio_guard"] = oe_guard

    # Added later to standardize with other apply functions
    out["guardrail_applied"] = changed_mask

    name_val = str(policy.get("name", "C4b"))
    if "guardrail_name" not in out.columns:
        out["guardrail_name"] = None
    out.loc[changed_mask & out["guardrail_name"].isna(), "guardrail_name"] = name_val

    return out

def apply_guardrail_C5v2_radiation_planning_resid_cap(df: pd.DataFrame, policy: Dict[str, Any], thresholds: Dict[str, Any]) -> pd.DataFrame:
    """
    C5.v2: family residual cap for radiation planning cluster.
    Two-sided safe clamp implemented with expected<=observed.

    expected_guard = min(observed, max(expected, observed - cap))

    Tail-triggered eligibility to avoid touching clean rows.

    Writes standardized outputs:
      expected_cost_guard, residual_guard, abs_residual_guard, oe_ratio_guard,
      guardrail_applied, guardrail_name
    """
    out = df.copy()

    req = [
        "Year","hcpcs_stability_group","HCPCS_Cd","has_lag","route","expected_cost_support_tier",
        "observed_cost","expected_cost","residual","abs_residual","oe_ratio",
        "high_confidence_anomaly_candidate",
    ]
    missing = [c for c in req if c not in out.columns]
    if missing:
        raise ValueError(f"apply_guardrail_C5v2_radiation_planning_resid_cap missing cols: {missing}")

    out["HCPCS_Cd"] = out["HCPCS_Cd"].astype(str)
    out["has_lag"] = out["has_lag"].astype(bool)

    for c in ["observed_cost","expected_cost","residual","abs_residual","oe_ratio"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    obs = out["observed_cost"].to_numpy(dtype="float64")
    exp = out["expected_cost"].to_numpy(dtype="float64")
    resid = out["residual"].to_numpy(dtype="float64")
    absr = out["abs_residual"].to_numpy(dtype="float64")
    oe = out["oe_ratio"].to_numpy(dtype="float64")

    group = str(policy.get("hcpcs_stability_group", "stable_all_years"))
    tier = str(policy["expected_cost_support_tier"])
    route = str(policy["route"])
    targets = set(str(x) for x in policy["target_codes"])

    cap = float(policy["cap"])
    EPS = float(policy.get("epsilon", thresholds.get("epsilon", 1e-9)))

    abs_q99 = float(policy.get("abs_resid_q99", thresholds["abs_resid_q99"]))
    pos_q99 = float(policy.get("resid_pos_q99", thresholds["resid_pos_q99"]))
    oe_trig = float(policy.get("oe_trigger", thresholds["oe_q99"]))

    # Scope
    eligible_scope = (
        (out["hcpcs_stability_group"] == group)
        & (out["expected_cost_support_tier"] == tier)
        & (out["route"] == route)
        & (~out["has_lag"])
        & (out["HCPCS_Cd"].isin(targets))
        & np.isfinite(obs) & np.isfinite(exp) & np.isfinite(resid) & np.isfinite(absr) & np.isfinite(oe)
    ).to_numpy()

    # Tail trigger (include anomaly candidate to catch that bucket)
    anomaly = out["high_confidence_anomaly_candidate"].astype(bool).to_numpy()
    trigger_tail = (
        (absr >= abs_q99)
        | (resid >= pos_q99)
        | (resid <= -abs_q99)
        | (oe >= oe_trig)
        | anomaly
    )

    eligible = eligible_scope & trigger_tail

    # Two-sided safe clamp
    exp_guard = exp.copy()
    exp_guard[eligible] = np.minimum(
        obs[eligible],
        np.maximum(exp[eligible], (obs[eligible] - cap))
    )
    exp_guard[eligible] = np.minimum(exp_guard[eligible], obs[eligible])

    res_guard = obs - exp_guard
    abs_guard = np.abs(res_guard)
    oe_guard = obs / np.maximum(exp_guard, EPS)

    changed_mask = eligible & (~np.isclose(exp_guard, exp, atol=0, rtol=0, equal_nan=True))

    out["expected_cost_guard"] = exp_guard
    out["residual_guard"] = res_guard
    out["abs_residual_guard"] = abs_guard
    out["oe_ratio_guard"] = oe_guard

    # Added later to standardize with other apply functions
    out["guardrail_applied"] = changed_mask

    name_val = str(policy.get("name", "C5v2"))
    if "guardrail_name" not in out.columns:
        out["guardrail_name"] = None
    out.loc[changed_mask & out["guardrail_name"].isna(), "guardrail_name"] = name_val

    return out