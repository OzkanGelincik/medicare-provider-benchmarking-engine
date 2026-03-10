# src/bench/guardrails_artifacts.py
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd

from . import guardrails_impl


# -----------------------------
# Helpers: JSON-safe conversion
# -----------------------------
def _json_safe(obj: Any) -> Any:
    """
    Convert common non-JSON types (numpy scalars/arrays, pandas types, sets, etc.)
    into JSON-serializable Python types.
    """
    # Numpy scalars
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    # Pandas / numpy NaN handling
    if obj is None:
        return None
    try:
        if pd.isna(obj):  # covers np.nan, pd.NA
            return None
    except Exception:
        pass

    # Containers
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, set):
        return [_json_safe(v) for v in sorted(list(obj))]
    if isinstance(obj, np.ndarray):
        return [_json_safe(v) for v in obj.tolist()]

    # Basic JSON types pass through
    if isinstance(obj, (str, int, float, bool)):
        return obj

    # Last-resort stringify
    return str(obj)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Registry (keys -> callables)
# -----------------------------
def build_guardrail_fn_registry() -> Dict[str, Callable]:
    """
    Stable keys are what you store in JSON. Values are real callables.

    This registry is the bridge between:
      - serialized spec (strings)
      - executable guardrail set (python functions)
    """
    reg = {
        "apply_guardrail_v0_A1c_2023_only": guardrails_impl.apply_guardrail_v0_A1c_2023_only,
        "apply_guardrail_B2_2b": guardrails_impl.apply_guardrail_B2_2b,
        "apply_guardrail_B2_2c": guardrails_impl.apply_guardrail_B2_2c,
        "apply_guardrail_C4a_J2469_uplift": guardrails_impl.apply_guardrail_C4a_J2469_uplift,
        "apply_guardrail_C4b_J2505_resid_cap": guardrails_impl.apply_guardrail_C4b_J2505_resid_cap,
        "apply_guardrail_C5v2_radiation_planning_resid_cap": guardrails_impl.apply_guardrail_C5v2_radiation_planning_resid_cap,
    }
    return reg


# -----------------------------
# Spec builder + save/load
# -----------------------------
def build_guardrail_spec(
    guardrail_set: dict,
    fn_registry: Dict[str, Callable],
    *,
    version: str,
    thresholds_version: str,
    created_at_utc: Optional[str] = None,
    git_commit: Optional[str] = None,
    notes: Optional[str] = None,
) -> dict:
    """
    Build a JSON-serializable spec:
      - stores apply_fn_key instead of apply_fn callable
      - stores policy as pure dict
    """
    if "components" not in guardrail_set:
        raise KeyError("guardrail_set missing 'components'")

    # reverse lookup callable -> key
    reverse = {fn: key for key, fn in fn_registry.items()}

    created_at_utc = created_at_utc or datetime.now(timezone.utc).isoformat()

    spec_components = []
    for comp in guardrail_set["components"]:
        if "name" not in comp or "policy" not in comp or "apply_fn" not in comp:
            raise KeyError(f"Component is missing required keys: {list(comp.keys())}")

        fn = comp["apply_fn"]
        if fn not in reverse:
            raise KeyError(
                f"apply_fn for component '{comp.get('name')}' is not in registry.\n"
                f"Function name: {getattr(fn, '__name__', str(fn))}\n"
                f"Registry keys: {list(fn_registry.keys())}"
            )

        spec_components.append(
            {
                "component_name": str(comp["name"]),
                "apply_fn_key": reverse[fn],
                "policy": _json_safe(comp["policy"]),
                "notes": None,
            }
        )

    spec = {
        "schema_version": 1,
        "version": str(version),
        "name": str(guardrail_set.get("name", "")),
        "phase": str(guardrail_set.get("phase", "")),
        "created_at_utc": created_at_utc,
        "git_commit": git_commit,
        "thresholds_version": str(thresholds_version),
        "notes": notes,
        "components": spec_components,
    }
    return spec


def save_guardrail_spec(
    spec: dict,
    thresholds: dict,
    *,
    out_dir: Path,
    spec_filename: str = "guardrail_v3_spec.json",
    thresholds_filename: str = "thresholds_v2.json",
) -> Dict[str, str]:
    """
    Write spec + thresholds JSON to disk. Returns paths.
    """
    _ensure_dir(out_dir)

    spec_path = out_dir / spec_filename
    thr_path = out_dir / thresholds_filename

    with open(spec_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(spec), f, indent=2, sort_keys=False)

    with open(thr_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(thresholds), f, indent=2, sort_keys=False)

    return {"spec_path": str(spec_path), "thresholds_path": str(thr_path)}


def load_guardrail_spec(
    spec_path: Path,
    thresholds_path: Path,
    fn_registry: Dict[str, Callable],
) -> Dict[str, Any]:
    """
    Load spec + thresholds, and rehydrate into a guardrail dict:
      guardrail = {"name","phase","components":[{"name","policy","apply_fn"}...]}
    """
    with open(spec_path, "r", encoding="utf-8") as f:
        spec = json.load(f)

    with open(thresholds_path, "r", encoding="utf-8") as f:
        thresholds = json.load(f)

    missing_keys = [
        c["apply_fn_key"]
        for c in spec["components"]
        if c["apply_fn_key"] not in fn_registry
    ]
    if missing_keys:
        raise KeyError(
            "Spec references apply_fn_key(s) not found in registry: "
            + ", ".join(sorted(set(missing_keys)))
        )

    guardrail = {
        "name": spec.get("name"),
        "phase": spec.get("phase"),
        "components": [
            {
                "name": c["component_name"],
                "policy": c["policy"],
                "apply_fn": fn_registry[c["apply_fn_key"]],
            }
            for c in spec["components"]
        ],
        "_spec_meta": {
            "schema_version": spec.get("schema_version"),
            "version": spec.get("version"),
            "created_at_utc": spec.get("created_at_utc"),
            "git_commit": spec.get("git_commit"),
            "thresholds_version": spec.get("thresholds_version"),
            "notes": spec.get("notes"),
        },
    }

    return {"guardrail": guardrail, "thresholds": thresholds, "spec": spec}