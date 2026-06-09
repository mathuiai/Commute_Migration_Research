"""Shared label semantics and preflight checks for commute targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LabelSemantics:
    contract_name: str = "commute_overlap_contract"
    contract_version: str = "v1.0"
    contract_mode: str = "overlap_allowed"
    total_col: str = "001E"
    overlap_allowed_targets: tuple[str, ...] = ()
    non_overlap_upper: float = 1.0
    overlap_upper: float = 2.0



TRIPMODE_V2_SEMANTICS = LabelSemantics(
    contract_name="commute_tripmode3_contract",
    contract_version="v2.0",
    contract_mode="mutually_exclusive_trip_modes_excluding_wfh",
    total_col="tripmode_denominator",
    overlap_allowed_targets=(),
    non_overlap_upper=1.0,
    overlap_upper=1.0,
)


def export_label_contract(semantics: LabelSemantics = TRIPMODE_V2_SEMANTICS) -> Dict[str, object]:
    return {
        "contract_name": semantics.contract_name,
        "contract_version": semantics.contract_version,
        "contract_mode": semantics.contract_mode,
        "total_col": semantics.total_col,
        "overlap_allowed_targets": list(semantics.overlap_allowed_targets),
        "non_overlap_upper": semantics.non_overlap_upper,
        "overlap_upper": semantics.overlap_upper,
    }


def _to_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, (int, float, np.floating, np.integer)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def ensure_geoid(df: pd.DataFrame, geoid_col: str = "GEOID") -> pd.DataFrame:
    if geoid_col in df.columns:
        df = df.copy()
        df[geoid_col] = df[geoid_col].astype(str).str.zfill(11)
    return df


def compute_targets_from_counts(
    df: pd.DataFrame,
    target_config: Mapping[str, Mapping[str, str]],
    semantics: LabelSemantics = TRIPMODE_V2_SEMANTICS,
) -> pd.DataFrame:
    """Compute target ratios from raw numerator columns and total commute column."""
    if semantics.total_col not in df.columns:
        raise KeyError(f"Missing total column: {semantics.total_col}")

    out = df.copy()
    out["total_commute"] = pd.to_numeric(out[semantics.total_col], errors="coerce")
    denom = out["total_commute"].replace(0, np.nan)

    for target, cfg in target_config.items():
        numerator_expr = cfg.get("numerator_col")
        if not numerator_expr:
            raise KeyError(f"Target {target} missing numerator_col in config")

        if "+" in numerator_expr:
            cols = [c.strip() for c in numerator_expr.split("+")]
            missing = [c for c in cols if c not in out.columns]
            if missing:
                raise KeyError(f"Target {target} missing numerator columns: {missing}")
            numerator = out[cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
        else:
            if numerator_expr not in out.columns:
                raise KeyError(f"Target {target} missing numerator column: {numerator_expr}")
            numerator = pd.to_numeric(out[numerator_expr], errors="coerce")

        out[target] = numerator / denom

    return out


# CBSA_MODE3_COUNT_COLS removed; v2 uses CBSA_TRIPMODE_V2_SOURCE_COLS
CBSA_TRIPMODE_V2_SOURCE_COLS = ("002E", "008E", "014E", "015E", "016E", "017E")


def _resolve_count_col(df: pd.DataFrame, short_col: str) -> str:
    prefixed = f"B08006_{short_col}"
    if short_col in df.columns:
        return short_col
    if prefixed in df.columns:
        return prefixed
    raise KeyError(f"缺少 ACS B08006 计数列: {short_col} / {prefixed}")


def add_cbsa_tripmode_v2_shares(df: pd.DataFrame) -> pd.DataFrame:
    """
    CBSA 三分类通勤方式 v2 标签：
    Car = B08006_002E
    Transit = B08006_008E
    NonTransit = B08006_014E + B08006_015E + B08006_016E

    该口径排除 B08006_017E worked from home，分母为三类出行方式之和。
    """
    out = df.copy()
    col_map = {short: _resolve_count_col(out, short) for short in CBSA_TRIPMODE_V2_SOURCE_COLS}
    for c in col_map.values():
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    car = out[col_map["002E"]]
    transit = out[col_map["008E"]]
    non_tr = out[col_map["014E"]] + out[col_map["015E"]] + out[col_map["016E"]]
    wfh = out[col_map["017E"]]
    denom = car + transit + non_tr
    safe = denom.replace(0, np.nan)

    out["Car_count"] = car
    out["Transit_count"] = transit
    out["NonTransit_count"] = non_tr
    out["ExcludedWFH_count"] = wfh
    out["class3_denominator"] = denom
    out["tripmode_denominator"] = denom
    out["Car_share"] = (car / safe).fillna(0.0)
    out["Transit_share"] = (transit / safe).fillna(0.0)
    out["NonTransit_share"] = (non_tr / safe).fillna(0.0)
    out["label_schema"] = TRIPMODE_V2_SEMANTICS.contract_name
    out["label_version"] = TRIPMODE_V2_SEMANTICS.contract_version
    out["wfh_policy"] = "excluded"

    total_col = None
    for candidate in ("001E", "B08006_001E", "total_commuters", "total_commute"):
        if candidate in out.columns:
            total_col = candidate
            break
    if total_col is not None:
        out["total_commute"] = pd.to_numeric(out[total_col], errors="coerce")
    else:
        out["total_commute"] = denom + wfh

    return out


def run_label_preflight(
    df: pd.DataFrame,
    target_names: Iterable[str],
    semantics: LabelSemantics = TRIPMODE_V2_SEMANTICS,
    city_hint: str | None = None,
    strict: bool = False,
) -> Dict[str, object]:
    """Validate label contract before train/inference. Raises on hard violations."""
    target_names = list(target_names)
    missing = [t for t in target_names if t not in df.columns]
    if missing:
        raise ValueError(f"Missing target columns: {missing}")

    if "total_commute" not in df.columns:
        raise ValueError("Missing total_commute column for preflight checks")

    target_df = df[target_names].apply(pd.to_numeric, errors="coerce")
    total_commute = pd.to_numeric(df["total_commute"], errors="coerce")

    nan_ratio = float(target_df.isna().any(axis=1).mean())
    negative_ratio = float((target_df < 0).any(axis=1).mean())
    total_le0_ratio = float((total_commute <= 0).mean())

    upper_violations: Dict[str, float] = {}
    for t in target_names:
        upper = semantics.overlap_upper if t in semantics.overlap_allowed_targets else semantics.non_overlap_upper
        upper_violations[t] = float((target_df[t] > upper).mean())

    sum_targets = target_df.sum(axis=1, skipna=True)
    report: Dict[str, object] = {
        "label_contract": export_label_contract(semantics),
        "city": city_hint,
        "n_rows": int(len(df)),
        "nan_ratio": nan_ratio,
        "negative_ratio": negative_ratio,
        "total_commute_le0_ratio": total_le0_ratio,
        "sum_mean": float(sum_targets.mean()),
        "sum_p50": float(sum_targets.quantile(0.5)),
        "sum_p95": float(sum_targets.quantile(0.95)),
        "sum_lt_0_8_ratio": float((sum_targets < 0.8).mean()),
        "sum_gt_1_2_ratio": float((sum_targets > 1.2).mean()),
        "upper_bound_violation_ratio": upper_violations,
    }

    hard_errors: List[str] = []
    if report["n_rows"] == 0:
        hard_errors.append("empty dataset")
    if total_le0_ratio >= 0.2:
        hard_errors.append(f"too many non-positive total_commute rows: {total_le0_ratio:.3f}")
    if negative_ratio > 0:
        hard_errors.append(f"negative target ratio detected: {negative_ratio:.3f}")

    non_overlap_bad = {k: v for k, v in upper_violations.items() if k not in semantics.overlap_allowed_targets and v > 0}
    if non_overlap_bad:
        hard_errors.append(f"non-overlap target > 1 violations: {non_overlap_bad}")

    if hard_errors:
        raise ValueError("Label preflight failed: " + "; ".join(hard_errors))

    warnings: List[str] = []
    if nan_ratio > 0:
        warnings.append(f"target NaN ratio={nan_ratio:.3f}")
    if total_le0_ratio > 0:
        warnings.append(f"total_commute<=0 ratio={total_le0_ratio:.3f}")
    sum_gt_ratio = _to_float(report["sum_gt_1_2_ratio"])
    if sum_gt_ratio > 0.3:
        warnings.append(
            f"sum(targets)>1.2 ratio is high ({sum_gt_ratio:.3f}); keep prediction postprocess as 'none'"
        )

    report["warnings"] = warnings
    report["strict"] = strict
    return report


def format_preflight_report(report: Mapping[str, object]) -> str:
    warnings_obj = report.get("warnings")
    warnings: List[str] = []
    if isinstance(warnings_obj, list):
        warnings = [str(w) for w in warnings_obj]
    warning_text = " | ".join(warnings) if warnings else "none"

    nan_ratio = _to_float(report.get("nan_ratio", 0.0), default=0.0)
    total_le0_ratio = _to_float(report.get("total_commute_le0_ratio", 0.0), default=0.0)
    sum_gt_ratio = _to_float(report.get("sum_gt_1_2_ratio", 0.0), default=0.0)
    return (
        f"n={report.get('n_rows')} "
        f"nan={nan_ratio:.3f} "
        f"total<=0={total_le0_ratio:.3f} "
        f"sum>1.2={sum_gt_ratio:.3f} "
        f"warnings={warning_text}"
    )
