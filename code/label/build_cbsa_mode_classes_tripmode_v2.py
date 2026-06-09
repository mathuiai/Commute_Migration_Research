"""Build CBSA 3-class commute labels with official B08006 trip-mode groups."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
SHARED_DIR = SCRIPT_DIR.parent / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from label_schema import (  # noqa: E402
    TRIPMODE_V2_SEMANTICS,
    add_cbsa_tripmode_v2_shares,
    export_label_contract,
)


OFFICIAL_B08006_SOURCE = "https://api.census.gov/data/2023/acs/acs5/groups/B08006.html"
TARGET_COLUMNS = ["Car_share", "Transit_share", "NonTransit_share"]
COUNT_COLUMNS = ["Car_count", "Transit_count", "NonTransit_count"]


def _safe_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def _first_existing(df: pd.DataFrame, names: list[str]) -> str:
    for name in names:
        if name in df.columns:
            return name
    raise KeyError(f"Missing required column, tried: {names}")


def _weighted_share(df: pd.DataFrame, count_cols: list[str]) -> dict:
    totals = {c: float(df[c].sum()) for c in count_cols}
    denom = sum(totals.values())
    if denom <= 0:
        return {c.replace("_count", ""): 0.0 for c in count_cols}
    return {c.replace("_count", ""): totals[c] / denom for c in count_cols}


def build_tripmode_v2_labels(
    input_csv: Path,
    output_dir: Path,
    min_total_commuters: float = 10.0,
    min_tripmode_denominator: float = 10.0,
) -> dict:
    df = pd.read_csv(input_csv, dtype={"GEOID": str, "cbsa_code": str})
    total_col = _first_existing(df, ["001E", "B08006_001E"])
    df["total_commuters"] = _safe_num(df[total_col])

    scored = add_cbsa_tripmode_v2_shares(df)
    scored["share_sum"] = scored[TARGET_COLUMNS].sum(axis=1)

    keep_mask = (
        (scored["total_commuters"] >= float(min_total_commuters))
        & (scored["tripmode_denominator"] >= float(min_tripmode_denominator))
    )
    filtered_df = scored[keep_mask].copy()
    removed_df = scored[~keep_mask].copy()

    key_cols = ["GEOID", "cbsa_code", "cbsa_title", "county_fips", "NAME", "total_commuters"]
    out_cols = [
        c for c in key_cols if c in filtered_df.columns
    ] + COUNT_COLUMNS + [
        "ExcludedWFH_count",
        "class3_denominator",
        "tripmode_denominator",
        "share_sum",
    ] + TARGET_COLUMNS + [
        "label_schema",
        "label_version",
        "wfh_policy",
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    class3_out = output_dir / "cbsa_mode_3class_tripmode_v2_ge10.csv"
    filtered_out = output_dir / "cbsa_labels_tripmode_v2_filtered_ge10.csv"
    removed_out = output_dir / "cbsa_labels_tripmode_v2_removed.csv"
    summary_out = output_dir / "summary.json"

    filtered_df[out_cols].to_csv(class3_out, index=False, encoding="utf-8-sig")
    filtered_df.to_csv(filtered_out, index=False, encoding="utf-8-sig")
    removed_df.to_csv(removed_out, index=False, encoding="utf-8-sig")

    summary = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "official_source": OFFICIAL_B08006_SOURCE,
        "label_contract": export_label_contract(TRIPMODE_V2_SEMANTICS),
        "formula": {
            "Car_share": "B08006_002E / (B08006_002E + B08006_008E + B08006_014E + B08006_015E + B08006_016E)",
            "Transit_share": "B08006_008E / tripmode_denominator",
            "NonTransit_share": "(B08006_014E + B08006_015E + B08006_016E) / tripmode_denominator",
            "wfh_policy": "B08006_017E is excluded from the three-class denominator.",
        },
        "min_total_commuters": float(min_total_commuters),
        "min_tripmode_denominator": float(min_tripmode_denominator),
        "total_column_used": total_col,
        "rows_before_filter": int(len(scored)),
        "rows_after_filter": int(len(filtered_df)),
        "rows_removed": int(len(removed_df)),
        "weighted_share": _weighted_share(filtered_df, COUNT_COLUMNS),
        "share_sum_max_abs_error": float((filtered_df["share_sum"] - 1.0).abs().max()) if len(filtered_df) else None,
        "files": {
            "class3": str(class3_out),
            "filtered_full": str(filtered_out),
            "removed": str(removed_out),
            "summary": str(summary_out),
        },
    }

    with summary_out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    default_workspace = Path(r"C:\Users\Admin\Desktop\ShiXiaoYu\codes\acsDataDownload")
    parser = argparse.ArgumentParser(description="Build CBSA trip-mode v2 labels")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=default_workspace / "datasets" / "CBSA" / "LabelData" / "cbsa_tract_labels_2023.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_workspace / "datasets" / "CBSA_mode_classes_tripmode_v2_20260605",
    )
    parser.add_argument("--min-total-commuters", type=float, default=10.0)
    parser.add_argument("--min-tripmode-denominator", type=float, default=10.0)
    args = parser.parse_args()

    summary = build_tripmode_v2_labels(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        min_total_commuters=args.min_total_commuters,
        min_tripmode_denominator=args.min_tripmode_denominator,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


