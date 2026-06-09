import json
import os
import time
from datetime import datetime
from typing import Any, Dict, Optional

from config import DATA_ROOT, FEATURES_ROOT, TEST_OUTPUT_ROOT


REGISTRY_PATH = os.path.join(DATA_ROOT, "FineTuneResNet", "experiment_registry.json")


def _ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _load_registry() -> Dict[str, Any]:
    if not os.path.exists(REGISTRY_PATH):
        return {"pipeline_runs": [], "batch_manifests": []}

    try:
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = {"pipeline_runs": [], "batch_manifests": []}

    if "pipeline_runs" not in data:
        data["pipeline_runs"] = []
    if "batch_manifests" not in data:
        data["batch_manifests"] = []
    return data


def _save_registry(data: Dict[str, Any]) -> None:
    _ensure_parent(REGISTRY_PATH)
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _dir_stats(root_dir: str) -> Dict[str, Any]:
    if not os.path.exists(root_dir):
        return {
            "exists": False,
            "total_files": 0,
            "total_dirs": 0,
            "total_size_bytes": 0,
            "top_level": [],
        }

    total_files = 0
    total_dirs = 0
    total_size = 0

    for cur, dirs, files in os.walk(root_dir):
        total_dirs += len(dirs)
        total_files += len(files)
        for name in files:
            p = os.path.join(cur, name)
            try:
                total_size += os.path.getsize(p)
            except OSError:
                pass

    top_level = []
    for name in sorted(os.listdir(root_dir)):
        p = os.path.join(root_dir, name)
        top_level.append({"name": name, "is_dir": os.path.isdir(p)})

    return {
        "exists": True,
        "total_files": total_files,
        "total_dirs": total_dirs,
        "total_size_bytes": total_size,
        "top_level": top_level,
    }


def write_batch_manifest(
    batch_dir: str,
    batch_name: Optional[str] = None,
    note: Optional[str] = None,
    source: str = "manual",
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    batch_name = batch_name or os.path.basename(batch_dir.rstrip("\\/"))
    manifest_path = os.path.join(batch_dir, "manifest.json")

    payload: Dict[str, Any] = {
        "batch_name": batch_name,
        "batch_dir": batch_dir,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": source,
        "note": note or "",
        "stats": _dir_stats(batch_dir),
        "key_subdirs": {
            "single_task_finetune": _dir_stats(os.path.join(batch_dir, "single_task_finetune")),
            "multi_task_finetune": _dir_stats(os.path.join(batch_dir, "multi_task_finetune")),
        },
    }
    if extra:
        payload["extra"] = extra

    _ensure_parent(manifest_path)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    reg = _load_registry()
    reg["batch_manifests"].append(
        {
            "batch_name": batch_name,
            "batch_dir": batch_dir,
            "manifest_path": manifest_path,
            "created_at": payload["created_at"],
            "source": source,
        }
    )
    _save_registry(reg)

    return manifest_path


def record_pipeline_run(
    *,
    step: str,
    finetune_mode: str,
    feature_type: str,
    compare_feature_types: Optional[list],
    aux_loss_weight: float,
    elapsed_seconds: float,
    status: str,
    error_message: Optional[str] = None,
) -> str:
    reg = _load_registry()
    reg["pipeline_runs"].append(
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "step": step,
            "finetune_mode": finetune_mode,
            "feature_type": feature_type,
            "compare_feature_types": compare_feature_types or [],
            "aux_loss_weight": aux_loss_weight,
            "elapsed_seconds": round(float(elapsed_seconds), 3),
            "status": status,
            "error_message": error_message or "",
            "paths_snapshot": {
                "features_root": FEATURES_ROOT,
                "features_root_exists": os.path.exists(FEATURES_ROOT),
                "test_output_root": TEST_OUTPUT_ROOT,
                "test_output_root_exists": os.path.exists(TEST_OUTPUT_ROOT),
            },
        }
    )
    _save_registry(reg)
    return REGISTRY_PATH


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="实验追踪工具：写入批次 manifest")
    parser.add_argument("--batch-dir", type=str, required=True, help="批次目录路径")
    parser.add_argument("--batch-name", type=str, default=None, help="批次名称")
    parser.add_argument("--note", type=str, default="", help="补充说明")
    args = parser.parse_args()

    p = write_batch_manifest(
        batch_dir=args.batch_dir,
        batch_name=args.batch_name,
        note=args.note,
        source="cli",
        extra={"generated_via": "experiment_tracking.py"},
    )
    print(f"manifest written: {p}")
