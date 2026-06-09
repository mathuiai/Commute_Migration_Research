"""Pipeline-level contracts: label semantics and feature manifest verification."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from typing import Dict, Optional


FEATURE_DIM_BY_TYPE = {
    "single_task_finetune": 128,
    "imagenet_pretrain": 2048,
    "satellite_pretrain": 2048,
    "multi_task_finetune": 2048,
    "single_task_backbone_finetune": 2048,
    "cbsa_commute_satellite_finetune": 2048,
    "simclr_vit_l16": 1024,
}


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def sha256_file(file_path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def feature_manifest_path(feature_root: str) -> str:
    return os.path.join(feature_root, "feature_manifest.json")


def write_feature_manifest(
    *,
    feature_root: str,
    feature_type: str,
    feature_dim: int,
    extractor_name: str,
    source_weight_path: Optional[str],
    image_size: int,
    note: str = "",
) -> str:
    os.makedirs(feature_root, exist_ok=True)

    source_weight_sha256 = None
    if source_weight_path and os.path.exists(source_weight_path):
        source_weight_sha256 = sha256_file(source_weight_path)

    payload = {
        "feature_type": feature_type,
        "feature_dim": int(feature_dim),
        "extractor_name": extractor_name,
        "source_weight_path": source_weight_path,
        "source_weight_sha256": source_weight_sha256,
        "image_size": int(image_size),
        "created_at": now_str(),
        "note": note,
    }

    manifest_path = feature_manifest_path(feature_root)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return manifest_path


def read_feature_manifest(feature_root: str) -> Dict[str, object]:
    path = feature_manifest_path(feature_root)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"缺少特征清单文件: {path}。请先运行 Step2/Step2i/Step2ii 重新生成该特征目录。"
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def verify_feature_manifest(feature_root: str, feature_type: str, expected_dim: Optional[int] = None) -> Dict[str, object]:
    payload = read_feature_manifest(feature_root)

    actual_type = str(payload.get("feature_type", ""))
    if actual_type != feature_type:
        raise RuntimeError(
            f"特征类型不匹配: 目录={feature_root}, manifest={actual_type}, 期望={feature_type}"
        )

    if expected_dim is None:
        expected_dim = FEATURE_DIM_BY_TYPE.get(feature_type)

    actual_dim_obj = payload.get("feature_dim")
    if isinstance(actual_dim_obj, int):
        actual_dim = actual_dim_obj
    elif isinstance(actual_dim_obj, str):
        actual_dim = int(actual_dim_obj)
    elif isinstance(actual_dim_obj, float):
        actual_dim = int(actual_dim_obj)
    else:
        raise RuntimeError(f"特征清单缺少有效 feature_dim: {actual_dim_obj}")

    if expected_dim is not None and actual_dim != int(expected_dim):
        raise RuntimeError(
            f"特征维度不匹配: 目录={feature_root}, manifest={actual_dim}, 期望={expected_dim}"
        )

    return payload
