"""Step6 runtime config: single entry for CLI/env overrides."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import os
from typing import Dict


@dataclass
class Step6RuntimeConfig:
    expert_feature_type: str = "single_task_backbone_finetune"
    expert_cluster_label_path: str | None = None
    expert_per_city_cluster_dir: str | None = None
    expert_model_save_dir: str | None = None
    expert_data_root: str | None = None
    expert_split_manifest_path: str | None = None
    expert_cbsa_label_path: str | None = None

    def apply_to_env(self) -> None:
        os.environ["EXPERT_FEATURE_TYPE"] = self.expert_feature_type
        if self.expert_cluster_label_path:
            os.environ["EXPERT_CLUSTER_LABEL_PATH"] = self.expert_cluster_label_path
        if self.expert_per_city_cluster_dir:
            os.environ["EXPERT_PER_CITY_CLUSTER_DIR"] = self.expert_per_city_cluster_dir
        if self.expert_model_save_dir:
            os.environ["EXPERT_MODEL_SAVE_DIR"] = self.expert_model_save_dir
        if self.expert_data_root:
            os.environ["EXPERT_DATA_ROOT"] = self.expert_data_root
        if self.expert_split_manifest_path:
            os.environ["EXPERT_SPLIT_MANIFEST_PATH"] = self.expert_split_manifest_path
        if self.expert_cbsa_label_path:
            os.environ["EXPERT_CBSA_LABEL_PATH"] = self.expert_cbsa_label_path

    def to_dict(self) -> Dict[str, str | None]:
        return asdict(self)
