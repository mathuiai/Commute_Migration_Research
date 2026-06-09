# =====================================================================
# Step3T 集成包教师封装器
# 用 Step3T 全局集成模型（LightGBM+XGBoost+CatBoost）作为 per-cluster 软标签来源
# 替代原来 Step6 内部训练的单 LightGBM 教师模型
# =====================================================================

from __future__ import annotations

import os
import sys
from typing import Dict, Optional

import numpy as np
import pandas as pd
import joblib

try:
    from .config import TARGET_NAMES, MODEL_SAVE_DIR
except ImportError:
    from config import TARGET_NAMES, MODEL_SAVE_DIR


class Step3TBundleTeacher:
    """
    用 Step3T 全局集成包为每个 city-cluster 生成软标签。

    不在 Step6 内部重新训练教师模型，而是直接调用 Step3T 优化好的
    全局集成（LightGBM + XGBoost + CatBoost）对每个 cluster 的表格特征做推理。

    Args:
        bundle_path: Step3T 生成的 .joblib bundle 路径
        tabular_path: CBSA 原始表格特征 CSV 路径（不含图像特征列）
    """

    def __init__(self, bundle_path: str, tabular_path: str):
        # 将 codes/FineTuneResNet 加入 sys.path 以import step3t模块
        code_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if code_root not in sys.path:
            sys.path.insert(0, code_root)

        # 导入 Step3T 的 TeacherEnsembleBundle 和 predict_teacher_bundle
        from step3t_cbsa_teacher_ensemble_opt import (
            TeacherEnsembleBundle,
            predict_teacher_bundle,
        )

        # 注册到 __main__ 以保证 joblib 能正确反序列化
        import __main__
        setattr(__main__, "TeacherEnsembleBundle", TeacherEnsembleBundle)

        self._predict_teacher_bundle = predict_teacher_bundle

        print(f"  加载 Step3T 集成包: {bundle_path}")
        self.bundle = joblib.load(bundle_path)
        print(f"  集成包加载成功 (targets: {list(self.bundle.models_by_target.keys())})")

        # 加载原始表格特征（仅表格列，不含 img_feat_*）
        print(f"  加载表格数据: {tabular_path}")
        tab_df = pd.read_csv(tabular_path)
        tab_df["GEOID"] = tab_df["GEOID"].astype(str).str.zfill(11)

        # 过滤掉图像特征列（如误传入含图像的联合文件）
        img_cols = [c for c in tab_df.columns if c.startswith("img_feat_")]
        if img_cols:
            tab_df = tab_df.drop(columns=img_cols)
            print(f"  已过滤 {len(img_cols)} 个 img_feat_* 列")

        self.tabular_df = tab_df.set_index("GEOID")
        print(f"  表格数据行数: {len(self.tabular_df)}, 列数: {len(self.tabular_df.columns)}")

    def generate_all_soft_labels(
        self,
        cluster_datasets: Dict[str, Dict],
        save_dir: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """
        为所有 cluster 生成 Step3T 软标签。

        Returns:
            {cluster_id: np.ndarray shape (n_samples, n_targets)}
            与 teacher_trainer.py 的 all_soft_labels 格式完全相同。
        """
        print("\n" + "=" * 60)
        print("Step3T Bundle: 为各 Cluster 推理软标签...")
        print("=" * 60)

        all_soft_labels: Dict[str, np.ndarray] = {}

        for cluster_id, dataset in cluster_datasets.items():
            geoids = dataset["geoids"]
            n = len(geoids)

            # 初始化为硬标签（fallback）
            soft = np.copy(dataset["y"]).astype(np.float64)

            # 找出在 tabular_df 中有记录的 GEOID
            found_mask = np.array([g in self.tabular_df.index for g in geoids])
            found_indices = np.where(found_mask)[0]
            found_geoids = [geoids[i] for i in found_indices]

            missing_count = n - len(found_geoids)
            if missing_count > 0:
                print(
                    f"  警告: {cluster_id} 中 {missing_count}/{n} 个 GEOID "
                    f"未在表格数据中找到，对应行使用硬标签"
                )

            if not found_geoids:
                print(f"  警告: {cluster_id} 无可用表格数据，全部使用硬标签")
                all_soft_labels[cluster_id] = soft
                continue

            # 取出表格子集
            tab_sub = self.tabular_df.loc[found_geoids].reset_index()
            if "GEOID" not in tab_sub.columns:
                tab_sub = tab_sub.rename(columns={"index": "GEOID"})

            # Step3T 集成推理：返回 pred_df 列 = TARGET_NAMES
            pred_df, _ = self._predict_teacher_bundle(self.bundle, tab_sub)

            # 填充软标签
            for t_idx, t_name in enumerate(TARGET_NAMES):
                if t_name in pred_df.columns:
                    soft[found_indices, t_idx] = pred_df[t_name].values.astype(np.float64)

            all_soft_labels[cluster_id] = soft
            print(
                f"  {cluster_id}: {len(found_geoids)}/{n} 样本由 Step3T Bundle 生成软标签"
            )

        # 保存到磁盘
        out_dir = save_dir or MODEL_SAVE_DIR
        os.makedirs(out_dir, exist_ok=True)
        soft_labels_path = os.path.join(out_dir, "step3t_bundle_soft_labels.pkl")
        joblib.dump(all_soft_labels, soft_labels_path)
        print(f"\n  Step3T 软标签已保存: {soft_labels_path}")
        print("=" * 60)

        return all_soft_labels
