# =====================================================================
# 模块5: 推理阶段接口（纯图像输入）
# =====================================================================

import os
import json
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm

try:
    from .config import (
        TARGET_NAMES, K_NEIGHBORS,
        STUDENT_MODEL_DIR, MULTITASK_STUDENT_DIR
    )
except ImportError:
    from config import (
        TARGET_NAMES, K_NEIGHBORS,
        STUDENT_MODEL_DIR, MULTITASK_STUDENT_DIR
    )


class ExpertEnsembleInference:
    """
    专家集成推理接口
    核心红线：全程纯图像输入，不触碰任何表格数据
    
    支持两种模式：
    1. 单任务学生模型（147个模型）
    2. 多任务模型（21个模型）
    """
    
    def __init__(self,
                 cluster_centers: Dict[str, np.ndarray],
                 model_type: str = 'student',  # 'student' or 'multitask-student'
                 model_dir: str = None,
                 k_neighbors: int = K_NEIGHBORS,
                 weighting_scope: str = 'knn',
                 prediction_postprocess: str = 'none',
                 distance_metric: str = 'cosine',
                 idw_power: float = 2.0,
                 perf_weight_mode: str = 'none',
                 teacher_report_path: str | None = None,
                 adaptive_similarity_threshold: float | None = None,
                 min_neighbors: int = 1):
        self.cluster_centers = cluster_centers
        self.model_type = model_type
        self.k_neighbors = k_neighbors
        self.weighting_scope = weighting_scope  # 'all' | 'knn'
        self.prediction_postprocess = prediction_postprocess  # 'none' | 'clip_non_negative' | 'clip01_renorm'
        self.distance_metric = str(distance_metric).strip().lower()  # 'l2' | 'cosine'
        self.idw_power = float(idw_power)
        self.perf_weight_mode = str(perf_weight_mode).strip().lower()  # 'none' | 'val_r2' | 'val_mae_inv'
        self.teacher_report_path = teacher_report_path
        self.adaptive_similarity_threshold = adaptive_similarity_threshold
        self.min_neighbors = max(1, int(min_neighbors))

        if self.distance_metric not in {'l2', 'cosine'}:
            raise ValueError(f"Unsupported distance_metric: {self.distance_metric}")
        if self.idw_power <= 0:
            raise ValueError(f"idw_power must be > 0, got {self.idw_power}")
        if self.perf_weight_mode not in {'none', 'val_r2', 'val_mae_inv'}:
            raise ValueError(f"Unsupported perf_weight_mode: {self.perf_weight_mode}")

        self.cluster_perf_weight = self._load_cluster_performance_weights(self.teacher_report_path, self.perf_weight_mode)
        
        # 加载模型
        if model_dir is None:
            if model_type == 'student':
                model_dir = STUDENT_MODEL_DIR
            elif model_type == 'multitask-student':
                model_dir = MULTITASK_STUDENT_DIR
            else:
                model_dir = MULTITASK_STUDENT_DIR  # 默认使用多任务学生
        self.models = self._load_models(model_dir)

    def _load_cluster_performance_weights(self, teacher_report_path: str | None, perf_weight_mode: str) -> Dict[str, float]:
        """从 teacher/training_report.json 读取 cluster 级性能权重。"""
        if perf_weight_mode == 'none':
            return {}
        if not teacher_report_path or not os.path.exists(teacher_report_path):
            raise FileNotFoundError(
                f"perf_weight_mode={perf_weight_mode} requires valid teacher_report_path, got: {teacher_report_path}"
            )

        with open(teacher_report_path, 'r', encoding='utf-8') as f:
            payload = json.load(f)

        rows = payload.get('results', [])
        if not rows:
            raise ValueError(f"No 'results' in teacher report: {teacher_report_path}")

        tmp: Dict[str, List[float]] = {}
        for r in rows:
            cid = str(r.get('cluster_id', '')).strip()
            if not cid:
                continue
            if perf_weight_mode == 'val_r2':
                v = float(r.get('val_r2', 0.0))
                # 负R2不赋予加成，避免差专家放大。
                score = max(0.0, v)
            else:
                v = float(r.get('val_mae', np.nan))
                if not np.isfinite(v):
                    continue
                score = 1.0 / max(v, 1e-6)
            tmp.setdefault(cid, []).append(score)

        if not tmp:
            raise ValueError(f"No usable cluster metrics in teacher report: {teacher_report_path}")

        cluster_scores = {cid: float(np.mean(vals)) for cid, vals in tmp.items() if len(vals) > 0}
        vals = np.array(list(cluster_scores.values()), dtype=float)
        if vals.size == 0:
            return {}

        s = float(vals.sum())
        if s <= 1e-12:
            n = len(cluster_scores)
            return {k: 1.0 / max(n, 1) for k in cluster_scores.keys()}
        return {k: float(v / s) for k, v in cluster_scores.items()}
        
    def _load_models(self, model_dir: str) -> Dict:
        """加载所有模型到内存"""
        models = {}
        expected_clusters = sorted(self.cluster_centers.keys())
        
        if self.model_type == 'student':
            # 单任务模型: 严格按当前 cluster_centers 期望集合加载，避免历史文件污染。
            missing = []
            for cluster_id in expected_clusters:
                for target_name in TARGET_NAMES:
                    model_file = f'student_{cluster_id}_{target_name}.pkl'
                    model_path = os.path.join(model_dir, model_file)
                    if not os.path.exists(model_path):
                        missing.append(model_file)
                        continue
                    models[f'{cluster_id}_{target_name}'] = joblib.load(model_path)

            if missing:
                preview = ', '.join(missing[:8])
                raise FileNotFoundError(
                    f"当前实验缺少学生模型文件({len(missing)}): {preview}"
                )

            expected_count = len(expected_clusters) * len(TARGET_NAMES)
            if len(models) != expected_count:
                raise RuntimeError(f"模型数量异常: expected={expected_count}, loaded={len(models)}")
            print(f"  加载了 {len(models)} 个单任务模型（严格匹配本轮cluster集合）")
        
        elif self.model_type == 'teacher':
            # 教师单任务模型: 与student加载逻辑一致，仅文件前缀不同
            missing = []
            for cluster_id in expected_clusters:
                for target_name in TARGET_NAMES:
                    model_file = f'teacher_{cluster_id}_{target_name}.pkl'
                    model_path = os.path.join(model_dir, model_file)
                    if not os.path.exists(model_path):
                        missing.append(model_file)
                        continue
                    models[f'{cluster_id}_{target_name}'] = joblib.load(model_path)

            if missing:
                preview = ', '.join(missing[:8])
                raise FileNotFoundError(
                    f'当前实验缺少教师模型文件({len(missing)}): {preview}'
                )

            expected_count = len(expected_clusters) * len(TARGET_NAMES)
            if len(models) != expected_count:
                raise RuntimeError(f'模型数量异常: expected={expected_count}, loaded={len(models)}')
            print(f'  加载了 {len(models)} 个教师单任务模型（严格匹配本轮cluster集合）')
        
        elif self.model_type in ['multitask', 'multitask-student']:
            # 多任务模型: 严格按当前 cluster_centers 期望集合加载。
            try:
                from .multitask_trainer import MultiTaskLGBM
            except ImportError:
                from multitask_trainer import MultiTaskLGBM

            missing = []
            for cluster_id in expected_clusters:
                candidate_files = [
                    os.path.join(model_dir, f'multitask_student_{cluster_id}.pkl'),
                    os.path.join(model_dir, f'multitask_{cluster_id}.pkl'),
                ]
                model_path = next((p for p in candidate_files if os.path.exists(p)), None)
                if model_path is None:
                    missing.append(f'multitask_student_{cluster_id}.pkl')
                    continue

                multitask_model = MultiTaskLGBM()
                multitask_model.load(model_path)
                models[cluster_id] = multitask_model

            if missing:
                preview = ', '.join(missing[:8])
                raise FileNotFoundError(
                    f"当前实验缺少多任务模型文件({len(missing)}): {preview}"
                )

            if len(models) != len(expected_clusters):
                raise RuntimeError(f"多任务模型数量异常: expected={len(expected_clusters)}, loaded={len(models)}")
            print(f"  加载了 {len(models)} 个多任务模型（严格匹配本轮cluster集合）")
        
        return models
    
    def find_cluster_distances(self, X_img: np.ndarray) -> List[Tuple[str, float]]:
        """计算测试样本到所有聚类中心的距离（升序）。"""
        distances = []
        
        for cluster_id, center in self.cluster_centers.items():
            if self.distance_metric == 'l2':
                dist = np.linalg.norm(X_img - center)
            else:
                # 余弦距离 = 1 - cos_sim，越小越相近。
                x_norm = np.linalg.norm(X_img)
                c_norm = np.linalg.norm(center)
                if x_norm <= 1e-12 or c_norm <= 1e-12:
                    dist = 1.0
                else:
                    cos_sim = float(np.dot(X_img, center) / (x_norm * c_norm))
                    cos_sim = float(np.clip(cos_sim, -1.0, 1.0))
                    dist = 1.0 - cos_sim
            distances.append((cluster_id, dist))
        
        distances.sort(key=lambda x: x[1])
        return distances

    def select_clusters_for_weighting(self, sorted_distances: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """根据加权范围选择参与IDW聚合的聚类集合。"""
        if self.weighting_scope == 'all':
            base = sorted_distances
        else:
            base = sorted_distances[:self.k_neighbors]

        if self.adaptive_similarity_threshold is None:
            return base

        selected = []
        thr = float(self.adaptive_similarity_threshold)
        for cid, dist in base:
            if self.distance_metric == 'cosine':
                sim = 1.0 - float(dist)
            else:
                sim = 1.0 / (float(dist) + 1e-6)
            if sim >= thr:
                selected.append((cid, dist))

        if len(selected) >= self.min_neighbors:
            return selected
        return base[:max(self.min_neighbors, min(len(base), self.k_neighbors))]

    def build_idw_weights(self, cluster_distance_pairs: List[Tuple[str, float]]) -> Dict[str, float]:
        """构建 cluster_id -> IDW 权重。"""
        if not cluster_distance_pairs:
            return {}

        d = np.array([dist for _, dist in cluster_distance_pairs], dtype=float)
        d = np.maximum(d, 1e-6)
        inv = (1.0 / d) ** self.idw_power
        w = inv / max(np.sum(inv), 1e-12)
        w_map = {cluster_distance_pairs[i][0]: float(w[i]) for i in range(len(cluster_distance_pairs))}

        if self.perf_weight_mode != 'none':
            for cid in list(w_map.keys()):
                perf = self.cluster_perf_weight.get(cid, 0.0)
                w_map[cid] = float(w_map[cid] * perf)
            s = float(sum(w_map.values()))
            if s > 1e-12:
                for cid in list(w_map.keys()):
                    w_map[cid] = float(w_map[cid] / s)
            else:
                # 当性能权重全为0时，回退到原始IDW。
                w_map = {cluster_distance_pairs[i][0]: float(w[i]) for i in range(len(cluster_distance_pairs))}

        return w_map
    
    def aggregate_single_task_predictions(
        self,
        X_img: np.ndarray,
        cluster_weights: Dict[str, float],
    ) -> np.ndarray:
        """
        单任务学生模型聚合：
        - 每个目标由 21 个 cluster 专家分别给出预测
        - 使用对应 cluster 的 IDW 权重聚合
        - 等价于按 147 个专家模型的原型相似度做加权
        """
        pred_vector = np.zeros(len(TARGET_NAMES), dtype=float)

        for target_idx, target_name in enumerate(TARGET_NAMES):
            w_sum = 0.0
            y_sum = 0.0
            for cluster_id, w in cluster_weights.items():
                model_key = f"{cluster_id}_{target_name}"
                model = self.models.get(model_key)
                if model is None:
                    continue

                pred = float(model.predict(X_img.reshape(1, -1))[0])
                y_sum += w * pred
                w_sum += w

            pred_vector[target_idx] = y_sum / max(w_sum, 1e-12)

        return pred_vector

    def aggregate_multitask_predictions(
        self,
        X_img: np.ndarray,
        cluster_weights: Dict[str, float],
    ) -> np.ndarray:
        """多任务学生模型聚合：每个 cluster 一个 7 目标模型。"""
        y_sum = np.zeros(len(TARGET_NAMES), dtype=float)
        w_sum = 0.0

        for cluster_id, w in cluster_weights.items():
            multitask_model = self.models.get(cluster_id)
            if multitask_model is None:
                continue
            pred = np.asarray(multitask_model.predict(X_img.reshape(1, -1))[0], dtype=float)
            y_sum += w * pred
            w_sum += w

        return y_sum / max(w_sum, 1e-12)
    
    def predict_single_student(self, X_img: np.ndarray) -> np.ndarray:
        """单任务学生模型预测。"""
        all_distances = self.find_cluster_distances(X_img)
        selected = self.select_clusters_for_weighting(all_distances)
        weights = self.build_idw_weights(selected)
        return self.aggregate_single_task_predictions(X_img, weights)
    

    def predict_single_teacher(self, X_img, X_tab=None):
        """教师单任务模型预测（需要表格+图像拼接）。"""
        if X_tab is None:
            raise ValueError("Teacher inference requires tabular features (X_tab)")
        # Handle 1D input by reshaping
        squeeze_out = X_img.ndim == 1
        if squeeze_out:
            X_img = X_img.reshape(1, -1)
            X_tab = X_tab.reshape(1, -1)
        all_distances = self.find_cluster_distances(X_img[0])
        selected = self.select_clusters_for_weighting(all_distances)
        X_teacher = np.concatenate([X_tab, X_img], axis=1)

        pred = np.zeros((X_img.shape[0], len(TARGET_NAMES)), dtype=np.float64)
        weight_sum = np.zeros(X_img.shape[0], dtype=np.float64)

        weights = self.build_idw_weights(selected)
        for cluster_id, w in weights.items():
            for t, target_name in enumerate(TARGET_NAMES):
                key = f'{cluster_id}_{target_name}'
                if key not in self.models:
                    continue
                model = self.models[key]
                y_pred = model.predict(X_teacher).ravel()
                pred[:, t] += w * y_pred
                weight_sum += w

        mask = weight_sum > 0
        for t in range(len(TARGET_NAMES)):
            pred[mask, t] /= weight_sum[mask]

        if squeeze_out:
            pred = pred[0]

        if self.prediction_postprocess != "none":
            pred = self.postprocess_predictions(pred)
        return pred
    def predict_single_multitask(self, X_img: np.ndarray) -> np.ndarray:
        """多任务学生模型预测。"""
        all_distances = self.find_cluster_distances(X_img)
        selected = self.select_clusters_for_weighting(all_distances)
        weights = self.build_idw_weights(selected)
        return self.aggregate_multitask_predictions(X_img, weights)
    
    def predict_single(self, X_img: np.ndarray) -> np.ndarray:
        """根据模型类型选择预测方法"""
        if self.model_type == "teacher":
            return self.predict_single_teacher(X_img)
        if self.model_type in ("student",):
            return self.predict_single_student(X_img)
        else:
            return self.predict_single_multitask(X_img)
    
    def postprocess_predictions(self, pred: np.ndarray) -> np.ndarray:
        """预测后处理（默认不做和为1约束，避免破坏非互斥标签口径）。"""
        mode = self.prediction_postprocess
        if mode == 'none':
            return pred
        if mode == 'clip_non_negative':
            return np.maximum(pred, 0.0)
        if mode == 'clip01_renorm':
            pred = np.clip(pred, 0, 1)
            return pred / pred.sum() if pred.sum() > 0 else pred
        raise ValueError(f"Unsupported prediction_postprocess: {mode}")
    
    def predict_batch(
        self,
        test_df: pd.DataFrame,
        img_feat_cols: List[str] = None
    ) -> pd.DataFrame:
        """
        批量预测
        
        Args:
            test_df: DataFrame with columns [GEOID, img_feat_0, img_feat_1, ...]
            img_feat_cols: 图像特征列名列表
            
        Returns:
            predictions_df: 包含预测结果的DataFrame
        """
        if img_feat_cols is None:
            img_feat_cols = [c for c in test_df.columns if c.startswith('img_feat_')]
        
        X = test_df[img_feat_cols].values.astype(np.float32)
        geoids = test_df['GEOID'].astype(str).tolist()
        return self.predict_batch_from_array(X_img_matrix=X, geoids=geoids)

    def predict_batch_from_array(
        self,
        X_img_matrix: np.ndarray,
        geoids: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """批量预测（直接输入预处理后的图像特征矩阵）。"""
        if geoids is None:
            geoids = [str(i) for i in range(len(X_img_matrix))]

        results = []
        for i in tqdm(range(len(X_img_matrix)), desc="Predicting"):
            geoid = geoids[i]
            X_img = np.asarray(X_img_matrix[i], dtype=np.float32)

            all_distances = self.find_cluster_distances(X_img)
            selected = self.select_clusters_for_weighting(all_distances)
            weights = self.build_idw_weights(selected)

            pred = self.predict_single(X_img)
            pred = self.postprocess_predictions(pred)

            nearest_cluster, nearest_distance = all_distances[0]

            results.append({
                'GEOID': geoid,
                'nearest_cluster': nearest_cluster,
                'nearest_distance': float(nearest_distance),
                'n_weighted_clusters': int(len(selected)),
                **{TARGET_NAMES[j]: float(pred[j]) for j in range(len(TARGET_NAMES))}
            })

        return pd.DataFrame(results)


def load_cluster_centers(cluster_datasets: Dict) -> Dict[str, np.ndarray]:
    """
    从训练数据计算聚类中心（纯图像特征均值）
    """
    centers = {}
    for cluster_id, dataset in cluster_datasets.items():
        # 使用PCA后的图像特征计算均值
        centers[cluster_id] = dataset['X_img'].mean(axis=0)
    return centers


def save_cluster_centers(centers: Dict[str, np.ndarray], save_path: str):
    """保存聚类中心"""
    joblib.dump(centers, save_path)
    print(f"聚类中心保存至: {save_path}")


def load_cluster_centers_from_file(load_path: str) -> Dict[str, np.ndarray]:
    """加载聚类中心"""
    return joblib.load(load_path)
