# =====================================================================
# 模块2: 教师专家模型训练（单任务）
# =====================================================================

import os
from typing import Dict, Tuple, List
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

try:
    from .config import (
        LGB_SINGLE_TASK_PARAMS, LGB_LONG_TAIL_PARAMS,
        TARGET_NAMES, LONG_TAIL_TARGETS,
        TEACHER_MODEL_DIR, EARLY_STOPPING_ROUNDS
    )
except ImportError:
    from config import (
        LGB_SINGLE_TASK_PARAMS, LGB_LONG_TAIL_PARAMS,
        TARGET_NAMES, LONG_TAIL_TARGETS,
        TEACHER_MODEL_DIR, EARLY_STOPPING_ROUNDS
    )


class TeacherExpertTrainer:
    """
    教师专家模型训练器（CBSA三分类）
    为每个city-cluster × 每个CBSA三分类目标训练独立的LGBM模型
    输入：图像特征 + 表格特征
    输出：3类CBSA通勤占比预测值（软标签）
    """
    
    def __init__(self, model_save_dir: str = TEACHER_MODEL_DIR):
        self.model_save_dir = model_save_dir
        os.makedirs(model_save_dir, exist_ok=True)
        self.training_results = []
        
    def get_lgb_params(self, target_name: str) -> Dict:
        """获取LGBM参数（CBSA三分类均用同一参数）"""
        return LGB_SINGLE_TASK_PARAMS.copy()
    
    def train_teacher_for_cluster_target(
        self,
        cluster_id: str,
        target_idx: int,
        target_name: str,
        X_teacher: np.ndarray,
        y: np.ndarray,
        train_idx: List[int],
        val_idx: List[int],
    ) -> Tuple[lgb.LGBMRegressor, Dict]:
        """
        为指定city-cluster和CBSA三分类通勤目标训练教师模型
        Args:
            cluster_id: city-cluster ID (如 "1_0")
            target_idx: 目标索引 (0-2)
            target_name: 目标名称
            X_teacher: 教师模型输入 (图像+表格)
            y: 真实标签 (3维)
            train_idx: 训练集索引
            val_idx: 验证集索引
        Returns:
            model: 训练好的LGBM模型
            metrics: 训练指标
        """
        y_target = y[:, target_idx]
        X_train, y_train = X_teacher[train_idx], y_target[train_idx]
        X_val, y_val = X_teacher[val_idx], y_target[val_idx]
        params = self.get_lgb_params(target_name)
        
        # 训练LGBM
        model = lgb.LGBMRegressor(**params)
        
        # 检查验证集是否为空
        if len(val_idx) > 0:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False),
                ]
            )
            best_iteration = model.best_iteration_
        else:
            # 验证集为空时，不使用early stopping
            model.fit(X_train, y_train)
            best_iteration = model.n_estimators
        
        # 评估
        y_pred_train = model.predict(X_train)
        
        if len(val_idx) > 0:
            y_pred_val = model.predict(X_val)
            val_mae = mean_absolute_error(y_val, y_pred_val)
            val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
            val_r2 = r2_score(y_val, y_pred_val)
        else:
            # 验证集为空时，使用训练集指标作为占位
            y_pred_val = y_pred_train
            val_mae = mean_absolute_error(y_train, y_pred_train)
            val_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            val_r2 = r2_score(y_train, y_pred_train)
        
        metrics = {
            'cluster_id': cluster_id,
            'target': target_name,
            'target_idx': target_idx,
            'n_train': len(train_idx),
            'n_val': len(val_idx),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'val_mae': val_mae,
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'val_rmse': val_rmse,
            'train_r2': r2_score(y_train, y_pred_train),
            'val_r2': val_r2,
            'best_iteration': best_iteration,
        }
        
        return model, metrics
    
    def generate_soft_labels(
        self,
        model: lgb.LGBMRegressor,
        X_teacher: np.ndarray,
    ) -> np.ndarray:
        """生成教师软标签"""
        return model.predict(X_teacher)
    
    def train_all_teachers(self, cluster_datasets: Dict[str, Dict]) -> Dict[str, np.ndarray]:
        """
        批量训练所有教师模型
        21 clusters × 7 targets = 147 个模型
        
        Returns:
            all_soft_labels: {cluster_id: soft_labels_array (n_samples × 7)}
        """
        print("\n" + "=" * 60)
        print("Step 4: 训练教师模型 (单任务)...")
        print("=" * 60)
        
        all_soft_labels = {}
        total_models = len(cluster_datasets) * len(TARGET_NAMES)
        trained = 0
        
        for cluster_id, dataset in cluster_datasets.items():
            n_samples = dataset['n_samples']
            cluster_soft_labels = np.zeros((n_samples, len(TARGET_NAMES)))
            
            for target_idx, target_name in enumerate(TARGET_NAMES):
                trained += 1
                print(f"  [{trained}/{total_models}] {cluster_id} - {target_name}")
                
                # 训练教师模型
                model, metrics = self.train_teacher_for_cluster_target(
                    cluster_id=cluster_id,
                    target_idx=target_idx,
                    target_name=target_name,
                    X_teacher=dataset['X_teacher'],
                    y=dataset['y'],
                    train_idx=dataset['train_idx'],
                    val_idx=dataset['val_idx'],
                )
                
                # 保存模型
                model_path = os.path.join(
                    self.model_save_dir,
                    f'teacher_{cluster_id}_{target_name}.pkl'
                )
                joblib.dump(model, model_path)
                
                # 记录结果
                self.training_results.append(metrics)
                
                # 生成软标签（全量数据）
                soft_labels = self.generate_soft_labels(model, dataset['X_teacher'])
                cluster_soft_labels[:, target_idx] = soft_labels
            
            all_soft_labels[cluster_id] = cluster_soft_labels
        
        # 保存训练报告
        self._save_training_report()
        
        print("=" * 60)
        return all_soft_labels
    
    def _save_training_report(self):
        """保存教师模型训练报告"""
        import json
        from datetime import datetime
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_type': 'teacher_single_task',
            'n_models': len(self.training_results),
            'avg_val_mae': np.mean([m['val_mae'] for m in self.training_results]),
            'avg_val_r2': np.mean([m['val_r2'] for m in self.training_results]),
            'results': self.training_results,
        }
        
        report_path = os.path.join(self.model_save_dir, 'training_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n  教师模型训练报告保存至: {report_path}")
        print(f"  平均验证MAE: {report['avg_val_mae']:.4f}")
        print(f"  平均验证R2: {report['avg_val_r2']:.4f}")
