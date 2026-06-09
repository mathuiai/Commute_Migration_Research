# =====================================================================
# 模块3: 学生专家模型训练（单任务，知识蒸馏）
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
        STUDENT_MODEL_DIR, EARLY_STOPPING_ROUNDS,
        ALPHA, TEMPERATURE
    )
except ImportError:
    from config import (
        LGB_SINGLE_TASK_PARAMS, LGB_LONG_TAIL_PARAMS,
        STUDENT_MODEL_DIR, EARLY_STOPPING_ROUNDS,
    )


class StudentExpertTrainer:
    """
    学生专家模型训练器(单任务, 知识蒸馏)
    输入: 纯图像特征
    """
    
    def __init__(self, 
                 alpha: float = ALPHA,
                 temperature: float = TEMPERATURE,
                 model_save_dir: str = STUDENT_MODEL_DIR):
        self.alpha = alpha  # 软标签权重
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.temperature = float(temperature)
        self.model_save_dir = model_save_dir
        os.makedirs(model_save_dir, exist_ok=True)
        self.training_results = []
    
    def get_lgb_params(self, target_name: str) -> Dict:
        """获取LGBM参数"""
        if target_name in LONG_TAIL_TARGETS:
            return LGB_LONG_TAIL_PARAMS.copy()
        return LGB_SINGLE_TASK_PARAMS.copy()
    
    def prepare_distillation_target(
        self,
        y_hard: np.ndarray,           # 真实硬标签
        y_teacher_soft: np.ndarray,   # 教师软标签
    ) -> np.ndarray:
        """
        准备蒸馏目标：混合软标签和硬标签。
        采用温度缩放教师偏移量，T>1 时减弱教师牵引，降低过度平滑风险。

        y_distill = y_hard + α * (y_soft - y_hard) / T
        """
        return y_hard + self.alpha * (y_teacher_soft - y_hard) / self.temperature
    
    def train_student_for_cluster_target(
        self,
        cluster_id: str,
        target_idx: int,
        target_name: str,
        X_img: np.ndarray,            # 纯图像特征
        y_hard: np.ndarray,           # 真实硬标签 (CBSA三分类)
        y_teacher_soft: np.ndarray,   # 教师软标签 (标量)
        train_idx: List[int],
        val_idx: List[int],
    ) -> Tuple[lgb.LGBMRegressor, Dict]:
        """
        为指定city-cluster和CBSA三分类通勤目标训练学生模型(知识蒸馏)
        """
        # 提取单目标硬标签
        y_hard_target = y_hard[:, target_idx]
        # 准备蒸馏目标
        y_distill = self.prepare_distillation_target(y_hard_target, y_teacher_soft)
        
        # 划分数据
        X_train, y_train = X_img[train_idx], y_distill[train_idx]
        X_val, y_val = X_img[val_idx], y_distill[val_idx]
        y_hard_val = y_hard_target[val_idx]  # 用于评估
        
        # 获取超参数
        params = self.get_lgb_params(target_name)
        
        # 训练学生LGBM(纯图像输入)
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
        
        # 评估(用真实硬标签评估)
        if len(val_idx) > 0:
            y_pred_val = model.predict(X_val)
            val_mae = mean_absolute_error(y_hard_val, y_pred_val)
            val_rmse = np.sqrt(mean_squared_error(y_hard_val, y_pred_val))
            val_r2 = r2_score(y_hard_val, y_pred_val)
        else:
            # 验证集为空时，使用训练集评估
            y_hard_train = y_hard[train_idx, target_idx]
            y_pred_train = model.predict(X_img[train_idx])
            val_mae = mean_absolute_error(y_hard_train, y_pred_train)
            val_rmse = np.sqrt(mean_squared_error(y_hard_train, y_pred_train))
            val_r2 = r2_score(y_hard_train, y_pred_train)
        
        metrics = {
            'cluster_id': cluster_id,
            'target': target_name,
            'target_idx': target_idx,
            'alpha': self.alpha,
            'temperature': self.temperature,
            'n_train': len(train_idx),
            'n_val': len(val_idx),
            'val_mae_hard': val_mae,
            'val_rmse_hard': val_rmse,
            'val_r2_hard': val_r2,
            'best_iteration': best_iteration,
        }
        
        return model, metrics
    
    def train_all_students(
        self, 
        cluster_datasets: Dict[str, Dict],
        all_teacher_soft_labels: Dict[str, np.ndarray]
    ):
        """
        批量训练所有学生模型(知识蒸馏, CBSA三分类)
        """
        print("\n" + "=" * 60)
        print(f"Step 5: 训练学生模型 (知识蒸馏, α={self.alpha}, T={self.temperature})...")
        print("=" * 60)
        
        total_models = len(cluster_datasets) * len(TARGET_NAMES)
        trained = 0
        
        for cluster_id, dataset in cluster_datasets.items():
            soft_labels = all_teacher_soft_labels[cluster_id]
            
            for target_idx, target_name in enumerate(TARGET_NAMES):
                trained += 1
                print(f"  [{trained}/{total_models}] {cluster_id} - {target_name}")
                
                # 训练学生模型
                model, metrics = self.train_student_for_cluster_target(
                    cluster_id=cluster_id,
                    target_idx=target_idx,
                    target_name=target_name,
                    X_img=dataset['X_img'],
                    y_hard=dataset['y'],
                    y_teacher_soft=soft_labels[:, target_idx],
                    train_idx=dataset['train_idx'],
                    val_idx=dataset['val_idx'],
                )
                
                # 保存模型
                model_path = os.path.join(
                    self.model_save_dir,
                    f'student_{cluster_id}_{target_name}.pkl'
                )
                joblib.dump(model, model_path)
                
                # 记录结果
                self.training_results.append(metrics)
        
        # 保存训练报告
        self._save_training_report()
        
        print("=" * 60)
    
    def _save_training_report(self):
        """保存学生模型训练报告"""
        import json
        from datetime import datetime
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_type': 'student_single_task',
            'alpha': self.alpha,
            'temperature': self.temperature,
            'n_models': len(self.training_results),
            'avg_val_mae_hard': np.mean([m['val_mae_hard'] for m in self.training_results]),
            'avg_val_r2_hard': np.mean([m['val_r2_hard'] for m in self.training_results]),
            'results': self.training_results,
        }
        
        report_path = os.path.join(self.model_save_dir, 'training_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n  学生模型训练报告保存至: {report_path}")
        print(f"  平均验证MAE: {report['avg_val_mae_hard']:.4f}")
        print(f"  平均验证R2: {report['avg_val_r2_hard']:.4f}")
