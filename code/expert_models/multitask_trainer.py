# =====================================================================
# 模块4: 多任务LGBM专家模型训练（教师-学生流程）
# =====================================================================
# 关键修改：多任务模型也采用教师-学生架构
# - 教师多任务：21个模型，输入图像+表格，联合输出7目标
# - 学生多任务：21个模型，输入纯图像，蒸馏教师知识，联合输出7目标
# =====================================================================

import os
from typing import Dict, Tuple, List
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

try:
    from .config import (
        LGB_MULTITASK_PARAMS,
        TARGET_NAMES,
        MULTITASK_MODEL_DIR,
        EARLY_STOPPING_ROUNDS,
        ALPHA, TEMPERATURE
    )
except ImportError:
    from config import (
        LGB_MULTITASK_PARAMS,
        TARGET_NAMES,
        MULTITASK_MODEL_DIR,
        EARLY_STOPPING_ROUNDS,
        ALPHA, TEMPERATURE
    )


class MultiTaskLGBM:
    """
    多任务LGBM包装器
    包装7个并行的LGBMRegressor，实现联合输入、联合输出
    """
    
    def __init__(self, lgb_params: Dict = None):
        self.lgb_params = lgb_params or LGB_MULTITASK_PARAMS.copy()
        self.models = {}  # {target_name: LGBMRegressor}
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            eval_set: Tuple = None,
            callbacks=None):
        """
        训练多任务模型
        
        Args:
            X: 输入特征
            y: 目标值 (n_samples × 7)
            eval_set: (X_val, y_val)
            callbacks: LGBM回调函数
        """
        n_targets = y.shape[1]
        
        for target_idx, target_name in enumerate(TARGET_NAMES):
            y_target = y[:, target_idx]
            
            # 准备验证集
            eval_set_target = None
            if eval_set is not None:
                X_val, y_val = eval_set
                eval_set_target = [(X_val, y_val[:, target_idx])]
            
            # 训练单目标模型
            model = lgb.LGBMRegressor(**self.lgb_params)
            
            # 检查验证集是否为空
            if eval_set_target is not None and len(eval_set_target[0][0]) > 0:
                model.fit(
                    X, y_target,
                    eval_set=eval_set_target,
                    callbacks=callbacks
                )
            else:
                # 验证集为空时，不使用early stopping
                model.fit(X, y_target)
            
            self.models[target_name] = model
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测所有目标
        
        Returns:
            predictions: (n_samples × 7) 的预测值
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet!")
        
        n_samples = X.shape[0]
        predictions = np.zeros((n_samples, len(TARGET_NAMES)))
        
        for target_idx, target_name in enumerate(TARGET_NAMES):
            model = self.models[target_name]
            predictions[:, target_idx] = model.predict(X)
        
        return predictions
    
    def save(self, path: str):
        """保存模型"""
        joblib.dump(self.models, path)
    
    def load(self, path: str):
        """加载模型"""
        self.models = joblib.load(path)
        self.is_fitted = True


class MultiTaskTeacherTrainer:
    """
    多任务教师模型训练器
    21个模型，每个模型输入图像+表格特征，联合输出7个通勤目标
    """
    
    def __init__(self, model_save_dir: str = None):
        if model_save_dir is None:
            model_save_dir = os.path.join(MULTITASK_MODEL_DIR, 'teacher')
        self.model_save_dir = model_save_dir
        os.makedirs(model_save_dir, exist_ok=True)
        self.training_results = []
    
    def train_teacher_for_cluster(
        self,
        cluster_id: str,
        X_teacher: np.ndarray,    # 图像+表格特征
        y: np.ndarray,            # 真实标签 (n_samples × 7)
        train_idx: List[int],
        val_idx: List[int],
    ) -> Tuple[MultiTaskLGBM, Dict]:
        """
        为指定city-cluster训练多任务教师模型
        
        Args:
            cluster_id: city-cluster ID
            X_teacher: 教师模型输入 (图像+表格)
            y: 真实标签 (7维)
            train_idx: 训练集索引
            val_idx: 验证集索引
            
        Returns:
            model: 训练好的多任务LGBM模型
            metrics: 训练指标
        """
        # 划分数据
        X_train, X_val = X_teacher[train_idx], X_teacher[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # 训练多任务模型
        model = MultiTaskLGBM(LGB_MULTITASK_PARAMS.copy())
        
        # 检查验证集是否为空
        if len(val_idx) > 0:
            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                callbacks=[
                    lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False),
                ]
            )
            # 评估
            y_pred_val = model.predict(X_val)
        else:
            # 验证集为空时，不使用early stopping，用训练集评估
            model.fit(X_train, y_train)
            y_pred_val = model.predict(X_train)
            y_val = y_train
        
        # 计算各目标的指标
        target_metrics = {}
        for target_idx, target_name in enumerate(TARGET_NAMES):
            target_metrics[target_name] = {
                'val_mae': mean_absolute_error(y_val[:, target_idx], y_pred_val[:, target_idx]),
                'val_rmse': np.sqrt(mean_squared_error(y_val[:, target_idx], y_pred_val[:, target_idx])),
                'val_r2': r2_score(y_val[:, target_idx], y_pred_val[:, target_idx]),
            }
        
        # 平均指标
        avg_mae = np.mean([m['val_mae'] for m in target_metrics.values()])
        avg_r2 = np.mean([m['val_r2'] for m in target_metrics.values()])
        
        metrics = {
            'cluster_id': cluster_id,
            'model_type': 'multitask_teacher',
            'n_train': len(train_idx),
            'n_val': len(val_idx),
            'avg_val_mae': avg_mae,
            'avg_val_r2': avg_r2,
            'target_metrics': target_metrics,
        }
        
        return model, metrics
    
    def train_all_teachers(self, cluster_datasets: Dict[str, Dict]) -> Dict[str, np.ndarray]:
        """
        批量训练所有多任务教师模型
        21 clusters = 21 个多任务模型
        
        Returns:
            all_soft_labels: {cluster_id: soft_labels_array (n_samples × 7)}
        """
        print("\n" + "=" * 60)
        print("训练多任务教师模型 (21个模型，每个输出7目标)...")
        print("=" * 60)
        
        all_soft_labels = {}
        total_clusters = len(cluster_datasets)
        
        for idx, (cluster_id, dataset) in enumerate(cluster_datasets.items(), 1):
            print(f"  [{idx}/{total_clusters}] {cluster_id} - {dataset['city_name']}")
            print(f"       样本数: {dataset['n_samples']}")
            
            # 训练多任务教师模型
            model, metrics = self.train_teacher_for_cluster(
                cluster_id=cluster_id,
                X_teacher=dataset['X_teacher'],
                y=dataset['y'],
                train_idx=dataset['train_idx'],
                val_idx=dataset['val_idx'],
            )
            
            # 保存模型
            model_path = os.path.join(
                self.model_save_dir,
                f'multitask_teacher_{cluster_id}.pkl'
            )
            model.save(model_path)
            
            # 记录结果
            self.training_results.append(metrics)
            
            # 生成软标签（全量数据）
            soft_labels = model.predict(dataset['X_teacher'])
            all_soft_labels[cluster_id] = soft_labels
            
            print(f"       平均验证MAE: {metrics['avg_val_mae']:.4f}")
            print(f"       平均验证R2: {metrics['avg_val_r2']:.4f}")
        
        # 保存训练报告
        self._save_training_report()
        
        print("=" * 60)
        return all_soft_labels
    
    def _save_training_report(self):
        """保存训练报告"""
        import json
        from datetime import datetime
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_type': 'multitask_teacher',
            'n_models': len(self.training_results),
            'avg_val_mae': np.mean([m['avg_val_mae'] for m in self.training_results]),
            'avg_val_r2': np.mean([m['avg_val_r2'] for m in self.training_results]),
            'results': self.training_results,
        }
        
        report_path = os.path.join(self.model_save_dir, 'training_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n  多任务教师模型训练报告保存至: {report_path}")
        print(f"  平均验证MAE: {report['avg_val_mae']:.4f}")
        print(f"  平均验证R2: {report['avg_val_r2']:.4f}")


class MultiTaskStudentTrainer:
    """
    多任务学生模型训练器（知识蒸馏）
    21个模型，每个模型输入纯图像特征，蒸馏教师知识，联合输出7目标
    """
    
    def __init__(self, 
                 alpha: float = ALPHA,
                 temperature: float = TEMPERATURE,
                 model_save_dir: str = None):
        if model_save_dir is None:
            model_save_dir = os.path.join(MULTITASK_MODEL_DIR, 'student')
        self.alpha = alpha  # 软标签权重
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.temperature = float(temperature)
        self.model_save_dir = model_save_dir
        os.makedirs(model_save_dir, exist_ok=True)
        self.training_results = []
    
    def prepare_distillation_target(
        self,
        y_hard: np.ndarray,           # 真实硬标签 (n_samples × 7)
        y_teacher_soft: np.ndarray,   # 教师软标签 (n_samples × 7)
    ) -> np.ndarray:
        """
        准备蒸馏目标
        使用温度缩放教师偏移量，T>1 时减弱教师牵引。

        y_distill = y_hard + α * (y_soft - y_hard) / T
        """
        return y_hard + self.alpha * (y_teacher_soft - y_hard) / self.temperature
    
    def train_student_for_cluster(
        self,
        cluster_id: str,
        X_img: np.ndarray,            # 纯图像特征
        y_hard: np.ndarray,           # 真实硬标签
        y_teacher_soft: np.ndarray,   # 教师软标签
        train_idx: List[int],
        val_idx: List[int],
    ) -> Tuple[MultiTaskLGBM, Dict]:
        """
        为指定city-cluster训练多任务学生模型（知识蒸馏）
        
        Args:
            cluster_id: city-cluster ID
            X_img: 纯图像特征
            y_hard: 真实硬标签
            y_teacher_soft: 教师软标签
            train_idx: 训练集索引
            val_idx: 验证集索引
            
        Returns:
            model: 训练好的多任务学生模型
            metrics: 训练指标
        """
        # 准备蒸馏目标
        y_distill = self.prepare_distillation_target(y_hard, y_teacher_soft)
        
        # 划分数据
        X_train, X_val = X_img[train_idx], X_img[val_idx]
        y_train, y_val = y_distill[train_idx], y_distill[val_idx]
        y_hard_val = y_hard[val_idx]  # 用于评估
        
        # 训练多任务学生模型
        model = MultiTaskLGBM(LGB_MULTITASK_PARAMS.copy())
        
        # 检查验证集是否为空
        if len(val_idx) > 0:
            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                callbacks=[
                    lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False),
                ]
            )
            # 评估（用真实硬标签评估）
            y_pred_val = model.predict(X_val)
        else:
            # 验证集为空时，不使用early stopping，用训练集评估
            model.fit(X_train, y_train)
            y_pred_val = model.predict(X_train)
            y_hard_val = y_hard[train_idx]
        
        # 计算各目标的指标
        target_metrics = {}
        for target_idx, target_name in enumerate(TARGET_NAMES):
            target_metrics[target_name] = {
                'val_mae_hard': mean_absolute_error(y_hard_val[:, target_idx], y_pred_val[:, target_idx]),
                'val_r2_hard': r2_score(y_hard_val[:, target_idx], y_pred_val[:, target_idx]),
            }
        
        # 平均指标
        avg_mae_hard = np.mean([m['val_mae_hard'] for m in target_metrics.values()])
        avg_r2_hard = np.mean([m['val_r2_hard'] for m in target_metrics.values()])
        
        metrics = {
            'cluster_id': cluster_id,
            'model_type': 'multitask_student',
            'alpha': self.alpha,
            'temperature': self.temperature,
            'n_train': len(train_idx),
            'n_val': len(val_idx),
            'avg_val_mae_hard': avg_mae_hard,
            'avg_val_r2_hard': avg_r2_hard,
            'target_metrics': target_metrics,
        }
        
        return model, metrics
    
    def train_all_students(
        self,
        cluster_datasets: Dict[str, Dict],
        all_teacher_soft_labels: Dict[str, np.ndarray]
    ):
        """
        批量训练所有多任务学生模型（知识蒸馏）
        21 clusters = 21 个多任务模型
        """
        print("\n" + "=" * 60)
        print(f"训练多任务学生模型 (知识蒸馏, α={self.alpha}, T={self.temperature}, 21个模型)...")
        print("=" * 60)
        
        total_clusters = len(cluster_datasets)
        
        for idx, (cluster_id, dataset) in enumerate(cluster_datasets.items(), 1):
            soft_labels = all_teacher_soft_labels[cluster_id]
            
            print(f"  [{idx}/{total_clusters}] {cluster_id} - {dataset['city_name']}")
            
            # 训练多任务学生模型
            model, metrics = self.train_student_for_cluster(
                cluster_id=cluster_id,
                X_img=dataset['X_img'],
                y_hard=dataset['y'],
                y_teacher_soft=soft_labels,
                train_idx=dataset['train_idx'],
                val_idx=dataset['val_idx'],
            )
            
            # 保存模型
            model_path = os.path.join(
                self.model_save_dir,
                f'multitask_student_{cluster_id}.pkl'
            )
            model.save(model_path)
            
            # 记录结果
            self.training_results.append(metrics)
            
            print(f"       平均验证MAE: {metrics['avg_val_mae_hard']:.4f}")
            print(f"       平均验证R2: {metrics['avg_val_r2_hard']:.4f}")
        
        # 保存训练报告
        self._save_training_report()
        
        print("=" * 60)
    
    def _save_training_report(self):
        """保存训练报告"""
        import json
        from datetime import datetime
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_type': 'multitask_student',
            'alpha': self.alpha,
            'temperature': self.temperature,
            'n_models': len(self.training_results),
            'avg_val_mae_hard': np.mean([m['avg_val_mae_hard'] for m in self.training_results]),
            'avg_val_r2_hard': np.mean([m['avg_val_r2_hard'] for m in self.training_results]),
            'results': self.training_results,
        }
        
        report_path = os.path.join(self.model_save_dir, 'training_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n  多任务学生模型训练报告保存至: {report_path}")
        print(f"  平均验证MAE: {report['avg_val_mae_hard']:.4f}")
        print(f"  平均验证R2: {report['avg_val_r2_hard']:.4f}")


def compare_single_vs_multitask(
    single_task_results: List[Dict],
    multitask_results: List[Dict],
    model_type: str = 'student'
):
    """
    对比单任务 vs 多任务的性能
    
    Args:
        single_task_results: 单任务模型训练结果
        multitask_results: 多任务模型训练结果
        model_type: 'student' 或 'teacher'
    """
    print("\n" + "=" * 70)
    print(f"单任务(147模型) vs 多任务(21模型) 性能对比 [{model_type}]")
    print("=" * 70)
    
    # 单任务平均性能
    if model_type == 'student':
        single_mae = np.mean([m['val_mae_hard'] for m in single_task_results])
        single_r2 = np.mean([m['val_r2_hard'] for m in single_task_results])
    else:
        single_mae = np.mean([m['val_mae'] for m in single_task_results])
        single_r2 = np.mean([m['val_r2'] for m in single_task_results])
    
    # 多任务平均性能
    if model_type == 'student':
        multi_mae = np.mean([m['avg_val_mae_hard'] for m in multitask_results])
        multi_r2 = np.mean([m['avg_val_r2_hard'] for m in multitask_results])
    else:
        multi_mae = np.mean([m['avg_val_mae'] for m in multitask_results])
        multi_r2 = np.mean([m['avg_val_r2'] for m in multitask_results])
    
    print(f"  单任务 (147个独立模型):")
    print(f"    平均验证MAE: {single_mae:.4f}")
    print(f"    平均验证R²:  {single_r2:.4f}")
    
    print(f"\n  多任务 (21个联合模型):")
    print(f"    平均验证MAE: {multi_mae:.4f}")
    print(f"    平均验证R²:  {multi_r2:.4f}")
    
    print(f"\n  性能差异:")
    mae_diff_pct = (multi_mae - single_mae) / single_mae * 100
    r2_diff = multi_r2 - single_r2
    print(f"    MAE变化: {mae_diff_pct:+.2f}%")
    print(f"    R²变化:  {r2_diff:+.4f}")
    
    print(f"\n  模型复杂度对比:")
    print(f"    单任务: 147个模型")
    print(f"    多任务: 21个模型（每个联合输出7目标）")
    print(f"    压缩比: 7.0x")
    
    if abs(mae_diff_pct) < 5:
        print(f"\n  ✅ 结论: 多任务模型在显著降低复杂度的同时，保持了相近的预测精度")
    elif mae_diff_pct < 10:
        print(f"\n  ⚠️ 结论: 多任务模型复杂度降低，但精度有轻微下降")
    else:
        print(f"\n  ❌ 结论: 多任务模型精度下降明显，建议优先使用单任务模型")
    
    print("=" * 70)
