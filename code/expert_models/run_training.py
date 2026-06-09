# =====================================================================
# 主入口脚本：训练所有专家模型
# =====================================================================
# 运行命令：
#   python run_training.py --mode all
#   python run_training.py --mode teacher
#   python run_training.py --mode student
#   python run_training.py --mode multitask-teacher
#   python run_training.py --mode multitask-student
#   python run_training.py --mode compare
# =====================================================================

import os
import sys
import argparse
import json
import joblib
from datetime import datetime

from .config import MODEL_SAVE_DIR, CLUSTER_LABEL_PATH, ALPHA, TEMPERATURE, TARGET_NAMES
from .data_loader import ExpertDataLoader
from .teacher_trainer import TeacherExpertTrainer
from .student_trainer import StudentExpertTrainer
from .multitask_trainer import (
    MultiTaskTeacherTrainer, 
    MultiTaskStudentTrainer,
    compare_single_vs_multitask
)
from .inference import load_cluster_centers, save_cluster_centers


def _save_run_manifest(cluster_datasets, mode: str) -> None:
    cluster_ids = sorted(cluster_datasets.keys())
    manifest = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "model_save_dir": MODEL_SAVE_DIR,
        "cluster_label_path": CLUSTER_LABEL_PATH,
        "n_clusters": len(cluster_ids),
        "cluster_ids": cluster_ids,
        "target_names": TARGET_NAMES,
        "expected_single_task_models": len(cluster_ids) * len(TARGET_NAMES),
    }
    out_path = os.path.join(MODEL_SAVE_DIR, "run_manifest.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"运行清单已写入: {out_path}")


def train_single_task_teachers(cluster_datasets):
    """训练单任务教师模型"""
    print("\n" + "="*70)
    print("【方案1】训练单任务教师模型（147个模型）")
    print("="*70)
    
    trainer = TeacherExpertTrainer()
    all_soft_labels = trainer.train_all_teachers(cluster_datasets)
    
    # 保存软标签
    soft_labels_path = os.path.join(MODEL_SAVE_DIR, 'teacher', 'teacher_soft_labels.pkl')
    joblib.dump(all_soft_labels, soft_labels_path)
    print(f"软标签保存至: {soft_labels_path}")
    
    return all_soft_labels, trainer.training_results


def train_single_task_students(cluster_datasets, all_teacher_soft_labels, alpha: float = ALPHA, temperature: float = TEMPERATURE):
    """训练单任务学生模型"""
    print("\n" + "="*70)
    print("【方案1】训练单任务学生模型（147个模型，知识蒸馏）")
    print("="*70)
    
    trainer = StudentExpertTrainer(alpha=alpha, temperature=temperature)
    trainer.train_all_students(cluster_datasets, all_teacher_soft_labels)
    
    return trainer.training_results


def train_multitask_teachers(cluster_datasets):
    """训练多任务教师模型"""
    print("\n" + "="*70)
    print("【方案2】训练多任务教师模型（21个模型，联合输出7目标）")
    print("="*70)
    
    trainer = MultiTaskTeacherTrainer()
    all_soft_labels = trainer.train_all_teachers(cluster_datasets)
    
    # 保存软标签
    soft_labels_path = os.path.join(MODEL_SAVE_DIR, 'multitask', 'teacher', 'multitask_teacher_soft_labels.pkl')
    joblib.dump(all_soft_labels, soft_labels_path)
    print(f"多任务软标签保存至: {soft_labels_path}")
    
    return all_soft_labels, trainer.training_results


def train_multitask_students(cluster_datasets, all_teacher_soft_labels, alpha: float = ALPHA, temperature: float = TEMPERATURE):
    """训练多任务学生模型"""
    print("\n" + "="*70)
    print("【方案2】训练多任务学生模型（21个模型，知识蒸馏，联合输出7目标）")
    print("="*70)
    
    trainer = MultiTaskStudentTrainer(alpha=alpha, temperature=temperature)
    trainer.train_all_students(cluster_datasets, all_teacher_soft_labels)
    
    return trainer.training_results


def main():
    parser = argparse.ArgumentParser(description='训练专家模型')
    parser.add_argument('--mode', type=str, default='all',
                       choices=['all', 'teacher', 'student',
                               'multitask-teacher', 'multitask-student', 'multitask-all',
                               'single-all', 'compare'],
                       help='训练模式')
    parser.add_argument('--skip-data', action='store_true',
                       help='跳过数据加载（已有预处理数据）')
    parser.add_argument('--data-path', type=str, default=None,
                       help='预处理数据路径')
    parser.add_argument('--alpha', type=float, default=ALPHA,
                       help='蒸馏软标签权重')
    parser.add_argument('--temperature', type=float, default=TEMPERATURE,
                       help='蒸馏温度 (必须 > 0)')
    args = parser.parse_args()

    if args.temperature <= 0:
        raise ValueError('--temperature must be > 0')
    
    print("="*70)
    print("专家模型训练管道")
    print("="*70)
    print(f"模式: {args.mode}")
    print(f"蒸馏参数: alpha={args.alpha}, temperature={args.temperature}")
    print("="*70)
    
    # 数据加载
    if args.skip_data and args.data_path:
        print(f"\n加载预处理数据: {args.data_path}")
        data_bundle = joblib.load(args.data_path)
        cluster_datasets = data_bundle['cluster_datasets']
        data_loader = data_bundle['data_loader']
    else:
        data_loader = ExpertDataLoader()
        df = data_loader.load_all_data()
        data_loader.fit_scalers(df)
        cluster_datasets = data_loader.get_cluster_datasets(df)
        
        # 保存预处理器和聚类中心
        data_loader.save_preprocessors(MODEL_SAVE_DIR)
        cluster_centers = load_cluster_centers(cluster_datasets)
        save_cluster_centers(cluster_centers, os.path.join(MODEL_SAVE_DIR, 'cluster_centers.pkl'))
        
        # 保存预处理数据（可选）
        data_bundle = {
            'cluster_datasets': cluster_datasets,
            'data_loader': data_loader,
        }
        joblib.dump(data_bundle, os.path.join(MODEL_SAVE_DIR, 'preprocessed_data.pkl'))

    _save_run_manifest(cluster_datasets, mode=args.mode)
    
    # 存储结果用于对比
    single_teacher_results = None
    single_student_results = None
    multitask_teacher_results = None
    multitask_student_results = None
    
    # 根据模式执行训练
    if args.mode in ['all', 'teacher', 'single-all']:
        # 单任务教师
        _, single_teacher_results = train_single_task_teachers(cluster_datasets)
    
    if args.mode in ['all', 'student', 'single-all']:
        # 单任务学生
        soft_labels_path = os.path.join(MODEL_SAVE_DIR, 'teacher', 'teacher_soft_labels.pkl')
        if os.path.exists(soft_labels_path):
            all_soft_labels = joblib.load(soft_labels_path)
            single_student_results = train_single_task_students(
                cluster_datasets,
                all_soft_labels,
                alpha=args.alpha,
                temperature=args.temperature,
            )
        else:
            print("错误: 未找到教师软标签，请先训练教师模型")
            return
    
    if args.mode in ['all', 'multitask-teacher', 'multitask-all']:
        # 多任务教师
        _, multitask_teacher_results = train_multitask_teachers(cluster_datasets)
    
    if args.mode in ['all', 'multitask-student', 'multitask-all']:
        # 多任务学生
        soft_labels_path = os.path.join(MODEL_SAVE_DIR, 'multitask', 'teacher', 'multitask_teacher_soft_labels.pkl')
        if os.path.exists(soft_labels_path):
            all_soft_labels = joblib.load(soft_labels_path)
            multitask_student_results = train_multitask_students(
                cluster_datasets,
                all_soft_labels,
                alpha=args.alpha,
                temperature=args.temperature,
            )
        else:
            print("错误: 未找到多任务教师软标签，请先训练多任务教师模型")
            return
    
    # 对比分析
    if args.mode in ['all', 'compare']:
        print("\n" + "="*70)
        print("对比分析: 单任务 vs 多任务")
        print("="*70)
        
        # 加载已有结果
        import json
        
        if single_student_results is None:
            try:
                with open(os.path.join(MODEL_SAVE_DIR, 'student', 'training_report.json')) as f:
                    single_student_results = json.load(f)['results']
            except:
                pass
        
        if multitask_student_results is None:
            try:
                with open(os.path.join(MODEL_SAVE_DIR, 'multitask', 'student', 'training_report.json')) as f:
                    multitask_student_results = json.load(f)['results']
            except:
                pass
        
        # 对比学生模型
        if single_student_results and multitask_student_results:
            compare_single_vs_multitask(
                single_student_results, 
                multitask_student_results,
                model_type='student'
            )
        
        # 对比教师模型
        if single_teacher_results and multitask_teacher_results:
            compare_single_vs_multitask(
                single_teacher_results,
                multitask_teacher_results,
                model_type='teacher'
            )
    
    print("\n" + "="*70)
    print("训练完成！")
    print("="*70)
    print(f"模型保存路径: {MODEL_SAVE_DIR}")
    print("\n目录结构:")
    print(f"  {MODEL_SAVE_DIR}/")
    print(f"    ├── teacher/          (147个单任务教师模型)")
    print(f"    ├── student/          (147个单任务学生模型)")
    print(f"    ├── multitask/")
    print(f"    │     ├── teacher/    (21个多任务教师模型)")
    print(f"    │     └── student/    (21个多任务学生模型)")
    print(f"    └── cluster_centers.pkl")
    print("="*70)


if __name__ == '__main__':
    main()
