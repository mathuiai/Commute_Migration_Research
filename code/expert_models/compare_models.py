# =====================================================================
# 模型对比分析脚本
# 对比：单任务教师模型 vs 学生模型 vs 多任务模型
# =====================================================================

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List

from .config import MODEL_SAVE_DIR, TARGET_NAMES


def load_training_report(model_type: str) -> Dict:
    """加载训练报告"""
    report_path = os.path.join(MODEL_SAVE_DIR, model_type, 'training_report.json')
    if not os.path.exists(report_path):
        return None
    
    with open(report_path, 'r') as f:
        return json.load(f)


def compare_training_performance():
    """对比训练阶段性能"""
    print("="*70)
    print("专家模型性能对比（训练阶段）")
    print("="*70)
    
    # 加载各模型的训练报告
    teacher_report = load_training_report('teacher')
    student_report = load_training_report('student')
    multitask_report = load_training_report('multitask')
    
    reports = {
        '教师模型(单任务)': teacher_report,
        '学生模型(单任务+蒸馏)': student_report,
        '多任务模型': multitask_report,
    }
    
    # 过滤掉未训练的模型
    reports = {k: v for k, v in reports.items() if v is not None}
    
    if not reports:
        print("未找到训练报告，请先运行训练！")
        return
    
    # 打印对比表
    print(f"\n{'模型类型':<25} {'模型数量':<12} {'平均MAE':<12} {'平均R2':<12}")
    print("-"*70)
    
    for name, report in reports.items():
        n_models = report.get('n_models', report.get('n_clusters', 'N/A'))
        mae = report.get('avg_val_mae', report.get('avg_val_mae_hard', 'N/A'))
        r2 = report.get('avg_val_r2', report.get('avg_val_r2_hard', 'N/A'))
        
        if isinstance(mae, (int, float)):
            print(f"{name:<25} {n_models:<12} {mae:<12.4f} {r2:<12.4f}")
        else:
            print(f"{name:<25} {n_models:<12} {mae:<12} {r2:<12}")
    
    print("="*70)
    
    # 计算性能差异
    if '教师模型(单任务)' in reports and '多任务模型' in reports:
        teacher_mae = reports['教师模型(单任务)']['avg_val_mae']
        multitask_mae = reports['多任务模型']['avg_val_mae']
        
        diff_pct = (multitask_mae - teacher_mae) / teacher_mae * 100
        
        print(f"\n性能对比分析:")
        print(f"  多任务 vs 教师模型 MAE差异: {diff_pct:+.2f}%")
        print(f"  模型复杂度对比: 147个模型 vs 21个模型 (压缩比: 7.0x)")
        
        if diff_pct < 5:
            print(f"  结论: 多任务模型在显著降低复杂度的同时，保持了相近的预测精度")
        elif diff_pct < 10:
            print(f"  结论: 多任务模型复杂度降低，但精度有轻微下降")
        else:
            print(f"  结论: 多任务模型精度下降明显，建议优先使用单任务模型")
    
    # 绘制对比图
    if len(reports) >= 2:
        plot_comparison(reports)


def compare_inference_performance(prediction_dir: str):
    """对比推理阶段性能（在测试集上）"""
    print("\n" + "="*70)
    print("专家模型性能对比（测试集推理）")
    print("="*70)
    
    # 查找预测结果文件
    student_preds = []
    multitask_preds = []
    
    for fname in os.listdir(prediction_dir):
        if fname.startswith('evaluation_'):
            path = os.path.join(prediction_dir, fname)
            with open(path, 'r') as f:
                eval_data = json.load(f)
                
                if eval_data['model_type'] == 'student':
                    student_preds.append(eval_data)
                elif eval_data['model_type'] == 'multitask':
                    multitask_preds.append(eval_data)
    
    if not student_preds and not multitask_preds:
        print("未找到预测结果，请先运行推理！")
        return
    
    # 汇总各模型的测试性能
    print(f"\n{'模型类型':<25} {'测试城市':<15} {'MAE':<12} {'R2':<12}")
    print("-"*70)
    
    for pred in student_preds:
        mae = pred['metrics']['avg_mae']
        r2 = pred['metrics']['avg_r2']
        city = pred['city_name']
        print(f"{'学生模型':<25} {city:<15} {mae:<12.4f} {r2:<12.4f}")
    
    for pred in multitask_preds:
        mae = pred['metrics']['avg_mae']
        r2 = pred['metrics']['avg_r2']
        city = pred['city_name']
        print(f"{'多任务模型':<25} {city:<15} {mae:<12.4f} {r2:<12.4f}")
    
    print("="*70)
    
    # 计算平均测试性能
    if student_preds:
        avg_student_mae = np.mean([p['metrics']['avg_mae'] for p in student_preds])
        avg_student_r2 = np.mean([p['metrics']['avg_r2'] for p in student_preds])
        print(f"\n学生模型平均测试性能: MAE={avg_student_mae:.4f}, R2={avg_student_r2:.4f}")
    
    if multitask_preds:
        avg_multitask_mae = np.mean([p['metrics']['avg_mae'] for p in multitask_preds])
        avg_multitask_r2 = np.mean([p['metrics']['avg_r2'] for p in multitask_preds])
        print(f"多任务模型平均测试性能: MAE={avg_multitask_mae:.4f}, R2={avg_multitask_r2:.4f}")


def plot_comparison(reports: Dict[str, Dict]):
    """绘制对比图"""
    try:
        import matplotlib
        matplotlib.use('Agg')  # 无头模式
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # MAE对比
        names = list(reports.keys())
        maes = []
        for name in names:
            report = reports[name]
            mae = report.get('avg_val_mae', report.get('avg_val_mae_hard', 0))
            maes.append(mae)
        
        axes[0].bar(names, maes, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[0].set_ylabel('MAE')
        axes[0].set_title('Validation MAE Comparison')
        axes[0].tick_params(axis='x', rotation=15)
        
        # R2对比
        r2s = []
        for name in names:
            report = reports[name]
            r2 = report.get('avg_val_r2', report.get('avg_val_r2_hard', 0))
            r2s.append(r2)
        
        axes[1].bar(names, r2s, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[1].set_ylabel('R²')
        axes[1].set_title('Validation R² Comparison')
        axes[1].tick_params(axis='x', rotation=15)
        
        plt.tight_layout()
        
        # 保存
        output_path = os.path.join(MODEL_SAVE_DIR, 'model_comparison.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n对比图保存至: {output_path}")
        
    except Exception as e:
        print(f"绘图失败: {e}")


def generate_summary_report(output_path: str = None):
    """生成综合对比报告"""
    if output_path is None:
        output_path = os.path.join(MODEL_SAVE_DIR, 'comparison_summary.md')
    
    teacher_report = load_training_report('teacher')
    student_report = load_training_report('student')
    multitask_report = load_training_report('multitask')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# 专家模型对比分析报告\n\n")
        f.write("## 模型架构对比\n\n")
        f.write("| 模型类型 | 模型数量 | 输入特征 | 训练策略 |\n")
        f.write("|---------|---------|---------|---------|\n")
        f.write("| 教师模型(单任务) | 147 | 图像+表格 | 传统监督学习 |\n")
        f.write("| 学生模型(单任务) | 147 | 纯图像 | 知识蒸馏(α=0.7) |\n")
        f.write("| 多任务模型 | 21 | 纯图像 | 多任务联合训练 |\n\n")
        
        f.write("## 训练性能对比\n\n")
        f.write("| 模型类型 | 平均MAE | 平均R² |\n")
        f.write("|---------|--------|--------|\n")
        
        if teacher_report:
            f.write(f"| 教师模型 | {teacher_report['avg_val_mae']:.4f} | {teacher_report['avg_val_r2']:.4f} |\n")
        if student_report:
            mae = student_report.get('avg_val_mae_hard', 'N/A')
            r2 = student_report.get('avg_val_r2_hard', 'N/A')
            f.write(f"| 学生模型 | {mae if isinstance(mae, str) else f'{mae:.4f}'} | {r2 if isinstance(r2, str) else f'{r2:.4f}'} |\n")
        if multitask_report:
            f.write(f"| 多任务模型 | {multitask_report['avg_val_mae']:.4f} | {multitask_report['avg_val_r2']:.4f} |\n")
        
        f.write("\n## 结论与建议\n\n")
        
        if teacher_report and multitask_report:
            teacher_mae = teacher_report['avg_val_mae']
            multitask_mae = multitask_report['avg_val_mae']
            diff_pct = (multitask_mae - teacher_mae) / teacher_mae * 100
            
            f.write(f"1. **精度对比**: 多任务模型相比单任务教师模型，MAE差异为 {diff_pct:+.2f}%\n")
            f.write(f"2. **复杂度对比**: 多任务模型将模型数量从147减少到21，压缩比为 7.0x\n")
            
            if diff_pct < 5:
                f.write(f"3. **推荐方案**: 多任务模型在显著降低复杂度的同时保持了相近的精度，推荐用于快速实验\n")
            else:
                f.write(f"3. **推荐方案**: 单任务模型精度更优，推荐用于最终模型\n")
        
        f.write(f"\n4. **推理阶段**: 所有模型均支持纯图像输入，符合研究核心设计\n")
    
    print(f"\n综合对比报告保存至: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='模型对比分析')
    parser.add_argument('--prediction-dir', type=str,
                       default=os.path.join(MODEL_SAVE_DIR, 'predictions'),
                       help='预测结果目录')
    parser.add_argument('--generate-report', action='store_true',
                       help='生成Markdown对比报告')
    args = parser.parse_args()
    
    # 训练阶段对比
    compare_training_performance()
    
    # 推理阶段对比
    if os.path.exists(args.prediction_dir):
        compare_inference_performance(args.prediction_dir)
    
    # 生成报告
    if args.generate_report:
        generate_summary_report()


if __name__ == '__main__':
    main()
