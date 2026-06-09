# 教师-学生专家模型训练框架

基于逐城市聚类的专家模型集成训练框架，支持知识蒸馏和多任务学习对比。

## 框架架构

```
step6_expert_models/
├── config.py              # 配置（特征、超参数、路径）
├── data_loader.py         # 模块1: 数据准备与特征对齐
├── teacher_trainer.py     # 模块2: 单任务教师模型（147模型）
├── student_trainer.py     # 模块3: 单任务学生模型（147模型，知识蒸馏）
├── multitask_trainer.py   # 模块4: 多任务教师-学生模型（21+21模型）
├── inference.py           # 模块5: 推理接口（纯图像输入）
├── run_training.py        # 主训练脚本
├── run_inference.py       # 推理脚本
├── compare_models.py      # 模型对比分析
└── README.md              # 本文件
```

## 两种方案对比

### 方案1: 单任务模型（细粒度）

| 阶段 | 模型数量 | 输入 | 输出 |
|------|---------|------|------|
| 教师 | 147 | 图像+表格(7维) | 每个目标独立预测 |
| 学生 | 147 | 纯图像(128PCA) | 每个目标独立预测 |

### 方案2: 多任务模型（高效率）

| 阶段 | 模型数量 | 输入 | 输出 |
|------|---------|------|------|
| 教师 | 21 | 图像+表格(7维) | 联合输出7目标 |
| 学生 | 21 | 纯图像(128PCA) | 联合输出7目标 |

**核心区别**: 多任务模型每个模型同时预测7个通勤目标，单任务每个模型只预测1个目标。

## 快速开始

### 1. 训练所有模型

```bash
# 训练全部模型（方案1 + 方案2）
python run_training.py --mode all

# 仅训练方案1（单任务）
python run_training.py --mode single-all

# 仅训练方案2（多任务）
python run_training.py --mode multitask-all

# 仅训练单任务教师
python run_training.py --mode teacher

# 仅训练单任务学生（需要已有教师软标签）
python run_training.py --mode student

# 仅训练多任务教师
python run_training.py --mode multitask-teacher

# 仅训练多任务学生（需要已有多任务教师软标签）
python run_training.py --mode multitask-student

# 仅进行对比分析
python run_training.py --mode compare
```

### 2. 推理测试

```bash
# 使用单任务学生模型预测指定CBSA
python run_inference.py --model-type student --test-cbsa 12060

# 使用多任务学生模型预测指定CBSA
python run_inference.py --model-type multitask-student --test-cbsa 41860

# 使用不同的K值
python run_inference.py --model-type student --test-cbsa 12060 --k-neighbors 15
```

### 3. 模型对比分析

```bash
# 对比训练性能并生成报告
python compare_models.py --generate-report
```

## 四种模型详细对比

| 维度 | 单任务教师 | 单任务学生 | 多任务教师 | 多任务学生 |
|------|-----------|-----------|-----------|-----------|
| **模型数量** | 147 | 147 | **21** | **21** |
| **输入特征** | 图像+表格 | 纯图像 | 图像+表格 | 纯图像 |
| **输出方式** | 独立7模型 | 独立7模型 | **联合7目标** | **联合7目标** |
| **训练策略** | 监督学习 | 知识蒸馏(α=0.7) | 监督学习 | 知识蒸馏(α=0.7) |
| **推理依赖** | 需表格数据 | **纯图像** | 需表格数据 | **纯图像** |
| **适用场景** | 生成软标签 | 最终部署 | 生成软标签 | 快速实验/部署 |
| **精度** | 高 | 高 | 中等 | 中等 |
| **复杂度** | 高 | 高 | **低(7x压缩)** | **低(7x压缩)** |

## 核心红线

1. **测试CBSA隔离**: TEST_CBSA_CODES 全程不进入训练
2. **学生模型纯图像**: 训练/推理绝不使用表格特征
3. **特征标准化**: 仅用训练城市拟合，防泄露
4. **知识蒸馏一致性**: 学生模型学习目标 = α×教师软标签 + (1-α)×真实硬标签

## 超参数配置

```python
# 知识蒸馏
ALPHA = 0.7  # 软标签权重 (推荐范围 0.6-0.8)

# LGBM
learning_rate = 0.05
num_leaves = 31
max_depth = 6

# 推理
K_NEIGHBORS = 10  # IDW最近邻
PCA_DIM = 128     # 图像特征降维
```

## 输出文件

```
expert_models/
├── teacher/                    # 单任务教师模型 (147个)
│   ├── teacher_{cluster}_{target}.pkl
│   ├── teacher_soft_labels.pkl
│   └── training_report.json
├── student/                    # 单任务学生模型 (147个)
│   ├── student_{cluster}_{target}.pkl
│   └── training_report.json
├── multitask/
│   ├── teacher/               # 多任务教师模型 (21个)
│   │     ├── multitask_teacher_{cluster}.pkl
│   │     ├── multitask_teacher_soft_labels.pkl
│   │     └── training_report.json
│   └── student/               # 多任务学生模型 (21个)
│         ├── multitask_student_{cluster}.pkl
│         └── training_report.json
├── cluster_centers.pkl         # 聚类中心（推理用）
├── image_scaler.pkl            # 图像标准化器
├── table_scaler.pkl            # 表格标准化器
├── pca.pkl                     # PCA降维器
└── preprocessed_data.pkl       # 预处理数据缓存
```

## MultiTaskLGBM 使用示例

```python
from multitask_trainer import MultiTaskLGBM
import numpy as np

# 创建模型
model = MultiTaskLGBM()

# 训练 (X: n_samples × n_features, y: n_samples × 7)
model.fit(X_train, y_train, eval_set=(X_val, y_val))

# 预测 (输出: n_samples × 7)
predictions = model.predict(X_test)

# 保存/加载
model.save('multitask_model.pkl')
model.load('multitask_model.pkl')
```

## 设计文档

详细设计见: `step6_expert_models_teacher_student_design.md`
