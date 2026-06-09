# Commute Mode Prediction from Satellite Imagery

Census-tract-level commuting mode share prediction (Car / Transit / Non-Transit) using satellite imagery via teacher-student knowledge distillation and IDW cross-city routing.

**Key features:**
- Zero-shot cross-city transfer: predict commuting patterns in unseen cities using only NAIP aerial imagery
- Teacher-student distillation: teacher uses ACS tabular data at training time; student learns from satellite imagery alone
- IDW (Inverse Distance Weighting) routing: matches target tracts to the most relevant training-city prototypes
- 6 feature backbones benchmarked under a 12-city 66-fold leave-two-out protocol

## Repository Structure

```
├── data/
│   ├── cbsa_mode_3class_tripmode_v2_ge10.csv   # Corrected v2 labels (Car=B08006_002E, Transit=B08006_008E)
│   └── summary.json
├── results/
│   ├── stage6_model_66fold_metrics.csv          # Expert model results across 66 folds × 6 feature types
│   └── naive_baselines_66fold_metrics.csv       # Baseline comparisons
├── code/
│   ├── label/                                   # Label construction from ACS raw data
│   │   └── build_cbsa_mode_classes_tripmode_v2.py
│   ├── expert_models/                           # Teacher-student expert model pipeline (core contribution)
│   │   ├── config.py                            # Paths and hyperparameters
│   │   ├── data_loader.py                       # Feature/label/cluster data merging
│   │   ├── teacher_trainer.py                   # 147 single-target teacher experts
│   │   ├── student_trainer.py                   # Student experts with knowledge distillation
│   │   ├── multitask_trainer.py                 # Multi-task teacher/student variants
│   │   ├── inference.py                         # Pure-image IDW inference
│   │   ├── run_training.py                      # Training orchestrator
│   │   ├── run_inference.py                     # Inference CLI
│   │   ├── compare_models.py                    # Model comparison analysis
│   │   └── compare_with_baseline.py             # Expert vs baseline comparison
│   └── shared/                                  # Shared utilities
│       ├── config.py                            # Global config (6 feature types, 12 CBSA codes)
│       ├── label_schema.py                      # Label semantics and preflight checks
│       ├── metrics_utils.py                     # Shared metric helpers
│       ├── reproducibility.py                   # Seed setting
│       ├── experiment_tracking.py               # Pipeline run registry
│       └── pipeline_contracts.py                # Feature manifest verification
└── README.md
```

## Data

### Labels (v2, corrected)
The label CSV uses the official ACS B08006 trip-mode groups:
- **Car** = B08006_002E (car, truck, or van)
- **Transit** = B08006_008E (public transportation, excluding taxicab)
- **NonTransit** = B08006_014E + B08006_015E + B08006_016E (bicycle, walked, other)

All tracts with total tripmode denominator < 10 are excluded. Work-from-home (B08006_017E) is excluded from the denominator.

**Note:** Earlier versions of this project incorrectly used B08006_004E (carpool) as Transit. This v2 corrected label set is the only supported version.

### Results
The `results/` directory contains:
- `stage6_model_66fold_metrics.csv`: Per-fold, per-city, per-feature-type metrics (MAE, RMSE, R²) for 6 feature backbones across all 66 leave-two-out folds
- `naive_baselines_66fold_metrics.csv`: Baseline methods (nearest train city mean, test city oracle mean) for comparison

### 12 Study Cities (CBSA Codes)
| Code | City | Role |
|------|------|------|
| 12060 | Atlanta | Train |
| 14460 | Boston | Train |
| 16980 | Chicago | Train |
| 26420 | Houston | Train |
| 31080 | Los Angeles | Train |
| 33100 | Miami | Train |
| 35620 | New York | Train |
| 38060 | Phoenix | Train |
| 41860 | San Francisco | Train |
| 42660 | Seattle | Train |
| 19100 | Dallas-Fort Worth | **Zero-shot test** |
| 47900 | Washington DC | **Zero-shot test** |

## Quick Start

### Prerequisites
```bash
pip install torch torchvision lightgbm scikit-learn pandas numpy pillow joblib tqdm
```

### 1. Build labels (optional — pre-built labels are in `data/`)
```bash
cd code/label
python build_cbsa_mode_classes_tripmode_v2.py --input <raw_acs_csv> --output <output_dir>
```

### 2. Train expert models
```bash
cd code/expert_models
# Train teacher experts (uses image + tabular features)
python run_training.py --mode teacher

# Train student experts (requires teacher soft labels)
python run_training.py --mode student
```

### 3. Run inference on a test city
```bash
python run_inference.py --model-type student --test-cbsa 41860
```

### 4. Compare with baselines
```bash
python compare_with_baseline.py --test-cbsa 41860
```

## 6 Feature Types

| Type | Dimension | Source |
|------|-----------|--------|
| `aef_annual_64d` | 64 | Google Earth Engine AEF |
| `clay_v1_5_768d` | 768 | Clay v1.5 foundation model |
| `prithvi_eo_v2_1024d` | 1024 | NASA Prithvi-EO v2 |
| `imagenet_pretrain` | 2048 | ImageNet pre-trained ResNet-50 |
| `satellite_pretrain` | 2048 | Satellite pre-trained ResNet-50 |
| `single_task_backbone_finetune` | 2048 | Fine-tuned ResNet-50 backbone |

## Configuration

Set environment variables to override paths:
- `EXPERT_DATA_ROOT`: Override data directory
- `EXPERT_FEATURE_TYPE`: Select feature type (default: `single_task_backbone_finetune`)
- `EXPERT_MODEL_SAVE_DIR`: Override model save directory
- `EXPERT_TEST_CBSA_CODES`: Override test city CBSA codes

See `code/shared/config.py` and `code/expert_models/config.py` for all configuration options.

## Citation

If you use this code or data, please cite the associated paper (TBD).

## License

TBD — Research code, no license specified.

