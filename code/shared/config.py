"""
Global configuration for Commute Mode Prediction from Satellite Imagery.

This config references v2 tripmode labels (Car = B08006_002E,
Transit = B08006_008E, NonTransit = B08006_014E+015E+016E)
and the six feature types benchmarked in the 12-city 66-fold
leave-two-out cross-validation protocol.
"""

import os

try:
    import torch
except ImportError:
    torch = None

# ==================== Root paths ====================
# Set these environment variables to override for your machine.
# Or edit them directly.
GIT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.environ.get("EXPERT_DATA_ROOT", os.path.join(GIT_ROOT, "data"))
RESULTS_DIR = os.path.join(GIT_ROOT, "results")

DEVICE = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"

# ==================== Label data ====================
LABEL_CSV = os.path.join(DATA_DIR, "cbsa_mode_3class_tripmode_v2_ge10.csv")

TARGET_NAMES = ["Car_share", "Transit_share", "NonTransit_share"]
COUNT_NAMES = ["Car_count", "Transit_count", "NonTransit_count"]

# Minimum tripmode denominator for a tract to be included
MIN_TRIPMODE_DENOM = 10
MIN_TOTAL_COMMUTERS = 10

# ==================== City identifiers ====================
# 12 CBSA codes used in this study.
# First 10 are training cities; last 2 (41860, 47900) are zero-shot test cities.
CBSA_CODES_ALL = [12060, 14460, 16980, 19100, 26420, 31080, 33100, 35620, 38060, 42660, 41860, 47900]
CBSA_NAMES = {
    12060: "Atlanta", 14460: "Boston", 16980: "Chicago", 19100: "Dallas",
    26420: "Houston", 31080: "Los Angeles", 33100: "Miami", 35620: "New York",
    38060: "Phoenix", 41860: "San Francisco", 42660: "Seattle", 47900: "Washington DC",
}
TRAIN_CBSA_CODES = [12060, 14460, 16980, 26420, 31080, 33100, 35620, 38060, 41860, 42660]
TEST_CBSA_CODES = [19100, 47900]

# ==================== Feature types ====================
FEATURE_TYPE_DIM = {
    "aef_annual_64d": 64,
    "clay_v1_5_768d": 768,
    "prithvi_eo_v2_1024d": 1024,
    "imagenet_pretrain": 2048,
    "satellite_pretrain": 2048,
    "single_task_backbone_finetune": 2048,
}
FEATURE_TYPES = list(FEATURE_TYPE_DIM.keys())

# ==================== Expert model defaults ====================
EXPERT_DEFAULTS = dict(
    alpha=0.7,                # Knowledge distillation: soft-label weight
    learning_rate=0.05,
    num_leaves=31,
    max_depth=6,
    n_estimators=500,
    k_neighbors=5,            # IDW nearest neighbors for inference
    distance_metric="cosine",
    pca_dim=128,              # PCA reduction for image features
    prediction_postprocess="none",
)

