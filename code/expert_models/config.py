# =====================================================================
# 鏁欏笀-瀛︾敓涓撳妯″瀷閰嶇疆
# =====================================================================

import os
from typing import List, Dict, Set

import pandas as pd

# ===================== 璺緞閰嶇疆 =====================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_ROOT = os.getenv("EXPERT_DATA_ROOT", os.path.join(PROJECT_ROOT, "data"))


def _latest_per_city_cluster_dir(base_dir: str) -> str | None:
    if not os.path.exists(base_dir):
        return None
    runs = []
    for name in os.listdir(base_dir):
        p = os.path.join(base_dir, name)
        label_path = os.path.join(p, "all_tract_city_cluster_labels.csv")
        if os.path.isdir(p) and name.startswith("per_city_cluster_") and os.path.exists(label_path):
            runs.append(p)
    if not runs:
        return None
    runs.sort()
    return runs[-1]

# 鐗瑰緛璺緞锛圕BSA-only锛?
EXPERT_FEATURE_TYPE = os.getenv("EXPERT_FEATURE_TYPE", "single_task_backbone_finetune")
CBSA_ROOT = os.path.join(DATA_ROOT, "CBSA")
CBSA_TRAIN_DATA_DIR = os.path.join(CBSA_ROOT, "TrainData")
CBSA_LABEL_DATA_DIR = os.path.join(CBSA_ROOT, "LabelData")



def _first_existing_path(*paths: str) -> str:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return paths[0]

CBSA_TRAIN_WITH_IMAGE_PATH = os.getenv(
    "EXPERT_CBSA_TRAIN_WITH_IMAGE_PATH",
    _first_existing_path(
        os.path.join(CBSA_TRAIN_DATA_DIR, f"cbsa_tract_train_features_with_image_{EXPERT_FEATURE_TYPE}.csv"),
    ),
)
CBSA_LABEL_PATH = os.getenv(
    "EXPERT_CBSA_LABEL_PATH",
    _first_existing_path(
        os.path.join(CBSA_LABEL_DATA_DIR, "cbsa_mode_3class_tripmode_v2_ge10.csv"),
        os.path.join(PROJECT_ROOT, "data", "cbsa_mode_3class_tripmode_v2_ge10.csv"),
        os.path.join(_LEGACY_DATA_ROOT, "CBSA", "LabelData", "cbsa_mode_3class_ge10.csv"),
    ),
)

# Backward-compat aliases kept for helper scripts.
IMAGE_FEATURE_DIR = os.path.join(DATA_ROOT, "FineTuneResNet", "features", EXPERT_FEATURE_TYPE)
TABULAR_DATA_DIR = CBSA_TRAIN_DATA_DIR
LABEL_DATA_DIR = CBSA_LABEL_DATA_DIR

# 鑱氱被缁撴灉璺緞
PER_CITY_CLUSTER_BASE_DIR = os.path.join(DATA_ROOT, "FineTuneResNet", "city_prototypes_per_city")
_DEFAULT_PER_CITY_CLUSTER_DIR = _latest_per_city_cluster_dir(PER_CITY_CLUSTER_BASE_DIR)
PER_CITY_CLUSTER_DIR = os.getenv("EXPERT_PER_CITY_CLUSTER_DIR", _DEFAULT_PER_CITY_CLUSTER_DIR or "")

if os.getenv("EXPERT_CLUSTER_LABEL_PATH"):
    CLUSTER_LABEL_PATH = os.getenv("EXPERT_CLUSTER_LABEL_PATH")
else:
    CLUSTER_LABEL_PATH = os.path.join(PER_CITY_CLUSTER_DIR, "all_tract_city_cluster_labels.csv") if PER_CITY_CLUSTER_DIR else ""

# 妯″瀷杈撳嚭璺緞
MODEL_SAVE_DIR = os.getenv("EXPERT_MODEL_SAVE_DIR", os.path.join(DATA_ROOT, "FineTuneResNet", "expert_models"))
TEACHER_MODEL_DIR = os.path.join(MODEL_SAVE_DIR, "teacher")
STUDENT_MODEL_DIR = os.path.join(MODEL_SAVE_DIR, "student")
MULTITASK_MODEL_DIR = os.path.join(MODEL_SAVE_DIR, "multitask")
MULTITASK_TEACHER_DIR = os.path.join(MULTITASK_MODEL_DIR, "teacher")
MULTITASK_STUDENT_DIR = os.path.join(MULTITASK_MODEL_DIR, "student")

# 鍒涘缓鐩綍
for d in [TEACHER_MODEL_DIR, STUDENT_MODEL_DIR, MULTITASK_MODEL_DIR, 
          MULTITASK_TEACHER_DIR, MULTITASK_STUDENT_DIR]:
    os.makedirs(d, exist_ok=True)

if not CLUSTER_LABEL_PATH or not os.path.exists(CLUSTER_LABEL_PATH):
    raise FileNotFoundError(
        "鏈壘鍒颁笓瀹舵ā鍨嬭仛绫绘爣绛炬枃浠?all_tract_city_cluster_labels.csv銆?
        "璇疯缃幆澧冨彉閲?EXPERT_CLUSTER_LABEL_PATH 鎴?EXPERT_PER_CITY_CLUSTER_DIR銆?
    )

# ===================== CBSA 鍒掑垎閰嶇疆 =====================
CBSA_CODE_COLUMN = "cbsa_code"

DEFAULT_SPLIT_MANIFEST_PATH = os.path.join(
    DATA_ROOT,
    "3.26plan",
    "protocol",
    "split_manifest_v2_train10_test2.csv",
)


def _parse_test_cbsa_codes_from_manifest(split_manifest_path: str) -> Set[int]:
    if not split_manifest_path or not os.path.exists(split_manifest_path):
        return set()
    try:
        manifest = pd.read_csv(split_manifest_path, dtype={"cbsa_code": str})
    except Exception:
        return set()
    needed = {"cbsa_code", "split"}
    if not needed.issubset(set(manifest.columns)):
        return set()
    test_df = manifest[manifest["split"].astype(str).str.lower() == "test"].copy()
    out: Set[int] = set()
    for x in test_df["cbsa_code"].astype(str).tolist():
        x = x.strip()
        if not x:
            continue
        out.add(int(x))
    return out

def _parse_test_cbsa_codes() -> Set[int]:
    raw = os.getenv("EXPERT_TEST_CBSA_CODES", "").strip()
    if raw:
        codes = set()
        for x in raw.split(","):
            x = x.strip()
            if not x:
                continue
            codes.add(int(x))
        return codes

    split_manifest_path = os.getenv("EXPERT_SPLIT_MANIFEST_PATH", DEFAULT_SPLIT_MANIFEST_PATH)
    from_manifest = _parse_test_cbsa_codes_from_manifest(split_manifest_path)
    if from_manifest:
        return from_manifest

    # Fallback to the current protocol policy if split manifest cannot be resolved.
    return {41860, 42660}


TEST_CBSA_CODES: Set[int] = _parse_test_cbsa_codes()

# Backward-compat alias names used by older helper scripts.
TEST_CITY_IDS: Set[int] = TEST_CBSA_CODES
TRAIN_CITY_LIST = []


def city_id_to_folder(city_id: int) -> str:
    # Deprecated in CBSA-only pipeline. Kept only to avoid import breakage.
    return str(int(city_id))


# ===================== 鐗瑰緛閰嶇疆 =====================

# 銆愮粷瀵圭鐢?- 鏁版嵁娉勯湶绾㈢嚎銆?
DISABLED_FEATURES = [
    "avg_commute_time",
    "avg_commute_distance",
    "morning_peak_commute_ratio",
]

# 銆愭暀甯堟ā鍨嬫帹鑽愮壒寰?- 7涓€?
TEACHER_RECOMMENDED_FEATURES = [
    "avg_car_per_household",
    "no_car_household_ratio",
    "household_median_income",
    "bachelor_above_ratio",
    "housing_ownership_ratio",
    "employment_rate",
]

# 銆愭暀甯堟ā鍨嬬畝鍖栫壒寰?- 5涓€?
TEACHER_SIMPLIFIED_FEATURES = [
    "avg_car_per_household",
    "no_car_household_ratio",
    "household_median_income",
    "bachelor_above_ratio",
    "housing_ownership_ratio",
]

# 閫夋嫨浣跨敤鍝釜閰嶇疆
TEACHER_FEATURES = TEACHER_RECOMMENDED_FEATURES  # 7鐗瑰緛鏂规


# ===================== CBSA涓夊垎绫荤洰鏍囬厤缃?=====================
TARGET_NAMES = [
    "Car_share",
    "Transit_share",
    "NonTransit_share",
]
TARGET_CONFIG = {
    "Car_share": {"desc": "灏忔苯杞﹀嚭琛屽崰姣?},
    "Transit_share": {"desc": "鍏叡浜ら€氬崰姣?},
    "NonTransit_share": {"desc": "闈炴満鍔?鍏朵粬鍗犳瘮"},
}
LONG_TAIL_TARGETS: Set[str] = set()

# ===================== 鐭ヨ瘑钂搁瓒呭弬鏁?=====================
ALPHA = 0.7              # 杞爣绛炬潈閲?
TEMPERATURE = 1.0        # 钂搁娓╁害

# ===================== LGBM瓒呭弬鏁?=====================

# 鍗曚换鍔GBM鍙傛暟锛堟暀甯堝拰瀛︾敓锛?
LGB_SINGLE_TASK_PARAMS = {
    "objective": "regression",
    "metric": ["rmse", "mae"],
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 6,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_data_in_leaf": 10,
    "verbose": -1,
    "random_state": 42,
    "n_estimators": 500,
}

# 闀垮熬鐩爣涓撶敤鍙傛暟
LGB_LONG_TAIL_PARAMS = {
    "objective": "regression_l1",  # MAE鎹熷け锛屽寮傚父鍊兼洿椴佹
    "metric": ["mae", "rmse"],
    "learning_rate": 0.02,
    "num_leaves": 15,
    "max_depth": 5,
    "min_data_in_leaf": 5,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "verbose": -1,
    "random_state": 42,
    "n_estimators": 500,
}

# 澶氫换鍔GBM鍙傛暟
LGB_MULTITASK_PARAMS = {
    "objective": "regression",
    "metric": ["rmse", "mae"],
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 6,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_data_in_leaf": 10,
    "verbose": -1,
    "random_state": 42,
    "n_estimators": 500,
}

# ===================== 鎺ㄧ悊閰嶇疆 =====================
K_NEIGHBORS = 10         # IDW鏈€杩戦偦鏁伴噺
PCA_DIM = 128            # PCA闄嶇淮缁村害

# ===================== 鍏朵粬閰嶇疆 =====================
RANDOM_STATE = 42
VAL_SPLIT = 0.2          # 楠岃瘉闆嗘瘮渚?
EARLY_STOPPING_ROUNDS = 20




