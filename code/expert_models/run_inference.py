# =====================================================================
# 鎺ㄧ悊鑴氭湰锛氫娇鐢ㄨ缁冨ソ鐨勪笓瀹舵ā鍨嬭繘琛岄娴?
# =====================================================================
# 杩愯鍛戒护锛?
#   python run_inference.py --model-type student --test-cbsa 12060
#   python run_inference.py --model-type multitask-student --test-cbsa 41860
# =====================================================================

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from label_schema import format_preflight_report, run_label_preflight

try:
    from .config import (
        MODEL_SAVE_DIR,
        CBSA_TRAIN_WITH_IMAGE_PATH,
        CBSA_LABEL_PATH,
        CBSA_CODE_COLUMN,
        EXPERT_FEATURE_TYPE,
        TARGET_NAMES,
        TEST_CBSA_CODES,
    )
    from .data_loader import ExpertDataLoader, load_label_data
    from .inference import ExpertEnsembleInference, load_cluster_centers_from_file
except ImportError:
    from config import (
        MODEL_SAVE_DIR,
        CBSA_TRAIN_WITH_IMAGE_PATH,
        CBSA_LABEL_PATH,
        CBSA_CODE_COLUMN,
        EXPERT_FEATURE_TYPE,
        TARGET_NAMES,
        TEST_CBSA_CODES,
    )
    from data_loader import ExpertDataLoader, load_label_data
    from inference import ExpertEnsembleInference, load_cluster_centers_from_file

from pipeline_contracts import FEATURE_DIM_BY_TYPE


def load_test_data(test_cbsa: int):
    """鍔犺浇娴嬭瘯 CBSA 鏁版嵁锛圕BSA-only锛夈€?""
    feat_df = pd.read_csv(CBSA_TRAIN_WITH_IMAGE_PATH)
    feat_df['GEOID'] = feat_df['GEOID'].astype(str).str.zfill(11)

    label_df = load_label_data(CBSA_LABEL_PATH)
    label_cols = ['GEOID'] + [c for c in TARGET_NAMES if c in label_df.columns]
    if 'total_commute' in label_df.columns:
        label_cols.append('total_commute')
    elif 'class3_denominator' in label_df.columns:
        label_df = label_df.copy()
        label_df['total_commute'] = pd.to_numeric(label_df['class3_denominator'], errors='coerce')
        label_cols.append('total_commute')
    elif 'cbsa_mode3_denom' in label_df.columns:
        label_df = label_df.copy()
        label_df['total_commute'] = pd.to_numeric(label_df['cbsa_mode3_denom'], errors='coerce')
        label_cols.append('total_commute')
    else:
        label_df = label_df.copy()
        label_df['total_commute'] = 1.0
        label_cols.append('total_commute')
    if CBSA_CODE_COLUMN in label_df.columns:
        label_cols.insert(1, CBSA_CODE_COLUMN)

    feat_cols = feat_df.columns.tolist()
    if CBSA_CODE_COLUMN in label_cols and CBSA_CODE_COLUMN in feat_cols:
        feat_cols = [c for c in feat_cols if c != CBSA_CODE_COLUMN]

    test_df = label_df[label_cols].merge(feat_df[feat_cols], on='GEOID', how='inner')
    if CBSA_CODE_COLUMN not in test_df.columns and CBSA_CODE_COLUMN in feat_df.columns:
        test_df = test_df.merge(feat_df[['GEOID', CBSA_CODE_COLUMN]], on='GEOID', how='left')
    test_df[CBSA_CODE_COLUMN] = pd.to_numeric(test_df[CBSA_CODE_COLUMN], errors='coerce')
    test_df = test_df[test_df[CBSA_CODE_COLUMN] == int(test_cbsa)].copy()

    preflight = run_label_preflight(test_df, TARGET_NAMES, city_hint=f"CBSA-{test_cbsa}")
    print(f"  鏍囩棰勬: {format_preflight_report(preflight)}")

    return test_df


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """璇勪及棰勬祴缁撴灉"""
    metrics = {}
    for i, target_name in enumerate(TARGET_NAMES):
        metrics[target_name] = {
            'mae': mean_absolute_error(y_true[:, i], y_pred[:, i]),
            'rmse': np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])),
            'r2': r2_score(y_true[:, i], y_pred[:, i]),
        }
    
    # 骞冲潎鎸囨爣
    metrics['avg_mae'] = np.mean([m['mae'] for m in metrics.values() if isinstance(m, dict)])
    metrics['avg_rmse'] = np.mean([m['rmse'] for m in metrics.values() if isinstance(m, dict)])
    metrics['avg_r2'] = np.mean([m['r2'] for m in metrics.values() if isinstance(m, dict)])
    
    return metrics


def resolve_model_dir(model_type: str) -> str:
    """瑙ｆ瀽妯″瀷鐩綍銆?""
    if model_type == 'student':
        return os.path.join(MODEL_SAVE_DIR, 'student')
    if model_type == 'teacher':
        return os.path.join(MODEL_SAVE_DIR, 'teacher')
    if model_type in ('multitask', 'multitask-student'):
        return os.path.join(MODEL_SAVE_DIR, 'multitask', 'student')
    raise ValueError(f"Unsupported model_type: {model_type}")


def run_expert_inference(
    test_city: int,
    model_type: str = 'student',
    k_neighbors: int = 10,
    weighting_scope: str = 'knn',
    output_dir: str = None,
    prediction_postprocess: str = 'none',
    distance_metric: str = 'cosine',
    idw_power: float = 2.0,
    perf_weight_mode: str = 'none',
    teacher_report_path: str | None = None,
    adaptive_similarity_threshold: float | None = None,
    min_neighbors: int = 1,
) -> tuple[pd.DataFrame, dict, str, str]:
    """鎵ц涓€娆′笓瀹舵ā鍨嬫帹鐞嗗苟淇濆瓨缁撴灉锛坱est_city 鍙傛暟鎸?CBSA code 瑙ｉ噴锛夈€?""
    test_cbsa = int(test_city)
    if test_cbsa not in TEST_CBSA_CODES:
        raise ValueError(f"娴嬭瘯CBSA蹇呴』鏄?{sorted(TEST_CBSA_CODES)}")

    data_loader = ExpertDataLoader()
    data_loader.load_preprocessors(MODEL_SAVE_DIR)

    test_df = load_test_data(test_cbsa)
    img_feat_cols = [c for c in test_df.columns if c.startswith('img_feat_')]

    expected_dim = FEATURE_DIM_BY_TYPE.get(EXPERT_FEATURE_TYPE)
    if expected_dim and len(img_feat_cols) != int(expected_dim):
        raise RuntimeError(f"鐗瑰緛缁村害涓嶅尮閰? expect={expected_dim}, got={len(img_feat_cols)}")

    X_img = test_df[img_feat_cols].values.astype(np.float32)
    X_img_scaled = data_loader.image_scaler.transform(X_img)
    if data_loader.pca is not None:
        X_img_scaled = data_loader.pca.transform(X_img_scaled)

    # Teacher model also needs tabular features
    TEACHER_TAB_COLS = ['avg_car_per_household', 'no_car_household_ratio', 'household_median_income', 'bachelor_above_ratio', 'housing_ownership_ratio', 'employment_rate']
    if model_type == 'teacher':
        for col in TEACHER_TAB_COLS:
            if col not in test_df.columns:
                raise RuntimeError(f"Teacher inference requires column '{col}' in test data")
        X_tab = test_df[TEACHER_TAB_COLS].values.astype(np.float32)
        # 浣跨敤璁粌闆嗗潎鍊煎～鍏?NaN
        col_means = np.nanmean(X_tab, axis=0)
        nan_mask = np.isnan(X_tab)
        X_tab[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
        X_tab_scaled = data_loader.table_scaler.transform(X_tab)
    else:
        X_tab_scaled = None

    cluster_centers = load_cluster_centers_from_file(
        os.path.join(MODEL_SAVE_DIR, 'cluster_centers.pkl')
    )

    inferencer = ExpertEnsembleInference(
        cluster_centers=cluster_centers,
        model_type=model_type,
        model_dir=resolve_model_dir(model_type),
        k_neighbors=k_neighbors,
        weighting_scope=weighting_scope,
        prediction_postprocess=prediction_postprocess,
        distance_metric=distance_metric,
        idw_power=idw_power,
        perf_weight_mode=perf_weight_mode,
        teacher_report_path=teacher_report_path,
        adaptive_similarity_threshold=adaptive_similarity_threshold,
        min_neighbors=min_neighbors,
    )

    if model_type == 'teacher':
        # Teacher: per-tract loop with tabular+image concatenation
        import pandas as pd
        pred_array = np.zeros((X_img_scaled.shape[0], len(TARGET_NAMES)), dtype=np.float64)
        for i in range(X_img_scaled.shape[0]):
            pred_array[i] = inferencer.predict_single_teacher(
                X_img_scaled[i], X_tab_scaled[i]
            )
        
        results_df = pd.DataFrame(pred_array, columns=TARGET_NAMES)
        results_df.insert(0, 'GEOID', test_df['GEOID'].astype(str).values)
    else:
        results_df = inferencer.predict_batch_from_array(
            X_img_matrix=X_img_scaled,
            geoids=test_df['GEOID'].astype(str).tolist(),
        )

    y_true = test_df[TARGET_NAMES].values
    y_pred = results_df[TARGET_NAMES].values
    metrics = evaluate_predictions(y_true, y_pred)

    if output_dir is None:
        output_dir = os.path.join(MODEL_SAVE_DIR, 'predictions')
    os.makedirs(output_dir, exist_ok=True)

    safe_model = model_type.replace('-', '_')
    suffix = f"{safe_model}_cbsa{test_cbsa}_{weighting_scope}_k{k_neighbors}_{prediction_postprocess}"
    pred_path = os.path.join(output_dir, f'predictions_{suffix}.csv')
    report_path = os.path.join(output_dir, f'evaluation_{suffix}.json')

    results_df.to_csv(pred_path, index=False, encoding='utf-8-sig')

    import json
    report = {
        'model_type': model_type,
        'test_cbsa': test_cbsa,
        'k_neighbors': k_neighbors,
        'weighting_scope': weighting_scope,
        'prediction_postprocess': prediction_postprocess,
        'distance_metric': distance_metric,
        'idw_power': idw_power,
        'perf_weight_mode': perf_weight_mode,
        'teacher_report_path': teacher_report_path,
        'adaptive_similarity_threshold': adaptive_similarity_threshold,
        'min_neighbors': min_neighbors,
        'n_samples': len(test_df),
        'metrics': metrics,
    }

    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return results_df, metrics, pred_path, report_path


def main():
    default_test_cbsa = sorted(TEST_CBSA_CODES)[0] if TEST_CBSA_CODES else 41860
    parser = argparse.ArgumentParser(description='涓撳妯″瀷鎺ㄧ悊')
    parser.add_argument('--model-type', type=str, default='student',
                       choices=['student', 'teacher', 'multitask', 'multitask-student'],
                       help='妯″瀷绫诲瀷 (student=鍗曚换鍔″鐢? multitask/multitask-student=澶氫换鍔″鐢?')
    parser.add_argument('--test-cbsa', type=int, default=default_test_cbsa,
                       help='娴嬭瘯CBSA code锛堝繀椤诲湪 TEST_CBSA_CODES锛?)
    parser.add_argument('--k-neighbors', type=int, default=10,
                       help='IDW鏈€杩戦偦鏁伴噺')
    parser.add_argument('--weighting-scope', type=str, default='knn', choices=['all', 'knn'],
                       help='IDW鍔犳潈鑼冨洿: all=鍏ㄩ噺涓撳(147瀵瑰簲鍘熷瀷), knn=浠匥杩戦偦')
    parser.add_argument('--prediction-postprocess', type=str, default='none',
                       choices=['none', 'clip_non_negative', 'clip01_renorm'],
                       help='棰勬祴鍚庡鐞嗘柟寮?)
    parser.add_argument('--output-dir', type=str, default=None,
                       help='杈撳嚭鐩綍')
    parser.add_argument('--distance-metric', type=str, default='cosine', choices=['l2', 'cosine'],
                       help='鍘熷瀷鐩镐技搴﹁窛绂? l2 鎴?cosine(1-cos)')
    parser.add_argument('--idw-power', type=float, default=2.0,
                       help='IDW骞傛鍙傛暟 p, 鏉冮噸~(1/d)^p')
    parser.add_argument('--perf-weight-mode', type=str, default='none', choices=['none', 'val_r2', 'val_mae_inv'],
                       help='鎬ц兘鏍″噯妯″紡: none / val_r2 / val_mae_inv')
    parser.add_argument('--teacher-report-path', type=str, default=None,
                       help='teacher/training_report.json 璺緞锛堝惎鐢ㄦ€ц兘鏍″噯鏃跺繀濉級')
    parser.add_argument('--adaptive-similarity-threshold', type=float, default=None,
                       help='鑷€傚簲K闃堝€硷紱浣欏鸡鐢╟os鐩镐技搴﹂槇鍊硷紝L2鐢?/(d+eps)闃堝€?)
    parser.add_argument('--min-neighbors', type=int, default=1,
                       help='鑷€傚簲绛涢€夊悗淇濆簳鏈€灏戣繎閭绘暟')
    args = parser.parse_args()
    
    # 楠岃瘉娴嬭瘯CBSA
    if args.test_cbsa not in TEST_CBSA_CODES:
        raise ValueError(f"娴嬭瘯CBSA蹇呴』鏄?{sorted(TEST_CBSA_CODES)}")
    
    print("="*70)
    print(f"涓撳妯″瀷鎺ㄧ悊")
    print("="*70)
    print(f"妯″瀷绫诲瀷: {args.model_type}")
    print(f"娴嬭瘯CBSA: {args.test_cbsa}")
    print(f"K鏈€杩戦偦: {args.k_neighbors}")
    print(f"鍔犳潈鑼冨洿: {args.weighting_scope}")
    print(f"鍚庡鐞? {args.prediction_postprocess}")
    print(f"璺濈搴﹂噺: {args.distance_metric}")
    print(f"IDW骞傛: {args.idw_power}")
    print(f"鎬ц兘鏍″噯: {args.perf_weight_mode}")
    print(f"鑷€傚簲闃堝€? {args.adaptive_similarity_threshold}")
    print("="*70)

    print("\n寮€濮嬮娴?..")
    results_df, metrics, output_path, report_path = run_expert_inference(
        test_city=args.test_cbsa,
        model_type=args.model_type,
        k_neighbors=args.k_neighbors,
        weighting_scope=args.weighting_scope,
        prediction_postprocess=args.prediction_postprocess,
        output_dir=args.output_dir,
        distance_metric=args.distance_metric,
        idw_power=args.idw_power,
        perf_weight_mode=args.perf_weight_mode,
        teacher_report_path=args.teacher_report_path,
        adaptive_similarity_threshold=args.adaptive_similarity_threshold,
        min_neighbors=args.min_neighbors,
    )
    
    print("\n" + "="*70)
    print("璇勪及缁撴灉")
    print("="*70)
    print(f"{'鐩爣':<25} {'MAE':<10} {'RMSE':<10} {'R2':<10}")
    print("-"*70)
    for target_name in TARGET_NAMES:
        m = metrics[target_name]
        print(f"{target_name:<25} {m['mae']:<10.4f} {m['rmse']:<10.4f} {m['r2']:<10.4f}")
    print("-"*70)
    print(f"{'骞冲潎':<25} {metrics['avg_mae']:<10.4f} {metrics['avg_rmse']:<10.4f} {metrics['avg_r2']:<10.4f}")
    print("="*70)
    
    print(f"\n棰勬祴缁撴灉淇濆瓨鑷? {output_path}")
    print(f"璇勪及鎶ュ憡淇濆瓨鑷? {report_path}")


if __name__ == '__main__':
    main()

