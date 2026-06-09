# =====================================================================
# 涓撳妯″瀷 vs 鍩虹嚎妯″瀷 瀵规瘮鍒嗘瀽
# =====================================================================
# 瀵规瘮鏂规锛?
# 1. 涓撳妯″瀷锛堣仛绫诲悗锛夛細21/147涓ā鍨嬶紝IDW闆嗘垚
# 2. 鍩虹嚎妯″瀷锛堜笉鑱氱被锛夛細1涓€氱敤LGBM妯″瀷锛堟潵鑷猻tep3a鎴杝tep4a锛?
# =====================================================================

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 娣诲姞鐖剁洰褰曞埌璺緞
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MODEL_SAVE_DIR, CBSA_LABEL_PATH, CBSA_CODE_COLUMN, TARGET_NAMES, TEST_CBSA_CODES
from data_loader import ExpertDataLoader, load_label_data
from inference import ExpertEnsembleInference, load_cluster_centers_from_file
from run_inference import run_expert_inference as run_expert_inference_cbsa


def load_baseline_predictions(
    test_cbsa: int,
    baseline_type: str = "universal",
    baseline_pred_path: str | None = None,
) -> pd.DataFrame:
    """
    鍔犺浇鍩虹嚎妯″瀷鐨勯娴嬬粨鏋?
    
    Args:
        test_cbsa: 娴嬭瘯CBSA code
        baseline_type: 'universal' (閫氱敤妯″瀷) 鎴?'single' (鍗曞煄甯傛ā鍨?
        baseline_pred_path: 鎸囧畾鍩虹嚎棰勬祴鏂囦欢锛堝彲閫夛級
    
    Returns:
        DataFrame with columns [GEOID, target1, target2, ...]
    """
    # 灏濊瘯澶氫釜鍙兘鐨勮矾寰?
    possible_paths = []
    if baseline_pred_path:
        possible_paths.append(baseline_pred_path)

    data_root = os.path.dirname(os.path.dirname(os.path.dirname(MODEL_SAVE_DIR)))
    possible_paths.extend(
        [
            os.path.join(data_root, "test_set", "cbsa_3class", f"predictions_cbsa{test_cbsa}.csv"),
            os.path.join(data_root, "FineTuneResNet", "expert_models", "predictions", f"predictions_student_cbsa{test_cbsa}_knn_k10_none.csv"),
        ]
    )
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"  鍔犺浇鍩虹嚎棰勬祴: {path}")
            df = pd.read_csv(path)
            df['GEOID'] = df['GEOID'].astype(str).str.zfill(11)
            
            # 妫€鏌ユ槸鍚︽槸 step4a 鏍煎紡鐨勯娴嬫枃浠?(鍖呭惈 actual 鍜?predicted 鍒?
            if any('_predicted' in col for col in df.columns):
                # 鎻愬彇 predicted 鍒楀苟閲嶅懡鍚嶄负鏍囧噯 target 鍚嶇О
                result = pd.DataFrame({'GEOID': df['GEOID']})
                for target in TARGET_NAMES:
                    pred_col = f"{target}_predicted"
                    if pred_col in df.columns:
                        result[target] = df[pred_col]
                return result
            else:
                return df
    
    raise FileNotFoundError(f"鏈壘鍒板熀绾挎ā鍨嬮娴嬬粨鏋滐紝灏濊瘯璺緞: {possible_paths}")


def run_expert_inference(test_cbsa: int, model_type: str = 'student', k_neighbors: int = 10) -> pd.DataFrame:
    """
    杩愯涓撳妯″瀷鎺ㄧ悊
    
    Args:
        test_cbsa: 娴嬭瘯CBSA code
        model_type: 'student' (鍗曚换鍔? 鎴?'multitask-student' (澶氫换鍔?
        k_neighbors: IDW鏈€杩戦偦鏁伴噺
    
    Returns:
        棰勬祴缁撴灉DataFrame
    """
    print(f"\n{'='*60}")
    print(f"Expert Model Inference [{model_type}]")
    print(f"{'='*60}")

    results_df, _metrics, _pred_path, _report_path = run_expert_inference_cbsa(
        test_city=int(test_cbsa),
        model_type=model_type,
        k_neighbors=k_neighbors,
        weighting_scope='knn',
        prediction_postprocess='none',
        output_dir=None,
    )
    return results_df


def evaluate_and_compare(
    y_true: np.ndarray,
    y_pred_expert: np.ndarray,
    y_pred_baseline: np.ndarray,
    target_names: List[str]
) -> Dict:
    """
    璇勪及骞跺姣斾笓瀹舵ā鍨嬪拰鍩虹嚎妯″瀷
    
    Returns:
        瀵规瘮缁撴灉瀛楀吀
    """
    results = {
        'expert': {},
        'baseline': {},
        'comparison': {}
    }
    
    for i, target in enumerate(target_names):
        y_t = y_true[:, i]
        y_e = y_pred_expert[:, i]
        y_b = y_pred_baseline[:, i]
        
        # 璁＄畻鎸囨爣
        expert_mae = mean_absolute_error(y_t, y_e)
        baseline_mae = mean_absolute_error(y_t, y_b)
        
        expert_rmse = np.sqrt(mean_squared_error(y_t, y_e))
        baseline_rmse = np.sqrt(mean_squared_error(y_t, y_b))
        
        expert_r2 = r2_score(y_t, y_e)
        baseline_r2 = r2_score(y_t, y_b)
        
        results['expert'][target] = {
            'mae': expert_mae,
            'rmse': expert_rmse,
            'r2': expert_r2
        }
        
        results['baseline'][target] = {
            'mae': baseline_mae,
            'rmse': baseline_rmse,
            'r2': baseline_r2
        }
        
        # 瀵规瘮
        mae_improvement = (baseline_mae - expert_mae) / baseline_mae * 100
        r2_improvement = expert_r2 - baseline_r2
        
        results['comparison'][target] = {
            'mae_improvement_pct': mae_improvement,
            'r2_improvement': r2_improvement,
            'winner': 'expert' if expert_mae < baseline_mae else 'baseline'
        }
    
    # 璁＄畻骞冲潎鎸囨爣
    results['expert']['avg'] = {
        'mae': np.mean([results['expert'][t]['mae'] for t in target_names]),
        'rmse': np.mean([results['expert'][t]['rmse'] for t in target_names]),
        'r2': np.mean([results['expert'][t]['r2'] for t in target_names])
    }
    
    results['baseline']['avg'] = {
        'mae': np.mean([results['baseline'][t]['mae'] for t in target_names]),
        'rmse': np.mean([results['baseline'][t]['rmse'] for t in target_names]),
        'r2': np.mean([results['baseline'][t]['r2'] for t in target_names])
    }
    
    avg_mae_improvement = (results['baseline']['avg']['mae'] - results['expert']['avg']['mae']) / results['baseline']['avg']['mae'] * 100
    avg_r2_improvement = results['expert']['avg']['r2'] - results['baseline']['avg']['r2']
    
    results['comparison']['avg'] = {
        'mae_improvement_pct': avg_mae_improvement,
        'r2_improvement': avg_r2_improvement,
        'winner': 'expert' if results['expert']['avg']['mae'] < results['baseline']['avg']['mae'] else 'baseline'
    }
    
    return results


def print_comparison_table(results: Dict, target_names: List[str], city_name: str):
    """鎵撳嵃瀵规瘮琛ㄦ牸"""
    print("\n" + "="*90)
    print(f"涓撳妯″瀷 vs 鍩虹嚎妯″瀷 瀵规瘮缁撴灉 [{city_name}]")
    print("="*90)
    
    # 琛ㄥご
    print(f"\n{'鐩爣':<25} {'鍩虹嚎MAE':<12} {'涓撳MAE':<12} {'MAE鎻愬崌':<12} {'鍩虹嚎R2':<10} {'涓撳R2':<10} {'鑳滆€?:<10}")
    print("-"*90)
    
    # 鍚勭洰鏍?
    for target in target_names:
        b_mae = results['baseline'][target]['mae']
        e_mae = results['expert'][target]['mae']
        improvement = results['comparison'][target]['mae_improvement_pct']
        b_r2 = results['baseline'][target]['r2']
        e_r2 = results['expert'][target]['r2']
        winner = results['comparison'][target]['winner']
        
        print(f"{target:<25} {b_mae:<12.4f} {e_mae:<12.4f} {improvement:>+10.2f}% {b_r2:<10.4f} {e_r2:<10.4f} {winner:<10}")
    
    # 骞冲潎
    print("-"*90)
    b_avg_mae = results['baseline']['avg']['mae']
    e_avg_mae = results['expert']['avg']['mae']
    avg_improvement = results['comparison']['avg']['mae_improvement_pct']
    b_avg_r2 = results['baseline']['avg']['r2']
    e_avg_r2 = results['expert']['avg']['r2']
    avg_winner = results['comparison']['avg']['winner']
    
    print(f"{'骞冲潎':<25} {b_avg_mae:<12.4f} {e_avg_mae:<12.4f} {avg_improvement:>+10.2f}% {b_avg_r2:<10.4f} {e_avg_r2:<10.4f} {avg_winner:<10}")
    print("="*90)
    
    # 缁撹
    print(f"\nConclusion:")
    if avg_improvement > 0:
        print(f"  [OK] Expert model better than baseline, MAE reduced by {avg_improvement:.2f}%")
    else:
        print(f"  [WARNING] Expert model worse than baseline, MAE increased by {-avg_improvement:.2f}%")
    
    # 缁熻鍚勭洰鏍囪儨璐?
    expert_wins = sum(1 for t in target_names if results['comparison'][t]['winner'] == 'expert')
    baseline_wins = len(target_names) - expert_wins
    print(f"  Target-level comparison: Expert wins on {expert_wins}/{len(target_names)} targets")


def save_comparison_report(results: Dict, city_id: int, city_name: str, output_dir: str):
    """淇濆瓨瀵规瘮鎶ュ憡"""
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = os.path.join(output_dir, f'comparison_expert_vs_baseline_city{city_id}.json')
    
    report = {
        'city_id': city_id,
        'city_name': city_name,
        'timestamp': pd.Timestamp.now().isoformat(),
        'results': results
    }
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n瀵规瘮鎶ュ憡淇濆瓨鑷? {report_path}")


def main():
    parser = argparse.ArgumentParser(description='涓撳妯″瀷 vs 鍩虹嚎妯″瀷瀵规瘮锛圕BSA锛?)
    parser.add_argument('--test-city', '--test-cbsa', dest='test_cbsa', type=int, default=sorted(TEST_CBSA_CODES)[0],
                       choices=sorted(TEST_CBSA_CODES),
                       help='娴嬭瘯CBSA code')
    parser.add_argument('--expert-type', type=str, default='student',
                       choices=['student', 'multitask-student'],
                       help='涓撳妯″瀷绫诲瀷')
    parser.add_argument('--k-neighbors', type=int, default=10,
                       help='IDW鏈€杩戦偦鏁伴噺')
    parser.add_argument('--baseline-type', type=str, default='universal',
                       choices=['universal', 'single'],
                       help='鍩虹嚎妯″瀷绫诲瀷')
    parser.add_argument('--baseline-pred-path', type=str, default=None,
                       help='鍙€夛細鏄惧紡鎸囧畾鍩虹嚎棰勬祴CSV璺緞')
    args = parser.parse_args()

    city_name = f"CBSA-{args.test_cbsa}"
    
    print("="*90)
    print("涓撳妯″瀷 vs 鍩虹嚎妯″瀷 瀵规瘮鍒嗘瀽")
    print("="*90)
    print(f"娴嬭瘯鍖哄煙: [{args.test_cbsa}] {city_name}")
    print(f"涓撳妯″瀷: {args.expert_type}")
    print(f"鍩虹嚎妯″瀷: {args.baseline_type}")
    print("="*90)
    
    try:
        # 1. 杩愯涓撳妯″瀷鎺ㄧ悊
        expert_preds = run_expert_inference(
            args.test_cbsa,
            model_type=args.expert_type,
            k_neighbors=args.k_neighbors
        )
        
        # 2. 鍔犺浇鍩虹嚎妯″瀷棰勬祴
        baseline_preds = load_baseline_predictions(
            args.test_cbsa,
            args.baseline_type,
            baseline_pred_path=args.baseline_pred_path,
        )
        
        # 3. 瀵归綈鏁版嵁
        common_geoids = set(expert_preds['GEOID']) & set(baseline_preds['GEOID'])
        print(f"\n鍏卞悓鏍锋湰鏁? {len(common_geoids)}")
        
        expert_preds = expert_preds[expert_preds['GEOID'].isin(common_geoids)].sort_values('GEOID')
        baseline_preds = baseline_preds[baseline_preds['GEOID'].isin(common_geoids)].sort_values('GEOID')
        
        # 4. 鍔犺浇鐪熷疄鏍囩
        label_df = load_label_data(CBSA_LABEL_PATH)
        label_df[CBSA_CODE_COLUMN] = pd.to_numeric(label_df[CBSA_CODE_COLUMN], errors='coerce')
        label_df = label_df[label_df[CBSA_CODE_COLUMN] == int(args.test_cbsa)]
        label_df = label_df[label_df['GEOID'].isin(common_geoids)].sort_values('GEOID')
        
        # 5. 鎻愬彇鏁板€?
        y_true = label_df[TARGET_NAMES].values
        y_pred_expert = expert_preds[TARGET_NAMES].values
        y_pred_baseline = baseline_preds[TARGET_NAMES].values
        
        # 6. 璇勪及瀵规瘮
        results = evaluate_and_compare(y_true, y_pred_expert, y_pred_baseline, TARGET_NAMES)
        
        # 7. 鎵撳嵃缁撴灉
        print_comparison_table(results, TARGET_NAMES, city_name)
        
        # 8. 淇濆瓨鎶ュ憡
        output_dir = os.path.join(MODEL_SAVE_DIR, 'comparisons')
        save_comparison_report(results, args.test_cbsa, city_name, output_dir)
        
    except FileNotFoundError as e:
        print(f"\n閿欒: {e}")
        print("\n璇风‘淇濆凡璁粌鍩虹嚎妯″瀷骞剁敓鎴愰娴嬬粨鏋溿€?)
        print("鍩虹嚎妯″瀷璁粌鍛戒护:")
        print("  python step3a_single_task_lgbm.py")
        print("  python step4a_universal_lgbm_test.py")


if __name__ == '__main__':
    main()

