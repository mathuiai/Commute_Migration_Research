# =====================================================================
# 妯″潡1: 鏁版嵁鍑嗗涓庣壒寰佸榻?
# =====================================================================

import os
import csv
import warnings
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

try:
    from .config import (
        CBSA_TRAIN_WITH_IMAGE_PATH, CBSA_LABEL_PATH,
        CBSA_CODE_COLUMN, CLUSTER_LABEL_PATH, TEST_CBSA_CODES,
        TEACHER_FEATURES, TARGET_NAMES, TARGET_CONFIG,
        PCA_DIM, RANDOM_STATE, VAL_SPLIT
    )
except ImportError:
    from config import (
        CBSA_TRAIN_WITH_IMAGE_PATH, CBSA_LABEL_PATH,
        CBSA_CODE_COLUMN, CLUSTER_LABEL_PATH, TEST_CBSA_CODES,
        TEACHER_FEATURES, TARGET_NAMES, TARGET_CONFIG,
        PCA_DIM, RANDOM_STATE, VAL_SPLIT
    )

warnings.filterwarnings('ignore')

try:
    from ..label_schema import (
        compute_targets_from_counts,
        ensure_geoid,
        format_preflight_report,
        run_label_preflight,
    )
except ImportError:
    from label_schema import (
        compute_targets_from_counts,
        ensure_geoid,
        format_preflight_report,
        run_label_preflight,
    )


def detect_delimiter(file_path: str) -> str:
    """妫€娴婥SV鍒嗛殧绗?""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            sample = "".join([line for _, line in zip(range(20), f) if line.strip()])
            d = csv.Sniffer().sniff(sample).delimiter
            if d not in [",", "\t", ";"]:
                d = ","
    except Exception:
        d = ","
    return d


def load_label_data(csv_path: str) -> pd.DataFrame:
    """鍔犺浇鏍囩鏁版嵁骞惰绠楅€氬嫟鐩爣"""
    df = pd.read_csv(csv_path, sep=detect_delimiter(csv_path), encoding="utf-8")
    df = ensure_geoid(df)

    # CBSA 棰勮绠楁爣绛炬枃浠跺凡鐩存帴鍖呭惈鐩爣鍒椼€?
    if all(c in df.columns for c in TARGET_NAMES):
        out = df.dropna(subset=list(TARGET_NAMES)).copy()
        if "class3_denominator" in out.columns:
            out = out[out["class3_denominator"] >= 10]
            if "total_commute" not in out.columns:
                out["total_commute"] = pd.to_numeric(out["class3_denominator"], errors="coerce")
        elif "cbsa_mode3_denom" in out.columns:
            out = out[out["cbsa_mode3_denom"] >= 10]
            if "total_commute" not in out.columns:
                out["total_commute"] = pd.to_numeric(out["cbsa_mode3_denom"], errors="coerce")
        elif "total_commute" not in out.columns:
            # Keep preflight checks functional for precomputed share labels.
            out["total_commute"] = 1.0
        return out

    try:
        # v2 label CSV already has share columns; this fallback is for raw ACS CSVs only
        out = df.dropna(subset=list(TARGET_NAMES)).copy()
        if "class3_denominator" in out.columns:
            out = out[out["class3_denominator"] >= 10]
            out["total_commute"] = pd.to_numeric(out["class3_denominator"], errors="coerce")
        return out
    return df


class ExpertDataLoader:
    """
    涓撳妯″瀷鏁版嵁鍔犺浇鍣?
    鏍稿績鍔熻兘锛?
    1. 鍔犺浇鍥惧儚鐗瑰緛銆佽〃鏍肩壒寰併€佽仛绫绘爣绛俱€侀€氬嫟鏍囩
    2. 鎸塩ity-cluster鍒掑垎鏁版嵁闆?
    3. 鐗瑰緛鏍囧噯鍖栵紙浠呯敤璁粌鍩庡競鎷熷悎锛?
    4. 淇濆瓨鏍囧噯鍖栧弬鏁颁緵鎺ㄧ悊浣跨敤
    """
    
    def __init__(self,
                 cbsa_train_with_image_path: str = CBSA_TRAIN_WITH_IMAGE_PATH,
                 cbsa_label_path: str = CBSA_LABEL_PATH,
                 cluster_label_path: str = CLUSTER_LABEL_PATH,
                 teacher_features: List[str] = None,
                 test_cbsa_codes: set = None,
                 pca_dim: int = PCA_DIM):
        self.cbsa_train_with_image_path = cbsa_train_with_image_path
        self.cbsa_label_path = cbsa_label_path
        self.cluster_label_path = cluster_label_path
        self.teacher_features = teacher_features or TEACHER_FEATURES
        self.test_cbsa_codes = test_cbsa_codes or TEST_CBSA_CODES
        self.pca_dim = pca_dim
        
        # 鏍囧噯鍖栧櫒
        self.table_scaler = StandardScaler()
        self.image_scaler = StandardScaler()
        self.pca = None
        
        # 鏁版嵁缂撳瓨
        self._cluster_df = None
        
    def load_all_data(self) -> pd.DataFrame:
        """鍔犺浇骞跺悎骞舵墍鏈夋暟鎹?""
        print("=" * 60)
        print("Step 1: 鍔犺浇鏁版嵁...")
        print("=" * 60)
        
        # 鍔犺浇鑱氱被鏍囩
        self._cluster_df = pd.read_csv(self.cluster_label_path)
        self._cluster_df['GEOID'] = self._cluster_df['GEOID'].astype(str).str.zfill(11)
        print(f"  鍔犺浇鑱氱被鏍囩: {len(self._cluster_df)} 涓猼ract")
        
        merged_df = self._load_cbsa_data()
        print(f"  杩囨护娴嬭瘯CBSA鍓? {len(merged_df)} 涓猼ract")

        if CBSA_CODE_COLUMN not in merged_df.columns:
            raise KeyError(f"缂哄皯鍒? {CBSA_CODE_COLUMN}")

        merged_df[CBSA_CODE_COLUMN] = pd.to_numeric(merged_df[CBSA_CODE_COLUMN], errors="coerce")
        merged_df = merged_df.dropna(subset=[CBSA_CODE_COLUMN]).copy()
        merged_df[CBSA_CODE_COLUMN] = merged_df[CBSA_CODE_COLUMN].astype(int)

        merged_df = merged_df[~merged_df[CBSA_CODE_COLUMN].isin(self.test_cbsa_codes)].copy()
        print(f"  杩囨护娴嬭瘯CBSA鍚? {len(merged_df)} 涓猼ract")
        
        # 涓庤仛绫绘爣绛惧悎骞?
        merged_df = merged_df.merge(
            self._cluster_df[['GEOID', 'city_cluster_id', 'city_cluster_name']],
            on='GEOID',
            how='inner'
        )
        
        print(f"  鍚堝苟鑱氱被鏍囩鍚? {len(merged_df)} 涓猼ract")
        print("=" * 60)
        
        return merged_df

    def _load_cbsa_data(self) -> pd.DataFrame:
        if not os.path.exists(self.cbsa_train_with_image_path):
            raise FileNotFoundError(self.cbsa_train_with_image_path)
        if not os.path.exists(self.cbsa_label_path):
            raise FileNotFoundError(self.cbsa_label_path)

        feat_df = pd.read_csv(
            self.cbsa_train_with_image_path,
            sep=detect_delimiter(self.cbsa_train_with_image_path),
            encoding="utf-8",
        )
        feat_df['GEOID'] = feat_df['GEOID'].astype(str).str.zfill(11)

        # 妫€鏌ユ暀甯堢壒寰佹槸鍚﹀瓨鍦?
        missing_features = [f for f in self.teacher_features if f not in feat_df.columns]
        if missing_features:
            raise KeyError(f"CBSA璁粌鍩熺己灏戞暀甯堢壒寰? {missing_features}")

        label_df = load_label_data(self.cbsa_label_path)
        preflight = run_label_preflight(label_df, TARGET_NAMES, city_hint="CBSA")
        print(f"  鏍囩棰勬: {format_preflight_report(preflight)}")

        keep_cols = ['GEOID'] + self.teacher_features
        img_cols = [c for c in feat_df.columns if c.startswith('img_feat_')]
        if not img_cols:
            raise RuntimeError("CBSA璁粌鍩熺己灏戝浘鍍忕壒寰佸垪 img_feat_*")

        if CBSA_CODE_COLUMN in feat_df.columns:
            keep_cols.append(CBSA_CODE_COLUMN)

        label_cols = ['GEOID'] + TARGET_NAMES
        if CBSA_CODE_COLUMN in label_df.columns:
            label_cols.insert(1, CBSA_CODE_COLUMN)

        feat_cols = keep_cols + img_cols
        # Avoid duplicate cbsa_code columns that would become cbsa_code_x/cbsa_code_y.
        if CBSA_CODE_COLUMN in label_cols and CBSA_CODE_COLUMN in feat_cols:
            feat_cols = [c for c in feat_cols if c != CBSA_CODE_COLUMN]

        merged = label_df[label_cols].merge(feat_df[feat_cols], on='GEOID', how='inner')
        if CBSA_CODE_COLUMN not in merged.columns and CBSA_CODE_COLUMN in feat_df.columns:
            merged = merged.merge(feat_df[['GEOID', CBSA_CODE_COLUMN]], on='GEOID', how='left')
        return merged
    
    def fit_scalers(self, df: pd.DataFrame):
        """浠呯敤璁粌鏁版嵁鎷熷悎鏍囧噯鍖栧櫒"""
        print("\n" + "=" * 60)
        print("Step 2: 鎷熷悎鏍囧噯鍖栧櫒...")
        print("=" * 60)
        
        img_feat_cols = [c for c in df.columns if c.startswith('img_feat_')]
        
        # 鍥惧儚鐗瑰緛鏍囧噯鍖?
        print(f"  鍥惧儚鐗瑰緛缁村害: {len(img_feat_cols)}")
        self.image_scaler.fit(df[img_feat_cols].values)
        
        # 琛ㄦ牸鐗瑰緛鏍囧噯鍖?
        print(f"  琛ㄦ牸鐗瑰緛: {self.teacher_features}")
        self.table_scaler.fit(df[self.teacher_features].values)
        
        # PCA闄嶇淮
        if len(img_feat_cols) > self.pca_dim:
            print(f"  PCA闄嶇淮: {len(img_feat_cols)} -> {self.pca_dim}")
            self.pca = PCA(n_components=self.pca_dim, random_state=RANDOM_STATE)
            self.pca.fit(df[img_feat_cols].values)
        else:
            print(f"  璺宠繃PCA (鐗瑰緛缁村害 {len(img_feat_cols)} <= {self.pca_dim})")
        
        print("=" * 60)
    
    def get_cluster_datasets(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        鎸塩ity-cluster鍒掑垎鏁版嵁闆?
        杩斿洖: {city_cluster_id: {'X_img': ..., 'X_tab': ..., 'y': ..., 'geoids': ...}}
        """
        print("\n" + "=" * 60)
        print("Step 3: 鍒掑垎cluster鏁版嵁闆?..")
        print("=" * 60)
        
        datasets = {}
        img_feat_cols = [c for c in df.columns if c.startswith('img_feat_')]
        
        unique_clusters = df['city_cluster_id'].unique()
        print(f"  鍏?{len(unique_clusters)} 涓猚ity-clusters")
        
        for cluster_id in unique_clusters:
            cluster_data = df[df['city_cluster_id'] == cluster_id].copy()
            
            # 鎻愬彇鐗瑰緛
            X_img = cluster_data[img_feat_cols].values.astype(np.float32)
            X_tab = cluster_data[self.teacher_features].values.astype(np.float32)
            
            # 鏍囧噯鍖?
            X_img_scaled = self.image_scaler.transform(X_img)
            X_tab_scaled = self.table_scaler.transform(X_tab)
            
            # PCA
            if self.pca is not None:
                X_img_scaled = self.pca.transform(X_img_scaled)
            
            # 7绫婚€氬嫟鐩爣
            y = cluster_data[TARGET_NAMES].values.astype(np.float32)
            
            # 鍒掑垎璁粌/楠岃瘉闆?
            n_samples = len(cluster_data)
            if n_samples < 10:
                print(f"  璀﹀憡: {cluster_id} 鏍锋湰鏁拌繃灏?({n_samples})锛岃烦杩囧垝鍒?)
                train_idx = list(range(n_samples))
                val_idx = []
            else:
                train_idx, val_idx = train_test_split(
                    range(n_samples), test_size=VAL_SPLIT, random_state=RANDOM_STATE
                )
            
            datasets[cluster_id] = {
                'X_img': X_img_scaled,                    # 瀛︾敓妯″瀷杈撳叆 (PCA鍚?
                'X_img_raw': X_img_scaled,                # 鐢ㄤ簬淇濆瓨鑱氱被涓績
                'X_tab': X_tab_scaled,                    # 琛ㄦ牸鐗瑰緛
                'X_teacher': np.hstack([X_img_scaled, X_tab_scaled]),  # 鏁欏笀妯″瀷杈撳叆
                'y': y,                                   # 鐪熷疄鏍囩 (7缁?
                'geoids': cluster_data['GEOID'].values,
                'n_samples': n_samples,
                'train_idx': train_idx,
                'val_idx': val_idx,
                'city_name': str(cluster_data[CBSA_CODE_COLUMN].iloc[0]) if CBSA_CODE_COLUMN in cluster_data.columns else "cbsa",
            }
            
            print(f"  {cluster_id}: {n_samples} samples ({len(train_idx)} train, {len(val_idx)} val)")
        
        print("=" * 60)
        return datasets
    
    def save_preprocessors(self, save_dir: str):
        """淇濆瓨鏍囧噯鍖栧櫒鍜孭CA"""
        import joblib
        joblib.dump(self.image_scaler, os.path.join(save_dir, "image_scaler.pkl"))
        joblib.dump(self.table_scaler, os.path.join(save_dir, "table_scaler.pkl"))
        if self.pca is not None:
            joblib.dump(self.pca, os.path.join(save_dir, "pca.pkl"))
        print(f"棰勫鐞嗗櫒淇濆瓨鑷? {save_dir}")
    
    def load_preprocessors(self, save_dir: str):
        """鍔犺浇鏍囧噯鍖栧櫒鍜孭CA"""
        import joblib
        self.image_scaler = joblib.load(os.path.join(save_dir, "image_scaler.pkl"))
        self.table_scaler = joblib.load(os.path.join(save_dir, "table_scaler.pkl"))
        pca_path = os.path.join(save_dir, "pca.pkl")
        if os.path.exists(pca_path):
            self.pca = joblib.load(pca_path)

