"""Shared regression metric helpers."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "n_samples": int(y_true.shape[0]),
    }


def group_metrics(
    df: pd.DataFrame,
    group_col: str,
    y_true_col: str,
    y_pred_col: str,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for group_value, grp in df.groupby(group_col):
        y_true = grp[y_true_col].to_numpy(dtype=float)
        y_pred = grp[y_pred_col].to_numpy(dtype=float)
        m = regression_metrics(y_true, y_pred)
        rows.append(
            {
                group_col: group_value,
                "y_true_mean": float(np.mean(y_true)),
                "y_pred_mean": float(np.mean(y_pred)),
                "pred_bias_mean": float(np.mean(y_pred) - np.mean(y_true)),
                **m,
            }
        )
    return pd.DataFrame(rows)
