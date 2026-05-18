


"""Model backend selection and estimator utilities for Step-7."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


@dataclass(frozen=True)
class ModelBackendInfo:
    """Resolved tree-model backend and availability details."""

    backend: str
    available: Dict[str, bool]
    fallback_chain: List[str]
    reason: str


def resolve_model_backend() -> ModelBackendInfo:
    """Resolve preferred model backend in order: xgboost -> lightgbm -> sklearn."""
    checks: List[Tuple[str, str]] = [
        ("xgboost", "xgboost"),
        ("lightgbm", "lightgbm"),
        ("sklearn", "sklearn"),
    ]
    available: Dict[str, bool] = {}
    fallback_chain: List[str] = []

    chosen = "sklearn"
    reason = "fallback_to_sklearn"
    for name, mod in checks:
        ok = importlib.util.find_spec(mod) is not None
        available[name] = bool(ok)
        fallback_chain.append(name)
        if ok:
            chosen = name
            reason = f"selected_{name}"
            break

    return ModelBackendInfo(
        backend=chosen,
        available=available,
        fallback_chain=fallback_chain,
        reason=reason,
    )


def ensure_numeric_frame(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Build numeric-safe feature frame from selected columns."""
    out = pd.DataFrame(index=df.index)
    for col in columns:
        if col not in df.columns:
            out[col] = np.nan
            continue
        series = df[col]
        if pd.api.types.is_bool_dtype(series.dtype):
            out[col] = series.astype("int64")
        else:
            out[col] = pd.to_numeric(series, errors="coerce")
    return out


def _xgb_estimator(task_type: str, seed: int, n_classes: int = 2):
    xgb = importlib.import_module("xgboost")
    if task_type == "regression":
        return xgb.XGBRegressor(
            n_estimators=120,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=int(seed),
            n_jobs=1,
        )

    params: Dict[str, Any] = {
        "n_estimators": 140,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "random_state": int(seed),
        "n_jobs": 1,
        "eval_metric": "mlogloss",
    }
    if int(n_classes) <= 2:
        params["objective"] = "binary:logistic"
    else:
        params["objective"] = "multi:softprob"
        params["num_class"] = int(n_classes)
    return xgb.XGBClassifier(**params)


def _lgbm_estimator(task_type: str, seed: int, n_classes: int = 2):
    lgbm = importlib.import_module("lightgbm")
    if task_type == "regression":
        return lgbm.LGBMRegressor(
            n_estimators=160,
            max_depth=4,
            num_leaves=15,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=int(seed),
        )

    params: Dict[str, Any] = {
        "n_estimators": 180,
        "max_depth": 4,
        "num_leaves": 15,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "random_state": int(seed),
    }
    if int(n_classes) <= 2:
        params["objective"] = "binary"
    else:
        params["objective"] = "multiclass"
        params["num_class"] = int(n_classes)
    return lgbm.LGBMClassifier(**params)


def _sklearn_estimator(task_type: str, seed: int):
    if task_type == "regression":
        return HistGradientBoostingRegressor(
            max_depth=3,
            learning_rate=0.05,
            max_iter=240,
            min_samples_leaf=4,
            random_state=int(seed),
        )
    return HistGradientBoostingClassifier(
        max_depth=3,
        learning_rate=0.05,
        max_iter=260,
        min_samples_leaf=4,
        random_state=int(seed),
    )


def build_estimator(
    task_type: str,
    seed: int,
    backend: str,
    n_classes: int = 2,
):
    """Build conservative tree-based estimator for regression/classification."""
    if task_type not in {"regression", "classification"}:
        raise ValueError(f"unsupported task_type: {task_type}")

    if backend == "xgboost":
        return _xgb_estimator(task_type=task_type, seed=seed, n_classes=n_classes)
    if backend == "lightgbm":
        return _lgbm_estimator(task_type=task_type, seed=seed, n_classes=n_classes)
    return _sklearn_estimator(task_type=task_type, seed=seed)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics used in Step-7."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    try:
        r2 = float(r2_score(y_true, y_pred))
    except Exception:
        r2 = float("nan")

    sp = spearmanr(y_true, y_pred, nan_policy="omit")
    spearman_val = float(sp.correlation) if sp is not None else float("nan")

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "spearman": spearman_val,
    }


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> Dict[str, float]:
    """Compute classification metrics used in Step-7."""
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        roc_auc_score,
    )

    out: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "roc_auc": float("nan"),
    }

    if y_proba is not None:
        try:
            if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                out["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
            elif y_proba.ndim == 2 and y_proba.shape[1] > 2:
                out["roc_auc"] = float(
                    roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
                )
        except Exception:
            out["roc_auc"] = float("nan")

    return out


def aligned_predict_proba(
    model: Any,
    x_df: pd.DataFrame,
    global_class_ids: Sequence[int],
) -> np.ndarray:
    """
    Predict class probabilities and align model output to global class id order.

    If estimator does not expose `predict_proba`, returns one-hot of `predict`.
    """
    n_samples = int(len(x_df))
    class_ids = list(global_class_ids)
    out = np.zeros((n_samples, len(class_ids)), dtype=float)

    if hasattr(model, "predict_proba"):
        raw = model.predict_proba(x_df)
        raw = np.asarray(raw, dtype=float)
        model_classes = list(getattr(model, "classes_", class_ids))
        if raw.ndim == 1:
            raw = raw.reshape(-1, 1)

        if raw.shape[1] == len(class_ids) and all(str(a) == str(b) for a, b in zip(model_classes, class_ids)):
            return raw

        col_map = {int(c): i for i, c in enumerate(model_classes)}
        for j, cid in enumerate(class_ids):
            idx = col_map.get(int(cid))
            if idx is not None and idx < raw.shape[1]:
                out[:, j] = raw[:, idx]
        row_sum = out.sum(axis=1)
        zero_mask = row_sum <= 0
        if np.any(zero_mask):
            out[zero_mask, :] = 1.0 / max(1, len(class_ids))
        non_zero = ~zero_mask
        if np.any(non_zero):
            out[non_zero, :] = out[non_zero, :] / row_sum[non_zero, None]
        return out

    preds = np.asarray(model.predict(x_df), dtype=int)
    id_to_pos = {int(cid): i for i, cid in enumerate(class_ids)}
    for i, p in enumerate(preds):
        j = id_to_pos.get(int(p))
        if j is None:
            out[i, :] = 1.0 / max(1, len(class_ids))
        else:
            out[i, j] = 1.0
    return out


def safe_import_optional(module_name: str) -> Tuple[bool, str]:
    """Utility for status reporting of optional dependencies."""
    try:
        importlib.import_module(module_name)
        return True, ""
    except Exception as exc:
        return False, str(exc)
