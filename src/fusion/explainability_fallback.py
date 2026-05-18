


"""Robust explainability fallback utilities for Step-7.5 refined evaluation."""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_absolute_error

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger("fusion.explainability_fallback")


def _predict_for_task(model: Any, x_df: pd.DataFrame, task_type: str) -> np.ndarray:
    if str(task_type) == "classification":
        pred = model.predict(x_df)
        return np.asarray(pred)
    pred = model.predict(x_df)
    return np.asarray(pred, dtype=float)


def permutation_importance_holdout(
    *,
    model: Any,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    task_type: str,
    random_seed: int,
    n_repeats: int = 5,
) -> Dict[str, Any]:
    """
    Compute permutation importance on held-out data.

    Regression importance:
    - importance = permuted_MAE - baseline_MAE (higher => more important)

    Classification importance:
    - importance = baseline_macroF1 - permuted_macroF1 (higher => more important)
    """
    x = x_test.copy()
    y = np.asarray(y_test)
    rng = np.random.default_rng(int(random_seed))

    if len(x) == 0 or x.shape[1] == 0:
        return {
            "ok": False,
            "reason": "empty_holdout_or_no_features",
            "rows": [],
            "baseline_metric": None,
            "metric_name": "",
        }

    base_pred = _predict_for_task(model=model, x_df=x, task_type=task_type)
    if str(task_type) == "classification":
        baseline_metric = float(f1_score(y, base_pred, average="macro", zero_division=0))
        metric_name = "macro_f1"
    else:
        baseline_metric = float(mean_absolute_error(y, base_pred))
        metric_name = "mae"

    rows: List[Dict[str, Any]] = []
    for feat in x.columns:
        deltas: List[float] = []
        for _ in range(int(max(1, n_repeats))):
            x_perm = x.copy()
            perm_idx = rng.permutation(len(x_perm))
            x_perm[feat] = x_perm[feat].to_numpy()[perm_idx]
            pred = _predict_for_task(model=model, x_df=x_perm, task_type=task_type)
            if str(task_type) == "classification":
                perm_metric = float(f1_score(y, pred, average="macro", zero_division=0))
                delta = float(baseline_metric - perm_metric)
            else:
                perm_metric = float(mean_absolute_error(y, pred))
                delta = float(perm_metric - baseline_metric)
            if np.isfinite(delta):
                deltas.append(delta)
        if not deltas:
            deltas = [0.0]
        rows.append(
            {
                "feature": str(feat),
                "importance_mean": float(np.mean(deltas)),
                "importance_std": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
                "importance_values": [float(d) for d in deltas],
            }
        )

    rows.sort(key=lambda r: (-float(r["importance_mean"]), str(r["feature"]).lower()))
    return {
        "ok": True,
        "reason": "",
        "metric_name": metric_name,
        "baseline_metric": float(baseline_metric),
        "rows": rows,
    }


def aggregate_permutation_records(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Aggregate fold-level permutation records to table output."""
    if not records:
        return pd.DataFrame(
            columns=[
                "target_name",
                "target_type",
                "target_tier",
                "model_group",
                "feature",
                "importance_mean",
                "importance_std",
                "n_records",
                "metric_name",
            ]
        )
    df = pd.DataFrame(list(records))
    if df.empty:
        return df

    grouped = (
        df.groupby(
            ["target_name", "target_type", "target_tier", "model_group", "feature", "metric_name"],
            dropna=False,
        )["importance"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "importance_mean",
                "std": "importance_std",
                "count": "n_records",
            }
        )
    )
    grouped["importance_std"] = grouped["importance_std"].fillna(0.0)
    grouped = grouped.sort_values(
        ["target_name", "model_group", "importance_mean", "feature"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)
    return grouped


def plot_permutation_importance(
    *,
    permutation_df: pd.DataFrame,
    target_name: str,
    model_group: str,
    out_path: Path,
    top_n: int = 20,
) -> bool:
    sub = permutation_df[
        (permutation_df["target_name"] == target_name)
        & (permutation_df["model_group"] == model_group)
    ].copy()
    if sub.empty:
        return False
    sub = sub.sort_values("importance_mean", ascending=False).head(max(1, int(top_n)))
    if sub.empty:
        return False

    fig_h = max(4.0, 0.35 * len(sub))
    plt.figure(figsize=(10, fig_h))
    plt.barh(
        sub["feature"].astype(str).tolist()[::-1],
        sub["importance_mean"].astype(float).tolist()[::-1],
        xerr=sub["importance_std"].astype(float).tolist()[::-1],
    )
    plt.xlabel("Permutation Importance")
    plt.title(f"Permutation Importance | {target_name} | {model_group}")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    return True


def try_shap_tree_summary(
    *,
    model: Any,
    x_df: pd.DataFrame,
    target_name: str,
    model_group: str,
    out_plots_dir: Path,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Try Tree SHAP; if unavailable/failing return empty dataframe and reason."""
    status: Dict[str, Any] = {
        "target_name": target_name,
        "model_group": model_group,
        "attempted": True,
        "available": False,
        "reason": "",
        "bar_plot": "",
        "beeswarm_plot": "",
    }
    try:
        shap = importlib.import_module("shap")
    except Exception as exc:
        status["reason"] = f"shap_import_failed:{exc}"
        return pd.DataFrame(), status

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_df)
        arr = np.asarray(shap_values, dtype=float)
        if arr.ndim == 3:

            arr = np.mean(np.abs(arr), axis=0)
        if arr.ndim != 2 or arr.shape[1] != x_df.shape[1]:
            status["reason"] = f"shap_shape_unexpected:{arr.shape}"
            return pd.DataFrame(), status

        mean_abs = np.mean(np.abs(arr), axis=0)
        shap_df = pd.DataFrame(
            {
                "target_name": str(target_name),
                "model_group": str(model_group),
                "feature": x_df.columns.astype(str),
                "mean_abs_shap": mean_abs.astype(float),
            }
        ).sort_values("mean_abs_shap", ascending=False)
        shap_df["rank"] = np.arange(1, len(shap_df) + 1)

        bar_path = out_plots_dir / f"target_{target_name}_shap_bar_refined.png"
        bee_path = out_plots_dir / f"target_{target_name}_shap_beeswarm_refined.png"

        shap.summary_plot(arr, x_df, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(bar_path, dpi=150)
        plt.close()

        shap.summary_plot(arr, x_df, show=False)
        plt.tight_layout()
        plt.savefig(bee_path, dpi=150)
        plt.close()

        status["available"] = True
        status["bar_plot"] = bar_path.as_posix()
        status["beeswarm_plot"] = bee_path.as_posix()
        return shap_df.reset_index(drop=True), status
    except Exception as exc:
        status["reason"] = f"shap_runtime_failed:{exc}"
        return pd.DataFrame(), status
