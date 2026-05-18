


"""Explainability outputs for Step-7 confirmatory models."""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib
import numpy as np
import pandas as pd

from .modeling import build_estimator, ensure_numeric_frame

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger("fusion.explainability")


def _safe_numeric_target(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, np.ndarray]:
    y = pd.to_numeric(df[target_col], errors="coerce")
    mask = y.notna()
    out_df = df.loc[mask].copy()
    return out_df, y.loc[mask].to_numpy(dtype=float)


def _feature_importance_from_model(model: Any, feature_names: Sequence[str]) -> np.ndarray:
    if hasattr(model, "feature_importances_"):
        arr = np.asarray(getattr(model, "feature_importances_"), dtype=float)
        if arr.size == len(feature_names):
            return arr
    if hasattr(model, "coef_"):
        arr = np.asarray(getattr(model, "coef_"), dtype=float)
        arr = np.ravel(arr)
        if arr.size == len(feature_names):
            return np.abs(arr)
    return np.zeros(len(feature_names), dtype=float)


def _plot_bar(
    rows: pd.DataFrame,
    out_path: Path,
    title: str,
    value_col: str,
    top_n: int = 20,
) -> None:
    if rows.empty:
        return
    data = rows.sort_values(value_col, ascending=False).head(top_n).copy()
    fig_h = max(4.0, 0.38 * len(data))
    plt.figure(figsize=(10, fig_h))
    plt.barh(data["feature"].astype(str).tolist()[::-1], data[value_col].astype(float).tolist()[::-1])
    plt.title(title)
    plt.xlabel(value_col)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_confirmatory_comparison(
    per_target_df: pd.DataFrame,
    confirmatory_targets: Sequence[str],
    out_path: Path,
) -> None:
    if per_target_df.empty:
        return
    subset = per_target_df[
        (per_target_df["target_type"] == "regression")
        & (per_target_df["target_name"].isin(list(confirmatory_targets)))
    ].copy()
    if subset.empty or "mae_mean" not in subset.columns:
        return

    targets = list(dict.fromkeys(subset["target_name"].astype(str).tolist()))
    groups = list(dict.fromkeys(subset["model_group"].astype(str).tolist()))
    x = np.arange(len(targets))
    width = 0.8 / max(1, len(groups))

    plt.figure(figsize=(12, 5.5))
    for i, group in enumerate(groups):
        vals = []
        for t in targets:
            row = subset[(subset["target_name"] == t) & (subset["model_group"] == group)]
            vals.append(float(row["mae_mean"].iloc[0]) if not row.empty else np.nan)
        offset = (i - (len(groups) - 1) / 2) * width
        plt.bar(x + offset, vals, width=width, label=group)

    plt.xticks(x, targets, rotation=20, ha="right")
    plt.ylabel("MAE (lower is better)")
    plt.title("Confirmatory Target Model Comparison")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def _compute_shap_for_regression(
    model: Any,
    x_df: pd.DataFrame,
    target_name: str,
    out_plots_dir: Path,
) -> Tuple[pd.DataFrame, List[str]]:
    warnings: List[str] = []
    try:
        shap = importlib.import_module("shap")
    except Exception as exc:
        return pd.DataFrame(), [f"shap_unavailable:{exc}"]

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_df)
        arr = np.asarray(shap_values, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != x_df.shape[1]:
            return pd.DataFrame(), [f"shap_shape_unexpected:{arr.shape}"]

        mean_abs = np.mean(np.abs(arr), axis=0)
        summary_df = pd.DataFrame(
            {
                "feature": x_df.columns.astype(str),
                "mean_abs_shap": mean_abs.astype(float),
            }
        ).sort_values("mean_abs_shap", ascending=False)

        bar_path = out_plots_dir / f"target_{target_name}_shap_bar.png"
        bee_path = out_plots_dir / f"target_{target_name}_shap_beeswarm.png"

        shap.summary_plot(arr, x_df, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(bar_path, dpi=150)
        plt.close()

        shap.summary_plot(arr, x_df, show=False)
        plt.tight_layout()
        plt.savefig(bee_path, dpi=150)
        plt.close()

        return summary_df, warnings
    except Exception as exc:
        warnings.append(f"shap_failed:{exc}")
        return pd.DataFrame(), warnings


def run_explainability(
    merged_df: pd.DataFrame,
    feature_groups: Mapping[str, Sequence[str]],
    target_registry: Mapping[str, Any],
    per_target_metrics_csv: str,
    out_dir: Path,
    backend: str,
    seed: int,
) -> Dict[str, Any]:
    """Generate Step-7 explainability artifacts for confirmatory regression targets."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.read_csv(per_target_metrics_csv) if Path(per_target_metrics_csv).is_file() else pd.DataFrame()

    confirmatory_targets = [
        str(t["target_name"])
        for t in target_registry.get("targets", [])
        if str(t.get("target_type")) == "regression"
        and str(t.get("tier")) == "confirmatory"
        and bool(t.get("enabled", False))
    ]

    _plot_confirmatory_comparison(
        per_target_df=metrics_df,
        confirmatory_targets=confirmatory_targets,
        out_path=plots_dir / "confirmatory_model_comparison.png",
    )

    importance_rows: List[Dict[str, Any]] = []
    shap_rows: List[Dict[str, Any]] = []
    warnings: List[str] = []

    for target_name in confirmatory_targets:
        if target_name not in merged_df.columns:
            warnings.append(f"missing_target_for_explainability:{target_name}")
            continue

        target_metrics = metrics_df[
            (metrics_df["target_name"] == target_name)
            & (metrics_df["target_type"] == "regression")
        ].copy()
        if target_metrics.empty or "mae_mean" not in target_metrics.columns:
            warnings.append(f"missing_metrics_for_target:{target_name}")
            continue
        target_metrics = target_metrics.sort_values("mae_mean", ascending=True)
        best_group = str(target_metrics.iloc[0]["model_group"])

        groups_to_train: List[str] = []
        for g in (best_group, "early_fusion", "visual_only", "audio_only"):
            if g in groups_to_train:
                continue
            cols = [c for c in feature_groups.get(g, []) if c in merged_df.columns]
            if cols:
                groups_to_train.append(g)

        for group_name in groups_to_train:
            feature_cols = [c for c in feature_groups.get(group_name, []) if c in merged_df.columns]
            if not feature_cols:
                continue

            work_df, y = _safe_numeric_target(merged_df, target_name)
            if len(work_df) < 8:
                warnings.append(f"target={target_name} group={group_name} insufficient_rows_for_explainability")
                continue
            x_df = ensure_numeric_frame(work_df, feature_cols)

            try:
                model = build_estimator(task_type="regression", seed=seed + 777, backend=backend)
                model.fit(x_df, y)
            except Exception as exc:
                warnings.append(f"target={target_name} group={group_name} fit_failed:{exc}")
                continue

            importances = _feature_importance_from_model(model, feature_cols)
            imp_df = pd.DataFrame({"feature": feature_cols, "importance": importances.astype(float)})
            imp_df = imp_df.sort_values("importance", ascending=False).reset_index(drop=True)

            for rank, (_, row) in enumerate(imp_df.iterrows(), start=1):
                importance_rows.append(
                    {
                        "target_name": target_name,
                        "model_group": group_name,
                        "feature": str(row["feature"]),
                        "importance": float(row["importance"]),
                        "rank": int(rank),
                    }
                )

            if group_name == best_group:
                _plot_bar(
                    rows=imp_df.rename(columns={"importance": "importance_value"}),
                    out_path=plots_dir / f"target_{target_name}_feature_importance.png",
                    title=f"Feature Importance | {target_name} | {group_name}",
                    value_col="importance_value",
                    top_n=20,
                )
                shap_df, shap_warn = _compute_shap_for_regression(
                    model=model,
                    x_df=x_df,
                    target_name=target_name,
                    out_plots_dir=plots_dir,
                )
                warnings.extend(shap_warn)
                if not shap_df.empty:
                    for rank, (_, row) in enumerate(shap_df.iterrows(), start=1):
                        shap_rows.append(
                            {
                                "target_name": target_name,
                                "model_group": group_name,
                                "feature": str(row["feature"]),
                                "mean_abs_shap": float(row["mean_abs_shap"]),
                                "rank": int(rank),
                            }
                        )

    feature_importance_df = pd.DataFrame(importance_rows)
    if not feature_importance_df.empty:
        feature_importance_df = feature_importance_df.sort_values(
            ["target_name", "model_group", "rank"]
        ).reset_index(drop=True)

    shap_summary_df = pd.DataFrame(shap_rows)
    if not shap_summary_df.empty:
        shap_summary_df = shap_summary_df.sort_values(
            ["target_name", "model_group", "rank"]
        ).reset_index(drop=True)

    feature_importance_path = out_dir / "feature_importance.csv"
    shap_summary_path = out_dir / "shap_summary.csv"
    feature_importance_df.to_csv(feature_importance_path, index=False, encoding="utf-8")
    shap_summary_df.to_csv(shap_summary_path, index=False, encoding="utf-8")

    logger.info(
        "step7 explainability done | confirmatory_targets=%d feature_rows=%d shap_rows=%d",
        len(confirmatory_targets),
        len(feature_importance_df),
        len(shap_summary_df),
    )
    return {
        "feature_importance_csv": feature_importance_path.as_posix(),
        "shap_summary_csv": shap_summary_path.as_posix(),
        "plots_dir": plots_dir.as_posix(),
        "warnings": warnings,
        "confirmatory_targets": confirmatory_targets,
    }
