


"""Step-7.5 refined fusion evaluation (additive, leakage-safe, explainability-robust)."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, RepeatedKFold, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from src.config import (
    STEP75_BOOTSTRAP_CI_ALPHA,
    STEP75_BOOTSTRAP_SAMPLES,
    STEP75_CLASS_MIN_COUNT,
    STEP75_REG_CV_REPEATS,
    STEP75_REG_CV_SPLITS,
    STEP75_REUSE_STEP7_SPLITS,
    STEP75_SCREEN_CORR_THRESHOLD,
    STEP75_SCREEN_MIN_MODALITY_EARLY,
    STEP75_SCREEN_MISSING_THRESHOLD,
    STEP75_SCREEN_TOPK_AUDIO,
    STEP75_SCREEN_TOPK_EARLY,
    STEP75_SCREEN_TOPK_VISUAL,
    STEP75_SCREEN_VARIANCE_THRESHOLD,
    STEP75_SEED,
)

from .explainability_fallback import (
    aggregate_permutation_records,
    permutation_importance_holdout,
    plot_permutation_importance,
    try_shap_tree_summary,
)
from .feature_screening import FeatureScreeningConfig, screen_features_for_fold
from .modeling import (
    aligned_predict_proba,
    build_estimator,
    classification_metrics,
    ensure_numeric_frame,
    regression_metrics,
    resolve_model_backend,
)
from .target_registry import (
    CONFIRMATORY_REGRESSION_TARGETS,
    EXPLORATORY_CLASSIFICATION_TARGETS,
    EXPLORATORY_REGRESSION_TARGETS,
    build_target_registry,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger("fusion.step75_runner")

VISUAL_SOURCE_GROUPS = {
    "visual_semantic",
    "visual_major",
    "green_view",
    "emotion",
    "people",
    "color",
    "ai_activity",
    "ai_activity_summary",
}
AUDIO_SOURCE_GROUPS = {"audio_events", "audio_signal", "audio_embedding"}

MODEL_GROUPS_REFINED: Tuple[str, ...] = (
    "visual_only_screened",
    "audio_only_screened",
    "early_fusion_screened",
    "late_fusion_legacy",
)
PAIR_COMPARISONS_REFINED: Tuple[Tuple[str, str], ...] = (
    ("early_fusion_screened", "visual_only_screened"),
    ("early_fusion_screened", "audio_only_screened"),
    ("visual_only_screened", "audio_only_screened"),
)
REG_METRICS: Tuple[str, ...] = ("mae", "rmse", "r2", "spearman")
CLS_METRICS: Tuple[str, ...] = ("accuracy", "macro_f1", "balanced_accuracy", "roc_auc")
ALL_METRICS: Tuple[str, ...] = (*REG_METRICS, *CLS_METRICS)
STEP75_LABEL_NUMERIC_COLUMNS: Tuple[str, ...] = (
    "comfort_score",
    "vitality_score",
    "soundscape_pleasantness",
    "soundscape_eventfulness",
    "overall_problem_severity",
)


def _emit_progress(
    progress_callback: Optional[Any],
    completed: float,
    total: float,
    description: str,
) -> None:
    if progress_callback is None:
        return
    try:
        progress_callback(float(completed), float(total), str(description))
    except Exception:
        logger.debug("step7.5 progress callback failed", exc_info=True)


def _inspect_validation_labels_schema(labels_df: pd.DataFrame) -> Dict[str, Any]:
    present_columns = [str(c) for c in labels_df.columns.tolist()]
    target_columns = [
        *list(CONFIRMATORY_REGRESSION_TARGETS),
        *list(EXPLORATORY_REGRESSION_TARGETS),
        *list(EXPLORATORY_CLASSIFICATION_TARGETS),
    ]
    non_missing_counts: Dict[str, int] = {}
    out_of_range_counts: Dict[str, int] = {}
    for col in STEP75_LABEL_NUMERIC_COLUMNS:
        if col not in labels_df.columns:
            continue
        ser = pd.to_numeric(labels_df[col], errors="coerce")
        non_missing_counts[col] = int(ser.notna().sum())
        in_range = ser.dropna().between(1, 5, inclusive="both")
        out_of_range_counts[col] = int((~in_range).sum())

    cls_non_missing_counts: Dict[str, int] = {}
    for col in EXPLORATORY_CLASSIFICATION_TARGETS:
        if col not in labels_df.columns:
            continue
        ser = labels_df[col].astype("string").str.strip()
        ser = ser.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
        cls_non_missing_counts[col] = int(ser.notna().sum())

    summary = {
        "present_columns": present_columns,
        "required_columns_present": "segment_id" in labels_df.columns,
        "target_columns_present": [col for col in target_columns if col in labels_df.columns],
        "target_columns_missing": [col for col in target_columns if col not in labels_df.columns],
        "numeric_target_non_missing_counts": non_missing_counts,
        "numeric_target_out_of_range_counts": out_of_range_counts,
        "classification_target_non_missing_counts": cls_non_missing_counts,
        "usable_target_columns": [
            col
            for col, count in {**non_missing_counts, **cls_non_missing_counts}.items()
            if int(count) >= 5
        ],
    }
    summary["compatible_for_step75"] = bool(
        summary["required_columns_present"] and len(summary["usable_target_columns"]) > 0
    )
    return summary


def _safe_read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return {}
    return {}


def _safe_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _normalize_segment_id(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if "segment_id" not in df.columns:
        raise ValueError(f"{name} missing required column: segment_id")
    out = df.copy()
    out["segment_id"] = pd.to_numeric(out["segment_id"], errors="coerce")
    out = out.dropna(subset=["segment_id"]).copy()
    out["segment_id"] = out["segment_id"].astype(int)
    return out


def _stable_text_seed(text: str, base: int = 0) -> int:
    acc = int(base)
    for i, ch in enumerate(str(text)):
        acc += (i + 1) * ord(ch)
    return int(acc)


def _source_group_for_col(col: str, feature_meta: Mapping[str, Any]) -> str:
    item = feature_meta.get(col, {})
    if isinstance(item, dict) and item.get("source_group"):
        return str(item["source_group"])
    if "__" in col:
        return str(col.split("__", 1)[0])
    return "unknown"


def _build_base_feature_groups(
    model_df: pd.DataFrame,
    model_feature_dict: Mapping[str, Any],
) -> Dict[str, Any]:
    feature_cols = [c for c in model_df.columns if c != "segment_id"]
    feature_meta = model_feature_dict.get("feature_metadata", {})
    if not isinstance(feature_meta, dict):
        feature_meta = {}

    visual: List[str] = []
    audio: List[str] = []
    group_assignments: Dict[str, str] = {}
    unknown: List[str] = []
    for col in feature_cols:
        group = _source_group_for_col(col, feature_meta)
        group_assignments[str(col)] = str(group)
        if group in VISUAL_SOURCE_GROUPS:
            visual.append(str(col))
        elif group in AUDIO_SOURCE_GROUPS:
            audio.append(str(col))
        else:
            unknown.append(str(col))

    visual = list(dict.fromkeys(visual))
    audio = list(dict.fromkeys(audio))
    early = list(dict.fromkeys([*visual, *audio]))
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "groups": {
            "visual_only": visual,
            "audio_only": audio,
            "early_fusion": early,
        },
        "feature_counts": {
            "all_model_features": int(len(feature_cols)),
            "visual_only": int(len(visual)),
            "audio_only": int(len(audio)),
            "early_fusion": int(len(early)),
            "unknown_group_features": int(len(unknown)),
        },
        "group_assignments": group_assignments,
        "unknown_group_features": unknown,
    }


def _resolve_paths(
    video_dir: str,
    feature_csv: Optional[str] = None,
    labels_csv: Optional[str] = None,
    out_dir: Optional[str] = None,
) -> Dict[str, Path]:
    vdir = Path(video_dir)
    feature_path = Path(feature_csv) if feature_csv else (vdir / "fusion" / "model_feature_table.csv")
    feature_dict_path = vdir / "fusion" / "model_feature_dictionary.json"

    adj_labels = vdir / "validation" / "final_annotation_labels_adjudicated.csv"
    fallback_labels = vdir / "validation" / "final_annotation_labels.csv"
    if labels_csv:
        label_path = Path(labels_csv)
        label_source = "explicit_cli"
    elif adj_labels.is_file():
        label_path = adj_labels
        label_source = "adjudicated_default"
    elif fallback_labels.is_file():
        label_path = fallback_labels
        label_source = "fallback_non_adjudicated"
    else:
        label_path = adj_labels
        label_source = "missing_default"

    return {
        "video_dir": vdir,
        "feature_csv": feature_path,
        "feature_dict_json": feature_dict_path,
        "labels_csv": label_path,
        "labels_source_type": Path(label_source),
        "adjudication_report_json": vdir / "validation" / "adjudication_report.json",
        "step7_target_registry_json": vdir / "fusion_eval" / "target_registry.json",
        "step7_cv_registry_json": vdir / "fusion_eval" / "cv_split_registry.json",
        "step7_per_target_metrics_csv": vdir / "fusion_eval" / "per_target_metrics.csv",
        "step7_model_comparison_csv": vdir / "fusion_eval" / "model_comparison.csv",
        "out_dir": Path(out_dir) if out_dir else (vdir / "fusion_eval_refined"),
    }


def _split_meta(fold_idx: int, n_splits: int) -> Tuple[int, int]:
    rep = int(fold_idx // max(1, n_splits))
    split = int(fold_idx % max(1, n_splits))
    return rep, split


def _build_reg_splits(seg_ids: np.ndarray, n_splits: int, n_repeats: int, seed: int) -> List[Dict[str, Any]]:
    n = int(len(seg_ids))
    if n < 2:
        return []
    n_splits_eff = int(min(max(2, int(n_splits)), n))
    n_repeats_eff = int(max(1, int(n_repeats)))
    rkf = RepeatedKFold(n_splits=n_splits_eff, n_repeats=n_repeats_eff, random_state=int(seed))
    rows: List[Dict[str, Any]] = []
    for i, (tr, te) in enumerate(rkf.split(np.arange(n))):
        rep, split = _split_meta(i, n_splits_eff)
        rows.append(
            {
                "fold_id": f"r{rep:02d}_f{split:02d}",
                "repeat_index": int(rep),
                "split_index": int(split),
                "train_idx": tr.astype(int),
                "test_idx": te.astype(int),
            }
        )
    return rows


def _build_cls_splits(
    seg_ids: np.ndarray,
    y_codes: np.ndarray,
    n_splits: int,
    n_repeats: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], str]:
    uniq, counts = np.unique(y_codes, return_counts=True)
    if uniq.size < 2:
        return [], "single_class_only"
    min_count = int(np.min(counts))
    n_splits_eff = int(min(max(2, int(n_splits)), min_count))
    if n_splits_eff < 2:
        return [], "class_support_too_sparse_for_stratified_kfold"
    n_repeats_eff = int(max(1, int(n_repeats)))
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits_eff,
        n_repeats=n_repeats_eff,
        random_state=int(seed),
    )
    rows: List[Dict[str, Any]] = []
    for i, (tr, te) in enumerate(rskf.split(np.arange(len(seg_ids)), y_codes)):
        rep, split = _split_meta(i, n_splits_eff)
        rows.append(
            {
                "fold_id": f"r{rep:02d}_f{split:02d}",
                "repeat_index": int(rep),
                "split_index": int(split),
                "train_idx": tr.astype(int),
                "test_idx": te.astype(int),
            }
        )
    return rows, ""


def _reuse_step7_splits_for_target(
    *,
    step7_cv_payload: Mapping[str, Any],
    target_name: str,
    target_type: str,
    seg_ids: np.ndarray,
    smoke_test: bool,
) -> Tuple[List[Dict[str, Any]], str]:
    section = "classification" if str(target_type) == "classification" else "regression"
    sec = step7_cv_payload.get(section, {})
    if not isinstance(sec, dict) or target_name not in sec:
        return [], "target_not_found_in_step7_cv_registry"
    item = sec.get(target_name, {})
    folds = item.get("folds", [])
    if not isinstance(folds, list) or not folds:
        return [], "step7_cv_folds_missing"

    idx_map = {int(sid): i for i, sid in enumerate(seg_ids.tolist())}
    rows: List[Dict[str, Any]] = []
    for f in folds:
        tr_ids = [int(x) for x in f.get("train_segment_ids", [])]
        te_ids = [int(x) for x in f.get("test_segment_ids", [])]
        if not tr_ids or not te_ids:
            return [], "invalid_fold_segment_ids"
        if any(sid not in idx_map for sid in tr_ids + te_ids):
            return [], "step7_split_segment_ids_not_subset_of_current_target_data"
        tr = np.asarray([idx_map[sid] for sid in tr_ids], dtype=int)
        te = np.asarray([idx_map[sid] for sid in te_ids], dtype=int)
        rows.append(
            {
                "fold_id": str(f.get("fold_id", "")) or f"r{int(f.get('repeat_index', 0)):02d}_f{int(f.get('split_index', 0)):02d}",
                "repeat_index": int(f.get("repeat_index", 0)),
                "split_index": int(f.get("split_index", 0)),
                "train_idx": tr,
                "test_idx": te,
            }
        )

    if bool(smoke_test):
        n_splits = int(item.get("n_splits", STEP75_REG_CV_SPLITS))
        keep = max(1, n_splits * 2)
        rows = rows[:keep]
        for i, row in enumerate(rows):
            rep, split = _split_meta(i, n_splits)
            row["repeat_index"] = int(rep)
            row["split_index"] = int(split)
            row["fold_id"] = f"r{rep:02d}_f{split:02d}"
    return rows, ""


def _bootstrap_ci(values: Sequence[float], n_bootstrap: int, alpha: float, seed: int) -> Dict[str, Any]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return {
            "ok": False,
            "reason": "insufficient_values",
            "n": int(arr.size),
            "mean": float(np.mean(arr)) if arr.size > 0 else None,
            "ci_lower": None,
            "ci_upper": None,
        }
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(n_bootstrap), dtype=float)
    for i in range(int(n_bootstrap)):
        draws[i] = float(np.mean(rng.choice(arr, size=arr.size, replace=True)))
    lo = float((1.0 - alpha) / 2.0)
    hi = float(1.0 - lo)
    return {
        "ok": True,
        "reason": "",
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "ci_lower": float(np.quantile(draws, lo)),
        "ci_upper": float(np.quantile(draws, hi)),
    }


def _plot_confirmatory_comparison(
    per_target_df: pd.DataFrame,
    confirm_targets: Sequence[str],
    out_path: Path,
) -> None:
    sub = per_target_df[
        (per_target_df["target_type"] == "regression")
        & (per_target_df["target_name"].isin(list(confirm_targets)))
    ].copy()
    if sub.empty or "mae_mean" not in sub.columns:
        return
    targets = list(dict.fromkeys(sub["target_name"].astype(str).tolist()))
    groups = list(dict.fromkeys(sub["model_group"].astype(str).tolist()))
    x = np.arange(len(targets))
    width = 0.82 / max(1, len(groups))
    plt.figure(figsize=(13, 5.6))
    for i, group in enumerate(groups):
        vals = []
        for t in targets:
            row = sub[(sub["target_name"] == t) & (sub["model_group"] == group)]
            vals.append(float(row["mae_mean"].iloc[0]) if not row.empty else np.nan)
        offset = (i - (len(groups) - 1) / 2) * width
        plt.bar(x + offset, vals, width=width, label=group)
    plt.xticks(x, targets, rotation=20, ha="right")
    plt.ylabel("MAE (lower is better)")
    plt.title("Step 7.5 Confirmatory Model Comparison (Refined)")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def _make_inner_split_regression(n_train: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    if n_train < 4:
        return []
    n_splits = int(min(5, max(2, n_train // 2)))
    if n_splits < 2:
        return []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    idx = np.arange(n_train)
    return [(tr.astype(int), te.astype(int)) for tr, te in kf.split(idx)]


def _make_inner_split_classification(y_train: np.ndarray, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    n_train = int(y_train.size)
    if n_train < 4:
        return []

    uniq, counts = np.unique(y_train, return_counts=True)
    if uniq.size >= 2:
        min_count = int(np.min(counts))
        n_splits = int(min(4, min_count))
        if n_splits >= 2:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
            idx = np.arange(n_train)
            return [(tr.astype(int), te.astype(int)) for tr, te in skf.split(idx, y_train)]

    kf = KFold(n_splits=2, shuffle=True, random_state=int(seed))
    idx = np.arange(n_train)
    return [(tr.astype(int), te.astype(int)) for tr, te in kf.split(idx)]


def _late_fusion_regression_predict(
    x_vis: pd.DataFrame,
    x_audio: pd.DataFrame,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    backend: str,
    seed: int,
) -> np.ndarray:
    x_v_tr = x_vis.iloc[train_idx].reset_index(drop=True)
    x_v_te = x_vis.iloc[test_idx].reset_index(drop=True)
    x_a_tr = x_audio.iloc[train_idx].reset_index(drop=True)
    x_a_te = x_audio.iloc[test_idx].reset_index(drop=True)
    y_tr = y[train_idx]

    inner_splits = _make_inner_split_regression(n_train=len(x_v_tr), seed=seed + 31)
    meta_train = np.full((len(x_v_tr), 2), np.nan, dtype=float)

    for inner_i, (sub_tr, sub_val) in enumerate(inner_splits):
        seed_i = int(seed + 101 + inner_i)

        mv = build_estimator("regression", seed=seed_i, backend=backend)
        mv.fit(x_v_tr.iloc[sub_tr], y_tr[sub_tr])
        meta_train[sub_val, 0] = mv.predict(x_v_tr.iloc[sub_val])

        ma = build_estimator("regression", seed=seed_i + 1000, backend=backend)
        ma.fit(x_a_tr.iloc[sub_tr], y_tr[sub_tr])
        meta_train[sub_val, 1] = ma.predict(x_a_tr.iloc[sub_val])

    mv_full = build_estimator("regression", seed=seed + 9001, backend=backend)
    mv_full.fit(x_v_tr, y_tr)
    pred_v_train = np.asarray(mv_full.predict(x_v_tr), dtype=float)
    pred_v_test = np.asarray(mv_full.predict(x_v_te), dtype=float)

    ma_full = build_estimator("regression", seed=seed + 9002, backend=backend)
    ma_full.fit(x_a_tr, y_tr)
    pred_a_train = np.asarray(ma_full.predict(x_a_tr), dtype=float)
    pred_a_test = np.asarray(ma_full.predict(x_a_te), dtype=float)

    nan_v = np.isnan(meta_train[:, 0])
    nan_a = np.isnan(meta_train[:, 1])
    if np.any(nan_v):
        meta_train[nan_v, 0] = pred_v_train[nan_v]
    if np.any(nan_a):
        meta_train[nan_a, 1] = pred_a_train[nan_a]

    x_meta_train = pd.DataFrame(meta_train, columns=["pred_visual", "pred_audio"])
    x_meta_test = pd.DataFrame(
        np.column_stack([pred_v_test, pred_a_test]),
        columns=["pred_visual", "pred_audio"],
    )

    meta_model = build_estimator("regression", seed=seed + 9900, backend=backend)
    meta_model.fit(x_meta_train, y_tr)
    return np.asarray(meta_model.predict(x_meta_test), dtype=float)


def _late_fusion_classification_predict(
    x_vis: pd.DataFrame,
    x_audio: pd.DataFrame,
    y_codes: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    backend: str,
    seed: int,
    class_ids: Sequence[int],
) -> np.ndarray:
    x_v_tr = x_vis.iloc[train_idx].reset_index(drop=True)
    x_v_te = x_vis.iloc[test_idx].reset_index(drop=True)
    x_a_tr = x_audio.iloc[train_idx].reset_index(drop=True)
    x_a_te = x_audio.iloc[test_idx].reset_index(drop=True)
    y_tr = y_codes[train_idx]

    class_ids = [int(c) for c in class_ids]
    n_cls = len(class_ids)
    inner_splits = _make_inner_split_classification(y_tr, seed=seed + 37)

    meta_vis = np.full((len(x_v_tr), n_cls), np.nan, dtype=float)
    meta_audio = np.full((len(x_v_tr), n_cls), np.nan, dtype=float)

    for inner_i, (sub_tr, sub_val) in enumerate(inner_splits):
        seed_i = int(seed + 151 + inner_i)

        mv = build_estimator("classification", seed=seed_i, backend=backend, n_classes=n_cls)
        mv.fit(x_v_tr.iloc[sub_tr], y_tr[sub_tr])
        meta_vis[sub_val, :] = aligned_predict_proba(mv, x_v_tr.iloc[sub_val], class_ids)

        ma = build_estimator("classification", seed=seed_i + 1000, backend=backend, n_classes=n_cls)
        ma.fit(x_a_tr.iloc[sub_tr], y_tr[sub_tr])
        meta_audio[sub_val, :] = aligned_predict_proba(ma, x_a_tr.iloc[sub_val], class_ids)

    mv_full = build_estimator("classification", seed=seed + 9101, backend=backend, n_classes=n_cls)
    mv_full.fit(x_v_tr, y_tr)
    prob_v_train = aligned_predict_proba(mv_full, x_v_tr, class_ids)
    prob_v_test = aligned_predict_proba(mv_full, x_v_te, class_ids)

    ma_full = build_estimator("classification", seed=seed + 9102, backend=backend, n_classes=n_cls)
    ma_full.fit(x_a_tr, y_tr)
    prob_a_train = aligned_predict_proba(ma_full, x_a_tr, class_ids)
    prob_a_test = aligned_predict_proba(ma_full, x_a_te, class_ids)

    nan_vis = np.isnan(meta_vis).any(axis=1)
    nan_audio = np.isnan(meta_audio).any(axis=1)
    if np.any(nan_vis):
        meta_vis[nan_vis, :] = prob_v_train[nan_vis, :]
    if np.any(nan_audio):
        meta_audio[nan_audio, :] = prob_a_train[nan_audio, :]

    meta_train = np.hstack([meta_vis, meta_audio])
    meta_test = np.hstack([prob_v_test, prob_a_test])

    meta_cols = [f"visual_cls_{i}" for i in range(n_cls)] + [f"audio_cls_{i}" for i in range(n_cls)]
    x_meta_train = pd.DataFrame(meta_train, columns=meta_cols)
    x_meta_test = pd.DataFrame(meta_test, columns=meta_cols)

    meta_model = build_estimator("classification", seed=seed + 9991, backend=backend, n_classes=n_cls)
    meta_model.fit(x_meta_train, y_tr)
    return aligned_predict_proba(meta_model, x_meta_test, class_ids)


def _build_oof_rows_regression(
    segment_ids: np.ndarray,
    target_name: str,
    target_tier: str,
    model_group: str,
    fold_id: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i in range(y_true.size):
        out.append(
            {
                "segment_id": int(segment_ids[i]),
                "target_name": target_name,
                "target_type": "regression",
                "target_tier": target_tier,
                "model_group": model_group,
                "fold_id": fold_id,
                "y_true": float(y_true[i]),
                "y_pred": float(y_pred[i]),
            }
        )
    return out


def _build_oof_rows_classification(
    segment_ids: np.ndarray,
    target_name: str,
    target_tier: str,
    model_group: str,
    fold_id: str,
    y_true_labels: Sequence[str],
    y_pred_labels: Sequence[str],
    proba: np.ndarray,
    class_labels: Sequence[str],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i in range(len(y_true_labels)):
        row: Dict[str, Any] = {
            "segment_id": int(segment_ids[i]),
            "target_name": target_name,
            "target_type": "classification",
            "target_tier": target_tier,
            "model_group": model_group,
            "fold_id": fold_id,
            "y_true": str(y_true_labels[i]),
            "y_pred": str(y_pred_labels[i]),
        }
        for cidx, cname in enumerate(class_labels):
            row[f"proba__{cname}"] = float(proba[i, cidx])
        out.append(row)
    return out


def _paired_delta_records(
    fold_metrics_df: pd.DataFrame,
    metric_cols: Sequence[str],
) -> Tuple[pd.DataFrame, Dict[str, List[float]]]:
    records: List[Dict[str, Any]] = []
    bootstrap_map: Dict[str, List[float]] = {}
    if fold_metrics_df.empty:
        return pd.DataFrame(), bootstrap_map

    for target_name in sorted(fold_metrics_df["target_name"].dropna().unique().tolist()):
        sub_t = fold_metrics_df[fold_metrics_df["target_name"] == target_name].copy()
        if sub_t.empty:
            continue
        t_type = str(sub_t["target_type"].dropna().iloc[0])
        active_metrics = [m for m in metric_cols if m in sub_t.columns]
        for lhs, rhs in PAIR_COMPARISONS_REFINED:
            ldf = sub_t[sub_t["model_group"] == lhs].copy()
            rdf = sub_t[sub_t["model_group"] == rhs].copy()
            if ldf.empty or rdf.empty:
                continue
            merged = ldf.merge(
                rdf,
                on=["target_name", "fold_id"],
                how="inner",
                suffixes=("_lhs", "_rhs"),
            )
            if merged.empty:
                continue
            for metric in active_metrics:
                m_l = f"{metric}_lhs"
                m_r = f"{metric}_rhs"
                if m_l not in merged.columns or m_r not in merged.columns:
                    continue
                vals = (merged[m_l] - merged[m_r]).replace([np.inf, -np.inf], np.nan).dropna()
                if vals.empty:
                    continue
                arr = vals.astype(float).tolist()
                key = f"{target_name}::{metric}::{lhs}_vs_{rhs}"
                bootstrap_map[key] = arr
                records.append(
                    {
                        "target_name": target_name,
                        "target_type": t_type,
                        "metric": metric,
                        "comparison": f"{lhs}_vs_{rhs}",
                        "delta_definition": f"{lhs}_minus_{rhs}",
                        "delta_mean": float(np.mean(arr)),
                        "delta_std": float(np.std(arr, ddof=1)) if len(arr) > 1 else float("nan"),
                        "n_folds": int(len(arr)),
                    }
                )
    out_df = pd.DataFrame(records)
    if not out_df.empty:
        out_df = out_df.sort_values(["target_name", "metric", "comparison"]).reset_index(drop=True)
    return out_df, bootstrap_map


def _build_step7_step75_comparison(
    step7_per_target_path: Path,
    per_target_refined_df: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "target_name",
        "target_type",
        "target_tier",
        "model_group_refined",
        "model_group_step7",
        "metric",
        "step7_value",
        "step75_value",
        "delta_step75_minus_step7",
    ]
    if not step7_per_target_path.is_file() or per_target_refined_df.empty:
        return pd.DataFrame(columns=columns)

    try:
        step7_df = pd.read_csv(step7_per_target_path)
    except Exception:
        return pd.DataFrame(columns=columns)
    if step7_df.empty:
        return pd.DataFrame(columns=columns)

    step7_map = {
        "visual_only_screened": "visual_only",
        "audio_only_screened": "audio_only",
        "early_fusion_screened": "early_fusion",
        "late_fusion_legacy": "late_fusion",
    }
    metrics = [m for m in [f"{k}_mean" for k in ALL_METRICS] if m in per_target_refined_df.columns and m in step7_df.columns]
    if not metrics:
        return pd.DataFrame(columns=columns)

    ref = per_target_refined_df.copy()
    ref["model_group_step7_target"] = ref["model_group"].map(step7_map)
    merged = ref.merge(
        step7_df,
        left_on=["target_name", "target_type", "model_group_step7_target"],
        right_on=["target_name", "target_type", "model_group"],
        how="left",
        suffixes=("_step75", "_step7"),
    )

    rows: List[Dict[str, Any]] = []
    for _, row in merged.iterrows():
        model_group_refined = row.get("model_group_step75", row.get("model_group", ""))
        model_group_step7 = row.get("model_group_step7_target", "")
        for metric in metrics:
            v75 = row.get(f"{metric}_step75", row.get(metric, np.nan))
            v7 = row.get(f"{metric}_step7", np.nan)
            if not (pd.notna(v75) and pd.notna(v7)):
                continue
            rows.append(
                {
                    "target_name": row["target_name"],
                    "target_type": row["target_type"],
                    "target_tier": row.get("target_tier_step75", row.get("target_tier", "")),
                    "model_group_refined": model_group_refined,
                    "model_group_step7": model_group_step7,
                    "metric": metric,
                    "step7_value": float(v7),
                    "step75_value": float(v75),
                    "delta_step75_minus_step7": float(v75) - float(v7),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=columns)
    return out.sort_values(["target_name", "model_group_refined", "metric"]).reset_index(drop=True)


def _build_summary_markdown(
    *,
    out_path: Path,
    label_path: Path,
    labels_source_type: str,
    target_registry: Mapping[str, Any],
    cv_registry_refined: Mapping[str, Any],
    screening_registry: Mapping[str, Any],
    per_target_df: pd.DataFrame,
    step7_vs_step75_df: pd.DataFrame,
    explainability_report: Mapping[str, Any],
    skipped_targets: Sequence[Mapping[str, Any]],
    late_fusion_notes: Sequence[Mapping[str, Any]],
) -> None:
    confirm_targets = list(target_registry.get("confirmatory_regression_targets", []))
    exploratory_reg = list(target_registry.get("exploratory_regression_targets", []))
    exploratory_cls = list(target_registry.get("exploratory_classification_targets", []))

    lines: List[str] = []
    lines.append("# Step 7.5 Refined Fusion Evaluation Summary")
    lines.append("")
    lines.append("## 1) Label Source")
    lines.append(f"- labels_csv: `{label_path.as_posix()}`")
    lines.append(f"- labels_source_type: `{labels_source_type}`")
    lines.append("- adjudicated_labels_preferred: true")
    lines.append("")
    lines.append("## 2) Confirmatory vs Exploratory Targets")
    lines.append(f"- confirmatory_regression: {', '.join(confirm_targets) if confirm_targets else 'none'}")
    lines.append(f"- exploratory_regression: {', '.join(exploratory_reg) if exploratory_reg else 'none'}")
    lines.append(f"- exploratory_classification: {', '.join(exploratory_cls) if exploratory_cls else 'none'}")
    lines.append("- note: confirmatory targets are primary evidence; exploratory targets are sensitivity analysis.")
    lines.append("")
    lines.append("## 3) CV Policy")
    target_items = cv_registry_refined.get("targets", {})
    reused = sum(1 for _, v in target_items.items() if isinstance(v, dict) and v.get("source") == "step7_reused")
    total = len(target_items) if isinstance(target_items, dict) else 0
    lines.append(f"- step7_split_reuse: {reused}/{total} targets reused Step-7 outer splits")
    lines.append("- fallback_when_needed: regenerate RepeatedKFold / RepeatedStratifiedKFold with fixed seed")
    lines.append("")
    lines.append("## 4) Feature Screening (Leakage-safe)")
    cfg = screening_registry.get("config", {})
    lines.append(f"- missing_ratio_threshold: {cfg.get('missing_ratio_threshold')}")
    lines.append(f"- variance_threshold: {cfg.get('variance_threshold')}")
    lines.append(f"- correlation_threshold: {cfg.get('correlation_threshold')}")
    lines.append(
        "- topk: visual={} audio={} early={} (min_each_modality_in_early={})".format(
            cfg.get("top_k_visual"),
            cfg.get("top_k_audio"),
            cfg.get("top_k_early"),
            cfg.get("min_modality_each_early"),
        )
    )
    lines.append("- screening_scope: performed inside each outer training fold only")
    lines.append("")
    lines.append("## 5) Model Groups")
    lines.append("- visual_only_screened")
    lines.append("- audio_only_screened")
    lines.append("- early_fusion_screened (primary fusion evidence)")
    lines.append("- late_fusion_legacy (exploratory benchmark only)")
    for n in list(late_fusion_notes)[:5]:
        lines.append(f"- late_fusion_note: target={n.get('target_name')} reason={n.get('reason')}")
    lines.append("")
    lines.append("## 6) Confirmatory Results")
    if per_target_df.empty:
        lines.append("- no per-target metrics exported.")
    else:
        for target in confirm_targets:
            sub = per_target_df[
                (per_target_df["target_name"] == target)
                & (per_target_df["target_type"] == "regression")
            ].copy()
            if sub.empty or "mae_mean" not in sub.columns:
                lines.append(f"- {target}: unavailable")
                continue
            sub = sub.sort_values("mae_mean", ascending=True)
            best = sub.iloc[0]
            lines.append(f"- {target}: best={best['model_group']} MAE={float(best['mae_mean']):.4f}")
            ev = sub[sub["model_group"] == "early_fusion_screened"]
            vv = sub[sub["model_group"] == "visual_only_screened"]
            if not ev.empty and not vv.empty:
                delta = float(ev.iloc[0]["mae_mean"]) - float(vv.iloc[0]["mae_mean"])
                lines.append(f"  early_vs_visual_MAE_delta={delta:.4f} (negative means early fusion better)")
    lines.append("")
    lines.append("## 7) Step 7 vs Step 7.5")
    if step7_vs_step75_df.empty:
        lines.append("- no comparable Step-7 rows found.")
    else:
        for target in confirm_targets:
            sub = step7_vs_step75_df[
                (step7_vs_step75_df["target_name"] == target)
                & (step7_vs_step75_df["model_group_refined"] == "early_fusion_screened")
                & (step7_vs_step75_df["metric"] == "mae_mean")
            ]
            if not sub.empty:
                lines.append(
                    f"- {target} early_fusion_screened MAE delta (Step7.5-Step7) = {float(sub['delta_step75_minus_step7'].mean()):.4f}"
                )
    lines.append("")
    lines.append("## 8) Explainability Status")
    perm = explainability_report.get("permutation", {})
    shap = explainability_report.get("shap", {})
    lines.append(
        f"- permutation_importance_available={perm.get('available')} rows={perm.get('rows')}"
    )
    lines.append(
        f"- shap_available_targets={shap.get('available_count')} attempted={shap.get('attempted_count')}"
    )
    lines.append("")
    lines.append("## 9) Skipped Targets / Tasks")
    if not skipped_targets:
        lines.append("- none")
    else:
        for item in skipped_targets:
            lines.append(
                f"- target={item.get('target_name')} type={item.get('target_type')} reason={item.get('reason')}"
            )
    lines.append("")
    lines.append("## 10) Caveats")
    lines.append("- small sample size; high variance remains possible.")
    lines.append("- labels are adjudicated human ratings; quality improved but still subjective.")
    lines.append("- late fusion is exploratory only in Step 7.5 conclusions.")
    lines.append("- exploratory targets should be interpreted cautiously.")
    lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def run_step75_refined_eval(
    video_dir: str,
    *,
    feature_csv: Optional[str] = None,
    labels_csv: Optional[str] = None,
    step75_outdir: Optional[str] = None,
    seed: Optional[int] = None,
    smoke_test: bool = False,
    reuse_step7_splits: Optional[bool] = None,
    progress_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Run additive Step-7.5 refined evaluation.

    - writes only to `fusion_eval_refined/` (or custom outdir)
    - keeps original Step-7 outputs untouched
    """
    run_seed = int(seed if seed is not None else STEP75_SEED)
    use_step7_splits = STEP75_REUSE_STEP7_SPLITS if reuse_step7_splits is None else bool(reuse_step7_splits)

    paths = _resolve_paths(
        video_dir=video_dir,
        feature_csv=feature_csv,
        labels_csv=labels_csv,
        out_dir=step75_outdir,
    )
    out_dir = paths["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    feature_path = paths["feature_csv"]
    feature_dict_path = paths["feature_dict_json"]
    labels_path = paths["labels_csv"]
    if not feature_path.is_file():
        raise FileNotFoundError(f"step7.5 missing feature table: {feature_path.as_posix()}")
    if not labels_path.is_file():
        raise FileNotFoundError(f"step7.5 missing labels table: {labels_path.as_posix()}")

    logger.info("step7.5 start | video=%s", video_dir)
    _emit_progress(progress_callback, 2, 100, "读取输入与标签")
    model_df = _normalize_segment_id(pd.read_csv(feature_path), "model_feature_table")
    labels_df = _normalize_segment_id(pd.read_csv(labels_path), "labels_table")
    label_schema_summary = _inspect_validation_labels_schema(labels_df)
    _emit_progress(progress_callback, 8, 100, "标签表 schema 检查")
    if not bool(label_schema_summary.get("compatible_for_step75", False)):
        raise ValueError(
            "step7.5 labels schema incompatible: missing segment_id or no usable target columns with >=5 non-missing samples"
        )

    model_seg = set(model_df["segment_id"].tolist())
    label_seg = set(labels_df["segment_id"].tolist())
    missing_in_labels = sorted(int(x) for x in (model_seg - label_seg))
    missing_in_features = sorted(int(x) for x in (label_seg - model_seg))
    if missing_in_labels:
        logger.warning("step7.5 segments missing labels: %d", len(missing_in_labels))
    if missing_in_features:
        logger.warning("step7.5 segments missing model features: %d", len(missing_in_features))

    merged = model_df.merge(labels_df, on="segment_id", how="inner", suffixes=("", "_label"))
    if merged.empty:
        raise RuntimeError("step7.5 merged dataset is empty")

    dataset_path = out_dir / "step75_modeling_dataset.csv"
    merged.to_csv(dataset_path, index=False, encoding="utf-8")

    model_feature_dict = _safe_read_json(feature_dict_path)
    base_group_registry = _build_base_feature_groups(model_df=model_df, model_feature_dict=model_feature_dict)
    base_groups = base_group_registry["groups"]
    feature_group_registry_path = out_dir / "feature_group_registry_refined.json"
    _safe_write_json(feature_group_registry_path, base_group_registry)

    target_registry, target_series_map = build_target_registry(
        labels_df=merged,
        min_class_count=int(STEP75_CLASS_MIN_COUNT),
    )
    target_registry["source_note"] = "same policy as Step 7 (confirmatory vs exploratory)"
    target_registry["inputs"] = {
        "feature_csv": feature_path.as_posix(),
        "feature_dict_json": feature_dict_path.as_posix(),
        "labels_csv": labels_path.as_posix(),
        "adjudication_report_json": paths["adjudication_report_json"].as_posix(),
        "step7_target_registry_json": paths["step7_target_registry_json"].as_posix(),
        "step7_cv_registry_json": paths["step7_cv_registry_json"].as_posix(),
    }
    target_registry["join_summary"] = {
        "model_rows": int(len(model_df)),
        "label_rows": int(len(labels_df)),
        "merged_rows": int(len(merged)),
        "missing_in_labels_count": int(len(missing_in_labels)),
        "missing_in_features_count": int(len(missing_in_features)),
        "missing_in_labels_segment_ids": missing_in_labels,
        "missing_in_features_segment_ids": missing_in_features,
    }
    target_registry["labels_schema_summary"] = label_schema_summary
    target_registry_path = out_dir / "target_registry_refined.json"
    _safe_write_json(target_registry_path, target_registry)

    step7_cv_payload = _safe_read_json(paths["step7_cv_registry_json"])
    backend_info = resolve_model_backend()
    backend = backend_info.backend

    repeats_eff = int(min(2, STEP75_REG_CV_REPEATS)) if bool(smoke_test) else int(STEP75_REG_CV_REPEATS)
    screening_cfg = FeatureScreeningConfig(
        missing_ratio_threshold=float(STEP75_SCREEN_MISSING_THRESHOLD),
        variance_threshold=float(STEP75_SCREEN_VARIANCE_THRESHOLD),
        correlation_threshold=float(STEP75_SCREEN_CORR_THRESHOLD),
        top_k_visual=int(STEP75_SCREEN_TOPK_VISUAL),
        top_k_audio=int(STEP75_SCREEN_TOPK_AUDIO),
        top_k_early=int(STEP75_SCREEN_TOPK_EARLY),
        min_modality_each_early=int(STEP75_SCREEN_MIN_MODALITY_EARLY),
        random_seed=int(run_seed),
    )

    cv_registry_refined: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(run_seed),
        "reuse_step7_splits_requested": bool(use_step7_splits),
        "targets": {},
    }
    screening_registry: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "missing_ratio_threshold": float(screening_cfg.missing_ratio_threshold),
            "variance_threshold": float(screening_cfg.variance_threshold),
            "correlation_threshold": float(screening_cfg.correlation_threshold),
            "top_k_visual": int(screening_cfg.top_k_visual),
            "top_k_audio": int(screening_cfg.top_k_audio),
            "top_k_early": int(screening_cfg.top_k_early),
            "min_modality_each_early": int(screening_cfg.min_modality_each_early),
        },
        "targets": {},
    }

    fold_metric_rows: List[Dict[str, Any]] = []
    oof_rows: List[Dict[str, Any]] = []
    permutation_records: List[Dict[str, Any]] = []
    shap_records: List[pd.DataFrame] = []
    shap_status_records: List[Dict[str, Any]] = []
    late_fusion_notes: List[Dict[str, Any]] = []
    skipped_targets: List[Dict[str, Any]] = []

    confirmatory_targets = [
        str(t["target_name"])
        for t in target_registry.get("targets", [])
        if t.get("enabled", False)
        and t.get("target_type") == "regression"
        and t.get("tier") == "confirmatory"
    ]
    enabled_targets = [t for t in target_registry.get("targets", []) if bool(t.get("enabled", False))]
    enabled_target_count = max(1, int(len(enabled_targets)))
    enabled_target_seen = 0

    for t_item in target_registry.get("targets", []):
        if not bool(t_item.get("enabled", False)):
            skipped_targets.append(
                {
                    "target_name": t_item.get("target_name"),
                    "target_type": t_item.get("target_type"),
                    "reason": t_item.get("reason", "disabled_by_registry"),
                }
            )
            continue

        target_name = str(t_item["target_name"])
        task_type = str(t_item["target_type"])
        target_tier = str(t_item.get("tier", "exploratory"))
        enabled_target_seen += 1
        _emit_progress(
            progress_callback,
            12 + ((enabled_target_seen - 1) / enabled_target_count) * 58.0,
            100,
            f"feature screening | target {enabled_target_seen}/{enabled_target_count} | {target_name}",
        )

        y_source = target_series_map.get(
            target_name,
            merged[target_name] if target_name in merged.columns else pd.Series(dtype=float),
        )
        if task_type == "classification":
            y_series = pd.Series(y_source, index=merged.index).astype("string").str.strip()
            y_series = y_series.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
            valid_mask = y_series.notna()
        else:
            y_series = pd.to_numeric(pd.Series(y_source, index=merged.index), errors="coerce")
            valid_mask = y_series.notna()

        if int(valid_mask.sum()) < 5:
            skipped_targets.append(
                {
                    "target_name": target_name,
                    "target_type": task_type,
                    "reason": "insufficient_non_missing_samples",
                }
            )
            continue

        local_df = merged.loc[valid_mask].reset_index(drop=True)
        seg_ids = local_df["segment_id"].astype(int).to_numpy()
        if task_type == "classification":
            labels = y_series.loc[valid_mask].astype(str).to_numpy()
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(labels).astype(int)
            class_ids = [int(x) for x in np.arange(len(label_encoder.classes_))]
            class_labels = [str(x) for x in label_encoder.classes_.tolist()]
        else:
            y = pd.to_numeric(y_series.loc[valid_mask], errors="coerce").to_numpy(dtype=float)
            label_encoder = None
            class_ids = []
            class_labels = []

        folds: List[Dict[str, Any]] = []
        split_source = "regenerated"
        split_reason = ""
        if bool(use_step7_splits):
            folds, split_reason = _reuse_step7_splits_for_target(
                step7_cv_payload=step7_cv_payload,
                target_name=target_name,
                target_type=task_type,
                seg_ids=seg_ids,
                smoke_test=bool(smoke_test),
            )
            if folds:
                split_source = "step7_reused"

        if not folds:
            local_seed = _stable_text_seed(target_name, base=run_seed + (991 if task_type == "classification" else 997))
            if task_type == "classification":
                folds, cls_reason = _build_cls_splits(
                    seg_ids=seg_ids,
                    y_codes=y,
                    n_splits=int(STEP75_REG_CV_SPLITS),
                    n_repeats=int(repeats_eff),
                    seed=int(local_seed),
                )
                if not folds:
                    skipped_targets.append(
                        {
                            "target_name": target_name,
                            "target_type": task_type,
                            "reason": f"unable_to_build_classification_splits:{cls_reason}",
                        }
                    )
                    continue
                split_reason = split_reason or cls_reason
            else:
                folds = _build_reg_splits(
                    seg_ids=seg_ids,
                    n_splits=int(STEP75_REG_CV_SPLITS),
                    n_repeats=int(repeats_eff),
                    seed=int(local_seed),
                )
                if not folds:
                    skipped_targets.append(
                        {
                            "target_name": target_name,
                            "target_type": task_type,
                            "reason": "unable_to_build_regression_splits",
                        }
                    )
                    continue

        n_splits_eff = int(max(1, len(set(int(f["split_index"]) for f in folds))))
        n_repeats_eff = int(max(1, len(set(int(f["repeat_index"]) for f in folds))))
        cv_registry_refined["targets"][target_name] = {
            "target_type": task_type,
            "target_tier": target_tier,
            "source": split_source,
            "source_reason": split_reason,
            "n_samples": int(len(seg_ids)),
            "n_splits": int(n_splits_eff),
            "n_repeats": int(n_repeats_eff),
            "folds": [
                {
                    "fold_id": str(f["fold_id"]),
                    "repeat_index": int(f["repeat_index"]),
                    "split_index": int(f["split_index"]),
                    "train_segment_ids": [int(seg_ids[i]) for i in f["train_idx"].tolist()],
                    "test_segment_ids": [int(seg_ids[i]) for i in f["test_idx"].tolist()],
                }
                for f in folds
            ],
        }
        screening_registry["targets"][target_name] = {
            "target_type": task_type,
            "target_tier": target_tier,
            "folds": [],
        }

        x_vis_legacy_full = ensure_numeric_frame(local_df, base_groups.get("visual_only", [])).fillna(0.0)
        x_aud_legacy_full = ensure_numeric_frame(local_df, base_groups.get("audio_only", [])).fillna(0.0)
        late_fusion_available = x_vis_legacy_full.shape[1] > 0 and x_aud_legacy_full.shape[1] > 0
        if not late_fusion_available:
            late_fusion_notes.append(
                {
                    "target_name": target_name,
                    "target_type": task_type,
                    "reason": "late_fusion_legacy_skipped_missing_visual_or_audio_features",
                }
            )

        shap_attempted_for_target = False
        for fold_pos, fold in enumerate(folds, start=1):
            fold_id = str(fold["fold_id"])
            rep = int(fold["repeat_index"])
            split = int(fold["split_index"])
            train_idx = np.asarray(fold["train_idx"], dtype=int)
            test_idx = np.asarray(fold["test_idx"], dtype=int)
            y_train = y[train_idx]
            y_test = y[test_idx]
            seg_test = seg_ids[test_idx]
            _emit_progress(
                progress_callback,
                18 + ((enabled_target_seen - 1) / enabled_target_count) * 52.0 + (fold_pos / max(1, len(folds))) * (52.0 / enabled_target_count),
                100,
                f"CV / OOF 训练评估 | target {enabled_target_seen}/{enabled_target_count} | fold {fold_pos}/{len(folds)} | {target_name}",
            )

            screening_seed = _stable_text_seed(f"{target_name}:{fold_id}", base=run_seed + 131)
            selected_map, fold_screen_info = screen_features_for_fold(
                x_train=local_df.iloc[train_idx].reset_index(drop=True),
                y_train=y_train,
                task_type=task_type,
                base_feature_groups=base_groups,
                config=screening_cfg,
                ranking_seed=int(screening_seed),
            )
            screening_registry["targets"][target_name]["folds"].append(
                {
                    "fold_id": fold_id,
                    "repeat_index": int(rep),
                    "split_index": int(split),
                    **fold_screen_info,
                }
            )

            x_group_frames: Dict[str, pd.DataFrame] = {
                "visual_only_screened": ensure_numeric_frame(local_df, selected_map.get("visual_only_screened", [])).fillna(0.0),
                "audio_only_screened": ensure_numeric_frame(local_df, selected_map.get("audio_only_screened", [])).fillna(0.0),
                "early_fusion_screened": ensure_numeric_frame(local_df, selected_map.get("early_fusion_screened", [])).fillna(0.0),
            }
            if late_fusion_available:
                x_group_frames["late_fusion_legacy"] = pd.DataFrame(index=local_df.index)

            eligible_groups: List[str] = []
            for g in MODEL_GROUPS_REFINED:
                if g == "late_fusion_legacy":
                    if late_fusion_available:
                        eligible_groups.append(g)
                    continue
                xg = x_group_frames.get(g)
                if xg is not None and xg.shape[1] > 0:
                    eligible_groups.append(g)

            if not eligible_groups:
                late_fusion_notes.append(
                    {
                        "target_name": target_name,
                        "target_type": task_type,
                        "reason": f"no_eligible_groups_after_screening fold={fold_id}",
                    }
                )
                continue

            for group in eligible_groups:
                model_seed = _stable_text_seed(f"{target_name}:{group}:{fold_id}", base=run_seed + 719)
                if group == "late_fusion_legacy":
                    if task_type == "regression":
                        y_pred = _late_fusion_regression_predict(
                            x_vis=x_vis_legacy_full,
                            x_audio=x_aud_legacy_full,
                            y=y,
                            train_idx=train_idx,
                            test_idx=test_idx,
                            backend=backend,
                            seed=int(model_seed),
                        )
                        met = regression_metrics(y_true=y_test, y_pred=y_pred)
                        fold_metric_rows.append(
                            {
                                "target_name": target_name,
                                "target_type": "regression",
                                "target_tier": target_tier,
                                "model_group": group,
                                "fold_id": fold_id,
                                "repeat_index": int(rep),
                                "split_index": int(split),
                                **met,
                            }
                        )
                        oof_rows.extend(
                            _build_oof_rows_regression(
                                segment_ids=seg_test,
                                target_name=target_name,
                                target_tier=target_tier,
                                model_group=group,
                                fold_id=fold_id,
                                y_true=y_test,
                                y_pred=y_pred,
                            )
                        )
                    else:
                        y_proba = _late_fusion_classification_predict(
                            x_vis=x_vis_legacy_full,
                            x_audio=x_aud_legacy_full,
                            y_codes=y,
                            train_idx=train_idx,
                            test_idx=test_idx,
                            backend=backend,
                            seed=int(model_seed),
                            class_ids=class_ids,
                        )
                        y_pred_codes = np.argmax(y_proba, axis=1).astype(int)
                        met = classification_metrics(y_true=y_test, y_pred=y_pred_codes, y_proba=y_proba)
                        fold_metric_rows.append(
                            {
                                "target_name": target_name,
                                "target_type": "classification",
                                "target_tier": target_tier,
                                "model_group": group,
                                "fold_id": fold_id,
                                "repeat_index": int(rep),
                                "split_index": int(split),
                                **met,
                            }
                        )
                        y_true_labels = label_encoder.inverse_transform(y_test).tolist() if label_encoder is not None else [str(v) for v in y_test.tolist()]
                        y_pred_labels = label_encoder.inverse_transform(y_pred_codes).tolist() if label_encoder is not None else [str(v) for v in y_pred_codes.tolist()]
                        oof_rows.extend(
                            _build_oof_rows_classification(
                                segment_ids=seg_test,
                                target_name=target_name,
                                target_tier=target_tier,
                                model_group=group,
                                fold_id=fold_id,
                                y_true_labels=y_true_labels,
                                y_pred_labels=y_pred_labels,
                                proba=y_proba,
                                class_labels=class_labels,
                            )
                        )
                    continue

                x_full = x_group_frames[group]
                x_train_fold = x_full.iloc[train_idx].reset_index(drop=True)
                x_test_fold = x_full.iloc[test_idx].reset_index(drop=True)
                if x_train_fold.shape[1] == 0:
                    continue

                if task_type == "classification":
                    model = build_estimator(
                        task_type="classification",
                        seed=int(model_seed),
                        backend=backend,
                        n_classes=len(class_ids),
                    )
                    model.fit(x_train_fold, y_train)
                    y_proba = aligned_predict_proba(model, x_test_fold, class_ids)
                    y_pred_codes = np.argmax(y_proba, axis=1).astype(int)
                    met = classification_metrics(y_true=y_test, y_pred=y_pred_codes, y_proba=y_proba)
                    fold_metric_rows.append(
                        {
                            "target_name": target_name,
                            "target_type": "classification",
                            "target_tier": target_tier,
                            "model_group": group,
                            "fold_id": fold_id,
                            "repeat_index": int(rep),
                            "split_index": int(split),
                            **met,
                        }
                    )
                    y_true_labels = label_encoder.inverse_transform(y_test).tolist() if label_encoder is not None else [str(v) for v in y_test.tolist()]
                    y_pred_labels = label_encoder.inverse_transform(y_pred_codes).tolist() if label_encoder is not None else [str(v) for v in y_pred_codes.tolist()]
                    oof_rows.extend(
                        _build_oof_rows_classification(
                            segment_ids=seg_test,
                            target_name=target_name,
                            target_tier=target_tier,
                            model_group=group,
                            fold_id=fold_id,
                            y_true_labels=y_true_labels,
                            y_pred_labels=y_pred_labels,
                            proba=y_proba,
                            class_labels=class_labels,
                        )
                    )
                    perm = permutation_importance_holdout(
                        model=model,
                        x_test=x_test_fold,
                        y_test=y_test,
                        task_type="classification",
                        random_seed=int(model_seed + 41),
                        n_repeats=3 if smoke_test else 5,
                    )
                else:
                    model = build_estimator(task_type="regression", seed=int(model_seed), backend=backend)
                    model.fit(x_train_fold, y_train)
                    y_pred = np.asarray(model.predict(x_test_fold), dtype=float)
                    met = regression_metrics(y_true=y_test, y_pred=y_pred)
                    fold_metric_rows.append(
                        {
                            "target_name": target_name,
                            "target_type": "regression",
                            "target_tier": target_tier,
                            "model_group": group,
                            "fold_id": fold_id,
                            "repeat_index": int(rep),
                            "split_index": int(split),
                            **met,
                        }
                    )
                    oof_rows.extend(
                        _build_oof_rows_regression(
                            segment_ids=seg_test,
                            target_name=target_name,
                            target_tier=target_tier,
                            model_group=group,
                            fold_id=fold_id,
                            y_true=y_test,
                            y_pred=y_pred,
                        )
                    )
                    perm = permutation_importance_holdout(
                        model=model,
                        x_test=x_test_fold,
                        y_test=y_test,
                        task_type="regression",
                        random_seed=int(model_seed + 43),
                        n_repeats=3 if smoke_test else 5,
                    )
                    if (
                        target_tier == "confirmatory"
                        and group == "early_fusion_screened"
                        and not shap_attempted_for_target
                    ):
                        shap_df, shap_status = try_shap_tree_summary(
                            model=model,
                            x_df=x_test_fold,
                            target_name=target_name,
                            model_group=group,
                            out_plots_dir=plots_dir,
                        )
                        shap_status_records.append(
                            {
                                **shap_status,
                                "target_type": task_type,
                                "target_tier": target_tier,
                                "fold_id": fold_id,
                            }
                        )
                        if not shap_df.empty:
                            sdf = shap_df.copy()
                            sdf["target_type"] = task_type
                            sdf["target_tier"] = target_tier
                            sdf["fold_id"] = fold_id
                            shap_records.append(sdf)
                        shap_attempted_for_target = True

                if perm.get("ok", False):
                    for prow in perm.get("rows", []):
                        permutation_records.append(
                            {
                                "target_name": target_name,
                                "target_type": task_type,
                                "target_tier": target_tier,
                                "model_group": group,
                                "fold_id": fold_id,
                                "feature": prow.get("feature", ""),
                                "importance": float(prow.get("importance_mean", 0.0)),
                                "importance_std_fold": float(prow.get("importance_std", 0.0)),
                                "metric_name": perm.get("metric_name", ""),
                                "baseline_metric": float(perm.get("baseline_metric", np.nan)),
                            }
                        )

        logger.info("step7.5 target done | target=%s type=%s folds=%d", target_name, task_type, len(folds))

    if not fold_metric_rows:
        raise RuntimeError("step7.5 produced no fold metrics; check inputs/targets")

    fold_metrics_df = pd.DataFrame(fold_metric_rows)
    metric_cols = [m for m in ALL_METRICS if m in fold_metrics_df.columns]
    agg_keys = ["target_name", "target_type", "target_tier", "model_group"]
    mean_df = fold_metrics_df.groupby(agg_keys, dropna=False)[metric_cols].mean().add_suffix("_mean")
    std_df = fold_metrics_df.groupby(agg_keys, dropna=False)[metric_cols].std().add_suffix("_std")
    cnt_df = fold_metrics_df.groupby(agg_keys, dropna=False).size().to_frame("fold_count")
    per_target_df = (
        mean_df.join(std_df, how="outer")
        .join(cnt_df, how="outer")
        .reset_index()
        .sort_values(["target_name", "model_group"])
        .reset_index(drop=True)
    )

    cmp_rows: List[Dict[str, Any]] = []
    for target_name in sorted(per_target_df["target_name"].dropna().unique().tolist()):
        sub = per_target_df[per_target_df["target_name"] == target_name].copy()
        if sub.empty:
            continue
        task_type = str(sub["target_type"].iloc[0])
        if task_type == "regression":
            metric_col = "mae_mean"
            direction = "lower_better"
            sub = sub.sort_values(metric_col, ascending=True)
        else:
            metric_col = "macro_f1_mean"
            direction = "higher_better"
            sub = sub.sort_values(metric_col, ascending=False)
        rank = 1
        for _, row in sub.iterrows():
            cmp_rows.append(
                {
                    "target_name": target_name,
                    "target_type": task_type,
                    "target_tier": row.get("target_tier", ""),
                    "model_group": row["model_group"],
                    "primary_metric": metric_col,
                    "primary_value": float(row.get(metric_col, np.nan)),
                    "rank": int(rank),
                    "direction": direction,
                }
            )
            rank += 1
    comparison_df = pd.DataFrame(cmp_rows)

    paired_df, bootstrap_source = _paired_delta_records(
        fold_metrics_df=fold_metrics_df,
        metric_cols=metric_cols,
    )
    bootstrap_payload: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(run_seed),
        "bootstrap_samples": int(STEP75_BOOTSTRAP_SAMPLES),
        "ci_alpha": float(STEP75_BOOTSTRAP_CI_ALPHA),
        "results": [],
    }
    bootstrap_items = sorted(bootstrap_source.items())
    for i, (key, vals) in enumerate(bootstrap_items):
        target_name, metric, comp = key.split("::", 2)
        _emit_progress(
            progress_callback,
            74 + ((i + 1) / max(1, len(bootstrap_items))) * 10.0,
            100,
            f"bootstrap CI | {i + 1}/{max(1, len(bootstrap_items))} | {target_name} | {metric}",
        )
        ci = _bootstrap_ci(
            values=vals,
            n_bootstrap=int(STEP75_BOOTSTRAP_SAMPLES if not smoke_test else min(300, STEP75_BOOTSTRAP_SAMPLES)),
            alpha=float(STEP75_BOOTSTRAP_CI_ALPHA),
            seed=int(run_seed + 5000 + i),
        )
        bootstrap_payload["results"].append(
            {
                "target_name": target_name,
                "metric": metric,
                "comparison": comp,
                **ci,
            }
        )

    oof_df = pd.DataFrame(oof_rows)
    if not oof_df.empty:
        fixed_cols = [
            "segment_id",
            "target_name",
            "target_type",
            "target_tier",
            "model_group",
            "fold_id",
            "y_true",
            "y_pred",
        ]
        prob_cols = sorted([c for c in oof_df.columns if c.startswith("proba__")])
        oof_df = oof_df[[c for c in fixed_cols if c in oof_df.columns] + prob_cols]

    permutation_df = aggregate_permutation_records(permutation_records)
    _emit_progress(progress_callback, 86, 100, "permutation importance")
    if permutation_df.empty:
        permutation_df = pd.DataFrame(
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

    if shap_records:
        shap_all = pd.concat(shap_records, axis=0, ignore_index=True)
        shap_summary = (
            shap_all.groupby(
                ["target_name", "target_type", "target_tier", "model_group", "feature"],
                dropna=False,
            )["mean_abs_shap"]
            .mean()
            .reset_index()
            .sort_values(["target_name", "mean_abs_shap"], ascending=[True, False])
            .reset_index(drop=True)
        )
        shap_summary["rank"] = shap_summary.groupby(["target_name", "model_group"]).cumcount() + 1
    else:
        shap_summary = pd.DataFrame(
            columns=[
                "target_name",
                "target_type",
                "target_tier",
                "model_group",
                "feature",
                "mean_abs_shap",
                "rank",
            ]
        )

    _plot_confirmatory_comparison(
        per_target_df=per_target_df,
        confirm_targets=confirmatory_targets,
        out_path=plots_dir / "confirmatory_model_comparison_refined.png",
    )
    permutation_plot_paths: List[str] = []
    for target in confirmatory_targets:
        preferred_group = "early_fusion_screened"
        if permutation_df[
            (permutation_df["target_name"] == target)
            & (permutation_df["model_group"] == preferred_group)
        ].empty:
            sub = permutation_df[permutation_df["target_name"] == target]
            if sub.empty:
                continue
            preferred_group = str(sub["model_group"].iloc[0])
        p_out = plots_dir / f"target_{target}_permutation_importance.png"
        ok = plot_permutation_importance(
            permutation_df=permutation_df,
            target_name=target,
            model_group=preferred_group,
            out_path=p_out,
            top_n=20,
        )
        if ok:
            permutation_plot_paths.append(p_out.as_posix())

    explainability_report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "permutation": {
            "available": bool(not permutation_df.empty),
            "rows": int(len(permutation_df)),
            "file": (out_dir / "permutation_importance.csv").as_posix(),
            "plots": permutation_plot_paths,
        },
        "shap": {
            "attempted_count": int(len(shap_status_records)),
            "available_count": int(sum(1 for x in shap_status_records if bool(x.get("available", False)))),
            "failed_count": int(sum(1 for x in shap_status_records if not bool(x.get("available", False)))),
            "records": shap_status_records,
            "file": (out_dir / "shap_summary_refined.csv").as_posix(),
        },
    }

    step7_vs_step75_df = _build_step7_step75_comparison(
        step7_per_target_path=paths["step7_per_target_metrics_csv"],
        per_target_refined_df=per_target_df,
    )

    per_target_path = out_dir / "per_target_metrics_refined.csv"
    comparison_path = out_dir / "model_comparison_refined.csv"
    paired_path = out_dir / "paired_deltas_refined.csv"
    bootstrap_path = out_dir / "bootstrap_ci_refined.json"
    oof_path = out_dir / "oof_predictions_refined.csv"
    cv_path = out_dir / "cv_split_registry_refined.json"
    screening_path = out_dir / "feature_screening_registry.json"
    permutation_path = out_dir / "permutation_importance.csv"
    shap_path = out_dir / "shap_summary_refined.csv"
    explainability_path = out_dir / "explainability_report.json"
    step7_vs_step75_path = out_dir / "step7_vs_step75_comparison.csv"
    summary_path = out_dir / "step75_summary.md"

    _emit_progress(progress_callback, 92, 100, "写出 fusion_eval_refined 各文件")
    per_target_df.to_csv(per_target_path, index=False, encoding="utf-8")
    comparison_df.to_csv(comparison_path, index=False, encoding="utf-8")
    if paired_df.empty:
        pd.DataFrame(
            columns=[
                "target_name",
                "target_type",
                "metric",
                "comparison",
                "delta_definition",
                "delta_mean",
                "delta_std",
                "n_folds",
            ]
        ).to_csv(paired_path, index=False, encoding="utf-8")
    else:
        paired_df.to_csv(paired_path, index=False, encoding="utf-8")
    _safe_write_json(bootstrap_path, bootstrap_payload)
    oof_df.to_csv(oof_path, index=False, encoding="utf-8")
    _safe_write_json(cv_path, cv_registry_refined)
    _safe_write_json(screening_path, screening_registry)
    permutation_df.to_csv(permutation_path, index=False, encoding="utf-8")
    shap_summary.to_csv(shap_path, index=False, encoding="utf-8")
    _safe_write_json(explainability_path, explainability_report)
    step7_vs_step75_df.to_csv(step7_vs_step75_path, index=False, encoding="utf-8")

    _emit_progress(progress_callback, 97, 100, "生成 step75_summary.md")
    _build_summary_markdown(
        out_path=summary_path,
        label_path=labels_path,
        labels_source_type=str(paths["labels_source_type"]),
        target_registry=target_registry,
        cv_registry_refined=cv_registry_refined,
        screening_registry=screening_registry,
        per_target_df=per_target_df,
        step7_vs_step75_df=step7_vs_step75_df,
        explainability_report=explainability_report,
        skipped_targets=skipped_targets,
        late_fusion_notes=late_fusion_notes,
    )
    _emit_progress(progress_callback, 100, 100, "Step 7.5 完成")

    logger.info("step7.5 done | out=%s", out_dir.as_posix())
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "video_dir": Path(video_dir).as_posix(),
        "labels_source_csv": labels_path.as_posix(),
        "labels_source_type": str(paths["labels_source_type"]),
        "labels_schema_summary": label_schema_summary,
        "feature_csv": feature_path.as_posix(),
        "feature_dict_json": feature_dict_path.as_posix(),
        "adjudication_report_json": paths["adjudication_report_json"].as_posix(),
        "step75_outdir": out_dir.as_posix(),
        "backend_info": {
            "backend": backend_info.backend,
            "available": backend_info.available,
            "fallback_chain": backend_info.fallback_chain,
            "reason": backend_info.reason,
        },
        "reuse_step7_splits": bool(use_step7_splits),
        "smoke_test": bool(smoke_test),
        "step75_modeling_dataset_csv": dataset_path.as_posix(),
        "target_registry_refined_json": target_registry_path.as_posix(),
        "feature_group_registry_refined_json": feature_group_registry_path.as_posix(),
        "cv_split_registry_refined_json": cv_path.as_posix(),
        "feature_screening_registry_json": screening_path.as_posix(),
        "per_target_metrics_refined_csv": per_target_path.as_posix(),
        "model_comparison_refined_csv": comparison_path.as_posix(),
        "paired_deltas_refined_csv": paired_path.as_posix(),
        "bootstrap_ci_refined_json": bootstrap_path.as_posix(),
        "oof_predictions_refined_csv": oof_path.as_posix(),
        "permutation_importance_csv": permutation_path.as_posix(),
        "shap_summary_refined_csv": shap_path.as_posix(),
        "explainability_report_json": explainability_path.as_posix(),
        "step7_vs_step75_comparison_csv": step7_vs_step75_path.as_posix(),
        "step75_summary_md": summary_path.as_posix(),
        "plots_dir": plots_dir.as_posix(),
        "skipped_targets": skipped_targets,
        "late_fusion_notes": late_fusion_notes,
        "missing_in_labels_count": int(len(missing_in_labels)),
        "missing_in_features_count": int(len(missing_in_features)),
    }
