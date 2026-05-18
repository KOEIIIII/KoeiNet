


"""Core training/evaluation engine for Step-7 fusion modeling."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, RepeatedKFold, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from .modeling import (
    aligned_predict_proba,
    build_estimator,
    classification_metrics,
    ensure_numeric_frame,
    regression_metrics,
    resolve_model_backend,
)

logger = logging.getLogger("fusion.train_eval")

MODEL_GROUP_ORDER: Tuple[str, ...] = ("visual_only", "audio_only", "early_fusion", "late_fusion")
PAIRWISE_COMPARISONS: Tuple[Tuple[str, str], ...] = (
    ("early_fusion", "visual_only"),
    ("audio_only", "visual_only"),
    ("late_fusion", "visual_only"),
)
REG_METRICS: Tuple[str, ...] = ("mae", "rmse", "r2", "spearman")
CLS_METRICS: Tuple[str, ...] = ("accuracy", "macro_f1", "balanced_accuracy", "roc_auc")


def _build_group_frames(
    local_df: pd.DataFrame,
    feature_groups: Mapping[str, Sequence[str]],
) -> Dict[str, pd.DataFrame]:
    """Build per-group numeric feature frames for one target-local dataset."""
    x_group_frames: Dict[str, pd.DataFrame] = {}
    for group in MODEL_GROUP_ORDER:
        cols = [c for c in feature_groups.get(group, []) if c in local_df.columns]
        if not cols:
            continue
        x_group_frames[group] = ensure_numeric_frame(local_df, cols)
    return x_group_frames


def _eligible_model_groups(x_group_frames: Mapping[str, pd.DataFrame]) -> List[str]:
    """Return eligible model groups in fixed order for current target data."""
    out: List[str] = []
    for group in MODEL_GROUP_ORDER:
        if group == "late_fusion":
            if "visual_only" in x_group_frames and "audio_only" in x_group_frames:
                out.append(group)
            continue
        if group in x_group_frames:
            out.append(group)
    return out


def _split_meta(fold_idx: int, n_splits: int) -> Tuple[int, int]:
    rep = int(fold_idx // max(1, n_splits))
    split = int(fold_idx % max(1, n_splits))
    return rep, split


def _build_regression_splits(
    n_samples: int,
    n_splits: int,
    n_repeats: int,
    seed: int,
) -> Tuple[int, int, List[Tuple[np.ndarray, np.ndarray]]]:
    if n_samples < 2:
        return 0, 0, []
    n_splits_eff = int(min(max(2, int(n_splits)), n_samples))
    n_repeats_eff = int(max(1, int(n_repeats)))
    rkf = RepeatedKFold(n_splits=n_splits_eff, n_repeats=n_repeats_eff, random_state=int(seed))
    splits = [(tr.astype(int), te.astype(int)) for tr, te in rkf.split(np.arange(n_samples))]
    return n_splits_eff, n_repeats_eff, splits


def _build_classification_splits(
    y_codes: np.ndarray,
    n_splits: int,
    n_repeats: int,
    seed: int,
) -> Tuple[int, int, List[Tuple[np.ndarray, np.ndarray]], str]:
    if y_codes.size < 2:
        return 0, 0, [], "insufficient_samples"

    uniq, counts = np.unique(y_codes, return_counts=True)
    if uniq.size < 2:
        return 0, 0, [], "single_class_only"

    min_count = int(np.min(counts))
    n_splits_eff = int(min(max(2, int(n_splits)), min_count))
    if n_splits_eff < 2:
        return 0, 0, [], "class_support_too_sparse_for_stratified_kfold"

    n_repeats_eff = int(max(1, int(n_repeats)))
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits_eff,
        n_repeats=n_repeats_eff,
        random_state=int(seed),
    )
    splits = [(tr.astype(int), te.astype(int)) for tr, te in rskf.split(np.arange(y_codes.size), y_codes)]
    return n_splits_eff, n_repeats_eff, splits, ""


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
        for lhs, rhs in PAIRWISE_COMPARISONS:
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

                key = f"{target_name}::{metric}::{lhs}_vs_{rhs}"
                arr = vals.astype(float).tolist()
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


def _bootstrap_ci(
    values: Sequence[float],
    n_bootstrap: int,
    alpha: float,
    seed: int,
) -> Dict[str, Any]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return {
            "ok": False,
            "reason": "insufficient_values",
            "n": int(arr.size),
            "ci_lower": None,
            "ci_upper": None,
            "mean": float(np.mean(arr)) if arr.size > 0 else None,
        }

    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(n_bootstrap), dtype=float)
    for i in range(int(n_bootstrap)):
        sample = rng.choice(arr, size=arr.size, replace=True)
        draws[i] = float(np.mean(sample))

    lo_q = float((1.0 - alpha) / 2.0)
    hi_q = float(1.0 - lo_q)
    return {
        "ok": True,
        "reason": "",
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "ci_lower": float(np.quantile(draws, lo_q)),
        "ci_upper": float(np.quantile(draws, hi_q)),
    }


def run_train_eval(
    merged_df: pd.DataFrame,
    feature_groups: Mapping[str, Sequence[str]],
    target_registry: Mapping[str, Any],
    target_series_map: Mapping[str, pd.Series],
    out_dir: Path,
    seed: int,
    reg_cv_splits: int = 5,
    reg_cv_repeats: int = 20,
    bootstrap_samples: int = 2000,
    bootstrap_ci_alpha: float = 0.95,
    enable_bootstrap: bool = True,
    smoke_test: bool = False,
    progress_callback: Optional[Callable[[Mapping[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Run Step-7 cross-validated training/evaluation and write output artifacts."""
    out_dir.mkdir(parents=True, exist_ok=True)

    backend_info = resolve_model_backend()
    backend = backend_info.backend

    if smoke_test:
        reg_cv_repeats = min(int(reg_cv_repeats), 2)
        bootstrap_samples = min(int(bootstrap_samples), 300)

    fold_metric_rows: List[Dict[str, Any]] = []
    oof_rows: List[Dict[str, Any]] = []
    cv_registry: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed),
        "regression": {},
        "classification": {},
    }
    skipped_targets: List[Dict[str, Any]] = []
    warnings: List[str] = []

    if "segment_id" not in merged_df.columns:
        raise ValueError("merged dataset missing segment_id")

    base_segment_ids = pd.to_numeric(merged_df["segment_id"], errors="coerce")
    if base_segment_ids.isna().any():
        raise ValueError("segment_id contains non-numeric values in merged dataset")

    reg_targets = [
        t
        for t in target_registry.get("targets", [])
        if t.get("enabled", False) and str(t.get("target_type")) == "regression"
    ]
    cls_targets = [
        t
        for t in target_registry.get("targets", [])
        if t.get("enabled", False) and str(t.get("target_type")) == "classification"
    ]

    def _emit(event: Mapping[str, Any]) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(dict(event))
        except Exception as exc:
            logger.debug("step7 progress callback failed: %s", exc)


    _emit({"event": "stage", "stage": "generating CV splits"})
    total_units = 0
    for target_info in reg_targets:
        target_name = str(target_info["target_name"])
        y_ser = target_series_map.get(target_name)
        if y_ser is None:
            y_ser = pd.to_numeric(merged_df[target_name], errors="coerce")
        else:
            y_ser = pd.to_numeric(y_ser, errors="coerce")
        mask = y_ser.notna()
        if int(mask.sum()) < 5:
            continue
        idx = np.flatnonzero(mask.to_numpy())
        y = y_ser.iloc[idx].to_numpy(dtype=float)
        local_df = merged_df.iloc[idx].reset_index(drop=True)
        n_splits_eff, _, splits = _build_regression_splits(
            n_samples=y.size,
            n_splits=reg_cv_splits,
            n_repeats=reg_cv_repeats,
            seed=seed + hash(target_name) % 997,
        )
        if not splits:
            continue
        x_group_frames = _build_group_frames(local_df=local_df, feature_groups=feature_groups)
        groups = _eligible_model_groups(x_group_frames)
        total_units += int(len(splits) * len(groups))

    for target_info in cls_targets:
        target_name = str(target_info["target_name"])
        y_ser = target_series_map.get(target_name)
        if y_ser is None:
            y_ser = merged_df[target_name]
        y_ser = y_ser.astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
        mask = y_ser.notna()
        if int(mask.sum()) < 10:
            continue
        idx = np.flatnonzero(mask.to_numpy())
        y_labels = y_ser.iloc[idx].astype(str).to_numpy()
        local_df = merged_df.iloc[idx].reset_index(drop=True)
        y_codes = LabelEncoder().fit_transform(y_labels)
        _, _, splits, _ = _build_classification_splits(
            y_codes=y_codes,
            n_splits=reg_cv_splits,
            n_repeats=reg_cv_repeats,
            seed=seed + hash(target_name) % 991,
        )
        if not splits:
            continue
        x_group_frames = _build_group_frames(local_df=local_df, feature_groups=feature_groups)
        groups = _eligible_model_groups(x_group_frames)
        total_units += int(len(splits) * len(groups))

    completed_units = 0
    _emit(
        {
            "event": "stage",
            "stage": "training/evaluating models",
            "completed_units": int(completed_units),
            "total_units": int(total_units),
        }
    )

    for target_info in reg_targets:
        target_name = str(target_info["target_name"])
        tier = str(target_info.get("tier", "exploratory"))

        y_ser = target_series_map.get(target_name)
        if y_ser is None:
            y_ser = pd.to_numeric(merged_df[target_name], errors="coerce")
        else:
            y_ser = pd.to_numeric(y_ser, errors="coerce")

        mask = y_ser.notna()
        if int(mask.sum()) < 5:
            skipped_targets.append(
                {
                    "target_name": target_name,
                    "target_type": "regression",
                    "reason": "insufficient_non_missing_samples",
                }
            )
            continue

        idx = np.flatnonzero(mask.to_numpy())
        y = y_ser.iloc[idx].to_numpy(dtype=float)
        seg_ids = base_segment_ids.iloc[idx].astype(int).to_numpy()
        local_df = merged_df.iloc[idx].reset_index(drop=True)

        n_splits_eff, n_repeats_eff, splits = _build_regression_splits(
            n_samples=y.size,
            n_splits=reg_cv_splits,
            n_repeats=reg_cv_repeats,
            seed=seed + hash(target_name) % 997,
        )
        if not splits:
            skipped_targets.append(
                {
                    "target_name": target_name,
                    "target_type": "regression",
                    "reason": "unable_to_build_cv_splits",
                }
            )
            continue

        cv_registry["regression"][target_name] = {
            "n_samples": int(y.size),
            "n_splits": int(n_splits_eff),
            "n_repeats": int(n_repeats_eff),
            "shared_across_model_groups": True,
            "folds": [
                {
                    "fold_id": f"r{_split_meta(i, n_splits_eff)[0]:02d}_f{_split_meta(i, n_splits_eff)[1]:02d}",
                    "repeat_index": int(_split_meta(i, n_splits_eff)[0]),
                    "split_index": int(_split_meta(i, n_splits_eff)[1]),
                    "train_segment_ids": [int(seg_ids[j]) for j in tr.tolist()],
                    "test_segment_ids": [int(seg_ids[j]) for j in te.tolist()],
                }
                for i, (tr, te) in enumerate(splits)
            ],
        }

        x_group_frames = _build_group_frames(local_df=local_df, feature_groups=feature_groups)
        eligible_groups = _eligible_model_groups(x_group_frames)
        if not eligible_groups:
            warnings.append(f"target={target_name} skipped: no eligible feature groups")
            continue

        for group in eligible_groups:

            x_df = x_group_frames.get(group, pd.DataFrame())
            x_vis = x_group_frames.get("visual_only")
            x_aud = x_group_frames.get("audio_only")

            for fold_idx, (tr, te) in enumerate(splits):
                rep, split = _split_meta(fold_idx, n_splits_eff)
                fold_id = f"r{rep:02d}_f{split:02d}"
                fit_status = "ok"

                y_true = y[te]
                sid_test = seg_ids[te]

                try:
                    if group == "late_fusion":
                        y_pred = _late_fusion_regression_predict(
                            x_vis=x_vis,
                            x_audio=x_aud,
                            y=y,
                            train_idx=tr,
                            test_idx=te,
                            backend=backend,
                            seed=seed + fold_idx + 5000,
                        )
                    else:
                        model = build_estimator(
                            task_type="regression",
                            seed=seed + fold_idx + 100,
                            backend=backend,
                        )
                        model.fit(x_df.iloc[tr], y[tr])
                        y_pred = np.asarray(model.predict(x_df.iloc[te]), dtype=float)
                except Exception as exc:
                    fit_status = f"fallback_baseline:{type(exc).__name__}"
                    fallback = float(np.mean(y[tr])) if len(tr) > 0 else float(np.mean(y))
                    y_pred = np.full(shape=y_true.shape, fill_value=fallback, dtype=float)

                mets = regression_metrics(y_true=y_true, y_pred=y_pred)
                fold_metric_rows.append(
                    {
                        "target_name": target_name,
                        "target_type": "regression",
                        "target_tier": tier,
                        "model_group": group,
                        "fold_id": fold_id,
                        "fit_status": fit_status,
                        **mets,
                        **{k: float("nan") for k in CLS_METRICS},
                    }
                )

                oof_rows.extend(
                    _build_oof_rows_regression(
                        segment_ids=sid_test,
                        target_name=target_name,
                        target_tier=tier,
                        model_group=group,
                        fold_id=fold_id,
                        y_true=y_true,
                        y_pred=y_pred,
                    )
                )
                completed_units += 1
                _emit(
                    {
                        "event": "unit",
                        "stage": "training/evaluating models",
                        "target_name": target_name,
                        "target_type": "regression",
                        "model_group": group,
                        "repeat_index": int(rep + 1),
                        "repeat_total": int(n_repeats_eff),
                        "fold_index": int(split + 1),
                        "fold_total": int(n_splits_eff),
                        "completed_units": int(completed_units),
                        "total_units": int(total_units),
                    }
                )

        _emit(
            {
                "event": "target_done",
                "stage": "training/evaluating models",
                "target_name": target_name,
                "target_type": "regression",
                "completed_units": int(completed_units),
                "total_units": int(total_units),
            }
        )

    for target_info in cls_targets:
        target_name = str(target_info["target_name"])
        tier = str(target_info.get("tier", "exploratory"))

        y_ser = target_series_map.get(target_name)
        if y_ser is None:
            y_ser = merged_df[target_name]
        y_ser = y_ser.astype("string").str.strip()
        y_ser = y_ser.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

        mask = y_ser.notna()
        if int(mask.sum()) < 10:
            skipped_targets.append(
                {
                    "target_name": target_name,
                    "target_type": "classification",
                    "reason": "insufficient_non_missing_samples",
                }
            )
            continue

        idx = np.flatnonzero(mask.to_numpy())
        y_labels = y_ser.iloc[idx].astype(str).to_numpy()
        seg_ids = base_segment_ids.iloc[idx].astype(int).to_numpy()
        local_df = merged_df.iloc[idx].reset_index(drop=True)

        label_encoder = LabelEncoder()
        y_codes = label_encoder.fit_transform(y_labels)
        class_ids = [int(x) for x in np.arange(len(label_encoder.classes_))]

        n_splits_eff, n_repeats_eff, splits, split_reason = _build_classification_splits(
            y_codes=y_codes,
            n_splits=reg_cv_splits,
            n_repeats=reg_cv_repeats,
            seed=seed + hash(target_name) % 991,
        )
        if not splits:
            skipped_targets.append(
                {
                    "target_name": target_name,
                    "target_type": "classification",
                    "reason": split_reason,
                }
            )
            continue

        cv_registry["classification"][target_name] = {
            "n_samples": int(y_codes.size),
            "n_classes": int(len(label_encoder.classes_)),
            "classes": [str(c) for c in label_encoder.classes_.tolist()],
            "n_splits": int(n_splits_eff),
            "n_repeats": int(n_repeats_eff),
            "shared_across_model_groups": True,
            "folds": [
                {
                    "fold_id": f"r{_split_meta(i, n_splits_eff)[0]:02d}_f{_split_meta(i, n_splits_eff)[1]:02d}",
                    "repeat_index": int(_split_meta(i, n_splits_eff)[0]),
                    "split_index": int(_split_meta(i, n_splits_eff)[1]),
                    "train_segment_ids": [int(seg_ids[j]) for j in tr.tolist()],
                    "test_segment_ids": [int(seg_ids[j]) for j in te.tolist()],
                }
                for i, (tr, te) in enumerate(splits)
            ],
        }

        x_group_frames = _build_group_frames(local_df=local_df, feature_groups=feature_groups)
        eligible_groups = _eligible_model_groups(x_group_frames)
        if not eligible_groups:
            warnings.append(f"target={target_name} skipped: no eligible feature groups")
            continue

        for group in eligible_groups:

            x_df = x_group_frames.get(group, pd.DataFrame())
            x_vis = x_group_frames.get("visual_only")
            x_aud = x_group_frames.get("audio_only")

            for fold_idx, (tr, te) in enumerate(splits):
                rep, split = _split_meta(fold_idx, n_splits_eff)
                fold_id = f"r{rep:02d}_f{split:02d}"
                fit_status = "ok"

                y_true_codes = y_codes[te]
                sid_test = seg_ids[te]

                try:
                    if group == "late_fusion":
                        y_proba = _late_fusion_classification_predict(
                            x_vis=x_vis,
                            x_audio=x_aud,
                            y_codes=y_codes,
                            train_idx=tr,
                            test_idx=te,
                            backend=backend,
                            seed=seed + fold_idx + 7000,
                            class_ids=class_ids,
                        )
                    else:
                        model = build_estimator(
                            task_type="classification",
                            seed=seed + fold_idx + 200,
                            backend=backend,
                            n_classes=len(class_ids),
                        )
                        model.fit(x_df.iloc[tr], y_codes[tr])
                        y_proba = aligned_predict_proba(model, x_df.iloc[te], class_ids)

                    y_pred_codes = np.argmax(y_proba, axis=1).astype(int)
                except Exception as exc:
                    fit_status = f"fallback_baseline:{type(exc).__name__}"
                    counts = np.bincount(y_codes[tr]) if len(tr) > 0 else np.bincount(y_codes)
                    majority = int(np.argmax(counts))
                    y_pred_codes = np.full(shape=y_true_codes.shape, fill_value=majority, dtype=int)
                    y_proba = np.zeros((y_true_codes.size, len(class_ids)), dtype=float)
                    y_proba[:, majority] = 1.0

                mets = classification_metrics(
                    y_true=y_true_codes,
                    y_pred=y_pred_codes,
                    y_proba=y_proba,
                )
                fold_metric_rows.append(
                    {
                        "target_name": target_name,
                        "target_type": "classification",
                        "target_tier": tier,
                        "model_group": group,
                        "fold_id": fold_id,
                        "fit_status": fit_status,
                        **{k: float("nan") for k in REG_METRICS},
                        **mets,
                    }
                )

                y_true_labels = label_encoder.inverse_transform(y_true_codes)
                y_pred_labels = label_encoder.inverse_transform(y_pred_codes)
                oof_rows.extend(
                    _build_oof_rows_classification(
                        segment_ids=sid_test,
                        target_name=target_name,
                        target_tier=tier,
                        model_group=group,
                        fold_id=fold_id,
                        y_true_labels=y_true_labels,
                        y_pred_labels=y_pred_labels,
                        proba=y_proba,
                        class_labels=label_encoder.classes_.tolist(),
                    )
                )
                completed_units += 1
                _emit(
                    {
                        "event": "unit",
                        "stage": "training/evaluating models",
                        "target_name": target_name,
                        "target_type": "classification",
                        "model_group": group,
                        "repeat_index": int(rep + 1),
                        "repeat_total": int(n_repeats_eff),
                        "fold_index": int(split + 1),
                        "fold_total": int(n_splits_eff),
                        "completed_units": int(completed_units),
                        "total_units": int(total_units),
                    }
                )

        _emit(
            {
                "event": "target_done",
                "stage": "training/evaluating models",
                "target_name": target_name,
                "target_type": "classification",
                "completed_units": int(completed_units),
                "total_units": int(total_units),
            }
        )

    fold_metrics_df = pd.DataFrame(fold_metric_rows)
    if fold_metrics_df.empty:
        raise RuntimeError("step7 train_eval produced no fold metrics; check input targets/features")

    all_metric_cols = [m for m in [*REG_METRICS, *CLS_METRICS] if m in fold_metrics_df.columns]
    agg_keys = ["target_name", "target_type", "target_tier", "model_group"]

    mean_df = fold_metrics_df.groupby(agg_keys, dropna=False)[all_metric_cols].mean().add_suffix("_mean")
    std_df = fold_metrics_df.groupby(agg_keys, dropna=False)[all_metric_cols].std().add_suffix("_std")
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
        metric_cols=all_metric_cols,
    )

    bootstrap_payload: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "enabled": bool(enable_bootstrap),
        "seed": int(seed),
        "bootstrap_samples": int(bootstrap_samples),
        "ci_alpha": float(bootstrap_ci_alpha),
        "results": [],
    }
    if enable_bootstrap:
        for i, (key, vals) in enumerate(sorted(bootstrap_source.items())):
            ci = _bootstrap_ci(
                values=vals,
                n_bootstrap=int(bootstrap_samples),
                alpha=float(bootstrap_ci_alpha),
                seed=seed + 10000 + i,
            )
            target_name, metric, comp = key.split("::", 2)
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

    per_target_path = out_dir / "per_target_metrics.csv"
    comparison_path = out_dir / "model_comparison.csv"
    paired_path = out_dir / "paired_deltas.csv"
    bootstrap_path = out_dir / "bootstrap_ci.json"
    oof_path = out_dir / "oof_predictions.csv"
    cv_path = out_dir / "cv_split_registry.json"

    per_target_df.to_csv(per_target_path, index=False, encoding="utf-8")
    comparison_df.to_csv(comparison_path, index=False, encoding="utf-8")
    if not paired_df.empty:
        paired_df.to_csv(paired_path, index=False, encoding="utf-8")
    else:
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

    bootstrap_path.write_text(json.dumps(bootstrap_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    oof_df.to_csv(oof_path, index=False, encoding="utf-8")
    cv_path.write_text(json.dumps(cv_registry, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(
        "step7 train_eval done | backend=%s targets(reg=%d cls=%d) folds=%d",
        backend,
        len(reg_targets),
        len(cls_targets),
        len(fold_metrics_df),
    )

    return {
        "backend": backend,
        "backend_info": {
            "backend": backend_info.backend,
            "available": backend_info.available,
            "fallback_chain": backend_info.fallback_chain,
            "reason": backend_info.reason,
        },
        "per_target_metrics_csv": per_target_path.as_posix(),
        "model_comparison_csv": comparison_path.as_posix(),
        "paired_deltas_csv": paired_path.as_posix(),
        "bootstrap_ci_json": bootstrap_path.as_posix(),
        "oof_predictions_csv": oof_path.as_posix(),
        "cv_split_registry_json": cv_path.as_posix(),
        "skipped_targets": skipped_targets,
        "warnings": warnings,
        "fold_metrics_rows": int(len(fold_metrics_df)),
        "oof_rows": int(len(oof_df)),
        "completed_units": int(completed_units),
        "total_units": int(total_units),
    }
