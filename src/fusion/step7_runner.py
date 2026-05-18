


"""Step-7 runner: adjudicated-label fusion modeling and evaluation pipeline."""

from __future__ import annotations

import importlib
import json
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.config import (
    STEP7_BOOTSTRAP_CI_ALPHA,
    STEP7_BOOTSTRAP_SAMPLES,
    STEP7_CLASS_MIN_COUNT,
    STEP7_ENABLE_BOOTSTRAP,
    STEP7_REG_CV_REPEATS,
    STEP7_REG_CV_SPLITS,
    STEP7_SEED,
)

from .explainability import run_explainability
from .target_registry import build_target_registry
from .train_eval import run_train_eval

logger = logging.getLogger("fusion.step7_runner")

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
STEP7_PROGRESS_FILENAME = "step7_progress.json"
STEP7_MANIFEST_FILENAME = "step7_run_manifest.json"


class Step7ProgressTracker:
    """Persistent + console progress tracker for Step-7 execution."""

    def __init__(self, out_dir: Path, show_progress: bool = False) -> None:
        self.out_dir = out_dir
        self.path = out_dir / STEP7_PROGRESS_FILENAME
        self.show_progress = bool(show_progress)
        self.start_dt = datetime.now(timezone.utc)
        self.start_ts = time.time()
        self.current_stage = "initializing"
        self.current_target = ""
        self.current_model_group = ""
        self.current_repeat = 0
        self.current_fold = 0
        self.completed_units = 0
        self.total_units = 0
        self.status = "running"
        self.error = ""
        self._last_stage_printed = ""
        self._last_plain_progress_line = ""
        self._pbar = None

        if self.show_progress:
            try:
                if importlib.util.find_spec("tqdm") is not None:
                    from tqdm import tqdm

                    self._pbar = tqdm(total=0, unit="unit", desc="step7 train/eval", leave=True)
            except Exception:
                self._pbar = None

        self._write()

    def close(self) -> None:
        if self._pbar is not None:
            try:
                self._pbar.close()
            except Exception:
                pass
            self._pbar = None

    def _now_elapsed(self) -> float:
        return float(max(0.0, time.time() - self.start_ts))

    def _eta_seconds(self) -> Optional[float]:
        if self.total_units <= 0 or self.completed_units <= 0:
            return None
        elapsed = self._now_elapsed()
        rate = self.completed_units / max(elapsed, 1e-9)
        remaining = max(0, self.total_units - self.completed_units)
        if rate <= 0:
            return None
        return float(remaining / rate)

    def _percent(self) -> float:
        if self.total_units <= 0:
            return 0.0
        return float(100.0 * self.completed_units / max(1, self.total_units))

    def _write(self) -> None:
        payload: Dict[str, Any] = {
            "status": self.status,
            "current_stage": self.current_stage,
            "current_target": self.current_target,
            "current_model_group": self.current_model_group,
            "current_repeat": int(self.current_repeat),
            "current_fold": int(self.current_fold),
            "completed_units": int(self.completed_units),
            "total_units": int(self.total_units),
            "percent": round(self._percent(), 6),
            "start_time": self.start_dt.isoformat(),
            "elapsed_seconds": round(self._now_elapsed(), 3),
            "eta_seconds": None if self._eta_seconds() is None else round(float(self._eta_seconds()), 3),
        }
        if self.error:
            payload["error"] = self.error

        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def _print_stage(self, stage: str) -> None:
        if stage == self._last_stage_printed:
            return
        self._last_stage_printed = stage
        print(f"[step7] stage={stage}", flush=True)

    def _print_unit_line(self, repeat_total: int, fold_total: int) -> None:
        if not self.show_progress:
            return
        if self._pbar is not None:
            return
        line = (
            f"[step7] target={self.current_target} model_group={self.current_model_group} "
            f"repeat={self.current_repeat}/{int(repeat_total)} fold={self.current_fold}/{int(fold_total)} "
            f"progress={self.completed_units}/{self.total_units}"
        )
        print(line, flush=True)
        self._last_plain_progress_line = line

    def stage(self, stage_name: str) -> None:
        self.current_stage = str(stage_name)
        self._print_stage(self.current_stage)
        self._write()

    def set_total_units(self, total_units: int) -> None:
        self.total_units = int(max(0, total_units))
        if self._pbar is not None:
            self._pbar.total = self.total_units
            self._pbar.refresh()
        self._write()

    def advance_unit(
        self,
        *,
        target_name: str,
        model_group: str,
        repeat_index: int,
        repeat_total: int,
        fold_index: int,
        fold_total: int,
        completed_units: int,
        total_units: int,
    ) -> None:
        self.current_target = str(target_name)
        self.current_model_group = str(model_group)
        self.current_repeat = int(repeat_index)
        self.current_fold = int(fold_index)
        self.completed_units = int(max(0, completed_units))
        self.total_units = int(max(self.total_units, total_units))
        self.current_stage = "training/evaluating models"

        if self._pbar is not None:
            delta = max(0, int(self.completed_units - int(self._pbar.n)))
            if delta > 0:
                self._pbar.update(delta)
            self._pbar.set_postfix_str(
                f"target={target_name} group={model_group} repeat={repeat_index}/{repeat_total} fold={fold_index}/{fold_total}"
            )
            self._pbar.refresh()
        self._print_unit_line(repeat_total=repeat_total, fold_total=fold_total)

        self._write()

    def target_done(self, target_name: str, target_type: str) -> None:
        print(
            f"[step7] target_done target={target_name} type={target_type} progress={self.completed_units}/{self.total_units}",
            flush=True,
        )
        self._write()

    def complete(self) -> None:
        self.status = "completed"
        self.current_stage = "completed"
        if self.total_units > 0 and self.completed_units < self.total_units:
            self.completed_units = self.total_units
        print(
            f"[step7] completed progress={self.completed_units}/{self.total_units} elapsed={self._now_elapsed():.1f}s",
            flush=True,
        )
        self._write()
        self.close()

    def fail(self, error: str) -> None:
        self.status = "failed"
        self.error = str(error)
        print(f"[step7] failed error={error}", flush=True)
        self._write()
        self.close()


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


def _normalize_segment_id(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if "segment_id" not in df.columns:
        raise ValueError(f"{name} missing required column: segment_id")
    out = df.copy()
    out["segment_id"] = pd.to_numeric(out["segment_id"], errors="coerce")
    bad = int(out["segment_id"].isna().sum())
    if bad > 0:
        logger.warning("%s has invalid segment_id rows: %d (dropped)", name, bad)
    out = out.dropna(subset=["segment_id"]).copy()
    out["segment_id"] = out["segment_id"].astype(int)
    return out


def _source_group_for_col(col: str, feature_meta: Mapping[str, Any]) -> str:
    item = feature_meta.get(col, {})
    if isinstance(item, dict) and item.get("source_group"):
        return str(item["source_group"])
    if "__" in col:
        return str(col.split("__", 1)[0])
    return "unknown"


def _build_feature_group_registry(
    model_df: pd.DataFrame,
    model_feature_dict: Mapping[str, Any],
) -> Dict[str, Any]:
    feature_cols = [c for c in model_df.columns if c != "segment_id"]
    feature_meta = model_feature_dict.get("feature_metadata", {})
    if not isinstance(feature_meta, dict):
        feature_meta = {}

    visual_only: List[str] = []
    audio_only: List[str] = []
    group_assignments: Dict[str, str] = {}
    unknown_group_features: List[str] = []

    for col in feature_cols:
        group = _source_group_for_col(col, feature_meta)
        group_assignments[col] = group
        if group in VISUAL_SOURCE_GROUPS:
            visual_only.append(col)
        elif group in AUDIO_SOURCE_GROUPS:
            audio_only.append(col)
        else:
            unknown_group_features.append(col)


    visual_only = list(dict.fromkeys(visual_only))
    audio_only = list(dict.fromkeys(audio_only))
    early_fusion = list(dict.fromkeys([*visual_only, *audio_only]))

    registry = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "visual_source_groups": sorted(VISUAL_SOURCE_GROUPS),
            "audio_source_groups": sorted(AUDIO_SOURCE_GROUPS),
            "note": "Membership derives from model_feature_dictionary.feature_metadata.source_group when available.",
        },
        "feature_counts": {
            "all_model_features": int(len(feature_cols)),
            "visual_only": int(len(visual_only)),
            "audio_only": int(len(audio_only)),
            "early_fusion": int(len(early_fusion)),
            "late_fusion_meta_features": 2,
            "unknown_group_features": int(len(unknown_group_features)),
        },
        "groups": {
            "visual_only": visual_only,
            "audio_only": audio_only,
            "early_fusion": early_fusion,
            "late_fusion": ["meta_pred_visual", "meta_pred_audio"],
        },
        "group_assignments": group_assignments,
        "unknown_group_features": unknown_group_features,
    }
    return registry


def _resolve_input_paths(
    video_dir: str,
    feature_csv: str | None = None,
    labels_csv: str | None = None,
    step7_outdir: str | None = None,
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

    adjudication_report = vdir / "validation" / "adjudication_report.json"
    reliability_report = vdir / "validation" / "reliability_report.json"
    out_dir = Path(step7_outdir) if step7_outdir else (vdir / "fusion_eval")

    return {
        "video_dir": vdir,
        "feature_csv": feature_path,
        "feature_dict_json": feature_dict_path,
        "labels_csv": label_path,
        "labels_source": label_source,
        "adjudication_report_json": adjudication_report,
        "reliability_report_json": reliability_report,
        "out_dir": out_dir,
    }


def _clean_step7_outdir(video_dir: Path, out_dir: Path, clean_enabled: bool) -> None:
    """Optionally remove only `<video_dir>/fusion_eval` before Step-7 run."""
    if not bool(clean_enabled):
        return

    default_out = (video_dir / "fusion_eval").resolve()
    target_out = out_dir.resolve()
    if target_out != default_out:
        raise ValueError(
            "--step7_clean_outdir 仅允许清理默认目录 output/<video>/fusion_eval；"
            f"当前 outdir={target_out.as_posix()}"
        )

    if target_out.is_dir():
        shutil.rmtree(target_out)
    target_out.mkdir(parents=True, exist_ok=True)


def _write_step7_run_manifest(out_dir: Path, payload: Mapping[str, Any]) -> str:
    path = out_dir / STEP7_MANIFEST_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return path.as_posix()


def _build_summary_markdown(
    *,
    label_path: Path,
    labels_source_type: str,
    target_registry: Mapping[str, Any],
    feature_registry: Mapping[str, Any],
    train_eval_result: Mapping[str, Any],
    explainability_result: Mapping[str, Any],
    per_target_metrics_path: Path,
    cv_split_registry_path: Path,
    out_path: Path,
) -> None:
    per_target_df = pd.read_csv(per_target_metrics_path) if per_target_metrics_path.is_file() else pd.DataFrame()
    confirmatory_targets = [
        t["target_name"]
        for t in target_registry.get("targets", [])
        if t.get("enabled", False)
        and t.get("target_type") == "regression"
        and t.get("tier") == "confirmatory"
    ]

    lines: List[str] = []
    lines.append("# Step 7 Fusion Evaluation Summary")
    lines.append("")
    lines.append("## 1) Label Source")
    lines.append(f"- labels_csv: `{label_path.as_posix()}`")
    lines.append(f"- labels_source_type: `{labels_source_type}`")
    lines.append("- primary policy: adjudicated labels are preferred when available.")
    lines.append("")
    lines.append("## 2) Confirmatory vs Exploratory Targets")
    lines.append(f"- confirmatory_regression: {', '.join(target_registry.get('confirmatory_regression_targets', []))}")
    lines.append(f"- exploratory_regression: {', '.join(target_registry.get('exploratory_regression_targets', []))}")
    lines.append(f"- exploratory_classification: {', '.join(target_registry.get('exploratory_classification_targets', []))}")
    lines.append("")
    lines.append("## 3) Classification Stability Decision")
    cls_items = [
        t
        for t in target_registry.get("targets", [])
        if t.get("target_type") == "classification"
    ]
    if not cls_items:
        lines.append("- no classification target configured.")
    else:
        for t in cls_items:
            pol = t.get("classification_policy", {})
            lines.append(
                "- target=`{}` enabled={} decision={} reason={}".format(
                    t.get("target_name"),
                    t.get("enabled"),
                    pol.get("decision", "n/a"),
                    t.get("reason", ""),
                )
            )
    lines.append("")
    lines.append("## 4) Feature Groups Compared")
    fc = feature_registry.get("feature_counts", {})
    lines.append(f"- visual_only: {fc.get('visual_only', 0)} features")
    lines.append(f"- audio_only: {fc.get('audio_only', 0)} features")
    lines.append(f"- early_fusion: {fc.get('early_fusion', 0)} features")
    lines.append("- late_fusion: meta-learner over out-of-fold visual/audio predictions only")
    lines.append("")
    lines.append("## 5) CV Design")
    cv_payload = _safe_read_json(cv_split_registry_path)
    reg_info = cv_payload.get("regression", {}) if isinstance(cv_payload, dict) else {}
    reg_n_splits = STEP7_REG_CV_SPLITS
    reg_n_repeats = STEP7_REG_CV_REPEATS
    if isinstance(reg_info, dict) and reg_info:
        first_key = sorted(reg_info.keys())[0]
        first_item = reg_info.get(first_key, {})
        if isinstance(first_item, dict):
            reg_n_splits = int(first_item.get("n_splits", reg_n_splits))
            reg_n_repeats = int(first_item.get("n_repeats", reg_n_repeats))
    lines.append(
        "- regression: RepeatedKFold with shared splits across model groups "
        f"(n_splits={reg_n_splits}, n_repeats={reg_n_repeats})"
    )
    lines.append("- classification: RepeatedStratifiedKFold when class support is adequate")
    lines.append(f"- model backend: `{train_eval_result.get('backend')}`")
    lines.append("")
    lines.append("## 6) Confirmatory Best Groups")
    if per_target_df.empty:
        lines.append("- no metrics table found.")
    else:
        for target in confirmatory_targets:
            sub = per_target_df[
                (per_target_df["target_name"] == target)
                & (per_target_df["target_type"] == "regression")
            ].copy()
            if sub.empty or "mae_mean" not in sub.columns:
                lines.append(f"- {target}: unavailable")
                continue
            sub = sub.sort_values("mae_mean", ascending=True)
            best = sub.iloc[0]
            lines.append(
                f"- {target}: best=`{best['model_group']}` (MAE={float(best['mae_mean']):.4f})"
            )
            v = sub[sub["model_group"] == "visual_only"]
            e = sub[sub["model_group"] == "early_fusion"]
            if not v.empty and not e.empty:
                lines.append(
                    f"  fusion_vs_visual (MAE delta early-visual) = {float(e.iloc[0]['mae_mean'] - v.iloc[0]['mae_mean']):.4f}"
                )
    lines.append("")
    lines.append("## 7) Explainability Highlights")
    imp_path = Path(str(explainability_result.get("feature_importance_csv", "")))
    if imp_path.is_file():
        imp_df = pd.read_csv(imp_path)
        if not imp_df.empty:
            for target in confirmatory_targets:
                sub = imp_df[imp_df["target_name"] == target].head(5)
                if sub.empty:
                    continue
                tops = ", ".join(sub["feature"].astype(str).tolist())
                lines.append(f"- {target}: top features -> {tops}")
        else:
            lines.append("- no feature importance rows exported.")
    else:
        lines.append("- feature importance file missing.")
    lines.append("")
    lines.append("## 8) Caveats")
    lines.append("- small labeled sample size; results may have high variance.")
    lines.append("- labels are adjudicated human ratings (improves quality but still subjective).")
    lines.append("- exploratory targets should be interpreted cautiously.")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def run_step7_fusion_eval(
    video_dir: str,
    *,
    feature_csv: str | None = None,
    labels_csv: str | None = None,
    step7_outdir: str | None = None,
    seed: int | None = None,
    smoke_test: bool = False,
    clean_outdir: bool = False,
    show_progress: bool = False,
) -> Dict[str, Any]:
    """
    Run updated Step-7 modeling/evaluation using adjudicated labels by default.

    This stage is read-only on prior artifacts and writes only to `fusion_eval/`.
    """
    run_start_dt = datetime.now(timezone.utc)
    run_start_time_iso = run_start_dt.isoformat()

    paths = _resolve_input_paths(
        video_dir=video_dir,
        feature_csv=feature_csv,
        labels_csv=labels_csv,
        step7_outdir=step7_outdir,
    )
    out_dir = paths["out_dir"]
    video_dir_path = Path(video_dir)
    _clean_step7_outdir(video_dir=video_dir_path, out_dir=out_dir, clean_enabled=bool(clean_outdir))
    out_dir.mkdir(parents=True, exist_ok=True)

    progress_tracker = Step7ProgressTracker(out_dir=out_dir, show_progress=bool(show_progress))

    feature_path = paths["feature_csv"]
    labels_path = paths["labels_csv"]
    feature_dict_path = paths["feature_dict_json"]
    run_seed = int(seed if seed is not None else STEP7_SEED)

    success = False
    failure_reason = ""
    dataset_path = out_dir / "step7_modeling_dataset.csv"
    target_registry_path = out_dir / "target_registry.json"
    feature_registry_path = out_dir / "feature_group_registry.json"
    summary_path = out_dir / "step7_summary.md"
    manifest_path = out_dir / STEP7_MANIFEST_FILENAME

    train_eval_result: Dict[str, Any] = {}
    explainability_result: Dict[str, Any] = {}
    target_registry: Dict[str, Any] = {}
    feature_registry: Dict[str, Any] = {}
    enabled_target_names: List[str] = []
    model_group_names: List[str] = []

    try:
        progress_tracker.stage("loading data")

        if not feature_path.is_file():
            raise FileNotFoundError(f"model feature table not found: {feature_path.as_posix()}")
        if not labels_path.is_file():
            raise FileNotFoundError(
                "label table not found (expected adjudicated labels first): "
                f"{labels_path.as_posix()}"
            )

        model_df = _normalize_segment_id(pd.read_csv(feature_path), "model_feature_table")
        labels_df = _normalize_segment_id(pd.read_csv(labels_path), "labels_table")
        model_feature_dict = _safe_read_json(feature_dict_path)
        adjudication_report = _safe_read_json(paths["adjudication_report_json"])
        reliability_report = _safe_read_json(paths["reliability_report_json"])

        model_seg = set(model_df["segment_id"].tolist())
        label_seg = set(labels_df["segment_id"].tolist())
        missing_in_labels = sorted(int(x) for x in (model_seg - label_seg))
        missing_in_features = sorted(int(x) for x in (label_seg - model_seg))
        if missing_in_labels:
            logger.warning("segments missing labels: %d", len(missing_in_labels))
        if missing_in_features:
            logger.warning("segments missing model features: %d", len(missing_in_features))

        merged = model_df.merge(labels_df, on="segment_id", how="inner", suffixes=("", "_label"))
        if merged.empty:
            raise RuntimeError("merged modeling dataset is empty after join on segment_id")
        merged.to_csv(dataset_path, index=False, encoding="utf-8")

        progress_tracker.stage("building target registry")
        target_registry, target_series = build_target_registry(
            labels_df=merged,
            min_class_count=int(STEP7_CLASS_MIN_COUNT),
        )
        target_registry["inputs"] = {
            "feature_csv": feature_path.as_posix(),
            "feature_dict_json": feature_dict_path.as_posix(),
            "labels_csv": labels_path.as_posix(),
            "adjudication_report_json": paths["adjudication_report_json"].as_posix(),
            "reliability_report_json": paths["reliability_report_json"].as_posix(),
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
        target_registry_path.write_text(
            json.dumps(target_registry, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        progress_tracker.stage("building feature groups")
        feature_registry = _build_feature_group_registry(
            model_df=model_df,
            model_feature_dict=model_feature_dict,
        )
        feature_registry_path.write_text(
            json.dumps(feature_registry, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        enabled_target_names = [
            str(t.get("target_name"))
            for t in target_registry.get("targets", [])
            if bool(t.get("enabled", False))
        ]
        model_group_names = list(feature_registry.get("groups", {}).keys())

        def _on_train_progress(event: Mapping[str, Any]) -> None:
            et = str(event.get("event", "")).strip().lower()
            if et == "stage":
                progress_tracker.stage(str(event.get("stage", "training/evaluating models")))
                if event.get("total_units") is not None:
                    progress_tracker.set_total_units(int(event.get("total_units", 0)))
                return
            if et == "unit":
                progress_tracker.advance_unit(
                    target_name=str(event.get("target_name", "")),
                    model_group=str(event.get("model_group", "")),
                    repeat_index=int(event.get("repeat_index", 0)),
                    repeat_total=int(event.get("repeat_total", 0)),
                    fold_index=int(event.get("fold_index", 0)),
                    fold_total=int(event.get("fold_total", 0)),
                    completed_units=int(event.get("completed_units", 0)),
                    total_units=int(event.get("total_units", 0)),
                )
                return
            if et == "target_done":
                progress_tracker.target_done(
                    target_name=str(event.get("target_name", "")),
                    target_type=str(event.get("target_type", "")),
                )

        progress_tracker.stage("generating CV splits")
        train_eval_result = run_train_eval(
            merged_df=merged,
            feature_groups=feature_registry["groups"],
            target_registry=target_registry,
            target_series_map=target_series,
            out_dir=out_dir,
            seed=run_seed,
            reg_cv_splits=int(STEP7_REG_CV_SPLITS),
            reg_cv_repeats=int(STEP7_REG_CV_REPEATS),
            bootstrap_samples=int(STEP7_BOOTSTRAP_SAMPLES),
            bootstrap_ci_alpha=float(STEP7_BOOTSTRAP_CI_ALPHA),
            enable_bootstrap=bool(STEP7_ENABLE_BOOTSTRAP),
            smoke_test=bool(smoke_test),
            progress_callback=_on_train_progress,
        )

        progress_tracker.stage("explainability export")
        explainability_result = run_explainability(
            merged_df=merged,
            feature_groups=feature_registry["groups"],
            target_registry=target_registry,
            per_target_metrics_csv=str(train_eval_result["per_target_metrics_csv"]),
            out_dir=out_dir,
            backend=str(train_eval_result["backend"]),
            seed=run_seed,
        )

        progress_tracker.stage("writing summary outputs")
        _build_summary_markdown(
            label_path=labels_path,
            labels_source_type=str(paths["labels_source"]),
            target_registry=target_registry,
            feature_registry=feature_registry,
            train_eval_result=train_eval_result,
            explainability_result=explainability_result,
            per_target_metrics_path=Path(str(train_eval_result["per_target_metrics_csv"])),
            cv_split_registry_path=Path(str(train_eval_result["cv_split_registry_json"])),
            out_path=summary_path,
        )

        progress_tracker.complete()
        success = True

        result = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "video_dir": Path(video_dir).as_posix(),
            "label_source_csv": labels_path.as_posix(),
            "label_source_type": str(paths["labels_source"]),
            "step7_outdir": out_dir.as_posix(),
            "step7_modeling_dataset_csv": dataset_path.as_posix(),
            "target_registry_json": target_registry_path.as_posix(),
            "feature_group_registry_json": feature_registry_path.as_posix(),
            "per_target_metrics_csv": train_eval_result["per_target_metrics_csv"],
            "model_comparison_csv": train_eval_result["model_comparison_csv"],
            "paired_deltas_csv": train_eval_result["paired_deltas_csv"],
            "bootstrap_ci_json": train_eval_result["bootstrap_ci_json"],
            "oof_predictions_csv": train_eval_result["oof_predictions_csv"],
            "cv_split_registry_json": train_eval_result["cv_split_registry_json"],
            "feature_importance_csv": explainability_result["feature_importance_csv"],
            "shap_summary_csv": explainability_result["shap_summary_csv"],
            "plots_dir": explainability_result["plots_dir"],
            "step7_summary_md": summary_path.as_posix(),
            "backend_info": train_eval_result.get("backend_info", {}),
            "missing_in_labels_count": int(len(missing_in_labels)),
            "missing_in_features_count": int(len(missing_in_features)),
            "adjudication_report_loaded": bool(adjudication_report),
            "reliability_report_loaded": bool(reliability_report),
            "smoke_test": bool(smoke_test),
            "step7_progress_json": (out_dir / STEP7_PROGRESS_FILENAME).as_posix(),
            "step7_run_manifest_json": (out_dir / STEP7_MANIFEST_FILENAME).as_posix(),
        }
        logger.info("step7 done | out=%s labels=%s", out_dir.as_posix(), labels_path.as_posix())
        return result
    except Exception as exc:
        failure_reason = str(exc)
        progress_tracker.fail(str(exc))
        raise
    finally:
        run_end_iso = datetime.now(timezone.utc).isoformat()
        manifest_payload: Dict[str, Any] = {
            "video_dir": video_dir_path.as_posix(),
            "fusion_eval_dir": out_dir.as_posix(),
            "feature_csv": feature_path.as_posix(),
            "labels_csv": labels_path.as_posix(),
            "seed": int(run_seed),
            "model_groups": model_group_names,
            "enabled_targets": enabled_target_names,
            "start_time": run_start_time_iso,
            "end_time": run_end_iso,
            "success": bool(success),
            "failure_reason": failure_reason,
            "clean_outdir": bool(clean_outdir),
            "show_progress": bool(show_progress),
            "smoke_test": bool(smoke_test),
            "progress_file": (out_dir / STEP7_PROGRESS_FILENAME).as_posix(),
            "summary_file": summary_path.as_posix() if summary_path.is_file() else "",
            "outputs": {
                "step7_modeling_dataset_csv": dataset_path.as_posix() if dataset_path.is_file() else "",
                "target_registry_json": target_registry_path.as_posix() if target_registry_path.is_file() else "",
                "feature_group_registry_json": feature_registry_path.as_posix()
                if feature_registry_path.is_file()
                else "",
                "per_target_metrics_csv": train_eval_result.get("per_target_metrics_csv", ""),
                "model_comparison_csv": train_eval_result.get("model_comparison_csv", ""),
                "paired_deltas_csv": train_eval_result.get("paired_deltas_csv", ""),
                "bootstrap_ci_json": train_eval_result.get("bootstrap_ci_json", ""),
                "oof_predictions_csv": train_eval_result.get("oof_predictions_csv", ""),
                "cv_split_registry_json": train_eval_result.get("cv_split_registry_json", ""),
                "feature_importance_csv": explainability_result.get("feature_importance_csv", ""),
                "shap_summary_csv": explainability_result.get("shap_summary_csv", ""),
                "plots_dir": explainability_result.get("plots_dir", ""),
            },
        }
        try:
            manifest_written = _write_step7_run_manifest(out_dir=out_dir, payload=manifest_payload)
            manifest_path = Path(manifest_written)
        except Exception:
            pass
        finally:
            progress_tracker.close()
