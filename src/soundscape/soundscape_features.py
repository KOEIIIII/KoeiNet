


"""Segment-level soundscape feature extraction from existing outputs."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .audio_features import compute_segment_audio_features, load_audio_waveform, slice_waveform
from .event_group_config import EVENT_GROUP_KEYWORDS, EVENT_GROUPS, classify_event_group
from .panns_embedder import PANNSEmbedder

logger = logging.getLogger("soundscape.soundscape_features")


def _to_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _load_segment_manifest(video_dir: Path) -> pd.DataFrame:
    csv_path = video_dir / "segments" / "segment_manifest.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"缺少 segment_manifest.csv: {csv_path.as_posix()}")
    df = pd.read_csv(csv_path)
    required = {"segment_id", "start_time_sec", "end_time_sec", "center_time_sec"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"segment_manifest.csv 缺少列: {sorted(missing)}")
    return df.sort_values("segment_id").reset_index(drop=True)


def _load_audio_event_timeline(video_dir: Path) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Load audio time-sync table and convert each row to event-score pairs.
    """
    audio_dir = video_dir / "audio_events"
    detailed = audio_dir / "audio_events_time_sync.csv"
    simple = audio_dir / "audio_events_time_sync_simple.csv"

    src: Optional[Path] = None
    if detailed.exists():
        src = detailed
    elif simple.exists():
        src = simple
    if src is None:
        return None, None

    try:
        df = pd.read_csv(src)
    except Exception as exc:
        logger.warning("读取音频时间同步文件失败: %s | %s", src.as_posix(), exc)
        return None, src.as_posix()

    if "start_time_sec" not in df.columns or "end_time_sec" not in df.columns:
        logger.warning("音频时间同步缺少 start_time_sec/end_time_sec: %s", src.as_posix())
        return None, src.as_posix()

    work = df.copy()
    work["start_time_sec"] = pd.to_numeric(work["start_time_sec"], errors="coerce")
    work["end_time_sec"] = pd.to_numeric(work["end_time_sec"], errors="coerce")
    work = work.dropna(subset=["start_time_sec", "end_time_sec"]).reset_index(drop=True)
    if work.empty:
        return None, src.as_posix()


    event_pairs: List[List[Tuple[str, float]]] = []
    top_class_cols = sorted([c for c in work.columns if c.startswith("top_") and c.endswith("_class")])
    if top_class_cols:
        for _, row in work.iterrows():
            pairs: List[Tuple[str, float]] = []
            for cls_col in top_class_cols:
                rank = cls_col.split("_")[1]
                score_col = f"top_{rank}_score"
                name = str(row.get(cls_col, "")).strip()
                if not name or name.lower() == "nan":
                    continue
                score = _to_float(row.get(score_col, 0.0), 0.0)
                if score <= 0:
                    continue
                pairs.append((name, float(score)))
            event_pairs.append(pairs)
    elif "top_class" in work.columns:
        for _, row in work.iterrows():
            name = str(row.get("top_class", "")).strip()
            score = _to_float(row.get("top_score", 0.0), 0.0)
            if name and name.lower() != "nan" and score > 0:
                event_pairs.append([(name, float(score))])
            else:
                event_pairs.append([])
    else:
        event_pairs = [[] for _ in range(len(work))]

    work = work[["start_time_sec", "end_time_sec"]].copy()
    work["event_pairs"] = event_pairs
    return work, src.as_posix()


def _aggregate_event_features(seg_audio_rows: pd.DataFrame, top_k: int) -> Dict[str, Any]:
    scores: Dict[str, float] = {}
    for _, row in seg_audio_rows.iterrows():
        pairs = row.get("event_pairs", [])
        if not isinstance(pairs, list):
            continue
        for item in pairs:
            if not isinstance(item, tuple) or len(item) != 2:
                continue
            name, score = item
            scores[name] = float(scores.get(name, 0.0) + float(score))

    total = float(sum(scores.values()))
    if total <= 0:
        distribution = {}
        top_events = []
    else:
        distribution = {k: float(v / total) for k, v in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)}
        top_events = [f"{k}:{distribution[k]:.4f}" for k in list(distribution.keys())[: max(1, int(top_k))]]

    group_scores = {g: 0.0 for g in EVENT_GROUPS}
    for event_name, score in scores.items():
        group = classify_event_group(event_name)
        if group not in group_scores:
            group = "other"
        group_scores[group] += float(score)

    group_total = float(sum(group_scores.values()))
    if group_total > 0:
        group_ratio = {k: float(v / group_total) for k, v in group_scores.items()}
    else:
        group_ratio = {k: float("nan") for k in group_scores}

    out: Dict[str, Any] = {
        "top_k_events": ";".join(top_events),
        "event_class_distribution_json": json.dumps(distribution, ensure_ascii=False),
        "audio_event_row_count": int(len(seg_audio_rows)),
    }
    for g in ("traffic", "human", "nature", "mechanical", "other"):
        out[f"group_ratio_{g}"] = float(group_ratio.get(g, float("nan")))
    return out


def _find_existing_wav(video_dir: Path) -> Optional[str]:
    audio_dir = video_dir / "audio_events"
    if not audio_dir.is_dir():
        return None
    wavs = sorted(audio_dir.glob("*.wav"))
    if not wavs:
        return None
    return wavs[0].as_posix()


def _normalization_meta(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    meta: Dict[str, Dict[str, float]] = {}
    num_df = df.select_dtypes(include=[np.number]).copy()
    if num_df.empty:
        return meta
    for col in num_df.columns:
        s = pd.to_numeric(num_df[col], errors="coerce").dropna()
        if s.empty:
            continue
        meta[col] = {
            "mean": float(s.mean()),
            "std": float(s.std(ddof=0)),
            "min": float(s.min()),
            "max": float(s.max()),
            "normalization": "zscore_recommended",
        }
    return meta


def extract_soundscape_features_for_video(
    video_dir: str,
    top_k_events: int = 5,
    panns_enabled: bool = True,
    panns_checkpoint_path: Optional[str] = None,
    panns_labels_path: Optional[str] = None,
    panns_force_local_resources: bool = True,
    panns_export_dims: int = 16,
    n_fft: int = 1024,
    hop: int = 512,
    rolloff_ratio: float = 0.85,
    progress_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Extract segment-level soundscape features from existing outputs.

    Exports:
    - `<video_dir>/soundscape/audio_segment_features.csv`
    - `<video_dir>/soundscape/audio_feature_meta.json`
    """
    vdir = Path(video_dir)
    soundscape_dir = vdir / "soundscape"
    soundscape_dir.mkdir(parents=True, exist_ok=True)

    seg_df = _load_segment_manifest(vdir)
    audio_timeline_df, audio_timeline_path = _load_audio_event_timeline(vdir)

    wav_path = _find_existing_wav(vdir)
    waveform, sr, waveform_loader = (None, None, "missing")
    if wav_path:
        waveform, sr, waveform_loader = load_audio_waveform(wav_path)
    if waveform is None or sr is None:
        logger.warning("未找到可用音频波形，信号特征与嵌入将使用占位值。")

    embedder = PANNSEmbedder(
        export_dims=panns_export_dims,
        enabled=panns_enabled,
        checkpoint_path=panns_checkpoint_path,
        labels_path=panns_labels_path,
        force_local_resources=bool(panns_force_local_resources),
    )
    if embedder.is_available:
        panns_status_line = (
            "[soundscape] panns_status=available "
            f"checkpoint={embedder.checkpoint_path} labels={embedder.labels_path}"
        )
    else:
        panns_status_line = f"[soundscape] panns_status=unavailable reason={embedder.reason_unavailable}"
    print(panns_status_line)
    logger.info(panns_status_line)

    rows: List[Dict[str, Any]] = []
    if progress_callback:
        progress_callback(0, len(seg_df), "soundscape | segment features")
    for idx, (_, seg) in enumerate(seg_df.iterrows()):
        seg_id = int(seg["segment_id"])
        start_sec = _to_float(seg["start_time_sec"], 0.0)
        end_sec = _to_float(seg["end_time_sec"], start_sec)
        center_sec = _to_float(seg["center_time_sec"], (start_sec + end_sec) / 2.0)
        frame_count = int(_to_float(seg.get("included_frame_count", 0), 0))

        if audio_timeline_df is not None and not audio_timeline_df.empty:
            arows = audio_timeline_df[
                (audio_timeline_df["end_time_sec"] >= start_sec)
                & (audio_timeline_df["start_time_sec"] <= end_sec)
            ].copy()
        else:
            arows = pd.DataFrame(columns=["start_time_sec", "end_time_sec", "event_pairs"])

        event_feat = _aggregate_event_features(arows, top_k=int(top_k_events))

        seg_wav: Optional[np.ndarray]
        if waveform is not None and sr is not None:
            seg_wav = slice_waveform(waveform, int(sr), start_sec, end_sec)
        else:
            seg_wav = None

        signal_feat = compute_segment_audio_features(
            seg_wav,
            sr,
            n_fft=int(n_fft),
            hop=int(hop),
            rolloff_ratio=float(rolloff_ratio),
        )

        panns_raw = embedder.extract(seg_wav, int(sr)) if (seg_wav is not None and sr is not None) else embedder.extract(
            np.asarray([], dtype=np.float32), 0
        )
        panns_feat = embedder.to_feature_columns(panns_raw)

        row: Dict[str, Any] = {
            "segment_id": seg_id,
            "start_time_sec": start_sec,
            "end_time_sec": end_sec,
            "center_time_sec": center_sec,
            "included_frame_count": frame_count,
            "linked_audio_start_time_sec": _to_float(seg.get("audio_start_time_sec"), float("nan")),
            "linked_audio_end_time_sec": _to_float(seg.get("audio_end_time_sec"), float("nan")),
            "waveform_available": bool(waveform is not None and sr is not None),
            "waveform_loader": waveform_loader,
            "sample_rate": int(sr) if sr is not None else 0,
            "roughness_proxy_method": "envelope_modulation_30_150Hz_proxy",
        }
        row.update(event_feat)
        row.update(signal_feat)
        row.update(panns_feat)
        rows.append(row)

        logger.info(
            "[soundscape] seg=%d events=%d rms=%.6f top=%s panns=%s",
            seg_id,
            int(row.get("audio_event_row_count", 0)),
            float(row.get("rms_energy", float("nan"))),
            str(row.get("top_k_events", ""))[:80],
            "on" if bool(row.get("panns_available")) else "off",
        )
        if progress_callback:
            progress_callback(idx + 1, len(seg_df), "soundscape | segment features")

    out_df = pd.DataFrame(rows).sort_values("segment_id").reset_index(drop=True)
    csv_path = soundscape_dir / "audio_segment_features.csv"
    out_df.to_csv(csv_path, index=False, encoding="utf-8")

    normalization = _normalization_meta(out_df)
    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "video_dir": vdir.as_posix(),
        "source_files": {
            "segment_manifest_csv": (vdir / "segments" / "segment_manifest.csv").as_posix(),
            "audio_event_timeline": audio_timeline_path,
            "waveform_path": wav_path,
        },
        "settings": {
            "top_k_events": int(top_k_events),
            "panns_enabled": bool(panns_enabled),
            "panns_force_local_resources": bool(panns_force_local_resources),
            "panns_export_dims": int(panns_export_dims),
            "stft_n_fft": int(n_fft),
            "stft_hop": int(hop),
            "spectral_rolloff_ratio": float(rolloff_ratio),
        },
        "panns": {
            "panns_available": bool(embedder.is_available),
            "panns_unavailable_reason": embedder.reason_unavailable,

            "available": bool(embedder.is_available),
            "unavailable_reason": embedder.reason_unavailable,
            "checkpoint_path": embedder.checkpoint_path,
            "labels_path": embedder.labels_path,
            "resource_mode": "local_only" if bool(panns_force_local_resources) else "default",
            "export_dims": int(embedder.export_dims),
        },
        "event_group_mapping": {
            "groups": list(EVENT_GROUPS),
            "keywords": EVENT_GROUP_KEYWORDS,
        },
        "normalization_metadata": normalization,
        "qa_summary": {
            "total_segments": int(len(out_df)),
            "segments_with_missing_audio_events": int((out_df["audio_event_row_count"] <= 0).sum())
            if "audio_event_row_count" in out_df.columns
            else int(len(out_df)),
            "segments_with_waveform": int(out_df["waveform_available"].astype(bool).sum())
            if "waveform_available" in out_df.columns
            else 0,
        },
        "csv_schema": list(out_df.columns),
    }
    meta_path = soundscape_dir / "audio_feature_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "csv_path": csv_path.as_posix(),
        "meta_path": meta_path.as_posix(),
        "schema": list(out_df.columns),
        "total_segments": int(len(out_df)),
        "preview_rows": out_df.head(5).to_dict("records"),
        "qa_summary": meta["qa_summary"],
        "panns_available": bool(embedder.is_available),
        "panns_reason": embedder.reason_unavailable,
        "panns_checkpoint_path": embedder.checkpoint_path,
        "panns_labels_path": embedder.labels_path,
    }
