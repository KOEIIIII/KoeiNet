


"""Generate two-rater blind annotation packs with hidden duplicates."""

from __future__ import annotations

import json
import logging
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .annotation_template import (
    ANNOTATOR_PACK_COLUMNS,
    build_anchor_examples_markdown,
    build_annotation_instructions_markdown,
)

logger = logging.getLogger("fusion.validation_pack")


@dataclass(frozen=True)
class ValidationPackConfig:
    """Configuration for two-rater validation pack generation."""

    unique_segments: int = 60
    hidden_duplicates_per_rater: int = 8
    random_seed: int = 20260310
    model_feature_top_var_cols: int = 80
    duplicate_min_gap: int = 12


def _parse_json_list(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw if str(x).strip()]
    text = str(raw).strip()
    if not text:
        return []
    try:
        out = json.loads(text)
        if isinstance(out, list):
            return [str(x) for x in out if str(x).strip()]
    except Exception:
        pass
    return []


def _load_inputs(video_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    seg_path = video_dir / "segments" / "segment_manifest.csv"
    model_path = video_dir / "fusion" / "model_feature_table.csv"
    if not seg_path.is_file():
        raise FileNotFoundError(f"missing segment manifest: {seg_path.as_posix()}")
    if not model_path.is_file():
        raise FileNotFoundError(f"missing model feature table: {model_path.as_posix()}")

    seg_df = pd.read_csv(seg_path)
    model_df = pd.read_csv(model_path)
    if "segment_id" not in seg_df.columns:
        raise ValueError("segment manifest missing segment_id")
    if "segment_id" not in model_df.columns:
        raise ValueError("model feature table missing segment_id")

    seg_df["segment_id"] = pd.to_numeric(seg_df["segment_id"], errors="coerce")
    seg_df = seg_df.dropna(subset=["segment_id"]).copy()
    seg_df["segment_id"] = seg_df["segment_id"].astype(int)

    model_df["segment_id"] = pd.to_numeric(model_df["segment_id"], errors="coerce")
    model_df = model_df.dropna(subset=["segment_id"]).copy()
    model_df["segment_id"] = model_df["segment_id"].astype(int)
    return seg_df, model_df


def _numeric_feature_matrix(
    merged_df: pd.DataFrame,
    top_var_cols: int,
) -> Tuple[np.ndarray, List[str]]:
    numeric_cols = [
        c
        for c in merged_df.columns
        if c != "segment_id" and pd.api.types.is_numeric_dtype(merged_df[c].dtype)
    ]
    if not numeric_cols:
        t = pd.to_numeric(merged_df.get("start_time_sec", pd.Series(np.arange(len(merged_df)))), errors="coerce")
        arr = np.nan_to_num(t.to_numpy(dtype=float).reshape(-1, 1), nan=0.0)
        return arr, ["start_time_sec_fallback"]

    work = merged_df[numeric_cols].copy()
    var = work.var(skipna=True)
    keep = var.sort_values(ascending=False).head(max(1, int(top_var_cols))).index.tolist()
    work = work[keep]


    for col in keep:
        s = pd.to_numeric(work[col], errors="coerce")
        med = float(s.median()) if s.notna().sum() > 0 else 0.0
        work[col] = s.fillna(med)

    arr = work.to_numpy(dtype=float)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    std[std == 0] = 1.0
    arr = (arr - mean) / std

    if "start_time_sec" in merged_df.columns:
        t = pd.to_numeric(merged_df["start_time_sec"], errors="coerce")
        t = t.fillna(float(t.median()) if t.notna().sum() > 0 else 0.0)
        tmin, tmax = float(t.min()), float(t.max())
        if tmax > tmin:
            t_norm = (t - tmin) / (tmax - tmin)
        else:
            t_norm = pd.Series(np.zeros(len(t)))
        arr = np.column_stack([arr, t_norm.to_numpy(dtype=float)])
        keep.append("timeline_norm")
    return arr, keep


def _timeline_anchor_indices(start_times: np.ndarray, anchor_count: int) -> List[int]:
    if len(start_times) == 0:
        return []
    if anchor_count <= 0:
        return []
    qs = np.linspace(0.0, 1.0, num=anchor_count)
    anchors: List[int] = []
    used = set()
    for q in qs:
        target = float(np.quantile(start_times, q))
        idx = int(np.argmin(np.abs(start_times - target)))
        if idx not in used:
            anchors.append(idx)
            used.add(idx)
    return anchors


def _greedy_diverse_sampling(
    merged_df: pd.DataFrame,
    unique_count: int,
    seed: int,
    top_var_cols: int,
) -> List[int]:
    if len(merged_df) <= unique_count:
        return merged_df["segment_id"].astype(int).tolist()

    rng = np.random.default_rng(seed)
    features, _ = _numeric_feature_matrix(merged_df, top_var_cols=top_var_cols)
    start_times = pd.to_numeric(merged_df.get("start_time_sec", pd.Series(np.arange(len(merged_df)))), errors="coerce")
    start_times = start_times.fillna(float(start_times.median()) if start_times.notna().sum() > 0 else 0.0)
    start_np = start_times.to_numpy(dtype=float)
    seg_ids = merged_df["segment_id"].astype(int).to_numpy()

    selected_idx: List[int] = []
    selected_seg_ids: List[int] = []


    anchor_count = min(10, max(4, unique_count // 8))
    for idx in _timeline_anchor_indices(start_np, anchor_count=anchor_count):
        sid = int(seg_ids[idx])
        if sid not in selected_seg_ids:
            selected_idx.append(idx)
            selected_seg_ids.append(sid)
        if len(selected_idx) >= unique_count:
            break


    if not selected_idx:
        first_idx = int(rng.integers(low=0, high=len(seg_ids)))
        selected_idx.append(first_idx)
        selected_seg_ids.append(int(seg_ids[first_idx]))

    while len(selected_idx) < unique_count:
        remaining = [i for i in range(len(seg_ids)) if i not in selected_idx]
        if not remaining:
            break

        best_i = None
        best_score = None
        need = unique_count - len(selected_idx)
        for i in remaining:
            sid = int(seg_ids[i])
            close_adjacent = any(abs(sid - s) <= 1 for s in selected_seg_ids)
            d_min = float(min(np.linalg.norm(features[i] - features[j]) for j in selected_idx))
            score = d_min - (0.15 if close_adjacent and len(remaining) > need else 0.0)
            if (best_score is None) or (score > best_score):
                best_score = score
                best_i = i
        assert best_i is not None
        selected_idx.append(int(best_i))
        selected_seg_ids.append(int(seg_ids[best_i]))

    return selected_seg_ids[:unique_count]


def _prepare_visual_assets(
    seg_df: pd.DataFrame,
    selected_segments: Sequence[int],
    previews_dir: Path,
    strips_dir: Path,
) -> Dict[int, Dict[str, str]]:
    previews_dir.mkdir(parents=True, exist_ok=True)
    strips_dir.mkdir(parents=True, exist_ok=True)
    asset_map: Dict[int, Dict[str, str]] = {}

    cv2 = None
    try:
        import cv2 as _cv2

        cv2 = _cv2
    except Exception:
        cv2 = None
        logger.warning("cv2 unavailable; frame strips will be skipped")

    seg_lookup = seg_df.set_index("segment_id")
    for sid in selected_segments:
        if sid not in seg_lookup.index:
            continue
        row = seg_lookup.loc[sid]
        frame_paths = _parse_json_list(row.get("included_frame_paths"))
        frame_paths = [p for p in frame_paths if Path(p).is_file()]

        preview_path = previews_dir / f"segment_{sid:04d}.jpg"
        strip_path = strips_dir / f"segment_{sid:04d}_strip.jpg"

        if frame_paths:
            center_path = Path(frame_paths[len(frame_paths) // 2])
            try:
                shutil.copy2(center_path.as_posix(), preview_path.as_posix())
            except Exception as exc:
                logger.warning("copy preview failed sid=%s err=%s", sid, exc)

            if cv2 is not None:
                try:
                    picks = [frame_paths[0], frame_paths[len(frame_paths) // 2], frame_paths[-1]]
                    imgs = []
                    for p in picks:
                        img = cv2.imread(str(p))
                        if img is not None:
                            img = cv2.resize(img, (320, 180))
                            imgs.append(img)
                    if len(imgs) >= 2:
                        strip = cv2.hconcat(imgs)
                        cv2.imwrite(str(strip_path), strip)
                except Exception as exc:
                    logger.warning("build frame strip failed sid=%s err=%s", sid, exc)

        asset_map[int(sid)] = {
            "preview_path": preview_path.as_posix() if preview_path.is_file() else "",
            "frame_strip_path": strip_path.as_posix() if strip_path.is_file() else "",
        }
    return asset_map


def _find_validation_source_audio(video_dir: Path) -> Optional[Path]:
    """Find the best local source audio file for validation clip export."""
    search_dirs = [
        video_dir / "audio_events",
        video_dir / "soundscape",
        video_dir / "split",
        video_dir,
    ]
    wav_candidates: List[Path] = []
    other_candidates: List[Path] = []
    for d in search_dirs:
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*")):
            if not p.is_file():
                continue
            suf = p.suffix.lower()
            if suf == ".wav":
                wav_candidates.append(p)
            elif suf in {".mp3", ".flac", ".ogg", ".m4a"}:
                other_candidates.append(p)

    def _pick_largest(paths: Sequence[Path]) -> Optional[Path]:
        if not paths:
            return None
        return sorted(paths, key=lambda x: int(x.stat().st_size) if x.is_file() else 0, reverse=True)[0]

    return _pick_largest(wav_candidates) or _pick_largest(other_candidates)


def _existing_clip_is_valid(clip_path: Path, expected_duration_sec: float) -> bool:
    if (not clip_path.is_file()) or clip_path.stat().st_size <= 0:
        return False
    try:
        import soundfile as sf

        info = sf.info(str(clip_path))
        if int(info.frames) <= 0 or int(info.samplerate) <= 0:
            return False
        actual = float(info.frames) / float(info.samplerate)
        tol = max(1.0 / float(info.samplerate), 0.02)
        return abs(actual - float(expected_duration_sec)) <= tol
    except Exception:

        return True


def _prepare_audio_assets(
    seg_df: pd.DataFrame,
    selected_segments: Sequence[int],
    video_dir: Path,
    audio_clips_dir: Path,
) -> Dict[int, str]:
    """
    Export one reusable audio clip per canonical segment_id.

    Clip naming is stable: validation/audio_clips/segment_<segment_id>.wav
    """
    audio_clips_dir.mkdir(parents=True, exist_ok=True)
    seg_lookup = seg_df.set_index("segment_id")
    audio_map: Dict[int, str] = {int(sid): "" for sid in selected_segments}

    source_audio = _find_validation_source_audio(video_dir)
    if source_audio is None:
        logger.warning("validation audio source not found under %s", video_dir.as_posix())
        return audio_map

    try:
        import soundfile as sf
    except Exception as exc:
        logger.warning("soundfile unavailable; skip validation audio clip export: %s", exc)
        return audio_map

    try:
        waveform, sample_rate = sf.read(str(source_audio), always_2d=True, dtype="float32")
    except Exception as exc:
        logger.warning("read validation source audio failed path=%s err=%s", source_audio.as_posix(), exc)
        return audio_map

    if waveform.ndim != 2 or waveform.shape[0] <= 0 or int(sample_rate) <= 0:
        logger.warning("invalid audio waveform for validation clips path=%s", source_audio.as_posix())
        return audio_map

    total_samples = int(waveform.shape[0])
    channels = int(waveform.shape[1])
    generated = 0
    reused = 0
    missing = 0

    for sid in selected_segments:
        sid_int = int(sid)
        if sid_int not in seg_lookup.index:
            missing += 1
            continue

        row = seg_lookup.loc[sid_int]
        start = pd.to_numeric(row.get("start_time_sec"), errors="coerce")
        end = pd.to_numeric(row.get("end_time_sec"), errors="coerce")
        if pd.isna(start) or pd.isna(end):
            missing += 1
            continue

        start_sec = max(0.0, float(start))
        end_sec = max(start_sec, float(end))
        expected_duration_sec = max(0.0, end_sec - start_sec)

        clip_path = audio_clips_dir / f"segment_{sid_int:04d}.wav"
        if _existing_clip_is_valid(clip_path, expected_duration_sec):
            audio_map[sid_int] = clip_path.as_posix()
            reused += 1
            continue

        start_idx = int(round(start_sec * float(sample_rate)))
        end_idx = int(round(end_sec * float(sample_rate)))
        expected_samples = max(1, end_idx - start_idx)

        src_start = max(0, min(total_samples, start_idx))
        src_end = max(0, min(total_samples, end_idx))
        clip = waveform[src_start:src_end]

        if clip.shape[0] < expected_samples:
            pad = np.zeros((expected_samples - clip.shape[0], channels), dtype=clip.dtype)
            clip = np.vstack([clip, pad])
        elif clip.shape[0] > expected_samples:
            clip = clip[:expected_samples]

        try:
            sf.write(str(clip_path), clip, int(sample_rate))
            if clip_path.is_file() and clip_path.stat().st_size > 0:
                audio_map[sid_int] = clip_path.as_posix()
                generated += 1
            else:
                missing += 1
        except Exception as exc:
            logger.warning("write validation audio clip failed sid=%s err=%s", sid_int, exc)
            missing += 1

    logger.info(
        "validation audio clips ready | source=%s generated=%d reused=%d missing=%d",
        source_audio.as_posix(),
        generated,
        reused,
        missing,
    )
    return audio_map


def _build_sequence_with_hidden_duplicates(
    unique_segments: Sequence[int],
    duplicate_segments: Sequence[int],
    rng: random.Random,
    min_gap: int,
) -> List[int]:
    seq = list(unique_segments)
    rng.shuffle(seq)

    for sid in duplicate_segments:
        if sid not in seq:
            continue
        orig_idx = seq.index(sid)
        positions = [p for p in range(len(seq) + 1) if abs(p - orig_idx) >= int(min_gap)]
        if not positions:
            positions = sorted(
                list(range(len(seq) + 1)),
                key=lambda p: abs(p - orig_idx),
                reverse=True,
            )
        pos = rng.choice(positions)
        seq.insert(pos, sid)
    return seq


def _build_rater_pack_and_admin(
    rater_id: str,
    sequence: Sequence[int],
    seg_meta: Mapping[int, Dict[str, Any]],
    duplicate_segments: Sequence[int],
    asset_map: Mapping[int, Mapping[str, str]],
    audio_map: Mapping[int, str],
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    dup_counter: Dict[int, int] = {}
    duplicate_set = set(int(x) for x in duplicate_segments)
    admin_rows: List[Dict[str, Any]] = []
    pack_rows: List[Dict[str, Any]] = []

    for idx, sid in enumerate(sequence, 1):
        sid_int = int(sid)
        dup_counter[sid_int] = dup_counter.get(sid_int, 0) + 1
        occur = dup_counter[sid_int]
        is_dup = bool(sid_int in duplicate_set and occur >= 2)
        dup_group = f"{rater_id}_dup_{sid_int:04d}" if sid_int in duplicate_set else ""
        displayed_item_id = f"{rater_id}_{idx:03d}"

        meta = seg_meta[sid_int]
        primary_preview_path = str(asset_map.get(sid_int, {}).get("preview_path") or "")
        context_strip_path = str(asset_map.get(sid_int, {}).get("frame_strip_path") or "")
        audio_clip_path = str(audio_map.get(sid_int) or "")

        preview_path = primary_preview_path or context_strip_path
        pack_row = {
            "displayed_item_id": displayed_item_id,
            "segment_id": sid_int,
            "start_time_sec": float(meta["start_time_sec"]),
            "end_time_sec": float(meta["end_time_sec"]),
            "preview_path": preview_path,
            "primary_preview_path": primary_preview_path,
            "context_strip_path": context_strip_path,
            "audio_clip_path": audio_clip_path,
            "safety_score": "",
            "comfort_score": "",
            "vitality_score": "",
            "overall_problem_severity": "",
            "soundscape_pleasantness": "",
            "soundscape_eventfulness": "",
            "primary_problem_label": "",
            "confidence_score": "",
            "notes": "",
        }
        pack_rows.append(pack_row)
        admin_rows.append(
            {
                "canonical_segment_id": sid_int,
                "displayed_item_id": displayed_item_id,
                "assigned_rater": rater_id,
                "duplicate_group_id": dup_group,
                "is_hidden_duplicate": bool(is_dup),
                "preview_path": preview_path,
                "primary_preview_path": primary_preview_path,
                "context_strip_path": context_strip_path,
                "audio_clip_path": audio_clip_path,
            }
        )

    pack_df = pd.DataFrame(pack_rows, columns=ANNOTATOR_PACK_COLUMNS)
    return pack_df, admin_rows


def generate_two_rater_validation_pack(
    video_dir: str,
    config: Optional[ValidationPackConfig] = None,
) -> Dict[str, Any]:
    """
    Generate blind two-rater annotation packs with hidden duplicates.

    Outputs:
    - validation/rater_A_annotation_pack.csv
    - validation/rater_B_annotation_pack.csv
    - validation/annotation_instructions.md
    - validation/anchor_examples.md
    - validation/sample_manifest_admin.csv
    - validation/previews/*
    - validation/frame_strips/*
    - validation/audio_clips/*
    - validation/session_randomization.json
    """
    cfg = config or ValidationPackConfig()
    vdir = Path(video_dir)
    val_dir = vdir / "validation"
    previews_dir = val_dir / "previews"
    strips_dir = val_dir / "frame_strips"
    audio_clips_dir = val_dir / "audio_clips"
    val_dir.mkdir(parents=True, exist_ok=True)

    seg_df, model_df = _load_inputs(vdir)
    merged = seg_df.merge(model_df, on="segment_id", how="inner")
    if merged.empty:
        raise RuntimeError("no overlapping segments between segment_manifest and model_feature_table")

    unique_n = min(int(cfg.unique_segments), int(len(merged)))
    selected = _greedy_diverse_sampling(
        merged_df=merged,
        unique_count=unique_n,
        seed=int(cfg.random_seed),
        top_var_cols=int(cfg.model_feature_top_var_cols),
    )
    selected = list(dict.fromkeys(int(x) for x in selected))
    if len(selected) < unique_n:
        extra_pool = [int(x) for x in merged["segment_id"].tolist() if int(x) not in set(selected)]
        selected.extend(extra_pool[: max(0, unique_n - len(selected))])
    selected = selected[:unique_n]

    dup_n = min(int(cfg.hidden_duplicates_per_rater), max(0, len(selected)))
    rng_a = random.Random(int(cfg.random_seed) + 101)
    rng_b = random.Random(int(cfg.random_seed) + 202)
    dup_a = rng_a.sample(selected, k=dup_n) if dup_n > 0 else []
    dup_b = rng_b.sample(selected, k=dup_n) if dup_n > 0 else []

    seq_a = _build_sequence_with_hidden_duplicates(
        unique_segments=selected,
        duplicate_segments=dup_a,
        rng=rng_a,
        min_gap=int(cfg.duplicate_min_gap),
    )
    seq_b = _build_sequence_with_hidden_duplicates(
        unique_segments=selected,
        duplicate_segments=dup_b,
        rng=rng_b,
        min_gap=int(cfg.duplicate_min_gap),
    )

    seg_meta = (
        seg_df.set_index("segment_id")[["start_time_sec", "end_time_sec", "included_frame_paths"]]
        .to_dict("index")
    )
    asset_map = _prepare_visual_assets(
        seg_df=seg_df,
        selected_segments=selected,
        previews_dir=previews_dir,
        strips_dir=strips_dir,
    )
    audio_map = _prepare_audio_assets(
        seg_df=seg_df,
        selected_segments=selected,
        video_dir=vdir,
        audio_clips_dir=audio_clips_dir,
    )

    pack_a, admin_a = _build_rater_pack_and_admin(
        rater_id="A",
        sequence=seq_a,
        seg_meta=seg_meta,
        duplicate_segments=dup_a,
        asset_map=asset_map,
        audio_map=audio_map,
    )
    pack_b, admin_b = _build_rater_pack_and_admin(
        rater_id="B",
        sequence=seq_b,
        seg_meta=seg_meta,
        duplicate_segments=dup_b,
        asset_map=asset_map,
        audio_map=audio_map,
    )

    admin_df = pd.DataFrame(admin_a + admin_b)
    admin_df = admin_df[
        [
            "canonical_segment_id",
            "displayed_item_id",
            "assigned_rater",
            "duplicate_group_id",
            "is_hidden_duplicate",
            "preview_path",
            "primary_preview_path",
            "context_strip_path",
            "audio_clip_path",
        ]
    ]

    pack_a_path = val_dir / "rater_A_annotation_pack.csv"
    pack_b_path = val_dir / "rater_B_annotation_pack.csv"
    admin_path = val_dir / "sample_manifest_admin.csv"
    inst_path = val_dir / "annotation_instructions.md"
    anchor_path = val_dir / "anchor_examples.md"
    rand_path = val_dir / "session_randomization.json"

    pack_a.to_csv(pack_a_path, index=False, encoding="utf-8")
    pack_b.to_csv(pack_b_path, index=False, encoding="utf-8")
    admin_df.to_csv(admin_path, index=False, encoding="utf-8")
    inst_path.write_text(build_annotation_instructions_markdown(), encoding="utf-8")
    anchor_path.write_text(build_anchor_examples_markdown(), encoding="utf-8")

    randomization = {
        "random_seed": int(cfg.random_seed),
        "protocol": {
            "raters": 2,
            "unique_segments": int(unique_n),
            "hidden_duplicates_per_rater": int(dup_n),
            "rows_per_rater": int(len(seq_a)),
        },
        "selected_segment_ids": selected,
        "rater_A_duplicate_segment_ids": dup_a,
        "rater_B_duplicate_segment_ids": dup_b,
        "rater_A_sequence_segment_ids": seq_a,
        "rater_B_sequence_segment_ids": seq_b,
    }
    rand_path.write_text(json.dumps(randomization, ensure_ascii=False, indent=2), encoding="utf-8")

    out = {
        "unique_segments": int(unique_n),
        "hidden_duplicates_per_rater": int(dup_n),
        "rows_per_rater": int(len(seq_a)),
        "random_seed": int(cfg.random_seed),
        "rater_A_annotation_pack_csv": pack_a_path.as_posix(),
        "rater_B_annotation_pack_csv": pack_b_path.as_posix(),
        "annotation_instructions_md": inst_path.as_posix(),
        "anchor_examples_md": anchor_path.as_posix(),
        "sample_manifest_admin_csv": admin_path.as_posix(),
        "previews_dir": previews_dir.as_posix(),
        "frame_strips_dir": strips_dir.as_posix(),
        "audio_clips_dir": audio_clips_dir.as_posix(),
        "session_randomization_json": rand_path.as_posix(),
        "downstream_schema_compatible_step7": True,
    }

    logger.info(
        "validation pack generated | unique=%d dup_per_rater=%d rows_per_rater=%d",
        out["unique_segments"],
        out["hidden_duplicates_per_rater"],
        out["rows_per_rater"],
    )
    return out
