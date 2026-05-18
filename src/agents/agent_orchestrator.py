


"""Step-6 multi-agent orchestrator (label-agnostic, diagnostics only)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.config import (
    AGENT_CACHE_ENABLED,
    AGENT_DISABLE_LLM,
    AGENT_MAX_RETRIES,
    ZHIPU_AGENT_MODEL,
)

from .critic_agent import CriticAgent
from .cross_modal_diagnostician import CrossModalDiagnosticianAgent
from .segment_profiler import SegmentProfilerAgent
from .soundscape_interpreter import SoundscapeInterpreterAgent
from .zhipu_client import ZhipuAgentClient

logger = logging.getLogger("agents.orchestrator")

INPUT_PATH_KEYS: Tuple[str, ...] = (
    "segment_manifest_csv",
    "audio_segment_features_csv",
    "raw_feature_table_csv",
    "model_feature_table_csv",
    "feature_dictionary_json",
    "model_feature_dictionary_json",
)


def _safe_read_csv(path: Path, required: bool = False) -> pd.DataFrame:
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"required csv missing: {path.as_posix()}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        if required:
            raise RuntimeError(f"read csv failed {path.as_posix()}: {exc}") from exc
        logger.warning("read csv failed path=%s err=%s", path.as_posix(), exc)
        return pd.DataFrame()


def _safe_read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception as exc:
        logger.warning("read json failed path=%s err=%s", path.as_posix(), exc)
        return {}


def _normalize_sid(df: pd.DataFrame, col: str = "segment_id") -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return df.copy()
    out = df.copy()
    out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=[col]).copy()
    out[col] = out[col].astype(int)
    return out


def _parse_json_list(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw if str(x).strip()]
    text = str(raw).strip()
    if not text or text.lower() == "nan":
        return []
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [str(x) for x in obj if str(x).strip()]
    except Exception:
        pass
    return []


def _to_num(v: Any) -> float:
    s = pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
    return float(s) if pd.notna(s) else float("nan")


def _top_numeric_from_row(
    row: Mapping[str, Any],
    *,
    prefix: str,
    suffix: Optional[str] = None,
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    items: List[Tuple[str, float]] = []
    for k, v in row.items():
        key = str(k)
        if not key.startswith(prefix):
            continue
        if suffix and (not key.endswith(suffix)):
            continue
        num = _to_num(v)
        if np.isnan(num):
            continue
        items.append((key, float(num)))
    items.sort(key=lambda kv: abs(kv[1]), reverse=True)
    out = []
    for key, val in items[: max(0, int(top_n))]:
        short = key[len(prefix) :]
        if suffix and short.endswith(suffix):
            short = short[: -len(suffix)]
        out.append({"feature": short, "value": round(float(val), 6)})
    return out


def _top_numeric_any(row: Mapping[str, Any], top_n: int = 8) -> List[Dict[str, Any]]:
    items: List[Tuple[str, float]] = []
    for k, v in row.items():
        key = str(k)
        if key == "segment_id":
            continue
        num = _to_num(v)
        if np.isnan(num):
            continue
        items.append((key, float(num)))
    items.sort(key=lambda kv: abs(kv[1]), reverse=True)
    return [{"feature": k, "value": round(v, 6)} for k, v in items[: max(0, int(top_n))]]


def _build_segment_payload(
    seg_row: Mapping[str, Any],
    soundscape_row: Mapping[str, Any],
    raw_row: Mapping[str, Any],
    model_row: Mapping[str, Any],
    feature_dict: Mapping[str, Any],
    model_feature_dict: Mapping[str, Any],
) -> Dict[str, Any]:
    sid = int(_to_num(seg_row.get("segment_id")))
    start_t = _to_num(seg_row.get("start_time_sec"))
    end_t = _to_num(seg_row.get("end_time_sec"))
    center_t = _to_num(seg_row.get("center_time_sec"))
    frame_count = _to_num(seg_row.get("included_frame_count"))

    audio_group = {}
    for g in ("traffic", "human", "nature", "mechanical", "other"):
        raw_key = f"audio_events__group_ratio_{g}"
        snd_key = f"group_ratio_{g}"
        val = _to_num(raw_row.get(raw_key, soundscape_row.get(snd_key)))
        if not np.isnan(val):
            audio_group[g] = round(float(val), 6)

    audio_top_events = _parse_json_list(
        raw_row.get("audio_events__top_k_events", soundscape_row.get("top_k_events"))
    )

    green = _to_num(raw_row.get("green_view__greenviewindex__mean"))
    people_mean = _to_num(raw_row.get("people__total_people__mean"))
    people_max = _to_num(raw_row.get("people__total_people__max"))

    payload = {
        "segment_id": sid,
        "timeline": {
            "start_time_sec": None if np.isnan(start_t) else float(start_t),
            "end_time_sec": None if np.isnan(end_t) else float(end_t),
            "center_time_sec": None if np.isnan(center_t) else float(center_t),
            "included_frame_count": None if np.isnan(frame_count) else float(frame_count),
        },
        "evidence": {

            "visual_major_top": _top_numeric_from_row(raw_row, prefix="visual_major__", suffix="__mean", top_n=5),
            "visual_semantic_top": _top_numeric_from_row(raw_row, prefix="visual_semantic__", suffix="__mean", top_n=6),
            "emotion_top": _top_numeric_from_row(raw_row, prefix="emotion__", suffix="__mean", top_n=4),
            "ai_activity_top": _top_numeric_from_row(raw_row, prefix="ai_activity__", suffix="__mean", top_n=4),
            "green_view_mean": None if np.isnan(green) else float(green),
            "people_total_mean": None if np.isnan(people_mean) else float(people_mean),
            "people_total_max": None if np.isnan(people_max) else float(people_max),
            "audio_group_ratios": audio_group,
            "audio_top_events": audio_top_events,
            "audio_signal_top": _top_numeric_from_row(raw_row, prefix="audio_signal__", suffix=None, top_n=8),

            "model_numeric_top": _top_numeric_any(model_row, top_n=10),
        },
        "feature_context": {
            "raw_feature_dictionary_keys": int(len(feature_dict)),
            "model_feature_dictionary_keys": int(len(model_feature_dict)),
        },
    }
    return payload


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _build_row_map(df: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
    if df.empty or "segment_id" not in df.columns:
        return {}
    out: Dict[int, Dict[str, Any]] = {}
    for _, r in df.iterrows():
        sid = int(r["segment_id"])
        out[sid] = r.to_dict()
    return out


def run_agents_stage(
    video_dir: str,
    options: Optional[Mapping[str, Any]] = None,
    progress_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Run Step-6 multi-agent reasoning in a label-agnostic way.

    Explicitly does NOT read validation/reliability/adjudication files.
    """
    opts = dict(options or {})
    vdir = Path(video_dir)
    diagnostics_dir = vdir / "diagnostics"
    cache_dir = diagnostics_dir / "cache"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    input_paths = {
        "segment_manifest_csv": (vdir / "segments" / "segment_manifest.csv").as_posix(),
        "audio_segment_features_csv": (vdir / "soundscape" / "audio_segment_features.csv").as_posix(),
        "raw_feature_table_csv": (vdir / "fusion" / "segment_feature_table.csv").as_posix(),
        "model_feature_table_csv": (vdir / "fusion" / "model_feature_table.csv").as_posix(),
        "feature_dictionary_json": (vdir / "fusion" / "feature_dictionary.json").as_posix(),
        "model_feature_dictionary_json": (vdir / "fusion" / "model_feature_dictionary.json").as_posix(),
    }

    seg_df = _normalize_sid(_safe_read_csv(Path(input_paths["segment_manifest_csv"]), required=True))
    sound_df = _normalize_sid(_safe_read_csv(Path(input_paths["audio_segment_features_csv"]), required=False))
    raw_df = _normalize_sid(_safe_read_csv(Path(input_paths["raw_feature_table_csv"]), required=False))
    model_df = _normalize_sid(_safe_read_csv(Path(input_paths["model_feature_table_csv"]), required=False))
    feat_dict = _safe_read_json(Path(input_paths["feature_dictionary_json"]))
    model_feat_dict = _safe_read_json(Path(input_paths["model_feature_dictionary_json"]))

    model_name = str(opts.get("ZHIPU_AGENT_MODEL") or ZHIPU_AGENT_MODEL or "glm-5")
    max_retries = int(opts.get("AGENT_MAX_RETRIES", AGENT_MAX_RETRIES))
    cache_enabled = bool(opts.get("AGENT_CACHE_ENABLED", AGENT_CACHE_ENABLED))
    disable_llm = bool(opts.get("AGENT_DISABLE_LLM", AGENT_DISABLE_LLM))
    if not cache_enabled:

        cache_dir = diagnostics_dir / "cache_disabled_runtime"
        cache_dir.mkdir(parents=True, exist_ok=True)

    if disable_llm:
        client = ZhipuAgentClient(available=False, unavailable_reason="agent_disable_llm=true")
    else:
        client = ZhipuAgentClient.from_apikey_env(root_dir=Path(__file__).resolve().parents[2])
    profiler = SegmentProfilerAgent(client=client, model_name=model_name, cache_dir=cache_dir.as_posix(), max_retries=max_retries)
    soundscape = SoundscapeInterpreterAgent(client=client, model_name=model_name, cache_dir=cache_dir.as_posix(), max_retries=max_retries)
    diagnoser = CrossModalDiagnosticianAgent(client=client, model_name=model_name, cache_dir=cache_dir.as_posix(), max_retries=max_retries)
    critic = CriticAgent(client=client, model_name=model_name, cache_dir=cache_dir.as_posix(), max_retries=max_retries)

    sound_map = _build_row_map(sound_df)
    raw_map = _build_row_map(raw_df)
    model_map = _build_row_map(model_df)

    profiles_rows: List[Dict[str, Any]] = []
    diagnosis_rows: List[Dict[str, Any]] = []
    critic_rows: List[Dict[str, Any]] = []

    status_counts: Dict[str, Dict[str, int]] = {
        "segment_profiler": {"ok": 0, "fallback": 0, "cache_hit": 0},
        "soundscape_interpreter": {"ok": 0, "fallback": 0, "cache_hit": 0},
        "cross_modal_diagnostician": {"ok": 0, "fallback": 0, "cache_hit": 0},
        "critic_agent": {"ok": 0, "fallback": 0, "cache_hit": 0},
    }

    ordered_segments = seg_df.sort_values("segment_id")
    if progress_callback:
        progress_callback(0, len(ordered_segments), "agents | segment reasoning")
    for idx, (_, seg_row) in enumerate(ordered_segments.iterrows()):
        sid = int(seg_row["segment_id"])
        payload = _build_segment_payload(
            seg_row=seg_row.to_dict(),
            soundscape_row=sound_map.get(sid, {}),
            raw_row=raw_map.get(sid, {}),
            model_row=model_map.get(sid, {}),
            feature_dict=feat_dict,
            model_feature_dict=model_feat_dict,
        )

        r_profile = profiler.run(payload)
        r_soundscape = soundscape.run(payload)
        diagnosis_input = {
            **payload,
            "profile_json": dict(r_profile.get("output", {})),
            "soundscape_json": dict(r_soundscape.get("output", {})),
        }
        r_diagnosis = diagnoser.run(diagnosis_input)
        critic_input = {
            **payload,
            "profile_json": dict(r_profile.get("output", {})),
            "soundscape_json": dict(r_soundscape.get("output", {})),
            "diagnosis_json": dict(r_diagnosis.get("output", {})),
        }
        r_critic = critic.run(critic_input)

        for item in (r_profile, r_soundscape, r_diagnosis, r_critic):
            agent = str(item.get("agent"))
            st = str(item.get("status"))
            if st == "ok":
                status_counts[agent]["ok"] += 1
            else:
                status_counts[agent]["fallback"] += 1
            if bool(item.get("cache_hit", False)):
                status_counts[agent]["cache_hit"] += 1

        profiles_rows.append(
            {
                "segment_id": sid,
                "start_time_sec": payload["timeline"]["start_time_sec"],
                "end_time_sec": payload["timeline"]["end_time_sec"],
                "profile_json": r_profile.get("output", {}),
                "soundscape_json": r_soundscape.get("output", {}),
                "profile_status": r_profile.get("status"),
                "soundscape_status": r_soundscape.get("status"),
                "profile_error": r_profile.get("error", ""),
                "soundscape_error": r_soundscape.get("error", ""),
            }
        )
        diagnosis_rows.append(
            {
                "segment_id": sid,
                "diagnosis_json": r_diagnosis.get("output", {}),
                "diagnosis_status": r_diagnosis.get("status"),
                "diagnosis_error": r_diagnosis.get("error", ""),
            }
        )
        critic_rows.append(
            {
                "segment_id": sid,
                "critic_json": r_critic.get("output", {}),
                "critic_status": r_critic.get("status"),
                "critic_error": r_critic.get("error", ""),
            }
        )
        if progress_callback:
            progress_callback(idx + 1, len(ordered_segments), "agents | segment reasoning")

    if not profiles_rows:
        profiles_rows = [
            {
                "segment_id": -1,
                "profile_json": {"segment_id": -1, "visual_facts": [], "audio_facts": [], "concise_summary": "empty_segment_manifest"},
                "soundscape_json": {
                    "segment_id": -1,
                    "dominant_sources": [],
                    "pleasantness_reasoning": "empty_segment_manifest",
                    "eventfulness_reasoning": "empty_segment_manifest",
                    "acoustic_risk_tags": [],
                },
                "profile_status": "fallback",
                "soundscape_status": "fallback",
                "profile_error": "empty_segment_manifest",
                "soundscape_error": "empty_segment_manifest",
            }
        ]
        diagnosis_rows = [
            {
                "segment_id": -1,
                "diagnosis_json": {
                    "segment_id": -1,
                    "problem_labels": ["mixed_or_unclear"],
                    "severity_scores": {},
                    "evidence_visual": [],
                    "evidence_audio": [],
                    "cross_modal_reason": "empty_segment_manifest",
                    "priority_actions": [],
                },
                "diagnosis_status": "fallback",
                "diagnosis_error": "empty_segment_manifest",
            }
        ]
        critic_rows = [
            {
                "segment_id": -1,
                "critic_json": {
                    "segment_id": -1,
                    "consistency_check": {"is_consistent": False, "issues": ["empty_segment_manifest"]},
                    "missing_evidence_check": {"has_missing_evidence": True, "missing_items": ["all"]},
                    "confidence_score": 1.0,
                },
                "critic_status": "fallback",
                "critic_error": "empty_segment_manifest",
            }
        ]

    profiles_path = diagnostics_dir / "segment_profiles.jsonl"
    diagnosis_path = diagnostics_dir / "segment_diagnosis.jsonl"
    critic_path = diagnostics_dir / "segment_critic.jsonl"
    _write_jsonl(profiles_path, profiles_rows)
    _write_jsonl(diagnosis_path, diagnosis_rows)
    _write_jsonl(critic_path, critic_rows)

    logger.info(
        "[agents] done | segments=%d model=%s api_available=%s profiles=%s diagnosis=%s critic=%s",
        len(profiles_rows),
        model_name,
        bool(client.available),
        profiles_path.as_posix(),
        diagnosis_path.as_posix(),
        critic_path.as_posix(),
    )
    return {
        "segment_profiles_jsonl": profiles_path.as_posix(),
        "segment_diagnosis_jsonl": diagnosis_path.as_posix(),
        "segment_critic_jsonl": critic_path.as_posix(),
        "segment_count": int(len(profiles_rows)),
        "model_name": model_name,
        "api_available": bool(client.available),
        "api_unavailable_reason": str(client.unavailable_reason or ""),
        "cache_enabled": bool(cache_enabled),
        "llm_disabled": bool(disable_llm),
        "status_counts": status_counts,
        "inputs_used": input_paths,
        "label_agnostic_step6": True,
    }
