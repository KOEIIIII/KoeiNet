


"""Editable event-to-group mapping for soundscape feature aggregation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List


EVENT_GROUPS = ("traffic", "human", "nature", "mechanical", "other")

logger = logging.getLogger("soundscape.event_group_config")



DEFAULT_EVENT_GROUP_KEYWORDS: Dict[str, List[str]] = {
    "traffic": [
        "car",
        "bus",
        "truck",
        "motor",
        "motorcycle",
        "vehicle",
        "train",
        "rail",
        "road",
        "traffic",
        "siren",
        "horn",
    ],
    "human": [
        "speech",
        "conversation",
        "talk",
        "voice",
        "child",
        "children",
        "crowd",
        "applause",
        "laugh",
        "cry",
        "scream",
        "footstep",
        "walk",
        "run",
    ],
    "nature": [
        "bird",
        "animal",
        "dog",
        "cat",
        "insect",
        "rain",
        "water",
        "wind",
        "thunder",
        "nature",
        "ocean",
    ],
    "mechanical": [
        "engine",
        "machine",
        "mechanical",
        "drill",
        "jackhammer",
        "construction",
        "compressor",
        "printer",
        "fan",
        "air",
        "hvac",
    ],
}

_MAPPING_PATH = Path(__file__).with_name("event_group_mapping.json")


def _sanitize_mapping(raw: object) -> Dict[str, List[str]]:
    """Validate mapping object and keep only supported groups/keywords."""
    out: Dict[str, List[str]] = {}
    if not isinstance(raw, dict):
        return out
    for group in EVENT_GROUPS:
        values = raw.get(group) if isinstance(raw, dict) else None
        if isinstance(values, list):
            clean = [str(v).strip().lower() for v in values if str(v).strip()]
            out[group] = clean
    return out


def load_event_group_keywords(mapping_path: Path = _MAPPING_PATH) -> Dict[str, List[str]]:
    """
    Load editable mapping file, fallback to defaults when unavailable/invalid.
    """
    merged = {k: list(v) for k, v in DEFAULT_EVENT_GROUP_KEYWORDS.items()}
    if not mapping_path.exists():
        return merged

    try:
        raw = json.loads(mapping_path.read_text(encoding="utf-8"))
        loaded = _sanitize_mapping(raw)
        for group, words in loaded.items():
            merged[group] = words
    except Exception as exc:
        logger.warning("加载事件分组映射失败，回退默认值: %s", exc)
    return merged




EVENT_GROUP_KEYWORDS: Dict[str, List[str]] = load_event_group_keywords()


def classify_event_group(event_name: str) -> str:
    """Return one of traffic/human/nature/mechanical/other."""
    name = str(event_name or "").lower()
    if not name:
        return "other"
    for group, keywords in EVENT_GROUP_KEYWORDS.items():
        if any(k in name for k in keywords):
            return group
    return "other"
