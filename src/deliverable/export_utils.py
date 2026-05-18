


"""Shared helpers for deliverable-layer export, parsing, and path resolution."""

from __future__ import annotations

import difflib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties

logger = logging.getLogger("deliverable.export_utils")


def sanitize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): sanitize_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_value(v) for v in value]
    if isinstance(value, tuple):
        return [sanitize_value(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def to_json_text(value: Any) -> str:
    clean = sanitize_value(value)
    if isinstance(clean, (dict, list)):
        return json.dumps(clean, ensure_ascii=False)
    if clean is None:
        return ""
    return str(clean)


def dataframe_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in out.columns:
        out[column] = out[column].map(
            lambda x: to_json_text(x) if isinstance(sanitize_value(x), (dict, list)) else sanitize_value(x)
        )
    return out


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception as exc:
                logger.warning("Failed to parse JSONL row from %s: %s", path.as_posix(), exc)
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(sanitize_value(dict(row)), ensure_ascii=False) + "\n")


def parse_json_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        loaded = json.loads(text)
        return loaded if isinstance(loaded, list) else []
    except Exception:
        return []


def split_semicolon_values(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    if not text:
        return []
    return [item.strip() for item in text.split(";") if item.strip()]


def mode_or_first(values: Iterable[Any], default: str = "") -> str:
    cleaned = [str(v).strip() for v in values if str(v).strip()]
    if not cleaned:
        return default
    counts = pd.Series(cleaned).value_counts()
    return str(counts.index[0]) if not counts.empty else default


def coalesce_texts(values: Iterable[Any], limit: int = 5) -> List[str]:
    seen: List[str] = []
    for value in values:
        for item in split_semicolon_values(value):
            if item not in seen:
                seen.append(item)
            if len(seen) >= int(limit):
                return seen
    return seen


def safe_relative_path(path: Path, base_dir: Path) -> str:
    try:
        return path.relative_to(base_dir).as_posix()
    except Exception:
        return path.as_posix()


def write_optional_excel(df: pd.DataFrame, path: Path) -> Tuple[bool, str]:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        dataframe_for_csv(df).to_excel(path, index=False)
        return True, ""
    except Exception as exc:
        return False, str(exc)


def _candidate_path(video_dir: Path, relative_path: str) -> Path:
    return video_dir / Path(relative_path)


def _closest_existing_path(video_dir: Path, relative_path: str) -> Optional[Path]:
    wanted = str(Path(relative_path).as_posix())
    all_files = [p for p in video_dir.rglob("*") if p.is_file()]
    if not all_files:
        return None

    exact_name = Path(relative_path).name.lower()
    name_matches = [p for p in all_files if p.name.lower() == exact_name]
    if name_matches:
        return sorted(name_matches)[0]

    relative_names = [str(p.relative_to(video_dir).as_posix()) for p in all_files]
    match = difflib.get_close_matches(wanted, relative_names, n=1, cutoff=0.45)
    if match:
        return video_dir / Path(match[0])
    return None


def resolve_artifact(
    video_dir: Path,
    relative_path: str,
    *,
    required: bool,
) -> Tuple[Optional[Path], Optional[str]]:
    candidate = _candidate_path(video_dir, relative_path)
    if candidate.exists():
        return candidate, None
    fallback = _closest_existing_path(video_dir, relative_path)
    if fallback and fallback.exists():
        note = f"`{relative_path}` not found; using closest existing path `{fallback.relative_to(video_dir).as_posix()}`."
        return fallback, note
    if required:
        raise FileNotFoundError(f"Deliverable layer requires existing artifact: {candidate.as_posix()}")
    return None, f"Optional artifact missing: `{relative_path}`."


def pick_cjk_font() -> Tuple[Optional[FontProperties], str]:
    preferred_tokens = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Source Han Sans CN",
        "PingFang SC",
        "Arial Unicode MS",
        "WenQuanYi Zen Hei",
    ]
    fonts = font_manager.fontManager.ttflist
    for token in preferred_tokens:
        for item in fonts:
            if token.lower() in str(item.name).lower():
                return FontProperties(fname=item.fname), item.name
    return None, "default_matplotlib_font"
