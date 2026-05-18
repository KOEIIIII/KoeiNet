


"""Lightweight integrity checks for stage output files."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Mapping

from .stage_contracts import STAGE_EXPECTED_OUTPUTS, expected_stage_paths


def _validate_csv(path: Path) -> Dict[str, object]:
    row_count = 0
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for _ in reader:
            row_count += 1

    valid = row_count >= 2
    return {"valid": valid, "reason": f"csv_rows={row_count}"}


def _validate_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return {"valid": len(data) > 0, "reason": f"json_list_items={len(data)}"}
    if isinstance(data, dict):
        return {"valid": len(data) > 0, "reason": f"json_dict_keys={len(data)}"}
    return {"valid": False, "reason": f"json_type={type(data).__name__}"}


def _validate_jsonl(path: Path) -> Dict[str, object]:
    item_count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            json.loads(raw)
            item_count += 1
    return {"valid": item_count > 0, "reason": f"jsonl_items={item_count}"}


def _validate_text(path: Path) -> Dict[str, object]:
    txt = path.read_text(encoding="utf-8").strip()
    return {"valid": len(txt) > 0, "reason": f"text_chars={len(txt)}"}


def validate_file_integrity(path: Path) -> Dict[str, object]:
    """
    Validate one output file:
    - exists
    - size > 0
    - parsable for csv/json/jsonl
    - has at least one row/item when applicable
    """
    if not path.exists():
        return {"path": path.as_posix(), "valid": False, "reason": "missing"}
    if path.stat().st_size <= 0:
        return {"path": path.as_posix(), "valid": False, "reason": "size_zero"}

    try:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            info = _validate_csv(path)
        elif suffix == ".json":
            info = _validate_json(path)
        elif suffix == ".jsonl":
            info = _validate_jsonl(path)
        else:
            info = _validate_text(path)
        return {"path": path.as_posix(), **info}
    except Exception as exc:
        return {
            "path": path.as_posix(),
            "valid": False,
            "reason": f"parse_error:{exc}",
        }


def validate_stage_outputs(video_dir: str, stage_name: str) -> Dict[str, object]:
    """Validate all expected outputs for one stage."""
    files = expected_stage_paths(video_dir, stage_name)
    checks = [validate_file_integrity(p) for p in files]
    valid = bool(checks) and all(bool(c.get("valid")) for c in checks)
    return {
        "stage": stage_name,
        "valid": valid,
        "checks": checks,
    }


def stage_validity_map(video_dir: str) -> Dict[str, bool]:
    """Return validity map for all stage output contracts."""
    out: Dict[str, bool] = {}
    for stage in STAGE_EXPECTED_OUTPUTS:
        out[stage] = bool(validate_stage_outputs(video_dir, stage)["valid"])
    return out


def stage_checks_map(video_dir: str) -> Mapping[str, List[Dict[str, object]]]:
    """Return detailed per-stage file checks."""
    out: Dict[str, List[Dict[str, object]]] = {}
    for stage in STAGE_EXPECTED_OUTPUTS:
        detail = validate_stage_outputs(video_dir, stage)
        out[stage] = list(detail["checks"])
    return out

