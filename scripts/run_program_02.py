"""CLI wrapper for Program 02: scoring, fusion review, and problem detection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from core.problem_detection import create_annotation_template, run_problem_detection


def _read_config(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    text = Path(path).read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        import yaml  # type: ignore

        payload = yaml.safe_load(text)
        return payload if isinstance(payload, dict) else {}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Program 02: annotation and problem-road detection")
    parser.add_argument("--config", default="configs/default_program_02_config.yaml")
    parser.add_argument("--program_01_output", default=None)
    parser.add_argument("--annotation_csv", default=None)
    parser.add_argument("--coefficient_config", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_percent", type=float, default=None)
    parser.add_argument("--priority_threshold", type=float, default=None)
    parser.add_argument("--max_gap_seconds", type=float, default=None)
    parser.add_argument("--create_annotation_template", action="store_true")
    parser.add_argument("--no_update_visualization_artifacts", action="store_true")
    parser.add_argument("--launch_gui", action="store_true")
    return parser.parse_args(argv)


def _merged_config(cli: argparse.Namespace) -> Dict[str, Any]:
    data = _read_config(cli.config)
    for key, value in vars(cli).items():
        if key in {"config", "launch_gui", "create_annotation_template", "no_update_visualization_artifacts"}:
            continue
        if value not in (None, ""):
            data[key] = value
    if cli.no_update_visualization_artifacts:
        data["update_visualization_artifacts"] = False
    return data


def main(argv: list[str] | None = None) -> int:
    cli = parse_args(argv)
    if cli.launch_gui:
        from apps.program_02_scoring_problem_detection.gui import main as gui_main

        return int(gui_main([]) or 0)
    data = _merged_config(cli)
    video_dir = data.get("program_01_output")
    if not video_dir:
        raise SystemExit("--program_01_output is required")
    if cli.create_annotation_template:
        path = create_annotation_template(video_dir, data.get("annotation_csv") or None)
        print(path)
        return 0
    result = run_problem_detection(
        video_dir=video_dir,
        annotation_csv=data.get("annotation_csv") or None,
        coefficient_config=data.get("coefficient_config") or "configs/street_type_coefficients.yaml",
        output_dir=data.get("output_dir") or None,
        top_k=data.get("top_k"),
        top_percent=data.get("top_percent"),
        priority_threshold=data.get("priority_threshold"),
        max_gap_seconds=float(data.get("max_gap_seconds", 5.0)),
        update_visualization_artifacts=bool(data.get("update_visualization_artifacts", True)),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
