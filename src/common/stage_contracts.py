


"""Contracts for stage outputs and required discovered inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping


STAGE_EXPECTED_OUTPUTS: Dict[str, List[str]] = {
    "segment": [
        "segments/segment_manifest.csv",
        "segments/segment_manifest.json",
    ],
    "visual": [
        "visual/segment_visual_features.csv",
    ],
    "geo_sync": [
        "geo_sync/frame_geo_metadata.csv",
        "geo_sync/segment_geo_metadata.csv",
        "geo_sync/geo_sync_summary.json",
    ],
    "soundscape": [
        "soundscape/audio_segment_features.csv",
        "soundscape/audio_feature_meta.json",
    ],
    "fusion": [
        "fusion/segment_feature_table.csv",
        "fusion/feature_dictionary.json",
        "fusion/feature_quality_report.json",
        "fusion/model_feature_table.csv",
        "fusion/model_feature_dictionary.json",
        "fusion/model_feature_report.json",
    ],
    "agents": [
        "diagnostics/segment_profiles.jsonl",
        "diagnostics/segment_diagnosis.jsonl",
        "diagnostics/segment_critic.jsonl",
    ],
    "design": [
        "design/step8_evidence_registry.json",
        "design/segment_priority_ranking.csv",
        "design/design_plan.jsonl",
        "design/intervention_matrix.csv",
        "design/edit_prompts.jsonl",
        "design/step8_design_summary.md",
    ],
    "deliverable": [
        "deliverable/problem_episodes.csv",
        "deliverable/problem_episode_summary.csv",
        "deliverable/final_problem_segments_table.csv",
    ],
    "gis_export": [
        "gis/frame_gis_export.csv",
        "gis/segment_gis_export.csv",
        "gis/problem_episode_gis_export.csv",
        "gis/gis_export_summary.json",
    ],
    "web_sync": [
        "web/sync_map_data.json",
    ],
}



STAGE_REQUIRED_INPUTS: Dict[str, List[str]] = {
    "segment": [],
    "visual": ["ai_evaluation", "stage:segment"],
    "geo_sync": ["stage:segment"],
    "soundscape": ["audio_events", "stage:segment"],


    "fusion": ["stage:segment"],
    "agents": ["stage:segment", "stage:soundscape", "stage:fusion"],
    "design": ["validation", "stage:agents"],
    "deliverable": ["stage:design"],
    "gis_export": ["stage:segment", "stage:geo_sync"],
    "web_sync": ["stage:segment", "stage:geo_sync"],
}

DISCOVERY_KEYS: Dict[str, str] = {
    "frames": "frames",
    "stats_visual_elements": "stats/visual_elements",
    "stats_green_view": "stats/green_view",
    "stats_emotion": "stats/emotion",
    "stats_people_count": "stats/people_count",
    "stats_color_analysis": "stats/color_analysis",
    "ai_evaluation": "ai_evaluation",
    "validation": "validation",
    "audio_events": "audio_events",
    "reproj": "reproj",
    "split": "split",
}


def expected_stage_paths(video_dir: str, stage_name: str) -> List[Path]:
    """Return absolute expected output paths for the given stage."""
    base = Path(video_dir)
    rels = STAGE_EXPECTED_OUTPUTS.get(stage_name, [])
    return [base / rel for rel in rels]


def stage_contract_as_posix(video_dir: str) -> Mapping[str, List[str]]:
    """Return expected output map with absolute POSIX-like paths."""
    out: Dict[str, List[str]] = {}
    for stage, rels in STAGE_EXPECTED_OUTPUTS.items():
        out[stage] = [(Path(video_dir) / r).as_posix() for r in rels]
    return out
