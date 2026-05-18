


"""Runner for the final deliverable layer built on top of Step-8 outputs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from .card_renderer import export_contact_sheet_pdf, render_problem_episode_cards
from .episode_builder import DEFAULT_TOP_K, build_problem_episodes, select_priority_segments
from .episode_evidence import EpisodeEvidenceAssembler
from .export_utils import (
    dataframe_for_csv,
    read_json,
    read_jsonl,
    resolve_artifact,
    safe_relative_path,
    to_json_text,
    write_jsonl,
    write_optional_excel,
)
from .html_report import (
    build_deliverable_onepage_markdown,
    build_deliverable_summary_markdown,
    build_shadow_eval_notes,
    write_problem_episode_html,
)
from .issue_summarizer import build_episode_summaries
from .prompt_translator import build_episode_prompts

logger = logging.getLogger("deliverable.runner")


def _resolve_paths(video_dir: str) -> Tuple[Dict[str, Any], List[str]]:
    vdir = Path(video_dir)
    notes: List[str] = []
    paths: Dict[str, Any] = {
        "video_dir": vdir,
        "deliverable_dir": vdir / "deliverable",
        "frames_dir": vdir / "frames",
    }
    if not paths["frames_dir"].is_dir():
        raise FileNotFoundError(f"Deliverable layer requires frames directory: {paths['frames_dir'].as_posix()}")

    expected = {
        "segment_manifest_csv": ("segments/segment_manifest.csv", True),
        "audio_segment_features_csv": ("soundscape/audio_segment_features.csv", True),
        "model_feature_table_csv": ("fusion/model_feature_table.csv", True),
        "model_feature_dictionary_json": ("fusion/model_feature_dictionary.json", True),
        "validation_csv": ("validation/final_annotation_labels_adjudicated.csv", True),
        "segment_profiles_jsonl": ("diagnostics/segment_profiles.jsonl", True),
        "segment_diagnosis_jsonl": ("diagnostics/segment_diagnosis.jsonl", True),
        "segment_critic_reviews_jsonl": ("diagnostics/segment_critic_reviews.jsonl", False),
        "segment_priority_ranking_csv": ("design/segment_priority_ranking.csv", True),
        "design_plan_jsonl": ("design/design_plan.jsonl", True),
        "intervention_matrix_csv": ("design/intervention_matrix.csv", True),
        "edit_prompts_jsonl": ("design/edit_prompts.jsonl", True),
        "step8_design_summary_md": ("design/step8_design_summary.md", True),
        "step8_evidence_registry_json": ("design/step8_evidence_registry.json", True),
        "proof_claim_registry_csv": ("proof/proof_claim_registry.csv", False),
        "proof_summary_md": ("proof/proof_summary.md", False),
        "relationship_summary_md": ("relationship/relationship_summary.md", False),
        "group_confirmatory_summary_md": ("relationship/group_confirmatory/group_confirmatory_summary.md", False),
        "step75_summary_md": ("fusion_eval_refined/step75_summary.md", False),
    }
    for key, (relative_path, required) in expected.items():
        actual, note = resolve_artifact(vdir, relative_path, required=required)
        paths[key] = actual
        if note:
            notes.append(note)
    for key, relative_path in {
        "segment_profile_summary_csv": "diagnostics/segment_profile_summary.csv",
        "segment_diagnosis_summary_csv": "diagnostics/segment_diagnosis_summary.csv",
    }.items():
        exact_path = Path(video_dir) / relative_path
        if exact_path.exists():
            paths[key] = exact_path
        else:
            paths[key] = None
            notes.append(f"Optional artifact missing: `{relative_path}`.")
    return paths, notes


def _jsonl_map(path: Optional[Path]) -> Dict[int, Dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    rows = read_jsonl(path)
    out: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        segment_id = pd.to_numeric(row.get("segment_id"), errors="coerce")
        if pd.notna(segment_id):
            out[int(segment_id)] = row
    return out


def _write_table_and_jsonl(df: pd.DataFrame, rows: Sequence[Mapping[str, Any]], csv_path: Path, jsonl_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe_for_csv(df).to_csv(csv_path, index=False, encoding="utf-8")
    write_jsonl(jsonl_path, rows)


def _merge_outputs(
    episodes_df: pd.DataFrame,
    evidence_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    prompts_df: pd.DataFrame,
) -> pd.DataFrame:
    merged = episodes_df.merge(evidence_df, on="episode_id", how="left", suffixes=("", "_evidence"))
    merged = merged.merge(summary_df, on="episode_id", how="left")
    merged = merged.merge(prompts_df, on="episode_id", how="left", suffixes=("", "_prompt"))
    for column in ["structured_prompt_input_json", "edit_prompt", "negative_prompt", "short_caption", "prompt_mode"]:
        prompt_col = f"{column}_prompt"
        if prompt_col in merged.columns:
            merged[column] = merged[prompt_col].where(merged[prompt_col].notna(), merged.get(column))
            merged = merged.drop(columns=[prompt_col])
    return merged


def run_deliverable_layer(
    *,
    video_dir: str,
    deliverable_top_k: Optional[int] = DEFAULT_TOP_K,
    deliverable_top_percent: Optional[float] = None,
    deliverable_priority_threshold: Optional[float] = None,
    deliverable_max_gap_seconds: float = 5.0,
    deliverable_export_cards: bool = False,
    deliverable_use_glm: bool = False,
    deliverable_render_html: bool = False,
    deliverable_render_pdf: bool = False,
    progress_callback: Optional[Any] = None,
) -> Dict[str, str]:
    paths, mapping_notes = _resolve_paths(video_dir=video_dir)
    if progress_callback:
        progress_callback(0, 6, "deliverable | load inputs")
    deliverable_dir = Path(paths["deliverable_dir"])
    deliverable_dir.mkdir(parents=True, exist_ok=True)

    manifest_df = pd.read_csv(paths["segment_manifest_csv"])
    ranking_df = pd.read_csv(paths["segment_priority_ranking_csv"])
    model_df = pd.read_csv(paths["model_feature_table_csv"])
    audio_df = pd.read_csv(paths["audio_segment_features_csv"])
    validation_df = pd.read_csv(paths["validation_csv"]) if paths.get("validation_csv") else pd.DataFrame()
    intervention_df = pd.read_csv(paths["intervention_matrix_csv"])
    step8_evidence_registry = read_json(paths["step8_evidence_registry_json"]) if paths.get("step8_evidence_registry_json") else {}
    proof_claim_df = pd.read_csv(paths["proof_claim_registry_csv"]) if paths.get("proof_claim_registry_csv") else pd.DataFrame()

    design_plan_map = _jsonl_map(paths.get("design_plan_jsonl"))
    edit_prompt_map = _jsonl_map(paths.get("edit_prompts_jsonl"))
    profile_map = _jsonl_map(paths.get("segment_profiles_jsonl"))
    diagnosis_map = _jsonl_map(paths.get("segment_diagnosis_jsonl"))
    critic_map = _jsonl_map(paths.get("segment_critic_reviews_jsonl"))

    selected_df, selection_meta = select_priority_segments(
        ranking_df,
        top_k=deliverable_top_k,
        top_percent=deliverable_top_percent,
        priority_threshold=deliverable_priority_threshold,
    )
    episodes_df, episodes_jsonl = build_problem_episodes(
        selected_df,
        manifest_df,
        max_gap_seconds=float(deliverable_max_gap_seconds),
    )
    if progress_callback:
        progress_callback(2, 6, "deliverable | build episodes")
    if episodes_df.empty:
        raise RuntimeError("Deliverable layer could not build any problem episodes from the selected segments.")

    evidence_assembler = EpisodeEvidenceAssembler(
        ranking_df=ranking_df,
        model_df=model_df,
        audio_df=audio_df,
        validation_df=validation_df,
        design_plan_map=design_plan_map,
        edit_prompt_map=edit_prompt_map,
        intervention_df=intervention_df,
        profile_map=profile_map,
        diagnosis_map=diagnosis_map,
        critic_map=critic_map,
        step8_evidence_registry=step8_evidence_registry,
        proof_claim_df=proof_claim_df,
    )
    evidence_df, evidence_jsonl = evidence_assembler.build(episodes_df)
    summary_df, summary_jsonl = build_episode_summaries(evidence_df)
    if progress_callback:
        progress_callback(4, 6, "deliverable | evidence and summary")
    prompts_df, prompts_jsonl, prompt_mode = build_episode_prompts(
        evidence_df,
        summary_df,
        use_glm=bool(deliverable_use_glm),
    )
    if progress_callback:
        progress_callback(5, 6, "deliverable | prompts and exports")
    final_df = _merge_outputs(episodes_df, evidence_df, summary_df, prompts_df)
    final_df["problem_summary"] = final_df["one_sentence_summary"]
    final_df = final_df.sort_values(["priority_rank", "start_time_sec"]).reset_index(drop=True)

    evidence_export_df = evidence_df.merge(prompts_df, on="episode_id", how="left", suffixes=("", "_prompt"))
    for column in ["structured_prompt_input_json", "edit_prompt", "negative_prompt", "short_caption", "prompt_mode"]:
        prompt_col = f"{column}_prompt"
        if prompt_col in evidence_export_df.columns:
            evidence_export_df[column] = evidence_export_df[prompt_col].where(evidence_export_df[prompt_col].notna(), evidence_export_df.get(column))
            evidence_export_df = evidence_export_df.drop(columns=[prompt_col])

    _write_table_and_jsonl(
        episodes_df,
        episodes_jsonl,
        deliverable_dir / "problem_episodes.csv",
        deliverable_dir / "problem_episodes.jsonl",
    )
    _write_table_and_jsonl(
        evidence_export_df,
        evidence_export_df.to_dict(orient="records"),
        deliverable_dir / "problem_episode_evidence.csv",
        deliverable_dir / "problem_episode_evidence.jsonl",
    )
    _write_table_and_jsonl(
        summary_df,
        summary_jsonl,
        deliverable_dir / "problem_episode_summary.csv",
        deliverable_dir / "problem_episode_summary.jsonl",
    )
    _write_table_and_jsonl(
        prompts_df,
        prompts_jsonl,
        deliverable_dir / "problem_episode_prompts.csv",
        deliverable_dir / "problem_episode_prompts.jsonl",
    )

    final_table = final_df[
        [
            "episode_id",
            "start_time_sec",
            "end_time_sec",
            "representative_segment_id",
            "representative_frame_indices",
            "representative_frame_paths",
            "problem_summary",
            "soundscape_problem",
            "visual_problem",
            "fused_problem",
            "intervention_theme",
            "edit_prompt",
            "negative_prompt",
            "short_caption",
            "priority_rank",
            "priority_score",
        ]
    ].copy()
    dataframe_for_csv(final_table).to_csv(deliverable_dir / "final_problem_segments_table.csv", index=False, encoding="utf-8")
    xlsx_ok, xlsx_error = write_optional_excel(final_table, deliverable_dir / "final_problem_segments_table.xlsx")

    render_cards = bool(deliverable_export_cards or deliverable_render_html or deliverable_render_pdf)
    card_records: List[Dict[str, str]] = []
    font_name = "cards_not_rendered"
    if render_cards:
        card_records, font_name = render_problem_episode_cards(
            final_df,
            out_dir=deliverable_dir / "problem_episode_cards",
        )
    html_path = ""
    if bool(deliverable_render_html):
        html_path = write_problem_episode_html(
            final_df,
            deliverable_dir=deliverable_dir,
            card_dir=deliverable_dir / "problem_episode_cards",
            out_path=deliverable_dir / "problem_episode_cards.html",
        )
    pdf_path = ""
    if bool(deliverable_render_pdf):
        pdf_path = export_contact_sheet_pdf(
            card_records,
            out_path=deliverable_dir / "problem_episode_contact_sheet.pdf",
        ) or ""

    shadow_eval_path = deliverable_dir / "shadow_eval_notes.md"
    shadow_eval_path.write_text(build_shadow_eval_notes(), encoding="utf-8")

    rendered_outputs = {
        "problem_episodes_csv": (deliverable_dir / "problem_episodes.csv").as_posix(),
        "problem_episode_evidence_csv": (deliverable_dir / "problem_episode_evidence.csv").as_posix(),
        "problem_episode_summary_csv": (deliverable_dir / "problem_episode_summary.csv").as_posix(),
        "problem_episode_prompts_csv": (deliverable_dir / "problem_episode_prompts.csv").as_posix(),
        "final_problem_segments_table_csv": (deliverable_dir / "final_problem_segments_table.csv").as_posix(),
        "final_problem_segments_table_xlsx": (deliverable_dir / "final_problem_segments_table.xlsx").as_posix() if xlsx_ok else "",
        "problem_episode_cards_html": html_path,
        "problem_episode_contact_sheet_pdf": pdf_path,
        "problem_episode_cards_dir": (deliverable_dir / "problem_episode_cards").as_posix() if render_cards else "",
        "shadow_eval_notes_md": shadow_eval_path.as_posix(),
    }

    if not xlsx_ok and xlsx_error:
        mapping_notes.append(f"XLSX export skipped: {xlsx_error}")
    if bool(deliverable_use_glm) and prompt_mode != "glm_refined":
        mapping_notes.append("GLM prompt refinement was requested but unavailable; fell back to deterministic template mode.")

    resolved_paths_for_summary = {
        key: value.as_posix()
        for key, value in paths.items()
        if isinstance(value, Path)
    }
    summary_md = build_deliverable_summary_markdown(
        resolved_paths=resolved_paths_for_summary,
        mapping_notes=mapping_notes,
        episodes_df=episodes_df,
        final_df=final_df,
        prompt_mode=prompt_mode,
        selection_mode=str(selection_meta.get("selection_mode", "")),
        font_name=font_name,
        rendered_outputs=rendered_outputs,
    )
    summary_path = deliverable_dir / "deliverable_summary.md"
    summary_path.write_text(summary_md, encoding="utf-8")

    best_view_file = html_path or pdf_path or (deliverable_dir / "final_problem_segments_table.csv").as_posix()
    onepage_md = build_deliverable_onepage_markdown(
        final_df,
        best_view_file=best_view_file,
    )
    onepage_path = deliverable_dir / "deliverable_onepage_report.md"
    onepage_path.write_text(onepage_md, encoding="utf-8")
    if progress_callback:
        progress_callback(6, 6, "deliverable | write outputs")

    logger.info("deliverable layer done | out=%s episodes=%s", deliverable_dir.as_posix(), len(episodes_df))
    return {
        "deliverable_dir": deliverable_dir.as_posix(),
        "problem_episodes_csv": (deliverable_dir / "problem_episodes.csv").as_posix(),
        "problem_episode_evidence_csv": (deliverable_dir / "problem_episode_evidence.csv").as_posix(),
        "problem_episode_summary_csv": (deliverable_dir / "problem_episode_summary.csv").as_posix(),
        "problem_episode_prompts_csv": (deliverable_dir / "problem_episode_prompts.csv").as_posix(),
        "final_problem_segments_table_csv": (deliverable_dir / "final_problem_segments_table.csv").as_posix(),
        "final_problem_segments_table_xlsx": (deliverable_dir / "final_problem_segments_table.xlsx").as_posix() if xlsx_ok else "",
        "problem_episode_cards_dir": (deliverable_dir / "problem_episode_cards").as_posix() if render_cards else "",
        "problem_episode_cards_html": html_path,
        "problem_episode_contact_sheet_pdf": pdf_path,
        "deliverable_summary_md": summary_path.as_posix(),
        "deliverable_onepage_report_md": onepage_path.as_posix(),
        "shadow_eval_notes_md": shadow_eval_path.as_posix(),
        "prompt_mode": prompt_mode,
        "card_font_name": font_name,
    }
