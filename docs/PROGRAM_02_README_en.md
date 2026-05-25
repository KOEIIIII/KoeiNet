# Program 02: Scoring, Fusion Review, and Problem Detection

Program 02 is the second-stage application. It reads Program 01 outputs and supports human annotation, street-type coefficient configuration, problem-segment scoring, and continuous problem-episode merging. It does not regenerate Program 01 base analysis results.

## Scope

- Load segment-level outputs from Program 01.
- Create or continue editing an annotation CSV.
- Browse and edit scoring fields by `segment_id`.
- Load, edit, save, and restore street-type coefficient configurations.
- Calculate `priority_score` from annotations, multimodal features, and coefficients.
- Reuse the existing continuous segment-merging logic to generate problem episodes.
- Export problem tables, summary reports, and visualization-compatible artifacts.

## Launch

Packaged local application:

```text
dist/Program02_ScoringProblemDetection/Program02_ScoringProblemDetection.exe
```

Development GUI:

```powershell
python scripts\run_program_02.py --launch_gui
```

Development CLI:

```powershell
python scripts\run_program_02.py --program_01_output examples\sample_outputs\minimal_program01_output --top_k 2 --priority_threshold 0.4
```

Create an annotation template:

```powershell
python scripts\run_program_02.py --program_01_output examples\sample_outputs\minimal_program01_output --create_annotation_template
```

## Recommended Workflow

Program 02 is normally used after manual scoring:

1. Select Program 01 Output.
2. Click Create Annotation File if no annotation CSV exists.
3. Fill or edit the Annotation table.
4. Click Save Annotation.
5. Load or modify coefficients in the Coefficients tab.
6. Set Top K, Priority Threshold, and Max Gap Seconds.
7. Click Run Problem Detection.
8. Review the files in `problem_detection/`.

If an annotation CSV already exists, load it directly, review or adjust values, and run detection.

## Annotation Fields

- `segment_id`: segment identifier; normally not edited.
- `street_type`: street type used to select coefficient settings.
- `comfort_score`: comfort score, recommended 1-5.
- `vitality_score`: vitality score, recommended 1-5.
- `soundscape_pleasantness`: soundscape pleasantness, recommended 1-5.
- `soundscape_eventfulness`: soundscape eventfulness, recommended 1-5.
- `overall_problem_severity`: overall problem severity, recommended 1-5.
- `main_problem_labels`: multiple labels may be separated by semicolons.
- `primary_problem_label`: primary issue label.
- `confidence_score`: annotation confidence, recommended 1-5.
- `annotator_notes`: reviewer notes.

## Coefficients

Default configuration:

```text
configs/street_type_coefficients.yaml
```

The file is marked as `default configuration`. Each `street_type` may define:

- `severity_threshold`: score threshold for problem segments.
- `desired_eventfulness`: expected eventfulness range.
- `coefficients`: weights for score components.

The GUI can load, edit, save, and restore the default configuration. Most users can start with the default configuration; research review can adjust weights by street type.

## Detection Logic

Program 02 merges:

- `segments/segment_manifest.csv`
- annotation CSV
- `fusion/segment_feature_table.csv`
- `visual/segment_visual_features.csv`
- `soundscape/audio_segment_features.csv`
- `geo_sync/segment_geo_metadata.csv`

Each segment receives `priority_score`, `priority_level`, and `is_problem_segment`. The application then calls the existing `src.deliverable.episode_builder` logic to merge adjacent selected segments into problem episodes.

## Outputs

Default output folder:

```text
<Program 01 output>/problem_detection/
```

Generated files:

- `segment_problem_priority.csv`
- `problem_episodes.csv`
- `problem_detection_summary.md`
- `problem_detection_run.json`

When Update visualization-compatible artifacts is enabled, the application also writes:

- `design/segment_priority_ranking.csv`
- `deliverable/problem_episodes.csv`
- `deliverable/problem_episode_summary.csv`

Existing compatible files are backed up before overwrite.

## Troubleshooting

- Missing `segment_manifest.csv`: run Program 01 first or select the correct Program 01 output folder.
- Annotation CSV cannot be read: keep `segment_id` and the standard annotation fields.
- Coefficient file cannot be parsed: keep it JSON-compatible YAML or install PyYAML.
- No problem episodes are generated: review thresholds, Top K, scores, and `main_problem_labels`.
