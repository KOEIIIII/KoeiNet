# KoeiNet

**KoeiNet is a panoramic-video-based multimodal street-space evaluation system for historic districts.**

KoeiNet supports street-space analysis, spatial anchoring, multimodal evidence organization, manual annotation, and problem-segment identification for historic district environments. It is designed as a research-to-application workflow: raw panoramic video and trajectory data are transformed into segment-level evidence, mapped outputs, annotation tables, and problem-road episodes that can be reviewed and exported.

## Research Orientation

Historic districts often require street-space evaluation that combines visual conditions, activity patterns, soundscape cues, spatial location, and expert judgment. KoeiNet organizes these signals into a reproducible workflow. The system does not replace professional planning interpretation; it provides structured evidence and traceable outputs for review, comparison, and reporting.

## Core Workflow

1. Panoramic video input.
2. GPS / trajectory input.
3. Basic visual and spatial analysis.
4. Segment-level evidence organization.
5. GIS anchoring.
6. Web-based visualization.
7. Manual scoring and adjudication.
8. Street-type-specific coefficient configuration.
9. Multimodal fusion and review.
10. Problem segment / problem episode detection.

## Application Modules

### Program 01: Basic Data Analysis and Spatial Visualization

Program 01 is the first-stage application. It reads panoramic video and GPS data, extracts analysis frames, organizes segment-level results, aligns segments to GPS trajectories, exports GIS-compatible files, and generates local visualization artifacts.

Typical outputs include:

- Segment-level analysis results.
- GPS-aligned spatial outputs.
- GIS export tables.
- Local web visualization files.
- Runtime logs and stage summaries.

### Program 02: Manual Scoring, Multimodal Fusion, and Problem-Segment Detection

Program 02 is the second-stage application. It loads Program 01 outputs, creates or edits manual annotation CSV files, supports street-type coefficient configuration, calculates problem priority scores, and merges adjacent problem segments into continuous problem episodes.

Typical outputs include:

- Manual annotation CSV.
- Coefficient configuration files.
- Segment priority table.
- Problem segment results.
- Problem episode results.
- Summary report.

## How To Use

1. Launch Program 01.
2. Select a panoramic video and GPS file.
3. Configure analysis parameters such as frame skip, segment length, overlap, and GPS offset.
4. Run analysis and generate spatial visualization results.
5. Launch Program 02.
6. Load the Program 01 output folder.
7. Complete or simulate manual scoring.
8. Configure street-type coefficients.
9. Run problem-segment detection.
10. Export and review the results.

## Key Outputs

- `segments/segment_manifest.csv`
- `visual/segment_visual_features.csv`
- `geo_sync/segment_geo_metadata.csv`
- `gis/segment_gis_export.csv`
- `web/` or `web_sync/` visualization artifacts
- `validation/final_annotation_labels_adjudicated.csv`
- `problem_detection/segment_problem_priority.csv`
- `problem_detection/problem_episodes.csv`
- `problem_detection/problem_detection_summary.md`

## Example Data

The repository includes minimal public samples for workflow validation:

- A tiny timestamped video sample.
- A minimal GPS CSV in the current `geo_sync` schema.
- A minimal Program 01 output folder for Program 02 testing.

The samples are not intended for accuracy evaluation.

## Notes And Limitations

- Full video processing depends on local model resources, ffmpeg, and suitable hardware.
- Raw panoramic videos and GPS trajectories may contain sensitive location or personal information and should not be committed to the public repository.
- Packaged desktop applications are large and should be published through GitHub Releases or a separate delivery archive rather than committed to Git.
- The system provides structured evidence and problem detection outputs; final interpretation should remain under expert review.
