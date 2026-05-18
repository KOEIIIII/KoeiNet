#!/usr/bin/env markdown

# visual segment features

`src.visual.segment_features` builds a manifest-aligned visual segment table
without redefining the segment system.

## Purpose

This module exists so the project has a stable segment-level visual table that
can be merged directly with:

- `geo_sync/segment_geo_metadata.csv`
- `soundscape/audio_segment_features.csv`
- `fusion/segment_feature_table.csv`

The output file is:

- `output/<video>/visual/segment_visual_features.csv`

## Inputs

The builder currently reuses existing project outputs instead of rerunning the
full panoramic segmentation stack:

- `segments/segment_manifest.csv`
- `frames/frame_*.jpg`
- `ai_evaluation/activity_scores.csv`

## Alignment rule

- `segment_manifest.csv` is the only segment anchor.
- `segment_id`, `start_time_sec`, `end_time_sec`, and `center_time_sec` are
  copied from the real manifest.
- Frame membership is derived from `included_frame_paths` /
  `included_frame_indices` in the manifest.

## Aggregation rule

Current segment-level fields are built as follows:

- low-cost image statistics are computed per frame from the saved frame JPGs
- numeric visual metrics are aggregated by segment with `mean` and `std`
- AI activity scores from `activity_scores.csv` are aligned by `frame_num` and
  aggregated by segment with `mean`
- `ai_activity_major_label` is the `argmax` over the six activity score means
- `ai_activity_suitable_labels` keeps labels whose segment mean is `>= 3.0`

## Current output contract

The output always contains:

- `segment_id`
- `start_time_sec`
- `end_time_sec`
- `center_time_sec`
- `included_frame_count`

And representative visual features such as:

- `vis_brightness_mean_mean`
- `vis_saturation_mean_mean`
- `vis_green_pixel_ratio_mean`
- `vis_sky_blue_ratio_top_mean`
- `vis_edge_density_mean`
- `vis_gray_entropy_mean`
- `ai_activity_*_score_mean`

## CLI

```bash
python -m src.visual.segment_features --video-dir output/<video_name>
```
