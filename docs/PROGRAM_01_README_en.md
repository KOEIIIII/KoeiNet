# Program 01: Base Analysis and Spatial Visualization

Program 01 is the first-stage application. It turns panoramic video and GPS/trajectory data into segment-level analysis outputs, spatial alignment files, GIS exports, and local visualization artifacts. It wraps the existing `main.py` pipeline without replacing the core algorithms.

## Scope

- Read panoramic video.
- Extract analysis frames.
- Run the existing cubemap/panoramic processing flow.
- Generate basic visual metrics and segment-level visual summaries.
- Read audio and generate basic audio-event outputs; optional soundscape extensions depend on local dependencies.
- Build `segment_manifest.csv`.
- Align video frames and segments to GPS trajectories.
- Export GIS tables and web-sync visualization files.

## Launch

Packaged local application:

```text
dist/Program01_AnalysisVisualization/Program01_AnalysisVisualization.exe
```

Development GUI:

```powershell
python scripts\run_program_01.py --launch_gui
```

Development CLI:

```powershell
python scripts\run_program_01.py --input_video examples\sample_inputs\VID_20250625_101458_00_006.mp4 --gps_file examples\sample_inputs\sample_gps.csv --output_dir output --frame_skip 20
```

Continue from an existing output:

```powershell
python scripts\run_program_01.py --from_existing_output output\VID_20250625_101458_00_006 --post_only --gps_file examples\sample_inputs\sample_gps.csv
```

## Input Requirements

The video must be readable by OpenCV/ffmpeg. If GPS alignment is enabled, the video needs a resolvable start time from filename, metadata, or sidecar. Filenames such as `VID_20250625_101458_00_006.mp4` are supported by the current parser.

The GPS CSV must include at least:

```text
groupTime,gps_longitude,gps_latitude
```

`groupTime` is a Unix timestamp in seconds. Coordinate semantics follow the existing `geo_sync` module.

## Parameters

- `input_video`: single video file.
- `gps_file`: GPS CSV.
- `output_dir`: output root.
- `frame_skip`: frame sampling interval.
- `segment_seconds`: segment window length.
- `segment_overlap`: overlap between adjacent segment windows.
- `gps_time_offset_seconds`: offset between video time and GPS time.
- `from_existing_output`: continue from an existing Program 01 output.
- `post_only`: run only post-processing stages.
- `resume_missing_only`: fill missing artifacts only.
- `enable_segment_pipeline`: enable segment processing.
- `enable_visual_segment_summary`: enable segment-level visual summaries.
- `enable_soundscape`: enable soundscape extension.
- `enable_geo_sync`: enable trajectory alignment.
- `enable_gis_export`: enable GIS export.
- `enable_web_sync_export`: enable web-sync export.

The GUI advanced toggles are wired to these CLI parameters.

## Outputs

A typical output folder is `output/<video_name>/` and may contain:

- `frames/`: extracted frames.
- `split/`, `mask/`, `overlay/`, `reproj/`: existing visual artifacts.
- `stats/`: statistics and charts.
- `audio_events/`: basic audio-event outputs.
- `segments/segment_manifest.csv`: segment manifest.
- `visual/segment_visual_features.csv`: segment-level visual features.
- `geo_sync/frame_geo_metadata.csv`: frame-level GPS alignment.
- `geo_sync/segment_geo_metadata.csv`: segment-level GPS alignment.
- `gis/segment_gis_export.csv`: segment GIS table.
- `web_sync/`, `web/`: local visualization and synchronization artifacts.

## GUI Workflow

1. Select Video.
2. Select GPS File.
3. Select Output Folder.
4. Set Frame Skip, Segment Length, Segment Overlap, and GPS Time Offset.
5. Enable or disable stages in Advanced Settings.
6. Click Start Analysis.
7. Use Open Output Folder or Open Visualization after completion.

The interface defaults to English and can be switched to Chinese.

## Troubleshooting

- GPS schema error: use `groupTime,gps_longitude,gps_latitude`.
- Start-time resolution error: check filename timestamp, metadata, or sidecar.
- Missing model or ffmpeg: confirm local external resources are present.
- Slow large-video processing: increase `frame_skip` or validate post-processing first with `post_only`.
