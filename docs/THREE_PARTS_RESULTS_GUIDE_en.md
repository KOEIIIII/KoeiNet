# How to interpret results — Emotional / Soundscape / Visual

This document explains, step-by-step, how the repository's Program 01 and Program 02 outputs produce the three result families that appear in the deliverables: Emotional (subjective annotation), Soundscape (audio features & events), and Visual (image-derived features). It tells users which files to look for, how to generate the outputs, and where to find example scripts for quick plotting and inspection.

---

## Quick summary

- Emotional: from Program 02 human annotations (annotation CSV). Key fields: `comfort_score`, `vitality_score`, `soundscape_pleasantness`, `soundscape_eventfulness`, `overall_problem_severity`, `confidence_score`.
- Soundscape: audio segment features produced by Program 01 (optional extension). Usually in `soundscape/audio_segment_features.csv` or `audio_events/` when `enable_soundscape` is used.
- Visual: visual segment features produced by Program 01, stored in `visual/segment_visual_features.csv` (and frames/ or overlay/ for visual artifacts).

## Required inputs

1. Panoramic video file (readable by ffmpeg/OpenCV).
2. GPS/trajectory CSV with `groupTime,gps_longitude,gps_latitude` (Unix seconds `groupTime`).
3. Program 01 output folder (run Program 01 first), example run:

```powershell
python scripts\run_program_01.py --input_video examples\sample_inputs\VID_...mp4 --gps_file examples\sample_inputs\sample_gps.csv --output_dir output --frame_skip 20
```

4. Enable soundscape features during Program 01 if needed: use `--enable_soundscape` or the GUI toggle.
5. Program 02 for annotation & fusion:

```powershell
python scripts\run_program_02.py --launch_gui
# or to prepare an annotation template
python scripts\run_program_02.py --program_01_output output\VID_... --create_annotation_template
```

## Where to find generated files

Typical Program 01 output path: `output/<video_name>/`

Important files (segment-level):

- `segments/segment_manifest.csv` — canonical segment list and identifiers.
- `visual/segment_visual_features.csv` — visual features per segment.
- `soundscape/audio_segment_features.csv` — soundscape/audio features per segment (if enabled).
- `geo_sync/segment_geo_metadata.csv` — GPS alignment and `matched_gps_*` fields for mapping.
- `web/` or `web_sync/` — local web visualization artifacts (optional).

Program 02 outputs (after annotation & detection):

- `problem_detection/segment_problem_priority.csv`
- `problem_detection/problem_episodes.csv`
- `validation/final_annotation_labels_adjudicated.csv` (or your saved annotation CSV)

## Step-by-step: obtain the three result families

1. Run Program 01 to create base outputs (visual, geo_sync, segments, optional soundscape). Enable `enable_soundscape` if you need audio features.
2. For subjective/emotional labels, run Program 02 (create annotation template if needed) and fill the annotation CSV.
3. Merge tables by `segment_id` to build a per-segment fused dataset for analysis:

   - `segments/segment_manifest.csv`
   - `visual/segment_visual_features.csv`
   - `soundscape/audio_segment_features.csv` (if present)
   - `geo_sync/segment_geo_metadata.csv`
   - annotation CSV

   Example merge (Python/pandas):

```python
import pandas as pd
segments = pd.read_csv('output/VID/.../segments/segment_manifest.csv')
visual = pd.read_csv('output/VID/.../visual/segment_visual_features.csv')
sound = pd.read_csv('output/VID/.../soundscape/audio_segment_features.csv')
geo = pd.read_csv('output/VID/.../geo_sync/segment_geo_metadata.csv')
anno = pd.read_csv('output/VID/.../validation/final_annotation_labels_adjudicated.csv')

df = segments.merge(visual, on='segment_id', how='left').merge(sound, on='segment_id', how='left').merge(geo[['segment_id','matched_gps_longitude_gcj02','matched_gps_latitude_gcj02']], on='segment_id', how='left').merge(anno, on='segment_id', how='left')
```

4. Basic checks: ensure `segment_id` alignment across files; convert annotation fields to numeric; drop or flag low `confidence_score` rows if desired.

## Recommended plots and quick explanations

Emotional
- Time series of `comfort_score` / `vitality_score` (detect low-score segments).
- Boxplot by `street_type` or annotator (compare distributions).

Soundscape
- Leq / RMS / energy time series per segment (identify noise hotspots).
- Event-type stacked counts or event frequency per segment.
- Scatter: subjective `soundscape_pleasantness` vs measured sound level.

Visual
- Time series / distribution of `vegetation_fraction`, `pedestrian_count`, `brightness`, `blur` etc.
- Correlation heatmap between visual features and subjective scores.
- Thumbnail montages for the lowest/highest scoring segments for manual validation.

## Spatial mapping

Plot segment points using `geo_sync/segment_geo_metadata.csv` `matched_gps_*` fields (or `_wgs84` if exported) with folium or geopandas. Color-code by any metric (comfort_score, Leq, vegetation_fraction, priority_score).

## Quick plotting snippets

- Time series (comfort):
```python
import matplotlib.pyplot as plt
plt.plot(df.sort_values('segment_index')['segment_index'], df.sort_values('segment_index')['comfort_score'])
plt.show()
```

- Correlation heatmap:
```python
import seaborn as sns
cols = ['comfort_score','vitality_score','soundscape_pleasantness','vegetation_fraction','leq_db']
sns.heatmap(df[cols].corr(), annot=True)
```

- Folium map (save to map.html):
```python
import folium
m = folium.Map(location=[df['matched_gps_latitude_gcj02'].mean(), df['matched_gps_longitude_gcj02'].mean()], zoom_start=15)
for _,r in df.iterrows():
    folium.CircleMarker([r['matched_gps_latitude_gcj02'], r['matched_gps_longitude_gcj02']], radius=3, popup=str(r['segment_id']), color='red' if r['comfort_score']<3 else 'green').add_to(m)
m.save('map.html')
```

## Checks & caveats

- `groupTime` timestamp in GPS is expected to be Unix seconds; geo_sync assumes GCJ-02 coordinates by default.
- Soundscape features may rely on optional dependencies or local audio models; enable `enable_soundscape` and install optional requirements if needed.
- Some features are aggregated per-frame -> per-segment; confirm aggregation method in `visual/segment_visual_features.csv` header if needed.
- Subjective scores should be interpreted alongside `confidence_score` and annotator agreement metrics.

## Next steps

- Use the merged dataset for regression or priority scoring; Program 02 already computes a `priority_score` using `configs/street_type_coefficients.yaml`.
- For reproducible analysis, export the merged CSV and generated plots into `deliverable/` or `problem_detection/` as needed.
