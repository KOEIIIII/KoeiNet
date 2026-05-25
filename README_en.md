# KoeiNet

KoeiNet is a panoramic-video-based multimodal street-space evaluation system for historic districts. It supports street-space analysis, spatial anchoring, multimodal evidence organization, manual annotation, and problem-segment identification.

The repository keeps the original research pipeline intact and organizes it into two staged applications:

- Program 01: Basic Data Analysis and Spatial Visualization
- Program 02: Manual Scoring, Multimodal Fusion, and Problem-Segment Detection

## Workflow

1. Run Program 01 on panoramic video and GPS data.
2. Review the generated segment-level outputs and web visualization.
3. Run Program 02 on the Program 01 output.
4. Create or load an annotation CSV, enter human scores, configure street-type coefficients, and run problem detection.

Program 02 is designed as the second stage. It can start from a newly generated Program 01 output or from an existing output folder and annotation file.

## Quick Start

The local delivery folder contains PyInstaller one-folder desktop applications:

```text
dist/Program01_AnalysisVisualization/Program01_AnalysisVisualization.exe
dist/Program02_ScoringProblemDetection/Program02_ScoringProblemDetection.exe
```

Double-click either executable to open the graphical interface. Keep the whole one-folder directory intact when sharing the application; the `_internal/` folder contains required runtime libraries and bundled resources.

For development:

```powershell
python scripts\run_program_01.py --launch_gui
python scripts\run_program_02.py --launch_gui
```

## Program 01

Program 01 handles panoramic video processing, basic segment analysis, GPS alignment, GIS export, and local web visualization. The interface defaults to English and provides a Chinese language option.

Inputs:

- Video File: panoramic video in mp4, mov, avi, insv, mkv, or a compatible format.
- GPS File: CSV trajectory file. The current `geo_sync` pipeline requires `groupTime`, `gps_longitude`, and `gps_latitude`.
- Output Folder: target folder for generated results.

Typical outputs:

- `segments/segment_manifest.csv`
- `visual/segment_visual_features.csv`
- `geo_sync/segment_geo_metadata.csv`
- `gis/segment_gis_export.csv`
- `web_sync/` or `web/` visualization artifacts

CLI example:

```powershell
python scripts\run_program_01.py --input_video examples\sample_inputs\VID_20250625_101458_00_006.mp4 --gps_file examples\sample_inputs\sample_gps.csv --output_dir output --frame_skip 20
```

## Program 02

Program 02 loads Program 01 outputs and supports annotation, coefficient tuning, problem-segment scoring, episode merging, and export.

Recommended workflow:

1. Select a Program 01 Output folder.
2. Click Create Annotation File if no annotation CSV exists.
3. Fill or edit the annotation table by `segment_id`.
4. Click Save Annotation.
5. Load or edit the street-type coefficient configuration.
6. Set Top K, Priority Threshold, and Max Gap Seconds.
7. Click Run Problem Detection.
8. Review the exported problem segments, problem episodes, and summary report.

Street-type coefficients are stored in:

```text
configs/street_type_coefficients.yaml
```

## Sample Data

- `examples/sample_inputs/VID_20250625_101458_00_006.mp4`: tiny 3-second timestamped video with an audio track.
- `examples/sample_inputs/sample_gps.csv`: minimal GPS file in the current `geo_sync` schema.
- `examples/sample_outputs/minimal_program01_output/`: minimal Program 01 output that can be loaded directly by Program 02.

The samples validate workflow and file I/O only; they are not intended for accuracy evaluation.

## Installation

Use Python 3.11 and a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements_gui.txt
```

Optional soundscape dependencies:

```powershell
pip install -r requirements_optional_soundscape.txt
```

Some full pipeline features require local external assets such as `ffmpeg.exe`, `yolo11m.pt`, `models/`, and `config/model_dir/`. These files are large and should not be committed directly to GitHub. Use release assets, Git LFS, or a local resource package when distributing them.

## Packaging

```powershell
python scripts\build_program_01.py --dry-run
python scripts\build_program_02.py --dry-run
python scripts\build_program_01.py
python scripts\build_program_02.py
```

Build outputs are written to `dist/`. The repository `.gitignore` excludes `dist/` by default because the generated executables and runtime libraries are too large for normal source control. Publish packaged applications through GitHub Releases or a separate delivery archive.

## Documentation

- Project page: [docs/PROJECT_PAGE_en.md](docs/PROJECT_PAGE_en.md)
- Chinese project page: [docs/PROJECT_PAGE_zh.md](docs/PROJECT_PAGE_zh.md)
- Program 01 guide: [docs/PROGRAM_01_README_en.md](docs/PROGRAM_01_README_en.md)
- Program 02 guide: [docs/PROGRAM_02_README_en.md](docs/PROGRAM_02_README_en.md)
- Test report: [docs/TEST_REPORT_en.md](docs/TEST_REPORT_en.md)

## Privacy And Limits

Raw panoramic videos and GPS trajectories may contain sensitive location or personal information. Do not commit real field data, private GPS tracks, API keys, local model caches, or packaged binaries to the source repository. The included examples are minimal samples for workflow validation.
