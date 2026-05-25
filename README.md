# KoeiNet

KoeiNet is a panoramic-video-based multimodal street-space evaluation system for historic districts.

KoeiNet 是一个面向历史街区的全景视频多模态街道空间智能评价系统。

The project keeps the existing command-line workflows and provides two desktop applications for staged delivery:

- Program 01: Basic Data Analysis and Spatial Visualization
- Program 02: Manual Scoring, Multimodal Fusion, and Problem-Segment Detection

Detailed documentation:

- [中文说明](README_zh.md)
- [English README](README_en.md)
- [Program 01 README](docs/PROGRAM_01_README_en.md)
- [Program 02 README](docs/PROGRAM_02_README_en.md)
- [项目展示页（中文）](docs/PROJECT_PAGE_zh.md)
- [Project page (English)](docs/PROJECT_PAGE_en.md)

## Quick Start

Development launch:

```powershell
.\venv\Scripts\python.exe scripts\run_program_01.py
.\venv\Scripts\python.exe scripts\run_program_02.py
```

Packaged local applications:

```text
dist/Program01_AnalysisVisualization/Program01_AnalysisVisualization.exe
dist/Program02_ScoringProblemDetection/Program02_ScoringProblemDetection.exe
```

Build packaged applications:

```powershell
.\venv\Scripts\python.exe scripts\build_program_01.py
.\venv\Scripts\python.exe scripts\build_program_02.py
```

## Files Not Included in Git

The repository does not commit local private data, temporary outputs, virtual environments, packaged executables, model weights, or API key files by default.

Ignored local resources include:

- `apikey.env`
- `venv/`
- `input/`
- `output/`
- `archive_unused/`
- `dist/`
- `build/`
- `models/`
- `ffmpeg.exe`
- `yolo11m.pt`
- raw video, audio, and large model files

Small public demonstration data is kept under `examples/sample_inputs/` and `examples/sample_outputs/`.
