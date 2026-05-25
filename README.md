# KoeiNet

KoeiNet is a panoramic-video-based multimodal street-space evaluation system for historic districts.

KoeiNet 是一个面向历史街区的全景视频多模态街道空间智能评价系统。

- 中文说明：[README_zh.md](README_zh.md)
- English README: [README_en.md](README_en.md)
- Project pages: [English](docs/PROJECT_PAGE_en.md) / [中文](docs/PROJECT_PAGE_zh.md)

Local double-click applications:

```text
dist/Program01_AnalysisVisualization/Program01_AnalysisVisualization.exe
dist/Program02_ScoringProblemDetection/Program02_ScoringProblemDetection.exe
```

The `dist/` directory is kept for local delivery but ignored by Git because the packaged executables are large. For GitHub releases, rebuild with `scripts/build_program_01.py` and `scripts/build_program_02.py`, or publish the packaged folders as release assets.
