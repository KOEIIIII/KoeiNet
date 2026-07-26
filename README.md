# KoeiNet

- [English README](README_en.md)
- [中文说明](README_zh.md)

# KoeiNet

**KoeiNet is a panoramic-video-based multimodal street-space evaluation system for historic districts.**

**KoeiNet 是一个面向步行空间的全景视频多模态街道空间智能评价系统。**

## Download And Run The Desktop Apps / 下载并运行桌面应用

**For most users, start here: [Download the latest Windows desktop release](https://github.com/KOEIIIII/KoeiNet/releases/latest).**

**普通用户请从这里开始：[下载最新版 Windows 桌面应用](https://github.com/KOEIIIII/KoeiNet/releases/latest)。**

On the release page, open **Assets** and download these files:

```text
KoeiNet_Windows.zip.part001
KoeiNet_Windows.zip.part002
KoeiNet_Windows.zip.part003
KoeiNet_Windows.zip.part004
Join_And_Extract.bat
README_FIRST.txt
```

Put all six files in the same folder, then double-click:

```text
Join_And_Extract.bat
```

After extraction, open either application:

```text
KoeiNet_Windows/Program01_AnalysisVisualization.exe
KoeiNet_Windows/Program02_ScoringProblemDetection.exe
```

Please download the files listed above from **Assets**. The automatically generated "Source code (zip)" and "Source code (tar.gz)" archives are for developers and do not contain the packaged desktop runtime.

KoeiNet supports street-space analysis, GPS spatial anchoring, segment-level multimodal evidence organization, manual annotation, street-type-specific coefficient configuration, problem-segment detection and export.

KoeiNet 面向历史街区街道空间评价，支持全景视频分析、GPS 空间锚定、多模态证据组织、人工裁定标签、不同街道类型的系数配置、问题路段识别与导出等功能。

## Application Modules / 应用模块

### Program 01: Basic Data Analysis and Spatial Visualization

Program 01 is the first-stage application. It reads panoramic video and GPS/trajectory files, extracts analysis frames, organizes segment-level results, aligns segments to spatial trajectories, and generates local web visualizations.

Program 01 是第一阶段应用，负责读取全景视频和 GPS/轨迹文件，完成分析帧抽取、片段级结果组织、轨迹对齐、GIS 文件导出以及本地网页可视化的生成。

### Program 02: Manual Scoring, Multimodal Fusion, and Problem-Segment Detection

Program 02 is the second-stage application. It loads Program 01 outputs, creates or edits manual annotation CSV files, supports street-type coefficient configuration, computes segment-level priority scores, and merges adjacent problem segments into continuous problem episodes for export and review.

Program 02 是第二阶段应用，基于 Program 01 的输出继续进行人工评分、多模态融合参数配置、问题优先级计算，并将相邻问题片段合并为连续的问题路段以便导出与审阅。

## Core Workflow / 核心流程

1. Input panoramic video / 输入全景视频。
2. Input GPS or trajectory data / 输入 GPS 或轨迹数据。
3. Run basic visual and spatial analysis / 进行基础视觉与空间分析。
4. Organize segment-level evidence / 组织时间片段级证据。
5. Anchor results to GIS space / 进行 GIS 空间锚定。
6. Generate local web visualization / 生成本地网页可视化。
7. Complete manual scoring and adjudication / 完成人工评分与裁定。
8. Configure street-type coefficients / 配置不同街道类型的问题识别系数。
9. Run multimodal fusion and review / 进行多模态融合与复核。
10. Detect problem segments and episodes / 识别问题片段与连续问题路段。

## Quick Start / 快速开始

Download desktop applications from GitHub Releases:

```text
https://github.com/KOEIIIII/KoeiNet/releases
```

For a packaged Windows release, download all parts named like the examples above and the `Join_And_Extract.bat` helper. Put the downloaded files in the same folder and run the batch file to join and extract the packaged runtime.

After extraction, run one of the executables:

```text
KoeiNet_Windows/Program01_AnalysisVisualization.exe
KoeiNet_Windows/Program02_ScoringProblemDetection.exe
```

Development launch (from a Windows PowerShell or compatible shell):

```powershell
python scripts\run_program_01.py --launch_gui
python scripts\run_program_02.py --launch_gui
```

Build packaged applications (development):

```powershell
python scripts\build_program_01.py
python scripts\build_program_02.py
python scripts\package_windows_release.py
```

## How To Use / 使用方式

Program 01:

1. Launch Program 01 / 打开 Program 01。
2. Select the panoramic video file / 选择全景视频文件。
3. Select the GPS file / 选择 GPS 文件。
4. Select the output folder and configure basic parameters / 选择输出目录并设置基础参数。
5. Start analysis / 开始分析。
6. Open the output folder or local visualization page / 打开输出文件夹或本地可视化网页。

Program 02:

1. Launch Program 02 / 打开 Program 02。
2. Load the Program 01 output folder / 加载 Program 01 输出结果。
3. Create or load an annotation CSV / 创建或加载人工评分 CSV。
4. Fill or edit scores such as `comfort_score`, `vitality_score`, `soundscape_pleasantness`, `soundscape_eventfulness`, `overall_problem_severity`, and `confidence_score` / 填写或修改相关评分字段。
5. Load, edit, save, or restore street-type coefficients / 读取、修改、保存或恢复街道类型系数。
6. Run problem-segment detection / 运行问题路段识别。
7. Export problem segment and problem episode results / 导出问题片段与连续问题路段结果。

## Key Outputs / 主要输出

- `segments/segment_manifest.csv`
- `visual/segment_visual_features.csv`
- `geo_sync/segment_geo_metadata.csv`
- `gis/segment_gis_export.csv`
- `web/` or `web_sync/` local visualization artifacts
- `validation/final_annotation_labels_adjudicated.csv`
- `configs/street_type_coefficients.yaml`
- `problem_detection/segment_problem_priority.csv`
- `problem_detection/problem_episodes.csv`
- `problem_detection/problem_detection_summary.md`

## Documentation / 文档

- [English README](README_en.md)
- [中文说明](README_zh.md)
- [Program 01 README](docs/PROGRAM_01_README_en.md)
- [Program 01 中文说明](docs/PROGRAM_01_README_zh.md)
- [Program 02 README](docs/PROGRAM_02_README_en.md)
- [Program 02 中文说明](docs/PROGRAM_02_README_zh.md)
- [Project Page (English)](docs/PROJECT_PAGE_en.md)
- [项目展示页（中文）](docs/PROJECT_PAGE_zh.md)
- [Download Desktop Apps](docs/DOWNLOAD_DESKTOP_APPS_en.md)
- [桌面应用下载说明](docs/DOWNLOAD_DESKTOP_APPS_zh.md)
- [Test Report](docs/TEST_REPORT_en.md)
- [测试报告](docs/TEST_REPORT_zh.md)

## Repository Contents / 仓库内容

```text
apps/       desktop application wrappers for Program 01 and Program 02
core/       lightweight shared wrappers around existing processing modules
configs/    default program configuration and street-type coefficients
docs/       bilingual documentation, reports, and project pages
examples/   minimal public sample inputs and outputs
scripts/    launch, build, and smoke-test scripts
src/        existing research pipeline implementation
web/        local web visualization and annotation resources
tests/      basic test placeholders and smoke-test support
```

## Files Not Included in Git / 未纳入 Git 的文件

The repository excludes local private data, temporary outputs, virtual environments, packaged executables, model weights, and API key files by default.

仓库默认不提交本地隐私数据、临时输出、虚拟环境、打包产物、模型权重和真实 API key 文件。

Ignored local resources include:

- `apikey.env`
- `venv/`, `.venv/`, `env/`
- `input/`, `output/`, `archive_unused/`
- `dist/`, `build/`, `release/`
- `release_packages/`
- `models/`, `config/model_dir/`
- `ffmpeg.exe`, `yolo11m.pt`
- raw video, audio, and large model files

Small public demonstration data is kept under `examples/sample_inputs/` and `examples/sample_outputs/`.

## Notes and Limitations / 注意事项与限制

- The downloadable Windows package is split into multiple parts because the shared runtime contains computer-vision and machine-learning libraries. Users must download every part before running the extractor.
- Raw panoramic videos and GPS trajectories may contain privacy-sensitive information. Only desensitized or minimal public samples should be uploaded.
- Some analysis paths may depend on local model files, optional soundscape dependencies, or GPU availability.
- The included smoke tests verify workflow connectivity and file I/O, not academic accuracy.

- Windows 桌面应用采用分卷压缩包下载，因为共享运行库包含计算机视觉和机器学习依赖。用户需要下载全部分卷后再运行 `Join_And_Extract.bat`。
- 原始全景视频和 GPS 轨迹可能包含隐私信息，公开仓库中只应保留脱敏或最小示例数据。
- 部分分析流程可能依赖本地模型文件、可选声景依赖或 GPU 环境。
- 当前 smoke test 主要验证流程连通性和文件读写链路，不代表算法精度评估。
