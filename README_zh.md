# KoeiNet

KoeiNet 是一个面向历史街区的全景视频多模态街道空间智能评价系统，支持街道空间分析、GPS 空间锚定、多模态证据组织、人工裁定标签和问题路段识别。

当前仓库保留原有研究型处理流程，并整理为两个递进式应用：

- Program 01：基础数据分析与空间可视化
- Program 02：人工评分、多模态融合与问题路段识别

## 工作流程

1. 使用 Program 01 处理全景视频和 GPS 数据。
2. 查看生成的片段级结果和网页可视化。
3. 使用 Program 02 加载 Program 01 输出。
4. 创建或加载人工评分 CSV，填写评分，配置街道类型系数，并运行问题路段识别。

Program 02 是第二阶段程序，可以基于新生成的 Program 01 输出运行，也可以基于已有输出和已有评分文件继续分析。

## 快速开始

本机交付目录中包含 PyInstaller one-folder 桌面应用：

```text
dist/Program01_AnalysisVisualization/Program01_AnalysisVisualization.exe
dist/Program02_ScoringProblemDetection/Program02_ScoringProblemDetection.exe
```

普通用户可以双击对应 exe 打开图形界面。分发时必须保留完整 one-folder 目录，不能只复制 exe 文件，因为 `_internal/` 中包含运行库和资源文件。

开发环境启动：

```powershell
python scripts\run_program_01.py --launch_gui
python scripts\run_program_02.py --launch_gui
```

## Program 01

Program 01 面向全景视频基础处理、时间片段组织、GPS 对齐、GIS 导出和本地网页可视化。界面默认英文，可切换中文。

输入：

- Video File：全景视频文件，支持 mp4、mov、avi、insv、mkv 等格式。
- GPS File：轨迹 CSV。当前 `geo_sync` 流程要求包含 `groupTime`、`gps_longitude`、`gps_latitude`。
- Output Folder：输出目录。

典型输出：

- `segments/segment_manifest.csv`
- `visual/segment_visual_features.csv`
- `geo_sync/segment_geo_metadata.csv`
- `gis/segment_gis_export.csv`
- `web_sync/` 或 `web/` 可视化文件

命令行示例：

```powershell
python scripts\run_program_01.py --input_video examples\sample_inputs\VID_20250625_101458_00_006.mp4 --gps_file examples\sample_inputs\sample_gps.csv --output_dir output --frame_skip 20
```

## Program 02

Program 02 读取 Program 01 输出，支持人工评分、系数配置、问题 segment 识别、连续问题片段合并和结果导出。

推荐流程：

1. 选择 Program 01 Output。
2. 没有评分文件时，点击 Create Annotation File。
3. 在 Annotation 表格中按 `segment_id` 填写或修改评分。
4. 点击 Save Annotation。
5. 加载或编辑街道类型系数配置。
6. 设置 Top K、Priority Threshold、Max Gap Seconds。
7. 点击 Run Problem Detection。
8. 查看导出的问题片段、连续问题路段和摘要报告。

街道类型系数文件：

```text
configs/street_type_coefficients.yaml
```

## 示例数据

- `examples/sample_inputs/VID_20250625_101458_00_006.mp4`：3 秒最小测试视频，文件名包含可解析时间戳。
- `examples/sample_inputs/sample_gps.csv`：符合当前 `geo_sync` 字段要求的最小 GPS 示例。
- `examples/sample_outputs/minimal_program01_output/`：Program 02 可直接加载的最小 Program 01 输出。

示例数据只用于验证流程和文件读写，不用于评价算法精度。

## 安装环境

建议使用 Python 3.11 和虚拟环境：

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements_gui.txt
```

可选声景依赖：

```powershell
pip install -r requirements_optional_soundscape.txt
```

部分完整流程依赖本地外置资源，例如 `ffmpeg.exe`、`yolo11m.pt`、`models/` 和 `config/model_dir/`。这些文件体量较大，不建议直接提交到 GitHub；可通过 release assets、Git LFS 或本地资源包分发。

## 打包

```powershell
python scripts\build_program_01.py --dry-run
python scripts\build_program_02.py --dry-run
python scripts\build_program_01.py
python scripts\build_program_02.py
```

打包结果位于 `dist/`。仓库 `.gitignore` 默认排除 `dist/`，因为 exe 和运行库体量较大。需要发布双击应用时，建议使用 GitHub Releases 或单独交付压缩包。

## 文档

- 项目展示页：[docs/PROJECT_PAGE_zh.md](docs/PROJECT_PAGE_zh.md)
- English project page: [docs/PROJECT_PAGE_en.md](docs/PROJECT_PAGE_en.md)
- Program 01 说明：[docs/PROGRAM_01_README_zh.md](docs/PROGRAM_01_README_zh.md)
- Program 02 说明：[docs/PROGRAM_02_README_zh.md](docs/PROGRAM_02_README_zh.md)
- 测试报告：[docs/TEST_REPORT_zh.md](docs/TEST_REPORT_zh.md)

## 隐私与限制

原始全景视频和 GPS 轨迹可能包含位置隐私或个人信息。不要将真实采集视频、私人 GPS 轨迹、API key、本地模型缓存或打包二进制文件提交到源码仓库。当前示例文件仅用于流程验证。
