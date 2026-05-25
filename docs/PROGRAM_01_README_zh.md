# Program 01：基础数据分析与空间可视化

Program 01 是第一阶段程序，用于将全景视频和 GPS/轨迹数据整理为可用于后续评分和问题识别的基础结果。它封装现有 `main.py` 流水线，不重写核心算法。

## 功能范围

- 读取全景视频。
- 抽取分析帧。
- 生成六视角 cubemap 或沿用现有全景处理流程。
- 生成基础视觉指标和片段级视觉汇总。
- 读取音频并生成基础音频事件结果；完整声景扩展可按依赖情况启用。
- 生成 `segment_manifest.csv`。
- 将视频片段与 GPS 轨迹对齐。
- 导出 GIS 表格和网页同步可视化文件。

## 启动方式

本机交付应用：

```text
dist/Program01_AnalysisVisualization/Program01_AnalysisVisualization.exe
```

开发环境 GUI：

```powershell
python scripts\run_program_01.py --launch_gui
```

开发环境 CLI：

```powershell
python scripts\run_program_01.py --input_video examples\sample_inputs\VID_20250625_101458_00_006.mp4 --gps_file examples\sample_inputs\sample_gps.csv --output_dir output --frame_skip 20
```

基于已有输出继续运行后处理：

```powershell
python scripts\run_program_01.py --from_existing_output output\VID_20250625_101458_00_006 --post_only --gps_file examples\sample_inputs\sample_gps.csv
```

## 输入要求

视频文件应为 OpenCV/ffmpeg 可读取格式。若启用 GPS 对齐，视频需要具备可解析的开始时间，可来自文件名、视频元数据或 sidecar。当前文件名解析支持类似 `VID_20250625_101458_00_006.mp4` 的时间戳。

GPS CSV 至少包含：

```text
groupTime,gps_longitude,gps_latitude
```

`groupTime` 使用 Unix 秒级时间戳；经纬度字段使用当前项目 geo_sync 模块约定的坐标来源。

## 参数说明

- `input_video`：单个视频文件。
- `gps_file`：GPS CSV。
- `output_dir`：输出根目录。
- `frame_skip`：抽帧间隔。
- `segment_seconds`：时间片段长度。
- `segment_overlap`：时间片段重叠长度。
- `gps_time_offset_seconds`：GPS 与视频时间偏移。
- `from_existing_output`：从已有 Program 01 输出继续。
- `post_only`：只运行后处理阶段。
- `resume_missing_only`：只补齐缺失产物。
- `enable_segment_pipeline`：启用片段流水线。
- `enable_visual_segment_summary`：启用片段级视觉汇总。
- `enable_soundscape`：启用声景扩展阶段。
- `enable_geo_sync`：启用轨迹对齐。
- `enable_gis_export`：启用 GIS 导出。
- `enable_web_sync_export`：启用网页同步数据导出。

GUI 中的高级开关已映射到上述 CLI 参数。

## 输出文件

典型输出目录为 `output/<video_name>/`，包括：

- `frames/`：抽取帧。
- `split/`、`mask/`、`overlay/`、`reproj/`：原有视觉处理产物。
- `stats/`：基础统计表和图表。
- `audio_events/`：基础音频事件结果。
- `segments/segment_manifest.csv`：片段清单。
- `visual/segment_visual_features.csv`：片段视觉特征。
- `geo_sync/frame_geo_metadata.csv`：帧级 GPS 对齐结果。
- `geo_sync/segment_geo_metadata.csv`：片段级 GPS 对齐结果。
- `gis/segment_gis_export.csv`：片段 GIS 导出表。
- `web_sync/`、`web/`：网页可视化和同步数据。

## 界面操作

1. 点击 Select Video 选择视频。
2. 点击 Select GPS File 选择 GPS 文件。
3. 点击 Select Output Folder 设置输出目录。
4. 设置 Frame Skip、Segment Length、Segment Overlap 和 GPS Time Offset。
5. 在 Advanced Settings 中按需启用或关闭后处理阶段。
6. 点击 Start Analysis。
7. 运行完成后点击 Open Output Folder 或 Open Visualization。

界面默认英文，可切换中文。

## 常见错误

- GPS 字段不匹配：请使用 `groupTime,gps_longitude,gps_latitude`。
- geo_sync 无法解析视频开始时间：请检查文件名时间戳、视频元数据或 sidecar。
- 缺少模型或 ffmpeg：确认本机交付目录或项目根目录中存在必要外置资源。
- 大型视频运行慢：调大 `frame_skip` 或先用 `post_only` 验证后处理链路。
