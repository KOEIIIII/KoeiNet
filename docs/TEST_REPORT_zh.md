# 测试报告

日期：2026-05-25

## 测试环境

- 系统：Windows / PowerShell
- Python：`.\venv\Scripts\python.exe`
- GPU：NVIDIA GeForce RTX 4060
- 打包工具：PyInstaller 6.20.0
- 打包程序：
  - `dist/Program01_AnalysisVisualization/Program01_AnalysisVisualization.exe`
  - `dist/Program02_ScoringProblemDetection/Program02_ScoringProblemDetection.exe`

## 测试数据

- 最小视频：`examples/sample_inputs/VID_20250625_101458_00_006.mp4`
- 最小 GPS：`examples/sample_inputs/sample_gps.csv`
- Program 02 样例输入：`examples/sample_outputs/minimal_program01_output`
- 人工评分测试使用合法模拟值，评分范围为 1-5，问题标签使用 `traffic_noise`、`pedestrian_discomfort`、`low_vitality` 等字段。

## 执行内容

1. 重新打包 Program 01 和 Program 02。
2. 运行 Program 01 最小视频真实流程。
3. 运行 GUI flow smoke，模拟文件选择、中文切换、打开输出/网页、人工评分、系数配置和问题识别。
4. 运行项目级 smoke test。
5. 运行两个打包 exe 的 `--smoke-test`。
6. 清理测试产生的临时目录、release smoke 输出、PyInstaller build/spec 和 Python 缓存。

## Program 01 结果

- GUI 可初始化，中英文切换通过。
- GUI 命令预览已包含高级阶段开关，确认界面参数可以传入 CLI。
- 使用最小视频和 GPS 真实运行基础流程，进程未中断。
- 已生成基础帧、视觉统计、音频事件、时间片段、视觉片段特征、GPS 对齐、GIS 导出和 web sync 文件。
- 运行阶段结果：`segment`、`visual`、`geo_sync`、`gis_export`、`web_sync` 均通过。
- Open Output Folder 和 Open Visualization 的路径处理函数已通过 offscreen GUI flow 验证。

## Program 02 结果

- GUI 可初始化，中英文切换通过。
- 成功加载 Program 01 示例输出。
- 成功创建人工评分 CSV，填写 3 个 segment 的模拟评分，保存、重新加载、继续编辑并再次保存。
- 成功读取、修改、保存问题识别系数，并恢复默认配置。
- 成功运行问题路段识别。
- 结果：3 个 segment 被评分，2 个 problem segments 被识别，合并生成 1 个 problem episode。
- 成功生成 `segment_problem_priority.csv`、`problem_episodes.csv`、`problem_detection_summary.md` 和运行记录。

## 打包验证

- Program 01 one-folder 打包成功。
- Program 02 one-folder 打包成功。
- 两个 exe 的 `--smoke-test` 均返回成功。
- 可见窗口的无参数双击等价测试尝试启动时，自动化环境的桌面启动权限审批未在时限内返回，因此该项没有声称通过。已通过 exe smoke 和 GUI offscreen 启动验证打包入口与资源路径。

## 打包警告

打包过程中出现以下环境警告，但未阻止构建或 smoke test：

- Ultralytics 无法写入用户目录下的 settings/cache 文件。
- 离线环境下无法自动安装可选依赖 `lap>=0.5.12`。
- PyInstaller 报告 Numba TBB 可选库 `tbb12.dll` 未解析。

## 限制

- 本测试关注流程、路径和文件读写，不验证模型精度。
- 最小视频只有 3 秒，不能代表大型真实全景视频的性能表现。
- 真实桌面可见窗口点击测试仍建议在最终演示机器上人工确认一次。
