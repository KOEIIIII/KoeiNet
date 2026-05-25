# 清理报告

日期：2026-05-25

## 清理目标

本轮清理目标是让项目达到可上传 GitHub 的源码仓库状态，同时保留本机交付所需的双击应用、核心代码、配置、示例数据和文档。

## 已删除内容

- 删除 `insta360_segmentation.log`：运行日志，不应提交。
- 删除 PyInstaller `build/` 和根目录 `.spec` 文件：打包中间产物，可由脚本重新生成。
- 删除 `dist/*/release_smoke/`：打包 exe smoke test 临时输出。
- 删除 `_tmp_smoke/`：本轮临时测试输出。
- 删除非 `venv/`、非 `dist/`、非 `archive_unused/` 下的 `__pycache__/`：Python 缓存。
- 删除 `archive_unused/` 中明确无保留价值的旧打包目录、release 包、build 中间产物、历史 smoke 输出、pycache 归档和生成 spec 文件。
- 删除 `archive_unused/local_apikey.env`：本地 API key 文件，不应保留或上传。

## 已归档内容

以下内容不属于 GitHub 源码仓库，但可能对本机追溯有用，因此移动到 `archive_unused/`。该目录已写入 `.gitignore`，默认不提交：

- `input/` → `archive_unused/local_input_raw_video/`：本地真实原始视频，体量大，不应上传。
- `output/` → `archive_unused/local_output_runtime_results/`：历史完整运行输出，体量大，不应上传。
- `output_gps.csv` → `archive_unused/local_output_gps.csv`：本地真实 GPS 输出，不应公开。
- `output_gps.sample.csv` → `archive_unused/legacy_output_gps_sample.csv`：旧版样例格式，已由 `examples/sample_inputs/sample_gps.csv` 替代。
- `docs/output_cleanup_strategy.md` 和旧技术说明文档 → `archive_unused/legacy_docs/`：旧说明与当前两个程序交付结构不完全一致。

## 已保留内容

- `apps/`、`core/`、`src/`、`web/`、`configs/`、`scripts/`：程序源码和入口。
- `examples/sample_inputs/`：最小测试视频和 GPS。
- `examples/sample_outputs/minimal_program01_output/`：Program 02 所需最小 Program 01 输出样例。
- `dist/Program01_AnalysisVisualization/` 和 `dist/Program02_ScoringProblemDetection/`：本机可双击交付应用。注意：`dist/` 被 `.gitignore` 排除，GitHub 建议用 Releases 或独立压缩包分发。
- `ffmpeg.exe`、`yolo11m.pt`、`models/`、`config/model_dir/`：本机运行可能需要的外置资源。注意：这些大文件被 `.gitignore` 排除。
- 中英文 README、Program README、测试报告和清理报告。

## .gitignore 更新

`.gitignore` 已覆盖：

- Python 缓存和测试缓存。
- 虚拟环境。
- IDE/系统文件。
- API key、token、env、本地配置。
- 临时日志和临时目录。
- 本地输入、输出和 archive。
- PyInstaller build/dist/spec 中间产物。
- 大型模型、权重和媒体文件。
- 明确放行最小示例视频 `examples/sample_inputs/VID_20250625_101458_00_006.mp4`。

## 影响评估

清理后，Program 01 和 Program 02 的源码入口、GUI、配置、示例数据、打包脚本和本机打包 exe 均保留。清理后的 smoke test 通过，未发现路径被破坏。

## GitHub 上传建议

上传前执行：

```powershell
git status --short
python scripts\smoke_test.py
```

不要提交 `archive_unused/`、`dist/`、`venv/`、`models/`、`input/`、`output/` 或真实 GPS/视频数据。若需要发布双击应用，请使用 GitHub Releases、Git LFS 或外部分发包。
