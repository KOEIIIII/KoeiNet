# StreetSmartEvaluator

全景街景视频多模态分析桌面启动器。项目保留现有 `main.py` 命令行流程，通过 `launcher_gui.py` 提供 Windows 桌面界面，用于选择视频、GPS、输出目录和常用分析参数，并可打包为便携版应用。

详细中文使用说明见：

- [README_使用说明.txt](README_%E4%BD%BF%E7%94%A8%E8%AF%B4%E6%98%8E.txt)

## Quick Start

开发环境启动 GUI：

```powershell
.\venv\Scripts\python.exe launcher_gui.py
```

Windows 打包：

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements_gui.txt
.\venv\Scripts\python.exe build_windows.py
```

打包完成后，普通用户使用整个目录：

```text
dist/StreetSmartEvaluator/
```

双击：

```text
StreetSmartEvaluator.exe
```

## Files Not Included in Git

仓库默认不提交以下本地资源：

- `apikey.env`
- `venv/`
- `input/`
- `output/`
- `dist/`
- `build/`
- `models/`
- `ffmpeg.exe`
- `yolo11m.pt`
- `output_gps.csv`
- 视频、音频和模型权重文件

GPS 示例文件见 `output_gps.sample.csv`。实际使用时可复制为 `output_gps.csv` 或在 GUI 中选择自己的 GPS 文件。

如果需要发布可直接运行的 Windows 应用，请将 `dist/StreetSmartEvaluator` 压缩后上传到 GitHub Releases。
