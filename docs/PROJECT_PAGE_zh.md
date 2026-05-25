# KoeiNet

**KoeiNet 是一个面向历史街区的全景视频多模态街道空间智能评价系统。**

KoeiNet 围绕历史街区街道空间评价，支持全景视频分析、GPS 空间锚定、多模态证据组织、人工裁定标签、问题路段识别和网页可视化表达。项目目标不是替代规划专业判断，而是把视频、轨迹、视觉、声景、人工评分和空间位置组织为可追溯、可复核、可导出的分析证据。

## 研究定位

历史街区的街道空间评价往往需要同时考虑视觉环境、活动特征、声景感受、空间位置和人工判断。KoeiNet 将这些信息整理为片段级证据链，支持研究者和规划设计人员进行问题路段筛查、结果复核和可视化表达。

## 核心工作流

1. 输入全景视频。
2. 输入 GPS / 轨迹数据。
3. 进行基础视觉与空间分析。
4. 组织时间片段级证据。
5. 进行 GIS 空间锚定。
6. 生成网页可视化结果。
7. 进行人工评分与裁定。
8. 配置不同街道类型的问题识别系数。
9. 进行多模态融合与复核。
10. 识别问题路段 / 连续问题片段。

## 应用模块

### Program 01：基础数据分析与空间可视化

Program 01 是第一阶段应用。它读取全景视频和 GPS 数据，完成分析帧抽取、片段级结果组织、GPS 轨迹对齐、GIS 文件导出和本地网页可视化。

典型输出包括：

- 片段级分析结果。
- GPS 对齐后的空间结果。
- GIS 导出表。
- 本地网页可视化文件。
- 运行日志和阶段摘要。

### Program 02：人工评分、多模态融合与问题路段识别

Program 02 是第二阶段应用。它读取 Program 01 输出，创建或编辑人工评分 CSV，支持不同街道类型的系数配置，计算片段问题优先级，并将相邻问题片段合并为连续问题路段。

典型输出包括：

- 人工裁定标签 CSV。
- 系数配置文件。
- 片段优先级结果表。
- 问题片段结果。
- 连续问题路段结果。
- 摘要报告。

## 使用方式

1. 打开 Program 01。
2. 选择全景视频和 GPS 文件。
3. 设置 frame skip、时间片段长度、重叠比例、GPS 时间偏移等参数。
4. 运行基础分析并生成空间可视化结果。
5. 打开 Program 02。
6. 加载 Program 01 的输出目录。
7. 完成或模拟人工评分。
8. 配置街道类型系数。
9. 运行问题路段识别。
10. 导出并查看结果。

## 桌面应用下载

Windows 桌面应用从仓库 Releases 页面下载：

```text
https://github.com/KOEIIIII/KoeiNet/releases
```

请下载所有 `KoeiNet_Windows.zip.part*` 分卷文件、`Join_And_Extract.bat` 和 `README_FIRST.txt`。把它们放在同一文件夹中，双击 `Join_And_Extract.bat` 自动合并并解压。解压完成后，打开：

```text
KoeiNet_Windows/Program01_AnalysisVisualization.exe
KoeiNet_Windows/Program02_ScoringProblemDetection.exe
```

## 主要输出

- `segments/segment_manifest.csv`
- `visual/segment_visual_features.csv`
- `geo_sync/segment_geo_metadata.csv`
- `gis/segment_gis_export.csv`
- `web/` 或 `web_sync/` 可视化文件
- `validation/final_annotation_labels_adjudicated.csv`
- `problem_detection/segment_problem_priority.csv`
- `problem_detection/problem_episodes.csv`
- `problem_detection/problem_detection_summary.md`

## 示例数据

仓库中保留了用于流程验证的最小公开样例：

- 一个带时间戳的极小视频样例。
- 一个最小 GPS CSV。
- 一个 Program 01 输出样例。

这些示例用于 smoke test 和界面流程演示，不代表真实研究数据规模。

## 注意事项与限制

- 原始全景视频和 GPS 轨迹可能包含位置隐私或个人信息，不应直接公开上传。
- 部分分析流程依赖本地模型、可选声景依赖或 GPU 环境。
- smoke test 主要验证流程连通性和文件读写链路，不代表算法精度评估。
- 如果需要复现实地研究结果，应使用经过脱敏和授权的数据，并记录模型、参数和运行环境。
