# Program 02：人工评分、多模态融合与问题路段识别

Program 02 是第二阶段程序。它读取 Program 01 输出，支持人工评分、街道类型系数配置、问题 segment 识别和连续问题路段合并。它不会重新生成 Program 01 的基础分析结果。

## 功能范围

- 加载 Program 01 的 segment-level 输出。
- 创建或继续编辑人工评分 CSV。
- 按 `segment_id` 浏览和修改评分字段。
- 加载、编辑、保存和恢复街道类型系数配置。
- 根据人工评分、多模态特征和系数计算 `priority_score`。
- 复用现有连续片段合并逻辑生成 problem episodes。
- 输出问题路段表、摘要报告和可视化兼容文件。

## 启动方式

本机交付应用：

```text
dist/Program02_ScoringProblemDetection/Program02_ScoringProblemDetection.exe
```

开发环境 GUI：

```powershell
python scripts\run_program_02.py --launch_gui
```

开发环境 CLI：

```powershell
python scripts\run_program_02.py --program_01_output examples\sample_outputs\minimal_program01_output --top_k 2 --priority_threshold 0.4
```

创建评分模板：

```powershell
python scripts\run_program_02.py --program_01_output examples\sample_outputs\minimal_program01_output --create_annotation_template
```

## 标准操作流程

Program 02 推荐先进行人工评分，再运行问题路段识别：

1. 选择 Program 01 Output。
2. 没有评分文件时，点击 Create Annotation File。
3. 在 Annotation 表格中填写或修改评分。
4. 点击 Save Annotation。
5. 在 Coefficients 页加载或修改系数。
6. 设置 Top K、Priority Threshold 和 Max Gap Seconds。
7. 点击 Run Problem Detection。
8. 查看 `problem_detection/` 输出。

如果已有评分文件，可以直接加载 Annotation CSV，检查或微调后运行识别。

## 人工评分字段

- `segment_id`：片段编号，不建议手动修改。
- `street_type`：街道类型，用于选择系数方案。
- `comfort_score`：舒适度评分，建议 1-5。
- `vitality_score`：活力评分，建议 1-5。
- `soundscape_pleasantness`：声景愉悦度，建议 1-5。
- `soundscape_eventfulness`：声景事件性，建议 1-5。
- `overall_problem_severity`：总体问题严重度，建议 1-5。
- `main_problem_labels`：主要问题标签，可用分号分隔多个标签。
- `primary_problem_label`：主问题标签。
- `confidence_score`：标注信心，建议 1-5。
- `annotator_notes`：评审备注。

## 系数配置

默认配置文件：

```text
configs/street_type_coefficients.yaml
```

该文件标注为 `default configuration`。每个 `street_type` 可配置：

- `severity_threshold`：判定为问题片段的阈值。
- `desired_eventfulness`：事件性期望范围。
- `coefficients`：各评分组成项的权重。

GUI 支持读取、编辑、保存和恢复默认配置。普通用户可以直接使用默认配置；研究复核时可按街道类型调整权重。

## 识别逻辑

Program 02 会合并以下信息：

- `segments/segment_manifest.csv`
- 人工评分 CSV
- `fusion/segment_feature_table.csv`
- `visual/segment_visual_features.csv`
- `soundscape/audio_segment_features.csv`
- `geo_sync/segment_geo_metadata.csv`

每个 segment 会得到 `priority_score`、`priority_level` 和 `is_problem_segment`。随后程序调用现有 `src.deliverable.episode_builder` 中的连续片段合并逻辑，生成 problem episodes。

## 输出文件

默认输出到 Program 01 输出目录下的 `problem_detection/`：

- `segment_problem_priority.csv`
- `problem_episodes.csv`
- `problem_detection_summary.md`
- `problem_detection_run.json`

启用 Update visualization-compatible artifacts 时，还会写入：

- `design/segment_priority_ranking.csv`
- `deliverable/problem_episodes.csv`
- `deliverable/problem_episode_summary.csv`

写入前会备份已有兼容文件。

## 常见错误

- 缺少 `segment_manifest.csv`：请先运行 Program 01 或选择正确的 Program 01 输出目录。
- 评分 CSV 无法读取：确认包含 `segment_id`，并保留标准评分字段。
- 系数配置无法解析：保持 JSON-compatible YAML，或安装 PyYAML。
- 没有识别出 problem episodes：检查阈值、Top K、评分值和 `main_problem_labels`。
