# Output 清理与重建策略

## 1. 最重要的安全规则

- 永远只操作单个视频目录：`output/<video_name>/`
- 不要为了重做一个视频而删除整个 `output/` 根目录
- 不要误删 `output/` 下其他视频的结果目录
- `--from_existing_output` 模式本来就是面向一个具体的 `output/<video_name>/` 做增量补跑

## 2. 可以安全删除后重建的派生目录

以下目录属于中下游派生产物，通常可以按需删除后重建：

- `output/<video_name>/segments/`
- `output/<video_name>/visual/`
- `output/<video_name>/geo_sync/`
- `output/<video_name>/soundscape/`
- `output/<video_name>/fusion/`
- `output/<video_name>/diagnostics/`
- `output/<video_name>/design/`
- `output/<video_name>/deliverable/`
- `output/<video_name>/web/`
- `output/<video_name>/multimodal/`

说明：

- `multimodal/` 主要保存阶段状态、run manifest 和 pipeline summary，可安全重建
- `deliverable/`、`web/`、`geo_sync/` 都属于典型下游派生产物

## 3. 不建议随便删除的上游关键资产

以下目录通常是多个下游阶段共同依赖的基础资产，不建议轻易删除：

- `output/<video_name>/frames/`
- `output/<video_name>/stats/`
- `output/<video_name>/audio_events/`
- `output/<video_name>/ai_evaluation/`
- `output/<video_name>/validation/`
- `output/<video_name>/reproj/`
- `output/<video_name>/split/`
- `output/<video_name>/mask/`
- `output/<video_name>/overlay/`

原因：

- `frames/` 是 segment、visual、geo_sync 对齐的共同基础
- `stats/` 是 fusion 的主要视觉输入
- `audio_events/` 是 soundscape 的主要输入
- `validation/` 是 design / deliverable / proof 链路的重要依据

## 4. 三种推荐清理模式

### 模式 A：仅重做 deliverable / web

推荐删除：

- `output/<video_name>/deliverable/`
- `output/<video_name>/web/`
- `output/<video_name>/multimodal/deliverable/`
- `output/<video_name>/multimodal/web_sync/`

保留：

- `segments/`
- `geo_sync/`
- `soundscape/`
- `fusion/`
- `design/`

### 模式 B：重做 geo_sync + web

推荐删除：

- `output/<video_name>/geo_sync/`
- `output/<video_name>/web/`
- `output/<video_name>/multimodal/geo_sync/`
- `output/<video_name>/multimodal/web_sync/`

如问题路段高亮依赖 deliverable，也可同时删除：

- `output/<video_name>/deliverable/`
- `output/<video_name>/multimodal/deliverable/`

必须保留：

- `frames/`
- `segments/`
- `stats/`
- `audio_events/`

### 模式 C：重跑单个视频全部结果

推荐做法：

- 只删除 `output/<video_name>/`
- 不要删除整个 `output/`
- 保持其他视频目录不动

如果你只想重做后处理，也可以保留上游资产，只删除派生目录后使用 `--from_existing_output` 补跑。

## 5. 推荐重建顺序

- 只重做 deliverable / web：
  - 保留 `segments/`、`geo_sync/`、`soundscape/`、`fusion/`、`design/`
  - 删除 `deliverable/`、`web/` 和对应 `multimodal/` 子状态目录
- 重做 geo_sync + web：
  - 保留 `frames/`、`segments/`、`stats/`、`audio_events/`
  - 删除 `geo_sync/`、`web/` 和对应 `multimodal/` 子状态目录
- 单视频全量重跑：
  - 只删除 `output/<video_name>/`
  - 重新运行 `main.py`
