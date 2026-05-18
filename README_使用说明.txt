街景智评桌面版使用说明
StreetSmartEvaluator User Guide

一、App 功能简介

StreetSmartEvaluator 是一个用于全景街景/普通视频分析的桌面启动器。它封装项目中已经跑通的 main.py 命令行流程，帮助普通用户通过图形界面完成视频处理、GPS 轨迹同步、街道空间指标分析、多模态后处理、GIS 数据导出、网页展示和交付物生成。

程序不会改变核心算法流程。桌面界面只负责选择输入、设置参数、检查资源、启动后台分析进程、显示日志，并帮助用户打开输出文件夹或生成的 HTML 页面。


二、运行环境说明

1. 已打包桌面版

普通用户不需要安装 Python，也不需要手动安装依赖。请把整个 StreetSmartEvaluator 文件夹复制到电脑上，然后双击：

StreetSmartEvaluator.exe

请不要只复制 exe 文件。ffmpeg.exe、models、web、config、_internal 等目录必须和 exe 放在同一个 StreetSmartEvaluator 文件夹中。

推荐系统：Windows 10 / Windows 11 64 位。

2. 开发环境运行

如果需要从源码启动 GUI，可在项目根目录运行：

.\venv\Scripts\python.exe launcher_gui.py

开发环境需要安装 requirements.txt 和 requirements_gui.txt。普通用户不需要执行这些步骤。

3. macOS / Linux

代码中保留了打开文件夹和网页的 macOS / Linux 兼容写法，但当前打包脚本面向 Windows onedir 发行。macOS / Linux 如需桌面版，需要在对应系统上重新打包。


三、App 界面使用说明

1. Input & Output / 输入与输出

Run Mode / 运行模式：
选择 Raw Video / Input Folder 时，程序从原始视频或输入文件夹开始分析。
选择 Existing Output Folder 时，程序从已有 output/<video_name> 结果目录继续运行后处理或交付物层，不重新做高成本帧级分析。

Input Video / Folder / 输入视频或文件夹：
选择单个视频文件，或选择包含多个视频的文件夹。支持的常见格式包括 mp4、mov、avi、insv。

Existing Output Folder / 已有输出目录：
用于选择已经分析过的视频结果目录，例如 output/VID_20250625_101458_00_006。只有 Run Mode 选择 Existing Output Folder 时使用。

GPS File / GPS 文件：
可选。GUI 支持选择 csv、xlsx、xls。核心 geo_sync 流程实际读取 CSV；如果选择 Excel，GUI 会在运行前自动转换成临时 CSV。

Output Root Folder / 输出根目录：
选择总输出目录。程序会在该目录下为每个视频建立结果子目录。

2. Analysis Options / 分析选项

Preset / 预设：
快速测试模式：使用较大的跳帧间隔，只验证流程是否能跑通。
标准分析模式：使用常规参数运行。
从已有结果生成交付物：适合已有分析结果，只生成最终交付层。
带 GPS 的 Web 展示：启用地理同步、GIS 导出和 Web 同步导出。
高质量完整模式：尽量启用正式分析和交付物输出，耗时最长。

3. Resource Settings / 资源设置

FFmpeg Path / FFmpeg 路径：
视频转码和音频抽取依赖 ffmpeg.exe。打包版默认在 exe 同级目录查找。

Models Folder / 模型文件夹：
模型资源目录。默认查找 exe 同级 models 文件夹，并检查 yolo11m.pt。

API Key Env File / API Key 环境文件：
可选。启用 GLM 或 Agents 时需要 apikey.env。常用键名为 ZHIPUAI_API_KEY 或 ZHIPU_API_KEY。不要把真实密钥写入代码。

4. Command Preview / 命令预览

这里显示 GUI 将要执行的 main.py 参数。路径会以参数列表方式传递，支持中文路径、空格路径和长路径。

5. Run Panel / 运行面板

Run Analysis / 运行分析：
开始后台分析。GUI 不会卡死，会持续显示日志。

Stop / 停止：
终止当前后台进程。已生成的中间文件不会自动删除。

Open Output Folder / 打开输出目录：
打开输出根目录或本次运行的视频结果目录。

Open Web Page / 打开网页：
自动查找输出目录中的 HTML。优先打开 deliverable、web_sync、index、report 等页面。如果有多个 HTML，会弹出选择框。程序会启动本地 127.0.0.1 HTTP 服务，避免 file:// 打开时的视频或资源加载问题。端口占用时会自动尝试下一个端口。

Save Config / Load Config / Reset Defaults：
保存、加载或恢复 GUI 参数。配置文件是 JSON，不包含真实 API 密钥。

Save Log / 保存日志：
保存当前日志面板内容。程序也会在 output/_launcher_logs 下写入本次运行日志。


四、可调参数详细说明

Run Mode / 运行模式：
raw 表示从原始视频或输入文件夹开始运行；existing 表示从已有输出目录继续。普通新任务使用 raw；已有结果补交付物或补后处理使用 existing。

Input Video / Folder：
原始视频或输入目录。单视频会传给 --video_path，目录会传给 --input_dir。

Existing Output Folder：
对应 --from_existing_output。用于复用已有 output/<video_name>。不会重新运行高成本 legacy 帧级流程，适合补生成多模态阶段、GIS、Web Sync 或交付物。

GPS File：
对应 --geo_sync_gps_csv。选择 GPS 后，GUI 会自动建议启用 Geo Sync、GIS Export 和 Web Sync。没有 GPS 时不要开启这些依赖坐标的流程。

Output Root Folder：
对应 --output_dir。输出越多，目录占用空间越大。建议选择磁盘空间充足、用户有写入权限的位置。

Frame Skip / 跳帧间隔：
对应 --frame_skip。默认推荐 20。数值越大，抽取和分析的帧越少，速度越快，输出文件更小，但细节精度降低；数值越小，分析更密集，速度更慢，结果更细，输出更大。快速测试可用 100-200；正式分析可用 10-20。

Segment Length / 分段长度：
对应 --segment_seconds。默认推荐 5.0 秒。数值越大，每个 segment 覆盖时间更长，片段数量更少，结果更概括；数值越小，片段更密，定位更细，但后处理文件更多。

Segment Overlap / 分段重叠：
对应 --segment_overlap。默认推荐 2.5 秒。增加重叠可让片段衔接更平滑，但会增加片段数量和后处理开销。通常不应大于 Segment Length。

GPS Time Offset / GPS 时间偏移：
对应 --geo_sync_time_offset_seconds。默认推荐 25.0 秒，具体取决于设备时间、视频起始时间和 GPS 记录方式。数值为正表示将视频时间向后偏移；为负表示向前偏移。视频和地图点位整体错位时调整此项。

Web Port / Web 端口：
用于 main.py 自动 Web 服务端口，默认 5000。GUI 的 Open Web Page 另会使用 8765 起的本地端口自动打开静态 HTML。端口被占用时可换一个。

Enable Segment Pipeline / 启用分段流水线：
对应 --enable_segment_pipeline。生成 segment_manifest 等片段级基础数据。后续 visual、geo_sync、soundscape、fusion、GIS、Web Sync 常依赖分段结果。会增加处理时间和中间文件。

Enable Visual Segment Summary / 启用视觉分段摘要：
对应 --enable_visual_segment_summary。把帧级视觉结果汇总到 segment 级。适合需要片段级展示、融合建模或交付物的场景。

Enable Soundscape / 启用声景分析：
对应 --enable_soundscape。提取音频和声景特征。会增加运行时间，并依赖音频处理、ffmpeg 和相关模型资源。视频没有音轨时可能无法生成有效结果。

Enable Fusion / 启用融合：
对应 --enable_fusion。整合视觉、声景、地理等特征，生成多模态特征表。适合需要综合分析或后续建模评估的任务。

Enable Agents / 启用智能体：
对应 --enable_agents。生成结构化诊断或质控类结果。通常需要 API key；没有 key 时应关闭。

Enable Design / 启用设计映射：
对应 --enable_design。基于诊断和优先级生成设计建议、干预映射等结果。适合正式交付。

Enable Deliverable / 启用交付物阶段：
对应 --enable_deliverable。启用 pipeline 内部交付物阶段。若只想从已有结果直接生成交付包，也可以使用 Run Deliverable Layer。

Enable Geo Sync / 启用地理同步：
对应 --enable_geo_sync。把视频帧/片段与 GPS 点进行时间对齐，输出 frame_geo_metadata.csv 和 segment_geo_metadata.csv。必须提供有效 GPS 文件。

Enable GIS Export / 启用 GIS 导出：
对应 --enable_gis_export。生成 GIS 友好的 CSV 文件，便于在 GIS 软件或地图工具中查看。依赖 Geo Sync 结果。

Enable Web Sync Export / 启用 Web 同步导出：
对应 --enable_web_sync_export。生成视频、图表和地图联动展示需要的 JSON 数据。依赖 Geo Sync 和分段结果。

Align to Analysis Frames / 对齐分析帧：
对应 --geo_sync_align_to_analysis_frames。默认开启。开启时优先按实际输出 frames 目录中的分析帧做 GPS 对齐，通常更稳定。关闭时使用 legacy frame step 方式回退。

Geo Export WGS84 / 导出 WGS84 坐标：
对应 --geo_sync_export_wgs84。默认开启。项目会保留原始 GCJ-02 字段，同时导出近似 WGS84 字段，便于 GIS 软件使用。

Web Sync Prefer WGS84 / Web 优先 WGS84：
对应 --web_sync_prefer_wgs84。默认关闭。开启后 Web 地图数据优先使用 derived WGS84 坐标；如果展示底图使用国内地图服务，可能更适合保留 GCJ-02。

GIS Prefer WGS84 / GIS 优先 WGS84：
对应 --gis_export_prefer_wgs84。默认开启。适合 QGIS、ArcGIS 等常见 GIS 工具。

Post Only / 仅后处理：
对应 --post_only。通常与 Existing Output Folder 一起使用，只运行新多模态后处理阶段，不重跑原始视频高成本流程。

Resume Missing Only / 仅补缺失结果：
对应 --resume_missing_only。用于已有输出中缺少旧 AI/音频产物时补生成，已存在且有效的结果会跳过。

Run Deliverable Layer / 运行交付物层：
对应 --run_deliverable_layer。从已有 Step-8 / design 等结果生成最终问题路段交付包。适合 Existing Output Folder 模式。

Use GLM / 使用 GLM：
对应 --deliverable_use_glm。启用后会尝试使用 GLM 润色交付物文本；需要 apikey.env。没有 API key 时建议关闭，核心流程会尽量回退到模板模式。

Export Cards / 导出卡片：
对应 --deliverable_export_cards。为每个问题 episode 生成 PNG 卡片。会增加输出文件数量。

Render HTML / 渲染 HTML：
对应 --deliverable_render_html。生成交付物 HTML 总览页，适合浏览器查看。

Render PDF / 渲染 PDF：
对应 --deliverable_render_pdf。生成 contact sheet PDF。会增加处理时间和输出文件体积。

Deliverable Top K / 交付物 Top K：
对应 --deliverable_top_k。默认 12。控制参与 episode 合并的高优先级 segment 数量。数值越大，覆盖问题越多，但交付物更长。

Deliverable Max Gap / 交付物最大间隔：
对应 --deliverable_max_gap_seconds。默认 5.0 秒。用于把相邻高优先级 segment 合并成 episode。数值越大，更容易合并成较长问题段；数值越小，episode 更碎。

No Web / 不自动启动 Web：
对应 --no_web。默认建议开启。处理完成后不让 main.py 自动启动 Flask Web 服务，避免后台进程一直占用端口。需要查看 HTML 时使用 GUI 的 Open Web Page。

Regen Charts / 重新生成图表：
对应 --regen_charts。GUI 当前不作为常用按钮暴露，命令行可用。只对已有输出重新生成统计图表，不重新分析视频。适合图表文件丢失或样式更新后重建。

FFmpeg Path：
指定 ffmpeg.exe。缺失时视频转码、H.264 生成、音频提取可能失败。

Models Folder：
指定 models 目录。主要用于 PANNs 等本地模型资源；yolo11m.pt 也可放在 exe 同级或 models 下。

API Key Env File：
指定 apikey.env。仅在 GLM/Agents 等需要外部 API 的功能中使用。普通本地分析可不填。


五、高级命令行参数说明

以下参数来自 main.py，主要用于维护、研究流程或人工标注流程。普通用户通常通过 GUI 使用常用参数即可。

--web_host：
自动启动 Flask Web 服务时的监听地址，默认 127.0.0.1。普通本机查看不需要修改。

--web_port：
自动启动 Flask Web 服务时的端口，默认 5000。端口被占用时可修改。

--check_panns：
只检查本地 PANNs 声景模型资源，不运行视频分析。用于排查 soundscape 依赖。

--video_name：
当 --from_existing_output 指向 output 根目录而不是具体视频目录时，用于指定子目录名。

--launch_validation_web：
启动 Step-5 本地标注 Web，用于人工评审 segment 样本。

--validation_rater：
选择评审者 A 或 B，默认 A。仅用于 validation web。

--launch_adjudication_web：
启动 Step-5.5 裁决 Web，用于处理两位评审之间的分歧。

--enable_validation_pack：
生成双评审盲标注包。会在 validation 目录下创建评审 CSV 和管理清单。

--compute_validation_reliability：
计算两位评审的一致性指标，例如 ICC、Spearman、MAE、Kappa。

--finalize_validation_labels：
汇总两位评审结果，生成最终 segment 级标签。

--build_adjudication_pack：
根据分歧结果生成裁决包。

--finalize_adjudicated_labels：
使用裁决结果生成 final_annotation_labels_adjudicated.csv。

--validation_rater_a_csv / --validation_rater_b_csv：
手动指定 A、B 评审填写后的 CSV。

--validation_admin_csv：
指定 validation 的管理员清单 CSV。

--reliability_report_json：
指定 reliability_report.json 路径。

--adjudication_pack_csv：
指定 adjudication_pack.csv 路径。

--baseline_final_labels_csv：
指定基线 final_annotation_labels.csv 路径。

--run_step7_fusion_eval：
运行 Step-7 融合建模评估，只消费已有特征和标签，不重新跑视频分析。

--labels_csv：
Step-7/Step-7.5 使用的标签 CSV。默认优先使用 validation/final_annotation_labels_adjudicated.csv。

--feature_csv：
Step-7/Step-7.5 使用的特征 CSV。默认使用 fusion/model_feature_table.csv。

--step7_outdir：
Step-7 输出目录，默认 output/<video>/fusion_eval。

--step7_seed：
Step-7 随机种子，默认来自 src/config.py。调整会影响交叉验证和随机过程的可复现结果。

--step7_smoke_test：
Step-7 烟雾测试模式，减少重复次数，适合快速检查流程。

--step7_clean_outdir：
运行 Step-7 前清理 fusion_eval 输出目录。只清理该目录。

--step7_show_progress：
显示更详细的 Step-7 进度，并写出 step7_progress.json。

--run_step75_refined_eval：
运行 Step-7.5 精细化融合评估，不覆盖 Step-7 输出。

--step75_outdir：
Step-7.5 输出目录，默认 output/<video>/fusion_eval_refined。

--step75_seed：
Step-7.5 随机种子。

--step75_smoke_test：
Step-7.5 快速测试模式。

--reuse_step7_splits / --no-reuse_step7_splits：
控制 Step-7.5 是否复用 Step-7 外层 CV 划分。复用时结果更便于比较。

--run_step8_design_mapping：
运行 Step-8 设计映射，生成优先级排序、干预矩阵和设计提示。

--step8_outdir：
Step-8 输出目录，默认 output/<video>/design。

--step8_top_n：
只对优先级前 N 个 segment 生成完整设计计划。0 表示全部。数值越大，输出越完整但耗时更长。

--step8_smoke_test：
Step-8 快速测试模式，只处理少量高优先级片段。

--run_relationship_analysis：
运行声景-视觉关系分析，输出到 relationship 目录。

--relationship_outdir：
指定 relationship analysis 输出目录。

--run_proof_package：
运行融合优于单模态的证明包，输出到 proof 目录。

--proof_outdir：
指定 proof package 输出目录。

--run_group_confirmatory_relationship：
运行 group-level confirmatory relationship 分析。

--group_confirmatory_outdir：
指定 group confirmatory 输出目录。

--run_paper_figures：
生成论文图包，通常消费 relationship/proof 等已有结果。

--paper_figures_outdir：
指定 paper_figures 输出目录。

--deliverable_top_percent：
按优先级百分比选择 segment 参与 episode 合并。与 Top K 相比，更适合不同视频长度之间保持比例一致。

--deliverable_priority_threshold：
只保留 priority_score 不低于该阈值的 segment。阈值越高，交付物越精简。

--geo_sync_sidecar_path：
指定单视频 sidecar JSON，可用于覆盖视频开始时间或时间偏移。

--geo_sync_max_gap_warning_sec：
GPS 有效重叠窗口的最大 gap 警告阈值。默认沿用配置。数值越小越严格，更容易提示 GPS 不连续。

--geo_sync_filename_tz_offset_hours：
从文件名解析视频开始时间时使用的时区偏移，默认通常为 8 小时。

--geo_sync_use_existing_segments / --no-geo_sync_use_existing_segments：
控制 geo_sync 是否复用已有 segment_manifest.csv。复用通常更稳定。

--geo_sync_frame_step：
geo_sync legacy 采样步长；仅在不对齐实际分析帧时作为回退参数使用。

--export_debug_json / --no-export_debug_json：
控制扩展阶段是否写出调试 JSON。开启会增加输出文件数量，适合排查 pipeline 阶段问题；普通用户建议关闭。


六、GPS 表格格式说明

核心 geo_sync 读取 CSV 文件，必须包含以下字段：

groupTime：必填。Unix 秒级时间戳，例如 1750817698。程序会按 UTC 解析为 groupTime_utc。
gps_longitude：必填。经度，范围 -180 到 180。
gps_latitude：必填。纬度，范围 -90 到 90。

可选字段：

speed：速度。若存在会转为数值，并用于质量统计。
horizontalAccuracy：水平精度。若存在会转为数值，并保留到质量检查中。

坐标系：

代码中把输入经纬度视为 GCJ-02，并在启用 Geo Export WGS84 时额外导出近似 WGS84 字段。原始 GCJ-02 字段仍会保留。国内地图底图通常适合 GCJ-02；QGIS/ArcGIS 等 GIS 工具通常适合 WGS84。

示例 CSV：

groupTime,gps_longitude,gps_latitude,speed,horizontalAccuracy
1750817698,116.403456,39.937123,1.2,5.0
1750817699,116.403462,39.937130,1.3,5.2
1750817700,116.403470,39.937136,1.1,5.1

仓库中的 output_gps.sample.csv 是示例文件。实际运行时请在 GUI 中选择自己的 GPS 文件，或复制示例文件为 output_gps.csv 后替换为真实数据。

Excel 文件：

GUI 可选择 .xlsx / .xls，但运行前会转换为临时 CSV。Excel 第一行仍必须使用上述字段名。

常见错误：

如果使用 timestamp、latitude、longitude 这类字段名，核心流程不会自动识别。请改名为 groupTime、gps_latitude、gps_longitude。


七、输出结果说明

输出根目录下通常会为每个视频生成一个子目录，例如：

output/VID_20250625_101458_00_006/

常见结果包括：

frames：
抽取的分析帧。

split / reproj / mask：
投影、切分、语义分割等中间图像结果。

stats：
视觉统计、绿视率、颜色、情绪评分、人数统计等图表和 CSV。

ai_evaluation：
AI 活动评估相关 CSV 和图表。

audio_events：
传统音频分析结果。

segments：
segment_manifest.csv / json，片段索引和时间范围。

visual：
segment_visual_features.csv，片段级视觉摘要。

geo_sync：
frame_geo_metadata.csv、segment_geo_metadata.csv、geo_sync_summary.json。用于查看视频帧/片段与 GPS 的对齐结果。

soundscape：
音频片段特征和声景结果。

fusion：
segment_feature_table.csv、model_feature_table.csv 等多模态融合特征表。

diagnostics / design：
诊断、优先级排序和设计映射结果。

deliverable：
最终交付物目录。常见文件包括 problem_episodes.csv、problem_episode_summary.csv、problem_episode_cards.html、problem_episode_contact_sheet.pdf、deliverable_summary.md。

gis_export：
GIS 友好的 frame_gis_export.csv、segment_gis_export.csv、problem_episode_gis_export.csv。

web_sync：
用于网页联动展示的 JSON 数据。

multimodal：
多模态 pipeline 的运行 manifest、stage_status 和 summary。

_launcher_logs：
GUI 启动器写入的运行日志。运行失败时可把这里的 txt 发给维护人员排查。


八、常见问题与排错

1. GPS 文件无法读取

确认文件存在，CSV 编码正常，字段名为 groupTime、gps_longitude、gps_latitude。Excel 文件第一行也必须是这些字段名。

2. 视频和 GPS 对不上

优先调整 GPS Time Offset。若整体提前或滞后，说明时间偏移不合适。若局部跳动大，检查 GPS 是否有大时间间隔、重复时间、坐标漂移或设备时间错误。

3. Web 页面打不开

点击 Open Web Page，而不是直接双击 HTML。程序会启动本地 HTTP 服务。若仍打不开，检查安全软件、防火墙或端口占用。

4. 输出文件夹为空

检查输入视频是否存在、输出目录是否有写权限、ffmpeg.exe 和模型是否存在。查看 Log Panel 或 output/_launcher_logs 中的错误。

5. 运行速度太慢

提高 Frame Skip，先用快速测试模式确认流程。关闭 Soundscape、Fusion、Agents、Design、Deliverable 等高成本阶段。高质量完整模式适合最终正式运行，不适合试错。

6. 打包后的程序无法启动

确认没有只复制 exe。必须保留整个 StreetSmartEvaluator 文件夹，包括 _internal、models、web、config、ffmpeg.exe、yolo11m.pt。路径可以包含中文和空格，但不要放在没有写权限的系统目录中。

7. 找不到模型

确认 yolo11m.pt 在 exe 同级目录，或放在 models/yolo11m.pt。PANNs 声音模型应位于 models/panns。

8. 找不到 ffmpeg

确认 ffmpeg.exe 在 exe 同级目录，或在 Resource Settings 中手动指定。

9. API key 缺失

只有 Use GLM 或 Enable Agents 等 API 功能需要 apikey.env。复制 apikey.env.template 为 apikey.env，填入 ZHIPUAI_API_KEY 或 ZHIPU_API_KEY。

10. 程序运行后没有自动弹出网页

默认 No Web 开启，main.py 不会自动启动 Web 服务。请点击 Open Web Page 查看生成的 HTML。


九、便携版使用注意事项

这个程序是便携目录，不是单文件 exe。可以把整个 StreetSmartEvaluator 文件夹移动到其他电脑或其他路径。不要改动 _internal、web、config、models 等目录名。不要把真实 apikey.env 发给无关人员。
