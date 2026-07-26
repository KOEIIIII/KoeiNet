# 三部分结果说明（中文）

- 视觉：`vegetation_fraction`、`pedestrian_count`、`brightness` 的时间序列或分布图；视觉与主观评分的相关热图；低分/高分段的代表帧拼图以便人工核验。

## 空间映射

使用 `geo_sync/segment_geo_metadata.csv` 的 `matched_gps_*`（或 `_wgs84` 字段）在 Folium 或 GeoPandas 地图上可视化，按任一指标着色（如 `comfort_score`、Leq、`vegetation_fraction`、`priority_score`）。

示例（Folium 简单示例）：

```python
import folium
m = folium.Map(location=[df['matched_gps_latitude_gcj02'].mean(), df['matched_gps_longitude_gcj02'].mean()], zoom_start=15)
for _, r in df.iterrows():
    folium.CircleMarker([r['matched_gps_latitude_gcj02'], r['matched_gps_longitude_gcj02']], radius=3,
                        popup=f"seg:{r['segment_id']} comfort:{r.get('comfort_score', 'NA')}",
                        color='red' if r.get('comfort_score', 5) < 3 else 'green').add_to(m)
m.save('map.html')
```

## 注意事项

- GPS 的 `groupTime` 需为 Unix 秒；`geo_sync` 默认以 GCJ-02 处理坐标。若需要 WGS84，请在导出时启用 `export_wgs84=True`。
- 声景功能依赖可选依赖或本地模型；如遇缺失字段或出错，请确认你在运行 Program 01 时启用了 `--enable_soundscape` 并安装了 `requirements_optional_soundscape.txt` 中的依赖。
- 某些特征由逐帧聚合为段级（frame → segment），请在 `visual/segment_visual_features.csv` 的 header 注释中确认聚合方式（mean/median/max）。
- 主观评分应结合 `confidence_score` 和多位注释者的一致性指标（例如相关系数、ICC）来解读。

## 后续建议

- 使用合并表进行回归或重要性分析；Program 02 已提供基于 `configs/street_type_coefficients.yaml` 的 `priority_score` 计算逻辑。
- 导出 `merged.csv`、图表与 `map.html` 到 `deliverable/` 或 `problem_detection/` 以便共享与审查。

---

如果你希望我做更多：
- 我可以把一段可执行的 Python 脚本放到 `examples/` 中，自动读取 Program01/02 输出并生成一套 PNG 图表与 `map.html`；
- 或者把本中文/英文文档拆分为更短的 FAQ 风格条目，放到 docs/ 目录的子文件夹中以便翻译与维护。
