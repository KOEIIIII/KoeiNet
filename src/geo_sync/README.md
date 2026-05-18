# geo_sync

`geo_sync` aligns video time with `output_gps.csv` and exports frame-level plus
segment-level coordinate metadata that can be merged back into the project by
`segment_id`.

## Coordinate fields

- Source GPS is treated as `GCJ-02`.
- Raw GCJ-02 values are preserved and remain authoritative.
- Approximate derived WGS84 fields can be exported for GIS interoperability.
- Unqualified longitude/latitude fields are intentionally removed from the CSV
  outputs to avoid silent CRS confusion.

Primary fields:

- `before_gps_longitude_gcj02`
- `before_gps_latitude_gcj02`
- `after_gps_longitude_gcj02`
- `after_gps_latitude_gcj02`
- `matched_gps_longitude_gcj02`
- `matched_gps_latitude_gcj02`
- `*_wgs84` counterparts when `export_wgs84=True`

CRS metadata fields:

- `source_coordinate_system`
- `derived_coordinate_system`
- `coordinate_system_source`
- `wgs84_conversion_method`

## Matching semantics

Each frame/segment row includes:

- before/after GPS bracket points
- an interpolated representative point
- `match_status`
- `confidence`
- `within_effective_overlap_window`
- `used_interpolation`

The interpolated matched point is the recommended field for downstream
segment-level spatial anchoring.

## Offset strategy

`time_offset_seconds` is intentionally per-video, not a project-wide truth.

Recommended order:

1. Store calibrated values in a per-video sidecar JSON
2. Use CLI override for one-off validation or research calibration
3. Fall back to config only as a convenience default

Summary JSON records both the configured default and the actually applied
offset.

## Segment alignment

`geo_sync` should preferentially reuse the real segment system instead of
inventing virtual segment ids.

Current preferred order:

1. Reuse `output/<video>/segments/segment_manifest.csv` when it exists
2. Only fall back to virtual segments when the real manifest is unavailable

For downstream merge, the canonical segment-level table is:

- `output/<video>/geo_sync/segment_geo_metadata.csv`

It is intended to be merged with:

- `output/<video>/segments/segment_manifest.csv`
- `output/<video>/soundscape/audio_segment_features.csv`
- `output/<video>/fusion/segment_feature_table.csv`
- `output/<video>/visual/segment_visual_features.csv`

All of the tables above should share the same `segment_id`.

## Partial-overlap semantics

Clips do not need to cover the full GPS track.

The expected workflow is:

1. Search across the full GPS table
2. Detect the local effective overlap window covered by the clip
3. Export `within_effective_overlap_window` for each frame/segment row

So a clip that only matches the middle portion of the full GPS trace is still a
successful synchronization case.
