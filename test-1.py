


"""Thin CLI wrapper for video-to-GPS synchronization."""

from __future__ import annotations

import argparse
import json

from src.geo_sync import run_geo_sync


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Align sampled video frames with output_gps.csv and export structured GPS matches."
    )
    parser.add_argument("--video-path", help="Video path. If omitted, auto-discover the newest supported file in input/.")
    parser.add_argument("--input-dir", default="input", help="Input directory used for auto-discovery. Default: input")
    parser.add_argument("--gps-csv", default="output_gps.csv", help="GPS CSV path. Default: output_gps.csv")
    parser.add_argument("--output-root", default="output", help="Output root. Default: output")
    parser.add_argument(
        "--frame-step",
        type=int,
        default=60,
        help="Sample one frame every N decoded frames. Default: 60",
    )
    parser.add_argument(
        "--time-offset-seconds",
        type=float,
        default=25.0,
        help="Additional offset applied after resolved start time. Default keeps backward compatibility with old test-1.py.",
    )
    parser.add_argument(
        "--start-time",
        help="Explicit ISO-8601 video start time, e.g. 2025-06-25T02:14:58Z. Overrides metadata and filename.",
    )
    parser.add_argument(
        "--sidecar",
        help="Optional JSON sidecar with keys like start_time, time_offset_seconds, filename_tz_offset_hours.",
    )
    parser.add_argument(
        "--filename-tz-offset-hours",
        type=float,
        default=8.0,
        help="Timezone offset used when falling back to filename timestamps. Default: 8",
    )
    parser.add_argument("--segment-manifest", help="Optional existing segment_manifest.csv path.")
    parser.add_argument("--segment-seconds", type=float, default=5.0, help="Virtual segment length. Default: 5.0")
    parser.add_argument("--segment-overlap", type=float, default=2.5, help="Virtual segment overlap. Default: 2.5")
    parser.add_argument(
        "--max-gap-warning-sec",
        type=float,
        default=60.0,
        help="Maximum GPS gap that still counts as effective overlap. Default: 60.0",
    )
    parser.add_argument("--max-samples", type=int, help="Optional cap on exported sampled frames.")
    parser.set_defaults(export_wgs84=True, use_existing_segments=True)
    parser.add_argument(
        "--export-wgs84",
        dest="export_wgs84",
        action="store_true",
        help="Export derived approximate WGS84 fields in addition to raw GCJ-02.",
    )
    parser.add_argument(
        "--no-export-wgs84",
        dest="export_wgs84",
        action="store_false",
        help="Do not export derived WGS84 fields; keep only GCJ-02.",
    )
    parser.add_argument(
        "--use-existing-segments",
        dest="use_existing_segments",
        action="store_true",
        help="Reuse output/<video>/segments/segment_manifest.csv when available.",
    )
    parser.add_argument(
        "--no-use-existing-segments",
        dest="use_existing_segments",
        action="store_false",
        help="Ignore existing segment manifests and build virtual segments instead.",
    )
    parser.add_argument("--display", action="store_true", help="Show sampled debug frames with GPS overlay.")
    parser.add_argument(
        "--save-preview-count",
        type=int,
        default=0,
        help="Save the first N annotated preview images under output/<video>/geo_sync/previews/. Default: 0",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    summary = run_geo_sync(
        video_path=args.video_path,
        input_dir=args.input_dir,
        gps_csv_path=args.gps_csv,
        output_root=args.output_root,
        frame_step=args.frame_step,
        start_time=args.start_time,
        sidecar_path=args.sidecar,
        time_offset_seconds=args.time_offset_seconds,
        filename_tz_offset_hours=args.filename_tz_offset_hours,
        segment_manifest_path=args.segment_manifest,
        segment_seconds=args.segment_seconds,
        segment_overlap=args.segment_overlap,
        max_samples=args.max_samples,
        display=args.display,
        save_preview_count=args.save_preview_count,
        export_wgs84=args.export_wgs84,
        max_gap_warning_sec=args.max_gap_warning_sec,
        use_existing_segments=args.use_existing_segments,
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
