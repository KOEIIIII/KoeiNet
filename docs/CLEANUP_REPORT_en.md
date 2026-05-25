# Cleanup Report

Date: 2026-05-25

## Goal

This cleanup prepares the project for a GitHub source repository while preserving local delivery assets, source code, configuration, sample data, and documentation.

## Removed

- Removed `insta360_segmentation.log`: runtime log.
- Removed PyInstaller `build/` and root `.spec` files: reproducible packaging intermediates.
- Removed `dist/*/release_smoke/`: temporary executable smoke outputs.
- Removed `_tmp_smoke/`: temporary test outputs.
- Removed `__pycache__/` folders outside `venv/`, `dist/`, and `archive_unused/`.
- Removed obsolete archived build folders, release packages, PyInstaller intermediates, historical smoke outputs, archived caches, and generated spec files from `archive_unused/`.
- Removed `archive_unused/local_apikey.env`: local API key file that should not be retained or uploaded.

## Archived

The following files are not suitable for a GitHub source repository but may be useful locally, so they were moved under `archive_unused/`. This directory is ignored by Git.

- `input/` -> `archive_unused/local_input_raw_video/`: local raw video data.
- `output/` -> `archive_unused/local_output_runtime_results/`: historical full runtime outputs.
- `output_gps.csv` -> `archive_unused/local_output_gps.csv`: local GPS output.
- `output_gps.sample.csv` -> `archive_unused/legacy_output_gps_sample.csv`: legacy sample schema replaced by `examples/sample_inputs/sample_gps.csv`.
- `docs/output_cleanup_strategy.md` and the legacy technical document -> `archive_unused/legacy_docs/`: older notes that no longer fully match the two-application delivery structure.

## Kept

- `apps/`, `core/`, `src/`, `web/`, `configs/`, `scripts/`: source code and entry points.
- `examples/sample_inputs/`: minimal video and GPS samples.
- `examples/sample_outputs/minimal_program01_output/`: minimal Program 01 output for Program 02.
- `dist/Program01_AnalysisVisualization/` and `dist/Program02_ScoringProblemDetection/`: local double-click applications. `dist/` is ignored by Git; publish packaged apps through Releases or a separate archive.
- `ffmpeg.exe`, `yolo11m.pt`, `models/`, `config/model_dir/`: local external runtime resources. These large assets are ignored by Git.
- Chinese and English README files, program guides, test reports, and cleanup reports.

## .gitignore

The root `.gitignore` now covers:

- Python caches and test caches.
- Virtual environments.
- IDE and OS metadata.
- API keys, tokens, env files, and local configuration.
- Runtime logs and temporary directories.
- Local inputs, outputs, and archive folders.
- PyInstaller build/dist/spec byproducts.
- Large models, weights, and media files.
- An explicit exception for the curated tiny sample video `examples/sample_inputs/VID_20250625_101458_00_006.mp4`.

## Impact

Program 01 and Program 02 source entry points, GUIs, configs, sample data, packaging scripts, and local packaged executables remain available. Post-cleanup smoke tests passed, and no broken paths were found.

## Before Uploading To GitHub

Run:

```powershell
git status --short
python scripts\smoke_test.py
```

Do not commit `archive_unused/`, `dist/`, `venv/`, `models/`, `input/`, `output/`, or real GPS/video data. Publish packaged desktop applications through GitHub Releases, Git LFS, or a separate delivery archive.
