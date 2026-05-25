# Test Report

Date: 2026-05-25

## Environment

- OS: Windows / PowerShell
- Python: `.\venv\Scripts\python.exe`
- GPU: NVIDIA GeForce RTX 4060
- Packager: PyInstaller 6.20.0
- Packaged applications:
  - `dist/Program01_AnalysisVisualization/Program01_AnalysisVisualization.exe`
  - `dist/Program02_ScoringProblemDetection/Program02_ScoringProblemDetection.exe`

## Test Data

- Minimal video: `examples/sample_inputs/VID_20250625_101458_00_006.mp4`
- Minimal GPS: `examples/sample_inputs/sample_gps.csv`
- Program 02 sample input: `examples/sample_outputs/minimal_program01_output`
- Annotation tests used valid simulated 1-5 scores and labels such as `traffic_noise`, `pedestrian_discomfort`, and `low_vitality`.

## What Was Run

1. Rebuilt Program 01 and Program 02.
2. Ran the real Program 01 minimal-video flow.
3. Ran GUI flow smoke tests for path selection simulation, Chinese switching, open handlers, annotation, coefficient editing, and problem detection.
4. Ran the project smoke test.
5. Ran `--smoke-test` on both packaged executables.
6. Removed temporary smoke outputs, executable release-smoke outputs, PyInstaller build/spec files, and Python caches.

## Program 01 Results

- GUI initialization passed, including English/Chinese switching.
- The command preview now includes advanced stage flags, confirming GUI stage toggles are passed to the CLI wrapper.
- The minimal video and GPS file were processed by the real Program 01 flow without interruption.
- The run generated frames, visual statistics, audio-event outputs, segment manifest, visual segment features, GPS alignment, GIS exports, and web-sync files.
- Stages `segment`, `visual`, `geo_sync`, `gis_export`, and `web_sync` completed successfully.
- Open Output Folder and Open Visualization path handlers were verified in the offscreen GUI flow.

## Program 02 Results

- GUI initialization passed, including English/Chinese switching.
- Program 02 loaded the Program 01 sample output.
- The test created an annotation CSV, filled simulated scores for 3 segments, saved it, reloaded it, edited it, and saved again.
- Coefficients were loaded, modified, saved, and restored to defaults.
- Problem detection completed successfully.
- Result: 3 segments scored, 2 problem segments detected, and 1 problem episode generated.
- The flow generated `segment_problem_priority.csv`, `problem_episodes.csv`, `problem_detection_summary.md`, and a run record.

## Packaging Verification

- Program 01 one-folder build completed.
- Program 02 one-folder build completed.
- Both packaged executables completed `--smoke-test`.
- A visible no-argument double-click-equivalent launch was attempted, but the desktop-launch approval did not return before timeout in the automation environment. This item is not claimed as passed. Executable smoke tests and offscreen GUI startup covered packaged entry points and resource paths.

## Packaging Warnings

The following environment warnings appeared during packaging but did not block builds or smoke tests:

- Ultralytics could not write settings/cache files under the user profile.
- The offline environment skipped auto-installing the optional `lap>=0.5.12` dependency.
- PyInstaller reported unresolved optional Numba TBB library `tbb12.dll`.

## Limits

- This test validates workflow, paths, and file I/O, not model accuracy.
- The minimal video is only 3 seconds and does not represent runtime behavior for large real panoramic videos.
- A final visible-window manual check is still recommended on the demonstration machine.
