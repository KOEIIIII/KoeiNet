"""Fast smoke tests for the two deliverable programs."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], *, env: dict[str, str] | None = None) -> tuple[int, str]:
    proc = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )
    return proc.returncode, proc.stdout


def record(results: list[dict[str, str]], name: str, ok: bool, detail: str) -> None:
    results.append({"test": name, "status": "passed" if ok else "failed", "detail": detail.strip()})


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def main() -> int:
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    results: list[dict[str, str]] = []
    sample_output = ROOT / "examples" / "sample_outputs" / "minimal_program01_output"
    sample_gps = ROOT / "examples" / "sample_inputs" / "sample_gps.csv"
    work_dir = ROOT / "_tmp_smoke" / "script_smoke"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    code, out = run([sys.executable, "-m", "py_compile", "scripts/run_program_01.py", "scripts/run_program_02.py"])
    record(results, "entrypoint_compile", code == 0, out or "compiled")

    code, out = run(
        [
            sys.executable,
            "scripts/run_program_01.py",
            "--dry_run",
            "--from_existing_output",
            str(sample_output),
            "--post_only",
        ]
    )
    record(results, "program_01_cli_dry_run", code == 0 and "main.py" in out, "Program 01 dry run completed.")

    record(results, "sample_gps_present", sample_gps.is_file(), rel(sample_gps))
    record(
        results,
        "sample_visualization_present",
        (sample_output / "web" / "index.html").is_file(),
        rel(sample_output / "web" / "index.html"),
    )

    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    code, out = run(
        [
            sys.executable,
            "-c",
            "from PySide6.QtWidgets import QApplication; from apps.program_01_analysis_visualization.gui import Program01Window; app=QApplication([]); w=Program01Window(); w.language.setCurrentIndex(1); print(w.windowTitle()); app.quit()",
        ],
        env=env,
    )
    record(results, "program_01_gui_start_and_language", code == 0 and "Program 01" in out, "Program 01 GUI initialized and language switch executed.")

    out_dir = work_dir / "program02_detection"
    code, out = run(
        [
            sys.executable,
            "scripts/run_program_02.py",
            "--program_01_output",
            str(sample_output),
            "--output_dir",
            str(out_dir),
            "--top_k",
            "2",
            "--priority_threshold",
            "0.4",
            "--no_update_visualization_artifacts",
        ]
    )
    ok = code == 0 and (out_dir / "segment_problem_priority.csv").is_file() and (
        out_dir / "problem_episodes.csv"
    ).is_file()
    record(results, "program_02_detection", ok, "Program 02 detection generated priority and episode outputs.")

    code, out = run(
        [
            sys.executable,
            "-c",
            "from PySide6.QtWidgets import QApplication; from apps.program_02_scoring_problem_detection.gui import Program02Window; app=QApplication([]); w=Program02Window(); w.language.setCurrentIndex(1); print(w.windowTitle()); app.quit()",
        ],
        env=env,
    )
    record(results, "program_02_gui_start_and_language", code == 0 and "Program 02" in out, "Program 02 GUI initialized and language switch executed.")

    for script in ["scripts/build_program_01.py", "scripts/build_program_02.py"]:
        code, out = run([sys.executable, script, "--dry-run"])
        record(results, f"{Path(script).stem}_dry_run", code == 0 and "PyInstaller" in out, f"{script} dry run completed.")

    report = {"results": results}
    report_path = ROOT / "_tmp_smoke" / "smoke_test_results.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.rmtree(work_dir, ignore_errors=True)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if all(item["status"] == "passed" for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
