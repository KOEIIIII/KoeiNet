"""Build Program 02 with PyInstaller in one-folder mode."""

from __future__ import annotations

import subprocess
import sys
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Program 02 with PyInstaller.")
    parser.add_argument("--dry-run", action="store_true", help="Print the PyInstaller command without executing it.")
    parser.add_argument("--check", action="store_true", help="Check whether PyInstaller can be imported.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.check:
        try:
            import PyInstaller  # type: ignore

            print(f"PyInstaller available: {getattr(PyInstaller, '__version__', 'unknown')}")
            return 0
        except Exception as exc:
            print(f"PyInstaller check failed: {exc}")
            return 1
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--onedir",
        "--windowed",
        "--name",
        "Program02_ScoringProblemDetection",
        "--hidden-import",
        "core.problem_detection.detector",
        "--collect-submodules",
        "src",
        "--add-data",
        f"{ROOT / 'configs'};configs",
        "--add-data",
        f"{ROOT / 'web'};web",
        "--add-data",
        f"{ROOT / 'examples'};examples",
        "--add-data",
        f"{ROOT / 'docs'};docs",
        str(ROOT / "apps" / "program_02_scoring_problem_detection" / "gui.py"),
    ]
    print("Running:", subprocess.list2cmdline(command))
    if args.dry_run:
        return 0
    return subprocess.run(command, cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
