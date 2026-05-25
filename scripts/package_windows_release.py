"""Create split Windows release packages for GitHub Releases.

The two PyInstaller applications share an identical ``_internal`` runtime
folder. This packager writes a single combined package with both executable
entry points and one shared runtime, then splits the resulting zip into
GitHub-friendly parts.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
DEFAULT_OUT = ROOT / "release_packages" / "KoeiNet_Windows"
PACKAGE_ROOT = "KoeiNet_Windows"


class SplitWriter:
    """A minimal non-seekable writer that splits bytes across part files."""

    def __init__(self, output_base: Path, part_size: int) -> None:
        self.output_base = output_base
        self.part_size = part_size
        self.part_index = 0
        self.part_offset = 0
        self.total = 0
        self.current = None
        self.parts: list[Path] = []
        self._open_next_part()

    def writable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    def tell(self) -> int:
        return self.total

    def flush(self) -> None:
        if self.current:
            self.current.flush()

    def close(self) -> None:
        if self.current:
            self.current.close()
            self.current = None

    def _open_next_part(self) -> None:
        if self.current:
            self.current.close()
        self.part_index += 1
        self.part_offset = 0
        part = self.output_base.with_suffix(self.output_base.suffix + f".part{self.part_index:03d}")
        part.parent.mkdir(parents=True, exist_ok=True)
        self.current = part.open("wb")
        self.parts.append(part)

    def write(self, data: bytes) -> int:
        view = memoryview(data)
        written = 0
        while written < len(view):
            if self.part_offset >= self.part_size:
                self._open_next_part()
            chunk_size = min(len(view) - written, self.part_size - self.part_offset)
            self.current.write(view[written : written + chunk_size])
            written += chunk_size
            self.part_offset += chunk_size
            self.total += chunk_size
        return len(data)


def parse_size(value: str) -> int:
    text = value.strip().lower()
    units = {"k": 1024, "m": 1024**2, "g": 1024**3}
    if text[-1:] in units:
        return int(float(text[:-1]) * units[text[-1]])
    return int(text)


def iter_files(path: Path) -> list[Path]:
    return [p for p in path.rglob("*") if p.is_file()]


def add_file(zf: zipfile.ZipFile, src: Path, arcname: str) -> None:
    info = zipfile.ZipInfo.from_file(src, arcname)
    info.compress_type = zf.compression
    zf.write(src, arcname)


def write_readme(out_dir: Path, part_base_name: str) -> None:
    readme = out_dir / "README_FIRST.txt"
    readme.write_text(
        f"""KoeiNet Windows Desktop Applications

Download all files whose names start with:

  {part_base_name}.part

Also download:

  Join_And_Extract.bat

Put all downloaded files in the same folder, then double-click
Join_And_Extract.bat. After extraction, open:

  KoeiNet_Windows\\Program01_AnalysisVisualization.exe
  KoeiNet_Windows\\Program02_ScoringProblemDetection.exe

Do not move the exe files away from the KoeiNet_Windows folder. The shared
_internal folder contains required runtime libraries and resources.

中文说明：

请下载所有以 {part_base_name}.part 开头的分卷文件，并下载
Join_And_Extract.bat。把它们放在同一个文件夹中，双击
Join_And_Extract.bat 自动合并并解压。解压完成后，打开：

  KoeiNet_Windows\\Program01_AnalysisVisualization.exe
  KoeiNet_Windows\\Program02_ScoringProblemDetection.exe

不要把 exe 单独移出 KoeiNet_Windows 文件夹，因为 _internal 文件夹中包含
程序运行所需的依赖库和资源。
""",
        encoding="utf-8",
    )


def write_joiner(out_dir: Path, archive_name: str, part_base_name: str) -> None:
    joiner = out_dir / "Join_And_Extract.bat"
    joiner.write_text(
        f"""@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
echo KoeiNet Windows package extractor
echo.
if exist "{archive_name}" del "{archive_name}"
copy /b "{part_base_name}.part*" "{archive_name}" >nul
if errorlevel 1 (
  echo Failed to join package parts. Please make sure all .part files are in this folder.
  pause
  exit /b 1
)
echo Extracting {archive_name} ...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -LiteralPath '{archive_name}' -DestinationPath '.' -Force"
if errorlevel 1 (
  echo PowerShell extraction failed. Please install 7-Zip and extract {archive_name} manually.
  pause
  exit /b 1
)
echo.
echo Done. Open KoeiNet_Windows\\Program01_AnalysisVisualization.exe or KoeiNet_Windows\\Program02_ScoringProblemDetection.exe
pause
""",
        encoding="ascii",
    )


def build_package(out_dir: Path, part_size: int, compression: int, dry_run: bool) -> int:
    program01 = DIST / "Program01_AnalysisVisualization"
    program02 = DIST / "Program02_ScoringProblemDetection"
    exe01 = program01 / "Program01_AnalysisVisualization.exe"
    exe02 = program02 / "Program02_ScoringProblemDetection.exe"
    internal01 = program01 / "_internal"
    internal02 = program02 / "_internal"

    required = [exe01, exe02, internal01, internal02]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print("Missing required packaged files:")
        for item in missing:
            print(f"  - {item}")
        return 1

    files01 = sorted(p.relative_to(internal01).as_posix() for p in iter_files(internal01))
    files02 = sorted(p.relative_to(internal02).as_posix() for p in iter_files(internal02))
    shared_runtime = files01 == files02
    if not shared_runtime:
        print("Warning: Program 01 and Program 02 runtime folders are not identical.")
        print("This script will still package Program 01 runtime and both executables.")

    runtime_size = sum(p.stat().st_size for p in iter_files(internal01))
    exe_size = exe01.stat().st_size + exe02.stat().st_size
    total_input = runtime_size + exe_size
    part_count_estimate = math.ceil(total_input / part_size)

    print(f"Shared runtime: {shared_runtime}")
    print(f"Input size: {total_input / (1024 ** 3):.2f} GiB")
    print(f"Split size: {part_size / (1024 ** 2):.0f} MiB")
    print(f"Estimated uncompressed part count: {part_count_estimate}")
    if dry_run:
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    archive_name = "KoeiNet_Windows.zip"
    archive_base = out_dir / archive_name
    for old in out_dir.glob("KoeiNet_Windows.zip.part*"):
        old.unlink()
    if archive_base.exists():
        archive_base.unlink()

    writer = SplitWriter(archive_base, part_size)
    try:
        with zipfile.ZipFile(
            writer,
            "w",
            compression=compression,
            compresslevel=1 if compression == zipfile.ZIP_DEFLATED else None,
            allowZip64=True,
        ) as zf:
            zf.write(exe01, f"{PACKAGE_ROOT}/Program01_AnalysisVisualization.exe")
            zf.write(exe02, f"{PACKAGE_ROOT}/Program02_ScoringProblemDetection.exe")
            for src in iter_files(internal01):
                rel = src.relative_to(internal01).as_posix()
                zf.write(src, f"{PACKAGE_ROOT}/_internal/{rel}")
    finally:
        writer.close()

    write_readme(out_dir, archive_name)
    write_joiner(out_dir, archive_name, archive_name)

    print("Created release package parts:")
    for part in writer.parts:
        print(f"  - {part.name} ({part.stat().st_size / (1024 ** 2):.1f} MiB)")
    print(f"  - README_FIRST.txt")
    print(f"  - Join_And_Extract.bat")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Package KoeiNet Windows apps into split release assets.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--part-size", default="1800m", help="Part size such as 1800m or 1.8g.")
    parser.add_argument("--store", action="store_true", help="Use no compression for faster packaging.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    compression = zipfile.ZIP_STORED if args.store else zipfile.ZIP_DEFLATED
    return build_package(args.out_dir, parse_size(args.part_size), compression, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
