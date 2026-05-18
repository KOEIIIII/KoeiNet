


from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
APP_NAME = "StreetSmartEvaluator"
DIST_APP = ROOT / "dist" / APP_NAME
BUILD_APP = ROOT / "build" / APP_NAME
SPEC_FILE = ROOT / f"{APP_NAME}.spec"


README_TEXT = "Double-click StreetSmartEvaluator.exe. Keep the whole StreetSmartEvaluator folder together."


ENV_TEMPLATE = """ZHIPUAI_API_KEY=
ZHIPU_API_KEY=
ZHIPU_AGENT_MODEL=glm-5
"""


EXAMPLE_CONFIG = {
    "run_mode": "existing",
    "preset": "deliverable_existing",
    "input_path": "",
    "existing_output": "output/VID_20250625_101458_00_006",
    "gps_file": "",
    "output_root": "output",
    "frame_skip": 20,
    "post_only": True,
    "run_deliverable_layer": True,
    "deliverable_use_glm": False,
    "deliverable_export_cards": True,
    "deliverable_render_html": True,
    "deliverable_render_pdf": True,
    "no_web": True,
}


def remove_path(path: Path) -> None:
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def copy_optional_tree(src: Path, dst: Path) -> None:
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")
        shutil.copytree(src, dst, ignore=ignore)


def copy_optional_file(src: Path, dst: Path) -> None:
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def run_pyinstaller() -> None:
    command = [sys.executable, "-m", "PyInstaller", "--noconfirm", str(SPEC_FILE)]
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def stage_external_resources() -> None:
    if not DIST_APP.is_dir():
        raise FileNotFoundError(f"PyInstaller output not found: {DIST_APP}")

    copy_optional_file(ROOT / "ffmpeg.exe", DIST_APP / "ffmpeg.exe")
    copy_optional_file(ROOT / "yolo11m.pt", DIST_APP / "yolo11m.pt")
    copy_optional_tree(ROOT / "models", DIST_APP / "models")
    copy_optional_tree(ROOT / "config", DIST_APP / "config")
    copy_optional_tree(ROOT / "web", DIST_APP / "web")
    copy_optional_tree(ROOT / "docs", DIST_APP / "docs")

    (DIST_APP / "apikey.env.template").write_text(ENV_TEMPLATE, encoding="utf-8")
    readme_source = ROOT / "README_使用说明.txt"
    readme_text = readme_source.read_text(encoding="utf-8") if readme_source.is_file() else README_TEXT
    (DIST_APP / "README_使用说明.txt").write_text(readme_text, encoding="utf-8")
    (DIST_APP / "example_config.json").write_text(
        json.dumps(EXAMPLE_CONFIG, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    try:
        remove_path(DIST_APP)
        remove_path(BUILD_APP)
        run_pyinstaller()
        stage_external_resources()
    except subprocess.CalledProcessError as exc:
        print(f"Build failed while running PyInstaller. Return code: {exc.returncode}")
        return int(exc.returncode or 1)
    except Exception as exc:
        print(f"Build failed: {exc}")
        return 1

    print(f"Build complete: {DIST_APP}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
