"""GUI-level smoke flow for Program 01 and Program 02.

The script runs Qt in offscreen mode, simulates user-selected paths, exercises
language switching, annotation save/reload, coefficient save/restore, problem
detection, and open-path handlers without leaving persistent test outputs.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pandas as pd
from PySide6.QtWidgets import QApplication, QMessageBox, QTableWidgetItem

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.program_01_analysis_visualization.gui import Program01Window
from apps.program_02_scoring_problem_detection.gui import Program02Window


def _no_dialog(*args, **kwargs):
    return QMessageBox.StandardButton.Ok


def _set_cell(table, row: int, column_name: str, value: str) -> None:
    columns = [table.horizontalHeaderItem(c).text() for c in range(table.columnCount())]
    table.setItem(row, columns.index(column_name), QTableWidgetItem(value))


def main() -> int:
    QMessageBox.information = _no_dialog
    QMessageBox.warning = _no_dialog
    QMessageBox.critical = _no_dialog
    opened: list[str] = []

    def fake_startfile(path: str) -> None:
        opened.append(str(path))

    os.startfile = fake_startfile  # type: ignore[attr-defined]

    work_dir = ROOT / "_tmp_smoke" / "gui_flow"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    app = QApplication([])
    try:
        sample_output = ROOT / "examples" / "sample_outputs" / "minimal_program01_output"
        sample_video = ROOT / "examples" / "sample_inputs" / "VID_20250625_101458_00_006.mp4"
        sample_gps = ROOT / "examples" / "sample_inputs" / "sample_gps.csv"

        p1 = Program01Window()
        p1.video.setText(str(sample_video))
        p1.gps.setText(str(sample_gps))
        p1.output.setText(str(sample_output))
        p1.language.setCurrentIndex(1)
        p1.enable_soundscape.setChecked(False)
        p1._update_command()
        p1._open_path(sample_output)
        p1._open_visualization()

        p2 = Program02Window()
        annotation = work_dir / "annotation.csv"
        coeff = work_dir / "coefficients.json"
        detection = work_dir / "problem_detection"
        p2.program_01_output.setText(str(sample_output))
        p2.annotation_csv.setText(str(annotation))
        p2.coeff_config.setText(str(ROOT / "configs" / "street_type_coefficients.yaml"))
        p2.output_dir.setText(str(detection))
        p2.language.setCurrentIndex(1)
        p2._create_annotation()
        p2._load_all()

        rows = [
            ("mixed_use", "4", "3", "3", "3", "2", "[]", "no_major_problem", "4", "smoke ok"),
            ("commercial", "2", "2", "2", "5", "4", "traffic_noise;pedestrian_discomfort", "traffic_noise", "4", "smoke problem"),
            ("residential", "3", "2", "2", "4", "4", "low_vitality", "low_vitality", "3", "smoke weak activity"),
        ]
        for row_index, values in enumerate(rows[: p2.annotation_table.rowCount()]):
            for column, value in zip(
                [
                    "street_type",
                    "comfort_score",
                    "vitality_score",
                    "soundscape_pleasantness",
                    "soundscape_eventfulness",
                    "overall_problem_severity",
                    "main_problem_labels",
                    "primary_problem_label",
                    "confidence_score",
                    "annotator_notes",
                ],
                values,
            ):
                _set_cell(p2.annotation_table, row_index, column, value)
        p2._save_annotation()
        p2._try_load_annotation()
        _set_cell(p2.annotation_table, 0, "annotator_notes", "smoke reload edit")
        p2._save_annotation()

        p2._load_coeff()
        p2.coeff_table.setItem(0, 2, QTableWidgetItem("0.44"))
        p2.coeff_config.setText(str(coeff))
        p2._save_coeff()
        p2._restore_coeff()
        p2.coeff_config.setText(str(coeff))
        p2._load_coeff()

        p2.top_k.setValue(2)
        p2.threshold.setValue(0.4)
        p2.gap.setValue(5.0)
        p2.update_viz.setChecked(False)
        p2._run_detection()
        p2._open_output()
        p2._open_visualization()

        priority = detection / "segment_problem_priority.csv"
        episodes = detection / "problem_episodes.csv"
        summary = detection / "problem_detection_summary.md"
        scored = pd.read_csv(priority)
        result = {
            "program01_title": p1.windowTitle(),
            "program01_command_has_stage_flags": "--enable_segment_pipeline" in p1.command.toPlainText(),
            "program02_title": p2.windowTitle(),
            "annotation_rows": int(pd.read_csv(annotation).shape[0]),
            "coeff_saved": coeff.is_file(),
            "priority_exists": priority.is_file(),
            "episodes_exists": episodes.is_file(),
            "summary_exists": summary.is_file(),
            "segments_scored": int(scored.shape[0]),
            "problem_segments": int((scored["is_problem_segment"] == True).sum()),
            "open_calls": opened,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        ok = (
            "基础数据分析" in result["program01_title"]
            and result["program01_command_has_stage_flags"]
            and "问题路段识别" in result["program02_title"]
            and result["annotation_rows"] >= 3
            and result["coeff_saved"]
            and result["priority_exists"]
            and result["episodes_exists"]
            and result["summary_exists"]
            and result["segments_scored"] >= 3
            and result["problem_segments"] >= 1
            and len(opened) >= 4
        )
        return 0 if ok else 1
    finally:
        app.quit()
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
