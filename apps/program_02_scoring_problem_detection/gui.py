"""Program 02 desktop interface."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def app_root() -> Path:
    return Path(sys.executable).resolve().parent if is_frozen() else Path(__file__).resolve().parents[2]


APP_DIR = app_root()
RESOURCE_ROOT = Path(getattr(sys, "_MEIPASS", APP_DIR))
ROOT = RESOURCE_ROOT
if str(RESOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(RESOURCE_ROOT))

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.problem_detection import ANNOTATION_COLUMNS, create_annotation_template, run_problem_detection


DISPLAY_COLUMNS = ["segment_id", "start_time_sec", "end_time_sec"] + [
    column for column in ANNOTATION_COLUMNS if column != "segment_id"
]


TEXT = {
    "en": {
        "title": "Program 02 - Scoring, Fusion, and Problem Detection",
        "subtitle": "Stage 2: load Program 01 outputs, edit annotations, tune coefficients, and export problem segments.",
        "language": "Language",
        "input": "Program 01 Output",
        "annotation": "Annotation CSV",
        "coeff": "Coefficient Config",
        "output": "Output Folder",
        "browse": "Browse",
        "load": "Load",
        "create": "Create Annotation File",
        "save_annotation": "Save Annotation",
        "run": "Run Problem Detection",
        "export": "Export Results",
        "open_output": "Open Output Folder",
        "open_web": "Open Visualization",
        "restore": "Restore Defaults",
        "save_coeff": "Save Coefficients",
        "load_coeff": "Load Coefficients",
        "paths": "Inputs",
        "params": "Detection Settings",
        "top_k": "Top K",
        "threshold": "Priority Threshold",
        "gap": "Max Gap Seconds",
        "update_viz": "Update visualization-compatible artifacts",
        "annotation_tab": "Annotation",
        "coeff_tab": "Coefficients",
        "log_tab": "Log",
    },
    "zh": {
        "title": "Program 02 - 人工评分、多模态融合与问题路段识别",
        "subtitle": "第二阶段：加载 Program 01 输出，编辑人工评分，配置系数并导出问题路段。",
        "language": "语言",
        "input": "Program 01 输出",
        "annotation": "评分文件",
        "coeff": "系数配置",
        "output": "输出文件夹",
        "browse": "浏览",
        "load": "加载",
        "create": "创建评分文件",
        "save_annotation": "保存评分",
        "run": "运行问题路段识别",
        "export": "导出结果",
        "open_output": "打开输出文件夹",
        "open_web": "打开可视化网页",
        "restore": "恢复默认系数",
        "save_coeff": "保存系数",
        "load_coeff": "加载系数",
        "paths": "输入",
        "params": "识别参数",
        "top_k": "Top K",
        "threshold": "优先级阈值",
        "gap": "最大合并间隔（秒）",
        "update_viz": "回写可视化兼容结果",
        "annotation_tab": "人工评分",
        "coeff_tab": "系数配置",
        "log_tab": "日志",
    },
}


class Program02Window(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.lang = "en"
        self.annotation_df = pd.DataFrame(columns=DISPLAY_COLUMNS)
        self.coeff_payload: dict[str, Any] = {}
        self.setWindowTitle(TEXT[self.lang]["title"])
        self.resize(1180, 820)
        self._build_ui()
        self._apply_style()
        self._load_defaults()
        self._translate()

    def tr(self, key: str) -> str:
        return TEXT[self.lang][key]

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QWidget { font-family: Segoe UI, Microsoft YaHei, sans-serif; font-size: 13px; color: #18202a; }
            QGroupBox { border: 1px solid #d6dee8; border-radius: 8px; margin-top: 12px; padding: 14px; font-weight: 700; }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; }
            QLineEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTableWidget { border: 1px solid #c8d1dc; border-radius: 6px; background: white; }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { padding: 7px; }
            QPushButton { border: 1px solid #b8c4d2; border-radius: 6px; padding: 8px 12px; background: #f7f9fc; }
            QPushButton:hover { background: #edf4ff; }
            QPushButton#primary { background: #1f6feb; color: white; border-color: #1f6feb; font-weight: 700; }
            QLabel#title { font-size: 24px; font-weight: 800; }
            QLabel#subtitle { color: #5d6b7a; }
            """
        )

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        header = QHBoxLayout()
        title_block = QVBoxLayout()
        self.title = QLabel()
        self.title.setObjectName("title")
        self.subtitle = QLabel()
        self.subtitle.setObjectName("subtitle")
        title_block.addWidget(self.title)
        title_block.addWidget(self.subtitle)
        self.language = QComboBox()
        self.language.addItem("English", "en")
        self.language.addItem("中文", "zh")
        self.language.currentIndexChanged.connect(self._change_language)
        header.addLayout(title_block, 1)
        self.language_label = QLabel()
        header.addWidget(self.language_label)
        header.addWidget(self.language)
        root.addLayout(header)

        paths_group = QGroupBox()
        paths = QGridLayout(paths_group)
        self.program_01_output = QLineEdit()
        self.annotation_csv = QLineEdit()
        self.coeff_config = QLineEdit()
        self.output_dir = QLineEdit()
        for row, (edit, slot) in enumerate(
            [
                (self.program_01_output, self._browse_program_output),
                (self.annotation_csv, self._browse_annotation),
                (self.coeff_config, self._browse_coeff),
                (self.output_dir, self._browse_output),
            ]
        ):
            label = QLabel()
            btn = QPushButton()
            btn.clicked.connect(slot)
            paths.addWidget(label, row, 0)
            paths.addWidget(edit, row, 1)
            paths.addWidget(btn, row, 2)
            setattr(self, f"path_label_{row}", label)
            setattr(self, f"path_button_{row}", btn)
        root.addWidget(paths_group)
        self.paths_group = paths_group

        params_group = QGroupBox()
        params = QHBoxLayout(params_group)
        self.top_k = QSpinBox()
        self.top_k.setRange(0, 10000)
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0, 1)
        self.threshold.setDecimals(3)
        self.threshold.setSingleStep(0.01)
        self.gap = QDoubleSpinBox()
        self.gap.setRange(0, 3600)
        self.gap.setDecimals(2)
        self.update_viz = QCheckBox()
        for key, widget in [("top_k", self.top_k), ("threshold", self.threshold), ("gap", self.gap)]:
            label = QLabel()
            setattr(self, f"param_label_{key}", label)
            params.addWidget(label)
            params.addWidget(widget)
        params.addWidget(self.update_viz)
        params.addStretch(1)
        root.addWidget(params_group)
        self.params_group = params_group

        actions = QHBoxLayout()
        self.load_btn = QPushButton()
        self.create_btn = QPushButton()
        self.save_annotation_btn = QPushButton()
        self.run_btn = QPushButton()
        self.run_btn.setObjectName("primary")
        self.export_btn = QPushButton()
        self.open_output_btn = QPushButton()
        self.open_web_btn = QPushButton()
        for btn in [
            self.load_btn,
            self.create_btn,
            self.save_annotation_btn,
            self.run_btn,
            self.export_btn,
            self.open_output_btn,
            self.open_web_btn,
        ]:
            actions.addWidget(btn)
        actions.addStretch(1)
        root.addLayout(actions)

        self.tabs = QTabWidget()
        self.annotation_table = QTableWidget()
        self.coeff_table = QTableWidget()
        coeff_widget = QWidget()
        coeff_layout = QVBoxLayout(coeff_widget)
        coeff_actions = QHBoxLayout()
        self.load_coeff_btn = QPushButton()
        self.save_coeff_btn = QPushButton()
        self.restore_coeff_btn = QPushButton()
        coeff_actions.addWidget(self.load_coeff_btn)
        coeff_actions.addWidget(self.save_coeff_btn)
        coeff_actions.addWidget(self.restore_coeff_btn)
        coeff_actions.addStretch(1)
        coeff_layout.addLayout(coeff_actions)
        coeff_layout.addWidget(self.coeff_table)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.tabs.addTab(self.annotation_table, "")
        self.tabs.addTab(coeff_widget, "")
        self.tabs.addTab(self.log, "")
        root.addWidget(self.tabs, 1)

        self.load_btn.clicked.connect(self._load_all)
        self.create_btn.clicked.connect(self._create_annotation)
        self.save_annotation_btn.clicked.connect(self._save_annotation)
        self.run_btn.clicked.connect(self._run_detection)
        self.export_btn.clicked.connect(self._run_detection)
        self.open_output_btn.clicked.connect(self._open_output)
        self.open_web_btn.clicked.connect(self._open_visualization)
        self.load_coeff_btn.clicked.connect(self._load_coeff)
        self.save_coeff_btn.clicked.connect(self._save_coeff)
        self.restore_coeff_btn.clicked.connect(self._restore_coeff)

    def _load_defaults(self) -> None:
        default_output = APP_DIR / "output" / "VID_20250625_101458_00_006"
        if default_output.is_dir():
            self.program_01_output.setText(str(default_output))
            self.annotation_csv.setText(str(default_output / "validation" / "final_annotation_labels_adjudicated.csv"))
            self.output_dir.setText(str(default_output / "problem_detection"))
        self.coeff_config.setText(str(RESOURCE_ROOT / "configs" / "street_type_coefficients.yaml"))
        self.top_k.setValue(12)
        self.threshold.setValue(0.45)
        self.gap.setValue(5.0)
        self.update_viz.setChecked(True)
        self._load_coeff()
        if self.program_01_output.text():
            self._try_load_annotation()

    def _translate(self) -> None:
        self.setWindowTitle(self.tr("title"))
        self.title.setText(self.tr("title"))
        self.subtitle.setText(self.tr("subtitle"))
        self.language_label.setText(self.tr("language"))
        self.paths_group.setTitle(self.tr("paths"))
        for row, key in enumerate(["input", "annotation", "coeff", "output"]):
            getattr(self, f"path_label_{row}").setText(self.tr(key))
            getattr(self, f"path_button_{row}").setText(self.tr("browse"))
        self.params_group.setTitle(self.tr("params"))
        self.param_label_top_k.setText(self.tr("top_k"))
        self.param_label_threshold.setText(self.tr("threshold"))
        self.param_label_gap.setText(self.tr("gap"))
        self.update_viz.setText(self.tr("update_viz"))
        self.load_btn.setText(self.tr("load"))
        self.create_btn.setText(self.tr("create"))
        self.save_annotation_btn.setText(self.tr("save_annotation"))
        self.run_btn.setText(self.tr("run"))
        self.export_btn.setText(self.tr("export"))
        self.open_output_btn.setText(self.tr("open_output"))
        self.open_web_btn.setText(self.tr("open_web"))
        self.load_coeff_btn.setText(self.tr("load_coeff"))
        self.save_coeff_btn.setText(self.tr("save_coeff"))
        self.restore_coeff_btn.setText(self.tr("restore"))
        self.tabs.setTabText(0, self.tr("annotation_tab"))
        self.tabs.setTabText(1, self.tr("coeff_tab"))
        self.tabs.setTabText(2, self.tr("log_tab"))

    def _change_language(self) -> None:
        self.lang = str(self.language.currentData())
        self._translate()

    def _browse_program_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, self.tr("input"), self.program_01_output.text() or str(APP_DIR / "output"))
        if path:
            self.program_01_output.setText(path)
            self.annotation_csv.setText(str(Path(path) / "validation" / "final_annotation_labels_adjudicated.csv"))
            self.output_dir.setText(str(Path(path) / "problem_detection"))

    def _browse_annotation(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, self.tr("annotation"), self.program_01_output.text() or str(APP_DIR), "CSV (*.csv);;All files (*)")
        if path:
            self.annotation_csv.setText(path)

    def _browse_coeff(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, self.tr("coeff"), str(RESOURCE_ROOT / "configs"), "Config (*.yaml *.yml *.json);;All files (*)")
        if path:
            self.coeff_config.setText(path)

    def _browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, self.tr("output"), self.output_dir.text() or str(APP_DIR))
        if path:
            self.output_dir.setText(path)

    def _read_config(self, path: Path) -> dict[str, Any]:
        text = path.read_text(encoding="utf-8")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            import yaml  # type: ignore

            payload = yaml.safe_load(text)
        return payload if isinstance(payload, dict) else {}

    def _load_all(self) -> None:
        self._try_load_annotation()
        self._load_coeff()

    def _try_load_annotation(self) -> None:
        path = Path(self.annotation_csv.text())
        if not path.is_file():
            self._log(f"Annotation CSV not found yet: {path}")
            return
        self.annotation_df = pd.read_csv(path)
        for column in ANNOTATION_COLUMNS:
            if column not in self.annotation_df.columns:
                self.annotation_df[column] = ""
            display_columns = [column for column in DISPLAY_COLUMNS if column in self.annotation_df.columns]
            for column in DISPLAY_COLUMNS:
                if column not in display_columns:
                    self.annotation_df[column] = ""
            self.annotation_df = self.annotation_df[DISPLAY_COLUMNS]
        self._render_annotation()
        self._log(f"Loaded annotation: {path}")

    def _render_annotation(self) -> None:
        df = self.annotation_df.fillna("")
        self.annotation_table.setRowCount(len(df))
        columns = list(self.annotation_df.columns)
        self.annotation_table.setColumnCount(len(columns))
        self.annotation_table.setHorizontalHeaderLabels(columns)
        for r, (_, row) in enumerate(df.iterrows()):
            for c, column in enumerate(columns):
                item = QTableWidgetItem(str(row.get(column, "")))
                if column in {"segment_id", "start_time_sec", "end_time_sec"}:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.annotation_table.setItem(r, c, item)
        self.annotation_table.resizeColumnsToContents()

    def _collect_annotation(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        columns = [self.annotation_table.horizontalHeaderItem(c).text() for c in range(self.annotation_table.columnCount())]
        for r in range(self.annotation_table.rowCount()):
            row = {}
            for c, column in enumerate(columns):
                item = self.annotation_table.item(r, c)
                row[column] = item.text() if item else ""
            rows.append(row)
        return pd.DataFrame(rows, columns=columns)

    def _create_annotation(self) -> None:
        try:
            target = create_annotation_template(self.program_01_output.text(), self.annotation_csv.text() or None)
            self.annotation_csv.setText(target)
            self._try_load_annotation()
        except Exception as exc:
            QMessageBox.critical(self, "Program 02", str(exc))

    def _save_annotation(self) -> None:
        path = Path(self.annotation_csv.text())
        if not path:
            QMessageBox.warning(self, "Program 02", "Annotation CSV path is required.")
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        self.annotation_df = self._collect_annotation()
        self.annotation_df.to_csv(path, index=False, encoding="utf-8-sig")
        self._log(f"Saved annotation: {path}")

    def _load_coeff(self) -> None:
        path = Path(self.coeff_config.text())
        if not path.is_file():
            return
        self.coeff_payload = self._read_config(path)
        self._render_coeff()
        self._log(f"Loaded coefficients: {path}")

    def _restore_coeff(self) -> None:
        self.coeff_config.setText(str(RESOURCE_ROOT / "configs" / "street_type_coefficients.yaml"))
        self._load_coeff()

    def _render_coeff(self) -> None:
        street_types = self.coeff_payload.get("street_types", {})
        rows = []
        for street_type, cfg in street_types.items():
            if not isinstance(cfg, dict):
                continue
            coeffs = cfg.get("coefficients", {})
            rows.append((street_type, "severity_threshold", cfg.get("severity_threshold", "")))
            if isinstance(coeffs, dict):
                for key, value in coeffs.items():
                    rows.append((street_type, key, value))
        self.coeff_table.setRowCount(len(rows))
        self.coeff_table.setColumnCount(3)
        self.coeff_table.setHorizontalHeaderLabels(["street_type", "parameter", "value"])
        for r, (street_type, key, value) in enumerate(rows):
            self.coeff_table.setItem(r, 0, QTableWidgetItem(str(street_type)))
            self.coeff_table.setItem(r, 1, QTableWidgetItem(str(key)))
            self.coeff_table.setItem(r, 2, QTableWidgetItem(str(value)))
        self.coeff_table.resizeColumnsToContents()

    def _collect_coeff(self) -> dict[str, Any]:
        payload = dict(self.coeff_payload)
        street_types = payload.setdefault("street_types", {})
        for r in range(self.coeff_table.rowCount()):
            street_type = self.coeff_table.item(r, 0).text().strip()
            key = self.coeff_table.item(r, 1).text().strip()
            value_text = self.coeff_table.item(r, 2).text().strip()
            try:
                value: Any = float(value_text)
            except ValueError:
                value = value_text
            cfg = street_types.setdefault(street_type, {"coefficients": {}})
            if key == "severity_threshold":
                cfg[key] = value
            else:
                cfg.setdefault("coefficients", {})[key] = value
        return payload

    def _save_coeff(self) -> None:
        path = Path(self.coeff_config.text())
        path.parent.mkdir(parents=True, exist_ok=True)
        self.coeff_payload = self._collect_coeff()
        path.write_text(json.dumps(self.coeff_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self._log(f"Saved coefficients: {path}")

    def _run_detection(self) -> None:
        try:
            self._save_annotation()
            self._save_coeff()
            result = run_problem_detection(
                video_dir=self.program_01_output.text(),
                annotation_csv=self.annotation_csv.text() or None,
                coefficient_config=self.coeff_config.text(),
                output_dir=self.output_dir.text() or None,
                top_k=self.top_k.value() or None,
                priority_threshold=self.threshold.value() if self.threshold.value() > 0 else None,
                max_gap_seconds=self.gap.value(),
                update_visualization_artifacts=self.update_viz.isChecked(),
            )
            self._log(json.dumps(result, ensure_ascii=False, indent=2))
            QMessageBox.information(self, "Program 02", "Problem detection completed.")
        except Exception as exc:
            self._log(f"ERROR: {exc}")
            QMessageBox.critical(self, "Program 02", str(exc))

    def _open_output(self) -> None:
        path = Path(self.output_dir.text() or Path(self.program_01_output.text()) / "problem_detection")
        path.mkdir(parents=True, exist_ok=True)
        os.startfile(str(path.resolve()))

    def _open_visualization(self) -> None:
        root = Path(self.program_01_output.text())
        html = sorted(root.rglob("*.html"), key=lambda p: (0 if p.name == "index.html" else 1, len(p.parts)))
        if html:
            os.startfile(str(html[0].resolve()))
        else:
            QMessageBox.information(self, "Program 02", "No visualization HTML was found.")

    def _log(self, text: str) -> None:
        self.log.appendPlainText(text)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--smoke-test":
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        app = QApplication([sys.argv[0]])
        win = Program02Window()
        win.language.setCurrentIndex(1)
        sample = RESOURCE_ROOT / "examples" / "sample_outputs" / "minimal_program01_output"
        out_dir = APP_DIR / "release_smoke" / "program02_detection"
        out_dir.mkdir(parents=True, exist_ok=True)
        annotation = out_dir / "smoke_annotation.csv"
        create_annotation_template(str(sample), str(annotation))
        win.program_01_output.setText(str(sample))
        win.annotation_csv.setText(str(annotation))
        win.coeff_config.setText(str(RESOURCE_ROOT / "configs" / "street_type_coefficients.yaml"))
        win.output_dir.setText(str(out_dir))
        win._load_all()
        if win.annotation_table.rowCount() > 0:
            notes_col = [win.annotation_table.horizontalHeaderItem(c).text() for c in range(win.annotation_table.columnCount())].index("annotator_notes")
            win.annotation_table.setItem(0, notes_col, QTableWidgetItem("smoke test annotation save"))
        win._save_annotation()
        result = run_problem_detection(
            video_dir=str(sample),
            annotation_csv=str(annotation),
            coefficient_config=str(RESOURCE_ROOT / "configs" / "street_type_coefficients.yaml"),
            output_dir=str(out_dir),
            top_k=2,
            priority_threshold=0.4,
            max_gap_seconds=5.0,
            update_visualization_artifacts=False,
        )
        ok = (
            sample.is_dir()
            and annotation.is_file()
            and int(result.get("segments_scored", 0)) >= 3
            and (out_dir / "segment_problem_priority.csv").is_file()
            and "Program 02" in win.windowTitle()
        )
        print(f"program02_smoke={'passed' if ok else 'failed'}")
        app.quit()
        return 0 if ok else 1
    app = QApplication(sys.argv[:1] + argv)
    win = Program02Window()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
