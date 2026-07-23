"""
Regex editor popup — appears when a dropped folder can't be parsed
by any built-in regex, or when the user clicks "Edit regex" from
the mask app.

Features:

* Editable regex line + "Try" button.
* Live tabular preview of the first 10 filenames parsed by the
  current regex.
* "Auto detect" button (heuristic — see
  :func:`spacr.qt.regex_detect.auto_detect_regex`).
* Dropdown to pick a built-in as a starting point.
* "Save" writes the regex into the target AppScreen's ``custom_regex``
  setting; "Cancel" closes without changes.

The dialog is pure UI — all logic lives in
:mod:`spacr.qt.regex_detect`.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QHBoxLayout, QLabel,
    QLineEdit, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget,
)

from . import regex_detect as rd

LOG = logging.getLogger("spacr.qt.regex_editor")


class RegexEditorDialog(QDialog):
    """Modal editor for the filename metadata regex.

    :param sample_filenames: bare filenames from the dropped folder
        used for the live preview (first 20 or so).
    :param initial_regex: pre-fill the input with this pattern; empty
        means we run auto-detect on show.
    :param multichannel: True for regular multi-channel plates; False
        for single-channel data (relaxes the warning set).
    :param parent: optional parent widget.
    :ivar regex: the final chosen regex; only meaningful after
        :py:meth:`exec` returned ``QDialog.Accepted``.
    """

    def __init__(
        self,
        sample_filenames: List[str],
        initial_regex: str = "",
        multichannel: bool = True,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("spaCR — Regex editor")
        self.setMinimumSize(760, 520)
        self.regex: str = ""
        self._samples = list(sample_filenames)[:20]
        self._multi = multichannel

        outer = QVBoxLayout(self)

        intro = QLabel(
            "<b>Filename metadata regex</b><br>"
            "<span style='color:gray;'>Named groups map onto spaCR's "
            "downstream fields — most importantly "
            "<code>plateID</code>, <code>wellID</code>, "
            "<code>fieldID</code>, and <code>chanID</code>.</span>"
        )
        intro.setTextFormat(Qt.RichText)
        intro.setWordWrap(True)
        outer.addWidget(intro)

        # ─── Regex input row ────────────────────────────────────────
        row = QHBoxLayout()
        self._regex_input = QLineEdit()
        mono = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        self._regex_input.setFont(mono)
        self._regex_input.setPlaceholderText(
            r"e.g. (?P<plateID>.*)_(?P<wellID>[A-Z]\d{2})_..."
        )
        self._regex_input.textChanged.connect(self._on_regex_changed)
        row.addWidget(self._regex_input, 1)

        self._auto_btn = QPushButton("Auto detect")
        self._auto_btn.clicked.connect(self._on_auto_detect)
        row.addWidget(self._auto_btn)

        wrap = QWidget(); wrap.setLayout(row)
        outer.addWidget(wrap)

        # ─── Preset dropdown ────────────────────────────────────────
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Or start from a preset:"))
        self._preset_combo = QComboBox()
        for label in rd.BUILTIN_REGEXES:
            self._preset_combo.addItem(label, label)
        self._preset_combo.addItem("(custom)", None)
        self._preset_combo.currentIndexChanged.connect(self._on_preset_pick)
        preset_row.addWidget(self._preset_combo)
        preset_row.addStretch(1)
        prewrap = QWidget(); prewrap.setLayout(preset_row)
        outer.addWidget(prewrap)

        # ─── Warnings + preview ──────────────────────────────────────
        self._warnings_lbl = QLabel("")
        self._warnings_lbl.setTextFormat(Qt.RichText)
        self._warnings_lbl.setWordWrap(True)
        outer.addWidget(self._warnings_lbl)

        outer.addWidget(QLabel("Preview:"))
        self._preview = QPlainTextEdit()
        self._preview.setReadOnly(True)
        self._preview.setFont(mono)
        outer.addWidget(self._preview, 1)

        # ─── Buttons ─────────────────────────────────────────────────
        buttons = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self._on_save)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

        # ─── Initial state ───────────────────────────────────────────
        if initial_regex:
            self._regex_input.setText(initial_regex)
        else:
            self._on_auto_detect()

    # -- reactive updates ------------------------------------------------
    def _on_regex_changed(self, text: str) -> None:
        # Match the dropdown to the current text if it matches a preset
        for i in range(self._preset_combo.count()):
            key = self._preset_combo.itemData(i)
            if key is not None and rd.BUILTIN_REGEXES.get(key) == text:
                self._preset_combo.blockSignals(True)
                self._preset_combo.setCurrentIndex(i)
                self._preset_combo.blockSignals(False)
                break
        else:
            # Custom
            self._preset_combo.blockSignals(True)
            self._preset_combo.setCurrentIndex(
                self._preset_combo.count() - 1
            )
            self._preset_combo.blockSignals(False)
        self._refresh_preview()

    def _on_preset_pick(self, _idx: int) -> None:
        key = self._preset_combo.currentData()
        if key is None:
            return
        pattern = rd.BUILTIN_REGEXES.get(key, "")
        self._regex_input.blockSignals(True)
        self._regex_input.setText(pattern)
        self._regex_input.blockSignals(False)
        self._refresh_preview()

    def _on_auto_detect(self) -> None:
        pattern, label, hits = rd.auto_detect_regex(self._samples)
        if pattern:
            self._regex_input.setText(pattern)
            self._preview.appendPlainText(
                f"[auto] chose regex `{label}` — matched {hits}/"
                f"{len(self._samples)} sampled filenames.\n"
            )
        else:
            self._preview.setPlainText(
                "[auto] no regex could be inferred from the sample."
            )

    def _refresh_preview(self) -> None:
        pattern = self._regex_input.text()
        records, missed = rd.apply_regex(self._samples, pattern)
        warnings = rd.validate_records(records, multichannel=self._multi)
        if warnings:
            self._warnings_lbl.setText(
                "<b style='color:#d29922;'>⚠ Warnings</b><br>"
                + "<br>".join(f"• {w}" for w in warnings)
            )
        else:
            self._warnings_lbl.setText(
                "<span style='color:#3fb950;'>✓ All required fields "
                "captured.</span>"
            )
        preview = rd.tabulate_records(records, max_rows=10)
        if missed:
            preview += (f"\n\n[{len(missed)} filenames did NOT match — "
                        f"first: {missed[0]}]")
        self._preview.setPlainText(preview)

    # -- accept ---------------------------------------------------------
    def _on_save(self) -> None:
        self.regex = self._regex_input.text().strip()
        self.accept()
