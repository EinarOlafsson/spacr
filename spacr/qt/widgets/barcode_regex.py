"""Interactive editor for sequencing barcode regular expressions.

Barcode regexes are not filename regexes.  The sequencing pipeline requires
three specifically named capture groups (``columnID``, ``grna`` and
``rowID``), so the generic filename-regex editor cannot validate this field.
This module keeps the compact settings row while providing a focused dialog
that compiles the expression and previews its captures against a pasted read.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict

from PySide6.QtCore import Qt, Signal

from ..i18n import tr
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QSizePolicy,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ...settings import DEFAULT_BARCODE_REGEX


REQUIRED_BARCODE_GROUPS = ("columnID", "grna", "rowID")

# A real column barcode, gRNA and row barcode from the bundled reference
# tables, surrounded by the constant sequence expected by the shipped regex.
EXAMPLE_BARCODE_READ = (
    "TCATAGGCTGCTGTAAACGTACTATAATGATATTACGAC"
    "AACTTAGAAGCAATGTCG"
)


@dataclass(frozen=True)
class BarcodeRegexResult:
    """Validation result shown by the inline field and test dialog."""

    valid: bool
    message: str
    captures: Dict[str, str]


def evaluate_barcode_regex(pattern: str, sample: str = "") -> BarcodeRegexResult:
    """Compile a barcode regex and optionally apply it to one sequence.

    A usable mapping expression must define every named group consumed by
    :mod:`spacr.sequencing`.  When a sample is supplied it must also match and
    produce a non-empty value for each required group.
    """
    pattern = str(pattern or "").strip()
    if not pattern:
        return BarcodeRegexResult(False, "Enter a regular expression.", {})
    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        return BarcodeRegexResult(False, f"Regex error: {exc}", {})

    missing = [
        name for name in REQUIRED_BARCODE_GROUPS
        if name not in compiled.groupindex
    ]
    if missing:
        return BarcodeRegexResult(
            False,
            "Missing named group" + ("s" if len(missing) != 1 else "")
            + ": " + ", ".join(missing),
            {},
        )

    sample = "".join(str(sample or "").split())
    if not sample:
        return BarcodeRegexResult(
            True,
            "Valid regex with columnID, grna and rowID groups.",
            {},
        )

    match = compiled.match(sample)
    if match is None:
        return BarcodeRegexResult(False, "The sample sequence did not match.", {})
    captures = {
        name: match.group(name) or ""
        for name in REQUIRED_BARCODE_GROUPS
    }
    empty = [name for name, value in captures.items() if not value]
    if empty:
        return BarcodeRegexResult(
            False,
            "Matched, but these groups were empty: " + ", ".join(empty),
            captures,
        )
    return BarcodeRegexResult(True, "Sample matched successfully.", captures)


class BarcodeRegexDialog(QDialog):
    """Edit and test a barcode regex against a representative read.

    :param initial_regex: the expression to open with. Empty opens the field
        empty; the dialog proposes nothing until the user tests a read.
    :param parent: parent widget.
    """

    def __init__(self, initial_regex: str = "", parent=None):
        """Build the barcode-regex tester.

        Sized in scaled pixels rather than raw ones: a size set from Python does
        not grow with the stylesheet's font size, and at the 200% scale the
        prose inside wrapped to more height than the window had.

        :param initial_regex: the pattern to open with; empty uses spaCR's
            default.
        :param parent: parent widget, or ``None``.
        """
        super().__init__(parent)
        self.setWindowTitle("spaCR — Barcode regex tester")
        from ..preferences import scaled_px

        # SIZED IN SCALED PIXELS, NOT RAW ONES. A dialog size set from
        # Python does not grow when the stylesheet's font size does, so at
        # the 200%% font scale the prose inside this window wrapped to more
        # height than the window had and the last line was cut off. The
        # size-policy fix on the label was necessary and not sufficient:
        # a policy stops a parent handing a label less than it asks for, but
        # it cannot make a window grow that has no room to give.
        self.setMinimumSize(scaled_px(760), scaled_px(430))
        self.regex = ""

        outer = QVBoxLayout(self)
        intro = QLabel(
            "<b>Barcode extraction regex</b><br>"
            "The expression must contain the named groups "
            "<code>columnID</code>, <code>grna</code> and "
            "<code>rowID</code>. Paste one extracted read window below to "
            "see exactly what will be written to the mapping table."
        )
        intro.setTextFormat(Qt.RichText)
        intro.setWordWrap(True)
        # A WRAPPED LABEL NEEDS (Preferred, Minimum): with Qt's default
        # Preferred height a parent is free to hand it less than its
        # heightForWidth. This is the house rule `prerun._label` documents.
        # NECESSARY BUT NOT SUFFICIENT HERE -- 350's sweep still reports this
        # label clipped at 2.0x, because the container above it does not grow
        # either. See 350; the remaining fix is the dialog's layout, not this.
        intro.setSizePolicy(QSizePolicy.Preferred,
                               QSizePolicy.Minimum)
        outer.addWidget(intro)

        mono = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        self._regex_input = QLineEdit(self)
        self._regex_input.setObjectName("BarcodeRegexPattern")
        self._regex_input.setFont(mono)
        self._regex_input.setText(initial_regex or DEFAULT_BARCODE_REGEX)
        outer.addWidget(self._regex_input)

        actions = QHBoxLayout()
        self._default_button = QPushButton("Reset spaCR default", self)
        self._example_button = QPushButton(tr("Load test data"), self)
        actions.addWidget(self._default_button)
        actions.addWidget(self._example_button)
        actions.addStretch(1)
        outer.addLayout(actions)

        outer.addWidget(QLabel("Sample read or extracted sequence window:"))
        self._sample_input = QPlainTextEdit(self)
        self._sample_input.setObjectName("BarcodeRegexSample")
        self._sample_input.setFont(mono)
        self._sample_input.setMaximumHeight(90)
        self._sample_input.setPlaceholderText(
            "Paste a DNA sequence here, or click “Load test data”."
        )
        outer.addWidget(self._sample_input)

        self._status = QLabel(self)
        self._status.setObjectName("BarcodeRegexStatus")
        self._status.setWordWrap(True)
        # A WRAPPED LABEL NEEDS (Preferred, Minimum): with Qt's default
        # Preferred height a parent is free to hand it less than its
        # heightForWidth. This is the house rule `prerun._label` documents.
        # NECESSARY BUT NOT SUFFICIENT HERE -- 350's sweep still reports this
        # label clipped at 2.0x, because the container above it does not grow
        # either. See 350; the remaining fix is the dialog's layout, not this.
        self._status.setSizePolicy(QSizePolicy.Preferred,
                                      QSizePolicy.Minimum)
        outer.addWidget(self._status)

        outer.addWidget(QLabel("Captured values:"))
        self._captures = QPlainTextEdit(self)
        self._captures.setObjectName("BarcodeRegexCaptures")
        self._captures.setReadOnly(True)
        self._captures.setFont(mono)
        outer.addWidget(self._captures, 1)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel,
            parent=self,
        )
        self._buttons.accepted.connect(self._save)
        self._buttons.rejected.connect(self.reject)
        outer.addWidget(self._buttons)

        self._regex_input.textChanged.connect(self._refresh)
        self._sample_input.textChanged.connect(self._refresh)
        self._default_button.clicked.connect(
            lambda: self._regex_input.setText(DEFAULT_BARCODE_REGEX)
        )
        self._example_button.clicked.connect(
            lambda: self._sample_input.setPlainText(EXAMPLE_BARCODE_READ)
        )
        self._refresh()

    def _refresh(self) -> None:
        """Re-evaluate the pattern against the pasted read and show the captures.

        The status line carries the verdict and is repolished so its style
        follows it, and the captures pane shows exactly what would be written to
        the mapping table.
        """
        result = evaluate_barcode_regex(
            self._regex_input.text(),
            self._sample_input.toPlainText(),
        )
        symbol = "✓" if result.valid else "⚠"
        self._status.setText(f"{symbol} {result.message}")
        self._status.setProperty("valid", result.valid)
        self._status.style().unpolish(self._status)
        self._status.style().polish(self._status)
        if result.captures:
            self._captures.setPlainText(
                "\n".join(
                    f"{name:<8} {result.captures.get(name, '')}"
                    for name in REQUIRED_BARCODE_GROUPS
                )
            )
        else:
            self._captures.setPlainText(
                "Add a sample sequence to preview the three captures."
            )
        self._buttons.button(QDialogButtonBox.Save).setEnabled(result.valid)

    def _save(self) -> None:
        """Accept the pattern, but only while it is valid.

        An invalid pattern does nothing rather than closing: saving one that
        cannot compile would fail on the next run instead of here.
        """
        result = evaluate_barcode_regex(
            self._regex_input.text(),
            self._sample_input.toPlainText(),
        )
        if not result.valid:
            return
        self.regex = self._regex_input.text().strip()
        self.accept()


class BarcodeRegexWidget(QWidget):
    """Compact settings-row field with inline validation and a test dialog.

    :param value: the regular expression already saved. Empty opens the field
        empty rather than with a suggestion, so a blank setting stays blank
        until the user or the test dialog fills it.
    :param parent: parent widget.
    """

    valueChanged = Signal(str)

    def __init__(self, value: str = "", parent=None):
        """Build the inline regex field with its verdict mark and Test button.

        :param value: the pattern to start with.
        :param parent: parent widget, or ``None``.
        """
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._line_edit = QLineEdit(self)
        self._line_edit.setObjectName("BarcodeRegexLineEdit")
        self._line_edit.setFont(
            QFontDatabase.systemFont(QFontDatabase.FixedFont)
        )
        self._line_edit.setText(value or "")
        layout.addWidget(self._line_edit, 1)

        self._status = QLabel(self)
        self._status.setObjectName("BarcodeRegexInlineStatus")
        self._status.setFixedWidth(18)
        layout.addWidget(self._status)

        self._test_button = QToolButton(self)
        self._test_button.setObjectName("BarcodeRegexTestButton")
        self._test_button.setText("Test…")
        self._test_button.setToolTip(
            "Compile this regex and preview column, gRNA and row captures."
        )
        layout.addWidget(self._test_button)

        self._line_edit.textChanged.connect(self._on_text_changed)
        self._test_button.clicked.connect(self._open_tester)
        self._on_text_changed(self._line_edit.text())

    def _on_text_changed(self, text: str) -> None:
        """Re-evaluate the typed pattern and update the inline verdict.

        The verdict's full message is the mark's tooltip, so why a pattern is
        refused is reachable without opening the tester.

        :param text: the pattern now in the field.
        """
        result = evaluate_barcode_regex(text)
        self._status.setText("✓" if result.valid else "⚠")
        self._status.setToolTip(result.message)
        self.valueChanged.emit(text)

    def _open_tester(self) -> None:
        """Open the full tester on the current pattern and take back what it saves."""
        dialog = BarcodeRegexDialog(self.get_value() or "", parent=self)
        if dialog.exec() == QDialog.Accepted:
            self.set_value(dialog.regex)

    def get_value(self):
        """Return the edited regex, or ``None`` for a blank field."""
        return self._line_edit.text().strip() or None

    def set_value(self, value) -> None:
        """Replace the current regex and immediately refresh validation."""
        self._line_edit.setText("" if value is None else str(value))
