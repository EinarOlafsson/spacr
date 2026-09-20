"""The window the FEATURES button in Make Masks opens.

Three things stacked, and the order is the order the questions come in:

1. :class:`spacr.qt.widgets.measure_input_table.MeasureInputTable` -- which
   file is which channel, and which is which object.
2. The Measure module's OWN settings form, built from
   :class:`spacr.qt.screens.settings_model.SettingsWidgets`. Not a copy of it
   and not a chosen subset: the same widgets the Measure screen builds, from
   the same defaults, so a setting that exists there exists here and a
   setting that gains a control there gains one here without this file being
   edited.
3. Run, off the GUI thread, through
   :func:`spacr.measure.measure_from_field_table` -- which writes the folders
   the Mask module would have written and then calls ``measure_crop``
   itself. The database and the folder tree are Measure's because Measure
   makes them.

THE SETTINGS THE TABLE DECIDES ARE SHOWN AND DISABLED rather than hidden.
``cell_mask_dim`` is not a question once the table says which column holds
the cell masks, but a user who has read the Measure documentation comes here
looking for it, and a control that is missing reads as a feature that is
missing. Shown, filled in, and not editable says the table already answered
it. :data:`spacr.measure.FIELD_TABLE_DECIDED_KEYS` is the list, and it is
derived from the plane order rather than written out again here.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ...measure import FIELD_TABLE_DECIDED_KEYS, field_table_settings
from ..job_runner import JobRunner
from ..widgets.card import Card
from ..widgets.collapsible_section import CollapsibleSection
from ..widgets.measure_input_table import MeasureInputTable

LOG = logging.getLogger(__name__)

#: The registry key whose settings form this window reuses. Naming it once
#: is what makes "the same settings as Measure" checkable rather than a
#: claim in a docstring.
SETTINGS_APP_KEY = "measure"


def write_setting_value(widget, value) -> bool:
    """Write one value into whichever kind of control holds it.

    THE TRAP THIS EXISTS FOR: only some of the settings controls carry a
    ``set_value`` method. ``cell_min_size`` is a plain ``QSpinBox``, and a
    writer that tests ``hasattr(widget, 'set_value')`` skips it -- silently,
    leaving the default 8000 in place. A run configured that way measures
    nothing and reports success, because every object was smaller than a
    minimum nobody chose.

    The coercions are the ones
    :meth:`spacr.qt.screens.app_screen.AppScreen._apply_value` makes, kept
    in step with it by hand. They are not shared yet because that method is
    bound to a screen this window is not one of; if a third caller appears,
    move it rather than copying it again.

    :param widget: the control to write into.
    :param value: the value, coerced to the control's own type and left
        alone when it cannot be.
    :returns: whether the control took it.
    """
    from PySide6.QtWidgets import (
        QCheckBox, QComboBox, QDoubleSpinBox, QLineEdit, QSpinBox)

    if isinstance(widget, QCheckBox):
        widget.setChecked(str(value).lower() in ("true", "1", "yes"))
        return True
    if isinstance(widget, QSpinBox):
        try:
            widget.setValue(int(float(value)))
        except (ValueError, TypeError):
            return False
        return True
    if isinstance(widget, QDoubleSpinBox):
        from .settings_model import AUTO_TEXT, _set_auto_or_number

        if str(widget.specialValueText() or "") == AUTO_TEXT:
            _set_auto_or_number(widget, value)
            return True
        try:
            widget.setValue(float(value))
        except (ValueError, TypeError):
            return False
        return True
    if isinstance(widget, QComboBox):
        index = widget.findData(value)
        if index < 0 and value is not None:
            index = widget.findData(str(value))
        if index < 0:
            index = widget.findText(str(value))
        if index < 0:
            return False
        widget.setCurrentIndex(index)
        return True
    setter = getattr(widget, "set_value", None)
    if setter is not None:
        setter(value)
        return True
    if isinstance(widget, QLineEdit):
        widget.setText("" if value is None else str(value))
        return True
    return False


class MeasureInputsScreen(QWidget):
    """Table, Measure's settings, and a Run that goes through Measure.

    :param parent: parent widget, or ``None``.
    :param threaded: ``False`` runs the measurement inline, emitting the same
        signals in the same order, so a test can drive the whole window
        synchronously without the behaviour diverging.

    :ivar run_finished: emitted with the result dict when a run completes,
        or with ``None`` when it failed.
    """

    run_finished = Signal(object)

    def __init__(self, parent: Optional[QWidget] = None, *,
                 threaded: bool = True) -> None:
        """Build the window and its three parts."""
        super().__init__(parent)
        self.setWindowTitle("Features -- measure hand-drawn masks")
        self._runner = JobRunner(self, threaded=threaded,
                                 app_key=SETTINGS_APP_KEY)
        self._destination: Optional[str] = None
        self._result: Optional[Dict[str, Any]] = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(8)

        splitter = QSplitter(Qt.Vertical, self)

        table_card = Card(
            "The files",
            "Drop images and masks here, or write one regex that sorts "
            "them. Rows are fields; columns are channels and mask types.",
            self)
        self.inputs = MeasureInputTable(table_card)
        self.inputs.table_changed.connect(self._on_table_changed)
        table_card.body_layout.addWidget(self.inputs)
        splitter.addWidget(table_card)

        settings_host = QWidget(self)
        settings_layout = QVBoxLayout(settings_host)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        self._settings_area = QScrollArea(settings_host)
        self._settings_area.setWidgetResizable(True)
        self._settings_area.setFrameShape(QScrollArea.NoFrame)
        settings_layout.addWidget(self._settings_area)
        splitter.addWidget(settings_host)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        outer.addWidget(splitter, 1)

        self.settings = self._build_settings_form()

        self._status = QLabel("", self)
        self._status.setObjectName("CardSubtitle")
        self._status.setWordWrap(True)
        outer.addWidget(self._status)

        self._log = QPlainTextEdit(self)
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(90)
        self._log.setPlaceholderText(
            "What the run writes appears here.")
        outer.addWidget(self._log)

        actions = QHBoxLayout()
        actions.addStretch(1)
        self.run_button = QPushButton("Measure", self)
        self.run_button.setToolTip(
            "Write the merged arrays this table describes and measure them "
            "with the Measure module itself. The output folders and the "
            "measurements database are the ones a Measure run produces.")
        self.run_button.clicked.connect(self.run)
        actions.addWidget(self.run_button)
        outer.addLayout(actions)

        self._on_table_changed()

    def _build_settings_form(self):
        """Build Measure's own settings form and disable what the table decides.

        :returns: the :class:`SettingsWidgets` holding every control.
        """
        from .settings_model import SettingsWidgets

        model = SettingsWidgets(SETTINGS_APP_KEY, self)
        body = QWidget(self._settings_area)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        try:
            sections = model.build_sections()
        except Exception:
            LOG.debug("Measure's settings form could not be built",
                      exc_info=True)
            sections = []
        for section in sections:
            title, rows = section[0], section[1]
            content = QWidget(body)
            form = QFormLayout(content)
            form.setContentsMargins(6, 6, 6, 6)
            for label, widget in rows:
                form.addRow(label, widget)
            layout.addWidget(
                CollapsibleSection(str(title), content, expanded=False,
                                   parent=body))
        layout.addStretch(1)
        self._settings_area.setWidget(body)
        self._decided_widgets = {
            key: model._widgets[key]
            for key in FIELD_TABLE_DECIDED_KEYS
            if key in getattr(model, "_widgets", {})}
        for widget in self._decided_widgets.values():
            widget.setEnabled(False)
            widget.setToolTip(
                "The file table decides this. Change the table's channels "
                "or mask columns and this follows.")
        return model

    def set_destination(self, path: Optional[str]) -> None:
        """Where the project is written. ``None`` puts it beside the files."""
        self._destination = str(path) if path else None

    def destination(self) -> Optional[str]:
        """Where the next run would write, as far as this window knows."""
        return self._destination

    def result(self) -> Optional[Dict[str, Any]]:
        """What the last run returned, or ``None`` if none has finished."""
        return self._result

    def derived_settings(self) -> Dict[str, Any]:
        """The settings a run would use: the panel's, over-written by the table.

        The same call the run makes, so what this returns is what would
        happen rather than a second opinion about it.
        """
        try:
            answers = self.settings.collect()
        except Exception:
            LOG.debug("Measure's settings form could not be read",
                      exc_info=True)
            answers = {}
        return field_table_settings(
            self.inputs.table(), answers, dst=self._destination)

    def _on_table_changed(self) -> None:
        """Refresh the decided controls, the status line and the Run button."""
        problems = self.inputs.problems()
        self.run_button.setEnabled(not problems)
        if problems:
            self._status.setText(problems[0] if len(problems) == 1 else (
                f"{problems[0]}  (+{len(problems) - 1} more)"))
        else:
            table = self.inputs.table()
            self._status.setText(
                f"{len(table.rows)} field(s), "
                f"{int(table.n_channels)} channel(s), "
                f"{len(table.ordered_roles())} mask type(s). "
                "The settings the table decides are filled in below.")
        self._show_decided_values()

    def _show_decided_values(self) -> None:
        """Write the table's answers into the controls it answers for."""
        if not getattr(self, "_decided_widgets", None):
            return
        values = field_table_settings(
            self.inputs.table(), {}, dst=self._destination)
        for key, widget in self._decided_widgets.items():
            try:
                write_setting_value(widget, values.get(key))
            except Exception:
                LOG.debug("Could not show the derived %s", key, exc_info=True)

    def apply_settings_dict(self, settings: Dict[str, Any]) -> int:
        """Write ``settings`` into the panel's controls.

        The same contract
        :meth:`spacr.qt.screens.app_screen.AppScreen.apply_settings_dict` has,
        so a settings pack saved from the Measure screen loads here.

        :param settings: the values to write. Keys with no control are
            ignored.
        :returns: how many controls took a value.
        """
        applied = 0
        for key, value in dict(settings or {}).items():
            widget = getattr(self.settings, "_widgets", {}).get(key)
            if widget is None:
                continue
            try:
                applied += bool(write_setting_value(widget, value))
            except Exception:
                LOG.debug("Could not apply %s", key, exc_info=True)
        self._show_decided_values()
        return applied

    def run(self) -> bool:
        """Measure the table, off the GUI thread.

        :returns: whether a run was started. A table with problems in it
            starts nothing and says what they are.
        """
        problems = self.inputs.problems()
        if problems:
            self._status.setText(
                "Nothing was measured: " + " ".join(problems[:3]))
            return False
        table = self.inputs.table()
        answers = self.settings.collect()
        destination = self._destination
        self.run_button.setEnabled(False)
        self._log.appendPlainText("Writing the merged arrays...")

        def work():
            """Run the measurement. Worker thread; touches no widget."""
            from ...measure import measure_from_field_table

            return measure_from_field_table(table, answers, dst=destination)

        return self._runner.submit(work, self._on_done)

    def _on_done(self, result: Any) -> None:
        """Report what the run wrote. GUI thread."""
        self._result = result if isinstance(result, dict) else None
        self.run_button.setEnabled(not self.inputs.problems())
        if self._result is None:
            self._log.appendPlainText("The run produced no result.")
            self.run_finished.emit(None)
            return
        db_path = self._result.get('db_path', '')
        self._log.appendPlainText(
            f"Measured {len(self._result.get('stems', []))} field(s).")
        self._log.appendPlainText(f"Database: {db_path}")
        if not os.path.isfile(str(db_path)):
            self._log.appendPlainText(
                "WARNING: the database is not where it was expected.")
        self.run_finished.emit(self._result)


def open_measure_inputs(owner: Optional[QWidget] = None, *,
                        folder: Optional[str] = None,
                        threaded: bool = True) -> MeasureInputsScreen:
    """Open the FEATURES window, owned by ``owner``'s window.

    :param owner: the screen the button is on, so Qt keeps the window alive
        for as long as that screen is and closes it with the application.
    :param folder: the folder the user is drawing in. Its files are offered
        to the table straight away, because a user who pressed FEATURES from
        an open folder means that folder.
    :param threaded: ``False`` runs the measurement inline; for tests.
    :returns: the window, already shown.
    """
    screen = MeasureInputsScreen(threaded=threaded)
    if owner is not None:
        screen.setParent(owner.window(), Qt.Window)
    if folder and os.path.isdir(str(folder)):
        screen.set_destination(os.path.join(str(folder), 'features'))
        screen.inputs.add_paths([
            os.path.join(str(folder), name)
            for name in sorted(os.listdir(str(folder)))
            if os.path.isfile(os.path.join(str(folder), name))])
    screen.show()
    return screen
