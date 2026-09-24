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

import copy
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
    QVBoxLayout,
    QWidget,
)

from ...measure import (
    FIELD_TABLE_DECIDED_KEYS,
    field_table_destination,
    field_table_settings,
)
from ..job_runner import JobRunner
from ..widgets.card import Card
from ..widgets.collapsible_splitter import CollapsibleSplitter, fold_card
from ..widgets.collapsible_section import CollapsibleSection
from ..widgets.measure_input_table import MeasureInputTable

LOG = logging.getLogger(__name__)

#: The registry key whose settings form this window reuses. Naming it once
#: is what makes "the same settings as Measure" checkable rather than a
#: claim in a docstring.
SETTINGS_APP_KEY = "measure"

#: Where this window remembers its folds and its dragged heights (item 471).
#: Its own key rather than :data:`SETTINGS_APP_KEY`, so folding the file
#: table here does not fold anything on the Measure module itself.
FOLD_KEY = "measure_inputs"


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
    #: A stage of the run started. Emitted FROM THE WORKER THREAD, which is
    #: safe because the receiver is a bound method of this GUI-thread object
    #: and Qt therefore queues the call -- the rule
    #: :mod:`spacr.qt.job_runner` sets out.
    progress = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None, *,
                 threaded: bool = True) -> None:
        """Build the window and its three parts."""
        super().__init__(parent)
        self.setWindowTitle("Features -- measure hand-drawn masks")
        self._runner = JobRunner(self, threaded=threaded,
                                 app_key=SETTINGS_APP_KEY)
        self._runner.job_failed.connect(self._on_failed)
        self._threaded = bool(threaded)
        self.progress.connect(self._on_progress)
        self._destination: Optional[str] = None
        self._result: Optional[Dict[str, Any]] = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(8)

        splitter = CollapsibleSplitter(Qt.Vertical, self,
                                       persist_key=f"{FOLD_KEY}::body")
        self._body = splitter

        table_card = Card(
            "The files",
            "Drop images and masks here, or write one regex that sorts "
            "them. Rows are fields; columns are channels and mask types.",
            self)
        self.inputs = MeasureInputTable(table_card, threaded=threaded)
        self.inputs.table_changed.connect(self._on_table_changed)
        table_card.body_layout.addWidget(self.inputs)
        self.table_folder = fold_card(table_card, "The files",
                                      persist_key=f"{FOLD_KEY}/The files")
        self.table_card = table_card
        splitter.add_pane(table_card, "The files", folder=self.table_folder,
                          stretch=1)

        settings_host = QWidget(self)
        settings_layout = QVBoxLayout(settings_host)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        self._settings_area = QScrollArea(settings_host)
        self._settings_area.setWidgetResizable(True)
        self._settings_area.setFrameShape(QScrollArea.NoFrame)
        settings_layout.addWidget(self._settings_area)
        splitter.add_pane(settings_host, "Settings", stretch=2)
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
        from .settings_model import retarget_field_tooltips
        retarget_field_tooltips(self)
        self._explain_the_decided_fields()

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
            layout.addWidget(self._nested_section(section, body))
        layout.addStretch(1)
        self._settings_area.setWidget(body)
        self._decided_widgets = {
            key: model._widgets[key]
            for key in FIELD_TABLE_DECIDED_KEYS
            if key in getattr(model, "_widgets", {})}
        for widget in self._decided_widgets.values():
            widget.setEnabled(False)
        return model

    def _explain_the_decided_fields(self) -> None:
        """Say on each table-decided field why it cannot be edited.

        Run after :func:`~.settings_model.retarget_field_tooltips` has moved
        each setting's help onto its name, so the name keeps the help and the
        disabled field carries the reason. The reason is marked with
        :data:`~.settings_model.DISABLED_REASON_TOOLTIP`, which is the
        convention for a note that explains the control itself.
        """
        from .settings_model import DISABLED_REASON_TOOLTIP

        for widget in self._decided_widgets.values():
            widget.setProperty(DISABLED_REASON_TOOLTIP, True)
            widget.setToolTip(
                "The file table decides this. Change the table's channels "
                "or mask columns and this follows.")

    def _nested_section(self, section, parent: QWidget) -> QWidget:
        """One heading of Measure's settings tree, with its sub-headings.

        The Measure module shows its settings three levels deep -- an
        umbrella, then one sub-heading per object -- and this window used to
        flatten them into one group per top-level title. ``own_rows`` are this heading's own controls and
        ``children`` the headings nested below it; a plain ``(title, rows)``
        pair has neither and is drawn flat, as before. Every control is
        still placed exactly once.

        :param section: a ``SettingsSection`` or a ``(title, rows)`` pair.
        :param parent: the widget the heading is drawn in.
        :returns: the collapsible heading.
        """
        title = getattr(section, "title", None)
        if title is None:
            title = section[0]
        own_rows = getattr(section, "own_rows", None)
        rows = section[1] if own_rows is None else own_rows
        children = tuple(getattr(section, "children", ()) or ())
        content = QWidget(parent)
        column = QVBoxLayout(content)
        column.setContentsMargins(6, 6, 6, 6)
        if rows:
            form_host = QWidget(content)
            form = QFormLayout(form_host)
            form.setContentsMargins(0, 0, 0, 0)
            for label, widget in rows:
                form.addRow(label, widget)
            column.addWidget(form_host)
        for child in children:
            column.addWidget(self._nested_section(child, content))
        return CollapsibleSection(str(title), content, expanded=False,
                                  parent=parent)

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
        table = self.inputs.table()
        return field_table_settings(
            table, answers,
            dst=field_table_destination(table, self._destination))

    def _on_table_changed(self) -> None:
        """Refresh the decided controls, the status line and the Run button."""
        problems = self.inputs.problems()
        self.run_button.setEnabled(not problems and not self._runner.is_busy())
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
        """Write the table's answers into the controls it answers for.

        The destination goes through
        :func:`spacr.measure.field_table_destination`, which is the same call
        the run makes. Passing the window's own ``_destination`` straight
        through showed ``src`` as the settings spec's ``path`` placeholder
        whenever the window had been opened without a folder, while the run
        wrote beside the first channel file -- a disabled box captioned "the
        table decides this" naming a folder the results are not in.
        """
        if not getattr(self, "_decided_widgets", None):
            return
        table = self.inputs.table()
        values = field_table_settings(
            table, {}, dst=field_table_destination(table, self._destination))
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
        table = copy.deepcopy(self.inputs.table())
        answers = self.settings.collect()
        destination = self._destination
        self._set_running(True)
        emit = self.progress.emit

        def work():
            """Run the measurement. Worker thread; touches no widget.

            The table it measures is a SNAPSHOT taken above, not the widget's
            live model. A measure run takes minutes and preparing the next
            batch while it runs is the natural thing to do, so the GUI thread
            can append rows and delete channel and mask keys underneath this
            -- which reaches the worker as a ``KeyError`` in the middle of
            writing, or as half a field that nobody asked for. The snapshot
            is plain data, so copying it is cheap and it cannot be edited
            from anywhere.
            """
            from ...measure import measure_from_field_table

            result = measure_from_field_table(
                table, answers, dst=destination, progress=emit)
            if isinstance(result, dict):
                result = dict(result)
                result['db_exists'] = os.path.isfile(
                    str(result.get('db_path', '')))
            return result

        return self._runner.submit(work, self._on_done)

    def _set_running(self, running: bool) -> None:
        """Lock or release the controls a run must not have changed under it.

        The table as well as the button: disabling only the button left the
        grid editable for the whole run, and it is the grid the worker is
        reading.

        :param running: whether a run is in flight.
        """
        self.run_button.setEnabled(
            not running and not self.inputs.problems())
        self.inputs.setEnabled(not running)

    def _on_progress(self, message: str) -> None:
        """Put one stage of the run in the log. GUI thread."""
        self._log.appendPlainText(str(message))

    def _on_failed(self, message: str) -> None:
        """Say why the run stopped, and give the window back. GUI thread.

        WITHOUT THIS THE WINDOW IS DEAD AND SILENT AFTER A FAILURE.
        ``JobRunner`` calls ``on_done`` only for a job that SUCCEEDED -- in
        both the threaded and the unthreaded path -- so :meth:`_on_done` is
        not the place the button comes back, and every refusal
        :func:`spacr.measure.write_field_table_project` raises (a float
        intensity image, a mask whose shape does not match its channels,
        label ids past 65535) passes :meth:`FieldTable.problems` and only
        fails once the run is under way. The carefully worded
        ``ConfigurationError`` arrives here, on ``job_failed``, and nowhere
        else: before this was connected it went nowhere at all and the user
        was left with a Measure button that never came back and no reason
        given.

        :param message: the failure, one line, from the runner.
        """
        try:
            self._result = None
            self._set_running(False)
            self._log.appendPlainText(f"The run stopped: {message}")
            self.run_finished.emit(None)
        except RuntimeError:
            pass

    def _on_done(self, result: Any) -> None:
        """Report what the run wrote. GUI thread."""
        self._result = result if isinstance(result, dict) else None
        self._set_running(False)
        if self._result is None:
            self._log.appendPlainText("The run produced no result.")
            self.run_finished.emit(None)
            return
        db_path = self._result.get('db_path', '')
        self._log.appendPlainText(
            f"Measured {len(self._result.get('stems', []))} field(s).")
        self._log.appendPlainText(f"Database: {db_path}")
        if not self._result.get('db_exists'):
            self._log.appendPlainText(
                "WARNING: the database is not where it was expected.")
        self.run_finished.emit(self._result)

    def closeEvent(self, event):  # noqa: N802 - Qt contract
        """Stop the run's threads before Qt destroys the widgets.

        Qt ABORTS THE PROCESS if a running ``QThread`` is destroyed, and this
        window is closable while a measure run -- minutes of work -- is in
        flight. :func:`spacr.qt.job_runner.shutdown_all` covers the
        application quitting; it does not cover one window being closed.

        :param event: the Qt close event.
        """
        for runner in (self._runner, getattr(self.inputs, '_scanner', None)):
            if runner is not None:
                try:
                    runner.shutdown()
                except Exception:                                # noqa: BLE001
                    LOG.debug("a runner would not shut down", exc_info=True)
        super().closeEvent(event)


def open_measure_inputs(owner: Optional[QWidget] = None, *,
                        folder: Optional[str] = None,
                        threaded: bool = True) -> MeasureInputsScreen:
    """Open the FEATURES window, owned by ``owner``'s window.

    :param owner: the screen the button is on, so Qt keeps the window alive
        for as long as that screen is and closes it with the application.
    :param folder: the folder the user is drawing in. Its files are offered
        to the table straight away, because a user who pressed FEATURES from
        an open folder means that folder.
    :param threaded: ``False`` runs the measurement and the folder walk
        inline; for tests.
    :returns: the window, already shown.

    THE FOLDER IS NOT READ HERE. This runs in the Make Masks button handler,
    on the GUI thread, and ``folder`` is a path the user chose -- on a
    microscope rig, the share the images live on. It used to ``isdir`` and
    ``listdir`` it and stat every entry before the window was even shown, so
    pressing FEATURES on a folder living on a sleeping automount froze the
    application with no traceback. The walk is
    :meth:`MeasureInputTable.add_dropped`'s, on a worker, and the table
    fills in a moment later.
    """
    screen = MeasureInputsScreen(threaded=threaded)
    if owner is not None:
        screen.setParent(owner.window(), Qt.Window)
    if folder:
        screen.set_destination(os.path.join(str(folder), 'features'))
        screen.inputs.add_dropped([str(folder)])
    screen.show()
    return screen
