"""The Voyager3 control in OPS's settings: a button, a progress bar and a status line (item 973, alpha).

The button runs only the ``voyager3`` phase of :func:`spacr.ops_engine.run_ops`
on the settings the panel holds, through the same worker thread a Run uses
(:func:`spacr.qt.bridge.make_thread`), so the run appears on Home and in the
console like any other. The bar follows the run's ``Progress: n/total`` lines.
The AI that answers is the one chosen under Preferences > AI, or
``ops_voyager3_provider``; Voyager3 never uses a vendor API key.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from PySide6.QtCore import QObject
from PySide6.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QPushButton, QVBoxLayout, QWidget

from .i18n import tr

LOG = logging.getLogger(__name__)

#: The setting whose section gets the control.
ANCHOR_KEY = "ops_voyager3_enabled"


def _run_voyager3_phase(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Run only the ``voyager3`` phase of the OPS module on ``settings``.

    :param settings: the OPS settings the panel holds.
    :returns: :func:`spacr.ops_engine.run_ops`' result.
    """
    from ..ops_engine import run_ops

    return run_ops(settings, phases=("voyager3",))


class Voyager3Controls(QWidget):
    """The button, bar and status line that start and follow a Voyager3 run.

    :param screen: the OPS module screen; its settings model gives the
        settings and its console, when it has one, shows the run's lines.
    :param parent: the Qt parent.
    """

    def __init__(self, screen: Any, parent: Optional[QWidget] = None) -> None:
        """Build the control, idle."""
        super().__init__(parent)
        self._screen = screen
        self._thread: Optional[QObject] = None
        self._worker: Optional[QObject] = None
        self.button = QPushButton(tr("Annotate clusters (Voyager3)"), self)
        self.button.setToolTip(tr(
            "Fetch every cluster gene's UniProt facts and ask the AI chosen under Preferences to name "
            "each cluster's biological role. Runs only this step, on the cluster table in the settings "
            "above, and writes results/ops_voyager3."))
        self.button.clicked.connect(self._start)
        self.bar = QProgressBar(self)
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.setTextVisible(False)
        self.bar.setVisible(False)
        self.status = QLabel(tr("Not run yet."), self)
        self.status.setWordWrap(True)
        row = QHBoxLayout()
        row.addWidget(self.button)
        row.addWidget(self.bar, 1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(row)
        layout.addWidget(self.status)

    def _refresh_visibility(self) -> bool:
        """Show the control only while Show alpha features shows Voyager3's settings.

        :returns: whether it is shown.
        """
        try:
            from .preferences import _is_alpha_visible

            shown = bool(_is_alpha_visible("settings", ANCHOR_KEY))
        except Exception:                                      # noqa: BLE001
            shown = True
        self.setVisible(shown)
        return shown

    def _is_running(self) -> bool:
        """Whether a Voyager3 run started here has not finished."""
        return self._thread is not None

    def _settings(self) -> Dict[str, Any]:
        """The settings the screen's panel holds now."""
        model = getattr(self._screen, "_settings_model", None)
        return dict(model.collect()) if model is not None else {}

    def _start(self, _checked: bool = False) -> bool:
        """Start a Voyager3 run on the current settings.

        :param _checked: swallows the bool a ``clicked`` signal passes.
        :returns: whether a run was started.
        """
        if self._is_running():
            return False
        try:
            settings = self._settings()
        except Exception as exc:                               # noqa: BLE001
            self.status.setText(tr("The settings could not be read: {error}", error=str(exc)))
            return False
        from .bridge import make_thread

        self._thread, self._worker = make_thread(_run_voyager3_phase, settings, app_key="ops")
        self._worker.line_ready.connect(self._on_line)
        console = getattr(self._screen, "_console", None)
        if console is not None and hasattr(console, "_display_stdout"):
            self._worker.line_ready.connect(console._display_stdout)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._on_finished)
        self._thread.finished.connect(self._release)
        self.button.setEnabled(False)
        self.bar.setValue(0)
        self.bar.setVisible(True)
        self.status.setText(tr("Voyager3 is fetching UniProt facts and asking the AI..."))
        self._thread.start()
        return True

    def _on_line(self, line: str) -> None:
        """Move the bar on a ``Progress:`` line.

        :param line: one line the run printed.
        """
        from .run_progress import parse_progress

        samples = parse_progress(str(line))
        if samples and samples[-1].total:
            last = samples[-1]
            self.bar.setValue(int(round(100 * last.done / last.total)))
            self.status.setText(tr("Voyager3: {done} of {total} clusters annotated",
                                   done=last.done, total=last.total))

    def _on_error(self, message: str) -> None:
        """Show a failed run's message.

        :param message: the error text the worker sent.
        """
        self.status.setText(tr("Voyager3 stopped: {error}", error=str(message).strip()[:400]))

    def _on_finished(self, ok: bool) -> None:
        """Say where the results are, or leave the error showing.

        :param ok: whether the run finished without an error.
        """
        self.button.setEnabled(True)
        if ok:
            self.bar.setValue(100)
            self.status.setText(tr("Voyager3 finished. The table, report and provenance are in "
                                   "results/ops_voyager3."))

    def _release(self) -> None:
        """Drop the finished thread and worker."""
        self._thread = None
        self._worker = None


def install_voyager3_controls(screen: Any, section: Any) -> Optional[Voyager3Controls]:
    """Add the Voyager3 control at the top of ``section`` once.

    :param screen: the OPS module screen.
    :param section: the settings section that holds :data:`ANCHOR_KEY`.
    :returns: the control, or None when the section already has one or
        cannot hold widgets.
    """
    if getattr(section, "_voyager3_controls", None) is not None:
        return None
    add = getattr(section, "add_prose", None)
    if not callable(add):
        return None
    controls = Voyager3Controls(screen, section if isinstance(section, QWidget) else None)
    add(controls, at_top=True)
    section._voyager3_controls = controls
    controls._refresh_visibility()
    return controls
