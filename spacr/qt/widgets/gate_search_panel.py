"""The hyperparameter search, beside the filter instead of behind a modal.

The last item of instruction 31:

    "Filter and Hyperparameter-search become TABS, the active one blue like
     the app's buttons, sitting above the console."

WHY IT WAS WORTH MOVING, and not just rearranging. A search is a thing you
ITERATE on -- change a parameter, look at the plot, change it again. It lived
in ``_ClusterSettingsDialog``, a modal, so *looking* meant closing the dialog
and reopening it to change anything. Every loop cost two clicks and the loss
of the previous view. Beside the filter it is one click away and the scatter
stays on screen while the numbers move.

ONE SOURCE OF TRUTH. Every control here reads and writes
:class:`~spacr.qt.widgets.gate_settings.GateEditorSettings`, which is what the
modal already does. That is deliberate: the modal's own docstring records the
bug from when it did NOT -- it opened on hardcoded 0.30/10 while Gate Settings
offered 0.5/20, so values the user set were discarded and the two disagreed
about the defaults as well. A second editor of the same numbers is only safe
when both edit the same object.

WHAT IS AND IS NOT HERE. DBSCAN's parameters and the Walk are, because they
are the search. The xD projection's hyperparameters are NOT: they live in the
settings dialog's xD tab beside the column picker they depend on, and a
reduction is chosen once per table rather than iterated on per gate. Splitting
them across two places would be the "two editors" mistake again.
"""
from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox, QFormLayout, QLabel, QPushButton, QSpinBox,
    QVBoxLayout, QWidget,
)

from ..theme import SPACING
from .toggle import Toggle

LOG = logging.getLogger("spacr.qt.gate_search_panel")

__all__ = ["GateSearchPanel"]


class GateSearchPanel(QWidget):
    """DBSCAN's parameters and the Walk toggle, live beside the plot."""

    #: A parameter changed. Carries ``{field: value}`` for the caller to
    #: fold into its settings -- the panel does not own them.
    settings_changed = Signal(dict)
    #: The user asked for the search to run now.
    run_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("GateSearchPanel")
        self._settings = None
        self._loading = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(SPACING["sm"])
        form = QFormLayout()

        self._eps = QDoubleSpinBox(self)
        self._eps.setRange(0.001, 100.0)
        self._eps.setSingleStep(0.05)
        self._eps.setDecimals(3)
        self._eps.setToolTip(
            "How close two objects must be to count as neighbours. While "
            "scaling is on this is in SCALED units, which is what makes one "
            "value work across measurements whose ranges differ by orders of "
            "magnitude — cell_area runs to thousands and eccentricity to "
            "one, and unscaled DBSCAN on that pair clusters on area alone.")
        self._eps.valueChanged.connect(
            lambda v: self._emit(cluster_eps=float(v)))
        form.addRow("Neighbour distance", self._eps)

        self._min_samples = QSpinBox(self)
        self._min_samples.setRange(2, 10000)
        self._min_samples.setToolTip(
            "How many neighbours an object needs before it can seed a "
            "cluster. Raise it to stop debris forming populations of its "
            "own; lower it to keep a small real population.")
        self._min_samples.valueChanged.connect(
            lambda v: self._emit(cluster_min_samples=int(v)))
        form.addRow("Minimum neighbours", self._min_samples)

        self._scale = Toggle("Scale the measurements first", self)
        self._scale.setToolTip(
            "Standardise each measurement before clustering. Off, whichever "
            "measurement has the larger numbers decides the clusters on its "
            "own, whatever it means.")
        self._scale.toggled.connect(
            lambda v: self._emit(cluster_scale=bool(v)))
        form.addRow("", self._scale)

        self._walk = Toggle("Walk the parameters instead", self)
        self._walk.setToolTip(
            "Search the space rather than taking the two numbers above: try "
            "a range, show each result as it arrives, and let you pick. What "
            "Walk means everywhere else in spaCR.\n\nThe numbers above are "
            "the STARTING POINT of the walk, not ignored.")
        self._walk.toggled.connect(self._on_walk_toggled)
        form.addRow("", self._walk)

        self._walk_steps = QSpinBox(self)
        self._walk_steps.setRange(2, 200)
        self._walk_steps.setToolTip(
            "How many parameter combinations the walk tries. Each is a full "
            "clustering pass, so this is the cost.")
        self._walk_steps.valueChanged.connect(
            lambda v: self._emit(cluster_walk_steps=int(v)))
        form.addRow("Walk steps", self._walk_steps)
        outer.addLayout(form)

        self._run = QPushButton("Run search", self)
        self._run.setToolTip(
            "Cluster with these parameters and turn what it finds into "
            "gates — the same gates a drawn shape makes, so they save, "
            "re-apply and export identically.")
        self._run.clicked.connect(self.run_requested.emit)
        outer.addWidget(self._run)

        self._note = QLabel(self)
        self._note.setObjectName("MutedNote")
        self._note.setWordWrap(True)
        outer.addWidget(self._note)
        outer.addStretch(1)
        self._refresh_gating()
        # Hover help belongs on a setting's NAME, not on the field the user
        # is about to type into (instruction 113). One post-pass rather than
        # a convention every hand-built row has to remember.
        from ..screens.settings_model import retarget_field_tooltips
        retarget_field_tooltips(self)

    # -- settings ---------------------------------------------------------
    def apply_settings(self, settings) -> None:
        """Take the screen's settings and show them.

        Signals are muted while loading: setting a spin box fires
        ``valueChanged``, and a load would otherwise report every control as
        freshly edited by the user and write the values it just read back
        out.
        """
        self._settings = settings
        self._loading = True
        try:
            self._eps.setValue(float(getattr(settings, "cluster_eps", 0.5)))
            self._min_samples.setValue(
                int(getattr(settings, "cluster_min_samples", 20)))
            self._scale.setChecked(bool(getattr(settings, "cluster_scale", True)))
            self._walk.setChecked(bool(getattr(settings, "cluster_walk", False)))
            self._walk_steps.setValue(
                int(getattr(settings, "cluster_walk_steps", 12)))
        finally:
            self._loading = False
        self._refresh_gating()

    def _emit(self, **changed) -> None:
        if self._loading:
            return
        self.settings_changed.emit(dict(changed))

    def _on_walk_toggled(self, on: bool) -> None:
        self._emit(cluster_walk=bool(on))
        self._refresh_gating()

    def _refresh_gating(self) -> None:
        """Grey what the other mode does not read.

        INVARIANTS 6 -- greyed, not removed. Walking makes the two numbers a
        starting point rather than the answer, and a control that vanished
        would take the value with it.
        """
        walking = self._walk.isChecked()
        self._walk_steps.setEnabled(walking)
        self._run.setText("Run walk" if walking else "Run search")
        self._note.setText(
            "The walk tries a range around the values above and shows each "
            "result as it lands."
            if walking else
            "One pass with exactly these values.")
