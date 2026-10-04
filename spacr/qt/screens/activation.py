"""Activation-map workflow embedded in the Classify screen.

The ``activation`` workflow remains distinct from classifier training: it
uses its own settings form, Run action, console, and hyperparameter-search
key, and :func:`spacr.qt.bridge.resolve_pipeline_entry` dispatches it to
``generate_activation_map``. The Classify masthead opens that workflow as a
folded page so attribution results remain associated with the trained model
without combining the two pipeline runs.

:class:`ExplainNavigator` routes requests from the Explain CV Model page to
the same folded activation page. The aliases in :data:`NAV_KEYS` cover the
pipeline key, command alias, and navigation spelling without constructing a
second screen or duplicating its settings and job state.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from PySide6.QtWidgets import QWidget

LOG = logging.getLogger(__name__)

#: Registry key of the module folded onto Classify.
APP_KEY = "activation"

#: Every name a request for the activation maps may arrive under.
#:
#: ``activation_maps`` is what :mod:`spacr.qt.screens.model_explanation`
#: asks for, ``activation_map`` is the spelling :data:`spacr.cli.ALIASES`
#: maps onto the key, and ``activation`` is the key itself -- what a saved
#: run journal or ``spacr-qt activation`` names. One set rather than three
#: call sites deciding separately which spelling counts.
NAV_KEYS = frozenset({"activation", "activation_map", "activation_maps"})


def build(host_window: Optional[QWidget] = None) -> QWidget:
    """The module's own screen, wired the way its sidebar row wired it.

    :param host_window: the main window, when there is one to connect to.
    :returns: the settings-driven screen for :data:`APP_KEY`, with its
        host connections -- "Explain error" and "Run on a cluster" -- made.
    """
    from .map_barcodes import build_settings_screen

    screen = build_settings_screen(APP_KEY, host_window)
    _add_counterfactual_viewer_button(screen)
    return screen


def _add_counterfactual_viewer_button(screen: Optional[QWidget]) -> Optional[QWidget]:
    """Put the counterfactual viewer button on ``screen``'s header row.

    An alpha widget, ``ActivationCounterfactualViewer`` in
    :data:`spacr.settings.ALPHA_FEATURES`, hidden unless alpha features are
    shown. It asks for a ``counterfactuals`` folder and opens
    :func:`_counterfactual_viewer` on it.

    :param screen: the activation screen; one without a header gets nothing.
    :returns: the button, or None.
    """
    from PySide6.QtWidgets import QFileDialog, QPushButton

    from ..i18n import tr
    from ..preferences import _apply_alpha_widgets

    header = getattr(screen, "_header", None)
    if header is None or not hasattr(header, "add_trailing"):
        return None
    button = QPushButton(tr("Counterfactuals…"), screen)
    button.setObjectName("ActivationCounterfactualViewer")
    button.setToolTip(tr(
        "Browse the counterfactual sequences of a finished run: pick its "
        "counterfactuals folder to step through each crop as it is morphed "
        "toward the target class or condition, with the classifier's score "
        "at every step. Default not opened."))

    def _open() -> None:
        """Ask for a counterfactuals folder and show what it holds."""
        folder = QFileDialog.getExistingDirectory(
            screen, tr("Choose a counterfactuals folder"))
        if folder:
            dialog = _counterfactual_viewer(folder, screen)
            dialog.show()
            screen._counterfactual_viewer = dialog

    button.clicked.connect(_open)
    header.add_trailing(button)
    _apply_alpha_widgets(button)
    return button


def _counterfactual_viewer(folder: str, parent: Optional[QWidget] = None) -> QWidget:
    """A dialog listing a run's counterfactual sequences and drawing one.

    Reads ``counterfactual_cells.csv`` and ``counterfactual_frames.npy``
    from the folder ``generate_activation_map`` wrote. The list shows each
    held-out crop with its source, target, score change and whether it
    flipped; the first sequences, the ones with saved frames, are drawn as
    a row of steps captioned with the score at each step.

    :param folder: the ``counterfactuals`` folder of a run.
    :param parent: the dialog's parent widget.
    :returns: the dialog, not yet shown.
    """
    import os

    import numpy as np
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QImage, QPixmap
    from PySide6.QtWidgets import (QDialog, QGridLayout, QLabel,
                                   QListWidget, QVBoxLayout, QWidget)

    from ...tabular import read_table
    from ..hidpi import scaled_for
    from ..i18n import tr

    dialog = QDialog(parent)
    dialog.setObjectName("CounterfactualViewerDialog")
    dialog.setWindowTitle(tr("Counterfactual sequences"))
    layout = QVBoxLayout(dialog)
    cells = os.path.join(folder, "counterfactual_cells.csv")
    frames_path = os.path.join(folder, "counterfactual_frames.npy")
    rows = read_table(cells).to_dict("records") if os.path.isfile(cells) else []
    frames = (np.load(frames_path) if os.path.isfile(frames_path)
              else np.zeros((0,)))
    status = QLabel(dialog)
    status.setWordWrap(True)
    layout.addWidget(status)
    listing = QListWidget(dialog)
    listing.setObjectName("CounterfactualViewerList")
    layout.addWidget(listing)
    strip = QWidget(dialog)
    grid = QGridLayout(strip)
    layout.addWidget(strip)
    dialog.rows, dialog.frames, dialog.listing = rows, frames, listing
    dialog.strip_labels = []
    if not rows:
        status.setText(tr("No counterfactual_cells.csv in this folder."))
        return dialog
    status.setText(tr("{n} sequences; the first {k} have saved frames.")
                   .format(n=len(rows), k=int(frames.shape[0])
                           if frames.ndim == 5 else 0))
    for row in rows:
        source = row.get("source_condition", row.get("source_class"))
        target = row.get("target_condition", row.get("target_class"))
        listing.addItem(
            f"{row.get('name', '')}  {source}→{target}  "
            f"{float(row.get('score_start', 0)):.2f}→"
            f"{float(row.get('score_end', 0)):.2f}"
            + ("  ✓" if str(row.get("flipped")) == "True" else ""))

    def _show(index: int) -> None:
        """Replace the strip with the sequence at ``index``."""
        for label in dialog.strip_labels:
            label.deleteLater()
        dialog.strip_labels = []
        if frames.ndim != 5 or not 0 <= index < frames.shape[0]:
            return
        seq = frames[index]
        lo, hi = float(seq[0].min()), float(seq[0].max())
        path = str(rows[index].get("score_path", "")).split(";")
        for step in range(seq.shape[0]):
            img = seq[step]
            img = img[0] if img.shape[0] != 3 else np.moveaxis(img, 0, -1)
            img = np.clip((img - lo) / ((hi - lo) or 1.0), 0, 1)
            img = np.ascontiguousarray((img * 255).astype(np.uint8))
            fmt = QImage.Format_RGB888 if img.ndim == 3 else QImage.Format_Grayscale8
            qimg = QImage(img.data, img.shape[1], img.shape[0],
                          img.strides[0], fmt).copy()
            picture = QLabel(strip)
            picture.setPixmap(scaled_for(QPixmap.fromImage(qimg), picture,
                                         96, 96))
            caption = QLabel(path[step] if step < len(path) else "", strip)
            caption.setAlignment(Qt.AlignHCenter)
            grid.addWidget(picture, 0, step)
            grid.addWidget(caption, 1, step)
            dialog.strip_labels += [picture, caption]

    listing.currentRowChanged.connect(_show)
    listing.setCurrentRow(0)
    return dialog


def opener_on(screen: Optional[QWidget]) -> Optional[Any]:
    """Return the activation-page opener registered on ``screen``, if present.

    The opener is owned by the host screen. Reusing it ensures that masthead
    actions and navigation requests address the same folded page and preserve
    its console, job runner, and settings state.

    :param screen: Candidate host screen.
    :returns: Matching fold opener, or ``None``.
    """
    for opener in getattr(screen, "_fold_openers", ()) or ():
        if getattr(opener, "key", "") == APP_KEY:
            return opener
    return None


def host_of(widget: Optional[QWidget]) -> Optional[QWidget]:
    """The fold host ``widget`` is sitting on, or None.

    Walks up from a folded page to the screen whose strip can open the
    activation maps. Derived from the widget tree rather than handed in,
    because the page is built before it is mounted -- and because a page
    that ended up in a window instead (the fold's last resort) has no such
    host above it, which is exactly what None says.

    :param widget: the widget to start from, usually a folded page; None gives
        None.
    """
    node = widget
    while node is not None:
        if opener_on(node) is not None:
            return node
        node = node.parent()
    return None


def open_page(host_screen: Optional[QWidget]) -> Optional[QWidget]:
    """Show the activation page on ``host_screen`` and raise it.

    :param host_screen: the screen that may carry the activation fold's opener;
        None gives None.
    :returns: the module's screen, or None when this host carries no
        activation fold.
    """
    opener = opener_on(host_screen)
    return opener.open() if opener is not None else None


def apply_seed(screen: Optional[QWidget], values: Dict[str, Any]) -> None:
    """Push ``values`` into ``screen``'s settings form.

    The rule ``MainWindow._on_train_requested`` applies, asked of the same
    function rather than written out a second time: a navigation that
    seeded differently from the sidebar's would be a second answer to one
    question. A key with no widget is skipped, as it is there.

    :param screen: the screen whose ``_settings_model`` widgets are set; None,
        or a screen without a settings model, does nothing.
    :param values: setting name to value; each is applied to the widget of that
        name, and names with no widget are skipped.
    """
    model = getattr(screen, "_settings_model", None)
    if model is None or not values:
        return
    widgets = getattr(model, "_widgets", {})
    try:
        from ..app import MainWindow

        apply_value = MainWindow._apply_seed_value
    except Exception:                                        # noqa: BLE001
        LOG.debug("Could not read the seeding rule", exc_info=True)
        return
    for key, value in values.items():
        widget = widgets.get(key)
        if widget is None:
            continue
        try:
            apply_value(widget, value)
        except Exception:                                    # noqa: BLE001
            LOG.debug("Could not seed %s with %r", key, value, exc_info=True)


class ExplainNavigator:
    """The host handed to Classify's Explain CV page.

    Explain CV Model sends the user on in one place -- "Open Activation
    Maps" -- and does it by calling ``host._on_train_requested``. Standing
    in for the window there is what lets that button land on the page
    beside it instead of on a key nothing knows, and it costs the
    explanation screen nothing: anything that is not a request for the
    activation maps is forwarded to the real window unchanged.

    A plain object rather than a ``QObject``: it is reached by attribute
    access only, and the page holds it, so it lives exactly as long as the
    page that uses it.

    :param window: the main window, when it can be navigated through --
        see :func:`spacr.qt.screens.classify._navigable`.
    """

    def __init__(self, window: Optional[QWidget] = None) -> None:
        """Record the window the explanation page will be opened on.

        :param window: the host window; the page itself is attached later, once
            it exists.
        """
        self.window = window
        #: The page this navigator was built for, once it exists.
        self.page: Optional[QWidget] = None

    def attach(self, page: QWidget) -> None:
        """Remember the page, so the host it lands on can be found later.

        :param page: the activation page widget, stored as ``page``.
        """
        self.page = page

    def _on_train_requested(self, target_key: str,
                            seed: Optional[Dict[str, Any]] = None
                            ) -> Optional[QWidget]:
        """Answer a navigation, or hand it on to the window.

        :param target_key: the module the page asked for.
        :param seed: settings to push into it, as the window would.
        :returns: the screen that was opened, or None when nothing could
            answer -- which is what the page did before it had a host.
        """
        values = dict(seed or {})
        key = str(target_key)
        if key in NAV_KEYS:
            opened = open_page(host_of(self.page))
            if opened is not None:
                apply_seed(opened, values)
                return opened
            key = APP_KEY
        window = self.window
        if window is None or not callable(
                getattr(window, "_on_train_requested", None)):
            return None
        return window._on_train_requested(key, values)
