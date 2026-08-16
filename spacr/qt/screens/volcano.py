"""Volcano Explorer — open a finished regression and interrogate its plot.

A regression run leaves a volcano PDF. That file answers "what was
significant" and nothing else: which guide is that dot, what would it look
like coloured by compartment, does the axis label match the correction that
was actually applied, and can I have it as a vector figure at the size the
journal asked for.

This screen opens a results folder (or a single results CSV) and hands it to
:class:`spacr.qt.widgets.volcano_explorer.VolcanoExplorer`, which draws the
plot through the same renderer the pipeline used -- so what is on screen is
the plot, not a preview of it.

It is deliberately a *reader*. It runs no analysis and writes nothing back
into the results folder; exports go wherever the user chooses. That means it
can be pointed at a finished run, including one produced months ago on another
machine, without any risk to it.
"""
from __future__ import annotations

import os

APP_KEY = "volcano_explorer"
APP_NAME = "Volcano Explorer"
APP_DESCRIPTION = (
    "Open a regression result, click any point for its full record, restyle "
    "the plot and export it as vector PDF or PNG")
APP_INTRO = (
    "Choose a regression results folder or CSV. Every point carries its whole "
    "row, so clicking one tells you the guide, gene, effect, P value, "
    "adjusted value and how many wells it was seen in. Colour and shape can "
    "be driven by any column, including columns merged in from your own "
    "annotation file. Axis scales, thresholds, a broken axis, colormap, "
    "marker, fonts, line weights and titles are all editable, and the export "
    "re-renders the figure rather than screenshotting it, so a PDF stays "
    "vector at publication size.")
APP_TRANSLATIONS = (
    "Vulkanutforskare", "Vulkan-Explorer", "Explorador de volcán",
    "火山图浏览器", "Explorador de vulcão", "वोल्केनो एक्सप्लोरर",
    "볼케이노 탐색기", "Eldfjallakönnuður", "Explorateur de volcan")
#: Why there is no ``spacr-run volcano_explorer``. Reaches
#: :data:`spacr.cli.INTERACTIVE_ONLY`, which is what the CLI prints instead of
#: "unknown module". This screen is a reader with no analysis behind it, so
#: the honest headless answer is the renderer it draws through -- the same one
#: the pipeline uses, which is why the exported figure is identical.
APP_CLI_NOTE = (
    "Volcano Explorer is an interactive reader for a finished regression -- "
    "clicking a point is the feature, so there is nothing to batch. Headless, "
    "call spacr.volcano_style.render_volcano(results, VolcanoStyle(...), "
    "save_path='volcano.pdf'); that is the renderer this screen draws "
    "through, so the figure is the same one, vector at publication size.")

__all__ = ["APP_KEY", "APP_NAME", "APP_DESCRIPTION", "APP_INTRO",
           "APP_CLI_NOTE", "find_results_table", "register"]

#: Result CSVs a regression folder may hold, best first. ``results_grna.csv``
#: leads because the volcano is a guide-level plot: ``results.csv`` is the
#: same rows for a permutation run but the gene-level table for a simultaneous
#: fit, and plotting genes where the user expects guides is a silent switch of
#: what a point means.
_RESULT_FILENAMES = (
    "guide_permutation_results_long.csv",
    "results_grna.csv",
    "results.csv",
)


def find_results_table(path):
    """Return the results CSV to plot for ``path``, or None.

    Accepts the CSV itself, a regression output folder, or a parent holding
    one -- the three things a user actually has to hand when they want to look
    at a volcano again.
    """
    path = os.path.abspath(os.path.expanduser(os.fspath(path)))
    if os.path.isfile(path):
        return path if path.lower().endswith(".csv") else None
    if not os.path.isdir(path):
        return None
    for name in _RESULT_FILENAMES:
        candidate = os.path.join(path, name)
        if os.path.exists(candidate):
            return candidate
    # One level down, so pointing at the run folder rather than its
    # `guide_permutation/list` leaf still works.
    for entry in sorted(os.listdir(path)):
        child = os.path.join(path, entry)
        if os.path.isdir(child):
            found = find_results_table(child)
            if found:
                return found
    return None


def load_results(path):
    """Read a results CSV and keep only the primary support family.

    A long permutation table holds every minimum-support family stacked, so
    plotting it unfiltered draws each guide once per family -- the same point
    two to four times, at different heights, which reads as extra hits.
    """
    import pandas as pd

    frame = pd.read_csv(path)
    if "minimum_wells_threshold" in frame.columns:
        primary = frame["minimum_wells_threshold"].min()
        frame = frame.loc[frame["minimum_wells_threshold"] == primary]
    if "outcome" in frame.columns and frame["outcome"].nunique() > 1:
        # Several responses were fitted. Show the first; the explorer's own
        # data controls can switch columns, and each response is its own
        # correction family so they must not be pooled into one plot.
        first = frame["outcome"].iloc[0]
        frame = frame.loc[frame["outcome"] == first]
    return frame.reset_index(drop=True)


def _make_screen(app_key=None, host=None):
    """Build the screen lazily; matplotlib and Qt widgets are not cheap."""
    from PySide6.QtWidgets import (
        QFileDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
    )

    from ..widgets.volcano_explorer import VolcanoExplorer

    class VolcanoScreen(QWidget):
        def __init__(self, host=None):
            # `host` is the main window, passed by the registry for
            # navigation -- NOT a Qt parent. Handing it to QWidget.__init__
            # raises, because the registry's host is not always a QWidget.
            super().__init__()
            self.host = host
            layout = QVBoxLayout(self)
            bar = QHBoxLayout()
            self._path_label = QLabel("No results loaded", self)
            self._path_label.setWordWrap(True)
            open_button = QPushButton("Open results…", self)
            open_button.setToolTip(
                "Choose a regression results folder or a results CSV")
            open_button.clicked.connect(self._open)
            bar.addWidget(open_button)
            bar.addWidget(self._path_label, 1)
            layout.addLayout(bar)
            self.explorer = VolcanoExplorer(parent=self)
            layout.addWidget(self.explorer, 1)
            self.setAcceptDrops(True)

        def load(self, path) -> bool:
            table = find_results_table(path)
            if table is None:
                self._path_label.setText(
                    f"No results CSV found under {path}")
                return False
            self.explorer.set_results(load_results(table))
            self._path_label.setText(table)
            return True

        def _open(self) -> None:
            folder = QFileDialog.getExistingDirectory(
                self, "Choose a regression results folder")
            if folder:
                self.load(folder)

        def dragEnterEvent(self, event):  # noqa: N802 - Qt name
            if event.mimeData().hasUrls():
                event.acceptProposedAction()

        def dropEvent(self, event):  # noqa: N802 - Qt name
            for url in event.mimeData().urls():
                if url.isLocalFile() and self.load(url.toLocalFile()):
                    break
            event.acceptProposedAction()

    return VolcanoScreen(host=host)


def register() -> bool:
    """Add the module through spaCR's single application-registration seam."""
    from ..app import APPS, SECTION_RESULTS, STAGE_ALPHA, register_app
    if any(row[0] == APP_KEY for row in APPS):
        return False
    register_app(
        APP_KEY, APP_NAME, APP_DESCRIPTION, SECTION_RESULTS,
        factory=_make_screen,
        stage=STAGE_ALPHA, title=APP_NAME, intro=APP_INTRO,
        cli_note=APP_CLI_NOTE,
        translations=APP_TRANSLATIONS)
    return True


register()
