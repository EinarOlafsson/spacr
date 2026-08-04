"""Run Compare — two runs of the same project, side by side.

Pick a project, pick two of its runs, and the screen answers the three
questions that follow a re-run:

1. **What did I change?** The settings diff, grouped under the same
   headings the settings panel groups them by, showing only what moved.
   Two hundred keys are not a diff; two keys under *Cellpose* are.
2. **What came out?** Objects, wells and fields per plate and overall.
   A run that produced 12% fewer cells has a problem, and this is where
   it is visible first.
3. **Which hits moved?** Appeared, vanished, and — just as importantly —
   changed rank. A hit list whose membership is stable but whose top ten
   reshuffles every run is not a stable result.

All three come from :mod:`spacr.run_compare`, which is headless: this
file picks the runs and draws the tables and knows nothing else. The run
lists come from the artifact registry (:mod:`spacr.artifacts`), never
from a filesystem scan — a run whose outputs were deleted still has its
settings recorded, and dropping it from the dropdown would lose the only
copy of what produced the numbers somebody is asking about.

**Incomparable runs are not diffed.** Two runs of different plates
subtract perfectly well and produce a table that looks exactly like a
regression report. When :func:`spacr.run_compare.comparability` raises a
blocker the tables stay empty and the banner says why, with a *Compare
anyway* button for the user who knows better. Warnings — a different
spaCR version above all — are shown with the tables rather than instead
of them, because a version change explains a count change on its own.
"""
from __future__ import annotations

import os
from typing import Any, List, Optional, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...run_compare import RunComparison, RunRef, compare_runs, runs_in
from ..theme import SPACING, pane_surface, register_widget_qss

__all__ = ["RunCompareScreen", "APP_KEY", "register"]

#: The app key this screen is registered under. Load-bearing: saved user
#: state, the command palette and ``spacr-qt run_compare`` all key off it.
APP_KEY = "run_compare"

#: The paragraph under this app's header, handed to the seam as ``intro``.
APP_INTRO = (
    "Two runs of the same project, side by side. What you changed (the "
    "settings diff, grouped the way the settings panel groups them, showing "
    "only what moved), what came out (objects, wells and fields per plate) "
    "and which hits moved — appeared, vanished, or just changed rank. Runs "
    "that are not comparable are not diffed: the banner says why, and you "
    "can override it.")

#: Why there is no ``spacr-run run_compare``; reaches
#: ``spacr.cli.INTERACTIVE_ONLY``, which prints it instead of "unknown
#: module".
APP_CLI_NOTE = (
    "Run Compare is an interactive side-by-side of two runs; headless, call "
    "spacr.run_compare.runs_in(project) to list them and "
    "spacr.run_compare.compare_runs(a, b) for the same three tables.")

#: "Run Compare" in the nine non-English UI languages, in
#: :data:`spacr.qt.i18n.LANGUAGES` order after English — sv, de, es, zh_CN,
#: pt, hi, ko, is, fr. "Run" is a pipeline run, not the verb.
APP_TRANSLATIONS = (
    "Jämför körningar",
    "Läufe vergleichen",
    "Comparar ejecuciones",
    "运行对比",
    "Comparar execuções",
    "रन तुलना",
    "실행 비교",
    "Bera saman keyrslur",
    "Comparer les exécutions",
)

#: Column layouts, so the tests and the drawing code cannot disagree.
_SETTINGS_COLUMNS = ("Setting", "A", "B", "Change")
_COUNT_COLUMNS = ("Count", "A", "B", "Δ", "%")
_HIT_COLUMNS = ("Hit", "A rank", "B rank", "Move", "A", "B")


def _banner_qss(palette: dict, opacity) -> str:
    """QSS for the verdict banner, registered through the theme seam.

    Two states, and the difference has to be legible at a glance because
    it is the difference between "these numbers mean something" and
    "these numbers are two different experiments": a blocked comparison
    takes the error colour, everything else the ordinary panel surface.
    The tables underneath are deliberately unstyled here — the shipped
    ``QHeaderView::section`` chips and ``::item:hover`` accent already
    carry the page opacity, and a screen that restyles them is a screen
    that stops following the theme.
    """
    surface = pane_surface("surface_alt", palette["theme"], opacity)
    return f"""
QFrame#RunCompareBanner {{
    background: {surface};
    border: 1px solid {palette["border_soft"]};
    border-radius: 8px;
}}
QFrame#RunCompareBanner[blocked="true"] {{
    border: 1px solid {palette["error"]};
}}
QLabel#RunCompareVerdict {{
    font-weight: 600;
}}
QLabel#RunCompareVerdict[blocked="true"] {{
    color: {palette["error"]};
}}
"""


# ``replace=True`` because this module owns the name: a reimport (a test
# that reloads it, a plugin that pulls it in twice) must re-register the
# same block rather than raise on the duplicate and leave the screen
# unstyled.
register_widget_qss("RunCompareBanner", _banner_qss, replace=True)


class RunCompareScreen(QWidget):
    """Put two runs of one project side by side.

    :param parent: Qt parent.
    :param project: open straight onto this project root, skipping the
        folder picker. Tests and the "compare with the run that just
        finished" path both use it.
    :ivar last_error: text of the most recent failure, ``""`` when the
        last operation worked. Failures land here and in the banner —
        never in a modal dialog, which hangs a headless run.
    """

    #: emitted with the :class:`~spacr.run_compare.RunComparison` after
    #: every comparison, including the ones that refused to diff.
    compared = Signal(object)
    #: emitted with the project root whenever a project's runs are loaded
    project_loaded = Signal(str)

    def __init__(self, parent=None, project: str = ""):
        super().__init__(parent)
        self._runs: List[RunRef] = []
        self._comparison: Optional[RunComparison] = None
        self._force = False
        self.last_error: str = ""

        self._build_ui()
        if project:
            self.load_project(project)
        else:
            self._set_verdict(
                "Choose a spaCR project folder to list the runs it recorded.",
                blocked=False)
        # Drop anywhere on this screen: the path is resolved through spaCR's
        # project layout, so the plate folder finds what this screen reads.
        from ..dnd import install_for
        install_for(self, "run_compare")

    # -- construction -----------------------------------------------------

    def _build_ui(self) -> None:
        """Lay the screen out: picker, banner, then the three tabs."""
        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACING["lg"], SPACING["lg"],
                                 SPACING["lg"], SPACING["lg"])
        outer.setSpacing(SPACING["md"])

        title = QLabel("Run Compare")
        title.setObjectName("ScreenTitle")
        outer.addWidget(title)

        subtitle = QLabel(
            "Two runs of the same project: what settings moved, what the "
            "counts did, and which hits changed.")
        subtitle.setObjectName("Muted")
        subtitle.setWordWrap(True)
        outer.addWidget(subtitle)

        picker = QHBoxLayout()
        picker.setSpacing(SPACING["sm"])
        self._project_edit = QLineEdit()
        self._project_edit.setPlaceholderText("Project folder")
        self._project_edit.returnPressed.connect(self._on_project_entered)
        picker.addWidget(QLabel("Project"))
        picker.addWidget(self._project_edit, 1)
        self._browse_button = QPushButton("Browse…")
        self._browse_button.clicked.connect(self._on_browse)
        picker.addWidget(self._browse_button)
        outer.addLayout(picker)

        runs = QHBoxLayout()
        runs.setSpacing(SPACING["sm"])
        self._a_combo = QComboBox()
        self._b_combo = QComboBox()
        for label, combo in (("A (baseline)", self._a_combo),
                             ("B (compared)", self._b_combo)):
            runs.addWidget(QLabel(label))
            combo.setMinimumWidth(240)
            combo.currentIndexChanged.connect(self._on_run_selected)
            runs.addWidget(combo, 1)
        self._compare_button = QPushButton("Compare")
        self._compare_button.clicked.connect(self.compare)
        runs.addWidget(self._compare_button)
        outer.addLayout(runs)

        self._banner = QFrame()
        self._banner.setObjectName("RunCompareBanner")
        banner_row = QHBoxLayout(self._banner)
        banner_row.setContentsMargins(SPACING["md"], SPACING["sm"],
                                      SPACING["md"], SPACING["sm"])
        banner_row.setSpacing(SPACING["sm"])
        self._verdict = QLabel("")
        self._verdict.setObjectName("RunCompareVerdict")
        self._verdict.setWordWrap(True)
        banner_row.addWidget(self._verdict, 1)
        self._force_button = QPushButton("Compare anyway")
        self._force_button.setToolTip(
            "Produce the three tables despite the blocker above. The "
            "numbers will be subtracted; whether that means anything is "
            "on you.")
        self._force_button.clicked.connect(self._on_force)
        self._force_button.setVisible(False)
        banner_row.addWidget(self._force_button)
        outer.addWidget(self._banner)

        options = QHBoxLayout()
        self._show_all = QCheckBox("Show unchanged settings")
        self._show_all.setToolTip(
            "Off, the settings tab shows only what differs — which is the "
            "question. On, it shows every key both runs set, so you can "
            "confirm that something you expected to change did not.")
        self._show_all.toggled.connect(self._on_show_all)
        options.addWidget(self._show_all)
        options.addStretch(1)
        outer.addLayout(options)

        self._tabs = QTabWidget()
        self._settings_tree = _tree(_SETTINGS_COLUMNS)
        self._counts_tree = _tree(_COUNT_COLUMNS)
        self._hits_tree = _tree(_HIT_COLUMNS)
        self._tabs.addTab(self._settings_tree, "Settings")
        self._tabs.addTab(self._counts_tree, "Counts")
        self._tabs.addTab(self._hits_tree, "Hits")
        outer.addWidget(self._tabs, 1)

    # -- project + runs ---------------------------------------------------

    def load_project(self, project: str) -> List[RunRef]:
        """Fill both dropdowns with the runs ``project`` has registered.

        :param project: the project root.
        :returns: the runs found, newest first. Empty when the project has
            no registry yet — which is what a project that predates the
            artifact registry looks like, and the banner says so rather
            than the screen looking broken.
        """
        self.last_error = ""
        self._project_edit.setText(project)
        self._runs = []
        try:
            from ...artifacts import Registry
            registry = Registry(project=project, create=False)
            self._runs = runs_in(registry, project)
        except FileNotFoundError:
            self._set_verdict(
                f"{os.path.basename(project) or project} has no artifact "
                f"registry, so nothing here recorded what it ran. Runs "
                f"started from this version of spaCR will appear.",
                blocked=False)
        except Exception as exc:                      # pragma: no cover
            self.last_error = str(exc)
            self._set_verdict(f"Could not read that project: {exc}",
                              blocked=True)

        self._fill_combos()
        self.project_loaded.emit(project)
        if self._runs and len(self._runs) < 2:
            self._set_verdict(
                "Only one run is recorded here, so there is nothing to "
                "compare it against yet.", blocked=False)
        elif self._runs:
            self.compare()
        return list(self._runs)

    def _fill_combos(self) -> None:
        """Repopulate both dropdowns, defaulting to the two newest runs."""
        for combo in (self._a_combo, self._b_combo):
            combo.blockSignals(True)
            combo.clear()
            for run in self._runs:
                combo.addItem(run.label, run.run_id)
            combo.blockSignals(False)
        if len(self._runs) >= 2:
            # Newest as B, the one before it as A: the question is almost
            # always "what did the run I just did do differently?", and
            # that reads better as a change *into* the newest.
            self._a_combo.setCurrentIndex(1)
            self._b_combo.setCurrentIndex(0)

    def runs(self) -> List[RunRef]:
        """The runs currently listed, newest first."""
        return list(self._runs)

    def selected_runs(self) -> Tuple[Optional[RunRef], Optional[RunRef]]:
        """``(A, B)`` as the dropdowns currently stand."""
        return (self._run_at(self._a_combo.currentIndex()),
                self._run_at(self._b_combo.currentIndex()))

    def _run_at(self, index: int) -> Optional[RunRef]:
        """The run at a combo index, or ``None``."""
        if 0 <= index < len(self._runs):
            return self._runs[index]
        return None

    def select(self, a: str, b: str) -> None:
        """Select two runs by run id and compare them.

        :param a: the baseline run's id.
        :param b: the compared run's id.
        """
        ids = [run.run_id for run in self._runs]
        for run_id, combo in ((a, self._a_combo), (b, self._b_combo)):
            if run_id in ids:
                combo.setCurrentIndex(ids.index(run_id))
        self.compare()

    # -- comparing --------------------------------------------------------

    def compare(self, *, force: Optional[bool] = None) -> Optional[RunComparison]:
        """Compare the two selected runs and redraw the three tables.

        :param force: diff even when the runs are not comparable. ``None``
            keeps whatever the *Compare anyway* button last set — which is
            reset every time the selection changes, so forcing one pair
            never silently forces the next.
        :returns: the :class:`~spacr.run_compare.RunComparison`, or
            ``None`` when two runs are not selected.
        """
        if force is not None:
            self._force = bool(force)
        a, b = self.selected_runs()
        if a is None or b is None:
            self._clear_tables()
            return None
        comparison = compare_runs(a, b,
                                  include_same_settings=self._show_all.isChecked(),
                                  force=self._force)
        self._comparison = comparison
        self._draw(comparison)
        self.compared.emit(comparison)
        return comparison

    def comparison(self) -> Optional[RunComparison]:
        """The most recent comparison, or ``None``."""
        return self._comparison

    def _on_run_selected(self, _index: int) -> None:
        """A dropdown moved: drop any forcing, then compare."""
        self._force = False
        self.compare()

    def _on_show_all(self, _checked: bool) -> None:
        """The unchanged-settings toggle moved."""
        if self._comparison is not None:
            self.compare()

    def _on_force(self) -> None:
        """*Compare anyway* was pressed."""
        self.compare(force=True)

    def _on_browse(self) -> None:
        """Pick a project folder."""
        chosen = QFileDialog.getExistingDirectory(self, "Choose a project")
        if chosen:
            self.load_project(chosen)

    def _on_project_entered(self) -> None:
        """The project field was committed with Return."""
        text = self._project_edit.text().strip()
        if text:
            self.load_project(text)

    # -- drawing ----------------------------------------------------------

    def _draw(self, comparison: RunComparison) -> None:
        """Redraw the banner and all three tabs."""
        blocked = not comparison.comparable
        self._set_verdict(comparison.headline(), blocked=blocked)
        self._force_button.setVisible(
            blocked and bool(comparison.comparability.blockers))
        if blocked:
            self._clear_tables()
            return
        _fill_settings(self._settings_tree, comparison)
        _fill_counts(self._counts_tree, comparison)
        _fill_hits(self._hits_tree, comparison)
        self._tabs.setTabText(0, f"Settings ({len(comparison.settings)})")
        self._tabs.setTabText(
            1, f"Counts ({len(comparison.counts.changed)})")
        self._tabs.setTabText(
            2, f"Hits ({len(comparison.hits.appeared)}"
               f"/{len(comparison.hits.vanished)})")

    def _clear_tables(self) -> None:
        """Empty all three tabs and reset their labels."""
        for tree in (self._settings_tree, self._counts_tree, self._hits_tree):
            tree.clear()
        for index, label in enumerate(("Settings", "Counts", "Hits")):
            self._tabs.setTabText(index, label)

    def _set_verdict(self, text: str, *, blocked: bool) -> None:
        """Write the banner and restyle it for the blocked state."""
        self._verdict.setText(text)
        for widget in (self._banner, self._verdict):
            widget.setProperty("blocked", "true" if blocked else "false")
            # A dynamic property only reaches the stylesheet after the
            # widget is re-polished; without this the error border never
            # appears until something else forces a restyle.
            widget.style().unpolish(widget)
            widget.style().polish(widget)

    def verdict_text(self) -> str:
        """What the banner currently says."""
        return self._verdict.text()


# ---------------------------------------------------------------------------
# Table filling — module functions so they are testable without a screen
# ---------------------------------------------------------------------------

def _tree(columns: Tuple[str, ...]) -> QTreeWidget:
    """A grouped, read-only table carrying the shipped table styling."""
    tree = QTreeWidget()
    tree.setColumnCount(len(columns))
    tree.setHeaderLabels(list(columns))
    tree.setAlternatingRowColors(True)
    tree.setRootIsDecorated(True)
    tree.setUniformRowHeights(True)
    header = tree.header()
    header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
    header.setStretchLastSection(True)
    return tree


def _fill_settings(tree: QTreeWidget, comparison: RunComparison) -> None:
    """Draw the settings diff, one top-level row per category."""
    tree.clear()
    diff = comparison.settings
    if diff is None:
        return
    if diff.identical and not diff.include_same:
        tree.addTopLevelItem(QTreeWidgetItem(["No setting changed.", "", "", ""]))
        return
    for block in diff.categories:
        header = QTreeWidgetItem([
            block.category,
            "", "",
            f"{len(block)} changed, {block.n_same} unchanged",
        ])
        tree.addTopLevelItem(header)
        for row in block.rows + block.same:
            header.addChild(QTreeWidgetItem([
                row.key, _render(row.a_val), _render(row.b_val), row.kind,
            ]))
        header.setExpanded(bool(block.rows))


def _fill_counts(tree: QTreeWidget, comparison: RunComparison) -> None:
    """Draw the count diff: overall, then one group per plate."""
    tree.clear()
    diff = comparison.counts
    if diff is None:
        return
    if not diff.available:
        tree.addTopLevelItem(QTreeWidgetItem([diff.note or "No counts.",
                                              "", "", "", ""]))
        return
    groups: List[Tuple[str, Tuple[Any, ...]]] = [("Overall", diff.overall())]
    groups += [(f"Plate {plate}", diff.for_plate(plate))
               for plate in diff.plates]
    for label, rows in groups:
        if not rows:
            continue
        moved = sum(1 for row in rows if row.changed)
        header = QTreeWidgetItem([label, "", "", "",
                                  f"{moved} of {len(rows)} moved"])
        tree.addTopLevelItem(header)
        for row in rows:
            pct = "—" if row.pct is None else f"{row.pct:+.1f}%"
            delta = "—" if row.delta is None else f"{row.delta:+,}"
            header.addChild(QTreeWidgetItem([
                row.metric, _number(row.a), _number(row.b), delta, pct,
            ]))
        header.setExpanded(True)


def _fill_hits(tree: QTreeWidget, comparison: RunComparison) -> None:
    """Draw the hit-list diff: appeared, vanished, then rank churn."""
    tree.clear()
    diff = comparison.hits
    if diff is None:
        return
    if not diff.available:
        tree.addTopLevelItem(QTreeWidgetItem([diff.note or "No hit list.",
                                              "", "", "", "", ""]))
        return
    groups = (
        ("Appeared", diff.appeared),
        ("Vanished", diff.vanished),
        ("Changed rank", diff.moved),
        ("Held rank", diff.held),
    )
    for label, changes in groups:
        if not changes:
            continue
        header = QTreeWidgetItem([label, "", "", str(len(changes)), "", ""])
        tree.addTopLevelItem(header)
        for change in changes:
            move = ("—" if change.rank_delta is None
                    else f"{-change.rank_delta:+d}")
            # No status column: the group heading already says what
            # happened, and repeating it in every row is noise.
            header.addChild(QTreeWidgetItem([
                change.key,
                _number(change.a_rank), _number(change.b_rank), move,
                _score(change.a_score), _score(change.b_score),
            ]))
        # Held ranks are the boring group and the biggest; collapsed so
        # the two that matter are what the tab opens on.
        header.setExpanded(label != "Held rank")


def _render(value: Any) -> str:
    """A settings value as one cell."""
    return "—" if value is None else repr(value)


def _number(value: Optional[int]) -> str:
    """An integer as one cell; an em dash when that side has no such row."""
    return "—" if value is None else f"{value:,}"


def _score(value: Optional[float]) -> str:
    """An effect size as one cell."""
    return "—" if value is None else f"{value:.4g}"


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register() -> bool:
    """Add Run Compare to the app registry. Idempotent.

    Called at import time so that importing this module is all it takes
    for the app to exist — the registration seam from ``1a5ac2ab``. It
    returns rather than raises on a duplicate key so a re-import (a
    reloaded module, a test that cleared the registry) is a no-op instead
    of taking the import down.

    It is also named in ``spacr.qt.app._SELF_REGISTERING_APPS`` and in
    :data:`spacr.qt.SELF_REGISTERING_MODULES`, which is belt and braces
    rather than a mistake: the first is what makes the row exist under a
    bare ``import spacr.qt.app`` — an inventory that depended on whether
    something else had imported this module is an inventory that fails on
    whichever file pytest collected first — and the second is the launch
    path, which must still work if the first ever fails. All three calls
    land on this function and it registers once.

    **GUI-only.** ``cli_note`` and no ``entry``: the answer this screen
    gives is three tables you read against each other, and
    :mod:`spacr.run_compare` is already the headless half — the note names
    the two functions rather than wrapping them in a settings file that
    would have to invent a spelling for "these two runs".

    :returns: True when this call is what registered it.
    """
    from ..app import APPS, SECTION_RESULTS, STAGE_ALPHA, register_app
    if any(row[0] == APP_KEY for row in APPS):
        return False
    register_app(
        APP_KEY, "Run Compare",
        "Put two runs side by side: settings, counts and hit-list diffs",
        SECTION_RESULTS,
        factory=lambda: RunCompareScreen(),
        stage=STAGE_ALPHA,
        title="Run Compare",
        intro=APP_INTRO,
        cli_note=APP_CLI_NOTE,
        api_module="qt/screens/run_compare",
        translations=APP_TRANSLATIONS,
    )
    return True


register()
