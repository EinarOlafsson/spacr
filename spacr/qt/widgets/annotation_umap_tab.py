"""Display held-out UMAP quality checks for annotated cells.

The panel delegates computation to :mod:`spacr.annotation_umap_qc`. It tunes
the embedding on one control subset and evaluates separation on held-out
controls, reports neighbour purity rather than cluster membership, and
refuses circular evaluation when phenotype scores were used to select the
annotated cells.

THE EMBEDDING IS DRAWN THROUGH THE SAME PLOT EVERY OTHER PANEL USES --
:class:`spacr.qt.widgets.fast_plots.FastPlot` beside
:class:`~spacr.qt.widgets.fast_plots.ResultsTable` -- rather than on a canvas
of its own. A scatter here needs exactly what those already carry: hovering a
point to see which guide it came from, the right-click restyle menu, the
export, and a table whose rows can be read against the picture. A second
canvas would be a second set of all of that, drifting from the first.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ...annotation_umap_qc import NEGATIVE, POSITIVE
from .fast_plots import FastPlot, ResultsTable

#: What a cell with no control label is called here.
ANNOTATED = "annotated"

#: What each cell is drawn as. Purity is the COLOUR, so the group has to be
#: the shape: a reader has to be able to tell a control from an annotated
#: cell, and a second colour scale over the first would fight the one that
#: carries the number.
#:
#: Keyed off the engine's own labels rather than off copies of them, so the
#: shapes cannot come to describe groups the scoring does not produce.
GROUP_SYMBOLS = {ANNOTATED: "o", POSITIVE: "t", NEGATIVE: "d"}

#: The colour scale purity is drawn on. One of
#: :data:`spacr.qt.widgets.fast_plots.COLORMAPS`, so the restyle menu can
#: offer the others without this panel keeping a list of its own.
PURITY_COLORMAP = "viridis"


class PurityScatter(FastPlot):
    """The embedding, one point per cell, coloured by neighbour purity.

    Colour is the reading and position is only the layout: a UMAP's cluster
    sizes and the distances between clusters are artefacts, so the picture
    exists to show WHERE a cell sits among the controls, and the number that
    says how positive that neighbourhood is comes off the colour scale.

    :param parent: parent widget.
    """

    def __init__(self, parent=None):
        super().__init__(title="Annotated cells among the controls",
                         x_label="UMAP 1", y_label="UMAP 2", parent=parent)

    def status(self) -> str:
        """What the plot is saying about itself right now.

        The status line is the scatter's legend -- what the colour means and
        what the shapes are -- so it is worth being able to read back rather
        than only to write.
        """
        return self._status.text()

    def clear_plot(self, message: str) -> None:
        """Take everything off the plot and say why it is empty."""
        self._reset_scene()
        self._frame = None
        self._style_note = ""
        self.set_status(str(message))

    def set_embedding(self, embedding, purity, guides, marks) -> int:
        """Draw ``embedding``, coloured by ``purity``. Returns points drawn.

        :param embedding: ``(n_cells, 2)`` coordinates.
        :param purity: one neighbour-purity value per cell, ``nan`` allowed.
        :param guides: the guide each cell was annotated with.
        :param marks: ``"PC"`` / ``"NC"`` / ``None`` per cell.

        A cell with no purity is drawn grey rather than at the bottom of the
        scale, which is what :meth:`FastPlot.colour_by_column` does with a
        missing value -- painting a ``nan`` dark would invent a measurement.
        """
        import numpy as np
        import pandas as pd

        points = np.asarray(embedding, dtype=float)
        self._reset_scene()
        if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] < 2:
            self.clear_plot("The embedding has no points to draw.")
            return 0
        groups = [ANNOTATED if mark is None else str(mark) for mark in marks]
        frame = pd.DataFrame({
            "umap_1": points[:, 0],
            "umap_2": points[:, 1],
            "purity": np.asarray(purity, dtype=float),
            "guide": [str(g) for g in guides],
            "group": groups,
        })
        self._frame = frame
        self.add_scatter(
            frame["umap_1"].to_numpy(), frame["umap_2"].to_numpy(),
            size=6.0,
            symbol_list=[GROUP_SYMBOLS.get(name, GROUP_SYMBOLS[ANNOTATED])
                         for name in groups],
            labels=[f"{guide} · {group}"
                    for guide, group in zip(frame["guide"], groups)])
        try:
            self.colour_by_column("purity", PURITY_COLORMAP)
        except ValueError as exc:
            # Every cell shares one purity, or none of them has a finite one.
            # The scatter is still worth showing; what is not true is that
            # the colour means anything, so it is said rather than implied.
            self.set_status(f"Purity is not drawn as a colour: {exc}")
            return int(len(frame))
        self.set_status(
            f"{len(frame):,} cells. Triangles are positive controls, "
            f"diamonds negative, circles the annotated cells; the brighter "
            f"a cell, the more of its control neighbours are positive.")
        return int(len(frame))


class AnnotationUmapTab(QWidget):
    """Embed the annotated cells with the controls and score where they sit.

    :param parent: the owning widget.

    Nothing is computed until :meth:`run` is called. A UMAP over a screen's
    cells is seconds of work, and a tab that started it on construction
    would pay that cost for every user who never opens it.
    """

    #: What the panel refuses to draw a verdict below. A held-out silhouette
    #: at or under zero means the controls did not separate on cells the
    #: search never saw, and every number after that describes the search
    #: rather than the screen.
    MINIMUM_SEPARATION = 0.0

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._frame = None
        self._embedding = None
        self._controls = []
        self._effects = {}

        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        row.addWidget(QLabel("Annotation method"))
        self.method = QComboBox()
        # THE ONES THIS CHECK CAN SPEAK ABOUT. `rank` is absent on purpose:
        # it takes the top-scoring cells in the well, so its cells sitting
        # near the positive controls restates how it chose them.
        from ...cell_montage import PICKING_MODES

        for name in PICKING_MODES:
            self.method.addItem(name, name)
        row.addWidget(self.method, 1)
        self.run_button = QPushButton("Embed and score")
        self.run_button.clicked.connect(self.run)
        row.addWidget(self.run_button)
        layout.addLayout(row)

        # The plot and the table are two views of ONE result, so they sit
        # side by side behind a divider the user owns: the picture says where
        # the cells landed, the table says by how much, and reading one
        # against the other is the whole job.
        self.body = QSplitter(Qt.Horizontal)
        self.body.setChildrenCollapsible(False)
        self.plot = PurityScatter()
        self.body.addWidget(self.plot)
        self.table = ResultsTable()
        self.body.addWidget(self.table)
        self.body.setStretchFactor(0, 3)
        self.body.setStretchFactor(1, 2)
        layout.addWidget(self.body, 1)

        self.report = QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setMaximumHeight(180)
        layout.addWidget(self.report)
        self.say("Pick a method and press Embed and score. Nothing is "
                 "computed until then.")
        self.clear_result("Nothing has been embedded yet.")

    # ------------------------------------------------------------------
    def say(self, text: str) -> None:
        """Put ``text`` in the report box, replacing what was there."""
        self.report.setPlainText(str(text))

    def clear_result(self, reason: str) -> None:
        """Empty the plot and the table, and say why they are empty.

        Called on every path that does not end in a verdict. A refusal that
        left the last run's scatter on screen would put a picture under a
        message saying this one means nothing, and the picture is what gets
        screenshotted.
        """
        self._embedding = None
        self.plot.clear_plot(reason)
        self.table.set_frame(None)

    def refuse(self, reason: str, said: str) -> None:
        """Show ``said`` and leave nothing drawn behind it."""
        self.clear_result(reason)
        self.say(said)

    def set_frame(self, frame, *, control_labels=None, effects=None) -> None:
        """Give the tab the cells to embed.

        :param frame: one row per cell, numeric measurement columns.
        :param control_labels: ``POSITIVE`` / ``NEGATIVE`` / ``None`` per
            row.
        :param effects: ``{guide: coefficient}`` for the agreement test.
        """
        self._frame = frame
        self._controls = list(control_labels or [])
        self._effects = dict(effects or {})

    # ------------------------------------------------------------------
    def run(self) -> dict:
        """Tune, embed, score, and report. Returns what it found."""
        from ...annotation_umap_qc import circularity_warning

        method = str(self.method.currentData() or "")
        warning = circularity_warning(method)
        if warning:
            # SHOWN INSTEAD OF THE PLOT, not beside it. A picture drawn
            # under a warning that it means nothing is still a picture
            # somebody will screenshot.
            self.refuse(
                "This method picked its cells by the phenotype score.",
                f"This check cannot judge {method!r}.\n\n{warning}\n\n"
                f"Choose a method whose cell selection did not use the "
                f"phenotype score.")
            return {"refused": warning}

        if self._frame is None or not len(self._frame):
            self.refuse("No cells are loaded.",
                        "No cells are loaded, so there is nothing to embed.")
            return {"error": "no cells"}

        return self._score(method)

    def _control_groups(self, control_rows):
        """Return one well identity per control row, or ``None``.

        :param control_rows: positions in the frame that carry a control
            label.
        :returns: ``(groups, level)``. ``groups`` is ``None`` when the frame
            carries no verifiable well identity, and ``level`` is then
            ``'cell'`` -- the leakiest rung of the shared ladder, chosen
            because it is the only one the data supports, never because the
            grouping was forgotten.
        """
        if self._frame is None:
            return None, "cell"
        try:
            from ...classifier_evaluation import split_group_values
            _, values = split_group_values(group_by="well",
                                           frame=self._frame.iloc[control_rows],
                                           table="control cells")
        except Exception:
            return None, "cell"
        return values, "well"

    def _score(self, method: str) -> dict:
        import numpy as np

        from ...annotation_umap_qc import (
            effect_agreement,
            fit_on_controls,
            neighbour_purity,
            purity_by_guide,
        )

        features = self._frame.select_dtypes(include=[np.number])
        marks = self._controls
        control_rows = [i for i, m in enumerate(marks) if m is not None]
        if len(control_rows) < 8:
            self.refuse(
                f"Only {len(control_rows)} control cell(s) are loaded.",
                f"Only {len(control_rows)} control cell(s) are loaded. "
                f"The embedding is tuned on half of them and scored on "
                f"the other half, so there is nothing to hold out.")
            return {"error": "too few controls"}

        recipes = [{"n_neighbors": n, "min_dist": d}
                   for n in (10, 15, 30) for d in (0.05, 0.1, 0.3)]
        # HOLD THE WELLS APART WHERE THE FRAME NAMES THEM. Sibling control
        # cells on both sides of the split would separate because they came
        # from the same well, and the held-out silhouette would report that
        # as biology. A frame that cannot name a well is split per object,
        # which the result records rather than hiding.
        groups, level = self._control_groups(control_rows)
        chosen = fit_on_controls(
            features.iloc[control_rows].to_numpy(dtype=float),
            [marks[i] for i in control_rows], recipes=recipes,
            groups=groups, group_by=level)
        if "error" in chosen:
            self.refuse(f"The embedding could not be tuned: {chosen['error']}",
                        f"The embedding could not be tuned: {chosen['error']}")
            return chosen

        held = float(chosen.get("holdout_silhouette", float("nan")))
        gap = float(chosen.get("overfit_gap", float("nan")))
        if not chosen.get("trustworthy") or held <= self.MINIMUM_SEPARATION:
            # REFUSED, WITH THE NUMBERS. This is the guard that matters: a
            # search that separates only the half it was tuned on has found
            # the split, not the biology.
            self.refuse(
                f"Held-out separation {held:.3f} is not enough to judge on.",
                f"The controls did not separate on cells the search never "
                f"saw.\n\n"
                f"  tuned silhouette    {chosen.get('tuned_silhouette'):.3f}\n"
                f"  held-out silhouette {held:.3f}\n"
                f"  gap                 {gap:.3f}\n\n"
                f"No verdict is drawn from this embedding. A search that "
                f"separates only the half it was tuned on has found the "
                f"split rather than the biology, and where the annotated "
                f"cells land in it would say nothing.")
            return dict(chosen, verdict="refused")

        from ...hyperparam import _default_umap_embed

        embedding = _default_umap_embed(
            features.to_numpy(dtype=float), dict(chosen["recipe"]), 0)
        self._embedding = embedding
        purity = neighbour_purity(embedding, marks)
        guides = [str(g) for g in self._frame.get(
            "montage_annotation", ["?"] * len(self._frame))]
        per_guide = purity_by_guide(purity, guides)
        agreement = effect_agreement(per_guide, self._effects)
        self.plot.set_embedding(embedding, purity, guides, marks)
        self.table.set_frame(self.guide_table(per_guide),
                             key_column="guide")

        lines = [
            f"Held-out separation of the controls: {held:.3f} "
            f"(gap to tuned {gap:.3f}).",
            f"{len(per_guide)} guide(s) had enough annotated cells to score.",
            "",
        ]
        if "error" in agreement:
            lines.append(agreement["error"])
        else:
            lines += [
                f"purity vs effect: rho = {agreement['correlation']:+.3f}, "
                f"p = {agreement['p_value']:.4f} "
                f"({agreement['permutations']:,} permutations)",
                f"  positive-effect guides mean purity "
                f"{agreement['positive_effect_purity']:.3f}",
                f"  negative-effect guides mean purity "
                f"{agreement['negative_effect_purity']:.3f}",
                "",
                ("The annotated cells land where their guide's effect says "
                 "they should." if agreement["separated"] else
                 "The annotated cells do NOT land where their guide's effect "
                 "says they should. Cells land somewhere whatever the "
                 "annotation says, which is why this is compared against "
                 "the effects shuffled between guides."),
            ]
        self.say("\n".join(lines))
        return {"separation": chosen, "purity": per_guide,
                "agreement": agreement}

    def guide_table(self, per_guide):
        """The per-guide rows as a table, purest first.

        :param per_guide: ``{guide: {"purity", "spread", "cells"}}`` from
            :func:`spacr.annotation_umap_qc.purity_by_guide`.

        The guide's effect is carried alongside its purity because agreeing
        is the claim being made: two columns a reader can put beside each
        other are what lets the correlation in the report be checked rather
        than believed. A guide with no effect gets an empty cell, not a
        zero -- zero is a coefficient somebody measured.
        """
        import numpy as np
        import pandas as pd

        rows = [{"guide": guide,
                 "purity": float(values.get("purity", float("nan"))),
                 "spread": float(values.get("spread", float("nan"))),
                 "cells": int(values.get("cells", 0)),
                 "effect": float(self._effects.get(guide, np.nan))}
                for guide, values in per_guide.items()]
        frame = pd.DataFrame(
            rows, columns=["guide", "purity", "spread", "cells", "effect"])
        if len(frame):
            frame = frame.sort_values("purity", ascending=False,
                                      ignore_index=True)
        return frame
