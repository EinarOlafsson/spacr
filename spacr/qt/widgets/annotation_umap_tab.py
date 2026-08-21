"""Where do the annotated cells land, among the controls?

Instruction 215. The engine is :mod:`spacr.annotation_umap_qc`; this is the
tab that drives it and shows what it found.

IT REFUSES BEFORE IT DRAWS. The three guards the engine implements only
matter if the panel honours them:

  * the embedding is tuned on HALF the controls and scored on the other
    half, and a poor held-out separation stops the verdict rather than
    decorating it -- searching until two labelled groups separate always
    succeeds, and the question is whether it survives on cells the search
    never saw;
  * the readout is neighbour purity, never cluster membership;
  * a method that PICKED cells by the phenotype score cannot be judged this
    way at all, and the warning is shown instead of the plot.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QPlainTextEdit,
                               QPushButton, QVBoxLayout, QWidget)


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

        self.canvas_holder = QVBoxLayout()
        layout.addLayout(self.canvas_holder, 1)

        self.report = QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setMaximumHeight(180)
        layout.addWidget(self.report)
        self.say("Pick a method and press Embed and score. Nothing is "
                 "computed until then.")

    # ------------------------------------------------------------------
    def say(self, text: str) -> None:
        """Put ``text`` in the report box, replacing what was there."""
        self.report.setPlainText(str(text))

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
            self.say(f"This check cannot judge {method!r}.\n\n{warning}\n\n"
                     f"Choose a method whose cell selection did not use the "
                     f"phenotype score.")
            return {"refused": warning}

        if self._frame is None or not len(self._frame):
            self.say("No cells are loaded, so there is nothing to embed.")
            return {"error": "no cells"}

        return self._score(method)

    def _score(self, method: str) -> dict:
        import numpy as np

        from ...annotation_umap_qc import (POSITIVE, effect_agreement,
                                           fit_on_controls, neighbour_purity,
                                           purity_by_guide)

        features = self._frame.select_dtypes(include=[np.number])
        marks = self._controls
        control_rows = [i for i, m in enumerate(marks) if m is not None]
        if len(control_rows) < 8:
            self.say(f"Only {len(control_rows)} control cell(s) are loaded. "
                     f"The embedding is tuned on half of them and scored on "
                     f"the other half, so there is nothing to hold out.")
            return {"error": "too few controls"}

        recipes = [{"n_neighbors": n, "min_dist": d}
                   for n in (10, 15, 30) for d in (0.05, 0.1, 0.3)]
        chosen = fit_on_controls(
            features.iloc[control_rows].to_numpy(dtype=float),
            [marks[i] for i in control_rows], recipes=recipes)
        if "error" in chosen:
            self.say(f"The embedding could not be tuned: {chosen['error']}")
            return chosen

        held = float(chosen.get("holdout_silhouette", float("nan")))
        gap = float(chosen.get("overfit_gap", float("nan")))
        if not chosen.get("trustworthy"):
            # REFUSED, WITH THE NUMBERS. This is the guard that matters: a
            # search that separates only the half it was tuned on has found
            # the split, not the biology.
            self.say(
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

        from ...annotation_umap_qc import fit_on_controls as _f  # noqa: F401
        from ...hyperparam import _default_umap_embed

        embedding = _default_umap_embed(
            features.to_numpy(dtype=float), dict(chosen["recipe"]), 0)
        self._embedding = embedding
        purity = neighbour_purity(embedding, marks)
        guides = [str(g) for g in self._frame.get(
            "montage_annotation", ["?"] * len(self._frame))]
        per_guide = purity_by_guide(purity, guides)
        agreement = effect_agreement(per_guide, self._effects)

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
