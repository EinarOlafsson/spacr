"""The whole regression as one publication-ready figure.

Asked for on 2026-08-16: "the all figures section should look like a
publication ready figure".

Not a gallery of separate pictures at separate sizes. One sheet, laid out the
way a journal figure is: 6-12 panels, bold upper-case letters top-left,
reading order matching the argument, related panels adjacent and sharing
scales, and more white space between groups than within them -- which is the
only hierarchy cue the published figures use.

READING ORDER IS THE ARGUMENT, and it is why the panel order is fixed rather
than alphabetical or whatever the dict happened to hold:

    A  volcano                what the screen found
    B  strongest effects      which genes, and how sure
    C  effect distribution    what the effects look like as a whole
    D  control separation     whether the assay worked at all
    E  guide agreement        whether the calls are corroborated
    F  p-value distribution   whether the correction means anything
    G  q-q                    whether the model was entitled to say it

A reader who stops after B has the result. One who reads to G knows whether
to believe it. A sheet ordered any other way asks them to take the result on
trust and audit it afterwards, which is not how anyone reads a figure.
"""

from __future__ import annotations

import string
from dataclasses import dataclass
from typing import List, Optional, Sequence

from .panels import REGISTRY, SHEET_ORDER, Panel
from .style import figure_style, hide_unused, panel_letter, theme_target

#: Single column, double column and full page, in inches. The published
#: figures are built to these and nothing else.
WIDTHS = {"single": 3.4, "double": 7.0, "full": 9.5}

#: A panel's height when it is one grid cell. Keeps the aspect near the
#: square the dense panels want without squashing the wide ones.
CELL_HEIGHT = 2.05


@dataclass
class Sheet:
    """A rendered sheet and everything needed to write its legend."""

    figure: object
    panels: List[Panel]
    skipped: List[Panel]

    def legend(self) -> str:
        """The figure legend, panel by panel.

        Generated from the panels themselves rather than written twice: a
        legend maintained by hand beside the code that draws the figure is a
        legend that describes last month's figure.
        """
        lines = []
        for letter, panel in zip(string.ascii_uppercase, self.panels):
            if panel.caption:
                lines.append(f"({letter}) {panel.caption}")
        if self.skipped:
            lines.append(
                "Not shown: "
                + "; ".join(f"{p.title} ({p.reason})" for p in self.skipped)
                + ".")
        return " ".join(lines)


def _grid(count: int, width: str) -> tuple:
    """Rows and columns for ``count`` panels at this width.

    Two columns at single width, three at double, four at full -- the
    proportions the published figures use. More columns than that and a panel
    is too small to read, which is the failure the aspect-ratio work
    (instruction 117) already fixed once for the grid view.
    """
    columns = {"single": 2, "double": 3, "full": 4}[width]
    columns = min(columns, max(count, 1))
    rows = -(-count // columns)
    return rows, columns


def build_sheet(frame, *, width: str = "double", target: Optional[str] = None,
                order: Sequence[str] = SHEET_ORDER, alpha: float = 0.05,
                effect_threshold: Optional[float] = None,
                highlight: Optional[str] = None) -> Sheet:
    """Draw every panel this table supports, as one figure.

    :param frame: the coefficient table.
    :param width: ``'single'``, ``'double'`` or ``'full'``.
    :param target: ``'screen'`` or ``'print'``; defaults to the user's own
        figure preference.
    :param highlight: a gene to ring on the volcano, so the sheet can follow
        the selection in the GUI.
    :returns: a :class:`Sheet`.

    Panels whose data is absent are SKIPPED AND NAMED, never drawn as an
    empty frame. A blank box in a figure sheet reads as a panel that failed,
    which is worse than a gap and much worse than a sentence saying why.
    """
    import matplotlib.pyplot as plt

    target = target or theme_target()
    with figure_style(target):
        # Draw once into a throwaway to find out which panels this table can
        # support, so the grid is sized for what will actually appear rather
        # than leaving holes.
        scratch = plt.figure()
        supported = []
        try:
            for key in order:
                ax = scratch.add_subplot(111)
                try:
                    result = REGISTRY[key](ax, frame)
                except Exception as error:      # noqa: BLE001
                    result = Panel(key, key, drawn=False, reason=str(error))
                if result.drawn:
                    supported.append(key)
                scratch.clear()
        finally:
            plt.close(scratch)

        rows, columns = _grid(len(supported), width)
        figure = plt.figure(figsize=(WIDTHS[width],
                                     max(rows, 1) * CELL_HEIGHT))
        axes = figure.subplots(rows, columns, squeeze=False).ravel()

        drawn: List[Panel] = []
        skipped: List[Panel] = []
        for index, key in enumerate(order):
            if key not in supported:
                ax = plt.figure().add_subplot(111)
                try:
                    skipped.append(REGISTRY[key](ax, frame))
                except Exception as error:      # noqa: BLE001
                    skipped.append(Panel(key, key, drawn=False,
                                         reason=str(error)))
                plt.close(ax.figure)
                continue
            ax = axes[len(drawn)]
            kwargs = {}
            if key == "volcano":
                kwargs = dict(alpha=alpha, effect_threshold=effect_threshold,
                              highlight=highlight)
            panel = REGISTRY[key](ax, frame, **kwargs)
            panel_letter(ax, string.ascii_uppercase[len(drawn)])
            drawn.append(panel)

        hide_unused(axes[len(drawn):])
        # More space between panel groups than within them: the only
        # hierarchy cue the published figures use.
        figure.subplots_adjust(left=.09, right=.98, top=.93, bottom=.09,
                               wspace=.42, hspace=.52)
        return Sheet(figure=figure, panels=drawn, skipped=skipped)


def build_panel(key: str, frame, *, target: Optional[str] = None,
                figsize=(3.4, 2.6), **kwargs):
    """One panel on its own figure, for the grid view and for saving.

    :returns: ``(figure, Panel)``.
    """
    import matplotlib.pyplot as plt

    with figure_style(target or theme_target()):
        figure = plt.figure(figsize=figsize)
        ax = figure.add_subplot(111)
        panel = REGISTRY[key](ax, frame, **kwargs)
        figure.subplots_adjust(left=.16, right=.97, top=.92, bottom=.16)
        attach(figure, panel)
        return figure, panel


def attach(figure, panel) -> None:
    """Hang a panel's name, data and groups on its figure.

    THE FIGURE HAS TO CARRY THEM because that is all the export sees. A user
    right-clicks a picture in the queue and asks to save it; nothing at that
    point knows which frame it came from or what was compared, unless the
    figure itself does.

    Private attributes on a matplotlib Figure rather than a wrapper object,
    because the figure is handed through the Qt bridge, the queue, a spill
    file and back, and a wrapper would be lost at the first of those.
    """
    if panel is None:
        return
    figure.set_label(panel.title or panel.key)
    figure._spacr_title = panel.title or panel.key
    figure._spacr_caption = panel.caption
    if getattr(panel, "data", None) is not None:
        figure._spacr_data = panel.data
    if getattr(panel, "groups", None):
        figure._spacr_groups = panel.groups


__all__ = ["CELL_HEIGHT", "Sheet", "WIDTHS", "attach", "build_panel", "build_sheet"]
