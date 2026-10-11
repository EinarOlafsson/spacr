"""Lay out the regression results as one multi-panel figure.

The sheet is laid out like a journal figure: 6-12 panels, bold upper-case
letters top-left, reading order matching the argument, related panels adjacent
and sharing scales, and more white space between groups than within them.

The panel order is fixed and follows the reading order:

    A  volcano                what the screen found
    B  strongest effects      which genes, and how sure
    C  effect distribution    what the effects look like as a whole
    D  control separation     whether the assay worked at all
    E  guide agreement        whether the calls are corroborated
    F  p-value distribution   whether the correction means anything
    G  q-q                    whether the model was entitled to say it

Panels A and B give the result; panels C to G show whether it can be trusted.
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
    """A rendered sheet and everything needed to write its legend.

    :param figure: Matplotlib figure containing the rendered multi-panel grid.
    :param panels: successfully rendered panel records in figure-reading
        order; their sequence supplies legend letters and captions.
    :param skipped: panel records omitted from the grid, retaining their
        titles and failure reasons for the legend.
    """

    figure: object
    panels: List[Panel]
    skipped: List[Panel]

    def legend(self) -> str:
        """The figure legend, panel by panel.

        Generated from the panels themselves, so it matches the drawn figure.
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
    proportions used by the published figures. The column cap keeps each
    panel large enough to read.
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

    Panels whose data is absent are skipped and listed with the reason, never
    drawn as an empty frame.
    """
    import matplotlib.pyplot as plt

    target = target or theme_target()
    with figure_style(target):
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
        axes = figure.subplots(max(rows, 1), columns, squeeze=False).ravel()
        from .bundle import _register_figure_data

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
        figure.subplots_adjust(left=.09, right=.98, top=.93, bottom=.09,
                               wspace=.42, hspace=.52)
        recipes = [dict(slot=index, regression=panel.key,
                        regression_options=dict(alpha=alpha, effect_threshold=effect_threshold,
                                                highlight=highlight) if panel.key == "volcano" else {},
                        measurement=panel.title, letter=string.ascii_uppercase[index],
                        rect=list(axes[index].get_position().bounds))
                   for index, panel in enumerate(drawn)]
        if recipes:
            _register_figure_data(figure, frame, kind="regression_panel",
                                  grid=[rows, columns], panels=recipes)
        else:
            _register_figure_data(figure, frame, kind="sheet")
        return Sheet(figure=figure, panels=drawn, skipped=skipped)


def build_panel(key: str, frame, *, target: Optional[str] = None,
                figsize=(3.4, 2.6), **kwargs):
    """One panel on its own figure, for the grid view and for saving.

    :param key: panel name from :data:`spacr.figures.panels.REGISTRY`.
    :param frame: coefficient table consumed by the selected panel.
    :returns: ``(figure, Panel)``.
    """
    import matplotlib.pyplot as plt

    with figure_style(target or theme_target(), kind=key):
        figure = plt.figure(figsize=figsize)
        ax = figure.add_subplot(111)
        panel = REGISTRY[key](ax, frame, **kwargs)
        from .bundle import _register_figure_data
        figure.subplots_adjust(left=.16, right=.97, top=.92, bottom=.16)
        attach(figure, panel)
        recipe = dict(regression=key, regression_options=dict(kwargs),
                      rect=list(ax.get_position().bounds))
        if kwargs.get("compartment"):
            from ..localisation import table

            recipe["localisations"] = dict(table())
        _register_figure_data(figure, frame, kind="regression_panel", **recipe)
        figure._spacr_drawn_data = panel.data
        return figure, panel


def attach(figure, panel) -> None:
    """Attach a panel's name, data and groups to its figure.

    The figure carries them because the export sees only the figure. A user
    right-clicks a picture in the queue and asks to save it; nothing at that
    point knows which frame it came from or what was compared, unless the
    figure itself does.

    Private attributes on a matplotlib Figure rather than a wrapper object,
    because the figure is handed through the Qt bridge, the queue, a spill
    file and back, and a wrapper would be lost at the first of those.

    :param figure: Matplotlib figure that will carry the export metadata.
    :param panel: panel metadata to attach; ``None`` leaves the figure
        unchanged.
    """
    if panel is None:
        return
    figure.set_label(panel.title or panel.key)
    figure._spacr_title = panel.title or panel.key
    figure._spacr_caption = panel.caption
    if getattr(panel, "data", None) is not None:
        from .bundle import _register_figure_data
        _register_figure_data(figure, panel.data, kind=panel.key,
                              title=panel.title or panel.key)
    if getattr(panel, "groups", None):
        figure._spacr_groups = panel.groups


__all__ = ["CELL_HEIGHT", "Sheet", "WIDTHS", "attach", "build_panel", "build_sheet"]
