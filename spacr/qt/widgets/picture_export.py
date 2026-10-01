"""Right-click a picture and save it, as a PNG or as a PDF.

A picture on screen is often the thing somebody wants in a slide an hour
later, and the route to it was a screenshot of a window, at the window's
resolution, with the surrounding chrome in it. This saves THE PICTURE, at
the resolution it was rendered at rather than the size it happens to be
shown at, into a file of the user's choosing.

TWO FORMATS, AND THE PDF IS NOT A PNG WITH A DIFFERENT EXTENSION. A raster
dropped into a PDF satisfies the file name and nothing else; a figure panel
is a page, and a page has a size in millimetres. :func:`save_picture` gives
the PDF a page the image's own shape and tells the writer the resolution
Preferences carries, so the picture lands at the size it was measured at and
prints at that resolution.

The resolution and the default format are the ones the Figures preferences
already own -- :func:`spacr.qt.preferences.get_figure_png_dpi` and
:func:`spacr.qt.preferences.get_figure_format` -- so a user who has already
said "300 DPI, PDF" is not asked a second time in different words.
"""
from __future__ import annotations

import logging
import os
from typing import Callable, Optional

from PySide6.QtCore import QMarginsF, QRectF, QSizeF, Qt
from PySide6.QtGui import QImage, QPageLayout, QPageSize, QPainter, QPixmap

LOG = logging.getLogger(__name__)

#: Millimetres per inch. Named because the conversion appears three times
#: and a wrong one is a page that is silently the wrong size.
MM_PER_INCH = 25.4

#: What the save dialog offers, in the order it offers it. PNG first: it is
#: what a picture of pixels usually wants to be, and the PDF is there for
#: the person putting it in a figure.
FILTERS = "PNG image (*.png);;PDF document (*.pdf)"

#: Fallback resolution when the preference cannot be read. The same 300 the
#: Figures preference defaults to.
FALLBACK_DPI = 300


def picture_dpi() -> int:
    """The resolution a saved picture is written at.

    :returns: the Figures preference, or :data:`FALLBACK_DPI`.
    """
    try:
        from ..preferences import get_figure_png_dpi
        dpi = int(get_figure_png_dpi())
    except Exception:                                       # noqa: BLE001
        LOG.debug("could not read the figure DPI", exc_info=True)
        return FALLBACK_DPI
    return dpi if dpi > 0 else FALLBACK_DPI


def preferred_suffix() -> str:
    """``".pdf"`` or ``".png"``, following the Figures format preference."""
    try:
        from ..preferences import get_figure_format
        chosen = str(get_figure_format() or "").strip().lower()
    except Exception:                                       # noqa: BLE001
        LOG.debug("could not read the figure format", exc_info=True)
        return ".png"
    return ".pdf" if chosen == "pdf" else ".png"


def as_image(picture) -> Optional[QImage]:
    """Whatever was handed over, as a :class:`QImage`, or ``None``.

    A caller may hold a ``QPixmap`` (what a view shows), a ``QImage`` (what
    a renderer produced) or nothing at all yet.

    :param picture: a ``QPixmap``, a ``QImage`` or ``None``; a null image or
        any other type also gives ``None``.
    """
    if picture is None:
        return None
    if isinstance(picture, QImage):
        return picture if not picture.isNull() else None
    if isinstance(picture, QPixmap):
        if picture.isNull():
            return None
        return picture.toImage()
    return None


def save_picture(picture, path, dpi: Optional[int] = None) -> bool:
    """Write ``picture`` to ``path``, as a PNG or a PDF by its suffix.

    :param picture: a ``QPixmap`` or ``QImage``.
    :param path: where to write. ``.pdf`` writes a page; anything else
        writes a PNG.
    :param dpi: resolution; :func:`picture_dpi` when omitted.
    :returns: whether a file was written.
    """
    image = as_image(picture)
    if image is None:
        return False
    resolution = int(dpi or picture_dpi())
    if resolution <= 0:
        resolution = FALLBACK_DPI
    path = str(path)
    if path.lower().endswith(".pdf"):
        return _save_pdf(image, path, resolution)
    return _save_png(image, path, resolution)


def _save_png(image: QImage, path: str, dpi: int) -> bool:
    """Write the raster, carrying the resolution in the file's own header.

    ``setDotsPerMeterX`` is not decoration: a PNG with no resolution in it
    is imported at 72 DPI by every layout program there is, and the figure
    then arrives four times too big and has to be scaled by hand.
    """
    stamped = QImage(image)
    try:
        per_metre = int(round(dpi / MM_PER_INCH * 1000.0))
        stamped.setDotsPerMeterX(per_metre)
        stamped.setDotsPerMeterY(per_metre)
    except Exception:                                       # noqa: BLE001
        LOG.debug("could not stamp the resolution", exc_info=True)
    try:
        return bool(stamped.save(path, "PNG"))
    except Exception:                                       # noqa: BLE001
        LOG.debug("could not write %s", path, exc_info=True)
        return False


def _save_pdf(image: QImage, path: str, dpi: int) -> bool:
    """Write a one-page PDF the image's own shape, at ``dpi``.

    The page is sized from the pixels and the resolution rather than fixed,
    so a tall field is a tall page and nothing is stretched or letterboxed.
    """
    from PySide6.QtGui import QPdfWriter

    width_mm = max(1.0, image.width() / float(dpi) * MM_PER_INCH)
    height_mm = max(1.0, image.height() / float(dpi) * MM_PER_INCH)
    painter = None
    try:
        writer = QPdfWriter(path)
        writer.setResolution(int(dpi))
        writer.setPageSize(QPageSize(QSizeF(width_mm, height_mm),
                                     QPageSize.Millimeter))
        writer.setPageMargins(QMarginsF(0, 0, 0, 0), QPageLayout.Millimeter)
        painter = QPainter(writer)
        target = QRectF(0, 0, writer.width(), writer.height())
        painter.drawImage(target, image, QRectF(image.rect()))
    except Exception:                                       # noqa: BLE001
        LOG.debug("could not write %s", path, exc_info=True)
        return False
    finally:
        if painter is not None:
            try:
                painter.end()
            except Exception:                                # noqa: BLE001
                LOG.debug("the PDF painter would not close", exc_info=True)
    return os.path.isfile(path)


def suggested_name(stem: str) -> str:
    """A file name for ``stem``, with the preferred suffix on it.

    :param stem: the base name; whitespace runs become underscores, and an
        empty result (or ``None``) becomes ``picture``.
    """
    cleaned = "_".join(str(stem or "picture").split()).strip("_")
    return f"{cleaned or 'picture'}{preferred_suffix()}"


def ask_where_to_save(parent, stem: str) -> str:
    """Ask for a path. ``""`` when the user walked away.

    THE SUFFIX IS PUT BACK IF THE USER DROPS IT. A name typed without one
    would otherwise be written as a PNG whatever filter was selected, which
    is how "save as PDF" quietly produces a raster.

    :param parent: the widget the save dialog is parented to.
    :param stem: the file name offered, without a suffix; cleaned by
        :func:`suggested_name`.
    """
    from PySide6.QtWidgets import QFileDialog

    start = suggested_name(stem)
    chosen, selected = QFileDialog.getSaveFileName(
        parent, "Save picture", start, FILTERS)
    if not chosen:
        return ""
    if not os.path.splitext(chosen)[1]:
        chosen += ".pdf" if "pdf" in str(selected).lower() else ".png"
    return chosen


def build_menu(parent, enabled: bool):
    """The right-click menu: save as PNG, save as PDF.

    Each action carries its suffix as its ``data``, so the caller reads the
    choice off the action rather than comparing it against two references
    it had to keep.

    :param parent: the widget that owns the menu.
    :param enabled: ``False`` when there is no picture yet. The actions are
        SHOWN AND GREYED rather than left out, and a line underneath says
        why: a menu that is empty on one field and full on the next reads
        as a bug, not as a state.
    """
    from PySide6.QtWidgets import QMenu

    menu = QMenu(parent)
    png = menu.addAction("Save as PNG…")
    png.setData(".png")
    pdf = menu.addAction("Save as PDF…")
    pdf.setData(".pdf")
    for action in (png, pdf):
        action.setEnabled(bool(enabled))
    if not enabled:
        menu.addSeparator()
        menu.addAction("There is no picture here yet").setEnabled(False)
    return menu


def choose_format(view, point, enabled: bool) -> str:
    """Pop the menu at ``point`` and return the chosen suffix, or ``""``.

    THE MODAL CALL IS ALONE IN HERE so everything around it can be driven
    by a test. ``QMenu.exec`` spins an event loop of its own, and a test
    that reached it would hang rather than fail; substituting this one
    function is how the rest of the gesture is checked at all.

    :param view: the widget the menu belongs to; ``point`` is mapped to
        global coordinates through it.
    :param point: where the right-click happened, in ``view`` coordinates.
    :param enabled: ``False`` greys out the save actions, as in
        :func:`build_menu`.
    """
    menu = build_menu(view, enabled)
    chosen = menu.exec(view.mapToGlobal(point))
    if chosen is None:
        return ""
    return str(chosen.data() or "")


def save_as(view, picture, stem: str, want: str) -> str:
    """Ask for a path and write ``picture`` there. ``""`` if nothing was.

    :param view: the widget the save dialog and any warning are parented to.
    :param picture: the ``QPixmap`` or ``QImage`` to write.
    :param stem: the file name offered, without a suffix.
    :param want: the suffix the user picked in the menu. It WINS over what
        the file dialog came back with, so "save as PDF" followed by a name
        ending in ``.png`` still writes a PDF rather than quietly changing
        format because of a typed extension.
    :returns: the path written, or ``""``.
    """
    from PySide6.QtWidgets import QMessageBox

    path = ask_where_to_save(view, stem)
    if not path:
        return ""
    if os.path.splitext(path)[1].lower() != want:
        path = os.path.splitext(path)[0] + want
    if save_picture(picture, path):
        return path
    QMessageBox.warning(view, "Not saved",
                        f"{os.path.basename(path)} could not be written.")
    return ""


def install_picture_save(view, picture: Callable[[], object],
                         stem: str = "picture",
                         unless: Optional[Callable[[], bool]] = None) -> bool:
    """Give ``view`` a right-click menu that saves what it is showing.

    :param view: the widget the user right-clicks.
    :param picture: called when the menu is used; returns the ``QPixmap`` or
        ``QImage`` to write. Called AT SAVE TIME rather than now, so the
        menu always writes the picture currently on screen.
    :param stem: the file name offered, without a suffix. A callable is
        asked at save time, so a view whose contents change -- Overlay one
        moment, Flows the next -- offers the name of what is on it now.
    :param unless: asked first; ``True`` means the right button belongs to
        something else at this moment and no menu is shown. The live
        preview's ruler is cleared with a right-click while it is active,
        and a menu appearing over that would take a tool away to add a file
        dialog nobody asked for.
    :returns: whether the menu was installed.
    """
    def _menu(point) -> str:
        """Offer the save menu at ``point``; the path written, or ``""``.

        :param point: where the right-click landed, in ``view`` coordinates.
        """
        if unless is not None:
            try:
                if unless():
                    return ""
            except Exception:                                # noqa: BLE001
                LOG.debug("a picture menu guard raised", exc_info=True)
        image = as_image(picture())
        want = choose_format(view, point, image is not None)
        if not want or image is None:
            return ""
        name = stem() if callable(stem) else stem
        return save_as(view, image, name, want)

    try:
        view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        view.customContextMenuRequested.connect(_menu)
        view._spacr_picture_menu = _menu
        return True
    except Exception:                                       # noqa: BLE001
        LOG.debug("could not install the picture menu", exc_info=True)
        return False
