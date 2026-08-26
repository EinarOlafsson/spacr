"""Image Scatter and PCA views integrated with Image UMAP.

Image UMAP, Image Scatter and PCA provide complementary projections of the
same objects. Image Scatter plots two measured variables with crop previews
and linked selection. PCA provides feature selection, a scree plot, a
loadings biplot, local filtering and CSV export.

Both views open as complete pages beside the UMAP settings and inherit the
measurements database selected by the Image UMAP screen. Shared page, signal
and masthead integration is implemented by
:mod:`spacr.qt.screens.map_barcodes`.
"""

from __future__ import annotations

import logging
import os
from functools import partial
from typing import Callable, Dict, Optional, Tuple

from PySide6.QtWidgets import QWidget

from ..widgets.fold_strip import FoldStrip
from .map_barcodes import install_fold_strip

LOG = logging.getLogger(__name__)

#: Registry key of the screen this module hangs its strip on. "umap" is
#: what Image UMAP has always been keyed as; the name is the display name.
HOST_KEY = "umap"

#: Registry keys of the modules folded into it, in the order the strip
#: draws them: the two measured columns first, then the whole feature
#: block -- which is the order a user narrows a table in.
FOLDED_APPS: Tuple[str, ...] = ("image_scatter", "pca")

# What each of those two said as a TILE -- the name, the sentence and the
# maturity colour a button has to go on carrying once the row is dropped --
# lives in `spacr.qt.screens.map_barcodes.FOLD_FALLBACK`, because
# `map_barcodes.fold_description` is what `install_fold_strip` restates
# these buttons through, and that is the only table it reads. A second copy
# stood here and nothing consulted it.

#: Where a measurements database sits relative to a project folder, best
#: first. Written out here rather than imported because the one existing
#: resolver is private to :mod:`spacr.qt.dnd_handlers` and answers a
#: different question -- what a DROPPED path means -- while this one only
#: has to turn a settings value into a file.
DATABASE_CANDIDATES: Tuple[Tuple[str, ...], ...] = (
    ("measurements", "measurements.db"),
    ("measurements.db",),
)


def source_path(screen) -> str:
    """The project folder ``screen`` is pointed at, or "".

    Read from the settings form rather than remembered, so the answer is
    whatever is in the box at the moment the button is pressed. A
    list-valued ``src`` -- several plates through one run -- yields the
    first, which is the one whose measurements the other two views can
    actually plot: a scatter over two plates' tables is two populations on
    one pair of axes.
    """
    if screen is None:
        return ""
    model = getattr(screen, "_settings_model", None)
    collect = getattr(model, "collect", None) if model is not None else None
    if not callable(collect):
        return ""
    try:
        source = (collect() or {}).get("src")
    except Exception:
        LOG.debug("Could not read the Image UMAP source", exc_info=True)
        return ""
    if isinstance(source, (list, tuple)):
        source = source[0] if source else ""
    source = str(source or "").strip()
    if not source:
        return ""
    return os.path.abspath(os.path.expanduser(source))


def measurements_database(source: str) -> str:
    """The measurements database for ``source``, or "".

    Accepts the database itself and the project folder above it, which are
    the two things the UMAP screen's source box actually holds. A path
    that resolves to nothing returns "", and the folded screen then opens
    on its own Browse button rather than on a path that is not there.
    """
    source = str(source or "").strip()
    if not source:
        return ""
    source = os.path.abspath(os.path.expanduser(source))
    if os.path.isfile(source):
        return source if source.lower().endswith(
            (".db", ".sqlite", ".sqlite3")) else ""
    for parts in DATABASE_CANDIDATES:
        candidate = os.path.join(source, *parts)
        if os.path.isfile(candidate):
            return candidate
    return ""


def _build_image_scatter(host_window: Optional[QWidget] = None,
                         screen: Optional[QWidget] = None) -> QWidget:
    """Image Scatter's own screen, pointed at the host's database.

    Built through the module's own :func:`make_image_scatter_screen`, which
    is the one constructor the module offers -- it was the ``factory=`` of
    the registry row, and it is still the only place the screen is made.
    """
    from .image_scatter import make_image_scatter_screen

    view = make_image_scatter_screen()
    view.set_database(measurements_database(source_path(screen)))
    return view


def _build_pca(host_window: Optional[QWidget] = None,
               screen: Optional[QWidget] = None) -> QWidget:
    """PCA's own screen, pointed at the host's database.

    Built through the module's own :func:`make_pca_screen`, which is the one
    constructor the module offers -- it was the ``factory=`` of the registry
    row, and it is still the only place the screen is made.
    """
    from .pca import make_pca_screen

    view = make_pca_screen()
    database = measurements_database(source_path(screen))
    if database:
        view.load_path(database)
    return view


#: One builder per folded module. Each takes the main window and the host
#: screen; :func:`install_folds` binds the screen, so a builder still has
#: the one-argument shape
#: :func:`spacr.qt.screens.map_barcodes.install_fold_strip` calls it with.
BUILDERS: Dict[str, Callable[..., QWidget]] = {
    "image_scatter": _build_image_scatter,
    "pca": _build_pca,
}


def install_folds(screen: QWidget) -> Optional[FoldStrip]:
    """Put Image UMAP's fold strip on ``screen``'s masthead."""
    builders = {key: partial(build, screen=screen)
                for key, build in BUILDERS.items()}
    return install_fold_strip(screen, HOST_KEY, FOLDED_APPS, builders)
