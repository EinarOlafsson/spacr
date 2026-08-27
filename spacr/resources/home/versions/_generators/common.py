"""Provide shared bootstrap data for the Home-screen variant generators.

These modules render thirty candidate Home screens from the real Qt
widgets (``spacr.qt.widgets.tile.HTile``, ``Card``, ``Section``, …) and
the real app registry (``spacr.qt.app.APPS``), then grab each one to a
PNG under ``spacr/resources/home/versions/``.

Nothing here is installed into the app. The output is a layout-review surface,
and using production widgets ensures each candidate is buildable.

Determinism
-----------
* ``QT_QPA_PLATFORM=offscreen`` and a throwaway ``QSettings`` path, so
  the renders do not depend on the reviewer's saved theme / font scale.
* No live system state. Every "recent run", disk figure and GPU number
  is a fixed literal in :data:`MOCK` — a home screen that renders
  differently every run cannot be reviewed.
"""
from __future__ import annotations

import os
import sys
import tempfile
from typing import Dict, List, Sequence, Tuple

#: The realistic laptop case. Every variant is grabbed at exactly this.
CANVAS_W = 1440
CANVAS_H = 900


def here() -> str:
    """Absolute path of this ``_generators`` directory."""
    return os.path.dirname(os.path.abspath(__file__))


def versions_dir() -> str:
    """Absolute path of ``spacr/resources/home/versions``."""
    return os.path.normpath(os.path.join(here(), ".."))


def repo_root() -> str:
    """Absolute path of the spacr checkout root (five levels up)."""
    return os.path.normpath(os.path.join(here(), *([".."] * 5)))


def _prefer_checkout_package() -> None:
    """Make this generator import ``spacr`` from the checkout it writes."""
    root = repo_root()
    normalized = os.path.normcase(os.path.realpath(root))
    sys.path[:] = [entry for entry in sys.path
                   if os.path.normcase(os.path.realpath(entry or os.getcwd()))
                   != normalized]
    sys.path.insert(0, root)

    loaded = sys.modules.get("spacr")
    if loaded is None:
        return
    origins = []
    loaded_file = getattr(loaded, "__file__", "")
    if loaded_file:
        origins.append(loaded_file)
    origins.extend(str(path) for path in getattr(loaded, "__path__", ()))
    package_root = os.path.join(normalized, "spacr")
    is_local = any(
        os.path.commonpath((package_root, os.path.realpath(origin)))
        == package_root
        for origin in origins
    )
    if not is_local:
        for module_name in [name for name in sys.modules
                            if name == "spacr" or name.startswith("spacr.")]:
            sys.modules.pop(module_name, None)


# The tables below read the registry during module import, before
# :func:`bootstrap` is called. Select the checkout now so those tables and the
# later renderer cannot disagree about which spaCR tree they represent.
_prefer_checkout_package()


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def bootstrap():
    """Create (or return) the offscreen QApplication, isolated + fonted.

    Redirects ``QSettings`` at a temp directory *before* anything reads
    a preference, so ``preferences.get_font_scale()`` (which
    ``HTile``/``scaled_px`` consult) always answers 1.0 regardless of
    what the reviewer has saved.
    """
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    _prefer_checkout_package()

    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    # QSettings.setDefaultFormat / setPath are PROCESS-GLOBAL. Redirecting
    # them when we did not create the QApplication reaches into a host that
    # is already running: under pytest-qt it repoints every other test's
    # preferences at a temp directory mid-session, which is how this file
    # took the whole tests/qt suite down with a segfault. Only isolate when
    # this really is our own standalone process.
    global _WE_OWN_THE_APP
    app = QApplication.instance()
    if app is None:
        # NativeFormat as well as Ini. `preferences._settings()` builds
        # `QSettings("spacr", "qt")`, which is a NativeFormat object and
        # ignores setDefaultFormat/setPath(IniFormat, ...) — redirecting only
        # Ini left every render reading the operator's own saved font scale
        # and theme, so "deterministic" renders differed per machine.
        sandbox = tempfile.mkdtemp(prefix="spacr-home-variants-")
        QSettings.setDefaultFormat(QSettings.IniFormat)
        for fmt in (QSettings.NativeFormat, QSettings.IniFormat):
            QSettings.setPath(fmt, QSettings.UserScope, sandbox)
        app = QApplication(sys.argv[:1])
        _WE_OWN_THE_APP = True
    _load_fonts()
    return app


def _load_fonts() -> None:
    """Register the bundled Open Sans faces so metrics match the app."""
    from PySide6.QtGui import QFontDatabase
    fonts = os.path.join(repo_root(), "spacr", "qt", "resources", "fonts")
    if not os.path.isdir(fonts):
        return
    for name in sorted(os.listdir(fonts)):
        if name.lower().endswith((".ttf", ".otf")):
            QFontDatabase.addApplicationFont(os.path.join(fonts, name))


def available_themes() -> Tuple[str, ...]:
    """Return dark and light themes, plus space when its palette is available."""
    out = ["dark", "light"]
    try:
        from spacr.qt.theme import palette_for
        pal = palette_for("space")
        if isinstance(pal, dict) and pal.get("bg") and pal is not palette_for("dark"):
            out.append("space")
    except Exception:
        pass
    return tuple(out)


# ---------------------------------------------------------------------------
# The real app registry
# ---------------------------------------------------------------------------

def _registry():
    # ``spacr.qt.run`` performs these registrations before constructing
    # MainWindow. Home and the sidebar therefore show this launched registry,
    # not the shorter import-time table from ``app.py`` alone.
    import spacr.qt

    spacr.qt.register_self_registering_modules()
    from spacr.qt.app import _FORCE_GLYPH, _ICON_OVERRIDES, APPS
    return APPS, _ICON_OVERRIDES, _FORCE_GLYPH


def apps() -> List[Tuple[str, str, str, str]]:
    """The real ``(key, name, blurb, section)`` list, unmodified."""
    return list(_registry()[0])


def app_map() -> Dict[str, Tuple[str, str, str, str]]:
    """``key -> (key, name, blurb, section)``."""
    return {row[0]: row for row in apps()}


def name_of(key: str) -> str:
    """Display name of an app key."""
    return app_map()[key][1]


def blurb_of(key: str) -> str:
    """One-line description of an app key."""
    return app_map()[key][2]


def all_keys() -> List[str]:
    """Every app key, in registry order."""
    return [row[0] for row in apps()]


def core_keys() -> List[str]:
    """Return Core-pipeline app keys in registry order.

    The registry's :data:`spacr.qt.app.SECTION_CORE` constant defines the
    section identity so display-name changes cannot empty the result.
    """
    from spacr.qt.app import SECTION_CORE
    return [row[0] for row in apps() if row[3] == SECTION_CORE]


def n_apps() -> int:
    """Return the current number of registered apps."""
    return len(all_keys())


def section_names() -> List[str]:
    """Return non-empty registry sections in their displayed order."""
    from spacr.qt.app import SECTIONS

    occupied = {row[3] for row in apps()}
    return [section for section in SECTIONS if section in occupied]


def n_sections() -> int:
    """Return the number of non-empty sections shown in the sidebar."""
    return len(section_names())


#: Icons are re-inked per theme by ``iconset`` (a PIL + numpy pass per
#: PNG), which is far too slow to repeat for each of the thirty
#: variants. Cached across contexts, keyed by ``(theme, key)``.
_ICON_CACHE: Dict[Tuple[str, str], object] = {}
_PIXMAP_CACHE: Dict[Tuple[str, str, int], object] = {}
_LOGO_CACHE: Dict[Tuple[str, int], object] = {}

#: True only when :func:`bootstrap` created the QApplication itself. When we
#: are a guest inside someone else's (pytest-qt), application-wide restyling
#: is off limits -- see :meth:`Ctx.apply_theme`.
_WE_OWN_THE_APP = False


class Ctx:
    """Per-theme rendering context: palette, stylesheet, icon cache."""

    def __init__(self, app, theme: str):
        from spacr.qt.theme import palette_for
        self.app = app
        self.theme = theme
        self.P = palette_for(theme)

    def qss(self) -> str:
        """This theme's stylesheet.

        background=None: the Space theme degrades to its gradient sky rather
        than depending on a cached generated image.
        """
        from spacr.qt.theme import stylesheet
        return stylesheet(self.theme, 1.0, background=None)

    def apply_theme(self, target=None) -> None:
        """Apply this theme, to ``target`` if given, else the application.

        QApplication.setStyleSheet re-polishes EVERY top-level widget. Inside
        pytest-qt the application is shared, so widgets belonging to other
        tests -- including ones mid-teardown whose C++ side is already gone --
        get re-polished, and the process SEGFAULTS. That is what took the
        tests/qt suite down here, and it is the same trap the theme work hit
        with QApplication.topLevelWidgets().

        A stylesheet set on a widget cascades to its children, so styling the
        root being rendered is equivalent for our purposes and touches nothing
        else. The application-wide path is used only when bootstrap() created
        the application, i.e. in the standalone generator.
        """
        from spacr.qt.theme import apply_qpalette
        if not _WE_OWN_THE_APP:
            # Guest inside someone else's QApplication: never touch it.
            if target is not None:
                target.setStyleSheet(self.qss())
            return
        apply_qpalette(self.app, self.theme)
        self.app.setStyleSheet(self.qss())

    def icon(self, key: str):
        """A themed :class:`QIcon` for an app key (same rules as the app)."""
        cache_key = (self.theme, key)
        if cache_key not in _ICON_CACHE:
            from spacr.qt import iconset
            _apps, overrides, force_glyph = _registry()
            if key in force_glyph:
                ic = iconset.icon(key, theme=self.theme)
            else:
                ic = iconset.app_icon(key, override=overrides.get(key),
                                      theme=self.theme)
            _ICON_CACHE[cache_key] = ic
        return _ICON_CACHE[cache_key]

    def pixmap(self, key: str, px: int):
        """The app icon rendered to a ``px`` square pixmap."""
        from PySide6.QtCore import QSize
        cache_key = (self.theme, key, px)
        if cache_key not in _PIXMAP_CACHE:
            _PIXMAP_CACHE[cache_key] = self.icon(key).pixmap(QSize(px, px))
        return _PIXMAP_CACHE[cache_key]

    def logo(self, px: int):
        """The spaCR wordmark logo, re-inked for this theme."""
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QPixmap

        from spacr.qt.iconset import themed_pixmap
        cache_key = (self.theme, px)
        if cache_key in _LOGO_CACHE:
            return _LOGO_CACHE[cache_key]
        path = os.path.join(repo_root(), "spacr", "resources", "icons",
                            "logo_spacr.png")
        if not os.path.isfile(path):
            return None
        pix = themed_pixmap(path, self.theme) or QPixmap(path)
        if pix.isNull():
            return None
        scaled = pix.scaled(px, px, Qt.KeepAspectRatio,
                            Qt.SmoothTransformation)
        _LOGO_CACHE[cache_key] = scaled
        return scaled


# ---------------------------------------------------------------------------
# Mock content for the elements that do not exist yet
# ---------------------------------------------------------------------------
# Fixed literals, never live state — see the module docstring. Anything
# drawn from these is *proposed* UI, not something spaCR reports today.
MOCK = {
    "project":   "toxo_mito_screen",
    "plates":    "12 plates",
    "images":    "48 320 images",
    "objects":   "1.42 M objects",
    "version":   "1.3.6",
    "last_run":  ("measure", "plate_07", "finished 18 min ago"),
    "recent": [
        ("mask",     "plate_07",  "18 min ago",  True,  "22 m 04 s"),
        ("measure",  "plate_07",  "1 h ago",     True,  "41 m 11 s"),
        ("classify_merged", "plate_06", "yesterday", False, "3 m 27 s"),
        ("annotate", "plate_06",  "yesterday",   True,  "—"),
    ],
    "system": [("GPU", 41, "RTX 4090"), ("VRAM", 62, "14.9 / 24 GB"),
               ("Disk", 68, "1.2 TB free"), ("RAM", 35, "22 / 64 GB")],
    "whats_new": [
        "Mask now runs on the Cellpose 4 (SAM) backend.",
        "Invasion Assay: two-colour outside/inside scoring.",
        "Make Masks can evaluate a segmentation model on selected fields.",
        "Report writes a shareable HTML/PDF with the QC verdict.",
    ],
    "queue": [("plate_08", "Mask → Measure", "queued"),
              ("plate_09", "Mask → Measure", "queued"),
              ("plate_10", "Measure", "queued")],
}

#: Six apps a returning user is assumed to have pinned.
PINNED = ["mask", "measure", "annotate", "classify_merged", "plate_view",
          "report"]

#: Invented but plausible run counts, used by the frequency-ordered
#: variants. Labelled as such wherever they are drawn.
#:
#: Must name every key in the registry: :func:`by_frequency` sorts on
#: ``USE_COUNTS.get(k, 0)``, so an app missing here silently sinks to
#: the bottom of every frequency-ordered variant instead of failing.
USE_COUNTS = {
    "mask": 412, "measure": 388, "annotate": 250, "classify_merged": 164,
    "plate_view": 131, "db_browser": 118, "report": 88,
    "queue": 71, "batch": 64, "map_barcodes": 59, "regression": 52,
    "convert": 47, "umap": 41, "make_masks": 38, "run_history": 35,
    "graph_builder": 31,
    "align": 24, "layer_viewer": 23, "distributed_jobs": 22, "foreign": 21,
    "external_masks": 20,
    "train_compare": 15, "recruitment": 9,
    "analyze_plaques": 8, "invasion": 6, "replication": 6,
}

#: What an app nobody has invented a count for is given. Below the
#: smallest real entry, so a newcomer sorts to the bottom of every
#: frequency-ordered variant, which is where a brand-new app belongs.
UNUSED_APP_COUNT = 4

for _key in all_keys():
    # Variant 14 reads ``USE_COUNTS[k]`` for the badge on every tile, so a
    # key missing here is not "sorts to the bottom", it is a KeyError that
    # takes all thirty variants down. That is a hand-edit a module which
    # registers itself from its own file cannot make, so the table fills
    # itself and the literals above stay a statement about the apps
    # somebody actually had an opinion on.
    USE_COUNTS.setdefault(_key, UNUSED_APP_COUNT)
del _key


# ---------------------------------------------------------------------------
# Categorisations — every one covers every real app key exactly once
# ---------------------------------------------------------------------------

def cats_current() -> "List[Tuple[str, List[str]]]":
    """The current non-empty sections, straight out of ``spacr.qt.app``."""
    sections = section_names()
    grouped = {s: [] for s in sections}
    for key, _n, _d, section in apps():
        grouped.setdefault(section, []).append(key)
    return [(s, grouped[s]) for s in sections]


def _with_late_registrations(
    cats: "Sequence[Tuple[str, Sequence[str]]]",
    fallback: str,
) -> "List[Tuple[str, List[str]]]":
    """``cats`` with every uncategorised registry key added to ``fallback``.

    Each table below is a hand-made judgement about where an app belongs
    and stays one: an app named in it lands where it was put. What this
    adds is that an app NOBODY has filed — one that registered itself
    from its own module after these literals were written — lands
    somewhere real instead of making :func:`check_coverage` raise and
    taking all thirty variants down with it.

    The fallback band is chosen per table as the one whose question a
    brand-new app is most likely to answer, and landing there is a
    prompt to file it properly, not an answer. Note that this does NOT
    relax the width rules: a band that overflows its grid still fails
    ``test_no_stage_band_exceeds_the_seven_column_grid_by_more_than_a_row``,
    which is the point — the layout decision has to be made by a person.

    :param cats: the literal categorisation, ``(title, keys)`` per band.
    :param fallback: the title of the band unfiled keys are appended to.
    :returns: a fresh list; the literal is not mutated.
    """
    live = set(all_keys())
    placed_literal = {key for _title, keys in cats for key in keys}
    retired = sorted(placed_literal - live)
    if retired:
        raise AssertionError(
            "categorisation still presents retired app keys: "
            f"{retired}. Remove each folded module from the Home table and "
            "route it through its host screen instead.")

    # New registrations still enter the declared fallback so the review
    # surface remains buildable. Retired rows are deliberately different:
    # retaining one would present a standalone Home tile that no longer
    # exists, so the explicit failure above makes that drift visible.
    result = [(title, list(keys)) for title, keys in cats]
    placed = {key for _title, keys in result for key in keys}
    missing = [key for key in all_keys() if key not in placed]
    if missing:
        for title, keys in result:
            if title == fallback:
                keys.extend(missing)
                break
    return result


CATS_BROAD3 = _with_late_registrations([
    # Power / Design is the only app in the registry that runs BEFORE the
    # images exist. "Prepare" is the closest of these three to that, and
    # it is where a screener would look for it.
    ("Prepare", ["power", "experiment_design", "convert", "align",
                 "foreign", "external_masks", "project_browser",
                 "make_masks"]),
    ("Run", ["mask", "measure", "annotate",
             "classify_merged",
             "map_barcodes", "regression", "queue", "batch", "distributed_jobs", "analyze_plaques",
             "recruitment", "invasion", "replication"]),
    ("Review", ["plate_view", "umap", "layer_viewer", "napari_bridge",
                "graph_builder", "tabulate", "trellis", "gate_editor",
                "feature_explorer", "outliers", "dose_response",
                "control_chart", "run_compare",
                "train_compare", "run_history", "db_browser", "data_manager", "report",
                "pipeline_graph", "hit_list", "profiler",
                "methods_export", "investigate_hit", "qc_dashboard",
                "lineage", "feature_dict"]),
], fallback="Review")

#: Five alternative workflow stages used by variants 02, 15, 23 and 30.
#: Seven tiles fit per row; the focused test pins the current widest band so
#: registry growth or consolidation cannot silently add or leave an empty row.
CATS_STAGE5 = _with_late_registrations([
    # Design precedes acquisition; project conversion, dispatch and storage
    # management all prepare inputs rather than interpret results.
    ("Acquire", ["power", "experiment_design", "convert", "align", "foreign",
                 "external_masks", "queue", "batch",
                 "distributed_jobs", "data_manager", "project_browser"]),
    # Mask creation, manual correction and registered layer inspection are
    # one segmentation stage; folded model tools are reached through Mask.
    ("Segment", ["mask", "make_masks", "layer_viewer", "napari_bridge"]),
    # These applications quantify, label or summarize measured objects.
    ("Measure", ["measure", "annotate", "lineage", "analyze_plaques",
                 "recruitment", "invasion", "replication", "tabulate",
                 "feature_dict"]),
    # Classification, barcode mapping, regression and exploratory model
    # interrogation produce analytical results.
    ("Analyse", ["classify_merged", "map_barcodes", "regression", "umap",
                 "graph_builder", "profiler", "investigate_hit", "trellis",
                 "gate_editor", "feature_explorer", "dose_response"]),
    # Provenance, QC, comparisons and export determine whether a result can
    # be reported and preserve the evidence used to reach that decision.
    ("Report", ["plate_view", "train_compare", "run_history", "run_compare",
                 "db_browser", "report",
                 "pipeline_graph", "hit_list", "methods_export",
                 "qc_dashboard", "outliers", "control_chart"]),
], fallback="Report")

CATS_NARROW8 = _with_late_registrations([
    # The bands stay deliberately narrow. Folded capabilities remain on
    # their host screens and therefore do not receive standalone entries.
    ("Segment",          ["mask", "make_masks", "napari_bridge"]),
    ("Measure",          ["measure", "tabulate", "feature_dict"]),
    ("Label",            ["annotate"]),
    ("Classify",         ["classify_merged", "train_compare"]),
    # The Prediction Profiler goes here rather than under "Classify":
    # what it sweeps is a screen's regression, which is this band's
    # subject, while Classify contains the classifier and training review.
    ("Screens & reports", ["map_barcodes", "regression",
                           "umap", "graph_builder", "layer_viewer",
                           "plate_view", "report",
                           "hit_list", "methods_export", "pipeline_graph",
                           "profiler", "investigate_hit", "qc_dashboard",
                           "lineage", "trellis", "gate_editor",
                           "feature_explorer", "outliers", "control_chart"]),
    ("Import & batch",   ["convert", "align", "foreign", "external_masks",
                          "queue", "batch",
                          "distributed_jobs", "run_history", "run_compare",
                          "db_browser", "data_manager", "project_browser"]),
    ("Toxoplasma",       ["analyze_plaques", "recruitment", "invasion",
                          "replication"]),
    ("Design",            ["power", "experiment_design", "dose_response"]),
], fallback="Screens & reports")

CATS_QUESTIONS = _with_late_registrations([
    # Power / Design answers the question BEFORE the first one here — "do
    # I have enough images?" — and the honest place for it is the band
    # about getting images, since that is the decision it feeds.
    ("I have images. Where are my objects?",
     ["mask", "make_masks", "align", "convert", "foreign",
      "external_masks", "power", "experiment_design", "project_browser",
      "napari_bridge"]),
    ("I have objects. What are they like?",
     ["measure", "annotate", "analyze_plaques", "recruitment",
      "invasion", "replication", "layer_viewer", "tabulate", "lineage",
      "feature_dict"]),
    # Hit List answers this band's question in the most direct way there
    # is — it IS the list of genes that matter — and the Prediction
    # Profiler is how you interrogate the model that produced it.
    ("I have a screen. Which genes matter?",
     ["classify_merged", "map_barcodes",
      "regression", "umap", "graph_builder", "hit_list", "profiler",
      "investigate_hit", "trellis", "gate_editor", "feature_explorer",
      "outliers", "dose_response"]),
    # Pipeline Graph belongs here for the literal reason: it marks the
    # outputs that no longer follow from their inputs, which is the
    # question in the heading. Methods & Results is the other half — what
    # you write down once you have decided you do believe it.
    ("Should I believe any of this?",
     ["plate_view", "train_compare", "report", "run_history", "run_compare", "db_browser", "data_manager",
      "queue", "batch", "distributed_jobs", "pipeline_graph",
      "methods_export", "qc_dashboard", "control_chart"]),
], fallback="Should I believe any of this?")

CATS_INTENT4 = [
    ("Segment images", CATS_QUESTIONS[0][1]),
    ("Measure objects", CATS_QUESTIONS[1][1]),
    ("Analyse a screen", CATS_QUESTIONS[2][1]),
    ("Check & share", CATS_QUESTIONS[3][1]),
]


def by_frequency() -> List[str]:
    """All keys, most-used first (see :data:`USE_COUNTS`)."""
    return sorted(all_keys(), key=lambda k: (-USE_COUNTS.get(k, 0), k))


def alphabetical() -> List[str]:
    """All keys sorted by display name."""
    return sorted(all_keys(), key=lambda k: name_of(k).lower())


def pinned_first() -> List[str]:
    """Pinned keys first, then everything else by frequency."""
    rest = [k for k in by_frequency() if k not in PINNED]
    return list(PINNED) + rest


def check_coverage(cats: Sequence[Tuple[str, Sequence[str]]]) -> None:
    """Raise if a categorisation drops, duplicates or invents a key."""
    seen: List[str] = []
    for _title, keys in cats:
        seen.extend(keys)
    known = set(all_keys())
    dupes = {k for k in seen if seen.count(k) > 1}
    if dupes:
        raise AssertionError(f"duplicate keys in categorisation: {sorted(dupes)}")
    unknown = set(seen) - known
    if unknown:
        raise AssertionError(f"unknown keys: {sorted(unknown)}")
    missing = known - set(seen)
    if missing:
        raise AssertionError(f"keys not categorised: {sorted(missing)}")
