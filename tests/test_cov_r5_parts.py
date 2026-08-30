"""What the Home-variant widget kit builds when a part is not supplied.

The kit under ``spacr/resources/home/versions/_generators/parts.py`` builds
the thirty review renders, and most of its pieces are optional: the logo, a
category rule, a heading's trailing note, a rail's own caption, a rail's item
counts.  Every one of those has a "not supplied" shape, and each of the tests
here builds the same widget twice -- once with the part and once without --
because a builder that quietly dropped the part on BOTH runs would look the
same as one that handled the omission correctly.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types

import pytest

pytest.importorskip("PySide6")

REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
GENERATORS = os.path.join(REPO_ROOT, "spacr", "resources", "home", "versions",
                          "_generators")


def _load(name: str, module_name: str):
    """Import one generator module under an explicit module name."""
    path = os.path.join(GENERATORS, f"{name}.py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def kit(qapp):
    """``common`` and ``parts``, loaded under the plain names they expect.

    Same arrangement as ``tests/test_cov_2_parts.py``: ``parts`` imports
    ``common`` by plain name, so both occupy those entries in
    :data:`sys.modules` while they load, and the originals go back afterwards.
    """
    if not os.path.isdir(GENERATORS):
        pytest.skip("home-screen variant generators not present")
    names = ("common", "parts")
    saved = {name: sys.modules.get(name) for name in names}
    try:
        common = _load("common", "common")
        common.bootstrap()
        parts = _load("parts", "parts")
        yield types.SimpleNamespace(common=common, parts=parts, app=qapp)
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


@pytest.fixture
def ctx(kit):
    """A dark rendering context, unthemed.

    ``apply_theme()`` is not called: these are structural assertions about
    which child widgets exist, which the stylesheet does not decide.
    """
    return kit.common.Ctx(kit.app, "dark")


@pytest.fixture
def strip_the_icons(kit, monkeypatch, tmp_path):
    """Call this to move the checkout to a tree with no icon resources.

    ``Ctx.logo`` returns None when ``logo_spacr.png`` is missing -- what an
    install stripped of its resources looks like.  The pixmap cache is
    emptied at the same time, or a size an earlier call already rendered
    would be served from it and the missing file never consulted.

    A callable rather than a plain fixture so one test can build the same
    widget with the logo and then without it.
    """
    def _strip():
        monkeypatch.setattr(kit.common, "repo_root", lambda: str(tmp_path))
        monkeypatch.setattr(kit.common, "_LOGO_CACHE", {})

    return _strip


def _pixmap_labels(widget):
    """The QLabels in ``widget`` that are actually drawing a pixmap."""
    from PySide6.QtWidgets import QLabel

    return [lbl for lbl in widget.findChildren(QLabel)
            if lbl.pixmap() is not None and not lbl.pixmap().isNull()]


def _texts(widget):
    from PySide6.QtWidgets import QLabel

    return [lbl.text() for lbl in widget.findChildren(QLabel) if lbl.text()]


# ---------------------------------------------------------------------------
# The logo, which an install may not have shipped
# ---------------------------------------------------------------------------

def test_the_hero_keeps_its_wordmark_when_there_is_no_logo(ctx, kit,
                                                          strip_the_icons):
    """No icon file means no pixmap label -- and the pitch still renders.

    The hero is the first thing on the Home screen; losing the logo must cost
    the logo and nothing else.
    """
    with_logo = kit.parts.hero(ctx)
    assert len(_pixmap_labels(with_logo)) == 1, "the logo is normally drawn"

    strip_the_icons()
    without = kit.parts.hero(ctx)
    assert _pixmap_labels(without) == []
    assert "spaCR" in _texts(without), "the wordmark is a label, not the logo"
    assert len(_texts(without)) == len(_texts(with_logo)), (
        "only the pixmap label went missing")


def test_the_top_bar_keeps_its_title_when_there_is_no_logo(ctx, kit,
                                                          strip_the_icons):
    """The slim bar's mark is optional; its title and actions are not."""
    from PySide6.QtWidgets import QPushButton

    with_logo = kit.parts.top_bar(ctx, title="Analyse", subtitle="step 2",
                                  actions=(("Run", True),))
    assert len(_pixmap_labels(with_logo)) == 1

    strip_the_icons()
    without = kit.parts.top_bar(ctx, title="Analyse", subtitle="step 2",
                                actions=(("Run", True),))
    assert _pixmap_labels(without) == []
    assert "Analyse" in _texts(without)
    assert "step 2" in _texts(without)
    assert [b.text() for b in without.findChildren(QPushButton)] == ["Run"]


# ---------------------------------------------------------------------------
# Optional trimmings
# ---------------------------------------------------------------------------

def test_a_category_header_draws_its_rule_only_when_asked(ctx, kit):
    """``rule=False`` is how a header sits directly above another one.

    The hairline is a ``Divider`` child, so its presence is countable.
    """
    from spacr.qt.widgets.divider import Divider

    ruled = kit.parts.cat_header(ctx, "Segment")
    assert len(ruled.findChildren(Divider)) == 1

    bare = kit.parts.cat_header(ctx, "Segment", rule=False)
    assert bare.findChildren(Divider) == []
    assert _texts(bare) == _texts(ruled), "the caption is untouched"


def test_a_plain_header_adds_a_note_only_when_it_is_given_one(ctx, kit):
    """An empty note must not leave an empty label behind.

    A zero-width label still takes the layout's spacing, which is what put a
    stray gap after the heading in a variant that passed ``note=""``.
    """
    annotated = kit.parts.plain_header(ctx, "Datasets", "12 found")
    assert _texts(annotated) == ["Datasets", "12 found"]

    plain = kit.parts.plain_header(ctx, "Datasets")
    assert _texts(plain) == ["Datasets"]


def test_the_category_rail_captions_and_counts_are_both_optional(ctx, kit):
    """A rail with no header has no caption label, and no counts prints none.

    Both are read off the built widget: the caption is the only QLabel the
    rail owns, and the counts are appended to each item's own text.
    """
    from PySide6.QtWidgets import QListWidget

    def _items(rail):
        lst = rail.findChild(QListWidget)
        return [lst.item(i).text() for i in range(lst.count())]

    titles = ["Segment", "Measure"]

    full = kit.parts.cat_rail(ctx, titles, header="Stages", counts=[3, 7])
    assert _texts(full) == ["STAGES"], "the caption is upper-cased"
    assert _items(full) == ["Segment    3", "Measure    7"]

    bare = kit.parts.cat_rail(ctx, titles)
    assert _texts(bare) == [], "no header means no caption label"
    assert _items(bare) == titles, "no counts means the plain titles"
