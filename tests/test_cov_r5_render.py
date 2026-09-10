"""The variant renderer's optional outputs and its optional font.

``render.py`` is a script, and everything pinned here is one of its "skip
that step" paths: the layout audit walking past a button that fits, the
``--no-sheet``/``--no-md`` flags that drop the contact sheet and VARIANTS.md
from a run, and the bundled font being unavailable so the contact-sheet
labels fall back to Pillow's built-in one.

Every one of these is a pair -- the step taken and the step skipped -- in a
single test, because a run that produced neither output looks exactly like a
run that skipped both on purpose.

The output directories are redirected into ``tmp_path`` throughout: this
module's subject writes PNGs and prunes directories, and the checked-in
renders are not test fixtures.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("PIL")

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


@pytest.fixture
def sandbox(qapp, monkeypatch, tmp_path):
    """``render``, with ``common``'s output directories inside ``tmp_path``.

    ``render`` does ``import common`` by plain name, so the redirected
    ``common`` has to be in :data:`sys.modules` before ``render`` executes.
    Without the redirect this test would overwrite the checked-in renders and
    the self-check would read them.
    """
    if not os.path.isdir(GENERATORS):
        pytest.skip("home-screen variant generators not present")
    names = ("common", "parts", "variants", "render")
    saved = {name: sys.modules.get(name) for name in names}
    versions = tmp_path / "versions"
    here = versions / "_generators"
    here.mkdir(parents=True)
    try:
        common = _load("common", "common")
        monkeypatch.setattr(common, "versions_dir", lambda: str(versions))
        monkeypatch.setattr(common, "here", lambda: str(here))
        render = _load("render", "render")
        render._test_versions_dir = str(versions)
        yield render
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


# ---------------------------------------------------------------------------
# The layout audit
# ---------------------------------------------------------------------------

def test_the_audit_names_only_the_buttons_that_actually_elided(qapp, sandbox):
    """A button that fits is walked past; one that does not is reported.

    The audit exists to fail a variant whose sidebar shows ``Classific…``
    instead of the app's name.  Reporting every button would make the finding
    useless, so the two cases have to be told apart on the same page.
    """
    from PySide6.QtWidgets import QWidget

    from spacr.qt.widgets.eliding import ElidingPushButton

    page = QWidget()
    page.resize(600, 200)
    roomy = ElidingPushButton("Measure", page)
    roomy.setFixedWidth(400)
    roomy.move(0, 0)
    cramped = ElidingPushButton("Classification and regression", page)
    cramped.move(0, 60)
    page.show()
    qapp.processEvents()
    # resize(), not setFixedWidth(). _refresh refuses to elide until
    # WA_Resized is set -- a deliberate guard against the default 100 px a
    # widget carries before its first layout pass -- and setFixedWidth
    # pins min == max == 52, so the follow-up resize is a no-op that fires
    # no resizeEvent and the button never shortens. A real geometry change
    # is what the widget is written to respond to.
    cramped.resize(52, cramped.sizeHint().height())
    qapp.processEvents()

    assert roomy.is_elided() is False
    assert cramped.is_elided() is True

    found = sandbox.audit(page)
    assert found["elided"] == ["Classification and regression"], (
        "the button that fits is not a finding")

    page.hide()
    page.setParent(None)
    page.deleteLater()


# ---------------------------------------------------------------------------
# The contact-sheet font
# ---------------------------------------------------------------------------

def test_the_sheet_font_falls_back_when_the_bundled_one_is_missing(
        sandbox, monkeypatch, tmp_path):
    """A checkout without ``OpenSans-SemiBold.ttf`` still labels the sheet.

    The contact sheet is a review artefact; losing the bundled face has to
    cost the typeface and not the run.  Pointing ``repo_root`` at an empty
    tree is exactly what an install stripped of its font resources looks
    like.
    """
    import common

    # `repo_root()` is "five levels up from this file", and the fixture
    # runs the generators out of tmp_path -- so it resolves outside the
    # checkout and the bundled face is not found there. Point it at the
    # real root to see the bundled case at all.
    monkeypatch.setattr(common, "repo_root", lambda: REPO_ROOT)
    bundled = sandbox._sheet_font(15)
    assert bundled.getname()[0] == "Open Sans"
    assert bundled.size == 15

    monkeypatch.setattr(common, "repo_root", lambda: str(tmp_path))
    fallback = sandbox._sheet_font(15)
    assert fallback.getname()[0] != "Open Sans", (
        "the missing file was not silently found somewhere else")

    # And the sheet is still drawn with it.
    import variants

    out = sandbox.build_sheet(variants.VARIANTS[:2])
    assert os.path.isfile(out)


# ---------------------------------------------------------------------------
# The CLI's optional outputs
# ---------------------------------------------------------------------------

def test_a_run_can_be_told_to_skip_the_sheet_and_the_markdown(sandbox, capsys):
    """``--no-sheet``/``--no-md`` drop the two whole-set artefacts.

    Both are derived from every variant, not just the rendered ones, so a
    partial re-render is the case they exist for.  ``--only 0`` matches no
    variant, which leaves the run as the bookkeeping around them.
    """
    import variants

    total = len(variants.VARIANTS)
    versions = sandbox._test_versions_dir

    assert sandbox.main(["--only", "0", "--no-sheet", "--no-md"]) == 0
    quiet = capsys.readouterr().out
    assert f"variants: 0 of {total}" in quiet
    # Line-anchored: the self-check below prints "  contact sheet:
    # MISSING", which a bare substring test matches even though it is
    # exactly the line that proves the sheet was skipped.
    printed = [line for line in quiet.splitlines()]
    assert not any(line.startswith("sheet:") for line in printed)
    assert not any(line.startswith("markdown:") for line in printed)
    assert "self-check" in quiet, "the run still reports on what is there"
    assert not os.path.isfile(os.path.join(versions, "_sheet.png"))
    assert not os.path.isfile(os.path.join(versions, "VARIANTS.md"))

    assert sandbox.main(["--only", "0"]) == 0
    full = capsys.readouterr().out
    assert "sheet:" in full
    assert "markdown:" in full
    assert os.path.isfile(os.path.join(versions, "_sheet.png"))
    assert os.path.isfile(os.path.join(versions, "VARIANTS.md"))
