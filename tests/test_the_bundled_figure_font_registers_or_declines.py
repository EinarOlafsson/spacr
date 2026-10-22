"""Registering the bundled Open Sans faces, including every way it declines.

``use_open_sans_for_figures`` runs before every figure is styled, and its whole
design is about NOT doing expensive work: it returns early once it has run, it
asks matplotlib what fonts it already has before adding any, and it survives a
matplotlib that will not answer. Almost none of that had been exercised,
because the first call in a test session takes the fast path and every later
one returns at the ``_registered`` guard.

The comment at line 85 records why this matters: ``addfont`` cost 23 SECONDS
of a 13-second module open on the Mask screen. Every branch below is either
that cost being avoided or a failure that must not stop a figure appearing.
"""
from __future__ import annotations

import builtins
import os
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def native_manager(monkeypatch, tmp_path):
    from spacr import figure_font

    known = str(tmp_path / 'known.ttf')
    core = str(tmp_path / 'core.ttf')
    custom = str(tmp_path / 'custom.otf')
    directory = str(tmp_path / 'directory.ttf')
    afm = str(tmp_path / 'directory.afm')
    manager = SimpleNamespace(ttflist=[SimpleNamespace(fname=known)], afmlist=[],
                              __version__='test')
    manager.addfont = lambda path: (
        manager.afmlist if path.endswith('.afm') else manager.ttflist
    ).append(SimpleNamespace(fname=path))
    writes, imports, probes = [], [], []
    fonts = SimpleNamespace(
        fontManager=manager, X11FontDirectories=['x11'], OSXFontDirectories=['mac'],
        _get_fontconfig_fonts=lambda: [custom],
        findSystemFonts=lambda **kw: [afm if kw['fontext'] == 'afm' else directory],
        json_dump=lambda obj, path: writes.append((obj, path)))
    root = SimpleNamespace(__version__='3.11.2', get_cachedir=lambda: str(tmp_path))
    monkeypatch.setitem(sys.modules, 'matplotlib', root)
    monkeypatch.delitem(sys.modules, 'matplotlib.font_manager', raising=False)
    monkeypatch.setattr(sys, 'platform', 'darwin')
    monkeypatch.delenv('MPL_IGNORE_SYSTEM_FONTS', raising=False)
    original = builtins.__import__

    def importing(name, globals=None, locals=None, fromlist=(), level=0):
        if name == 'matplotlib' and 'font_manager' in (fromlist or ()):
            imports.append(os.environ.get('MPL_IGNORE_SYSTEM_FONTS'))
            if getattr(root, 'fail_import', False):
                raise ImportError('unavailable renderer')
            return SimpleNamespace(font_manager=fonts)
        return original(name, globals, locals, fromlist, level)

    def probe(*args, **kwargs):
        probes.append(args)
        return json.dumps([known, core, core])

    monkeypatch.setattr(builtins, '__import__', importing)
    monkeypatch.setattr(subprocess, 'check_output', probe)
    return SimpleNamespace(module=figure_font, fonts=fonts, root=root, manager=manager,
                           writes=writes, imports=imports, probes=probes,
                           expected={known, core, custom, directory, afm})


def test_native_discovery_keeps_coretext_directory_and_fontconfig_fonts(native_manager):
    case = native_manager
    assert case.module._figure_font_manager() is case.fonts
    paths = [entry.fname for entry in [*case.manager.ttflist, *case.manager.afmlist]]
    assert set(paths) == case.expected
    assert len(paths) == len(set(paths))
    assert case.imports == ['1']
    assert 'MPL_IGNORE_SYSTEM_FONTS' not in os.environ
    assert len(case.writes) == 1
    assert case.writes[0][1].name == 'fontlist-vtest.json'


@pytest.mark.parametrize('reason', ['missing_api', 'old_matplotlib', 'optout'])
def test_native_discovery_falls_back_without_hiding_fonts(native_manager, monkeypatch, reason):
    case = native_manager
    if reason == 'missing_api':
        def unavailable(*args, **kwargs):
            raise subprocess.TimeoutExpired('osascript', 3)
        monkeypatch.setattr(subprocess, 'check_output', unavailable)
    elif reason == 'old_matplotlib':
        case.root.__version__ = '3.10.0'
    else:
        monkeypatch.setenv('MPL_IGNORE_SYSTEM_FONTS', 'user-choice')
    assert case.module._figure_font_manager() is case.fonts
    assert not case.writes
    assert case.imports == ['user-choice' if reason == 'optout' else None]
    assert os.environ.get('MPL_IGNORE_SYSTEM_FONTS') == (
        'user-choice' if reason == 'optout' else None)


@pytest.mark.parametrize("version", [None, "3.development"])
def test_an_unreadable_matplotlib_version_keeps_the_existing_fonts(
        native_manager, version):
    case = native_manager
    before = list(case.manager.ttflist)
    case.root.__version__ = version

    assert case.module._figure_font_manager() is case.fonts
    assert case.manager.ttflist == before
    assert case.manager.afmlist == []
    assert case.probes == []
    assert case.imports == [None]
    assert case.writes == []
    assert "MPL_IGNORE_SYSTEM_FONTS" not in os.environ


@pytest.mark.parametrize("paths", [None, [], ["relative.ttf"], [17]])
def test_an_invalid_coretext_list_keeps_normal_font_discovery(
        native_manager, monkeypatch, paths):
    case = native_manager
    before = list(case.manager.ttflist)
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **kw: json.dumps(paths))

    assert case.module._figure_font_manager() is case.fonts
    assert case.manager.ttflist == before
    assert case.manager.afmlist == []
    assert case.imports == [None]
    assert case.writes == []
    assert "MPL_IGNORE_SYSTEM_FONTS" not in os.environ


@pytest.mark.parametrize("refusal", ["unsupported_suffix", "one_unreadable", "all_unreadable"])
def test_native_face_refusals_preserve_existing_and_usable_fonts(
        native_manager, monkeypatch, refusal):
    case = native_manager
    known = case.manager.ttflist[0].fname
    expected = set(case.expected)
    real_add = case.manager.addfont
    if refusal == "unsupported_suffix":
        unsupported = os.path.join(os.path.dirname(known), "not-a-font.txt")
        monkeypatch.setattr(subprocess, "check_output", lambda *a, **kw:
                            json.dumps([*case.expected, unsupported]))
    else:
        def add_if_readable(path):
            if refusal == "all_unreadable" or os.path.basename(path) == "core.ttf":
                raise OSError("unreadable native face")
            return real_add(path)

        monkeypatch.setattr(case.manager, "addfont", add_if_readable)
        expected = ({known} if refusal == "all_unreadable" else
                    {path for path in expected if os.path.basename(path) != "core.ttf"})

    assert case.module._figure_font_manager() is case.fonts
    fonts = [entry.fname for entry in [*case.manager.ttflist, *case.manager.afmlist]]
    assert set(fonts) == expected
    assert len(fonts) == len(expected)
    assert known in fonts
    assert case.imports == ["1"]
    assert "MPL_IGNORE_SYSTEM_FONTS" not in os.environ
    if refusal == "all_unreadable":
        assert case.writes == []
    else:
        assert len(case.writes) == 1
        assert case.writes[0][0] is case.manager
        assert case.writes[0][1].name == "fontlist-vtest.json"


def test_a_cache_write_failure_does_not_discard_registered_fonts(
        native_manager, monkeypatch):
    case = native_manager
    real_dump = case.fonts.json_dump

    def unwritable(manager, path):
        real_dump(manager, path)
        raise OSError("font cache is read-only")

    monkeypatch.setattr(case.fonts, "json_dump", unwritable)
    assert case.module._figure_font_manager() is case.fonts
    fonts = [entry.fname for entry in [*case.manager.ttflist, *case.manager.afmlist]]
    assert set(fonts) == case.expected
    assert len(fonts) == len(case.expected)
    assert len(case.writes) == 1
    assert case.writes[0][0] is case.manager
    assert case.writes[0][1].name == "fontlist-vtest.json"
    assert case.imports == ["1"]
    assert "MPL_IGNORE_SYSTEM_FONTS" not in os.environ


def test_a_failed_native_font_import_restores_the_discovery_setting(native_manager):
    native_manager.root.fail_import = True
    with pytest.raises(ImportError, match='unavailable renderer'):
        native_manager.module._figure_font_manager()
    assert 'MPL_IGNORE_SYSTEM_FONTS' not in os.environ


@pytest.fixture
def cold(monkeypatch):
    """A module that has not registered anything yet, restored afterwards."""
    from spacr import figure_font

    monkeypatch.setattr(figure_font, "_registered", False, raising=False)
    monkeypatch.setattr(figure_font, "_resolved", False, raising=False)
    return figure_font


# ---------------------------------------------------------------------------
# bundled_faces — line 53, the directory is not there
# ---------------------------------------------------------------------------

def test_a_missing_font_directory_yields_no_faces(monkeypatch):
    """The ``if not os.path.isdir(directory):`` branch.

    Documented behaviour, and the reason it is documented: a figure drawn in
    the wrong font is a blemish and never a reason for a plot not to appear.
    An installation that dropped the resources folder still plots.
    """
    from spacr import figure_font

    monkeypatch.setattr(figure_font, "font_dir",
                        lambda: "/nonexistent/font/dir")
    assert figure_font.bundled_faces() == []


def test_the_shipped_directory_holds_ttf_faces():
    """The other side: the package really does carry the faces."""
    from spacr import figure_font

    faces = figure_font.bundled_faces()
    assert faces
    assert all(f.lower().endswith((".ttf", ".otf")) for f in faces)


def test_a_readable_font_directory_is_filtered_and_sorted(tmp_path,
                                                           monkeypatch):
    """Readable directories return only supported faces in filename order."""
    from spacr import figure_font

    for name in ("zeta.ttf", "ignore.txt", "Alpha.OTF"):
        (tmp_path / name).write_bytes(b"font fixture")
    monkeypatch.setattr(figure_font, "font_dir", lambda: str(tmp_path))

    assert [os.path.basename(path) for path in figure_font.bundled_faces()] == [
        "Alpha.OTF",
        "zeta.ttf",
    ]


def test_an_unreadable_font_directory_yields_no_faces(monkeypatch):
    """A directory race or permissions error must not prevent a plot."""
    from spacr import figure_font

    monkeypatch.setattr(figure_font.os.path, "isdir", lambda _path: True)

    def unreadable(_path):
        raise OSError("permission denied")

    monkeypatch.setattr(figure_font.os, "listdir", unreadable)
    assert figure_font.bundled_faces() == []


# ---------------------------------------------------------------------------
# use_open_sans_for_figures — the early return, and every refusal
# ---------------------------------------------------------------------------

def test_a_second_call_returns_the_first_answer_without_asking_again(monkeypatch):
    """The ``if _registered:`` guard, which is the whole performance argument.

    The check it skips is a set comprehension over every font matplotlib
    knows, and this runs before every figure is styled.
    """
    from spacr import figure_font

    monkeypatch.setattr(figure_font, "_registered", True, raising=False)
    monkeypatch.setattr(figure_font, "_resolved", True, raising=False)

    def explode(*_a, **_k):                       # must not be reached
        raise AssertionError("the registered guard did not short-circuit")

    monkeypatch.setattr(figure_font, "bundled_faces", explode)
    assert figure_font.use_open_sans_for_figures() is True


def test_a_matplotlib_that_cannot_be_imported_is_declined(cold, monkeypatch):
    """Lines 77-78: no font_manager, so no font, and no exception either."""
    real_import = builtins.__import__

    def refusing(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "matplotlib" and "font_manager" in (fromlist or ()):
            raise ImportError("matplotlib is not importable here")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", refusing)
    assert cold.use_open_sans_for_figures() is False


def _font_manager_where(monkeypatch, cold, *, ttflist, addfont=None):
    """Install a stand-in ``font_manager`` for the duration of one call."""
    import types

    class _FontManager:
        def __init__(self):
            self.added = []

        @property
        def ttflist(self):
            value = ttflist() if callable(ttflist) else ttflist
            if isinstance(value, Exception):
                raise value
            return value

        def addfont(self, path):
            self.added.append(path)
            if addfont is not None:
                addfont(path)

    module = types.ModuleType("matplotlib.font_manager")
    module.fontManager = _FontManager()
    monkeypatch.setattr(cold, "_figure_font_manager", lambda: module)
    return module.fontManager


class _Face:
    def __init__(self, name):
        self.name = name


def test_an_unreadable_font_directory_is_declined_by_registration(
        cold, monkeypatch):
    """The user-facing registration path must contain enumeration failure."""
    manager = _font_manager_where(
        monkeypatch, cold, ttflist=[_Face("DejaVu Sans")])
    monkeypatch.setattr(cold.os.path, "isdir", lambda _path: True)

    def unreadable(_path):
        raise OSError("stale font directory")

    monkeypatch.setattr(cold.os, "listdir", unreadable)

    assert cold.use_open_sans_for_figures() is False
    assert manager.added == []


def test_a_font_manager_that_will_not_list_its_fonts_still_answers(cold,
                                                                   monkeypatch):
    """Lines 82-83 and 99-100: an unreadable ttflist becomes an empty set.

    Both reads of ``ttflist`` are guarded, and both must be: the second one
    runs after faces have been added, so a failure there would otherwise throw
    away the work rather than merely failing to confirm it.
    """
    boom = RuntimeError("this font manager will not enumerate")
    manager = _font_manager_where(monkeypatch, cold, ttflist=lambda: boom)

    assert cold.use_open_sans_for_figures() is False
    # It still TRIED to add the bundled faces -- the empty set means "I do not
    # know what is installed", not "nothing is installed".
    assert manager.added


def test_an_already_installed_open_sans_costs_no_addfont(cold, monkeypatch):
    """The ``if FAMILY not in available:`` branch NOT taken.

    This is the 23-second saving the comment records. A machine with Open Sans
    installed must add nothing at all.
    """
    manager = _font_manager_where(
        monkeypatch, cold, ttflist=[_Face(cold.FAMILY)])

    assert cold.use_open_sans_for_figures() is True
    assert manager.added == []


def test_one_unreadable_face_does_not_cost_the_others(cold, monkeypatch):
    """Lines 94-96: ``continue`` past a face that will not load.

    Eight faces ship. One corrupt file must cost that one face, not the whole
    family -- which is the difference between a figure in the wrong weight and
    a figure in the wrong font.
    """
    state = {"seen": 0}
    faces_before = [_Face("DejaVu Sans")]

    def addfont(path):
        state["seen"] += 1
        if state["seen"] == 1:
            raise OSError("this face is corrupt")

    manager = _font_manager_where(
        monkeypatch, cold,
        ttflist=lambda: faces_before if state["seen"] == 0
        else [_Face("DejaVu Sans"), _Face(cold.FAMILY)],
        addfont=addfont)

    assert cold.use_open_sans_for_figures() is True
    assert len(manager.added) == len(cold.bundled_faces())


def test_faces_that_add_without_resolving_are_reported_as_unresolved(cold,
                                                                     monkeypatch):
    """Lines 103-104: added everything and the family still is not there.

    The honest answer is False. Returning True would tell the style helpers to
    ask for a family matplotlib cannot resolve, and matplotlib answers that
    with its default font and a warning per text object.
    """
    manager = _font_manager_where(
        monkeypatch, cold, ttflist=[_Face("DejaVu Sans")])

    assert cold.use_open_sans_for_figures() is False
    assert manager.added                          # it did try
