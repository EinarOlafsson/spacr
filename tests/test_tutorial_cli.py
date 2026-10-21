"""``spacr-tutorial`` lists the lessons and opens the online library.

The maintainer's decision, 2026-10-12: the command opens the tutorials rather
than rendering videos. With no argument it prints the lesson list (number,
title, screen) from the bundled lesson catalog and opens the library at the
URL the Help menu opens; with a lesson number, slug or screen name it opens
that lesson; an unknown name gets suggestions; ``--list`` prints only and
``--print-url`` prints the address instead of opening a browser.

``webbrowser.open`` is replaced in every test: a test that opened a real
browser would be a test nobody can run twice.
"""
from __future__ import annotations

import json
import re
import runpy
import sys
from pathlib import Path

import pytest

from spacr.qt import tutorials
from spacr.qt.tutorial import __main__ as cli

REPO = Path(__file__).resolve().parents[1]
LIBRARY = "https://einarolafsson.github.io/spacr/tutorials/"


@pytest.fixture(autouse=True)
def _english_and_a_fresh_catalog(monkeypatch):
    """English output, and no lesson list cached by an earlier test."""
    monkeypatch.setenv("SPACR_LANGUAGE", "en")
    monkeypatch.setattr(tutorials, "_LESSONS", None)


@pytest.fixture
def opened(monkeypatch):
    """Every URL handed to the browser; the browser reports success."""
    urls = []

    def _open(url, *_args, **_kwargs):
        urls.append(url)
        return True

    monkeypatch.setattr(cli.webbrowser, "open", _open)
    return urls


def _bundled_catalog(monkeypatch, payload):
    """Serve ``payload`` as the bundled ``tutorial_index.json``."""
    class _Raw:
        def read_text(self, encoding=None):
            return json.dumps(payload)

    class _Anchor:
        def __truediv__(self, _name):
            return _Raw()

    monkeypatch.setattr("importlib.resources.files", lambda _pkg: _Anchor())


def test_the_url_is_the_one_the_help_menu_opens():
    """Read from app.py's source, so the check costs no Qt import."""
    source = (REPO / "spacr" / "qt" / "app.py").read_text(encoding="utf-8")
    base = re.search(r'^DOCS_BASE_URL = "([^"]+)"', source, re.M).group(1)
    assert re.search(r'^TUTORIALS_URL = f"\{DOCS_BASE_URL\}/tutorials/"',
                     source, re.M)
    assert tutorials.TUTORIALS_URL == f"{base}/tutorials/" == LIBRARY


def test_the_lessons_are_the_published_catalog_in_number_order():
    """Every lesson of the docs' catalog, numbered as the library numbers
    them, with the screen each covers."""
    catalog = (REPO / "docs" / "source" / "_extra" / "tutorials"
               / "lesson_catalog.js").read_text(encoding="utf-8")
    published = json.loads(catalog[catalog.index("(") + 1:
                                   catalog.rindex(")")])["lessons"]
    rows = tutorials.lessons()

    assert [row["lesson"] for row in rows] == [
        lesson["id"] for lesson in sorted(published,
                                          key=lambda item: item["number"])]
    assert [row["number"] for row in rows] == sorted(
        row["number"] for row in rows)
    mask = next(row for row in rows if row["lesson"] == "07_mask")
    assert mask == {"number": 7, "lesson": "07_mask", "slug": "mask",
                    "title": "Mask: segment images and inspect the results",
                    "app_key": "mask"}


def test_the_lesson_list_is_read_once_and_handed_out_as_a_copy(monkeypatch):
    first = tutorials.lessons()
    first.clear()
    _bundled_catalog(monkeypatch, {"catalog": []})

    assert tutorials.lessons(), "a second read replaced the cached list"


def test_a_malformed_row_is_dropped_and_a_missing_number_sorts_first(
        monkeypatch):
    _bundled_catalog(monkeypatch, {"catalog": [
        {"number": 3, "lesson": "03_three", "title": "Three",
         "app_key": "three", "slug": "three"},
        "not a lesson",
        {"number": 4, "title": "No id"},
        {"lesson": "00_intro"},
    ]})

    assert tutorials.lessons() == [
        {"number": 0, "lesson": "00_intro", "slug": "", "title": "",
         "app_key": ""},
        {"number": 3, "lesson": "03_three", "slug": "three",
         "title": "Three", "app_key": "three"},
    ]


@pytest.mark.parametrize("payload", [{"lessons": {}}, "not json at all"])
def test_a_table_without_a_catalog_lists_nothing(monkeypatch, payload):
    """An older table, and an unreadable one, both cost the list only."""
    if isinstance(payload, str):
        def explode(*_args, **_kwargs):
            raise OSError("no such file: tutorial_index.json")

        monkeypatch.setattr("importlib.resources.files", explode)
    else:
        _bundled_catalog(monkeypatch, payload)

    assert tutorials.lessons() == []
    assert tutorials.find_lesson("mask") == (None, [])


@pytest.mark.parametrize("name, lesson", [
    ("7", "07_mask"),
    ("07", "07_mask"),
    ("07_mask", "07_mask"),
    ("mask", "07_mask"),
    ("Make-Masks", "14_make_masks"),
    ("plaque assay", "24_plaque"),
    ("classify", "10_classify_cv"),
    ("classify_merged", "41_classify"),
])
def test_a_lesson_is_found_by_number_id_screen_or_slug(name, lesson):
    found, suggestions = tutorials.find_lesson(name)

    assert found["lesson"] == lesson
    assert suggestions == []


def test_an_unknown_name_gets_close_and_containing_names():
    assert tutorials.find_lesson("mesure") == (None, ["measure"])
    found, suggestions = tutorials.find_lesson("gate")
    assert found is None and "gate_editor" in suggestions
    assert len(tutorials.find_lesson("mask_")[1]) <= 5
    assert tutorials.find_lesson("zzzz") == (None, [])
    assert tutorials.find_lesson("999") == (None, [])
    assert tutorials.find_lesson("") == (None, [])


def test_the_lesson_url_carries_the_fragment_the_library_reads():
    assert tutorials.lesson_url("07_mask") == f"{LIBRARY}#lesson=07_mask"
    assert tutorials.tutorial_url("mask") == f"{LIBRARY}#lesson=07_mask"


def test_no_argument_lists_every_lesson_and_opens_the_library(opened, capsys):
    assert cli.main([]) == 0

    out = capsys.readouterr().out
    lines = out.splitlines()
    assert lines[0].split() == ["No.", "Lesson", "Screen"]
    for row in tutorials.lessons():
        assert any(line.split()[:1] == [str(row["number"])]
                   and row["title"] in line
                   and line.rstrip().endswith(row["app_key"] or "-")
                   for line in lines), row
    assert "spacr-tutorial mask" in out
    assert f"Opened {LIBRARY}" in out
    assert opened == [LIBRARY]


@pytest.mark.parametrize("name", ["8", "measure", "08_measure"])
def test_a_lesson_by_number_slug_or_screen_opens_that_lesson(
        opened, capsys, name):
    assert cli.main([name]) == 0

    out = capsys.readouterr().out
    assert "Measure: extract object features and crops" in out
    assert "Mask: segment" not in out
    assert "spacr-tutorial mask" not in out
    assert opened == [f"{LIBRARY}#lesson=08_measure"]


def test_a_lesson_by_slug_opens_it(opened):
    assert cli.main(["plaque-assay"]) == 0
    assert opened == [f"{LIBRARY}#lesson=24_plaque"]


def test_an_unknown_name_is_answered_with_suggestions(opened, capsys):
    assert cli.main(["mesure"]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "No lesson matches 'mesure'." in captured.err
    assert "Did you mean: measure?" in captured.err
    assert "spacr-tutorial --list" in captured.err
    assert opened == []


def test_an_unknown_name_with_nothing_close_points_at_the_list(opened,
                                                               capsys):
    assert cli.main(["zzzz"]) == 2

    err = capsys.readouterr().err
    assert "Did you mean" not in err
    assert "spacr-tutorial --list" in err
    assert opened == []


def test_list_prints_the_lessons_and_opens_nothing(opened, capsys):
    assert cli.main(["--list"]) == 0

    out = capsys.readouterr().out
    assert len(out.splitlines()) == len(tutorials.lessons()) + 1
    assert "Opened" not in out
    assert opened == []


def test_list_with_a_lesson_prints_that_lesson_only(opened, capsys):
    assert cli.main(["mask", "--list"]) == 0

    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2
    assert lines[1].split()[0] == "7" and lines[1].rstrip().endswith("mask")
    assert opened == []


@pytest.mark.parametrize("argv, url", [
    (["--print-url"], LIBRARY),
    (["mask", "--print-url"], f"{LIBRARY}#lesson=07_mask"),
])
def test_print_url_prints_the_address_instead_of_opening_it(
        opened, capsys, argv, url):
    assert cli.main(argv) == 0

    assert capsys.readouterr().out == url + "\n"
    assert opened == []


def test_a_machine_without_a_browser_is_given_the_address(monkeypatch,
                                                          capsys):
    monkeypatch.setattr(cli.webbrowser, "open", lambda *_a, **_k: False)

    assert cli.main(["mask"]) == 1

    captured = capsys.readouterr()
    assert f"Open this address yourself: {LIBRARY}#lesson=07_mask" in (
        captured.err)
    assert "Opened" not in captured.out


def test_an_installation_without_the_list_still_opens_the_library(
        monkeypatch, opened, capsys):
    monkeypatch.setattr(cli, "lessons", lambda: [])

    assert cli.main([]) == 0

    assert "The lesson list is not available" in capsys.readouterr().out
    assert opened == [LIBRARY]


def test_the_old_renderer_options_are_gone(capsys):
    with pytest.raises(SystemExit) as stopped:
        cli.main(["mask", "--out", "/tmp/x"])

    assert stopped.value.code == 2
    assert "unrecognized arguments: --out" in capsys.readouterr().err


def test_running_the_module_exits_with_mains_code(monkeypatch, opened):
    """``python -m spacr.qt.tutorial`` propagates main()'s exit code."""
    monkeypatch.setattr(sys, "argv", ["spacr-tutorial", "no-such-lesson"])
    monkeypatch.setattr("webbrowser.open", opened.append)

    with pytest.raises(SystemExit) as stopped:
        runpy.run_module("spacr.qt.tutorial.__main__", run_name="__main__")

    assert stopped.value.code == 2
    assert opened == []


CONTRACT = (REPO / "tests" / "data" / "release_contracts"
            / "tutorial_cli_api_changes_2026-10-12.json")


def _resolve(symbol: str):
    """The object ``symbol`` names, or None when nothing answers to it."""
    import importlib

    parts = symbol.split(".")
    for cut in range(len(parts), 0, -1):
        try:
            target = importlib.import_module(".".join(parts[:cut]))
        except ImportError:
            continue
        for name in parts[cut:]:
            target = getattr(target, name, None)
            if target is None:
                return None
        return target
    return None


def test_the_release_contract_names_what_arrived_and_what_left(capsys, api_catalogs):
    import inspect

    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    manifest = json.loads((api_catalogs / "en.json").read_text(encoding="utf-8"))
    setup = (REPO / "setup.py").read_text(encoding="utf-8")

    assert contract["console_script"] in setup
    assert contract["added"], "the contract names no arrival"
    for symbol, docstring in contract["added"].items():
        assert inspect.cleandoc(_resolve(symbol).__doc__) == docstring
        assert manifest["symbols"][symbol]["text"] == docstring, symbol
    for symbol in contract["removed"]:
        assert _resolve(symbol) is None, f"{symbol} is still importable"
        assert symbol not in manifest["symbols"], symbol
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    usage = " ".join(capsys.readouterr().out.split("\n\n")[0].split())
    assert usage == "usage: " + contract["cli"]["usage"]
