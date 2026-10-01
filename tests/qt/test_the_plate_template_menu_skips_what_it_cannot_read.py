"""The plate-template menu keeps the templates it can read.

Pinned behaviour of :func:`spacr.qt.widgets.plate_layout.plate_templates`:

* when the templates folder is missing from the installed package the menu
  is simply empty, with no exception;
* a template file that is not valid JSON, or whose design does not parse, is
  left off the menu while the good templates stay on it.
"""
from __future__ import annotations

import importlib.resources
import json

import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets import plate_layout  # noqa: E402

pytestmark = pytest.mark.qt


def test_a_package_without_the_templates_folder_offers_no_templates(
        monkeypatch):
    def _missing(package):
        raise ModuleNotFoundError(f"No module named {package!r}")

    monkeypatch.setattr(importlib.resources, "files", _missing)

    assert plate_layout.plate_templates() == []


def test_templates_that_do_not_parse_are_left_off_the_menu(
        tmp_path, monkeypatch):
    folder = tmp_path / "plate_templates"
    folder.mkdir()
    (folder / "a_good.json").write_text(json.dumps({
        "title": "Good layout", "description": "works",
        "plate_id": "plate1", "plate_format": 96, "conditions": [],
    }), encoding="utf-8")
    (folder / "b_not_json.json").write_text("{oops", encoding="utf-8")
    (folder / "c_bad_format.json").write_text(json.dumps({
        "title": "Bad", "plate_format": "ninety-six"}), encoding="utf-8")
    (folder / "notes.txt").write_text("not a template", encoding="utf-8")
    monkeypatch.setattr(importlib.resources, "files", lambda package: tmp_path)

    templates = plate_layout.plate_templates()

    assert [t.key for t in templates] == ["a_good"]
    assert templates[0].title == "Good layout"
    assert templates[0].description == "works"
