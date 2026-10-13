"""Small guards that keep a window opening when something around it fails.

Each pins a fallback by what the user still gets:

* on Python 3.9 ``importlib.resources.files`` raises ``TypeError`` for
  ``spacr.resources`` (a namespace package), and the layout policy and the
  plate templates are then read from the folder beside the package -- the
  same data, not nothing;
* a measured-width table whose rows are all unreadable answers "nothing
  measured", and a policy whose matrix height is not a number opens at the
  fallback height;
* the help strip's four-row cache is bounded: past 64 fonts it starts over
  rather than growing for the session;
* the shortcut editor is raised, not doubled, while it is open, and a
  window that refuses to remember it still gets an editor;
* the glass filter comes off even when the theme will not drop its claim;
* mapping PyTorch's libraries early stops at the first thing it does not
  expect and leaves the rest to ``import torch``.
"""
from __future__ import annotations

import importlib.resources
import logging
import os
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QMainWindow                            # noqa: E402

pytestmark = pytest.mark.qt


def _files_like_python_3_9(_package):
    raise TypeError("'NoneType' object is not subscriptable")


def test_the_layout_policy_is_read_beside_the_package_on_python_3_9(
        monkeypatch):
    from spacr.qt import _layout_policy

    monkeypatch.setattr(_layout_policy, "_CACHE", None)
    expected = _layout_policy.read_policy(refresh=True)
    assert expected, "the bundled layout policy did not load at all"

    monkeypatch.setattr(importlib.resources, "files", _files_like_python_3_9)
    assert _layout_policy.read_policy(refresh=True) == expected


def test_a_width_table_with_no_readable_row_measures_nothing():
    from spacr.qt import _layout_policy

    policy = {"minimum_width": {"one": "wide", "1.5": None}}
    assert _layout_policy.minimum_width_for(1.0, policy) is None


def test_a_matrix_height_that_is_not_a_number_opens_at_the_fallback():
    from spacr.qt import _layout_policy

    policy = {"minimum_width": {"1.0": 1000}, "matrix": {"height": "tall"}}
    assert _layout_policy.recommended_window_size((2000, 1400), 1.0,
                                                  policy) == (
        1000, _layout_policy.FALLBACK[1])


def test_the_plate_templates_are_read_beside_the_package_on_python_3_9(
        monkeypatch):
    from spacr.qt.widgets import plate_layout

    expected = [template.key for template in plate_layout.plate_templates()]
    assert expected, "no plate template shipped"
    monkeypatch.setattr(importlib.resources, "files", _files_like_python_3_9)
    assert [template.key
            for template in plate_layout.plate_templates()] == expected


def test_the_four_row_cache_starts_over_past_64_fonts(qtbot, monkeypatch):
    from spacr.qt.widgets import hint_bar

    strip = hint_bar.HintBar("Hover something.")
    qtbot.addWidget(strip)
    monkeypatch.setattr(hint_bar, "_FOUR_ROWS", {})
    fresh = strip._four_painted_rows()
    measured = dict(hint_bar._FOUR_ROWS)
    assert len(measured) == 1

    crowded = {f"font-{index}": 1 for index in range(65)}
    monkeypatch.setattr(hint_bar, "_FOUR_ROWS", crowded)
    assert strip._four_painted_rows() == fresh
    assert hint_bar._FOUR_ROWS == measured, (
        "the cache kept growing past its bound")


def test_an_open_shortcut_editor_is_raised_not_doubled(qtbot):
    from spacr.qt import shortcuts

    window = QMainWindow()
    qtbot.addWidget(window)
    first = shortcuts._open_keymap(window)
    try:
        assert first.isVisible()
        assert shortcuts._open_keymap(window) is first
        assert len(window.findChildren(shortcuts._KeymapDialog)) == 1
    finally:
        first.close()


class _ForgetfulWindow(QMainWindow):
    """A window that will not hold on to its shortcut editor."""

    def __setattr__(self, name, value):
        if name == "_spacr_keymap_dialog":
            raise AttributeError("this window keeps no editor")
        super().__setattr__(name, value)


def test_a_window_that_cannot_remember_its_editor_still_gets_one(qtbot):
    from spacr.qt import shortcuts

    window = _ForgetfulWindow()
    qtbot.addWidget(window)
    dialog = shortcuts._open_keymap(window)
    try:
        assert isinstance(dialog, shortcuts._KeymapDialog)
        assert dialog.isVisible()
        assert dialog.parent() is window
    finally:
        dialog.hide()


def test_the_glass_filter_comes_off_when_the_theme_keeps_its_claim(
        qapp, monkeypatch, caplog):
    from spacr.qt import theme
    from spacr.qt.widgets import glass

    removed = []
    installed = object()
    monkeypatch.setattr(glass, "_INSTALLED", installed)
    monkeypatch.setattr(glass, "_INSTALLED_APP", qapp)
    monkeypatch.setattr(glass, "_stop_watching_application_events",
                        lambda app, target: removed.append((app, target)))

    def keeps_its_claim(_wants):
        raise RuntimeError("the theme is already torn down")

    monkeypatch.setattr(theme, "_stop_letting_a_decorator_sheet",
                        keeps_its_claim)
    with caplog.at_level(logging.DEBUG, logger=glass.LOG.name):
        assert glass.uninstall_glass_everywhere() is True
    assert removed == [(qapp, installed)]
    assert glass._INSTALLED is None and glass._INSTALLED_APP is None
    assert "the theme kept the glass filter's claim" in caplog.text


def test_a_wheel_cuda_library_is_found_past_an_empty_nvidia_folder(
        tmp_path, monkeypatch):
    from spacr.qt import gil_priority

    empty = tmp_path / "first"
    (empty / "nvidia").mkdir(parents=True)
    wheel = tmp_path / "second"
    lib = wheel / "nvidia" / "cuda_nvrtc" / "lib"
    lib.mkdir(parents=True)
    (lib / "libnvrtc.so.12").write_bytes(b"")
    monkeypatch.setattr(sys, "path", [str(empty), str(tmp_path / "absent"),
                                      str(wheel)])

    assert gil_priority._wheel_cuda_library(
        "cuda_nvrtc", "libnvrtc.so.*[0-9]") == str(lib / "libnvrtc.so.12")
    assert gil_priority._wheel_cuda_library(
        "nvjitlink", "libnvJitLink.so.*[0-9]") is None


@pytest.fixture
def torch_build(tmp_path, monkeypatch):
    """A fake torch install whose ``lib`` folder each test fills."""
    from spacr.qt import gil_priority

    if not sys.platform.startswith("linux"):
        pytest.skip("the early mapping is Linux only")
    package = tmp_path / "torch"
    lib = package / "lib"
    lib.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    spec = SimpleNamespace(origin=str(package / "__init__.py"))
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec",
        lambda name, *args: spec if name == "torch"
        else real_find_spec(name, *args))
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr(gil_priority, "_TORCH_LIBRARIES_TRIED", False)
    monkeypatch.setattr(gil_priority, "_open_wheel_cuda_libraries",
                        lambda dlopen: True)
    opened = []
    answers = {}

    def dlopen(path, mode):
        name = os.path.basename(os.fsdecode(path))
        opened.append((name, mode))
        return answers.get(name, 1)

    monkeypatch.setattr(gil_priority, "_dlopen_without_the_lock",
                        lambda: dlopen)
    return SimpleNamespace(lib=lib, opened=opened, answers=answers,
                           module=gil_priority)


def test_a_build_without_its_global_library_is_left_to_import_torch(
        torch_build):
    assert torch_build.module._open_torch_libraries() is False
    assert torch_build.opened == []
    assert torch_build.module._open_torch_libraries() is False, (
        "a second attempt in the same process tried again")


def test_a_library_the_linker_refuses_stops_the_mapping(torch_build):
    for name in ("libtorch_global_deps.so", "libtorch_cpu.so"):
        (torch_build.lib / name).write_bytes(b"")
    torch_build.answers["libtorch_global_deps.so"] = 0

    assert torch_build.module._open_torch_libraries() is False
    assert torch_build.opened == [
        ("libtorch_global_deps.so", os.RTLD_NOW | os.RTLD_GLOBAL)]


def test_a_build_without_cuda_maps_the_rest_in_torchs_order(torch_build):
    for name in ("libtorch_global_deps.so", "libtorch_cpu.so"):
        (torch_build.lib / name).write_bytes(b"")

    assert torch_build.module._open_torch_libraries() is True
    assert torch_build.opened == [
        ("libtorch_global_deps.so", os.RTLD_NOW | os.RTLD_GLOBAL),
        ("libtorch_cpu.so", os.RTLD_NOW)]


def test_a_wheel_cuda_library_that_cannot_be_found_stops_the_mapping(
        torch_build, monkeypatch):
    for name in ("libtorch_global_deps.so", "libtorch_cpu.so"):
        (torch_build.lib / name).write_bytes(b"")
    monkeypatch.setattr(torch_build.module, "_open_wheel_cuda_libraries",
                        lambda dlopen: False)

    assert torch_build.module._open_torch_libraries() is False
    assert [name for name, _mode in torch_build.opened] == [
        "libtorch_global_deps.so"], "it went on past a missing NVRTC"


def test_a_dlopen_that_cannot_be_reached_leaves_it_to_import_torch(
        torch_build, monkeypatch, caplog):
    def no_c_library():
        raise OSError("libc.so.6: cannot open shared object file")

    (torch_build.lib / "libtorch_global_deps.so").write_bytes(b"")
    monkeypatch.setattr(torch_build.module, "_dlopen_without_the_lock",
                        no_c_library)
    with caplog.at_level(logging.DEBUG, logger=torch_build.module.LOG.name):
        assert torch_build.module._open_torch_libraries() is False
    assert "could not map the torch libraries early" in caplog.text
    assert torch_build.opened == []
