"""Item 700: what an anonymous report says about the run's settings and input.

A report may carry the settings a run used, a description of its source
folder and the computer it ran on, each reduced to values that name nobody.
These tests feed each reducer the awkward values a real run produces (paths,
names, non-finite numbers, long lists, unreadable images, a damaged
summary edited on disk) and check exactly what survives.
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from spacr import crashreport as cr


def test_long_problem_lines_are_cut_and_repeats_dropped():
    """A long problem is cut with an ellipsis; a repeat or blank is dropped."""
    problem = types.SimpleNamespace(setting="channels",
                                    message="must list " + "x " * 80)
    lines = cr.public_problem_lines([problem, problem, ""], limit=30)
    assert len(lines) == 1
    assert lines[0].startswith("channels: must list")
    assert lines[0].endswith("…") and len(lines[0]) <= 30


def test_a_path_marker_says_only_whether_the_path_exists(tmp_path):
    """A path becomes exists=yes or exists=no, never its own text."""
    assert cr._path_marker(str(tmp_path)) == "<path: exists=yes>"
    assert cr._path_marker(str(tmp_path / "gone")) == "<path: exists=no>"
    assert cr._path_marker("bad\0path") == "<path: exists=no>"


@pytest.mark.parametrize("key, value, expected", [
    ("diameter", None, None),
    ("save", True, True),
    ("batch_size", 8, 8),
    ("threshold", 0.5, 0.5),
    ("threshold", float("nan"), "nan"),
    ("threshold", float("inf"), "inf"),
    ("model", "   ", ""),
    ("model", "<text>", "<text>"),
    ("model", "<path: exists=no>", "<path: exists=no>"),
    ("model", "cyto3", "cyto3"),
    ("model", "/opt/nowhere-spacr/models/cyto3", "<path: exists=no>"),
    ("src", "plate1", "<path: exists=no>"),
    ("experiment", "plate1", "<text>"),
    ("model", "two words", "<text>"),
    ("channels", [0, 1, "dapi"], [0, 1, "dapi"]),
    ("channels", list(range(17)), "<list: n=17>"),
    ("channels", [[1, 2], {"a": 1}], ["<list: n=2>", "<mapping: n=1>"]),
    ("extra", {"a": 1, "b": 2}, "<mapping: n=2>"),
    ("extra", object(), "<object>"),
])
def test_each_setting_value_keeps_only_what_names_nobody(key, value,
                                                          expected):
    """Numbers and plain options stay; names, paths and objects are markers."""
    assert cr._public_setting_value(key, value) == expected


def test_public_settings_skip_odd_names_and_stop_at_the_cap(monkeypatch):
    """Settings are taken in name order, odd names skipped, at most the cap."""
    monkeypatch.setattr(cr, "MAX_SETTINGS", 2)
    out = cr.public_settings({"b": 2, "a": 1, "bad name": 3, "c": 4})
    assert out == {"a": 1, "b": 2}
    monkeypatch.setattr(cr, "MAX_SETTINGS", 150)
    assert cr.public_settings({"bad name": 3, "c": 4}) == {"c": 4}
    assert cr.public_settings(None) == {}


def test_the_first_source_reads_lists_and_skips_empty_values(tmp_path):
    """A list source gives its first folder; empty values give nothing."""
    assert cr._first_source({"src": [str(tmp_path)]}) == str(tmp_path)
    assert cr._first_source({"src": [], "source": "  "}) == ""
    assert cr._first_source(None) == ""


def test_image_shapes_are_read_from_tiff_numpy_and_png_headers(tmp_path):
    """Each format reports its shape and dtype or mode from its header."""
    import tifffile
    from PIL import Image

    tifffile.imwrite(tmp_path / "a.tif", np.zeros((3, 8, 9), np.uint16))
    np.save(tmp_path / "b.npy", np.zeros((5, 6), np.float32))
    Image.new("RGB", (7, 4)).save(tmp_path / "c.png")
    Image.new("L", (7, 4)).save(tmp_path / "d.png")
    assert cr._image_shape(str(tmp_path / "a.tif")) == ([3, 8, 9], "uint16")
    assert cr._image_shape(str(tmp_path / "b.npy")) == ([5, 6], "float32")
    assert cr._image_shape(str(tmp_path / "c.png")) == ([4, 7, 3], "RGB")
    assert cr._image_shape(str(tmp_path / "d.png")) == ([4, 7], "L")


def test_input_facts_count_images_and_read_the_first_header(tmp_path):
    """Only image files are counted; the first one gives shape and channels."""
    np.save(tmp_path / "a.npy", np.zeros((4, 5, 2), np.uint8))
    np.save(tmp_path / "b.npy", np.zeros((4, 5, 2), np.uint8))
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    (tmp_path / "README").write_text("x", encoding="utf-8")
    (tmp_path / "folder.tif").mkdir()
    facts = cr.public_input_facts({"src": str(tmp_path)})
    assert facts == {"images": 2, "extensions": ["npy"], "shape": [4, 5, 2],
                     "dtype": "uint8", "channels": 2}


def test_input_facts_name_one_channel_for_a_plane_and_a_wide_stack(tmp_path):
    """A 2-D image is one channel; a stack wider than 16 both ways is one."""
    plane = tmp_path / "plane"
    plane.mkdir()
    np.save(plane / "a.npy", np.zeros((4, 5), np.uint8))
    assert cr.public_input_facts({"src": str(plane)})["channels"] == 1
    wide = tmp_path / "wide"
    wide.mkdir()
    np.save(wide / "a.npy", np.zeros((20, 30, 40), np.uint8))
    assert cr.public_input_facts({"src": str(wide)})["channels"] == 1
    flat = tmp_path / "flat"
    flat.mkdir()
    np.save(flat / "a.npy", np.zeros((7,), np.uint8))
    assert "channels" not in cr.public_input_facts({"src": str(flat)})


def test_input_facts_survive_an_unreadable_image_and_stop_at_the_limit(
        tmp_path):
    """A damaged first image still counts; ``limit`` caps the entries seen."""
    (tmp_path / "a.tif").write_bytes(b"not a tiff")
    (tmp_path / "b.tif").write_bytes(b"not a tiff")
    facts = cr.public_input_facts({"src": str(tmp_path)})
    assert facts == {"images": 2, "extensions": ["tif"]}
    assert cr.public_input_facts({"src": str(tmp_path)}, limit=1)["images"] \
        == 1


def test_input_facts_are_empty_without_a_readable_folder(monkeypatch,
                                                         tmp_path):
    """No folder, a missing folder or a scan that fails gives no facts."""
    assert cr.public_input_facts({}) == {}
    assert cr.public_input_facts({"src": str(tmp_path)}) == {
        "images": 0, "extensions": []}
    assert cr.public_input_facts({"src": str(tmp_path / "gone")}) == {}

    def refuse(_path):
        """Fail the way an unreadable network folder does."""
        raise PermissionError("denied")

    monkeypatch.setattr(cr.os, "scandir", refuse)
    assert cr.public_input_facts({"src": str(tmp_path)}) == {}


def _distribution(text=None, error=None):
    """A stand-in for ``importlib.metadata.distribution``."""
    def distribution(_name):
        """Return a distribution whose direct_url.json reads ``text``."""
        if error is not None:
            raise error
        return types.SimpleNamespace(read_text=lambda _file: text)
    return distribution


def test_install_type_names_each_way_spacr_can_be_installed(monkeypatch,
                                                            tmp_path):
    """Frozen, editable, conda, pip and an unreadable install are told apart."""
    import importlib.metadata

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert cr._install_type() == "installer"
    monkeypatch.delattr(sys, "frozen")
    monkeypatch.setattr(importlib.metadata, "distribution",
                        _distribution("{'dir_info': {'editable': True}}"))
    assert cr._install_type() == "editable"
    monkeypatch.setattr(importlib.metadata, "distribution",
                        _distribution(error=LookupError("gone")))
    assert cr._install_type() == "unknown"
    monkeypatch.setattr(importlib.metadata, "distribution",
                        _distribution(None))
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    assert cr._install_type() == "pip"
    (tmp_path / "conda-meta").mkdir()
    assert cr._install_type() == "conda"


def _fake_torch(cuda=True, mps=True, broken=False):
    """A loaded-torch stand-in with one GPU of 8 GB, or one that breaks."""
    def available():
        """Report CUDA, or fail like a broken driver."""
        if broken:
            raise RuntimeError("driver")
        return cuda

    props = types.SimpleNamespace(name="NVIDIA RTX A4000 (user@host)",
                                  total_memory=8 * 2 ** 30)
    return types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=available,
                                   get_device_properties=lambda _i: props),
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: mps)))


def test_environment_reports_the_gpu_and_libraries_torch_already_knows(
        monkeypatch):
    """A loaded torch gives the GPU model, its memory, CUDA and MPS."""
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    monkeypatch.setitem(sys.modules, "PySide6.QtCore",
                        types.SimpleNamespace(qVersion=lambda: "6.7.2"))
    monkeypatch.setattr(cr, "_distribution_version",
                        lambda name: "1.0" if name == "numpy" else "")
    env = cr.public_environment()
    assert env["cuda"] is True and env["mps"] is True
    assert env["vram_gb"] == 8
    assert env["gpu_model"].startswith("NVIDIA RTX A4000")
    assert "@" not in env["gpu_model"]
    assert env["versions"] == {"numpy": "1.0", "qt": "6.7.2"}
    assert env["install"] in ("pip", "conda", "installer", "editable",
                              "unknown")


def test_environment_survives_broken_torch_qt_and_psutil(monkeypatch):
    """Facts that cannot be read are left out, never raised."""
    import psutil

    def broken():
        """Fail like a missing binding."""
        raise RuntimeError("broken")

    monkeypatch.setitem(sys.modules, "torch", _fake_torch(broken=True))
    monkeypatch.setitem(sys.modules, "PySide6.QtCore",
                        types.SimpleNamespace(qVersion=broken))
    monkeypatch.setattr(psutil, "virtual_memory", broken)
    env = cr.public_environment()
    assert "ram_gb" not in env and "cuda" not in env
    assert "qt" not in env["versions"]
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=False,
                                                          mps=False))
    env = cr.public_environment()
    assert env["cuda"] is False and env["mps"] is False
    assert "gpu_model" not in env
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "PySide6.QtCore", None)
    env = cr.public_environment()
    assert "cuda" not in env and "qt" not in env["versions"]


def test_a_chain_is_read_from_traceback_text_skipping_unparseable_blocks():
    """Each earlier block names its exception; one with none is skipped."""
    text = ("no traceback here\n\n"
            "During handling of the above exception, another exception "
            "occurred:\n\n"
            "Traceback (most recent call last):\n"
            '  File "x.py", line 1, in f\n'
            "KeyError: 'a'\n\n"
            "The above exception was the direct cause of the following "
            "exception:\n\n"
            "Traceback (most recent call last):\n"
            '  File "x.py", line 2, in g\n'
            "ValueError: bad\n")
    chain = cr._exception_chain(traceback_text=text)
    assert [link["type"] for link in chain] == ["KeyError"]
    assert cr._exception_chain() == []


def test_the_run_block_keeps_stage_time_and_preview_flag():
    """A negative or unreadable time is dropped; no preview flag is not sent."""
    assert cr._public_run({"stage": "measure", "elapsed_s": -1}) == {
        "stage": "measure"}
    assert cr._public_run({"elapsed_s": "soon", "dry_run": 1}) == {
        "dry_run": True}
    assert cr._public_run({"elapsed_s": 12.345}) == {"elapsed_s": 12.3}


def test_an_environment_block_from_disk_keeps_only_allowlisted_values():
    """Wrong types, unknown installs and odd versions are dropped."""
    assert cr._clean_environment("not a mapping") == {}
    out = cr._clean_environment({
        "os_version": "6.8", "arch": "x86_64", "gpu_model": "RTX",
        "cores": 8, "ram_gb": True, "vram_gb": 10 ** 9, "cuda": True,
        "mps": "yes", "install": "stolen", "versions": {"torch": "2.4",
                                                         "evil": "1"}})
    assert out == {"os_version": "6.8", "arch": "x86_64", "gpu_model": "RTX",
                   "cores": 8, "vram_gb": 100000, "cuda": True,
                   "versions": {"torch": "2.4"}}
    assert cr._clean_environment({"versions": "2.4"}) == {}


def test_input_facts_from_disk_keep_counts_extensions_shape_and_dtype():
    """Unknown extensions, booleans and non-integer sizes are dropped."""
    assert cr._clean_inputs(["not", "a", "mapping"]) == {}
    out = cr._clean_inputs({"images": -3, "channels": True,
                            "extensions": ["TIF", "exe"],
                            "shape": [4, "5", True, 6], "dtype": "uint16"})
    assert out == {"images": 0, "extensions": ["tif"], "shape": [4, 6],
                   "dtype": "uint16"}
    assert cr._clean_inputs({"shape": "4x5"}) == {}


def test_settings_from_disk_are_checked_again(monkeypatch):
    """Odd names are dropped, values reduced and the cap still applies."""
    assert cr._clean_settings(None) == {}
    assert cr._clean_settings({"bad name": 1, "src": "/x/y",
                               "batch": 2}) == {
        "src": "<path: exists=no>", "batch": 2}
    monkeypatch.setattr(cr, "MAX_SETTINGS", 1)
    assert cr._clean_settings({"a": 1, "b": 2}) == {"a": 1}


def test_a_summary_edited_on_disk_loses_a_chain_with_no_named_exception():
    """Chain entries without an exception type are dropped, then the chain."""
    out = cr._validated_summary({"chain": [{"type": "", "message": "x"},
                                           "not a mapping"]})
    assert "chain" not in out
    out = cr._validated_summary({
        "chain": [{"type": "KeyError", "message": "a"}],
        "settings": {"batch": 2}, "inputs": {"images": 3},
        "environment": {"cores": 4}, "run": {"stage": "mask"}})
    assert out["chain"] == [{"type": "KeyError", "message": "a"}]
    assert out["settings"] == {"batch": 2}
    assert out["inputs"] == {"images": 3}
    assert out["environment"] == {"cores": 4}
    assert out["run"] == {"stage": "mask"}


def test_the_relay_summary_is_kept_only_when_something_survives_scrubbing(
        monkeypatch):
    """A relay-written summary is scrubbed; an empty one is not stored."""
    answers = iter([
        (200, {"issue_number": 7, "ai_summary_source": "relay",
               "ai_summary": "The channel list is too short."}),
        (200, {"issue_number": 8, "ai_summary_source": "relay",
               "ai_summary": "   "}),
    ])
    monkeypatch.setattr(cr, "_post_json", lambda *_a: next(answers))
    sent = cr._send_one({"message": "x"}, 1.0)
    assert sent["status"] == cr.REPORT_SENT
    assert sent["ai_summary"] == "The channel list is too short."
    assert sent["ai_summary_source"] == "relay"
    sent = cr._send_one({"message": "x"}, 1.0)
    assert "ai_summary" not in sent and sent["issue_number"] == 8


def test_a_problem_that_is_only_a_number_does_not_enter_the_signature():
    """A bullet that is empty once numbers are removed is not a problem."""
    assert cr._normalized_problems("Problems:\n- 123\n- src is missing") == [
        "src is missing"]
