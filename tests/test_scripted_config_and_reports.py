"""Table-driven tests of the scripted configuration, installer profile and crash report.

One general test per unit, each walking a case table with pytest's
``subtests`` fixture: how ``spacr.api.MaskConfig`` expands its sources and
refuses a configuration that cannot mean anything; how
``spacr.install_profile.write_profile`` writes atomically, cleans up after a
failed write and reads back only a profile that matches the schema; and what
``spacr.crashreport._gpu_facts`` reports for each state a torch install can be
in.  torch is replaced in ``sys.modules`` by a small stand-in so no GPU is
touched, and every file lives under ``tmp_path``.
"""
from __future__ import annotations

import json
import os
import re
import sys
import types
from pathlib import Path

import pytest

from spacr import crashreport
from spacr import install_profile as IP
from spacr.api import MaskConfig


def _case_dir(tmp_path, name):
    """A fresh per-case directory under ``tmp_path``."""
    folder = tmp_path / re.sub(r"[^A-Za-z0-9_]+", "_", name)
    folder.mkdir()
    return folder


# ---------------------------------------------------------------------------
# MaskConfig
# ---------------------------------------------------------------------------

def test_mask_config_expands_sources_and_refuses_meaningless_runs(
        subtests, tmp_path):
    """Sources keep their plurality as strings; empty, unknown or repeated settings are refused."""
    plates = [tmp_path / "plate01", tmp_path / "plate02"]
    for plate in plates:
        plate.mkdir()
    expand_cases = [
        ("a Path is one source", dict(src=Path(tmp_path), nucleus_channel=1),
         {"src": str(tmp_path), "nucleus_channel": 1, "cell_channel": None}),
        ("a str is one source", dict(src=str(tmp_path), cell_channel=0),
         {"src": str(tmp_path), "cell_channel": 0, "pipeline_style": "v1"}),
        ("a list of Paths is one string per plate",
         dict(src=plates, cell_channel=0),
         {"src": [str(p) for p in plates]}),
        ("a tuple of plates becomes a list",
         dict(src=tuple(plates), pathogen_channel=2, pipeline_style="v2"),
         {"src": [str(p) for p in plates], "pathogen_channel": 2,
          "pipeline_style": "v2"}),
        ("a new extra setting is merged",
         dict(src=tmp_path, cell_channel=0, extra={"my_extra": 3}),
         {"my_extra": 3}),
    ]
    for case_id, kwargs, expected in expand_cases:
        with subtests.test(msg=case_id):
            settings = MaskConfig(**kwargs).to_settings()
            assert {k: settings[k] for k in expected} == expected

    refuse_cases = [
        ("no channel at all", dict(src=tmp_path),
         "select at least one segmentation channel"),
        ("an unknown pipeline style",
         dict(src=tmp_path, cell_channel=0, pipeline_style="fast"),
         "pipeline_style must be 'v1' or 'v2'"),
        ("extra repeats one typed setting",
         dict(src=tmp_path, cell_channel=0, extra={"cell_channel": 2}),
         "extra repeats typed setting(s): cell_channel. Set those values on "
         "the configuration object instead."),
        ("extra repeats two typed settings",
         dict(src=tmp_path, cell_channel=0,
              extra={"nucleus_channel": 1, "cell_channel": 2}),
         "extra repeats typed setting(s): cell_channel, nucleus_channel. Set "
         "those values on the configuration object instead."),
    ]
    for case_id, kwargs, message in refuse_cases:
        with subtests.test(msg=case_id):
            with pytest.raises(ValueError) as excinfo:
                MaskConfig(**kwargs).to_settings()
            assert str(excinfo.value) == message


# ---------------------------------------------------------------------------
# install_profile.write_profile / read_profile
# ---------------------------------------------------------------------------

def _fail_replace(src, dst):
    raise OSError("the volume is read-only")


def _vanish_then_fail(src, dst):
    os.unlink(src)
    raise OSError("the volume went away mid-write")


def test_write_profile_is_atomic_and_read_profile_is_fussy(
        subtests, tmp_path, monkeypatch):
    """A failed write leaves nothing; a written profile reads back whole; a half-right one reads as absent."""
    for case_id, replace, message in [
            ("replace fails", _fail_replace, "the volume is read-only"),
            ("temporary already gone", _vanish_then_fail,
             "the volume went away mid-write")]:
        with subtests.test(msg=case_id):
            folder = _case_dir(tmp_path, case_id)
            with monkeypatch.context() as mp:
                mp.setattr(os, "replace", replace)
                with pytest.raises(OSError) as excinfo:
                    IP.write_profile(folder / "install-profile.json", "cpu",
                                     "none")
            assert str(excinfo.value) == message
            assert list(folder.iterdir()) == []

    with subtests.test(msg="a written profile reads back whole"):
        target = _case_dir(tmp_path, "roundtrip") / "install-profile.json"
        written = IP.write_profile(target, "CPU", "None",
                                   consent_collected=True, report_issues=True)
        assert IP.read_profile(target) == written
        assert json.loads(target.read_text(encoding="utf-8")) == written
        assert written["requested_backend"] == "cpu"
        assert written["detected_accelerator"] == "none"
        assert written["consent"] == {"collected": True,
                                      "share_diagnostics": False,
                                      "report_issues": True}
        assert written["without_wgpu"] is False
        assert [p.name for p in target.parent.iterdir()] == [target.name]

    absent_cases = [
        ("a schema from another version", {"schema": 99}),
        ("a backend that is not a token",
         {"requested_backend": "CUDA 12 please"}),
        ("a backend spaCR cannot run on", {"active_backend": "vulkan"}),
    ]
    for case_id, mutate in absent_cases:
        with subtests.test(msg=case_id):
            target = _case_dir(tmp_path, case_id) / "install-profile.json"
            IP.write_profile(target, "cpu", "none")
            payload = json.loads(target.read_text(encoding="utf-8"))
            payload.update(mutate)
            target.write_text(json.dumps(payload), encoding="utf-8")
            assert IP.read_profile(target) is None


# ---------------------------------------------------------------------------
# crashreport._gpu_facts
# ---------------------------------------------------------------------------

def _fake_torch(**over):
    torch = types.ModuleType("torch")
    torch.__version__ = "2.9.0"
    torch.version = types.SimpleNamespace(cuda="12.4")
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 2,
        get_device_name=lambda i: f"Fake GPU {i}",
        get_device_capability=lambda i: (8, 6 + i),
    )
    for key, value in over.items():
        setattr(torch.cuda, key, value)
    return torch


def _raise(message):
    def _explode(*args):
        raise RuntimeError(message)
    return _explode


def test_gpu_facts_answer_in_place_for_every_torch_state(subtests,
                                                         monkeypatch):
    """Each torch state yields exactly its facts, and a failure stops further probing."""
    base = {"torch": "2.9.0", "cuda_built": "12.4"}
    cases = [
        ("torch not importable", None, None),
        ("cuda probe fails", _fake_torch(is_available=_raise("driver mismatch")),
         {**base, "cuda_available": "unreadable: driver mismatch"}),
        ("no cuda", _fake_torch(is_available=lambda: False),
         {**base, "cuda_available": False}),
        ("devices cannot be enumerated",
         _fake_torch(get_device_name=_raise("device lost")),
         {**base, "cuda_available": True, "devices": "unreadable: device lost"}),
        ("devices listed", _fake_torch(),
         {**base, "cuda_available": True, "devices": [
             {"index": 0, "name": "Fake GPU 0", "capability": "8.6"},
             {"index": 1, "name": "Fake GPU 1", "capability": "8.7"}]}),
    ]
    for case_id, torch, expected in cases:
        with subtests.test(msg=case_id):
            with monkeypatch.context() as mp:
                mp.setitem(sys.modules, "torch", torch)
                facts = crashreport._gpu_facts()
            if torch is None:
                assert list(facts) == ["torch"]
                assert facts["torch"].startswith("not importable: ")
            else:
                assert facts == expected


def test_read_state_answers_a_dict_for_every_state_file(subtests, tmp_path,
                                                         monkeypatch):
    """The saved state is returned as written; anything unreadable is ``{}``."""
    monkeypatch.setattr(crashreport, "_reports_dir", lambda: tmp_path)
    saved = {"enabled": True, "notice_shown": True, "install_id": "abc"}
    cases = [
        ("no state file", None, {}),
        ("a saved state", json.dumps(saved), saved),
        ("text that is not JSON", "{not json", {}),
        ("JSON that is not an object", json.dumps([1, 2]), {}),
    ]
    for case_id, text, expected in cases:
        with subtests.test(msg=case_id):
            state = tmp_path / "state.json"
            state.unlink(missing_ok=True)
            if text is not None:
                state.write_text(text, encoding="utf-8")
            assert crashreport._read_state() == expected
