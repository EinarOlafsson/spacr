"""Item 475: SpotNet's DeepCell token, and SpotNet as an OPS spot detector.

THE TOKEN is read from DEEPCELL_ACCESS_TOKEN or ~/.spacr/deepcell_token and
handed to SpotNet's own worker and to nothing else: not pip, not the
install's self-test, not another backend. Every token here is fake and every
home is a scratch folder.

THE SETTING picks the detector. Native is the default; SpotNet is refused
with the reason when it cannot run, never silently swapped for native.
"""
from __future__ import annotations

import logging
import os

import numpy as np
import pytest

from spacr import _segmentation_backends as SB
from spacr import ops_engine

FAKE = "fake-token-0123456789"


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A scratch home with no token anywhere."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.delenv(SB._DEEPCELL_TOKEN_ENV, raising=False)
    (tmp_path / ".spacr").mkdir()
    return tmp_path


def _write_token(home, text, mode=0o600):
    path = home / ".spacr" / "deepcell_token"
    path.write_text(text, encoding="utf-8")
    os.chmod(path, mode)
    return path


def _installed(root):
    """A SpotNet environment that looks installed, with no packages."""
    env = root / "spotnet"
    python = SB._env_python(str(env))
    os.makedirs(os.path.dirname(python), exist_ok=True)
    open(python, "w").close()
    SB._write_marker(str(env), {"backend": "spotnet"})
    return env


def test_no_token_anywhere_is_none(home):
    assert SB._deepcell_token() == (None, None)


def test_the_file_is_read_and_stripped(home, caplog):
    path = _write_token(home, f"\n  {FAKE}  \n")
    with caplog.at_level(logging.WARNING):
        assert SB._deepcell_token() == (FAKE, str(path))
    assert not caplog.records


def test_the_variable_wins_over_the_file(home, monkeypatch):
    _write_token(home, "from-the-file")
    monkeypatch.setenv(SB._DEEPCELL_TOKEN_ENV, f" {FAKE} ")
    assert SB._deepcell_token() == (FAKE, SB._DEEPCELL_TOKEN_ENV)


def test_an_empty_file_is_no_token(home):
    _write_token(home, "  \n")
    assert SB._deepcell_token() == (None, None)


@pytest.mark.skipif(os.name == "nt", reason="POSIX permissions")
def test_a_file_others_can_read_warns_without_saying_the_token(home, caplog):
    path = _write_token(home, FAKE, mode=0o644)
    with caplog.at_level(logging.WARNING):
        assert SB._deepcell_token()[0] == FAKE
    text = " ".join(record.getMessage() for record in caplog.records)
    assert str(path) in text and "chmod 600" in text
    assert FAKE not in text


def test_only_spotnets_worker_gets_the_token(home, monkeypatch, tmp_path):
    _write_token(home, FAKE)
    monkeypatch.setenv(SB._DEEPCELL_TOKEN_ENV, FAKE)
    env = str(tmp_path / "envs" / "spotnet")
    served = SB._serve_env("spotnet", env)
    assert served[SB._DEEPCELL_TOKEN_ENV] == FAKE
    assert served["HOME"] == SB._spotnet_home(env)
    assert served["HOME"].startswith(env)
    assert SB._DEEPCELL_TOKEN_ENV not in SB._worker_env("spotnet", env)
    assert SB._DEEPCELL_TOKEN_ENV not in SB._clean_env(env)
    for name in SB._SPECS:
        if name != "spotnet":
            assert SB._DEEPCELL_TOKEN_ENV not in SB._serve_env(name, env)


def test_without_a_token_the_worker_env_has_none(home, tmp_path):
    served = SB._serve_env("spotnet", str(tmp_path / "spotnet"))
    assert SB._DEEPCELL_TOKEN_ENV not in served


def test_the_zoo_row_says_where_the_token_goes_and_never_what_it_is(home):
    note = SB._credential_note("spotnet")
    assert "deepcell_token" in note and SB._DEEPCELL_TOKEN_ENV in note
    assert "No token was found." in note
    path = _write_token(home, FAKE)
    note = SB._credential_note("spotnet")
    assert f"A token was found in {path}." in note
    assert FAKE not in note
    assert SB._credential_note("cellpose3") == ""


def test_the_zoo_lists_the_token_note_on_spotnets_row(home, monkeypatch,
                                                     tmp_path):
    from spacr import model_zoo

    monkeypatch.setenv(SB._ROOT_ENV, str(tmp_path / "backends"))
    rows = {entry.uri: entry for entry in model_zoo.installable_backend_entries()}
    notes = " ".join(rows["backend:spotnet"].notes)
    assert "deepcell_token" in notes
    assert "deepcell_token" not in " ".join(rows["backend:cellpose3"].notes)


def test_readiness_names_what_is_missing(home, tmp_path):
    root = tmp_path / "backends"
    root.mkdir()
    ready, reason = SB._spotnet_readiness(str(root))
    assert not ready and "not installed" in reason

    env = _installed(root)
    ready, reason = SB._spotnet_readiness(str(root))
    assert not ready
    assert "deepcell_token" in reason and SB._DEEPCELL_TOKEN_ENV in reason

    _write_token(home, FAKE)
    ready, reason = SB._spotnet_readiness(str(root))
    assert ready and FAKE not in reason

    os.remove(home / ".spacr" / "deepcell_token")
    cache = os.path.join(SB._spotnet_home(str(env)), ".deepcell", "models")
    os.makedirs(cache)
    open(os.path.join(cache, SB._SPOTNET_ARCHIVE), "wb").close()
    assert SB._spotnet_readiness(str(root))[0]


def test_detect_spots_sends_the_image_to_spotnets_worker(home, tmp_path):
    root = tmp_path / "backends"
    root.mkdir()
    _installed(root)
    _write_token(home, FAKE)
    seen = {}

    class _Worker:
        def request(self, op, **payload):
            seen["op"] = op
            seen["image"] = np.load(payload["image"])
            seen["threshold"] = payload["threshold"]
            return {"spots": [[1.5, 2.25], [3.0, 4.0]]}

    def worker_for(name, env):
        seen["name"] = name
        return _Worker()

    image = np.arange(12, dtype=np.uint16).reshape(3, 4)
    spots = SB._detect_spots(image, threshold=0.9, root=str(root),
                             worker_for=worker_for)
    assert seen["name"] == "spotnet" and seen["op"] == "detect_spots"
    assert seen["threshold"] == 0.9
    np.testing.assert_array_equal(seen["image"], image.astype(np.float32))
    np.testing.assert_array_equal(spots, [[1.5, 2.25], [3.0, 4.0]])


def test_detect_spots_refuses_with_the_reason(home, tmp_path):
    with pytest.raises(ImportError, match="not installed"):
        SB._detect_spots(np.zeros((4, 4)), root=str(tmp_path))


def test_native_is_the_default_detector():
    assert ops_engine._spot_detector({}) == "native"
    assert ops_engine._spot_detector({"ops_spot_detector": None}) == "native"
    assert ops_engine._spot_detector({"ops_spot_detector": " Native "}) == "native"


def test_an_unknown_detector_is_refused():
    with pytest.raises(ValueError, match="ops_spot_detector must be one of"):
        ops_engine._spot_detector({"ops_spot_detector": "bigfish"})


def test_spotnet_that_cannot_run_is_refused_not_swapped(monkeypatch):
    monkeypatch.setattr(SB, "_spotnet_readiness",
                        lambda: (False, "no DeepCell access token"))
    with pytest.raises(ValueError, match="no DeepCell access token"):
        ops_engine._spot_detector({"ops_spot_detector": "spotnet"})
    monkeypatch.setattr(SB, "_spotnet_readiness", lambda: (True, ""))
    assert ops_engine._spot_detector({"ops_spot_detector": "SpotNet"}) == "spotnet"


def test_spotnet_positions_become_whole_unique_pixels_inside_the_field():
    stack = np.zeros((3, 4, 10, 12), np.float32)
    stack[:, 2, 4, 5] = [100, 200, 400]
    seen = {}

    def detect(image, threshold):
        seen["image"], seen["threshold"] = image, threshold
        return [[4.4, 5.4], [3.6, 4.6], [-2.0, 30.0]]

    peaks = ops_engine._spotnet_peaks(stack, detect=detect)
    assert seen["threshold"] == ops_engine._SPOTNET_THRESHOLD
    assert seen["image"].shape == (10, 12)
    assert np.unravel_index(seen["image"].argmax(), (10, 12)) == (4, 5)
    assert peaks.dtype.kind == "i"
    np.testing.assert_array_equal(peaks, [[0, 11], [4, 5]])
    assert ops_engine._spotnet_peaks(
        stack, detect=lambda image, threshold: []).shape == (0, 2)


def test_the_setting_is_registered_with_native_default_and_its_licence():
    from spacr import ops_settings

    assert ops_settings.OPS_DEFAULTS["ops_spot_detector"] == "native"
    assert ops_settings.OPS_TYPES["ops_spot_detector"] is str
    assert "ops_spot_detector" in ops_settings.OPS_CATEGORIES["OPS decoding"]
    tip = ops_settings.OPS_TOOLTIPS["ops_spot_detector"]
    assert "NON-COMMERCIAL ACADEMIC USE ONLY" in tip
    assert "deepcell_token" in tip
