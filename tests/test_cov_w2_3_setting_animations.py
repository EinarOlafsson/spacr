"""The packaged setting-animation manifest, and every way it can be wrong.

The registry is deliberately Qt-free: the GUI, the docs build and the release
tests all resolve the same setting key to the same GIF through it. That makes
the manifest a contract, and the value of the contract is entirely in what it
refuses -- an unsafe path, a digest that is not one, a slug claimed twice, a
GIF that is not the size it was generated at.
"""
from __future__ import annotations

import builtins
import hashlib
import json

import pytest

from spacr import setting_animations as SA


@pytest.fixture
def caches_restored():
    """Drop the manifest caches before and after, so a fake one can be read."""
    SA.setting_animations.cache_clear()
    SA.animations_by_setting.cache_clear()
    try:
        yield
    finally:
        SA.setting_animations.cache_clear()
        SA.animations_by_setting.cache_clear()


def _write_manifest(tmp_path, monkeypatch, payload):
    """Point the registry at a hand-written manifest in ``tmp_path``."""
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(SA, "_MANIFEST_PATH", path)
    monkeypatch.setattr(SA, "_RESOURCE_ROOT", tmp_path)
    return path


def _entry(tmp_path, **overrides):
    """A valid manifest entry, with a real GIF-shaped file behind it."""
    gifs = tmp_path / "gifs"
    gifs.mkdir(exist_ok=True)
    body = b"GIF89a" + b"\0" * 40
    (gifs / "probe.gif").write_bytes(body)
    entry = {
        "slug": "probe_scene",
        "title": "Probe",
        "category": "Cropping",
        "scene": "border",
        "settings": ["probe_setting"],
        "file": "gifs/probe.gif",
        "validation": {
            "sha256": hashlib.sha256(body).hexdigest(),
            "frames": 4,
            "unique_frames": 3,
            "bytes": len(body),
        },
    }
    entry.update(overrides)
    return entry


# ---------------------------------------------------------------------------
# one entry at a time
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value,fragment", [
    ("", "must be a string"),
    (17, "must be a string"),
    ("/etc/passwd.gif", "unsafe animation path"),
    ("gifs/../../secret.gif", "unsafe animation path"),
    ("elsewhere/probe.gif", "unsafe animation path"),
    ("gifs/probe.png", "is not a GIF"),
])
def test_a_manifest_file_path_that_could_escape_the_package_is_refused(
        value, fragment):
    """The path is data from a file, so it is validated as data."""
    with pytest.raises(SA.SettingAnimationError, match=fragment):
        SA._safe_relative_file(value, "probe_scene")


@pytest.mark.parametrize("value", [0, -3, True, "4", 2.0, None])
def test_a_validation_count_that_is_not_a_positive_integer_is_refused(value):
    """``True`` is not 1 here: a frame count has to be a counted number."""
    with pytest.raises(SA.SettingAnimationError, match="frames"):
        SA._positive_integer(value, "probe_scene", "frames")


# ---------------------------------------------------------------------------
# the manifest as a whole
# ---------------------------------------------------------------------------

def test_a_valid_manifest_resolves_its_key_to_its_gif(tmp_path, monkeypatch,
                                                      caches_restored):
    """The whole point: an exact setting key becomes a packaged path."""
    _write_manifest(tmp_path, monkeypatch,
                    {"schema_version": SA.SCHEMA_VERSION,
                     "animations": [_entry(tmp_path)]})
    animation = SA.animation_for_setting("probe_setting")
    assert animation is not None and animation.slug == "probe_scene"
    assert SA.animation_path_for_setting("probe_setting") == animation.path
    assert SA.animation_path_for_setting("probe_settings") is None
    assert animation.docs_anchor == "setting-animation-probe-scene"
    assert animation.docs_url.endswith("#setting-animation-probe-scene")
    assert [a.slug for a in SA.iter_setting_animations()] == ["probe_scene"]


# ---------------------------------------------------------------------------
# the shipped assets
# ---------------------------------------------------------------------------

def test_every_shipped_animation_falls_below_an_impossible_bar():
    """The reporter collects every offender rather than stopping at the first.

    A fraction cannot exceed 1.0, so a minimum above it makes every shipped
    animation an offender -- which is how the collecting behaviour, not the
    assets, is put under test.
    """
    failures = SA.validate_animations_show_something(minimum=1.5)
    slugs = {a.slug for a in SA.setting_animations()}
    assert set(failures) == slugs
    assert all(0.0 <= value <= 1.0 for value in failures.values())


def test_a_border_animation_that_removes_an_interior_object_is_named(
        monkeypatch):
    """Only border-scene animations are judged, and each offender is listed."""
    border = [a for a in SA.setting_animations() if a.scene == "border"]
    if not border:
        pytest.skip("no border-scene animation is packaged")

    monkeypatch.setattr(SA, "measure_border_object_removal",
                        lambda path: {"interior": 2, "edge": 1})
    offenders = SA.validate_border_animations_remove_only_edge_objects()
    assert set(offenders) == {a.slug for a in border}
    assert set(offenders.values()) == {2}


def test_a_file_that_is_not_an_animation_shows_nothing(tmp_path):
    """An unreadable file reports "shows nothing" rather than raising."""
    not_a_gif = tmp_path / "text.gif"
    not_a_gif.write_bytes(b"this is not a GIF")
    assert SA.measure_visible_change(not_a_gif) == 0.0
    assert SA._animation_frames(not_a_gif) == []


def test_an_installation_without_pillow_reports_no_frames_rather_than_raising(
        monkeypatch, tmp_path):
    """Decoding is optional; the measurements say so by returning ``None``.

    The decoder imports Pillow inside the function, so a machine with no
    imaging stack reads ``None`` back rather than an ImportError.
    """
    real_import = builtins.__import__

    def no_pillow(name, *args, **kwargs):
        if name.split(".")[0] == "PIL":
            raise ImportError("No module named 'PIL'")
        return real_import(name, *args, **kwargs)

    gif = tmp_path / "probe.gif"
    gif.write_bytes(b"GIF89a")
    monkeypatch.setattr(builtins, "__import__", no_pillow)
    assert SA._animation_frames(gif) is None
