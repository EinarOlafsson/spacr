"""Table-driven tests of the setting-animation manifest in ``spacr.setting_animations``.

The manifest is a contract shared by the GUI, the docs build and the release
tests, so its value is in what it refuses.  One general test per unit, each
walking a case table with pytest's ``subtests`` fixture: one manifest entry
(``_parse_entry``), the manifest as a whole (``setting_animations``), and the
release check over the packaged GIFs (``validate_setting_animation_assets``).
Hand-written manifests live in ``tmp_path``; the module's caches are cleared
around every case so a fake manifest never leaks into another test.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from spacr import setting_animations as SA

_BODY = b"GIF89a" + b"\0" * 40
_DIGEST = hashlib.sha256(_BODY).hexdigest()


def _clear_caches():
    SA.setting_animations.cache_clear()
    SA.animations_by_setting.cache_clear()


@pytest.fixture(autouse=True)
def _fresh_caches():
    """Drop the manifest caches before and after every test."""
    _clear_caches()
    yield
    _clear_caches()


def _entry(root, **overrides):
    """A valid manifest entry, with a real GIF-shaped file behind it."""
    gifs = root / "gifs"
    gifs.mkdir(parents=True, exist_ok=True)
    (gifs / "probe.gif").write_bytes(_BODY)
    entry = {
        "slug": "probe_scene", "title": " Probe ", "category": "Cropping",
        "scene": "border", "settings": ["probe_setting"],
        "file": "gifs/probe.gif",
        "validation": {"sha256": _DIGEST, "frames": 4, "unique_frames": 3,
                       "bytes": len(_BODY)},
    }
    entry.update(overrides)
    return entry


def _with_validation(root, **fields):
    entry = _entry(root)
    entry["validation"] = dict(entry["validation"], **fields)
    return entry


def test_parse_entry_accepts_a_valid_entry_and_refuses_each_broken_field(
        tmp_path, monkeypatch, subtests):
    """A valid entry becomes a SettingAnimation; each fault names its field."""
    monkeypatch.setattr(SA, "_RESOURCE_ROOT", tmp_path)

    with subtests.test(msg="a valid entry"):
        animation = SA._parse_entry(_entry(tmp_path))
        assert animation == SA.SettingAnimation(
            slug="probe_scene", title="Probe", category="Cropping",
            scene="border", settings=("probe_setting",),
            relative_file="gifs/probe.gif", sha256=_DIGEST, frames=4,
            unique_frames=3, byte_size=len(_BODY))
        assert animation.path == tmp_path / "gifs" / "probe.gif"
        assert animation.docs_anchor == "setting-animation-probe-scene"
        assert animation.docs_url == (SA.ANIMATION_DOCS_BASE
                                      + "#setting-animation-probe-scene")

    refused = [
        ("not an object", lambda: ["gifs/probe.gif"],
         "animation entries must be JSON objects"),
        ("empty slug", lambda: _entry(tmp_path, slug=""),
         "invalid animation slug ''"),
        ("slug with spaces", lambda: _entry(tmp_path, slug="not a slug"),
         "invalid animation slug 'not a slug'"),
        ("no settings", lambda: _entry(tmp_path, settings=[]),
         "probe_scene: settings must be non-empty strings"),
        ("an empty setting", lambda: _entry(tmp_path, settings=["ok", ""]),
         "probe_scene: settings must be non-empty strings"),
        ("settings as a string", lambda: _entry(tmp_path, settings="one_key"),
         "probe_scene: settings must be non-empty strings"),
        ("no validation", lambda: _entry(tmp_path, validation=None),
         "probe_scene: validation metadata is missing"),
        ("blank title", lambda: _entry(tmp_path, title="  "),
         "probe_scene: title must be a non-empty string"),
        ("no category", lambda: _entry(tmp_path, category=None),
         "probe_scene: category must be a non-empty string"),
        ("empty scene", lambda: _entry(tmp_path, scene=""),
         "probe_scene: scene must be a non-empty string"),
        ("unsafe file path", lambda: _entry(tmp_path, file="../secret.gif"),
         "probe_scene: unsafe animation path '../secret.gif'"),
        ("file that is not a GIF", lambda: _entry(tmp_path, file="gifs/p.png"),
         "probe_scene: animation is not a GIF: 'gifs/p.png'"),
        ("frames not positive", lambda: _with_validation(tmp_path, frames=0),
         "probe_scene: validation field 'frames' must be a positive integer"),
        ("bytes as a bool", lambda: _with_validation(tmp_path, bytes=True),
         "probe_scene: validation field 'bytes' must be a positive integer"),
        ("gif not packaged", lambda: _entry(tmp_path, file="gifs/absent.gif"),
         "probe_scene: packaged GIF is missing: gifs/absent.gif"),
    ]
    for digest in ("", "abc", "z" * 64, 12345, "A" * 64, None):
        refused.append((f"digest {digest!r}",
                        lambda d=digest: _with_validation(tmp_path, sha256=d),
                        "probe_scene: invalid SHA-256 digest"))
    for case_id, build, message in refused:
        with subtests.test(msg=case_id):
            with pytest.raises(SA.SettingAnimationError) as caught:
                SA._parse_entry(build())
            assert str(caught.value) == message


def test_setting_animations_loads_or_refuses_the_whole_manifest(
        tmp_path, monkeypatch, subtests):
    """Unreadable, misshapen or ambiguous manifests are refused; a valid one resolves."""
    def manifest(payload, raw=None):
        path = tmp_path / "manifest.json"
        path.write_text(raw if raw is not None else json.dumps(payload),
                        encoding="utf-8")
        return path

    version = SA.SCHEMA_VERSION
    refused = [
        ("missing file", tmp_path / "not_there.json",
         "Could not load setting-animation manifest: "),
        ("not json", lambda: manifest(None, raw="{not json"),
         "Could not load setting-animation manifest: "),
        ("not an object", lambda: manifest([]),
         "setting-animation manifest must be an object"),
        ("unknown schema", lambda: manifest({"schema_version": 99,
                                             "animations": []}),
         "unsupported setting-animation manifest schema 99"),
        ("animations not a list",
         lambda: manifest({"schema_version": version, "animations": {}}),
         "manifest animations must be a list"),
        ("a slug claimed twice", lambda: manifest({
            "schema_version": version,
            "animations": [_entry(tmp_path),
                           _entry(tmp_path, settings=["other_setting"])]}),
         "setting-animation manifest has duplicate slugs"),
        ("a setting mapped twice", lambda: manifest({
            "schema_version": version,
            "animations": [_entry(tmp_path),
                           _entry(tmp_path, slug="other_scene")]}),
         "one setting key is mapped to more than one animation"),
        ("a broken entry", lambda: manifest({
            "schema_version": version, "animations": ["gifs/probe.gif"]}),
         "animation entries must be JSON objects"),
    ]
    monkeypatch.setattr(SA, "_RESOURCE_ROOT", tmp_path)
    for case_id, path, message in refused:
        with subtests.test(msg=case_id):
            _clear_caches()
            monkeypatch.setattr(SA, "_MANIFEST_PATH",
                                path() if callable(path) else path)
            with pytest.raises(SA.SettingAnimationError) as caught:
                SA.setting_animations()
            if message.endswith(": "):          # the OS / JSON error follows
                assert str(caught.value).startswith(message)
                assert isinstance(caught.value.__cause__, (OSError, ValueError))
            else:
                assert str(caught.value) == message

    with subtests.test(msg="a valid manifest resolves its key to its gif"):
        _clear_caches()
        second = _entry(tmp_path, slug="second_scene",
                        settings=["second_a", "second_b"])
        monkeypatch.setattr(SA, "_MANIFEST_PATH", manifest({
            "schema_version": version,
            "animations": [_entry(tmp_path), second]}))
        loaded = SA.setting_animations()
        assert [a.slug for a in loaded] == ["probe_scene", "second_scene"]
        assert SA.setting_animations() is loaded            # cached
        assert sorted(SA.animations_by_setting()) == ["probe_setting",
                                                      "second_a", "second_b"]
        assert SA.animation_for_setting("second_b").slug == "second_scene"
        assert SA.animation_path_for_setting("probe_setting") == \
            tmp_path / "gifs" / "probe.gif"
        assert SA.animation_path_for_setting("probe_settings") is None
        assert [a.slug for a in SA.iter_setting_animations()] == [
            "probe_scene", "second_scene"]


def _copy_layout(root, animations, size_of):
    """Write a file of ``size_of(animation)`` bytes at each animation's path."""
    for animation in animations:
        target = root.joinpath(*animation.relative_file.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"\0" * size_of(animation))


def test_validate_setting_animation_assets_checks_sizes_and_digests(
        tmp_path, monkeypatch, subtests):
    """The shipped GIFs pass; a missing, resized or edited GIF is named."""
    animations = SA.setting_animations()          # the real, shipped manifest
    first = animations[0]

    with subtests.test(msg="the shipped assets match their manifest"):
        counts = SA.validate_setting_animation_assets(check_hashes=True)
        assert counts == {
            "animations": len(animations),
            "setting_keys": len({k for a in animations for k in a.settings}),
            "bytes": sum(a.byte_size for a in animations)}
        assert SA.validate_setting_animation_assets() == counts

    cases = [
        ("an asset that cannot be inspected", "empty", None, False,
         f"Could not inspect {first.relative_file}: "),
        ("an asset of the wrong size", "bigger", lambda a: a.byte_size + 1,
         False, f"{first.slug}: expected {first.byte_size} bytes, "
                f"got {first.byte_size + 1}"),
        ("right size, wrong bytes, without hashing", "same_size",
         lambda a: a.byte_size, False, None),
        ("right size, wrong bytes, with hashing", "same_size",
         lambda a: a.byte_size, True,
         f"{first.slug}: packaged GIF digest does not match manifest"),
    ]
    for case_id, folder, size_of, check_hashes, message in cases:
        with subtests.test(msg=case_id):
            root = tmp_path / folder
            root.mkdir(exist_ok=True)
            if size_of is not None:
                _copy_layout(root, animations, size_of)
            with monkeypatch.context() as mp:
                mp.setattr(SA, "_RESOURCE_ROOT", root)
                if message is None:
                    got = SA.validate_setting_animation_assets(
                        check_hashes=check_hashes)
                    assert got["bytes"] == sum(a.byte_size for a in animations)
                    continue
                with pytest.raises(SA.SettingAnimationError) as caught:
                    SA.validate_setting_animation_assets(
                        check_hashes=check_hashes)
            assert str(caught.value).startswith(message)
