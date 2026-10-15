"""Native read-only installer inventory and unsupported-execution boundaries."""
from pathlib import Path
import json

import pytest

from spacr import install_cleanup as ic


@pytest.fixture
def cache(tmp_path):
    home = tmp_path / "home"
    folder = home / ".spacr" / "news"
    folder.mkdir(parents=True)
    body = json.dumps({"fetched": 1, "releases": []}).encode()
    (folder / "releases.json").write_bytes(body)
    machine = ic._Machine(environ={"HOME": str(home)}, platform="linux")
    return machine, folder, body


def selected(preview):
    return {"operation_id": preview["operation_id"],
            "snapshot_id": preview["snapshot_id"], "keys": ["news"]}


def test_preview_actual_size_content_and_immutable_operation(cache):
    machine, folder, body = cache
    preview = ic.preview_optional_cleanup("fresh-operation", system=machine)
    row = preview["folders"][0]
    assert row["path"] == str(folder)
    assert row["size_bytes"] == len(body)
    assert dict(row["contents"][0]) == {"name": "releases.json", "size_bytes": len(body)}
    assert row["reason"] == "coordinator_unavailable"
    assert not any(row["selectable"] for row in preview["folders"])
    with pytest.raises(TypeError):
        preview["operation_id"] = "replay"
    with pytest.raises(TypeError):
        row["contents"][0]["size_bytes"] = 0
    assert (folder / "releases.json").read_bytes() == body


def test_no_cross_process_lease_never_deletes_even_valid_fresh_choice(cache):
    machine, folder, body = cache
    preview = ic.preview_optional_cleanup("fresh-operation", system=machine)
    assert ic.apply_optional_cleanup(preview, selected(preview), system=machine) == {
        "ok": False, "reason": "coordinator_unavailable"}
    assert (folder / "releases.json").read_bytes() == body


@pytest.mark.parametrize("change", ["operation", "snapshot", "unknown", "duplicate", "serialized"])
def test_replayed_or_invalid_choices_cannot_authorize_cleanup(cache, change):
    machine, folder, body = cache
    preview = ic.preview_optional_cleanup("fresh-operation", system=machine)
    choice = selected(preview)
    if change == "operation":
        choice["operation_id"] = "other-operation"
    elif change == "snapshot":
        choice["snapshot_id"] = "stale"
    elif change == "unknown":
        choice["keys"] = ["models"]
    elif change == "duplicate":
        choice["keys"] *= 2
    else:
        preview = dict(preview)
    assert ic.apply_optional_cleanup(preview, choice, system=machine)["reason"] == "invalid_selection"
    assert (folder / "releases.json").read_bytes() == body


def test_changed_native_file_refuses_stale_preview(cache):
    machine, folder, body = cache
    preview = ic.preview_optional_cleanup("fresh-operation", system=machine)
    replacement = body + b" "
    (folder / "releases.json").write_bytes(replacement)
    from contextlib import contextmanager
    @contextmanager
    def guard(_paths):
        yield None
    assert ic.apply_optional_cleanup(preview, selected(preview), system=machine, guard=guard)["reason"] == "preview_changed"
    assert (folder / "releases.json").read_bytes() == replacement


@pytest.mark.parametrize("kind", ["external", "unknown", "symlink"])
def test_unverified_roots_and_unknown_contents_are_not_inspected_as_owned(cache, tmp_path, kind):
    machine, folder, body = cache
    if kind == "external":
        external = tmp_path / "spacr-news"
        external.mkdir()
        (external / "releases.json").write_bytes(body)
        machine.environ["SPACR_NEWS_CACHE"] = str(external)
    elif kind == "unknown":
        (folder / "irreplaceable-user-file").write_bytes(b"user-data")
    else:
        (folder / "releases.json").unlink()
        target = tmp_path / "user-file"
        target.write_bytes(body)
        (folder / "releases.json").symlink_to(target)
    row = ic.preview_optional_cleanup("fresh-operation", system=machine)["folders"][0]
    assert row["reason"] == "unverified" and row["contents"] == ()
    assert row["size_bytes"] is None and not row["selectable"]
    assert (folder / "releases.json").read_bytes() == body


def test_protected_models_and_shared_caches_are_never_recursively_scanned(cache, monkeypatch):
    machine, folder, body = cache
    models = folder.parent / "models"
    models.mkdir()
    (models / "user-model.pt").write_bytes(b"keep")
    original = ic.os.scandir
    visited = []
    def scan(path):
        visited.append(str(path))
        assert str(path) != str(models)
        return original(path)
    monkeypatch.setattr(ic.os, "scandir", scan)
    preview = ic.preview_optional_cleanup("fresh-operation", system=machine)
    protected = {row["key"]: row for row in preview["folders"][4:]}
    assert protected["models"]["path"] == str(models)
    assert all(row["size_bytes"] is None and not row["selectable"] for row in protected.values())
    assert (models / "user-model.pt").read_bytes() == b"keep"
    assert str(models) not in visited


def test_blank_operation_is_rejected_before_filesystem_inspection(monkeypatch):
    monkeypatch.setattr(ic.os, "scandir", lambda *_: pytest.fail("unexpected scan"))
    with pytest.raises(ValueError, match="operation_id"):
        ic.preview_optional_cleanup(" ")


def test_directory_swap_before_scan_never_follows_foreign_symlink(cache, tmp_path, monkeypatch):
    machine, folder, body = cache
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "releases.json").write_bytes(b"private foreign bytes")
    old = folder.with_name("saved-news")
    original = ic.os.scandir
    swapped = []
    def scan(descriptor):
        assert isinstance(descriptor, int)
        if not swapped:
            folder.rename(old)
            folder.symlink_to(foreign, target_is_directory=True)
            swapped.append(True)
        return original(descriptor)
    monkeypatch.setattr(ic.os, "scandir", scan)
    row = ic.preview_optional_cleanup("operation", system=machine)["folders"][0]
    assert row["reason"] == "unverified" and row["contents"] == ()
    assert (foreign / "releases.json").read_bytes() == b"private foreign bytes"
    assert (old / "releases.json").read_bytes() == body and folder.is_symlink()


def test_sound_child_swap_before_open_is_refused(cache, tmp_path, monkeypatch):
    machine, news, body = cache
    sounds = news.parent / "sounds"
    theme = sounds / "space-0123456789ab"
    theme.mkdir(parents=True)
    (theme / "bed.wav").write_bytes(b"cached sound")
    foreign = tmp_path / "foreign-sounds"
    foreign.mkdir()
    (foreign / "bed.wav").write_bytes(b"keep foreign")
    original = ic.os.open
    swapped = []
    def open_path(path, flags, *args, **kwargs):
        if path == theme.name and not swapped:
            theme.rename(sounds / "saved-theme")
            theme.symlink_to(foreign, target_is_directory=True)
            swapped.append(True)
        return original(path, flags, *args, **kwargs)
    monkeypatch.setattr(ic.os, "open", open_path)
    row = next(row for row in ic.preview_optional_cleanup("operation", system=machine)["folders"] if row["key"] == "sounds")
    assert row["reason"] == "unverified" and row["contents"] == ()
    assert (foreign / "bed.wav").read_bytes() == b"keep foreign"


def test_retained_paths_honor_external_overrides_xdg_and_relocation_without_scanning(cache, tmp_path):
    machine, folder, body = cache
    relocated = tmp_path / "relocated-news"
    relocated.mkdir()
    (relocated / "user-file").write_bytes(b"retain")
    (folder.parent / "cache_locations.json").write_text(json.dumps({"SPACR_NEWS_CACHE": str(relocated), "TORCH_HOME": "~/relocated-torch"}))
    machine.environ.update({"SPACR_BACKENDS_DIR": " ~/private-backends ", "XDG_CACHE_HOME": str(tmp_path / "xdg")})
    rows = {row["key"]: row for row in ic.preview_optional_cleanup("op", system=machine)["folders"]}
    assert rows["news"]["path"] == str(relocated) and rows["news"]["contents"] == ()
    assert rows["news"]["reason"] == "unverified"
    assert rows["backends"]["path"] == str(Path(machine.home) / "private-backends")
    assert rows["torch"]["path"] == str(Path(machine.home) / "relocated-torch")
    assert rows["huggingface"]["path"] == str(tmp_path / "xdg" / "huggingface")
    assert (relocated / "user-file").read_bytes() == b"retain"
    machine.environ["TORCH_HOME"] = str(tmp_path / "external-torch")
    rows = {row["key"]: row for row in ic.preview_optional_cleanup("op", system=machine)["folders"]}
    assert rows["torch"]["path"] == str(tmp_path / "external-torch")


def test_sound_override_strips_whitespace_while_other_writer_paths_remain_raw(cache, tmp_path):
    machine, folder, body = cache
    target = str(tmp_path / "external-sounds")
    machine.environ.update({"SPACR_SOUND_CACHE": " " + target + " ",
                            "SPACR_ICON_CACHE": " " + target + " "})
    rows = {row["key"]: row for row in ic.preview_optional_cleanup("op", system=machine)["folders"]}
    assert rows["sounds"]["path"] == target
    assert rows["sounds"]["reason"] == "unverified"
    assert rows["icons"]["path"] == str(Path(" " + target + " ").absolute())
    assert not rows["sounds"]["selectable"] and not rows["icons"]["selectable"]
