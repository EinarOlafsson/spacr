"""Only live, current, explicitly selected interface caches may be emptied."""
from contextlib import contextmanager
from pathlib import Path
import json
import threading

import pytest

from spacr import install_cleanup as ic

pytestmark = pytest.mark.posix_only


@pytest.fixture
def caches(tmp_path):
    home = tmp_path / "home"
    base = home / ".spacr"
    files = {}
    for key, name in [("news", "releases.json"), ("icons", "button-0123456789abcdef0123.png"),
                      ("backgrounds", "space-stars-64x64-s1-v3.png")]:
        directory = base / key
        directory.mkdir(parents=True)
        path = directory / name
        body = ("original-" + key).encode()
        path.write_bytes(body)
        files[key] = (path, body)
    models = base / "models"
    models.mkdir()
    (models / "user-model.pt").write_bytes(b"irreplaceable")
    machine = ic._Machine(environ={"HOME": str(home)}, platform="linux")
    state = {"active": False, "entered": 0, "inside": False}
    lock = threading.RLock()
    @contextmanager
    def guard(paths):
        with lock:
            state["entered"] += 1
            state["inside"] = True
            try:
                assert set(paths).issubset({str(base / key) for key in files})
                yield "active" if state["active"] else None
            finally:
                state["inside"] = False
    return machine, base, files, state, guard


def request(preview, keys):
    return {"operation_id": preview["operation_id"], "snapshot_id": preview["snapshot_id"], "keys": keys}


def fresh(caches):
    machine, base, files, state, guard = caches
    preview = ic.preview_optional_cleanup("op-" + str(base), system=machine, guard=guard)
    return preview


def test_two_selected_caches_clear_under_live_guard_without_touching_unselected_data(caches, monkeypatch):
    machine, base, files, state, guard = caches
    preview = fresh(caches)
    assert all(row["selectable"] for row in preview["folders"][:2])
    original = ic.os.unlink
    def unlink(path, *args, **kwargs):
        assert state["inside"], "mutation escaped actual coordinator context"
        return original(path, *args, **kwargs)
    monkeypatch.setattr(ic.os, "unlink", unlink)
    outcome = ic.apply_optional_cleanup(preview, request(preview, ["news", "icons"]), system=machine, guard=guard)
    assert outcome["ok"] and outcome["removed"] == 2
    assert list((base / "news").iterdir()) == [] and list((base / "icons").iterdir()) == []
    assert files["backgrounds"][0].read_bytes() == files["backgrounds"][1]
    assert (base / "models" / "user-model.pt").read_bytes() == b"irreplaceable"
    assert not list(base.glob(".spacr-cleanup-*"))


def test_same_operation_cannot_replay_even_with_fresh_empty_cache_snapshot(caches):
    machine, base, files, state, guard = caches
    preview = fresh(caches)
    assert ic.apply_optional_cleanup(preview, request(preview, ["news"]), system=machine, guard=guard)["ok"]
    replay = fresh(caches)
    assert ic.apply_optional_cleanup(replay, request(replay, ["news"]), system=machine, guard=guard) == {
        "ok": False, "reason": "operation_used"}


def test_live_active_guard_disables_preview_and_refuses_previously_selected_action(caches):
    machine, base, files, state, guard = caches
    preview = fresh(caches)
    state["active"] = True
    active = fresh(caches)
    assert not any(row["selectable"] for row in active["folders"])
    assert active["folders"][0]["reason"] == "active"
    result = ic.apply_optional_cleanup(preview, request(preview, ["news"]), system=machine, guard=guard)
    assert result == {"ok": False, "reason": "active"}
    assert files["news"][0].read_bytes() == files["news"][1]
    assert not list(base.glob(".spacr-cleanup-*"))


def test_changed_selected_file_refuses_before_any_detachment(caches):
    machine, base, files, state, guard = caches
    preview = fresh(caches)
    files["news"][0].write_bytes(b"new cache contents")
    result = ic.apply_optional_cleanup(preview, request(preview, ["news"]), system=machine, guard=guard)
    assert result == {"ok": False, "reason": "preview_changed"}
    assert files["news"][0].read_bytes() == b"new cache contents"
    assert not list(base.glob(".spacr-cleanup-*"))


def test_new_original_leaf_during_detachment_is_never_overwritten(caches, monkeypatch):
    machine, base, files, state, guard = caches
    preview = fresh(caches)
    rename = ic.os.rename
    def move(source, destination, *args, **kwargs):
        result = rename(source, destination, *args, **kwargs)
        (base / "news").mkdir()
        (base / "news" / "foreign-user-file").write_bytes(b"foreign replacement")
        return result
    monkeypatch.setattr(ic.os, "rename", move)
    result = ic.apply_optional_cleanup(preview, request(preview, ["news"]), system=machine, guard=guard)
    assert not result["ok"] and result["removed"] == 0
    retained = Path(result["folders"][0]["retained_path"]) / "cache" / "releases.json"
    assert retained.read_bytes() == files["news"][1]
    assert (base / "news" / "foreign-user-file").read_bytes() == b"foreign replacement"


def test_unknown_new_detached_entry_retains_entire_private_tree(caches, monkeypatch):
    machine, base, files, state, guard = caches
    preview = fresh(caches)
    rename = ic.os.rename
    def move(source, destination, *args, **kwargs):
        result = rename(source, destination, *args, **kwargs)
        for private in base.glob(".spacr-cleanup-*"):
            (private / "cache" / "user-data").write_bytes(b"new unknown")
        return result
    monkeypatch.setattr(ic.os, "rename", move)
    result = ic.apply_optional_cleanup(preview, request(preview, ["news"]), system=machine, guard=guard)
    assert not result["ok"] and result["removed"] == 0
    retained = Path(result["folders"][0]["retained_path"]) / "cache"
    assert (retained / "user-data").read_bytes() == b"new unknown"
    assert (retained / "releases.json").read_bytes() == files["news"][1]


@pytest.mark.parametrize("failure", ["rename", "unlink"])
def test_native_io_failure_keeps_cache_bytes_in_original_or_private_tree(caches, monkeypatch, failure):
    machine, base, files, state, guard = caches
    preview = fresh(caches)
    def refuse(*_args, **_kwargs):
        raise PermissionError("native locked cache fixture")
    monkeypatch.setattr(ic.os, failure, refuse)
    result = ic.apply_optional_cleanup(preview, request(preview, ["news"]), system=machine, guard=guard)
    assert not result["ok"] and result["removed"] == 0
    assert any(path.read_bytes() == files["news"][1] for path in base.rglob("releases.json"))


def test_sound_child_directory_contents_are_selected_and_cleaned_without_recursive_delete(caches):
    machine, base, files, state, old_guard = caches
    folder = base / "sounds" / "orbit-0123456789ab"
    folder.mkdir(parents=True)
    (folder / "bed.wav").write_bytes(b"sound cache")
    @contextmanager
    def guard(paths):
        state["inside"] = True
        try:
            yield None
        finally:
            state["inside"] = False
    preview = ic.preview_optional_cleanup("sound-op-" + str(base), system=machine, guard=guard)
    outcome = ic.apply_optional_cleanup(preview, request(preview, ["sounds"]), system=machine, guard=guard)
    assert outcome["ok"] and outcome["removed"] == 1
    assert list((base / "sounds").iterdir()) == []
    assert files["news"][0].read_bytes() == files["news"][1]


def test_serialized_guard_cannot_enable_or_execute_cleanup(caches):
    machine, base, files, state, guard = caches
    preview = ic.preview_optional_cleanup("op", system=machine, guard={"idle": True})
    assert not any(row["selectable"] for row in preview["folders"])
    assert ic.apply_optional_cleanup(preview, request(preview, ["news"]), system=machine,
                                     guard={"idle": True})["reason"] == "coordinator_unavailable"
    assert files["news"][0].read_bytes() == files["news"][1]


def test_foreign_root_swapped_at_rename_is_retained_without_deleting_either_tree(caches, monkeypatch):
    machine, base, files, state, guard = caches
    preview = fresh(caches)
    rename = ic.os.rename
    def swap(source, destination, *args, **kwargs):
        rename(base / "news", base / "original-news")
        (base / "news").mkdir()
        (base / "news" / "foreign-user-file").write_bytes(b"foreign body")
        return rename(source, destination, *args, **kwargs)
    monkeypatch.setattr(ic.os, "rename", swap)
    outcome = ic.apply_optional_cleanup(preview, request(preview, ["news"]), system=machine, guard=guard)
    assert not outcome["ok"] and outcome["removed"] == 0
    retained = Path(outcome["folders"][0]["retained_path"]) / "cache"
    assert (retained / "foreign-user-file").read_bytes() == b"foreign body"
    assert (base / "original-news" / "releases.json").read_bytes() == files["news"][1]


def test_dangling_link_inserted_after_detach_is_retained_as_link(caches, monkeypatch, tmp_path):
    machine, base, files, state, guard = caches
    preview = fresh(caches)
    rename = ic.os.rename
    target = tmp_path / "absent-user-path"
    def insert(source, destination, *args, **kwargs):
        result = rename(source, destination, *args, **kwargs)
        for private in base.glob(".spacr-cleanup-*"):
            (private / "cache" / "foreign-link").symlink_to(target)
        return result
    monkeypatch.setattr(ic.os, "rename", insert)
    outcome = ic.apply_optional_cleanup(preview, request(preview, ["news"]), system=machine, guard=guard)
    assert not outcome["ok"] and outcome["removed"] == 0
    retained = Path(outcome["folders"][0]["retained_path"]) / "cache"
    assert (retained / "foreign-link").is_symlink() and not target.exists()
    assert (retained / "releases.json").read_bytes() == files["news"][1]


def test_file_changed_after_detached_inventory_is_not_unlinked(caches, monkeypatch):
    machine, base, files, state, guard = caches
    preview = fresh(caches)
    tree = ic._optional_cleanup_tree
    changed = []
    def inspect(descriptor, key):
        result = tree(descriptor, key)
        private = list(base.glob(".spacr-cleanup-*"))
        if private and not changed:
            (private[0] / "cache" / "releases.json").write_bytes(b"foreign changed body")
            changed.append(True)
        return result
    monkeypatch.setattr(ic, "_optional_cleanup_tree", inspect)
    outcome = ic.apply_optional_cleanup(preview, request(preview, ["news"]), system=machine, guard=guard)
    assert not outcome["ok"] and outcome["removed"] == 0
    retained = Path(outcome["folders"][0]["retained_path"]) / "cache" / "releases.json"
    assert retained.read_bytes() == b"foreign changed body"


def test_actual_run_journal_guard_refuses_registered_worker_then_cleans_when_finalized(caches):
    from spacr import run_journal as journal
    machine, base, files, state, _guard = caches
    @contextmanager
    def guard(paths):
        with journal._ACTIVE_RUNS_LOCK:
            refusal = next((reason for path in paths if (reason := journal._cache_refusal(Path(path)))), None)
            yield refusal
    selected_before_worker = ic.preview_optional_cleanup("before-worker-" + str(base), system=machine, guard=guard)
    assert selected_before_worker["folders"][0]["selectable"]
    ready = threading.Event()
    release = threading.Event()
    run_path = str(base / "native-worker-run")
    def worker():
        with journal._ACTIVE_RUNS_LOCK:
            journal._ACTIVE_RUN_DIRS.add(run_path)
        ready.set()
        try:
            assert release.wait(5)
        finally:
            with journal._ACTIVE_RUNS_LOCK:
                journal._ACTIVE_RUN_DIRS.discard(run_path)
    thread = threading.Thread(target=worker)
    thread.start()
    try:
        assert ready.wait(5)
        active = ic.preview_optional_cleanup("worker-op-" + str(base), system=machine, guard=guard)
        assert not active["folders"][0]["selectable"]
        assert active["folders"][0]["reason"] == "active"
        assert ic.apply_optional_cleanup(selected_before_worker, request(selected_before_worker, ["news"]),
                                         system=machine, guard=guard) == {"ok": False, "reason": "active"}
        assert files["news"][0].read_bytes() == files["news"][1]
    finally:
        release.set()
        thread.join(5)
        assert not thread.is_alive()
    idle = ic.preview_optional_cleanup("worker-finalized-" + str(base), system=machine, guard=guard)
    assert idle["folders"][0]["selectable"]
    outcome = ic.apply_optional_cleanup(idle, request(idle, ["news"]), system=machine, guard=guard)
    assert outcome["ok"] and outcome["removed"] == 1
    assert (base / "models" / "user-model.pt").read_bytes() == b"irreplaceable"


def test_unselected_cache_refresh_does_not_invalidate_unchanged_selected_cache(caches):
    machine, base, files, state, guard = caches
    preview = fresh(caches)
    files["icons"][0].write_bytes(b"newly rendered unselected icon")
    outcome = ic.apply_optional_cleanup(preview, request(preview, ["news"]), system=machine, guard=guard)
    assert outcome["ok"] and outcome["removed"] == 1
    assert list((base / "news").iterdir()) == []
    assert files["icons"][0].read_bytes() == b"newly rendered unselected icon"
