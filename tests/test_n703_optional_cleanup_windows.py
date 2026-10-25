"""Windows optional cache cleanup inspects and deletes through no-follow native handles.

The handle layer is injected, so this logic runs on Linux: an ``O_PATH``
descriptor stands in for a ``CreateFileW`` handle opened with
``FILE_FLAG_OPEN_REPARSE_POINT``. It stays on the object it opened when the
path is renamed or replaced, sees a link as itself, and ``/proc/self/fd``
answers what ``GetFinalPathNameByHandleW`` would. The last test drives the
real ctypes layer on Windows.
"""
from contextlib import contextmanager
import os
import stat

import pytest

from spacr import install_cleanup as ic


class _Handle:
    def __init__(self, path, delete):
        self.fd = os.open(path, os.O_PATH | os.O_NOFOLLOW)
        self.path = path
        self.delete = delete
        self.pending = False


class FakeHandles:
    """The native layer over ``O_PATH`` descriptors, with hooks for races."""

    def __init__(self):
        self.reparse = set()
        self.moved = {}
        self.before_open = {}
        self.before_delete = []
        self.live = set()

    def open(self, path, *, delete=False):
        hook = self.before_open.pop(path, None)
        if hook is not None:
            hook()
        if delete and self.before_delete:
            self.before_delete.pop(0)(path)
        if not os.path.lexists(path):
            raise FileNotFoundError(2, "missing", path)
        handle = _Handle(path, delete)
        self.live.add(handle)
        return handle

    def information(self, handle):
        details = os.fstat(handle.fd)
        attributes = 0x10 if stat.S_ISDIR(details.st_mode) else 0x20
        if stat.S_ISLNK(details.st_mode) or handle.path in self.reparse:
            attributes |= 0x400
        return (attributes, details.st_dev, details.st_ino, details.st_size,
                details.st_mtime_ns, details.st_ctime_ns)

    def final_path(self, handle):
        actual = os.readlink(f"/proc/self/fd/{handle.fd}")
        return self.moved.get(actual, actual)

    def delete(self, handle):
        assert handle.delete, "deletion asked of a handle opened without DELETE access"
        handle.pending = True

    def close(self, handle):
        if handle.pending:
            actual = os.readlink(f"/proc/self/fd/{handle.fd}")
            held, current = os.fstat(handle.fd), os.lstat(actual)
            assert (held.st_dev, held.st_ino) == (current.st_dev, current.st_ino)
            (os.rmdir if stat.S_ISDIR(held.st_mode) else os.unlink)(actual)
        os.close(handle.fd)
        self.live.discard(handle)


_injected = pytest.mark.skipif(not hasattr(os, "O_PATH") or not os.path.isdir("/proc/self/fd"),
                               reason="the injected handle layer needs O_PATH and /proc")

_FILES = {"news": "releases.json", "icons": "button-0123456789abcdef0123.png",
          "sounds": "space-0123456789ab/bed.wav"}


@pytest.fixture
def windows(tmp_path, monkeypatch):
    home = tmp_path / "home"
    base = home / ".spacr"
    for key, name in _FILES.items():
        path = base / key / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("original-" + key).encode())
    (base / "models").mkdir()
    (base / "models" / "user-model.pt").write_bytes(b"irreplaceable")
    handles = FakeHandles()
    monkeypatch.setattr(ic, "_optional_cleanup_windows_api", lambda: handles)
    monkeypatch.setattr(ic, "_optional_cleanup_directory",
                        lambda *_: pytest.fail("the POSIX descriptor walk ran on the Windows path"))
    machine = ic._Machine(environ={"HOME": str(home)}, platform="linux")
    yield machine, base, handles, tmp_path
    assert not handles.live, "a native handle was left open"


@contextmanager
def _permitted(paths):
    yield None


def _rows(preview):
    return {row["key"]: row for row in preview["folders"]}


def _preview(machine, guard=_permitted):
    return ic.preview_optional_cleanup("windows-op-" + os.urandom(4).hex(), system=machine, guard=guard)


def _choose(preview, keys):
    return {"operation_id": preview["operation_id"], "snapshot_id": preview["snapshot_id"], "keys": keys}


@_injected
def test_windows_preview_lists_known_caches_through_handles(windows):
    machine, base, handles, _tmp = windows
    rows = _rows(ic.preview_optional_cleanup("windows-op", system=machine))
    for key, name in _FILES.items():
        body = ("original-" + key).encode()
        assert rows[key]["reason"] == "coordinator_unavailable", key
        assert rows[key]["size_bytes"] == len(body)
        assert [dict(row) for row in rows[key]["contents"]] == [{"name": name, "size_bytes": len(body)}]
    assert rows["backgrounds"]["reason"] == "missing"
    assert rows["models"]["reason"] == "protected"
    assert all(row["selectable"] for key, row in _rows(_preview(machine)).items() if key in _FILES)


@_injected
def test_windows_apply_empties_only_the_selected_caches(windows):
    machine, base, handles, _tmp = windows
    preview = _preview(machine)
    outcome = ic.apply_optional_cleanup(preview, _choose(preview, ["news", "sounds"]),
                                        system=machine, guard=_permitted)
    assert outcome["ok"] and outcome["removed"] == 2
    assert list((base / "news").iterdir()) == [] and list((base / "sounds").iterdir()) == []
    assert (base / "icons" / _FILES["icons"]).read_bytes() == b"original-icons"
    assert (base / "models" / "user-model.pt").read_bytes() == b"irreplaceable"
    replay = _preview(machine)
    again = dict(_choose(replay, ["icons"]), operation_id=preview["operation_id"])
    assert ic.apply_optional_cleanup(replay, again, system=machine, guard=_permitted)["reason"] == "invalid_selection"


@_injected
@pytest.mark.parametrize("where", ["root_link", "entry_link", "root_junction", "home_junction", "sound_junction"])
def test_windows_preview_refuses_every_reparse_point(windows, where):
    machine, base, handles, tmp = windows
    foreign = tmp / "foreign"
    foreign.mkdir()
    (foreign / "releases.json").write_bytes(b"private foreign bytes")
    if where == "root_link":
        (base / "news" / "releases.json").unlink()
        (base / "news").rmdir()
        (base / "news").symlink_to(foreign, target_is_directory=True)
    elif where == "entry_link":
        (base / "news" / "releases.json").unlink()
        (base / "news" / "releases.json").symlink_to(foreign / "releases.json")
    elif where == "root_junction":
        handles.reparse.add(str(base / "news"))
    elif where == "home_junction":
        handles.reparse.add(str(base))
    else:
        handles.reparse.add(str(base / "sounds" / "space-0123456789ab"))
    rows = _rows(_preview(machine))
    refused = ["news"] if where in ("root_link", "entry_link", "root_junction") else (
        list(_FILES) if where == "home_junction" else ["sounds"])
    for key in _FILES:
        row = rows[key]
        if key in refused:
            assert row["reason"] == "unverified" and row["contents"] == () and not row["selectable"], key
        else:
            assert row["selectable"], key
    assert (foreign / "releases.json").read_bytes() == b"private foreign bytes"


@_injected
def test_windows_preview_refuses_a_handle_whose_final_path_leaves_the_home(windows):
    machine, base, handles, tmp = windows
    handles.moved[str(base / "news" / "releases.json")] = str(tmp / "elsewhere" / "releases.json")
    rows = _rows(_preview(machine))
    assert rows["news"]["reason"] == "unverified" and rows["icons"]["selectable"]


@_injected
def test_windows_sound_pack_swapped_before_its_open_is_refused(windows):
    machine, base, handles, tmp = windows
    pack = base / "sounds" / "space-0123456789ab"
    foreign = tmp / "foreign-sounds"
    foreign.mkdir()
    (foreign / "bed.wav").write_bytes(b"keep foreign")

    def swap():
        pack.rename(base / "sounds" / "saved-pack")
        pack.symlink_to(foreign, target_is_directory=True)

    handles.before_open[str(pack)] = swap
    rows = _rows(_preview(machine))
    assert rows["sounds"]["reason"] == "unverified" and rows["sounds"]["contents"] == ()
    assert (foreign / "bed.wav").read_bytes() == b"keep foreign"


@_injected
def test_windows_cache_root_replaced_during_inspection_is_unverified(windows, monkeypatch):
    machine, base, handles, _tmp = windows
    real_tree = ic._optional_cleanup_windows_tree

    def tree(api, handle, root, key):
        identity, entries, contents, total = real_tree(api, handle, root, key)
        (base / "news" / "late-writer.json").write_bytes(b"concurrent")
        return identity, entries, contents, total

    monkeypatch.setattr(ic, "_optional_cleanup_windows_tree", tree)
    rows = _rows(_preview(machine))
    assert rows["news"]["reason"] == "unverified" and rows["news"]["contents"] == ()


@_injected
def test_windows_spacr_home_changed_during_inspection_is_unverified(windows, monkeypatch):
    machine, base, handles, _tmp = windows
    real_tree = ic._optional_cleanup_windows_tree

    def tree(api, handle, root, key):
        result = real_tree(api, handle, root, key)
        (base / ("new-entry-" + key)).write_bytes(b"concurrent")
        return result

    monkeypatch.setattr(ic, "_optional_cleanup_windows_tree", tree)
    rows = _rows(_preview(machine))
    assert all(rows[key]["reason"] == "unverified" and not rows[key]["selectable"] for key in _FILES)


@_injected
def test_windows_file_swapped_before_its_deletion_is_kept(windows):
    machine, base, handles, _tmp = windows
    preview = _preview(machine)
    target = base / "news" / "releases.json"

    def swap(path):
        if path == str(target):
            target.unlink()
            target.write_bytes(b"a user's replacement")

    handles.before_delete.append(swap)
    outcome = ic.apply_optional_cleanup(preview, _choose(preview, ["news"]), system=machine, guard=_permitted)
    assert not outcome["ok"]
    assert outcome["folders"] == [{"key": "news", "ok": False, "reason": "retained", "removed": 0,
                                   "retained_path": str(base / "news")}]
    assert target.read_bytes() == b"a user's replacement"


@_injected
def test_windows_file_turned_into_a_link_before_its_deletion_is_kept(windows):
    machine, base, handles, tmp = windows
    preview = _preview(machine)
    target = base / "news" / "releases.json"
    outside = tmp / "outside.json"
    outside.write_bytes(b"outside the cache")

    def swap(path):
        target.unlink()
        target.symlink_to(outside)

    handles.before_delete.append(swap)
    outcome = ic.apply_optional_cleanup(preview, _choose(preview, ["news"]), system=machine, guard=_permitted)
    assert not outcome["ok"] and outcome["folders"][0]["reason"] == "retained"
    assert target.is_symlink() and outside.read_bytes() == b"outside the cache"


@_injected
def test_windows_root_swapped_for_a_link_after_preview_is_never_followed(windows):
    machine, base, handles, tmp = windows
    preview = _preview(machine)
    foreign = tmp / "foreign"
    foreign.mkdir()
    (foreign / "releases.json").write_bytes(b"original-news")
    (base / "news").rename(base / "saved-news")
    (base / "news").symlink_to(foreign, target_is_directory=True)
    outcome = ic.apply_optional_cleanup(preview, _choose(preview, ["news"]), system=machine, guard=_permitted)
    assert outcome == {"ok": False, "reason": "preview_changed"}
    assert (foreign / "releases.json").read_bytes() == b"original-news"
    assert (base / "saved-news" / "releases.json").read_bytes() == b"original-news"


def test_windows_layer_is_chosen_only_on_windows():
    assert (ic._optional_cleanup_windows_api() is None) == (os.name != "nt")


def _native_home(tmp_path):
    home = tmp_path / "home"
    base = home / ".spacr"
    for key, name in (("news", "releases.json"), ("icons", _FILES["icons"])):
        (base / key).mkdir(parents=True)
        (base / key / name).write_bytes(("original-" + key).encode())
    return ic._Machine(environ={"HOME": str(home), "USERPROFILE": str(home)}, platform="linux"), base


@pytest.mark.skipif(os.name != "nt", reason="drives the real Windows handle layer")
def test_native_windows_handles_preview_and_empty_a_selected_cache(tmp_path):
    machine, base = _native_home(tmp_path)
    preview = _preview(machine)
    rows = _rows(preview)
    assert rows["news"]["selectable"] and rows["news"]["size_bytes"] == len(b"original-news")
    outcome = ic.apply_optional_cleanup(preview, _choose(preview, ["news"]), system=machine, guard=_permitted)
    assert outcome["ok"] and outcome["removed"] == 1
    assert list((base / "news").iterdir()) == []
    assert (base / "icons" / _FILES["icons"]).read_bytes() == b"original-icons"


@pytest.mark.skipif(os.name != "nt", reason="drives the real Windows handle layer")
def test_native_windows_junction_cache_root_is_refused(tmp_path):
    import subprocess

    machine, base = _native_home(tmp_path)
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "releases.json").write_bytes(b"private foreign bytes")
    (base / "news" / "releases.json").unlink()
    (base / "news").rmdir()
    subprocess.run(["cmd", "/c", "mklink", "/J", str(base / "news"), str(foreign)],
                   check=True, capture_output=True)
    rows = _rows(_preview(machine))
    assert rows["news"]["reason"] == "unverified" and not rows["news"]["selectable"]
    assert rows["icons"]["selectable"]
    assert (foreign / "releases.json").read_bytes() == b"private foreign bytes"
