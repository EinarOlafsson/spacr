"""Pruning the spaCR home folder, the disk-cache manager and the opt-in purge.

Every test runs against a temporary home, so nothing real is deleted.
"""
import json
import os
import time
from pathlib import Path

import pytest

from spacr import install_cleanup as ic
from spacr import run_journal as rj

DAY = 86400.0
CACHE_VARS = ("CELLPOSE_LOCAL_MODELS_PATH", "HF_HOME", "TORCH_HOME",
              "SPACR_BACKENDS_DIR", "SPACR_NEWS_CACHE", "SPACR_HOME")


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A temporary home with spaCR's log folder inside it."""
    root = tmp_path / "home"
    root.mkdir()
    monkeypatch.setenv("HOME", str(root))
    monkeypatch.setenv("USERPROFILE", str(root))
    monkeypatch.setenv("XDG_CACHE_HOME", str(root / ".cache"))
    monkeypatch.setenv("SPACR_PORTABLE", "0")
    monkeypatch.setenv("SPACR_LOG_DIR", str(root / ".spacr" / "logs"))
    monkeypatch.delenv("SPACR_RUN_ID", raising=False)
    for name in CACHE_VARS:
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    return root


def _aged(path: Path, days: float, size: int = 1000) -> Path:
    """Write ``size`` bytes at ``path`` and date it ``days`` ago."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix or path.name.startswith("spacr"):
        path.write_bytes(b"x" * size)
    else:
        path.mkdir(exist_ok=True)
        (path / "log.txt").write_bytes(b"x" * size)
        os.utime(path / "log.txt", (time.time() - days * DAY,) * 2)
    os.utime(path, (time.time() - days * DAY,) * 2)
    return path


def test_age_alone_deletes_only_older_daily_logs(home):
    logs = home / ".spacr" / "logs"
    old = _aged(logs / "spacr-20200101.log", 60)
    rotated = _aged(logs / "spacr-20200101.log.1", 60)
    new = _aged(logs / "spacr-20991231.log", 2)
    other = _aged(logs / "gui-stalls.log", 60)
    plan = rj._prune_plan("logs", 30, 0)
    assert {Path(p).name for p, _ in plan["delete"]} == {old.name, rotated.name}
    deleted, freed, refused = rj._prune(plan)
    assert (deleted, freed, refused) == (2, 2000, [])
    assert new.exists() and other.exists() and not old.exists()


def test_size_cap_deletes_oldest_first_and_never_inside_the_age_floor(home):
    runs = rj.runs_root()
    oldest = _aged(runs / "2020-01-01_000000_aaaaaaaa__mask", 400, 3 * 2**20)
    older = _aged(runs / "2020-02-01_000000_bbbbbbbb__mask", 300, 3 * 2**20)
    recent = _aged(runs / "2026-01-01_000000_cccccccc__mask", 5, 30 * 2**20)
    plan = rj._prune_plan("run_folders", 90, 4)
    assert [Path(p).name for p, _ in plan["delete"]] == [oldest.name, older.name]
    plan = rj._prune_plan("run_folders", 90, 34)
    assert [Path(p).name for p, _ in plan["delete"]] == [oldest.name]
    rj._prune(plan)
    assert not oldest.exists() and older.exists() and recent.exists()


def test_a_folder_under_its_cap_is_left_alone(home):
    _aged(home / ".spacr" / "logs" / "runs" / "abc.jsonl", 100)
    assert rj._prune_plan("run_logs", 30, 500)["delete"] == []


def test_the_current_run_and_its_log_are_never_pruned(home, monkeypatch):
    run_logs = home / ".spacr" / "logs" / "runs"
    live = _aged(run_logs / "live123.jsonl", 100)
    _aged(run_logs / "live123.resources.json", 100)
    gone = _aged(run_logs / "done456.jsonl", 100)
    monkeypatch.setenv("SPACR_RUN_ID", "live123")
    plan = rj._prune_plan("run_logs", 1, 0)
    assert [Path(p).name for p, _ in plan["delete"]] == [gone.name]
    plan["delete"].append((str(live), 1000))
    deleted, _freed, refused = rj._prune(plan)
    assert deleted == 1 and live.exists() and refused == ["live123.jsonl: in use"]


def test_an_entry_touched_after_planning_is_kept(home):
    log = _aged(home / ".spacr" / "logs" / "spacr-20200101.log", 60)
    plan = rj._prune_plan("logs", 30, 0)
    os.utime(log, None)
    deleted, _freed, refused = rj._prune(plan)
    assert deleted == 0 and log.exists() and "changed" in refused[0]


def test_a_path_outside_the_folder_is_refused(home, tmp_path):
    outside = _aged(tmp_path / "spacr-elsewhere.log", 60)
    plan = rj._prune_plan("logs", 30, 0)
    plan["delete"] = [(str(outside), 1000)]
    assert rj._prune(plan)[0] == 0 and outside.exists()


def test_the_cache_list_measures_each_cache(home):
    (home / ".cache" / "huggingface" / "hub").mkdir(parents=True)
    (home / ".cache" / "huggingface" / "hub" / "w.bin").write_bytes(b"1" * 4096)
    rows = {row["key"]: row for row in rj._disk_caches()}
    assert set(rows) == {"models", "cellpose", "huggingface", "torch",
                         "backends", "news"}
    assert rows["huggingface"]["size"] >= 4096 and rows["huggingface"]["exists"]
    assert rows["torch"]["exists"] is False and rows["torch"]["size"] == 0
    assert rows["models"]["relocatable"] is False


def test_clearing_a_cache_keeps_the_folder_and_link_targets(home, tmp_path):
    torch_dir = home / ".cache" / "torch"
    (torch_dir / "hub").mkdir(parents=True)
    (torch_dir / "hub" / "m.pt").write_bytes(b"1")
    precious = tmp_path / "precious.txt"
    precious.write_text("keep")
    try:
        (torch_dir / "link").symlink_to(precious)
    except OSError:
        pass
    removed, refused = rj._clear_cache("torch")
    assert refused == [] and removed >= 1
    assert torch_dir.is_dir() and list(torch_dir.iterdir()) == []
    assert precious.read_text() == "keep"


def test_clearing_refuses_the_home_folder(home, monkeypatch):
    monkeypatch.setenv("TORCH_HOME", str(home))
    (home / "thesis.docx").write_text("mine")
    removed, refused = rj._clear_cache("torch")
    assert removed == 0 and refused and (home / "thesis.docx").exists()


def test_relocating_moves_records_and_reapplies(home, tmp_path, monkeypatch):
    news = home / ".spacr" / "news"
    news.mkdir(parents=True)
    (news / "releases.json").write_text("[]")
    target = rj._relocate_cache("news", tmp_path / "bigdisk")
    assert target.name == "spacr-news" and (target / "releases.json").exists()
    assert not news.exists()
    assert os.environ["SPACR_NEWS_CACHE"] == str(target)
    monkeypatch.delenv("SPACR_NEWS_CACHE")
    assert rj._apply_cache_locations() == 1
    assert rj._cache_folder("news") == target
    recorded = json.loads((home / ".spacr" / "cache_locations.json").read_text())
    assert recorded == {"SPACR_NEWS_CACHE": str(target)}


def test_relocation_is_refused_when_set_outside_spacr(home, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "theirs"))
    with pytest.raises(ValueError):
        rj._relocate_cache("huggingface", tmp_path / "bigdisk")
    with pytest.raises(ValueError):
        rj._relocate_cache("models", tmp_path / "bigdisk")


def _machine(home: Path, **env) -> "ic._Machine":
    environ = {"HOME": str(home), "XDG_CACHE_HOME": str(home / ".cache")}
    environ.update(env)
    return ic._Machine(platform="linux", environ=environ,
                       fs_root=str(home.parent), running_prefix=None)


def _user_files(home: Path) -> None:
    for folder in (".spacr/logs", ".spacr/backends/cp3", ".cache/spacr",
                   ".config/spacr", "spacr-demos", ".cache/huggingface/hub",
                   ".cache/torch", "Documents"):
        (home / folder).mkdir(parents=True, exist_ok=True)
        (home / folder / "f.txt").write_text("x")


def test_purge_lists_only_spacr_owned_paths(home, tmp_path):
    _user_files(home)
    backends = tmp_path / "spacr-backends"
    backends.mkdir()
    shared = tmp_path / "data"
    shared.mkdir()
    (home / ".spacr" / "cache_locations.json").write_text(json.dumps(
        {"SPACR_BACKENDS_DIR": str(backends), "HF_HOME": str(shared)}))
    targets = ic._purge_targets(_machine(home))
    names = {os.path.relpath(p, tmp_path) for p in targets}
    assert names == {"home/.spacr", "home/.cache/spacr", "home/.config/spacr",
                     "home/spacr-demos", "spacr-backends"}


def test_purge_asks_and_keeps_everything_unless_confirmed(home, capsys):
    _user_files(home)
    machine = _machine(home)
    assert ic._purge_command(machine, ask=lambda _q: None) == 3
    assert ic._purge_command(machine, ask=lambda _q: "no") == 3
    assert ic._purge_command(machine, dry_run=True) == 0
    assert (home / ".spacr").exists()
    assert ic._purge_command(machine, ask=lambda _q: "purge") == 0
    assert not (home / ".spacr").exists() and not (home / "spacr-demos").exists()
    assert (home / "Documents" / "f.txt").exists()
    assert (home / ".cache" / "huggingface" / "hub" / "f.txt").exists()
    assert "Nothing was deleted" in capsys.readouterr().out


def test_purge_never_deletes_the_running_environment(home):
    _user_files(home)
    machine = _machine(home)
    machine.running_prefix = str(home / ".spacr" / "backends" / "cp3")
    assert ic._purge_targets(machine) == [
        p for p in ic._purge_targets(machine) if ".spacr" not in p]


def test_purge_deletes_the_settings_registry_tree(home):
    class Registry:
        def __init__(self):
            self.keys = {"Software\\spacr\\qt": ["General"],
                         "Software\\spacr\\qt\\General": []}

        def values(self, key):
            return {"a": "1"} if key in self.keys else {}

        def subkeys(self, key):
            return list(self.keys.get(key, []))

        def delete_key(self, key):
            assert not self.keys[key]
            del self.keys[key]
            parent, _sep, name = key.rpartition("\\")
            if parent in self.keys:
                self.keys[parent].remove(name)

    machine = _machine(home)
    machine.registry = Registry()
    assert ic._purge_command(machine, yes=True) == 0
    assert machine.registry.keys == {}


def test_the_purge_command_line_is_opt_in(home, capsys, monkeypatch):
    _user_files(home)
    monkeypatch.setattr(ic, "_ask_on_terminal", lambda _q: None)
    root = str(home.parent)
    assert ic._main(["purge", "--root", root]) == 3
    assert (home / ".spacr").exists()
    assert ic._main(["purge", "--dry-run", "--root", root]) == 0
    assert (home / ".spacr").exists()
    assert ic._main(["purge", "--yes", "--root", root]) == 0
    assert not (home / ".spacr").exists()


def test_the_uninstallers_offer_purge():
    root = Path(__file__).resolve().parents[1] / "packaging" / "online"
    unix = (root / "install_spacr_unix.sh").read_text(encoding="utf-8")
    assert "spacr.install_cleanup purge" in unix
    assert unix.index("spacr.install_cleanup purge") < unix.index('rm -rf "$INSTALL_ROOT"')
    nsis = (root / "spacr_online_installer.nsi").read_text(encoding="utf-8")
    uninstall = nsis[nsis.index('Section "Uninstall"'):]
    assert '"/PURGE"' in uninstall and "install_cleanup purge" in uninstall
    mac = (root / "build_macos_online.sh").read_text(encoding="utf-8")
    assert "spacr.install_cleanup purge" in mac
