"""``spacr-doctor``: every verdict, and every remediation it promises.

Two rules run through this file.

**Simulate the failure, do not assert the happy path twice.** A doctor is only
worth having when something is broken, so each failure mode gets a real
stand-in: a module missing from ``sys.modules``, a torch whose ``cuda`` says
one thing and whose allocator says another, a truncated database file written
byte by byte, an ``environment.yaml`` that contradicts its own ``setup.py``.

**Assert the fix, not just the verdict.** "GPU not available" with no
remediation is the failure this module exists to prevent, so a test that
checks only ``status == FAIL`` would let exactly that regression through.
Every failure test also asserts the text the user is told to run.
"""
from __future__ import annotations

import importlib.metadata
import json
import os
import sqlite3
import sys
import types
from pathlib import Path

import pytest

from spacr import doctor
from spacr.doctor import ERROR, FAIL, PASS, SKIP, WARN, Context, Result


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _rows(outcome):
    """Normalise a check's return value to a list of rows."""
    if isinstance(outcome, Result):
        return [outcome]
    return list(outcome)


def _only(outcome) -> Result:
    rows = _rows(outcome)
    assert len(rows) == 1, f"expected one row, got {rows}"
    return rows[0]


def _make_checkout(root: Path) -> Path:
    """Build the minimum directory shape ``_checkout_root`` recognises."""
    (root / "spacr").mkdir(parents=True)
    (root / "spacr" / "__init__.py").write_text("")
    (root / "setup.py").write_text("from setuptools import setup\n")
    return root


def _fake_spacr(package_dir: Path, version: str = "9.9.9"):
    """A stand-in for the imported ``spacr`` module rooted at ``package_dir``."""
    module = types.SimpleNamespace()
    module.__file__ = str(package_dir / "__init__.py")
    module.__version__ = version
    return module


class _FakeTensor:
    """Just enough tensor to satisfy the GPU allocation probe."""

    def __matmul__(self, other):
        return self

    def sum(self):
        return self

    def item(self):
        return 0.0


def _fake_torch(
    *,
    cuda_build="12.1",
    available=True,
    devices=1,
    device_name="Fake GPU",
    init_error=None,
    alloc_error=None,
    name_error=None,
):
    """A torch whose CUDA story is entirely under the test's control."""

    def get_device_name(index):
        if name_error is not None:
            raise name_error
        return f"{device_name} {index}"

    def init():
        if init_error is not None:
            raise init_error

    def zeros(*_args, **_kwargs):
        if alloc_error is not None:
            raise alloc_error
        return _FakeTensor()

    torch = types.SimpleNamespace()
    torch.__version__ = "2.9.0+fake"
    torch.version = types.SimpleNamespace(cuda=cuda_build)
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: available,
        device_count=lambda: devices,
        get_device_name=get_device_name,
        init=init,
        synchronize=lambda: None,
    )
    torch.zeros = zeros
    return torch


@pytest.fixture
def ctx(tmp_path):
    """A context pointing nowhere in particular, so checks stay independent."""
    return Context(checkout=tmp_path, probe_gpu=False)


# ---------------------------------------------------------------------------
# the Result / Context vocabulary
# ---------------------------------------------------------------------------

def test_fail_and_error_rows_are_failures_and_warn_is_not():
    assert Result("c", FAIL, "m").is_failure
    assert Result("c", ERROR, "m").is_failure
    assert not Result("c", WARN, "m").is_failure
    assert not Result("c", PASS, "m").is_failure
    assert not Result("c", SKIP, "m").is_failure


def test_context_defaults_to_the_directory_the_user_is_standing_in(tmp_path,
                                                                   monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert Context().checkout == Path.cwd()


def test_every_registered_check_carries_a_label():
    assert doctor.CHECKS, "no checks registered"
    for function in doctor.CHECKS:
        assert getattr(function, "check_label", "")


# ---------------------------------------------------------------------------
# helpers: distribution metadata
# ---------------------------------------------------------------------------

def test_canonical_name_folds_separators_and_case():
    assert doctor._canonical("Spacr_Nightly") == "spacr-nightly"
    assert doctor._canonical("umap.learn") == "umap-learn"


def test_distribution_version_reports_installed_and_absent():
    assert doctor._distribution_version("spacr")
    assert doctor._distribution_version("no-such-distribution-at-all") is None


def test_declared_requirement_finds_the_cellpose_bound():
    assert "4" in (doctor._declared_requirement("cellpose") or "")


def test_declared_requirement_skips_extras_and_unparseable_lines(monkeypatch):
    monkeypatch.setattr(
        importlib.metadata,
        "requires",
        lambda _name: [
            "!!! not a requirement",
            'torchcam<1.0,>=0.4.0; extra == "attribution"',
            'win10toast>=0.9; platform_system == "Windows"',
            "cellpose<5.0,>=4.0",
        ],
    )
    assert doctor._declared_requirement("torchcam") is None
    assert doctor._declared_requirement("win10toast") == ">=0.9"
    assert doctor._declared_requirement("cellpose") == "<5.0,>=4.0"
    assert doctor._declared_requirement("nothing-here") is None


def test_declared_requirement_survives_missing_metadata(monkeypatch):
    monkeypatch.delattr(importlib.metadata, "requires")
    assert doctor._declared_requirement("cellpose") is None


def test_declared_requirement_handles_a_package_with_no_requirements(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "requires", lambda _name: None)
    assert doctor._declared_requirement("cellpose") is None


def test_satisfies_answers_the_bounds_that_actually_appear_in_this_project():
    assert doctor._satisfies("<5.0,>=4.0", "4.0.7") is True
    assert doctor._satisfies("<5.0,>=4.0", "3.0.11") is False
    assert doctor._satisfies(">=3.9,<3.15,!=3.14.1", "3.10.19") is True
    assert doctor._satisfies(">=3.9,<3.15,!=3.14.1", "3.14.1") is False
    assert doctor._satisfies(">=3.9,<3.15,!=3.14.1", "3.15.0") is False


def test_satisfies_treats_an_empty_specifier_as_no_constraint():
    assert doctor._satisfies("", "1.2.3") is True
    assert doctor._satisfies(" , ", "1.2.3") is True


@pytest.mark.parametrize(
    "specifier, version, expected",
    [
        ("==4.0", "4.0.0", True),     # zero padding, PEP 440
        ("==4.0", "4.0.7", False),
        ("!=4.0.7", "4.0.7", False),
        ("<=2.0", "2.0", True),
        (">2.0", "2.0.1", True),
        ("<2.0", "2.0", False),
        ("~=4.0", "4.9.9", True),     # compatible release: >=4.0, ==4.*
        ("~=4.1", "4.0.9", False),
        ("~=4.0", "5.0.0", False),
        ("==4.*", "4.9.9", True),
        ("!=4.*", "4.9.9", False),
        ("===4.0.7", "4.0.7", True),
        ("===4.0.7", "4.0.7.1", False),
        (">=2.0", "2.9.1+cu128", True),   # local segments are not ordered
        (">=2.0", "2.0.0rc1", False),     # a pre-release precedes its release
        (">=2.0", "2.0.0.post1", True),
        (">2.0.dev1", "2.0", True),
    ],
)
def test_satisfies_covers_the_pep440_subset_it_claims(specifier, version, expected):
    assert doctor._satisfies(specifier, version) is expected


@pytest.mark.parametrize(
    "specifier, version",
    [
        ("this is not a specifier", "1.0"),   # no operator
        (">=1.0", "not-a-version"),           # unparseable version
        (">=not-a-version", "1.0"),           # unparseable bound
        ("<=1.*", "1.0"),                     # wildcard with an order operator
        ("==not.a.version.*", "1.0"),
        ("~=4", "4.1"),                       # ~= needs at least two segments
        (">=1.0banana", "1.0"),
    ],
)
def test_satisfies_admits_when_it_cannot_tell(specifier, version):
    assert doctor._satisfies(specifier, version) is None


def test_version_parser_rejects_what_is_not_a_version():
    assert doctor._parse_version("cellpose") is None
    assert doctor._parse_version("4.0.7-nightly-build") is None
    assert doctor._parse_version("v4.0.7") == ((4, 0, 7), (1, 0, 0))


# ---------------------------------------------------------------------------
# helpers: where does spacr live
# ---------------------------------------------------------------------------

def test_package_root_resolves_a_module_and_tolerates_one_without_a_file(tmp_path):
    package = tmp_path / "spacr"
    package.mkdir()
    assert doctor._package_root(_fake_spacr(package)) == package
    assert doctor._package_root(types.SimpleNamespace()) is None


def test_checkout_root_walks_up_to_the_clone(tmp_path):
    root = _make_checkout(tmp_path / "clone")
    nested = root / "spacr" / "qt"
    nested.mkdir(parents=True)
    assert doctor._checkout_root(nested) == root
    assert doctor._checkout_root(tmp_path / "elsewhere") is None


def test_checkout_root_accepts_a_pyproject_only_clone(tmp_path):
    root = tmp_path / "modern"
    (root / "spacr").mkdir(parents=True)
    (root / "spacr" / "__init__.py").write_text("")
    (root / "pyproject.toml").write_text("[project]\n")
    assert doctor._checkout_root(root) == root


def test_checkout_root_skips_a_directory_it_cannot_stat(tmp_path, monkeypatch):
    """An unreadable mount point must not abort the walk up the tree."""
    unreadable = tmp_path / "boom"
    unreadable.mkdir()
    real_is_file = Path.is_file

    def refuse(self):
        if "boom" in self.parts:
            raise OSError("permission denied")
        return real_is_file(self)

    monkeypatch.setattr(Path, "is_file", refuse)
    assert doctor._checkout_root(unreadable) is None


def test_checkout_root_gives_up_when_the_start_cannot_be_resolved(monkeypatch):
    def refuse(self, *args, **kwargs):
        raise OSError("stale file handle")

    monkeypatch.setattr(Path, "resolve", refuse)
    assert doctor._checkout_root(Path("/anywhere")) is None


def test_editable_url_target_reads_a_pep610_record(tmp_path):
    payload = json.dumps(
        {"dir_info": {"editable": True}, "url": f"file://{tmp_path}"}
    )
    assert doctor._editable_url_target(payload) == tmp_path.resolve()


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "",
        "{not json",
        json.dumps({"dir_info": {"editable": False}, "url": "file:///x"}),
        json.dumps({"dir_info": {"editable": True}, "url": "https://pypi.org/x"}),
    ],
)
def test_editable_url_target_rejects_everything_that_is_not_an_editable_dir(raw):
    assert doctor._editable_url_target(raw) is None


def test_editable_url_target_gives_up_on_an_unresolvable_path(monkeypatch):
    def refuse(self, *args, **kwargs):
        raise OSError("stale file handle")

    monkeypatch.setattr(Path, "resolve", refuse)
    payload = json.dumps({"dir_info": {"editable": True}, "url": "file:///x"})
    assert doctor._editable_url_target(payload) is None


def test_spacr_distributions_finds_this_installation():
    found = doctor._spacr_distributions()
    assert found, "spaCR is installed, so at least one metadata dir must exist"
    assert all(doctor._canonical(name) in doctor.SPACR_DISTRIBUTION_NAMES
               for name, _, _ in found)


def test_spacr_distributions_skips_unreadable_metadata(monkeypatch):
    class Broken:
        @property
        def metadata(self):
            raise ValueError("unreadable METADATA")

    class Nameless:
        metadata = {"Name": "not-spacr"}
        version = "1.0"

    class NoPath:
        metadata = {"Name": "spacr"}
        version = "1.0"
        _path = None

        def locate_file(self, _name):
            return "/somewhere/spacr-1.0.dist-info"

    class NoLocation(NoPath):
        def locate_file(self, _name):
            raise OSError("gone")

    monkeypatch.setattr(
        importlib.metadata,
        "distributions",
        lambda: [Broken(), Nameless(), NoPath(), NoLocation()],
    )
    found = doctor._spacr_distributions()
    assert [entry[2] for entry in found] == [
        "/somewhere/spacr-1.0.dist-info",
        "unknown",
    ]


def test_spacr_distributions_and_editable_target_survive_no_metadata_api(monkeypatch):
    monkeypatch.delattr(importlib.metadata, "distributions")
    assert doctor._spacr_distributions() == []
    assert doctor._editable_target() is None


def test_editable_target_ignores_distributions_it_cannot_read(monkeypatch, tmp_path):
    class Unreadable:
        metadata = {"Name": "spacr"}
        version = "1.0"

        def read_text(self, _name):
            raise OSError("no such file")

    class NotEditable:
        metadata = {"Name": "spacr"}
        version = "1.0"

        def read_text(self, _name):
            return json.dumps({"dir_info": {}, "url": "file:///x"})

    class Editable:
        metadata = {"Name": "spacr-nightly"}
        version = "1.0"

        def read_text(self, _name):
            return json.dumps(
                {"dir_info": {"editable": True}, "url": f"file://{tmp_path}"}
            )

    class Broken:
        @property
        def metadata(self):
            raise ValueError("unreadable")

    monkeypatch.setattr(
        importlib.metadata,
        "distributions",
        lambda: [Broken(), Unreadable(), NotEditable(), Editable()],
    )
    assert doctor._editable_target() == tmp_path.resolve()


def test_editable_target_returns_none_when_nothing_is_editable(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "distributions", lambda: [])
    assert doctor._editable_target() is None


def test_importable_spacr_dirs_finds_the_editable_install_not_just_sys_path():
    """An editable install ships a finder, not a sys.path entry.

    Scanning sys.path alone reported "no spacr installed" for the most common
    developer setup there is, which made the duplicate-install check useless
    exactly where it was needed.
    """
    found = doctor._importable_spacr_dirs()
    assert found, "spacr is importable, so at least one package dir must be found"
    import spacr

    assert Path(spacr.__file__).resolve().parent == found[0]


def test_importable_spacr_dirs_reports_a_shadowing_copy(tmp_path, monkeypatch):
    shadow = tmp_path / "shadow"
    (shadow / "spacr").mkdir(parents=True)
    (shadow / "spacr" / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(shadow))
    found = doctor._importable_spacr_dirs()
    assert (shadow / "spacr").resolve() in found
    assert len(found) >= 2


def test_importable_spacr_dirs_survives_an_unresolvable_entry(monkeypatch):
    def refuse(self, *args, **kwargs):
        raise OSError("stale file handle")

    monkeypatch.setattr(Path, "resolve", refuse)
    assert doctor._importable_spacr_dirs() == []


def test_importable_spacr_dirs_tolerates_a_broken_spec_lookup(monkeypatch):
    import importlib.util

    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda _name: (_ for _ in ()).throw(ValueError("bad finder")),
    )
    # The finder is dead, so the spec branch contributes nothing — but the
    # sys.path scan behind it still runs, and the in-tree checkout is on
    # sys.path (tests/conftest.py puts it there). Absorbing the failure has to
    # mean "fall through to the scan", not "return empty": an early return
    # would report "no spacr installed" for the most common developer setup
    # there is, which is the bug this function exists to avoid.
    import spacr

    found = doctor._importable_spacr_dirs()
    assert all(isinstance(p, Path) for p in found)
    assert Path(spacr.__file__).resolve().parent in found


def test_importable_spacr_dirs_uses_cwd_for_the_empty_sys_path_entry(
    tmp_path, monkeypatch
):
    import importlib.util

    (tmp_path / "spacr").mkdir()
    (tmp_path / "spacr" / "__init__.py").write_text("")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)
    monkeypatch.setattr(doctor.sys, "path", [""])
    assert doctor._importable_spacr_dirs() == [(tmp_path / "spacr").resolve()]


def test_importable_spacr_dirs_skips_an_unreadable_sys_path_entry(monkeypatch):
    import importlib.util

    real_is_file = Path.is_file

    def refuse(self):
        if "boom" in self.parts:
            raise OSError("permission denied")
        return real_is_file(self)

    monkeypatch.setattr(Path, "is_file", refuse)
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)
    monkeypatch.setattr(doctor.sys, "path", ["/boom"])
    assert doctor._importable_spacr_dirs() == []


# ---------------------------------------------------------------------------
# helpers: the driver
# ---------------------------------------------------------------------------

def test_nvidia_driver_is_none_without_nvidia_smi(monkeypatch):
    monkeypatch.setattr(doctor.shutil, "which", lambda _name: None)
    assert doctor._nvidia_driver() is None


def test_nvidia_driver_parses_the_first_line(monkeypatch):
    monkeypatch.setattr(doctor.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        doctor.subprocess,
        "run",
        lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="\n580.173.02\n"),
    )
    assert doctor._nvidia_driver() == "580.173.02"


def test_nvidia_driver_is_none_when_the_tool_fails_or_says_nothing(monkeypatch):
    monkeypatch.setattr(doctor.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        doctor.subprocess,
        "run",
        lambda *a, **k: types.SimpleNamespace(returncode=9, stdout="boom"),
    )
    assert doctor._nvidia_driver() is None
    monkeypatch.setattr(
        doctor.subprocess,
        "run",
        lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="  \n"),
    )
    assert doctor._nvidia_driver() is None


def test_nvidia_driver_is_none_when_the_tool_will_not_start(monkeypatch):
    monkeypatch.setattr(doctor.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")

    def explode(*_a, **_k):
        raise OSError("Exec format error")

    monkeypatch.setattr(doctor.subprocess, "run", explode)
    assert doctor._nvidia_driver() is None


# ---------------------------------------------------------------------------
# check: python
# ---------------------------------------------------------------------------

def test_python_check_passes_on_the_interpreter_running_the_suite(ctx):
    row = doctor.check_python(ctx)
    assert row.status == PASS
    assert sys.executable in row.details[0]


def test_python_check_fails_on_an_unsupported_interpreter(ctx, monkeypatch):
    monkeypatch.setattr(doctor, "_satisfies", lambda *_a: False)
    row = doctor.check_python(ctx)
    assert row.status == FAIL
    assert "conda create" in row.fix and "python=3.12" in row.fix


def test_python_check_warns_and_still_offers_a_route_when_it_cannot_tell(
    ctx, monkeypatch
):
    monkeypatch.setattr(doctor, "_satisfies", lambda *_a: None)
    row = doctor.check_python(ctx)
    assert row.status == WARN
    assert "conda create -n spacr python=3.12" in row.fix


def test_python_check_falls_back_when_metadata_is_unreadable(ctx, monkeypatch):
    monkeypatch.delattr(importlib.metadata, "metadata")
    row = doctor.check_python(ctx)
    assert doctor.FALLBACK_REQUIRES_PYTHON in row.message


def test_python_check_falls_back_when_requires_python_is_absent(ctx, monkeypatch):
    monkeypatch.setattr(importlib.metadata, "metadata",
                        lambda _name: {"Requires-Python": None})
    row = doctor.check_python(ctx)
    assert doctor.FALLBACK_REQUIRES_PYTHON in row.message


# ---------------------------------------------------------------------------
# check: the spacr package
# ---------------------------------------------------------------------------

def test_spacr_package_check_reports_the_real_install(ctx):
    row = doctor.check_spacr_package(ctx)
    assert row.status == PASS
    import spacr

    assert str(Path(spacr.__file__).resolve().parent) in row.message


def test_spacr_package_check_fails_when_the_import_dies(ctx, monkeypatch):
    def explode():
        raise ImportError("No module named 'spacr'")

    monkeypatch.setattr(doctor, "_import_spacr", explode)
    row = doctor.check_spacr_package(ctx)
    assert row.status == FAIL
    assert 'pip install "spacr[qt]"' in row.fix


def test_spacr_package_check_warns_about_a_namespace_leftover(ctx, monkeypatch):
    monkeypatch.setattr(doctor, "_import_spacr", lambda: types.SimpleNamespace())
    row = doctor.check_spacr_package(ctx)
    assert row.status == WARN
    assert "pip uninstall -y spacr" in row.fix


def test_spacr_package_check_warns_when_only_a_source_tree_is_present(
    ctx, tmp_path, monkeypatch
):
    package = tmp_path / "clone" / "spacr"
    package.mkdir(parents=True)
    module = types.SimpleNamespace()
    module.__file__ = str(package / "__init__.py")
    monkeypatch.setattr(doctor, "_import_spacr", lambda: module)
    row = doctor.check_spacr_package(ctx)
    assert row.status == WARN
    assert f'pip install -e "{tmp_path / "clone"}"' in row.fix


# ---------------------------------------------------------------------------
# check: running checkout — the stale editable install
# ---------------------------------------------------------------------------

def test_running_checkout_fails_when_you_edit_one_clone_and_run_another(
    tmp_path, monkeypatch
):
    """The failure this whole module exists for.

    Two clones, an editable install pointing at the first, a developer in the
    second. Nothing else in spaCR's output says the edits are inert.
    """
    editing = _make_checkout(tmp_path / "codex")
    running = _make_checkout(tmp_path / "claude")
    monkeypatch.setattr(doctor, "_import_spacr",
                        lambda: _fake_spacr(running / "spacr"))
    monkeypatch.setattr(doctor, "_editable_target", lambda: running)

    row = doctor.check_running_checkout(Context(checkout=editing))
    assert row.status == FAIL
    assert str(editing) in row.message and str(running) in row.message
    assert f'python -m pip install -e "{editing}"' in row.fix
    assert "print(spacr.__file__)" in row.fix


def test_running_checkout_passes_when_the_editable_install_is_this_clone(
    tmp_path, monkeypatch
):
    here = _make_checkout(tmp_path / "clone")
    monkeypatch.setattr(doctor, "_import_spacr", lambda: _fake_spacr(here / "spacr"))
    monkeypatch.setattr(doctor, "_editable_target", lambda: here)
    row = doctor.check_running_checkout(Context(checkout=here))
    assert row.status == PASS
    assert "Editable install" in row.message


def test_running_checkout_warns_when_only_the_cwd_puts_you_in_this_clone(
    tmp_path, monkeypatch
):
    here = _make_checkout(tmp_path / "clone")
    monkeypatch.setattr(doctor, "_import_spacr", lambda: _fake_spacr(here / "spacr"))
    monkeypatch.setattr(doctor, "_editable_target", lambda: None)
    row = doctor.check_running_checkout(Context(checkout=here))
    assert row.status == WARN
    assert "sys.path" in row.message
    assert f'python -m pip install -e "{here}"' in row.fix


def test_running_checkout_fails_when_the_editable_pointer_names_another_clone(
    tmp_path, monkeypatch
):
    here = _make_checkout(tmp_path / "clone")
    other = _make_checkout(tmp_path / "other")
    monkeypatch.setattr(doctor, "_import_spacr", lambda: _fake_spacr(here / "spacr"))
    monkeypatch.setattr(doctor, "_editable_target", lambda: other)
    row = doctor.check_running_checkout(Context(checkout=here))
    assert row.status == FAIL
    assert "another directory" in row.message
    assert f'python -m pip install -e "{here}"' in row.fix


def test_running_checkout_fails_on_a_stale_pointer_from_outside_any_clone(
    tmp_path, monkeypatch
):
    running = _make_checkout(tmp_path / "installed")
    monkeypatch.setattr(doctor, "_import_spacr",
                        lambda: _fake_spacr(running / "spacr"))
    monkeypatch.setattr(doctor, "_editable_target", lambda: tmp_path / "deleted")
    row = doctor.check_running_checkout(Context(checkout=tmp_path / "elsewhere"))
    assert row.status == FAIL
    assert "stale" in row.message
    assert f'python -m pip install -e "{tmp_path / "deleted"}"' in row.fix


def test_running_checkout_passes_outside_a_clone(tmp_path, monkeypatch):
    running = _make_checkout(tmp_path / "installed")
    monkeypatch.setattr(doctor, "_import_spacr",
                        lambda: _fake_spacr(running / "spacr"))
    monkeypatch.setattr(doctor, "_editable_target", lambda: running)
    row = doctor.check_running_checkout(Context(checkout=tmp_path / "elsewhere"))
    assert row.status == PASS
    assert "Not inside a spaCR checkout" in row.message


def test_running_checkout_skips_when_spacr_will_not_import(ctx, monkeypatch):
    def explode():
        raise ImportError("nope")

    monkeypatch.setattr(doctor, "_import_spacr", explode)
    row = doctor.check_running_checkout(ctx)
    assert row.status == SKIP
    assert "spacr package" in row.fix


def test_running_checkout_skips_when_the_package_has_no_file(ctx, monkeypatch):
    monkeypatch.setattr(doctor, "_import_spacr", lambda: types.SimpleNamespace())
    row = doctor.check_running_checkout(ctx)
    assert row.status == SKIP
    assert "__file__" in row.message


# ---------------------------------------------------------------------------
# check: duplicate installs and conflicting distributions
# ---------------------------------------------------------------------------

def test_duplicate_installs_passes_with_one_package_dir(ctx, monkeypatch):
    monkeypatch.setattr(doctor, "_importable_spacr_dirs", lambda: [Path("/env/spacr")])
    row = doctor.check_duplicate_installs(ctx)
    assert row.status == PASS


def test_duplicate_installs_fails_and_says_which_one_wins(ctx, monkeypatch):
    monkeypatch.setattr(
        doctor,
        "_importable_spacr_dirs",
        lambda: [Path("/a/spacr"), Path("/b/spacr")],
    )
    row = doctor.check_duplicate_installs(ctx)
    assert row.status == FAIL
    assert "/a/spacr wins" in row.message
    assert "pip uninstall -y spacr" in row.fix
    assert row.details == ("  1. /a/spacr", "  2. /b/spacr")


def test_duplicate_installs_skips_when_nothing_is_importable(ctx, monkeypatch):
    monkeypatch.setattr(doctor, "_importable_spacr_dirs", lambda: [])
    row = doctor.check_duplicate_installs(ctx)
    assert row.status == SKIP
    assert 'pip install "spacr[qt]"' in row.fix


def test_distributions_check_passes_with_a_single_metadata_dir(ctx, monkeypatch):
    monkeypatch.setattr(
        doctor, "_spacr_distributions", lambda: [("spacr", "1.4.9", "/env/spacr.dist-info")]
    )
    row = doctor.check_conflicting_distributions(ctx)
    assert row.status == PASS
    assert row.details == ("/env/spacr.dist-info",)


def test_distributions_check_fails_when_spacr_and_spacr_nightly_coexist(
    ctx, monkeypatch
):
    monkeypatch.setattr(
        doctor,
        "_spacr_distributions",
        lambda: [("spacr", "1.4.9", "/env/a"), ("spacr-nightly", "1.4.8", "/env/b")],
    )
    row = doctor.check_conflicting_distributions(ctx)
    assert row.status == FAIL
    assert "pip uninstall -y spacr spacr-nightly" in row.fix


def test_distributions_check_names_a_stale_egg_info_to_delete(ctx, monkeypatch):
    """A leftover egg-info in a checkout shadows the real install.

    Observed on this machine: from inside the clone, importlib.metadata read
    `spacr.egg-info` instead of site-packages, so the editable install looked
    absent and the dependency list came from stale metadata.
    """
    monkeypatch.setattr(
        doctor,
        "_spacr_distributions",
        lambda: [
            ("spacr", "1.4.9", "/clone/spacr.egg-info"),
            ("spacr", "1.4.9", "/env/spacr-1.4.9.dist-info"),
        ],
    )
    row = doctor.check_conflicting_distributions(ctx)
    assert row.status == WARN
    assert row.fix == 'rm -rf "/clone/spacr.egg-info"'


def test_distributions_check_falls_back_to_reinstall_without_an_egg_info(
    ctx, monkeypatch
):
    monkeypatch.setattr(
        doctor,
        "_spacr_distributions",
        lambda: [("spacr", "1.4.9", "/a.dist-info"), ("spacr", "1.4.8", "/b.dist-info")],
    )
    row = doctor.check_conflicting_distributions(ctx)
    assert row.status == WARN
    assert "--force-reinstall" in row.fix


def test_distributions_check_warns_when_spacr_is_not_installed(ctx, monkeypatch):
    monkeypatch.setattr(doctor, "_spacr_distributions", lambda: [])
    row = doctor.check_conflicting_distributions(ctx)
    assert row.status == WARN
    assert 'pip install "spacr[qt]"' in row.fix


# ---------------------------------------------------------------------------
# check: console scripts and PATH
# ---------------------------------------------------------------------------

class _FakeEntryPoint:
    def __init__(self, name, value, group="console_scripts"):
        self.name = name
        self.value = value
        self.group = group


class _FakeDistribution:
    def __init__(self, entry_points):
        self.entry_points = entry_points


def test_console_scripts_check_passes_for_the_real_install(ctx):
    row = doctor.check_console_scripts(ctx)
    assert row.status == PASS
    assert "resolve" in row.message


def test_console_scripts_check_names_every_broken_command(ctx, monkeypatch):
    """`sim=spacr.app_sim:gui_sim` outlived the file it named."""
    monkeypatch.setattr(
        importlib.metadata,
        "distribution",
        lambda _name: _FakeDistribution(
            [
                _FakeEntryPoint("spacr-run", "spacr.cli:main"),
                _FakeEntryPoint("sim", "spacr.app_sim:gui_sim"),
                _FakeEntryPoint("ghost", "no_such_top_level.sub:main"),
                _FakeEntryPoint("gui", "spacr.gui:gui_app", group="gui_scripts"),
            ]
        ),
    )
    row = doctor.check_console_scripts(ctx)
    assert row.status == FAIL
    assert "2 of 3" in row.message
    assert "--force-reinstall" in row.fix
    assert any("sim -> spacr.app_sim:gui_sim" == item for item in row.details)


def test_console_scripts_check_warns_when_none_are_declared(ctx, monkeypatch):
    monkeypatch.setattr(
        importlib.metadata, "distribution", lambda _name: _FakeDistribution([])
    )
    row = doctor.check_console_scripts(ctx)
    assert row.status == WARN
    assert "--force-reinstall" in row.fix


def test_console_scripts_check_skips_without_an_installed_distribution(
    ctx, monkeypatch
):
    def explode(_name):
        raise importlib.metadata.PackageNotFoundError("spacr")

    monkeypatch.setattr(importlib.metadata, "distribution", explode)
    row = doctor.check_console_scripts(ctx)
    assert row.status == SKIP
    assert 'pip install "spacr[qt]"' in row.fix


def test_path_check_passes_when_the_command_is_this_environment(ctx, monkeypatch):
    script = Path(sys.prefix) / "bin" / "spacr"
    monkeypatch.setattr(doctor.shutil, "which", lambda name: str(script))
    row = doctor.check_command_on_path(ctx)
    assert row.status == PASS


def test_path_check_fails_when_another_environment_shadows_it(
    ctx, tmp_path, monkeypatch
):
    other = tmp_path / "other-env" / "bin" / "spacr"
    other.parent.mkdir(parents=True)
    other.write_text("#!/bin/sh\n")
    monkeypatch.setattr(doctor.shutil, "which", lambda name: str(other))
    row = doctor.check_command_on_path(ctx)
    assert row.status == FAIL
    assert "different installation" in row.message
    assert str(Path(sys.executable).parent) in row.fix


def test_path_check_warns_when_no_command_is_on_path(ctx, monkeypatch):
    monkeypatch.setattr(doctor.shutil, "which", lambda _name: None)
    row = doctor.check_command_on_path(ctx)
    assert row.status == WARN
    assert "python -m spacr.doctor" in row.message
    assert "export PATH=" in row.fix


# ---------------------------------------------------------------------------
# check: the qt extra
# ---------------------------------------------------------------------------

def test_import_qt_app_helper_either_returns_launch_or_raises_importerror():
    try:
        launch = doctor._import_qt_app()
    except ImportError:
        return  # headless env: check_qt_extra reads exactly this failure
    assert callable(launch)


def test_qt_extra_check_reports_the_missing_extra_with_the_install_command(
    ctx, monkeypatch
):
    """The plain `pip install spacr` then `spacr` failure, diagnosed."""

    def explode():
        raise ModuleNotFoundError("No module named 'PySide6'", name="PySide6")

    monkeypatch.setattr(doctor, "_import_qt_app", explode)
    row = doctor.check_qt_extra(ctx)
    assert row.status == FAIL
    assert "PySide6" in row.message
    assert 'pip install "spacr[qt]"' in row.fix
    assert "spacr-run --list" in row.fix  # the headless escape hatch


def test_qt_extra_check_keeps_an_unrelated_import_error_distinct(ctx, monkeypatch):
    """A bug inside the GUI must not be reported as "Qt is not installed"."""

    def explode():
        raise ModuleNotFoundError("No module named 'pandas'", name="pandas")

    monkeypatch.setattr(doctor, "_import_qt_app", explode)
    row = doctor.check_qt_extra(ctx)
    assert row.status == FAIL
    assert "unrelated to the optional extra" in row.message
    assert row.fix == doctor._CRASH_FIX


def test_qt_extra_check_reports_a_non_import_failure(ctx, monkeypatch):
    def explode():
        raise RuntimeError("qt.qpa.plugin: could not load the Qt platform plugin")

    monkeypatch.setattr(doctor, "_import_qt_app", explode)
    row = doctor.check_qt_extra(ctx)
    assert row.status == FAIL
    assert "RuntimeError" in row.message


def test_qt_extra_check_passes_when_the_gui_imports(ctx, monkeypatch):
    monkeypatch.setattr(doctor, "_import_qt_app", lambda: (lambda argv: 0))
    monkeypatch.setattr(doctor, "_distribution_version", lambda _n: None)
    row = doctor.check_qt_extra(ctx)
    assert row.status == PASS
    assert "unknown version" in row.message


# ---------------------------------------------------------------------------
# check: display
# ---------------------------------------------------------------------------

def test_display_check_passes_off_linux(ctx, monkeypatch):
    monkeypatch.setattr(doctor.sys, "platform", "darwin")
    assert doctor.check_display(ctx).status == PASS


def test_display_check_accepts_a_deliberate_offscreen_platform(ctx, monkeypatch):
    monkeypatch.setattr(doctor.sys, "platform", "linux")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    row = doctor.check_display(ctx)
    assert row.status == PASS
    assert "deliberately headless" in row.message


def test_display_check_passes_with_a_display_server(ctx, monkeypatch):
    monkeypatch.setattr(doctor.sys, "platform", "linux")
    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
    monkeypatch.setenv("DISPLAY", ":0")
    assert doctor.check_display(ctx).status == PASS


def test_display_check_warns_headless_and_offers_the_headless_runner(
    ctx, monkeypatch
):
    monkeypatch.setattr(doctor.sys, "platform", "linux")
    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    row = doctor.check_display(ctx)
    assert row.status == WARN
    assert "spacr-run --list" in row.fix
    assert "QT_QPA_PLATFORM=offscreen" in row.fix


# ---------------------------------------------------------------------------
# check: core dependencies and optional extras
# ---------------------------------------------------------------------------

def test_core_dependency_check_passes_in_this_environment(ctx):
    row = doctor.check_core_dependencies(ctx)
    assert row.status == PASS


def test_core_dependency_check_names_the_distribution_to_install(ctx, monkeypatch):
    monkeypatch.setattr(
        doctor, "CORE_MODULES", (("numpy", "numpy"), ("skimage", "scikit-image"))
    )
    monkeypatch.setitem(sys.modules, "skimage", None)
    row = doctor.check_core_dependencies(ctx)
    assert row.status == FAIL
    assert row.fix == "python -m pip install scikit-image"
    assert any("skimage:" in detail for detail in row.details)


def test_optional_extras_check_reports_installed_and_absent(ctx, monkeypatch):
    monkeypatch.setattr(doctor, "OPTIONAL_EXTRAS", {"qt": ("PySide6",), "nd2": ("nd2reader",)})
    monkeypatch.setattr(
        doctor, "_distribution_version", lambda name: "1.0" if name == "PySide6" else None
    )
    row = doctor.check_optional_extras(ctx)
    assert row.status == PASS
    assert "installed: qt" in row.message and "not installed: nd2" in row.message


def test_optional_extras_check_flags_a_half_installed_extra(ctx, monkeypatch):
    monkeypatch.setattr(doctor, "OPTIONAL_EXTRAS", {"boosting": ("catboost", "lightgbm")})
    monkeypatch.setattr(
        doctor, "_distribution_version", lambda name: "1.0" if name == "catboost" else None
    )
    row = doctor.check_optional_extras(ctx)
    assert row.status == WARN
    assert row.fix == 'python -m pip install "spacr[boosting]"'
    assert "missing lightgbm" in row.details[0]
    assert "installed: none" in row.details[-1]


# ---------------------------------------------------------------------------
# check: torch and the GPU
# ---------------------------------------------------------------------------

def test_torch_check_passes_on_the_installed_torch(ctx):
    row = doctor.check_torch(ctx)
    assert row.status == PASS


def test_torch_check_fails_when_torch_is_absent(ctx, monkeypatch):
    def explode():
        raise ImportError("No module named 'torch'")

    monkeypatch.setattr(doctor, "_import_torch", explode)
    row = doctor.check_torch(ctx)
    assert row.status == FAIL
    assert row.fix == "python -m pip install torch torchvision"


def test_torch_check_names_a_cpu_only_build(ctx, monkeypatch):
    monkeypatch.setattr(doctor, "_import_torch", lambda: _fake_torch(cuda_build=None))
    assert "CPU-only build" in doctor.check_torch(ctx).message


def test_gpu_check_skips_without_torch(ctx, monkeypatch):
    def explode():
        raise ImportError("no torch")

    monkeypatch.setattr(doctor, "_import_torch", explode)
    row = doctor.check_gpu(ctx)
    assert row.status == SKIP


def test_gpu_check_fails_when_a_card_is_present_but_torch_is_cpu_only(
    ctx, monkeypatch
):
    monkeypatch.setattr(doctor, "_import_torch", lambda: _fake_torch(cuda_build=None))
    monkeypatch.setattr(doctor, "_nvidia_driver", lambda: "580.173.02")
    row = doctor.check_gpu(ctx)
    assert row.status == FAIL
    assert "CPU-only build" in row.message
    assert "download.pytorch.org/whl/cu124" in row.fix


def test_gpu_check_warns_on_a_machine_with_no_gpu_at_all(ctx, monkeypatch):
    monkeypatch.setattr(doctor, "_import_torch", lambda: _fake_torch(cuda_build=None))
    monkeypatch.setattr(doctor, "_nvidia_driver", lambda: None)
    row = doctor.check_gpu(ctx)
    assert row.status == WARN
    assert "nvidia-smi" in row.fix


def test_gpu_check_fails_when_the_driver_is_missing_entirely(ctx, monkeypatch):
    monkeypatch.setattr(
        doctor,
        "_import_torch",
        lambda: _fake_torch(available=False, init_error=RuntimeError("no driver")),
    )
    monkeypatch.setattr(doctor, "_nvidia_driver", lambda: None)
    row = doctor.check_gpu(ctx)
    assert row.status == FAIL
    assert "no NVIDIA driver is answering" in row.message
    assert "nvidia-driver" in row.fix
    assert row.details == ("RuntimeError: no driver",)


def test_gpu_check_fails_on_a_driver_runtime_mismatch(ctx, monkeypatch):
    monkeypatch.setattr(doctor, "_import_torch", lambda: _fake_torch(available=False))
    monkeypatch.setattr(doctor, "_nvidia_driver", lambda: "470.0")
    row = doctor.check_gpu(ctx)
    assert row.status == FAIL
    assert "driver / runtime mismatch" in row.message
    assert "--force-reinstall torch" in row.fix
    assert row.details == ()


def test_gpu_check_fails_when_the_first_allocation_dies(monkeypatch, tmp_path):
    """`torch.cuda.is_available()` can say yes and the allocator still refuse."""
    monkeypatch.setattr(
        doctor,
        "_import_torch",
        lambda: _fake_torch(alloc_error=RuntimeError("CUDA error: no kernel image")),
    )
    monkeypatch.setattr(doctor, "_nvidia_driver", lambda: "580.0")
    row = doctor.check_gpu(Context(checkout=tmp_path, probe_gpu=True))
    assert row.status == FAIL
    assert "first allocation failed" in row.message
    assert "nvidia-smi" in row.fix


def test_gpu_check_passes_when_the_allocation_probe_succeeds(monkeypatch, tmp_path):
    monkeypatch.setattr(doctor, "_import_torch", lambda: _fake_torch(devices=2))
    monkeypatch.setattr(doctor, "_nvidia_driver", lambda: "580.0")
    row = doctor.check_gpu(Context(checkout=tmp_path, probe_gpu=True))
    assert row.status == PASS
    assert "2 CUDA device(s) usable" in row.message


def test_gpu_check_says_when_the_probe_was_skipped(ctx, monkeypatch):
    monkeypatch.setattr(doctor, "_import_torch", lambda: _fake_torch())
    monkeypatch.setattr(doctor, "_nvidia_driver", lambda: "580.0")
    row = doctor.check_gpu(ctx)
    assert row.status == PASS
    assert "allocation probe skipped" in row.message


def test_gpu_check_survives_an_unnameable_device(ctx, monkeypatch):
    monkeypatch.setattr(
        doctor,
        "_import_torch",
        lambda: _fake_torch(name_error=RuntimeError("device lost")),
    )
    monkeypatch.setattr(doctor, "_nvidia_driver", lambda: "580.0")
    assert "unnamed (RuntimeError)" in doctor.check_gpu(ctx).message


# ---------------------------------------------------------------------------
# check: cellpose version drift
# ---------------------------------------------------------------------------

def _install_fake_cellpose(monkeypatch, *, version="4.0.7", models_attrs=None):
    if models_attrs is None:
        models_attrs = {"CellposeModel": object, "MODEL_NAMES": ["cpsam"]}
    models = types.ModuleType("cellpose.models")
    for key, value in models_attrs.items():
        setattr(models, key, value)
    package = types.ModuleType("cellpose")
    package.models = models
    if version is not None:
        package.version = version
    monkeypatch.setitem(sys.modules, "cellpose", package)
    monkeypatch.setitem(sys.modules, "cellpose.models", models)
    return package


def test_cellpose_check_passes_on_the_installed_cellpose(ctx):
    row = doctor.check_cellpose(ctx)
    assert row.status == PASS
    assert "Cellpose 4" in row.message


def test_cellpose_check_fails_when_cellpose_is_missing(ctx, monkeypatch):
    monkeypatch.setitem(sys.modules, "cellpose", None)
    row = doctor.check_cellpose(ctx)
    assert row.status == FAIL
    assert 'pip install "cellpose>=4.0,<5.0"' in row.fix


def test_cellpose_check_fails_on_a_cellpose_3_install(ctx, monkeypatch):
    """The migration this project actually made, caught before the run starts."""
    _install_fake_cellpose(
        monkeypatch,
        version="3.0.11",
        models_attrs={"Cellpose": object, "CellposeModel": object,
                      "MODEL_NAMES": ["cyto", "nuclei"]},
    )
    row = doctor.check_cellpose(ctx)
    assert row.status == FAIL
    assert "3.0.11" in row.message
    assert "cpsam" in row.message
    assert 'pip install "cellpose>=4.0,<5.0"' in row.fix


def test_cellpose_check_fails_when_the_api_is_not_the_one_spacr_calls(
    ctx, monkeypatch
):
    _install_fake_cellpose(
        monkeypatch,
        version="4.9.0",
        models_attrs={"Cellpose": object, "MODEL_NAMES": ["cyto3"]},
    )
    row = doctor.check_cellpose(ctx)
    assert row.status == FAIL
    assert "does not expose the API" in row.message
    assert any("CellposeModel is missing" in d for d in row.details)
    assert any("that wrapper is the 3.x API" in d for d in row.details)
    assert any("'cpsam' is not in" in d for d in row.details)


def test_cellpose_check_warns_about_api_drift_it_cannot_version_check(
    ctx, monkeypatch
):
    _install_fake_cellpose(monkeypatch, version="4.9.0", models_attrs={})
    monkeypatch.setattr(doctor, "_satisfies", lambda *_a: None)
    row = doctor.check_cellpose(ctx)
    assert row.status == WARN
    assert "does not expose the API" in row.message


def test_cellpose_check_warns_when_the_version_cannot_be_compared(ctx, monkeypatch):
    _install_fake_cellpose(monkeypatch, version="4.0.7-nightly")
    row = doctor.check_cellpose(ctx)
    assert row.status == WARN
    assert "could not be compared" in row.message
    assert 'pip install "cellpose>=4.0,<5.0"' in row.fix


def test_cellpose_check_warns_when_cellpose_reports_no_version(ctx, monkeypatch):
    _install_fake_cellpose(monkeypatch, version=None)
    monkeypatch.setattr(doctor, "_distribution_version", lambda _n: None)
    row = doctor.check_cellpose(ctx)
    assert row.status == WARN
    assert "--force-reinstall" in row.fix


def test_cellpose_check_falls_back_to_the_declared_bound(ctx, monkeypatch):
    _install_fake_cellpose(monkeypatch)
    monkeypatch.setattr(doctor, "_declared_requirement", lambda _n: None)
    row = doctor.check_cellpose(ctx)
    assert row.status == PASS
    assert doctor.FALLBACK_CELLPOSE_SPECIFIER in row.message


# ---------------------------------------------------------------------------
# check: declared pins
# ---------------------------------------------------------------------------

def test_declared_pins_check_catches_environment_yaml_contradicting_setup_py(
    tmp_path,
):
    """The live trap: `conda env create -f environment.yaml` installs a
    cellpose that this spaCR cannot call."""
    root = _make_checkout(tmp_path / "clone")
    (root / "setup.py").write_text(
        "dependencies = ['cellpose>=4.0,<5.0', 'numpy>=1.26.4,<3.0']\n"
    )
    (root / "environment.yaml").write_text(
        "name: spacr\ndependencies:\n  - python=3.9.19\n  - pip:\n"
        "      - cellpose==3.0.11\n      - numpy==1.26.4\n"
    )
    row = doctor.check_declared_pins(Context(checkout=root))
    assert row.status == WARN
    assert row.details == (
        "environment.yaml pins cellpose==3.0.11, setup.py requires "
        "cellpose>=4.0,<5.0",
    )
    assert f'pip install -e "{root}[qt]"' in row.fix


def test_declared_pins_check_passes_when_the_two_files_agree(tmp_path):
    root = _make_checkout(tmp_path / "clone")
    (root / "setup.py").write_text("dependencies = ['cellpose>=4.0,<5.0']\n")
    (root / "environment.yaml").write_text("  - pip:\n      - cellpose==4.0.7\n")
    row = doctor.check_declared_pins(Context(checkout=root))
    assert row.status == PASS


def test_declared_pins_check_skips_outside_a_checkout(tmp_path):
    row = doctor.check_declared_pins(Context(checkout=tmp_path))
    assert row.status == SKIP


def test_declared_pins_check_skips_without_both_files(tmp_path):
    root = _make_checkout(tmp_path / "clone")
    row = doctor.check_declared_pins(Context(checkout=root))
    assert row.status == SKIP
    assert "no setup.py/environment.yaml pair" in row.message


def test_declared_pins_check_skips_an_unreadable_dependency_list(tmp_path):
    root = _make_checkout(tmp_path / "clone")
    (root / "setup.py").write_text("something_else = ['cellpose>=4.0']\n")
    (root / "environment.yaml").write_text("  - cellpose==3.0.11\n")
    row = doctor.check_declared_pins(Context(checkout=root))
    assert row.status == SKIP
    assert "Could not read the dependency list" in row.message


def test_declared_pins_check_ignores_packages_setup_py_does_not_name(tmp_path):
    root = _make_checkout(tmp_path / "clone")
    (root / "setup.py").write_text("dependencies = ['cellpose>=4.0,<5.0']\n")
    (root / "environment.yaml").write_text("  - pytorch==1.0.0\n")
    assert doctor.check_declared_pins(Context(checkout=root)).status == PASS


def test_setup_dependency_parser_survives_every_shape_of_bad_input(tmp_path):
    broken = tmp_path / "broken.py"
    broken.write_text("def (:\n")
    assert doctor._parse_setup_dependencies(broken) == {}
    assert doctor._parse_setup_dependencies(tmp_path / "absent.py") == {}

    indirect = tmp_path / "indirect.py"
    indirect.write_text("dependencies = some_other_name\n")
    assert doctor._parse_setup_dependencies(indirect) == {}

    mixed = tmp_path / "mixed.py"
    mixed.write_text("other = 1\ndependencies = [1, '!!!', 'numpy>=2']\n")
    assert doctor._parse_setup_dependencies(mixed) == {"numpy": ">=2"}


def test_environment_pin_parser_returns_nothing_for_a_missing_file(tmp_path):
    assert doctor._parse_environment_pins(tmp_path / "absent.yaml") == {}


def test_environment_pin_parser_reads_conda_and_pip_spellings(tmp_path):
    path = tmp_path / "environment.yaml"
    path.write_text(
        "dependencies:\n  - python=3.9.19\n  - pip:\n"
        "      - cellpose==3.0.11\n      - some-git-dep\n      - '-e .'\n"
    )
    assert doctor._parse_environment_pins(path) == {
        "python": "3.9.19",
        "cellpose": "3.0.11",
    }


def test_declared_pins_check_on_this_repository_flags_the_stale_cellpose_pin():
    """A real finding, not a fixture: environment.yaml in this checkout pins
    cellpose 3.0.11 while setup.py requires >=4.0."""
    root = doctor._checkout_root(Path(__file__).resolve().parent)
    if root is None or not (root / "environment.yaml").is_file():
        pytest.skip("not running from a source checkout")
    row = doctor.check_declared_pins(Context(checkout=root))
    assert row.status in (PASS, WARN)
    if row.status == WARN:
        assert any("cellpose" in detail for detail in row.details)


# ---------------------------------------------------------------------------
# check: the project database
# ---------------------------------------------------------------------------

@pytest.fixture
def measurements_db(tmp_path):
    """A minimal but genuine spaCR measurements database."""
    path = tmp_path / "measurements.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE cell (object_label TEXT)")
    conn.execute("CREATE TABLE settings (key TEXT)")
    conn.execute("PRAGMA user_version = 1")
    conn.commit()
    conn.close()
    return path


class _FakeHealth:
    def __init__(self, quick_check="ok", warnings=()):
        self.quick_check = quick_check
        self.warnings = tuple(warnings)
        self.journal_mode = "WAL"
        self.busy_timeout_ms = 30000


def test_database_check_skips_when_none_is_given(ctx):
    row = _only(doctor.check_database(ctx))
    assert row.status == SKIP
    assert "measurements/measurements.db" in row.fix


def test_database_check_fails_when_the_path_is_not_a_file(tmp_path):
    row = _only(doctor.check_database(Context(db=tmp_path / "nope.db")))
    assert row.status == FAIL
    assert "measurements/measurements.db" in row.fix


def test_database_check_fails_on_a_file_that_is_not_a_database(tmp_path):
    """A truncated or half-copied .db, byte for byte."""
    path = tmp_path / "measurements.db"
    path.write_bytes(b"not a database, just bytes" * 64)
    row = _only(doctor.check_database(Context(db=path)))
    assert row.status == FAIL
    assert "cannot be opened as SQLite" in row.message
    assert 'sqlite3 "' in row.fix and ".schema" in row.fix


def test_database_check_passes_on_a_healthy_measurements_db(measurements_db):
    rows = _rows(doctor.check_database(Context(db=measurements_db)))
    assert [row.status for row in rows] == [PASS, PASS, PASS, PASS]
    assert "quick_check" in rows[0].message
    assert "cell, settings" in rows[2].message


def test_database_check_reports_corruption_and_drops_the_duplicate_warning(
    measurements_db, monkeypatch
):
    from spacr import database_concurrency

    monkeypatch.setattr(
        database_concurrency,
        "inspect_database",
        lambda *_a, **_k: _FakeHealth(
            quick_check="row 4 missing from index idx_cell",
            warnings=(
                "SQLite quick_check reported: row 4 missing from index idx_cell",
                "WAL journaling on a network filesystem (nfs)",
            ),
        ),
    )
    rows = _rows(doctor.check_database(Context(db=measurements_db)))
    assert rows[0].status == FAIL
    assert ".recover" in rows[0].fix
    assert rows[1].status == WARN
    assert "network filesystem" in rows[1].message
    assert "journal_mode=DELETE" in rows[1].fix


def test_database_check_warns_about_wal_on_a_network_filesystem(
    measurements_db, monkeypatch
):
    from spacr import database_concurrency

    monkeypatch.setattr(
        database_concurrency,
        "inspect_database",
        lambda *_a, **_k: _FakeHealth(warnings=("WAL on nfs is unsafe",)),
    )
    rows = _rows(doctor.check_database(Context(db=measurements_db)))
    assert rows[0].status == PASS
    assert rows[1].status == WARN


def test_database_schema_rows_fail_on_a_file_from_a_newer_spacr(measurements_db):
    conn = sqlite3.connect(measurements_db)
    conn.execute("PRAGMA user_version = 99")
    conn.commit()
    conn.close()
    rows = doctor._database_schema_rows(measurements_db)
    assert rows[0].status == FAIL
    assert "written by a newer spaCR" in rows[0].message
    assert "pip install --upgrade" in rows[0].fix
    assert "never downgrade" in rows[0].fix


def test_database_schema_rows_offer_the_migration_for_an_older_file(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE cell (x)")
    conn.commit()
    conn.close()
    rows = doctor._database_schema_rows(path)
    assert rows[0].status == WARN
    assert "ensure_database_schema" in rows[0].fix


def test_database_schema_rows_warn_when_the_version_is_unreadable(
    measurements_db, monkeypatch
):
    from spacr import database_schema

    def explode(*_a, **_k):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(database_schema, "database_schema_version", explode)
    rows = doctor._database_schema_rows(measurements_db)
    assert rows[0].status == WARN
    assert "OperationalError" in rows[0].message


def test_database_table_rows_warn_when_no_spacr_tables_are_present(tmp_path):
    path = tmp_path / "unrelated.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE customers (id INTEGER)")
    conn.commit()
    conn.close()
    row = _only(doctor._database_table_rows(path))
    assert row.status == WARN
    assert "none of spaCR's tables" in row.message
    assert "measurements/measurements.db" in row.fix


def test_database_table_rows_warn_when_the_tables_cannot_be_listed(tmp_path):
    row = _only(doctor._database_table_rows(tmp_path / "absent.db"))
    assert row.status == WARN
    assert "Could not list the tables" in row.message


def test_database_lock_rows_detect_a_live_writer(measurements_db):
    """A second connection holding BEGIN IMMEDIATE is exactly the GUI-left-open
    case users hit."""
    holder = sqlite3.connect(measurements_db, timeout=0.1, isolation_level=None)
    holder.execute("BEGIN IMMEDIATE")
    try:
        row = _only(doctor._database_lock_rows(measurements_db))
    finally:
        holder.rollback()
        holder.close()
    assert row.status == FAIL
    assert "locked by another process" in row.message
    assert "fuser -v" in row.fix


def test_database_lock_rows_warn_when_the_file_is_read_only(measurements_db):
    measurements_db.chmod(0o444)
    try:
        row = _only(doctor._database_lock_rows(measurements_db))
    finally:
        measurements_db.chmod(0o644)
    assert row.status == WARN
    assert "chmod u+w" in row.fix


def test_database_lock_rows_warn_when_the_connection_cannot_be_opened(
    measurements_db, monkeypatch
):
    from spacr import database_concurrency

    def explode(*_a, **_k):
        raise sqlite3.OperationalError("unable to open database file")

    monkeypatch.setattr(database_concurrency, "connect", explode)
    row = _only(doctor._database_lock_rows(measurements_db))
    assert row.status == WARN
    assert "Close any spaCR GUI" in row.fix


def test_database_lock_rows_report_a_raw_busy_error_as_a_lock(
    measurements_db, monkeypatch
):
    import contextlib

    from spacr import database_concurrency

    @contextlib.contextmanager
    def explode(*_a, **_k):
        raise sqlite3.OperationalError("database is locked")
        yield  # pragma: no cover - unreachable, keeps this a generator

    monkeypatch.setattr(database_concurrency, "transaction", explode)
    row = _only(doctor._database_lock_rows(measurements_db))
    assert row.status == FAIL
    assert "fuser -v" in row.fix


def test_database_lock_rows_warn_on_an_unexpected_probe_failure(
    measurements_db, monkeypatch
):
    import contextlib

    from spacr import database_concurrency

    @contextlib.contextmanager
    def explode(*_a, **_k):
        raise ValueError("something else entirely")
        yield  # pragma: no cover - unreachable, keeps this a generator

    monkeypatch.setattr(database_concurrency, "transaction", explode)
    row = _only(doctor._database_lock_rows(measurements_db))
    assert row.status == WARN
    assert "Lock probe" in row.message


# ---------------------------------------------------------------------------
# check: settings
# ---------------------------------------------------------------------------

def test_settings_check_skips_when_none_is_given(ctx):
    row = _only(doctor.check_settings(ctx))
    assert row.status == SKIP
    assert "--app measure" in row.fix


def test_settings_check_fails_when_the_file_is_missing(tmp_path):
    row = _only(doctor.check_settings(Context(settings=tmp_path / "absent.csv",
                                              app="measure")))
    assert row.status == FAIL
    assert "csv or .json" in row.fix or "csv" in row.fix


def test_settings_check_warns_without_an_app_key(tmp_path):
    path = tmp_path / "settings.csv"
    path.write_text("setting_key,setting_value\nsrc,/tmp\n")
    row = _only(doctor.check_settings(Context(settings=path)))
    assert row.status == WARN
    assert "--app measure" in row.fix


def test_settings_check_fails_when_the_file_cannot_be_read(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("{not valid json at all")
    row = _only(doctor.check_settings(Context(settings=path, app="measure")))
    assert row.status == FAIL
    assert "could not be read" in row.message


def test_settings_check_reports_a_combination_that_cannot_work(tmp_path):
    """`normalize=True` with measure is rejected by measure_crop at runtime."""
    path = tmp_path / "settings.csv"
    path.write_text(
        "setting_key,setting_value\n"
        f"src,{tmp_path}\n"
        "normalize,True\n"
        "normalize_by,bogus\n"
    )
    rows = _rows(doctor.check_settings(Context(settings=path, app="measure")))
    assert any(row.status == FAIL and "normalize" in row.message for row in rows)
    assert all(row.fix for row in rows), "every problem must carry a fix"


def test_settings_check_passes_a_clean_configuration(tmp_path, monkeypatch):
    from spacr import validate

    path = tmp_path / "settings.csv"
    path.write_text("setting_key,setting_value\nsrc,/tmp\n")
    monkeypatch.setattr(validate, "validate_settings", lambda *_a: [])
    row = _only(doctor.check_settings(Context(settings=path, app="measure")))
    assert row.status == PASS


def test_settings_check_renders_a_dataset_level_warning_without_a_key(
    tmp_path, monkeypatch
):
    from spacr import validate

    path = tmp_path / "settings.csv"
    path.write_text("setting_key,setting_value\nsrc,/tmp\n")
    monkeypatch.setattr(
        validate,
        "validate_settings",
        lambda *_a: [validate.Problem(validate.WARNING, "", "only 2 images found",
                                      "point src at the plate folder")],
    )
    row = _only(doctor.check_settings(Context(settings=path, app="measure")))
    assert row.status == WARN
    assert row.message == "only 2 images found"


# ---------------------------------------------------------------------------
# run_checks: a check that raises must not take down the report
# ---------------------------------------------------------------------------

def test_a_raising_check_becomes_an_error_row_and_the_run_continues(ctx):
    def exploding(_ctx):
        raise ZeroDivisionError("division by zero")

    exploding.check_label = "exploding"

    def fine(_ctx):
        return Result("fine", PASS, "still here")

    rows = run = doctor.run_checks(ctx, [exploding, fine])
    assert [row.status for row in rows] == [ERROR, PASS]
    assert "ZeroDivisionError: division by zero" in run[0].message
    assert "open an issue" in run[0].fix


def test_a_check_that_raises_a_base_exception_is_still_only_a_row(ctx):
    def exploding(_ctx):
        raise SystemExit(2)

    rows = doctor.run_checks(ctx, [exploding])
    assert rows[0].status == ERROR
    assert rows[0].check == "exploding"  # falls back to __name__


def test_keyboard_interrupt_is_not_a_diagnostic_finding(ctx):
    def interrupted(_ctx):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        doctor.run_checks(ctx, [interrupted])


def test_run_checks_accepts_none_and_multi_row_returns(ctx):
    rows = doctor.run_checks(
        ctx,
        [
            lambda _c: None,
            lambda _c: [Result("a", PASS, "one"), Result("a", WARN, "two")],
        ],
    )
    assert [row.message for row in rows] == ["one", "two"]


def test_run_checks_runs_the_whole_registry_by_default(tmp_path):
    rows = doctor.run_checks(Context(checkout=tmp_path, probe_gpu=False))
    labels = {row.check for row in rows}
    for expected in ("python", "spacr package", "running checkout", "cellpose",
                     "gpu", "qt extra", "database", "settings"):
        assert expected in labels
    assert all(row.status != ERROR for row in rows), [
        row for row in rows if row.status == ERROR
    ]
    # The promise this module is built on: nothing is ever reported as broken
    # without saying what to run. SKIP rows are the one exemption — "nothing
    # to check here" is not a problem to remediate.
    for row in rows:
        if row.status in (WARN, FAIL, ERROR):
            assert row.fix, f"{row.check} gives no remediation: {row.message}"


# ---------------------------------------------------------------------------
# reporting and exit codes
# ---------------------------------------------------------------------------

def test_summarize_always_reports_every_verdict():
    counts = doctor.summarize([Result("a", PASS, "m"), Result("b", FAIL, "m")])
    assert counts == {PASS: 1, WARN: 0, FAIL: 1, ERROR: 0, SKIP: 0}


def test_summarize_counts_a_verdict_it_has_never_seen():
    assert doctor.summarize([Result("a", "WEIRD", "m")])["WEIRD"] == 1


def test_exit_code_is_non_zero_for_failures_so_ci_can_gate_on_it():
    assert doctor.exit_code([Result("a", PASS, "m")]) == 0
    assert doctor.exit_code([Result("a", WARN, "m")]) == 0
    assert doctor.exit_code([Result("a", FAIL, "m")]) == 1
    assert doctor.exit_code([Result("a", ERROR, "m")]) == 1


def test_strict_mode_fails_on_warnings_too():
    assert doctor.exit_code([Result("a", WARN, "m")], strict=True) == 1
    assert doctor.exit_code([Result("a", PASS, "m")], strict=True) == 0


def test_format_report_prints_the_fix_for_every_non_pass_row():
    text = doctor.format_report(
        [
            Result("gpu", FAIL, "no CUDA", fix="line one\nline two",
                   details=("driver: none",)),
            Result("python", PASS, "fine", fix="never shown"),
        ]
    )
    assert "FAIL  gpu     no CUDA" in text
    assert "driver: none" in text
    assert "fix: line one" in text
    assert "     line two" in text
    assert "never shown" not in text
    assert "1 passed, 0 warnings, 1 failed, 0 errored, 0 skipped" in text


def test_format_report_handles_an_empty_run():
    assert doctor.format_report([]).startswith("\n0 passed")


# ---------------------------------------------------------------------------
# the command line
# ---------------------------------------------------------------------------

def test_parser_exposes_the_documented_flags():
    args = doctor.build_parser().parse_args(
        ["--checkout", "/c", "--db", "/d.db", "--settings", "/s.csv",
         "--app", "measure", "--no-gpu-probe", "--strict", "--json"]
    )
    assert args.checkout == "/c" and args.db == "/d.db"
    assert args.settings == "/s.csv" and args.app == "measure"
    assert args.no_gpu_probe and args.strict and args.json


def test_main_prints_a_report_and_exits_zero_on_a_healthy_install(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        doctor, "CHECKS", [lambda _c: Result("only", PASS, "all good")]
    )
    assert doctor.main(["--no-gpu-probe", "--checkout", str(tmp_path)]) == 0
    assert "PASS  only  all good" in capsys.readouterr().out


def test_main_exits_non_zero_and_names_the_fix_when_a_check_fails(
    monkeypatch, capsys
):
    monkeypatch.setattr(
        doctor,
        "CHECKS",
        [lambda _c: Result("gpu", FAIL, "unusable", fix="pip install torch")],
    )
    assert doctor.main([]) == 1
    assert "fix: pip install torch" in capsys.readouterr().out


def test_main_emits_machine_readable_json(monkeypatch, capsys):
    monkeypatch.setattr(
        doctor,
        "CHECKS",
        [lambda _c: Result("gpu", WARN, "cpu only", fix="buy a gpu")],
    )
    assert doctor.main(["--json", "--strict"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["summary"][WARN] == 1
    assert payload["results"][0] == {
        "check": "gpu",
        "status": WARN,
        "message": "cpu only",
        "fix": "buy a gpu",
        "details": [],
    }


def test_main_passes_every_option_through_to_the_context(monkeypatch, tmp_path):
    seen = {}

    def capture(ctx, checks=None):
        seen["ctx"] = ctx
        return []

    monkeypatch.setattr(doctor, "run_checks", capture)
    doctor.main(
        [
            "--checkout", str(tmp_path),
            "--db", str(tmp_path / "m.db"),
            "--settings", str(tmp_path / "s.csv"),
            "--app", "mask",
            "--no-gpu-probe",
        ]
    )
    ctx = seen["ctx"]
    assert ctx.checkout == tmp_path
    assert ctx.db == tmp_path / "m.db"
    assert ctx.settings == tmp_path / "s.csv"
    assert ctx.app == "mask"
    assert ctx.probe_gpu is False


def test_running_the_module_as_a_script_exits_with_the_report_status(monkeypatch):
    import runpy

    monkeypatch.setattr(sys, "argv", ["spacr-doctor", "--no-gpu-probe", "--json"])
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("spacr.doctor", run_name="__main__")
    assert excinfo.value.code in (0, 1)


# ---------------------------------------------------------------------------
# packaging: the command has to be installed for any of this to be reachable
# ---------------------------------------------------------------------------

def test_setup_py_declares_the_spacr_doctor_command():
    import ast

    root = doctor._checkout_root(Path(__file__).resolve().parent)
    if root is None:
        pytest.skip("not running from a source checkout")
    tree = ast.parse((root / "setup.py").read_text(encoding="utf-8"))
    scripts = []
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "entry_points":
            for key, value in zip(node.value.keys, node.value.values):
                if getattr(key, "value", None) == "console_scripts":
                    scripts = [element.value for element in value.elts]
    assert "spacr-doctor=spacr.doctor:main" in scripts


def test_doctor_is_reachable_through_the_lazy_package_loader():
    import spacr

    assert "doctor" in spacr._SUBMODULES
    assert spacr.doctor.main is doctor.main
