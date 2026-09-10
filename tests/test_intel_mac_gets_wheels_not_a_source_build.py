"""An Intel Mac must not be sent to a source build for llvmlite.

Reported from a clean `pip install -e .` on an iMac, minutes after the
packaged installer had SUCCEEDED on the same machine::

    FileNotFoundError: llvmlite needs CMake tools to build.
    ERROR: Failed building wheel for llvmlite

llvmlite 0.46+ publishes no macOS x86_64 wheel and numba 0.63+ requires
that line, so without a ceiling pip takes the newest of each, finds no
wheel, and falls back to a build that needs CMake.

`install_spacr_unix.sh` already applied exactly this pair as an
architecture-specific resolver guard. Declaring it only there is what
made the shipped installer work and the git install fail: the same
knowledge written in one of the two places that need it.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

from packaging.requirements import Requirement

ROOT = pathlib.Path(__file__).resolve().parents[1]

INTEL_MAC = {"sys_platform": "darwin", "platform_machine": "x86_64",
             "python_version": "3.12"}
APPLE_SILICON = {"sys_platform": "darwin", "platform_machine": "arm64",
                 "python_version": "3.12"}
LINUX = {"sys_platform": "linux", "platform_machine": "x86_64",
         "python_version": "3.12"}
WINDOWS = {"sys_platform": "win32", "platform_machine": "AMD64",
           "python_version": "3.12"}


def _requirements(name: str):
    """Every declared requirement for ``name``, parsed.

    READ FROM THE AST, not by pairing quotes. A regex over ``'([^']+)'``
    is offset by every apostrophe in a comment -- and setup.py is mostly
    comment -- so it silently found nothing at all and the whole file
    passed vacuously until one assertion happened to notice.
    """
    tree = ast.parse((ROOT / "setup.py").read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            continue
        raw = node.value.strip()
        if not raw.lower().startswith(name):
            continue
        try:
            requirement = Requirement(raw)
        except Exception:                                    # noqa: BLE001
            continue
        if requirement.name.lower() == name:
            found.append(requirement)
    return found


def _for(name: str, environment: dict):
    """The requirement that applies to one environment. Exactly one must."""
    applies = [r for r in _requirements(name)
               if r.marker is None or r.marker.evaluate(environment)]
    assert len(applies) == 1, (
        f"{len(applies)} {name} requirements apply to {environment}: "
        f"{[str(r) for r in applies]}. Two that both apply is a conflict "
        f"the resolver reports as unsatisfiable; none is an undeclared "
        f"dependency.")
    return applies[0]


class TestTheIntelMacCeiling:

    @pytest.mark.parametrize("name,ceiling", [("llvmlite", "0.46"),
                                              ("numba", "0.63")])
    def test_the_version_with_no_intel_wheel_is_excluded(self, name, ceiling):
        """THE DEFECT. Without this the resolver reaches a source build
        and stops on a missing `cmake` -- on a machine where the packaged
        installer works."""
        requirement = _for(name, INTEL_MAC)

        assert not requirement.specifier.contains(ceiling), (
            f"{name} {ceiling} is admitted on an Intel Mac, and it has no "
            f"wheel there -- pip will try to build it from source")

    @pytest.mark.parametrize("name,version", [("llvmlite", "0.45.1"),
                                              ("numba", "0.62.1")])
    def test_the_newest_version_that_does_have_one_is_admitted(self, name,
                                                               version):
        """The ceiling must not be so low that it excludes the wheels
        that DO exist -- these two are what the installer resolved to on
        the reporting machine."""
        assert _for(name, INTEL_MAC).specifier.contains(version), (
            f"{name} {version} has an Intel Mac wheel and is excluded")


class TestEveryOtherPlatformIsUnaffected:

    @pytest.mark.parametrize("environment,label", [
        (APPLE_SILICON, "Apple silicon"), (LINUX, "Linux"),
        (WINDOWS, "Windows")])
    @pytest.mark.parametrize("name,version", [("llvmlite", "0.49.0"),
                                              ("numba", "0.67.0")])
    def test_the_newest_release_is_still_allowed(self, environment, label,
                                                 name, version):
        """A ceiling for one architecture must not become a ceiling for
        every architecture. These are the versions this Linux box has
        installed today."""
        assert _for(name, environment).specifier.contains(version), (
            f"{name} {version} is excluded on {label}, which has a wheel "
            f"for it")

    @pytest.mark.parametrize("environment", [APPLE_SILICON, LINUX, WINDOWS,
                                             INTEL_MAC])
    @pytest.mark.parametrize("name", ["llvmlite", "numba"])
    def test_exactly_one_requirement_applies_everywhere(self, name,
                                                        environment):
        """Two that both apply is a conflict the resolver reports as
        unsatisfiable; none is an undeclared dependency that shows up as
        an ImportError at runtime."""
        _for(name, environment)


class TestTheInstallerAndThePackageAgree:

    def test_the_shell_guard_and_setup_py_carry_the_same_pair(self):
        """The bug was the two disagreeing. Read from the installer
        rather than repeated, so they cannot drift apart again."""
        script = (ROOT / "packaging" / "online"
                  / "install_spacr_unix.sh").read_text(encoding="utf-8")

        assert 'platform == "macos" && "$(uname -m)" == "x86_64"' in script \
            or 'PLATFORM" == "macos"' in script, (
            "the installer no longer special-cases Intel macOS")

        for guard in ('"numba>=0.60,<0.63"', '"llvmlite>=0.43,<0.46"'):
            assert guard in script, (
                f"the installer no longer carries {guard}; if the ceiling "
                f"moved, setup.py has to move with it")

        for name, bound in (("numba", "0.63"), ("llvmlite", "0.46")):
            assert not _for(name, INTEL_MAC).specifier.contains(bound)

    def test_the_reason_is_written_down_where_it_is_declared(self):
        """A bare version ceiling with no reason is one the next reader
        raises to 'get the newest', which is exactly how this returns."""
        text = (ROOT / "setup.py").read_text(encoding="utf-8")

        assert "no macOS x86_64 wheel" in text
        assert "cmake" in text.lower()
