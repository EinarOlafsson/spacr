"""The version module resolves from metadata and stays torch-free on import.

The module body is executed here from its own source file under a private
module name. A plain ``import spacr.version`` is a no-op after the package
has been imported once, so the statements that compute ``__version__`` and
define the lazy ``__getattr__`` would never run inside a test at all.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_fresh(name="spacr_version_under_test"):
    """Execute spacr/version.py again under a private module name."""
    import spacr.version as installed

    spec = importlib.util.spec_from_file_location(
        name, Path(installed.__file__))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_fresh_import_computes_version_from_the_resolver():
    """``__version__`` is whatever ``get_version()`` answers, not a literal."""
    module = _load_fresh()

    assert module.__version__ == module.get_version()
    assert module.__version__ != ""


def test_importing_the_module_does_not_import_torch(monkeypatch):
    """The lazy ``__getattr__`` exists so ``import spacr.version`` is cheap.

    Executing the module body with ``torch`` removed from ``sys.modules``
    and blocked from being imported must still succeed; if the body called
    ``format_version_info()`` the block would raise.
    """
    blocked = []

    class _Blocker:
        def find_module(self, fullname, path=None):
            return None

        def find_spec(self, fullname, path=None, target=None):
            if fullname == "torch" or fullname.startswith("torch."):
                blocked.append(fullname)
                raise ImportError("torch is blocked for this test")
            return None

    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr(sys, "meta_path", [_Blocker(), *sys.meta_path])

    module = _load_fresh("spacr_version_no_torch")

    assert blocked == []
    assert module.get_torch_version() == "not available"
    assert blocked == ["torch"]


def test_an_unknown_distribution_reports_unknown(monkeypatch):
    """Every candidate distribution missing resolves to ``"unknown"``."""
    import spacr.version as version
    from importlib.metadata import PackageNotFoundError

    def _absent(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr(version, "package_version", _absent)

    assert version.get_version() == "unknown"


def test_the_fallback_distribution_is_consulted_second(monkeypatch):
    """``spacr-nightly`` answers when the canonical ``spacr`` does not."""
    import spacr.version as version
    from importlib.metadata import PackageNotFoundError

    asked = []

    def _only_nightly(name):
        asked.append(name)
        if name == "spacr-nightly":
            return "9.9.9-nightly"
        raise PackageNotFoundError(name)

    monkeypatch.setattr(version, "package_version", _only_nightly)

    assert version.get_version() == "9.9.9-nightly"
    assert asked == ["spacr", "spacr-nightly"]


def test_the_report_carries_every_field_it_documents():
    """``format_version_info`` prints one labelled line per info key."""
    import spacr.version as version

    info = version.get_version_info()
    text = version.format_version_info()

    assert set(info) == {"spacr_version", "platform", "python_version",
                         "torch_version"}
    lines = text.splitlines()
    assert [line.split(":\t")[0] for line in lines] == [
        "spacr version", "platform", "python version", "torch version"]
    for value in info.values():
        assert value in text


def test_version_str_is_resolved_lazily_and_matches_the_report():
    """``from spacr.version import version_str`` still works via PEP 562."""
    import spacr.version as version

    assert version.version_str == version.format_version_info()


def test_an_unknown_attribute_still_raises_attribute_error():
    """The PEP 562 hook must not swallow genuine typos."""
    import spacr.version as version

    with pytest.raises(AttributeError) as excinfo:
        version.verison_str  # noqa: B018 - the typo is the point

    assert "verison_str" in str(excinfo.value)


def test_a_torch_that_fails_to_report_is_not_fatal(monkeypatch):
    """A torch whose ``__version__`` blows up degrades, it does not raise."""
    import spacr.version as version

    class _Exploding:
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "torch":
                raise RuntimeError("CUDA driver mismatch")
            return None

    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr(sys, "meta_path", [_Exploding(), *sys.meta_path])

    assert version.get_torch_version() == "not available"
    assert version.get_version_info()["torch_version"] == "not available"
