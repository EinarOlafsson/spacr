"""Coverage-fill for spacr.run_journal: installed-package versions."""
from __future__ import annotations

from spacr import run_journal as RJ


# ---------------------------------------------------------------------------
# _pkg_version
# ---------------------------------------------------------------------------


def test_pkg_version():
    assert isinstance(RJ._pkg_version("numpy"), str)
    assert RJ._pkg_version("definitely-not-a-package-xyz") == "not installed"
