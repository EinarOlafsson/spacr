"""`runs_root` creates the journal folder on first access."""
from __future__ import annotations

import spacr.run_journal as rj


def test_the_runs_root_is_made_on_first_access(tmp_path, monkeypatch):
    """`runs_root` creates ~/.spacr/runs rather than assuming it.

    Every other test in this file replaces it, so nothing here reached
    its three lines -- and a first-ever launch is exactly when they run.
    """
    import pathlib

    # THE REAL ONE. `_isolated_dot_spacr_store` replaces `rj.runs_root`
    # with a lambda returning the session sandbox, so calling the module
    # attribute here would test the fixture and not the three lines this
    # test exists for. That fixture stashes the original under
    # `unsandboxed_runs_root` for exactly this.
    runs_root = rj.unsandboxed_runs_root

    monkeypatch.setattr(pathlib.Path, "home", staticmethod(lambda: tmp_path))
    expected = tmp_path / ".spacr" / "runs"
    assert not expected.exists()
    got = runs_root()
    assert got == expected
    assert expected.is_dir(), "the journal root was not created"
    assert runs_root() == expected         # idempotent on a second call
