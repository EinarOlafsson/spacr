"""`check_reviewed_runtime_evidence.py` must classify, not merely complain.

WHY A TEST AND NOT JUST THE TOOL. On a clean tree the reporter prints "every
reviewed runtime record still matches its source" and returns 0, which is
exactly what it printed while carrying a bug that made it useless on the only
tree it was written for. The GATE check called `_contextualize`, which calls
`_reviewed_translation`, which calls `reviewed_runtime_translations` -- the
function that RAISES on the first stale record. So the reporter re-entered the
loader it exists to replace and died with the traceback it exists to turn into
a list. A green run proved nothing; only simulating a rename found it.

So these tests do what that calibration did: mutate the SOURCE TABLES in
memory, never on disk, and assert the reporter survives and classifies.

THE DISTINCTION IT EXISTS TO DRAW, from `check_reviewed_api_evidence.py`:

    MOVED  the English still exists under a different key. A rename. Re-bind
           the record's `key` and the human review is preserved.
    STALE  the English is gone. Retire -- which for the runtime store means
           DELETE, because the loader validates the field set exactly and a
           record carrying `retired` is invalid rather than retired.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

pytest.importorskip("build_i18n_catalogs")

import build_i18n_catalogs as runtime  # noqa: E402
import check_reviewed_runtime_evidence as reporter  # noqa: E402


@pytest.fixture
def sources():
    """The real canonical source tables, so the corpus is the real one."""
    return runtime.canonical_sources()


def _rename_keys(sources, old_suffix, new_suffix):
    """A copy of the source tables with one suffix renamed on every key."""
    out = {}
    for name, table in sources.items():
        if isinstance(table, dict):
            out[name] = {
                (key[: -len(old_suffix)] + new_suffix
                 if key.endswith(old_suffix) else key): value
                for key, value in table.items()
            }
        else:
            out[name] = table
    return out


def _drop_keys(sources, suffix):
    """A copy of the source tables with every key carrying ``suffix`` removed."""
    out = {}
    for name, table in sources.items():
        if isinstance(table, dict):
            out[name] = {k: v for k, v in table.items()
                         if not k.endswith(suffix)}
        else:
            out[name] = table
    return out


def test_the_clean_tree_is_reported_clean(capsys):
    """The baseline. If this fails the tree has stale evidence already."""
    assert reporter.main([]) == 0
    assert "still matches its source" in capsys.readouterr().out


def test_a_key_rename_is_reported_as_MOVED_not_STALE(monkeypatch, capsys,
                                                     sources):
    """A rename with byte-identical prose must not read as rewritten prose.

    THE PREMISE THIS REFUTES, and it cost a wrong plan before it was measured:
    that a record survives a key rename when the English does not change. It
    does not. `setting_tooltips` is keyed by the SETTING KEY, so the loader
    resolves `sources[table].get(key)` to None and reports a STALE SOURCE for
    a string nobody touched -- the most confusing possible failure, because
    the obvious diagnosis is wrong. The reporter's job is to say "this moved".
    """
    # A SYNTHETIC TARGET, DELIBERATELY. This used to simulate the real
    # `_min_distance -> _min_watershed_distance` rename, and then instruction
    # 391 performed that rename for real -- so the simulation renamed nothing,
    # found nothing, and the test failed for a reason that had nothing to do
    # with the reporter. A test that pins a live setting name is a test any
    # future rename breaks; the suffix below is a live one with 39 reviewed records, renamed
    # to a spelling nobody will ever ship.
    renamed = _rename_keys(sources, "_chann_dim", "_chann_dim_ZZSYNTHETIC")
    monkeypatch.setattr(runtime, "canonical_sources", lambda: renamed)

    assert reporter.main([]) == 1
    out = capsys.readouterr().out
    assert "MOVED" in out, out
    assert "re-bind" in out
    assert "_chann_dim_ZZSYNTHETIC" in out, (
        "the report has to name the key to re-bind TO, or it is not a work "
        "list")
    moved_block = out.split("MOVED", 1)[1].split("STALE", 1)[0]
    assert "_chann_dim" in moved_block


def test_a_removed_setting_is_reported_as_STALE(monkeypatch, capsys, sources):
    """A key that is gone entirely cannot be re-bound, only deleted."""
    dropped = _drop_keys(sources, "_chann_dim")
    monkeypatch.setattr(runtime, "canonical_sources", lambda: dropped)

    assert reporter.main([]) == 1
    out = capsys.readouterr().out
    assert "STALE" in out, out
    stale_block = out.split("STALE", 1)[1]
    assert "_chann_dim" in stale_block


def test_the_advice_is_delete_rather_than_retire_in_place(monkeypatch,
                                                          capsys, sources):
    """The one place this differs from the API tool, and getting it wrong
    costs a build.

    The API store retires a record in place with ``"retired": "<date>"``. The
    runtime loader validates ``set(record) == {...}`` by EXACT equality, so a
    runtime record carrying `retired` is an "invalid reviewed runtime record"
    -- a hard failure, not a retired record.
    """
    monkeypatch.setattr(runtime, "canonical_sources",
                        lambda: _drop_keys(sources, "_chann_dim"))
    reporter.main([])
    out = capsys.readouterr().out
    assert "DELETE" in out, (
        "a reader who retires these in place the way the API store does will "
        "break the build, so the report has to say delete")


def test_the_reporter_does_not_re_enter_the_loader_it_replaces(monkeypatch,
                                                               sources):
    """THE BUG THIS FILE EXISTS FOR.

    With any record stale, the reporter used to raise `ValueError` out of
    `reviewed_runtime_translations` instead of printing a list -- inheriting
    the exact failure it was written to replace. It survives only because it
    holds the loader's own `_REVIEWED_RUNTIME_LOADING` guard.
    """
    monkeypatch.setattr(runtime, "canonical_sources",
                        lambda: _drop_keys(sources, "_chann_dim"))
    try:
        status = reporter.main([])
    except ValueError as exc:  # pragma: no cover - the regression
        pytest.fail(
            "the reporter re-entered reviewed_runtime_translations and raised "
            f"instead of reporting: {exc}")
    assert status == 1


def test_every_language_on_disk_is_checked_by_default(sources):
    """A reporter that silently skipped a locale would under-report the work."""
    languages = reporter._languages([])
    on_disk = sorted(p.name for p in reporter.REVIEWED_RUNTIME.iterdir()
                     if p.is_dir())
    assert languages == on_disk
    assert len(languages) >= 9, languages
