"""Tests for the Batch-6a "deep features" — notebook export, custom
per-object features, and the auto-updater.

These modules are network- and filesystem-boundary code, so tests
focus on:

* The pure logic (version comparison, feature discovery).
* Structural correctness of the generated notebook (valid JSON, has
  the cells we promised).
* Graceful degradation when the network is unreachable.

Nothing here hits the real PyPI / GitHub — the network path is
monkey-patched with a stub.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from spacr import custom_features, notebook_export, updater


# ---------------------------------------------------------------------------
# updater — version comparison
# ---------------------------------------------------------------------------

class TestVersionCompare:
    """The custom ``_lt`` handles mixed-length 3 / 4-part versions."""

    def test_lt_three_part_ordering(self):
        assert updater._lt("1.4.0", "1.4.1") is True
        assert updater._lt("1.4.1", "1.4.1") is False
        assert updater._lt("1.4.2", "1.4.1") is False

    def test_lt_four_part_beats_three_part(self):
        # 1.4.1 < 1.4.1.1: the 4-part is a strictly newer polish
        assert updater._lt("1.4.1", "1.4.1.1") is True
        assert updater._lt("1.4.1.1", "1.4.1") is False

    def test_lt_four_part_relative(self):
        assert updater._lt("1.4.1.1", "1.4.1.2") is True
        assert updater._lt("1.4.1.9", "1.4.2") is True


class TestUpdaterCheck:
    """The check_for_updates path degrades gracefully."""

    def test_network_error_returns_error_field(self, monkeypatch):
        """When both requests raise, we still return an UpdateInfo
        with an error field — never raise."""
        import urllib.request

        def boom(*args, **kwargs):
            raise RuntimeError("no network")

        monkeypatch.setattr(urllib.request, "urlopen", boom)
        info = updater.check_for_updates(timeout=0.1)
        assert info.latest_release is None
        assert info.nightly_sha is None
        assert info.error is not None
        assert info.upgrade_available is False

    def test_pypi_hit_marks_upgrade_when_older(self, monkeypatch):
        """Stub urlopen to return a canned PyPI payload that reports
        a newer version — upgrade_available should flip True."""
        import io
        import urllib.request

        class _Resp:
            def __init__(self, payload: bytes):
                self._p = payload
            def read(self):
                return self._p
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False

        installed = updater._installed_version()
        # Fake latest = installed + one polish bump
        parts = installed.split(".")
        while len(parts) < 4:
            parts.append("0")
        try:
            parts[-1] = str(int(parts[-1]) + 1)
        except ValueError:
            parts[-1] = "99"
        fake_latest = ".".join(parts)
        payload = json.dumps({"info": {"version": fake_latest}}).encode()

        def stub(req, timeout=None):
            url = getattr(req, "full_url", str(req))
            if "pypi" in url:
                return _Resp(payload)
            raise RuntimeError("no github in this test")

        monkeypatch.setattr(urllib.request, "urlopen", stub)
        info = updater.check_for_updates(timeout=0.1)
        assert info.latest_release == fake_latest
        assert info.upgrade_available is True


# ---------------------------------------------------------------------------
# notebook_export — structural correctness
# ---------------------------------------------------------------------------

class TestNotebookExport:
    """Ensure a run folder -> valid nbformat v4 notebook JSON."""

    def _make_run(self, tmp_path: Path, app_key: str = "mask") -> Path:
        run_dir = tmp_path / "20260722_010203_abcdef__mask"
        run_dir.mkdir()
        (run_dir / "manifest.json").write_text(json.dumps({
            "app_key":   app_key,
            "start_utc": "2026-07-22T01:02:03Z",
            "elapsed_s": 12.5,
            "status":    "ok",
            "env":       {"spacr": "1.4.1.2",
                          "torch": "2.4.0",
                          "cellpose": "3.0.10"},
        }))
        (run_dir / "settings.json").write_text(json.dumps({
            "src": "/tmp/dataset",
            "cell_channel": 0,
        }))
        return run_dir

    def test_export_run_writes_valid_notebook(self, tmp_path):
        run_dir = self._make_run(tmp_path, "mask")
        out = notebook_export.export_run(
            run_dir, out_path=tmp_path / "out.ipynb")
        assert out.exists()
        nb = json.loads(out.read_text())
        # nbformat v4 shape
        assert nb["nbformat"] == 4
        assert "cells" in nb and len(nb["cells"]) >= 4
        # First cell mentions the app key
        first = "".join(nb["cells"][0]["source"])
        assert "mask" in first.lower()
        # A code cell should reference the recorded SETTINGS
        code_texts = [
            "".join(c["source"])
            for c in nb["cells"] if c["cell_type"] == "code"
        ]
        assert any("SETTINGS" in t for t in code_texts)

    def test_export_run_missing_folder_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            notebook_export.export_run(tmp_path / "does_not_exist")

    def test_export_run_measure_scaffolds_sqlite_cell(self, tmp_path):
        run_dir = self._make_run(tmp_path, "measure")
        out = notebook_export.export_run(
            run_dir, out_path=tmp_path / "measure.ipynb")
        nb = json.loads(out.read_text())
        code = "\n".join(
            "".join(c["source"])
            for c in nb["cells"] if c["cell_type"] == "code"
        )
        # The measure entrypoint should be imported and a sqlite call
        # should be scaffolded — that's the point of the export.
        assert "measure_crop" in code
        assert "sqlite3" in code

    def test_export_run_unknown_app_still_produces_notebook(self, tmp_path):
        """A run whose app_key is not in _ENTRYPOINTS should still
        produce a notebook — falling back to a "no known entrypoint"
        stub rather than crashing."""
        run_dir = self._make_run(tmp_path, "definitely_not_a_real_app")
        out = notebook_export.export_run(
            run_dir, out_path=tmp_path / "unknown.ipynb")
        nb = json.loads(out.read_text())
        code = "\n".join(
            "".join(c["source"])
            for c in nb["cells"] if c["cell_type"] == "code"
        )
        assert "_run" in code


# ---------------------------------------------------------------------------
# custom_features — discovery + safe invocation
# ---------------------------------------------------------------------------

class TestCustomFeatures:
    """Discovery must load valid features + skip malformed ones."""

    @pytest.fixture(autouse=True)
    def _redirect_features_dir(self, tmp_path, monkeypatch):
        """Point ``features_dir()`` at a throwaway tmp dir so tests
        don't touch the real ``~/.spacr/features/``."""
        fake = tmp_path / "features"
        fake.mkdir()
        monkeypatch.setattr(custom_features, "features_dir",
                            lambda: fake)
        return fake

    def test_discover_no_files_returns_empty(self):
        assert custom_features.discover_features() == []

    def test_discover_picks_up_valid_feature(self, _redirect_features_dir):
        (_redirect_features_dir / "myfeat.py").write_text(textwrap.dedent("""
            def asymmetry(mask, image):
                return 0.5
        """))
        feats = custom_features.discover_features()
        names = [f.name for f in feats]
        assert "asymmetry" in names

    def test_discover_ignores_private_and_underscore(self,
                                                      _redirect_features_dir):
        (_redirect_features_dir / "_private.py").write_text(
            "def priv(mask, image): return 1\n")
        (_redirect_features_dir / "keep.py").write_text(
            "def _hidden(mask, image): return 1\n"
            "def public(mask, image): return 2\n"
        )
        feats = custom_features.discover_features()
        names = [f.name for f in feats]
        assert "priv" not in names
        assert "_hidden" not in names
        assert "public" in names

    def test_discover_skips_broken_file(self, _redirect_features_dir):
        (_redirect_features_dir / "broken.py").write_text(
            "raise RuntimeError('bad file')\n")
        (_redirect_features_dir / "good.py").write_text(
            "def ok(mask, image): return 42\n")
        feats = custom_features.discover_features()
        assert any(f.name == "ok" for f in feats)

    def test_call_feature_scalar_result(self, _redirect_features_dir):
        (_redirect_features_dir / "s.py").write_text(
            "def m(mask, image): return 7\n")
        cf = custom_features.discover_features()[0]
        out = custom_features.call_feature(cf, [], [])
        assert out == {"m": 7}

    def test_call_feature_dict_result_gets_prefix(self,
                                                    _redirect_features_dir):
        (_redirect_features_dir / "d.py").write_text(
            "def stats(mask, image):\n"
            "    return {'mean': 1.0, 'std': 0.5}\n"
        )
        cf = custom_features.discover_features()[0]
        out = custom_features.call_feature(cf, [], [])
        assert out == {"stats_mean": 1.0, "stats_std": 0.5}

    def test_call_feature_swallows_exception(self, _redirect_features_dir):
        (_redirect_features_dir / "e.py").write_text(
            "def bad(mask, image): raise ValueError('nope')\n")
        cf = custom_features.discover_features()[0]
        assert custom_features.call_feature(cf, [], []) == {}
