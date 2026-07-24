"""Coverage-fill tests for the small utility modules — drives version,
updater, _v1_v2_bridge, mask_io, custom_features, notebook_export to
100% by exercising their error/edge branches.

Each test targets specific previously-uncovered lines (see the
per-line comments); together with the existing suites these modules
reach full statement coverage.
"""
from __future__ import annotations

import builtins
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# version.py
# ---------------------------------------------------------------------------

class TestVersion:
    def test_get_version_unknown_when_no_distribution(self, monkeypatch):
        # Both candidate distributions raise → "unknown" (lines 24-26).
        from spacr import version as V
        from importlib.metadata import PackageNotFoundError

        def _raise(name):
            raise PackageNotFoundError(name)
        monkeypatch.setattr(V, "package_version", _raise)
        assert V.get_version() == "unknown"

    def test_get_torch_version_not_available(self, monkeypatch):
        # torch import fails → "not available" (lines 37-38).
        from spacr import version as V
        real_import = builtins.__import__

        def _no_torch(name, *a, **k):
            if name == "torch":
                raise ImportError("no torch")
            return real_import(name, *a, **k)
        monkeypatch.setattr(builtins, "__import__", _no_torch)
        assert V.get_torch_version() == "not available"

    def test_get_version_info_shape(self):
        from spacr import version as V
        info = V.get_version_info()
        assert set(info) == {"spacr_version", "platform",
                              "python_version", "torch_version"}

    def test_format_version_info_multiline(self):
        from spacr import version as V
        out = V.format_version_info()
        assert "spacr version:" in out and "torch version:" in out


# ---------------------------------------------------------------------------
# updater.py
# ---------------------------------------------------------------------------

class TestUpdater:
    def test_installed_version_falls_back_to_nightly(self, monkeypatch):
        # spacr raises, spacr-nightly resolves (lines 107-112).
        import importlib.metadata as M
        from spacr import updater as U

        def _ver(name):
            if name == "spacr":
                raise M.PackageNotFoundError(name)
            return "9.9.9"
        monkeypatch.setattr(M, "version", _ver)
        assert U._installed_version() == "9.9.9"

    def test_installed_version_unknown(self, monkeypatch):
        import importlib.metadata as M
        from spacr import updater as U

        def _raise(name):
            raise M.PackageNotFoundError(name)
        monkeypatch.setattr(M, "version", _raise)
        assert U._installed_version() == "unknown"

    def test_lt_handles_bad_version_strings(self):
        from spacr.updater import _lt
        # Non-numeric parts are filtered; unparseable → False (125-126).
        assert _lt("abc", "def") is False
        assert _lt("1.0", "1.0.1") is True

    def test_run_pip_upgrade_invokes_pip(self, monkeypatch):
        # Lines 141-146 — build args + subprocess.call.
        import subprocess
        from spacr import updater as U
        captured = {}
        def _fake_call(args):
            captured["args"] = args
            return 0
        monkeypatch.setattr(subprocess, "call", _fake_call)
        rc = U.run_pip_upgrade(pre_release=True)
        assert rc == 0
        assert "--pre" in captured["args"]
        assert "spacr" in captured["args"]

    def test_check_for_updates_github_branch(self, monkeypatch):
        # Exercise the github nightly parse (lines 89-90) + error absorb.
        import urllib.request
        from spacr import updater as U

        class _Resp:
            def __init__(self, payload): self._p = payload
            def read(self): return self._p
            def __enter__(self): return self
            def __exit__(self, *a): return False

        import json
        def _stub(req, timeout=None):
            url = getattr(req, "full_url", str(req))
            if "pypi" in url:
                return _Resp(json.dumps({"info": {"version": "1.0.0"}}).encode())
            return _Resp(json.dumps({"sha": "abcdef1234567"}).encode())
        monkeypatch.setattr(urllib.request, "urlopen", _stub)
        info = U.check_for_updates(timeout=0.1)
        assert info.nightly_sha == "abcdef1"


# ---------------------------------------------------------------------------
# _v1_v2_bridge.py
# ---------------------------------------------------------------------------

class TestV1V2Bridge:
    def test_channels_default_when_empty(self):
        # No channels resolvable → 4-channel default (lines ~108-109).
        from spacr._v1_v2_bridge import v2_channels_from_settings
        chans, names = v2_channels_from_settings({})
        assert chans == [0, 1, 2, 3]
        assert names == ["ch0", "ch1", "ch2", "ch3"]

    def test_channels_skips_non_int(self):
        # Non-int channel entries are skipped (lines 73-74).
        from spacr._v1_v2_bridge import v2_channels_from_settings
        chans, names = v2_channels_from_settings(
            {"channels": [0, "x", 2, None]})
        assert 0 in chans and 2 in chans
        assert "x" not in chans

    def test_report_disk_savings_on_empty(self, tmp_path):
        from spacr._v1_v2_bridge import report_disk_savings
        out = report_disk_savings(tmp_path, [])
        assert set(out) >= {"v2_bytes", "v1_estimated_bytes"}
        assert out["v2_bytes"] == 0


# ---------------------------------------------------------------------------
# mask_io.py
# ---------------------------------------------------------------------------

class TestMaskIO:
    def test_save_and_load_npy(self, tmp_path):
        from spacr.mask_io import save_mask, load_mask
        mask = np.zeros((16, 16), dtype=np.uint16); mask[4:8, 4:8] = 1
        p = save_mask(str(tmp_path / "m"), mask, fmt="npy")
        assert p.suffix == ".npy"
        loaded = load_mask(str(p))
        assert loaded.shape == (16, 16)
        assert loaded.max() == 1

    def test_save_tif(self, tmp_path):
        from spacr.mask_io import save_mask, load_mask
        mask = np.zeros((16, 16), dtype=np.uint16); mask[2:6, 2:6] = 3
        p = save_mask(str(tmp_path / "m"), mask, fmt="tif")
        assert p.suffix in (".tif", ".tiff")
        assert load_mask(str(p)).max() == 3

    def test_save_unknown_format_raises(self, tmp_path):
        from spacr.mask_io import save_mask
        with pytest.raises(ValueError):
            save_mask(str(tmp_path / "m"), np.zeros((4, 4), np.uint16),
                      fmt="bogus")  # line ~84

    def test_load_missing_raises_filenotfound(self, tmp_path):
        from spacr.mask_io import load_mask
        with pytest.raises(FileNotFoundError):
            load_mask(str(tmp_path / "does_not_exist"))

    def test_read_one_unsupported_extension_raises(self, tmp_path):
        # _read_one's else branch (line ~124) for an unknown suffix.
        from spacr.mask_io import _read_one
        bad = tmp_path / "m.qqq"
        bad.write_bytes(b"x")
        with pytest.raises(ValueError):
            _read_one(bad)

    def test_save_tif_falls_back_to_npy_without_tifffile(
            self, tmp_path, monkeypatch):
        # tifffile import fails → npy fallback (lines 74-76).
        from spacr import mask_io
        real_import = builtins.__import__

        def _no_tifffile(name, *a, **k):
            if name == "tifffile":
                raise ImportError("no tifffile")
            return real_import(name, *a, **k)
        monkeypatch.setattr(builtins, "__import__", _no_tifffile)
        p = mask_io.save_mask(str(tmp_path / "m"),
                              np.zeros((8, 8), np.uint16), fmt="tif")
        assert p.suffix == ".npy"


# ---------------------------------------------------------------------------
# custom_features.py
# ---------------------------------------------------------------------------

class TestCustomFeatures:
    @pytest.fixture
    def _fd(self, tmp_path, monkeypatch):
        d = tmp_path / "features"; d.mkdir()
        from spacr import custom_features as CF
        monkeypatch.setattr(CF, "features_dir", lambda: d)
        return d

    def test_features_dir_creates(self, tmp_path, monkeypatch):
        from spacr import custom_features as CF
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        d = CF.features_dir()
        assert d.exists() and d.name == "features"

    def test_discover_skips_single_param_function(self, _fd):
        # Function with <2 params is skipped (lines 105-107).
        (_fd / "one.py").write_text("def only_mask(mask): return 1\n")
        from spacr.custom_features import discover_features
        assert not any(f.name == "only_mask" for f in discover_features())

    def test_discover_skips_imported_symbol(self, _fd):
        # A symbol imported (not defined) in the file is skipped (93-99).
        (_fd / "imp.py").write_text(
            "from math import hypot\n"
            "def local_feat(mask, image): return 1.0\n")
        from spacr.custom_features import discover_features
        names = [f.name for f in discover_features()]
        assert "hypot" not in names
        assert "local_feat" in names

    def test_discover_handles_broken_module(self, _fd):
        (_fd / "boom.py").write_text("raise RuntimeError('x')\n")
        (_fd / "ok.py").write_text("def good(mask, image): return 2\n")
        from spacr.custom_features import discover_features
        assert any(f.name == "good" for f in discover_features())


# ---------------------------------------------------------------------------
# notebook_export.py
# ---------------------------------------------------------------------------

class TestNotebookExport:
    def test_missing_manifest_and_settings_are_tolerated(self, tmp_path):
        # A run dir with NO manifest/settings → the readers return {}
        # (lines 50, 53-54, 60, 63-64) and export still works.
        from spacr.notebook_export import export_run
        run_dir = tmp_path / "20260101_000000_x__mask"
        run_dir.mkdir()
        (run_dir / "settings.json").write_text("{}")
        out = export_run(run_dir, out_path=tmp_path / "nb.ipynb")
        assert out.exists()

    def test_corrupt_json_is_tolerated(self, tmp_path):
        from spacr.notebook_export import export_run
        run_dir = tmp_path / "20260101_000000_x__mask"
        run_dir.mkdir()
        (run_dir / "manifest.json").write_text("{bad json")
        (run_dir / "settings.json").write_text("{also bad")
        out = export_run(run_dir, out_path=tmp_path / "nb.ipynb")
        assert out.exists()

    def test_default_out_path(self, tmp_path):
        # out_path=None defaults to <run_dir>/notebook.ipynb (line 224).
        from spacr.notebook_export import export_run
        run_dir = tmp_path / "20260101_000000_x__measure"
        run_dir.mkdir()
        (run_dir / "manifest.json").write_text('{"app_key": "measure"}')
        (run_dir / "settings.json").write_text('{"src": "/tmp/x"}')
        out = export_run(run_dir)
        assert out.name == "notebook.ipynb"


# ---------------------------------------------------------------------------
# Edge-line closers (the last stubborn branches)
# ---------------------------------------------------------------------------

class TestEdgeLines:
    def test_disk_savings_bad_stack_path(self, tmp_path):
        # stat() on a missing stack path → except: continue (108-109).
        from spacr._v1_v2_bridge import report_disk_savings
        class _S:
            path = str(tmp_path / "missing.npy")
        out = report_disk_savings(tmp_path, [_S()])
        assert out["v2_bytes"] == 0

    def test_disk_savings_counts_sidecars(self, tmp_path):
        # A present filename_map.csv is added (117-118).
        from spacr._v1_v2_bridge import report_disk_savings
        (tmp_path / "filename_map.csv").write_text("a,b\n1,2\n")
        out = report_disk_savings(tmp_path, [])
        assert out["v2_bytes"] > 0

    def test_custom_feature_non_callable_public_symbol(
            self, tmp_path, monkeypatch):
        from spacr import custom_features as CF
        d = tmp_path / "f"; d.mkdir()
        monkeypatch.setattr(CF, "features_dir", lambda: d)
        # CONSTANT is public + non-callable → skipped (line 93).
        (d / "c.py").write_text(
            "CONSTANT = 42\ndef feat(mask, image): return 1\n")
        names = [f.name for f in CF.discover_features()]
        assert "CONSTANT" not in names and "feat" in names

    def test_custom_feature_builtin_signature_unavailable(
            self, tmp_path, monkeypatch):
        # A callable whose signature() raises is skipped (106-107).
        from spacr import custom_features as CF
        d = tmp_path / "f"; d.mkdir()
        monkeypatch.setattr(CF, "features_dir", lambda: d)
        # Re-export a C builtin (len) — inspect.signature raises for it,
        # and its __module__ ('builtins') != the module, so it's skipped.
        (d / "b.py").write_text(
            "mylen = len\ndef feat(mask, image): return 1\n")
        names = [f.name for f in CF.discover_features()]
        assert "mylen" not in names and "feat" in names

    def test_read_settings_corrupt_json_returns_empty(self, tmp_path):
        # notebook_export._read_settings except branch (line 60).
        from spacr.notebook_export import _read_settings
        run_dir = tmp_path
        (run_dir / "settings.json").write_text("{not valid")
        assert _read_settings(run_dir) == {}

    def test_read_manifest_corrupt_json_returns_empty(self, tmp_path):
        from spacr.notebook_export import _read_manifest
        (tmp_path / "manifest.json").write_text("{bad")
        assert _read_manifest(tmp_path) == {}

    def test_lt_non_string_returns_false(self):
        # _lt with a non-string arg → .split raises → except (125-126).
        from spacr.updater import _lt
        assert _lt(None, "1.0") is False
        assert _lt("1.0", 123) is False
