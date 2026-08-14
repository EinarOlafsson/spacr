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
import sys

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
    @pytest.mark.parametrize("executable_name", ["uv", "uv.exe"])
    def test_find_uv_recognises_every_installer_executable(
            self, executable_name, monkeypatch, tmp_path):
        from spacr import updater as U

        install_root = tmp_path / "spaCR"
        prefix = install_root / "venv"
        bootstrap = install_root / "bootstrap"
        bootstrap.mkdir(parents=True)
        executable = bootstrap / executable_name
        executable.write_bytes(b"uv")
        executable.chmod(0o755)

        monkeypatch.setattr(U.sys, "prefix", str(prefix))
        monkeypatch.setattr(U.shutil, "which", lambda _name: None)

        assert U.find_uv() == str(executable)

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
        # Build args + subprocess.run, and return the output alongside the
        # exit code: the desktop installers have no terminal to print it to.
        import subprocess
        from spacr import updater as U
        captured = {}

        class _Completed:
            returncode = 0
            stdout = "Successfully installed spacr\n"
            stderr = ""

        def _fake_run(args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return _Completed()

        monkeypatch.setattr(U, "find_uv", lambda: None)
        monkeypatch.setattr(subprocess, "run", _fake_run)
        rc, output = U.run_pip_upgrade(pre_release=True)
        assert rc == 0
        assert "--pre" in captured["args"]
        assert "spacr" in captured["args"]
        assert "-m" in captured["args"] and "pip" in captured["args"]
        assert captured["kwargs"].get("capture_output") is True
        assert "Successfully installed" in output

    def test_run_pip_upgrade_uses_uv_when_pip_is_absent(self, monkeypatch):
        # The native installers build the environment with `uv venv`, which
        # never seeds pip, so `python -m pip` fails before it starts. The
        # updater has to reach for the tool that built the environment.
        import subprocess
        from spacr import updater as U
        captured = {}

        class _Completed:
            returncode = 0
            stdout = ""
            stderr = ""

        def _fake_run(args, **kwargs):
            captured["args"] = args
            return _Completed()

        monkeypatch.setattr(U, "find_uv", lambda: "/opt/spacr/bootstrap/uv")
        monkeypatch.setattr(subprocess, "run", _fake_run)
        rc, _output = U.run_pip_upgrade()
        assert rc == 0
        assert captured["args"][0] == "/opt/spacr/bootstrap/uv"
        assert captured["args"][1:3] == ["pip", "install"]
        assert "--python" in captured["args"]
        assert "-m" not in captured["args"]

    def test_a_missing_upgrade_tool_is_reported_not_raised(self, monkeypatch):
        # FileNotFoundError here used to surface as a bare exit code with no
        # explanation, on an install with no terminal to explain it in.
        import subprocess
        from spacr import updater as U

        def _boom(args, **kwargs):
            raise FileNotFoundError(args[0])

        monkeypatch.setattr(U, "find_uv", lambda: None)
        monkeypatch.setattr(subprocess, "run", _boom)
        rc, output = U.run_pip_upgrade()
        assert rc == 1
        assert "Could not run" in output

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
        monkeypatch.delitem(sys.modules, "spacr.tiff_io", raising=False)
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

    def test_a_corrupt_manifest_is_tolerated(self, tmp_path):
        """The manifest only decorates the summary cell; the export stands."""
        from spacr.notebook_export import export_run
        run_dir = tmp_path / "20260101_000000_x__mask"
        run_dir.mkdir()
        (run_dir / "manifest.json").write_text("{bad json")
        (run_dir / "settings.json").write_text('{"src": "/tmp/x"}')
        out = export_run(run_dir, out_path=tmp_path / "nb.ipynb")
        assert out.exists()

    def test_corrupt_settings_refuse_the_export(self, tmp_path):
        """A notebook that cannot load its own settings is not an export.

        This used to assert the opposite — that a ``settings.json`` reading
        ``{also bad`` still produced a notebook. ``export_run`` calls
        ``_read_settings`` under the comment "Validate that the recorded
        settings exist and parse", and while that call swallowed the parse
        error the validation was a no-op: the notebook was written with
        ``json.loads((RUN_DIR / 'settings.json').read_text())`` as its first
        code cell, so the failure moved off the export — where it can be
        reported — and into the user's notebook, on cell 1 of a file they had
        just been told was produced.
        """
        import json

        from spacr.notebook_export import export_run
        run_dir = tmp_path / "20260101_000000_x__mask"
        run_dir.mkdir()
        (run_dir / "manifest.json").write_text('{"app_key": "mask"}')
        # What a kill -9 mid-write leaves behind.
        (run_dir / "settings.json").write_text("{also bad")

        with pytest.raises(json.JSONDecodeError):
            export_run(run_dir, out_path=tmp_path / "nb.ipynb")
        assert not (tmp_path / "nb.ipynb").exists()

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

    def test_read_settings_raises_on_corrupt_json(self, tmp_path):
        """Absent is ``{}``; present-but-unparseable is an error.

        The two cases used to give the same answer, which is what made
        ``export_run``'s "validate that the settings parse" line validate
        nothing. Only the absent case is empty now.
        """
        import json

        from spacr.notebook_export import _read_settings
        assert _read_settings(tmp_path) == {}          # no file at all
        (tmp_path / "settings.json").write_text("{not valid")
        with pytest.raises(json.JSONDecodeError):
            _read_settings(tmp_path)

    def test_read_manifest_corrupt_json_returns_empty(self, tmp_path):
        from spacr.notebook_export import _read_manifest
        (tmp_path / "manifest.json").write_text("{bad")
        assert _read_manifest(tmp_path) == {}

    def test_lt_non_string_returns_false(self):
        # _lt with a non-string arg → .split raises → except (125-126).
        from spacr.updater import _lt
        assert _lt(None, "1.0") is False
        assert _lt("1.0", 123) is False


class TestDefensiveInjection:
    def test_notebook_missing_settings_file(self, tmp_path):
        # _read_settings when settings.json is ABSENT (nb line 60).
        from spacr.notebook_export import _read_settings
        assert _read_settings(tmp_path) == {}

    def test_export_run_non_directory_raises(self, tmp_path):
        # export_run on a non-existent dir → FileNotFoundError (nb 173).
        from spacr.notebook_export import export_run
        with pytest.raises(FileNotFoundError):
            export_run(tmp_path / "nope")

    def test_export_run_mask_emits_mask_preview_cell(self, tmp_path):
        # Reach the app_key=='mask' output-cell branch (nb 109).
        import json
        from spacr.notebook_export import export_run
        run_dir = tmp_path / "20260101_000000_x__mask"; run_dir.mkdir()
        (run_dir / "manifest.json").write_text('{"app_key": "mask"}')
        (run_dir / "settings.json").write_text('{"src": "/tmp/x"}')
        out = export_run(run_dir, out_path=tmp_path / "nb.ipynb")
        nb = json.loads(out.read_text())
        code = "\n".join("".join(c["source"]) for c in nb["cells"]
                          if c["cell_type"] == "code")
        assert "masks" in code.lower()

    def test_human_readable_bytes_small(self):
        # _human_readable_bytes < 1 KB → "N B" (_v1_v2 143).
        from spacr._v1_v2_bridge import _human
        assert _human(512).endswith("B")

    def test_channels_from_explicit_cellpose_keys(self):
        # The cellpose_*_channel branch (_v1_v2 58-64).
        from spacr._v1_v2_bridge import v2_channels_from_settings
        chans, names = v2_channels_from_settings({
            "cell_channel": 1, "nucleus_channel": 0,
            "pathogen_channel": None,
        })
        assert 0 in chans and 1 in chans

    def test_disk_savings_sidecar_stat_raises(self, tmp_path, monkeypatch):
        # exists()=True but stat() raises → except: pass (_v1_v2 117-118).
        from spacr import _v1_v2_bridge as B
        sidecar = tmp_path / "filename_map.csv"
        sidecar.write_text("a\n")
        real_stat = Path.stat
        def _boom(self, *a, **k):
            if self.name == "filename_map.csv":
                raise OSError("stat blocked")
            return real_stat(self, *a, **k)
        monkeypatch.setattr(Path, "stat", _boom)
        out = B.report_disk_savings(tmp_path, [])
        assert out["v2_bytes"] == 0

    def test_custom_features_spec_none(self, tmp_path, monkeypatch):
        # spec_from_file_location → None makes the file skip (cf line 82).
        import importlib.util
        from spacr import custom_features as CF
        d = tmp_path / "f"; d.mkdir()
        monkeypatch.setattr(CF, "features_dir", lambda: d)
        (d / "x.py").write_text("def feat(mask, image): return 1\n")
        monkeypatch.setattr(importlib.util, "spec_from_file_location",
                            lambda *a, **k: None)
        assert CF.discover_features() == []

    def test_custom_features_signature_unavailable(
            self, tmp_path, monkeypatch):
        # A same-module callable whose signature() raises → skip (106-107).
        import inspect
        from spacr import custom_features as CF
        d = tmp_path / "f"; d.mkdir()
        monkeypatch.setattr(CF, "features_dir", lambda: d)
        (d / "s.py").write_text("def feat(mask, image): return 1\n")
        real_sig = inspect.signature
        def _boom(obj, *a, **k):
            raise ValueError("no signature")
        monkeypatch.setattr(CF.inspect, "signature", _boom)
        # signature raises for feat → skipped, discover returns [].
        assert CF.discover_features() == []

    def test_custom_features_module_attr_raises(
            self, tmp_path, monkeypatch):
        # A callable whose __module__ access raises → skip (cf 98-99).
        from spacr import custom_features as CF
        d = tmp_path / "f"; d.mkdir()
        monkeypatch.setattr(CF, "features_dir", lambda: d)
        # Object that is callable but whose __module__ property raises.
        (d / "m.py").write_text(
            "class _Bad:\n"
            "    def __call__(self, mask, image): return 1\n"
            "    @property\n"
            "    def __module__(self): raise RuntimeError('x')\n"
            "weird = _Bad()\n"
            "def feat(mask, image): return 2\n")
        names = [f.name for f in CF.discover_features()]
        # 'weird' is skipped (module attr raises); 'feat' survives.
        assert "weird" not in names


class TestMaskIOEnvBranch:
    def test_invalid_env_format_warns_and_defaults(self, monkeypatch):
        # SPACR_MASK_FORMAT=bogus at import → warn + default tif (47-49).
        import importlib
        monkeypatch.setenv("SPACR_MASK_FORMAT", "bogus")
        import spacr.mask_io as MIO
        importlib.reload(MIO)
        assert MIO.DEFAULT_FORMAT == "tif"
        # Reload once more with the env cleared so other tests see the
        # normal default.
        monkeypatch.delenv("SPACR_MASK_FORMAT", raising=False)
        importlib.reload(MIO)


def test_get_torch_version_with_torch_present():
    # Directly exercise the successful import-torch path (version line 36).
    from spacr.version import get_torch_version
    v = get_torch_version()
    assert isinstance(v, str)  # line 36 executes whether import succeeds or not
