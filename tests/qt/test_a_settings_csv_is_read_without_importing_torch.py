"""Reading a settings CSV in the GUI does not import spacr.utils.

The tutorial capture of lesson 08 (clip tpl_33_templates_imported, Measure)
froze the window for about 2 s after the settings CSV was picked in
Templates > Import. Measured offscreen on 2026-10-13 at 2560x1440 with a
2 ms heartbeat and cProfile: ``RecipeDialog._on_import`` read the file with
``AppScreen._load_settings_csv``, which called
:func:`spacr.utils.load_settings`, and the first ``import spacr.utils`` of a
session brings in torch, torchvision, scikit-image and statsmodels: 4.4 s on
the GUI thread, nothing after it. The parse itself takes milliseconds.

The parser now lives in :mod:`spacr.tabular` (pandas and :mod:`spacr.schema`
only), :func:`spacr.utils.load_settings` delegates to it, and the screen
reads the file once through it: 4,449 ms -> 9 ms for the same import.

The block test runs in a fresh interpreter, where nothing has imported
spacr.utils yet, and compares the read with the import it used to need in
the same process, so a loaded runner slows both sides alike.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr import tabular                                       # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def _old_load_settings(csv_file_path, setting_key='setting_key',
                       setting_value='setting_value'):
    """spacr.utils.load_settings as it was before 2026-10-13, verbatim."""
    df = tabular.read_table(csv_file_path, report=None)
    if setting_key not in df.columns or setting_value not in df.columns:
        if 'Key' in df.columns and 'Value' in df.columns:
            setting_key, setting_value = 'Key', 'Value'
        else:
            raise ValueError(
                f"CSV file must contain {setting_key} and {setting_value} "
                f"columns (or the Key/Value pair save_settings writes); "
                f"{os.path.basename(str(csv_file_path))} has "
                f"{list(df.columns)}.")

    def parse_value(value):
        if pd.isna(value) or value == '':
            return None
        if not isinstance(value, str):
            return value
        if value.strip().lower() == 'true':
            return True
        if value.strip().lower() == 'false':
            return False
        if value.startswith(('(', '[', '{')):
            try:
                parsed_value = ast.literal_eval(value)
                if isinstance(parsed_value, dict):
                    parsed_value = {k: parse_value(v) for k, v in parsed_value.items()}
                return parsed_value
            except (ValueError, SyntaxError):
                pass
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            pass
        return value

    return {key: parse_value(value) for key, value in zip(df[setting_key], df[setting_value])}


def _old_screen_reader(columns, path):
    """AppScreen._load_settings_csv as it was, on the old load_settings."""
    first_error = None
    for key_col, value_col in columns:
        try:
            return _old_load_settings(path, setting_key=key_col,
                                      setting_value=value_col)
        except ValueError as e:
            if first_error is None:
                first_error = e
    raise first_error


_ROWS = [
    ("src", "/data/plate1"), ("test_mode", "False"), ("verbose", "TRUE"),
    ("channels", "[0, 1, 2, 3]"), ("png_size", "[224, 224]"),
    ("crop_mode", "['cell']"), ("nested", "{'a': '1', 'b': 'true', 'c': ''}"),
    ("tuple", "(1, 2)"), ("broken", "[1, 2"), ("n_jobs", "25"),
    ("sigma", "2.5"), ("version", "1.5.1"), ("empty", ""),
    ("normalize_by", "png"), ("home", "~/plates"),
]


def _write(path, header, rows=_ROWS):
    lines = [",".join(header)]
    for key, value in rows:
        quoted = '"' + value.replace('"', '""') + '"' if value else ""
        lines.append(f"{key},{quoted}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


HEADERS = [("Key", "Value"), ("setting_key", "setting_value"), ("key", "value"),
           ("Setting", "Value"), ("name", "value")]


def _same(a, b):
    return json.dumps(a, sort_keys=True, default=repr) == json.dumps(b, sort_keys=True, default=repr)


@pytest.mark.parametrize("header", HEADERS)
def test_the_screen_reads_every_header_spelling_as_before(tmp_path, header):
    from spacr.qt.screens.app_screen import AppScreen
    path = _write(tmp_path / "settings.csv", header)
    new = AppScreen._load_settings_csv(path)
    old = _old_screen_reader(AppScreen._CSV_COLUMNS, path)
    assert _same(new, old)
    assert new["nested"] == {"a": 1, "b": True, "c": None}
    assert new["broken"] == "[1, 2" and new["empty"] is None


def test_an_unknown_header_fails_with_the_same_message(tmp_path):
    from spacr.qt.screens.app_screen import AppScreen
    path = _write(tmp_path / "odd.csv", ("parameter", "setting"))
    with pytest.raises(ValueError) as new:
        AppScreen._load_settings_csv(path)
    with pytest.raises(ValueError) as old:
        _old_screen_reader(AppScreen._CSV_COLUMNS, path)
    assert str(new.value) == str(old.value)


def test_an_empty_file_fails_as_before(tmp_path):
    from spacr.qt.screens.app_screen import AppScreen
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError) as new:
        AppScreen._load_settings_csv(str(path))
    with pytest.raises(ValueError) as old:
        _old_screen_reader(AppScreen._CSV_COLUMNS, str(path))
    assert type(new.value) is type(old.value) and str(new.value) == str(old.value)


@pytest.mark.parametrize("header", HEADERS + [("parameter", "setting")])
@pytest.mark.parametrize("asked", [("setting_key", "setting_value"), ("Key", "Value"),
                                   ("key", "value")])
def test_load_settings_is_unchanged(tmp_path, header, asked):
    from spacr.utils import load_settings
    path = _write(tmp_path / "s.csv", header)
    try:
        old = _old_load_settings(path, *asked)
    except ValueError as exc:
        with pytest.raises(ValueError) as new:
            load_settings(path, setting_key=asked[0], setting_value=asked[1])
        assert str(new.value) == str(exc)
        return
    assert _same(load_settings(path, setting_key=asked[0], setting_value=asked[1]), old)


def test_a_real_run_settings_file_reads_the_same(tmp_path):
    from spacr.qt.screens.app_screen import AppScreen
    settings = {"src": "/data/plate1", "channels": [0, 1, 2, 3], "png_size": [224, 224],
                "crop_mode": ["cell"], "normalize": False, "n_jobs": 25,
                "manders_thresholds": [15, 85, 95], "dialate_png_ratios": [0.2],
                "nucleus_mask_dim": None, "timelapse_objects": ["cell"]}
    path = tmp_path / "measure_crop_settings.csv"
    pd.DataFrame({"Key": list(settings), "Value": [str(v) for v in settings.values()]}).to_csv(path, index=False)
    new = AppScreen._load_settings_csv(str(path))
    assert _same(new, _old_screen_reader(AppScreen._CSV_COLUMNS, str(path)))
    assert new["channels"] == [0, 1, 2, 3] and new["nucleus_mask_dim"] is None


def test_the_templates_import_files_the_same_template(qtbot, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog, QWidget
    from spacr.qt import recipes
    from spacr.qt.screens.app_screen import AppScreen
    monkeypatch.setenv(recipes.RECIPE_DIR_ENV, str(tmp_path / "recipes"))
    path = _write(tmp_path / "measure_crop_settings.csv", ("Key", "Value"))

    class Screen(QWidget):
        app_key = "measure"
        _load_settings_csv = staticmethod(AppScreen._load_settings_csv)

    screen = Screen()
    qtbot.addWidget(screen)
    dialog = recipes.RecipeDialog(screen)
    qtbot.addWidget(dialog)
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (path, "Settings CSV (*.csv)")))
    dialog._on_import()
    [recipe] = dialog.recipes()
    assert recipe.name == "measure_crop_settings" and recipe.app_key == "measure"
    assert _same(recipe.settings, _old_screen_reader(AppScreen._CSV_COLUMNS, path))


_PROBE = textwrap.dedent("""
    import json, sys, time
    import pandas, spacr.tabular
    from spacr.qt.screens.app_screen import AppScreen
    before = {"utils": "spacr.utils" in sys.modules, "torch": "torch" in sys.modules}
    t = time.perf_counter()
    settings = AppScreen._load_settings_csv(sys.argv[1])
    read = time.perf_counter() - t
    after = {"utils": "spacr.utils" in sys.modules, "torch": "torch" in sys.modules}
    t = time.perf_counter()
    import spacr.utils
    imported = time.perf_counter() - t
    print(json.dumps({"before": before, "after": after, "read": read,
                      "import": imported, "n": len(settings)}))
""")


def test_reading_a_settings_csv_does_not_import_spacr_utils(tmp_path):
    """Self-calibrated: the read against the import it used to trigger."""
    path = _write(tmp_path / "measure_crop_settings.csv", ("Key", "Value"))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", QT_QPA_PLATFORM="offscreen")
    out = subprocess.run([sys.executable, "-c", _PROBE, path], cwd=str(REPO), env=env,
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-3000:]
    result = json.loads(out.stdout.strip().splitlines()[-1])
    if result["before"]["utils"]:
        pytest.skip("building the screen imported spacr.utils already")
    assert result["after"] == {"utils": False, "torch": False}
    assert result["n"] == len(_ROWS)
    assert result["read"] * 10 < result["import"], result
