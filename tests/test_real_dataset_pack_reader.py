"""The opt-in real-data E2E consumes the pack it was actually given.

Only the actual helper's AST is executed: collecting this file cannot import
the opt-in suite, inspect a user's dataset, download data, or start a model.
CSV discovery, coercion, migration and pipeline defaults remain real.
"""
from __future__ import annotations

import ast
import csv
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def load_settings_for():
    source = Path(__file__).with_name("test_e2e_real_dataset.py")
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    helpers = [node for node in tree.body
               if isinstance(node, ast.FunctionDef)
               and node.name == "_load_settings_for"]
    assert len(helpers) == 1, "the actual E2E settings helper moved"
    namespace = {"Path": Path}
    module = ast.Module(body=helpers, type_ignores=[])
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["_load_settings_for"]


@pytest.fixture
def shared_reader_calls(monkeypatch):
    from spacr.qt import settings_pack

    original = settings_pack.settings_from_pack
    calls = []

    def traced(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append((args, kwargs, result[1]))
        return result

    monkeypatch.setattr(settings_pack, "settings_from_pack", traced)
    return calls


def _pack(tmp_path, filename, rows):
    root = tmp_path / "settings"
    root.mkdir()
    with (root / filename).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Key", "Value"])
        writer.writerows(rows)
    return root


@pytest.mark.parametrize("app_key,filename", [
    ("mask", "gen_masks_settings.csv"),
    ("mask", "gen_mask_settings.csv"),
    ("measure", "crop_measure_settings.csv"),
    ("measure", "measure_crop_settings.csv"),
])
def test_real_dataset_helper_reads_published_pack_and_keeps_pipeline_defaults(
        tmp_path, load_settings_for, shared_reader_calls, capsys,
        app_key, filename):
    rows = [("src", "/another/machine/plate/merged"),
            ("channels", "[2, 0]"), ("plot", "false")]
    if app_key == "mask":
        rows += [("cell_channel", "2"), ("nucleus_channel", "None"),
                 ("timelapse", "true"), ("fps", "7")]
    else:
        rows += [("cell_mask_dim", "2"), ("nucleus_mask_dim", "None")]
    settings_root = _pack(tmp_path, filename, rows)
    copied_plate = tmp_path / "copied_plate"
    copied_plate.mkdir()

    settings = load_settings_for(app_key, settings_root, copied_plate)

    assert settings["src"] == str(copied_plate)
    assert settings["channels"] == [2, 0]
    assert settings["plot"] is False
    assert "Key" not in settings
    if app_key == "mask":
        assert settings["cell_channel"] == 2
        assert settings["nucleus_channel"] is None
        # The pipeline consumes these; the Mask GUI intentionally hides them.
        assert settings["timelapse"] is True
        assert settings["fps"] == 7
        assert settings["preprocess"] is True
    else:
        assert settings["cell_mask_dim"] == 2
        assert settings["nucleus_mask_dim"] is None
        assert settings["save_measurements"] is True
    assert len(shared_reader_calls) == 1
    args, kwargs, report = shared_reader_calls[0]
    assert args == (app_key, str(settings_root))
    assert kwargs["src"] == str(copied_plate)
    assert report.source == filename
    assert not report.dropped and not report.malformed
    assert filename in capsys.readouterr().out


@pytest.mark.parametrize("app_key,expected_file", [
    ("mask", "gen_masks_settings.csv"),
    ("measure", "crop_measure_settings.csv"),
])
def test_real_dataset_helper_refuses_an_absent_stage_pack(
        tmp_path, load_settings_for, app_key, expected_file):
    root = _pack(tmp_path, "unrelated_settings.csv", [("channels", "[2, 0]")])

    with pytest.raises(AssertionError) as failure:
        load_settings_for(app_key, root, tmp_path / "copied_plate")

    message = str(failure.value)
    assert str(root) in message
    assert expected_file in message
    assert "unrelated_settings.csv" in message
    assert "defaults" in message


def test_real_dataset_helper_migrates_names_and_reports_dropped_keys(
        tmp_path, load_settings_for, shared_reader_calls, capsys):
    root = _pack(tmp_path, "gen_masks_settings.csv", [
        ("cell_FT", "0.25"), ("cell_channel", "2"),
        ("a_setting_that_never_existed", "9"),
    ])

    settings = load_settings_for("mask", root, tmp_path / "copied_plate")

    assert settings["cell_flow_threshold"] == 0.25
    assert settings["cell_channel"] == 2
    assert "cell_FT" not in settings
    assert "a_setting_that_never_existed" not in settings
    assert len(shared_reader_calls) == 1
    report = shared_reader_calls[0][2]
    assert report.renamed == [("cell_FT", "cell_flow_threshold")]
    assert report.dropped == ["a_setting_that_never_existed"]
    message = capsys.readouterr().out
    assert "cell_FT" in message and "cell_flow_threshold" in message
    assert "a_setting_that_never_existed" in message
