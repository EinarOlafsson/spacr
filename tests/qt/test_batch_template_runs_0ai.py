"""Batch Runner template runs (item 0ai).

A finished run folder is imported as a template; new image folders become
jobs once they match it; a Mask+Measure template makes two chained jobs per
folder; a job opens on its module screen and every edit is saved to the job
settings file; the queue runs Mask before Measure. Nothing real is run: the
queue uses a fake runner.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from spacr import batch as bt
from spacr.settings import SETTINGS_SCHEMA_VERSION, SETTINGS_VERSION_KEY


class FakeRunner:
    """Records the order jobs run in; runs nothing."""

    def __init__(self):
        self.calls = []

    def __call__(self, job, settings_path, log_path):
        self.calls.append((job.id, job.module))
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text(f"log for {job.id}\n", encoding="utf-8")
        return 0


def _plate(root, name, channels=(1, 2, 3), shape=(48, 64), dtype=np.uint16,
           pattern="W{well}F0001T0001Z000C{channel}.tif"):
    folder = Path(root) / name
    folder.mkdir(parents=True, exist_ok=True)
    for well in ("0001", "0002"):
        for channel in channels:
            tifffile.imwrite(folder / pattern.format(well=well, channel=channel),
                             np.zeros(shape, dtype))
    return folder


def _template(root, measure=True):
    run = _plate(root, "template_run")
    settings = run / "settings"
    settings.mkdir()
    mask = {"src": str(run), "metadata_type": "cq1", "nucleus_channel": 0,
            "pathogen_channel": 2, "cell_channel": None, "nucleus_background": 100,
            "_mask_batch_membership": {"x": 1}, SETTINGS_VERSION_KEY: SETTINGS_SCHEMA_VERSION}
    (settings / "gen_mask_settings.json").write_text(json.dumps(mask), encoding="utf-8")
    (settings / "gen_mask_settings.csv").write_text(
        "Key,Value\nsrc,/elsewhere\n", encoding="utf-8")
    if measure:
        values = {"src": str(run), "nucleus_mask_dim": 3,
                  SETTINGS_VERSION_KEY: SETTINGS_SCHEMA_VERSION}
        (settings / "measure_crop_settings.json").write_text(
            json.dumps(values), encoding="utf-8")
    return run


@pytest.fixture
def screen(qtbot, qt_theme_applied, tmp_path):
    from spacr.qt.screens.batch import BatchScreen

    widget = BatchScreen(threaded=False, runner=FakeRunner())
    widget._template_job_dir = str(tmp_path / "jobs")
    qtbot.addWidget(widget)
    return widget


def test_a_mask_only_template_reads_its_mask_settings(tmp_path):
    run = _template(tmp_path, measure=False)
    template = bt.read_template_run(run)
    assert template.modules == ("mask",)
    assert template.files["mask"].endswith("gen_mask_settings.json")
    assert template.settings["mask"]["nucleus_background"] == 100
    assert "_mask_batch_membership" not in template.settings["mask"]


def test_a_mask_and_measure_template_reads_both_in_run_order(tmp_path):
    run = _template(tmp_path)
    assert bt.read_template_run(run / "settings").modules == ("mask", "measure")


def test_a_folder_without_settings_is_not_a_template(tmp_path):
    with pytest.raises(bt.TemplateError, match="No module settings"):
        bt.read_template_run(_plate(tmp_path, "raw"))


def test_one_folder_makes_a_mask_job_and_a_chained_measure_job(screen, tmp_path):
    assert screen._import_template(str(_template(tmp_path)))
    new = _plate(tmp_path, "plate_b")
    assert screen._add_template_folders([str(new)]) == 1
    mask, measure = screen.queue().jobs
    assert (mask.module, measure.module) == ("mask", "measure")
    assert measure.depends_on == [mask.id]
    stored = bt.read_job_settings(mask)
    assert stored["src"] == str(new)
    assert stored["nucleus_background"] == 100
    assert stored["pathogen_channel"] == 2
    assert bt.read_job_settings(measure)["src"] == str(new)
    assert screen._template_jobs.count() == 2


def test_three_folders_make_three_job_pairs(screen, tmp_path):
    screen._import_template(str(_template(tmp_path)))
    folders = [str(_plate(tmp_path, f"plate_{n}")) for n in "cde"]
    assert screen._add_template_folders(folders) == 3
    jobs = screen.queue().jobs
    assert [j.module for j in jobs] == ["mask", "measure"] * 3
    for mask, measure in zip(jobs[::2], jobs[1::2]):
        assert measure.depends_on == [mask.id]
        assert bt.read_job_settings(mask)["src"] == bt.read_job_settings(measure)["src"]
    assert sorted(bt.read_job_settings(j)["src"] for j in jobs[::2]) == sorted(folders)


def test_a_folder_that_does_not_match_is_refused_in_plain_words(screen, tmp_path):
    screen._import_template(str(_template(tmp_path)))
    fewer = _plate(tmp_path, "two_channels", channels=(1, 2), shape=(32, 32))
    other = _plate(tmp_path, "other_names", pattern="img_{well}_c{channel}.tif")
    assert screen._add_template_folders([str(fewer), str(other)]) == 0
    assert not screen.queue().jobs
    problems = screen.problems_text()
    assert "The channels differ: the template has 1, 2, 3, this folder has 1, 2." in problems
    assert "The image size differs" in problems
    assert "do not follow the file-name pattern of the template (cq1)" in problems
    assert screen.last_error
    assert not list((tmp_path / "jobs").glob("*.json"))


def test_the_pixel_type_and_z_planes_are_checked(tmp_path):
    template = bt.read_template_run(_template(tmp_path))
    eight_bit = _plate(tmp_path, "eight_bit", dtype=np.uint8)
    assert any("pixel type differs" in p for p in bt.check_template_folder(template, eight_bit))
    stacks = _plate(tmp_path, "stacks", pattern="W{well}F0001T0001Z001C{channel}.tif")
    _plate(tmp_path, "stacks", pattern="W{well}F0001T0001Z002C{channel}.tif")
    assert any("z planes differ" in p for p in bt.check_template_folder(template, stacks))


def test_the_queue_runs_mask_before_measure(screen, tmp_path):
    screen._import_template(str(_template(tmp_path)))
    screen._add_template_folders([str(_plate(tmp_path, "plate_b"))])
    assert screen.run()
    assert [module for _id, module in screen._runner.calls] == ["mask", "measure"]
    assert all(j.status == bt.STATUS_SUCCESS for j in screen.queue().jobs)


def test_the_drop_sections_take_a_run_folder_and_image_folders(screen, tmp_path):
    from spacr.qt.dnd_handlers import _TemplateJobsDropHandler, _TemplateRunDropHandler

    run = _template(tmp_path)
    raw = _plate(tmp_path, "plate_b")
    assert _TemplateRunDropHandler().can_accept(run)
    assert not _TemplateRunDropHandler().can_accept(raw)
    _TemplateRunDropHandler().apply(run, screen)
    assert screen._template is not None
    assert _TemplateJobsDropHandler().apply_all([raw], screen)
    assert len(screen.queue().jobs) == 2


def test_editing_a_job_on_its_module_screen_saves_to_the_job(qtbot, qt_theme_applied,
                                                             tmp_path):
    from spacr.qt import app as qt_app

    window = qt_app.MainWindow()
    qtbot.addWidget(window)
    window.open_module("batch")
    batch = window._screens["batch"]
    batch._template_job_dir = str(tmp_path / "jobs")
    batch.set_runner(FakeRunner())
    assert batch._import_template(str(_template(tmp_path)))
    batch._add_template_folders([str(_plate(tmp_path, "plate_b"))])
    mask = batch.queue().jobs[0]

    batch._template_jobs.itemClicked.emit(batch._template_jobs.item(0))
    screen = window._screens["mask"]
    model = screen._settings_model
    assert model.collect()["nucleus_background"] == 100
    assert "mask-1" in screen._batch_job_bar.text()

    widget = model._widgets["nucleus_background"]
    assert model.set_value_for_key("nucleus_background", 250)
    widget.editingFinished.emit()
    assert bt.read_job_settings(mask)["nucleus_background"] == 250
    assert json.loads(Path(mask.settings_path).read_text())["nucleus_background"] == 250

    screen._batch_job_bar._stop()
    assert window._batch_job_editor is None
    model.set_value_for_key("nucleus_background", 300)
    widget.editingFinished.emit()
    assert bt.read_job_settings(mask)["nucleus_background"] == 250
