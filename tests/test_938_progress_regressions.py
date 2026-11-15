"""Item 938: the Progress lines the run bar reads count the whole run.

Each test pins one defect the timing audit found: a total that restarted,
a count that never reached its total, or a Time_left computed with the wrong
unit.
"""
import re

import numpy as np
import pytest

from spacr import deep_spacr, io, plot, utils


def _progress(text):
    return [(int(a), int(b)) for a, b in re.findall(r"Progress: (\d+)/(\d+)", text)]


def test_time_left_counts_the_batches_left_when_the_time_is_per_batch(capsys):
    utils.print_progress(10, 100, 1, time_ls=[2.0], batch_size=10, operation_type="x")
    out = capsys.readouterr().out
    # 90 images left = 9 batches of 2 s = 18 s = 0.300 min
    assert "Time_left: 0.300 min" in out
    assert "Time/batch: 2.000sec" in out


def test_time_left_is_per_image_when_there_is_no_batch_size(capsys):
    utils.print_progress(10, 100, 2, time_ls=[2.0], operation_type="x")
    # 90 images x 2 s over 2 jobs = 90 s = 1.500 min
    assert "Time_left: 1.500 min" in capsys.readouterr().out


def test_normalizing_counts_across_all_the_batches_of_a_run(capsys):
    for batch in range(3):
        report = io._normalizing_reporter(batch, 3)
        for channel in (1, 2):
            report(channel, 2)
    lines = _progress(capsys.readouterr().out)
    assert lines == [(n, 6) for n in range(1, 7)]


def test_normalizing_reporter_names_the_stage(capsys):
    io._normalizing_reporter(0, 1)(1, 1)
    assert "operation_type: Normalizing" in capsys.readouterr().out


def test_the_activation_count_reaches_its_total_on_the_last_batch():
    counts = [deep_spacr._activation_progress_counts(i, 8, 4) for i in range(4)]
    assert counts == [(8, 32), (16, 32), (24, 32), (32, 32)]


def test_a_partial_last_batch_never_counts_past_the_total():
    # the loader has 3 batches of 8 slots: the count is capped, never above
    assert deep_spacr._activation_progress_counts(2, 8, 3) == (24, 24)


def _scimg_dir(tmp_path):
    from PIL import Image
    src = tmp_path / "plate" / "cell"
    src.mkdir(parents=True)
    for index in range(2):
        Image.fromarray(np.full((16, 16), 200, dtype=np.uint8)).save(src / f"c{index}.png")
    return str(src)


def test_saved_figures_are_numbered_across_folders_not_always_one_of_one(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import spacr.io

    seen = []

    def fake_save(fig, src, text, dpi=300, i=1, all_folders=1):
        seen.append((text, i, all_folders))
        plt.close(fig)

    def fake_grid(*args, **kwargs):
        return plt.figure(figsize=(1, 1))

    monkeypatch.setattr(spacr.io, "_save_figure", fake_save)
    monkeypatch.setattr(plot, "_plot_images_on_grid", fake_grid)
    plot._save_scimg_plot(_scimg_dir(tmp_path), channel_indices=[0, 1],
                          standardize=False, i=2, all_folders=3)
    # one folder = the all-channels figure plus one per channel = 3 figures;
    # this is the second of three folders, so figures 4, 5 and 6 of 9
    assert seen == [("all_channels", 4, 9), ("channel_0", 5, 9), ("channel_1", 6, 9)]
