"""A save location of None must never become a file called ``None.pdf``.

``spacr.plot.save_figure`` formatted its ``path`` argument with ``str()``,
so a caller that passed None got ``None.pdf`` in the working directory.
Its contract for None is the one ``spacr.figure_sink.publish`` states: no
file was requested, so nothing is written. ``visualize_cellpose_masks``
with ``save=True`` and no ``filename`` built ``results/None.pdf`` the same
way; its contract is the documented ``<src>/results/<filename>.pdf``
location, so the figure lands there under a default stem.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _listing(root):
    found = []
    for folder, _dirs, files in os.walk(root):
        found.extend(os.path.join(folder, name) for name in files)
    return sorted(found)


def test_save_figure_with_no_path_writes_nothing(tmp_path, monkeypatch):
    from spacr.plot import save_figure

    monkeypatch.chdir(tmp_path)
    fig, _ax = plt.subplots()
    try:
        assert save_figure(fig, None) is None
    finally:
        plt.close(fig)
    assert _listing(tmp_path) == []
    assert not list(tmp_path.glob("None.*"))


def test_visualize_masks_without_filename_saves_under_results(tmp_path, monkeypatch):
    from spacr.plot import visualize_cellpose_masks

    monkeypatch.chdir(tmp_path)
    visualize_cellpose_masks([np.zeros((6, 6), dtype=int)], save=True)
    written = _listing(tmp_path)
    assert written, "save=True must still write the figure"
    assert not [p for p in written if os.path.basename(p).startswith("None.")]
    assert all(os.path.dirname(p) == str(tmp_path / "results") for p in written)
    plt.close("all")
