"""The deterministic advection geometry changes its local flow over time."""

import numpy as np

from spacr.qt.widgets import ambient


def test_advection_structure_and_velocity_change_without_changing_seed(monkeypatch):
    engine = ambient.make_engine('data_art_genetic_advection', 'spacr', '#101418', seed=42)
    snapshots = []

    def capture(width, height, x, y, intensity, spread=False):
        snapshots.append((np.asarray(x).copy(), np.asarray(y).copy()))
        from PySide6.QtGui import QImage

        image = QImage(width, height, QImage.Format_RGB32)
        image.fill(engine.identity)
        return image

    monkeypatch.setattr(engine, '_point_material', capture)
    for stamp in [1, 1.2, 31, 31.2, 1]:
        engine.set_time(stamp)
        engine.shade(960, 540)
    early = snapshots[1][0] - snapshots[0][0]
    late = snapshots[3][0] - snapshots[2][0]
    assert np.count_nonzero(early != late) > early.size / 10
    assert np.array_equal(snapshots[0][0], snapshots[-1][0])
    assert np.array_equal(snapshots[0][1], snapshots[-1][1])
    assert all(x.size == 34000 * 12 for x, _ in snapshots)
