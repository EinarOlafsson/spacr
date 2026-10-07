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


def test_density_changes_the_number_and_spacing_of_actual_advection_grains(monkeypatch):
    """Even at native high Detail, more ink comes from more seeded trails."""
    from PySide6.QtGui import QImage

    engine = ambient.make_engine('data_art_genetic_advection', 'spacr', '#101418', seed=42)
    engine.set_max_pixels(960 * 540)
    engine.set_time(7.0)
    captured = []

    def capture(width, height, x, y, intensity, spread=False):
        captured.append((np.asarray(x).copy(), np.asarray(y).copy(),
                         np.asarray(intensity).copy()))
        image = QImage(width, height, QImage.Format_RGB32)
        image.fill(engine.identity)
        return image

    monkeypatch.setattr(engine, '_point_material', capture)
    for detail in (1.0, 2.0):
        engine.set_resolution(detail)
        populations = []
        occupied = []
        for density in (.01, .10, 1.0, 2.0, 3.0):
            engine.set_density(density)
            engine.shade(960, 540)
            x, y, light = captured[-1]
            populations.append(x.shape[-1])
            occupied.append(np.unique(y[0] * 960 + x[0]).size)
            assert np.isfinite(light).all()
        assert populations == [340, 3400, 34000, 47000, 60000]
        assert all(occupied[i] < occupied[i + 1]
                   for i in range(len(occupied) - 1))
        assert all(np.array_equal(captured[i][0][:, :populations[i]],
                                  captured[i + 1][0][:, :populations[i]])
                   for i in range(len(populations) - 1))
        captured.clear()
