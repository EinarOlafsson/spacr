"""One percent density reaches engines and widgets without a hidden clamp."""

import pytest

from spacr.qt.widgets import ambient


@pytest.mark.parametrize('density', [.01, .10, .50])
def test_density_percentages_survive_construct_set_and_widget_boundary(density, qapp):
    assert ambient.DENSITY_RANGE == (.01, 3.0)
    engine = ambient.make_engine('data_art_impulse_lens', 'spacr', '#101418',
                                 density=density, seed=42)
    assert engine.density == density
    engine.set_density(1.0)
    engine.set_density(density)
    assert engine.density == density
    widget = ambient.AmbientWidget(theme='data_art_impulse_lens', palette='spacr',
                                   density=density, seed=42)
    try:
        assert widget.density() == density
        assert widget.engine.density == density
        widget.set_density(1.0)
        widget.set_density(density)
        assert widget.density() == density
        assert widget.engine.density == density
    finally:
        widget.close()


def test_density_minimum_preserves_default_maximum_and_actual_advection_counts():
    engine = ambient.make_engine('data_art_genetic_advection', 'spacr', '#101418', seed=42)
    assert engine.density == 1.0
    counts = []
    for density in [.01, .10, .50]:
        engine.set_density(density)
        counts.append(engine.element_count(34000, 60000))
    assert counts == [340, 3400, 17000]
    engine.set_density(0)
    assert engine.density == .01
    engine.set_density(100)
    assert engine.density == 3.0
