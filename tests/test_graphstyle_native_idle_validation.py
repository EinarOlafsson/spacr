"""Live validation coalesces redraws while file writes always observe current marks."""
import numpy as np
import pytest
from spacr.figures import style
from spacr.qt.widgets.fast_plots import PValueHistogram

pytestmark = pytest.mark.qt


def test_a_burst_of_native_redraws_invalidates_old_certification_and_observes_only_the_latest_scene(qtbot, monkeypatch):
    """Three synchronous histogram updates must queue one real observation after the event loop resumes; neither a stale certificate nor an intermediate count array may be published."""
    widget = PValueHistogram()
    qtbot.addWidget(widget)
    widget.resize(760, 560)
    widget.show()
    qtbot.waitExposed(widget)
    real = style.finalize_figure
    observed = []

    def capture(figure, *args, **kwargs):
        """Retain actual latest histogram counts at the real finalization boundary."""
        if figure is widget:
            observed.append(widget._counts.copy())
        return real(figure, *args, **kwargs)

    monkeypatch.setattr(style, 'finalize_figure', capture)
    widget._spacr_style_diagnostics = style.FigureDiagnostics(style.POLICY_VERSION, (), ('completed presentation checks',))
    for values in ([.1,.2,.3], [.4,.5,.6], [.8,.9,.95]):
        widget.set_p_values(values, bins=3)
        assert not widget._spacr_style_diagnostics.compliant
        assert 'Validate the updated native plot' in widget._spacr_style_diagnostics.problems[0]
    assert not observed
    expected = widget._counts.copy()
    qtbot.waitUntil(lambda: len(observed) == 1)
    np.testing.assert_array_equal(observed[0], expected)
    np.testing.assert_array_equal(expected, [0,0,3])
    assert not widget._spacr_style_validation_timer.isActive()
    assert 'actual native plot area' in widget._spacr_style_diagnostics.checks


def test_native_export_validates_current_scene_before_the_writer_even_while_idle_validation_is_pending(qtbot, monkeypatch, tmp_path):
    """An immediate export cannot use pending or old diagnostics: its actual writer boundary sees synchronous mark checks and cancels the redundant idle timer while source counts stay exact."""
    widget = PValueHistogram()
    qtbot.addWidget(widget)
    widget.resize(760, 560)
    widget.set_p_values([.1,.2,.8], bins=3)
    expected = widget._counts.copy()
    assert widget._spacr_style_validation_timer.isActive()
    written = []

    def capture(item, path, width, height, exporters):
        """Read the real diagnostics at the synchronous file-painter boundary."""
        written.append(widget._spacr_style_diagnostics)
        assert not widget._spacr_style_validation_timer.isActive()
        np.testing.assert_array_equal(widget._counts, expected)

    monkeypatch.setattr(widget, '_write_export', capture)
    widget.export(str(tmp_path/'current.svg'))
    assert len(written) == 1
    assert 'actual native plot area' in written[0].checks
    assert not any('updated native plot before certifying' in value for value in written[0].problems)
    np.testing.assert_array_equal(widget._counts, expected)
