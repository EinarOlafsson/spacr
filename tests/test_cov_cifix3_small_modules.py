"""Coverage for small guard branches of illumination, normalization, notebook export, gene comparison and the classifier utilities."""
import numpy as np
import pandas as pd
import pytest

from spacr import illumination
from spacr.illumination import IlluminationCorrector
from spacr.normalization import normalization_stats


def _corrector(verbose):
    corrector = IlluminationCorrector.__new__(IlluminationCorrector)
    corrector.verbose = verbose
    corrector._warned = 0
    return corrector


class _Ctx:
    file_name = "field.tif"


def test_a_quiet_corrector_reports_no_rescale(capsys):
    """A corrector built with verbose off prints nothing and counts no warning."""
    corrector = _corrector(False)
    corrector._report_rescale(0.5, _Ctx())
    assert capsys.readouterr().out == ""
    assert corrector._warned == 0


def test_rescale_warnings_stop_after_the_limit(capsys):
    """Only the first few rescales are printed and later ones stay silent."""
    corrector = _corrector(True)
    for _ in range(illumination._MAX_CLIP_WARNINGS + 3):
        corrector._report_rescale(0.5, _Ctx())
    out = capsys.readouterr().out
    assert out.count("illumination correction rescaled") == illumination._MAX_CLIP_WARNINGS
    assert corrector._warned == illumination._MAX_CLIP_WARNINGS + 3


@pytest.mark.parametrize("mode", [
    "not-a-mode",   # a mistyped mode gives the same statistics as symmetric
    "auto",         # the auto mode resolves to the symmetric statistics
])
def test_an_unknown_or_auto_normalization_means_symmetric(mode):
    assert normalization_stats(mode) == normalization_stats("symmetric")


def test_a_setting_whose_literal_differs_has_no_notebook():
    """A value whose repr evaluates to something unequal yields no literal."""
    from spacr.notebook_export import _settings_literal

    class Odd(str):
        def __repr__(self):
            return "'other'"

    assert _settings_literal({"a": Odd("x")}) is None
    assert _settings_literal({"a": float("nan")}) is None
    assert _settings_literal({"a": [1, "x"]}) is not None


def test_a_notebook_is_written_to_a_given_folder(tmp_path):
    """An explicit folder is used as given and the notebook lands in it."""
    from spacr.notebook_export import _write_launch_notebook

    folder = tmp_path / "out"
    path = _write_launch_notebook("replication", {"src": "x"}, folder=folder)
    assert path is not None
    assert path.parent == folder
    assert path.exists()


def test_the_calibrated_null_needs_control_cells():
    """With no finite control cells the calibrated null cannot be estimated."""
    from spacr.gene_measurement_compare import _calibrated_null

    guides = pd.DataFrame({"f": [1.0, 2.0, 3.0], "n_cells": [5, 5, 5]})
    assert _calibrated_null(pd.Series([np.nan, np.nan]), guides, "f") is None


def test_a_classifier_needs_at_least_one_channel():
    """A custom classifier refuses zero input channels."""
    from spacr.utils import CustomCellClassifier

    with pytest.raises(ValueError, match="at least 1"):
        CustomCellClassifier(in_channels=0)


def test_stem_adaptation_variants():
    """The stem keeps three channels, widens with mean filters, narrows, or refuses."""
    import torch.nn as nn
    from spacr.utils import _adapt_stem_channels

    def net(bias):
        return nn.Sequential(nn.ReLU(), nn.Sequential(nn.Conv2d(3, 4, 3, bias=bias)))

    same = net(True)
    assert _adapt_stem_channels(same, 3) is same
    assert same[1][0].in_channels == 3
    wide = _adapt_stem_channels(net(True), 5)
    conv = wide[1][0]
    assert conv.in_channels == 5
    assert np.allclose(conv.weight[:, 3].detach().numpy(),
                       conv.weight[:, :3].mean(dim=1).detach().numpy())
    nobias = _adapt_stem_channels(net(False), 2)
    assert nobias[1][0].in_channels == 2 and nobias[1][0].bias is None
    with pytest.raises(ValueError, match="No 3-channel"):
        _adapt_stem_channels(nn.Sequential(nn.Conv2d(1, 2, 3)), 4)


def test_the_calibrated_null_needs_some_variance():
    """Constant cells and equal guide medians leave no variance to calibrate."""
    from spacr.gene_measurement_compare import _calibrated_null

    guides = pd.DataFrame({"f": [2.0, 2.0, 2.0], "n_cells": [5, 5, 5]})
    assert _calibrated_null(pd.Series([2.0, 2.0, 2.0]), guides, "f") is None
