"""The demo generator clips, fills and refuses rather than producing nonsense.

The synthetic datasets are what a new user's first run is made of, so every
edge in them has to produce something a pipeline can actually read:

* an object whose centre has drifted off the field contributes nothing rather
  than indexing backwards into the array, which in numpy is a silent write at
  the wrong end of the image;
* a channel the caller asked for that carries none of the four roles is filled
  with background, because a merged stack missing a plane is a KeyError deep
  inside measure;
* a barcode pool is unique by construction, so two wells cannot be handed the
  same barcode;
* a read whose adapter constants no longer add up stops the generator instead
  of writing a FASTQ that maps to nothing.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest

from spacr.qt import synthetic


def test_a_spot_that_has_left_the_field_draws_nothing():
    """A Gaussian centred off the image leaves only background behind."""
    rng = np.random.default_rng(0)

    empty = synthetic._draw_spots((16, 16), [], rng)
    gone = synthetic._draw_spots(
        (16, 16), [(-500.0, 8.0, 2.0, 40000.0)], np.random.default_rng(0))

    assert gone.shape == (16, 16)
    assert int(gone.max()) < 40000
    assert gone.dtype == empty.dtype


def test_a_disc_that_has_left_the_field_paints_nothing():
    """A label whose centre is off the mask leaves the mask untouched."""
    mask = np.zeros((16, 16), dtype=np.uint16)

    synthetic._paint_disc(mask, -500.0, 8.0, 4.0, 3)

    assert not mask.any()


def test_a_disc_inside_the_field_is_painted():
    """The same helper does paint when the disc overlaps the mask."""
    mask = np.zeros((16, 16), dtype=np.uint16)

    synthetic._paint_disc(mask, 8.0, 8.0, 4.0, 3)

    assert set(np.unique(mask)) == {0, 3}


def test_a_channel_with_no_role_is_filled_with_background():
    """An extra channel gets a plane, so nothing downstream hits a KeyError."""
    field = synthetic._synth_field(seed=0, channels=(0, 1, 2, 3, 7),
                                   shape=(64, 64))

    assert 7 in field.images
    assert field.images[7].shape == (64, 64)
    assert int(field.images[7].max()) < 40000


def test_a_barcode_pool_is_unique_even_when_the_space_is_small():
    """Redrawing a barcode already in the pool does not put it there twice.

    Two wells sharing a barcode makes their reads indistinguishable, which is
    a silently wrong mapping table rather than a failure.
    """
    pool = synthetic.barcode_pool(20, 4, seed=0)

    assert len(pool) == 20
    assert len(set(pool)) == 20
    assert all(len(bc) == 4 for bc in pool)


def test_an_unknown_app_gets_only_the_shared_settings():
    """A key with no tailored block falls back to the shared base."""
    settings = synthetic.demo_settings("not_an_app", "/demo")

    assert settings["src"] == "/demo"
    assert settings["channels"] == [0, 1, 2, 3]
    assert "grna_csv" not in settings
    assert "timelapse" not in settings


def test_adapter_constants_that_no_longer_add_up_stop_the_generator(
        monkeypatch):
    """A window of the wrong length raises rather than writing bad reads.

    Every downstream field is positioned by offset from the window start, so
    a window one base long writes a FASTQ that maps to nothing at all.
    """
    monkeypatch.setattr(synthetic, "SEQ_TAIL", synthetic.SEQ_TAIL + "A")

    column = "A" * synthetic.WELL_BARCODE_LENGTH
    row = "C" * synthetic.WELL_BARCODE_LENGTH
    grna = "G" * synthetic.GRNA_LENGTH

    with pytest.raises(ValueError, match="no longer add up"):
        synthetic.synthetic_read(column, grna, row)


def test_a_well_formed_read_is_the_documented_length():
    """The unmodified constants do produce a full-length read."""
    column = "A" * synthetic.WELL_BARCODE_LENGTH
    row = "C" * synthetic.WELL_BARCODE_LENGTH
    grna = "G" * synthetic.GRNA_LENGTH

    read = synthetic.synthetic_read(column, grna, row)

    assert len(read) == synthetic.FASTQ_READ_LENGTH
    assert grna in read


def test_the_cli_generates_one_named_demo(tmp_path, capsys):
    """Naming one app builds that app's dataset in the given folder."""
    assert synthetic.main(["mask", str(tmp_path)]) == 0

    assert (tmp_path / "1").exists() or any(tmp_path.iterdir())


def test_the_module_entry_point_exits_with_the_cli_status(tmp_path,
                                                          monkeypatch):
    """``python -m spacr.qt.synthetic`` exits with the CLI's return code.

    The guard at the bottom of the file is the whole command-line surface;
    executing the module as ``__main__`` is the only way to prove it wires
    ``main`` to the process exit status rather than dropping it.
    """
    import runpy

    monkeypatch.setattr(sys, "argv", ["synthetic", "mask", str(tmp_path)])

    with pytest.raises(SystemExit) as exited:
        runpy.run_path(synthetic.__file__, run_name="__main__")

    assert exited.value.code == 0
    assert any(tmp_path.iterdir())
