"""spaCR's own console decoration must not be able to end a run.

``pretty_print_settings`` frames its output in box-drawing characters and
prefixes each category with ``▸`` (U+25B8). Every spaCR entry point prints it
through ``save_settings(..., show=True)`` before doing any work --
:func:`spacr.measure.measure_crop` does it at the top of every source folder,
ahead of the multiprocessing pool.

No Windows codepage encodes that character. Not cp1252 (the Western ANSI
default), not cp437 or cp850 (the OEM console defaults), not cp932 or cp936.
The box-drawing frame additionally fails on cp1252. On Windows that is fatal
the moment stdout is not a UTF-8 stream -- a redirected console, a batch-queue
child, ``spacr-run`` writing to a pipe -- and it takes the run down with a
``UnicodeEncodeError`` before the first field is read.

The same applies to what the *values* contain, which is why this is not only
about decoration: cp1252 cannot encode ``Δku80``, the standard *T. gondii*
parental strain, and cp932/cp936 cannot encode ``µm``. Both routinely appear
in a settings dict.

These tests stand a real cp1252 text stream in for ``sys.stdout``, which is
exactly what CPython gives a Windows process whose output is redirected.
"""

from __future__ import annotations

import io
import os
import sqlite3
import sys

import numpy as np
import pytest

from spacr import utils as U


def _codepage_stdout(encoding='cp1252'):
    """A text stream that behaves like a redirected Windows console."""
    return io.TextIOWrapper(io.BytesIO(), encoding=encoding,
                            errors='strict', newline='')


# ---------------------------------------------------------------------------
# the primitives
# ---------------------------------------------------------------------------

def test_console_encoding_reads_it_off_the_stream():
    assert U.console_encoding(_codepage_stdout()) == 'cp1252'


def test_console_encoding_falls_back_when_the_stream_declares_none():
    """A queue-backed GUI console has no ``.encoding``; assume UTF-8."""
    class _NoEncoding:
        def write(self, text):
            return len(text)

    assert U.console_encoding(_NoEncoding()) == 'utf-8'


def test_console_encoding_defaults_to_stdout(monkeypatch):
    monkeypatch.setattr(sys, 'stdout', _codepage_stdout('cp437'))
    assert U.console_encoding() == 'cp437'


@pytest.mark.parametrize('codepage', ['cp1252', 'cp437', 'cp850', 'cp932'])
def test_the_category_bullet_is_unprintable_in_every_windows_codepage(codepage):
    """The premise of the whole fix, asserted rather than assumed."""
    stream = _codepage_stdout(codepage)
    assert not U.console_can_encode('▸', stream)
    assert U.console_can_encode('Measurements', stream)


def test_box_drawing_survives_the_oem_codepages_but_not_cp1252():
    assert not U.console_can_encode('┌─┐', _codepage_stdout('cp1252'))
    assert U.console_can_encode('┌─┐', _codepage_stdout('cp437'))


def test_console_safe_replaces_what_cannot_be_encoded():
    out = U.console_safe('strain Δku80 at 0.65 µm', _codepage_stdout('cp932'))
    assert 'strain' in out and 'ku80' in out
    out.encode('cp932')  # must not raise -- that is the entire point


def test_console_safe_leaves_printable_text_untouched():
    text = 'strain RH at 0.65 um'
    assert U.console_safe(text, _codepage_stdout()) is text
    assert U.console_safe('Δku80', _codepage_stdout('utf-8')) == 'Δku80'


def test_console_safe_survives_a_stream_naming_a_codec_python_lacks():
    class _Bogus:
        encoding = 'not-a-real-codec'

    assert U.console_safe('Δku80', _Bogus()) == '?ku80'
    assert not U.console_can_encode('Δku80', _Bogus())


# ---------------------------------------------------------------------------
# the printer
# ---------------------------------------------------------------------------

def test_pretty_print_settings_degrades_to_ascii_on_cp1252(monkeypatch):
    fake = _codepage_stdout('cp1252')
    monkeypatch.setattr(sys, 'stdout', fake)
    U.pretty_print_settings({'src': '/data/plate1', 'n_jobs': 4},
                            title='Measure Crop Settings')
    fake.flush()
    text = fake.buffer.getvalue().decode('cp1252')
    assert '▸' not in text and '┌' not in text
    assert '> ' in text or '+---' in text
    assert 'Measure Crop Settings' in text
    assert 'n_jobs' in text


def test_pretty_print_settings_keeps_the_pretty_frame_on_utf8(capsys):
    """The nice output is the point; it must survive where it can be printed."""
    U.pretty_print_settings({'src': '/data/plate1'}, title='Run')
    out = capsys.readouterr().out
    assert out.startswith('┌')
    assert '│ Run' in out


def test_pretty_print_settings_does_not_raise_on_a_non_encodable_value(monkeypatch):
    """``pathogen: RH Δhxgprt`` is an ordinary spaCR setting."""
    fake = _codepage_stdout('cp1252')
    monkeypatch.setattr(sys, 'stdout', fake)
    U.pretty_print_settings({'pathogen': 'RH Δhxgprt', 'voxel_size_xy_um': '0.65 µm'},
                            title='Settings')
    fake.flush()
    assert b'hxgprt' in fake.buffer.getvalue()


def test_save_settings_show_does_not_raise_on_a_windows_console(monkeypatch, tmp_path):
    """``save_settings(show=True)`` is what ``measure_crop`` calls first."""
    fake = _codepage_stdout('cp437')
    monkeypatch.setattr(sys, 'stdout', fake)
    U.save_settings({'src': str(tmp_path), 'experiment': 'exp'},
                    name='measure_crop_settings', show=True)
    assert (tmp_path / 'settings' / 'measure_crop_settings.csv').is_file()


# ---------------------------------------------------------------------------
# and the run it used to kill
# ---------------------------------------------------------------------------

def test_measure_crop_completes_with_a_windows_console(tmp_path, monkeypatch):
    """End to end: a whole measure run whose stdout cannot encode cp1252.

    Before the fix this raised ``UnicodeEncodeError`` inside
    ``save_settings`` -- measure.py's first call after resolving settings, and
    well before the pool -- so nothing at all was measured.
    """
    from spacr.measure import measure_crop
    from spacr.settings import get_measure_crop_settings

    merged = tmp_path / 'merged'
    merged.mkdir(parents=True)
    (tmp_path / 'measurements').mkdir(parents=True)

    yy, xx = np.mgrid[:96, :96]
    cell = np.zeros((96, 96), np.uint16)
    nucleus = np.zeros((96, 96), np.uint16)
    for i, (cy, cx) in enumerate([(28, 28), (28, 68), (68, 28)], start=1):
        cell[(yy - cy) ** 2 + (xx - cx) ** 2 <= 14 ** 2] = i
        nucleus[(yy - cy) ** 2 + (xx - cx) ** 2 <= 5 ** 2] = i
    rng = np.random.default_rng(0)
    chans = []
    for _ in range(4):
        base = rng.integers(50, 200, size=(96, 96)).astype(np.uint16)
        base[cell > 0] += 3000
        chans.append(base)
    data = np.stack(chans + [cell, nucleus, np.zeros_like(cell)],
                    axis=-1).astype(np.uint16)
    np.save(merged / 'plate1_A01_F001.npy', data)

    settings = get_measure_crop_settings(settings={})
    settings.update({
        'src': str(merged), 'channels': [0, 1, 2, 3],
        'cell_mask_dim': 4, 'nucleus_mask_dim': 5, 'pathogen_mask_dim': None,
        'png_dims': [0, 1, 2], 'png_size': [32, 32],
        'save_measurements': True, 'save_png': False, 'save_arrays': False,
        'plot': False, 'verbose': False, 'timelapse': False,
        'crop_mode': ['cell'], 'normalize': [1, 99], 'normalize_by': 'png',
        # A real, spaCR-shaped value that cp1252 cannot encode.
        'experiment': 'RH Δku80', 'n_jobs': 1, 'test_mode': False,
    })

    fake = _codepage_stdout('cp1252')
    monkeypatch.setattr(sys, 'stdout', fake)
    try:
        measure_crop(settings)
    finally:
        monkeypatch.undo()

    db = tmp_path / 'measurements' / 'measurements.db'
    assert db.is_file(), 'the run died before it wrote anything'
    con = sqlite3.connect(db)
    try:
        assert con.execute('SELECT COUNT(*) FROM cell').fetchone()[0] == 3
    finally:
        con.close()
