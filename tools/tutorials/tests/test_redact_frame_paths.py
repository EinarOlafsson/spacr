"""Offline tests for the frame path redaction tool (item 447); no OCR model is loaded."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

MODULE = Path(__file__).resolve().parents[1] / 'redact_frame_paths.py'
spec = importlib.util.spec_from_file_location('redact_frame_paths', MODULE)
redact = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = redact  # dataclasses resolve their module
spec.loader.exec_module(redact)

STAGE_ROOT = '/mnt/disk9/Someone/toxoplasma_projects/tutorials/refresh_2026-09-09'


@pytest.mark.parametrize('token, expected, kind', [
    (STAGE_ROOT + '/replication_runs/X', '/data/example/replication_runs/X', 'local_root->/data/example'),
    ('ude/toxoplasma_projects/tutorials/refresh_2026-09-09/runs/a.csv', '/data/example/runs/a.csv',
     'local_root->/data/example'),
    ('/home/olafsson/.spacr/runs/2026-09-10_run', '/home/user/.spacr/runs/2026-09-10_run',
     'account_home->/home/user'),
    ('/mnt/fi', '/data/example', 'local_root->/data/example'),
    ('olafsson', 'user', 'account_name->user'),
])
def test_neutral_path_keeps_trailing_names(token, expected, kind):
    replacement, found = redact.neutral_path(token)
    assert (replacement, found) == (expected, kind)
    assert 'olafsson' not in found and 'mnt' not in found  # kinds never carry the original text


def test_neutral_paths_and_public_names_are_left_alone():
    assert redact.path_spans('Output /tmp/spacr-tutorials/profiler_runs/a.csv') == []
    assert redact.path_spans('Author: Einar Olafsson  https://einarolafsson.github.io/spacr') == []
    assert redact.path_spans('/home/user/.cache/spacr/example_data') == []


def test_span_starts_at_the_root_when_ocr_drops_a_space():
    text = 'Results saved to/home/olafsson/.cache/x.csv'
    [(start, end, token)] = redact.path_spans(text)
    assert text[start:end] == token == '/home/olafsson/.cache/x.csv'


def _font():
    for name in ('OpenSans-Light.ttf', 'OpenSans-Regular.ttf'):
        for path in redact.font_files():
            if path.endswith(name):
                return path
    pytest.skip('no Open Sans font available')


def _line_image(text, font_path, size=20, width=1900, height=60):
    """A console-like line on a smooth dark gradient, with exact character boxes."""
    ramp = np.linspace(0, 1, width, dtype=np.float32)[None, :, None]
    base = (np.array([43, 32, 53], np.float32) * (1 - ramp) + np.array([28, 40, 44], np.float32) * ramp)
    image = Image.fromarray(np.repeat(base, height, axis=0).astype(np.uint8))
    font = ImageFont.truetype(font_path, size)
    x0, baseline = 40, 38
    ImageDraw.Draw(image).text((x0, baseline), text, font=font, fill=(72, 155, 248), anchor='ls')
    chars = []
    for index in range(len(text)):
        left = x0 + font.getlength(text[:index])
        right = x0 + font.getlength(text[:index + 1])
        chars.append([int(left), baseline - 16, int(np.ceil(right)), baseline + 5])
    line = {'text': text, 'box': [x0, baseline - 17, int(x0 + font.getlength(text)), baseline + 6],
            'chars': chars}
    return np.asarray(image).copy(), line, font


def _similarity(a, b):
    """Correlation of lightly blurred luminance (sub-pixel placement is not the question)."""
    import cv2
    a = cv2.GaussianBlur(a.astype(np.float32).mean(-1), (0, 0), 1.5)
    b = cv2.GaussianBlur(b.astype(np.float32).mean(-1), (0, 0), 1.5)
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def test_redacts_only_the_local_root_and_moves_the_rest():
    font_path = _font()
    original = f'Saved to {STAGE_ROOT}/runs/a.csv and done'
    pixels, line, _ = _line_image(original, font_path)
    before = pixels.copy()
    records = redact.redact_line(pixels, line)
    assert [r.kind for r in records] == ['local_root->/data/example']
    assert records[0].shift > 0 and records[0].fit_score >= redact.MIN_FIT
    assert redact.changed_outside(before, pixels, records) == 0
    expected, _, _ = _line_image('Saved to /data/example/runs/a.csv and done', font_path)
    assert _similarity(pixels, expected) > 0.9
    assert _similarity(pixels, expected) > _similarity(before, expected)
    # the words before the path are untouched
    assert np.array_equal(before[:, :40 + 60], pixels[:, :40 + 60])


def test_account_name_in_a_home_path():
    font_path = _font()
    pixels, line, _ = _line_image('Reproducibility manifest: /home/olafsson/.spacr/runs/r1', font_path)
    before = pixels.copy()
    records = redact.redact_line(pixels, line)
    assert [r.kind for r in records] == ['account_home->/home/user']
    assert redact.changed_outside(before, pixels, records) == 0
    expected, _, _ = _line_image('Reproducibility manifest: /home/user/.spacr/runs/r1', font_path)
    assert _similarity(pixels, expected) > 0.9


def test_changed_outside_counts_pixels_beyond_the_boxes():
    before = np.zeros((10, 10, 3), np.uint8)
    after = before.copy()
    after[1, 1] = 9
    after[8, 8] = 9
    record = redact.Redaction(box=[0, 0, 4, 4], kind='k', font='f', size=1, fit_score=1, shift=0,
                              clipped_left=False, replacement_len=1, original_len=1)
    assert redact.changed_outside(before, after, [record]) == 1


def test_account_inside_a_name_and_wrapped_rows():
    [(start, end, token)] = redact.path_spans('/tmp/pytest-of-olafsson/pytest-8/x')
    assert token.startswith('olafsson') and redact.neutral_path(token)[0] == 'user/pytest-8/x'
    above = {'text': 'saved to /mnt/disk9/Someone/toxo', 'box': [100, 100, 900, 125]}
    below = {'text': 'plasma_projects/tutorials/refresh_2026-09-09/runs/a', 'box': [100, 128, 900, 153]}
    assert redact.is_continuation(below, [above, below])
    assert not redact.is_continuation(above, [above, below])
