"""The capture-host clip gate on small synthetic clips with the real cursor sprite."""
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
pytest.importorskip('cv2')
if shutil.which('ffmpeg') is None or not Path('/usr/share/icons/Yaru/cursors/left_ptr').exists():
    pytest.skip('needs ffmpeg and the Yaru cursor theme', allow_module_level=True)
import capture_clip_gate  # noqa: E402
from measure_capture_host_proof import xcursor_image  # noqa: E402

W, H, FPS = 640, 360, 30
SPRITE, HOT = xcursor_image(Path('/usr/share/icons/Yaru/cursors/left_ptr').resolve(), 32)


def draw_cursor(frame, x, y):
    h, w = SPRITE.shape[:2]
    x0, y0 = int(x) - HOT[0], int(y) - HOT[1]
    region = frame[y0:y0 + h, x0:x0 + w].astype(np.float32)
    alpha = SPRITE[..., 3:4].astype(np.float32) / 255.0
    frame[y0:y0 + h, x0:x0 + w] = (SPRITE[..., :3] * alpha + region * (1 - alpha)).astype(np.uint8)


def field(i, animate=True):
    """A dark backdrop whose dots move every frame (or not, when ``animate`` is False)."""
    frame = np.full((H, W, 3), 22, np.uint8)
    rng = np.random.default_rng(i if animate else 0)
    ys = rng.integers(0, H - 3, 120)
    xs = rng.integers(0, W - 3, 120)
    for y, x in zip(ys, xs):
        frame[y:y + 3, x:x + 3] = (90, 160, 220)
    return frame


def write(path, frames, fps=FPS):
    proc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                             '-video_size', f'{W}x{H}', '-framerate', str(fps), '-i', '-',
                             '-c:v', 'libx264rgb', '-qp', '0', '-preset', 'ultrafast', str(path)],
                            stdin=subprocess.PIPE)
    for frame in frames:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    assert proc.wait() == 0
    return path


def clip(tmp_path, name, *, n=45, path=None, animate_every=1, top_bar=False, fps=FPS):
    path = path or (lambda i: (100 + 6 * i, 150 + 2 * i))
    frames = []
    for i in range(n):
        frame = field(i // animate_every)
        if top_bar:
            frame[:32] = 0
            frame[10:22, 8:80] = 255                    # "Activities"
            frame[10:22, W // 2 - 30:W // 2 + 30] = 255  # the clock
        draw_cursor(frame, *path(i))
        frames.append(frame)
    return write(tmp_path / f'{name}.mkv', frames, fps)


def test_a_smooth_clip_with_a_gliding_pointer_passes(tmp_path):
    result = capture_clip_gate.evaluate(clip(tmp_path, 'smooth'))
    assert result['passed'], result['reasons']
    assert result['distinct_fps'] >= 28
    assert result['cursor']['found_fraction'] == 1.0
    assert 5 <= result['cursor']['max_step_px'] <= 7
    assert result['top_bar']['hidden']


def test_a_pointer_jump_fails(tmp_path):
    def jumpy(i):
        return (100 + 6 * i + (200 if i >= 20 else 0), 150)
    result = capture_clip_gate.evaluate(clip(tmp_path, 'jump', path=jumpy))
    assert not result['passed']
    assert result['cursor']['jump_count'] == 1
    assert result['cursor']['jumps'][0]['frame'] == 20
    assert any(r.startswith('pointer jump') for r in result['reasons'])


def test_a_choppy_field_fails_even_when_the_pointer_moves(tmp_path):
    # The field repaints every third frame (10 fps); the pointer still moves
    # every frame, which must not count as animation.
    result = capture_clip_gate.evaluate(clip(tmp_path, 'choppy', animate_every=3))
    assert not result['passed']
    assert 8 <= result['distinct_fps'] <= 12
    assert any(r.startswith('choppy') for r in result['reasons'])


def test_a_visible_top_bar_fails(tmp_path):
    result = capture_clip_gate.evaluate(clip(tmp_path, 'bar', top_bar=True))
    assert not result['passed']
    assert not result['top_bar']['hidden']
    assert any(r.startswith('top bar') for r in result['reasons'])


def test_an_adapter_bypass_in_the_receipt_fails(tmp_path):
    receipt = {'adapter_bypasses': [{'qtest': 'mouseClick', 'reason': 'target is hidden', 't': 1.0}]}
    result = capture_clip_gate.evaluate(clip(tmp_path, 'bypass'), receipt=receipt)
    assert not result['passed']
    assert any('bypassed the human pointer' in r for r in result['reasons'])


def test_top_bar_detector_on_a_dark_app_menu_bar_is_negative():
    frame = np.full((H, W, 3), 14, np.uint8)        # a dark (not black) app menu bar
    frame[10:22, 8:60] = 230
    assert not capture_clip_gate.top_bar_visible(frame)['visible']
    frame[:32] = 0
    frame[10:22, 8:60] = 255
    assert capture_clip_gate.top_bar_visible(frame)['visible']


def test_cli_exit_status(tmp_path):
    good = clip(tmp_path, 'cli')
    assert capture_clip_gate.main(['--video', str(good), '--out', str(tmp_path / 'g.json')]) == 0
    assert (tmp_path / 'g.json').exists()


def test_the_threshold_follows_the_clip_frame_rate(tmp_path):
    # 60 fps clip whose field repaints every second frame: 30 distinct fps
    # would pass at 30 fps but is choppy at 60 (needs 0.93 x 60 = 55.8).
    def slow(i):
        return (100 + 3 * i, 150 + i)
    half = capture_clip_gate.evaluate(clip(tmp_path, 'half60', n=90, animate_every=2, fps=60, path=slow))
    assert half['fps'] == 60 and half['limits']['min_fps'] == 55.8
    assert not half['passed'] and any(r.startswith('choppy') for r in half['reasons'])
    full = capture_clip_gate.evaluate(clip(tmp_path, 'full60', n=90, fps=60, path=slow))
    assert full['passed'], full['reasons']
    # The pointer bound is per frame: half the 30 fps bound at 60 fps.
    assert full['cursor']['bound_px'] == round(1.5 * 0.02 * W * 0.5, 1)


def test_painted_per_second_and_labels():
    samples = [(0.0, 0), (1.0, 30), (2.0, 32), (3.0, 62), (4.0, 89)]   # second 1: the app froze
    painted = capture_clip_gate.app_painted_per_second(samples, 4)
    assert painted == [30.0, 2.0, 30.0, 27.0]
    # Second 3: the app painted 27 (its usual pace, just under 27.9): slow, not a freeze.
    freeze, loss, slow = capture_clip_gate.label_seconds([29, 3, 12, 20], painted, 27.9)
    assert freeze == [1] and loss == [2] and slow == [3]
    assert capture_clip_gate.app_painted_per_second(None, 3) is None


def test_choppy_clip_is_labelled_app_freeze_or_capture_loss(tmp_path):
    video = clip(tmp_path, 'label', n=150, animate_every=3,       # 10 distinct fps for 5 s
                 path=lambda i: (100 + 2 * i, 150 + i))
    # The app paints 30/s but stalls (2 frames a second) in seconds 0 and 1.
    stalled = {'app_paint_samples': [[0, 0], [1, 2], [2, 4], [3, 34], [4, 64], [5, 94]]}
    painting = {'app_paint_samples': [[t, 30 * t] for t in range(6)]}
    a = capture_clip_gate.evaluate(video, receipt=stalled)
    b = capture_clip_gate.evaluate(video, receipt=painting)
    assert a['app_freeze_seconds'] == [0, 1] and a['capture_loss_seconds'] == [2, 3, 4]
    assert any(r.startswith('choppy (app freeze + capture loss)') for r in a['reasons']), a['reasons']
    assert b['app_freeze_seconds'] == [] and b['capture_loss_seconds'] == [0, 1, 2, 3, 4]
    assert any(r.startswith('choppy (capture loss)') for r in b['reasons']), b['reasons']


def test_lost_frames_are_matched_to_app_paint_stalls():
    samples = [[0.0, 0], [0.1, 4], [0.2, 4], [0.3, 8], [0.4, 12]]     # nothing painted 0.1-0.2 s
    windows = capture_clip_gate.app_stall_windows(samples)
    assert windows == [(0.1, 0.2)]
    changed = [True] * 12
    changed[4] = changed[5] = False        # 0.133 s, 0.167 s: in the stall
    changed[10] = False                    # 0.333 s: not
    lost, inside = capture_clip_gate.lost_in_stalls(changed, 30.0, windows)
    assert lost == [4, 5, 10] and inside == [4, 5]
