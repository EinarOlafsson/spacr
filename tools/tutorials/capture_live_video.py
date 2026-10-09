"""Record real Qt painting with wallclock timestamps and no invented frames.

The caller owns the fresh private application profile and schedules real UI
actions. Slow captures skip target slots; they never speed the animation up.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path


def record_widget_video(window, process_events, output: Path, seconds: float,
                        *, target_fps: int = 30, actions=()) -> dict:
    """Capture a widget while its actual event loop continues to run.

    ``actions`` are (seconds, name, callback) tuples, recorded when executed.
    A clip is silent. Its receipt reports achieved rate and real timestamps.
    """
    from PySide6.QtGui import QImage

    if seconds <= 0 or target_fps <= 0:
        raise ValueError("Video duration and target rate must be positive")
    scheduled = sorted(actions, key=lambda item: item[0])
    if any(not 0 <= when < seconds for when, _, _ in scheduled):
        raise ValueError("A video action leaves the recording interval")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    size = window.grab().size()
    command = [
        "ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "rawvideo",
        "-pixel_format", "rgba", "-video_size", f"{size.width()}x{size.height()}",
        "-use_wallclock_as_timestamps", "1", "-framerate", str(target_fps),
        "-i", "pipe:0", "-copyts", "-start_at_zero", "-vsync", "0", "-an",
        "-c:v", "libx264", "-preset", "ultrafast", "-threads", "2",
        "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(output),
    ]
    frames, performed = [], []
    error_path = output.with_suffix(".encoder.log")
    with error_path.open("wb") as error:
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=error)
        begin = time.monotonic()
        slot = 0
        try:
            while time.monotonic() - begin < seconds:
                slot = max(slot, int((time.monotonic() - begin) * target_fps))
                due = begin + slot / target_fps
                while time.monotonic() < due:
                    process_events()
                    time.sleep(.001)
                process_events()
                elapsed = time.monotonic() - begin
                while scheduled and scheduled[0][0] <= elapsed:
                    when, name, callback = scheduled.pop(0)
                    callback()
                    process_events()
                    performed.append({"name": name, "scheduled_seconds": when,
                                      "actual_seconds": time.monotonic() - begin})
                timestamp = time.monotonic() - begin
                image = window.grab().toImage().convertToFormat(QImage.Format_RGBA8888)
                if image.size() != size:
                    raise ValueError("Application surface resized during its clip")
                data = bytes(image.constBits())
                encoder.stdin.write(data)
                frames.append({"target_slot": slot, "seconds": timestamp,
                               "rgba_sha256": hashlib.sha256(data).hexdigest()})
                slot += 1
            capture_seconds = time.monotonic() - begin
            encoder.stdin.close()
            encoder.wait(timeout=60)
        finally:
            if encoder.poll() is None:
                encoder.kill()
                encoder.wait()
    if encoder.returncode:
        raise RuntimeError(error_path.read_text())
    if scheduled:
        raise RuntimeError("The recording ended before a scheduled action")
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json",
        str(output),
    ], text=True))
    streams = probe["streams"]
    if len(streams) != 1 or streams[0]["codec_type"] != "video":
        raise ValueError("A tutorial clip must contain one silent video stream")
    if int(streams[0]["nb_frames"]) != len(frames):
        raise ValueError("The encoder changed the captured frame count")
    duration = float(probe["format"]["duration"])
    if abs(duration - (frames[-1]["seconds"] - frames[0]["seconds"])) > .4:
        raise ValueError("Encoded duration does not preserve actual capture time")
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-threads", "2",
                    "-i", str(output), "-f", "null", "-"], check=True)
    receipt = {
        "schema": 1, "video": output.name,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "size": [size.width(), size.height()], "duration": duration,
        "capture_seconds": capture_seconds, "target_fps": target_fps,
        "achieved_fps": len(frames) / capture_seconds,
        "frames": frames, "actions": performed,
        "unique_rgba_frames": len({frame["rgba_sha256"] for frame in frames}),
        "timing": "actual wallclock VFR; target slots may be skipped",
        "surface": "Qt application widget; no native compositor claim",
        "full_decode_passed": True, "audio_streams": 0,
    }
    output.with_suffix(".capture.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def record_x11_video(window, process_events, output: Path, seconds: float,
                     *, target_fps: int = 30, actions=()) -> dict:
    """Capture the real X11 surface while Qt handles paints and interactions."""
    from PySide6.QtCore import QPoint
    if not os.environ.get('DISPLAY') or os.environ.get('QT_QPA_PLATFORM') == 'offscreen':
        raise ValueError('X11 recording requires an actual isolated X11 application surface')
    if seconds <= 0 or target_fps <= 0:
        raise ValueError('Video duration and frame rate must be positive')
    scheduled = sorted(actions, key=lambda item: item[0])
    if any(not 0 <= when < seconds for when, _, _ in scheduled):
        raise ValueError('A video action leaves the recording interval')
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    origin = window.mapToGlobal(QPoint(0, 0))
    size = window.size()
    command = ['ffmpeg', '-nostdin', '-y', '-v', 'error', '-f', 'x11grab',
               '-framerate', str(target_fps), '-video_size', f'{size.width()}x{size.height()}',
               '-i', f"{os.environ['DISPLAY']}+{origin.x()},{origin.y()}",
               '-frames:v', str(math.ceil(seconds * target_fps)),
               '-an', '-c:v', 'libx264', '-preset', 'ultrafast',
               '-threads', '2', '-crf', '18', '-pix_fmt', 'yuv420p',
               '-movflags', '+faststart', str(output)]
    performed = []
    with output.with_suffix('.encoder.log').open('wb') as error:
        begin = time.monotonic()
        encoder = subprocess.Popen(command, stderr=error)
        try:
            while encoder.poll() is None:
                process_events()
                elapsed = time.monotonic() - begin
                while scheduled and scheduled[0][0] <= elapsed:
                    when, name, callback = scheduled.pop(0)
                    callback()
                    process_events()
                    performed.append({'name': name, 'scheduled_seconds': when,
                                      'actual_seconds': time.monotonic() - begin})
                if elapsed > seconds + 30:
                    raise TimeoutError('X11 encoder did not finish its bounded recording')
                time.sleep(.001)
            elapsed = time.monotonic() - begin
        finally:
            if encoder.poll() is None:
                encoder.kill()
                encoder.wait()
    if encoder.returncode:
        raise RuntimeError(output.with_suffix('.encoder.log').read_text())
    if scheduled:
        raise RuntimeError('Recording ended before a scheduled action')
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams',
                                              '-show_format', '-of', 'json', str(output)]))
    streams = probe['streams']
    if len(streams) != 1 or streams[0]['codec_type'] != 'video':
        raise ValueError('A tutorial clip requires one silent video stream')
    duration = float(probe['format']['duration'])
    expected_frames = math.ceil(seconds * target_fps)
    if (int(streams[0]['nb_frames']) != expected_frames
            or abs(duration - expected_frames / target_fps) > .002
            or abs(elapsed - seconds) > 2):
        raise ValueError('X11 video duration does not preserve recording wall time')
    decoded = subprocess.check_output(['ffmpeg', '-nostdin', '-v', 'error', '-xerror',
                                      '-threads', '2', '-i', str(output), '-an',
                                      '-f', 'framemd5', '-'], text=True)
    frames = [line.split(',') for line in decoded.splitlines() if not line.startswith('#')]
    if len(frames) != int(streams[0]['nb_frames']):
        raise ValueError('Independent full decode has a different frame count')
    receipt = {'schema': 1, 'video': output.name,
               'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
               'size': [size.width(), size.height()], 'duration': duration,
               'capture_seconds': elapsed, 'target_fps': target_fps,
               'encoded_fps': streams[0]['avg_frame_rate'], 'actions': performed,
               'frames': [{'pts': int(row[2]), 'decoded_md5': row[-1].strip()} for row in frames],
               'unique_decoded_frames': len({row[-1].strip() for row in frames}),
               'timing': 'real X11 frames at original wallclock speed',
               'surface': 'isolated native Qt application on X11; no compositor claim',
               'full_decode_passed': True, 'audio_streams': 0}
    output.with_suffix('.capture.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt
