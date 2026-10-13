#!/usr/bin/env python3
"""Record the current Preferences scenes without touching desktop settings.

Run through tools/run_capped.sh. Outputs are private review assets; this tool
never publishes media or changes a lesson's narration/audio timing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

SCENES = (
    ("09_performance", "Choose a Performance profile for this computer."),
    ("10_appearance_categories", "Open Appearance. Theme and Animation are folded categories."),
    ("11_appearance_theme", "Expand Theme to choose the application theme."),
    ("12_appearance_animation", "Expand Animation to adjust the backdrop and its motion."),
)


def main():
    """Capture real Qt interactions and encode a captioned review clip."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--platform', choices=('offscreen', 'xcb'), default='offscreen')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    stage = args.stage.resolve()
    stage.mkdir(parents=True, exist_ok=True)
    if (stage / 'frames.json').exists():
        raise RuntimeError('Choose a fresh stage to preserve previous capture evidence')
    with tempfile.TemporaryDirectory(prefix='spacr-preferences-recording-') as config:
        for key, value in {
            'XDG_CONFIG_HOME': config, 'XDG_STATE_HOME': config + '/state',
            'SPACR_LOG_DIR': config + '/logs', 'QT_QPA_PLATFORM': args.platform,
            'QT_SCALE_FACTOR': '1', 'QT_SCREEN_SCALE_FACTORS': '1',
            'QT_AUTO_SCREEN_SCALE_FACTOR': '0', 'QT_FONT_DPI': '96', 'SPACR_LANGUAGE': 'en',
            'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2',
        }.items():
            os.environ[key] = value
        from PySide6.QtCore import QPoint
        from PySide6.QtGui import QPainter
        from PySide6.QtWidgets import QApplication, QDialog, QTabWidget, QWidget
        from spacr.qt import preferences
        from spacr.qt.widgets.ambient import AmbientWidget
        from spacr.qt.widgets.glass import install_glass_everywhere
        from capture_home import record_performance
        from capture_policy import configure_appearance, verify_appearance, verify_visible_paths

        app = QApplication([])
        configure_appearance()
        preferences.set_font_scale(1.5)
        preferences.apply_preferences_to_app(app)
        install_glass_everywhere(app)
        host = QWidget()
        host.setWindowTitle('spaCR Preferences')
        host.resize(3840, 2160)
        backdrop = AmbientWidget(host, theme='blobs', resolution=0.25, fps=15)
        backdrop.setGeometry(host.rect())
        host.show()
        frames = {}

        def settle(seconds=0.35):
            """Allow actual layout, animation and painting to finish."""
            until = time.monotonic() + seconds
            while time.monotonic() < until:
                app.processEvents()
                time.sleep(0.01)

        def capture(name):
            """Save the actual dialog and its source-bound scene receipt."""
            appearance = verify_appearance(host)
            dialogs = [w for w in app.topLevelWidgets()
                       if isinstance(w, QDialog) and w.isVisible()]
            if len(dialogs) != 1:
                raise RuntimeError('Expected exactly one visible Preferences dialog')
            dialog = dialogs[0]
            verify_visible_paths([dialog], stage)
            image = host.grab()
            painter = QPainter(image)
            position = dialog.mapToGlobal(QPoint()) - host.mapToGlobal(QPoint())
            painter.drawPixmap(position, dialog.grab())
            painter.end()
            path = stage / (name + '.png')
            if not image.save(str(path)):
                raise RuntimeError(f'Cannot save {path}')
            tabs = dialog.findChild(QTabWidget, 'PreferencesTabs')
            frames[name] = {
                'image': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'size': [image.width(), image.height()], 'appearance': appearance,
                'tabs': [tabs.tabText(i) for i in range(tabs.count())],
                'active_tab': tabs.tabText(tabs.currentIndex()),
                'dialog_rect': [position.x(), position.y(), dialog.width(), dialog.height()],
            }

        try:
            settle(1)
            record_performance(host, capture, settle, dialog_size=(1800, 1700))
        finally:
            host.close()
            app.processEvents()
        if set(frames) != {name for name, _ in SCENES}:
            raise RuntimeError('The recording did not capture every required scene')
        (stage / 'frames.json').write_text(json.dumps(frames, indent=2) + '\n')
        (stage / 'provenance.json').write_text(json.dumps({
            'completed_capture': True, 'published': False,
            'platform': args.platform, 'settings': 'temporary isolated Qt configuration',
            'source': 'tools/tutorials/capture_home.py:record_performance',
            'source_sha256': {str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in (Path(__file__).resolve(),
                                           repo / 'tools/tutorials/capture_home.py',
                                           repo / 'spacr/qt/preferences.py')},
            'existing_scene_consumers': ['05_home', '04_platform_installers'],
            'existing_scene': '09_performance',
            'scope': 'Current Preferences captures and review clip; existing full lessons/audio unchanged',
        }, indent=2) + '\n')

    # Relative filenames keep ffmpeg's concat parser independent of stage names.
    lines = []
    captions = ['WEBVTT', '']
    for index, (name, caption) in enumerate(SCENES):
        lines.extend([f"file '{name}.png'", 'duration 6'])
        captions.extend([f'00:00:{index*6:02d}.000 --> 00:00:{(index+1)*6:02d}.000', caption, ''])
    lines.append(f"file '{SCENES[-1][0]}.png'")
    (stage / 'frames.concat').write_text('\n'.join(lines) + '\n')
    (stage / 'preferences_appearance.vtt').write_text('\n'.join(captions))
    subprocess.run([
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
        '-filter_threads', '2', '-f', 'concat', '-safe', '1', '-i', 'frames.concat',
        '-vf', 'fps=15,crop=1920:1080:960:200', '-frames:v', '360',
        '-c:v', 'libx264', '-threads', '2', '-preset', 'fast', '-crf', '20',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', 'preferences_appearance.mp4',
    ], cwd=stage, check=True)
    (stage / 'review.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>Preferences: Appearance</title>'
        '<style>body{margin:0;background:#161719;color:white;font:18px sans-serif}'
        'video{width:100%;max-height:90vh}</style>'
        '<video controls><source src="preferences_appearance.mp4" type="video/mp4">'
        '<track default kind="captions" srclang="en" label="English" '
        'src="preferences_appearance.vtt"></video>'
        '<p>Preferences: Performance, Appearance, Theme, and Animation.</p>')
    print(f'Captured four current Preferences scenes and a 24-second review clip in {stage}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
