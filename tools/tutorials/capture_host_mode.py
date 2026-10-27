"""Lesson recorder in the headless GNOME capture host: the real app, a human pointer, GNOME video.

``run_native_capture.py --host gnome`` runs this inside
:mod:`gnome_capture_host` (and its private bwrap stage). The flow:

1. the capture profile is written into the private profile by an offscreen
   child process (:func:`prepare_profile`): Dark theme, the spaCR field
   ``data_art_impulse_lens`` at its shipped density on the ``spacr`` palette,
   the shipped settings-window backdrop with its default motion, page
   opacity 60 %, BOTH alpha gates off (Feature maturity "Show alpha" and
   Show alpha features/species), ``--fresh``, What's new and tours seen,
   font scale 1.5 x width / 3840;
2. spaCR starts through its real entry point, ``spacr.qt.run(['--fresh',
   '--no-setup'])``, so every launch-time behaviour (dialog filters: rounded,
   translucent glass dialogs; fonts; the ambient start) is present;
3. once its window is up it is maximised (the capture host hides GNOME's
   top bar, so it fills the 2560x1440 monitor), the QTest adapter
   (:mod:`capture_host_adapter`) is installed, and the unchanged scene
   drivers of :mod:`capture_refresh` run from inside the app's event loop;
4. each ``capture(name, actions=...)`` becomes :meth:`HostRuntime.record_scene`:
   a GNOME ScreenCast recording (real cursor embedded) while the scene's
   actions run, the :mod:`capture_clip_gate` verdict, and the hand-off files
   the Xvfb path writes (``NAME.mp4`` at 3840x2160, ``NAME.capture.json``,
   ``NAME.png``, ``frames.json``), so ``stage_lesson.py`` and the renderer
   work unchanged. The native 2560x1440 recording stays next to it as
   ``NAME.native.mp4`` with ``NAME.gate.json`` and ``NAME.pointer.json``.

A clip that fails the gate is listed in ``host_gate_summary.json`` and the
capture is marked incomplete (``provenance.json``: ``completed_capture``
false, ``status`` ``gate_failed``), which ``stage_lesson.py`` refuses.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
import weakref

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
HANDOFF_SIZE = (3840, 2160)
GUI_SAMPLE_INTERVAL_S = 0.1
#: A GUI-sampler gap the gate counts: fixed, because capture_clip_gate checks
#: every recorded gap against max(0.35, 3.5 x the sampler interval).
GUI_SAMPLE_DELAY_S = 0.35
#: When the stall watcher writes all-thread stacks. Lowering it (e.g. 0.12
#: for a diagnostic retake) adds stacks only; the gate evidence is unchanged.
STALL_DUMP_AFTER_S = float(os.environ.get('SPACR_TUTORIAL_STALL_AFTER_S', os.environ.get('SPACR_CAPTURE_STALL_AFTER_S', '0.35')))
#: Stack dumps per clip (a stalled GUI thread is dumped once per stall).
GIL_STALL_S = float(os.environ.get('SPACR_TUTORIAL_GIL_STALL_S', '1.0'))
NATIVE_STALL_S = float(os.environ.get('SPACR_TUTORIAL_NATIVE_STALL_S', '0'))
STALL_DUMP_LIMIT = int(os.environ.get('SPACR_TUTORIAL_STALL_LIMIT', os.environ.get('SPACR_CAPTURE_STALL_LIMIT', '6')))
PAINT_COUNTER_SEMANTICS = 'observed_per_widget_deltas'

#: The capture profile (maintainer decisions, 2026-10-11).
CAPTURE_PROFILE = {
    'theme': 'dark',
    'field': 'data_art_impulse_lens',
    # Maintainer 2026-10-24: record the app's CURRENT defaults -- the field at its shipped density
    # (_default_density_for, 0.1 since nightly 0b36236f31) and the shipped settings-window backdrop
    # (drift) with its own default motion and darkness. Both are resolved from the app in prepare_profile.
    'field_density': None,
    'field_palette': 'spacr',
    'popup_backdrop': None,
    'pane_opacity_pct': 60,
    # Feature maturity -> Show alpha: off, except for a lesson whose SUBJECT module is alpha-stage (Home and the Help
    # index hide those otherwise): its job sets SPACR_TUTORIAL_SHOW_ALPHA=1 (only SPACR_TUTORIAL_* names reach the app).
    # Show alpha features / species stay off.
    'show_alpha': os.environ.get('SPACR_TUTORIAL_SHOW_ALPHA') == '1',
    'show_alpha_features': os.environ.get('SPACR_TUTORIAL_SHOW_ALPHA_FEATURES') == '1',   # only the alpha-features lessons (76 OPS, 87)
    'show_alpha_species': False,
}

_RUNTIME = None


def runtime():
    """The active :class:`HostRuntime`, or None outside the capture host."""
    return _RUNTIME


def default_font_scale(width: int) -> float:
    """The 4K lessons use 1.5; keep the same proportion of the screen."""
    return round(1.5 * width / 3840, 3)


def to_handoff(rect, scale: float):
    if rect is None:
        return None
    return [int(round(v * scale)) for v in rect]


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# -- 1. profile ----------------------------------------------------------------
def use_this_checkout():
    """Import spaCR from this checkout, never an editable install elsewhere."""
    sys.meta_path[:] = [f for f in sys.meta_path
                        if '__editable__' not in (getattr(f, '__module__', '') or type(f).__module__)]
    sys.path[:0] = [str(REPO), str(HERE)]


def prepare_report_notice() -> dict:
    """Disable reports and acknowledge their notice only in the private capture profile."""
    home = Path(os.environ.get('HOME', ''))
    app_home = Path(os.environ.get('SPACR_HOME', ''))
    config = Path(os.environ.get('XDG_CONFIG_HOME', ''))
    stage = home.parent.resolve()
    if (os.environ.get('SPACR_TUTORIAL_CACHE_ISOLATED') != '1'
            or not all(path.is_absolute() for path in (home, app_home, config))
            or home.resolve() != stage / 'profile'
            or app_home.resolve() != stage / 'app-state'
            or not config.resolve().is_relative_to(stage / 'config')):
        raise RuntimeError('Report notice setup requires the isolated capture profile')
    state = app_home / 'anonymous_reports' / 'state.json'
    if not state.resolve().is_relative_to(app_home.resolve()):
        raise RuntimeError('Report notice state escapes the private capture profile')
    from spacr import crashreport

    crashreport.set_anonymous_reports_enabled(False)
    crashreport.mark_anonymous_report_notice_shown()
    actual = {
        'anonymous_reports_enabled': crashreport.anonymous_reports_enabled(),
        'anonymous_report_notice_due': crashreport.anonymous_report_notice_due(),
        'anonymous_report_state': str(state),
    }
    if (actual['anonymous_reports_enabled'] or actual['anonymous_report_notice_due']
            or not state.is_file()):
        raise RuntimeError('Private capture report notice settings did not take')
    return actual


def prepare_profile(font_scale: float) -> dict:
    """Offscreen child: write the capture profile into the private Qt store, read it back."""
    use_this_checkout()
    report_notice = prepare_report_notice()
    import spacr
    assert Path(spacr.__file__).resolve().parent == REPO / 'spacr', spacr.__file__
    from PySide6.QtWidgets import QApplication

    import spacr.qt
    spacr.qt.register_self_registering_modules()
    from spacr.qt import app as gui
    from spacr.qt import preferences as prefs
    from spacr.qt.first_run import mark_tour_seen
    from spacr.qt.walkthrough import mark_seen
    from spacr.qt.widgets.ambient import _default_density_for
    from spacr.qt.widgets.fold_strip import folded_modules
    from spacr.updater import _installed_version
    from capture_policy import configure_appearance

    app = QApplication.instance() or QApplication(['spacr'])  # noqa: F841 - Qt settings need it
    mark_tour_seen()
    for key, *_ in gui.APPS:
        mark_seen(key)
    for key in folded_modules():
        mark_seen(key)
    prefs.set_preload_policy('on_demand')
    prefs.set_ai_on_by_default(False)
    # Theme, field, its shipped density and default palette, alpha features
    # and species off, --fresh (no remembered session, no crash drafts).
    configure_appearance(CAPTURE_PROFILE['theme'], CAPTURE_PROFILE['field'], default_field=True)
    prefs.set_show_alpha(CAPTURE_PROFILE['show_alpha'])          # Feature maturity -> Show alpha
    # Alpha SPECIES (Plasmodium, Candida, ... item 634) and alpha FEATURES have their own preferences, separate from
    # module maturity: they stay off even when a lesson turns alpha modules on (SPACR_TUTORIAL_SHOW_ALPHA=1).
    for setter, key in (('_set_show_alpha_species', 'show_alpha_species'), ('_set_show_alpha_features', 'show_alpha_features')):
        getattr(prefs, setter, lambda _on: None)(CAPTURE_PROFILE[key])
    CAPTURE_PROFILE['field_density'] = float(_default_density_for(CAPTURE_PROFILE['field']))
    CAPTURE_PROFILE['popup_backdrop'] = prefs.DEFAULT_POPUP_BACKDROP
    prefs.set_pane_opacity(CAPTURE_PROFILE['pane_opacity_pct'] / 100.0)
    prefs.set_font_scale(font_scale)
    prefs._note_running_version(_installed_version())
    prefs._note_running_version(_installed_version())
    prefs._settings().sync()
    actual = {
        'theme': prefs.get_theme() if hasattr(prefs, 'get_theme') else None,
        'field': prefs.get_ambient_animation(),
        'field_density': round(prefs.get_ambient_density(CAPTURE_PROFILE['field']), 4),
        'field_density_shipped': _default_density_for(CAPTURE_PROFILE['field']),
        'field_palette': prefs.get_ambient_palette(),
        'popup_backdrop': prefs.get_popup_backdrop(),
        'popup_density': round(prefs._popup_backdrop_motion()['density'], 4),
        'popup_backdrop_darkness': round(prefs._popup_backdrop_darkness(), 4),
        'pane_opacity_pct': int(round(prefs.get_pane_opacity() * 100)),
        'show_beta': prefs.get_show_beta(),
        'show_alpha': prefs.get_show_alpha(),
        'show_alpha_features': prefs._get_show_alpha_features(),
        'show_alpha_species': prefs._get_show_alpha_species(),
        'font_scale': font_scale,
        'store': prefs._settings().fileName(),
    }
    skip = {'theme'}
    actual['spaceout'] = False
    actual.update(report_notice)
    wrong = {k: (v, actual.get(k)) for k, v in CAPTURE_PROFILE.items()
             if k not in skip and (actual.get(k) != v if not isinstance(v, float)
                                  else not math.isclose(actual.get(k), v, abs_tol=1e-3))}
    if wrong:
        raise RuntimeError(f'Capture profile did not take: {wrong}')
    return actual


# -- 2./4. the runtime the recorder talks to --------------------------------------
class HostRuntime:
    """What :mod:`capture_refresh` uses in host mode: the window, the pump, the clip recorder."""

    def __init__(self, window, adapter, *, stage: Path, fps: int = 60, python: str = sys.executable):
        self.window = window
        self.adapter = adapter
        self.pump = adapter.pump
        self.stage = Path(stage)
        self.python = python
        self.fps = int(fps)
        self.width, self.height = (int(v) for v in os.environ['SPACR_CAPTURE_HOST_SIZE'].split('x'))
        self.scale = HANDOFF_SIZE[0] / self.width
        if abs(HANDOFF_SIZE[1] / self.height - self.scale) > 1e-6:
            raise ValueError('The host monitor must have the 16:9 shape of the 3840x2160 hand-off')
        self.gates = {}
        self.failed = []
        self.current = None
        self.pending = []
        self._paint_baselines = weakref.WeakKeyDictionary()
        self._observed_field_frames = 0
        self._next_field_identity = 0

    def to_handoff(self, rect):
        """A rectangle on the host monitor, in the 3840x2160 hand-off frame."""
        return to_handoff(rect, self.scale)

    def field_sample(self):
        """Accumulate observed widget deltas without adding historical counters.

        Hidden widgets keep their baselines, so revealing an existing page
        cannot inject all its earlier paints. A new or reset counter starts
        a baseline; unobserved paints are never invented. Weak keys avoid
        confusing a deleted widget with a new widget reusing its identity.
        """
        from spacr.qt.widgets.ambient import AmbientWidget
        widgets = [w for w in self.window.findChildren(AmbientWidget)
                   if not w.property('spacrPopupBackdrop')]
        present = set(widgets)
        for widget, (identity, count, shown, observed) in list(self._paint_baselines.items()):
            if widget not in present:
                self._paint_baselines[widget] = (identity, count, False, observed)
        visible = []
        identities = []
        for widget in widgets:
            count, shown = int(widget.frames_painted), widget.isVisible()
            previous = self._paint_baselines.get(widget)
            if previous is None:
                self._next_field_identity += 1
                identity, observed = self._next_field_identity, 0
            else:
                identity, before, was_shown, observed = previous
                if shown and was_shown:
                    delta = max(0, count - before)
                    observed += delta
                    self._observed_field_frames += delta
            self._paint_baselines[widget] = (identity, count, shown, observed)
            identities.append({'id': identity, 'frames_painted': count,
                               'visible': shown, 'observed_frames_painted': observed,
                               'counter_reset': previous is not None and count < previous[1]})
            if shown:
                visible.append(widget)
        return {'t': time.monotonic(), 'frames_painted': self._observed_field_frames,
                'fps_cap': [w.fps() for w in visible], 'widgets': identities}

    def _sample_scene(self, scene):
        """Record GUI timer delivery separately from the paint delta count."""
        sample = self.field_sample()
        samples = scene['samples']
        if samples and sample['t'] - samples[-1][0] >= GUI_SAMPLE_DELAY_S:
            scene['gui_sampler_gaps'].append((samples[-1][0], sample['t']))
        samples.append((sample['t'], sample['frames_painted']))
        scene['widget_samples'].append((sample['t'], sample['widgets']))
        # A C-level watchdog (faulthandler runs it WITHOUT the GIL): when no sample has come for GIL_STALL_S, every
        # thread's stack is written. The Python stall watcher below cannot run while some thread holds the GIL in C
        # (clip 01_module of lesson 26: a 7.7 s paint gap with no stack on record), this one can.
        handle = scene.get('gil_stall_file')
        if handle is not None:
            import faulthandler
            faulthandler.dump_traceback_later(GIL_STALL_S, file=handle)
        return sample

    def _wait(self, procs, timeout=900, pump=True):
        deadline = time.monotonic() + timeout
        while any(p.poll() is None for p in procs):
            if time.monotonic() > deadline:
                for p in procs:
                    p.kill()
                raise TimeoutError('clip post-processing did not finish')
            if pump:
                self.pump.once()
            time.sleep(0.01)

    def start_scene(self, captures: Path, name: str, seconds: float, *, appearance=None,
                    hidden_backdrops=None, buttons=(), dialogs=()) -> dict:
        """Start recording one scene; :meth:`end_scene` stops it once ``seconds`` have passed.

        Scene scripts that must run inside an application dialog's own event
        loop (a dialog opened by a real click runs ``exec()``) use this pair
        directly; :meth:`record_scene` wraps it for scheduled actions.
        """
        captures = Path(captures)
        native = captures / f'{name}.native.mp4'
        for path in (native, captures / f'{name}.mp4', captures / f'{name}.capture.json'):
            if path.exists():
                raise FileExistsError(path)
        if self.current is not None:
            raise RuntimeError(f'Scene {self.current["name"]} is still recording')
        scene = {'captures': captures, 'name': name, 'seconds': float(seconds), 'native': native,
                 'appearance': appearance, 'hidden_backdrops': hidden_backdrops,
                 'buttons': list(buttons), 'dialogs': list(dialogs), 'performed': [],
                 'mark': self.adapter.call('trace_mark'), 'first_bypass': len(self.adapter.bypasses),
                 'load_before': os.getloadavg()}
        self.pump.run(0.2)
        scene['field_before'] = self.field_sample()
        # The app's own paint counter, sampled by a Qt timer during the clip:
        # when the GUI thread freezes the samples stop, so the gate can tell
        # an app freeze from frames lost by the capture.
        from PySide6.QtCore import QTimer
        samples = scene['samples'] = []
        scene['gui_sampler_gaps'] = []
        scene['widget_samples'] = []
        # Parented to the window and torn down in end_scene: a parentless QTimer whose lambda closes over the
        # scene dict that holds it is a reference cycle only the garbage collector frees, and freeing a QObject
        # inside gc is the lesson 14 segfault.
        sampler = scene['sampler'] = QTimer(self.window)
        sampler.setInterval(round(GUI_SAMPLE_INTERVAL_S * 1000))
        sampler.timeout.connect(lambda: self._sample_scene(scene))
        # Ask mutter for its own frame-clock rate (the monitor refresh): a
        # max-framerate above what the stream offers never negotiates.
        self.adapter.call('start_recording', str(native), fps=self.fps,
                          source_fps=int(os.environ.get('SPACR_CAPTURE_HOST_REFRESH', '60')))
        scene['begin'] = time.monotonic()
        scene['field_before'] = self._sample_scene(scene)
        sampler.start()
        scene['gil_stall_file'] = open(captures / f'{name}.gil_stalls.txt', 'w')
        scene['stall_watch'] = self._watch_stalls(captures / f'{name}.stalls.txt', samples, scene['begin'])
        self.current = scene
        return scene

    @staticmethod
    def _watch_stalls(path, samples, begin, *, after=None, limit=None):
        """Dump every thread stack while the GUI thread is frozen during a clip.

        The 100 ms paint sampler runs on the GUI thread, so a sample older than
        ``after`` seconds means the GUI thread is stuck: this watcher thread
        then writes all Python stacks (faulthandler) once per stall, so the
        main-thread frame of an app freeze seen by the gate is on record.
        """
        import faulthandler
        import threading
        after = STALL_DUMP_AFTER_S if after is None else after
        limit = STALL_DUMP_LIMIT if limit is None else limit
        stop = threading.Event()

        native_after = NATIVE_STALL_S

        def native_dump(age):
            # The Python stacks above stop at processEvents when the GUI thread sits in C++ (a Qt paint, layout or
            # style pass, a GL swap): no spaCR frame shows. gdb names the C++ frames. Opt-in (SPACR_TUTORIAL_NATIVE_STALL_S),
            # twice per clip at most, because attaching pauses the process for a few seconds.
            import ctypes
            import subprocess
            try:
                ctypes.CDLL(None).prctl(0x59616d61, ctypes.c_ulong(0xffffffffffffffff))     # PR_SET_PTRACER, any
                with open(str(path).replace('.stalls.txt', '.native_stalls.txt'), 'a') as handle:
                    handle.write(f'--- native stacks after a {age:.1f} s GUI stall ---\n')
                    handle.flush()
                    subprocess.run(['timeout', '40', 'gdb', '-batch', '-p', str(os.getpid()),
                                    '-ex', 'thread apply 1 bt 40'], stdout=handle, stderr=subprocess.STDOUT, timeout=50)
            except Exception:
                pass

        def run():
            dumped_for, count, native = None, 0, 0
            with open(path, 'a') as handle:
                while not stop.wait(0.05) and count < limit:
                    last = samples[-1][0] if samples else begin
                    age = time.monotonic() - last
                    if native_after and age > native_after and native < 2:
                        native += 1
                        native_dump(age)
                    if age > after and dumped_for != last:
                        dumped_for, count = last, count + 1
                        handle.write(f'--- GUI thread stalled {age:.2f} s at clip second '
                                     f'{time.monotonic() - begin:.2f} ---\n')
                        handle.flush()
                        faulthandler.dump_traceback(file=handle, all_threads=True)
                        handle.flush()

        thread = threading.Thread(target=run, name='capture-stall-watch', daemon=True)
        thread.start()
        return stop

    def elapsed(self) -> float:
        return time.monotonic() - self.current['begin'] if self.current else 0.0

    def note_action(self, label, started, when=None):
        if self.current is not None:
            self.current['performed'].append({'name': label, 'scheduled_seconds': when,
                                              'actual_seconds': round(started, 3),
                                              'finished_seconds': round(self.elapsed(), 3)})

    def end_scene(self, *, at_least: float | None = None) -> dict:
        """Hold until the scene's duration has passed, stop recording, keep it for post-processing."""
        scene = self.current
        if scene is None:
            raise RuntimeError('No scene is recording')
        hold = scene['seconds'] if at_least is None else at_least
        while self.elapsed() < hold:
            self.pump.once()
            time.sleep(0.001)
        scene['sampler'].stop()
        import faulthandler
        faulthandler.cancel_dump_traceback_later()
        gil_file = scene.pop('gil_stall_file', None)
        if gil_file is not None:
            gil_file.close()
        if scene.get('stall_watch') is not None:
            scene['stall_watch'].set()
        field_after = self._sample_scene(scene)
        scene['recording'] = self.adapter.call('stop_recording')
        begin = scene['begin']
        scene['wall'] = time.monotonic() - begin
        trace = self.adapter.call('trace_since', scene['mark']['index'])
        bypasses = self.adapter.bypasses[scene['first_bypass']:]
        captures, name = scene['captures'], scene['name']
        (captures / f'{name}.pointer.json').write_text(json.dumps(
            [{**e, 't': round(e['t'] - begin, 4)} for e in trace], indent=0) + '\n')
        fb = scene['field_before']
        scene['host_facts'] = {
            'adapter_bypasses': [{**b, 't': round(b['t'] - begin, 3)} for b in bypasses],
            'actions': scene['performed'], 'loadavg_before': scene['load_before'],
            'loadavg_after': os.getloadavg(),
            'app_paint_samples': [[round(t - begin, 3), n] for t, n in scene['samples']],
            'app_paint_counter_semantics': PAINT_COUNTER_SEMANTICS,
            'app_paint_counter_scope': 'sum_of_observed_visible_widget_deltas',
            'app_field_widget_samples': [[round(t - begin, 3), widgets]
                                         for t, widgets in scene['widget_samples']],
            'app_gui_sampler_interval_s': GUI_SAMPLE_INTERVAL_S,
            'app_gui_sampler_gap_threshold_s': GUI_SAMPLE_DELAY_S,
            'app_gui_sampler_gaps_s': [[round(a - begin, 3), round(b - begin, 3)]
                                       for a, b in scene['gui_sampler_gaps']],
            'app_field': {'fps_cap': fb['fps_cap'],
                          'painted_fps_scope': 'sum_of_observed_visible_widget_deltas',
                          'painted_fps': round((field_after['frames_painted'] - fb['frames_painted'])
                                               / max(1e-6, field_after['t'] - fb['t']), 2)},
        }
        (captures / f'{name}.host.json').write_text(json.dumps(scene['host_facts'], indent=2, default=str) + '\n')
        sampler = scene.pop('sampler')
        sampler.stop()
        try:
            sampler.timeout.disconnect()          # breaks timer -> lambda -> scene -> timer
        except (RuntimeError, TypeError):
            pass
        sampler.setParent(None)
        sampler.deleteLater()
        for key in ('samples', 'gui_sampler_gaps', 'widget_samples'):
            scene.pop(key)
        self.pending.append(scene)
        self.current = None
        print(f'recorded {name} ({scene["wall"]:.1f} s; post-processing after the session)', flush=True)
        return {'image': f'{name}.png', 'appearance': scene['appearance'],
                'hidden_decorative_backdrops': scene['hidden_backdrops'],
                'capture_surface': 'gnome_capture_host_screencast', 'sha256': None,
                'buttons': scene['buttons'], 'dialogs': scene['dialogs'],
                'clip': {'pending_post_processing': True, 'native_video': scene['native'].name}}

    def record_scene(self, captures: Path, name: str, seconds: float, actions=(), **facts) -> dict:
        """Record one scene with the GNOME screencast while its scheduled actions run."""
        scheduled = sorted(actions, key=lambda item: item[0])
        if any(not 0 <= when < seconds for when, _, _ in scheduled):
            raise ValueError('A video action leaves the recording interval')
        self.start_scene(captures, name, seconds, **facts)
        while scheduled:
            when, label, callback = scheduled[0]
            if self.elapsed() < when:
                self.pump.once()
                time.sleep(0.001)
                continue
            scheduled.pop(0)
            started = self.elapsed()
            callback()
            self.note_action(label, started, when)
        return self.end_scene()

    def post_process(self, scene: dict, pump: bool = True) -> dict:
        """Gate, 3840x2160 hand-off, receipt and poster of one recorded scene."""
        captures, name, native = scene['captures'], scene['name'], scene['native']
        video = captures / f'{name}.mp4'
        receipt_path = captures / f'{name}.capture.json'
        facts_path = captures / f'{name}.host.json'
        gate_path = captures / f'{name}.gate.json'
        gate = subprocess.Popen([self.python, str(HERE / 'capture_clip_gate.py'), '--video', str(native),
                                 '--receipt', str(facts_path), '--out', str(gate_path)],
                                stdout=subprocess.DEVNULL, stderr=(captures / f'{name}.gate.log').open('wb'))
        upscale = subprocess.Popen(
            ['ffmpeg', '-nostdin', '-y', '-v', 'error', '-i', str(native),
             '-vf', f'scale={HANDOFF_SIZE[0]}:{HANDOFF_SIZE[1]}:flags=lanczos', '-r', str(self.fps), '-an',
             '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '16', '-threads', '4',
             '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video)],
            stderr=video.with_suffix('.encoder.log').open('wb'))
        self._wait([gate, upscale], timeout=3600, pump=pump)
        if upscale.returncode:
            raise RuntimeError(video.with_suffix('.encoder.log').read_text())
        verdict = json.loads(gate_path.read_text())
        host_facts = scene['host_facts']
        receipt = handoff_receipt(video, native=native, wall_seconds=scene['wall'], recording=scene['recording'],
                                  host_facts=host_facts, verdict=verdict, fps=self.fps)
        receipt_path.write_text(json.dumps(receipt, indent=2, default=str) + '\n')
        poster = captures / f'{name}.png'
        subprocess.run(['ffmpeg', '-nostdin', '-y', '-v', 'error', '-i', str(video), '-frames:v', '1',
                        '-pix_fmt', 'rgb24', str(poster)], check=True)
        summary = {'passed': verdict['passed'], 'reasons': verdict['reasons'],
                   'distinct_fps': verdict['distinct_fps'], 'fps': self.fps,
                   'min_fps': verdict['limits']['min_fps'],
                   'cursor_max_step_px': verdict['cursor']['max_step_px'],
                   'cursor_bound_px': verdict['cursor']['bound_px'],
                   'cursor_found_fraction': verdict['cursor']['found_fraction'],
                   'top_bar_hidden': verdict['top_bar']['hidden'],
                   'app_painted_fps': host_facts['app_field']['painted_fps'],
                   'app_freeze_seconds': verdict.get('app_freeze_seconds'),
                   'capture_loss_seconds': verdict.get('capture_loss_seconds'),
                   'loadavg': [scene['load_before'][0], host_facts['loadavg_after'][0]],
                   'file': gate_path.name}
        self.gates[name] = summary
        if not verdict['passed']:
            self.failed.append(name)
            print(f'GATE FAILED {name}: ' + '; '.join(verdict['reasons']), flush=True)
        return {'image': poster.name, 'sha256': sha256(poster),
                'clip': {'video': video.name, 'receipt': receipt_path.name,
                         'receipt_sha256': sha256(receipt_path), 'sha256': receipt['sha256'],
                         'native_video': native.name, 'native_sha256': sha256(native),
                         'gate': summary}}

    def finish(self, captures: Path):
        """Post-process every recorded scene, complete frames.json, summarise the gate.

        Done after the last scene so gate and encoder work never competes with
        a recording. A capture with a failed clip is not complete.
        """
        captures = Path(captures)
        if self.current is not None:
            raise RuntimeError(f'Scene {self.current["name"]} was never ended')
        frames_path = captures / 'frames.json'
        frames = json.loads(frames_path.read_text()) if frames_path.exists() else {}
        # Three scenes at a time (each runs the gate and the 4K encode); worker
        # threads only wait on subprocesses, the Qt thread keeps pumping.
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [(scene, pool.submit(self.post_process, scene, False)) for scene in self.pending]
            while not all(f.done() for _, f in futures):
                self.pump.once()
                time.sleep(0.05)
        for scene, future in futures:
            entry = frames.setdefault(scene['name'], {})
            entry.update(future.result())
        frames_path.write_text(json.dumps(frames, ensure_ascii=False, indent=2) + '\n')
        self.pending = []
        summary = {'clips': self.gates, 'failed': self.failed, 'passed': not self.failed}
        (captures / 'host_gate_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        if self.failed:
            provenance_path = captures / 'provenance.json'
            provenance = json.loads(provenance_path.read_text())
            provenance.update(completed_capture=False, status='gate_failed', gate_failed_clips=self.failed)
            provenance_path.write_text(json.dumps(provenance, indent=2) + '\n')
        return summary


def handoff_receipt(video: Path, *, native: Path, wall_seconds: float, recording: dict,
                    host_facts: dict, verdict: dict, fps: int = 60) -> dict:
    """The receipt ``stage_lesson.py`` checks, for the 3840x2160 hand-off clip."""
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams',
                                                '-show_format', '-of', 'json', str(video)]))
    streams = probe['streams']
    if len(streams) != 1 or streams[0]['codec_type'] != 'video':
        raise ValueError('A tutorial clip requires one silent video stream')
    decoded = subprocess.check_output(['ffmpeg', '-nostdin', '-v', 'error', '-xerror', '-threads', '2',
                                       '-i', str(video), '-an', '-f', 'framemd5', '-'], text=True)
    rows = [line.split(',') for line in decoded.splitlines() if not line.startswith('#')]
    if len(rows) != int(streams[0]['nb_frames']):
        raise ValueError('Independent full decode has a different frame count')
    return {'schema': 1, 'video': video.name, 'sha256': sha256(video),
            'size': [int(streams[0]['width']), int(streams[0]['height'])],
            'duration': float(probe['format']['duration']), 'capture_seconds': wall_seconds,
            'target_fps': fps, 'encoded_fps': streams[0]['avg_frame_rate'],
            'actions': host_facts['actions'],
            'frames': [{'pts': int(row[2]), 'decoded_md5': row[-1].strip()} for row in rows],
            'unique_decoded_frames': len({row[-1].strip() for row in rows}),
            'timing': f'GNOME ScreenCast frames at original wallclock speed ({fps} fps constant-rate)',
            'surface': ('actual spaCR (spacr.qt.run) in isolated software-rendered nested Wayland GNOME Shell: '
                        'real cursor embedded; pointer and keys via mutter RemoteDesktop'
                        if os.environ.get('SPACR_CAPTURE_HOST_BACKEND') == 'software-nested-wayland' else
                        'actual spaCR (spacr.qt.run) in isolated software-rendered GNOME Shell: '
                        'X11, real cursor embedded; pointer and keys via mutter RemoteDesktop'
                        if os.environ.get('SPACR_CAPTURE_HOST_BACKEND') == 'software-x11' else
                        'actual spaCR (spacr.qt.run) in headless GNOME Shell 42: mutter compositor, '
                        'Xwayland, real cursor embedded; pointer and keys via mutter RemoteDesktop'),
            'native': {'video': native.name, 'sha256': sha256(native), 'recorder': recording,
                       'size': verdict['size'], 'scaled_to_handoff': 'lanczos'},
            'gate': {'passed': verdict['passed'], 'reasons': verdict['reasons']},
            'adapter_bypasses': host_facts['adapter_bypasses'],
            'full_decode_passed': True, 'audio_streams': 0}


#: What ``--fps auto`` records at. 30, measured 2026-10-12 on an idle machine
#: (capture_host_bench.py, load ~5 on 32 cores): with the field painting
#: 57-60 fps, mutter's ScreenCast handed over only ~38-43 frames per second
#: at 2560x1440 with nothing encoding (any cursor mode, with or without
#: always-copy), and gnome-shell's main thread sat at 80-100 % of a core:
#: every frame is a full-surface software-Xwayland upload plus a synchronous
#: read-back (no DMA-buf screencast on this NVIDIA stack). The encoder is not
#: the limit (ultrafast and superfast give the same ~40 distinct fps). So a
#: 60 fps recording is reliably choppy (~40 distinct of 60), while 30 fps
#: gets every frame (29-30 distinct). ``--fps 60`` stays available for a
#: faster capture stack.
AUTO_FPS = 30


def recording_fps(window, requested: str = 'auto') -> int:
    """``30``/``60`` as asked; ``auto`` = :data:`AUTO_FPS` (30; see why there)."""
    if requested != 'auto':
        return int(requested)
    return AUTO_FPS


# -- 2./3. launch the real app and run the recorder inside it -----------------
def run_child(stage: Path, capture_args: list[str]) -> int:
    """Inside the capture host and the neutral stage: profile, real launch, scenes."""
    global _RUNTIME
    use_this_checkout()
    import capture_refresh
    from human_input import GuiCaller, HumanProcess

    width, height = (int(v) for v in os.environ['SPACR_CAPTURE_HOST_SIZE'].split('x'))
    capture_args = list(capture_args)
    if '--platform' not in capture_args:
        capture_args += ['--platform', 'xcb']
    if '--font-scale' not in capture_args:
        capture_args += ['--font-scale', str(default_font_scale(width))]
    args = capture_refresh.build_parser().parse_args(['--stage', str(stage), *capture_args])
    os.environ.update(capture_refresh.capture_environment(args, stage))
    os.environ.update(SPACR_TUTORIAL_CACHE_ISOLATED='1', SPACR_CAPTURE_HOST_MODE='gnome')
    os.environ.pop('SPACR_CHAINING_PINS', None)
    logs = Path(os.environ['SPACR_LOG_DIR'])
    logs.mkdir(parents=True, exist_ok=True)
    # `kill -USR1 <pid>` writes every thread's Python stack (diagnosing a stuck scene).
    import faulthandler
    import signal as _signal
    faulthandler.register(_signal.SIGUSR1, file=(logs / 'stacks.txt').open('a'), all_threads=True)
    profile = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--prepare-profile',
                              '--font-scale', str(args.font_scale), '--out', str(stage / 'capture-profile.json')],
                             env={**os.environ, 'QT_QPA_PLATFORM': 'offscreen'},
                             capture_output=True, text=True)
    if profile.returncode:
        raise RuntimeError('Capture profile failed:\n' + profile.stderr[-4000:])

    import spacr
    assert Path(spacr.__file__).resolve().parent == REPO / 'spacr', spacr.__file__
    import spacr.qt

    human = HumanProcess(log=(logs / 'human_input.log').open('w'))
    human.press('Escape')                          # no overview, whatever the shell did
    human.warp(width * 0.62, height * 0.58)
    gui = GuiCaller()                              # main thread, before spaCR's QApplication
    outcome = {'failure': None, 'rc': None}

    def driver():
        global _RUNTIME
        from PySide6.QtWidgets import QApplication
        from capture_host_adapter import EventPump, QTestAdapter
        adapter = None
        try:
            from spacr.qt import app as spacr_app
            window = next(w for w in QApplication.topLevelWidgets()
                          if isinstance(w, spacr_app.MainWindow) and w.isVisible())
            window.showMaximized()
            pump = EventPump()
            pump.run(6.0)                          # startup work and the field settle
            from PySide6.QtCore import QPoint
            origin = window.mapToGlobal(QPoint(0, 0))
            screen = window.screen().geometry()
            placed = {'window': [origin.x(), origin.y(), window.width(), window.height()],
                      'screen': [screen.x(), screen.y(), screen.width(), screen.height()],
                      'top_bar': os.environ.get('SPACR_CAPTURE_HOST_TOP_BAR')}
            (stage / 'host-window.json').write_text(json.dumps(placed, indent=2) + '\n')
            if placed['window'] != [0, 0, width, height]:
                raise RuntimeError(f'spaCR does not fill the monitor (top bar not hidden?): {placed}')
            fps = recording_fps(window, os.environ.get('SPACR_CAPTURE_FPS', 'auto'))
            placed['recording_fps'] = fps
            (stage / 'host-window.json').write_text(json.dumps(placed, indent=2) + '\n')
            adapter = QTestAdapter(human, pump=pump, log=(logs / 'adapter.jsonl').open('w')).install()
            _RUNTIME = HostRuntime(window, adapter, stage=stage, fps=fps)
            sys.argv = ['capture_refresh', '--stage', str(stage), *capture_args]
            outcome['rc'] = capture_refresh.main()
        except BaseException:                      # noqa: BLE001 - reported, then the app quits
            outcome['failure'] = traceback.format_exc()
            print(outcome['failure'], file=sys.stderr, flush=True)
        finally:
            if adapter is not None:
                adapter.uninstall()
            QApplication.quit()

    def watcher():
        try:
            deadline = time.monotonic() + 180
            from PySide6.QtWidgets import QApplication
            while time.monotonic() < deadline:
                if QApplication.instance() is not None:
                    from spacr.qt import app as spacr_app
                    try:
                        shown = gui(lambda: any(isinstance(w, spacr_app.MainWindow) and w.isVisible()
                                                for w in QApplication.topLevelWidgets()), timeout=10)
                    except TimeoutError:           # the event loop has not started yet
                        continue
                    if shown:
                        gui.post(driver)
                        return
                time.sleep(0.2)
            raise TimeoutError('spaCR did not open its window')
        except BaseException:                      # noqa: BLE001
            outcome['failure'] = traceback.format_exc()
            os._exit(4)

    threading.Thread(target=watcher, name='capture-host-watcher', daemon=True).start()
    app_rc = spacr.qt.run(['--fresh', '--no-setup'])
    human.close()
    summary = _RUNTIME.gates if _RUNTIME is not None else {}
    failed = _RUNTIME.failed if _RUNTIME is not None else []
    (stage / 'host-run.json').write_text(json.dumps({
        'app_exit_code': app_rc, 'recorder_rc': outcome['rc'], 'failure': outcome['failure'],
        'clips': summary, 'gate_failed': failed}, indent=2) + '\n')
    if outcome['failure']:
        return 1
    return 2 if failed else (outcome['rc'] or 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--prepare-profile', action='store_true', required=True)
    parser.add_argument('--font-scale', type=float, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(json.dumps(prepare_profile(args.font_scale), indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
