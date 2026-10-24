"""N673 speed follow-up (2026-10-16): the True 3D view on the GPU.

The maintainer's review of the napari-like True 3D: "much better, now
speed". Turning, zooming and panning must feel instant. The view now draws
on the GPU (VisPy gloo ray-marching, OpenGL) with the CPU renderer as the
fallback; a Render backend list (Auto, OpenGL, CPU) and a Render scale
slider (1-100 %) replace "Render at full resolution"; the mask surfaces
have a Finish (Matte, Satin, Glossy, Metallic); and the magnifier works in
3D as a cube that magnifies what is inside it and dims the rest.

GPU tests run only where an OpenGL context can be made offscreen and are
skipped elsewhere. The GPU renderer draws from the same rays as the CPU
one, so its frames are compared with the CPU's pixel for pixel.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QApplication

from spacr import layers
from spacr.layers import (CUBE_EDGES, GPU_MAX_VOXELS, RENDER_DRAG_VOXELS,
                          RENDER_SCALE_MAX_PIXEL, SURFACE_FINISHES, VOLUME_FINISHES,
                          VOLUME_RENDERINGS, ImageLayer, LabelsLayer, LayerError,
                          SurfaceLight, VolumeLens, VolumeRays, render_scale,
                          render_volume, shade_surface, surface_finish)
from spacr.qt import preferences
from spacr.qt.screens import make_masks as mm

from .test_673_napari_true3d import _settle, _spheres, _wait_render, screen, view

__all__ = ["screen", "view"]


@pytest.mark.parametrize(('x', 'y', 'expected'), [
    (9.999, 35, 0), (15, 40.001, 0), (9.999, 40.001, 0),
    (30.001, 35, 0), (15, 19.999, 0),
    (10.001, 39.999, 7), (29.999, 20.001, 10),
    (10, 35, 7), (15, 40, 7), (30, 35, 0), (15, 20, 0),
])
def test_volume_picking_respects_the_rendered_pixel_grid(x, y, expected):
    from types import SimpleNamespace

    stage = SimpleNamespace(
        _render_state=SimpleNamespace(
            rect=(10, 30, 20, 40), picked=np.array([[7, 8], [9, 10]])),
        _labels=SimpleNamespace(findData=lambda label: -1),
        _say=lambda message: None)
    assert mm._VolumeStage.pick_at(stage, x, y) == expected


def _gl_context():
    """An offscreen OpenGL context for the GPU renderer, or skip the test.

    2026-10-16 (wgpu follow-up): never where the GPU is hidden
    (CUDA_VISIBLE_DEVICES empty or SPACR_NO_GPU_RENDER set), as at a
    computer whose GPU another job owns.
    """
    refusal = layers.gpu_render_refusal()
    if refusal:
        pytest.skip(f"GPU rendering is turned off here ({refusal})")
    try:
        return mm._GLVolumeContext()
    except RuntimeError as exc:
        if str(exc) not in {
            "VisPy is not installed",
            "no OpenGL context could be created",
            "only OpenGL ES is available, and the shaders need desktop OpenGL",
            "no offscreen OpenGL surface could be created",
            "the OpenGL context could not be made current",
        }:
            raise
        pytest.skip(f"no OpenGL context offscreen: {exc}")


def _rays(shape=(30, 36, 30)):
    """Oblique rays that never land on a voxel boundary exactly."""
    return VolumeRays(origin=(-4.031, -2.073, -2.117), across=(0.0, 0.1, 0.9),
                      down=(0.2, 0.9, 0.0), depth=(0.8, 0.1, 0.3), shape=shape)


def _layers(rendering="mip", labels_rendering="surface", finish="matte"):
    labels = _spheres()
    image = labels.astype(np.float32) * 100 + np.indices(labels.shape)[2]
    channel = ImageLayer(image.astype(np.float32), name="c", contrast_limits=[(0, 120)],
                         colormaps=["green"], metadata={"rendering": rendering})
    masks = LabelsLayer(labels, name="m", opacity=0.6,
                        metadata={"rendering": labels_rendering, "finish": finish})
    return channel, masks


def _use(stage, backend, qtbot):
    stage._backend_box.setCurrentIndex(stage._backend_box.findData(backend))
    _wait_render(stage, qtbot)


def test_the_render_scale_slider_maps_to_voxels_pixels_and_samples():
    voxels = 64 * 256 * 256
    full = render_scale(100, voxels)
    assert full.max_voxels is None and full.pixel == 1.0 and full.sample == 1.0
    half = render_scale(50, voxels)
    assert half.max_voxels == voxels // 8 and half.pixel == 2.0
    quarter = render_scale(25, voxels)
    assert quarter.max_voxels == voxels // 64 and quarter.pixel == 4.0
    least = render_scale(1, voxels)
    assert least.max_voxels == layers.RENDER_SCALE_MIN_VOXELS
    assert least.pixel == RENDER_SCALE_MAX_PIXEL
    assert render_scale(0, voxels).percent == 1 and render_scale(150, voxels).percent == 100
    dragged = render_scale(100, voxels, dragging=True)
    assert dragged.max_voxels == RENDER_DRAG_VOXELS and dragged.pixel >= 4 and dragged.sample > 1
    gpu = render_scale(100, voxels, gpu=True)
    gpu_dragged = render_scale(100, voxels, dragging=True, gpu=True)
    assert gpu.max_voxels is None and gpu_dragged.max_voxels is None
    assert gpu_dragged.pixel == 2.0 and gpu_dragged.sample == 1.0
    huge = render_scale(100, 4 * GPU_MAX_VOXELS, gpu=True)
    assert huge.max_voxels == GPU_MAX_VOXELS


def test_the_slider_replaces_full_resolution_and_is_remembered(view, qtbot):
    stage = view
    assert stage.findChild(object, "VolumeRenderFull") is None
    slider = stage._scale_slider
    assert (slider.minimum(), slider.maximum()) == (1, 100)
    _use(stage, "cpu", qtbot)
    slider.setValue(25)
    _wait_render(stage, qtbot)
    request = stage._render_request("preview")
    assert request.scale.percent == 25 and request.scale.pixel == 4.0
    assert np.prod([-(-n // s) for n, s in zip(stage.session.image.shape, request.steps)]) \
        <= max(layers.RENDER_SCALE_MIN_VOXELS, stage.session.image.size // 64)
    assert stage._scale_value.text() == "25%"
    assert stage._render_state.scale.percent == 25 and "25% scale" in stage._detail.text()
    slider.setValue(100)
    _wait_render(stage, qtbot)
    assert stage._render_request("preview").steps == (1, 1, 1)
    assert preferences.get_volume_view()["scale"] == 100


def test_backends_offered_and_the_choice_is_remembered(view, qtbot):
    stage = view
    offered = [stage._backend_box.itemData(i) for i in range(stage._backend_box.count())]
    assert offered == list(mm.volume_render_backends())
    assert offered[0] == "auto" and offered[-1] == "cpu"
    # 2026-10-16: wgpu ships by default, so Vulkan, Metal and Direct3D 12 are
    # offered where wgpu finds an adapter (tests/qt/test_673_true3d_wgpu.py).
    assert set(offered) <= set(mm.VOLUME_BACKENDS)
    _use(stage, "cpu", qtbot)
    assert preferences.get_volume_view()["backend"] == "cpu"
    assert stage._render_state.backend == "cpu" and "the CPU" in stage._detail.text()
    assert mm._volume_view_settings()[0] == "cpu"


def test_native_metal_and_user_scale_survive_same_build_restart(tmp_path):
    """Check two real Cocoa processes sharing only their owned portable profile."""
    _run_native_volume_probe(tmp_path, scenario='restart')


def test_native_image_opacity_changes_only_its_layer_and_restores_the_frame(tmp_path):
    """Check the actual image slider and exact Metal frame restoration."""
    _run_native_volume_probe(tmp_path, scenario='image_opacity')


def test_native_image_visibility_contrast_and_rendering_restore_the_frame(tmp_path):
    """Check seven real image-control states without changing sibling layers."""
    _run_native_volume_probe(tmp_path, scenario='image_controls')


def test_native_masks_opacity_rendering_and_finish_restore_the_frame(tmp_path):
    """Check seven real mask-layer states and the intentional finish preference."""
    _run_native_volume_probe(tmp_path, scenario='mask_controls')


def test_native_masks_visibility_restores_the_frame(tmp_path):
    """Use the actual Masks eye without changing sibling layers or voxel data."""
    _run_native_volume_probe(tmp_path, scenario='mask_visibility')


def test_native_image_colour_restores_the_frame_without_changing_label_ids(tmp_path):
    """Change the actual Image colour combo without changing Masks or label IDs."""
    _run_native_volume_probe(tmp_path, scenario='image_colour')


def test_native_remaining_image_renderings_each_restore_mip(tmp_path):
    """Check the three other offered Image renderings without replaying attenuation."""
    _run_native_volume_probe(tmp_path, scenario='image_renderings')


def test_native_remaining_masks_finishes_each_restore_matte(tmp_path):
    """Check the two other offered Masks finishes without replaying satin."""
    _run_native_volume_probe(tmp_path, scenario='mask_finishes')


def _run_native_volume_probe(tmp_path, *, scenario):
    """Share only normal startup, owned data and current-frame verification."""
    import ast
    import hashlib
    import json
    import os
    from pathlib import Path
    import shutil
    import subprocess
    import sys
    import textwrap

    assert scenario in ('restart', 'image_opacity', 'image_controls', 'mask_controls', 'mask_visibility',
        'image_colour', 'image_renderings', 'mask_finishes')
    if sys.platform != 'darwin' or os.environ.get('QT_QPA_PLATFORM', '').lower() != 'cocoa':
        pytest.skip('Requires the owned native Cocoa acceptance runner')
    root = Path(__file__).resolve().parents[2]
    retained = Path(os.environ.get('SPACR_RESTART_FIBSEM_FIXTURE',
        '/Users/olafsson/Library/Caches/spacr-acceptance/fibsem-native.aGEZyt/make_masks_fibsem'))
    required = ('hela_fibsem.tif', 'manifest.csv', 'ground_truth_masks/hela_fibsem.tif',
                'corrected_masks/hela_fibsem.tif')
    assert all((retained / name).is_file() for name in required), retained
    fixture = tmp_path / 'fibsem'
    fixture.mkdir()
    for name in required:
        target = fixture / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(retained / name, target)
    checksums = {name: hashlib.sha256((fixture / name).read_bytes()).hexdigest() for name in required}
    source = ast.parse((root / 'tests/qt/test_the_macos_application_menu.py').read_text())
    bootstrap = next(node.value.args[0] for node in source.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == '_OWNED_SETTINGS_PROBE'
                for target in node.targets))
    child = textwrap.dedent(ast.literal_eval(bootstrap)) + textwrap.dedent(r'''
        import hashlib, subprocess, time, traceback
        from pathlib import Path
        import numpy as np
        import shiboken6
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QScrollArea
        from spacr import layers
        from spacr.qt import preferences, timing
        from spacr.qt.app import MainWindow
        from spacr.qt.bridge import registry, thread_has_stopped
        from spacr.qt.first_run import _TourOverlay, was_tour_shown
        from spacr.qt.make_masks_demo import load_the_fibsem_test_data
        from spacr.qt.screens import make_masks as mm

        root = Path(os.environ['SPACR_RESTART_ROOT'])
        assert Path(__import__('spacr').__file__).resolve().parent.parent == root
        assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip() == os.environ['SPACR_RESTART_HEAD']
        assert timing.ENABLED and not layers.gpu_render_refusal()
        phase = os.environ['SPACR_RESTART_PHASE']
        scenario = os.environ['SPACR_VOLUME_SCENARIO']
        assert scenario in ('restart', 'image_opacity', 'image_controls', 'mask_controls', 'mask_visibility',
            'image_colour', 'image_renderings', 'mask_finishes')
        control_operations = ['initial', 'visibility_hidden', 'visibility_restored',
            'contrast_changed', 'contrast_restored', 'rendering_changed', 'rendering_restored']
        if scenario == 'image_colour':
            control_operations = ['initial', 'colour_changed', 'colour_restored']
        elif scenario == 'image_renderings':
            control_operations = ['initial', 'translucent_changed', 'translucent_restored',
                'average_changed', 'average_restored', 'minip_changed', 'minip_restored']
        mask_operations = ['initial', 'mask_opacity_changed', 'mask_opacity_restored',
            'mask_rendering_changed', 'mask_rendering_restored', 'mask_finish_changed', 'mask_finish_restored']
        if scenario == 'mask_visibility':
            mask_operations = ['initial', 'mask_visibility_hidden', 'mask_visibility_restored']
        elif scenario == 'mask_finishes':
            mask_operations = ['initial', 'glossy_changed', 'glossy_restored', 'metallic_changed', 'metallic_restored']
        folder = Path(os.environ['SPACR_RESTART_FIXTURE'])
        output = Path(os.environ['SPACR_RESTART_OUTPUT'])
        report = {'phase': phase, 'errors': [], 'initial_saved': preferences.get_volume_view(),
                  'source_head': os.environ['SPACR_RESTART_HEAD'], 'runtime': sys.executable,
                  'settings_file': preferences._settings().fileName(),
                  'home_tour_seen_before_run': was_tour_shown(),
                  'pick_scope': 'inside-frame background only; positive selection not exercised'}
        if scenario == 'image_opacity':
            report.update(scenario=scenario, pick_scope='not exercised', opacity_states=[])
        elif scenario in ('image_controls', 'image_colour', 'image_renderings'):
            report.update(scenario=scenario, pick_scope='not exercised', image_control_states=[], image_control_actions=[])
        elif scenario in ('mask_controls', 'mask_visibility', 'mask_finishes'):
            report.update(scenario=scenario, pick_scope='not exercised', mask_control_states=[], mask_control_actions=[])
        if phase == 'B':
            assert report['initial_saved']['backend'] == 'metal' and report['initial_saved']['scale'] == 75
        else:
            assert phase == 'A' and not report['initial_saved']
        preferences.set_ai_on_by_default(False)
        state = {}

        def digest(array):
            return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()

        def save():
            output.write_text(json.dumps(report, indent=2))

        def idle(stage):
            return stage.render_idle() and all(worker.idle() for worker in
                (stage._worker, stage._mesher, stage._renderer))

        def registry_snapshot(moment):
            rows = []
            for handle in registry().active():
                valid = shiboken6.isValid(handle)
                thread, worker = handle.thread, handle.worker
                thread_valid = thread is not None and shiboken6.isValid(thread)
                owner = handle.parent() if valid else None
                owner_valid = owner is not None and shiboken6.isValid(owner)
                rows.append({'handle_id': id(handle), 'handle_type': type(handle).__name__,
                    'handle_valid': valid, 'handle_cpp': shiboken6.getCppPointer(handle) if valid else None,
                    'name': handle.app_key, 'native_name': handle.objectName() if valid else None,
                    'owner_id': id(owner) if owner is not None else None,
                    'owner_type': type(owner).__name__ if owner is not None else None,
                    'owner_cpp': shiboken6.getCppPointer(owner) if owner_valid else None,
                    'kind': handle.kind, 'stop_on_quit': getattr(handle, 'stop_on_quit', True),
                    'blocks_shutdown': handle.blocks_shutdown,
                    'is_running': handle.is_running() if valid and (thread is None or thread_valid) else None,
                    'thread_id': id(thread) if thread is not None else None,
                    'thread_valid': thread_valid,
                    'thread_cpp': shiboken6.getCppPointer(thread) if thread_valid else None,
                    'thread_name': thread.objectName() if thread_valid else None,
                    'thread_is_running': thread.isRunning() if thread_valid else None,
                    'thread_is_finished': thread.isFinished() if thread_valid else None,
                    'thread_has_stopped': thread_has_stopped(thread) if thread_valid else True,
                    'thread_captured_before_close': thread is not None and any(
                        thread is captured for captured in state.get('threads', ())),
                    'worker_id': id(worker) if worker is not None else None,
                    'worker_type': type(worker).__name__ if worker is not None else None,
                    'worker_valid': worker is not None and shiboken6.isValid(worker)})
            return {'moment': moment, 'operation': state['operation'],
                'elapsed': time.perf_counter()-state['started'], 'handles': rows}

        def close():
            state['timer'].stop()
            report['closing_phase'] = state['operation']
            report['closing_elapsed'] = time.perf_counter()-state['started']
            state['threads'] = [handle.thread for handle in registry().active()]
            state.setdefault('registry_snapshots', []).append(registry_snapshot('before_close'))
            report['normal_close_accepted'] = state['window'].close()
            state['registry_snapshots'].append(registry_snapshot('after_close'))
            save()

        def advance(operation):
            elapsed = time.perf_counter()-state['started']
            assert elapsed < (30 if state['operation'] in ('startup', 'initial') else 20)
            report[state['operation']+'_seconds'] = elapsed
            state.update(operation=operation, started=time.perf_counter())
            save()

        def dismiss_owned_tour(window):
            modal = QApplication.activeModalWidget()
            assert modal is None, 'No consent/provider dialog may be dismissed'
            if window._startup_sequence_running():
                return False
            home_context = state['operation'] == 'startup' and not state.get('module_requested', False)
            tours = [tour for tour in window.findChildren(_TourOverlay) if tour.isVisible()]
            observation = {'operation': state['operation'], 'home_seen': was_tour_shown(),
                'stack_is_home': window._stack.currentWidget() is window._startup,
                'stack_class': type(window._stack.currentWidget()).__name__,
                'tours': [{'pointer': shiboken6.getCppPointer(tour),
                    'key': getattr(tour, '_walkthrough_app_key', None),
                    'owned_by_window': tour.parentWidget() is window,
                    'in_module_owner_list': tour in getattr(window, '_module_walkthroughs', ()),
                    'skip_enabled': tour._skip_btn.isEnabled(), 'skip_visible': tour._skip_btn.isVisible(),
                    'skip_visible_region': tour._skip_btn.visibleRegion().boundingRect().getRect()}
                    for tour in tours]}
            if observation != report.get('tour_observation'):
                report['tour_observation'] = observation
                report.setdefault('tour_history', []).append(observation)
                save()
            if home_context and not tours and not was_tour_shown():
                return False
            if tours:
                assert len(tours) == 1
                tour = tours[0]
                assert tour.parentWidget() is window and tour._skip_btn.isVisible() and tour._skip_btn.isEnabled()
                assert not tour._skip_btn.visibleRegion().isEmpty()
                if home_context:
                    assert getattr(tour, '_walkthrough_app_key', None) is None
                    assert window._stack.currentWidget() is window._startup
                    assert not was_tour_shown()
                else:
                    assert getattr(tour, '_walkthrough_app_key', None) == 'make_masks'
                    assert window._module_walkthroughs == [tour]
                    assert window._stack.currentWidget() is state['screen']
                report.setdefault('owned_tour_skips', []).append(tour._skip_btn.text())
                tour._skip_btn.click()
                return False
            if home_context:
                assert was_tour_shown()
                report['home_tour_hidden_and_seen_before_module'] = True
            return True

        def install_paint_observer(stage):
            canvas = stage._panel.canvas._canvas
            cls, original = type(canvas), type(canvas).paintEvent
            state.update(canvas=canvas, paint_class=cls, original_paint=original, painted=[])

            def paint(widget, event):
                original(widget, event)
                if widget is canvas and stage._render_state is not None:
                    rendered, artist = stage._render_state, stage._render_artist
                    state['painted'].append({'serial': stage._shown_serial, 'state_id': id(rendered),
                        'artist_visible': artist is not None and artist.get_visible(),
                        'image_exact': artist is not None and np.array_equal(np.asarray(artist.get_array()), rendered.rgb),
                        'offset_exact': artist is not None and (artist.ox, artist.oy) ==
                            (int(rendered.rect[0]), int(rendered.rect[2])),
                        'rect': list(rendered.rect), 'backend': rendered.backend, 'scale': rendered.scale.percent})
            cls.paintEvent = paint

        def layer_snapshot(stage):
            rows = []
            for layer in stage._layers:
                image = isinstance(layer, layers.ImageLayer)
                controls = stage._layer_rows[layer.name]
                rows.append({'name': layer.name, 'image': image, 'identity': id(layer),
                    'visible': layer.visible, 'opacity': layer.opacity,
                    'colormap': layer.colormap.name if image else None,
                    'contrast': tuple(layer.contrast_limits(0)) if image else None,
                    'label_seed': None if image else layer._seed,
                    'rendering': layer.metadata['rendering'], 'finish': layer.metadata.get('finish'),
                    'ui': {'eye': controls.eye.isChecked(), 'opacity': controls.opacity.value(),
                        'colour': controls.colour.currentData() if controls.colour is not None else None,
                        'low': controls.low.value() if controls.low is not None else None,
                        'high': controls.high.value() if controls.high is not None else None,
                        'rendering': controls.rendering.currentData(),
                        'finish': controls.finish.currentData() if controls.finish is not None else None}})
            return rows

        def request_pose(stage, request):
            axes = stage._view_axes()
            assert request.lens is None, 'This opacity probe does not enable the magnifier'
            return {'session': id(request.session), 'cache': id(request.cache), 'quality': request.quality,
                'rect': tuple(request.rect), 'steps': tuple(request.steps), 'version': request.version,
                'highlight': request.highlight, 'voxel_size': tuple(request.voxel_size),
                'rays': (tuple(request.rays.origin), tuple(request.rays.across), tuple(request.rays.down),
                    tuple(request.rays.depth), tuple(request.rays.shape)),
                'scale': (request.scale.percent, request.scale.pixel, request.scale.max_voxels),
                'lens': request.lens,
                'camera': (axes.elev, axes.azim, axes.roll, tuple(axes.get_xlim3d()),
                    tuple(axes.get_ylim3d()), tuple(axes.get_zlim3d()))}

        def current_request(stage, quality):
            bookkeeping = (stage._render_serial, stage._render_floor, stage._render_turning)
            try:
                return stage._render_request(quality)
            finally:
                stage._render_serial, stage._render_floor, stage._render_turning = bookkeeping

        def image_state(stage, context, request):
            session = stage.session
            return {'operation': state['operation'], 'pose': request_pose(stage, request),
                'layers': layer_snapshot(stage), 'frame': dict(report['current_frame']),
                'preferences': preferences.get_volume_view(), 'selected': stage._selected,
                'selected_ui': stage._labels.currentData(), 'context': id(context),
                'data_history_unchanged': (
                    (id(session.image), id(session.mask)) == state['session_arrays']
                    and digest(session.image) == state['image'] and digest(session.mask) == state['mask']
                    and (tuple(map(id, session._undo)), tuple(map(id, session._redo))) == state['history']
                    and stage.dirty == state['dirty'])}

        def begin_control_action(stage, operation, target, baseline, ledger):
            advance(operation)
            ancestor = target.parentWidget()
            reveals = []
            while ancestor is not None and ancestor is not state['window']:
                if isinstance(ancestor, QScrollArea):
                    ancestor.ensureWidgetVisible(target, 0, 0)
                    reveals.append(ancestor.objectName())
                ancestor = ancestor.parentWidget()
            action = {'operation': operation, 'target': target.objectName(), 'reveals': reveals,
                'enabled': target.isEnabled(), 'visible': target.isVisible(),
                'visible_region': target.visibleRegion().boundingRect().getRect(),
                'elapsed_at_reveal': time.perf_counter()-state['started']}
            report[ledger].append(action)
            save()
            assert 'VolumeLayerScroll' in reveals
            assert target.isEnabled() and target.isVisible() and not target.visibleRegion().isEmpty()
            assert request_pose(stage, current_request(stage, 'preview')) == baseline['pose']
            return action

        def act_on_image_control(stage, operation):
            controls = state['image_controls']
            target = controls.colour if operation.startswith('colour_') else (
                controls.eye if operation.startswith('visibility_') else (
                    controls.high if operation.startswith('contrast_') else controls.rendering))
            action = begin_control_action(stage, operation, target, state['controls_baseline'], 'image_control_actions')
            if operation.startswith('colour_'):
                value = 'green' if operation == 'colour_changed' else 'gray'
                index = target.findData(value)
                assert index >= 0 and target.currentIndex() != index
                target.setCurrentIndex(index)
            elif operation.startswith('visibility_'):
                expected = operation == 'visibility_restored'
                assert target.isChecked() != expected
                target.click()
                assert target.isChecked() == expected
            elif operation.startswith('contrast_'):
                target.setValue(state['contrast_target'] if operation == 'contrast_changed' else state['contrast_original'])
            else:
                value = 'attenuated_mip' if operation == 'rendering_changed' else 'mip'
                if scenario == 'image_renderings' and operation.endswith('_changed'):
                    value = operation.rsplit('_', 1)[0]
                index = target.findData(value)
                assert index >= 0 and target.currentIndex() != index
                target.setCurrentIndex(index)
            action['elapsed_after_action'] = time.perf_counter()-state['started']
            save()
            assert action['elapsed_after_action'] < 20

        def act_on_mask_control(stage, operation):
            controls = state['mask_controls']
            target = controls.eye if operation.startswith('mask_visibility_') else (
                controls.opacity if operation.startswith('mask_opacity_') else (
                    controls.rendering if operation.startswith('mask_rendering_') else controls.finish))
            action = begin_control_action(stage, operation, target, state['mask_baseline'], 'mask_control_actions')
            changed = operation.endswith('_changed')
            if operation.startswith('mask_visibility_'):
                expected = operation == 'mask_visibility_restored'
                assert target.isChecked() != expected
                target.click()
                assert target.isChecked() == expected
            elif operation.startswith('mask_opacity_'):
                value = 37 if changed else 60
                assert target.value() != value
                target.setValue(value)
            else:
                value = ('translucent' if changed else 'surface') if operation.startswith('mask_rendering_') else (
                    'satin' if changed else 'matte')
                if scenario == 'mask_finishes' and changed:
                    value = operation.rsplit('_', 1)[0]
                index = target.findData(value)
                assert index >= 0 and target.currentIndex() != index
                target.setCurrentIndex(index)
            action['elapsed_after_action'] = time.perf_counter()-state['started']
            save()
            assert action['elapsed_after_action'] < 20

        def tick():
            try:
                elapsed = time.perf_counter()-state['started']
                assert elapsed < (30 if state['operation'] in ('startup', 'initial') else 20), state['operation']
                window = state['window']
                assert QApplication.activeModalWidget() is None, 'No consent/provider dialog may be dismissed'
                if state['operation'] == 'startup' and state.get('module_requested', False):
                    screen = window._screens.get('make_masks')
                    publication = {'registered_keys': sorted(window._screens),
                        'opening': bool(getattr(window, '_opening_a_screen', False)),
                        'screen_pointer': shiboken6.getCppPointer(screen) if screen is not None else None,
                        'screen_class': type(screen).__name__ if screen is not None else None,
                        'current': screen is not None and window._stack.currentWidget() is screen,
                        'visible': screen is not None and screen.isVisible()}
                    if publication != report.get('module_publication'):
                        report['module_publication'] = publication
                        report.setdefault('module_publication_history', []).append(publication)
                        save()
                    if screen is None:
                        return
                    assert isinstance(screen, mm.MakeMasksScreen)
                    if publication['opening'] or not publication['current'] or not publication['visible']:
                        return
                    state['screen'] = screen
                if not dismiss_owned_tour(window):
                    return
                if state['operation'] == 'startup':
                    if not state.get('module_requested', False):
                        assert report['home_tour_hidden_and_seen_before_module']
                        state['module_requested'] = True
                        report['module_request'] = {'opening_before': bool(getattr(window, '_opening_a_screen', False)),
                            'registered_before': sorted(window._screens),
                            'stack_before': type(window._stack.currentWidget()).__name__}
                        save()
                        report['module_request']['returned'] = window.open_module('make_masks')
                        report['module_request']['opening_after'] = bool(getattr(window, '_opening_a_screen', False))
                        report['module_request']['registered_after'] = sorted(window._screens)
                        save()
                        assert report['module_request']['returned'] == 'make_masks'
                        return
                    screen = state['screen']
                    assert load_the_fibsem_test_data(screen, with_masks=True, folder=folder,
                        ask=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError('No download allowed')))
                    stage = screen._mask_stage
                    session = stage.session
                    assert session.image.shape == (128, 256, 256) and session.axes == 'ZYX'
                    assert set(np.unique(session.mask)) == set(range(44))
                    state.update(stage=stage, image=digest(session.image), mask=digest(session.mask),
                        history=(tuple(map(id, session._undo)), tuple(map(id, session._redo))), dirty=stage.dirty)
                    if scenario in ('image_opacity', 'image_controls', 'mask_controls', 'mask_visibility',
                            'image_colour', 'image_renderings', 'mask_finishes'):
                        state['session_arrays'] = (id(session.image), id(session.mask))
                    assert 'metal' in mm.volume_render_backends()
                    report['before_true3d_saved'] = preferences.get_volume_view()
                    if phase == 'B':
                        assert report['before_true3d_saved']['backend'] == 'metal' and report['before_true3d_saved']['scale'] == 75
                    assert stage.set_view('true3d')
                    install_paint_observer(stage)
                    if phase == 'A':
                        index = stage._backend_box.findData('metal')
                        assert index >= 0 and stage._backend_box.currentData() != 'metal'
                        stage._backend_box.setCurrentIndex(index)
                        assert stage._scale_slider.value() != 75
                        stage._scale_slider.setValue(75)
                    advance('initial')
                    return
                stage = state['stage']
                if state['operation'] in ('initial', 'opacity_changed', 'opacity_restored',
                        'visibility_hidden', 'visibility_restored', 'contrast_changed', 'contrast_restored',
                        'rendering_changed', 'rendering_restored', 'mask_opacity_changed', 'mask_opacity_restored',
                        'mask_rendering_changed', 'mask_rendering_restored', 'mask_finish_changed', 'mask_finish_restored',
                        'mask_visibility_hidden', 'mask_visibility_restored', 'colour_changed', 'colour_restored',
                        *control_operations, *mask_operations):
                    if not idle(stage) or stage._render_state is None:
                        return
                    if scenario == 'image_opacity' and 'opacity_slider' not in state:
                        images = [layer for layer in stage._layers if isinstance(layer, layers.ImageLayer)]
                        masks = [layer for layer in stage._layers if isinstance(layer, layers.LabelsLayer)]
                        assert len(images) == len(masks) == 1
                        slider = stage._layer_rows[images[0].name].opacity
                        ancestor = slider.parentWidget()
                        reveals = []
                        while ancestor is not None and ancestor is not window:
                            if isinstance(ancestor, QScrollArea):
                                ancestor.ensureWidgetVisible(slider, 0, 0)
                                reveals.append(ancestor.objectName())
                            ancestor = ancestor.parentWidget()
                        report['opacity_control'] = {'name': slider.objectName(),
                            'enabled': slider.isEnabled(), 'visible': slider.isVisible(),
                            'visible_region': slider.visibleRegion().boundingRect().getRect(), 'reveals': reveals}
                        save()
                        assert 'VolumeLayerScroll' in reveals
                        assert slider.isEnabled() and slider.isVisible() and not slider.visibleRegion().isEmpty()
                        assert slider.value() != 37
                        state.update(opacity_slider=slider, opacity_layer=images[0], original_opacity=slider.value())
                        return
                    assert stage._backend_box.currentData() == stage._backend_choice == 'metal'
                    assert stage._scale_slider.value() == stage._scale == stage._render_percent() == 75
                    assert preferences.get_volume_view()['backend'] == 'metal' and preferences.get_volume_view()['scale'] == 75
                    context, rendered = stage._gl, stage._render_state
                    assert context is not None and context.backend == rendered.backend == 'metal'
                    assert rendered.scale.percent == 75 and np.any(rendered.rgb)
                    if scenario == 'mask_visibility' and state['operation'] == 'mask_visibility_hidden':
                        assert not np.any(rendered.picked)
                    else:
                        assert np.any(rendered.picked > 0)
                    if state['canvas']._draw_pending:
                        return
                    matching = [row for row in state['painted'] if row['state_id'] == id(rendered)
                        and row['serial'] == stage._shown_serial and row['image_exact'] and row['offset_exact']
                        and row['artist_visible']]
                    if not matching:
                        state['canvas'].draw_idle()
                        return
                    request = current_request(stage, rendered.quality)
                    rgb, picked = context.render(request)
                    assert tuple(request.rect) == tuple(rendered.rect) and request.scale.percent == 75
                    assert np.array_equal(rgb, rendered.rgb) and np.array_equal(picked, rendered.picked)
                    report['current_frame'] = {'serial': stage._shown_serial, 'rgb': digest(rgb), 'picked': digest(picked),
                        'backend': context.backend, 'context_name': context.name(), 'scale': stage._scale,
                        'positive_original_ids': sorted(map(int, np.unique(picked[picked > 0]))),
                        'fresh_request_exact': True, 'paint': matching[-1]}
                    assert set(report['current_frame']['positive_original_ids']) <= set(range(1, 44))
                    if scenario in ('mask_controls', 'mask_visibility', 'mask_finishes'):
                        assert request.quality == 'preview'
                        row = image_state(stage, context, request)
                        row['effective_finish'] = stage._finish
                        report['mask_control_states'].append(row)
                        save()
                        assert row['data_history_unchanged']
                        state['context'] = context
                        if state['operation'] == 'initial':
                            masks = [layer for layer in stage._layers if isinstance(layer, layers.LabelsLayer)]
                            assert len(masks) == 1
                            mask = masks[0]
                            controls = stage._layer_rows[mask.name]
                            assert mask.visible and controls.eye.isChecked()
                            assert mask.opacity == .6 and controls.opacity.value() == 60
                            assert mask.metadata['rendering'] == controls.rendering.currentData() == 'surface'
                            assert mask.metadata['finish'] == stage._finish == controls.finish.currentData() == 'matte'
                            state.update(mask_baseline=row, mask_rgb=rgb.copy(), mask_picked=picked.copy(),
                                mask_layer=mask, mask_controls=controls)
                        else:
                            baseline = state['mask_baseline']
                            assert row['pose'] == baseline['pose'] and row['context'] == baseline['context']
                            assert row['selected'] == baseline['selected'] and row['selected_ui'] == baseline['selected_ui']
                            assert row['frame']['serial'] > report['mask_control_states'][-2]['frame']['serial']
                            finish = 'satin' if state['operation'] == 'mask_finish_changed' else 'matte'
                            if scenario == 'mask_finishes' and state['operation'].endswith('_changed'):
                                finish = state['operation'].rsplit('_', 1)[0]
                            assert stage._finish == finish
                            expected_prefs = dict(baseline['preferences'])
                            if state['operation'].startswith('mask_finish_') or scenario == 'mask_finishes':
                                expected_prefs['finish'] = finish
                            assert row['preferences'] == expected_prefs
                            expected = []
                            for original in baseline['layers']:
                                copied = dict(original, ui=dict(original['ui']))
                                if copied['name'] == state['mask_layer'].name:
                                    if state['operation'] == 'mask_visibility_hidden':
                                        copied['visible'] = copied['ui']['eye'] = False
                                    elif state['operation'] == 'mask_opacity_changed':
                                        copied['opacity'], copied['ui']['opacity'] = .37, 37
                                    elif state['operation'] == 'mask_rendering_changed':
                                        copied['rendering'] = copied['ui']['rendering'] = 'translucent'
                                    elif state['operation'] == 'mask_finish_changed':
                                        copied['finish'] = copied['ui']['finish'] = 'satin'
                                    elif scenario == 'mask_finishes' and state['operation'].endswith('_changed'):
                                        copied['finish'] = copied['ui']['finish'] = finish
                                expected.append(copied)
                            assert row['layers'] == expected
                            if state['operation'].endswith('_restored'):
                                assert row['layers'] == baseline['layers']
                                assert np.array_equal(rgb, state['mask_rgb']) and np.array_equal(picked, state['mask_picked'])
                            else:
                                assert not np.array_equal(rgb, state['mask_rgb'])
                        index = mask_operations.index(state['operation'])
                        if index == len(mask_operations)-1:
                            report['mask_controls_restored'] = True
                            advance('current_idle')
                        else:
                            act_on_mask_control(stage, mask_operations[index+1])
                        return
                    if scenario in ('image_controls', 'image_colour', 'image_renderings'):
                        assert request.quality == 'preview'
                        row = image_state(stage, context, request)
                        report['image_control_states'].append(row)
                        save()
                        assert row['data_history_unchanged']
                        state['context'] = context
                        if state['operation'] == 'initial':
                            images = [layer for layer in stage._layers if isinstance(layer, layers.ImageLayer)]
                            masks = [layer for layer in stage._layers if isinstance(layer, layers.LabelsLayer)]
                            assert len(images) == len(masks) == 1
                            image = images[0]
                            controls = stage._layer_rows[image.name]
                            low, high = image.contrast_limits(0)
                            assert image.visible and controls.eye.isChecked()
                            assert image.metadata['rendering'] == controls.rendering.currentData() == 'mip'
                            assert controls.low.value() == low == 60 and controls.high.value() == high == 192
                            assert controls.high.minimum() <= 126 <= controls.high.maximum() and low < 126 < high
                            if scenario == 'image_colour':
                                assert image.colormap.name == controls.colour.currentData() == 'gray'
                            state.update(controls_baseline=row, controls_rgb=rgb.copy(), controls_picked=picked.copy(),
                                image_layer=image, image_controls=controls, contrast_original=high, contrast_target=126)
                        else:
                            baseline = state['controls_baseline']
                            assert row['pose'] == baseline['pose'] and row['context'] == baseline['context']
                            assert row['selected'] == baseline['selected'] and row['selected_ui'] == baseline['selected_ui']
                            assert row['preferences'] == baseline['preferences']
                            assert row['frame']['serial'] > report['image_control_states'][-2]['frame']['serial']
                            expected = []
                            for original in baseline['layers']:
                                copied = dict(original, ui=dict(original['ui']))
                                if copied['name'] == state['image_layer'].name:
                                    if state['operation'] == 'visibility_hidden':
                                        copied['visible'], copied['ui']['eye'] = False, False
                                    elif state['operation'] == 'contrast_changed':
                                        copied['contrast'] = (copied['contrast'][0], 126.)
                                        copied['ui']['high'] = 126.
                                    elif state['operation'] == 'rendering_changed':
                                        copied['rendering'] = copied['ui']['rendering'] = 'attenuated_mip'
                                    elif state['operation'] == 'colour_changed':
                                        copied['colormap'] = copied['ui']['colour'] = 'green'
                                    elif scenario == 'image_renderings' and state['operation'].endswith('_changed'):
                                        copied['rendering'] = copied['ui']['rendering'] = state['operation'].rsplit('_', 1)[0]
                                expected.append(copied)
                            assert row['layers'] == expected
                            if scenario == 'image_colour':
                                assert np.array_equal(picked, state['controls_picked'])
                            if state['operation'].endswith('_restored'):
                                assert row['layers'] == baseline['layers']
                                assert np.array_equal(rgb, state['controls_rgb']) and np.array_equal(picked, state['controls_picked'])
                            else:
                                assert not np.array_equal(rgb, state['controls_rgb'])
                        index = control_operations.index(state['operation'])
                        if index == len(control_operations)-1:
                            report['image_controls_restored'] = True
                            advance('current_idle')
                        else:
                            act_on_image_control(stage, control_operations[index+1])
                        return
                    if scenario == 'image_opacity':
                        assert request.quality == 'preview'
                        assert state['opacity_slider'].isEnabled() and state['opacity_slider'].isVisible()
                        assert not state['opacity_slider'].visibleRegion().isEmpty()
                        row = image_state(stage, context, request)
                        pose, rows = row['pose'], row['layers']
                        report['opacity_states'].append(row)
                        save()
                        assert row['data_history_unchanged']
                        state['context'] = context
                        if state['operation'] == 'initial':
                            state.update(opacity_baseline=row, baseline_rgb=rgb.copy(), baseline_picked=picked.copy())
                            state['opacity_slider'].setValue(37)
                            advance('opacity_changed')
                            return
                        baseline = state['opacity_baseline']
                        assert pose == baseline['pose'] and id(context) == baseline['context']
                        assert stage._selected == baseline['selected'] and stage._labels.currentData() == baseline['selected_ui']
                        assert stage._shown_serial > report['opacity_states'][-2]['frame']['serial']
                        assert preferences.get_volume_view() == baseline['preferences']
                        expected = []
                        for original in baseline['layers']:
                            copied = dict(original, ui=dict(original['ui']))
                            if copied['name'] == state['opacity_layer'].name and state['operation'] == 'opacity_changed':
                                copied['opacity'], copied['ui']['opacity'] = .37, 37
                            expected.append(copied)
                        assert rows == expected
                        if state['operation'] == 'opacity_changed':
                            assert not np.array_equal(rgb, state['baseline_rgb'])
                            report['opacity_rgb_changed'] = True
                            state['opacity_slider'].setValue(state['original_opacity'])
                            advance('opacity_restored')
                            return
                        assert rows == baseline['layers']
                        assert np.array_equal(rgb, state['baseline_rgb'])
                        assert np.array_equal(picked, state['baseline_picked'])
                        report['opacity_frame_restored'] = True
                        advance('current_idle')
                        return
                    row, column = map(int, np.argwhere(picked == 0)[0])
                    x0, x1, y0, y1 = rendered.rect
                    rows, columns = picked.shape
                    x, y = x0+(column+.5)*(x1-x0)/columns, y1-(row+.5)*(y1-y0)/rows
                    assert x0 < x < x1 and y0 < y < y1
                    returned = int(stage.pick_at(x, y))
                    report['pick'] = {'row': row, 'column': column, 'expected': 0, 'returned': returned,
                        'x': x, 'y': y, 'shown_serial': stage._shown_serial}
                    assert returned == 0
                    state['context'] = context
                    advance('current_idle')
                    return
                if not idle(stage):
                    return
                session = stage.session
                assert digest(session.image) == state['image'] and digest(session.mask) == state['mask']
                assert (tuple(map(id, session._undo)), tuple(map(id, session._redo))) == state['history']
                assert stage.dirty == state['dirty']
                report.update(data_history_unchanged=True, saved=preferences.get_volume_view(),
                    current_idle_seconds=time.perf_counter()-state['started'])
                preferences._settings().sync()
                close()
            except Exception:
                report['errors'].append(traceback.format_exc())
                save()
                close()

        def prepare(entry):
            if entry.get('detail') != '__home__' or state:
                return
            windows = [window for window in QApplication.instance().topLevelWidgets() if isinstance(window, MainWindow)]
            assert len(windows) == 1
            window = windows[0]
            report['native'] = {'platform': QApplication.instance().platformName(),
                'exposed': window.windowHandle().isExposed(),
                'dpr': window.windowHandle().devicePixelRatio(), 'work': window.screen().availableGeometry().getRect()}
            save()
            assert report['native']['platform'] == 'cocoa' and report['native']['dpr'] == 2. and report['native']['exposed']
            state.update(window=window, operation='startup', started=time.perf_counter())
            timer = QTimer(window)
            timer.setInterval(50)
            timer.timeout.connect(tick)
            state['timer'] = timer
            timer.start()
            save()

        def ready(entry):
            try:
                prepare(entry)
            except Exception:
                report['errors'].append(traceback.format_exc())
                windows = [window for window in QApplication.instance().topLevelWidgets() if isinstance(window, MainWindow)]
                assert len(windows) == 1
                state.setdefault('window', windows[0])
                state.setdefault('operation', 'startup')
                state.setdefault('started', time.perf_counter())
                state.setdefault('timer', QTimer(windows[0]))
                close()

        timing.subscribe_readiness(ready)
        from spacr.qt import run
        result = run(['--no-setup', '--fresh'])
        timing.unsubscribe_readiness(ready)
        if 'paint_class' in state:
            state['paint_class'].paintEvent = state['original_paint']
        from spacr.qt.widgets import ambient
        context = state.get('context')
        renderer = context.renderer if context is not None else None
        report.update(normal_return=result, remaining_jobs=len(registry().active()),
            owned_threads_alive=sum(not thread_has_stopped(thread) for thread in state.get('threads', [])),
            compiler_threads=[thread.name for thread in ambient._COMPILER_THREADS if thread.is_alive()],
            compiler_helper_present=ambient._PRIME_PROCESS is not None,
            gpu_resources_released=renderer is not None and not any((renderer._textures, renderer._buffers,
                renderer._pipelines, renderer._luts, renderer._dispatch_slots)))
        state.setdefault('registry_snapshots', []).append(registry_snapshot('after_run_return'))
        if report['remaining_jobs']:
            report['registry_diagnostic'] = state['registry_snapshots']
        save()
        assert result == 0 and not report['errors'] and report.get('normal_close_accepted'), report
        assert report['startup_seconds'] < 30 and report['initial_seconds'] < 30
        assert report['current_idle_seconds'] < 20 and report['closing_elapsed'] < 20
        if scenario == 'image_opacity':
            assert report['opacity_changed_seconds'] < 20 and report['opacity_restored_seconds'] < 20
            assert report['opacity_rgb_changed'] and report['opacity_frame_restored']
            assert [row['operation'] for row in report['opacity_states']] == ['initial', 'opacity_changed', 'opacity_restored']
            assert all(row['data_history_unchanged'] for row in report['opacity_states'])
            assert 'pick' not in report
        elif scenario in ('image_controls', 'image_colour', 'image_renderings'):
            assert report['image_controls_restored'] and 'pick' not in report
            assert [row['operation'] for row in report['image_control_states']] == control_operations
            assert [row['operation'] for row in report['image_control_actions']] == control_operations[1:]
            baseline = report['image_control_states'][0]
            for i, row in enumerate(report['image_control_states']):
                assert report[row['operation']+'_seconds'] < (30 if i == 0 else 20)
                assert row['data_history_unchanged'] and row['frame']['fresh_request_exact']
                assert row['pose'] == baseline['pose'] and row['preferences'] == baseline['preferences']
                if scenario == 'image_colour':
                    assert row['frame']['picked'] == baseline['frame']['picked']
                if i:
                    assert row['frame']['serial'] > report['image_control_states'][i-1]['frame']['serial']
                    if row['operation'].endswith('_restored'):
                        assert row['frame']['rgb'] == baseline['frame']['rgb'] and row['frame']['picked'] == baseline['frame']['picked']
                    else:
                        assert row['frame']['rgb'] != baseline['frame']['rgb']
            assert all(row['enabled'] and row['visible'] and row['visible_region'][2] > 0
                and row['visible_region'][3] > 0 and row['elapsed_after_action'] < 20
                for row in report['image_control_actions'])
        if scenario in ('mask_controls', 'mask_visibility', 'mask_finishes'):
            assert report['scenario'] == scenario and report['mask_controls_restored'] and 'pick' not in report
            operations = ['initial', 'mask_opacity_changed', 'mask_opacity_restored',
                'mask_rendering_changed', 'mask_rendering_restored', 'mask_finish_changed', 'mask_finish_restored']
            if scenario == 'mask_visibility':
                operations = ['initial', 'mask_visibility_hidden', 'mask_visibility_restored']
            elif scenario == 'mask_finishes':
                operations = ['initial', 'glossy_changed', 'glossy_restored', 'metallic_changed', 'metallic_restored']
            assert [row['operation'] for row in report['mask_control_states']] == operations
            assert [row['operation'] for row in report['mask_control_actions']] == operations[1:]
            baseline = report['mask_control_states'][0]
            for i, row in enumerate(report['mask_control_states']):
                assert report[row['operation']+'_seconds'] < (30 if i == 0 else 20)
                assert row['data_history_unchanged'] and row['frame']['fresh_request_exact']
                assert row['pose'] == baseline['pose'] and row['context'] == baseline['context']
                finish = 'satin' if row['operation'] == 'mask_finish_changed' else 'matte'
                if scenario == 'mask_finishes' and row['operation'].endswith('_changed'):
                    finish = row['operation'].rsplit('_', 1)[0]
                assert row['effective_finish'] == finish
                expected_prefs = dict(baseline['preferences'])
                if row['operation'].startswith('mask_finish_') or (scenario == 'mask_finishes' and i):
                    expected_prefs['finish'] = finish
                assert row['preferences'] == expected_prefs
                if scenario == 'mask_visibility':
                    assert bool(row['frame']['positive_original_ids']) == (row['operation'] != 'mask_visibility_hidden')
                if i:
                    assert row['frame']['serial'] > report['mask_control_states'][i-1]['frame']['serial']
                    if row['operation'].endswith('_restored'):
                        assert row['layers'] == baseline['layers']
                        assert row['frame']['rgb'] == baseline['frame']['rgb'] and row['frame']['picked'] == baseline['frame']['picked']
                    else:
                        assert row['frame']['rgb'] != baseline['frame']['rgb']
            assert report['saved'] == (baseline['preferences'] if scenario == 'mask_visibility'
                else dict(baseline['preferences'], finish='matte'))
            assert all(row['enabled'] and row['visible'] and row['visible_region'][2] > 0
                and row['visible_region'][3] > 0 and row['elapsed_after_action'] < 20
                for row in report['mask_control_actions'])
        assert report.get('data_history_unchanged') and report['current_frame']['fresh_request_exact']
        assert report['gpu_resources_released'] and report['remaining_jobs'] == report['owned_threads_alive'] == 0
        assert not report['compiler_threads'] and not report['compiler_helper_present']
    ''')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    env = os.environ.copy()
    for name in ('CUDA_VISIBLE_DEVICES', 'SPACR_NO_GPU_RENDER', 'SPACR_GPU_RENDER',
                 'SPACR_ISOLATED_PROFILE', 'SPACR_HOME'):
        env.pop(name, None)
    env.update(QT_QPA_PLATFORM='cocoa', SPACR_ISOLATED_PROFILE='1', SPACR_HOME=str(tmp_path / 'profile'),
        SPACR_MENU_TEST_SETTINGS=str(tmp_path.resolve()), SPACR_RESTART_ROOT=str(root), SPACR_RESTART_HEAD=head,
        SPACR_RESTART_FIXTURE=str(fixture), SPACR_NO_NETWORK='1', SPACR_TIMING='1',
        XDG_CONFIG_HOME=str(tmp_path / 'settings'), XDG_CACHE_HOME=str(tmp_path / 'cache'),
        MPLCONFIGDIR=str(tmp_path / 'cache' / 'matplotlib'), NUMBA_CACHE_DIR=str(tmp_path / 'numba'))
    env['SPACR_VOLUME_SCENARIO'] = scenario
    reports = []
    for phase in (('A', 'B') if scenario == 'restart' else ('A',)):
        output = tmp_path / ('restart-'+phase+'.json')
        env.update(SPACR_RESTART_PHASE=phase, SPACR_RESTART_OUTPUT=str(output),
                   SPACR_TIMING_LOG=str(tmp_path / ('timing-'+phase+'.txt')))
        completed = subprocess.run([sys.executable, '-c', child], cwd=root, env=env,
                                   capture_output=True, text=True, timeout=180)
        (tmp_path / ('restart-'+phase+'-stdout.txt')).write_text(completed.stdout+completed.stderr)
        assert completed.returncode == 0, completed.stdout+completed.stderr
        for marker in ('Error calling Python override', 'Traceback (most recent call last):',
                       'spaCR wrote a crash report'):
            assert marker not in completed.stdout+completed.stderr, completed.stdout+completed.stderr
        report = json.loads(output.read_text())
        assert report['phase'] == phase and report['normal_return'] == 0 and report['normal_close_accepted']
        assert report['native']['platform'] == 'cocoa' and report['native']['dpr'] == 2. and report['native']['exposed']
        assert report['home_tour_hidden_and_seen_before_module']
        assert report['startup_seconds'] < 30 and report['initial_seconds'] < 30
        assert report['current_idle_seconds'] < 20 and report['closing_elapsed'] < 20
        assert report['saved']['backend'] == 'metal' and report['saved']['scale'] == 75
        assert report['current_frame']['backend'] == 'metal' and report['current_frame']['scale'] == 75
        assert report['current_frame']['fresh_request_exact']
        if scenario == 'restart':
            assert report['pick']['returned'] == report['pick']['expected'] == 0
        elif scenario == 'image_opacity':
            assert report['scenario'] == scenario and report['opacity_changed_seconds'] < 20 and report['opacity_restored_seconds'] < 20
            assert report['opacity_rgb_changed'] and report['opacity_frame_restored'] and 'pick' not in report
            assert [row['operation'] for row in report['opacity_states']] == ['initial', 'opacity_changed', 'opacity_restored']
            assert all(row['data_history_unchanged'] for row in report['opacity_states'])
        elif scenario in ('mask_controls', 'mask_visibility', 'mask_finishes'):
            assert report['scenario'] == scenario and report['mask_controls_restored'] and 'pick' not in report
            operations = ['initial', 'mask_opacity_changed', 'mask_opacity_restored',
                'mask_rendering_changed', 'mask_rendering_restored', 'mask_finish_changed', 'mask_finish_restored']
            if scenario == 'mask_visibility':
                operations = ['initial', 'mask_visibility_hidden', 'mask_visibility_restored']
            elif scenario == 'mask_finishes':
                operations = ['initial', 'glossy_changed', 'glossy_restored', 'metallic_changed', 'metallic_restored']
            assert [row['operation'] for row in report['mask_control_states']] == operations
            assert [row['operation'] for row in report['mask_control_actions']] == operations[1:]
            baseline = report['mask_control_states'][0]
            for i, row in enumerate(report['mask_control_states']):
                assert report[row['operation']+'_seconds'] < (30 if i == 0 else 20)
                assert row['data_history_unchanged'] and row['frame']['fresh_request_exact']
                assert row['pose'] == baseline['pose'] and row['context'] == baseline['context']
                finish = 'satin' if row['operation'] == 'mask_finish_changed' else 'matte'
                if scenario == 'mask_finishes' and row['operation'].endswith('_changed'):
                    finish = row['operation'].rsplit('_', 1)[0]
                assert row['effective_finish'] == finish
                expected_prefs = dict(baseline['preferences'])
                if row['operation'].startswith('mask_finish_') or (scenario == 'mask_finishes' and i):
                    expected_prefs['finish'] = finish
                assert row['preferences'] == expected_prefs
                if scenario == 'mask_visibility':
                    assert bool(row['frame']['positive_original_ids']) == (row['operation'] != 'mask_visibility_hidden')
                if i:
                    assert row['frame']['serial'] > report['mask_control_states'][i-1]['frame']['serial']
                    if row['operation'].endswith('_restored'):
                        assert row['layers'] == baseline['layers']
                        assert row['frame']['rgb'] == baseline['frame']['rgb'] and row['frame']['picked'] == baseline['frame']['picked']
                    else:
                        assert row['frame']['rgb'] != baseline['frame']['rgb']
            assert report['saved'] == (baseline['preferences'] if scenario == 'mask_visibility'
                else dict(baseline['preferences'], finish='matte'))
            assert all(row['enabled'] and row['visible'] and row['visible_region'][2] > 0
                and row['visible_region'][3] > 0 and row['elapsed_after_action'] < 20
                for row in report['mask_control_actions'])
        else:
            assert report['scenario'] == scenario and report['image_controls_restored'] and 'pick' not in report
            operations = ['initial', 'visibility_hidden', 'visibility_restored',
                'contrast_changed', 'contrast_restored', 'rendering_changed', 'rendering_restored']
            if scenario == 'image_colour':
                operations = ['initial', 'colour_changed', 'colour_restored']
            elif scenario == 'image_renderings':
                operations = ['initial', 'translucent_changed', 'translucent_restored',
                    'average_changed', 'average_restored', 'minip_changed', 'minip_restored']
            assert [row['operation'] for row in report['image_control_states']] == operations
            assert [row['operation'] for row in report['image_control_actions']] == operations[1:]
            baseline = report['image_control_states'][0]
            for i, row in enumerate(report['image_control_states']):
                assert report[row['operation']+'_seconds'] < (30 if i == 0 else 20)
                assert row['data_history_unchanged'] and row['frame']['fresh_request_exact']
                assert row['pose'] == baseline['pose'] and row['preferences'] == baseline['preferences']
                if scenario == 'image_colour':
                    assert row['frame']['picked'] == baseline['frame']['picked']
                if i:
                    assert row['frame']['serial'] > report['image_control_states'][i-1]['frame']['serial']
                    if row['operation'].endswith('_restored'):
                        assert row['frame']['rgb'] == baseline['frame']['rgb'] and row['frame']['picked'] == baseline['frame']['picked']
                    else:
                        assert row['frame']['rgb'] != baseline['frame']['rgb']
            assert all(row['enabled'] and row['visible'] and row['visible_region'][2] > 0
                and row['visible_region'][3] > 0 and row['elapsed_after_action'] < 20
                for row in report['image_control_actions'])
        assert report['data_history_unchanged'] and report['gpu_resources_released']
        assert report['remaining_jobs'] == report['owned_threads_alive'] == 0
        assert not report['errors'] and not report['compiler_threads'] and not report['compiler_helper_present']
        reports.append(report)
        assert {name: hashlib.sha256((fixture / name).read_bytes()).hexdigest() for name in required} == checksums
    if scenario == 'restart':
        assert reports[0]['settings_file'] == reports[1]['settings_file']
        assert reports[0]['source_head'] == reports[1]['source_head'] == head
        assert reports[0]['runtime'] == reports[1]['runtime'] == sys.executable
        assert reports[1]['initial_saved'] == reports[0]['saved']


def test_auto_falls_back_to_the_cpu_with_a_one_line_note(view, qtbot, monkeypatch, caplog):
    stage = view

    def broken(*_args, **_kwargs):
        raise RuntimeError("no OpenGL context could be created")

    monkeypatch.setattr(mm, "_GLVolumeContext", broken)
    # 2026-10-16: the chain is OpenGL alone here, with the GPU allowed; the
    # wgpu steps of Auto are covered in test_673_true3d_wgpu.py.
    monkeypatch.setattr(mm, "volume_gpu_refusal", lambda: "")
    monkeypatch.setattr(mm, "volume_render_backends", lambda: ("auto", "opengl", "cpu"))
    if stage._gl is not None:
        stage._gl.close()
    stage._gl = None
    with caplog.at_level(logging.WARNING, logger="spacr.qt.make_masks"):
        _use(stage, "cpu", qtbot)
        _use(stage, "auto", qtbot)
    assert stage._active_backend() == "cpu" and stage._gl is None
    notes = [record.getMessage() for record in caplog.records if "True 3D" in record.getMessage()]
    assert len(notes) == 1 and "\n" not in notes[0] and "CPU" in notes[0]
    assert "no OpenGL context" in stage._detail.text()
    assert stage._render_state.backend == "cpu" and stage._render_state.rgb.any()


def test_a_gpu_that_fails_mid_frame_hands_the_frame_to_the_cpu(view, qtbot, monkeypatch):
    stage = view
    stage._gl = None
    stage._gl_tried = True
    monkeypatch.setattr(mm, "volume_render_backends", lambda: ("auto", "opengl", "cpu"))
    _gl = _gl_context()
    stage._gl = _gl

    def failing(_request):
        raise RuntimeError("GL_OUT_OF_MEMORY")

    monkeypatch.setattr(_gl, "render", failing)
    assert stage._submit_render("preview")
    _wait_render(stage, qtbot)
    assert stage._gl is None and stage._render_state.backend == "cpu"
    assert "GL_OUT_OF_MEMORY" in stage._backend_note


def test_auto_does_not_choose_a_software_rasteriser(monkeypatch):
    _gl_context()
    real = layers.GLVolumeRenderer.limits

    def software(self):
        found = real(self)
        found["renderer"] = "llvmpipe (LLVM 15.0.7, 256 bits)"
        return found

    monkeypatch.setattr(layers.GLVolumeRenderer, "limits", software)
    with pytest.raises(RuntimeError, match="software"):
        mm._GLVolumeContext(allow_software=False)
    assert mm._GLVolumeContext(allow_software=True).name().startswith("llvmpipe")


def test_finishes_are_ordered_from_dull_to_shiny_and_checked():
    assert VOLUME_FINISHES == ("matte", "satin", "glossy", "metallic")
    speculars = [SURFACE_FINISHES[name].specular for name in VOLUME_FINISHES]
    assert speculars == sorted(speculars) and speculars[0] == 0.0
    assert SURFACE_FINISHES["glossy"].shininess > SURFACE_FINISHES["satin"].shininess
    assert SURFACE_FINISHES["metallic"].tint > 0.5
    assert all(SURFACE_FINISHES[name].tint == 0.0 for name in ("matte", "satin", "glossy"))
    assert surface_finish(None).name == "matte"
    with pytest.raises(LayerError):
        surface_finish("chrome")
    with pytest.raises(LayerError):
        render_volume([_layers(finish="chrome")[1]], _rays())


def test_finish_lighting_matte_has_no_highlight_and_metal_is_tinted():
    rays = _rays()
    light = SurfaceLight.for_rays(rays)
    red = np.array([[0.9, 0.1, 0.1]])
    mirror = np.asarray([light.half])
    away = np.asarray([light.view]) * 0 + np.asarray([[0.0, 0.0, 0.0]])
    matte = shade_surface(red, mirror, light, surface_finish("matte"))
    glossy = shade_surface(red, mirror, light, surface_finish("glossy"))
    metal = shade_surface(red, mirror, light, surface_finish("metallic"))
    flat = SURFACE_FINISHES["matte"]
    expected = red * (flat.ambient + flat.diffuse * float(np.dot(light.half, light.light)))
    assert np.allclose(matte, expected)
    assert glossy[0, 1] - red[0, 1] * 0.6 > 0.5
    assert metal[0, 0] / metal[0, 1] > glossy[0, 0] / glossy[0, 1]
    assert np.allclose(shade_surface(red, away, light, surface_finish("glossy")),
                       red * SURFACE_FINISHES["glossy"].ambient)
    pictures = {}
    for name in VOLUME_FINISHES:
        masks = _layers(finish=name)[1]
        masks.opacity = 1.0
        pictures[name] = render_volume([masks], rays)[0]
    assert all(not np.allclose(pictures["matte"], pictures[name]) for name in VOLUME_FINISHES[1:])
    assert pictures["glossy"].max() > pictures["matte"].max()


def test_the_finish_list_sets_the_masks_layer_and_is_remembered(view, qtbot):
    stage = view
    row = stage._layer_rows["Masks"]
    assert row.finish is not None and stage._layer_rows["Channel 1"].finish is None
    assert [row.finish.itemData(i) for i in range(row.finish.count())] == list(VOLUME_FINISHES)
    row.finish.setCurrentIndex(row.finish.findData("metallic"))
    _wait_render(stage, qtbot)
    assert stage._layers["Masks"].metadata["finish"] == "metallic"
    finishes = {entry[0]: entry[-1] for entry in stage._render_request("preview").layers}
    assert finishes["Masks"] == "metallic"
    assert preferences.get_volume_view()["finish"] == "metallic"


def test_the_lens_cube_geometry_warp_and_dimming_mask():
    lens = VolumeLens((6.0, 12.0, 14.0), (3.0, 4.0, 5.0), zoom=2.0, dim=0.25)
    corners = lens.corners()
    assert corners.shape == (8, 3)
    assert np.allclose(corners.mean(axis=0), lens.centre)
    assert np.allclose(corners.max(axis=0) - corners.min(axis=0), (6.0, 8.0, 10.0))
    assert len(CUBE_EDGES) == 12 and len({tuple(sorted(e)) for e in CUBE_EDGES}) == 12
    for a, b in CUBE_EDGES:
        assert np.count_nonzero(corners[a] != corners[b]) == 1
    points = np.array([[6.0, 8.0, 0.0], [12.0, 14.0, 0.0], [14.0, 18.0, 0.0]], np.float32)
    where, weight = lens.warp(points)
    assert np.allclose(where[:, 0], (6.0, 12.0, 14.0)) and weight[0] == 1.0
    assert np.allclose(where[:, 1], (7.0, 13.0, 16.0)) and weight[1] == 1.0
    assert np.allclose(where[:, 2], points[:, 2]) and weight[2] == pytest.approx(0.25)
    with pytest.raises(LayerError):
        VolumeLens((0.0, 0.0, 0.0), (0.0, 1.0, 1.0))
    with pytest.raises(LayerError):
        VolumeLens((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), zoom=0.5)
    rays = _rays()
    covered = lens.covers(rays)
    assert covered.any() and not covered.all()
    channel = ImageLayer(np.random.default_rng(1).random(_spheres().shape).astype(np.float32),
                         name="g", contrast_limits=[(0.0, 1.0)], colormaps=["gray"])
    plain, _ = render_volume([channel], rays)
    magnified, _ = render_volume([channel], rays, lens=lens)
    assert np.allclose(magnified[~covered], plain[~covered] * 0.25, atol=1e-5)
    assert not np.allclose(magnified[covered], plain[covered] * 0.25)


def test_the_m_key_turns_the_3d_magnifier_cube_on_and_shift_ctrl_resize_and_zoom(view, screen, qtbot):
    stage = view
    _use(stage, "cpu", qtbot)
    assert not stage._cube.on and stage.cube_corners() is None
    assert screen._toggle_magnifier_key()
    assert stage._cube.on and stage._cube_button.isChecked()
    assert not screen._canvas.magnifier.enabled
    corners = stage.cube_corners()
    side = stage._cube.side
    assert np.allclose(corners.max(axis=0) - corners.min(axis=0), side)
    assert np.allclose(corners.mean(axis=0), [sum(pair) / 2 for pair in stage.session.limits])
    ax = stage._view_axes()
    cube = next(artist for artist in ax.collections if artist.get_gid() == "VolumeMagnifierCube")
    assert cube.get_visible() and len(cube._segments3d) == 12
    _wait_render(stage, qtbot)
    lens = stage._render_request("preview").lens
    assert lens is not None and lens.zoom == stage._cube.zoom and lens.dim == mm.VOLUME_CUBE_DIM
    widget = stage._panel.canvas._canvas
    speed = stage._wheel_speed()

    def wheel(modifiers, up=True):
        centre = QPointF(widget.width() / 2, widget.height() / 2)
        event = QWheelEvent(centre, widget.mapToGlobal(centre), QPoint(0, 0),
                            QPoint(0, 120 if up else -120), Qt.NoButton, modifiers,
                            Qt.NoScrollPhase, False)
        QApplication.sendEvent(widget, event)

    zoom = stage._cube.zoom
    view_zoom = stage._panel.canvas._volume_zoom
    wheel(Qt.ShiftModifier)
    assert stage._cube.side == pytest.approx(side * speed)
    wheel(Qt.ShiftModifier, up=False)
    wheel(Qt.ShiftModifier, up=False)
    assert stage._cube.side == pytest.approx(side / speed)
    wheel(Qt.ControlModifier)
    assert stage._cube.zoom == pytest.approx(min(mm._MAGNIFIER_ZOOM_RANGE[1], zoom * speed))
    assert stage._panel.canvas._volume_zoom == view_zoom
    for _ in range(80):
        stage.step_cube_size(True)
        stage.step_cube_zoom(True)
    assert stage._cube.side == pytest.approx(stage._cube_side_range()[1])
    assert stage._cube.zoom == mm._MAGNIFIER_ZOOM_RANGE[1]
    for _ in range(200):
        stage.step_cube_size(False)
    assert stage._cube.side == pytest.approx(stage._cube_side_range()[0])
    assert screen._toggle_magnifier_key()
    assert not stage._cube.on and not cube.get_visible()
    assert stage._render_request("preview").lens is None


def test_the_cube_follows_the_pointer_on_the_slice_plane_and_dims_outside(view, qtbot):
    stage = view
    _use(stage, "cpu", qtbot)
    canvas = stage._panel.canvas
    canvas._view_angles = (89.0, -90.0, 0.0)
    canvas.render_now()
    canvas._canvas.draw()
    stage.set_slice(6)
    stage.set_magnifier(True)
    sx, sy, sz = stage.session.xyz_spacing
    target = np.array([8.5 * sx, 8.5 * sy, 6.5 * sz])
    origin, matrix, screen0 = stage._projection(stage._view_axes())
    point = screen0 + matrix @ (target - origin)
    assert stage._move_cube(point[0], point[1])
    assert np.allclose(stage._cube.centre, target, atol=1e-6)
    stage._layer_rows["Masks"].eye.setChecked(False)
    _wait_render(stage, qtbot)
    dimmed = stage._render_state.rgb.astype(float)
    stage.set_magnifier(False)
    _wait_render(stage, qtbot)
    plain = stage._render_state.rgb.astype(float)
    assert dimmed.sum() < plain.sum()
    x0, x1, y0, y1 = stage._render_state.rect
    column, row = int(round(point[0] - x0)), int(round(y1 - point[1]))
    assert dimmed[row, column].max() == pytest.approx(plain[row, column].max(), abs=3)


def test_cpu_fallback_meets_its_frame_budget_at_low_scale(view, qtbot):
    stage = view
    _use(stage, "cpu", qtbot)
    stage._scale_slider.setValue(10)
    _wait_render(stage, qtbot)
    times = []
    for _ in range(3):
        request = stage._render_request("interactive")
        started = time.perf_counter()
        assert mm._render_volume_view(request) is not None
        times.append(time.perf_counter() - started)
    assert request.scale.pixel == RENDER_SCALE_MAX_PIXEL
    assert sorted(times)[1] < 0.1


def test_gpu_frames_match_the_cpu_for_every_rendering_finish_and_lens():
    gl = _gl_context()
    rays = _rays()
    lens = VolumeLens((6.0, 12.0, 14.0), (4.0, 4.0, 4.0), zoom=2.0)
    with gl:
        for rendering in VOLUME_RENDERINGS:
            channel, _masks = _layers(rendering=rendering)
            cpu, _ = render_volume([channel], rays)
            gpu, _ = gl.renderer.render([channel], rays)
            assert np.abs(cpu * 255 - gpu).max() <= 3, rendering
        for labels_rendering in ("surface", "translucent"):
            for finish in VOLUME_FINISHES:
                for magnifier in (None, lens):
                    channel, masks = _layers(labels_rendering=labels_rendering, finish=finish)
                    cpu, cpu_picked = render_volume([channel, masks], rays, highlight=2,
                                                    voxel_size=(2.0, 1.0, 1.0), lens=magnifier)
                    gpu, gpu_picked = gl.renderer.render(
                        [channel, masks], rays, highlight=2, voxel_size=(2.0, 1.0, 1.0),
                        lens=magnifier)
                    assert np.abs(cpu * 255 - gpu).max() <= 3, (labels_rendering, finish)
                    assert np.array_equal(cpu_picked, gpu_picked)
        channel, masks = _layers()
        big, _ = gl.renderer.render([channel, masks], rays, size=(60, 72))
        assert big.shape == (60, 72, 3) and big.dtype == np.uint8
    gl.close()


def test_the_gpu_view_uses_the_cpu_camera_and_picks_the_same_objects(view, qtbot, monkeypatch):
    stage = view
    monkeypatch.setattr(mm, "volume_render_backends", lambda: ("auto", "opengl", "cpu"))
    _use(stage, "opengl", qtbot)
    if stage._active_backend() != "opengl":
        pytest.skip("no OpenGL context offscreen")
    stage._scale_slider.setValue(99)
    stage._scale_slider.setValue(100)
    _wait_render(stage, qtbot)
    gpu_request = stage._render_request("preview")
    gpu_state = stage._render_state
    assert gpu_request.backend == "opengl" and gpu_state.backend == "opengl"
    assert "the GPU (OpenGL" in stage._detail.text()
    projection = stage._projection(stage._view_axes())
    _use(stage, "cpu", qtbot)
    cpu_request = stage._render_request("preview")
    cpu_state = stage._render_state
    for gpu_part, cpu_part in zip(projection, stage._projection(stage._view_axes())):
        assert np.array_equal(gpu_part, cpu_part)
    assert gpu_request.rays == cpu_request.rays and gpu_request.rect == cpu_request.rect
    assert gpu_state.rgb.shape == cpu_state.rgb.shape
    assert np.array_equal(gpu_state.picked, cpu_state.picked)
    assert np.abs(gpu_state.rgb.astype(int) - cpu_state.rgb.astype(int)).mean() < 2.0
    x0, x1, y0, y1 = gpu_state.rect
    rows, columns = gpu_state.picked.shape
    row, column = np.argwhere(gpu_state.picked == 2)[0]
    x = x0 + (column + .5) * (x1 - x0) / columns
    y = y1 - (row + .5) * (y1 - y0) / rows
    _use(stage, "opengl", qtbot)
    assert stage._render_state.backend == "opengl"
    assert stage.pick_at(x, y) == 2


def test_gpu_drag_frames_stay_inside_an_interactive_budget(view, qtbot, monkeypatch):
    stage = view
    monkeypatch.setattr(mm, "volume_render_backends", lambda: ("auto", "opengl", "cpu"))
    _use(stage, "opengl", qtbot)
    if stage._active_backend() != "opengl":
        pytest.skip("no OpenGL context offscreen")
    stage._scale_slider.setValue(100)
    _wait_render(stage, qtbot)
    times = []
    for _ in range(5):
        started = time.perf_counter()
        assert stage._submit_render("interactive")
        times.append(time.perf_counter() - started)
    assert stage._render_state.quality == "interactive"
    assert sorted(times)[2] < 0.1
    _settle()
