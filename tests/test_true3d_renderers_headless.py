"""The True 3D renderers, checked where no GPU may be used.

CI hides the GPU (``CUDA_VISIBLE_DEVICES`` empty), so the frame-for-frame
comparisons in ``tests/qt/test_673_true3d_gpu.py`` and
``tests/qt/test_673_true3d_wgpu.py`` skip there. These tests drive the
OpenGL (VisPy gloo) and wgpu renderers against recording stand-ins for
those libraries, so what the renderers decide in Python -- which uniforms
and parameters a camera, lens and finish become, which volumes are uploaded,
kept or freed, how a picture is scaled and how picked IDs are decoded, and
what they refuse -- is checked on any computer. The CPU renderer's lens,
glass and single-ray paths, which those comparisons also reached, are
checked here against the CPU renderer itself.
"""
from __future__ import annotations

import importlib.machinery
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from spacr import layers
from spacr.layers import (LABEL_DENSITY, SURFACE_FINISHES, VOLUME_ATTENUATION, VOLUME_DENSITY,
                          GLVolumeRenderer, ImageLayer, LabelsLayer, LayerError, SurfaceLight,
                          VolumeLens, VolumeRays, WGPUVolumeRenderer, render_volume)


def _labels(shape=(10, 12, 14)):
    z, y, x = np.indices(shape)
    labels = np.zeros(shape, np.uint16)
    labels[(z - 5) ** 2 + (y - 4) ** 2 + (x - 4) ** 2 <= 9] = 1
    labels[(z - 5) ** 2 + (y - 8) ** 2 + (x - 10) ** 2 <= 4] = 2
    return labels


def _rays(shape=(6, 7, 9)):
    return VolumeRays(origin=(-0.5, 0.3, 0.2), across=(0.0, 0.0, 2.0),
                      down=(0.0, 2.0, 0.0), depth=(1.2, 0.0, 0.0), shape=shape)


def _scene(labels_rendering="surface", finish="glossy", rendering="mip"):
    labels = _labels()
    image = (labels.astype(np.float32) * 50 + 10).astype(np.float32)
    channel = ImageLayer(image, name="c", contrast_limits=[(0, 120)], colormaps=["green"],
                         metadata={"rendering": rendering})
    masks = LabelsLayer(labels, name="m", opacity=0.6,
                        metadata={"rendering": labels_rendering, "finish": finish})
    return channel, masks


class _GL:
    """A recording stand-in for ``vispy.gloo`` and its context module."""

    def __init__(self):
        self.current = None
        self.made_current = 0
        self.forgotten = 0
        self.flushed = 0
        self.bound = []
        self.objects = []
        self.draws = []
        self.states = []
        self.pick_codes = None

    def install(self, monkeypatch):
        gl = self
        vispy = types.ModuleType("vispy")
        gloo = types.ModuleType("vispy.gloo")
        context = types.ModuleType("vispy.gloo.context")
        glmod = types.ModuleType("vispy.gloo.gl")

        class FakeCanvas:
            def __init__(self):
                gl.current = self

            def flush(self):
                gl.flushed += 1

        def set_current_canvas(canvas):
            gl.made_current += 1
            gl.current = canvas

        def forget_canvas(canvas):
            gl.forgotten += 1
            if gl.current is canvas:
                gl.current = None

        context.FakeCanvas = FakeCanvas
        context.get_current_canvas = lambda: gl.current
        context.set_current_canvas = set_current_canvas
        context.forget_canvas = forget_canvas
        glmod.GL_RENDERER, glmod.GL_VERSION = "RENDERER", "VERSION"
        glmod.glGetParameter = {"RENDERER": "Fake GL", "VERSION": "4.6 fake",
                                0x8073: 512}.__getitem__

        class _Object:
            def __init__(self, *args, **kwargs):
                self.args, self.kwargs, self.deleted = args, kwargs, False
                gl.objects.append(self)

            def delete(self):
                self.deleted = True

        class Texture3D(_Object):
            pass

        class Texture2D(_Object):
            @property
            def shape(self):
                return self.args[0].shape if self.args else self.kwargs["shape"]

        class RenderBuffer(_Object):
            pass

        class FrameBuffer(_Object):
            @property
            def color_buffer(self):
                return self.kwargs["color"]

            def __enter__(self):
                gl.bound.append(self)
                return self

            def __exit__(self, *exc):
                gl.bound.pop()

            def read(self, mode, alpha=True):
                assert mode == "color" and alpha is False
                colour = self.kwargs["color"]
                rows, columns = (colour.shape if isinstance(colour, Texture2D)
                                 else colour.args[0])[:2]
                if self.kwargs["depth"] is not None:
                    if gl.pick_codes is not None:
                        return gl.pick_codes
                    return np.zeros((rows, columns, 3), np.uint8)
                return np.full((rows, columns, 3), 7 + len(gl.bound), np.uint8)

        class Program(_Object):
            def __init__(self, vertex, fragment):
                super().__init__(vertex, fragment)
                self.fragment, self.uniforms = fragment, {}

            def __setitem__(self, key, value):
                self.uniforms[key] = value

            def __getitem__(self, key):
                return self.uniforms[key]

            def draw(self, mode):
                assert mode == "triangle_strip"
                gl.draws.append((self, dict(self.uniforms), gl.bound[-1]))

        gloo.Texture3D, gloo.Texture2D, gloo.RenderBuffer = Texture3D, Texture2D, RenderBuffer
        gloo.FrameBuffer, gloo.Program = FrameBuffer, Program
        gloo.set_viewport = lambda *args: gl.states.append(("viewport", args))
        gloo.set_state = lambda **kwargs: gl.states.append(("state", kwargs))
        gloo.clear = lambda **kwargs: gl.states.append(("clear", kwargs))
        gloo.context, gloo.gl, vispy.gloo = context, glmod, gloo
        for name, module in (("vispy", vispy), ("vispy.gloo", gloo),
                             ("vispy.gloo.context", context), ("vispy.gloo.gl", glmod)):
            monkeypatch.setitem(sys.modules, name, module)
        self.Texture3D, self.Texture2D, self.FrameBuffer = Texture3D, Texture2D, FrameBuffer
        return self

    def of(self, kind):
        return [obj for obj in self.objects if isinstance(obj, kind)]


@pytest.fixture
def fake_gl(monkeypatch):
    return _GL().install(monkeypatch)


def _kernel(fragment):
    if fragment == layers._GL_COPY:
        return "copy"
    return "labels" if fragment.endswith(layers._GL_LABELS) else "image"


def test_gl_limits_read_the_context_and_release_it(fake_gl):
    renderer = GLVolumeRenderer()
    assert renderer.limits() == {"renderer": "Fake GL", "version": "4.6 fake", "max_3d": 512}
    assert fake_gl.flushed == 1 and fake_gl.forgotten == 1 and fake_gl.current is None
    renderer.limits()
    assert fake_gl.made_current == 1


def test_gl_frame_hands_the_camera_lens_and_finish_to_the_shaders(fake_gl):
    channel, masks = _scene(finish="metallic")
    rays = _rays()
    lens = VolumeLens((5.0, 6.0, 7.0), (2.0, 3.0, 4.0), zoom=3.0, dim=0.25)
    fake_gl.pick_codes = np.zeros((6, 7, 3), np.uint8)
    fake_gl.pick_codes[0, 0] = (70000 & 255, (70000 >> 8) & 255, 70000 >> 16)
    fake_gl.pick_codes[2, 3] = (2, 0, 0)
    renderer = GLVolumeRenderer()
    rgb, picked = renderer.render([channel, masks], rays, highlight=2,
                                  voxel_size=(2.0, 1.0, 1.0), lens=lens)
    assert rgb.shape == (6, 7, 3) and rgb.flags["C_CONTIGUOUS"]
    assert picked.dtype == np.int64 and picked[0, 0] == 70000 and picked[2, 3] == 2
    assert picked.sum() == 70002
    kinds = [(_kernel(program.fragment), uniforms.get("u_mode"), target.kwargs["depth"] is None)
             for program, uniforms, target in fake_gl.draws]
    assert kinds == [("image", 0, True), ("labels", 0, True), ("labels", 2, False)]
    image = fake_gl.draws[0][1]
    assert image["u_origin"] == (-0.5, 0.3, 0.2) and image["u_depth"] == (1.2, 0.0, 0.0)
    assert image["u_shape"] == (10.0, 12.0, 14.0)
    assert image["u_rows"] == 6.0 and image["u_samples"] == 9
    assert image["u_lens_centre"] == (5.0, 6.0, 7.0) and image["u_lens_half"] == (2.0, 3.0, 4.0)
    assert image["u_lens_zoom"] == 3.0 and image["u_lens_dim"] == 0.25
    assert (image["u_low"], image["u_high"]) == (0.0, 120.0)
    assert image["u_length"] == pytest.approx(1.2)
    assert image["u_attenuation"] == VOLUME_ATTENUATION and image["u_density"] == VOLUME_DENSITY
    surface = fake_gl.draws[1][1]
    metal = SURFACE_FINISHES["metallic"]
    assert (surface["u_ambient"], surface["u_specular"], surface["u_tint"]) == (
        metal.ambient, metal.specular, metal.tint)
    assert surface["u_highlight"] == 2.0 and surface["u_opacity"] == pytest.approx(0.6)
    assert surface["u_alpha"] == pytest.approx(min(1.0, LABEL_DENSITY * 1.2))
    light = SurfaceLight.for_rays(rays, (2.0, 1.0, 1.0))
    assert surface["u_size"] == (2.0, 1.0, 1.0)
    assert surface["u_light"] == pytest.approx(tuple(light.light))
    assert surface["u_colours_size"] == (float(layers._GL_COLOUR_WIDTH), 1.0)
    colours = surface["u_colours"].args[0]
    assert colours.shape == (1, layers._GL_COLOUR_WIDTH, 4)
    assert np.allclose(colours[0, 1, :3], layers.label_colors(np.array([1]))[0])
    assert np.all(colours[0, :, 3] == 1.0)
    occupancy = surface["u_occupancy"].args[0]
    assert occupancy.dtype == np.float32 and set(np.unique(occupancy)) == {0.0, 1.0}
    assert surface["u_ids"].kwargs["interpolation"] == "nearest"
    assert image["u_volume"].kwargs["interpolation"] == "linear"
    assert fake_gl.current is None and fake_gl.flushed == 1


def test_gl_without_a_lens_sends_a_lens_that_changes_nothing(fake_gl):
    channel, masks = _scene(labels_rendering="translucent")
    GLVolumeRenderer().render([channel, masks], _rays())
    image, glass = fake_gl.draws[0][1], fake_gl.draws[1][1]
    assert image["u_lens_zoom"] == 0.0 and image["u_lens_dim"] == 1.0
    assert image["u_lens_half"] == (1.0, 1.0, 1.0)
    assert glass["u_mode"] == 1


@pytest.mark.parametrize("rendering", layers.VOLUME_RENDERINGS)
def test_gl_image_modes_follow_the_layer_rendering(fake_gl, rendering):
    channel, _ = _scene(rendering=rendering)
    GLVolumeRenderer().render([channel], _rays())
    assert fake_gl.draws[0][1]["u_mode"] == layers._GL_IMAGE_MODES[rendering]


def test_gl_volumes_are_uploaded_once_and_freed_when_a_frame_stops_drawing_them(fake_gl):
    channel, masks = _scene()
    renderer = GLVolumeRenderer()
    rays = _rays()
    renderer.render([channel, masks], rays)
    uploads = fake_gl.of(fake_gl.Texture3D)
    assert len(uploads) == 3
    renderer.render([channel, masks], rays)
    assert len(fake_gl.of(fake_gl.Texture3D)) == 3
    assert fake_gl.made_current == 1
    luts = [obj for obj in fake_gl.of(fake_gl.Texture2D) if obj.kwargs.get("internalformat")
            == "rgba32f" and obj.args and obj.args[0].shape == (1, 256, 4)]
    assert len(luts) == 1
    renderer.render([channel], rays)
    assert [texture.deleted for texture in uploads] == [False, True, True]
    edited = LabelsLayer(masks.data.copy(), name="m", metadata=dict(masks.metadata))
    renderer.render([channel, edited], rays)
    assert len(fake_gl.of(fake_gl.Texture3D)) == 5


def test_gl_a_new_frame_size_replaces_only_targets_of_that_kind(fake_gl):
    channel, masks = _scene()
    renderer = GLVolumeRenderer()
    renderer.render([channel, masks], _rays())
    first = dict(renderer._targets)
    renderer.render([channel, masks], _rays((4, 5, 9)))
    assert all(target.deleted for target in first.values())
    assert sorted(key[:2] for key in renderer._targets) == [(4, 5), (4, 5)]


def test_gl_a_bigger_picture_is_scaled_up_by_the_copy_program(fake_gl):
    channel, masks = _scene()
    renderer = GLVolumeRenderer()
    rgb, picked = renderer.render([channel, masks], _rays(), size=(12, 15))
    assert rgb.shape == (12, 15, 3) and picked.shape == (6, 7)
    copy, uniforms, target = fake_gl.draws[-1]
    assert _kernel(copy.fragment) == "copy"
    frame = renderer._targets[(6, 7, "frame")]
    assert uniforms["u_image"] is frame.color_buffer
    assert target is renderer._targets[(12, 15, "out")]
    assert ("viewport", (0, 0, 15, 12)) in fake_gl.states
    same, _ = renderer.render([channel, masks], _rays(), size=(6, 7))
    assert same.shape == (6, 7, 3) and (6, 7, "out") not in renderer._targets


def test_gl_refuses_mixed_shapes_flat_layers_and_huge_ids(fake_gl):
    channel, masks = _scene()
    other = LabelsLayer(np.zeros((3, 4, 5), np.uint16), name="small")
    renderer = GLVolumeRenderer()
    with pytest.raises(LayerError, match="share one shape"):
        renderer.render([channel, other], _rays())
    flat = ImageLayer(np.ones((5, 6), np.float32), name="flat")
    with pytest.raises(LayerError, match="3-D layers only"):
        renderer.render([flat], _rays())
    huge = LabelsLayer(np.full((3, 4, 5), 1 << 24, np.uint32), name="huge")
    with pytest.raises(LayerError, match="2\\*\\*24"):
        renderer.render([huge], _rays())
    assert fake_gl.current is None


def test_gl_hidden_layers_draw_nothing_and_close_frees_everything(fake_gl):
    channel, masks = _scene()
    masks.visible = False
    renderer = GLVolumeRenderer()
    renderer.close()
    assert fake_gl.objects == []
    renderer.render([channel, masks], _rays())
    assert [_kernel(draw[0].fragment) for draw in fake_gl.draws] == ["image"]
    made = [obj for obj in fake_gl.objects
            if isinstance(obj, (fake_gl.Texture3D, fake_gl.FrameBuffer))
            or type(obj).__name__ == "Program" or obj in renderer._luts.values()]
    renderer.close()
    assert len(made) == 1 + 2 + 1 + 1 and all(obj.deleted for obj in made)
    assert renderer._textures == {} and renderer._programs == {}
    assert renderer._targets == {} and renderer._luts == {}


class _WGPU:
    """A recording stand-in for the ``wgpu`` package and one device."""

    def __init__(self, limits=None, info=None, destroyable=True):
        self.dispatches = []
        self.writes = []
        self.buffer_writes = []
        self.created = []
        self.submitted = 0
        self.copies = []
        self.mapping = []
        self.reader = None
        self.asked = None
        state = self
        module = types.ModuleType("wgpu")
        module.__spec__ = importlib.machinery.ModuleSpec("wgpu", None)
        module.TextureFormat = SimpleNamespace(r32uint="r32uint")
        module.TextureUsage = SimpleNamespace(TEXTURE_BINDING=4, COPY_DST=2)
        module.BufferUsage = SimpleNamespace(MAP_READ=1, STORAGE=128, COPY_SRC=4, COPY_DST=8, UNIFORM=64)

        class GPUBuffer:
            def __init__(self, size, usage, data=None):
                self.size, self.usage, self.data, self.destroyed = size, usage, data, False
                self.copies = {}
                self.storage = bytearray(size)
                self.mapped = None
                self.submission = 0
                state.created.append(self)

            def map_sync(self, mode, *, offset=0, size=None):
                size = self.size - offset if size is None else size
                assert not self.destroyed and self.mapped is None
                assert self.usage == module.BufferUsage.MAP_READ | module.BufferUsage.COPY_DST
                assert mode == "READ" and 0 <= offset < offset + size <= self.size
                assert self.submission == state.submitted and self.submission > 0
                self.mapped = (offset, size)
                state.mapping.append(("map", self, mode, offset, size))

            def read_mapped(self, *, offset=0, size=None, copy=True):
                size = self.size - offset if size is None else size
                assert self.mapped == (offset, size) and copy is True
                assert not self.destroyed
                state.mapping.append(("read", self, offset, size, copy))
                return bytes(self.storage[offset:offset + size])

            def unmap(self):
                assert self.mapped is not None and not self.destroyed
                self.mapped = None
                state.mapping.append(("unmap", self))

            def destroy(self):
                assert self.mapped is None
                self.destroyed = True

        class Texture:
            def __init__(self, **kwargs):
                self.kwargs, self.destroyed = kwargs, False
                state.created.append(self)

            def create_view(self):
                return SimpleNamespace(texture=self)

            def destroy(self):
                self.destroyed = True

        class Pipeline:
            def __init__(self, code):
                self.name = next(name for name, kernel in layers._WGSL_KERNELS.items()
                                 if code == layers._WGSL_COMMON + kernel)

            def get_bind_group_layout(self, index):
                return (self.name, index)

        class Pass:
            def set_pipeline(self, pipeline):
                self.pipeline = pipeline

            def set_bind_group(self, index, group):
                self.group = group

            def dispatch_workgroups(self, x, y, z):
                state.dispatches.append((self.pipeline.name, self.group, (x, y, z)))

            def end(self):
                self.ended = True

        class Queue:
            def write_texture(self, destination, data, layout, size):
                state.writes.append((destination, np.array(data), layout, size))

            def write_buffer(self, buffer, buffer_offset, data):
                values = np.array(data, copy=True, order="C")
                payload = values.tobytes()
                assert not buffer.destroyed and buffer.mapped is None
                assert buffer.usage & module.BufferUsage.COPY_DST
                assert buffer_offset >= 0 and buffer_offset % 4 == 0 and len(payload) % 4 == 0
                assert buffer_offset + len(payload) <= buffer.size
                buffer.storage[buffer_offset:buffer_offset + len(payload)] = payload
                buffer.data = values
                state.buffer_writes.append((buffer, buffer_offset, values.copy()))

            def submit(self, buffers):
                state.submitted += len(buffers)
                for commands in buffers:
                    for source, source_offset, target, target_offset, size in commands:
                        assert not source.destroyed and not target.destroyed
                        assert target.mapped is None
                        payload = self._contents(source)[source_offset:source_offset + size]
                        assert len(payload) == size
                        target.storage[target_offset:target_offset + size] = payload
                        target.submission = state.submitted

            def _contents(self, buffer):
                if state.reader is not None:
                    found = state.reader(buffer)
                    if found is not None:
                        return found
                return bytes(buffer.size)

        class Device:
            queue = Queue()

            def create_shader_module(self, code):
                return code

            def create_compute_pipeline(self, layout, compute):
                assert layout == "auto" and compute["entry_point"] == "main"
                return Pipeline(compute["module"])

            def create_texture(self, **kwargs):
                return Texture(**kwargs)

            def create_buffer_with_data(self, data, usage):
                return GPUBuffer(np.asarray(data).nbytes, usage, np.array(data))

            def create_buffer(self, size, usage):
                return GPUBuffer(size, usage)

            def create_bind_group(self, layout, entries):
                return {"layout": layout, "entries": entries}

            def create_command_encoder(self):
                copies = []

                def copy_buffer_to_buffer(source, source_offset, target, target_offset, size):
                    assert source.usage & module.BufferUsage.COPY_SRC
                    assert target.usage & module.BufferUsage.COPY_DST
                    assert source_offset + size <= source.size
                    assert target_offset + size <= target.size
                    target.copies[target_offset] = (source, source_offset, size)
                    state.copies.append((source, source_offset, target, target_offset, size))
                    copies.append((source, source_offset, target, target_offset, size))

                return SimpleNamespace(begin_compute_pass=Pass, finish=lambda: copies,
                                       copy_buffer_to_buffer=copy_buffer_to_buffer)

        if destroyable:
            Device.destroy = lambda self: setattr(state, "device_destroyed", True)
        self.device_destroyed = False

        def request_device_sync(required_limits):
            state.asked = required_limits
            return Device()

        self.adapter = SimpleNamespace(
            info=info or {"backend_type": "Vulkan", "adapter_type": "DiscreteGPU",
                          "device": "Fake RTX", "description": "fake driver 1.0"},
            limits=limits if limits is not None else {
                "max_texture_dimension_3d": 256, "max-buffer-size": 1 << 30,
                "max_bind_groups": 4},
            request_device_sync=request_device_sync)
        module.GPUBuffer = GPUBuffer
        self.module, self.GPUBuffer, self.Texture = module, GPUBuffer, Texture

    def install(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "wgpu", self.module)
        return self


def test_wgpu_asks_for_the_adapters_own_largest_limits_and_reports_them(monkeypatch):
    fake = _WGPU().install(monkeypatch)
    renderer = WGPUVolumeRenderer(fake.adapter)
    assert renderer.backend == "vulkan"
    assert fake.asked == {"max_texture_dimension_3d": 256, "max-buffer-size": 1 << 30}
    assert renderer.limits() == {"renderer": "Fake RTX", "version": "fake driver 1.0",
                                 "max_3d": 256, "adapter_type": "DiscreteGPU"}
    assert renderer._limit("max-storage-buffer-binding-size", 99) == 99


def test_wgpu_an_adapter_with_only_the_async_api_is_still_asked(monkeypatch):
    fake = _WGPU(limits={}).install(monkeypatch)
    adapter = SimpleNamespace(info=fake.adapter.info,
                              request_device=fake.adapter.request_device_sync)
    renderer = WGPUVolumeRenderer(adapter)
    assert fake.asked == {} and renderer.limits()["max_3d"] == 2048


def test_wgpu_params_pack_camera_shape_mode_and_lens():
    rays = _rays()
    plain = WGPUVolumeRenderer._params(rays, (10, 12, 14), None, mode=3)
    assert plain.shape == (17, 4) and plain.dtype == np.float32
    assert tuple(plain[0, :3]) == pytest.approx((-0.5, 0.3, 0.2))
    assert tuple(plain[3, :3]) == pytest.approx((1.2, 0.0, 0.0))
    assert tuple(plain[4, :3]) == (10.0, 12.0, 14.0)
    assert tuple(plain[5]) == (6.0, 7.0, 9.0, 3.0)
    assert not plain[6:].any()
    lens = VolumeLens((1.0, 2.0, 3.0), (0.5, 1.5, 2.5), zoom=4.0, dim=0.5)
    magnified = WGPUVolumeRenderer._params(rays, (5, 6), lens)
    assert tuple(magnified[4, :3]) == (1.0, 1.0, 1.0)
    assert tuple(magnified[6, :3]) == (1.0, 2.0, 3.0)
    assert tuple(magnified[7, :3]) == (0.5, 1.5, 2.5)
    assert tuple(magnified[8, :2]) == (4.0, 0.5)


def test_wgpu_frame_runs_clear_image_labels_and_pack_and_decodes_the_picture(monkeypatch):
    fake = _WGPU().install(monkeypatch)
    renderer = WGPUVolumeRenderer(fake.adapter)
    channel, masks = _scene(finish="satin")
    rays = _rays()
    lens = VolumeLens((5.0, 6.0, 7.0), (2.0, 3.0, 4.0), zoom=2.0, dim=0.3)

    def reader(buffer):
        if buffer is renderer._buffers["picture"][1]:
            picture = np.zeros((12, 15, 4), np.uint8)
            picture[..., 0] = np.arange(15)
            picture[..., 3] = 255
            return picture.tobytes()
        if buffer is renderer._buffers["pick"][1]:
            codes = np.zeros((6, 7, 2), np.uint32)
            codes[1, 2] = (2, 5)
            codes[5, 6] = (70000, 1)
            return codes.tobytes()
        return None

    fake.reader = reader
    rgb, picked = renderer.render([channel, masks], rays, highlight=1,
                                  voxel_size=(2.0, 1.0, 1.0), lens=lens, size=(12, 15))
    assert [name for name, _group, _groups in fake.dispatches] == [
        "clear", "image", "labels", "pack"]
    assert [groups for *_rest, groups in fake.dispatches] == [
        (1, 1, 1), (1, 1, 1), (1, 1, 1), (2, 2, 1)]
    assert fake.submitted == 1
    staging = renderer._buffers["readback"][1]
    mapped_bytes = 12 * 15 * 4 + 6 * 7 * 8
    assert staging.usage == fake.module.BufferUsage.MAP_READ | fake.module.BufferUsage.COPY_DST
    assert not staging.usage & fake.module.BufferUsage.STORAGE
    assert fake.mapping == [("map", staging, "READ", 0, mapped_bytes),
                            ("read", staging, 0, mapped_bytes, True),
                            ("unmap", staging)]
    assert staging.mapped is None and not staging.destroyed
    assert [(copy[1], copy[3], copy[4]) for copy in fake.copies] == [
        (0, 0, 12 * 15 * 4), (0, 12 * 15 * 4, 6 * 7 * 8)]
    assert rgb.shape == (12, 15, 3) and np.array_equal(rgb[3, :, 0], np.arange(15))
    assert picked.dtype == np.int64 and picked[1, 2] == 2 and picked[5, 6] == 70000
    assert picked.sum() == 70002
    image_entries = fake.dispatches[1][1]["entries"]
    params = image_entries[0]["resource"]["buffer"].data
    assert params[5, 3] == layers._GL_IMAGE_MODES["mip"]
    assert tuple(params[9]) == pytest.approx((0.0, 120.0, 1.0, 1.2))
    assert tuple(params[10, :2]) == pytest.approx((VOLUME_ATTENUATION, VOLUME_DENSITY))
    assert tuple(params[8, :2]) == pytest.approx((2.0, 0.3))
    assert isinstance(image_entries[2]["resource"], SimpleNamespace)
    assert image_entries[3]["resource"]["size"] == 256 * 16
    labels_params = fake.dispatches[2][1]["entries"][0]["resource"]["buffer"].data
    satin = SURFACE_FINISHES["satin"]
    assert tuple(labels_params[15]) == pytest.approx(
        (satin.ambient, satin.diffuse, satin.specular, satin.shininess))
    assert labels_params[10, 3] == 1.0 and labels_params[9, 2] == pytest.approx(0.6)
    assert tuple(labels_params[11, :3]) == (2.0, 1.0, 1.0)
    pack_params = fake.dispatches[3][1]["entries"][0]["resource"]["buffer"].data
    assert tuple(pack_params[16, 1:3]) == (12.0, 15.0)
    ids_write = [write for write in fake.writes if write[1].shape == (10, 12, 14)
                 and np.array_equal(write[1], masks.data)]
    assert ids_write and ids_write[0][3] == (14, 12, 10)
    assert ids_write[0][2] == {"offset": 0, "bytes_per_row": 56, "rows_per_image": 12}
    bits = np.ascontiguousarray(channel.channel_data(0), dtype=np.float32).view(np.uint32)
    assert any(write[1].dtype == np.uint32 and np.array_equal(write[1], bits)
               for write in fake.writes)


def test_wgpu_translucent_masks_and_the_rays_own_picture_size(monkeypatch):
    fake = _WGPU().install(monkeypatch)
    renderer = WGPUVolumeRenderer(fake.adapter)
    channel, masks = _scene(labels_rendering="translucent")
    rgb, picked = renderer.render([channel, masks], _rays())
    assert rgb.shape == (6, 7, 3) and picked.shape == (6, 7) and not picked.any()
    labels_params = fake.dispatches[2][1]["entries"][0]["resource"]["buffer"].data
    assert labels_params[5, 3] == 1.0 and not labels_params[6:9].any()


def test_wgpu_volumes_and_buffers_are_kept_resized_and_freed(monkeypatch):
    fake = _WGPU().install(monkeypatch)
    renderer = WGPUVolumeRenderer(fake.adapter)
    channel, masks = _scene()
    rays = _rays()
    renderer.render([channel, masks], rays)
    textures = [obj for obj in fake.created if isinstance(obj, fake.Texture)]
    frame = renderer._buffers["frame"][1]
    staging = renderer._buffers["readback"][1]
    assert staging.size == 6 * 7 * (4 + 8)
    assert len(textures) == 2
    uniforms = tuple(slot["uniform"] for slot in renderer._dispatch_slots)
    groups = tuple(slot["group"] for slot in renderer._dispatch_slots)
    assert len(uniforms) == 4
    assert all(buffer.usage == fake.module.BufferUsage.UNIFORM | fake.module.BufferUsage.COPY_DST
               for buffer in uniforms)
    assert [buffer for buffer, offset, _values in fake.buffer_writes] == list(uniforms)
    assert all(offset == 0 for _buffer, offset, _values in fake.buffer_writes)
    assert all(bytes(buffer.storage) == values.tobytes()
               for buffer, _offset, values in fake.buffer_writes)
    renderer.render([channel, masks], rays)
    assert all(slot["uniform"] is uniform and slot["group"] is group
               for slot, uniform, group in zip(renderer._dispatch_slots, uniforms, groups))
    assert [buffer for buffer, _offset, _values in fake.buffer_writes] == list(uniforms) * 2
    assert len([obj for obj in fake.created if isinstance(obj, fake.Texture)]) == 2
    assert renderer._buffers["frame"][1] is frame and not frame.destroyed
    colours = renderer._textures[(layers._array_key(masks.data), "colours")][1][0]
    staging.storage[:] = b"\xa5" * staging.size
    renderer.render([channel], _rays((4, 5, 9)))
    mapped_bytes = 4 * 5 * (4 + 8)
    assert renderer._buffers["readback"][1] is staging
    assert staging.storage[mapped_bytes:] == b"\xa5" * (staging.size - mapped_bytes)
    assert fake.mapping[-3:] == [("map", staging, "READ", 0, mapped_bytes),
                                 ("read", staging, 0, mapped_bytes, True),
                                 ("unmap", staging)]
    assert fake.submitted == 3 and staging.mapped is None
    assert renderer._buffers["frame"] == (6 * 7 * 16, frame) and not frame.destroyed
    assert [texture.destroyed for texture in textures] == [False, True]
    assert colours.destroyed
    renderer.render([channel], _rays((8, 9, 9)))
    assert frame.destroyed and renderer._buffers["frame"][0] == 8 * 9 * 16
    grown_staging = renderer._buffers["readback"][1]
    mapped_bytes = 8 * 9 * (4 + 8)
    assert staging.destroyed and staging.mapped is None
    assert grown_staging is not staging and grown_staging.size == mapped_bytes
    assert fake.mapping[-3:] == [("map", grown_staging, "READ", 0, mapped_bytes),
                                 ("read", grown_staging, 0, mapped_bytes, True),
                                 ("unmap", grown_staging)]
    assert fake.submitted == 4 and grown_staging.mapped is None
    lut = next(iter(renderer._luts.values()))
    held = [buffer for _size, buffer in renderer._buffers.values()]
    renderer.close()
    assert all(texture.destroyed for texture in textures) and lut.destroyed
    assert held and all(buffer.destroyed for buffer in held)
    assert renderer._textures == {} and renderer._buffers == {} and renderer._pipelines == {}
    assert fake.device_destroyed
    assert staging.destroyed and staging.mapped is None
    assert grown_staging.destroyed and grown_staging.mapped is None


def test_wgpu_close_without_a_device_destroy(monkeypatch):
    fake = _WGPU(destroyable=False).install(monkeypatch)
    renderer = WGPUVolumeRenderer(fake.adapter)
    renderer.render([_scene()[0]], _rays())
    buffers = [buffer for _size, buffer in renderer._buffers.values()]
    renderer.close()
    assert buffers and all(buffer.destroyed for buffer in buffers)
    assert not fake.device_destroyed


def test_wgpu_refuses_what_the_gpu_cannot_hold(monkeypatch):
    fake = _WGPU(limits={"max-texture-dimension-3d": 8,
                         "max_storage_buffer_binding_size": 64}).install(monkeypatch)
    renderer = WGPUVolumeRenderer(fake.adapter)
    channel, masks = _scene()
    with pytest.raises(LayerError, match="share one shape"):
        renderer.render([channel, LabelsLayer(np.zeros((2, 3, 4), np.uint16))], _rays())
    with pytest.raises(LayerError, match="3-D layers only"):
        renderer.render([ImageLayer(np.ones((4, 5), np.float32))], _rays())
    with pytest.raises(LayerError, match="exceeds this GPU's 8"):
        renderer.render([masks], _rays())
    renderer.close()
    renderer = WGPUVolumeRenderer(fake.adapter)
    negative = LabelsLayer(np.full((2, 3, 4), -1, np.int32), name="neg")
    with pytest.raises(LayerError, match="between 0 and 2\\*\\*32"):
        renderer.render([negative], _rays())
    renderer.close()
    renderer = WGPUVolumeRenderer(fake.adapter)
    many = np.zeros((2, 3, 4), np.uint16)
    many[0, 0, 0] = 4
    with pytest.raises(LayerError, match="too many for this GPU's colour table"):
        renderer.render([LabelsLayer(many, name="many")], _rays())
    renderer.close()
    renderer = WGPUVolumeRenderer(fake.adapter)
    few = np.zeros((2, 3, 4), np.uint16)
    few[0, 0, :3] = (1, 2, 3)
    rgb, _ = renderer.render([LabelsLayer(few, name="few")], _rays())
    assert rgb.shape == (6, 7, 3)


def test_wgpu_enumerate_without_wgpu_with_either_api_and_when_it_breaks(monkeypatch):
    import importlib.util

    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a: None if name == "wgpu" else real_find_spec(name, *a))
    assert layers._wgpu_enumerate() == []
    monkeypatch.setattr(importlib.util, "find_spec", real_find_spec)
    fake = _WGPU().install(monkeypatch)
    fake.module.gpu = SimpleNamespace(enumerate_adapters_sync=lambda: iter(["a", "b"]))
    assert layers._wgpu_enumerate() == ["a", "b"]
    fake.module.gpu = SimpleNamespace(enumerate_adapters=lambda: ("c",))
    assert layers._wgpu_enumerate() == ["c"]

    def broken():
        raise RuntimeError("no driver")

    fake.module.gpu = SimpleNamespace(enumerate_adapters_sync=broken)
    assert layers._wgpu_enumerate() == []


def test_wgpu_adapter_refuses_a_backend_wgpu_does_not_offer():
    with pytest.raises(RuntimeError, match="'opengl' is not a wgpu backend"):
        layers.wgpu_adapter("opengl")


def test_volume_steps_reduce_until_the_budget_holds():
    assert layers.volume_steps((1, 1, 1), 1) == (1, 1, 1)
    assert layers.volume_steps((3, 1, 1), 1) == (3, 1, 1)
    assert layers.volume_steps((40, 1000, 1000), 40 * 250 * 250) == (1, 4, 4)
    assert layers.volume_steps((5, 5), None) == (1, 1)


def test_one_ray_draws_and_picks_what_its_pixel_of_the_frame_shows():
    channel, masks = _scene(labels_rendering="translucent")
    rays = _rays()
    full, picked = render_volume([channel, masks], rays)
    for row, column in ((2, 1), (4, 5), (0, 0)):
        one = rays.ray(row, column)
        assert one.shape == (1, 1, rays.shape[2]) and one.depth == rays.depth
        rgb, pick = render_volume([channel, masks], one)
        assert np.allclose(rgb[0, 0], full[row, column], atol=1e-5)
        assert pick[0, 0] == picked[row, column]


def test_a_lens_dims_what_lies_outside_it_and_keeps_what_it_covers():
    image = np.full((10, 12, 14), 100.0, np.float32)
    channel = ImageLayer(image, name="c", contrast_limits=[(0, 100)], colormaps=["gray"])
    rays = _rays()
    lens = VolumeLens((5.0, 6.0, 7.0), (2.0, 2.0, 2.0), zoom=2.0, dim=0.4)
    plain, _ = render_volume([channel], rays)
    seen, _ = render_volume([channel], rays, lens=lens)
    covered = lens.covers(rays)
    assert covered.any() and (~covered).any()
    assert np.allclose(seen[covered], plain[covered], atol=1e-5)
    assert np.allclose(seen[~covered], 0.4 * plain[~covered], atol=1e-5)


def test_a_lens_dims_glass_masks_outside_it():
    _, masks = _scene(labels_rendering="translucent")
    rays = _rays()
    lens = VolumeLens((5.0, 4.0, 4.0), (1.0, 1.0, 1.0), zoom=1.0, dim=0.5)
    plain, picked = render_volume([masks], rays)
    seen, seen_picked = render_volume([masks], rays, lens=lens)
    assert np.array_equal(picked, seen_picked)
    outside = (picked != 0) & ~lens.covers(rays)
    assert outside.any()
    assert np.all(seen[outside].sum(axis=-1) < plain[outside].sum(axis=-1))


def test_a_highlighted_glass_object_is_drawn_lighter_and_no_other():
    _, masks = _scene(labels_rendering="translucent")
    rays = _rays()
    plain, picked = render_volume([masks], rays)
    lit, _ = render_volume([masks], rays, highlight=1)
    chosen = picked == 1
    assert chosen.any() and (picked == 2).any()
    assert np.all(lit[chosen].sum(axis=-1) > plain[chosen].sum(axis=-1))
    assert np.allclose(lit[~chosen], plain[~chosen])


def test_rays_flat_along_an_axis_miss_a_lens_they_pass_beside():
    rays = VolumeRays(origin=(0.0, 0.0, 0.0), across=(0.0, 0.0, 1.0), down=(0.0, 1.0, 0.0),
                      depth=(1.0, 0.0, 0.0), shape=(6, 6, 8))
    lens = VolumeLens((3.0, 2.0, 2.0), (1.0, 1.0, 1.0))
    covered = lens.covers(rays)
    assert np.array_equal(np.argwhere(covered).min(axis=0), (1, 1))
    assert np.array_equal(np.argwhere(covered).max(axis=0), (3, 3))
    assert covered.sum() == 9


def test_lighting_refuses_a_voxel_size_that_does_not_fit_the_rays():
    with pytest.raises(LayerError, match="3 positive lengths"):
        SurfaceLight.for_rays(_rays(), (1.0, 2.0))
    with pytest.raises(LayerError, match="3 positive lengths"):
        SurfaceLight.for_rays(_rays(), (1.0, 0.0, 1.0))


def test_the_cpu_renderer_refuses_layers_of_different_shapes():
    channel, _ = _scene()
    other = LabelsLayer(np.zeros((3, 4, 5), np.uint16))
    with pytest.raises(LayerError, match="share one shape"):
        render_volume([channel, other], _rays())


def test_gl_two_views_of_one_cached_volume_share_one_upload(fake_gl):
    cache = np.zeros((2,) + _labels().shape, np.float32)
    cache[0] = _labels() * 50.0
    first = cache[0]
    second = cache[0]
    assert first is not second and layers._array_owner(second) is cache
    assert layers._array_key(first) == layers._array_key(second)
    assert layers._array_key(cache[1]) != layers._array_key(first)
    renderer = GLVolumeRenderer()
    for view in (first, second):
        renderer.render([ImageLayer(view, name="c", contrast_limits=[(0, 100)])], _rays())
    assert len(fake_gl.of(fake_gl.Texture3D)) == 1
