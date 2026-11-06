"""CPU contracts for renderer resources and commands; no graphics backend runs."""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from spacr import layers as L


class Resource:
    def __init__(self, kind, data=None, **options):
        self.kind = kind
        self.data = None if data is None else np.array(data, copy=True)
        self.options = options
        self.shape = tuple(options.get('shape', () if data is None else self.data.shape))
        self.deleted = 0

    def delete(self):
        self.deleted += 1


class Program(dict):
    def __init__(self, driver, vertex, fragment):
        super().__init__()
        self.driver, self.vertex, self.fragment = driver, vertex, fragment
        self.deleted = 0

    def draw(self, primitive):
        if self.driver.fail_draw:
            raise RuntimeError('recorded draw failure')
        self.driver.draws.append((primitive, dict(self)))

    def delete(self):
        self.deleted += 1


@pytest.fixture
def gl(monkeypatch):
    """Replace the complete VisPy import boundary before any driver is asked."""
    driver = types.SimpleNamespace(resources=[], programs=[], draws=[], states=[],
                                   current=None, flushes=0, forgotten=[], fail_draw=False)
    gloo = types.ModuleType('vispy.gloo')
    context = types.ModuleType('vispy.gloo.context')
    native = types.ModuleType('vispy.gloo.gl')
    native.GL_RENDERER, native.GL_VERSION = 1, 2
    native.glGetParameter = lambda key: {1: 'recorded device', 2: 'recorded GL', 0x8073: 128}[key]
    gloo.gl = native

    def resource(kind, data=None, **options):
        obj = Resource(kind, data, **options)
        driver.resources.append(obj)
        return obj

    gloo.Texture3D = lambda data, **kw: resource('volume', data, **kw)
    gloo.Texture2D = lambda data=None, **kw: resource('texture', data, **kw)
    gloo.RenderBuffer = lambda shape, **kw: resource('renderbuffer', shape=shape, **kw)

    class Frame:
        def __init__(self, color, depth):
            self.color_buffer, self.depth_buffer = color, depth
            self.deleted = 0
            driver.resources.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, kind, alpha):
            assert kind == 'color' and alpha is False
            rows, columns = self.color_buffer.shape[:2]
            pixel = [2, 1, 1] if self.depth_buffer is not None else [7, 11, 19]
            return np.tile(np.array(pixel, np.uint8), (rows, columns, 1))

        def delete(self):
            self.deleted += 1

    def program(vertex, fragment):
        obj = Program(driver, vertex, fragment)
        driver.programs.append(obj)
        return obj

    class Canvas:
        def __init__(self):
            driver.current = self

        def flush(self):
            driver.flushes += 1

    gloo.FrameBuffer, gloo.Program = Frame, program
    gloo.set_viewport = lambda *args: driver.states.append(('viewport', args))
    gloo.set_state = lambda **kw: driver.states.append(('state', kw))
    gloo.clear = lambda **kw: driver.states.append(('clear', kw))
    context.FakeCanvas = Canvas
    context.get_current_canvas = lambda: driver.current
    context.set_current_canvas = lambda canvas: setattr(driver, 'current', canvas)
    context.forget_canvas = lambda canvas: driver.forgotten.append(canvas)
    vispy = types.ModuleType('vispy')
    vispy.gloo = gloo
    for name, module in [('vispy', vispy), ('vispy.gloo', gloo),
                         ('vispy.gloo.context', context), ('vispy.gloo.gl', native)]:
        monkeypatch.setitem(sys.modules, name, module)
    driver.api, driver.native = gloo, native
    return driver


def rays():
    return L.VolumeRays((0., 0., 0.), (0., 0., 1.), (0., 1., 0.),
                        (1., 0., 0.), (2, 3, 4))


def scene(label_mode='surface'):
    image = L.ImageLayer(np.arange(24, dtype=np.float32).reshape(4, 2, 3),
                         contrast_limits=[(2, 20)], colormaps=['red'], opacity=.4)
    labels = L.LabelsLayer(np.full((4, 2, 3), 3, np.uint32), opacity=.7,
                          metadata={'rendering': label_mode, 'finish': 'metallic'})
    return image, labels


def test_gl_context_queue_is_reused_and_released_even_when_limits_fail(gl, monkeypatch):
    renderer = L.GLVolumeRenderer()
    renderer.close()
    assert gl.resources == [] and gl.current is None
    assert renderer.limits() == {'renderer': 'recorded device', 'version': 'recorded GL', 'max_3d': 128}
    first = renderer._canvas
    assert gl.flushes == 1 and gl.forgotten == [first]
    assert renderer._gloo() is gl.api and renderer._canvas is first
    gl.current = object()
    renderer._gloo()
    assert gl.current is first
    def failed(_key):
        raise RuntimeError('lost context')
    monkeypatch.setattr(gl.native, 'glGetParameter', failed)
    with pytest.raises(RuntimeError, match='lost context'):
        renderer.limits()
    assert gl.flushes == 2 and gl.forgotten == [first, first]


def test_gl_uploads_preserve_values_and_release_only_unreferenced_volumes(gl):
    renderer = L.GLVolumeRenderer()
    source = np.arange(48, dtype=np.int16).reshape(4, 3, 4)[:, :, ::2]
    uploaded = {}
    for kind in ('image', 'ids', 'occupancy'):
        texture = renderer._texture(source, kind)
        expected = (source != 0).astype(np.float32) if kind == 'occupancy' else source.astype(np.float32)
        np.testing.assert_array_equal(texture.data, expected)
        assert texture.data.flags.c_contiguous
        assert texture.options['interpolation'] == ('nearest' if kind == 'ids' else 'linear')
        assert texture.options['internalformat'] == 'r32f'
        assert renderer._texture(source, kind) is texture
        uploaded[kind] = texture
    renderer._forget_unused()
    assert all(t.deleted == 0 for t in uploaded.values())
    assert renderer._texture(source, 'image') is uploaded['image']
    renderer._forget_unused()
    assert uploaded['image'].deleted == 0
    assert uploaded['ids'].deleted == uploaded['occupancy'].deleted == 1
    renderer.close()
    assert uploaded['image'].deleted == 1 and not renderer._textures


def test_gl_frame_sizes_colours_and_luts_are_cached_without_stale_resources(gl):
    renderer = L.GLVolumeRenderer()
    frame = renderer._target(2, 3, 'frame')
    pick = renderer._target(2, 3, 'pick')
    out = renderer._target(4, 6, 'out')
    assert frame.depth_buffer is out.depth_buffer is None
    assert pick.depth_buffer.options['format'] == 'depth'
    assert renderer._target(2, 3, 'frame') is frame
    larger = renderer._target(4, 6, 'frame')
    assert larger is not frame and frame.deleted == 1 and pick.deleted == 0
    cmap = L.colormap('red')
    lut = renderer._lut(cmap)
    assert renderer._lut(cmap) is lut and lut.data.shape == (1, 256, 4)
    np.testing.assert_array_equal(lut.data[0, [0, -1]], [[0, 0, 0, 1], [1, 0, 0, 1]])
    ids = np.array([0, 3, 1025], np.uint32)
    colours = renderer._colours(ids)
    np.testing.assert_allclose(colours.data.reshape(-1, 4)[ids, :3], L.label_colors(ids))
    assert renderer._colours(ids) is colours
    empty = renderer._colours(np.array([], np.uint32))
    assert empty.shape == (1, L._GL_COLOUR_WIDTH, 4)
    with pytest.raises(L.LayerError, match=r'2\*\*24'):
        renderer._colours(np.array([1 << 24], np.uint32))
    renderer.close()
    assert pick.deleted == out.deleted == larger.deleted == lut.deleted == colours.deleted == 1
    assert not renderer._targets and not renderer._luts


@pytest.mark.parametrize('label_mode,size', [('surface', None), ('translucent', (2, 3)),
                                            ('surface', (4, 6))])
def test_gl_command_order_uniforms_and_readback_are_independent_of_a_driver(gl, label_mode, size):
    renderer = L.GLVolumeRenderer()
    image, labels = scene(label_mode)
    hidden = L.ImageLayer(np.zeros((1, 1, 1)), visible=False)
    lens = L.VolumeLens((1., 1., 1.), (.5, .5, .5), zoom=3, dim=.2)
    rgb, picked = renderer.render([hidden, image, labels], rays(), highlight=3,
                                  voxel_size=(2, 1, 1), lens=lens, size=size)
    assert rgb.shape == (*((2, 3) if size is None else size), 3)
    assert rgb.dtype == np.uint8 and rgb.flags.c_contiguous
    np.testing.assert_array_equal(rgb, np.broadcast_to([7, 11, 19], rgb.shape))
    np.testing.assert_array_equal(picked, np.full((2, 3), 65794))
    image_pass, labels_pass, pick_pass = [d[1] for d in gl.draws[:3]]
    assert image_pass['u_low'] == 2 and image_pass['u_high'] == 20
    assert image_pass['u_opacity'] == .4 and image_pass['u_samples'] == 4
    assert image_pass['u_lens_zoom'] == 3 and image_pass['u_lens_dim'] == .2
    assert labels_pass['u_mode'] == (0 if label_mode == 'surface' else 1)
    assert labels_pass['u_highlight'] == 3 and labels_pass['u_size'] == (2., 1., 1.)
    assert labels_pass['u_tint'] == L.surface_finish('metallic').tint
    assert pick_pass['u_mode'] == 2
    assert ('state', {'blend': False, 'depth_test': True, 'depth_func': 'less'}) in gl.states
    assert len(gl.draws) == (4 if size == (4, 6) else 3)
    assert gl.flushes == 1
    renderer.close()
    assert all(p.deleted == 1 for p in gl.programs)


def test_gl_draw_error_releases_queue_and_empty_frame_has_no_layer_commands(gl):
    renderer = L.GLVolumeRenderer()
    gl.fail_draw = True
    with pytest.raises(RuntimeError, match='recorded draw failure'):
        renderer.render(scene(), rays())
    assert gl.flushes == 1 and gl.forgotten == [renderer._canvas]
    gl.fail_draw = False
    rgb, picked = renderer.render([], rays())
    assert rgb.shape == (2, 3, 3) and picked.shape == (2, 3)
    assert not gl.draws
    renderer.close()


@pytest.mark.parametrize('objects,message', [
    ([L.ImageLayer(np.zeros((2, 2, 2))), L.LabelsLayer(np.zeros((3, 2, 2), int))], 'share one shape'),
    ([L.ImageLayer(np.zeros((2, 2)))], '3-D layers only'),
])
def test_gl_rejects_incompatible_dimensions_before_allocating(gl, objects, message):
    with pytest.raises(L.LayerError, match=message):
        L.GLVolumeRenderer().render(objects, rays())
    assert gl.current is None and gl.resources == []


def test_cpu_reductions_ignore_outside_samples_and_preserve_empty_rays():
    values = np.array([[.2, .8], [0., 0.]])
    inside = np.array([[True, True], [False, False]])
    np.testing.assert_allclose(L._reduce_image(values, inside, 'minip', 1), [.2, 0])
    np.testing.assert_allclose(L._reduce_image(values, inside, 'average', 1), [.5, 0])
    with pytest.raises(L.LayerError, match='positive lengths'):
        L.SurfaceLight.for_rays(rays(), voxel_size=(1, 0, 1))
    with pytest.raises(L.LayerError, match='share one shape'):
        L.render_volume([L.ImageLayer(np.zeros((2, 2, 2))),
                         L.LabelsLayer(np.zeros((3, 2, 2), int))], rays())


def test_cpu_lens_covers_forward_reverse_and_parallel_rays():
    lens = L.VolumeLens((1., 1., 1.), (.5, .5, .5))
    forward = rays()
    reverse = L.VolumeRays((3., 0., 0.), forward.across, forward.down, (-1., 0., 0.), forward.shape)
    expected = np.array([[False, False, False], [False, True, False]])
    np.testing.assert_array_equal(lens.covers(forward), expected)
    np.testing.assert_array_equal(lens.covers(reverse), expected)


def test_single_pixel_rays_preserve_the_full_grid_samples_and_fractional_pick():
    grid = rays()
    single = grid.ray(1, 2)
    assert single.shape == (1, 1, 4)
    assert single.across == grid.across and single.down == grid.down
    assert single.depth == grid.depth
    np.testing.assert_array_equal(single.positions(0, 1), grid.positions(1, 2)[:, 2:3])
    fractional = grid.ray(.25, 1.5)
    np.testing.assert_array_equal(fractional.origin, [0., .25, 1.5])


@pytest.mark.parametrize('mode', ['surface', 'translucent'])
def test_cpu_label_compositing_highlights_only_the_picked_object_and_dims_outside_lens(mode):
    _, labels = scene(mode)
    plain, picked = L.render_volume([labels], rays())
    highlighted, other_picked = L.render_volume([labels], rays(), highlight=3)
    np.testing.assert_array_equal(picked, np.full((2, 3), 3))
    np.testing.assert_array_equal(other_picked, picked)
    assert np.all(highlighted >= plain) and np.any(highlighted > plain)
    lens = L.VolumeLens((100., 100., 100.), (1., 1., 1.), dim=.25)
    dimmed, dim_picked = L.render_volume([labels], rays(), lens=lens)
    np.testing.assert_allclose(dimmed, plain * .25, atol=1e-7)
    np.testing.assert_array_equal(dim_picked, picked)
    empty = L.LabelsLayer(np.zeros((4, 2, 3), int), metadata={'rendering': mode})
    rgb, ids = L.render_volume([empty], rays())
    assert not rgb.any() and not ids.any()


class Buffer:
    def __init__(self, size=None, data=None, usage=0):
        self.data = None if data is None else np.array(data, copy=True)
        self.size = int(self.data.nbytes if size is None else size)
        self.usage = usage
        self.destroyed = 0

    def destroy(self):
        self.destroyed += 1


@pytest.fixture
def webgpu(monkeypatch):
    """Record WebGPU descriptors and byte transport without importing wgpu."""
    fake = types.ModuleType('wgpu')
    fake.GPUBuffer = Buffer
    fake.BufferUsage = types.SimpleNamespace(STORAGE=1, COPY_SRC=2, UNIFORM=4, MAP_READ=8, COPY_DST=16)
    fake.TextureUsage = types.SimpleNamespace(TEXTURE_BINDING=1, COPY_DST=2)
    fake.TextureFormat = types.SimpleNamespace(r32uint='r32uint')
    monkeypatch.setitem(sys.modules, 'wgpu', fake)
    device = types.SimpleNamespace(resources=[], shaders=[], pipelines=[], bindings=[],
                                   writes=[], buffer_writes=[], reads=[], read_targets=[], submissions=[],
                                   commands=[], destroyed=0)

    class Texture:
        def __init__(self, **descriptor):
            if any(side <= 0 for side in descriptor['size']):
                raise RuntimeError('zero-sized texture refused')
            self.descriptor = descriptor
            self.destroyed = 0
            self.view = types.SimpleNamespace(texture=self)

        def create_view(self):
            return self.view

        def destroy(self):
            self.destroyed += 1

    def make(cls, **kw):
        obj = cls(**kw)
        device.resources.append(obj)
        return obj

    def shader(**descriptor):
        device.shaders.append(descriptor)
        return descriptor

    def pipeline(**descriptor):
        obj = types.SimpleNamespace(descriptor=descriptor,
                                    get_bind_group_layout=lambda index: ('layout', index))
        device.pipelines.append(obj)
        return obj

    def group(**descriptor):
        device.bindings.append(descriptor)
        return descriptor

    def write_texture(destination, data, layout, size):
        device.writes.append((destination, np.array(data, copy=True), layout, size))

    def write_buffer(buffer, offset, data):
        buffer.data = np.array(data, copy=True)
        device.buffer_writes.append((buffer, offset, buffer.data))

    def read_mapped(size, copy):
        assert copy is True
        data = b''.join(device.reads)
        device.reads = []
        assert len(data) == size
        return data

    class Compute:
        def set_pipeline(self, pipeline):
            device.commands.append(('pipeline', pipeline))

        def set_bind_group(self, index, group):
            device.commands.append(('bindings', index, group))

        def dispatch_workgroups(self, *grid):
            device.commands.append(('dispatch', grid))

        def end(self):
            device.commands.append(('end',))

    class Encoder:
        def begin_compute_pass(self):
            device.commands.append(('begin',))
            return Compute()

        def copy_buffer_to_buffer(self, source, source_offset, target, target_offset, size):
            device.read_targets.append((source, source_offset, target, target_offset, size))

        def finish(self):
            device.commands.append(('finish',))
            return 'recorded commands'

    device.create_texture = lambda **kw: make(Texture, **kw)
    def create_buffer(**kw):
        buffer = make(Buffer, **kw)
        buffer.map_sync = lambda mode, size: device.commands.append(('map', mode, size))
        buffer.read_mapped = read_mapped
        buffer.unmap = lambda: device.commands.append(('unmap',))
        return buffer

    device.create_buffer = create_buffer
    device.create_buffer_with_data = lambda **kw: make(Buffer, **kw)
    device.create_shader_module = shader
    device.create_compute_pipeline = pipeline
    device.create_bind_group = group
    device.create_command_encoder = Encoder
    device.destroy = lambda: setattr(device, 'destroyed', device.destroyed + 1)
    device.queue = types.SimpleNamespace(write_texture=write_texture, write_buffer=write_buffer,
                                        
                                         submit=lambda commands: device.submissions.append(commands))
    asked = []
    def request(**kw):
        asked.append(kw)
        return device
    adapter = types.SimpleNamespace(
        info={'backend_type': 'Vulkan', 'adapter_type': 'DiscreteGPU',
              'device': 'recorded device', 'description': 'recorded driver'},
        limits={'max_texture_dimension_3d': 64, 'max-storage-buffer-binding-size': 4096,
                'max-buffer-size': 8192, 'unrequested-limit': 123},
        request_device_sync=request, request_device=request)
    return types.SimpleNamespace(module=fake, device=device, adapter=adapter, asked=asked)


def test_wgpu_requests_only_supported_limits_and_handles_both_adapter_apis(webgpu):
    renderer = L.WGPUVolumeRenderer(webgpu.adapter)
    assert webgpu.asked == [{'required_limits': {
        'max_texture_dimension_3d': 64, 'max-storage-buffer-binding-size': 4096,
        'max-buffer-size': 8192}}]
    assert renderer.limits() == {'renderer': 'recorded device', 'version': 'recorded driver',
                                 'max_3d': 64, 'adapter_type': 'DiscreteGPU'}
    assert renderer.backend == 'vulkan' and renderer._limit('unknown', 12) == 12
    webgpu.adapter.request_device_sync = None
    webgpu.adapter.limits = None
    other = L.WGPUVolumeRenderer(webgpu.adapter)
    assert webgpu.asked[-1] == {'required_limits': {}}
    assert other.limits()['max_3d'] == 2048
    renderer.close()
    assert webgpu.device.destroyed == 1
    webgpu.device.destroy = None
    other.close()


def test_wgpu_volume_descriptors_preserve_float_bits_ids_and_noncontiguous_data(webgpu):
    renderer = L.WGPUVolumeRenderer(webgpu.adapter)
    source = np.linspace(-2, 4, 24, dtype=np.float32).reshape(4, 2, 3)[:, :, ::-1]
    texture, view = renderer._volume(source, 'image')
    destination, bits, layout, size = webgpu.device.writes[-1]
    assert texture.descriptor == {'size': (3, 2, 4), 'dimension': '3d', 'format': 'r32uint', 'usage': 3}
    assert destination == {'texture': texture, 'mip_level': 0, 'origin': (0, 0, 0)}
    assert layout == {'offset': 0, 'bytes_per_row': 12, 'rows_per_image': 2}
    assert size == (3, 2, 4) and bits.flags.c_contiguous and bits.dtype == np.uint32
    np.testing.assert_array_equal(bits.view(np.float32), source)
    assert renderer._volume(source, 'image') == (texture, view)
    assert len(webgpu.device.writes) == 1
    ids = np.array([0, (1 << 32) - 1], np.uint64).reshape(1, 1, 2)
    renderer._volume(ids, 'ids')
    np.testing.assert_array_equal(webgpu.device.writes[-1][1], ids)
    renderer._forget_unused()
    renderer._volume(source, 'image')
    renderer._forget_unused()
    assert texture.destroyed == 0
    assert webgpu.device.resources[1].destroyed == 1
    renderer.close()
    assert texture.destroyed == 1


@pytest.mark.parametrize('array,message', [
    (np.array([[[-1]]]), 'object IDs must lie'),
    (np.array([[[1 << 32]]], np.uint64), 'object IDs must lie'),
    (np.zeros((65, 1, 1), np.uint32), 'exceeds this GPU'),
])
def test_wgpu_rejects_unrepresentable_volumes_before_upload(webgpu, array, message):
    renderer = L.WGPUVolumeRenderer(webgpu.adapter)
    with pytest.raises(L.LayerError, match=message):
        renderer._volume(array, 'ids')
    assert webgpu.device.resources == [] and webgpu.device.writes == []


def test_wgpu_empty_volume_colours_luts_and_buffer_replacement_are_bounded(webgpu):
    renderer = L.WGPUVolumeRenderer(webgpu.adapter)
    with pytest.raises(RuntimeError, match='zero-sized texture refused'):
        renderer._volume(np.zeros((0, 1, 1), np.uint32), 'ids')
    assert not webgpu.device.resources and not webgpu.device.writes
    ids = np.array([0, 3], np.uint32)
    colours, no_view = renderer._colours(ids)
    assert no_view is None and colours.data.shape == (4, 4)
    np.testing.assert_allclose(colours.data[ids, :3], L.label_colors(ids))
    assert np.all(colours.data[:, 3] == 1)
    assert renderer._colours(ids)[0] is colours
    empty, _ = renderer._colours(np.array([], np.uint32))
    assert empty.data.shape == (1, 4)
    with pytest.raises(L.LayerError, match='too many'):
        renderer._colours(np.array([256], np.uint32))
    cmap = L.colormap('red')
    lut = renderer._lut(cmap)
    assert renderer._lut(cmap) is lut and lut.data.shape == (256, 4)
    np.testing.assert_array_equal(lut.data[[0, -1]], [[0, 0, 0, 1], [1, 0, 0, 1]])
    first = renderer._buffer('pick', 32, True)
    assert first.usage == 3 | 16 and renderer._buffer('pick', 32, True) is first
    second = renderer._buffer('pick', 64, True)
    assert first.destroyed == 1 and second is not first
    frame = renderer._buffer('frame', 64, False)
    assert frame.usage == 1
    renderer.close()
    assert second.destroyed == frame.destroyed == lut.destroyed == colours.destroyed == 1
    assert not renderer._textures and not renderer._buffers and not renderer._luts


@pytest.mark.parametrize('label_mode,size', [('surface', None), ('translucent', (4, 9))])
def test_wgpu_serializes_camera_light_and_picking_without_executing_shaders(webgpu, label_mode, size):
    renderer = L.WGPUVolumeRenderer(webgpu.adapter)
    height, width = (2, 3) if size is None else size
    webgpu.device.reads = [np.tile(np.array([7, 11, 19, 255], np.uint8), height * width).tobytes(),
                          np.tile(np.array([0xfedcba98, 17], np.uint32), 6).tobytes()]
    lens = L.VolumeLens((1., 1., 1.), (.5, .5, .5), zoom=3, dim=.2)
    hidden = L.ImageLayer(np.zeros((1, 1, 1)), visible=False)
    rgb, picked = renderer.render([hidden, *scene(label_mode)], rays(), highlight=3,
                                  voxel_size=(2, 1, 1), lens=lens, size=size)
    np.testing.assert_array_equal(rgb, np.broadcast_to([7, 11, 19], (height, width, 3)))
    np.testing.assert_array_equal(picked, np.full((2, 3), 0xfedcba98, np.int64))
    assert rgb.dtype == np.uint8 and rgb.flags.c_contiguous and picked.dtype == np.int64
    picture, pick, readback = (renderer._buffers[name][1] for name in ('picture', 'pick', 'readback'))
    assert webgpu.device.read_targets == [(picture, 0, readback, 0, height * width * 4),
                                          (pick, 0, readback, height * width * 4, 48)]
    assert webgpu.device.commands[-3:] == [
        ('finish',), ('map', 'READ', height * width * 4 + 48), ('unmap',)]
    assert webgpu.device.submissions == [['recorded commands']]
    groups = webgpu.device.bindings
    assert len(groups) == 4
    uniforms = [data for _buffer, _offset, data in webgpu.device.buffer_writes]
    assert all(p.shape == (17, 4) and p.dtype == np.float32 for p in uniforms)
    np.testing.assert_array_equal(uniforms[1][9], [2, 20, np.float32(.4), 1])
    np.testing.assert_allclose(uniforms[1][8, :2], [3, .2])
    assert uniforms[2][5, 3] == (0 if label_mode == 'surface' else 1)
    np.testing.assert_array_equal(uniforms[2][11, :3], [2, 1, 1])
    assert uniforms[2][10, 3] == 3
    finish = L.surface_finish('metallic')
    np.testing.assert_allclose(uniforms[2][15], [finish.ambient, finish.diffuse, finish.specular, finish.shininess])
    np.testing.assert_array_equal(uniforms[-1][16, 1:3], [height, width])
    assert groups[1]['entries'][2]['resource'].texture is webgpu.device.writes[0][0]['texture']
    colours = groups[2]['entries'][4]['resource']
    assert colours['offset'] == 0 and colours['size'] == 4 * 4 * 4
    np.testing.assert_allclose(colours['buffer'].data[3, :3], L.label_color(3))
    dispatches = [event[1] for event in webgpu.device.commands if event[0] == 'dispatch']
    assert dispatches == [(1, 1, 1)] * 3 + [((-(-width // 8)), -(-height // 8), 1)]
    assert renderer._pipeline('image') is renderer._pipelines['image']
    assert len(webgpu.device.pipelines) == 4
    renderer.close()


def test_wgpu_empty_frame_and_parameter_fallback_do_not_allocate_volume_textures(webgpu):
    renderer = L.WGPUVolumeRenderer(webgpu.adapter)
    webgpu.device.reads = [bytes(4), bytes(6 * 8)]
    rgb, picked = renderer.render([], rays(), size=(0, -1))
    assert rgb.shape == (1, 1, 3) and not rgb.any() and not picked.any()
    assert not renderer._textures and not webgpu.device.writes
    params = renderer._params(rays(), (), None)
    np.testing.assert_array_equal(params[4, :3], [1, 1, 1])
    assert not params[6:9].any()
    renderer.close()


@pytest.mark.parametrize('objects,message', [
    ([L.ImageLayer(np.zeros((2, 2, 2))), L.LabelsLayer(np.zeros((3, 2, 2), int))], 'share one shape'),
    ([L.ImageLayer(np.zeros((2, 2)))], '3-D layers only'),
])
def test_wgpu_invalid_dimensions_do_not_start_commands(webgpu, objects, message):
    renderer = L.WGPUVolumeRenderer(webgpu.adapter)
    with pytest.raises(L.LayerError, match=message):
        renderer.render(objects, rays())
    assert not webgpu.device.resources and not webgpu.device.commands


def test_wgpu_upload_failure_can_be_closed_without_leaking_owned_buffers(webgpu, monkeypatch):
    renderer = L.WGPUVolumeRenderer(webgpu.adapter)
    def failed(**kwargs):
        raise RuntimeError('texture allocation refused')
    monkeypatch.setattr(webgpu.device, 'create_texture', failed)
    with pytest.raises(RuntimeError, match='allocation refused'):
        renderer.render(scene(), rays())
    owned = [buffer for _size, buffer in renderer._buffers.values()]
    renderer.close()
    assert len(owned) == 3 and all(buffer.destroyed == 1 for buffer in owned)
    assert webgpu.device.destroyed == 1


def test_wgpu_enumeration_optional_dependency_and_broken_adapter_are_empty(webgpu, monkeypatch):
    import importlib.util
    monkeypatch.setattr(importlib.util, 'find_spec', lambda name: None)
    assert L._wgpu_enumerate() == []
    monkeypatch.setattr(importlib.util, 'find_spec', lambda name: object())
    webgpu.module.gpu = types.SimpleNamespace(enumerate_adapters_sync=lambda: [webgpu.adapter])
    assert L._wgpu_enumerate() == [webgpu.adapter]
    webgpu.module.gpu = types.SimpleNamespace(enumerate_adapters=lambda: [webgpu.adapter])
    assert L._wgpu_enumerate() == [webgpu.adapter]
    def failed():
        raise RuntimeError('adapter unavailable')
    webgpu.module.gpu.enumerate_adapters = failed
    assert L._wgpu_enumerate() == []
    with pytest.raises(RuntimeError, match='not a wgpu backend'):
        L.wgpu_adapter('invalid')
