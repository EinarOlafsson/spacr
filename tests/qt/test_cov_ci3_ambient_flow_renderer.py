"""N684: the optional OpenGL renderer drives its context as documented.

No graphics driver is used. ``moderngl`` is replaced by a recording stand-in
that keeps buffers, textures, framebuffers and draw calls in memory, so
each renderer method can be checked for the state it sets (targets, blend
functions, scissor boxes, uniforms, instance counts), the resources it
owns and releases, and the pixels it reads back.
"""
import re
import sys
import threading
import types

import numpy as np
import pytest
from PySide6.QtGui import QColor, QImage

from spacr.qt.widgets import ambient


class _Member:
    def __init__(self):
        self.value = None
        self.written = None

    def write(self, raw):
        self.written = bytes(raw)


class _Resource:
    def __init__(self, ctx, kind):
        self.ctx = ctx
        self.kind = kind
        self.released = False
        ctx.made.append(self)

    def release(self):
        self.released = True
        self.ctx.released.append(self)


class _Program(_Resource):
    def __init__(self, ctx, vertex_shader, fragment_shader):
        super().__init__(ctx, "program")
        if ctx.fail_program:
            raise RuntimeError("shader compile failed")
        self.source = vertex_shader + fragment_shader
        self.members = {name: _Member() for name in
                        re.findall(r"uniform\s+\w+\s+(\w+)", self.source)}
        self.attributes = set(re.findall(r"^\s*in\s+\w+\s+(\w+);",
                                         vertex_shader, re.M))
        self.attributes -= set(ctx.dropped)

    def __getitem__(self, name):
        return self.members[name]

    def __contains__(self, name):
        return name in self.members or name in self.attributes


class _Buffer(_Resource):
    def __init__(self, ctx, reserve):
        super().__init__(ctx, "buffer")
        self.size = reserve
        self.data = b""
        self.binding = None
        self.orphaned = []

    def orphan(self, size):
        self.orphaned.append(size)
        self.size = size

    def write(self, raw):
        assert len(raw) <= self.size
        self.data = bytes(raw)
        self.ctx.writes.append(self)

    def bind_to_storage_buffer(self, binding):
        self.binding = binding
        self.ctx.bindings[binding] = self


class _VertexArray(_Resource):
    def __init__(self, ctx, program, content):
        super().__init__(ctx, "vertex_array")
        self.program = program
        self.content = content

    def render(self, mode, vertices, instances=1):
        ctx = self.ctx
        buffer = self.content[0][0] if self.content else None
        ctx.renders.append(types.SimpleNamespace(
            program=self.program, mode=mode, vertices=vertices,
            instances=instances, data=buffer.data if buffer else None,
            layout=self.content[0][1:] if self.content else (),
            target=ctx.bound, scissor=ctx.scissor,
            blend=(ctx.blend_equation, ctx.blend_func),
            viewport=ctx.viewport, blending=ctx.FLAG_BLEND in ctx.enabled))


class _Texture(_Resource):
    def __init__(self, ctx, size, components, data=None, dtype="f1"):
        super().__init__(ctx, "texture")
        self.size = tuple(size)
        self.components = components
        self.data = data
        self.dtype = dtype
        self.repeat_x = self.repeat_y = True
        self.filter = None
        self.location = None

    def use(self, location=0):
        self.location = location


class _Framebuffer(_Resource):
    def __init__(self, ctx, attachments):
        super().__init__(ctx, "framebuffer")
        self.texture = attachments[0]
        self.colour = None

    def use(self):
        self.ctx.bound = self

    def clear(self, red, green, blue, alpha):
        self.colour = (red, green, blue, alpha)

    def read_into(self, buffer, components=4, alignment=1):
        width, height = self.texture.size
        pixel = bytes(int(round(channel * 255)) for channel in self.colour)
        view = memoryview(buffer).cast("B")
        view[:width * height * components] = pixel * (width * height)


class _Context:
    FLAG_POINT = 1
    FLAG_BLEND = 2

    def __init__(self, options, fail_program=False):
        self.options = options
        self.version_code = options["require"]
        self.fail_program = fail_program
        self.made, self.released, self.writes, self.renders = [], [], [], []
        self.bindings = {}
        self.enabled = set()
        self.bound = None
        self.viewport = None
        self.scissor = "unset"
        self.blend_equation = None
        self.blend_func = None
        self.info = {"GL_MAX_TEXTURE_SIZE": 8192}
        self.context_released = False
        self.dropped = ()

    def program(self, vertex_shader, fragment_shader):
        return _Program(self, vertex_shader, fragment_shader)

    def buffer(self, reserve=0):
        return _Buffer(self, reserve)

    def vertex_array(self, program, content):
        return _VertexArray(self, program, content)

    def texture(self, size, components, data=None, dtype="f1"):
        return _Texture(self, size, components, data, dtype)

    def framebuffer(self, attachments):
        return _Framebuffer(self, attachments)

    def enable(self, flags):
        for flag in (self.FLAG_POINT, self.FLAG_BLEND):
            if flags & flag:
                self.enabled.add(flag)

    def disable(self, flags):
        self.enabled.discard(flags)

    def release(self):
        self.context_released = True
        self.released.append(self)


def _fake_moderngl(fail_program=False):
    module = types.ModuleType("moderngl")
    module.contexts = []
    module.PROGRAM_POINT_SIZE = _Context.FLAG_POINT
    module.BLEND = _Context.FLAG_BLEND
    for name in ("MAX", "MIN", "FUNC_ADD", "ONE", "ZERO", "DST_COLOR",
                 "ONE_MINUS_SRC_ALPHA", "LINEAR", "NEAREST", "POINTS",
                 "TRIANGLE_STRIP", "TRIANGLES"):
        setattr(module, name, name)

    def create_context(**options):
        context = _Context(options, fail_program)
        module.contexts.append(context)
        return context

    module.create_context = create_context
    return module


@pytest.fixture
def gl(monkeypatch):
    module = _fake_moderngl()
    monkeypatch.setitem(sys.modules, "moderngl", module)
    return module


@pytest.fixture
def renderer(gl):
    made = ambient._FlowPointGraphics()
    yield made
    if made._context is not None:
        made._close()


def _ctx(renderer):
    return renderer._context


def _rgb(image, x=0, y=0):
    return image.pixel(x, y) & 0xFFFFFF


def test_construction_builds_the_point_program_on_the_platform_context(gl, renderer):
    context = _ctx(renderer)
    assert context.options == ambient._graphics_context_options()
    assert renderer._owner == threading.get_ident()
    assert renderer._vertices.size == 12
    assert renderer._palette.size == 768 * 4
    assert context.enabled == {_Context.FLAG_POINT, _Context.FLAG_BLEND}
    assert set(renderer._program.members) == {"canvas", "spread"}
    assert renderer._vao.content == [(renderer._vertices, "3i", "position")]


def test_a_failing_compile_releases_the_context_and_raises(monkeypatch):
    module = _fake_moderngl(fail_program=True)
    monkeypatch.setitem(sys.modules, "moderngl", module)
    with pytest.raises(RuntimeError, match="shader compile failed"):
        ambient._FlowPointGraphics()
    context, = module.contexts
    assert context.context_released


def test_point_frames_blend_by_theme_brightness_and_upload_once(renderer):
    context = _ctx(renderer)
    lookup = np.arange(256, dtype=np.uint32) | np.uint32(0xFF000000)
    px = np.array([1, 2, 3], dtype=np.int32)
    py = np.array([0, 1, 2], dtype=np.int32)
    levels = np.array([10, 20, 30], dtype=np.uint8)
    image = renderer._draw(8, 4, px, py, levels, lookup, True, True)
    assert (image.width(), image.height()) == (8, 4)
    assert image.format() == QImage.Format_RGB32
    assert _rgb(image) == 0x000000
    assert context.blend_equation == "MAX"
    assert renderer._program["spread"].value is True
    assert renderer._program["canvas"].value == (8.0, 4.0)
    render = context.renders[-1]
    assert (render.mode, render.vertices) == ("POINTS", 3)
    points = np.frombuffer(renderer._vertices.data, dtype=np.int32).reshape(-1, 3)
    assert points.tolist() == [[1, 0, 10], [2, 1, 20], [3, 2, 30]]
    assert renderer._vertices.orphaned == [36]
    table = np.frombuffer(renderer._palette.data, dtype=np.uint32)
    assert len(table) == 768 and table[255] == lookup[255]
    assert table[256 + 255] == lookup[int(np.rint(255 * .68))]
    assert renderer._palette.binding == 2
    palette_writes = sum(1 for buffer in context.writes if buffer is renderer._palette)
    framebuffer = renderer._framebuffer
    light = renderer._draw(8, 4, px[:1], py[:1], levels[:1], lookup, False, False)
    assert renderer._framebuffer is framebuffer
    assert context.blend_equation == "MIN"
    assert _rgb(light) == 0xFFFFFF
    assert renderer._program["spread"].value is False
    assert sum(1 for buffer in context.writes
               if buffer is renderer._palette) == palette_writes


def test_point_frames_resize_their_target_and_skip_empty_draws(renderer):
    context = _ctx(renderer)
    lookup = np.zeros(256, dtype=np.uint32)
    empty = np.zeros(0, dtype=np.int32)
    renderer._draw(4, 4, empty, empty, empty.astype(np.uint8), lookup, False, True)
    first_fb, first_tex = renderer._framebuffer, renderer._texture
    renders = len(context.renders)
    image = renderer._draw(6, 3, empty, empty, empty.astype(np.uint8), lookup, False, True)
    assert len(context.renders) == renders
    assert (image.width(), image.height()) == (6, 3)
    assert first_fb.released and first_tex.released
    assert renderer._framebuffer.texture.size == (6, 3)
    assert renderer._vertices.data == b""


def test_drawing_and_targets_refuse_another_thread(renderer):
    errors = []

    def attempt():
        for call in (lambda: renderer._draw(2, 2, [], [], [], np.zeros(256, np.uint32),
                                            False, True),
                     lambda: renderer._target(2, 2, (0, 0, 0, 1)),
                     renderer._close):
            try:
                call()
            except RuntimeError as error:
                errors.append(str(error))

    thread = threading.Thread(target=attempt)
    thread.start()
    thread.join()
    assert errors == ["flow graphics context belongs to another thread"] * 2 + [
        "flow graphics release belongs to another thread"]
    assert renderer._context is not None


def test_programs_streams_and_arrays_are_built_once_and_reused(renderer):
    context = _ctx(renderer)
    blobs = renderer._shader("blobs")
    assert renderer._shader("blobs") is blobs
    assert blobs.source.startswith("#version 430")
    data = np.ones((2, 7), dtype=np.float32)
    stream = renderer._stream("s", data)
    assert stream.size == 64 and stream.orphaned == []
    big = np.ones((10, 7), dtype=np.float32)
    assert renderer._stream("s", big) is stream
    assert stream.orphaned == [big.nbytes]
    writes = len(context.writes)
    renderer._stream("s", np.zeros(0, dtype=np.float32))
    assert len(context.writes) == writes
    array = renderer._array(blobs, stream, "4f 3f/i", "shape", "tint")
    assert renderer._array(blobs, stream, "4f 3f/i", "shape", "tint") is array
    assert array.content == [(stream, "4f 3f/i", "shape", "tint")]
    bare = renderer._bare(blobs)
    assert renderer._bare(blobs) is bare and bare.content == []
    assert bare is not array


def test_uniforms_ignore_names_the_compiler_dropped(renderer):
    program = renderer._shader("blobs")
    renderer._uniform(program, "missing", 3.0)
    assert "missing" not in program.members
    renderer._uniform(program, "canvas", (2.0, 3.0))
    renderer._uniform(program, "stops", b"\x01\x02")
    renderer._uniform(program, "stops", bytearray(b"\x03"))
    assert program["canvas"].value == (2.0, 3.0)
    assert program["stops"].written == b"\x03" and program["stops"].value is None


def test_scratch_targets_follow_their_size_and_filter(renderer):
    texture, framebuffer = renderer._scratch("a", (5, 4))
    assert texture.filter == ("NEAREST", "NEAREST")
    assert (texture.repeat_x, texture.repeat_y) == (False, False)
    assert framebuffer.texture is texture and texture.components == 4
    assert renderer._scratch("a", (5, 4)) == (texture, framebuffer)
    smooth, smooth_fb = renderer._scratch("a", (7, 4), 1, filtered=True)
    assert texture.released and framebuffer.released
    assert smooth.filter == ("LINEAR", "LINEAR") and smooth.components == 1


@pytest.mark.parametrize("mode,func", [
    ("plus", ("ONE", "ONE", "ZERO", "ONE")),
    ("multiply", ("DST_COLOR", "ONE_MINUS_SRC_ALPHA", "ZERO", "ONE")),
    ("over", ("ONE", "ONE_MINUS_SRC_ALPHA", "ONE", "ONE_MINUS_SRC_ALPHA"))])
def test_compositions_match_the_painter_modes(renderer, mode, func):
    renderer._blend(mode)
    assert _ctx(renderer).blend_equation == "FUNC_ADD"
    assert _ctx(renderer).blend_func == func


def test_target_binds_clears_and_resets_scissor(renderer):
    context = _ctx(renderer)
    context.scissor = (1, 1, 2, 2)
    context.blend_equation = "MAX"
    renderer._target(9, 5, (0.25, 0.5, 0.75, 1.0))
    assert context.bound is renderer._framebuffer
    assert renderer._framebuffer.colour == (0.25, 0.5, 0.75, 1.0)
    assert context.viewport == (0, 0, 9, 5) and context.scissor is None
    assert context.blend_equation == "FUNC_ADD"
    old = renderer._framebuffer
    renderer._target(9, 5, (0, 0, 0, 0))
    assert renderer._framebuffer is old and not old.released
    texture = renderer._texture
    renderer._target(3, 2, (1, 1, 1, 1))
    assert old.released and texture.released
    assert renderer._framebuffer.texture.size == (3, 2)
    assert context.viewport == (0, 0, 3, 2)


def test_discs_draw_one_instance_per_row_and_nothing_for_none(renderer):
    context = _ctx(renderer)
    renderer._target(10, 10, (0, 0, 0, 0))
    renderer._discs(10, 10, np.zeros((0, 7)))
    assert context.renders == []
    rows = np.arange(14, dtype=np.float64).reshape(2, 7)
    renderer._discs(10, 10, rows, scale=(0.5, 2.0), smooth=False)
    render, = context.renders
    program = renderer._shader("discs")
    assert (render.mode, render.vertices, render.instances) == ("TRIANGLE_STRIP", 4, 2)
    assert program["scale"].value == (0.5, 2.0)
    assert program["smooth_edge"].value is False
    assert np.frombuffer(render.data, np.float32).tolist() == list(range(14))


@pytest.mark.parametrize("dark,identity,func", [
    (True, 0x000000, ("ONE", "ONE", "ZERO", "ONE")),
    (False, 0xFFFFFF, ("DST_COLOR", "ONE_MINUS_SRC_ALPHA", "ZERO", "ONE"))])
def test_blobs_add_on_dark_and_multiply_on_light(renderer, dark, identity, func):
    context = _ctx(renderer)
    rows = [(10, 10, 5, .5, .1, .2, .3), (20, 5, 4, .4, .3, .2, .1)]
    stops = [(0.0, 1.0), (0.4, 0.6), (0.8, 0.2), (1.0, 0.0)]
    image = renderer._draw_blobs(32, 16, rows, stops, dark)
    assert _rgb(image, 31, 15) == identity
    render, = context.renders
    assert render.instances == 2 and render.blend[1] == func
    program = renderer._shader("blobs")
    assert program["stops"].written == np.asarray(stops, np.float32).tobytes()
    assert program["canvas"].value == (32.0, 16.0)
    nothing = renderer._draw_blobs(32, 16, [], stops, dark)
    assert len(context.renders) == 1 and _rgb(nothing) == identity


def test_stars_are_drawn_over_a_transparent_premultiplied_frame(renderer):
    rows = np.array([[3, 3, 2, 1, 1, 1, 1]], dtype=np.float32)
    image = renderer._draw_stars(8, 8, rows, (1.0, 1.0), True)
    assert image.format() == QImage.Format_ARGB32_Premultiplied
    assert image.pixelColor(0, 0).alpha() == 0
    render, = _ctx(renderer).renders
    assert render.blend[1] == ("ONE", "ONE_MINUS_SRC_ALPHA", "ONE", "ONE_MINUS_SRC_ALPHA")


def test_aurora_generates_each_curtain_texture_then_warps_it(renderer):
    context = _ctx(renderer)
    lookup = np.full((256, 3), 0.5, dtype=np.float32)
    columns = np.ones((12, 4), dtype=np.float32)
    strips = np.ones((6, 4), dtype=np.float32)
    curtains = [(np.zeros((0, 4)), lookup, False, strips),
                (columns, lookup, True, np.zeros((0, 4))),
                (columns, lookup, True, strips)]
    image = renderer._draw_aurora(40, 20, curtains, True)
    assert _rgb(image) == 0
    generate, warp = context.renders
    assert generate.program is renderer._shader("aurora_texture")
    assert (generate.mode, generate.vertices) == ("TRIANGLES", 3)
    assert generate.viewport == (0, 0, 12, ambient._AURORA_TEXTURE_ROWS)
    assert not generate.blending and warp.blending
    assert generate.target.texture.size == (12, ambient._AURORA_TEXTURE_ROWS)
    assert generate.target.texture.filter == ("LINEAR", "LINEAR")
    assert warp.target is renderer._framebuffer and warp.vertices == 6
    assert warp.blend[1] == ("ONE", "ONE", "ZERO", "ONE")
    table = np.frombuffer(context.bindings[4].data, np.float32).reshape(-1, 4)
    assert table.shape == (256, 4) and set(table[:, 3]) == {0.0}
    assert np.frombuffer(context.bindings[3].data, np.float32).size == 48
    generator = renderer._shader("aurora_texture")
    assert generator["per_column"].value is True and generator["dark"].value is True
    assert generator["rows"].value == float(ambient._AURORA_TEXTURE_ROWS)
    assert renderer._shader("strips")["source"].value == (12.0, float(
        ambient._AURORA_TEXTURE_ROWS))
    light = renderer._draw_aurora(40, 20, curtains[2:], False)
    assert _rgb(light) == 0xFFFFFF
    assert context.renders[-1].blend[1][0] == "DST_COLOR"


def test_filaments_stroke_hairlines_directly_and_unite_thick_groups(renderer):
    context = _ctx(renderer)
    segments = np.array([
        [1, 1, 5, 1, 0.5, 3],
        [2, 2, 30, 12, 2.0, 1],
        [30, 12, 34, 14, 2.0, 2],
        [0, 0, 1, 1, 2.0, 3]], dtype=np.float32)
    groups = [(0, 1, 0, 0, 6, 2, .1, .2, .3, .4),
              (1, 2, -4, -3, 38.2, 12.5, .5, .6, .7, .8),
              (3, 1, 50, 50, 60, 60, 1, 1, 1, 1),
              (3, 0, 0, 0, 4, 4, 1, 1, 1, 1)]
    tips = np.array([[4, 4, 1, 1, 1, 1, 1]], dtype=np.float32)
    image = renderer._draw_filaments(36, 14, groups, segments, tips, True)
    assert _rgb(image) == 0
    hair, capsules, combine, tip = context.renders
    assert hair.program is renderer._shader("hairlines") and hair.instances == 1
    assert hair.scissor is None and hair.blend[1][0] == "ONE"
    assert capsules.program is renderer._shader("capsules")
    assert capsules.instances == 2 and capsules.blend == ("MAX", ("ONE", "ONE"))
    assert capsules.scissor == (0, 0, 36, 13)
    assert capsules.target.texture.components == 1
    assert combine.target is renderer._framebuffer and combine.scissor == (0, 0, 36, 13)
    assert renderer._shader("coverage")["bounds"].value == (0.0, 0.0, 36.0, 13.0)
    assert renderer._shader("coverage")["colour"].value == (.5, .6, .7, .8)
    assert renderer._shader("hairlines")["colour"].value == (.1, .2, .3, .4)
    assert tip.program is renderer._shader("discs") and tip.scissor is None
    context.renders.clear()
    renderer._draw_filaments(36, 14, [], segments, np.zeros((0, 7)), False)
    assert context.renders == []


def _tile(colour, width, height):
    image = QImage(width, height, QImage.Format_RGB32)
    image.fill(QColor(colour))
    return image


def test_tiles_pack_into_one_atlas_and_place_rotated_or_snapped(renderer):
    context = _ctx(renderer)
    material = ((0, 0, 0, 0, 0, _tile("#ff0000", 3, 2), 1.5, 1.0),
                (0, 0, 0, 0, 0, _tile("#00ff00", 2, 4), 1.0, 2.0))
    placements = [(0, 10.2, 8.0, 0.0), (1, 20.0, 9.0, 90.0)]
    image = renderer._draw_tiles(32, 16, material, placements, True)
    assert _rgb(image) == 0
    _material, atlas, rects = renderer._atlas
    assert rects == [(1, 1, 3, 2), (6, 1, 2, 4)]
    width, height = atlas.size
    assert width == 4096 and height == 6
    pixels = np.frombuffer(atlas.data, np.uint8).reshape(height, width, 4)
    assert pixels[1, 1, :3].tolist() == [0, 0, 255]
    assert pixels[2, 7, :3].tolist() == [0, 255, 0]
    assert pixels[0, 0].tolist() == [0, 0, 0, 255]
    render, = context.renders
    rows = np.frombuffer(render.data, np.float32).reshape(-1, 9)
    assert rows[0].tolist() == pytest.approx([9.0, 7.0, 0.0, 1, 1, 3, 2, 0.0, 0.0])
    assert rows[1].tolist() == pytest.approx(
        [20.0, 9.0, np.pi / 2, 6, 1, 2, 4, -1.0, -2.0])
    assert render.instances == 2 and atlas.location == 0
    renderer._draw_tiles(32, 16, material, placements[:1], True)
    assert renderer._atlas[1] is atlas
    replacement = material[:1]
    light = renderer._draw_tiles(32, 16, replacement, [], False)
    assert atlas.released and renderer._atlas[0] is replacement
    assert _rgb(light) == 0xFFFFFF
    fresh = np.frombuffer(renderer._atlas[1].data, np.uint8).reshape(-1, 4096, 4)
    assert fresh[0, 0].tolist() == [255, 255, 255, 255]


def test_tile_atlas_wraps_shelves_and_refuses_an_oversized_budget(renderer):
    texture, rects = renderer._pack_tiles((), True)
    assert rects == [(1, 1, 1, 1)] and texture.size == (4096, 3)
    wide = [(0, 0, 0, 0, 0, _tile("#123456", 3000, 2), 0, 0)] * 2
    _texture, rects = renderer._pack_tiles(tuple(wide), True)
    assert rects == [(1, 1, 3000, 2), (1, 5, 3000, 2)]
    _ctx(renderer).info["GL_MAX_TEXTURE_SIZE"] = 16
    with pytest.raises(RuntimeError, match="atlas budget"):
        renderer._pack_tiles(tuple(wide), True)


def test_close_releases_everything_once_and_survives_a_failing_release(renderer, caplog):
    context = _ctx(renderer)
    renderer._target(4, 4, (0, 0, 0, 1))
    scratch = renderer._scratch("x", (2, 2))
    material = ((0, 0, 0, 0, 0, _tile("#ffffff", 1, 1), 0.5, 0.5),)
    renderer._draw_tiles(4, 4, material, [], True)
    atlas = renderer._atlas[1]
    stubborn = renderer._shader("discs")

    def refuse():
        raise RuntimeError("driver lost")

    stubborn.release = refuse
    framebuffer = renderer._framebuffer
    with caplog.at_level("DEBUG", logger=ambient.LOG.name):
        renderer._close()
    assert all(item.released for item in (*scratch, atlas, framebuffer))
    assert context.context_released
    assert "could not release a flow graphics resource" in caplog.text
    assert renderer._context is None and renderer._resources == []
    for name in ("_programs", "_streams", "_arrays", "_scratch_targets", "_atlas"):
        assert name not in renderer.__dict__
    renderer._close()
    assert context.released.count(context) == 1


def test_arrays_pad_attributes_the_linker_removed(renderer):
    context = _ctx(renderer)
    context.dropped = ("tint",)
    program = renderer._shader("blobs")
    assert "shape" in program and "tint" not in program
    stream = renderer._stream("pad", np.zeros((1, 7), np.float32))
    array = renderer._array(program, stream, "4f 3f/i", "shape", "tint")
    assert array.content == [(stream, "4f 12x/i", "shape")]
    assert renderer._array(program, stream, "4f 3f/i", "shape", "tint") is array
    strips = renderer._shader("strips")
    plain = renderer._array(strips, stream, "4f", "vertex")
    assert plain.content == [(stream, "4f", "vertex")]
