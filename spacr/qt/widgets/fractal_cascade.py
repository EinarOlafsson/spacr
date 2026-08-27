"""The fold-inversion cascade: the second spaceout pattern.

From the maintainer's `fold-inversion fractal cascade` v1.0.0, ported to
PySide6. It is a genuinely different fractal from the orbit-fold in
:mod:`spacr.qt.widgets.fractal_travel`, not the same one with other numbers:

* the map is a Kaliset-like fold/inversion -- reflect, swap, rotate, invert
  through a sphere, displace -- and the colour comes from THREE ORBIT TRAPS
  through that set (a ring, a diagonal and a slant) rather than from an
  escape count;
* the endless inward travel is two overlapping logarithmic scale windows.
  The outgoing scale is dropped only once the incoming one has reached the
  same view, so there is no reset and no microscopic number to lose
  precision on;
* the CPU renderer does TRUE SPATIAL 2x2 supersampling inside one frame --
  four samples of the same instant, averaged. The orbit pattern's CPU path
  jitters across four frames instead, which is cheaper but blends different
  animation times. That is the reason this one costs four times as much per
  pixel and renders a quarter of the pixels to pay for it.
"""
from __future__ import annotations

import math
from typing import Final

import numpy as np

try:
    from numba import njit, prange
except Exception:                                            # pragma: no cover
    njit = None
    prange = range


VERSION: Final[str] = "1.0.0"

#: The scale ratio between the two windows, and its logarithm. A window is
#: retired exactly when the next has reached the same view.
SCALE_BASE: Final[float] = 2.35
LOG_SCALE_BASE: Final[float] = 0.8544153281560676

_FAST_PI: Final[float] = math.pi
_FAST_TWO_PI: Final[float] = 2.0 * math.pi


FRAGMENT_SHADER: Final[str] = """
uniform vec2 u_resolution;
uniform float u_time;
uniform float u_speed;
uniform float u_dream;
uniform float u_palette_phase;
uniform float u_tx;
uniform float u_ty;
uniform float u_rotation;
uniform float u_shear_x;
uniform float u_shear_y;
uniform float u_stretch_x;
uniform float u_stretch_y;
uniform int u_detail;

const float SCALE_BASE = 2.35;
const float LOG_SCALE_BASE = 0.8544153281560676;

vec2 rotate2(vec2 p, float a) {
    float cs = cos(a);
    float sn = sin(a);
    return vec2(cs * p.x - sn * p.y, sn * p.x + cs * p.y);
}

vec3 palette(float x) {
    vec3 a = vec3(0.50, 0.49, 0.48);
    vec3 b = vec3(0.47, 0.45, 0.52);
    vec3 c = vec3(1.00, 1.00, 1.00);
    vec3 d = vec3(0.02, 0.30, 0.62)
        + vec3(0.11, 0.07, 0.13)
        * sin(0.055 * u_time + vec3(0.0, 1.7, 3.1));
    return a + b * cos(6.28318530718 * (c * x + d));
}

vec4 fractal_layer(vec2 z) {
    float trap_ring = 10.0;
    float trap_diagonal = 10.0;
    float trap_slant = 10.0;
    float energy = 0.0;
    float amplitude = 1.0;

    float theta = 0.39
        + 0.095 * sin(0.071 * u_time)
        + 0.050 * sin(0.019 * u_time + 1.4);
    float cs = cos(theta);
    float sn = sin(theta);
    mat2 rotation = mat2(cs, -sn, sn, cs);

    vec2 constant = vec2(
        0.715
            + 0.052 * sin(0.083 * u_time)
            + 0.025 * sin(0.023 * u_time + 2.1),
        0.475
            + 0.061 * cos(0.091 * u_time + 0.7)
            + 0.022 * sin(0.029 * u_time)
    );

    for (int i = 0; i < 9; ++i) {
        if (i >= u_detail) {
            break;
        }
        z = abs(z);
        if (z.x < z.y) {
            z = z.yx;
        }
        z = rotation * z;
        float radius2 = dot(z, z) + 0.078;
        z = 1.325 * z / radius2 - constant;

        float fi = float(i);
        z += 0.022 * u_dream * vec2(
            sin(1.55 * z.y + 0.13 * u_time + 1.17 * fi),
            cos(1.43 * z.x - 0.11 * u_time - 1.03 * fi)
        );

        float radius = length(z);
        trap_ring = min(trap_ring, abs(radius - 0.575));
        trap_diagonal = min(trap_diagonal, abs(z.x - z.y));
        trap_slant = min(trap_slant, abs(z.x + 0.36 * z.y));
        energy += amplitude / (1.0 + 4.4 * abs(radius2 - 0.70));
        amplitude *= 0.72;
    }
    return vec4(trap_ring, trap_diagonal, trap_slant, energy);
}

vec4 fractal_window(vec2 p) {
    float depth = u_time * u_speed / 12.0;
    float phase = fract(depth);
    float scale_a = exp(-LOG_SCALE_BASE * phase);
    float scale_b = scale_a * SCALE_BASE;
    vec4 a = fractal_layer(p * scale_a);
    vec4 b = fractal_layer(p * scale_b);
    float blend = phase * phase * (3.0 - 2.0 * phase);
    return mix(a, b, blend);
}

vec3 render_sample(vec2 fragment_position) {
    float denominator = min(u_resolution.x, u_resolution.y);
    vec2 uv = (2.0 * fragment_position - u_resolution) / denominator;
    uv *= 1.08;

    uv = rotate2(uv, u_rotation);
    uv = mat2(u_stretch_x, u_shear_x, u_shear_y, u_stretch_y) * uv;
    uv += vec2(u_tx, u_ty);
    uv += 0.045 * u_dream * vec2(
        sin(0.19 * u_time + 1.35 * uv.y),
        cos(0.17 * u_time - 1.25 * uv.x)
    );

    vec4 traps = fractal_window(uv);
    float ring = exp(-20.0 * traps.x);
    float diagonal = exp(-17.0 * traps.y);
    float slant = exp(-12.5 * traps.z);
    float orbit = 0.19 * traps.w;

    float structure = 2.15 * ring + 1.55 * diagonal + 1.15 * slant + orbit;
    float fine = 0.5 + 0.5 * sin(
        2.65 * structure + 1.25 * diagonal - 0.92 * slant
        + u_palette_phase + 0.075 * u_time
    );
    float flow = 0.5 + 0.5 * cos(
        1.75 * structure - 0.105 * u_time + 1.8 * ring
    );
    float palette_index = 0.145 * structure + 0.18 * fine + 0.11 * flow;

    vec3 color = palette(palette_index);
    color += vec3(0.16, 0.27, 0.34) * diagonal * (0.35 + 0.65 * flow);
    color += vec3(0.31, 0.12, 0.28) * ring * (0.30 + 0.70 * fine);
    color += vec3(0.08, 0.19, 0.13) * slant;
    color = pow(max(color, vec3(0.0)), vec3(0.84));

    float vignette = 1.0 - smoothstep(0.60, 1.80, length(uv));
    color *= 0.76 + 0.24 * vignette;
    return clamp(color, 0.0, 1.0);
}

void main() {
    vec3 color = vec3(0.0);
    color += render_sample(gl_FragCoord.xy + vec2(-0.25, -0.25));
    color += render_sample(gl_FragCoord.xy + vec2( 0.25, -0.25));
    color += render_sample(gl_FragCoord.xy + vec2(-0.25,  0.25));
    color += render_sample(gl_FragCoord.xy + vec2( 0.25,  0.25));
    gl_FragColor = vec4(0.25 * color, 1.0);
}
"""


if njit is not None:

    @njit(inline="always", fastmath=True)
    def _fast_sin(value):
        value -= math.floor((value + _FAST_PI) / _FAST_TWO_PI) * _FAST_TWO_PI
        result = (1.2732395447351627 * value
                  - 0.4052847345693511 * value * abs(value))
        return 0.225 * (result * abs(result) - result) + result

    @njit(inline="always", fastmath=True)
    def _fast_cos(value):
        return _fast_sin(value + 0.5 * _FAST_PI)

    @njit(inline="always", fastmath=True)
    def _layer(zx, zy, iterations, rotation_cs, rotation_sn,
               constant_x, constant_y):
        """One scale window: the fold/inversion map and its three traps."""
        trap_ring = 10.0
        trap_diagonal = 10.0
        trap_slant = 10.0
        energy = 0.0
        amplitude = 1.0
        for _iteration in range(iterations):
            zx = abs(zx)
            zy = abs(zy)
            if zx < zy:
                zx, zy = zy, zx
            old_x = zx
            zx = rotation_cs * zx - rotation_sn * zy
            zy = rotation_sn * old_x + rotation_cs * zy
            radius2 = zx * zx + zy * zy + 0.078
            inverse = 1.325 / radius2
            zx = zx * inverse - constant_x
            zy = zy * inverse - constant_y
            radius = math.sqrt(zx * zx + zy * zy)
            ring_distance = abs(radius - 0.575)
            diagonal_distance = abs(zx - zy)
            slant_distance = abs(zx + 0.36 * zy)
            if ring_distance < trap_ring:
                trap_ring = ring_distance
            if diagonal_distance < trap_diagonal:
                trap_diagonal = diagonal_distance
            if slant_distance < trap_slant:
                trap_slant = slant_distance
            energy += amplitude / (1.0 + 4.4 * abs(radius2 - 0.70))
            amplitude *= 0.72
        return trap_ring, trap_diagonal, trap_slant, energy

    @njit(inline="always", fastmath=True)
    def _sample(px, py, width, height, t, dream, iterations,
                camera_cs, camera_sn, tx, ty, shear_x, shear_y,
                stretch_x, stretch_y, rotation_cs, rotation_sn,
                constant_x, constant_y, scale_a, scale_b, blend,
                palette_phase):
        denominator = float(min(width, height))
        x = (2.0 * px - width) / denominator * 1.08
        y = (height - 2.0 * py) / denominator * 1.08

        old_x = x
        x = camera_cs * x - camera_sn * y
        y = camera_sn * old_x + camera_cs * y
        old_x = x
        x = stretch_x * x + shear_x * y + tx
        y = shear_y * old_x + stretch_y * y + ty
        x += 0.045 * dream * _fast_sin(0.19 * t + 1.35 * y)
        y += 0.045 * dream * _fast_cos(0.17 * t - 1.25 * x)

        a0, a1, a2, a3 = _layer(x * scale_a, y * scale_a, iterations,
                                rotation_cs, rotation_sn,
                                constant_x, constant_y)
        b0, b1, b2, b3 = _layer(x * scale_b, y * scale_b, iterations,
                                rotation_cs, rotation_sn,
                                constant_x, constant_y)
        inverse_blend = 1.0 - blend
        trap_ring = inverse_blend * a0 + blend * b0
        trap_diagonal = inverse_blend * a1 + blend * b1
        trap_slant = inverse_blend * a2 + blend * b2
        energy = inverse_blend * a3 + blend * b3

        # Rational trap profiles: four exponentials per subpixel is the
        # single most expensive thing this kernel could do, and these stay
        # smooth and bounded for a fraction of it.
        ring_base = 1.0 / (1.0 + 20.0 * trap_ring)
        diagonal_base = 1.0 / (1.0 + 17.0 * trap_diagonal)
        slant_base = 1.0 / (1.0 + 12.5 * trap_slant)
        ring = ring_base * ring_base
        diagonal = diagonal_base * diagonal_base
        slant = slant_base * slant_base
        structure = (2.15 * ring + 1.55 * diagonal + 1.15 * slant
                     + 0.19 * energy)

        fine = 0.5 + 0.5 * _fast_sin(
            2.65 * structure + 1.25 * diagonal - 0.92 * slant
            + palette_phase + 0.075 * t)
        flow = 0.5 + 0.5 * _fast_cos(
            1.75 * structure - 0.105 * t + 1.8 * ring)
        palette_index = 0.145 * structure + 0.18 * fine + 0.11 * flow

        phase_r = 6.28318530718 * (palette_index + 0.02)
        phase_g = 6.28318530718 * (palette_index + 0.30)
        phase_b = 6.28318530718 * (palette_index + 0.62)
        red = 0.50 + 0.47 * _fast_cos(phase_r + 0.11 * _fast_sin(0.055 * t))
        green = 0.49 + 0.45 * _fast_cos(
            phase_g + 0.07 * _fast_sin(0.055 * t + 1.7))
        blue = 0.48 + 0.52 * _fast_cos(
            phase_b + 0.13 * _fast_sin(0.055 * t + 3.1))

        diagonal_light = diagonal * (0.35 + 0.65 * flow)
        ring_light = ring * (0.30 + 0.70 * fine)
        red += 0.16 * diagonal_light + 0.31 * ring_light + 0.08 * slant
        green += 0.27 * diagonal_light + 0.12 * ring_light + 0.19 * slant
        blue += 0.34 * diagonal_light + 0.28 * ring_light + 0.13 * slant

        screen_radius = math.sqrt(x * x + y * y)
        vignette = 1.0 - max(0.0, min(1.0, (screen_radius - 0.60) / 1.20))
        brightness = 0.76 + 0.24 * vignette
        red = max(0.0, min(1.0, red)) * brightness
        green = max(0.0, min(1.0, green)) * brightness
        blue = max(0.0, min(1.0, blue)) * brightness
        return int(255.0 * red), int(255.0 * green), int(255.0 * blue)

    @njit(cache=True, parallel=True, fastmath=True, nogil=True)
    def render_into(output, t, speed, dream, iterations):
        """One complete frame, four spatial samples per pixel.

        Every sample is of the SAME instant, so the result is a true
        supersample rather than a blend across animation times.
        """
        camera_rotation = (
            0.26 * _fast_sin(_FAST_TWO_PI * t / 59.0)
            + 0.11 * _fast_sin(_FAST_TWO_PI * t / 211.0 + 0.7)
        ) * (0.55 + 0.75 * dream)
        camera_cs = _fast_cos(camera_rotation)
        camera_sn = _fast_sin(camera_rotation)
        tx = dream * (0.090 * _fast_sin(_FAST_TWO_PI * t / 47.0)
                      + 0.035 * _fast_sin(_FAST_TWO_PI * t / 131.0 + 1.1)
                      + 0.025 * _fast_cos(_FAST_TWO_PI * t / 307.0 + 0.6))
        ty = dream * (0.080 * _fast_cos(_FAST_TWO_PI * t / 53.0 + 0.5)
                      + 0.040 * _fast_sin(_FAST_TWO_PI * t / 149.0 + 0.2)
                      + 0.025 * _fast_sin(_FAST_TWO_PI * t / 283.0 + 1.8))
        shear_x = 0.18 * dream * _fast_sin(_FAST_TWO_PI * t / 73.0 + 0.2)
        shear_y = 0.16 * dream * _fast_cos(_FAST_TWO_PI * t / 89.0 + 0.8)
        stretch_x = math.exp(
            0.17 * dream * _fast_sin(_FAST_TWO_PI * t / 97.0 + 0.2))
        stretch_y = math.exp(
            0.15 * dream * _fast_cos(_FAST_TWO_PI * t / 107.0 + 1.4))

        theta = (0.39 + 0.095 * _fast_sin(0.071 * t)
                 + 0.050 * _fast_sin(0.019 * t + 1.4))
        rotation_cs = _fast_cos(theta)
        rotation_sn = _fast_sin(theta)
        constant_x = (0.715 + 0.052 * _fast_sin(0.083 * t)
                      + 0.025 * _fast_sin(0.023 * t + 2.1))
        constant_y = (0.475 + 0.061 * _fast_cos(0.091 * t + 0.7)
                      + 0.022 * _fast_sin(0.029 * t))

        depth = t * speed / 12.0
        phase = depth - math.floor(depth)
        scale_a = math.exp(-LOG_SCALE_BASE * phase)
        scale_b = scale_a * SCALE_BASE
        blend = phase * phase * (3.0 - 2.0 * phase)
        palette_phase = (0.38 * _fast_sin(_FAST_TWO_PI * t / 173.0)
                         + 0.22 * _fast_cos(_FAST_TWO_PI * t / 337.0 + 0.3))

        height, width, _channels = output.shape
        for y in prange(height):
            for x in range(width):
                red = 0
                green = 0
                blue = 0
                for dy in range(2):
                    for dx in range(2):
                        r, g, b = _sample(
                            x + 0.25 + 0.5 * dx, y + 0.25 + 0.5 * dy,
                            width, height, t, dream, iterations,
                            camera_cs, camera_sn, tx, ty, shear_x, shear_y,
                            stretch_x, stretch_y, rotation_cs, rotation_sn,
                            constant_x, constant_y, scale_a, scale_b, blend,
                            palette_phase)
                        red += r
                        green += g
                        blue += b
                output[y, x, 0] = red // 4
                output[y, x, 1] = green // 4
                output[y, x, 2] = blue // 4

else:                                                        # pragma: no cover

    def render_into(*_args, **_kwargs):
        raise RuntimeError("numba is required for the cascade CPU backend")


class CascadeEngine:
    """One supersampled buffer. NO temporal history at all.

    The orbit pattern keeps a four-frame ring because its antialiasing walks
    the sub-pixel grid over time. This one takes all four samples inside a
    single frame, so there is nothing to remember between frames -- which is
    also why pausing and resuming it cannot show a seam.
    """

    def __init__(self, thread_count: int) -> None:
        self.thread_count = max(1, int(thread_count))
        self.width = 0
        self.height = 0
        self.output = None

    def _ensure_size(self, width: int, height: int) -> None:
        if width == self.width and height == self.height and self.output is not None:
            return
        self.width = width
        self.height = height
        self.output = np.empty((height, width, 3), dtype=np.uint8)

    def render(self, width: int, height: int, t: float, speed: float,
               dream: float, iterations: int) -> np.ndarray:
        from numba import set_num_threads

        set_num_threads(self.thread_count)
        self._ensure_size(width, height)
        render_into(self.output, t, speed, dream, iterations)
        return self.output.copy()
