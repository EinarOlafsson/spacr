"""Forward flight through a dark star field, with sparse celestial objects.

The third spaceout pattern, beside the orbit fold and the fold-inversion
cascade. It is the quiet one: predominantly black, with a very low-amplitude
broad hue field, six parallax star layers travelling toward the viewer, and
three object slots that pass by -- mostly stars, occasionally a lit planet or
a bright sun with a halo.

WHY IT SUITS A BACKDROP better than the other two. The orbit fold and the
cascade fill the frame with structure, which is what they are for and also
what makes them compete with the interface in front of them. Space is mostly
empty, so the thing a user is reading stays the brightest object on screen.

Both backends draw the same scene: the GLSL below and the Numba kernels in
`fractal_travel`'s CPU path share the star-field and object maths, so a
machine without a GPU sees the same flight rather than a different one.
"""
from __future__ import annotations

import math
from typing import Final

import numpy as np

try:
    from numba import njit, prange
except Exception:
    njit = None
    prange = range

#: Slots that carry a passing object. Three, because the scene is meant to be
#: sparse: the star field is the constant and an object is an event.
OBJECT_SLOTS: Final[int] = 3

#: Parallax layers in the star field. Each is a plane at its own depth, so
#: near stars sweep past while far ones barely move -- which is the whole
#: cue that says "forward" rather than "drifting".
STAR_LAYERS: Final[int] = 6

#: Above this hash value a cell holds a star. High, because the field has to
#: read as sparse: at 0.87 the grid becomes a texture rather than stars.
STAR_THRESHOLD: Final[float] = 0.935


FRAGMENT_SHADER: Final[str] = r"""
uniform vec2 u_resolution;
uniform float u_pointer_x;
uniform float u_pointer_y;
uniform float u_pull;
uniform float u_push;
uniform float u_time;
uniform float u_speed;
uniform float u_intensity;
uniform float u_palette_phase;
uniform float u_tx;
uniform float u_ty;
uniform float u_rotation;
uniform float u_shear_x;
uniform float u_shear_y;
uniform float u_stretch_x;
uniform float u_stretch_y;
uniform int u_detail;

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

vec2 rotate2(vec2 p, float a) {
    float cs = cos(a);
    float sn = sin(a);
    return vec2(cs * p.x - sn * p.y, sn * p.x + cs * p.y);
}

vec3 space_object(vec2 uv, float slot, float travel) {
    float phase = fract(travel + 0.33 * slot);
    float epoch = floor(travel + 0.33 * slot);
    float object_id = epoch * 3.0 + slot;
    float seed = hash21(vec2(object_id, 17.31));
    float type_value = hash21(vec2(object_id + 7.1, 3.77));

    float z = mix(7.2, 0.16, phase);
    float angle = 6.28318 * hash21(vec2(object_id + 2.2, 8.8));
    float radius = 0.70 + 1.00 * hash21(vec2(object_id + 4.7, 5.6));
    vec2 world = radius * vec2(cos(angle), sin(angle));
    vec2 projected = world / max(z, 0.16);
    vec2 local = (uv - projected) * z;
    float enter = smoothstep(0.02, 0.12, phase);
    float pass_out = 1.0 - smoothstep(0.90, 0.995, phase);

    float r = length(local) + 1e-5;
    float a = atan(local.y, local.x);
    vec3 color = vec3(0.0);

    if (type_value < 0.45) {
        float disc_radius = 0.34 + 0.12 * seed;
        float body = exp(-8.2 * r * r);
        float disc = step(r, disc_radius);
        float nz = sqrt(max(0.0, 1.0 - min(1.0, r * r / (disc_radius * disc_radius))));
        vec3 normal = normalize(vec3(local.x, local.y, nz));
        vec3 light_dir = normalize(vec3(-0.82, 0.25, 0.52));
        float light = max(0.0, dot(normal, light_dir)) * 0.92 + 0.08;
        float band = 0.86 + 0.14 * sin(6.2 * local.y + seed * 6.0);
        vec3 base = mix(vec3(0.40, 0.30, 0.16), vec3(0.78, 0.66, 0.40), seed);
        color += base * body * disc * light * band;
        color += vec3(0.10, 0.12, 0.20) * body * disc * (1.0 - light) * 0.65;
        color += vec3(0.94, 0.88, 0.70) * exp(-340.0 * pow(abs(r - disc_radius), 2.0)) * 0.16;
    } else {
        float core = exp(-920.0 * r * r);
        float halo = exp(-16.0 * r * r);
        float rays = exp(-36.0 * abs(local.x)) * exp(-8.0 * abs(local.y))
                   + exp(-36.0 * abs(local.y)) * exp(-8.0 * abs(local.x));
        float cloud = exp(-2.4 * r * r)
                    * (0.60 + 0.40 * sin(4.2 * a + 2.4 * log(r + 0.03) + seed * 6.0));
        vec3 star_col = mix(vec3(1.00, 0.92, 0.76), vec3(0.86, 0.90, 1.00), seed);
        vec3 cloud_col = mix(vec3(0.10, 0.12, 0.26), vec3(0.18, 0.20, 0.34), seed);
        color += cloud_col * cloud * 0.34;
        color += star_col * (2.9 * core + 0.95 * halo + 0.10 * rays);
    }

    return color * enter * pass_out;
}

vec3 space_star_field(vec2 uv, float depth) {
    vec3 color = vec3(0.0);
    for (int layer = 0; layer < 6; ++layer) {
        float lf = float(layer);
        float phase = fract(depth * (0.20 + 0.045 * lf) + 0.19 * lf);
        float z = mix(2.5, 0.12, phase);
        vec2 drift = vec2(
            0.060 * u_time * (0.10 + 0.025 * lf),
            -0.035 * u_time * (0.08 + 0.020 * lf)
        );
        vec2 plane = uv * z * 12.0 + drift + vec2(1.9 * lf, -1.4 * lf);
        vec2 base = floor(plane);
        float near_factor = 1.0 / (0.18 + z);
        for (int jy = -1; jy <= 1; ++jy) {
            for (int ix = -1; ix <= 1; ++ix) {
                vec2 cell = base + vec2(float(ix), float(jy));
                float h = hash21(cell + vec2(3.3 * lf, 7.9));
                if (h > 0.935) {
                    vec2 star = cell + vec2(
                        hash21(cell + vec2(1.2, 9.3)),
                        hash21(cell + vec2(5.4, 2.7))
                    );
                    vec2 d = (plane - star) / z;
                    float size = 0.0035 + 0.018 * near_factor;
                    float core = exp(-dot(d, d) / (size * size));
                    float ray_x = exp(-abs(d.x) / (0.008 + 0.025 * near_factor))
                        * exp(-abs(d.y) / (0.0014 + 0.006 * near_factor));
                    float ray_y = exp(-abs(d.y) / (0.008 + 0.025 * near_factor))
                        * exp(-abs(d.x) / (0.0014 + 0.006 * near_factor));
                    float brightness = (2.8 * core + 0.08 * (ray_x + ray_y))
                        * (0.38 + 0.62 * h) * pow(near_factor, 1.18);
                    vec3 star_color = mix(
                        vec3(1.00, 0.88, 0.74),
                        vec3(0.70, 0.83, 1.00),
                        hash21(cell + vec2(8.1, 1.4))
                    );
                    color += star_color * brightness;
                }
            }
        }
    }
    return color;
}


// THE POINTER IS THE POINT EVERYTHING FLOWS TO. Shifting the coordinate
// ORIGIN toward the cursor moves the centre the pattern radiates from,
// rather than adding a second warp on top of the one it already has --
// which would read as a smear rather than as a centre. A click pushes the
// origin away instead, so the flow reverses around it.
vec2 toward_pointer(vec2 uv) {
    vec2 target = vec2(u_pointer_x, u_pointer_y);
    return uv - target * (u_pull - 0.85 * u_push);
}

vec3 render_sample(vec2 fragment_position) {
    float denominator = min(u_resolution.x, u_resolution.y);
    vec2 uv = (2.0 * fragment_position - u_resolution) / denominator;
    uv *= 1.08;
    uv = toward_pointer(uv);

    float depth = u_time * u_speed / 14.0;
    float roll = 0.025 * sin(0.009 * u_time);
    vec2 p = rotate2(uv, roll);
    p += 0.025 * vec2(sin(0.006 * u_time), cos(0.005 * u_time + 0.7));

    float hue = 0.5 + 0.5 * sin(
        0.38 * p.x - 0.31 * p.y + 0.004 * u_time + 0.7 * sin(0.55 * p.y));
    vec3 color = mix(vec3(0.0015, 0.0022, 0.0060),
                     vec3(0.0055, 0.0028, 0.0105), hue);
    color += space_star_field(p, depth);

    float object_travel = u_time * (0.35 + 1.05 * u_speed) / 520.0;
    color += space_object(p, 0.0, object_travel);
    color += 0.82 * space_object(p, 1.0, object_travel + 0.27);
    color += 0.60 * space_object(p, 2.0, object_travel + 0.61);

    float vignette = 1.0 - smoothstep(1.0, 2.2, length(p));
    color *= 0.86 + 0.14 * vignette;
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
    @njit(cache=True, fastmath=True, inline="always")
    def _hash2(x: float, y: float) -> float:
        value = math.sin(x * 127.1 + y * 311.7) * 43758.5453123
        return value - math.floor(value)

    @njit(cache=True, fastmath=True)
    def _object_color(x: float, y: float, t: float, speed: float, slot: int):
        travel = t * (0.35 + 1.05 * speed) / 520.0 + 0.33 * slot
        epoch = math.floor(travel)
        phase = travel - epoch
        object_id = epoch * 3.0 + slot
        seed = _hash2(object_id, 17.31)
        type_value = _hash2(object_id + 7.1, 3.77)
        z = 7.2 + (0.16 - 7.2) * phase
        angle = 2.0 * math.pi * _hash2(object_id + 2.2, 8.8)
        radius = 0.70 + 1.00 * _hash2(object_id + 4.7, 5.6)
        projected_x = radius * math.cos(angle) / max(z, 0.16)
        projected_y = radius * math.sin(angle) / max(z, 0.16)
        local_x = (x - projected_x) * z
        local_y = (y - projected_y) * z
        enter = max(0.0, min(1.0, (phase - 0.02) / 0.10))
        pass_out = 1.0 - max(0.0, min(1.0, (phase - 0.90) / 0.095))
        r = math.sqrt(local_x * local_x + local_y * local_y) + 1e-5
        a = math.atan2(local_y, local_x)
        red = green = blue = 0.0
        if type_value < 0.45:
            disc_radius = 0.34 + 0.12 * seed
            body = math.exp(-8.2 * r * r)
            disc = 1.0 if r <= disc_radius else 0.0
            denom = max(1e-6, disc_radius * disc_radius)
            nz = math.sqrt(max(0.0, 1.0 - min(1.0, r * r / denom)))
            nx = local_x / max(disc_radius, 1e-5)
            ny = local_y / max(disc_radius, 1e-5)
            light = max(0.0, nx * -0.82 + ny * 0.25 + nz * 0.52) * 0.92 + 0.08
            band = 0.86 + 0.14 * math.sin(6.2 * local_y + seed * 6.0)
            lit = body * disc * light * band
            shadow = body * disc * (1.0 - light) * 0.65
            red += (0.40 * (1.0 - seed) + 0.78 * seed) * lit + 0.10 * shadow
            green += (0.30 * (1.0 - seed) + 0.66 * seed) * lit + 0.12 * shadow
            blue += (0.16 * (1.0 - seed) + 0.40 * seed) * lit + 0.20 * shadow
            rim = math.exp(-340.0 * (abs(r - disc_radius) ** 2)) * 0.16
            red += 0.94 * rim
            green += 0.88 * rim
            blue += 0.70 * rim
        else:
            core = math.exp(-920.0 * r * r)
            halo = math.exp(-16.0 * r * r)
            rays = (math.exp(-36.0 * abs(local_x)) * math.exp(-8.0 * abs(local_y))
                    + math.exp(-36.0 * abs(local_y)) * math.exp(-8.0 * abs(local_x)))
            cloud = math.exp(-2.4 * r * r) * (
                0.60 + 0.40 * math.sin(4.2 * a + 2.4 * math.log(r + 0.03)
                                       + seed * 6.0))
            glow = 2.9 * core + 0.95 * halo + 0.10 * rays
            red += (0.10 * (1.0 - seed) + 0.18 * seed) * cloud * 0.34 \
                + (1.00 * (1.0 - seed) + 0.86 * seed) * glow
            green += (0.12 * (1.0 - seed) + 0.20 * seed) * cloud * 0.34 \
                + (0.92 * (1.0 - seed) + 0.90 * seed) * glow
            blue += (0.26 * (1.0 - seed) + 0.34 * seed) * cloud * 0.34 \
                + (0.76 * (1.0 - seed) + 1.00 * seed) * glow
        return red * enter * pass_out, green * enter * pass_out, \
            blue * enter * pass_out

    @njit(cache=True, fastmath=True)
    def sample_space(x: float, y: float, t: float, speed: float):
        """One pixel of the flight, in scene coordinates.

        Kept a free function so a test can compare it with the shader and so
        the frame kernel below stays a loop and nothing else.
        """
        depth = t * speed / 14.0
        roll = 0.025 * math.sin(0.009 * t)
        cs = math.cos(roll)
        sn = math.sin(roll)
        px = cs * x - sn * y + 0.025 * math.sin(0.006 * t)
        py = sn * x + cs * y + 0.025 * math.cos(0.005 * t + 0.7)
        hue = 0.5 + 0.5 * math.sin(
            0.38 * px - 0.31 * py + 0.004 * t + 0.7 * math.sin(0.55 * py))
        red = 0.0015 * (1.0 - hue) + 0.0055 * hue
        green = 0.0022 * (1.0 - hue) + 0.0028 * hue
        blue = 0.0060 * (1.0 - hue) + 0.0105 * hue

        for layer in range(6):
            lf = float(layer)
            phase = (depth * (0.20 + 0.045 * lf) + 0.19 * lf) % 1.0
            z = 2.5 + (0.12 - 2.5) * phase
            plane_x = px * z * 12.0 + 0.060 * t * (0.10 + 0.025 * lf) + 1.9 * lf
            plane_y = py * z * 12.0 - 0.035 * t * (0.08 + 0.020 * lf) - 1.4 * lf
            base_x = math.floor(plane_x)
            base_y = math.floor(plane_y)
            near_factor = 1.0 / (0.18 + z)
            for jy in range(-1, 2):
                for ix in range(-1, 2):
                    cell_x = base_x + ix
                    cell_y = base_y + jy
                    h = _hash2(cell_x + 3.3 * lf, cell_y + 7.9)
                    if h > 0.935:
                        star_x = cell_x + _hash2(cell_x + 1.2, cell_y + 9.3)
                        star_y = cell_y + _hash2(cell_x + 5.4, cell_y + 2.7)
                        dx = (plane_x - star_x) / z
                        dy = (plane_y - star_y) / z
                        size = 0.0035 + 0.018 * near_factor
                        core = math.exp(-(dx * dx + dy * dy) / (size * size))
                        ray_x = math.exp(-abs(dx) / (0.008 + 0.025 * near_factor)) \
                            * math.exp(-abs(dy) / (0.0014 + 0.006 * near_factor))
                        ray_y = math.exp(-abs(dy) / (0.008 + 0.025 * near_factor)) \
                            * math.exp(-abs(dx) / (0.0014 + 0.006 * near_factor))
                        brightness = (2.8 * core + 0.08 * (ray_x + ray_y)) \
                            * (0.38 + 0.62 * h) * (near_factor ** 1.18)
                        mixv = _hash2(cell_x + 8.1, cell_y + 1.4)
                        red += (1.00 * (1.0 - mixv) + 0.70 * mixv) * brightness
                        green += (0.88 * (1.0 - mixv) + 0.83 * mixv) * brightness
                        blue += (0.74 * (1.0 - mixv) + 1.00 * mixv) * brightness

        r0, g0, b0 = _object_color(px, py, t, speed, 0)
        r1, g1, b1 = _object_color(px, py, t, speed, 1)
        r2, g2, b2 = _object_color(px, py, t, speed, 2)
        red += r0 + 0.82 * r1 + 0.60 * r2
        green += g0 + 0.82 * g1 + 0.60 * g2
        blue += b0 + 0.82 * b1 + 0.60 * b2
        radius = math.sqrt(px * px + py * py)
        vignette = 1.0 - max(0.0, min(1.0, (radius - 1.0) / 1.2))
        fac = 0.86 + 0.14 * vignette
        return (max(0.0, min(1.0, red * fac)),
                max(0.0, min(1.0, green * fac)),
                max(0.0, min(1.0, blue * fac)))

    @njit(cache=True, parallel=True, fastmath=True, nogil=True)
    def render_space_frame(width: int, height: int, t: float, speed: float,
                           offset_x: float, offset_y: float,
                           samples: int) -> np.ndarray:
        """A whole frame of the flight.

        ``nogil`` because this runs on the shading thread and the GUI thread
        has to keep answering while it does.
        """
        out = np.empty((height, width, 3), dtype=np.uint8)
        denominator = float(min(width, height))
        for row in prange(height):
            for col in range(width):
                if samples <= 1:
                    x = (2.0 * (col + 0.5) - width) / denominator * 1.08
                    y = (height - 2.0 * (row + 0.5)) / denominator * 1.08
                    r, g, b = sample_space(x + offset_x, y + offset_y,
                                           t, speed)
                    out[row, col, 0] = int(255.0 * r)
                    out[row, col, 1] = int(255.0 * g)
                    out[row, col, 2] = int(255.0 * b)
                else:
                    ar = ag = ab = 0.0
                    for oy in (0.25, 0.75):
                        for ox in (0.25, 0.75):
                            x = (2.0 * (col + ox) - width) / denominator * 1.08
                            y = (height - 2.0 * (row + oy)) / denominator * 1.08
                            r, g, b = sample_space(x + offset_x,
                                                   y + offset_y, t, speed)
                            ar += r
                            ag += g
                            ab += b
                    out[row, col, 0] = int(255.0 * ar / 4.0)
                    out[row, col, 1] = int(255.0 * ag / 4.0)
                    out[row, col, 2] = int(255.0 * ab / 4.0)
        return out

else:
    def sample_space(*_args, **_kwargs):
        raise RuntimeError("numba is required for the CPU space renderer")

    def render_space_frame(*_args, **_kwargs):
        raise RuntimeError("numba is required for the CPU space renderer")


class SpaceEngine:
    """The CPU side of the flight, driven exactly like the other engines.

    :param thread_count: worker threads numba may use.

    The widget builds and calls every pattern engine the same way, so this
    takes the same arguments even where the scene has no use for one. A
    pattern that needed a different call would put a branch in the one place
    all three are meant to look alike.
    """

    def __init__(self, thread_count: int) -> None:
        self.thread_count = max(1, int(thread_count))
        try:
            from numba import set_num_threads

            set_num_threads(self.thread_count)
        except Exception:                                    # noqa: BLE001
            # A thread cap that cannot be set is a slower frame, never a
            # reason for the backdrop not to draw.
            pass

    def render(self, width: int, height: int, t: float, speed: float,
               dream: float = 0.0, iterations: int = 0,
               pointer_x: float = 0.0, pointer_y: float = 0.0,
               pull: float = 0.0, push: float = 0.0) -> np.ndarray:
        """One finished frame as ``(height, width, 3)`` uint8.

        :param dream: unused. The flight has no dream term -- the scene is a
            star field, and warping it toward a hallucination is what the
            other two patterns are for.
        :param iterations: unused. The cost here is six parallax layers and
            three object slots, all fixed, so there is no depth to trade.
        :param pointer_x: where the pointer is, in scene coordinates.
        :param pointer_y: as above.
        :param pull: how strongly the pointer draws the flight toward it.
        :param push: a click's shove, decaying.

        THE POINTER STEERS RATHER THAN WARPS. The other patterns bend their
        field toward the cursor; bending a star field would make the stars
        curve, which reads as a fault rather than as attention. Here it
        nudges the flight's heading, so the field slides the way a camera
        pans and every star stays a point.
        """
        offset_x = float(pointer_x) * float(pull) * 0.22 - float(push) * float(pointer_x) * 0.35
        offset_y = float(pointer_y) * float(pull) * 0.22 - float(push) * float(pointer_y) * 0.35
        return render_space_frame(
            max(1, int(width)), max(1, int(height)), float(t), float(speed),
            float(offset_x), float(offset_y), self._samples(width, height))

    @staticmethod
    def _samples(width: int, height: int) -> int:
        """Two samples a side on a small frame, one on a large one.

        A star is a sub-pixel point, so it aliases worse than anything the
        other patterns draw -- but supersampling a big frame costs four
        times as much for a backdrop nobody is looking straight at.
        """
        return 2 if width * height <= 320_000 else 1
