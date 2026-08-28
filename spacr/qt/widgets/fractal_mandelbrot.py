"""A continuous deep zoom into one point on the Mandelbrot boundary.

The fourth spaceout pattern. The viewport is recomputed every frame -- no
previous frame is reused as image data -- and the only camera motion is zoom.

WHY IT NEEDS MORE THAN A SHADER. Past about fifteen decades a double has no
bits left to tell neighbouring pixels apart, and the picture dissolves into
blocks. This uses PERTURBATION around one high-precision reference orbit:

    dz[n+1] = 2*Z[n]*dz[n] + dz[n]^2 + dc

``Z`` is iterated once in arbitrary precision and handed to the renderer;
every pixel then iterates only its small OFFSET from it, in the precision the
hardware actually has. That is what buys hundreds of decades from float32.

THE TARGET IS A MISIUREWICZ POINT, preperiod 4 period 1, refined at startup
by solving ``f_c^5(0) = f_c^4(0)``. It has to be ON the boundary: an interior
point fades to flat colour as you descend into it and an exterior one escapes,
and either way the zoom stops finding anything. A boundary point keeps
revealing structure at any magnification.
"""
from __future__ import annotations

import math
from typing import Final, Optional

import numpy as np

#: How deep the dive goes before it starts again.
#:
#: THE ZOOM HAS AN END, and past it the screen goes black. Perturbation
#: buys precision for the CENTRE, but the per-pixel offset is still a
#: float32 in the shader, and that is what runs out. Measured, at the
#: default starting scale of 1.25 and a 1080-tall window:
#:
#:     depth 34: pixel step 2.3e-37   fine
#:     depth 38: scale is denormal    losing bits
#:     depth 45: pixel step is ZERO   every pixel samples one point
#:
#: A zero pixel step means the whole frame is one sample of one point, which
#: is the black screen. Thirty-four is chosen before the denormals rather
#: than at the cliff, because precision degrades through that range rather
#: than failing at a line -- the picture goes mushy before it goes black.
#:
#: At the default twenty-four seconds a decade that is about fourteen
#: minutes of descent before it begins again.
MAX_USEFUL_DEPTH: Final[float] = 34.0

#: The published defaults, as asked for on 2026-08-28.
DEFAULTS: Final[dict] = {
    "supersampling": 2,
    "render_scale": 1.0,
    "fps": 30,
    "zoom_rate": 1.0,
    "seconds_per_decade": 24.0,
    "base_iterations": 300,
    "iterations_per_decade": 55.0,
    "max_iterations": 2200,
    "precision_digits": 320,
    "initial_scale": 1.25,
    "tile_rows": 32,
    # FP64 OFF DELIBERATELY. The double-precision shader needs GLSL 400, and
    # many drivers either lack it or emulate it at a cost far larger than the
    # precision is worth here -- perturbation is what buys the depth, not the
    # shader's float width. Float32 plus perturbation runs everywhere.
    "gpu_fp64": False,
    # GUIDED, not fixed: every 0.40 decades it looks for a nearby bounded
    # boundary point and eases the camera onto it, so the zoom keeps finding
    # new structure instead of descending one shaft forever.
    "path": "guided",
    "steering_strength": 0.09,
    "steering_interval_decades": 0.40,
    "steering_duration": 3.8,
    "candidate_count": 24,
    "max_depth": MAX_USEFUL_DEPTH,
}

#: A boundary point to start from, refined at startup.
MISIUREWICZ_GUESS_REAL: Final[str] = "-0.10109636384562"
MISIUREWICZ_GUESS_IMAG: Final[str] = "0.95628651080914"

#: The most iterations any shader will run. Bounds the orbit texture.
HARD_MAX_ITERATIONS: Final[int] = 4096


FRAGMENT_SHADER: Final[str] = r"""
uniform sampler2D u_orbit;
uniform vec2 u_resolution;
uniform float u_scale;
uniform vec2 u_center_offset;
uniform float u_depth;
uniform float u_orbit_length;
uniform int u_max_iter;
uniform float u_pointer_x;
uniform float u_pointer_y;
uniform float u_pull;
uniform float u_push;

const int HARD_MAX = 4096;
const float ESCAPE2 = 256.0;

vec2 cmul(vec2 a, vec2 b) {
    return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// THE REFERENCE ORBIT, split high/low across two channel pairs. A float32
// texture cannot hold Z to the precision the deep zoom needs, so each
// component is stored as a sum of two floats and added back here.
vec2 refz(int n) {
    float u = (float(n) + 0.5) / max(1.0, u_orbit_length);
    vec4 p = texture2D(u_orbit, vec2(u, 0.5));
    return vec2(p.r + p.g, p.b + p.a);
}

vec3 palette(float x) {
    vec3 a = vec3(0.30, 0.22, 0.30);
    vec3 b = vec3(0.62, 0.58, 0.68);
    vec3 c = vec3(0.86, 0.75, 0.94);
    vec3 d = vec3(0.00, 0.30, 0.65);
    return a + b * cos(6.28318530718 * (c * x + d));
}

// The pointer moves the point the zoom descends into, so the mouse steers
// the dive rather than smearing the picture.
vec2 toward_pointer(vec2 q) {
    vec2 target = vec2(u_pointer_x, u_pointer_y);
    return q - target * (u_pull - 0.85 * u_push);
}

vec3 sample_mandel(vec2 pixel) {
    float aspect = u_resolution.x / max(1.0, u_resolution.y);
    vec2 q = vec2(
        (pixel.x / u_resolution.x) * 2.0 - 1.0,
        (pixel.y / u_resolution.y) * 2.0 - 1.0
    );
    q = toward_pointer(q);
    q.x *= aspect;
    vec2 dc = u_center_offset + q * u_scale;
    vec2 dz = vec2(0.0);
    float trap = 1e30;
    float mu = 0.0;
    bool escaped = false;

    for (int n = 0; n < HARD_MAX; ++n) {
        if (n >= u_max_iter) break;
        vec2 Z = refz(n);
        vec2 z = Z + dz;
        float m2 = dot(z, z);
        trap = min(trap, abs(z.x * z.y));
        if (m2 > ESCAPE2) {
            mu = float(n) + 1.0 - log(log(sqrt(m2))) / log(2.0);
            escaped = true;
            break;
        }
        dz = 2.0 * cmul(Z, dz) + cmul(dz, dz) + dc;
    }

    if (!escaped) return vec3(0.003, 0.004, 0.010);
    float tg = exp(-25.0 * min(trap, 0.18));
    float phase = 0.025 * mu + 0.014 * u_depth + 0.052 * tg;
    vec3 col = palette(phase) + tg * vec3(0.16, 0.09, 0.22);
    return pow(clamp(col, 0.0, 1.0), vec3(0.93));
}

vec3 render_sample(vec2 fragment_position) {
    return sample_mandel(fragment_position);
}

void main() {
    vec3 col = vec3(0.0);
    col += sample_mandel(gl_FragCoord.xy + vec2(-0.25, -0.25));
    col += sample_mandel(gl_FragCoord.xy + vec2( 0.25, -0.25));
    col += sample_mandel(gl_FragCoord.xy + vec2(-0.25,  0.25));
    col += sample_mandel(gl_FragCoord.xy + vec2( 0.25,  0.25));
    gl_FragColor = vec4(0.25 * col, 1.0);
}
"""


def exact_misiurewicz_center(digits: int = 320):
    """Refine the boundary target to ``digits`` decimal places.

    :param digits: working precision for the solve.
    :returns: an ``mpmath.mpc`` on the Mandelbrot boundary.
    :raises RuntimeError: when mpmath is not installed.

    Solves ``f_c^5(0) = f_c^4(0)``, which is the defining equation of a
    Misiurewicz point of preperiod 4 and period 1. Newton is given two
    nearby starting points rather than one, because the derivative of a
    fifth iterate is stiff enough that a single-point secant wanders.
    """
    try:
        import mpmath as mp
    except Exception as error:                               # noqa: BLE001
        raise RuntimeError("the Mandelbrot pattern needs mpmath") from error

    def _iterate(c, n):
        z = mp.mpc(0)
        for _ in range(n):
            z = z * z + c
        return z

    mp.mp.dps = int(digits)
    guess = mp.mpc(MISIUREWICZ_GUESS_REAL, MISIUREWICZ_GUESS_IMAG)
    delta = mp.mpc(mp.mpf("1e-8"), mp.mpf("1e-8"))
    return mp.findroot(
        lambda c: _iterate(c, 5) - _iterate(c, 4),
        (guess, guess + delta),
        tol=mp.power(10, -(max(60, int(digits) - 30))),
        maxsteps=100,
    )


class ReferenceOrbit:
    """``Z[n]`` for one centre, in a form both renderers can use.

    :param max_iter: how many points to iterate.
    :param digits: working precision.
    :param center: the centre; refined from the Misiurewicz guess when
        omitted.

    BUILD IT OFF THE GUI THREAD. Iterating a few thousand points at 320
    decimal digits takes seconds, and the backdrop has to keep drawing while
    it happens.
    """

    def __init__(self, max_iter: int = 2200, digits: int = 320,
                 center=None) -> None:
        import mpmath as mp

        self.max_iter = max(1, min(int(max_iter), HARD_MAX_ITERATIONS))
        self.digits = int(digits)
        mp.mp.dps = self.digits
        self.center = (exact_misiurewicz_center(self.digits)
                       if center is None else mp.mpc(center))
        self.escaped_at: Optional[int] = None
        self.packed = np.zeros((1, self.max_iter + 1, 4), dtype=np.float32)
        self._build()

    def _build(self) -> None:
        import mpmath as mp

        mp.mp.dps = self.digits
        z = mp.mpc(0)
        for n in range(self.max_iter + 1):
            real, imag = mp.re(z), mp.im(z)
            # HIGH AND LOW, because one float32 cannot hold Z at this depth.
            # The shader adds the pair back; the residual is what a single
            # float would have thrown away.
            re_hi = np.float32(float(real))
            im_hi = np.float32(float(imag))
            re_lo = np.float32(float(real - mp.mpf(float(re_hi))))
            im_lo = np.float32(float(imag - mp.mpf(float(im_hi))))
            self.packed[0, n] = (re_hi, re_lo, im_hi, im_lo)
            z = z * z + self.center
            if abs(z) > mp.mpf("256"):
                # A REFERENCE THAT ESCAPES IS NOT A REFERENCE. Every pixel
                # perturbs around it, so the rest is zeroed rather than left
                # holding numbers that mean nothing.
                self.escaped_at = n + 1
                self.packed[0, n + 1:] = 0.0
                break

    @property
    def is_bounded(self) -> bool:
        """Whether the orbit stayed bounded for its whole length."""
        return self.escaped_at is None


def depth_decades(seconds: float, zoom_rate: float = 1.0,
                  seconds_per_decade: float = 24.0) -> float:
    """How many decades of magnification ``seconds`` of flight is worth."""
    return max(0.0, float(seconds) * float(zoom_rate)
               / max(1e-6, float(seconds_per_decade)))


def iteration_budget(depth: float, base: int = 300,
                     per_decade: float = 55.0, ceiling: int = 2200) -> int:
    """How many iterations a given depth needs.

    :returns: at least ``base`` and at most ``ceiling``.

    DEEPER NEEDS MORE. Near the boundary the escape time grows with
    magnification, so a fixed budget draws the deep frames as solid interior
    -- the picture stops changing and looks broken rather than deep.
    """
    wanted = int(base) + int(round(float(per_decade) * float(depth)))
    return max(int(base), min(int(ceiling), wanted))


def depth_after_restart(depth: float,
                        max_depth: float = MAX_USEFUL_DEPTH) -> float:
    """Where the dive is, having started again if it reached the end.

    :param depth: decades descended so far.
    :param max_depth: how deep it may go; see :data:`MAX_USEFUL_DEPTH`.
    :returns: a depth within range.

    A RESTART, NOT A STOP. The alternative is a backdrop that spends
    fourteen minutes getting somewhere and then holds a black frame for the
    rest of the session, which reads as the application having died. Going
    back to the surface and descending again is what the pattern is for.

    Wrapped with a modulo rather than reset to zero on a comparison, so a
    frame that arrives late -- the machine was asleep, or a run took the
    CPU -- lands where it should instead of skipping a whole descent.
    """
    limit = max(0.1, float(max_depth))
    return float(depth) % limit


def scale_at(depth: float, initial_scale: float = 1.25) -> float:
    """The viewport's half-height at ``depth`` decades.

    Clamped at 307 decades, which is where a float64 underflows -- past it
    the scale would silently become zero and every pixel would sample the
    same point.
    """
    return float(initial_scale) * 10.0 ** (-min(float(depth), 307.0))
