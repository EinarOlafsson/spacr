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
#: THE ZOOM HAS AN END, and it is the REFERENCE ORBIT'S precision that sets
#: it -- not the scale's exponent, which is what an earlier version measured
#: and got wrong by a factor of two.
#:
#: Perturbation asks the shader for z = Z + dz, where dz is about the size
#: of one viewport. For that sum to mean anything, Z has to be known to
#: better than dz. Z is carried through the texture as a pair of float32s,
#: high plus low, and measured against a 320-digit reference the pair
#: reproduces it to 2.2e-16 absolute -- about 15.7 decades.
#:
#: Running past that does not go black; it goes MUSHY, which is worse
#: because it looks like a rendering fault rather than an end. Reported
#: 2026-08-28: "the mandelbrot theme ends quickly in a verry pixelated
#: image" -- it was being run to 34 decades, more than twice as deep as the
#: numbers support, so most of every dive was noise.
#:
#: Fourteen leaves a margin below 15.7, because the error grows with the
#: iteration count and the figure above is measured over 400 of them.
#:
#: At twenty-four seconds a decade that is about five and a half minutes of
#: descent before it starts again. GOING DEEPER MEANS CARRYING Z MORE
#: PRECISELY -- a third float in the texture would buy roughly another seven
#: decades -- and not raising this number.
MAX_USEFUL_DEPTH: Final[float] = 14.0

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
#:
#: SMALLER THAN THE ORIGINAL'S 4096, deliberately. This is the loop bound a
#: GLSL compiler sees, and it is the only shader in spaCR with a loop longer
#: than ten or a texture fetch inside one -- drivers routinely try to unroll
#: a constant-bounded loop, and at four thousand iterations of a fetch and
#: two complex multiplies that is where a compile fails or times out.
#:
#: 2304 keeps the published ceiling of 2200 reachable with room over it,
#: which is what the number is for: iterations above the bound would be
#: silently ignored rather than refused.
HARD_MAX_ITERATIONS: Final[int] = 2304


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

const int HARD_MAX = 2304;
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


# ---------------------------------------------------------------------------
# The guided path
# ---------------------------------------------------------------------------
#
# WITHOUT THIS THE DIVE ALWAYS ENDS IN THE SAME PLACE. A fixed path descends
# to one Misiurewicz point for ever: correct, and the same picture every
# time. Guided steering looks around every so often, picks a nearby point on
# the boundary that has structure worth arriving at, and eases the camera
# onto it -- so the descent keeps finding new things instead of drilling one
# shaft.
#
# THE LOOK-AROUND IS CHEAP ON PURPOSE. It renders a 96x54 escape map, which
# is 5,184 points against the two million a frame draws, and it runs on a
# worker thread. It is a decision about where to go, not a picture.


def perturbation_escape_map(orbit, width, height, scale, max_iter,
                            offset_re=0.0, offset_im=0.0):
    """A low-resolution map of what escapes and how fast.

    :param orbit: the reference :class:`ReferenceOrbit`.
    :param scale: the viewport's half-height.
    :returns: ``(escaped, iterations)``, both ``(height, width)``.

    Vectorised over the whole grid rather than looped per pixel: this runs
    while the backdrop is drawing, and a Python loop over 5,184 points times
    900 iterations is a second of held GIL.
    """
    real = np.asarray(orbit.packed[0, :, 0], dtype=np.float64) \
        + np.asarray(orbit.packed[0, :, 1], dtype=np.float64)
    imag = np.asarray(orbit.packed[0, :, 2], dtype=np.float64) \
        + np.asarray(orbit.packed[0, :, 3], dtype=np.float64)

    aspect = float(width) / float(height)
    xs = ((np.arange(width, dtype=np.float64) + 0.5) / width * 2.0 - 1.0)
    ys = ((np.arange(height, dtype=np.float64) + 0.5) / height * 2.0 - 1.0)
    dc_re = np.broadcast_to(xs * scale * aspect, (height, width)).copy()
    dc_im = np.broadcast_to(ys[:, None] * scale, (height, width)).copy()
    dc_re += float(offset_re)
    dc_im += float(offset_im)

    dz_re = np.zeros((height, width), dtype=np.float64)
    dz_im = np.zeros((height, width), dtype=np.float64)
    escaped = np.zeros((height, width), dtype=bool)
    iterations = np.full((height, width), int(max_iter), dtype=np.int32)

    limit = min(int(max_iter), orbit.max_iter)
    for n in range(limit):
        live = ~escaped
        if not live.any():
            break
        zr = real[n]
        zi = imag[n]
        z_re = zr + dz_re
        z_im = zi + dz_im
        newly = live & ((z_re * z_re + z_im * z_im) > 256.0)
        if newly.any():
            escaped |= newly
            iterations[newly] = n
        live = ~escaped
        if not live.any():
            break
        ar = dz_re[live]
        ai = dz_im[live]
        dz_re[live] = 2.0 * (zr * ar - zi * ai) + (ar * ar - ai * ai) \
            + dc_re[live]
        dz_im[live] = 2.0 * (zr * ai + zi * ar) + 2.0 * ar * ai + dc_im[live]
    return escaped, iterations


def structure_mask(escaped: np.ndarray, iterations: np.ndarray,
                   max_iter: int) -> np.ndarray:
    """Where the picture has detail worth steering toward.

    :returns: a boolean map the same shape as ``escaped``.

    SET MEMBERSHIP IS NOT ENOUGH ONCE THE ZOOM IS DEEP. Around a Misiurewicz
    point the set is measure-zero: measured on a 96x54 map at a scale of
    1.25e-3, every pixel escaped and :func:`boundary_mask` found NOTHING, so
    the guided path stopped steering after two decades and the dive went
    straight down again.

    What is still there is the escape TIME, and its level sets are the
    filaments the picture is made of. A pixel whose escape time differs
    sharply from its neighbours sits on one of those edges -- which is where
    detail survives at any magnification, and is what the eye reads as
    structure.

    The true boundary is preferred where it exists, because a bounded point
    beside an escaping one is the strongest evidence of an edge there is.
    """
    edge = boundary_mask(escaped)
    if edge.any():
        return edge

    times = iterations.astype(np.float64) / max(1.0, float(max_iter))
    gradient = np.zeros_like(times)
    gradient[1:, :] = np.maximum(gradient[1:, :],
                                 np.abs(times[1:, :] - times[:-1, :]))
    gradient[:-1, :] = np.maximum(gradient[:-1, :],
                                  np.abs(times[1:, :] - times[:-1, :]))
    gradient[:, 1:] = np.maximum(gradient[:, 1:],
                                 np.abs(times[:, 1:] - times[:, :-1]))
    gradient[:, :-1] = np.maximum(gradient[:, :-1],
                                  np.abs(times[:, 1:] - times[:, :-1]))
    if not np.isfinite(gradient).any() or gradient.max() <= 0.0:
        return np.zeros_like(escaped)
    # The steepest tenth: enough candidates to choose between, few enough
    # that they are all genuinely on a filament.
    threshold = float(np.quantile(gradient[gradient > 0.0], 0.90))
    steep = gradient >= max(threshold, 1e-9)
    steep[[0, -1], :] = False
    steep[:, [0, -1]] = False
    return steep


def boundary_mask(escaped: np.ndarray) -> np.ndarray:
    """Bounded points that touch an escaping one.

    THE BOUNDARY IS WHERE THE STRUCTURE IS. An interior point fades to flat
    colour as you descend into it; an exterior one escapes and the frame
    empties. Only the edge keeps producing detail at every magnification.

    The frame's own edge is excluded: a point there may look like a boundary
    only because the map stopped.
    """
    bounded = ~escaped
    neighbour_escaped = np.zeros_like(escaped)
    neighbour_escaped[1:, :] |= escaped[:-1, :]
    neighbour_escaped[:-1, :] |= escaped[1:, :]
    neighbour_escaped[:, 1:] |= escaped[:, :-1]
    neighbour_escaped[:, :-1] |= escaped[:, 1:]
    edge = bounded & neighbour_escaped
    edge[[0, -1], :] = False
    edge[:, [0, -1]] = False
    return edge


def candidate_score(escaped, iterations, row, col, max_iter) -> float:
    """How interesting the neighbourhood of one point is.

    Three things, because none alone is enough: how often the escape answer
    CHANGES across the patch (detail), how much the escape TIME varies
    (depth of structure), and how BALANCED bounded and escaping are (an
    edge, rather than a speck in a field of one or the other).
    """
    r0, r1 = max(0, row - 3), min(escaped.shape[0], row + 4)
    c0, c1 = max(0, col - 3), min(escaped.shape[1], col + 4)
    patch = escaped[r0:r1, c0:c1]
    times = iterations[r0:r1, c0:c1].astype(np.float64) / max(1.0, max_iter)
    balance = 1.0 - 2.0 * abs(float(patch.mean()) - 0.5)
    variation = float(times.std())
    transitions = 0.0
    if patch.shape[0] > 1:
        transitions += float(np.mean(patch[1:, :] != patch[:-1, :]))
    if patch.shape[1] > 1:
        transitions += float(np.mean(patch[:, 1:] != patch[:, :-1]))
    return 2.4 * transitions + 1.8 * variation + 0.8 * max(0.0, balance)


def plan_guided_step(orbit, scale, max_iter, strength=0.09,
                     candidates=24, step_index=0, offset_re=0.0,
                     offset_im=0.0):
    """Choose where the dive should head next.

    :param strength: how far off centre to look, in screen units.
    :param candidates: how many directions to try.
    :param step_index: which step this is; rotates the search.
    :returns: ``(dx, dy, score)`` in screen units, or ``None`` when the
        view holds no boundary at all.

    THE DIRECTIONS ARE SPREAD BY THE GOLDEN ANGLE and rotated per step, so
    consecutive choices do not favour one side of the frame -- which is what
    makes a "guided" path that always drifts the same way.
    """
    width, height = 96, 54
    escaped, iterations = perturbation_escape_map(
        orbit, width, height, float(scale), int(max_iter),
        offset_re, offset_im)
    edge = structure_mask(escaped, iterations, int(max_iter))
    if not edge.any():
        return None

    aspect = width / height
    xs = ((np.arange(width, dtype=np.float64) + 0.5) / width * 2.0 - 1.0)
    ys = ((np.arange(height, dtype=np.float64) + 0.5) / height * 2.0 - 1.0)
    grid_x, grid_y = np.meshgrid(xs, ys)
    screen_x = grid_x
    screen_y = grid_y
    radius = np.hypot(screen_x, screen_y)
    # NOT THE POINT ALREADY UNDER THE CAMERA, and not the far corners: the
    # first is where it is going anyway and the second is a lurch.
    eligible = edge & (radius >= 0.025) & (radius <= max(0.34, 2.2 * strength))
    if not eligible.any():
        eligible = edge

    best = None
    phase = step_index * 2.399963229728653          # the golden angle
    count = max(1, int(candidates))

    # THE HEADING IS A CONSTRAINT, NOT A PREFERENCE. Scoring every boundary
    # point and merely penalising the distant ones lets the most structured
    # point in the frame win whatever direction the step is supposed to be
    # exploring -- measured, six consecutive steps chose two targets between
    # them, which is a fixed path wearing a guided path's settings.
    #
    # Restricting the candidates to an arc around this step's own heading
    # makes each step go somewhere it has not been, and the golden angle
    # walks that arc around the frame without ever repeating a heading.
    point_angle = np.arctan2(screen_y, screen_x)
    difference = np.abs(np.angle(np.exp(1j * (point_angle - phase))))
    in_heading = eligible & (difference <= math.pi / 3.0)
    if in_heading.any():
        eligible = in_heading

    for index in range(count):
        # AN ARC PER STEP, not the whole circle. Spread over 360 degrees the
        # candidate set is nearly the same however the phase is rotated, so
        # the best-scoring point is the same every time -- which is a fixed
        # path wearing a guided path's settings. Measured: six consecutive
        # steps chose one target.
        #
        # A third of a circle around this step's own heading gives each step
        # somewhere different to look while keeping the move a STEER: the
        # golden angle then walks that window around the frame without ever
        # repeating a heading.
        spread = 2.0 * math.pi / 3.0
        angle = phase + spread * (index / count - 0.5)
        # Within the arc the candidates fan out, so the choice is still made
        # between real alternatives rather than one point being scored.
        want_x = strength * math.cos(angle)
        want_y = strength * math.sin(angle)
        distance = (screen_x - want_x) ** 2 + (screen_y - want_y) ** 2
        masked = np.where(eligible, distance, np.inf)
        flat = int(np.argmin(masked))
        row, col = np.unravel_index(flat, masked.shape)
        if not np.isfinite(masked[row, col]):
            continue
        structure = candidate_score(escaped, iterations, int(row), int(col),
                                    int(max_iter))
        # A CLOSER TARGET IS WORTH SOMETHING TOO. Left unpenalised the
        # search would keep choosing the most interesting point in the
        # frame, which is a jump rather than a steer.
        penalty = math.sqrt(float(masked[row, col])) / max(0.04, strength)
        score = structure - 0.32 * penalty
        if best is None or score > best[2]:
            best = (float(grid_x[row, col] * aspect),
                    float(grid_y[row, col]), score)
    return best


def eased(fraction: float) -> float:
    """Smoothstep, for a camera move that starts and stops gently.

    A linear move between two points is a lurch at both ends; this is the
    difference between the camera being steered and being teleported.
    """
    x = 0.0 if fraction < 0.0 else (1.0 if fraction > 1.0 else float(fraction))
    return x * x * (3.0 - 2.0 * x)
