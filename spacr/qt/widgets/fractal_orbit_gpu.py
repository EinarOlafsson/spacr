"""The orbit fold on the GPU: the fifth spaceout pattern.

Instruction 327 (5). The CPU orbit fold in
:mod:`spacr.qt.widgets.fractal_travel` is the one renderer the maintainer
singled out as right -- "the orbit fold cpu effect is like a magnigying
glass, which looks cool" -- and it is the one the pointer complaint
excluded. This is that same map, on the GPU.

WHY IT IS A SECOND ENTRY RATHER THAN A BACKEND SWITCH. The CPU path
jitters four samples ACROSS four frames, which averages four different
animation times; a GPU pass can afford the four samples inside one
instant. They are not the same picture, so offering them as one option
with a hidden backend would mean the same setting drawing two things.
"Orbit fold" is taken, so this is "Orbit fold (sharp)" -- descriptive of
what actually differs rather than a version number.

THE MAP IS TRANSLITERATED, not reinvented: same fold, same inversion
constants, same three orbit traps, same palette. A test asserts the
constants match the CPU function character for character, because two
renderers that drift apart are two patterns wearing one name.
"""
from __future__ import annotations

from typing import Final

#: What the pattern is called in Preferences and in the README.
PATTERN_KEY: Final[str] = "orbit_gpu"
PATTERN_LABEL: Final[str] = "Orbit fold (sharp, GPU 2x2)"

#: Fold iterations per sample. The CPU path takes this from the quality
#: setting; on the GPU the cost is a loop bound the compiler unrolls, so
#: it is fixed at what the CPU calls "high".
ITERATIONS: Final[int] = 24

FRAGMENT_SHADER: Final[str] = """
#version 120

uniform vec2 u_resolution;
uniform float u_time;
uniform float u_speed;
uniform float u_dream;
uniform float u_pointer_x;
uniform float u_pointer_y;
uniform float u_pull;
uniform float u_push;

// THE POINTER BENDS THE PLANE, IT DOES NOT MOVE THE CAMERA. Identical to
// every other pattern since instruction 327 (4): displacement toward the
// pointer with a 1/r^2 strength, so it is firm under the cursor and gone
// by the far corner, and nothing is displaced globally to spring back.
vec2 toward_pointer(vec2 uv) {
    vec2 target = vec2(u_pointer_x, u_pointer_y);
    vec2 to_pointer = target - uv;
    float distance2 = dot(to_pointer, to_pointer) + 0.05;
    float strength = (0.55 * u_pull - 0.95 * u_push) / distance2;
    strength = clamp(strength, -1.4, 0.9);
    return uv + strength * to_pointer;
}

vec3 orbit_sample(vec2 fragment_position) {
    float denominator = min(u_resolution.x, u_resolution.y);
    vec2 p = (2.0 * fragment_position - u_resolution) / denominator;
    float screen_radius = length(p);
    p = toward_pointer(p);

    float t = u_time;

    float rotation = 0.24 * sin(0.17 * t) + 0.11 * sin(0.043 * t + 1.2);
    float cs = cos(rotation);
    float sn = sin(rotation);
    vec2 tp = vec2(cs * p.x - sn * p.y, sn * p.x + cs * p.y);

    vec2 drift = u_dream * vec2(
        0.10 * sin(0.071 * t) + 0.04 * sin(0.019 * t + 1.3),
        0.09 * cos(0.063 * t + 0.4) + 0.04 * sin(0.023 * t + 2.1));
    float stretch_x = exp(0.10 * u_dream * sin(0.041 * t));
    float stretch_y = exp(0.09 * u_dream * cos(0.037 * t + 0.8));
    float shear_x = 0.12 * u_dream * sin(0.052 * t + 0.6);
    float shear_y = 0.07 * u_dream * cos(0.047 * t);

    float old_x = tp.x;
    tp = vec2(stretch_x * tp.x + shear_x * tp.y + drift.x,
              stretch_y * tp.y + shear_y * old_x + drift.y);

    float radius_squared = dot(tp, tp) + 1e-4;
    float inverse_radius = inversesqrt(radius_squared);
    float radial_phase = 0.80 * sin(
        5.5 * log(radius_squared + 0.03) + 0.42 * t * u_speed);
    tp += 0.10 * u_dream * radial_phase * tp * inverse_radius;

    float constant_x = 0.73 + 0.08 * sin(0.11 * t) + 0.05 * sin(0.031 * t + 2.0);
    float constant_y = 0.48 + 0.10 * cos(0.13 * t + 0.7) + 0.04 * sin(0.037 * t);

    float orbit_a = 0.0;
    float orbit_b = 0.0;
    float orbit_c = 0.0;
    float previous_radius = 1e9;
    vec2 o = tp;

    for (int iteration = 0; iteration < ITERATIONS; iteration++) {
        o = abs(o);
        if (o.x < o.y) { o = o.yx; }
        o.x = abs(o.x - 0.45 * o.y);
        float current_radius = dot(o, o) + 0.055;
        o = o / current_radius - vec2(constant_x, constant_y);

        float radius_change = abs(current_radius - previous_radius);
        previous_radius = current_radius;
        orbit_a += 1.0 / (1.0 + 12.0 * abs(current_radius - 0.42));
        orbit_b += 1.0 / (1.0 + 9.0 * abs(o.x - o.y));
        orbit_c += 1.0 / (1.0 + 18.0 * radius_change);

        float step = float(iteration);
        o = vec2(o.x + 0.035 * u_dream * sin(1.7 * o.y + 0.19 * t + step),
                 o.y + 0.035 * u_dream * cos(1.5 * o.x - 0.17 * t - step));
    }

    float inverse_iterations = 1.0 / float(ITERATIONS);
    orbit_a *= inverse_iterations;
    orbit_b *= inverse_iterations;
    orbit_c *= inverse_iterations;

    // Three orbit traps drive the phase rather than one escape count,
    // which is what keeps the colour moving where a Mandelbrot would band.
    float palette_phase = 5.2 * orbit_a + 3.7 * orbit_b + 2.3 * orbit_c
                        + 0.075 * t;
    vec3 colour = vec3(
        0.50 + 0.43 * cos(palette_phase + 0.15) + 0.12 * orbit_c,
        0.48 + 0.42 * cos(palette_phase + 2.25) + 0.11 * orbit_a,
        0.50 + 0.45 * cos(palette_phase + 4.35) + 0.13 * orbit_b);

    float glow = clamp(1.4 * orbit_a * orbit_b, 0.0, 1.0);
    colour += vec3(0.15, 0.10, 0.24) * glow;

    // The vignette is what lets controls sit on top and stay readable.
    float vignette = 1.0 - clamp((screen_radius - 0.55) / 1.30, 0.0, 1.0);
    return clamp(colour, 0.0, 1.0) * (0.78 + 0.22 * vignette);
}

void main() {
    // TRUE SPATIAL 2x2, four samples of ONE instant. The CPU path jitters
    // across four FRAMES instead, which is cheaper and blends four
    // different animation times -- that difference is why this is a
    // separate entry rather than a backend switch on the same one.
    vec2 base = gl_FragCoord.xy;
    vec3 total = orbit_sample(base + vec2(0.25, 0.25))
               + orbit_sample(base + vec2(0.75, 0.25))
               + orbit_sample(base + vec2(0.25, 0.75))
               + orbit_sample(base + vec2(0.75, 0.75));
    gl_FragColor = vec4(total * 0.25, 1.0);
}
""".replace("ITERATIONS", str(ITERATIONS))
