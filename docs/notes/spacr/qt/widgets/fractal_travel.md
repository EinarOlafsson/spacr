# Notes from `spacr/qt/widgets/fractal_travel.py`

Prose lifted out of `spacr/qt/widgets/fractal_travel.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (4 entries)
- [Pointer.sample](#pointersample) (4 entries)
- [RuntimeControls.speed_at](#runtimecontrolsspeed_at) (1 entry)
- [resolved_quality](#resolved_quality) (2 entries)
- [platform_can_do_opengl](#platform_can_do_opengl) (2 entries)
- [_orbit_sample](#_orbit_sample) (4 entries)
- [OrbitEngine.render](#orbitenginerender) (1 entry)
- [_quit_and_join_thread](#_quit_and_join_thread) (2 entries)
- [_make_cpu_widget](#_make_cpu_widget) (4 entries)
- [_make_cpu_widget._Worker._say_something](#_make_cpu_widget_worker_say_something) (1 entry)
- [_make_cpu_widget.CpuFractalWidget.__init__](#_make_cpu_widgetcpufractalwidget__init__) (4 entries)
- [_make_cpu_widget.CpuFractalWidget._target_size](#_make_cpu_widgetcpufractalwidget_target_size) (2 entries)
- [_make_cpu_widget.CpuFractalWidget._request_frame](#_make_cpu_widgetcpufractalwidget_request_frame) (2 entries)
- [_make_cpu_widget.CpuFractalWidget._adapt_resolution](#_make_cpu_widgetcpufractalwidget_adapt_resolution) (1 entry)
- [_make_cpu_widget.CpuFractalWidget._accept_frame](#_make_cpu_widgetcpufractalwidget_accept_frame) (1 entry)
- [target_render_size](#target_render_size) (1 entry)
- [RegionTour.target_at](#regiontourtarget_at) (1 entry)
- [_TourPilot.__init__](#_tourpilot__init__) (1 entry)
- [_make_gpu_widget](#_make_gpu_widget) (5 entries)
- [_make_gpu_widget._Canvas.__init__](#_make_gpu_widget_canvas__init__) (5 entries)
- [_make_gpu_widget._Canvas._update_uniforms](#_make_gpu_widget_canvas_update_uniforms) (5 entries)
- [_make_gpu_widget._Canvas._start_the_reference_orbit._work](#_make_gpu_widget_canvas_start_the_reference_orbit_work) (1 entry)
- [_make_gpu_widget._Canvas._mandelbrot_uniforms](#_make_gpu_widget_canvas_mandelbrot_uniforms) (6 entries)
- [_make_gpu_widget._Canvas._steer](#_make_gpu_widget_canvas_steer) (6 entries)
- [_make_gpu_widget._Canvas._refine_the_reference](#_make_gpu_widget_canvas_refine_the_reference) (1 entry)
- [_make_gpu_widget._Canvas._upload_the_orbit_if_it_arrived](#_make_gpu_widget_canvas_upload_the_orbit_if_it_arrived) (2 entries)
- [_make_gpu_widget._Canvas._pointer_state](#_make_gpu_widget_canvas_pointer_state) (1 entry)
- [_make_gpu_widget._Canvas.on_draw](#_make_gpu_widget_canvason_draw) (1 entry)
- [_make_gpu_widget._Canvas._on_timer](#_make_gpu_widget_canvas_on_timer) (2 entries)
- [_make_gpu_widget.GpuFractalWidget.__init__](#_make_gpu_widgetgpufractalwidget__init__) (1 entry)
- [nudge_zoom_rate](#nudge_zoom_rate) (3 entries)
- [create_fractal_widget](#create_fractal_widget) (6 entries)

## Module level

### lines 79-81

```python
from ..fractal_defaults import (GPU_ONLY_PATTERNS,  # noqa: E402
```

ONE LIST, in the Qt-free module, because `preferences` needs it at import time and cannot import this one. Two copies is how a pattern came to be selectable in code and absent from the Preferences combo.

### lines 86-90

```python
"orbit_gpu": "Orbit fold (sharp, GPU 2x2)",
```

A SECOND ENTRY, NOT A BACKEND SWITCH. Same map, but the CPU path jitters four samples across four FRAMES -- averaging four different animation times -- while this takes four samples of one instant. They are not the same picture, so one setting drawing either would mean the setting no longer says what appears. Instruction 327 (5).

### lines 488-490  _(unsure)_

```python
_FAST_PI: Final[float] = math.pi
```

CPU backend -- orbit-fold, evaluated at animation rate

### lines 1290-1292  _(unsure)_

```python
VERTEX_SHADER: Final[str] = """
```

GPU backend -- the GLSL field, unchanged from the script

## Pointer.sample

### lines 205-206

```python
self.pull = max(0.0, self.pull - 0.08)
```

Off the widget it neither pulls nor pushes, rather than pulling toward an edge it is nowhere near.

### lines 210-213

```python
distance = math.hypot(self.x, self.y)
```

SIZE IS A REACH, not a hard edge. Beyond it the pull falls to nothing smoothly, so a pointer crossing the boundary does not snap the picture. `size` is in the same -1..1 space as the coordinates, so 1.0 reaches the short edge of the widget.

### lines 219-221

```python
self.pull += 0.06 * (wanted_pull * float(strength) - self.pull)
```

EASED, not switched. A pull that appears the instant the pointer enters reads as a glitch; this reaches full strength over about a second and lets go about as fast.

### lines 225-232

```python
held = left or right
```

DRAG THE VIEW. Asked for 2026-08-28: "best would be if the user could drag the visual field with the mouse by clicking and mooving the mouse."

ACCUMULATED, not assigned: the renderer consumes this and zeroes it, so a frame dropped under load does not lose the movement -- it arrives with the next frame instead, and a slow machine pans by the same total distance as a fast one.

## RuntimeControls.speed_at

### line 327

```python
period = max(1.0, float(self.speed_period or _VARIABLE_SPEED_PERIOD))
```

A period of zero would divide by zero; a tiny one is a strobe.

## resolved_quality

### lines 372-380

```python
return "balanced"
```

CONSERVATIVE ON A GPU TOO. A first impression that stutters is worse than one that is merely plain, and this cannot know whether the card is a workstation's or a laptop's -- "high" here would have asked every machine ever made for four samples a pixel at native resolution before anyone chose it.

A GPU still gets more than a CPU: the per-backend budgets below give it a wider frame and a higher frame-rate cap at the same level, which is the headroom, without guessing at the hardware.

### lines 382-383

```python
return "high" if hardware.logical_cpus >= 16 else "balanced"
```

AND THE CPU IS ASKED FOR EVIDENCE FIRST. Sixteen cores is a machine that can spare some; anything less gets the light profile.

## platform_can_do_opengl

### lines 421-424

```python
if os.environ.get("SPACR_NO_GL"):
```

SAFE MODE REFUSES GL OUTRIGHT. `safespacr` exists because the crash log points here, and a safe start that still built a GL canvas would be no safer than the ordinary one. Read from the environment because the context can be created before any preference has been read.

### lines 433-445

```python
if not platform and sys.platform in ("darwin", "win32"):
```

AN EMPTY VALUE MEANS QT WILL CHOOSE, and what it chooses decides whether DISPLAY is the right question. On X11 and Wayland it is: no DISPLAY, no GL. On macOS and Windows it is not -- neither sets DISPLAY, both always have a window server, and Qt picks cocoa or windows without being told.

THIS TEST USED TO ASK DISPLAY REGARDLESS, so on every Mac -- where QT_QPA_PLATFORM is normally unset -- it answered no and the spaceout fractal ran its Numba CPU renderer instead of its shader. Measured on the reporting iMac: VisPy opens a context on that machine reporting GL_RENDERER "AMD Radeon Pro 5300 OpenGL Engine", so the card was there and working the whole time and the environment heuristic was what said otherwise.

## _orbit_sample

### lines 534-537

```python
if pull > 0.0 or push > 0.0:
```

THE POINTER BENDS THE PLANE, it does not move the camera. Warping the sample position pulls the STRUCTURE toward the cursor and leaves the travel alone, so the fractal still goes where it was going -- which a camera shove would not.

### line 542  _(unsure)_

```python
strength = (0.55 * pull - 0.95 * push) / distance2
```

1/r falloff: firm near the pointer, gone by the far corner.

### lines 618-619

```python
palette_phase = (5.2 * orbit_a + 3.7 * orbit_b + 2.3 * orbit_c
```

Three orbit traps drive the phase rather than one escape count, which is what keeps the colour moving where a Mandelbrot would band.

### line 631  _(unsure)_

```python
screen_radius = math.sqrt(x * x + y * y)
```

The vignette is what lets controls sit on top and stay readable.

## OrbitEngine.render

### lines 771-772

```python
for index in range(4):
```

Fill the unused history with the first real sample, so the opening frame is the picture rather than a fade up from black.

## _quit_and_join_thread

### lines 782-784

```python
def _quit_and_join_thread(thread) -> None:
```

The CPU widget -- a render thread, and a paint that never computes

### lines 801-802

```python
pass
```

Shutdown is also reached while Qt is tearing its own wrappers down. A wrapper that is already gone has no live thread left to join.

## _make_cpu_widget

### lines 857-861

```python
if settings.pattern == "space":
```

EACH PATTERN'S OWN BUDGET. The cascade evaluates four samples per pixel inside one frame where the orbit evaluates one and averages across frames, so it renders roughly a quarter of the pixels and holds a lower cap to spend the same wall-clock. Sharing one set of numbers would make one of them either wasteful or unusable.

### lines 866-868

```python
iterations = 0
```

MOSTLY EMPTY SKY IS CHEAP. Each pixel walks six layers of a 3x3 neighbourhood and three object slots, and almost every cell misses so it carries a wider frame and a higher cap than either fold.

### lines 882-884

```python
target_fps = max(15, min(settings.fps, 30))
```

Capped at 30 because every displayed frame is newly evaluated there is no keyframe to re-project, so a higher cap only burns cores.

### lines 886-887  _(unsure)_

```python
base_pixels = 460_000.0 if quality == "balanced" else 680_000.0
```

A pixel COUNT, not a percentage: on a 4K display a percentage is an unbounded promise, and this is a backdrop.

## _make_cpu_widget._Worker._say_something

### lines 953-954  _(unsure)_

```python
pass
```

"Signal source has been deleted" -- the widget went away while this frame was being shaded. There is nobody to tell.

## _make_cpu_widget.CpuFractalWidget.__init__

### lines 978-981

```python
self._thread = QThread()
```

NOT PARENTED TO self. A QThread whose parent is deleted while it runs prints "Destroyed while thread is still running" and takes the process down; the backdrop is reparented and deleted with its screen, so that is the ordinary path, not an edge case.

### lines 991-993

```python
_join_on_destroy(self, self._thread)
```

JOINED WHENEVER QT FREES THE WIDGET. `destroyed` fires during destruction, so the handler must hold the THREAD and never `self` -- reaching for a half-destroyed widget is its own crash.

### lines 995-1000

```python
self._app_quit_join = (
```

QApplication can tear down its native widgets before Python releases their wrappers.  In that order ``destroyed`` is too late to protect an unparented QThread, so join at the earlier, explicit application-shutdown boundary as well.  The closure captures only the thread; keeping it on ``self`` merely lets an explicit shutdown disconnect the now-unneeded application hook.

### lines 1029-1031

```python
self._depth_phase = DepthPhase()
```

Integrated travel, so a speed change does not teleport the camera. One per canvas: it is this canvas's own position on the trajectory.

## _make_cpu_widget.CpuFractalWidget._target_size

### lines 1080-1083

```python
"""The pixel size to shade at, from the render scale and the window.
```

RENDER SCALE, which was a setting nobody read. It is the fraction of the window's own pixels to shade, so 1.0 is native and anything less trades sharpness for speed -- the direct answer to "how do i get the image sharper".

### lines 1110-1112

```python
return target_render_size(
```

THE ARITHMETIC LIVES IN `target_render_size`, so instruction 327's "measure before you change anything" can be done without a GL context.

## _make_cpu_widget.CpuFractalWidget._request_frame

### lines 1128-1132

```python
pointer = self._pointer.sample(
```

SAMPLED ON THE GUI THREAD, sent to the worker as numbers. QCursor and QApplication are not safe to touch from the render thread, and by the time the frame is drawn the pointer has moved anyway -- so the position that matters is the one when the frame was ASKED for.

### lines 1135-1151

```python
speed = controls.speed_at(self._sim_time)
```

THE CLOCK CARRIES THE SPEED, and the kernel is handed 1.0. Every CPU pattern positions the picture with `t * speed` the orbit's radial phase, the cascade's depth, the flight's depth and its object travel -- so a scroll moved the picture by the whole elapsed time at once. Measured at t=57.3s, a change from speed 1 to 2 moved the orbit as far as a second of ordinary travel, the cascade as far as two, and the star flight further than thirty seconds' worth: 30 to 900 frames arriving between two frames. That is the reported jump.

`DepthPhase` integrates instead, so the change is continuous by construction: the phase is WHERE the picture is, and speed only changes how fast it leaves. This widget already held a DepthPhase for that reason and never asked it anything only the GPU canvas was wired up.

Passing the speed on as well would apply it twice.

## _make_cpu_widget.CpuFractalWidget._adapt_resolution

### lines 1180-1181

```python
budget = 0.78 * target_period
```

About 22% of the period is left for Qt's conversion, the scale and the rest of the application.

## _make_cpu_widget.CpuFractalWidget._accept_frame

### lines 1199-1200

```python
self._image_array = frame
```

The array is kept alive alongside the QImage: QImage does not copy, and a freed buffer paints garbage or crashes.

## target_render_size

### lines 1497-1514

```python
requested = native * render_scale * render_scale * adaptive_scale ** 2
```

MULTIPLIED IN, NOT REPLACED. This branch used to overwrite

`requested`, which threw the adaptive scale away -- and since `render_scale` defaults above zero, that was every launch.

`_adapt_resolution` measures the render time, compares it with the frame budget and computes a new scale between 0.58 and 1.35. All of that ran, and none of it reached the renderer. Measured: sweeping the adaptive scale across its whole range left the shaded size at 1280x720 every time.

That is the answer to instruction 327 (1). Fullscreen shades 5.12x a panel's pixels at 2560x1440 and 11.53x at 4K -- the cost IS the area -- but the machinery meant to compensate was disconnected, so the frame rate fell instead of the resolution.

The user's own `scale` still means what it says: 0.5 is half native when frames are comfortable. The adaptive term only ever takes it further down, or back up toward it.

## RegionTour.target_at

### lines 1633-1634  _(unsure)_

```python
position = float(seconds) % total
```

Modulo, so the tour is a loop and a long session does not run off the end of the list.

## _TourPilot.__init__

### lines 1682-1687

```python
self._floor_px = self._measure_the_floor()
```

COMPUTED ONCE, BECAUSE THIS IS NOW THE DEFAULT PATH. The floor is a property of the itinerary and the itinerary does not change while the pilot exists, so recomputing it per frame was twenty float conversions sixty times a second for an answer that never moves. Cheap either way; on the frame path of the backdrop every user now gets, "cheap" is not the standard.

## _make_gpu_widget

### lines 1851-1852

```python
vispy_app.use_app("pyside6")
```

PySIDE6, not pyqt6. Two Qt bindings in one process segfault, and vispy will happily import the other one if asked.

### lines 1862-1865

```python
base_detail = 6
```

ITERATIONS, NOT FOLD DEPTH. The adaptive loop turns `_detail` down when a frame runs long, and here that is the iteration budget which is also what the zoom needs MORE of as it descends, so the floor is high enough that a deep frame does not go solid.

### lines 1871-1874

```python
base_detail = 4
```

THE SCENE HAS NO ITERATION COUNT. Its cost is six parallax star layers and three object slots, all fixed, so the adaptive detail loop has nothing to turn down. The numbers are equal so a frame that runs long cannot make the picture change.

### lines 1885-1911

```python
base_detail = 4
```

THE ITERATION COUNT IS FIXED IN THE SHADER, so the adaptive detail loop has nothing to turn down here -- equal numbers mean a frame that runs long cannot change the picture.

THIS COMMENT USED TO SAY "resolution is what gives way instead, through the adaptive render scale". IT DOES NOT, ON THIS PATH. The GPU canvas shades `self.physical_size` at every one of its three uses and never calls `target_render_size`; every `scale` in it is the camera zoom. So for `orbit_gpu` -- and for `space`, which has equal numbers for the same reason -- a long frame has NO lever at all on the GPU.

AND THAT IS MEASURED AS NOT WORTH FIXING, so do not go and write it. This used to say "the fix is render-to-texture and it needs a GPU to verify". A GPU has since been measured -- RTX 3090 Ti, real GL context, 2 s a row -- and every shader has between seven and twenty-five times the headroom it needs even at 4K:

orbit_gpu 2.56 ms   cascade 1.34 ms   space 4.87 ms at 3840x2160, against a 33.3 ms budget

So an FBO, a second shader program and a resize-time allocation would optimise something already far inside its budget, and would add a failure surface to the one path that cannot currently fail that way. The choppiness instruction 327 opens with is the CPU path -- 68.51 ms a frame at 4K, 14.6 fps -- and its adaptive scale, which is where anyone reading this for what to do next should go.

### lines 1919-1921

```python
_DECLARED = frozenset(
```

WHAT THIS SHADER ACTUALLY DECLARES, read out of its source once. The patterns share one uniform update and not one uniform list, and vispy warns per frame per unknown name rather than ignoring it.

## _make_gpu_widget._Canvas.__init__

### lines 1960-1963

```python
self._pointer = Pointer()
```

THE SAME POINTER THE CPU PATH USES. It samples QCursor rather than receiving events, so it needs nothing from the widget except a rectangle to be relative to -- which is why one class serves both backends.

### lines 1965-1967

```python
self._depth_phase = DepthPhase()
```

Integrated travel, so a speed change does not teleport the camera. One per canvas: it is this canvas's own position on the trajectory.

### lines 1969-1974

```python
self._orbit = None
```

THE REFERENCE ORBIT, for the Mandelbrot pattern only. Built on a worker thread because iterating a few thousand points at 320 decimal digits takes seconds, and the backdrop has to keep drawing while it happens -- until it arrives the shader has an all-zero orbit, which renders as the flat interior colour rather than as a stall.

### lines 1992-2001

```python
try:
```

A PLACEHOLDER UNTIL THE REAL ORBIT ARRIVES, and AFTER the program exists: this used to run before `self._program` was assigned, so it raised AttributeError on every build.

vispy warns once per DRAW for a uniform a linked program has never been given, and the real orbit takes seconds to iterate on its thread. One black texel costs nothing and the shader reads it as an orbit at the origin, which draws the interior colour -- what an unset sampler drew anyway, without sixty warnings a second.

### lines 2014-2016

```python
self.native.destroyed.connect(self._on_native_destroyed)
```

STOPPED WHEN QT FREES THE WIDGET, not only when someone calls shutdown. A backdrop is reparented and deleted with its screen, which never runs closeEvent.

## _make_gpu_widget._Canvas._update_uniforms

### lines 2029-2031

```python
phase = self._depth_phase.advance(elapsed, speed)
```

THE PHASE, not `elapsed * speed`. Scrolling changes how fast the trajectory is travelled; it must not change WHERE on the trajectory the camera is. See DepthPhase.

### lines 2033-2038

```python
state = state_at_seconds(phase, 1.0, controls.dream,
```

THE PHASE IS THE CLOCK THE PICTURE IS DRAWN AT, not only the depth. `state_at_seconds` was still handed the wall clock, so the drift, rotation and stretch ignored the speed control entirely while the depth obeyed it -- one camera moving at two rates. At speed 1 the two clocks are the same number, so this changes nothing about how the backdrop looks by default.

### lines 2042-2047

```python
for name, value in (
```

ONLY WHAT THIS SHADER DECLARES. The three patterns share this update but not their uniforms -- space has no dream term, since a star field has nothing to warp -- and vispy warns once per frame for every value handed to a name it cannot find. GPU space printed "Value provided for 'u_dream'" sixty times a second, into the terminal AND the console.

### lines 2050-2058

```python
("u_time", np.float32(phase)),
```

THE PHASE REACHES THE SHADERS TOO, and the speed they are told is 1.0. Every GPU pattern but the Mandelbrot computes its position as `u_time * u_speed` -- the cascade's and the flight's depth, the orbit fold's radial phase -- which is the same teleport the CPU kernels had: measured, a scroll from 1 to 2 moved the star flight further than thirty seconds of travel. The Mandelbrot never reads either, so its dive keeps its own integrator.

### lines 2071-2073

```python
("u_pointer_x", np.float32(pointer_x)),
```

THE POINTER REACHES THE GPU TOO. These were fed on the

CPU path only, so the backdrop followed the mouse in one backend and ignored it in the other.

## _make_gpu_widget._Canvas._start_the_reference_orbit._work

### lines 2103-2105

```python
self._orbit = orbit
```

HANDED OVER BY ASSIGNMENT, which is atomic, rather than touched into the GL program from this thread: a GL call off the thread that owns the context is undefined.

## _make_gpu_widget._Canvas._mandelbrot_uniforms

### lines 2124-2127

```python
now = time.perf_counter()
```

THE DEPTH IS INTEGRATED, not recomputed from the elapsed time: Up and Down change the rate, and a depth derived from `elapsed * rate` would jump backwards the moment the rate was lowered, because the whole flight so far would be re-scaled.

### lines 2132-2134

```python
step = (now - previous) * controls.speed * controls.zoom_rate \
```

SIGNED, and floored at the surface: Down past zero backs out of the zoom, and there is nothing above the starting scale to back out into.

### lines 2139-2146

```python
token = getattr(controls, "restart_token", 0)
```

THE DIVE STARTS AGAIN RATHER THAN ENDING IN A BLACK FRAME. The per-pixel offset is a float32 whatever the reference orbit's precision, and past about forty-five decades the step between neighbouring pixels underflows to zero -- one sample of one point, filling the screen. ASKED TO START AGAIN. The settings changed, so the dive goes back to the surface rather than applying new numbers thirty decades down where they have nothing recognisable to act on.

### lines 2152-2154

```python
camera = getattr(self, "_camera", None)
```

BACK TO THE ANCHOR AS WELL. A restart that kept the course would begin at the surface but already pointed thirty decades of steering away from the centre.

### lines 2158-2160

```python
pilot = getattr(self, "_pilot", None)
```

CTRL+R HANDS THE CAMERA BACK TO THE TOUR, which is what instruction 327 asked for in the same sentence that asked a drag to stop it.

### lines 2182-2186

```python
try:
```

A FAULT IN THE STEERING MUST NOT STOP THE PICTURE. This returns the uniforms for the whole frame, and a NameError in the course-plotting once left every one of them unset -- so the pattern drew nothing at all, silently, which is a far worse failure than a dive that goes straight down.

## _make_gpu_widget._Canvas._steer

### lines 2242-2254

```python
span = scale_at(depth, float(_mandel_setting("initial_scale")))
```

FIXED MEANS FIXED -- but not "aimed at the least interesting place in the frame". The anchor is chosen ONCE, before anything moves, and then never again: the dive is exactly as steady as a fixed path because it IS one, and the survey costs a fraction of a second on a worker thread while the backdrop is already drawing.

Continuous steering is what shook; choosing where to point before the descent starts moves nothing. DRAGGING WORKS ON EITHER PATH. It is the user moving the camera, and refusing that on the steady path would mean the only way to look somewhere else was to turn on the search that shook.

### lines 2262-2264

```python
pilot = getattr(self, "_pilot", None)
```

AND THE TOUR STOPS ARGUING. A drag is a statement about where the user wants to be; a tour that steered over it would be the application disagreeing every frame.

### lines 2268-2272

```python
self._refine_due = 0.0
```

THE REFERENCE FOLLOWS THE CAMERA. Perturbation measures every pixel as a small offset from ONE orbit, so a camera that walks away from it takes the picture with it: measured, a reference 0.3 away escapes at iteration six and the detail in the dragged view falls to nothing.

### lines 2276-2280

```python
self._refine_the_reference(camera, orbit, budget, depth, span)
```

AND KEEPS FOLLOWING IT DOWN. Each refinement is picked out of the current view, and the view shrinks -- so a reference accurate to a pixel now is accurate to a hundredth of that two decades on. Measured: refining every decade holds the picture sharp to eleven, against one or two without.

### lines 2285-2291

```python
pilot = getattr(self, "_pilot", None)
```

THE TWENTY PLACES WORTH LOOKING AT, ON SCREEN AT LAST. `RegionTour` and `fractal_regions.REGIONS` were built, tested and never called; this is the caller. The pilot only writes the camera's TARGET -- the camera's own exponential follow is what moves it, so the tour's smoothstep between regions and the follow compose into one motion rather than a slide that starts and stops.

### lines 2299-2309

```python
return camera.centre
```

NO AUTOMATIC AIMING. Choosing a "more interesting" point by surveying the surface was tried and made it worse: a point on a busy edge at the starting scale was measured going completely flat three decades in -- entirely interior at one candidate, entirely exterior at another -- because surface structure does not predict what survives a descent. Only a genuinely special point does, and the reference centre already is one.

The camera is steered BY HAND instead: drag to move it, and the arrow keys change the speed and the direction.

## _make_gpu_widget._Canvas._refine_the_reference

### lines 2365-2366  _(unsure)_

```python
camera.centre = (0.0, 0.0)
```

The camera is now sitting ON the new reference, so its offset starts again from nothing.

## _make_gpu_widget._Canvas._upload_the_orbit_if_it_arrived

### lines 2414-2419

```python
self._program["u_orbit"] = gloo.Texture2D(
```

CLAMPED AND NEAREST, said outright. The orbit is one row of 2,201 texels -- not a power of two -- and a driver that defaults to REPEAT wrapping can refuse a non-power-of-two texture outright. Nearest because every texel is one iteration of the reference orbit: interpolating between two of them is a number that is not on the orbit at all.

### lines 2426-2427

```python
self._orbit_uploaded = orbit
```

Marked done regardless, or a driver that refuses the format would be asked again sixty times a second.

## _make_gpu_widget._Canvas._pointer_state

### line 2457

```python
return 0.0, 0.0, 0.0, 0.0
```

A backdrop that cannot find the mouse still draws.

## _make_gpu_widget._Canvas.on_draw

### lines 2481-2490

```python
self._dead = True
```

ONE COMPLAINT, NOT A STORM. vispy catches whatever a

DrawEvent handler raises, logs it as an ERROR, and RETRIES doubling a repeat counter each time. A draw that cannot succeed once cannot succeed at all, so the retries only fill the terminal and, because these are logged at ERROR, raise a "spaCR ERROR" panel per retry while a module runs.

This machine's context is OpenGL ES; vispy compiles these shaders as desktop GLSL 120, which ES rejects. Stopping is the honest response: the backdrop cannot draw here.

## _make_gpu_widget._Canvas._on_timer

### lines 2517-2520

```python
if a_popup_is_on_screen():
```

HOLD STILL UNDER A POPUP. A menu or a tooltip composited over this native GL surface makes the widgets around it repaint, which is the flicker of the dock and the header. The last frame stays up; only the clock stops moving.

### lines 2527-2531

```python
self._dead = True
```

"Internal C++ object already deleted". vispy's Timer is not a QTimer and is not destroyed with the widget, so it goes on firing at a canvas Qt has freed -- and vispy's own handler catches, logs and RETRIES, which is where the 2,4,8...4096 repeat storm comes from. Stop the timer at the first one.

## _make_gpu_widget.GpuFractalWidget.__init__

### lines 2576-2597

```python
lock = _heavy_import_lock()
```

UNDER THE HEAVY-IMPORT LOCK. Creating a GL context while the preloader is bringing torch (and therefore CUDA) up is exactly the concurrent initialisation `_PipelinePreloader` used to stay on the GUI thread to avoid. It is on a worker thread now, so the two take turns instead.

THE WAIT IS BOUNDED, and it has to be. This constructor runs on the GUI thread, and the lock's other holder is an import that takes seconds -- so `with lock:` here was a priority inversion: a background task with no deadline holding up the one thread that has one. Measured at 2,130 ms of blocked GUI thread for a 2,000 ms hold, against ~130 ms of actual construction. That is the whole of the difference the maintainer reported between `spaceout` and `spacr` on opening a module: only spaceout builds this widget, and only this widget takes the lock. The ordinary ambient backdrop never does, which is why `spacr` opened the same screen without the compositor offering to force-quit.

`AppScreen._heavy_lock_is_free` cannot prevent it. That peek is a check, not a reservation, and the preloader re-takes the lock between two imports -- so landing in the gap is not rare, it is the ordinary case for a click made while the preloader runs.

## nudge_zoom_rate

### lines 2756-2765

```python
rate = float(controls.zoom_rate)
```

SIGNED, AND THE KEY CHANGES THE VALUE, NOT THE MAGNITUDE. Asked for 2026-08-28: "so i could go slow and fast forward and back". Down means "less", all the way through zero into backing out of the zoom; Up means "more".

Stepping the magnitude multiplicatively and flipping the sign at the floor OSCILLATES: every press at the floor swaps the direction, so holding Down never gets anywhere. Which side of zero the rate is on has to decide whether a step grows or shrinks it.

### line 2770  _(unsure)_

```python
rate = rate * ZOOM_STEP if rate > 0 else rate * ZOOM_STEP
```

Away from zero on the side it is already on.

### lines 2773-2774  _(unsure)_

```python
shrunk = rate / ZOOM_STEP
```

Toward zero, and through it when there is nowhere left to shrink to.

## create_fractal_widget

### lines 2846-2848

```python
settings = replace(
```

THE PATTERN THIS MACHINE CAN ACTUALLY DRAW. Mandelbrot is GPU-only, so a machine with no usable context gets the orbit fold rather than a backdrop that draws nothing.

### lines 2852-2863

```python
if not any(existing is controls for existing in _LIVE_CONTROLS):
```

Registered so a key press can reach it; the backdrop itself must not accept events, or it would eat the clicks meant for the interface in front of it. BY IDENTITY, NOT BY VALUE. `RuntimeControls` is a dataclass, so `in` compares field by field: a new backdrop whose settings happen to match a previous one was never added, and the keys then drove a stale object that no canvas reads. That is why Ctrl+R and Up and Down stopped working after the first backdrop was replaced.

The list is also trimmed here, because nothing else can: a backdrop that has been destroyed leaves its controls behind, and updating a few dead ones is harmless while letting the list grow without bound is not.

### lines 2869-2870

```python
if settings.backend in ("auto", "gpu") and gpu_is_available():
```

`gpu_is_available` covers the explicit 'gpu' request as well: asking for a renderer this platform would crash on is still a crash.

### lines 2875-2878

```python
raise
```

NOT A GPU FAILURE, so not the CPU renderer's cue. The context was never attempted; the lock was busy. Falling through here would trade a 0.3 s wait for the twenty-core fallback, and would do it every time a module is opened during startup.

### lines 2881-2885

```python
LOG.warning("the GPU backdrop could not be built; falling back "
```

SAID OUT LOUD. This swallowed every GPU failure without a word, so a shader that would not compile looked exactly like a machine with no GPU -- and the Mandelbrot pattern, which has no CPU renderer, came out as the orbit fold with nothing anywhere to say why.

### lines 2889-2892

```python
LOG.warning("the Mandelbrot pattern needs the GPU renderer; "
```

AND THE CPU CANNOT DRAW THIS ONE. Handing it to the CPU builder silently produced the orbit fold, because that is what its final `else` does -- so the user chose Mandelbrot and got something else with no indication anything had happened.
