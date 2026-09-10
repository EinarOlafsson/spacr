#!/usr/bin/env python3
"""Render polished, geometry-safe 4K keyframes for installation lessons."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
LOGICAL_SIZE = (1920, 1080)
SCALE = 2
SIZE = tuple(value * SCALE for value in LOGICAL_SIZE)
FONT_PATH = ROOT / "web" / "fonts" / "OpenSans-Regular.ttf"
RELEASE_VERSION = "1.5.0.4"

BG_TOP = (5, 9, 16)
BG_BOTTOM = (10, 17, 28)
PANEL = "#080d14"
PANEL_HEADER = "#111927"
PANEL_LINE = "#273448"
TEXT = "#f5f7fb"
MUTED = "#aab6c7"
DIM = "#718097"
CYAN = "#4a9eff"
GREEN = "#42d392"
MAGENTA = "#ff0099"
AMBER = "#e6b450"
RED = "#ff6b6b"


def scaled(value: int | float) -> int:
    return int(round(value * SCALE))


def point(x: int | float, y: int | float) -> tuple[int, int]:
    return scaled(x), scaled(y)


def box(values) -> tuple[int, int, int, int]:
    return tuple(scaled(value) for value in values)


def focus_box(values) -> list[int]:
    return [scaled(value) for value in values]


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_PATH), scaled(size))


TITLE = font(48)
SUBTITLE = font(24)
COMMAND = font(28)
OUTPUT = font(24)
BODY = font(26)
SMALL = font(20)
TINY = font(16)


def gradient_background() -> Image.Image:
    image = Image.new("RGB", SIZE)
    draw = ImageDraw.Draw(image)
    for y in range(SIZE[1]):
        amount = y / max(1, SIZE[1] - 1)
        colour = tuple(
            int(round(start + (end - start) * amount))
            for start, end in zip(BG_TOP, BG_BOTTOM)
        )
        draw.line((0, y, SIZE[0], y), fill=colour)
    return image


def draw_text(draw, xy, text, text_font, fill, anchor=None):
    draw.text(point(*xy), text, font=text_font, fill=fill, anchor=anchor)


def canvas(title: str, subtitle: str):
    image = gradient_background()
    draw = ImageDraw.Draw(image)
    draw.ellipse(box((1450, -380, 2200, 440)), fill="#0d2238")
    draw.ellipse(box((-350, 810, 350, 1510)), fill="#10182b")
    draw_text(draw, (150, 60), title, TITLE, TEXT)
    draw_text(draw, (152, 126), subtitle, SUBTITLE, MUTED)
    draw_text(draw, (150, 1020), "spaCR tutorial series", SMALL, MUTED)
    draw_text(draw, (1770, 1020), "Official installation routes", SMALL,
              MUTED, anchor="ra")
    return image, draw


def panel(draw, bounds=(150, 190, 1770, 925), label="spaCR install"):
    x1, y1, x2, y2 = bounds
    for offset, colour in ((14, "#03060a"), (8, "#050910"), (3, "#070b12")):
        draw.rounded_rectangle(box((x1 + offset, y1 + offset,
                                    x2 + offset, y2 + offset)),
                               radius=scaled(19), fill=colour)
    draw.rounded_rectangle(box(bounds), radius=scaled(19), fill=PANEL,
                           outline=PANEL_LINE, width=scaled(1))
    draw.rounded_rectangle(box((x1, y1, x2, y1 + 66)), radius=scaled(19),
                           fill=PANEL_HEADER)
    draw.rectangle(box((x1, y1 + 35, x2, y1 + 66)), fill=PANEL_HEADER)
    draw.line(box((x1, y1 + 66, x2, y1 + 66)), fill=PANEL_LINE,
              width=scaled(1))
    for index, colour in enumerate(("#ff5f57", "#febc2e", "#28c840")):
        x = x1 + 37 + index * 30
        draw.ellipse(box((x, y1 + 26, x + 13, y1 + 39)), fill=colour)
    draw.rounded_rectangle(box((x1 + 165, y1 + 13, x1 + 455, y1 + 55)),
                           radius=scaled(8), fill="#0a111c",
                           outline="#324158", width=scaled(1))
    draw.ellipse(box((x1 + 186, y1 + 30, x1 + 194, y1 + 38)), fill=GREEN)
    draw_text(draw, (x1 + 211, y1 + 34), label, SMALL, "#dce5f2",
              anchor="lm")
    draw_text(draw, (x2 - 43, y1 + 34), "bash", TINY, DIM, anchor="rm")


def note(draw, text: str, colour=CYAN) -> list[int]:
    bounds = (205, 815, 1715, 885)
    draw.rounded_rectangle(box(bounds), radius=scaled(9), fill="#0b1b2a",
                           outline="#24415d", width=scaled(1))
    draw.ellipse(box((232, 844, 242, 854)), fill=colour)
    draw_text(draw, (262, 849), text, SMALL, "#d9edff", anchor="lm")
    return focus_box(bounds)


def terminal_frame(path: Path, title: str, subtitle: str,
                   lines: list[tuple[str, str]], active: int,
                   note_text: str = "", note_colour=CYAN) -> list[int]:
    image, draw = canvas(title, subtitle)
    panel(draw)
    draw_text(draw, (203, 278), "spacr-env", TINY, CYAN, anchor="lm")
    draw_text(draw, (300, 278), "in", TINY, DIM, anchor="lm")
    draw_text(draw, (325, 278), "~", TINY, AMBER, anchor="lm")
    row_top = 315
    row_step = 78
    row_height = 64
    for index, (kind, value) in enumerate(lines):
        top = row_top + index * row_step
        center = top + row_height / 2
        is_active = index == active
        if is_active:
            draw.rounded_rectangle(box((195, top - 3, 1722, top + row_height + 3)),
                                   radius=scaled(7), fill="#101c2a")
            draw.rectangle(box((195, top + 7, 198, top + row_height - 7)),
                           fill=CYAN)
        elif index:
            draw.line(box((220, top - 7, 1690, top - 7)), fill="#152131",
                      width=scaled(1))
        if kind == "command":
            draw_text(draw, (218, center), "$", COMMAND, GREEN, anchor="lm")
            draw_text(draw, (268, center), value, COMMAND,
                      TEXT if is_active else "#d3dbe7", anchor="lm")
        else:
            draw_text(draw, (222, center), "│", OUTPUT, "#34445b", anchor="lm")
            draw_text(draw, (268, center), value, OUTPUT,
                      MUTED if is_active else DIM, anchor="lm")
    focus = focus_box((195, row_top + active * row_step - 3, 1527,
                       row_height + 6))
    if note_text:
        note(draw, note_text, note_colour)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, compress_level=3)
    return focus


def route_frame(path: Path, title: str, subtitle: str,
                steps: list[tuple[str, str, str]], active: int | None,
                footer: str) -> list[int] | None:
    image, draw = canvas(title, subtitle)
    panel(draw, label="installation map")
    x_positions = [220, 755, 1290]
    for index, ((heading, detail, badge), x) in enumerate(zip(steps, x_positions)):
        selected = active == index
        fill = "#142b41" if selected else "#0e1724"
        outline = CYAN if selected else "#34485f"
        draw.rounded_rectangle(box((x, 330, x + 410, 690)),
                               radius=scaled(16), fill=fill,
                               outline=outline, width=scaled(1))
        draw.rounded_rectangle(box((x + 28, 365, x + 146, 398)),
                               radius=scaled(16), fill="#173a58")
        draw_text(draw, (x + 87, 381), badge, TINY, "#d9edff", anchor="mm")
        draw_text(draw, (x + 28, 460), heading, BODY, TEXT)
        draw.multiline_text(point(x + 28, 520), detail, font=SMALL, fill=MUTED,
                            spacing=scaled(10))
        if index < len(steps) - 1:
            draw_text(draw, (x + 468, 510), "→", font(40), DIM, anchor="mm")
    note(draw, footer, GREEN if active is None else CYAN)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, compress_level=3)
    return None if active is None else focus_box(
        (x_positions[active], 330, 410, 360))


def platform_frame(path: Path, active: str | None, footer: str,
                   detail: tuple[str, ...] = ()) -> list[int] | None:
    image, draw = canvas(
        "Native spaCR installers",
        "No existing Python or Conda installation is required",
    )
    panel(draw, label="GitHub Release")
    cards = [
        ("windows", "Windows 10 / 11", ".exe",
         "Automatic CUDA\nCPU-only fallback"),
        ("macos", "macOS 11+", ".pkg", "Intel + Apple silicon\nApple MPS"),
        ("linux", "Linux x86-64", ".run",
         "Automatic CUDA\nCPU-only fallback"),
    ]
    x_positions = (220, 755, 1290)
    focus = None
    for (key, name, suffix, description), x in zip(cards, x_positions):
        selected = active == key
        fill = "#183752" if selected else "#101b29"
        outline = CYAN if selected else "#34485f"
        draw.rounded_rectangle(box((x, 315, x + 410, 690)),
                               radius=scaled(16), fill=fill,
                               outline=outline, width=scaled(1))
        draw_text(draw, (x + 28, 375), name, BODY, TEXT)
        draw_text(draw, (x + 28, 470), suffix, font(58), MAGENTA)
        draw.multiline_text(point(x + 28, 565), description, font=SMALL,
                            fill=MUTED, spacing=scaled(8))
        if selected:
            focus = focus_box((x, 315, 410, 375))
    if detail:
        y = 726
        for item in detail:
            draw.ellipse(box((235, y + 8, 245, y + 18)), fill=CYAN)
            draw_text(draw, (264, y + 13), item, SMALL, "#d9edff", anchor="lm")
            y += 42
    else:
        note(draw, footer, GREEN)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, compress_level=3)
    return focus


def write_geometry(lesson_root: Path, scenes: list[dict]) -> None:
    target = lesson_root / "keyframes" / "geometry.json"
    target.write_text(json.dumps({
        "schema": 2,
        "frame_size": list(SIZE),
        "scenes": scenes,
    }, indent=2) + "\n")


def build_sources(production: Path, captures: Path | None) -> None:
    root = production / "01_pypi_github"
    out = root / "keyframes"
    out.mkdir(parents=True, exist_ok=True)
    capture_geometry_path = out / "capture_geometry.json"
    if captures:
        mapping = {
            "pypi.png": "01_pypi_top.png",
            "github_main.png": "04_github_main.png",
            "github_release.png": "05_github_release.png",
        }
        for source, target in mapping.items():
            candidate = captures / source
            if not candidate.is_file():
                raise FileNotFoundError(candidate)
            shutil.copy2(candidate, out / target)
        source_geometry = captures / "geometry.json"
        if not source_geometry.is_file():
            raise FileNotFoundError(source_geometry)
        shutil.copy2(source_geometry, capture_geometry_path)
    if not capture_geometry_path.is_file():
        raise FileNotFoundError(
            "lesson 01 needs a fresh capture bundle; run "
            "tools/capture_install_web.py first"
        )
    capture_geometry = json.loads(capture_geometry_path.read_text())
    if capture_geometry.get("frame_size") != list(SIZE):
        raise ValueError("lesson 01 web captures are not native 4K")
    if capture_geometry.get("release_version") != RELEASE_VERSION:
        raise ValueError(
            "lesson 01 capture version does not match RELEASE_VERSION: "
            f"{capture_geometry.get('release_version')!r}"
        )

    # PyPI supplies both its release-header and Project-links scenes.
    pypi = out / "01_pypi_top.png"
    github = out / "04_github_main.png"
    release = out / "05_github_release.png"
    for required in (pypi, github, release):
        if not required.is_file():
            raise FileNotFoundError(required)
        with Image.open(required) as image:
            if image.size != SIZE:
                raise ValueError(f"web capture is not native 4K: {required}")

    route_frame(
        out / "06_conda_status.png",
        "Official conda-forge package",
        "spaCR and desktop dependencies  •  published on conda-forge",
        [
            ("Create environment", "Python 3.12\nIsolated dependencies", "CONDA"),
            ("Install spaCR", "conda install\nconda-forge::spacr", "OFFICIAL"),
            ("Launch", "spacr\nDesktop application", "READY"),
        ],
        None,
        "The channel-qualified command selects the official conda-forge build.",
    )
    route_frame(
        out / "07_decision.png",
        "Choose an official installation route",
        "Stable packages, native installers, or development source",
        [
            ("conda-forge", "Desktop package\nand dependencies", "CONDA"),
            ("PyPI", "Python package\nand Qt extra", "PYTHON"),
            ("GitHub", "Native installers\nor nightly source", "RELEASES"),
        ],
        None,
        "Use nightly only for unreleased changes that are still under test.",
    )
    write_geometry(root, [
        {"image": "keyframes/01_pypi_top.png", "focus": None},
        {"image": "keyframes/01_pypi_top.png",
         "focus": capture_geometry["pypi_release"]},
        {"image": "keyframes/01_pypi_top.png",
         "focus": capture_geometry["pypi_links"]},
        {"image": "keyframes/04_github_main.png",
         "focus": capture_geometry["github_branch"]},
        {"image": "keyframes/05_github_release.png",
         "focus": capture_geometry["github_assets"]},
        {"image": "keyframes/06_conda_status.png", "focus": None},
        {"image": "keyframes/07_decision.png", "focus": None},
    ])


def build_conda(production: Path) -> None:
    root = production / "02_conda_install"
    out = root / "keyframes"
    scenes = []
    route_frame(out / "01.png", "Install spaCR with Conda",
                "Isolated environment  •  Python 3.12 recommended",
                [("Create", "A clean named\nenvironment", "1"),
                 ("Install", "Official conda-forge\ndesktop package", "2"),
                 ("Validate", "Check Qt, PyTorch,\nand hardware", "3")],
                None, "conda-forge supplies spaCR and its declared dependencies.")
    scenes.append({"image": "keyframes/01.png", "focus": None})
    frames = [
        ([('command', 'conda create -n spacr python=3.12 -y')], 0,
         "Create one environment dedicated to spaCR."),
        ([('command', 'conda activate spacr'),
          ('command', 'python -c \"import sys; print(sys.executable)\"'),
          ('output', '.../envs/spacr/bin/python')], 1,
         "The reported Python path must belong to the spacr environment."),
        ([('command', 'conda install conda-forge::spacr')], 0,
         "Install the official conda-forge package into the active environment."),
        ([('command', 'conda list spacr'),
          ('output', 'spacr  ' + RELEASE_VERSION + '  pyhd8ed1ab_0  conda-forge'),
          ('command', 'python -c \"import spacr; print(spacr.__version__)\"'),
          ('output', RELEASE_VERSION)], 2,
         "Confirm the package record and imported spaCR version."),
        ([('command', 'spacr-doctor'),
          ('output', 'PASS  Python  •  Qt  •  PyTorch  •  package consistency')], 0,
         "Resolve failed checks before the first application launch."),
    ]
    for index, item in enumerate(frames, start=2):
        lines, active, note_text, *colour = item
        focus = terminal_frame(out / f"{index:02d}.png",
                               "Install spaCR with Conda",
                               "Isolated environment  •  Python 3.12 recommended",
                               lines, active, note_text,
                               colour[0] if colour else CYAN)
        scenes.append({"image": f"keyframes/{index:02d}.png", "focus": focus})
    focus = terminal_frame(
        out / "07.png", "Launch spaCR",
        "Keep the spacr environment active for every session",
        [('command', 'spacr'),
         ('output', 'spaCR desktop application')],
        0,
        "The conda-forge package includes the desktop interface.",
        GREEN,
    )
    scenes.append({"image": "keyframes/07.png", "focus": focus})
    write_geometry(root, scenes)


def build_pip(production: Path) -> None:
    root = production / "03_pip_install"
    out = root / "keyframes"
    scenes = []
    route_frame(out / "01.png", "Install spaCR with pip",
                "Use an isolated virtual environment on every operating system",
                [("Create", "Python 3.12\nvirtual environment", "1"),
                 ("Install", "Desktop or\nheadless package", "2"),
                 ("Verify", "Version and\ndiagnostics", "3")],
                None, "Do not modify the operating system's Python installation.")
    scenes.append({"image": "keyframes/01.png", "focus": None})
    frames = [
        ([('command', 'python3.12 -m venv spacr-env'),
          ('command', 'source spacr-env/bin/activate')], 1,
         "macOS and Linux"),
        ([('command', 'py -3.12 -m venv spacr-env'),
          ('command', r'spacr-env\Scripts\activate')], 1,
         "Windows PowerShell or Command Prompt"),
        ([('command', 'python -m pip install --upgrade pip')], 0,
         "The active Python chooses the matching pip."),
        ([('command', 'python -m pip install spacr'),
          ('command', 'python -m pip install \"spacr[qt]\"')], 1,
         "Plain package: headless. Qt extra: desktop application."),
        ([('command', 'python -c \"import spacr; print(spacr.__version__)\"'),
          ('output', RELEASE_VERSION),
          ('command', 'spacr-doctor')], 2,
         "Verify the package and resolve diagnostics before launch."),
        ([('command', 'python -m pip install --upgrade spacr'),
          ('output', 'Update only when you intend to change versions')], 0,
         "Keep the environment activated whenever you use spaCR."),
    ]
    for index, (lines, active, note_text) in enumerate(frames, start=2):
        focus = terminal_frame(out / f"{index:02d}.png",
                               "Install spaCR with pip",
                               "Standard virtual environment  •  Python 3.12 recommended",
                               lines, active, note_text)
        scenes.append({"image": f"keyframes/{index:02d}.png", "focus": focus})
    write_geometry(root, scenes)


def build_installers(production: Path) -> None:
    root = production / "04_platform_installers"
    out = root / "keyframes"
    scenes = []
    platform_frame(out / "01.png", None,
                   "Each installer creates and validates a private Python 3.12 runtime.")
    scenes.append({"image": "keyframes/01.png", "focus": None})
    release_focus = route_frame(
        out / "02.png", "Download from GitHub Releases",
        "Choose the latest release and verify its checksum when required",
        [("Windows", "Online Setup\nexecutable", ".EXE"),
         ("macOS", "Universal online\npackage", ".PKG"),
         ("Linux", "x86-64 online\ninstaller", ".RUN")],
        None, "SHA256SUMS.txt is published beside the three installers.")
    scenes.append({"image": "keyframes/02.png", "focus": release_focus})
    for index, (active, detail) in enumerate([
        ("windows", ("Open the .exe on Windows 10 or 11.",
                     "Automatic acceleration is selected by default.")),
        ("macos", ("Open the universal .pkg on macOS 11 or newer.",
                   "If blocked: Privacy & Security → Open Anyway.")),
    ], start=3):
        focus = platform_frame(out / f"{index:02d}.png", active, "", detail)
        scenes.append({"image": f"keyframes/{index:02d}.png", "focus": focus})
    linux_lines = [
        ('command', 'chmod +x SpaCR-*-Linux-x86_64-Online.run'),
        ('command', './SpaCR-*-Linux-x86_64-Online.run'),
    ]
    focus = terminal_frame(out / "05.png", "Install spaCR on Linux",
                           "64-bit x86-64 online installer", linux_lines, 1,
                           "Automatic backend selection is the default.")
    scenes.append({"image": "keyframes/05.png", "focus": focus})
    focus = terminal_frame(
        out / "06.png", "CPU-only installation on Linux",
        "Override the accelerated default only when required",
        [('command', './SpaCR-*-Linux-x86_64-Online.run --torch-backend cpu')],
        0, "The default uses CUDA automatically on compatible NVIDIA hardware.")
    scenes.append({"image": "keyframes/06.png", "focus": focus})
    route_frame(out / "07.png", "Safe installer updates",
                "Validate first, replace only after every check passes",
                [("Download", "Private Python, Qt,\nPyTorch, spaCR", "1"),
                 ("Validate", "Imports and package\nconsistency", "2"),
                 ("Switch", "Preserve previous\nworking version", "3")],
                1, "Diagnostics are saved as install.log in the private installation.")
    scenes.append({"image": "keyframes/07.png",
                   "focus": focus_box((755, 330, 410, 360))})

    # Installation ends at the same real Preferences surface introduced by
    # the Home lesson. Reusing the native 4K capture keeps this lesson tied to
    # the shipped five-level control instead of drawing a second, drift-prone
    # imitation of the application.
    home_keyframes = production / "05_home" / "keyframes"
    performance_frame = home_keyframes / "02_performance_4k.png"
    home_geometry_path = home_keyframes / "geometry.json"
    if not performance_frame.is_file() or not home_geometry_path.is_file():
        raise FileNotFoundError(
            "the current Home Preferences capture is required before "
            "rendering the installer lesson"
        )
    home_geometry = json.loads(home_geometry_path.read_text(encoding="utf-8"))
    performance_focus = home_geometry.get("performance")
    if not performance_focus:
        raise RuntimeError("Home capture has no Performance-level focus box")
    shutil.copy2(performance_frame, out / "08.png")
    scenes.append({"image": "keyframes/08.png", "focus": performance_focus})

    platform_frame(out / "09.png", None,
                   "Installation complete — use the operating system launcher.")
    scenes.append({"image": "keyframes/09.png", "focus": None})
    write_geometry(root, scenes)


def build_api(production: Path) -> None:
    """Render the headless lesson from the CLI's current public contract."""
    root = production / "06_api"
    out = root / "keyframes"
    frames = [
        ([('command', 'python -c "import spacr; print(spacr.__version__)"'),
          ('output', RELEASE_VERSION)], 0,
         "The active interpreter imports the expected spaCR release."),
        ([('command', 'spacr-run --list'),
          ('output', 'mask  measure  annotate  classify  report  …')], 0,
         "List the registered headless modules before choosing one."),
        ([('command', 'spacr-run mask --settings mask_settings.csv'),
          ('output', 'Pre-flight passed  •  starting mask')], 0,
         "An exported GUI settings file is also a reproducible CLI input."),
        ([('command', 'spacr-run mask --settings mask_settings.csv --dry-run'),
          ('output', 'Resolved settings  •  pre-flight passed  •  no data written')], 0,
         "Dry-run validates the exact plan without processing a field."),
        ([('command', 'spacr-run mask --settings mask_settings.csv'),
          ('output', 'Run complete  •  manifest, settings, outputs and log saved')], 0,
         "Execute only after the dry-run plan and pre-flight checks pass."),
    ]
    scenes = []
    for index, (lines, active, note) in enumerate(frames, start=1):
        focus = terminal_frame(
            out / f"{index:02d}.png",
            "Automate spaCR with the Python API",
            "The same reproducible workflows on workstations, servers, and clusters",
            lines, active, note,
        )
        scenes.append({"image": f"keyframes/{index:02d}.png", "focus": focus})
    write_geometry(root, scenes)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("production_root", type=Path)
    parser.add_argument("--captures", type=Path,
                        help="Optional folder containing pypi.png, "
                             "github_main.png, and github_release.png")
    args = parser.parse_args()
    build_sources(args.production_root, args.captures)
    build_conda(args.production_root)
    build_pip(args.production_root)
    build_installers(args.production_root)
    build_api(args.production_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
