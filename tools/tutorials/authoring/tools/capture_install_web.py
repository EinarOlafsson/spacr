#!/usr/bin/env python3
"""Capture the live spaCR release pages and their semantic focus geometry.

The installation overview previously painted release numbers onto archived
screenshots.  Apart from looking artificial, those fixed-coordinate edits
made the focus rectangles drift away from the page elements.  This tool uses
Chrome's DevTools protocol to capture the real release pages and records the
bounding rectangles returned by the live DOM in the same pass.

The output is a small capture bundle consumed by
``render_install_keyframes.py --captures``::

    python tools/capture_install_web.py \
        --output generated/install_web_captures

No logged-in browser profile is used.  The screenshots therefore contain
only public release information and are reproducible on another workstation.
"""
from __future__ import annotations

import argparse
import base64
import json
import shutil
import socket
import struct
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "generated" / "install_web_captures"
RELEASE_VERSION = "1.5.0.4"
VIEWPORT = (1920, 1080)
DEVICE_SCALE = 2
FRAME_SIZE = tuple(value * DEVICE_SCALE for value in VIEWPORT)


class DevTools:
    """Minimal synchronous Chrome DevTools client."""

    def __init__(self, websocket_url: str):
        try:
            import websocket
        except ImportError as error:  # pragma: no cover - environment guard
            raise RuntimeError(
                "capture_install_web.py needs the websocket-client package"
            ) from error
        self._socket = websocket.create_connection(
            websocket_url,
            timeout=30,
            origin="http://localhost",
        )
        self._next_id = 0

    def close(self) -> None:
        self._socket.close()

    def call(self, method: str, params: dict | None = None) -> dict:
        self._next_id += 1
        request_id = self._next_id
        self._socket.send(json.dumps({
            "id": request_id,
            "method": method,
            "params": params or {},
        }))
        while True:
            message = json.loads(self._socket.recv())
            if message.get("id") != request_id:
                continue
            if "error" in message:
                raise RuntimeError(
                    f"Chrome DevTools {method} failed: {message['error']}"
                )
            return message.get("result", {})

    def evaluate(self, expression: str):
        result = self.call("Runtime.evaluate", {
            "expression": expression,
            "returnByValue": True,
            "awaitPromise": True,
        })
        exception = result.get("exceptionDetails")
        if exception:
            raise RuntimeError(f"browser JavaScript failed: {exception}")
        return result.get("result", {}).get("value")


def _free_port() -> int:
    with socket.socket() as candidate:
        candidate.bind(("127.0.0.1", 0))
        return int(candidate.getsockname()[1])


def _page_websocket(port: int, timeout: float = 20.0) -> str:
    deadline = time.monotonic() + timeout
    endpoint = f"http://127.0.0.1:{port}/json/list"
    while time.monotonic() < deadline:
        try:
            with urlopen(endpoint, timeout=1) as response:
                targets = json.load(response)
            page = next(item for item in targets if item.get("type") == "page")
            return str(page["webSocketDebuggerUrl"])
        except (OSError, URLError, StopIteration, ValueError, KeyError):
            time.sleep(0.1)
    raise RuntimeError("Chrome did not expose a DevTools page target")


def _wait_ready(browser: DevTools, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if browser.evaluate("document.readyState") == "complete":
            # Web fonts, badges, and GitHub's deferred fragments settle after
            # the load event.  This is short and bounded rather than a network
            # idle wait, which third-party analytics can keep open forever.
            time.sleep(2.0)
            return
        time.sleep(0.1)
    raise RuntimeError("page did not finish loading")


def _navigate(browser: DevTools, url: str) -> None:
    result = browser.call("Page.navigate", {"url": url})
    if result.get("errorText"):
        raise RuntimeError(f"could not load {url}: {result['errorText']}")
    _wait_ready(browser)


def _wait_for(browser: DevTools, expression: str,
              description: str, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if browser.evaluate(expression):
            return
        time.sleep(0.2)
    raise RuntimeError(f"timed out waiting for {description}")


def _png_size(data: bytes) -> tuple[int, int]:
    if data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        raise ValueError("Chrome did not return a PNG")
    return struct.unpack(">II", data[16:24])


def _capture(browser: DevTools, path: Path) -> None:
    payload = browser.call("Page.captureScreenshot", {
        "format": "png",
        "fromSurface": True,
        "captureBeyondViewport": False,
    })
    data = base64.b64decode(payload["data"])
    if _png_size(data) != FRAME_SIZE:
        raise RuntimeError(
            f"captured {_png_size(data)}, expected native {FRAME_SIZE}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _scaled_rect(value, name: str) -> list[int]:
    if not isinstance(value, dict):
        raise RuntimeError(f"browser did not return the {name} rectangle")
    rect = [
        int(round(float(value["x"]) * DEVICE_SCALE)),
        int(round(float(value["y"]) * DEVICE_SCALE)),
        int(round(float(value["width"]) * DEVICE_SCALE)),
        int(round(float(value["height"]) * DEVICE_SCALE)),
    ]
    x, y, width, height = rect
    if (x < 0 or y < 0 or width <= 0 or height <= 0 or
            x + width > FRAME_SIZE[0] or y + height > FRAME_SIZE[1]):
        raise RuntimeError(f"{name} rectangle leaves the frame: {rect}")
    return rect


RECT_HELPERS = r"""
(() => {
  const normal = value => (value || '').replace(/\s+/g, ' ').trim();
  const visible = element => {
    if (!element) return false;
    const style = getComputedStyle(element);
    const rect = element.getBoundingClientRect();
    return style.display !== 'none' && style.visibility !== 'hidden' &&
      rect.width > 1 && rect.height > 1;
  };
  const exact = (selector, text) => Array.from(document.querySelectorAll(selector))
    .filter(visible).filter(element => normal(element.innerText) === text);
  const nearest = (elements, y) => elements.sort((left, right) =>
    Math.abs(left.getBoundingClientRect().top - y) -
    Math.abs(right.getBoundingClientRect().top - y))[0];
  const union = elements => {
    const rects = elements.filter(visible).map(element => element.getBoundingClientRect());
    if (!rects.length) return null;
    const left = Math.min(...rects.map(rect => rect.left));
    const top = Math.min(...rects.map(rect => rect.top));
    const right = Math.max(...rects.map(rect => rect.right));
    const bottom = Math.max(...rects.map(rect => rect.bottom));
    return {x: left, y: top, width: right - left, height: bottom - top};
  };
  window.__tutorialGeometry = {normal, visible, exact, nearest, union};
  return true;
})()
"""


def _pypi(browser: DevTools, output: Path) -> dict[str, list[int]]:
    _navigate(browser, f"https://pypi.org/project/spacr/{RELEASE_VERSION}/")
    _wait_for(
        browser,
        f"document.body.innerText.includes('spacr {RELEASE_VERSION}')",
        "the current PyPI release header",
    )
    # At 135%, the real header and verified Project links remain comfortably
    # visible while the immutable long-description links from older releases
    # stay below the viewport.  Browser zoom changes layout only; no page text
    # or pixels are edited.
    browser.evaluate(r"""
(() => {
  document.documentElement.style.zoom = '1.35';
  scrollTo(0, 0);
  // Release pages can retain README badges whose cached version and current
  // branch status no longer describe the published package. The tutorial is
  // about installation sources, not transient CI, so omit that entire badge
  // paragraph instead of repainting an individual badge green or pasting a
  // release number over it.
  const paragraph = Array.from(document.querySelectorAll('p')).find(element =>
    element.querySelectorAll('img').length >= 4);
  if (paragraph) paragraph.style.display = 'none';
  return true;
})()
""")
    time.sleep(0.5)
    browser.evaluate(RECT_HELPERS)
    geometry = browser.evaluate(f"""
(() => {{
  const h = window.__tutorialGeometry;
  const title = h.exact('h1', 'spacr {RELEASE_VERSION}')[0];
  const summary = title && title.parentElement.querySelector(
    '.project-header__summary'
  );
  const textBox = (element, phrase) => {{
    const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT);
    while (walker.nextNode()) {{
      const node = walker.currentNode;
      const index = node.data.indexOf(phrase);
      if (index < 0) continue;
      const range = document.createRange();
      range.setStart(node, index);
      range.setEnd(node, index + phrase.length);
      const rect = range.getBoundingClientRect();
      return {{x: rect.left, y: rect.top, width: rect.width, height: rect.height}};
    }}
    return null;
  }};
  const unionBoxes = boxes => {{
    const valid = boxes.filter(Boolean);
    const left = Math.min(...valid.map(rect => rect.x));
    const top = Math.min(...valid.map(rect => rect.y));
    const right = Math.max(...valid.map(rect => rect.x + rect.width));
    const bottom = Math.max(...valid.map(rect => rect.y + rect.height));
    return {{x: left, y: top, width: right - left, height: bottom - top}};
  }};
  const commandText = 'pip install spacr=={RELEASE_VERSION}';
  const commandCandidates = Array.from(document.querySelectorAll('code, pre, span, div'))
    .filter(h.visible)
    .filter(element => h.normal(element.innerText) === commandText);
  const command = commandCandidates.sort((a, b) =>
    a.getBoundingClientRect().width - b.getBoundingClientRect().width)[0];
  const copyButton = title && title.parentElement.querySelector(
    '.project-header__pip-instructions button'
  );
  const heading = h.exact('h1, h2, h3, h4, h5, h6', 'Project links')[0];
  const links = ['Documentation', 'Homepage', 'Issues', 'Source'].map(label =>
    h.nearest(h.exact('a', label), heading.getBoundingClientRect().top));
  return {{
    release: unionBoxes([
      textBox(title, 'spacr {RELEASE_VERSION}'),
      summary && textBox(summary, h.normal(summary.innerText)),
      command && command.getBoundingClientRect(),
      copyButton && copyButton.getBoundingClientRect(),
    ]),
    links: h.union([heading, ...links]),
  }};
}})()
""")
    result = {
        "pypi_release": _scaled_rect(geometry.get("release"), "PyPI release"),
        "pypi_links": _scaled_rect(geometry.get("links"), "PyPI links"),
    }
    _capture(browser, output / "pypi.png")
    return result


def _github_main(browser: DevTools, output: Path) -> dict[str, list[int]]:
    _navigate(browser, "https://github.com/EinarOlafsson/spacr/tree/main")
    _wait_for(
        browser,
        "document.body.innerText.includes('EinarOlafsson') && "
        "document.body.innerText.includes('spacr')",
        "the GitHub repository",
    )
    browser.evaluate(RECT_HELPERS)
    rect = browser.evaluate(r"""
(() => {
  const h = window.__tutorialGeometry;
  const labels = h.exact('span, button, summary', 'main');
  for (const label of labels) {
    const control = label.closest('button, summary, a');
    if (h.visible(control) && control.getBoundingClientRect().top > 100) {
      return h.union([control]);
    }
  }
  return null;
})()
""")
    _capture(browser, output / "github_main.png")
    return {"github_branch": _scaled_rect(rect, "GitHub branch selector")}


def _github_release(browser: DevTools, output: Path) -> dict[str, list[int]]:
    _navigate(
        browser,
        f"https://github.com/EinarOlafsson/spacr/releases/tag/v{RELEASE_VERSION}",
    )
    _wait_for(
        browser,
        f"document.body.innerText.includes('SpaCR {RELEASE_VERSION}')",
        "the current GitHub release",
    )
    browser.evaluate(RECT_HELPERS)
    browser.evaluate(r"""
(() => {
  const h = window.__tutorialGeometry;
  // Keep this release tutorial about the currently published artifacts.  The
  // autogenerated comparison link includes the preceding version and was
  // easy to mistake for the old version that the former capture path painted
  // over.  Hiding that optional history row leaves every 1.5.0.4 identifier
  // and every downloadable asset untouched.
  const changelog = Array.from(document.querySelectorAll('p')).find(element =>
    h.normal(element.innerText).startsWith('Full Changelog:'));
  if (changelog) changelog.style.display = 'none';
  const summary = Array.from(document.querySelectorAll('summary')).find(element =>
    h.visible(element) && h.normal(element.innerText).startsWith('Assets'));
  if (!summary) return false;
  const details = summary.closest('details');
  details.open = true;
  details.scrollIntoView({block: 'center', inline: 'nearest'});
  return true;
})()
""")
    _wait_for(
        browser,
        r"""(() => {
          const summary = Array.from(document.querySelectorAll('summary')).find(element =>
            (element.innerText || '').replace(/\s+/g, ' ').trim().startsWith('Assets'));
          const details = summary && summary.closest('details');
          return !!details && details.querySelectorAll(
            'a[href*="/releases/download/"], a[href*="/archive/refs/tags/"]'
          ).length >= 8;
        })()""",
        "all eight release assets",
        timeout=30,
    )
    # The lazy asset fragment changes the details height. Recenter after it is
    # present, then measure and capture the exact same layout.
    rect = browser.evaluate(r"""
(() => {
  const h = window.__tutorialGeometry;
  const summary = Array.from(document.querySelectorAll('summary')).find(element =>
    h.visible(element) && h.normal(element.innerText).startsWith('Assets'));
  const details = summary && summary.closest('details');
  if (!details) return null;
  details.scrollIntoView({block: 'center', inline: 'nearest'});
  return h.union([details]);
})()
""")
    time.sleep(0.5)
    # Scrolling can shift the viewport by a fraction after the first measured
    # layout, so obtain the final viewport-relative rectangle once more.
    rect = browser.evaluate(r"""
(() => {
  const h = window.__tutorialGeometry;
  const summary = Array.from(document.querySelectorAll('summary')).find(element =>
    h.visible(element) && h.normal(element.innerText).startsWith('Assets'));
  return h.union([summary.closest('details')]);
})()
""")
    _capture(browser, output / "github_release.png")
    return {"github_assets": _scaled_rect(rect, "GitHub release assets")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chrome", default=shutil.which("google-chrome"))
    args = parser.parse_args()
    if not args.chrome:
        raise RuntimeError("Google Chrome is required for web captures")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    port = _free_port()
    with tempfile.TemporaryDirectory(prefix="spacr_install_chrome_") as profile:
        process = subprocess.Popen([
            args.chrome,
            "--headless=new",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-gpu",
            "--hide-scrollbars",
            "--force-dark-mode",
            "--remote-allow-origins=*",
            f"--remote-debugging-port={port}",
            f"--user-data-dir={profile}",
            "about:blank",
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        browser = None
        try:
            browser = DevTools(_page_websocket(port))
            browser.call("Page.enable")
            browser.call("Runtime.enable")
            browser.call("Network.enable")
            browser.call("Network.setCacheDisabled", {"cacheDisabled": True})
            browser.call("Emulation.setDeviceMetricsOverride", {
                "width": VIEWPORT[0],
                "height": VIEWPORT[1],
                "deviceScaleFactor": DEVICE_SCALE,
                "mobile": False,
                "screenWidth": VIEWPORT[0],
                "screenHeight": VIEWPORT[1],
            })
            browser.call("Emulation.setEmulatedMedia", {
                "features": [{"name": "prefers-color-scheme", "value": "dark"}],
            })
            geometry = {"frame_size": list(FRAME_SIZE)}
            geometry.update(_pypi(browser, output))
            geometry.update(_github_main(browser, output))
            geometry.update(_github_release(browser, output))
            geometry["release_version"] = RELEASE_VERSION
            geometry["urls"] = {
                "pypi": f"https://pypi.org/project/spacr/{RELEASE_VERSION}/",
                "github_main": "https://github.com/EinarOlafsson/spacr/tree/main",
                "github_release": (
                    "https://github.com/EinarOlafsson/spacr/releases/tag/"
                    f"v{RELEASE_VERSION}"
                ),
            }
            (output / "geometry.json").write_text(
                json.dumps(geometry, indent=2) + "\n"
            )
        finally:
            if browser is not None:
                browser.close()
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    for name in ("pypi.png", "github_main.png", "github_release.png",
                 "geometry.json"):
        print(output / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
