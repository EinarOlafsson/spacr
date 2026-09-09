#!/usr/bin/env python3
"""Read an accepted Report HTML in real Chrome; never generate or rewrite it.

Run only after the GUI recorder has succeeded. Supply its receipt and SHA256,
and a new private output directory. This standalone browser step neither clicks
the Report Open button nor certifies science, publication or human readability.
No application/Qt import, browser download, web server or network is needed.
"""
from __future__ import annotations

import argparse
import hashlib
from html.parser import HTMLParser
import importlib.util
import json
import math
from pathlib import Path
import re
import struct
import time
from urllib.parse import unquote, urlsplit


_SPEC = importlib.util.spec_from_file_location(
    '_report_browser_source_guards', Path(__file__).with_name('capture_report.py'))
_REPORT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_REPORT)
VIEWPORT = {'width': 1920, 'height': 1080}
SCALE = 2
MAX_FRAMES = 64


def _normal(text):
    return ' '.join(text.split())


def _read_bound(path, expected, maximum=8 * 1024 * 1024):
    path = Path(path)
    if (path.is_symlink() or not path.is_file() or path.stat().st_size > maximum
            or not re.fullmatch('[0-9a-f]{64}', str(expected))):
        raise RuntimeError('Expected a bounded regular file and explicit SHA256')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise RuntimeError('Bound file SHA256 differs: ' + str(path))
    return raw


class _StaticHTML(HTMLParser):
    """Expected section text and complete tables, including closed details.

    This vector-only report must contain no active content or resource loads.
    Native fragment links and the existing inline stylesheet are allowed.
    """
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.sections, self.section = [], None
        self.table, self.row, self.cell = None, None, None
        self.caption, self.style, self.forbidden = None, False, []

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if tag in ('script', 'iframe', 'object', 'embed', 'link', 'base', 'img', 'form'):
            self.forbidden.append(tag)
        if (any(key.lower().startswith('on') for key in attrs)
                or 'src' in attrs or 'srcset' in attrs
                or ('href' in attrs and not attrs['href'].startswith('#'))
                or (tag == 'meta' and attrs.get('http-equiv', '').lower() == 'refresh')):
            self.forbidden.append('active attribute')
        if re.search(r'@import|url\s*\(', attrs.get('style', ''), re.I):
            self.forbidden.append('style resource')
        if tag == 'style':
            self.style = True
        if tag == 'section':
            if self.section is not None:
                raise RuntimeError('Nested report sections are outside this profile')
            self.section = {'key': attrs.get('id'), 'classes': attrs.get('class', '').split(),
                            'text': [], 'tables': []}
            self.sections.append(self.section)
        if tag == 'table' and self.section is not None:
            if self.table is not None:
                raise RuntimeError('Nested report tables are outside this profile')
            self.table = {'caption': '', 'rows': []}
            self.section['tables'].append(self.table)
        if tag == 'caption' and self.table is not None:
            self.caption = []
        if tag == 'tr' and self.table is not None:
            self.row = []
        if tag in ('th', 'td') and self.row is not None:
            self.cell = []

    def handle_data(self, text):
        if self.style and re.search(r'@import|url\s*\(', text, re.I):
            self.forbidden.append('stylesheet resource')
        if self.section is not None:
            self.section['text'].append(text)
        if self.cell is not None:
            self.cell.append(text)
        if self.caption is not None:
            self.caption.append(text)

    def handle_endtag(self, tag):
        if tag == 'style':
            self.style = False
        if tag in ('th', 'td') and self.cell is not None:
            self.row.append(_normal(''.join(self.cell)))
            self.cell = None
        if tag == 'tr' and self.row is not None:
            self.table['rows'].append(self.row)
            self.row = None
        if tag == 'caption' and self.caption is not None:
            self.table['caption'] = _normal(''.join(self.caption))
            self.caption = None
        if tag == 'table':
            self.table = None
        if tag == 'section' and self.section is not None:
            self.section['text'] = _normal(''.join(self.section['text']))
            self.section = None


def _expected_sections(html):
    parsed = _StaticHTML()
    parsed.feed(html)
    parsed.close()
    if parsed.forbidden or parsed.section is not None:
        raise RuntimeError('HTML contains active/resource content or incomplete sections')
    if [s['key'] for s in parsed.sections] != _REPORT.CORE_SECTIONS:
        raise RuntimeError('HTML is not the approved eight-section profile')
    return parsed.sections


def _allowed_request(url, html_path):
    """Only the exact accepted local document, optionally a native fragment."""
    parsed = urlsplit(url)
    return (parsed.scheme == 'file' and not parsed.netloc and not parsed.query
            and Path(unquote(parsed.path)) == Path(html_path).resolve())


def _verify_dom(dom, expected, summary, facts):
    """Pure checks on read-only DOM observations; never repair mismatches."""
    if dom['sections'] != expected:
        raise RuntimeError('Browser section text, classes, table rows or captions differ from bound HTML')
    if dom['title'] != summary['title'] or dom['source'] != summary['src']:
        raise RuntimeError('Browser title or source identity differs from accepted receipt')
    if ('complete' not in dom['banner_classes']
            or _normal(summary['status_detail']) not in dom['banner_text']
            or '0 of 6 figure(s) are embedded in this file.' not in dom['banner_text']):
        raise RuntimeError('Browser completion or vector-only disclosure differs')
    if dom['toc'] != ['#' + key for key in _REPORT.CORE_SECTIONS] or dom['images']:
        raise RuntimeError('Browser contents or image inventory differs')
    for section, accepted in zip(dom['sections'], summary['sections']):
        if ('missing' in section['classes']) != (accepted['status'] == 'missing'):
            raise RuntimeError('Browser missing-section disclosure differs')
    sections = {section['key']: section for section in dom['sections']}
    for key in ('segmentation_qc', 'plate_qc'):
        if ('missing' not in sections[key]['classes']
                or 'not available' not in sections[key]['text']):
            raise RuntimeError('Browser fails to disclose unavailable QC')
    # Compare the complete six-PDF data table, not only name presence.
    pdf_rows = [[name, 'vector — not embeddable'] for name in facts['vector_figures']]
    if [row for table in sections['figures']['tables'] for row in table['rows'][1:]] != pdf_rows:
        raise RuntimeError('Browser six-PDF identities differ from the source profile')
    for relative, counts in facts['database_counts'].items():
        candidates = [table for table in sections['statistics']['tables']
                      if table['caption'] == relative + ' — tables and row counts']
        expected_rows = [['table', 'rows']] + [[name, str(count)] for name, count in counts.items()]
        if len(candidates) != 1 or candidates[0]['rows'] != expected_rows:
            raise RuntimeError('Browser exact SQLite table identities/counts differ from source')
    return True


def _png_geometry(path):
    with Path(path).open('rb') as stream:
        header = stream.read(24)
    if len(header) != 24 or header[:8] != b'\x89PNG\r\n\x1a\n' or header[12:16] != b'IHDR':
        raise RuntimeError('Browser screenshot is not a PNG')
    dimensions = list(struct.unpack('>II', header[16:24]))
    if dimensions != [3840, 2160]:
        raise RuntimeError('Browser screenshot is not genuine 3840 by 2160 output')
    return dimensions


def _require_accepted_receipt(receipt):
    if (receipt.get('accepted') is not True or receipt.get('lesson') != '29_report'
            or receipt.get('original_preserved') is not True
            or receipt.get('private_material_files_unchanged') is not True
            or receipt.get('html', {}).get('accepted') is not True
            or receipt.get('html', {}).get('section_count') != 8
            or receipt.get('html', {}).get('embedded_images') != 0
            or receipt.get('html', {}).get('external_dependencies') is not False
            or receipt.get('synthetic_status_created') is not False
            or receipt.get('source_pipeline_rerun') is not False
            or receipt.get('active_jobs') != []
            or not receipt.get('job_results') or any(value is not True for value in receipt['job_results'])):
        raise RuntimeError('A complete accepted real GUI Report receipt is required')


def _verify_snapshots(receipt, private_now, original_now, sidecar_changes):
    _REPORT.require_unchanged(receipt['source_files_after'], private_now)
    _REPORT.require_unchanged(receipt['original_source_files_before'], receipt['source_files_before'])
    _REPORT.require_unchanged(receipt['original_source_files_before'], original_now)
    if sidecar_changes != receipt['private_sqlite_sidecar_changes']:
        raise RuntimeError('Private SQLite sidecar changes differ from the accepted receipt')


def _load_accepted(receipt_path, receipt_sha256):
    receipt_path = Path(receipt_path).resolve()
    receipt = json.loads(_read_bound(receipt_path, receipt_sha256))
    _require_accepted_receipt(receipt)
    source, output = Path(receipt['source']).resolve(), Path(receipt['output'])
    original = Path(receipt['original_source']).resolve()
    if source == original or source.is_relative_to(original) or original.is_relative_to(source):
        raise RuntimeError('Report source must be an isolated private clone, not the original')
    if (output.is_symlink() or output.resolve().parent != Path(receipt['destination']).resolve()
            or output.suffix.lower() != '.html' or output.resolve().is_relative_to(source)):
        raise RuntimeError('Receipt must name a separate generated HTML file')
    output = output.resolve()
    html = _read_bound(output, receipt['output_sha256']).decode('utf-8')
    before = _REPORT.snapshot_source(source)
    changes = _REPORT.verify_private_sqlite_changes(source, receipt['source_files_before'], before)
    _verify_snapshots(receipt, before, _REPORT.snapshot_source(original), changes)
    facts = _REPORT.read_source_facts(source)
    if facts != receipt['source_facts']:
        raise RuntimeError('The independently read source profile differs from the receipt')
    _REPORT.verify_report_summary(receipt['report'], source, facts)
    _REPORT.verify_html(html, receipt['report'], facts)
    return receipt, source, output, before, _expected_sections(html)


_DOM = """() => {
  const norm = text => text.replace(/\\s+/g, ' ').trim();
  return {
    title: document.querySelector('header.doc h1')?.textContent.trim(),
    source: document.querySelector('header.doc .meta code')?.textContent.trim(),
    banner_classes: [...document.querySelector('.banner').classList],
    banner_text: norm(document.querySelector('.banner').textContent),
    toc: [...document.querySelectorAll('nav.toc a')].map(n => n.getAttribute('href')),
    images: document.querySelectorAll('img').length,
    sections: [...document.querySelectorAll('section')].map(s => ({
      key: s.id, classes: [...s.classList], text: norm(s.textContent),
      tables: [...s.querySelectorAll('table')].map(t => ({
        caption: norm(t.querySelector('caption')?.textContent || ''),
        rows: [...t.querySelectorAll('tr')].map(r => [...r.querySelectorAll('th,td')].map(c => norm(c.textContent)))
      }))
    }))
  };
}"""

_LAYOUT = """() => ({
  x: window.scrollX, y: window.scrollY,
  width: window.innerWidth, height: window.innerHeight, scale: window.devicePixelRatio,
  document_width: document.documentElement.scrollWidth,
  document_height: document.documentElement.scrollHeight,
  body_font_px: parseFloat(getComputedStyle(document.body).fontSize),
  visible_sections: [...document.querySelectorAll('section')].map(n => {
    const r = n.getBoundingClientRect();
    return {id: n.id, x: r.x, y: r.y, width: r.width, height: r.height};
  }).filter(r => r.y < innerHeight && r.y + r.height > 0),
  horizontal_overflows: [...document.querySelectorAll('body *')].filter(n => {
    const r = n.getBoundingClientRect();
    return r.width > 0 && r.height > 0 && n.scrollWidth > n.clientWidth + 2
      && ['auto','scroll'].includes(getComputedStyle(n).overflowX);
  }).map(n => ({tag: n.tagName, section: n.closest('section')?.id || null,
                width: n.clientWidth, scroll_width: n.scrollWidth}))
})"""


def capture_report_browser(receipt_path, receipt_sha256, output_dir, *,
                           chrome='/opt/google/chrome/chrome', timeout=120):
    """Write only a fresh browser inventory/receipt and real viewport PNGs."""
    if not math.isfinite(float(timeout)) or timeout <= 0:
        raise ValueError('Browser timeout must be positive and finite')
    started = time.monotonic()
    receipt_path = Path(receipt_path).resolve()
    receipt, source, html_path, before, expected = _load_accepted(receipt_path, receipt_sha256)
    destination = Path(output_dir)
    if (destination.exists() or destination.is_symlink()
            or destination.resolve().is_relative_to(source)
            or destination.resolve().is_relative_to(Path(receipt['original_source']).resolve())
            or destination.resolve().is_relative_to(html_path.parent)
            or destination.resolve() == receipt_path.parent):
        raise ValueError('Use a new separate private browser output directory')
    destination.mkdir()  # Parent must already exist; never overwrite a capture.
    evidence = {'accepted': False, 'browser_reviewed': False, 'lesson': '29_report',
                'native_human_publish': False, 'native_speaker_signoff': False,
                'published': False, 'open_control_clicked': False,
                'scope': 'Technical Chrome rendering, source-bound DOM identities and screenshot geometry only; not scientific validation or human publication approval.',
                'receipt': str(receipt_path), 'receipt_sha256': receipt_sha256,
                'html': str(html_path), 'html_sha256': receipt['output_sha256'],
                'source': str(source), 'original_source': receipt['original_source'],
                'source_profile': receipt['source_facts'],
                'verification_code_sha256': {
                    'capture_report_browser.py': _REPORT._digest(__file__),
                    'capture_report.py': _REPORT._digest(Path(__file__).with_name('capture_report.py'))},
                'private_sqlite_sidecar_changes_during_gui_collection': receipt['private_sqlite_sidecar_changes'],
                'collection_zero_file_writes_claimed': False, 'report_recomputed': False,
                'viewport_css': VIEWPORT, 'device_scale_factor': SCALE,
                'html_css_dom_injected': False, 'frames': [], 'requests': [],
                'blocked_requests': [], 'javascript_errors': [], 'console_errors': [],
                'request_failures': [], 'downloads': [], 'layout_observations': [],
                'native_scroll_actions': [], 'native_disclosure_actions': []}

    def write(name, value):
        (destination / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')

    def remaining():
        left = timeout - (time.monotonic() - started)
        if left <= 0:
            raise TimeoutError('Report browser exceeded its lifecycle timeout')
        return max(1, int(left * 1000))

    write('browser_review.json', evidence)
    try:
        from playwright.sync_api import sync_playwright
        if not Path(chrome).is_file():
            raise RuntimeError('Existing Chrome executable is required; no download is permitted')
        with sync_playwright() as engine:
            browser = engine.chromium.launch(executable_path=chrome, headless=True,
                timeout=remaining(), args=['--disable-background-networking', '--disable-component-update',
                    '--disable-sync', '--no-first-run', '--no-default-browser-check',
                    '--host-resolver-rules=MAP * ~NOTFOUND'])
            try:
                evidence['browser_version'] = browser.version
                evidence['chrome_executable'] = str(Path(chrome).resolve())
                context = browser.new_context(viewport=VIEWPORT, device_scale_factor=SCALE,
                    color_scheme='light', offline=True, service_workers='block', accept_downloads=False)

                def route(request_route):
                    url = request_route.request.url
                    if _allowed_request(url, html_path):
                        request_route.continue_()
                    else:
                        evidence['blocked_requests'].append(url)
                        request_route.abort('blockedbyclient')

                context.route('**/*', route)
                context.on('request', lambda r: evidence['requests'].append(r.url))
                context.on('requestfailed', lambda r: evidence['request_failures'].append(
                    {'url': r.url, 'failure': r.failure}))
                page = context.new_page()
                page.on('pageerror', lambda error: evidence['javascript_errors'].append(str(error)))
                page.on('console', lambda message: evidence['console_errors'].append(message.text)
                        if message.type == 'error' else None)
                page.on('download', lambda download: (evidence['downloads'].append(download.suggested_filename),
                                                     download.cancel()))
                page.goto(html_path.as_uri(), wait_until='load', timeout=remaining())
                if page.url != html_path.as_uri():
                    raise RuntimeError('Chrome did not open the exact accepted local HTML')
                initial_dom = page.evaluate(_DOM)
                _verify_dom(initial_dom, expected, receipt['report'], receipt['source_facts'])
                write('browser_dom.json', initial_dom)
                evidence['browser_dom_sha256'] = _REPORT._digest(destination / 'browser_dom.json')

                def layout():
                    observed = page.evaluate(_LAYOUT)
                    if ([observed['width'], observed['height'], observed['scale']] != [1920, 1080, 2]
                            or observed['body_font_px'] < 14
                            or observed['document_height'] > 100000):
                        raise RuntimeError('Unexpected browser viewport, scale, font size or document size')
                    return observed

                def scroll_to(y):
                    for _ in range(32):
                        remaining()
                        observed = layout()
                        target = max(0, min(y, observed['document_height'] - observed['height']))
                        delta = target - observed['y']
                        if abs(delta) <= 2:
                            return
                        page.mouse.move(1800, 900)
                        wheel_y = max(-850, min(850, delta))
                        page.mouse.wheel(0, wheel_y)
                        page.wait_for_timeout(min(100, remaining()))
                        evidence['native_scroll_actions'].append({'before_y': observed['y'],
                            'wheel_y': wheel_y, 'after_y': layout()['y']})
                    raise RuntimeError('Actual browser wheel did not reach the requested document region')

                def frame(label):
                    if len(evidence['frames']) >= MAX_FRAMES:
                        raise RuntimeError('Browser frame budget exceeded')
                    observed = layout()
                    name = f"{len(evidence['frames']) + 1:02d}_{label}.png"
                    image = destination / name
                    page.screenshot(path=str(image), full_page=False, scale='device', timeout=remaining())
                    item = {'name': name, 'path': str(image.resolve()), 'sha256': _REPORT._digest(image),
                            'bytes': image.stat().st_size, 'dimensions': _png_geometry(image),
                            'viewport_css': VIEWPORT, 'device_scale_factor': SCALE,
                            'scroll_css': [observed['x'], observed['y']]}
                    evidence['frames'].append(item)
                    evidence['layout_observations'].append({'frame': name, **observed})
                    write('frames.json', {'kind': 'standalone_chrome_report', 'frames': evidence['frames']})

                frame('report_top_completion_and_contents')
                for key in _REPORT.CORE_SECTIONS:
                    # Only native mouse scrolling and summary clicks alter view state.
                    section = page.locator('section#' + key)
                    bounds = section.evaluate('(n) => ({top: n.getBoundingClientRect().top + scrollY, bottom: n.getBoundingClientRect().bottom + scrollY})')
                    scroll_to(bounds['top'] - 30)
                    frame(key)
                    while bounds['bottom'] > layout()['y'] + VIEWPORT['height'] - 30:
                        old_y = layout()['y']
                        scroll_to(old_y + 850)
                        if layout()['y'] <= old_y + 2:
                            raise RuntimeError('Unable to show the remainder of a report section')
                        frame(key + '_continued')
                    # A bounded native settings disclosure, if one exists. Other
                    # closed tables are DOM-checked but not claimed fully viewed.
                    disclosures = section.locator('details > summary')
                    if key == 'settings' and disclosures.count():
                        first = disclosures.first
                        summary_y = first.evaluate('(n) => n.getBoundingClientRect().top + scrollY')
                        scroll_to(summary_y - 80)
                        first.click(timeout=remaining())
                        if not first.evaluate('(n) => n.parentElement.open'):
                            raise RuntimeError('Native settings disclosure did not open')
                        evidence['native_disclosure_actions'].append({'section': key,
                            'summary': first.inner_text(), 'action': 'opened by native click'})
                        frame('settings_first_native_disclosure')
                        first.click(timeout=remaining())
                        if first.evaluate('(n) => n.parentElement.open'):
                            raise RuntimeError('Native settings disclosure did not close')
                        evidence['native_disclosure_actions'].append({'section': key,
                            'summary': first.inner_text(), 'action': 'closed by native click'})
                final_dom = page.evaluate(_DOM)
                _verify_dom(final_dom, expected, receipt['report'], receipt['source_facts'])
                if final_dom != initial_dom:
                    raise RuntimeError('Report content changed during native viewing')
                evidence['dom_verified'] = True
                evidence['dom_sections'] = _REPORT.CORE_SECTIONS
                evidence['horizontal_overflow'] = any(
                    item['document_width'] > item['width'] + 2 or item['horizontal_overflows']
                    for item in evidence['layout_observations'])
                if evidence['horizontal_overflow']:
                    raise RuntimeError('Real report has horizontal overflow; retained frames document it without CSS repair')
                if (evidence['javascript_errors'] or evidence['console_errors']
                        or evidence['blocked_requests'] or evidence['request_failures']
                        or evidence['downloads']
                        or any(not _allowed_request(url, html_path) for url in evidence['requests'])):
                    raise RuntimeError('Browser errors, disallowed resource requests or downloads occurred')
                context.close()
            finally:
                browser.close()
        _read_bound(receipt_path, receipt_sha256)
        _read_bound(html_path, receipt['output_sha256'])
        _REPORT.require_unchanged(before, _REPORT.snapshot_source(source))
        _REPORT.require_unchanged(receipt['original_source_files_before'],
                                  _REPORT.snapshot_source(receipt['original_source']))
        evidence.update(accepted=True, browser_reviewed=True, original_preserved=True,
                        private_source_exact_post_generation_state_preserved=True,
                        frames_manifest_sha256=_REPORT._digest(destination / 'frames.json'),
                        receipt_unchanged=True, html_unchanged=True,
                        reason='Unmodified accepted HTML rendered in Chrome; eight sections, missing QC, actual completion, exact table/PDF identities and HiDPI frames verified. Human publication review remains pending.')
    except Exception as error:
        evidence.update(accepted=False, browser_reviewed=False, reason=str(error))
        raise
    finally:
        evidence['elapsed_seconds'] = round(time.monotonic() - started, 3)
        write('browser_review.json', evidence)
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', required=True, type=Path)
    parser.add_argument('--receipt-sha256', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--chrome', default='/opt/google/chrome/chrome')
    parser.add_argument('--timeout', type=float, default=120)
    args = parser.parse_args()
    evidence = capture_report_browser(args.receipt, args.receipt_sha256, args.output,
                                      chrome=args.chrome, timeout=args.timeout)
    print(json.dumps({'accepted': evidence['accepted'], 'frames': len(evidence['frames']),
                      'output': str(args.output)}, indent=2))


if __name__ == '__main__':
    main()
