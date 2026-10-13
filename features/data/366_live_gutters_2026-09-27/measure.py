"""Measure actual nightly README tile gutters in Chromium; never alter the page."""
import hashlib
import io
import json
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

root = Path('/mnt/wd4tb/scratch/spacr-completion/worktree')
out = Path(__file__).resolve().parent
url = 'https://github.com/EinarOlafsson/spacr/tree/nightly'
api = 'https://api.github.com/repos/EinarOlafsson/spacr/commits/nightly'
local = (root / 'README.rst').read_bytes()
source_sha = hashlib.sha256(local).hexdigest()
text = local.decode()
grid = text.split('.. spacr-workflow-begin', 1)[1].split('.. spacr-workflow-end', 1)[0]
labels = dict(re.findall(r'(?ms)^\.\. \|Module_([^|]+)\| image::[^\n]+\n.*?^   :alt: Open the ([^\n]+) API$', grid))
rows = []
section = None
lines = grid.splitlines()
for index, line in enumerate(lines):
    if index + 1 < len(lines) and re.fullmatch(r'\^{3,}', lines[index + 1]):
        section = line
    if line.startswith('| '):
        rows.append((section, [labels[key] for key in re.findall(r'\|Module_([^|]+)\|', line)]))
expected = [label for _, row in rows for label in row]
assert len(expected) == len(set(expected)) == 21
assert [(section, len(row)) for section, row in rows] == [('Core', 6), ('Data', 6), ('Data', 1), ('Tools', 5), ('Assays', 3)]
receipt = {'item': 366, 'url': url, 'started_utc': datetime.now(timezone.utc).isoformat(),
           'readme_sha256': source_sha, 'expected_rows': rows, 'accepted': False,
           'method': 'Actual GitHub page at two desktop widths. Visible tile rectangles combine DOM image geometry with alpha bounds of the exact loaded PNG bytes. Compare horizontal neighbors and vertical neighbors in the same section; headings between sections are intentional separators.',
           'tolerance_css_px': 1.0, 'viewports': [], 'images': {}}
selector = 'img[alt^="Open the "][alt$=" API"]'
(out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
try:
    with sync_playwright() as engine:
        browser = engine.chromium.launch(executable_path='/opt/google/chrome/chrome', headless=True,
                                          args=['--disable-gpu', '--disable-dev-shm-usage'])
        context = browser.new_context(device_scale_factor=1, locale='en-US')
        response = context.request.get(api, timeout=30000)
        assert response.ok, f'Nightly revision request: {response.status}'
        commit = response.json()['sha']
        receipt['nightly_commit_before'] = commit
        canonical = context.request.get(f'https://raw.githubusercontent.com/EinarOlafsson/spacr/{commit}/README.rst', timeout=30000)
        assert canonical.ok and hashlib.sha256(canonical.body()).hexdigest() == source_sha, 'Live README differs from inspected source'
        page = context.new_page()
        for width in (1440, 1024):
            page.set_viewport_size({'width': width, 'height': 1100})
            navigation = page.goto(url, wait_until='domcontentloaded', timeout=60000)
            assert navigation and navigation.ok, f'GitHub page status: {navigation.status if navigation else None}'
            page.wait_for_function('(s) => document.querySelectorAll(s).length === 21', arg=selector, timeout=60000)
            for tile in page.locator(selector).all():
                tile.scroll_into_view_if_needed()
            page.wait_for_function('(s) => document.querySelectorAll(s).length === 21 && [...document.querySelectorAll(s)].every(i => i.complete && i.naturalWidth > 0)', arg=selector, timeout=60000)
            page.evaluate('document.fonts.ready')
            page.locator(selector).first.scroll_into_view_if_needed()
            rectangles = page.locator(selector).evaluate_all('''images => images.map(image => {
                const b = image.getBoundingClientRect();
                const css = getComputedStyle(image);
                return {label: image.alt.slice(9,-4), src: image.currentSrc,
                        x:b.x+scrollX, y:b.y+scrollY, width:b.width, height:b.height,
                        naturalWidth:image.naturalWidth, naturalHeight:image.naturalHeight,
                        css:{verticalAlign:css.verticalAlign, margin:css.margin, display:css.display}};
            })''')
            assert [row['label'] for row in rectangles] == expected
            for row in rectangles:
                if row['src'] not in receipt['images']:
                    image_response = context.request.get(row['src'], timeout=30000)
                    assert image_response.ok, f'Loaded image unavailable: {row["src"]}'
                    data = image_response.body()
                    digest = hashlib.sha256(data).hexdigest()
                    (out / f'{digest}.png').write_bytes(data)
                    with Image.open(io.BytesIO(data)) as image:
                        bounds = image.convert('RGBA').getchannel('A').getbbox()
                        size = image.size
                    assert bounds
                    receipt['images'][row['src']] = {'sha256':digest, 'alpha_bounds':bounds, 'size':size}
                artifact = receipt['images'][row['src']]
                assert artifact['size'] == [row['naturalWidth'], row['naturalHeight']] or tuple(artifact['size']) == (row['naturalWidth'], row['naturalHeight'])
                left, top, right, bottom = artifact['alpha_bounds']
                sx, sy = row['width']/row['naturalWidth'], row['height']/row['naturalHeight']
                row['visible'] = {'left':row['x']+left*sx, 'top':row['y']+top*sy,
                                  'right':row['x']+right*sx, 'bottom':row['y']+bottom*sy}
            by_label = {row['label']:row for row in rectangles}
            horizontal, vertical = [], []
            for section, row_labels in rows:
                for first, second in zip(row_labels, row_labels[1:]):
                    a,b = by_label[first]['visible'],by_label[second]['visible']
                    assert abs(a['top']-b['top']) <= 1, 'A six-column source row wrapped on the live page'
                    horizontal.append({'section':section,'from':first,'to':second,'gap_css_px':b['left']-a['right']})
            for (first_section, first_row),(second_section, second_row) in zip(rows,rows[1:]):
                if first_section == second_section:
                    for first,second in zip(first_row,second_row):
                        a,b = by_label[first]['visible'],by_label[second]['visible']
                        assert abs(a['left']-b['left']) <= 1
                        vertical.append({'section':first_section,'from':first,'to':second,'gap_css_px':b['top']-a['bottom']})
            assert horizontal and vertical
            values = [pair['gap_css_px'] for pair in horizontal+vertical]
            left = min(row['x'] for row in rectangles)
            top = min(row['y'] for row in rectangles)
            right = max(row['x']+row['width'] for row in rectangles)
            bottom = max(row['y']+row['height'] for row in rectangles)
            screenshot = out / f'nightly-module-grid-{width}.png'
            measured = {'viewport_width':width, 'rectangles':rectangles,'horizontal_pairs':horizontal,
                        'vertical_pairs':vertical,'horizontal_median':statistics.median(pair['gap_css_px'] for pair in horizontal),
                        'vertical_median':statistics.median(pair['gap_css_px'] for pair in vertical),
                        'minimum':min(values),'maximum':max(values),'spread_css_px':max(values)-min(values),
                        'screenshot':str(screenshot),
                        'accepted':min(values)>=0 and max(values)-min(values)<=1.0}
            receipt['viewports'].append(measured)
            page.screenshot(path=str(screenshot), full_page=True,
                            clip={'x':max(0,left-8),'y':max(0,top-12),'width':right-left+16,'height':bottom-top+24})
            measured['screenshot_sha256'] = hashlib.sha256(screenshot.read_bytes()).hexdigest()
        after = context.request.get(api, timeout=30000)
        assert after.ok
        receipt['nightly_commit_after'] = after.json()['sha']
        assert receipt['nightly_commit_after'] == commit, 'Nightly moved during measurement'
        receipt['accepted'] = all(view['accepted'] for view in receipt['viewports'])
        browser.close()
        assert receipt['accepted'], 'Measured horizontal and vertical gutters differ'
except Exception as error:
    receipt['error'] = str(error)
    raise
finally:
    receipt['finished_utc'] = datetime.now(timezone.utc).isoformat()
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
