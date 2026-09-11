"""Real-browser regression for long filenames, API names and source hashes."""
from html import escape
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright


STYLES = Path(__file__).resolve().parents[3] / 'docs/source/_extra/tutorials/styles.css'


@pytest.mark.parametrize('width', [320, 390, 768, 1440])
def test_prerequisite_is_visible_without_horizontal_scrolling(width):
    # This is the real grid and stylesheet, not a word-wrap property assertion.
    # The empty-prefix counterpart ensures the container itself fits as well.
    strings = ['Ready.', 'SHA256 ' + '9bbfd6de409ec501' * 5,
               'plate_1_unique_combinations.csv spacr.sequencing_qc.threshold_sweep',
               '参照ライブラリがなければ完全に欠けたガイドは分かりません。' * 4]
    with sync_playwright() as engine:
        browser = engine.chromium.launch(executable_path='/opt/google/chrome/chrome', headless=True)
        page = browser.new_page(viewport={'width': width, 'height': 844})
        for text in strings:
            page.set_content('<style>' + STYLES.read_text() + '</style>'
                '<main style="width:100%;padding:16px">'
                '<div class="prerequisite"><svg></svg><div><strong>Before you start</strong>'
                '<p id="prerequisite-copy">' + escape(text) + '</p></div></div></main>')
            result = page.evaluate('''() => {
                const p = document.querySelector('#prerequisite-copy'), r = p.getBoundingClientRect();
                return {width: document.documentElement.scrollWidth, viewport: innerWidth,
                    left: r.left, right: r.right, height: r.height, text: p.textContent,
                    overflowX: getComputedStyle(p).overflowX};
            }''')
            assert result['text'] == text and result['height'] > 0
            assert result['overflowX'] == 'visible'  # Do not pass by clipping the text.
            assert result['width'] <= width and 0 <= result['left'] < result['right'] <= width, result
        browser.close()
