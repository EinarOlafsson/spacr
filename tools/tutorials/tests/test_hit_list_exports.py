"""Positive tables paired with actual changed cells, omissions and script tags."""
from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hit_list_exports import check_markdown, check_html

ROW = dict(rank=1, gene='100', name='100', effect=.25, p_value=.01,
           q_value=.03, n_agree=2, n_guides=3, agreement=2/3, flags='control')
CELLS = ['1', '100', '100', '0.25', '0.01', '0.03', '2/3', '0.6667', 'control']
MD = '| rank | gene | name | effect | p | q | guides | agree | flags |\n' + '|---' * 9 + '|\n| ' + ' | '.join(CELLS) + ' |\n'
HTML = '<table><tr>' + ''.join('<td>'+cell+'</td>' for cell in CELLS) + '</tr></table>'


def test_markdown_positive_and_changed_value():
    assert check_markdown(MD, [ROW]) == 9
    with pytest.raises(ValueError, match='Markdown rows'):
        check_markdown(MD.replace('0.25', '-0.25'), [ROW])


def test_html_positive_and_missing_row():
    assert check_html(HTML, [ROW]) == 9
    with pytest.raises(ValueError, match='HTML rows'):
        check_html('<table></table>', [ROW])


def test_html_positive_and_added_script():
    assert check_html(HTML, [ROW]) == 9
    with pytest.raises(ValueError, match='script-free'):
        check_html(HTML + '<script>alert(1)</script>', [ROW])
