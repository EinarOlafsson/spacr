"""Check actual human-readable exports independently of spaCR's exporters."""
from html.parser import HTMLParser
import math


def expected_cells(row, empty_flag):
    def number(value):
        if not math.isfinite(value):
            return '—'
        return format(value, '.3g' if value and (abs(value) < .001 or abs(value) >= 1e5) else '.4g')
    return [str(row['rank']), row['gene'], row['name'], number(row['effect']),
            number(row['p_value']), number(row['q_value']),
            f"{row['n_agree']}/{row['n_guides']}", number(row['agreement']),
            row['flags'].replace(';', ', ') or empty_flag]


def check_markdown(text, expected):
    rows = [[cell.strip() for cell in line.strip('|').split('|')]
            for line in text.splitlines() if line.startswith('|')][2:]
    if rows != [expected_cells(row, '-') for row in expected]:
        raise ValueError('Markdown rows, identities or rounded values differ')
    return len(rows) * 9


class TableReader(HTMLParser):
    def __init__(self):
        super().__init__(); self.rows = []; self.row = []; self.cell = None
        self.external_or_script = False

    def handle_starttag(self, tag, attrs):
        if tag == 'script' or any(key in ('src', 'href') for key, _ in attrs):
            self.external_or_script = True
        if tag == 'tr': self.row = []
        if tag == 'td': self.cell = ''

    def handle_data(self, value):
        if self.cell is not None: self.cell += value

    def handle_endtag(self, tag):
        if tag == 'td' and self.cell is not None:
            self.row.append(self.cell); self.cell = None
        if tag == 'tr' and self.row: self.rows.append(self.row)


def check_html(text, expected):
    parser = TableReader(); parser.feed(text)
    if parser.rows != [expected_cells(row, '') for row in expected]:
        raise ValueError('HTML rows, identities or rounded values differ')
    if parser.external_or_script:
        raise ValueError('Expected script-free self-contained HTML')
    return len(parser.rows) * 9
