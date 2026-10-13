"""Pagination must complete metadata lookup without partial or cross-origin reads."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from spacr import plate_qc as qc


def test_later_page_failure_preserves_existing_measurement_outputs(tmp_path, capsys):
    from spacr.measure import _run_plate_barcode_step

    merged = tmp_path / 'plate1' / 'merged'
    merged.mkdir(parents=True)
    np.save(merged / 'plate1_A01_1.npy', np.zeros((2, 2, 1)))
    output = merged.parent / 'measurements'
    output.mkdir()
    originals = {'plate_map_lims.csv': b'previous complete map\n',
                 'plate_barcode_mismatches.csv': b'previous mismatch report\n',
                 'measurements.db': b'existing measurement data\n'}
    for name, content in originals.items():
        (output / name).write_bytes(content)
    settings = {'src': str(merged), 'plate_barcode_source': 'https://example.test/plates',
                'profiling_metadata': '', 'viability_plate_map': '', 'timelapse': False}
    before = settings.copy()
    calls = []

    def fetch(url, headers):
        calls.append(url)
        if len(calls) == 2:
            raise OSError('second page unavailable')
        return json.dumps({'wells': [{'well': 'A01', 'compound': 'new', 'concentration': 1}],
                           'next': '?page=2'})

    assert _run_plate_barcode_step(settings, fetch=fetch) is None
    assert len(calls) == 2
    assert settings == before
    assert {p.name: p.read_bytes() for p in output.iterdir()} == originals
    assert 'second page unavailable' in capsys.readouterr().out


@pytest.mark.parametrize('next_field', ['next', 'next_url', 'links', '@odata.nextLink'])
def test_multiple_pages_keep_order_metadata_and_each_barcode(next_field):
    calls = []

    def fetch(url, headers):
        calls.append(url)
        if 'page=2' in url:
            return json.dumps({'wells': [{'well': 'A02'}], 'next': None})
        link = {'next': {'href': '?page=2'}} if next_field == 'links' else '?page=2'
        return json.dumps({'operator': 'Ada', 'wells': [{'well': 'A01'}], next_field: link})

    frame = qc._lims_records('https://example.test/plates/{barcode}', ['BC1', 'BC2', 'BC1'], fetch=fetch)
    assert calls == ['https://example.test/plates/BC1', 'https://example.test/plates/BC1?page=2',
                     'https://example.test/plates/BC2', 'https://example.test/plates/BC2?page=2']
    assert frame[['barcode', 'well', 'operator']].values.tolist() == [
        ['BC1', 'A01', 'Ada'], ['BC1', 'A02', 'Ada'], ['BC2', 'A01', 'Ada'], ['BC2', 'A02', 'Ada']]
    assert not set(frame.columns) & qc._LIMS_PAGING_FIELDS


@pytest.mark.parametrize('link', [
    'https://other.test/page2', 'http://example.test/page2', '//other.test/page2',
    'https://example.test:8443/page2', 'https://user:password@example.test/page2',
    'file:///tmp/records', 42, {'href': []}, {'cursor': 'opaque'}, False,
])
def test_bad_links_fail_before_a_second_request_or_returning_partial_rows(link, monkeypatch):
    monkeypatch.setenv('TEST_LIMS_TOKEN', 'test-only-secret')
    calls = []

    def fetch(url, headers):
        calls.append((url, headers))
        return json.dumps({'items': [{'well': 'A01'}], 'next': link})

    with pytest.raises(ValueError):
        qc._lims_records('https://example.test/plates/{barcode}', ['BC1'],
                         token_env='TEST_LIMS_TOKEN', fetch=fetch)
    assert len(calls) == 1
    assert calls[0][1]['Authorization'] == 'Bearer test-only-secret'


def test_cycles_are_rejected_before_fetching_a_page_twice():
    calls = []

    def fetch(url, headers):
        calls.append(url)
        return json.dumps({'items': [{'well': 'A01'}], 'next': '/plates/BC1'})

    with pytest.raises(ValueError, match='repeats'):
        qc._lims_records('https://example.test/plates/{barcode}', ['BC1'], fetch=fetch)
    assert len(calls) == 1


@pytest.mark.parametrize('limit, value, message', [
    ('_LIMS_MAX_PAGE_BYTES', 10, 'byte limit'),
    ('_LIMS_MAX_TOTAL_BYTES', 120, 'byte limit'),
    ('_LIMS_MAX_PAGES', 1, 'page limit'),
    ('_LIMS_MAX_RECORDS', 1, 'record limit'),
])
def test_bounded_collection_never_silently_truncates(limit, value, message, monkeypatch):
    monkeypatch.setattr(qc, limit, value)
    calls = []

    def fetch(url, headers):
        calls.append(url)
        return json.dumps({'items': [{'well': 'A01'}], 'operator': 'A' * 10,
                           'next': f'?page={len(calls) + 1}'})

    with pytest.raises(ValueError, match=message):
        qc._lims_records('https://example.test/plates/{barcode}', ['BC1'], fetch=fetch)
    assert len(calls) <= 2


def test_conflicting_pagination_urls_are_not_guessed():
    with pytest.raises(ValueError, match='conflicting'):
        qc._lims_records('https://example.test/plates/{barcode}', ['BC1'], fetch=lambda *_:
                         json.dumps({'wells': [], 'next': '?page=2', 'next_url': '?page=3'}))


def test_real_http_paging_populates_both_imaged_wells(tmp_path, monkeypatch):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            calls.append((self.path, self.headers.get('Authorization')))
            data = ({'wells': [{'well': 'A02', 'compound': 'drug'}], 'next': None}
                    if 'page=2' in self.path else
                    {'strain': 'RH', 'wells': [{'well': 'A01', 'compound': 'control'}],
                     'next': '?page=2'})
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())

        def log_message(self, *args):
            pass

    src = tmp_path / 'plate1' / 'merged'
    src.mkdir(parents=True)
    for well in ('A01', 'A02'):
        np.save(src / f'plate1_{well}_1.npy', np.zeros((2, 2, 1)))
    server = HTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv('TEST_LIMS_TOKEN', 'test-only-secret')
    try:
        plate_map, mismatches = qc._link_plate_barcodes(
            str(src), f'http://127.0.0.1:{server.server_port}/plates/{{barcode}}',
            token_env='TEST_LIMS_TOKEN')
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert plate_map['compound'].tolist() == ['control', 'drug']
    assert plate_map['strain'].tolist() == ['RH', 'RH']
    assert mismatches.empty
    assert calls == [('/plates/plate1', 'Bearer test-only-secret'),
                     ('/plates/plate1?page=2', 'Bearer test-only-secret')]


def test_http_redirect_cannot_forward_credentials_to_another_origin():
    destination_calls = []

    class Destination(BaseHTTPRequestHandler):
        def do_GET(self):
            destination_calls.append(self.headers.get('Authorization'))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'[]')

        def log_message(self, *args):
            pass

    destination = HTTPServer(('127.0.0.1', 0), Destination)

    class Redirect(Destination):
        def do_GET(self):
            self.send_response(302)
            self.send_header('Location', f'http://127.0.0.1:{destination.server_port}/')
            self.end_headers()

    redirect = HTTPServer(('127.0.0.1', 0), Redirect)
    threads = [threading.Thread(target=s.serve_forever, daemon=True) for s in (destination, redirect)]
    for thread in threads:
        thread.start()
    try:
        with pytest.raises(ValueError, match='redirect'):
            qc._default_lims_fetch(f'http://127.0.0.1:{redirect.server_port}/',
                                   {'Authorization': 'Bearer test-only-secret'})
    finally:
        for server, thread in zip((destination, redirect), threads):
            server.shutdown()
            server.server_close()
            thread.join()
    assert destination_calls == []


def test_next_link_uses_effective_url_after_same_origin_redirect():
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            calls.append(self.path)
            if self.path == '/plates/BC1':
                self.send_response(302)
                self.send_header('Location', '/pages/first')
                self.end_headers()
                return
            self.send_response(200)
            self.end_headers()
            data = ({'wells': [{'well': 'A01'}], 'next': 'second'}
                    if self.path == '/pages/first' else [{'well': 'A02'}])
            self.wfile.write(json.dumps(data).encode())

        def log_message(self, *args):
            pass

    server = HTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        frame = qc._lims_records(f'http://127.0.0.1:{server.server_port}/plates/{{barcode}}', ['BC1'])
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert calls == ['/plates/BC1', '/pages/first', '/pages/second']
    assert frame['well'].tolist() == ['A01', 'A02']
