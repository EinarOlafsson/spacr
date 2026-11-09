"""Item 750: the public dataset fetcher and its completeness check.

Every transfer here is served by a local HTTP server on 127.0.0.1 or read
from ``file://``; nothing reaches the network. The archive is a handful of
files of a few kilobytes.
"""
from __future__ import annotations

import ftplib
import hashlib
import http.server
import io
import json
import os
import threading
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from spacr import public_datasets as pdx


def _zip_bytes():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        archive.writestr('plate/a.tif', b'x' * 3000)
    return buffer.getvalue()


FILES = {
    'plate1/a.tif': bytes(range(256)) * 20,
    'plate1/b.tif': b'b' * 3001,
    'plate2/c.tif': b'c' * 1500,
    'images.zip': _zip_bytes(),
}


class _Archive(http.server.BaseHTTPRequestHandler):
    """Serves FILES under /Files/ with Range; can cut or refuse transfers."""

    files = FILES
    cut_after = {}
    fail_first = set()
    ignore_range = False
    index = {}
    requests = []

    def log_message(self, *args):
        """Keep the test output quiet."""

    def do_GET(self):
        """Serve one file, one range of it, an index page or an error."""
        type(self).requests.append((self.path, self.headers.get('Range')))
        if self.path in self.index:
            body = self.index[self.path].encode()
            self.send_response(200)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        name = self.path[len('/Files/'):]
        if name not in self.files:
            self.send_error(404)
            return
        if name in self.fail_first:
            type(self).fail_first = self.fail_first - {name}
            self.send_error(503)
            return
        data = self.files[name]
        start = 0
        span = self.headers.get('Range')
        if span and not self.ignore_range:
            start = int(span.split('=')[1].split('-')[0])
            if start >= len(data):
                self.send_response(416)
                self.end_headers()
                return
            self.send_response(206)
            self.send_header('Content-Range', f'bytes {start}-{len(data) - 1}/{len(data)}')
        else:
            self.send_response(200)
        self.send_header('Content-Length', str(len(data) - start))
        self.end_headers()
        limit = self.cut_after.get(name)
        if limit is not None:
            type(self).cut_after = {k: v for k, v in self.cut_after.items() if k != name}
            self.wfile.write(data[start:start + limit])
            self.wfile.flush()
            self.connection.shutdown(2)
            return
        self.wfile.write(data[start:])


@pytest.fixture
def archive():
    """A local archive server, reset for each test."""
    _Archive.cut_after, _Archive.fail_first = {}, set()
    _Archive.ignore_range, _Archive.index, _Archive.requests = False, {}, []
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), _Archive)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f'http://127.0.0.1:{server.server_address[1]}/Files/'
    server.shutdown()
    server.server_close()


def _manifest(tmp_path, *, sizes=True, md5=True, name='filelist.tsv'):
    rows = []
    for path, data in FILES.items():
        row = {'Files': path}
        if sizes:
            row['Size'] = len(data)
        if md5:
            row['md5'] = hashlib.md5(data).hexdigest()
        rows.append(row)
    path = tmp_path / name
    frame = pd.DataFrame(rows)
    if name.endswith('.json'):
        path.write_text(json.dumps({'files': rows}), encoding='utf-8')
    else:
        frame.to_csv(path, sep='\t', index=False)
    return path


def _settings(tmp_path, base, manifest, **extra):
    settings = {'dataset_archive': 'url', 'dataset_manifest': str(manifest),
                'dataset_base_url': base, 'dataset_dst': str(tmp_path / 'mirror'),
                'dataset_workers': 2, 'dataset_retries': 3}
    settings.update(extra)
    return settings


def _no_wait(_seconds):
    pass


def test_interrupted_twice_then_resumed_reports_nothing_missing(tmp_path, archive):
    """The spec's check: two interruptions, one resume, zero missing or short."""
    manifest = _manifest(tmp_path)
    _Archive.cut_after = {'plate1/a.tif': 1000, 'plate1/b.tif': 10}
    first = pdx.fetch_dataset(_settings(tmp_path, archive, manifest, dataset_retries=0),
                              sleep=_no_wait)
    assert not first['complete'] and set(first['failed']) == {'plate1/a.tif', 'plate1/b.tif'}
    mirror = tmp_path / 'mirror'
    assert list((mirror / 'plate1').glob('a.tif.*.part'))
    _Archive.cut_after = {'plate1/a.tif': 2000}
    second = pdx.fetch_dataset(_settings(tmp_path, archive, manifest, dataset_retries=0),
                               sleep=_no_wait)
    assert set(second['failed']) == {'plate1/a.tif'}
    assert second['before']['partial'] == 2
    _Archive.requests = []
    third = pdx.fetch_dataset(_settings(tmp_path, archive, manifest), sleep=_no_wait)
    assert third['complete'] and third['after']['missing'] == 0 and third['after']['short'] == 0
    assert third['fetched'] == ['plate1/a.tif']
    assert _Archive.requests == [('/Files/plate1/a.tif', 'bytes=3000-')]
    for path, data in FILES.items():
        assert (mirror / path).read_bytes() == data
    assert not list(mirror.rglob('*.part'))
    state = mirror / pdx.STATE_FOLDER
    summary = json.loads((state / 'summary.json').read_text())
    assert summary['complete'] and summary['after']['ok'] == 4
    table = pd.read_csv(state / 'completeness.csv')
    assert set(table['status']) == {'ok'}
    _Archive.requests = []
    again = pdx.fetch_dataset(_settings(tmp_path, archive, manifest), sleep=_no_wait)
    assert again['fetched'] == [] and _Archive.requests == []


def test_the_check_names_every_kind_of_incomplete_file(tmp_path):
    manifest = pdx.read_manifest(_manifest(tmp_path))
    mirror = tmp_path / 'mirror'
    (mirror / 'plate1').mkdir(parents=True)
    (mirror / 'plate2').mkdir()
    (mirror / 'plate1' / 'a.tif').write_bytes(b'')
    (mirror / 'plate1' / 'b.tif').write_bytes(FILES['plate1/b.tif'])
    (mirror / 'plate1' / 'b.tif.lftp-pget-status').write_text('size=3001\n')
    (mirror / 'plate2' / 'c.tif').write_bytes(b'c' * 10)
    (mirror / 'images.zip').write_bytes(FILES['images.zip'][:-30] + b'z' * 30)
    (mirror / 'stray.txt').write_text('left over')
    table = pdx.check_completeness(manifest, mirror)
    status = dict(zip(table['path'], table['status']))
    assert status == {'plate1/a.tif': 'empty', 'plate1/b.tif': 'partial',
                      'plate2/c.tif': 'short', 'images.zip': 'corrupt_zip',
                      'stray.txt': 'extra'}
    assert table.set_index('path').loc['plate1/b.tif', 'detail'] == 'plate1/b.tif.lftp-pget-status'


def test_long_and_wrong_content_files_are_named(tmp_path):
    manifest = pdx.read_manifest(_manifest(tmp_path))
    mirror = tmp_path / 'mirror'
    for path, data in FILES.items():
        (mirror / path).parent.mkdir(parents=True, exist_ok=True)
        (mirror / path).write_bytes(data)
    (mirror / 'plate1' / 'a.tif').write_bytes(FILES['plate1/a.tif'] + b'!')
    (mirror / 'plate2' / 'c.tif').write_bytes(b'd' * 1500)
    table = pdx.check_completeness(manifest, mirror)
    status = dict(zip(table['path'], table['status']))
    assert status['plate1/a.tif'] == 'long' and status['plate2/c.tif'] == 'checksum'
    assert status['plate1/b.tif'] == 'ok'
    cache = json.loads((mirror / pdx.STATE_FOLDER / 'checksums.json').read_text())
    assert cache['plate1/b.tif'][3] == hashlib.md5(FILES['plate1/b.tif']).hexdigest()
    again = pdx.check_completeness(manifest, mirror)
    assert dict(zip(again['path'], again['status'])) == status
    unchecked = pdx.check_completeness(manifest, mirror, checksums=False)
    assert dict(zip(unchecked['path'], unchecked['status']))['plate2/c.tif'] == 'ok'


def test_bad_files_are_fetched_again_and_leftovers_cleared(tmp_path, archive):
    manifest = _manifest(tmp_path, name='filelist.json')
    mirror = tmp_path / 'mirror'
    (mirror / 'plate1').mkdir(parents=True)
    (mirror / 'plate2').mkdir()
    (mirror / 'plate1' / 'a.tif').write_bytes(FILES['plate1/a.tif'] + b'extra')
    (mirror / 'plate1' / 'b.tif').write_bytes(b'\0' * 3001)
    (mirror / 'plate1' / 'b.tif.lftp-pget-status').write_text('pos=0')
    (mirror / 'plate2' / 'c.tif').write_bytes(FILES['plate2/c.tif'][:700])
    (mirror / 'images.zip').write_bytes(b'not a zip')
    _Archive.fail_first = {'plate2/c.tif'}
    result = pdx.fetch_dataset(_settings(tmp_path, archive, manifest), sleep=_no_wait)
    assert result['complete'], result['completeness']
    assert sorted(result['fetched']) == sorted(FILES)
    assert not (mirror / 'plate1' / 'b.tif.lftp-pget-status').exists()
    for path, data in FILES.items():
        assert (mirror / path).read_bytes() == data
    ranges = [r for p, r in _Archive.requests if p.endswith('c.tif')]
    assert ranges == ['bytes=700-', 'bytes=700-']


def test_a_server_that_ignores_ranges_restarts_the_file(tmp_path, archive):
    manifest = _manifest(tmp_path, md5=False)
    mirror = tmp_path / 'mirror' / 'plate2'
    mirror.mkdir(parents=True)
    (mirror / 'c.tif').write_bytes(b'c' * 200)
    _Archive.ignore_range = True
    result = pdx.fetch_dataset(_settings(tmp_path, archive, manifest,
                                         dataset_include='plate2/'), sleep=_no_wait)
    assert result['complete'] and list(result['manifest']['path']) == ['plate2/c.tif']
    assert (mirror / 'c.tif').read_bytes() == FILES['plate2/c.tif']


def test_sizes_are_learned_from_transfers_and_kept_for_the_next_run(tmp_path, archive):
    urls = tmp_path / 'urls.txt'
    urls.write_text('# BBBC-style list\n\n' + '\n'.join(archive + p for p in FILES) + '\n')
    result = pdx.fetch_dataset(_settings(tmp_path, archive, urls, dataset_checksums=False),
                               sleep=_no_wait)
    assert result['complete']
    written = pd.read_csv(tmp_path / 'mirror' / pdx.STATE_FOLDER / 'manifest.csv')
    assert dict(zip(written['path'], written['size'])) == {p: len(d) for p, d in FILES.items()}
    (tmp_path / 'mirror' / 'plate2' / 'c.tif').write_bytes(b'c' * 3)
    check = pdx.fetch_dataset(_settings(tmp_path, archive, urls, dataset_check_only=True),
                              sleep=_no_wait)
    assert check['before']['short'] == 1 and check['fetched'] == [] and not check['complete']


def test_a_missing_file_on_the_server_is_reported_as_failed(tmp_path, archive):
    listing = tmp_path / 'list.csv'
    pd.DataFrame({'path': ['plate1/a.tif', 'gone.tif'], 'size': [len(FILES['plate1/a.tif']), 5]}
                 ).to_csv(listing, index=False)
    result = pdx.fetch_dataset(_settings(tmp_path, archive, listing), sleep=_no_wait)
    assert set(result['failed']) == {'gone.tif'} and 'HTTP Error 404' in result['failed']['gone.tif']
    row = result['completeness'].set_index('path').loc['gone.tif']
    assert row['status'] == 'failed' and result['after']['failed'] == 1
    assert sum(1 for p, _ in _Archive.requests if p.endswith('gone.tif')) == 1


def test_file_urls_and_a_complete_stale_partial(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'x.bin').write_bytes(b'q' * 900)
    target = tmp_path / 'out' / 'x.bin'
    target.parent.mkdir()
    stale = target.with_name('x.bin.999999999.part')
    stale.write_bytes(b'q' * 400)
    smaller = target.with_name('x.bin.999999998.part')
    smaller.write_bytes(b'q' * 10)
    got = pdx.download_file(source.joinpath('x.bin').as_uri(), target, size=900,
                            sleep=_no_wait)
    assert got == 900 and target.read_bytes() == b'q' * 900
    assert not stale.exists() and not smaller.exists()
    whole = target.with_name('y.bin.999999997.part')
    whole.write_bytes(b'w' * 50)
    calls = []
    assert pdx.download_file('http://unused/y.bin', target.with_name('y.bin'), size=50,
                             open_url=lambda *a, **k: calls.append(a)) == 50
    assert calls == []


def test_a_partial_owned_by_a_running_process_is_left_alone(tmp_path):
    target = tmp_path / 'z.bin'
    busy = tmp_path / f'z.bin.{os.getppid()}.part'
    busy.write_bytes(b'1')
    with pytest.raises(pdx.DownloadError, match='another running process'):
        pdx.download_file('http://unused/z.bin', target)
    assert busy.exists()


def test_a_partial_longer_than_the_archive_file_is_dropped(tmp_path, archive):
    target = tmp_path / 'c.tif'
    target.with_name(f'c.tif.{os.getpid()}.part').write_bytes(b'c' * 5000)
    got = pdx.download_file(archive + 'plate2/c.tif', target, size=1500,
                            digest=('md5', hashlib.md5(FILES['plate2/c.tif']).hexdigest()))
    assert got == 1500


def test_a_range_past_the_end_and_a_wrong_digest_are_retried(tmp_path, archive):
    target = tmp_path / 'c.tif'
    target.with_name(f'c.tif.{os.getpid()}.part').write_bytes(b'c' * 1600)
    assert pdx.download_file(archive + 'plate2/c.tif', target, sleep=_no_wait) == 1500
    waits = []
    with pytest.raises(pdx.DownloadError, match='md5 does not match'):
        pdx.download_file(archive + 'plate2/c.tif', tmp_path / 'd.tif',
                          digest=('md5', '0' * 32), retries=2, sleep=waits.append)
    assert waits == [2.0, 4.0]
    with pytest.raises(pdx.DownloadError, match='refused'):
        pdx.download_file('http://127.0.0.1:9/none', tmp_path / 'e.tif', retries=0)


def test_archive_addresses():
    assert pdx.archive_base_url('bia', 'S-BIAD394') == (
        'https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/394/S-BIAD394/Files/')
    assert pdx.archive_base_url('bia', 'S-BSST12') == (
        'https://ftp.ebi.ac.uk/biostudies/fire/S-BSST/012/S-BSST12/Files/')
    assert pdx.archive_base_url('idr', 'idr0094-ellinger-sarscov2/') == (
        'https://ftp.ebi.ac.uk/pub/databases/IDR/idr0094-ellinger-sarscov2/')
    assert pdx.archive_base_url('bbbc', 'BBBC021') == 'https://data.broadinstitute.org/bbbc/BBBC021/'
    for archive, accession in (('bia', ''), ('bia', 'BIAD'), ('zenodo', 'x')):
        with pytest.raises(ValueError):
            pdx.archive_base_url(archive, accession)


def test_manifest_reading_rules(tmp_path):
    table = tmp_path / 'list.tsv'
    pd.DataFrame({'Files': ['./a/b.tif', 'a/b.tif', None, 'dir'],
                  'url': ['', '', 'https://h/base/c/d.tif', ''],
                  'type': ['file', 'file', 'file', 'directory'],
                  'SHA256': ['AA', 'BB', '', '']}).to_csv(table, sep='\t', index=False)
    frame = pdx.read_manifest(table, base_url='https://h/base/')
    assert list(frame['path']) == ['a/b.tif', 'c/d.tif']
    assert list(frame['sha256']) == ['bb', ''] and frame['size'].isna().all()
    for bad in ('../up.tif', '/abs.tif', 'C:/x.tif', ''):
        with pytest.raises(ValueError):
            pdx._safe_relative(bad)
    nocol = tmp_path / 'nocol.csv'
    pd.DataFrame({'other': [1]}).to_csv(nocol, index=False)
    with pytest.raises(ValueError, match='no path column'):
        pdx.read_manifest(nocol)
    listed = tmp_path / 'list.json'
    listed.write_text(json.dumps([{'path': 'x.tif', 'size': 3}]))
    assert pdx.read_manifest(listed)['size'].tolist() == [3]
    assert pdx._path_of_url('https://other/q/z%20y.tif?x=1', 'https://h/') == 'z y.tif'


class _FakeFTP:
    tree = {'/pub/S': [('a.tif', {'type': 'file', 'size': '5'}),
                       ('.', {'type': 'cdir'}), ('sub', {'type': 'dir'})],
            '/pub/S/sub': [('b.tif', {'type': 'file', 'size': '7'})]}
    refuse = False
    quit_fails = False

    def __init__(self, host, timeout):
        self.host, self.timeout, self.closed = host, timeout, False

    def login(self, user, password):
        self.user = user

    def mlsd(self, folder, facts):
        if self.refuse:
            raise ftplib.error_perm('500 MLSD not understood')
        return iter(self.tree[folder])

    def quit(self):
        if self.quit_fails:
            raise OSError('gone')

    def close(self):
        self.closed = True


def test_ftp_listing_walks_folders_with_sizes():
    frame = pdx.list_ftp('ftp://ftp.example/pub/S/', ftp_factory=_FakeFTP)
    assert dict(zip(frame['path'], frame['size'])) == {'a.tif': 5, 'sub/b.tif': 7}
    _FakeFTP.refuse, _FakeFTP.quit_fails = True, True
    try:
        with pytest.raises(ValueError, match='cannot list'):
            pdx.list_ftp('ftp://ftp.example/pub/S', ftp_factory=_FakeFTP)
    finally:
        _FakeFTP.refuse, _FakeFTP.quit_fails = False, False


def test_web_index_listing_follows_sub_folders(archive):
    _Archive.index = {
        '/Files/': '<a href="../">up</a><a href="plate1/">p</a><a href="images.zip">z</a>'
                   '<a href="?C=N">sort</a><a href="https://elsewhere/x.zip">x</a><a>none</a>',
        '/Files/plate1/': '<a href="a.tif">a</a><a href="/Files/">top</a><a href="deep/">d</a>',
    }
    frame = pdx.list_http_index(archive[:-1], depth=0)
    assert sorted(frame['path']) == ['images.zip']
    frame = pdx.list_http_index(archive, depth=1)
    assert sorted(frame['path']) == ['images.zip', 'plate1/a.tif']
    assert set(frame['url']) == {archive + 'images.zip', archive + 'plate1/a.tif'}


def test_archive_settings_pick_the_listing(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(pdx, 'list_ftp', lambda url: seen.append(url) or pdx._frame([]))
    monkeypatch.setattr(pdx, 'list_http_index', lambda url: seen.append(url) or pdx._frame([]))
    for archive, accession in (('bia', 'S-BIAD394'), ('idr', 'idr0094'), ('bbbc', 'BBBC021')):
        result = pdx.fetch_dataset({'dataset_archive': archive, 'dataset_accession': accession,
                                    'dataset_dst': str(tmp_path / archive)})
        assert result['complete'] and len(result['manifest']) == 0
    assert seen == ['ftp://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/394/S-BIAD394/Files/',
                    'ftp://ftp.ebi.ac.uk/pub/databases/IDR/idr0094/',
                    'https://data.broadinstitute.org/bbbc/BBBC021/']
    base, _ = pdx._resolve({'dataset_archive': 'url', 'dataset_base_url': 'https://m/x',
                            'dataset_manifest': str(_manifest(tmp_path))})
    assert base == 'https://m/x/'
    with pytest.raises(ValueError, match='needs a file list'):
        pdx._resolve({'dataset_archive': 'url'})
    with pytest.raises(ValueError, match='must be one of'):
        pdx._resolve({'dataset_archive': 'nas'})
    with pytest.raises(ValueError, match='dataset_dst'):
        pdx.fetch_dataset({})


def test_settings_are_registered_and_alpha():
    from spacr import settings as S

    defaults = S.defaults_for(pdx.APP_KEY, {})
    assert set(defaults) == set(pdx._EXPECTED_TYPES) == set(pdx._TOOLTIPS)
    for key, kind in pdx._EXPECTED_TYPES.items():
        assert S.expected_types[key] is kind and isinstance(defaults[key], kind)
        assert S.tooltips[key].startswith(f'({kind.__name__}) - ') and len(S.tooltips[key]) <= 600
    categorised = {key for keys in pdx._CATEGORIES.values() for key in keys}
    assert categorised == set(defaults)
    for name, keys in pdx._CATEGORIES.items():
        assert set(keys) <= set(S.categories[name])
    assert set(S.ALPHA_FEATURES[750]['module_settings'][pdx.APP_KEY]) == set(defaults)
    assert set(defaults) <= S._alpha_names('settings', app_key=pdx.APP_KEY)
    pdx._register()


def test_command_line(tmp_path, archive, capsys):
    manifest = _manifest(tmp_path)
    args = ['--archive', 'url', '--manifest', str(manifest), '--base-url', archive,
            '--dst', str(tmp_path / 'cli'), '--workers', '2']
    assert pdx.main(args + ['--check-only', '--no-checksums']) == 1
    assert pdx.main(args) == 0
    assert '4 of 4 files complete' in capsys.readouterr().out


def test_unreadable_state_and_odd_headers(tmp_path):
    state = tmp_path / pdx.STATE_FOLDER
    state.mkdir()
    (state / 'checksums.json').write_text('[1, 2]')
    assert pdx._load_cache(tmp_path) == {}
    (state / 'checksums.json').write_text('{')
    assert pdx._load_cache(tmp_path) == {}
    assert pdx._scan(tmp_path / 'absent') == {}

    class _Response:
        headers = {'Content-Range': 'bytes 0-1/*', 'Content-Length': 'x'}

    assert pdx._total_from(_Response(), 0) is None
    assert pdx._total_from(object(), 0) is None
    assert pdx._leftover_of('a.tif.aspx.partial') == 'a.tif'
    assert pdx._leftover_of('a.tif.123.part') == 'a.tif'
    assert pdx._leftover_of('a.tif') is None
    assert pdx._pid_alive(os.getpid())


def test_pid_check_without_psutil(monkeypatch):
    import builtins

    real = builtins.__import__

    def refuse(name, *args, **kwargs):
        """Fail the psutil import only."""
        if name == 'psutil':
            raise ImportError(name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', refuse)
    assert pdx._pid_alive(1) is True


def test_a_fetch_that_cannot_write_is_reported(tmp_path, monkeypatch):
    record = {'path': 'a.tif', 'url': 'http://unused/a.tif', 'size': 1, 'md5': '', 'sha256': ''}

    def broken(*args, **kwargs):
        """Fail the way a full disk does."""
        raise OSError('No space left on device')

    monkeypatch.setattr(pdx, 'download_file', broken)
    assert pdx._fetch_one(record, 'missing', '', tmp_path, 0, None, _no_wait) == (
        'a.tif', None, 'No space left on device')
