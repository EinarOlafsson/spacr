"""Fetch a public imaging dataset and check it against the archive's file list.

A reanalysis of a public screen starts with a transfer of hundreds of
gigabytes to several terabytes. Interrupted transfers leave files that look
present: a zero-byte TIFF, a truncated ZIP, or a full-size file that lftp
was still filling in when it stopped (it keeps a ``.lftp-pget-status`` file
beside it). A run started on such a mirror fails hours later, or loses a
well without failing. This module finds those files before the run starts,
and fetches only them.

Three archives are named, and any other one works from a file list:

* the BioImage Archive (``'bia'``, an accession such as ``S-BIAD394``), whose
  files are under ``https://ftp.ebi.ac.uk/biostudies/fire/``;
* the Image Data Resource (``'idr'``, a study such as
  ``idr0094-ellinger-sarscov2``), under
  ``https://ftp.ebi.ac.uk/pub/databases/IDR/``;
* the Broad Bioimage Benchmark Collection (``'bbbc'``, a set such as
  ``BBBC021``), whose ZIP archives are under
  ``https://data.broadinstitute.org/bbbc/``;
* ``'url'``: a file list whose rows are full addresses.

The file list (the manifest) is a TSV, CSV or JSON table with a path column
and, where the archive publishes them, a size and an MD5 or SHA-256
column, or a text file with one address per line (the form of the
2026-10-31 ``bbbc021_urls.txt``). Without a file list, a BioImage Archive
or IDR folder is listed over FTP, which gives every file's size, and a BBBC
folder from its web index.

:func:`check_completeness` compares the local folder with the manifest and
gives every file one status: ``ok``, ``missing``, ``empty`` (zero bytes),
``short`` or ``long`` (size differs from the archive's), ``checksum``
(published checksum differs), ``partial`` (a leftover ``.part``,
``.partial``, ``.aspx`` or ``.lftp-pget-status`` file beside it),
``corrupt_zip`` (a ZIP whose directory cannot be read) or ``extra`` (on disk,
not in the manifest). :func:`fetch_dataset` runs that check, fetches every
file that is not ``ok`` with ranged, resumed transfers and retries, checks
again and writes the manifest, the table and a summary under
``_spacr_dataset_check`` in the destination. Running it again fetches only
what is still missing or bad.

Each transfer writes to ``<name>.<process id>.part`` and renames it into
place only when it is complete, so two processes fetching into one folder
never rename each other's file away. A ``.part`` left by a process that has
stopped is taken over and resumed from its last byte.

Parallelism is across files, not within one file: an archive of many
TIFFs keeps every worker busy, and a single large ZIP is resumed rather
than split. The worker count is clamped by
:func:`spacr.resource_log._guard_workers`.
"""
from __future__ import annotations

import argparse
import ftplib
import glob
import hashlib
import html.parser
import http.client
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd

#: Key the settings of this module are registered under.
APP_KEY = 'public_datasets'

#: The archives :func:`archive_base_url` knows how to address.
ARCHIVES = ('bia', 'idr', 'bbbc', 'url')

#: Folder inside the destination that holds the manifest, the completeness
#: table, the summary and the checksum cache. It is never reported as extra.
STATE_FOLDER = '_spacr_dataset_check'

#: Suffixes of files that an interrupted transfer leaves beside its target:
#: lftp's segment status, Aspera's partial and resume files, spaCR's and
#: browsers' partial downloads.
LEFTOVER_SUFFIXES = ('.lftp-pget-status', '.partial', '.aspx', '.part',
                     '.crdownload')

#: Every status :func:`check_completeness` gives, in report order.
STATUSES = ('ok', 'missing', 'empty', 'short', 'long', 'checksum',
            'partial', 'corrupt_zip', 'failed', 'extra')

#: The statuses that :func:`fetch_dataset` fetches again.
REFETCH = frozenset({'missing', 'empty', 'short', 'long', 'checksum',
                     'partial', 'corrupt_zip', 'failed'})

_PATH_COLUMNS = ('path', 'files', 'file', 'filename', 'file_name', 'name',
                 'relative_path')
_URL_COLUMNS = ('url', 'href', 'link', 'download_url')
_SIZE_COLUMNS = ('size', 'bytes', 'file_size', 'size_bytes', 'length')
_MD5_COLUMNS = ('md5', 'md5sum', 'md5_checksum', 'checksum')
_SHA256_COLUMNS = ('sha256', 'sha256sum', 'sha256_checksum')
_CHUNK = 1 << 20

_DEFAULTS: Dict[str, Any] = {
    'dataset_archive': 'bia',
    'dataset_accession': '',
    'dataset_manifest': '',
    'dataset_base_url': '',
    'dataset_dst': '',
    'dataset_include': '',
    'dataset_workers': 4,
    'dataset_retries': 5,
    'dataset_checksums': True,
    'dataset_check_only': False,
}

_EXPECTED_TYPES: Dict[str, Any] = {
    'dataset_archive': str,
    'dataset_accession': str,
    'dataset_manifest': str,
    'dataset_base_url': str,
    'dataset_dst': str,
    'dataset_include': str,
    'dataset_workers': int,
    'dataset_retries': int,
    'dataset_checksums': bool,
    'dataset_check_only': bool,
}

_TOOLTIPS: Dict[str, str] = {
    'dataset_archive': "(str) - Which public archive the files come from: 'bia' (BioImage Archive), 'idr' (Image Data Resource), 'bbbc' (Broad Bioimage Benchmark Collection) or 'url' (a file list of full addresses). It decides the download address built from the accession and how the file list is found when none is given. Default 'bia'.",
    'dataset_accession': "(str) - The archive's name for the dataset, such as S-BIAD394, idr0094-ellinger-sarscov2 or BBBC021. With a blank base URL it decides where files are downloaded from. Not needed for 'url'. Default blank.",
    'dataset_manifest': "(str) - The archive's file list: a TSV, CSV or JSON table with a path column and, where published, size and MD5 or SHA-256 columns, or a text file of addresses, one per line. Sizes and checksums in it are what the completeness check compares against. Blank lists the archive folder instead (FTP for 'bia' and 'idr', the web index for 'bbbc'). Default blank.",
    'dataset_base_url': "(str) - Address that the manifest's relative paths are joined to. Blank builds it from the archive and accession, for example https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/394/S-BIAD394/Files/. Set it for a mirror or when the archive moves its files. Default blank.",
    'dataset_dst': "(str) - Local folder the dataset is fetched into and checked in. Files keep the manifest's relative paths. The manifest, the completeness table and the summary are written to its _spacr_dataset_check folder. Default blank.",
    'dataset_include': "(str) - Regular expression a manifest path must contain to be fetched and checked, for example 20200202_6W-LaC024A/ for one plate of a screen. Files outside it are neither fetched nor reported missing. Blank includes every file. Default blank.",
    'dataset_workers': "(int) - Files transferred at the same time. More workers fill a fast link sooner; each opens one connection, and archives may throttle many. Lowered when free memory cannot hold them. Default 4.",
    'dataset_retries': "(int) - Further attempts for a file whose transfer fails or stops early. Each attempt resumes from the last byte received, waiting longer each time, up to one minute. 0 tries each file once. Default 5.",
    'dataset_checksums': "(bool) - Compare each file with the MD5 or SHA-256 the manifest publishes. Catches a file of the right size with wrong content, at the cost of reading every file once; results are cached by size and modification time. Off checks names and sizes only. Default True.",
    'dataset_check_only': "(bool) - Write the completeness table without downloading anything, for example to audit a mirror made with another tool. Off fetches every missing or bad file, then checks again. Default False.",
}

_CATEGORIES: Dict[str, List[str]] = {
    'Public dataset': ['dataset_archive', 'dataset_accession',
                       'dataset_manifest', 'dataset_base_url',
                       'dataset_dst', 'dataset_include'],
    'Public dataset transfer': ['dataset_workers', 'dataset_retries',
                                'dataset_checksums', 'dataset_check_only'],
}

_DESCRIPTION = (
    "Fetch a public imaging dataset from the BioImage Archive, the Image "
    "Data Resource or the Broad Bioimage Benchmark Collection, resuming "
    "interrupted transfers, and check every file against the archive's own "
    "file list before an analysis starts."
)


def public_dataset_defaults(settings: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The settings of :func:`fetch_dataset`, filled in with their defaults.

    :param settings: settings given so far; a key already present is kept.
    :returns: a new dict with every key of this module.
    """
    merged = dict(settings or {})
    for key, value in _DEFAULTS.items():
        merged.setdefault(key, value)
    return merged


def _register() -> None:
    """Register this module's settings through the defaults seam.

    Guarded so a reload of this module in one process is not a second
    registration.
    """
    from .settings import has_registered_defaults, register_defaults

    if has_registered_defaults(APP_KEY):
        return
    register_defaults(APP_KEY, public_dataset_defaults,
                      expected_types=_EXPECTED_TYPES, tooltips=_TOOLTIPS,
                      categories=_CATEGORIES, description=_DESCRIPTION)


_register()


def _say(text: str, **values: Any) -> None:
    """Print one translated progress line.

    :param text: the English message, with ``str.format`` placeholders.
    :param values: the placeholder values.
    """
    from .qt.i18n import tr

    print(tr(text, **values), flush=True)


def archive_base_url(archive: str, accession: str) -> str:
    """The HTTPS folder an archive serves one dataset's files from.

    The BioImage Archive files accession ``S-BIAD394`` under
    ``fire/S-BIAD/394/S-BIAD394/Files/``: the letters, then the last three
    digits of the number, then the accession.

    :param archive: one of ``'bia'``, ``'idr'`` or ``'bbbc'``.
    :param accession: the dataset's accession, such as ``S-BIAD394``,
        ``idr0094-ellinger-sarscov2`` or ``BBBC021``.
    :returns: the address, ending in ``/``.
    :raises ValueError: for another archive, a blank accession or a
        BioImage Archive accession without a number.
    """
    accession = str(accession or '').strip().strip('/')
    if not accession:
        raise ValueError('a dataset accession is needed to build its address')
    if archive == 'bia':
        match = re.fullmatch(r'([A-Za-z-]+?)(\d+)', accession)
        if match is None:
            raise ValueError(
                f'{accession!r} is not a BioImage Archive accession such as S-BIAD394')
        prefix, number = match.group(1), match.group(2)
        return ('https://ftp.ebi.ac.uk/biostudies/fire/'
                f'{prefix}/{number[-3:].zfill(3)}/{accession}/Files/')
    if archive == 'idr':
        return f'https://ftp.ebi.ac.uk/pub/databases/IDR/{accession}/'
    if archive == 'bbbc':
        return f'https://data.broadinstitute.org/bbbc/{accession}/'
    raise ValueError(f'no address is known for archive {archive!r}; '
                     f'expected one of {ARCHIVES[:3]} or a base URL')


def _pick(columns: Iterable[str], names: Sequence[str]) -> Optional[str]:
    """The first column whose lower-case name is in ``names``.

    :param columns: the table's column names.
    :param names: accepted names, in order of preference.
    :returns: the column as spelled in the table, or None.
    """
    lowered = {str(column).strip().lower(): column for column in columns}
    for name in names:
        if name in lowered:
            return lowered[name]
    return None


def _safe_relative(path: str) -> str:
    """A manifest path as a relative POSIX path that stays inside the folder.

    :param path: the path as the archive lists it.
    :returns: the cleaned path.
    :raises ValueError: for an empty path, or one that is absolute or climbs
        out with ``..``.
    """
    text = str(path).replace('\\', '/').strip()
    while text.startswith('./'):
        text = text[2:]
    parts = [part for part in text.split('/') if part not in ('', '.')]
    if not parts or text.startswith('/') or '..' in parts or re.match(r'^[A-Za-z]:', text):
        raise ValueError(f'manifest path {path!r} is empty or leaves the destination')
    return '/'.join(parts)


def _frame(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """The manifest frame of ``rows``, with its five columns in order.

    :param rows: dicts with any of ``path``, ``url``, ``size``, ``md5`` and
        ``sha256``.
    :returns: a frame with one row per distinct path.
    """
    frame = pd.DataFrame(rows, columns=['path', 'url', 'size', 'md5', 'sha256'])
    frame['size'] = pd.to_numeric(frame['size'], errors='coerce').astype('Int64')
    for column in ('url', 'md5', 'sha256'):
        frame[column] = frame[column].fillna('').astype(str).str.strip()
    frame['md5'] = frame['md5'].str.lower()
    frame['sha256'] = frame['sha256'].str.lower()
    return frame.drop_duplicates('path', keep='last').reset_index(drop=True)


def _rows_from_table(table: pd.DataFrame, base_url: str) -> List[Dict[str, Any]]:
    """Manifest rows of a file-list table.

    :param table: the table as the archive publishes it.
    :param base_url: address its relative paths are joined to.
    :returns: one dict per file.
    :raises ValueError: when the table has neither a path nor an address
        column.
    """
    path_col = _pick(table.columns, _PATH_COLUMNS)
    url_col = _pick(table.columns, _URL_COLUMNS)
    if path_col is None and url_col is None:
        raise ValueError(
            'the file list has no path column; expected one of '
            f'{", ".join(_PATH_COLUMNS)} or an address column')
    size_col = _pick(table.columns, _SIZE_COLUMNS)
    md5_col = _pick(table.columns, _MD5_COLUMNS)
    sha_col = _pick(table.columns, _SHA256_COLUMNS)
    type_col = _pick(table.columns, ('type',))
    rows = []
    for record in table.to_dict('records'):
        if type_col is not None and str(record.get(type_col, '')).lower() in ('directory', 'dir'):
            continue
        url = str(record.get(url_col) or '') if url_col is not None else ''
        path = record.get(path_col) if path_col is not None else None
        if path is None or (isinstance(path, float) and pd.isna(path)) or str(path).strip() == '':
            if not url:
                continue
            path = _path_of_url(url, base_url)
        rows.append({'path': _safe_relative(str(path)), 'url': url,
                     'size': record.get(size_col) if size_col else None,
                     'md5': record.get(md5_col) if md5_col else '',
                     'sha256': record.get(sha_col) if sha_col else ''})
    return rows


def _path_of_url(url: str, base_url: str) -> str:
    """The local relative path of a full address.

    :param url: the address.
    :param base_url: when the address starts with it, the rest is the path;
        otherwise the address's file name is.
    :returns: the relative path.
    """
    if base_url and url.startswith(base_url):
        rest = url[len(base_url):]
    else:
        rest = urllib.parse.urlsplit(url).path.rsplit('/', 1)[-1]
    return urllib.parse.unquote(rest.split('?', 1)[0])


def read_manifest(source: Any, *, base_url: str = '') -> pd.DataFrame:
    """Read an archive's file list into spaCR's manifest table.

    A ``.json`` file holds a list of records, or a dict with the list under
    ``files``; a ``.txt`` or ``.urls`` file holds one address per line
    (blank lines and lines starting with ``#`` are skipped); anything else
    is read with :func:`spacr.tabular.read_table`, so TSV, CSV, Parquet and
    Excel all work. Column names are matched without regard to case:
    ``path`` (or ``Files``, ``file``, ``name``), ``url``, ``size`` (or
    ``bytes``), ``md5`` and ``sha256``. Directory rows of a BioImage Archive
    JSON list are skipped.

    :param source: path of the file list.
    :param base_url: address relative paths are joined to; for a list of
        addresses, the prefix stripped to make each local path.
    :returns: a frame with ``path``, ``url``, ``size`` (nullable integer),
        ``md5`` and ``sha256``, one row per path.
    :raises ValueError: for a path that is absolute or leaves the folder,
        or a table with no path or address column.
    """
    path = Path(os.path.expanduser(str(source)))
    suffix = path.suffix.lower()
    if suffix in ('.txt', '.urls', '.list'):
        rows = []
        for line in path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            rows.append({'path': _safe_relative(_path_of_url(line, base_url)),
                         'url': line})
        return _frame(rows)
    if suffix == '.json':
        payload = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(payload, dict):
            payload = payload.get('files', [])
        return _frame(_rows_from_table(pd.DataFrame(list(payload)), base_url))
    from .tabular import read_table

    table = read_table(str(path), canonicalise=False, report=None, dtype=str)
    return _frame(_rows_from_table(table, base_url))


def _ftp_entries(ftp: Any, folder: str) -> Iterable[tuple]:
    """``(name, facts)`` of one FTP folder, from MLSD.

    :param ftp: a connected :class:`ftplib.FTP`.
    :param folder: the folder.
    :returns: the entries, without ``.`` and ``..``.
    """
    for name, facts in ftp.mlsd(folder, facts=['type', 'size']):
        if facts.get('type') in ('cdir', 'pdir') or name in ('.', '..'):
            continue
        yield name, facts


def list_ftp(url: str, *, ftp_factory: Callable[..., Any] = ftplib.FTP,
             timeout: float = 60.0) -> pd.DataFrame:
    """List every file below an FTP folder, with its size.

    :param url: ``ftp://host/folder/``.
    :param ftp_factory: called as ``ftp_factory(host, timeout=timeout)`` to
        open the connection; :class:`ftplib.FTP` by default.
    :param timeout: seconds to wait on the server.
    :returns: a manifest frame (see :func:`read_manifest`) with paths
        relative to ``url`` and no addresses.
    :raises ValueError: when the server cannot list folders with MLSD.
    """
    parts = urllib.parse.urlsplit(url)
    root = parts.path.rstrip('/') or '/'
    ftp = ftp_factory(parts.hostname, timeout=timeout)
    rows = []
    try:
        ftp.login(parts.username or 'anonymous', parts.password or '')
        pending = ['']
        while pending:
            relative = pending.pop()
            folder = f'{root}/{relative}'.rstrip('/') if relative else root
            try:
                entries = list(_ftp_entries(ftp, folder))
            except ftplib.error_perm as exc:
                raise ValueError(
                    f'{parts.hostname} cannot list {folder} ({exc}); give the '
                    'archive file list as dataset_manifest instead') from exc
            for name, facts in entries:
                child = f'{relative}/{name}' if relative else name
                if facts.get('type') == 'dir':
                    pending.append(child)
                else:
                    rows.append({'path': _safe_relative(child), 'size': facts.get('size')})
    finally:
        try:
            ftp.quit()
        except Exception:
            ftp.close()
    return _frame(rows)


class _LinkParser(html.parser.HTMLParser):
    """Collect the ``href`` of every link on a web folder index."""

    def __init__(self) -> None:
        """Start with no links."""
        super().__init__()
        self.links: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[tuple]) -> None:
        """Keep the address of an ``<a>`` tag.

        :param tag: the tag name.
        :param attrs: its ``(name, value)`` attributes.
        """
        if tag == 'a':
            for name, value in attrs:
                if name == 'href' and value:
                    self.links.append(value)


def list_http_index(url: str, *, open_url: Callable[..., Any] = urllib.request.urlopen,
                    timeout: float = 60.0, depth: int = 4) -> pd.DataFrame:
    """List the files a web folder index links to, following sub-folders.

    Links that leave the folder (to a parent, another host, a query or a
    fragment) are ignored. Sizes are not known from an index; they are
    learned from each transfer.

    :param url: the folder address, such as
        ``https://data.broadinstitute.org/bbbc/BBBC021/``.
    :param open_url: called as ``open_url(url, timeout=timeout)``.
    :param timeout: seconds to wait on the server.
    :param depth: how many levels of sub-folders to follow.
    :returns: a manifest frame (see :func:`read_manifest`) with addresses.
    """
    base = url if url.endswith('/') else url + '/'
    rows, pending, seen = [], [(base, 0)], set()
    while pending:
        folder, level = pending.pop()
        if folder in seen:
            continue
        seen.add(folder)
        with open_url(folder, timeout=timeout) as response:
            parser = _LinkParser()
            parser.feed(response.read().decode('utf-8', 'replace'))
        for href in parser.links:
            target = urllib.parse.urljoin(folder, href)
            if (not target.startswith(base) or target == folder
                    or '?' in href or '#' in href):
                continue
            if target.endswith('/'):
                if level < depth:
                    pending.append((target, level + 1))
                continue
            rows.append({'path': _safe_relative(_path_of_url(target, base)),
                         'url': target})
    return _frame(rows)


def _scan(dst: Path) -> Dict[str, int]:
    """``{relative path: size}`` of every file under ``dst``.

    The state folder is skipped.

    :param dst: the destination folder.
    :returns: the sizes; empty when ``dst`` does not exist.
    """
    sizes: Dict[str, int] = {}
    pending = [(dst, '')]
    while pending:
        folder, prefix = pending.pop()
        try:
            entries = list(os.scandir(folder))
        except (FileNotFoundError, NotADirectoryError):
            continue
        for entry in entries:
            relative = f'{prefix}{entry.name}'
            if entry.is_dir(follow_symlinks=False):
                if relative != STATE_FOLDER:
                    pending.append((Path(entry.path), relative + '/'))
            elif entry.is_file():
                sizes[relative] = entry.stat().st_size
    return sizes


def _leftover_of(name: str) -> Optional[str]:
    """The file an interrupted-transfer leftover belongs to.

    :param name: a relative path.
    :returns: the target's relative path, or None when ``name`` is not a
        leftover.
    """
    for suffix in LEFTOVER_SUFFIXES:
        if name.endswith(suffix):
            stem = name[:-len(suffix)]
            if suffix == '.part':
                stem = re.sub(r'\.\d+$', '', stem)
            elif suffix == '.partial' and stem.endswith('.aspx'):
                stem = stem[:-len('.aspx')]
            return stem
    return None


def _file_digest(path: Path, kind: str) -> str:
    """The hex digest of a file.

    :param path: the file.
    :param kind: ``'md5'`` or ``'sha256'``.
    :returns: the lower-case hex digest.
    """
    digest = hashlib.new(kind)
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(_CHUNK), b''):
            digest.update(block)
    return digest.hexdigest()


def _zip_readable(path: Path) -> bool:
    """Whether a ZIP archive's directory can be read.

    A truncated archive loses its directory, which sits at the end.

    :param path: the archive.
    :returns: True when :class:`zipfile.ZipFile` opens it.
    """
    try:
        with zipfile.ZipFile(path) as archive:
            archive.infolist()
        return True
    except (zipfile.BadZipFile, OSError, EOFError):
        return False


def _load_cache(dst: Path) -> Dict[str, list]:
    """The verified-checksum cache of a destination.

    :param dst: the destination folder.
    :returns: ``{path: [size, mtime_ns, kind, digest]}``; empty when absent
        or unreadable.
    """
    try:
        cache = json.loads((dst / STATE_FOLDER / 'checksums.json').read_text(encoding='utf-8'))
        return cache if isinstance(cache, dict) else {}
    except (OSError, ValueError):
        return {}


def _verify(path: Path, kind: str, expected: str, cache: Dict[str, list], key: str) -> bool:
    """Whether a file has the expected digest, using the cache when it can.

    :param path: the file.
    :param kind: ``'md5'`` or ``'sha256'``.
    :param expected: the published digest.
    :param cache: the cache from :func:`_load_cache`, updated in place.
    :param key: the file's relative path, the cache key.
    :returns: True when the digests agree.
    """
    stat = path.stat()
    stamp = [stat.st_size, stat.st_mtime_ns, kind]
    cached = cache.get(key)
    if cached and list(cached[:3]) == stamp:
        return cached[3] == expected
    digest = _file_digest(path, kind)
    cache[key] = stamp + [digest]
    return digest == expected


def _check_row(row: Mapping[str, Any], dst: Path, sizes: Mapping[str, int],
               leftovers: Mapping[str, List[str]], checksums: bool,
               cache: Dict[str, list]) -> Dict[str, Any]:
    """The completeness row of one manifest file.

    :param row: the manifest row.
    :param dst: the destination folder.
    :param sizes: local sizes from :func:`_scan`.
    :param leftovers: ``{target: [leftover paths]}``.
    :param checksums: compare published checksums.
    :param cache: the checksum cache.
    :returns: dict with ``path``, ``status``, ``expected_size``,
        ``local_size`` and ``detail``.
    """
    path = row['path']
    expected = row['size']
    expected = None if pd.isna(expected) else int(expected)
    local = sizes.get(path)
    out = {'path': path, 'status': 'ok', 'expected_size': expected,
           'local_size': local, 'detail': ''}
    if path in leftovers:
        out.update(status='partial', detail=', '.join(sorted(leftovers[path])))
    elif local is None:
        out['status'] = 'missing'
    elif local == 0 and expected != 0:
        out['status'] = 'empty'
    elif expected is not None and local < expected:
        out['status'] = 'short'
    elif expected is not None and local > expected:
        out['status'] = 'long'
    elif path.lower().endswith('.zip') and not _zip_readable(dst / path):
        out['status'] = 'corrupt_zip'
    elif checksums:
        for kind in ('sha256', 'md5'):
            if row[kind]:
                if not _verify(dst / path, kind, row[kind], cache, path):
                    out.update(status='checksum', detail=kind)
                break
    return out


def check_completeness(manifest: pd.DataFrame, dst: Any, *, checksums: bool = True,
                       workers: int = 4) -> pd.DataFrame:
    """Compare a local folder with the archive's file list.

    Every manifest file gets one status (see :data:`STATUSES`); every other
    file under ``dst`` is ``extra``, except the leftovers of manifest files,
    which make their file ``partial``, and the state folder.

    :param manifest: frame from :func:`read_manifest` or :func:`list_ftp`.
    :param dst: the local folder.
    :param checksums: compare published MD5 or SHA-256 digests; results
        are cached in the state folder by size and modification time.
    :param workers: files hashed at the same time.
    :returns: a frame with ``path``, ``status``, ``expected_size``,
        ``local_size`` and ``detail``, manifest files first.
    """
    dst = Path(os.path.expanduser(str(dst)))
    sizes = _scan(dst)
    wanted = set(manifest['path'])
    leftovers: Dict[str, List[str]] = {}
    extra = []
    for name in sorted(sizes):
        target = _leftover_of(name)
        if target is not None and target in wanted:
            leftovers.setdefault(target, []).append(name)
        elif name not in wanted:
            extra.append(name)
    cache = _load_cache(dst) if checksums else {}
    records = manifest.to_dict('records')
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        rows = list(pool.map(
            lambda record: _check_row(record, dst, sizes, leftovers, checksums, cache),
            records))
    rows.extend({'path': name, 'status': 'extra', 'expected_size': None,
                 'local_size': sizes[name], 'detail': ''} for name in extra)
    if checksums and cache:
        state = dst / STATE_FOLDER
        state.mkdir(parents=True, exist_ok=True)
        (state / 'checksums.json').write_text(json.dumps(cache), encoding='utf-8')
    table = pd.DataFrame(rows, columns=['path', 'status', 'expected_size',
                                        'local_size', 'detail'])
    for column in ('expected_size', 'local_size'):
        table[column] = pd.to_numeric(table[column], errors='coerce').astype('Int64')
    return table


def _pid_alive(pid: int) -> bool:
    """Whether a process with this id is running.

    :param pid: the process id.
    :returns: True when it is running, or when that cannot be told.
    """
    try:
        import psutil

        return bool(psutil.pid_exists(pid))
    except Exception:
        return True


def _partial_for(target: Path) -> Optional[Path]:
    """This process's partial file for ``target``, adopting a stale one.

    A ``.part`` left by a process that is no longer running is renamed to
    this process's name so its bytes are resumed; the largest one wins and
    the others are removed.

    :param target: the final file.
    :returns: the partial path, or None when another running process is
        fetching the same file.
    """
    own = target.with_name(f'{target.name}.{os.getpid()}.part')
    others = []
    for candidate in target.parent.glob(f'{glob.escape(target.name)}.*.part'):
        match = re.fullmatch(re.escape(target.name) + r'\.(\d+)\.part', candidate.name)
        if match is None or candidate == own:
            continue
        if _pid_alive(int(match.group(1))):
            return None
        others.append(candidate)
    others.sort(key=lambda path: path.stat().st_size, reverse=True)
    for index, stale in enumerate(others):
        if index == 0 and not (own.exists() and own.stat().st_size >= stale.stat().st_size):
            os.replace(stale, own)
        else:
            stale.unlink(missing_ok=True)
    return own


def _total_from(response: Any, have: int) -> Optional[int]:
    """The full file size a response announces.

    :param response: the open response.
    :param have: bytes already on disk, counted when the response is a
        range.
    :returns: the size, or None when the server does not say.
    """
    headers = getattr(response, 'headers', None) or {}
    span = headers.get('Content-Range') if hasattr(headers, 'get') else None
    if span and '/' in span and not span.endswith('/*'):
        return int(span.rsplit('/', 1)[1])
    length = headers.get('Content-Length') if hasattr(headers, 'get') else None
    if length is not None and str(length).isdigit():
        return int(length) + have
    return None


class DownloadError(OSError):
    """A file could not be fetched after every retry."""


def download_file(url: str, target: Any, *, size: Optional[int] = None,
                  digest: Optional[tuple] = None, retries: int = 5,
                  timeout: float = 60.0,
                  open_url: Callable[..., Any] = urllib.request.urlopen,
                  sleep: Callable[[float], None] = time.sleep) -> int:
    """Fetch one file, resuming from a partial copy and retrying.

    The bytes go to ``<target>.<process id>.part``; a ``.part`` left by a
    stopped process is resumed with an HTTP ``Range`` request. A server
    that ignores the range is restarted from the first byte. The partial
    file is renamed onto ``target`` only once it has the announced size and,
    when given, the expected digest.

    :param url: the address (``https://``, ``http://``, ``ftp://`` or
        ``file://``).
    :param target: the final path.
    :param size: the size the archive publishes; None trusts the server's.
    :param digest: ``(kind, hex)`` the finished file must have, or None.
    :param retries: further attempts after the first.
    :param timeout: seconds to wait on the server.
    :param open_url: called as ``open_url(request, timeout=timeout)``.
    :param sleep: called with the seconds to wait between attempts.
    :returns: the size of the finished file.
    :raises DownloadError: when every attempt failed, or another running
        process is fetching the same file.
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = _partial_for(target)
    if partial is None:
        raise DownloadError(f'another running process is fetching {target.name}')
    last: Optional[BaseException] = None
    for attempt in range(max(0, int(retries)) + 1):
        if attempt:
            sleep(min(60.0, 2.0 ** attempt))
        have = partial.stat().st_size if partial.exists() else 0
        if size is not None and have > size:
            partial.unlink()
            have = 0
        try:
            if size is None or have < size:
                request = urllib.request.Request(url)
                if have:
                    request.add_header('Range', f'bytes={have}-')
                with open_url(request, timeout=timeout) as response:
                    status = getattr(response, 'status', None)
                    if have and status != 206:
                        have = 0
                    total = _total_from(response, have)
                    with open(partial, 'ab' if have else 'wb') as handle:
                        shutil.copyfileobj(response, handle, _CHUNK)
                expected = size if size is not None else total
            else:
                expected = size
            got = partial.stat().st_size
            if expected is not None and got != expected:
                raise DownloadError(f'{target.name}: received {got} of {expected} bytes')
            if digest is not None and _file_digest(partial, digest[0]) != digest[1]:
                partial.unlink()
                raise DownloadError(f'{target.name}: {digest[0]} does not match the archive')
            os.replace(partial, target)
            return got
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code == 416:
                partial.unlink(missing_ok=True)
            elif exc.code in (401, 403, 404, 410):
                break
        except (urllib.error.URLError, http.client.HTTPException, OSError) as exc:
            last = exc
    raise DownloadError(f'{target.name}: {last}')


def _resolve(settings: Mapping[str, Any]) -> tuple:
    """The base address and the manifest frame a run uses.

    :param settings: the run settings.
    :returns: ``(base_url, manifest)``.
    :raises ValueError: without a destination, or for an archive with
        neither a file list nor a known way to list it.
    """
    archive = str(settings.get('dataset_archive') or 'bia').lower()
    if archive not in ARCHIVES:
        raise ValueError(f'dataset_archive must be one of {ARCHIVES}, not {archive!r}')
    base_url = str(settings.get('dataset_base_url') or '')
    if not base_url and archive != 'url':
        base_url = archive_base_url(archive, str(settings.get('dataset_accession') or ''))
    if base_url and not base_url.endswith('/'):
        base_url += '/'
    source = str(settings.get('dataset_manifest') or '')
    if source:
        manifest = read_manifest(source, base_url=base_url)
    elif archive in ('bia', 'idr'):
        parts = urllib.parse.urlsplit(base_url)
        manifest = list_ftp(urllib.parse.urlunsplit(('ftp', parts.netloc, parts.path, '', '')))
    elif archive == 'bbbc':
        manifest = list_http_index(base_url)
    else:
        raise ValueError("dataset_archive 'url' needs a file list in dataset_manifest")
    pattern = str(settings.get('dataset_include') or '')
    if pattern:
        manifest = manifest[manifest['path'].str.contains(pattern, regex=True)].reset_index(drop=True)
    return base_url, manifest


def _learned_sizes(manifest: pd.DataFrame, dst: Path) -> pd.DataFrame:
    """Fill sizes the archive did not publish from an earlier run's manifest.

    :param manifest: the manifest of this run.
    :param dst: the destination folder.
    :returns: the manifest, with sizes learned from earlier transfers.
    """
    earlier = dst / STATE_FOLDER / 'manifest.csv'
    if not earlier.is_file() or not manifest['size'].isna().any():
        return manifest
    from .tabular import read_table

    previous = read_table(str(earlier), canonicalise=False, report=None,
                          dtype={'path': str})
    known = dict(zip(previous['path'], pd.to_numeric(previous['size'], errors='coerce')))
    missing = manifest['size'].isna()
    manifest.loc[missing, 'size'] = [
        known.get(path) if not pd.isna(known.get(path, float('nan'))) else pd.NA
        for path in manifest.loc[missing, 'path']]
    manifest['size'] = manifest['size'].astype('Int64')
    return manifest


def _fetch_one(record: Mapping[str, Any], status: str, base_url: str, dst: Path,
               retries: int, open_url: Callable[..., Any],
               sleep: Callable[[float], None]) -> tuple:
    """Fetch one manifest file whose status asks for it.

    A ``short`` file is moved to this process's partial name and resumed;
    a file with another fault, or one beside lftp or Aspera leftovers, is
    fetched again from the first byte, and the leftovers are removed once it
    is in place.

    :param record: the manifest row.
    :param status: its completeness status.
    :param base_url: address relative paths are joined to.
    :param dst: the destination folder.
    :param retries: further attempts after the first.
    :param open_url: the opener passed to :func:`download_file`.
    :param sleep: the wait passed to :func:`download_file`.
    :returns: ``(path, size or None, error message)``.
    """
    path = record['path']
    target = dst / path
    url = record['url'] or urllib.parse.urljoin(base_url, urllib.parse.quote(path))
    size = None if pd.isna(record['size']) else int(record['size'])
    digest = next(((kind, record[kind]) for kind in ('sha256', 'md5') if record[kind]), None)
    own = target.with_name(f'{target.name}.{os.getpid()}.part')
    try:
        if status == 'short' and target.exists() and not own.exists():
            os.replace(target, own)
        elif status in ('long', 'empty', 'checksum', 'corrupt_zip'):
            own.unlink(missing_ok=True)
        if status == 'partial':
            for suffix in ('.lftp-pget-status', '.aspx', '.partial', '.aspx.partial'):
                if (target.parent / f'{target.name}{suffix}').exists():
                    own.unlink(missing_ok=True)
                    break
        got = download_file(url, target, size=size, digest=digest, retries=retries,
                            open_url=open_url, sleep=sleep)
        for suffix in ('.lftp-pget-status', '.aspx', '.partial', '.aspx.partial'):
            (target.parent / f'{target.name}{suffix}').unlink(missing_ok=True)
        return path, got, ''
    except (DownloadError, OSError) as exc:
        return path, None, str(exc)


def _counts(table: pd.DataFrame) -> Dict[str, int]:
    """Files per status, every status listed.

    :param table: a completeness table.
    :returns: ``{status: count}`` in :data:`STATUSES` order.
    """
    counted = table['status'].value_counts().to_dict()
    return {status: int(counted.get(status, 0)) for status in STATUSES}


def fetch_dataset(settings: Optional[Mapping[str, Any]] = None, *,
                  open_url: Callable[..., Any] = urllib.request.urlopen,
                  sleep: Callable[[float], None] = time.sleep) -> Dict[str, Any]:
    """Fetch a public dataset's missing and bad files, then check it.

    Reads or lists the archive's file list, checks the destination against
    it (:func:`check_completeness`), fetches every file whose status is in
    :data:`REFETCH` (unless ``dataset_check_only``), and checks again. A
    second run fetches only what the first left missing or bad.

    Written to ``<dataset_dst>/_spacr_dataset_check/``, through
    :func:`spacr.tabular.write_table`: ``manifest.csv`` (the file list
    with the sizes learned from transfers), ``completeness.csv`` (one row
    per file with its status), and ``summary.json`` (the counts before and
    after, and the files that failed).

    :param settings: the keys of :func:`public_dataset_defaults`; missing
        keys take their defaults.
    :param open_url: opener for every transfer, ``urllib.request.urlopen``
        by default.
    :param sleep: wait between retries, ``time.sleep`` by default.
    :returns: dict with ``manifest`` and ``completeness`` (the frames),
        ``before`` and ``after`` (counts per status), ``fetched`` (paths
        fetched), ``failed`` (``{path: reason}``), ``complete`` (no file
        missing, empty, short, long, partial or bad) and ``paths`` (the
        three files written).
    :raises ValueError: without ``dataset_dst``, or for settings
        :func:`archive_base_url` or :func:`read_manifest` reject.
    """
    from .resource_log import _guard_workers
    from .tabular import write_table

    settings = public_dataset_defaults(settings)
    if not settings['dataset_dst']:
        raise ValueError('dataset_dst must name the folder to fetch into')
    dst = Path(os.path.expanduser(str(settings['dataset_dst'])))
    dst.mkdir(parents=True, exist_ok=True)
    base_url, manifest = _resolve(settings)
    manifest = _learned_sizes(manifest, dst)
    checksums = bool(settings['dataset_checksums'])
    workers = max(1, int(_guard_workers('public_datasets', int(settings['dataset_workers']),
                                        _CHUNK * 4, settings=settings) or 1))
    _say('Checking {count} files against the archive file list.', count=len(manifest))
    before = check_completeness(manifest, dst, checksums=checksums, workers=workers)
    fetched: List[str] = []
    failed: Dict[str, str] = {}
    todo = before[before['status'].isin(REFETCH)]
    if not settings['dataset_check_only'] and len(todo):
        _say('Fetching {count} missing or incomplete files with {workers} workers.',
             count=len(todo), workers=workers)
        status_of = dict(zip(todo['path'], todo['status']))
        records = [record for record in manifest.to_dict('records')
                   if record['path'] in status_of]
        retries = int(settings['dataset_retries'])
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(
                lambda record: _fetch_one(record, status_of[record['path']], base_url,
                                          dst, retries, open_url, sleep),
                records))
        learned = {}
        for path, got, error in results:
            if error:
                failed[path] = error
            else:
                fetched.append(path)
                learned[path] = got
        unknown = manifest['size'].isna() & manifest['path'].isin(learned)
        manifest.loc[unknown, 'size'] = [learned[path] for path in manifest.loc[unknown, 'path']]
        manifest['size'] = manifest['size'].astype('Int64')
        after = check_completeness(manifest, dst, checksums=checksums, workers=workers)
    else:
        after = before
    if failed:
        after.loc[after['path'].isin(failed) & (after['status'] == 'missing'), 'status'] = 'failed'
        after['detail'] = [failed.get(path, detail)
                           for path, detail in zip(after['path'], after['detail'])]
    state = dst / STATE_FOLDER
    paths = {'manifest': write_table(manifest, state / 'manifest.csv', canonicalise=False),
             'completeness': write_table(after, state / 'completeness.csv', canonicalise=False)}
    counts_before, counts_after = _counts(before), _counts(after)
    complete = all(counts_after[status] == 0 for status in STATUSES
                   if status not in ('ok', 'extra'))
    summary = {'base_url': base_url, 'files': int(len(manifest)), 'before': counts_before,
               'after': counts_after, 'fetched': len(fetched), 'failed': failed,
               'complete': complete}
    (state / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    paths['summary'] = str(state / 'summary.json')
    _say('{ok} of {count} files complete; {bad} missing or incomplete, {extra} extra. '
         'Table: {table}', ok=counts_after['ok'], count=len(manifest),
         bad=sum(counts_after.values()) - counts_after['ok'] - counts_after['extra'],
         extra=counts_after['extra'], table=paths['completeness'])
    return {'manifest': manifest, 'completeness': after, 'before': counts_before,
            'after': counts_after, 'fetched': fetched, 'failed': failed,
            'complete': complete, 'paths': paths}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run :func:`fetch_dataset` from the command line.

    ``python -m spacr.public_datasets --archive bbbc --accession BBBC021
    --manifest bbbc021_urls.txt --dst /data/BBBC021`` fetches what is
    missing; ``--check-only`` only writes the completeness table.

    :param argv: the arguments; None reads :data:`sys.argv`.
    :returns: 0 when every file is complete, 1 otherwise.
    """
    parser = argparse.ArgumentParser(prog='python -m spacr.public_datasets',
                                     description=_DESCRIPTION)
    parser.add_argument('--archive', default=_DEFAULTS['dataset_archive'], choices=ARCHIVES)
    parser.add_argument('--accession', default='')
    parser.add_argument('--manifest', default='')
    parser.add_argument('--base-url', default='')
    parser.add_argument('--dst', required=True)
    parser.add_argument('--include', default='')
    parser.add_argument('--workers', type=int, default=_DEFAULTS['dataset_workers'])
    parser.add_argument('--retries', type=int, default=_DEFAULTS['dataset_retries'])
    parser.add_argument('--no-checksums', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args(argv)
    result = fetch_dataset({
        'dataset_archive': args.archive, 'dataset_accession': args.accession,
        'dataset_manifest': args.manifest, 'dataset_base_url': args.base_url,
        'dataset_dst': args.dst, 'dataset_include': args.include,
        'dataset_workers': args.workers, 'dataset_retries': args.retries,
        'dataset_checksums': not args.no_checksums,
        'dataset_check_only': args.check_only})
    return 0 if result['complete'] else 1


if __name__ == '__main__':
    sys.exit(main())
