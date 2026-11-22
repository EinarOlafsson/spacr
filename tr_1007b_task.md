# Item 1007B translation handover

Branch `ws-offload-codex-work-1007b-int`; assigned base `b9af1a272ba6f01a3aa8f6fd732e45c4cd2f0f3e`; integrated lane `c49ab2979d`.

Home owns the nine translation records and catalog rebuilds. No translation records or pins change on this branch. The companion `tr_1007b_sources.json` contains exact bodies, block layouts, hashes and source locations.

Locales: sv, de, es, zh_CN, pt, hi, ko, is, fr. Preserve paths, identifiers, NGFF versions, format fields and md5.

The converter writes a separate store, links or copies unchanged pixel chunks, verifies each md5 and checks the metadata round trip. The Convert row exposes the forward upgrade as alpha. The existing registration control remains available. The alpha registry still has 178 item IDs because this control extends existing item 1007.

## Public API

`spacr.ome_zarr_convert.upgrade_ngff` — `spacr/ome_zarr_convert.py:1781`; SHA-256 `99be2a49714440a125259e91fc194813122a1e67f82f2c922828c6d85cd5722a`.

Convert an OME-Zarr 0.5 store to 0.6, or a 0.6 store back to 0.5, without touching a pixel.

Item 1007. Every chunk file is hard-linked into ``dst`` (copied when ``link`` is False or
the filesystem refuses the link) and only the group ``zarr.json`` metadata is rewritten,
by :func:`spacr.ome_zarr.multiscale_to_ngff06` (or
:func:`spacr.ome_zarr.multiscale_to_ngff05` for ``to='0.5'``): ``ome.version`` becomes the
target, ``axes`` the intrinsic coordinate system and the transformations of each level the
``{"path"} -> {"name"}`` form. A ``version`` repeated inside ``multiscales``, ``plate``,
``well`` or ``labels`` that equals the store version is dropped, as 0.6 keeps it only at
``ome.version``, and listed in the report. Array metadata, the other OME keys and every
non-OME group are kept byte for byte or key for key.

Anything that has no lossless form in the target version is refused with the reason
before anything is written, and ``src`` is never changed. A hard-linked chunk is the same
file as in the source, so ``dst`` is a new view of the same pixels; edit neither in place.

After writing, the md5 of every chunk file in ``src`` and ``dst`` is compared and every
group is mapped back to the source version, which must give the original metadata (key
order aside). The report is written to ``dst/ngff_upgrade.json``. A run stopped part-way
resumes when called again with the same ``src``, ``dst`` and ``to``.

:param src: the store to convert, a Zarr v3 OME-Zarr 0.5 (or 0.6) group.
:param dst: the new store; it must not exist, or hold an earlier run of the same upgrade.
:param to: ``'0.6'`` (default) or ``'0.5'``.
:param link: hard-link the chunk files (True) or copy them (False); None (default) reads
    ``ome_zarr_upgrade_link`` from ``settings``, else its default in
    :data:`spacr.ops_settings.OPS_DEFAULTS` (True).
:param settings: a settings dict, as supplied by a settings file or the Upgrade row in the Convert tab; read for ``ome_zarr_upgrade_link``.
:returns: the report: ``to``, ``groups``, ``arrays``, ``chunks``, ``linked``, ``copied``,
    ``chunk_list_md5`` (the md5 of the sorted ``path<TAB>md5`` list),
    ``removed_version_keys`` and ``verdict`` (``'verified'``).
:raises ValueError: for an unknown target, a store that is not the source version, a
    Zarr v2 store, a destination inside the source or holding other files, metadata
    without a lossless form, or a failed verification.

`spacr.qt.screens.convert.ConvertScreen.run_ngff_upgrade` — `spacr/qt/screens/convert.py:924`; SHA-256 `a960f9b900e866a5f26eba6f590227125b3fbe8367ec13c6b09f1f4bd5a33e3b`.

Upgrade the chosen OME-Zarr 0.5 store to 0.6 off the GUI thread (item 1007, alpha).

:func:`spacr.ome_zarr_convert.upgrade_ngff` writes a new store beside the source and
verifies every chunk; the report goes to the Summary pane, a refusal to the status
line.

:returns: True when the job was started (or, unthreaded, succeeded).

## Runtime sources

`setting_labels/ome_zarr_upgrade_link`; `spacr/ops_settings.py:73`.

Ome zarr upgrade link

`setting_tooltips/ome_zarr_upgrade_link`; `spacr/ops_settings.py:1887`.

How the OME-Zarr 0.5 to 0.6 converter (Convert, Upgrade OME-Zarr) puts the pixel chunks into the new store: True hard-links each chunk file, so the new store takes almost no space and is the same pixels; False copies them. Copies are made anyway where the disk refuses a link. Only the metadata is rewritten; every chunk is checked byte for byte. Default True. Alpha.

`ui/Array metadata verification failed: {path}`; `spacr/ome_zarr_convert.py:1765`.

Array metadata verification failed: {path}

`ui/Choose the OME-Zarr store and a new destination.`; `spacr/qt/screens/convert.py:937`.

Choose the OME-Zarr store and a new destination.

`ui/Chunk verification failed: {path}`; `spacr/ome_zarr_convert.py:1761`.

Chunk verification failed: {path}

`ui/Destination…`; `spacr/qt/screens/convert.py:217`.

Destination…

`ui/Hard-link chunks`; `spacr/qt/screens/convert.py:220`.

Hard-link chunks

`ui/Metadata round trip failed: {path}`; `spacr/ome_zarr_convert.py:1777`.

Metadata round trip failed: {path}

`ui/New 0.6 store`; `spacr/qt/screens/convert.py:213`.

New 0.6 store

`ui/OME-Zarr 0.5 store`; `spacr/ome_zarr_convert.py:1782`, `spacr/qt/screens/convert.py:925`, `spacr/qt/screens/convert.py:211`, `spacr/qt/screens/convert.py:216`.

OME-Zarr 0.5 store

`ui/OME-Zarr {version} store written and verified.`; `spacr/qt/screens/convert.py:251`.

OME-Zarr {version} store written and verified.

`ui/On: the new store links to the same chunk files and takes almost no space. Off: the chunks are copied. Only the metadata is rewritten, and every chunk is checked.`; `spacr/qt/screens/convert.py:223`.

On: the new store links to the same chunk files and takes almost no space. Off: the chunks are copied. Only the metadata is rewritten, and every chunk is checked.

`ui/Select the OME-Zarr 0.5 store`; `spacr/qt/screens/convert.py:216`.

Select the OME-Zarr 0.5 store

`ui/Select the folder for the 0.6 store`; `spacr/qt/screens/convert.py:219`.

Select the folder for the 0.6 store

`ui/Store…`; `spacr/qt/screens/convert.py:214`.

Store…

`ui/The destination {dst} must be outside the source store {src}.`; `spacr/ome_zarr_convert.py:1827`.

The destination {dst} must be outside the source store {src}.

`ui/Upgrade an OME-Zarr store to 0.6 (Alpha)`; `spacr/qt/screens/convert.py:208`.

Upgrade an OME-Zarr store to 0.6 (Alpha)

`ui/Upgrade verified: {path}`; `spacr/qt/screens/convert.py:947`.

Upgrade verified: {path}

`ui/Upgrading {path} to OME-Zarr 0.6…`; `spacr/qt/screens/convert.py:940`.

Upgrading {path} to OME-Zarr 0.6…

`ui/md5 of the chunk list: {digest}`; `spacr/qt/screens/convert.py:254`.

md5 of the chunk list: {digest}

`ui/upgrade_ngff converts to '0.6' or '0.5'; got {target}`; `spacr/ome_zarr_convert.py:1825`.

upgrade_ngff converts to '0.6' or '0.5'; got {target}

`ui/{dst} already holds files that are not an earlier upgrade; choose a new destination.`; `spacr/ome_zarr_convert.py:1834`.

{dst} already holds files that are not an earlier upgrade; choose a new destination.

`ui/{dst} holds an upgrade of {other} to {version}; choose a new destination.`; `spacr/ome_zarr_convert.py:1838`.

{dst} holds an upgrade of {other} to {version}; choose a new destination.

`ui/{groups} group(s), {arrays} array(s), {chunks} chunk file(s): {linked} linked, {copied} copied, {kept} kept from an earlier run.`; `spacr/qt/screens/convert.py:252`.

{groups} group(s), {arrays} array(s), {chunks} chunk file(s): {linked} linked, {copied} copied, {kept} kept from an earlier run.

`ui/{path} holds Zarr v2 metadata ({name}); upgrade_ngff converts OME-Zarr 0.5 and 0.6 stores, which are Zarr v3. Nothing was converted.`; `spacr/ome_zarr_convert.py:1712`.

{path} holds Zarr v2 metadata ({name}); upgrade_ngff converts OME-Zarr 0.5 and 0.6 stores, which are Zarr v3. Nothing was converted.

`ui/{path} holds a coordinateTransformations group, which OME-Zarr 0.5 cannot hold. Nothing was converted.`; `spacr/ome_zarr_convert.py:1723`.

{path} holds a coordinateTransformations group, which OME-Zarr 0.5 cannot hold. Nothing was converted.

`ui/{path} is not an OME-Zarr 0.5 or 0.6 store: it has no zarr.json (Zarr v3).`; `spacr/ome_zarr_convert.py:1702`.

{path} is not an OME-Zarr 0.5 or 0.6 store: it has no zarr.json (Zarr v3).

`ui/{where} is OME-Zarr {version}; upgrade_ngff converts {source} to {target} only, so nothing was converted.`; `spacr/ome_zarr_convert.py:1675`.

{where} is OME-Zarr {version}; upgrade_ngff converts {source} to {target} only, so nothing was converted.

`ui/{where}: {key} carries version {inner} inside a {version} store; the converter cannot tell which one is right, so nothing was converted.`; `spacr/ome_zarr_convert.py:1654`.

{where}: {key} carries version {inner} inside a {version} store; the converter cannot tell which one is right, so nothing was converted.

## Other documentation

35 changed private or test docstrings are inventoried in the companion JSON; they create no public API identities.

## Validation

Final test counts and inherited landing deferrals are recorded in the commit body and WORK_STATUS.md.
