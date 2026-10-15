from pathlib import Path
import hashlib,json,difflib
r=Path('/mnt/wd4tb/spacr-worktrees/codex-measured-channel-identity-20261015');s=Path('/mnt/wd4tb/scratch/measured-channel-identity-20261015');original=(r/'spacr/measure.py').read_text();updated=original
needle='def _cell_cycle_channel(settings):\n'
helper='''def _measurement_channel_index(value):
    """Normalize a physical image-channel identity without truncating it.

    :param value: Numeric scalar or integer string/bytes; zero-dimensional
        arrays are unboxed before checking their scalar value.
    :returns: a nonnegative Python integer with the same identity.
    :raises ValueError: boolean, nonintegral, nonfinite, nonscalar, negative
        or malformed channel identities.
    """
    try:
        if np.ndim(value) != 0:
            raise ValueError
        if isinstance(value, np.ndarray):
            value = value.item()
        if isinstance(value, (bool, np.bool_, complex, np.complexfloating, np.ndarray)):
            raise ValueError
        index = int(value)
        if index < 0 or (not isinstance(value, (str, bytes)) and value != index):
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        from .qt.i18n import tr

        raise ValueError(tr("Measurement channels must be nonnegative integer indices.")) from None
    return index


'''
assert updated.count(needle)==1;updated=updated.replace(needle,helper+needle)
a='''            return int(value)
    channels = settings.get('channels') or [0]
    return int(channels[0])
''';b='''            return _measurement_channel_index(value)
    channels = settings.get('channels') or [0]
    return _measurement_channel_index(channels[0])
''';assert updated.count(a)==1;updated=updated.replace(a,b)
a="    measured = [int(c) for c in (settings.get('channels') or [])]\n    if not measured:\n        return int(channel)\n    if int(channel) not in measured:\n";b="    channel = _measurement_channel_index(channel)\n    measured = [_measurement_channel_index(c) for c in (settings.get('channels') or [])]\n    if not measured:\n        return channel\n    if channel not in measured:\n";assert updated.count(a)==1;updated=updated.replace(a,b).replace('    return measured.index(int(channel))\n','    return measured.index(channel)\n')
a="    if 'nucleus' not in database_tables(db_path):\n";b="    channel = _cell_cycle_channel(settings)\n    column = _measured_channel_column(settings, channel)\n    fucci = settings.get('cell_cycle_fucci_channels')\n    fucci_columns = [_measured_channel_column(settings, c) for c in (fucci or [])]\n    if 'nucleus' not in database_tables(db_path):\n";assert updated.count(a)==1;updated=updated.replace(a,b)
a="    nuclei = nuclei.reset_index(drop=True)\n    channel = _cell_cycle_channel(settings)\n    column = _measured_channel_column(settings, channel)\n";b="    nuclei = nuclei.reset_index(drop=True)\n";assert updated.count(a)==1;updated=updated.replace(a,b)
a="        fucci = settings.get('cell_cycle_fucci_channels') or []\n        columns = [column] + [_measured_channel_column(settings, c)\n                              for c in fucci]\n";b="        columns = [column] + fucci_columns\n";assert updated.count(a)==1;updated=updated.replace(a,b)
a="    fucci = settings.get('cell_cycle_fucci_channels')\n    if fucci:\n        table['fucci_state']";b="    if fucci:\n        table['fucci_state']";assert updated.count(a)==1;updated=updated.replace(a,b)
old="    dead_channel = settings.get('viability_dead_channel')\n    live_channel = settings.get('viability_live_channel')\n    dead_channel = None if dead_channel in (None, '') else int(dead_channel)\n    live_channel = None if live_channel in (None, '') else int(live_channel)\n    method = ('stain' if dead_channel is not None or live_channel is not None\n              else 'morphology')\n";assert updated.count(old)==1;updated=updated.replace(old,'')
pref='''    dead_channel = settings.get('viability_dead_channel')
    live_channel = settings.get('viability_live_channel')
    dead_channel = None if dead_channel in (None, '') else _measurement_channel_index(dead_channel)
    live_channel = None if live_channel in (None, '') else _measurement_channel_index(live_channel)
    method = ('stain' if dead_channel is not None or live_channel is not None
              else 'morphology')
    if method == 'morphology':
        channel = settings.get('nucleus_channel')
        if channel in (None, ''):
            channel = (settings.get('channels') or [0])[0]
        morphology_column = _measured_channel_column(settings, channel,
                                                     setting='nucleus_channel')
    else:
        dead_column = (None if dead_channel is None else _measured_channel_column(
            settings, dead_channel, setting='viability_dead_channel'))
        live_column = (None if live_channel is None else _measured_channel_column(
            settings, live_channel, setting='viability_live_channel'))
'''
a="    manual_dead, manual_live = _viability_manual(\n        settings.get('viability_thresholds'))\n    tables = database_tables(db_path)";b="    manual_dead, manual_live = _viability_manual(\n        settings.get('viability_thresholds'))\n"+pref+"    tables = database_tables(db_path)";assert updated.count(a)==1;updated=updated.replace(a,b)
a="        channel = settings.get('nucleus_channel')\n        if channel in (None, ''):\n            channel = (settings.get('channels') or [0])[0]\n        column = _measured_channel_column(settings, int(channel),\n                                          setting='nucleus_channel')\n";b="        column = morphology_column\n";assert updated.count(a)==1;updated=updated.replace(a,b)
a="            column = _measured_channel_column(\n                settings, dead_channel, setting='viability_dead_channel')\n";assert updated.count(a)==1;updated=updated.replace(a,'            column = dead_column\n')
a="            column = _measured_channel_column(\n                settings, live_channel, setting='viability_live_channel')\n";assert updated.count(a)==1;updated=updated.replace(a,'            column = live_column\n')
(s/'proposed-measure.py').write_text(updated);(s/'production.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),updated.splitlines(True),fromfile='a/spacr/measure.py',tofile='b/spacr/measure.py')));(s/'proposal.json').write_text(json.dumps({'original_sha256':hashlib.sha256(original.encode()).hexdigest(),'proposed_sha256':hashlib.sha256(updated.encode()).hexdigest(),'source':'Measurement channels must be nonnegative integer indices.','scope':['new private _measurement_channel_index (no new module/API prose)','_cell_cycle_channel: normalize before returning physical index','_measured_channel_column: normalize requested/all measured identities without float conversion; preserve ordering/empty/duplicate mapping','_classify_cell_cycle: resolve DNA/FUCCI column identities before DB listing/reads; use same resolved columns downstream','_classify_viability: normalize active stain/morphology channels and resolve columns before DB listing/reads; reuse exact positions downstream'],'limits':'No numeric changes, phase fitting/training, stain cut or calibration math. Only invalid-setting refusal precedes DB errors/migrations. Existing automatic cut/default channel conventions preserved; channel literal auto remains existing malformed input. None inactive stains/nucleus fallback preserved.'},indent=2)+'\n')
