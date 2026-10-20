from pathlib import Path
import difflib,hashlib,json
r=Path('/mnt/wd4tb/spacr-worktrees/codex-f537-parent-identity-20261015');s=Path('/mnt/wd4tb/scratch/f537-parent-identity-20261015')
p=r/'spacr/_lineage_trees.py';original=p.read_text()
before="    firsts = df.dropna(subset=['parent_track_id']).groupby('track_id')['parent_track_id'].first()\n"
after='''    grouped = df.dropna(subset=['parent_track_id']).groupby('track_id')['parent_track_id']
    for track, values in grouped:
        known = set()
        for value in values:
            try:
                parent = int(value)
            except (TypeError, ValueError, OverflowError):
                continue
            if not isinstance(value, (str, bytes)) and value != parent:
                continue
            if parent > 0 and parent != int(track) and parent in spans.index:
                known.add(parent)
        if len(known) > 1:
            from .qt.i18n import tr

            raise ValueError(tr("Track {track} has conflicting known parent IDs in the lineage table.").format(track=track))
    firsts = grouped.first()
'''
assert original.count(before)==1
updated=original.replace(before,after)
(s/'proposed-_lineage_trees.py').write_text(updated)
(s/'production.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),updated.splitlines(True),fromfile='a/spacr/_lineage_trees.py',tofile='b/spacr/_lineage_trees.py')))
(s/'proposal.json').write_text(json.dumps({'original_sha256':hashlib.sha256(original.encode()).hexdigest(),'proposed_sha256':hashlib.sha256(updated.encode()).hexdigest(),'source':'Track {track} has conflicting known parent IDs in the lineage table.','scope':'Exactly _lineage_explicit_parents; firsts and existing selection unchanged, new prepass only refuses >1 known positive nonself ID present in spans; no pd.to_numeric/float coercion.'},indent=2)+'\n')
