from pathlib import Path
import difflib,hashlib,json
r=Path('/mnt/wd4tb/spacr-worktrees/codex-f540-finite-threshold-20261015');s=Path('/mnt/wd4tb/scratch/f540-finite-threshold-20261015');p=r/'spacr/measure.py';original=p.read_text()
a='''        try:
            out.append(float(item))
        except (TypeError, ValueError):
            raise ValueError(
                f"Setting: viability_thresholds entries must be numbers or "
                f"None; got {item!r}.") from None
    return out[0], out[1]
'''
b='''        try:
            number = float(item)
        except (TypeError, ValueError):
            raise ValueError(
                f"Setting: viability_thresholds entries must be numbers or "
                f"None; got {item!r}.") from None
        if not np.isfinite(number):
            from .qt.i18n import tr

            raise ValueError(tr("Manual viability thresholds must be finite numbers or automatic."))
        out.append(number)
    return out[0], out[1]
'''
assert original.count(a)==1;updated=original.replace(a,b);(s/'proposed-measure.py').write_text(updated);(s/'production.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),updated.splitlines(True),fromfile='a/spacr/measure.py',tofile='b/spacr/measure.py')));(s/'proposal.json').write_text(json.dumps({'original_sha256':hashlib.sha256(original.encode()).hexdigest(),'proposed_sha256':hashlib.sha256(updated.encode()).hexdigest(),'source':'Manual viability thresholds must be finite numbers or automatic.','scope':'Exactly _viability_manual; float parsing and malformed-input/automatic conventions unchanged; require finite converted numeric scalar before returning explicit manual cuts.'},indent=2)+'\n')
