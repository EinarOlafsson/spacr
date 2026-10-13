"""Package unchanged proven snapshot bytes and an explicit relocation helper."""
import json
from pathlib import Path
import zipfile
from build_evaluation_example import sha
from stage_lesson import DEFAULT_STAGE,REPO,read


def build():
    proof=read(DEFAULT_STAGE/'run_compare_verified_snapshots_v1/preparation.json')
    if not proof['accepted'] or not proof['original_sources_unchanged']:
        raise ValueError('Require accepted actual snapshots')
    root=Path(__file__).resolve().parent
    contents={'README.txt':(root/'run_compare_README.txt').read_bytes(),
              'prepare_run_compare_download.py':(root/'prepare_run_compare_download.py').read_bytes()}
    records=[]
    for record in proof['records']:
        original=record['original'];transported=record['transported'];source=Path(transported['path'])
        if sha(source)!=original['fingerprint']:raise ValueError('A real snapshot changed')
        contents['snapshots/'+original['run_id']+'/measurements.db']=source.read_bytes()
        note=json.loads(transported['extra_json'])['tutorial_relocation']
        records.append(dict(original=original,original_registry_sha256=note['original_registry_sha256']))
    contents['source_records.json']=(json.dumps(dict(records=records),indent=2)+'\n').encode()
    target=REPO/'docs/source/_extra/tutorials/examples/Run_Compare_real_snapshots.zip'
    if target.exists():raise FileExistsError('Preserve the previous bundle')
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,data in contents.items():
            info=zipfile.ZipInfo(name,date_time=(2026,9,11,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(info,data)
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() or any(archive.read(name)!=data for name,data in contents.items()):
            raise ValueError('Archived bytes differ')
    print(target,sha(target))


if __name__=='__main__':build()
