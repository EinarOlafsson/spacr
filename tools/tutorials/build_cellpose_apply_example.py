"""Bundle unchanged Apply inputs and the real resolved settings."""
import json
from pathlib import Path
import zipfile

from build_evaluation_example import sha
from stage_lesson import DEFAULT_STAGE, REPO, read


def build():
    root=Path(__file__).resolve().parent
    capture=DEFAULT_STAGE/'captures/cellpose_apply_native_zoom_v1'
    proof=read(capture/'scientific_acceptance.json')
    reference=read(capture/'independent_reference.json')
    if not proof['pipeline']['accepted'] or not reference['accepted']:
        raise ValueError('Require the successful actual run and independent checks')
    contents={'README.txt':(root/'cellpose_apply_README.txt').read_bytes(),
              'recorded_settings.json':(capture/'configured_settings.json').read_bytes()}
    manifest=dict(scope='Real demonstration crops, not ground truth',images=[])
    for path,value in proof['original_inputs'].items():
        source=Path(path)
        if sha(source)!=value:raise ValueError('A recorded image changed')
        contents['images/'+source.name]=source.read_bytes()
        manifest['images'].append(dict(file=source.name,sha256=value))
    if len(manifest['images'])!=3:raise ValueError('Expected exactly three actual images')
    contents['source_manifest.json']=(json.dumps(manifest,indent=2)+'\n').encode()
    target=REPO/'docs/source/_extra/tutorials/examples/Apply_Cellpose_stock_example.zip'
    if target.exists():raise FileExistsError('Preserve the previous bundle')
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,data in contents.items():
            info=zipfile.ZipInfo(name,date_time=(2026,9,11,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(info,data)
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() or set(archive.namelist())!=set(contents):
            raise ValueError('Archive structure or CRC failed')
        if any(archive.read(name)!=data for name,data in contents.items()):
            raise ValueError('Archived bytes differ from the recorded example')
    print(target,sha(target))


if __name__=='__main__':build()
