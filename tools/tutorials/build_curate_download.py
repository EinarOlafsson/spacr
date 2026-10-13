"""Package exact synthetic teaching inputs, not the modified exercise output."""
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import tifffile
from build_evaluation_example import sha
from stage_lesson import DEFAULT_STAGE,REPO,read


def build():
    proof=read(DEFAULT_STAGE/'captures/curate_disclosed_verified_v1/scientific_acceptance.json')
    if not proof['accepted'] or not proof['synthetic']:raise ValueError('Require verified disclosed synthetic example')
    sources=proof['native_observation']['original_inputs'];contents={}
    for path,value in sources.items():
        if sha(path)!=value:raise ValueError('Original input changed')
        if path.endswith('.npy'):
            raw=np.load(path,allow_pickle=False);buffer=io.BytesIO()
            tifffile.imwrite(buffer,raw[...,2]);contents['mask.tif']=buffer.getvalue()
            if not np.array_equal(tifffile.imread(io.BytesIO(contents['mask.tif'])),raw[...,2]):
                raise ValueError('TIFF conversion changed label pixels')
        elif path.endswith('.csv'):contents['tracks.csv']=Path(path).read_bytes()
    root=Path(__file__).resolve().parent
    contents.update({'README.txt':(root/'curate_README.txt').read_bytes(),
        'curate_checkpoint.py':(root/'curate_checkpoint.py').read_bytes(),
        'source_manifest.json':(json.dumps(dict(synthetic=True,original_sha256=sources,label_plane=2),indent=2)+'\n').encode()})
    target=REPO/'docs/source/_extra/tutorials/examples/Curate_SYNTHETIC_practice.zip'
    if target.exists():raise FileExistsError('Preserve previous bundle')
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,data in contents.items():
            info=zipfile.ZipInfo(name,date_time=(2026,9,11,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(info,data)
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() or any(archive.read(name)!=data for name,data in contents.items()):raise ValueError('Archive differs')
    print(target,sha(target))


if __name__=='__main__':build()
