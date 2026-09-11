"""Package unchanged verified cell pairs and the exact recorded API helper."""
import hashlib
from pathlib import Path
import zipfile

from stage_lesson import DEFAULT_STAGE, REPO, read


def build():
    root=Path(__file__).resolve().parent
    receipt=read(root/'evidence/2026-09-11_cellpose_explicit_training_checks.json')
    helper=root/'train_cellpose_example.py'
    if hashlib.sha256(helper.read_bytes()).hexdigest()!=receipt['helper_sha256']:
        raise ValueError('The downloadable helper must match the recorded training')
    source=DEFAULT_STAGE/'derived/train_cellpose_corrected'
    contents={'train_cellpose_example.py':helper.read_bytes(),
              'README.txt':(root/'cellpose_training_README.txt').read_bytes(),
              'recorded_settings.json':(Path(receipt['training']['destination'])/'requested_settings.json').read_bytes()}
    for name,digest in receipt['training']['source_hashes'].items():
        path=Path(name)
        data=path.read_bytes()
        if not path.is_relative_to(source) or hashlib.sha256(data).hexdigest()!=digest:
            raise ValueError('The verified source manifest or pair changed')
        contents['data/'+path.relative_to(source).as_posix()]=data
    if len(contents)!=16:
        raise ValueError('Expected twelve TIFFs, manifest, recorded settings, helper and README')
    target=REPO/'docs/source/_extra/tutorials/examples/Train_Cellpose_API_example.zip'
    if target.exists():raise FileExistsError('Preserve the existing training bundle')
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,data in contents.items():
            info=zipfile.ZipInfo(name,date_time=(2026,9,11,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(info,data)
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() or set(archive.namelist())!=set(contents):
            raise ValueError('The archive structure or CRC failed')
        if any(archive.read(name)!=value for name,value in contents.items()):
            raise ValueError('Archived source differs from the verified training input')
    print(target,hashlib.sha256(target.read_bytes()).hexdigest())


if __name__=='__main__':build()
