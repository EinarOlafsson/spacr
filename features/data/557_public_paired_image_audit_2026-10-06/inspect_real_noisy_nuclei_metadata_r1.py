from pathlib import Path, PurePosixPath
import hashlib
import io
import json
import zipfile

import requests
import tifffile

scratch=Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
root=scratch/'557-noisy-nuclei-primary-inspection-r1'
root.mkdir(exist_ok=False)
url='https://zenodo.org/records/5750174'
response=requests.get(url,timeout=90)
response.raise_for_status()
(root/'primary-record.html').write_text(response.text)
archive_url=url+'/files/Denoising_Dataset.zip?download=1'
archive=root/'Denoising_Dataset.zip'
md5=hashlib.md5();sha=hashlib.sha256();size=0
with requests.get(archive_url,stream=True,timeout=90) as response:
    response.raise_for_status()
    with archive.open('wb') as stream:
        for block in response.iter_content(1024*1024):
            stream.write(block);md5.update(block);sha.update(block);size+=len(block)
assert md5.hexdigest()=='5af4937e6a43d5bce5e6119abb118a2a'
rows=[];texts={}
with zipfile.ZipFile(archive) as source:
    for member in source.infolist():
        path=PurePosixPath(member.filename)
        assert '..' not in path.parts and not path.is_absolute()
        if member.is_dir() or any(p=='__MACOSX' or p.startswith('.') for p in path.parts):continue
        payload=source.read(member)
        if path.suffix.lower() in ('.txt','.md','.csv','.xml','.json'):
            texts[member.filename]=payload.decode('utf-8',errors='replace')
        elif path.suffix.lower() in ('.tif','.tiff'):
            with tifffile.TiffFile(io.BytesIO(payload)) as image:
                tags={str(tag.name):str(tag.value)[:20000] for tag in image.pages[0].tags.values()}
                array=image.asarray()
                rows.append({'archive_member':member.filename,'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest(),'shape':list(array.shape),'dtype':str(array.dtype),'original_pixels_sha256':hashlib.sha256(array.tobytes()).hexdigest(),'tags':tags,'OME_metadata':image.ome_metadata,'imagej_metadata':image.imagej_metadata})
record={'original_primary_URL':url,'original_DOI':'10.5281/zenodo.5750174','archive_URL':archive_url,'published_MD5_verified':True,'MD5':md5.hexdigest(),'sha256':sha.hexdigest(),'bytes':size,'TIFF_originals':rows,'text_originals':texts,'actual_noise_exposure_replication_annotation_and_pixel_pair_alignment_still_require_review':True,'no_low_light_calibration_GPU_training_or_F1_acceptance_claim':True}
(root/'inspection.json').write_text(json.dumps(record,indent=2,default=str)+'\n')
print('PASS: original public real noisy/high-SNR archive MD5 verified; metadata inventoried without alteration; TIFF originals',len(rows),'text files',list(texts),'; calibration/pair alignment and scientific acceptance not inferred.',flush=True)
