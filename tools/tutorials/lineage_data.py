"""Isolate the genuine Annotate database and every existing crop for lesson56."""
from pathlib import Path
import shutil
import tempfile

from capture_database import _digest, _readonly, prepare_database_copy, require_unchanged_source


def prepare(stage):
    stage=Path(stage);base=stage/'annotate_fresh/example_data'
    source=base/'plate1/measurements/measurements.db'
    parent=stage/'lineage_runs';parent.mkdir(exist_ok=True)
    work=Path(tempfile.mkdtemp(prefix='real-containment-',dir=parent))
    cache=work/'example_data';database=cache/'plate1/measurements/measurements.db'
    database.parent.mkdir(parents=True)
    original=prepare_database_copy(source,database,
        expected_sha256='7b18161f0161d39b3ecedf92cfb0ccf9fee2328980da8e43167555a8f6fd27cd')
    prefix=Path('/home/olafsson/.cache/spacr/example_data');crops=[]
    with _readonly(source) as db:
        rows=db.execute('SELECT png_path FROM png_list').fetchall()
    if len(rows)!=2341:raise ValueError('Expected all2341 genuine downloaded cell crops')
    for (name,) in rows:
        relative=Path(name).relative_to(prefix)
        if not relative.parts or relative.parts[0]!='plate1' or '..' in relative.parts:
            raise ValueError('Crop must remain within the private plate1 copy')
        path=(base/relative).resolve(strict=True)
        if not path.is_relative_to(base.resolve()):raise ValueError('Crop escapes original dataset')
        destination=cache/relative
        if destination.exists():raise ValueError('Repeated source crop path')
        digest=_digest(path);destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,destination)
        if _digest(destination)!=digest or _digest(path)!=digest:raise ValueError('Real crop changed during private copy')
        crops.append(dict(relative=str(relative),source=str(path),sha256=digest,bytes=path.stat().st_size))
    require_unchanged_source(source,original['source_bundle'])
    return dict(source=original,cache=str(cache),project=str(cache/'plate1'),work=str(work),
                crops=crops,source_images_generated=False,published=False)


def verify_preserved(prepared):
    original=prepared['source'];require_unchanged_source(original['source'],original['source_bundle'])
    for record in prepared['crops']:
        if _digest(record['source'])!=record['sha256'] or _digest(Path(prepared['cache'])/record['relative'])!=record['sha256']:
            raise ValueError('Original or private crop changed during Lineage demonstration')
    return dict(original_database_and_sidecars_unchanged=True,all_original_and_private_crops_unchanged=True,
                crop_count=len(prepared['crops']),crop_bytes=sum(r['bytes'] for r in prepared['crops']),
                private_database_bytes_unchanged=_digest(original['database'])==original['database_sha256'])
