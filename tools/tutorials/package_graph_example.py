"""Package the exact recorded database; no derived or invented measurement rows."""
import json
from pathlib import Path
import zipfile
from build_evaluation_example import sha
from stage_lesson import DEFAULT_STAGE, REPO, read, write


def package(capture_name='graph_native_review_v2'):
    capture = DEFAULT_STAGE/'captures'/capture_name
    proof = read(capture/'scientific_acceptance.json')
    provenance = read(capture/'provenance.json')
    if not provenance['completed_capture'] or not proof['accepted']:
        raise ValueError('Native chart checks must finish first')
    if proof['annotation_handoff_fixed'] or proof['brush_review']['annotation_handoff_works']:
        raise ValueError('The recorded handoff defect must not be reported repaired')
    database = Path(proof['database'])
    if sha(database) != proof['database_sha256']:
        raise ValueError('Recorded source database changed')
    manifest = dict(database_sha256=sha(database), database_bytes=database.stat().st_size,
                    native_verification=proof, provenance=provenance,
                    native_chart_export=False, synthetic_data=False, published=False)
    target = REPO/'docs/source/_extra/tutorials/examples/Graph_Builder_real_measurements.zip'
    if target.exists():
        raise FileExistsError('Preserve existing downloadable evidence')
    entries = {'measurements.db':database.read_bytes(),
               'README.txt':(Path(__file__).with_name('graph_README.txt')).read_bytes(),
               'source_manifest.json':(json.dumps(manifest,indent=2)+'\n').encode()}
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in entries.items():
            info=zipfile.ZipInfo(name,date_time=(2026,9,11,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED;archive.writestr(info,data)
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() or any(archive.read(name)!=data for name,data in entries.items()):
            raise ValueError('Packaged bytes differ')
    manifest.update(download_sha256=sha(target), download=str(target))
    write(REPO/'tools/tutorials/evidence/2026-09-11_graph_native_verification.json',manifest)
    print(target,sha(target))


if __name__ == '__main__':package()
