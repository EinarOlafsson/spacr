"""Reuse genuine simulator footage to teach its limitations, not endorse advice."""
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import subprocess
import zipfile
from build_evaluation_example import sha
from compose_report_capture import _frame, _read
from power_evidence import check_curves
from power_review import review_baselines
from stage_lesson import DEFAULT_STAGE, REPO, write

APPLICATION_FILES = ('spacr/qt/screens/power.py', 'spacr/qt/widgets/power_design.py',
                     'spacr/power_model.py', 'spacr/power_simulate.py')


def prepare():
    source = DEFAULT_STAGE / 'captures/power_real'
    destination = DEFAULT_STAGE / 'captures/power_critical_review_v1'
    if destination.exists(): raise FileExistsError('Preserve previous composition')
    hashes = {}; provenance = _read(source/'provenance.json', hashes)
    hold = _read(source/'scientific_acceptance.json', hashes)
    records = _read(source/'simulation_result.json', hashes)
    if not (provenance['completed_capture'] and hold['accepted'] is False
            and hold['fit_kwargs_overridden'] is False and hold['caveat_toggle_then_restoration']):
        raise ValueError('Require actual completed controls and the preserved scientific-advice hold')
    for name in APPLICATION_FILES:
        old = subprocess.check_output(['git','show',provenance['commit']+':'+name], cwd=REPO)
        if old != (REPO/name).read_bytes(): raise ValueError('Application source changed: '+name)
        hashes[str(REPO/name)] = sha(REPO/name)
    count = check_curves(records, records['spec']['detection_auroc'], records['spec']['n_replicates'])
    if count != 18 or records['n_clipped_screens'] != 2:
        raise ValueError('Expected the recorded eighteen fits and two clipped simulations')
    review = review_baselines(records)
    curves = records['cells_curve'] + records['wells_curve']
    table = records['table']
    if len(table) != len(curves): raise ValueError('Actual table omits a curve point')
    for index, (shown, curve) in enumerate(zip(table, curves)):
        coordinates = ([curve['value'], 96] if index < len(records['cells_curve']) else [120, curve['value']])
        if [float(shown[0]),float(shown[1])] != coordinates:
            raise ValueError('Actual table coordinates differ')
        if shown[2] != f"{curve['power']:.0%}" or shown[3] != f"{curve['n_detected']}/{curve['n_replicates']}":
            raise ValueError('Actual table detections or denominator differ')
        for column, key, tolerance in [(4,'mean_auroc',.005000001),(5,'mean_ap',.005000001),(6,'ap_baseline',.000500001)]:
            value = curve[key]
            if value is None:
                if shown[column] != '—': raise ValueError('Unavailable mean presented as a number')
            elif not math.isclose(float(shown[column]),value,rel_tol=0,abs_tol=tolerance):
                raise ValueError('Actual table mean differs beyond its displayed precision')
        if [int(shown[7]),int(shown[8])] != [curve['n_not_converged'],curve['n_failed']]:
            raise ValueError('Actual table fit-status counts differ')
    frames = {}
    for key, original in _read(source/'frames.json', hashes).items():
        frame = deepcopy(original)
        path = _frame(source/frame['image'],frame['sha256'],source,hashes)
        frame['image'] = os.path.relpath(path,destination); frames[key] = frame
    if any(sha(path) != value for path,value in hashes.items()): raise ValueError('Evidence changed during review')
    destination.mkdir()
    write(destination/'frames.json',frames)
    write(destination/'provenance.json',dict(completed_capture=True,module='power',sources=[provenance],composition_only=True,app_source_modified=False))
    accepted = dict(accepted=True,scope='Critical reading of actual simulator output, explicitly REJECTING contradictory advice; not calibrated power or sample-size guidance',
        original_scientific_advice_hold=hold,review=review,source_hashes=hashes,
        actual_fit_count=count,actual_table_cells_checked=81,new_simulation=False,
        recommendation_fixed=False,scientific_calibration_approved=False,published=False)
    write(destination/'scientific_acceptance.json',accepted)
    write(REPO/'tools/tutorials/evidence/2026-09-11_power_critical_review_verification.json',accepted)
    contents={'README.txt':(Path(__file__).parent/'power_README.txt').read_bytes(),
              'recorded_simulation_result.json':(source/'simulation_result.json').read_bytes()}
    contents['review.json']=(json.dumps(review,indent=2)+'\n').encode()
    target=REPO/'docs/source/_extra/tutorials/examples/Power_recorded_simulation_review.zip'
    if target.exists():raise FileExistsError('Preserve existing download')
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,data in contents.items():
            info=zipfile.ZipInfo(name,date_time=(2026,9,11,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(info,data)
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() or any(archive.read(name)!=data for name,data in contents.items()):
            raise ValueError('Packaged evidence differs')
    print(destination);print(target,sha(target))


if __name__ == '__main__':prepare()
