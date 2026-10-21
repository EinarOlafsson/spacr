#!/usr/bin/env python3
"""Verify saved sweep capture values without fitting or changing the experiment."""
import argparse
import csv
import json
import hashlib
from pathlib import Path

from build_evaluation_example import sha
from sweep_evidence import (check_display,check_result_family,count_result_rows,
                            check_trial_set,check_fitted_support,check_support_geometry)


def check_saved_summary(displayed, saved, path):
    """Check the selected trial's saved summary verbatim; never fit a model."""
    if not isinstance(saved, str) or not saved.strip():
        raise ValueError('Saved model summary must be nonempty')
    body = saved.strip()
    prefix = (f"Read from {path}: this is the run's own summary, written when it was "
              "fitted, not recomputed here.\n\n")
    if not isinstance(displayed, str) or not displayed.startswith(prefix):
        raise ValueError('Displayed summary does not name the selected trial source')
    if displayed[len(prefix):] != body:
        raise ValueError('Displayed model summary differs from the saved verbatim body')
    return dict(passed=True, source=str(path), characters=len(body),
                body_sha256=hashlib.sha256(body.encode('utf-8')).hexdigest(),
                saved_summary_checked=True, new_fit_requested=False)


def verify(capture):
    proof=json.loads((capture/'scientific_acceptance.json').read_text(encoding="utf-8"))
    provenance=json.loads((capture/'provenance.json').read_text(encoding="utf-8"))
    work=Path(proof['private_folder'])
    if not provenance['completed_capture']:
        raise ValueError('Expected a completed sweep capture')
    for name,digest in proof['original_inputs'].items():
        if sha(name)!=digest or sha(work/'inputs'/Path(name).name)!=digest:
            raise ValueError('Original or private example input changed')
    rows=list(csv.DictReader((work/'trials/sweep_results.csv').open()))
    trial_set=check_trial_set(rows)
    if len({Path(row['folder']).resolve() for row in rows}) != 2:
        raise ValueError('Trials share the same resolved output folder')
    report=dict(lesson='73_parameter_sweep',source_commit=provenance['commit'],
        capture=str(capture),accepted=False,scope='saved execution and display consistency only',
        inference_validated=False,published=False,new_fits_requested=proof['new_sweep_requested'],
        display=check_display(proof['displayed_headers'],proof['displayed_rows'],rows),
        trial_set=trial_set,trials=[])
    for row in rows:
        folder=Path(row['folder']);path=folder/'results/ridge/results.csv'
        result=list(csv.DictReader(path.open()))
        actual=json.loads((folder/'settings/regression.json').read_text(encoding="utf-8"))
        requested=json.loads((folder/'_trial_settings.json').read_text(encoding="utf-8"))['settings']
        prepared_path=path.with_name('regression_data.csv')
        prepared=list(csv.DictReader(prepared_path.open()))
        qc={}
        qc_hashes={}
        for level in ('grna','gene'):
            qc_path=path.parent/level/'regression_qc/regression_qc_numbers.json'
            if qc_path.exists():
                qc[level]=json.loads(qc_path.read_text(encoding="utf-8"))
                qc_hashes[level]=sha(qc_path)
        fitted_path=path.with_name('regression_fit_designs.json')
        fit_designs=json.loads(fitted_path.read_text(encoding="utf-8"))
        if not isinstance(fit_designs,dict):
            raise ValueError('Missing measured fit records')
        support=check_fitted_support(prepared,row,actual,requested,qc,fit_designs)
        counts=count_result_rows(result)
        if any(counts[k]!=int(row[k]) for k in counts):raise ValueError('Sweep summary count differs')
        if float(actual['alpha'])!=float(row['alpha']) or float(requested['alpha'])!=float(row['alpha']):
            raise ValueError('Requested and actual ridge penalties differ')
        report['trials'].append(dict(trial_id=row['trial_id'],alpha=float(row['alpha']),
            results_sha256=sha(path),counts=counts,qc_inference=row['qc_inference'],
            qc_verdict=row['qc_verdict'],fitted_support=support,
            prepared_data_sha256=sha(prepared_path),qc_sha256=qc_hashes,
            fit_designs_sha256=sha(fitted_path),
            actual_settings_sha256=sha(folder/'settings/regression.json'),
            requested_settings_sha256=sha(folder/'_trial_settings.json')))
    summary_path=Path(rows[0]['folder'])/'results/ridge/model_summary.txt'
    report['saved_model_summary']=check_saved_summary(
        proof['model_summary_text'],summary_path.read_text(encoding='utf-8'),summary_path)
    first=list(csv.DictReader((Path(rows[0]['folder'])/'results/ridge/results.csv').open()))
    report['family_checks']={name:check_result_family(snapshot,first,snapshot['level'])
        for name,snapshot in proof['result_snapshots'].items()}
    if list(report['family_checks'])!=['10_saved_guide_family','11_saved_gene_family','12_saved_guide_family_restored']:
        raise ValueError('Missing guide/gene/restoration evidence')
    geometry=proof['trial_table_geometry']
    if not geometry['both_rows_visible'] or geometry['viewport_height']<2*geometry['row_height']:
        raise ValueError('The recorded trial rows do not fit')
    report.update(accepted=True,original_inputs_preserved=True,trial_table_geometry=geometry,
        visible_fitted_support=check_support_geometry(proof['support_geometry'],rows),
        checked_result_cells=sum(c['table']['cells'] for c in report['family_checks'].values()),
        checked_plot_points=sum(c['plotted_points'] for c in report['family_checks'].values()))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture',type=Path)
    args=parser.parse_args()
    print(json.dumps(verify(args.capture),indent=2))
