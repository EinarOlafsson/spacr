#!/usr/bin/env python3
"""Verify saved sweep capture values without fitting or changing the experiment."""
import argparse
import csv
import json
from pathlib import Path

from build_evaluation_example import sha
from sweep_evidence import check_display,check_result_family,count_result_rows


def verify(capture):
    proof=json.loads((capture/'scientific_acceptance.json').read_text())
    provenance=json.loads((capture/'provenance.json').read_text())
    work=Path(proof['private_folder'])
    if not provenance['completed_capture'] or proof['new_sweep_requested']:
        raise ValueError('Expected a completed read-only saved-sweep replay')
    for name,digest in proof['original_inputs'].items():
        if sha(name)!=digest or sha(work/'inputs'/Path(name).name)!=digest:
            raise ValueError('Original or private example input changed')
    rows=list(csv.DictReader((work/'trials/sweep_results.csv').open()))
    report=dict(lesson='73_parameter_sweep',source_commit=provenance['commit'],
        capture=str(capture),accepted=False,scope='saved execution and display consistency only',
        inference_validated=False,published=False,new_fits_requested=False,
        display=check_display(proof['displayed_headers'],proof['displayed_rows'],rows),trials=[])
    for row in rows:
        folder=Path(row['folder']);path=folder/'results/ridge/results.csv'
        result=list(csv.DictReader(path.open()))
        actual=json.loads((folder/'settings/regression.json').read_text())
        requested=json.loads((folder/'_trial_settings.json').read_text())['settings']
        counts=count_result_rows(result)
        if any(counts[k]!=int(row[k]) for k in counts):raise ValueError('Sweep summary count differs')
        if float(actual['alpha'])!=float(row['alpha']) or float(requested['alpha'])!=float(row['alpha']):
            raise ValueError('Requested and actual ridge penalties differ')
        report['trials'].append(dict(trial_id=row['trial_id'],alpha=float(row['alpha']),
            results_sha256=sha(path),counts=counts,qc_inference=row['qc_inference'],
            qc_verdict=row['qc_verdict']))
    first=list(csv.DictReader((Path(rows[0]['folder'])/'results/ridge/results.csv').open()))
    report['family_checks']={name:check_result_family(snapshot,first,snapshot['level'])
        for name,snapshot in proof['result_snapshots'].items()}
    if list(report['family_checks'])!=['10_saved_guide_family','11_saved_gene_family','12_saved_guide_family_restored']:
        raise ValueError('Missing guide/gene/restoration evidence')
    geometry=proof['trial_table_geometry']
    if not geometry['both_rows_visible'] or geometry['viewport_height']<2*geometry['row_height']:
        raise ValueError('The recorded trial rows do not fit')
    report.update(accepted=True,original_inputs_preserved=True,trial_table_geometry=geometry,
        checked_result_cells=sum(c['table']['cells'] for c in report['family_checks'].values()),
        checked_plot_points=sum(c['plotted_points'] for c in report['family_checks'].values()))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture',type=Path)
    args=parser.parse_args()
    print(json.dumps(verify(args.capture),indent=2))
