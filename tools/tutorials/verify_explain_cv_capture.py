#!/usr/bin/env python3
"""Verify the recorded surrogate's saved artifacts without repeating its fit."""
import argparse
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from build_evaluation_example import sha
from surrogate_evidence import binary_counts, check_metric_values, check_formatted_table, check_well_partition


def read_saved_table(path):
    # Five-digit GUI rounding can straddle a boundary after the default CSV
    # parser loses a final bit. Preserve the exact saved floating-point value.
    return pd.read_csv(path,index_col=0,float_precision='round_trip')


def verify(capture):
    capture = Path(capture)
    proof = json.loads((capture/'scientific_acceptance.json').read_text())
    provenance = json.loads((capture/'provenance.json').read_text())
    work = Path(proof['private_folder']); output = work/'surrogate'
    if not provenance['completed_capture'] or not proof.get('computational_run_completed'):
        raise ValueError('The actual surrogate recording did not finish')
    if not all(sha(path)==digest for path,digest in proof['original_inputs'].items()):
        raise ValueError('An original surrogate teaching input changed')
    predictions = pd.read_csv(work/'recorded_predictions.csv')
    if len(predictions)!=234:
        raise ValueError('Expected the original 234 recorded CV predictions')
    source = {Path(row.filename).name:int(row.predicted_label) for row in predictions.itertuples()}
    if len(source)!=len(predictions): raise ValueError('Duplicate recorded prediction basename')
    with sqlite3.connect((work/'measurements.db').as_uri()+'?mode=ro',uri=True) as connection:
        connection.row_factory=sqlite3.Row
        metadata=[dict(row) for row in connection.execute(
            'SELECT png_path,prcfo,plateID,rowID,columnID FROM png_list')]
    matched=[row for row in metadata if Path(row['png_path']).name in source]
    if len(matched)!=len(source): raise ValueError('Prediction-to-database population differs')
    manifest=json.loads((output/'manifest.json').read_text())
    # Reuse the unfitted application's database join, not its split, scorer or
    # model. State this boundary: this is not an independent feature-join audit.
    from spacr.surrogate import build_surrogate_frame
    joined=build_surrogate_frame(str(work/'measurements.db'),predictions,
        path_column='filename',prediction_column='predicted_label')
    complete=np.isfinite(joined[manifest['feature_columns']].to_numpy(float)).all(axis=1)
    by_name={Path(row['png_path']).name:row for row in matched}
    complete_names=joined.loc[complete,'png_path'].map(lambda p:Path(p).name).tolist()
    if len(set(complete_names))!=len(complete_names):raise ValueError('Duplicate complete feature object')
    population=[by_name[name] for name in complete_names]
    if manifest['n_objects']!=len(population):raise ValueError('Complete-feature accounting differs')
    frames={role:read_saved_table(output/filename) for role,filename in {
        'importance':'feature_importance.csv','metrics':'class_metrics.csv',
        'confusion':'confusion_matrix.csv','held_out':'held_out_predictions.csv',
        'correlations':'correlated_feature_pairs.csv','distributions':'held_out_feature_distributions.csv',
    }.items() if (output/filename).exists()}
    shap_path=output/'held_out_signed_shap.csv'
    if shap_path.exists():frames['shap']=read_saved_table(shap_path)
    held=frames['held_out'].to_dict('records')
    for row in held:
        original=by_name[Path(row['png_path']).name]
        if any(str(row[key])!=str(original[key]) for key in ('plateID','rowID','columnID','png_path')):
            raise ValueError('A held-out cell identity differs from the source database')
        if row['cv_prediction']!=source[Path(original['png_path']).name]:
            raise ValueError('A CV class call changed between source and surrogate')
        row['prcfo']=original['prcfo']
    partition=check_well_partition(population,held,manifest['split_report'])
    counted=binary_counts(held)
    metrics=check_metric_values(manifest,{k:counted[k] for k in ('fidelity','balanced_accuracy','f1_macro')})
    all_counts=pd.Series([source[name] for name in complete_names]).value_counts()
    baseline=float(all_counts.max()/len(population))
    check_metric_values(manifest,{'majority_baseline':baseline,
        'fidelity_improvement':counted['fidelity']-baseline})
    if {int(k):v for k,v in manifest['class_counts'].items()}!=all_counts.to_dict():
        raise ValueError('Surrogate class counts differ from the saved CV calls')
    expected_faithful=counted['fidelity']>=baseline+manifest['minimum_fidelity_improvement']
    if manifest['importance_presented']!=expected_faithful or proof['faithful']!=expected_faithful:
        raise ValueError('Fidelity gate differs from the displayed/saved result')
    for entry in counted['class_metrics']:
        row=frames['metrics'].loc[entry['class_id']].to_dict()
        check_metric_values(row,{k:v for k,v in entry.items() if k!='class_id'})
    if frames['confusion'].loc[[0,1],['0','1']].to_numpy().tolist()!=counted['confusion']:
        raise ValueError('Saved confusion counts differ from held-out predictions')
    checked={}
    for name,table in proof['tables'].items():
        if name not in frames:
            if table['rows']:raise ValueError('Displayed data has no saved counterpart: '+name)
            checked[name]=dict(passed=True,displayed_rows=0,cells=0,artifact_absent=True)
            continue
        frame=frames[name]
        if name in ('metrics','confusion','held_out','shap'):frame=frame.reset_index()
        if name=='importance' and not expected_faithful:frame=frame.iloc[0:0]
        try:
            checked[name]=check_formatted_table(table,list(map(str,frame.columns)),
                [list(row) for row in frame.itertuples(index=False,name=None)])
        except ValueError as exc:
            raise ValueError(f'{name}: {exc}') from exc
    return dict(lesson='70_explain_cv',accepted=True,source_commit=provenance['commit'],
        capture=str(capture),private_folder=str(work),original_inputs_preserved=True,
        scope='real saved prediction identities, surrogate arithmetic and display consistency only',
        independent_classifier_validation=False,biological_mechanism_inferred=False,published=False,
        prediction_rows=len(predictions),joined_rows=len(joined),complete_feature_rows=len(population),
        excluded_feature_rows=int((~complete).sum()),
        feature_join_independently_reimplemented=False,
        population_basis='Unfitted application join; finite values checked separately across every selected feature',
        partition=partition,metrics=metrics,independent_counts=counted,
        displayed_baseline_uses='all complete objects, not the held-out subset',
        source_majority_baseline=baseline,importance_presented=expected_faithful,
        tables=checked,checked_cells=sum(item['cells'] for item in checked.values()),
        saved_artifacts={f.name:sha(f) for f in output.iterdir() if f.is_file()})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('capture',type=Path)
    args=parser.parse_args();print(json.dumps(verify(args.capture),indent=2))
