"""Scoped source/split/arithmetic/display checks, never lesson/publication acceptance."""
from pathlib import Path
import argparse
from collections import Counter
import hashlib
import json
import math
import statistics
import shlex
import re
import pandas as pd
from surrogate_source_evidence import reconstruct,identity,number,compare,spearman,partial_scope
from surrogate_evidence import binary_counts,check_metric_values,check_formatted_table,check_well_partition


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(1024*1024),b''):result.update(block)
    return result.hexdigest()

def read(path):return json.loads(Path(path).read_text())

def real(stage,value):
    value=Path(value)
    if value.is_relative_to('/tmp/spacr-tutorial-current'):value=stage/value.relative_to('/tmp/spacr-tutorial-current')
    if not value.is_absolute() or '..' in value.parts or not value.resolve().is_relative_to(stage.resolve()):
        raise ValueError('Artifact escapes actual capture stage')
    return value

def table(path):return pd.read_csv(path,index_col=0,float_precision='round_trip')

def terminal_job(stage,queue,job_stem):
    """Bind the current first/retry job, log and declared media scope without running it."""
    if job_stem not in ('c09','d09') or queue.name!='queue' or queue.parent.name!='t70':
        raise ValueError('Expected the actual lesson70 c09/d09 queue job')
    done=queue/(job_stem+'.job.done')
    if not done.is_file() or any((queue/(job_stem+suffix)).exists() for suffix in ('.job','.job.running')):
        raise ValueError('Actual requested original/retry job is not terminal')
    terminal=shlex.split(done.read_text());rc=(queue/(job_stem+'.rc')).read_text().strip()
    actual_output=(queue/(job_stem+'.out')).read_text(errors='replace')
    if re.search(r'Fatal Python error|(?:received signal|terminated with signal)\s+SIG(?:SEGV|ABRT|BUS|ILL|FPE)\b',actual_output) or 'sending signal INT to command' in actual_output:
        raise ValueError('Fatal/deadline actual job is not a gate-only partial')
    ends=re.findall(r'^end \S+ rc=(\d+)$',actual_output,re.M)
    if ends!=[rc] or rc not in ('0','2'):
        raise ValueError('Actual output has no matching successful/gate-only terminal result')
    if job_stem=='d09':
        parent=queue/'c09.job.done'
        if len(terminal)!=2 or Path(terminal[0]).name!='retry_failed.sh' or Path(terminal[1]).resolve()!=parent.resolve():
            raise ValueError('Retry job does not own the actual original job')
        command=shlex.split(parent.read_text())
    else:command=terminal
    try:
        module=command[command.index('--module')+1]
        requested=command[command.index('--record-scenes')+1].split(',')
        launcher=next(i for i,arg in enumerate(command) if Path(arg).name=='live18.sh')
        original_stage=Path(command[launcher+1])
    except (ValueError,IndexError,StopIteration) as exc:
        raise ValueError('Actual terminal job lacks the normal capture arguments') from exc
    expected=Path(str(original_stage)+('-b' if job_stem=='d09' else ''))
    if module!='explain_cv' or not expected.is_absolute() or expected.resolve()!=stage:
        raise ValueError('Actual terminal job does not own the requested stage/module')
    routes=re.findall(r'^capture stage (.+) -> (.+)$',actual_output,re.M)
    if len(routes)!=1 or Path(routes[0][0]).resolve()!=stage or Path(routes[0][1]).resolve()!=stage:
        raise ValueError('Actual output belongs to another capture stage')
    if job_stem=='d09':
        terminal_job(original_stage.resolve(strict=True),queue,'c09')
        requested=read(original_stage/'captures/explain_cv/host_gate_summary.json')['failed']
        if f"re-recording {','.join(requested)} from {original_stage}" not in actual_output.splitlines():
            raise ValueError('Actual retry output does not own the original failed scene scope')
    if not requested or len(set(requested))!=len(requested):
        raise ValueError('Actual declared media scope is empty/duplicated')
    return rc,requested


def verify(stage,queue,allow_partial,job_stem):
    stage=Path(stage).resolve(strict=True);queue=Path(queue).resolve(strict=True);capture=stage/'captures/explain_cv'
    proof_path=capture/'scientific_acceptance.json';provenance_path=capture/'provenance.json'
    originals={p:sha(p) for p in (proof_path,provenance_path)}
    proof=read(proof_path);provenance=read(provenance_path)
    rc,requested=terminal_job(stage,queue,job_stem)
    frames=read(capture/'frames.json')
    if not set(requested)<=set(frames):raise ValueError('Actual declared media/tour scope is incomplete')
    for name in requested:
        row=frames[name];image=real(stage,str(capture/row['image']))
        if sha(image)!=row['sha256']:raise ValueError('Actual source-frame identity changed')
    scope=partial_scope(provenance,proof,rc,allow_partial)
    if proof['source_commit']!=provenance['commit'] or provenance['app_source_modified'] is not False:raise ValueError('Actual scientific/source provenance differs')
    for key in ('cv_model_retrained','independent_classifier_validation','synthetic_predictions','application_modified','published'):
        if proof[key] is not False:raise ValueError('Original non-biological/non-retraining scope changed: '+key)
    work=real(stage,proof['private_folder']);output=work/'surrogate'
    input_hashes={real(stage,name):digest for name,digest in proof['original_inputs'].items()}
    if len(input_hashes)!=2 or not all(sha(path)==h for path,h in input_hashes.items()):raise ValueError('Original source input differs')
    prepared=read(stage/'exact-source.json')['prepared_inputs']
    if prepared['source']!='/media/carruthers/mnt3/claude/n676_stages/host/inputs/explain_cv_in2':raise ValueError('Actual genuine source dataset differs')
    for path,h in input_hashes.items():
        if prepared['sha256'].get(str(path.relative_to(stage)))!=h:raise ValueError('Scientific input is not the actual prepared-source identity')
    private_hashes={}
    for name in ('measurements.db','recorded_predictions.csv'):
        source=work/name
        private_hashes[source]=sha(source)
        if private_hashes[source] not in input_hashes.values():raise ValueError('Private input differs from genuine original source')
    artifact_hashes={p:sha(p) for p in output.iterdir() if p.is_file()}
    manifest=read(output/'manifest.json');features=manifest['feature_columns']
    if len(features)!=1126:raise ValueError('Actual selected feature scope differs from full genuine tutorial run')
    source=reconstruct(work/'measurements.db',work/'recorded_predictions.csv',features)
    population=source['population'];by_name={r['basename']:r for r in population}
    if source['prediction_rows']!=234 or len(population)!=89 or manifest['n_objects']!=89 or proof['n_objects']!=89:
        raise ValueError('Independent full genuine finite-feature population differs; no imputation or forced accounting')
    frames={name:table(output/file) for name,file in {'importance':'feature_importance.csv','metrics':'class_metrics.csv',
        'confusion':'confusion_matrix.csv','held_out':'held_out_predictions.csv','correlations':'correlated_feature_pairs.csv',
        'distributions':'held_out_feature_distributions.csv','shap':'held_out_signed_shap.csv'}.items()}
    held_frame=frames['held_out'];held=[];index_to_row={}
    if not held_frame.index.is_unique:raise ValueError('Duplicate saved held-out object index')
    for index,row in held_frame.iterrows():
        record=row.to_dict();name=Path(str(record['png_path'])).name
        if name not in by_name:raise ValueError('Held-out identity absent from independent eligible source population')
        expected=by_name[name]
        if identity(record)!=expected['key'] or str(record['png_path'])!=str(expected['png_path']):raise ValueError('Held-out full identity or crop path differs from source')
        if number(record['cell_id'])!=expected['object_label'] or record['cv_prediction']!=expected['cv_prediction']:raise ValueError('Held-out cell identity/CV call changed')
        record['prcfo']=expected['prcfo'];held.append(record);index_to_row[index]=expected
        probabilities=[float(record['probability_0']),float(record['probability_1'])]
        if any(not math.isfinite(x) or not 0<=x<=1 for x in probabilities):raise ValueError('Invalid saved class probability')
        compare(sum(probabilities),1.,'probability sum')
        if int(record['surrogate_prediction'])!=(0 if probabilities[0]>=probabilities[1] else 1):raise ValueError('Saved predicted class disagrees with saved probabilities')
    partition=check_well_partition(population,held,manifest['split_report'])
    if manifest['split_report']['requested_fraction']!=.3 or manifest['random_seed']!=0:raise ValueError('Original held-out split policy differs')
    if len(held)!=21 or partition['training_objects']!=68 or partition['shared_wells']!=0:raise ValueError('Actual whole-well split differs')
    counted=binary_counts(held)
    metrics=check_metric_values(manifest,{k:counted[k] for k in ('fidelity','balanced_accuracy','f1_macro')})
    classes=Counter(r['cv_prediction'] for r in population);baseline=max(classes.values())/len(population)
    check_metric_values(manifest,{'majority_baseline':baseline,'fidelity_improvement':counted['fidelity']-baseline})
    if {int(k):v for k,v in manifest['class_counts'].items()}!=dict(classes):raise ValueError('Class population counts differ')
    faithful=counted['fidelity']>=baseline+manifest['minimum_fidelity_improvement']
    check_metric_values(proof,{'fidelity':counted['fidelity'],'baseline':baseline,'n_objects':89,'features':1126})
    if manifest['model_family']!='random_forest' or manifest['model_parameters']!={'n_estimators':300,'n_jobs':-1,'random_state':0}:raise ValueError('Actual unchanged surrogate model policy differs')
    if manifest['minimum_fidelity_improvement']!=.05 or manifest['importance_presented']!=faithful or proof['faithful']!=faithful:
        raise ValueError('Original fidelity gate/withheld importance differs')
    for wanted in counted['class_metrics']:
        row=frames['metrics'].loc[wanted['class_id']].to_dict()
        check_metric_values(row,{k:v for k,v in wanted.items() if k!='class_id'})
    if frames['confusion'].loc[[0,1],['0','1']].to_numpy().tolist()!=counted['confusion']:raise ValueError('Saved confusion differs from independent counts')
    # The saved source values are checked independently by full held-out identity,
    # not by reproducing the application's intermediate DataFrame row order.
    values=table(output/'held_out_shap_feature_values.csv')
    if list(values.columns)!=features or set(values.index)!=set(index_to_row) or not values.index.is_unique:
        raise ValueError('Saved SHAP input values have unknown/duplicate/incomplete held-out identities/features')
    for index,row in values.iterrows():
        expected=index_to_row[index]['values']
        for feature in features:compare(row[feature],expected[feature],'held-out source '+feature)
    if list(frames['shap'].columns)!=features or set(frames['shap'].index)!=set(index_to_row) or not frames['shap'].index.is_unique or set(manifest['shap_object_indices'])!={str(index) for index in index_to_row}:raise ValueError('Saved SHAP values do not identify the exact held-out cells/features')
    if not all(math.isfinite(float(x)) for row in frames['shap'].itertuples(index=False,name=None) for x in row):raise ValueError('Saved SHAP matrix has nonfinite values')
    # Independently recompute the held-out per-class feature summary arithmetic.
    distributions=frames['distributions'];expected_rows=len(features)*len({r['cv_prediction'] for r in held})
    if len(distributions)!=expected_rows:raise ValueError('Saved feature distributions omit or duplicate source groups')
    seen=set()
    for _,row in distributions.iterrows():
        if row['cv_class'] not in (0,1):raise ValueError('Distribution has nonbinary source class')
        key=(row['feature'],int(row['cv_class']))
        if key in seen or key[0] not in features:raise ValueError('Duplicate/unknown distribution identity')
        seen.add(key);numbers=[r['values'][key[0]] for r in index_to_row.values() if r['cv_prediction']==key[1]]
        if not numbers:raise ValueError('Distribution class has no source cells')
        wanted={'n':len(numbers),'mean':math.fsum(numbers)/len(numbers),'median':statistics.median(numbers),
                'std':statistics.stdev(numbers) if len(numbers)>1 else float('nan')}
        for field,value in wanted.items():compare(row[field],value,'distribution '+key[0]+' '+field)
    # Every reported training-split correlation is numerically checked. Top500
    # selection/completeness is not claimed independently by this narrow scope.
    held_names={r['prcfo'] for r in held};training=[r for r in population if r['prcfo'] not in held_names];seen=set()
    for _,row in frames['correlations'].iterrows():
        pair=(row['feature_a'],row['feature_b'])
        pair_identity=tuple(sorted(pair))
        if pair_identity in seen or pair[0]==pair[1] or not set(pair)<=set(features):raise ValueError('Duplicate/unknown correlation pair')
        seen.add(pair_identity);expected=spearman([r['values'][pair[0]] for r in training],[r['values'][pair[1]] for r in training])
        compare(row['spearman'],expected,'training Spearman')
        if abs(float(row['spearman']))<.9:raise ValueError('Saved strong correlation is below documented threshold')
    displayed={}
    for name,saved in frames.items():
        expected=saved.reset_index() if name in ('metrics','confusion','held_out','shap') else saved
        if name=='importance' and not faithful:expected=expected.iloc[0:0]
        displayed[name]=check_formatted_table(proof['tables'][name],list(map(str,expected.columns)),
            [list(row) for row in expected.itertuples(index=False,name=None)])
    if not all(sha(path)==h for path,h in {**originals,**input_hashes,**private_hashes,**artifact_hashes}.items()):raise ValueError('An original proof/input/result artifact changed during verification')
    return {'scope_passed':True,'scope':'independent source identity/declared finite eligibility/whole-well split/arithmetic/saved-table display consistency only',
        'lesson_accepted':False,'publication_eligible':False,'raw_scientific_accepted':proof['accepted'],'raw_artifacts_unchanged':True,
        'raw_provenance_completed_unchanged':provenance['completed_capture'],'partial_scope':scope,
        'source_population':len(population),'declared_source_features':len(features),'source_prediction_rows':source['prediction_rows'],
        'joined_rows':source['joined_rows'],'excluded_rows':len(source['excluded']),'partition':partition,'metrics':metrics,
        'independent_counts':counted,'majority_baseline':baseline,'importance_presented':faithful,'tables':displayed,
        'independent_reference_training':False,'independent_model_probabilities':False,'independent_shap_computation':False,
        'independent_top500_correlation_selection':False,'feature_selection_policy_independently_reimplemented':False,
        'biological_mechanism_inferred':False,'fit_repeated':False}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',type=Path,required=True);parser.add_argument('--queue',type=Path,required=True)
    parser.add_argument('--job-stem',choices=['c09','d09'],required=True)
    parser.add_argument('--allow-terminal-gate-partial',action='store_true');args=parser.parse_args()
    print(json.dumps(verify(args.stage,args.queue,args.allow_terminal_gate_partial,args.job_stem),indent=2,sort_keys=True))
