"""Read-only, independent source reconstruction for the fixed real tutorial schema.

No application join, scorer, estimator, schema migration or imputation is used.
Unsupported identity/schema variants are errors rather than guessed conversions.
"""
from collections import Counter, defaultdict
import csv
import math
from pathlib import Path
import re
import sqlite3

KEY=('plateID','rowID','columnID','fieldID')
TABLES=('cell','cytoplasm','nucleus','pathogen','png_list')
MAX_ROWS=200000

def number(value):
    if isinstance(value,bool):raise ValueError('Boolean object identity')
    if isinstance(value,(int,float)):
        if not math.isfinite(value) or value<=0 or int(value)!=value:raise ValueError('Invalid numeric object identity')
        if int(value)>2**53-1:raise ValueError('Object identity exceeds exact tutorial numeric range')
        return int(value)
    text=str(value)
    if not re.fullmatch(r'o?[1-9][0-9]*(?:\.0+)?',text):raise ValueError('Unsupported object identity spelling')
    result=int(text.removeprefix('o').split('.')[0])
    if result>2**53-1:raise ValueError('Object identity exceeds exact tutorial numeric range')
    return result

def coordinates(row):
    values=tuple(row[k] for k in KEY)
    patterns=(r'plate[0-9]+',r'r[0-9]+',r'c[0-9]+',r'f[0-9]+')
    if not all(isinstance(v,str) and re.fullmatch(p,v) for p,v in zip(patterns,values)):
        raise ValueError('Unsupported or missing full acquisition identity')
    return values

def identity(row,object_column='object_label'):
    return (*coordinates(row),number(row[object_column]))

def composed(key):return '_'.join(key[:4])+'_o'+str(key[4])

def numeric(value):
    if value is None:return None
    if isinstance(value,bool) or not isinstance(value,(int,float)):
        raise ValueError('Selected measurement is not a native numeric SQLite value')
    return float(value)

def predictions(path):
    result={}
    with Path(path).open(encoding='utf-8',newline='') as source:
        reader=csv.DictReader(source)
        if not {'filename','predicted_label'}<=set(reader.fieldnames or []):raise ValueError('Missing actual prediction columns')
        for row in reader:
            name=Path(row['filename']).name
            if not name or name in result:raise ValueError('Ambiguous or duplicate prediction basename')
            value=float(row['predicted_label'])
            if value not in (0,1):raise ValueError('Prediction is not a finite binary call')
            result[name]=int(value)
    if not result:raise ValueError('No genuine predictions')
    return result

def reconstruct(database,prediction_file,features):
    """Join source identities and compute declared features using raw tables.

    Parent measurements are direct source cells/cytoplasm. Child numeric
    measurements use the documented arithmetic mean of all linked objects,
    ignoring SQL NULL, with nucleus required and pathogen/cytoplasm optional.
    Missing/nonfinite values exclude eligibility; no replacement value is made.
    """
    database=Path(database).resolve(strict=True)
    if database.stat().st_size>64*1024*1024:raise ValueError('Database exceeds fixed tutorial read budget')
    for suffix in ('-wal','-journal'):
        side=Path(str(database)+suffix)
        if side.exists() and side.stat().st_size:raise ValueError('Source has pending writes')
    if not features or len(features)>1500 or len(set(features))!=len(features):raise ValueError('Invalid declared feature set')
    wanted=predictions(prediction_file)
    connection=sqlite3.connect(database.as_uri()+'?mode=ro&immutable=1',uri=True)
    connection.row_factory=sqlite3.Row
    try:
        connection.execute('PRAGMA query_only=ON');connection.execute('PRAGMA cache_size=-8192')
        schemas={}
        for table in TABLES:
            entry=connection.execute('SELECT type FROM sqlite_master WHERE name=?',(table,)).fetchone()
            if entry is None or entry[0]!='table':raise ValueError('Missing genuine object table: '+table)
            columns=[r[1] for r in connection.execute('PRAGMA table_info("'+table+'")')]
            if any(c in columns for c in ('timeID','time_id')):raise ValueError('Timepoint schema requires separately reviewed full time identity')
            if not set(KEY)<=set(columns):raise ValueError('Missing acquisition identity: '+table)
            if len(set(columns))!=len(columns):raise ValueError('Ambiguous source columns')
            schemas[table]=columns
        owners={}
        for feature in features:
            if feature in ('measurement_ndim','n_z'):owner='cell'
            elif feature in ('count_nucleus','count_pathogen'):owner=feature[6:]
            else:
                owner=next((t for t in TABLES[:4] if feature.startswith(t+'_')),None)
            if owner is None or (not feature.startswith('count_') and feature not in schemas[owner]):
                raise ValueError('Unsupported or missing selected source feature: '+feature)
            owners[feature]=owner
        def rows(table):
            cursor=connection.execute('SELECT * FROM "'+table+'" ORDER BY rowid')
            for i,row in enumerate(cursor):
                if i>=MAX_ROWS:raise ValueError('Source table exceeds bounded row budget')
                yield dict(row)
        png={};png_keys=set()
        if 'cell_id' not in schemas['png_list']:raise ValueError('Crop list is not keyed to real cell objects')
        for row in rows('png_list'):
            name=Path(str(row.get('png_path',''))).name
            if name not in wanted:continue
            key=identity(row,'cell_id')
            if name in png or key in png_keys:raise ValueError('Duplicate/ambiguous crop-to-object identity')
            if row.get('prcfo') not in (None,composed(key)):raise ValueError('Stored crop key disagrees with full object identity')
            png[name]={'key':key,'png_path':row['png_path'],'cv_prediction':wanted[name]};png_keys.add(key)
        if set(png)!=set(wanted):raise ValueError('A genuine prediction has no unique source crop')
        cell={};seen=set()
        for row in rows('cell'):
            key=identity(row)
            if key in seen:raise ValueError('Duplicate full cell identity')
            seen.add(key)
            if key not in png_keys:continue
            if row.get('prcf')!='_'.join(key[:4]):raise ValueError('Cell field key disagrees with full identity')
            cell[key]=row
        if set(cell)!=png_keys:raise ValueError('Prediction crop has no source cell measurement')
        cyto={};seen=set()
        for row in rows('cytoplasm'):
            key=identity(row)
            if key in seen:raise ValueError('Duplicate full cytoplasm identity')
            seen.add(key)
            if key in png_keys:
                if row.get('prcf')!='_'.join(key[:4]):raise ValueError('Cytoplasm field key disagrees')
                cyto[key]=row
        children={}
        for table in ('nucleus','pathogen'):
            if 'cell_id' not in schemas[table]:raise ValueError('Child table has no explicit parent cell identity')
            groups=defaultdict(list);seen=set()
            for row in rows(table):
                own=identity(row)
                if own in seen:raise ValueError('Duplicate child object identity')
                seen.add(own)
                if row['cell_id'] is None or row['cell_id']==0:continue  # explicitly unlinked child, no parent value invented
                key=identity(row,'cell_id')
                if key not in png_keys:continue
                if row.get('prcf')!='_'.join(key[:4]):raise ValueError('Child field key disagrees with parent acquisition')
                groups[key].append(row)
            children[table]=groups
        population=[];joined=0;excluded={}
        for name,bridge in png.items():
            key=bridge['key']
            if not children['nucleus'].get(key):
                excluded[name]='no linked nucleus';continue
            joined+=1;values={}
            for feature,owner in owners.items():
                if feature.startswith('count_'):
                    value=len(children[owner].get(key,[])) if children[owner].get(key) else None
                elif owner in ('cell','cytoplasm'):
                    row=cell[key] if owner=='cell' else cyto.get(key)
                    value=None if row is None else numeric(row[feature])
                else:
                    observed=[numeric(row[feature]) for row in children[owner].get(key,[]) if row[feature] is not None]
                    if not observed:value=None
                    elif not all(math.isfinite(v) for v in observed):value=None
                    else:value=math.fsum(observed)/len(observed)
                values[feature]=value
            bad=[feature for feature,value in values.items() if value is None or not math.isfinite(value)]
            if bad:excluded[name]={'nonfinite_declared_features':len(bad)};continue
            population.append({'prcfo':composed(key),'key':key,'basename':name,'png_path':bridge['png_path'],
                **dict(zip(KEY,key[:4])),'object_label':key[4],'cv_prediction':bridge['cv_prediction'],'values':values})
        return {'population':population,'prediction_rows':len(wanted),'matched_crops':len(png),
                'joined_rows':joined,'excluded':excluded,'feature_count':len(features)}
    finally:connection.close()

def compare(value,wanted,description):
    value=float(value);wanted=float(wanted)
    if math.isnan(wanted):
        if not math.isnan(value):raise ValueError('Expected undefined statistic: '+description)
    elif not math.isfinite(value) or not math.isclose(value,wanted,rel_tol=1e-12,abs_tol=1e-12):
        raise ValueError('Independent source numeric mismatch: '+description)

def ranks(values):
    ordered=sorted(range(len(values)),key=values.__getitem__);result=[0.]*len(values);i=0
    while i<len(ordered):
        j=i+1
        while j<len(ordered) and values[ordered[j]]==values[ordered[i]]:j+=1
        for k in ordered[i:j]:result[k]=(i+j-1)/2+1
        i=j
    return result

def spearman(left,right):
    a=ranks(left);b=ranks(right);am=math.fsum(a)/len(a);bm=math.fsum(b)/len(b)
    aa=[v-am for v in a];bb=[v-bm for v in b]
    denominator=math.sqrt(math.fsum(v*v for v in aa)*math.fsum(v*v for v in bb))
    return math.fsum(x*y for x,y in zip(aa,bb))/denominator if denominator else float('nan')

def partial_scope(provenance,proof,job_rc,allow_partial):
    """Eligibility for scoped data checks only; never changes lesson acceptance."""
    if proof.get('computational_run_completed') is not True or proof.get('original_inputs_preserved') is not True or proof.get('private_inputs_preserved') is not True:
        raise ValueError('Actual original scientific callbacks did not complete')
    if len(proof.get('snapshots',{}))!=8 or set(proof.get('tables',{}))!={'importance','metrics','confusion','shap','correlations','held_out','distributions'}:
        raise ValueError('Actual full result tour/table scope incomplete')
    if provenance.get('completed_capture') is True:
        if job_rc not in ('0','2'):raise ValueError('Actual capture process did not complete')
        return 'completed_capture'
    if not allow_partial or job_rc!='2' or provenance.get('status')!='gate_failed':
        raise ValueError('No explicit actual terminal gate-only partial scope')
    return 'terminal_gate_failed_media_only; scientific source/arithmetic scope only'
