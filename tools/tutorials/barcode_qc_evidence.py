"""Independent count-table checks for the real Barcode QC tutorial inputs.

These functions use the downloaded CSV, not the application's normalisation,
threshold or plotting code. A target is a demonstration choice, not proof of
the experiment's intended guide multiplicity or biological attribution.
"""
from collections import Counter, defaultdict
import csv
import hashlib
import math
from pathlib import Path
import statistics

def read_counts(path):
    """Read the one-plate published example without pooling other plates."""
    path=Path(path); grouped=Counter(); source_rows=0
    with path.open(newline='') as stream:
        reader=csv.DictReader(stream)
        required={'row_name','column_name','grna_name','count'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('Downloaded count table lacks its required columns')
        for row in reader:
            source_rows+=1
            key=tuple(row[name] for name in ('row_name','column_name','grna_name'))
            if any(not value or value.strip()!=value for value in key):
                raise ValueError('Downloaded count identity is missing or ambiguous')
            value=int(row['count'])
            if value<=0:raise ValueError('This tutorial requires positive counted reads')
            grouped[key]+=value
    if not grouped:raise ValueError('No counted reads exist in the tutorial input')
    wells=defaultdict(dict); guides=Counter()
    for (row,column,guide),count in grouped.items():
        wells[(row,column)][guide]=count;guides[guide]+=count
    totals={key:sum(values.values()) for key,values in wells.items()}
    return dict(source=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                source_rows=source_rows,grouped=grouped,wells=dict(wells),totals=totals,
                guides=guides,total_reads=sum(totals.values()))


def per_well(source):
    return [dict(prc=f'plate1_{row}_{col}',plateID='plate1',rowID=row,columnID=col,
                 reads=source['totals'][(row,col)],n_grnas=len(values))
            for (row,col),values in source['wells'].items()]


def positions(source,ratio=2.):
    middle=statistics.median(source['totals'].values());result=[]
    for axis,index in [('row',0),('column',1)]:
        groups=defaultdict(list)
        for key,count in source['totals'].items():groups[key[index]].append(count)
        for label,counts in groups.items():
            median=statistics.median(counts);fold=median/middle
            result.append(dict(plateID='plate1',axis=axis,label=label,n_wells=len(counts),
                median_reads=median,plate_median=middle,ratio_to_plate=fold,
                flagged=fold>=ratio or fold<=1/ratio))
    return result


def sweep_row(source,threshold,target=5,starved_fraction=.1):
    cutoff=statistics.median(source['totals'].values())*starved_fraction
    population={key:values for key,values in source['wells'].items() if source['totals'][key]>=cutoff}
    if not population:raise ValueError('No non-starved wells remain in the tutorial population')
    survivors=[];kept_reads=0;total_reads=0
    for key,values in population.items():
        total=source['totals'][key];total_reads+=total
        kept=[n for n in values.values() if n/total>=threshold]
        survivors.append(len(kept));kept_reads+=sum(kept)
    kept_wells=[n for n in survivors if n];retained=len(kept_wells)
    over=sum(n>target for n in survivors);number=len(survivors)
    return dict(threshold=threshold,grnas_per_well=statistics.median(survivors),
                grnas_per_well_retained=statistics.median(kept_wells) if kept_wells else float('nan'),
                wells_retained=retained,well_retention=retained/number,
                wells_over_budget=over,collision_rate=over/number,
                collision_rate_retained=over/retained if retained else float('nan'),
                n_calls=sum(survivors),reads_retained=kept_reads/total_reads)


def compare_rows(actual,expected,keys):
    """Check every expected field keyed by full identity; no row-order guess."""
    def identity(row):return tuple(str(row[k]) for k in keys)
    rows={identity(r):r for r in actual};want={identity(r):r for r in expected}
    if len(rows)!=len(actual) or set(rows)!=set(want):
        raise ValueError('Output rows duplicate, omit or substitute a source identity')
    checked=0;maximum=0.
    for key,expected_row in want.items():
        for field,value in expected_row.items():
            got=rows[key].get(field)
            if isinstance(value,bool):
                if str(got).lower()!=str(value).lower():raise ValueError('Output flag differs: '+field)
            elif isinstance(value,(int,float)):
                try:number=float(got)
                except (TypeError,ValueError):
                    if str(got)=='' and math.isnan(value):number=float('nan')
                    else:raise ValueError('Output numeric value is missing: '+field)
                if not (math.isnan(number) and math.isnan(value)):
                    if not math.isfinite(number) or not math.isclose(number,value,rel_tol=1e-10,abs_tol=1e-10):
                        raise ValueError('Output numeric value differs: '+field)
                    maximum=max(maximum,abs(number-value))
            elif got!=value:raise ValueError('Output text differs: '+field)
            checked+=1
    return dict(rows=len(rows),fields_checked=checked,maximum_numeric_error=maximum)


def read_csv(path):
    with Path(path).open(newline='') as stream:return list(csv.DictReader(stream))


def verify_outputs(folder,source,target=5):
    folder=Path(folder)
    checks=dict(per_well=compare_rows(read_csv(folder/'reads_per_well.csv'),per_well(source),['prc']),
                positions=compare_rows(read_csv(folder/'position_effects.csv'),positions(source),['plateID','axis','label']))
    cutoff=statistics.median(source['totals'].values())*.1
    expected=[r for r in per_well(source) if r['reads']<cutoff]
    checks['starved']=compare_rows(read_csv(folder/'starved_wells.csv'),expected,['prc'])
    number_starved=len(expected)
    rows=read_csv(folder/'threshold_sweep.csv')
    if not rows:raise ValueError('No actual threshold sweep was saved')
    expected=[sweep_row(source,float(r['threshold']),target) for r in rows]
    checks['sweep']=compare_rows(rows,expected,['threshold'])
    unchanged=hashlib.sha256(Path(source['source']).read_bytes()).hexdigest()==source['sha256']
    if not unchanged:raise ValueError('The recorded input changed during the QC run')
    return dict(checks=checks,total_reads=source['total_reads'],wells=len(source['wells']),
                observed_guides=len(source['guides']),starved_wells=number_starved,
                starvation_cutoff=cutoff,source_unchanged=unchanged)


def check_native_run(run):
    """Saved tables alone do not demonstrate a working GUI plot workflow."""
    outcome=run['outcome']
    if not outcome['finished'] or not outcome['ok'] or outcome['errors']:
        raise ValueError('The native Barcode QC worker did not finish cleanly')
    if run['gui_figure_count']<2 or not run['figures_card_visible']:
        raise ValueError('Both Barcode QC figures must be visible in the GUI')
    if run['settings_errors']:
        raise ValueError('The visible input must pass settings validation')
