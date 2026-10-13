"""Independent two-class ranking arithmetic over immutable measurement rows."""
from bisect import bisect_left, bisect_right
import math
from pathlib import Path
import sqlite3


def read_measurements(database):
    path = Path(database).resolve()
    with sqlite3.connect(path.as_uri()+'?mode=ro&immutable=1', uri=True) as con:
        con.row_factory = sqlite3.Row
        schema = list(con.execute('PRAGMA table_info(cell)'))
        numeric = [r['name'] for r in schema if r['type'].upper() in ('REAL','INTEGER','FLOAT','DOUBLE')
                   and r['name'] != 'object_label']
        rows = [dict(r) for r in con.execute('SELECT * FROM cell')]
    return rows, numeric


def independent_scores(rows, features, *, label='rowID', statistic='auc'):
    """Pair counting via bisect and empirical CDFs, not spaCR's rank implementation."""
    if statistic not in ('auc','ks'):
        raise ValueError('Only AUC and KS are independently checked here')
    levels = sorted({str(r[label]) for r in rows if r[label] is not None})
    if len(levels) != 2:
        raise ValueError('The tutorial arithmetic requires exactly two real classes')
    result = []
    for name in features:
        groups = {level: [] for level in levels}
        for row in rows:
            value = row.get(name)
            if row[label] is not None and value is not None and math.isfinite(float(value)):
                groups[str(row[label])].append(float(value))
        left, right = (sorted(groups[level]) for level in levels)
        if not left or not right or min(left[0],right[0]) == max(left[-1],right[-1]):
            continue
        auc = sum(bisect_left(left,v)+.5*(bisect_right(left,v)-bisect_left(left,v))
                  for v in right)/(len(left)*len(right))
        ks = max(abs(bisect_right(left,v)/len(left)-bisect_right(right,v)/len(right))
                 for v in set(left+right))
        result.append(dict(feature=name, auc=auc, ks=ks,
                           score=abs(2*auc-1) if statistic=='auc' else ks,
                           n_by_class={levels[0]:len(left),levels[1]:len(right)},
                           higher_in=levels[1] if auc>=.5 else levels[0]))
    return sorted(result,key=lambda r:(-r['score'],r['feature']))


def verify_ranking(result, rows, features, *, statistic='auc', top=20, label='rowID'):
    expected = independent_scores(rows,features,label=label,statistic=statistic)
    if (result.n_rows != len(rows) or result.label != label or
            result.spec.statistic != statistic or result.spec.top != top or
            result.n_considered != len(expected) or len(result.scores) != min(top,len(expected))):
        raise ValueError('Ranking population, statistic, Top or considered count differs')
    errors = []
    for score, wanted in zip(result.scores, expected[:top]):
        if (score.feature != wanted['feature'] or dict(score.n_by_class) != wanted['n_by_class'] or
                score.higher_in != wanted['higher_in']):
            raise ValueError('Ranking feature order, class identity or finite counts differ')
        for field in ('auc','ks','score'):
            error = abs(getattr(score,field)-wanted[field])
            if not math.isfinite(error) or error > 1e-12:
                raise ValueError('Ranking differs from independent pair-count/CDF arithmetic')
            errors.append(error)
    return {'rows':len(rows),'label':label,'classes':sorted({r[label] for r in rows}),
            'considered':len(expected),'returned':len(result.scores),'statistic':statistic,
            'max_numeric_error':max(errors,default=0), 'top_features':[r['feature'] for r in expected[:8]],
            'no_independent_well_level_or_causal_inference_claimed':True}
