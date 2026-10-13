"""Independent checks for one real, unannotated permutation-result family.

This oracle deliberately does not call spaCR's hit parsing, correction,
agreement, filtering, sorting, formatting or export implementations.
"""
import csv
import math
from pathlib import Path


def read_rows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def expected_hits(folder):
    genes = read_rows(Path(folder) / 'results_gene.csv')
    guides = read_rows(Path(folder) / 'results_grna.csv')
    if not genes or len({r['gene'] for r in genes}) != len(genes):
        raise ValueError('Expected a nonempty unique gene family')
    if len({r['grna'] for r in guides}) != len(guides):
        raise ValueError('Guide identities must be unique')
    if any(not math.isfinite(float(r['coefficient'])) or
           not math.isclose(float(r['coefficient']), float(r['standardized_marginal_effect']), abs_tol=1e-12)
           for r in guides):
        raise ValueError('Guide effect differs from source marginal effect')
    p = [float(r['permutation_p_value']) for r in genes]
    order = sorted(range(len(p)), key=lambda i: p[i])
    adjusted = [0.] * len(p)
    ceiling = 1.
    for rank in range(len(p), 0, -1):
        i = order[rank - 1]
        ceiling = min(ceiling, len(p) * p[i] / rank)
        adjusted[i] = ceiling
    expected = []
    for i, row in enumerate(genes):
        if (row['feature'] != 'gene_fraction:gene[' + row['gene'] + ']'
                or row['multiple_testing_method'] != 'fdr_bh'
                or row['level'] != 'gene'):
            raise ValueError('Wrong gene identity or correction family')
        effect = float(row['standardized_marginal_effect'])
        probability = (int(row['permutation_exceedances']) + 1) / (int(row['permutations']) + 1)
        if (not 0 < p[i] <= 1 or not math.isfinite(effect)
                or abs(p[i] - probability) > 1e-12
                or abs(p[i] - float(row['p_value'])) > 1e-12
                or abs(effect - float(row['coefficient'])) > 1e-12
                or abs(adjusted[i] - float(row['adjusted_p_value'])) > 1e-12):
            raise ValueError('Source permutation probability, effect or gene BH differs')
        members = [g for g in guides if g['grna'].rsplit('_', 1)[0] == row['gene']]
        agree = sorted(g['grna'] for g in members if float(g['coefficient']) * effect > 0)
        n = len(members)
        flags = []
        if row['condition'] in ('nc', 'pc', 'control'):
            flags.append('control')
        if n == 0:
            flags.append('no-guide-rows')
        elif n == 1:
            flags.append('single-guide')
        elif len(agree) / n < .5:
            flags.append('guides-disagree')
        expected.append(dict(gene=row['gene'], name=row['gene'], effect=effect,
            std_err=float('nan'), ci_low=float('nan'), ci_high=float('nan'),
            p_value=p[i], q_value=adjusted[i], selection_frequency=float('nan'),
            n_guides=n, n_agree=len(agree), agreement=len(agree)/n if n else float('nan'),
            agreeing_guides=';'.join(agree), n_obs=0, condition=row['condition'],
            direction='up' if effect > 0 else 'down' if effect < 0 else '',
            flags=';'.join(flags), feature=row['feature']))
    return ranked(expected)


def ranked(rows):
    ordered = sorted(rows, key=lambda r: (r['q_value'], -abs(r['effect']), r['gene']))
    return [dict(r, rank=i + 1) for i, r in enumerate(ordered)]


def filtered(rows, *, max_q=1., min_effect=0., min_agreement=0., min_guides=0,
             direction='any', exclude_controls=False, query=''):
    result = [r for r in rows if r['q_value'] <= max_q and abs(r['effect']) >= min_effect
              and (not min_agreement or r['agreement'] >= min_agreement)
              and r['n_guides'] >= min_guides and (direction == 'any' or r['direction'] == direction)
              and (not exclude_controls or r['condition'] not in ('nc', 'pc', 'control'))
              and query.casefold().strip() in r['gene'].casefold()]
    return ranked(result)


def check_rows(actual, expected):
    if len(actual) != len(expected):
        raise ValueError('Hit row count differs')
    compared = 0
    for got, wanted in zip(actual, expected):
        if set(got) != set(wanted):
            raise ValueError('Hit columns differ')
        for key, value in wanted.items():
            observed = got[key]
            if isinstance(value, (int, float)):
                observed = float(observed) if observed != '' else float('nan')
                same = (math.isnan(value) and math.isnan(observed)) or math.isclose(value, observed, rel_tol=1e-11, abs_tol=1e-12)
            else:
                same = str(observed) == value
            if not same:
                raise ValueError('Hit value or order differs: ' + key)
            compared += 1
    return compared


def check_screen(screen, expected):
    if screen.last_error or screen.is_busy():
        raise ValueError('Hit List worker has not completed successfully')
    shown = screen.filtered()
    if shown is None or shown.ranking != 'q-value':
        raise ValueError('No completed q-ranked hit list')
    compared = check_rows([h.to_dict() for h in shown], expected)
    if screen._table.topLevelItemCount() != len(expected):
        raise ValueError('Displayed hit population differs')
    displayed = [screen._table.topLevelItem(i).text(1) for i in range(len(expected))]
    if displayed != [r['gene'] for r in expected]:
        raise ValueError('Displayed gene order differs')
    for i, row in enumerate(expected):
        def number(value):
            return format(value, '.3g' if value and (abs(value) < .001 or abs(value) >= 1e5) else '.4g')
        wanted = [str(row['rank']), row['gene'], row['name'], number(row['effect']),
                  '—', number(row['p_value']), number(row['q_value']),
                  f"{row['n_agree']}/{row['n_guides']}",
                  '—' if math.isnan(row['agreement']) else format(row['agreement'], '.0%'),
                  row['condition'] or '—', row['flags'].replace(';', ', ')]
        if [screen._table.topLevelItem(i).text(j) for j in range(11)] != wanted:
            raise ValueError('Displayed hit text differs')
    return dict(rows=len(expected), values_checked=compared,
                summary=screen._summary.text(), filters=screen.current_filters())
