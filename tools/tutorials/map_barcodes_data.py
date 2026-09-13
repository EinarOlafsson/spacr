"""Prepare independent reference copies and verify real tutorial mapping output.

No reference name or read is invented. The two reversed tables retain exactly
the identifiers in the maintainer's original primers_3 and guide tables.
"""
from collections import Counter
import csv
import hashlib
from pathlib import Path
import shutil


REFERENCES = {
    'column': 'primers_3_column_barecodes.csv',
    'row': 'primers_3_row_barecodes.csv',
    'grna': 'grna_barcodes.csv',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference_rows(path):
    with Path(path).open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(not row.get('name') or not row.get('sequence') for row in rows):
        raise ValueError('A reference needs nonempty name and sequence columns')
    sequences = [row['sequence'].upper() for row in rows]
    if any(set(sequence) - set('ACGT') for sequence in sequences):
        raise ValueError('The recorded reference must contain concrete DNA bases')
    return [(row['sequence'].upper(), row['name']) for row in rows]


def table(path):
    """Resolve only unambiguous sequence identities; never choose a duplicate."""
    names = {}
    for sequence, name in reference_rows(path):
        names.setdefault(sequence, set()).add(name)
    return {sequence: next(iter(values)) for sequence, values in names.items()
            if len(values) == 1}


def require_reverse_copy(original, reversed_path):
    """Check sequence AND identifier identity, independently of the app helper."""
    complement = str.maketrans('ACGT', 'TGCA')
    expected = Counter((sequence.translate(complement)[::-1], name)
                       for sequence, name in reference_rows(original))
    if Counter(reference_rows(reversed_path)) != expected:
        raise ValueError('The reverse-complement copy changed a sequence or its name')
    return sum(expected.values())


def prepare_references(source, destination):
    """Copy existing plain/RC pairs to a new private tutorial directory."""
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError('Use a new reference directory; never overwrite originals')
    selected = {}
    for role, name in REFERENCES.items():
        original = source / name
        flipped = original.with_name(original.stem + '_RC.csv')
        require_reverse_copy(original, flipped)
        selected[role] = (original, flipped)
    destination.mkdir(parents=True)
    result = {}
    for role, pair in selected.items():
        result[role] = {}
        for direction, original in zip(('plain', 'reverse_complement'), pair):
            target = destination / original.name
            before = digest(original)
            shutil.copyfile(original, target)
            if digest(target) != before or digest(original) != before:
                raise ValueError('A reference changed while being copied')
            result[role][direction] = {'source': str(original), 'path': str(target),
                                       'sha256': before, 'rows': len(table(target))}
    return result


def verify_counts(folder, references, expected_pairs):
    """Rebuild counts from saved reads, checking every sequence-to-name join."""
    import pandas as pd
    folder = Path(folder)
    reads = pd.read_hdf(folder / 'annotated_reads.h5', key='df')
    counts = pd.read_csv(folder / 'unique_combinations.csv')
    columns = {'column': ('column_sequence', 'columnID'),
               'row': ('row_sequence', 'rowID'), 'grna': ('grna_sequence', 'grna_name')}
    # Legacy output uses these historically misspelled sequence-column names.
    aliases = {'column_sequence': 'column', 'row_sequence': 'row', 'grna_sequence': 'grna'}
    ids = []
    for role, path in references.items():
        sequence, identity = columns[role]
        if sequence not in reads:
            sequence = aliases[sequence]
        lookup = table(path)
        expected = reads[sequence].map(lookup)
        actual = reads[identity]
        if not ((expected == actual) | (expected.isna() & actual.isna())).all():
            raise ValueError('A saved barcode name does not match its reference sequence')
        ids.append(identity)
    actual_counts = Counter({tuple(str(row[key]) for key in ids): int(row['count'])
                             for row in counts.to_dict('records')})
    if len(actual_counts) != len(counts) or (counts['count'] <= 0).any():
        raise ValueError('Count rows must be unique and positive')
    expected_counts = Counter(tuple(str(value) for value in row)
                              for row in reads.dropna(subset=ids)[ids].itertuples(index=False, name=None))
    if actual_counts != expected_counts or not 0 < sum(actual_counts.values()) <= len(reads) <= expected_pairs:
        raise ValueError('Saved counts do not reconcile with the annotated reads')
    return {'accepted': True, 'scope': 'Exact saved barcode joins and count reconciliation',
            'requested_pairs': expected_pairs, 'extracted_rows': len(reads),
            'mapped_reads': sum(actual_counts.values()), 'count_rows': len(counts),
            'reference_sizes': {role: len(table(path)) for role, path in references.items()},
            'artifacts': {name: digest(folder / name) for name in
                          ('annotated_reads.h5', 'unique_combinations.csv', 'qc.csv')},
            'biological_validation_claimed': False}
