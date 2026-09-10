"""Check the demonstration's split against its actual database, not filename guesses."""
from __future__ import annotations

import sqlite3
from pathlib import Path


def inspect_database_split(database, dataset):
    """Measure real well overlap; exclude copied misclassification review folders."""
    database, dataset = Path(database), Path(dataset)
    with sqlite3.connect(f'{database.resolve().as_uri()}?mode=ro', uri=True) as connection:
        rows = connection.execute(
            'SELECT png_path, plateID, rowID, columnID FROM png_list').fetchall()
    identity = {Path(path).name: tuple(str(value) for value in values)
                for path, *values in rows}
    if len(identity) != len(rows):
        raise RuntimeError('Duplicate crop basenames prevent an unambiguous database check')
    groups, counts = {}, {}
    for split in ('train', 'test'):
        paths = list((dataset / split).glob('*/*.png'))
        if not paths:
            raise RuntimeError(f'No real class crops in the {split} split')
        unknown = [path.name for path in paths if path.name not in identity]
        if unknown:
            raise RuntimeError(f'Unknown crop lineage: {unknown[:3]}')
        groups[split] = {identity[path.name] for path in paths}
        counts[split] = len(paths)
    overlap = sorted(groups['train'] & groups['test'])
    return {
        'accepted': not overlap, 'source': 'png_list plateID, rowID, columnID',
        'sample_counts': counts,
        'actual_wells': {key: sorted(value) for key, value in groups.items()},
        'overlapping_actual_wells': overlap,
        'reason': 'Actual database wells occur in both train and test.' if overlap else '',
    }
