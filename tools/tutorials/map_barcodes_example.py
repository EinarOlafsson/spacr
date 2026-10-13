"""Reproduce the two-barcode API example on the genuine downloaded read pair."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

from map_barcodes_data import digest, verify_counts


def run(stage, destination):
    import pandas as pd
    from spacr.settings import barcode_set_from_settings
    from spacr.sequencing import process_chunk
    stage, destination = Path(stage), Path(destination)
    if destination.exists():
        raise FileExistsError('Use a new output folder; recorded results are never overwritten')
    capture = stage / 'captures/map_search_v2'
    settings = json.loads((capture / 'batch_settings.json').read_text())
    references = json.loads((capture / 'mapping_reference_selection.json').read_text())
    source = stage / 'example_data/sequencing'
    full = verify_counts(source / 'SRR33531217_paired', references, 10000)
    download = json.loads((capture / 'sequencing_download.json').read_text())
    inputs = {str(source / row['file']): row['sha256'] for row in download['files']}
    inputs.update({str(path): digest(path) for path in references.values()})
    if any(digest(path) != value for path, value in inputs.items()):
        raise ValueError('A recorded input changed')
    mates = []
    for index in (1, 2):
        with gzip.open(source / f'SRR33531217_{index}.fastq.gz', 'rt') as handle:
            mates.append([''.join(handle.readline() for _ in range(4)) for _ in range(1000)])
    settings['barcode_set'] = ['column', 'grna']
    selected = barcode_set_from_settings(settings)
    chunk = {'r1_chunk': mates[0], 'r2_chunk': mates[1], 'regex': settings['regex'],
             'target_sequence': settings['target_sequence'],
             'offset_start': settings['offset_start'], 'window_length': settings['window_length'],
             'barcode_set': selected, 'fill_na': False}
    frame, counts, qc = process_chunk(chunk)
    if list(counts.columns) != ['columnID', 'grna_name', 'count']:
        raise ValueError('The two-entry set did not determine the output columns')
    expected = Counter(frame.dropna(subset=['columnID', 'grna_name'])[
                       ['columnID', 'grna_name']].itertuples(index=False, name=None))
    actual = Counter({(row.columnID, row.grna_name): row.count for row in counts.itertuples()})
    if actual != expected or not 0 < sum(actual.values()) <= len(frame) <= 1000:
        raise ValueError('The two-barcode count did not reconcile')
    destination.mkdir(parents=True)
    frame.to_hdf(destination / 'annotated_reads.h5', key='df', format='table')
    counts.to_csv(destination / 'unique_combinations.csv', index=False)
    qc.to_csv(destination / 'qc.csv', index=False)
    two = verify_counts(destination, {key: references[key] for key in ('column', 'grna')}, 1000)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    plt.style.use('dark_background')
    original_counts = pd.read_csv(source / 'SRR33531217_paired/unique_combinations.csv')
    wells = original_counts.pivot_table(index='rowID', columns='columnID', values='count',
                                       aggfunc='sum', fill_value=0)
    fig, ax = plt.subplots(figsize=(13, 8), constrained_layout=True)
    image = ax.imshow(wells.to_numpy(), aspect='auto', cmap='viridis')
    ax.set_xticks(range(len(wells.columns)), labels=wells.columns, rotation=90)
    ax.set_yticks(range(len(wells.index)), labels=wells.index)
    ax.set_xlabel('Column barcode name'); ax.set_ylabel('Row barcode name')
    ax.set_title(f'Recorded GUI output: {full["mapped_reads"]:,} mapped reads\n'
                 'Python verification plot — read depth is not a biological effect')
    fig.colorbar(image, ax=ax, label='Mapped reads')
    fig.savefig(destination / 'mapped_read_depth.png', dpi=180); plt.close(fig)
    if any(digest(path) != value for path, value in inputs.items()):
        raise ValueError('An original input changed during the example')
    proof = {'accepted': True, 'scope': 'Recorded three-barcode GUI result and two-entry API set',
             'gui': full, 'api_two_barcodes': two, 'barcode_set': settings['barcode_set'],
             'input_hashes': inputs, 'inputs_unchanged': True,
             'helper_sha256': digest(__file__), 'gui_barcode_set_control_claimed': False,
             'biological_validation_claimed': False,
             'artifacts': {p.name: digest(p) for p in destination.iterdir()}}
    (destination / 'run.json').write_text(json.dumps(proof, indent=2) + '\n')
    print(f'GUI: {full["requested_pairs"]:,} pairs; {full["extracted_rows"]:,} extracted; '
          f'{full["mapped_reads"]:,} mapped; {full["count_rows"]:,} count rows.')
    print("API barcode_set = ['column', 'grna']")
    print(f'API: 1,000 pairs; {len(frame)} extracted; {sum(actual.values())} mapped.')
    print('Output columns:', ', '.join(counts.columns))
    print('Every saved count reconciles. Original inputs unchanged.')
    print('Two-barcode counts omit row identity; they are not per-well counts.')
    return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.stage, args.output)
