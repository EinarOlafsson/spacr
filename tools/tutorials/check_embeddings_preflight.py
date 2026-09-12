"""Reconcile actual encoder artifacts and guards before recording narration."""
from pathlib import Path
import json

from check_embeddings_example_mutations import run as mutations
from embeddings_example import digest, verify_saved
from stage_lesson import DEFAULT_STAGE, write


def main():
    import numpy as np
    root = DEFAULT_STAGE / 'embeddings_1507_preflight'
    reports = []
    for policy, width in [('per_channel', 1536), ('project', 512)]:
        output = root / policy
        report = json.loads((output / 'run.json').read_text())
        if (report['helper_sha256'] != digest(Path(__file__).with_name('embeddings_example.py'))
                or report['shape'] != [16, width] or report['accepted'] is not True):
            raise ValueError('Actual preflight report is stale or incomplete')
        for name, sha in report['artifacts'].items():
            if digest(output / name) != sha:
                raise ValueError('Changed actual artifact: ' + name)
        for record in report['sources']:
            if digest(Path(report['source_folder']) / record['name']) != record['sha256']:
                raise ValueError('Source crop changed')
        values = np.load(output / 'vectors.npy', allow_pickle=False)
        prefix = 'emb_rgb_' if policy == 'project' else 'emb_c'
        names = ([f'emb_rgb_{i:03d}' for i in range(512)] if policy == 'project' else
                 [f'emb_c{c}_{i:03d}' for c in range(3) for i in range(512)])
        assert all(n.startswith(prefix) for n in names)
        verify_saved(output, values, names, [s['name'] for s in report['sources']])
        reports.append(report)
    if reports[0]['sources'] != reports[1]['sources']:
        raise ValueError('The channel comparison must use the same actual crops')
    gui = DEFAULT_STAGE / 'captures/embeddings_1507_gui'
    proof = json.loads((gui / 'scientific_acceptance.json').read_text())
    if (proof['accepted'] is not True or proof['crops_injected'] is not False
            or proof['gui_workflow_completed'] is not False):
        raise ValueError('Expected honest navigation-only GUI introduction')
    frames = json.loads((gui / 'frames.json').read_text())
    for frame in frames.values():
        if digest(gui / frame['image']) != frame['sha256']:
            raise ValueError('Actual GUI frame changed')
    receipt = {'scope': 'Embeddings 1.5.0.7 real API preflight and GUI navigation only',
               'accepted': True, 'runs': reports, 'gui': proof, 'gui_frames': frames,
               'mutations': mutations(), 'video_complete': False,
               'multilingual_audio_complete': False, 'published': False}
    path = Path(__file__).with_name('evidence') / '2026-09-12_embeddings_api_preflight.json'
    write(path, receipt)
    print(path)


if __name__ == '__main__':
    main()
