"""Join genuine preparation and Classify frames with their preserved inputs.

This demonstrates an explicit canonical-name preparation workaround, not a
repair to the application's legacy filename parser or a useful trained model.
"""
from copy import deepcopy
import argparse
import hashlib
import os
import shutil
from pathlib import Path

from classify_split_evidence import inspect_finished, inspect_metrics, mapped_source_path
from compose_report_capture import _frame, _read, _same_hash
from stage_lesson import DEFAULT_STAGE, REPO, read, write


def compose(stage=DEFAULT_STAGE, *, terminal=None, gui=None, dataset=None, destination=None,
            recorded_stage=None, preparation_references=None, copy_inputs=False, gui_sources=None):
    stage = Path(stage).resolve()
    terminal = Path(terminal or stage / 'captures/classify_canonical_preparation').resolve()
    gui = Path(gui or stage / 'classify_canonical_capture_v2/captures/classify_canonical_existing_v2').resolve()
    destination = Path(destination or stage / 'captures/classify_canonical_verified').resolve()
    if destination.exists():
        raise FileExistsError('Preserve the existing accepted composition')
    hashes = {}
    preparation = _read(terminal / 'scientific_acceptance.json', hashes)
    original_sources=[Path(p).resolve() for p in gui_sources] if gui_sources is not None else [gui]
    if not original_sources or len(set(original_sources))!=len(original_sources):
        raise ValueError('Expected unique original GUI capture sources')
    recorded = _read(original_sources[0] / 'scientific_acceptance.json', hashes)
    batch = _read(original_sources[0] / 'batch_acceptance.json', hashes)
    dataset = Path(dataset or stage / 'classify_canonical_split_v2').resolve()
    mapping = None if recorded_stage is None else (Path(recorded_stage),stage)
    proof = inspect_finished(dataset,path_mapping=mapping)
    metrics = inspect_metrics(dataset,path_mapping=mapping)
    if (not batch.get('accepted') or preparation.get('accepted') is not True
            or preparation.get('preparation_only') is not True
            or preparation.get('model_started') is not False
            or preparation.get('filename_parser_fixed') is not False
            or (preparation.get('destination') != str(dataset) if mapping is None else
                mapped_source_path(preparation.get('destination'),mapping,file=False) != dataset)
            or preparation.get('inputs') != proof['inputs']
            or recorded.get('inputs') != proof['inputs']
            or recorded.get('native_audits') != proof['native_audits']
            or recorded.get('independent_test_metrics') != metrics):
        raise ValueError('Preparation, native run and independently checked outputs must agree')
    original_proofs=[]
    for original in original_sources:
        scientific=_read(original/'scientific_acceptance.json',hashes)
        accepted=_read(original/'batch_acceptance.json',hashes)
        provenance=_read(original/'provenance.json',hashes)
        if (accepted.get('accepted') is not True or provenance.get('module')!='classify_merged'
                or scientific.get('inputs')!=proof['inputs']
                or scientific.get('native_audits')!=proof['native_audits']
                or scientific.get('independent_test_metrics')!=metrics):
            raise ValueError('Every GUI take must match the exact inspected inputs, audits and saved metrics')
        original_proofs.append(dict(capture=str(original),scientific=scientific,batch=accepted))
    if gui_sources is not None:
        union=_read(gui/'provenance.json',hashes)
        identity=_read(gui/'exact-source.json',hashes)
        members=union.get('composition')
        if (union.get('completed_capture') is not True or not isinstance(members,list)
                or [row.get('source') for row in members]!=list(map(str,original_sources))
                or identity.get('composition')!=members):
            raise ValueError('Expected the normal exact-source GUI capture union')
        for original,row in zip(original_sources,members):
            for name,key in [('frames.json','manifest_sha256'),('provenance.json','provenance_sha256')]:
                _same_hash(original/name,row[key],hashes)
            source_identity=original/'exact-source.json'
            if not source_identity.is_file():source_identity=original.parent.parent/'exact-source.json'
            _same_hash(source_identity,row['exact_source_sha256'],hashes)
            if _read(source_identity,hashes).get('sha256')!=row['source_files_sha256']:
                raise ValueError('GUI union source identity changed')
    _same_hash(Path(__file__).with_name('prepare_classify_split.py'),
               preparation['helper_sha256'], hashes)
    references = None
    if preparation_references is not None:
        from render_classify_input_reference_cards import validate_cards
        preparation_references=Path(preparation_references).resolve()
        references=validate_cards(preparation_references,
            read(REPO/'tools/tutorials/lessons/10_classify_cv.json'),REPO)
        for path in [preparation_references/'cards.json',
                     *(preparation_references/'source-results').iterdir()]:
            _same_hash(path,hashlib.sha256(path.read_bytes()).hexdigest(),hashes)
    frames, sources, copies = {}, [], {}
    def destination_input(path,group):
        if not copy_inputs:return os.path.relpath(path,destination)
        relative=Path('_inputs')/group/path.name
        if relative in copies and copies[relative]!=path:
            raise ValueError('Different source files would share a staged name')
        copies[relative]=path
        return str(relative)

    for prefix, root in [('prepare', terminal), ('gui', gui)]:
        provenance = _read(root / 'provenance.json', hashes)
        if (provenance.get('completed_capture') is not True or
                (provenance.get('module') != 'classify_merged' and not (prefix=='gui' and gui_sources is not None))):
            raise ValueError('Expected completed genuine Classify captures')
        sources.append(provenance)
        original_frames=_read(root / 'frames.json', hashes)
        if prefix=='prepare' and references is not None:
            for visual,card in references.items():
                path=_frame(preparation_references/card['image'],card['sha256'],preparation_references,hashes)
                frames[visual]=dict(image=destination_input(path,'references'),sha256=card['sha256'],
                    kind='authored_reference_card',narration_sha256=card['narration_sha256'],
                    source_reference_contract=str(preparation_references/'cards.json'),buttons=[])
            continue
        for key, original in original_frames.items():
            frame = deepcopy(original)
            path = _frame(root / frame['image'], frame['sha256'], root, hashes)
            frame['image'] = destination_input(path,prefix)
            frame['source_capture'] = str(root)
            if frame.get('clip'):
                clip=frame['clip']
                for field,sha_field in [('video','sha256'),('receipt','receipt_sha256'),
                                        ('native_video','native_sha256')]:
                    if field not in clip:
                        if field!='native_video':raise ValueError('Native clip source binding missing')
                        continue
                    original_root=root
                    original_path=root/clip[field]
                    if field=='native_video' and not original_path.is_file() and gui_sources is not None:
                        origin=frame.get('composition_source',{})
                        original_root=Path(origin.get('capture','')).resolve()
                        if original_root not in original_sources:
                            raise ValueError('Native source is not an original GUI union member')
                        source_frame=_read(original_root/'frames.json',hashes)[origin['visual']]
                        if (source_frame.get('clip',{}).get(field)!=clip[field]
                                or source_frame['clip'].get(sha_field)!=clip[sha_field]):
                            raise ValueError('Original native source binding changed')
                        original_path=original_root/clip[field]
                    if (original_path.is_symlink() or not original_path.resolve().is_relative_to(original_root)):
                        raise ValueError('Native clip source leaves its original capture')
                    _same_hash(original_path,clip[sha_field],hashes)
                    clip[field]=destination_input(original_path.resolve(),prefix)
            frames[prefix + '_' + key] = frame
    for path, digest in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError('A composition input changed while being read')
    if copy_inputs:
        identity=gui/'exact-source.json'
        if not identity.is_file():identity=gui.parent.parent/'exact-source.json'
        if identity.is_symlink() or not _read(identity,hashes).get('sha256'):
            raise ValueError('GUI capture needs its actual exact source identity')
        copies[Path('exact-source.json')]=identity
        if gui_sources is None:
            gate=gui/'host_gate_summary.json'
            _read(gate,hashes)
            copies[Path('host_gate_summary.json')]=gate
        else:
            source_gates={}
            for index,original in enumerate(original_sources):
                gate=original/'host_gate_summary.json'
                source_gates[str(original)]=_read(gate,hashes)
                copies[Path('_inputs/gui-gates')/f'{index}.json']=gate
            union_clips={}
            for visual,frame in frames.items():
                if not visual.startswith('gui_'):continue
                origin=frame['composition_source']
                union_clips[visual]=source_gates[origin['capture']]['clips'][origin['visual']]
    for path,expected in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:
            raise ValueError('A composition input changed before stage copy')
    destination.mkdir()
    for relative,path in copies.items():
        target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True)
        try:os.link(path,target)
        except OSError:shutil.copy2(path,target)
        if hashlib.sha256(target.read_bytes()).hexdigest()!=hashes[str(path)]:
            raise ValueError('Staged copy differs from its original source')
    if copy_inputs and gui_sources is not None:
        write(destination/'host_gate_summary.json',dict(clips=union_clips,
            failed=[name for name,gate in union_clips.items() if gate.get('passed') is not True],
            composition_sources=[dict(capture=str(original),
                sha256=hashes[str(original/'host_gate_summary.json')]) for original in original_sources]))
    write(destination / 'frames.json', frames)
    write(destination / 'provenance.json', dict(sources[-1], sources=sources,
          composition_only=True, app_source_modified=False))
    write(destination / 'scientific_acceptance.json', dict(accepted=True,
          scope='Explicit canonical-name preparation and native existing-split CV run',
          preparation=preparation, native=proof, independent_test_metrics=metrics,
          batch_acceptance=batch, source_hashes=hashes, filename_parser_fixed=False,
          biological_validation=False, useful_model_claimed=False, published=False,
          **(dict(recorded_stage=str(recorded_stage),physical_stage=str(stage)) if mapping else {}),
          **(dict(preparation_reference_contract=str(preparation_references/'cards.json')) if references else {}),
          **(dict(original_gui_proofs=original_proofs) if gui_sources is not None else {})))
    print(destination)
    return frames


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, default=DEFAULT_STAGE)
    for name in ('terminal', 'gui', 'dataset', 'destination', 'recorded-stage', 'preparation-references'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args()
    compose(args.stage, terminal=args.terminal, gui=args.gui,
            dataset=args.dataset, destination=args.destination, recorded_stage=args.recorded_stage,
            preparation_references=args.preparation_references)
