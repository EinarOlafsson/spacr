"""A bounded published-card repair must not silently replan legacy footage."""
from copy import deepcopy
from pathlib import Path
import struct
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import retained_visual_refresh as refresh
import build_appended_candidate as append
from audit_staged_catalogs import CATALOGS
from stage_lesson import read, write


@pytest.fixture
def retained(tmp_path, monkeypatch):
    identity = '02_install_spacr'
    baseline, original, stage, repo = [tmp_path / name for name in ('baseline', 'original', 'stage', 'repo')]
    root = stage / 'production' / identity
    monkeypatch.setattr(refresh, 'REPO', repo)
    frames = [700, 700, 700, 700, 700, 920, 692, 700, 700, 700, 760, 886,
              700, 700, 700, 700, 1243]
    assert sum(frames) == 12901
    old_scenes, clocks, origins = [], [], {}
    elapsed = 0

    def put(path, content):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def record(path):
        item = refresh.binding(path)
        return dict(path=path.relative_to(baseline).as_posix(), bytes=item['bytes'], sha256=item['sha256'])

    for index, count in enumerate(frames):
        visual = 'pip_conda_01_create' if index == 6 else f'visual_{index}'
        relative = f'images/{visual}.png'
        image = put(original / relative, b'original source' + bytes([index]))
        put(root / relative, image.read_bytes())
        scene = dict(image=relative, capture_sha256=refresh.digest(image), narration=f'Text {index}')
        if index == 11:
            video = put(original / 'clips/home.mp4', b'complete sixty second original clip')
            proof = put(original / 'clips/home.json', b'original clip proof')
            for path in (video, proof):
                put(root / path.relative_to(original), path.read_bytes())
            scene['clip'] = dict(video='clips/home.mp4', sha256=refresh.digest(video),
                                 receipt='clips/home.json', receipt_sha256=refresh.digest(proof), duration=60.)
        old_scenes.append(scene)
        origins[visual] = dict(kind='published_installation', image_sha256=scene['capture_sha256'],
                               installed_identity={'version': '1.5.1.3'})
        clocks.append(dict(scene=index + 1, speech_start=elapsed / 30, duration=count / 30,
                           scene_end=(elapsed + count) / 30, visual=visual, native_live_video=index == 11))
        elapsed += count
    old_spec = dict(lesson=identity, fps=30, size=[3840, 2160], scenes=old_scenes,
                    capture_source=dict(completed_capture=True, frame_pixels_modified=False,
                                        frame_origins=origins, other_historical_proof='untouched'))
    write(original / 'scenes.json', old_spec)
    lesson = dict(id=identity, scenes=[dict(narration=s['narration']) for s in old_scenes])
    write(repo / 'tools/tutorials/lessons' / f'{identity}.json', lesson)
    write(root / 'lesson.en.json', lesson)
    for name in CATALOGS:
        catalog = dict(lessons=[{**lesson, 'narration_voices': {'en': ['af_heart']}}])
        write(baseline / 'web/catalog' / name, catalog)
        write(stage / 'catalog' / name, catalog)
    old_timing = dict(lesson=identity, fps=30, total_duration=12901 / 30, scenes=clocks,
                      visual_spec_sha256=refresh.digest(original / 'scenes.json'),
                      coverage=dict(accepted=True, old_selected_native=True))
    timing_source = baseline / 'media_host' / identity / 'video/native-live-timings.json'
    write(timing_source, old_timing)
    old_master = put(baseline / 'media_host' / identity / 'video' / f'{identity}_silent.mp4', b'old encoded master')
    audio = []
    for index in range(50):
        for suffix in ('.m4a', '.json'):
            relative = f'en/voice{index}{suffix}'
            path = put(baseline / 'media_host' / identity / 'audio' / relative,
                       f'{index}{suffix} retained'.encode())
            put(root / 'audio' / relative, path.read_bytes())
            audio.append(record(path))
    records = [record(timing_source), record(old_master), *audio]
    write(baseline / 'release-manifest.json', dict(files=records))
    commit = 'a' * 40
    write(baseline / 'publication-receipt.json', dict(
        manifest_sha256=refresh.digest(baseline / 'release-manifest.json'), commit=commit,
        media_root='https://huggingface.co/datasets/einarolafsson/spacr-tutorials/resolve/' + commit,
        tag='verified', media_files=len(records), readback=dict(commit=commit, passed=True,
            files_expected=len(records), metadata_matched=len(records), downloaded_sha256_matched=len(records),
            metadata_failures=[], download_failures=[])))
    card_dir = root / 'cards'
    card = put(card_dir / 'pip_conda_01_create.png',
               b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR' + struct.pack('>II', 3840, 2160) + b'unit card')
    author = put(repo / 'tools/tutorials/render_installation_route_cards.py', b'generator source')
    style = put(repo / 'tools/tutorials/authoring/tools/render_install_keyframes.py', b'style source')
    write(card_dir / 'frames.json', {'pip_conda_01_create': dict(image=card.name,
        sha256=refresh.digest(card), buttons=[], kind='generated_command_reference_not_terminal_output')})
    write(card_dir / 'provenance.json', dict(author_source_sha256=refresh.digest(author),
                                           style_source_sha256=refresh.digest(style)))
    new_origin = dict(kind='reference_guidance', capture=str(card_dir), source_visual='pip_conda_01_create',
                      image_sha256=refresh.digest(card), installed_identity=None,
                      representation='generated_command_reference_not_terminal_output',
                      generator_provenance=refresh.binding(card_dir / 'provenance.json'))
    spec = deepcopy(old_spec)
    spec['scenes'][6].update(image='cards/' + card.name, capture_sha256=refresh.digest(card))
    spec['capture_source']['frame_origins']['pip_conda_01_create'] = new_origin
    spec['capture_source'].update(completed_capture=False, frame_pixels_modified=True)
    write(root / 'scenes.json', spec)
    master = put(root / 'video' / f'{identity}_silent.mp4', b'new encoded master')
    log = put(stage / 'overlay.log', b'actual execution output\nTerminal exit status: 0\n')
    render_path = stage / 'render.json'
    write(render_path, dict(command=refresh.overlay_command(old_master, card, master, 4420, 692, 12901, 30),
                            returncode=0, output_log=refresh.binding(log)))
    receipt = dict(schema=1, kind='retained_visual_refresh', lesson=identity,
        baseline_root=str(baseline), original_root=str(original),
        baseline_manifest=refresh.binding(baseline / 'release-manifest.json'),
        publication_receipt=refresh.binding(baseline / 'publication-receipt.json'),
        original_spec=refresh.binding(original / 'scenes.json'), original_timing=refresh.binding(timing_source),
        original_master=refresh.binding(old_master), new_spec=refresh.binding(root / 'scenes.json'),
        new_master=refresh.binding(master), render_log=refresh.binding(render_path),
        replacements=[dict(scene=7, first_frame=4420, frame_count=692,
            original_image_sha256=old_scenes[6]['capture_sha256'], new_image_sha256=refresh.digest(card),
            original_origin=origins['pip_conda_01_create'], new_origin=new_origin)])
    receipt_path = root / 'retained-visual-refresh.json'

    def seal():
        receipt['new_spec'] = refresh.binding(root / 'scenes.json')
        receipt['render_log'] = refresh.binding(render_path)
        write(receipt_path, receipt)
        write(root / 'video/native-live-timings.json', refresh.retained_timing(old_timing,
            refresh.digest(root / 'scenes.json'), refresh.digest(receipt_path),
            refresh.digest(baseline / 'release-manifest.json')))
    seal()
    return dict(stage=stage, root=root, baseline=baseline, original=original,
                identity=identity, receipt=receipt, receipt_path=receipt_path, spec=spec,
                timing=old_timing, render=render_path, seal=seal)


def test_published_clocks_native_selection_audio_and_all_catalogs_are_retained(retained):
    r = retained
    result = refresh.checked_retained_visual_timing(r['stage'], r['identity'], r['baseline'])
    actual = read(result)
    assert actual['scenes'] == r['timing']['scenes']
    assert actual['total_duration'] == 430.03333333333336
    assert actual['scenes'][11]['duration'] == 29.533333333333335
    assert 'complete_native_clips' not in actual
    assert len(list((r['root'] / 'audio').rglob('*.*'))) == 100
    assert read(r['root'] / 'scenes.json')['capture_source']['completed_capture'] is False
    command = read(r['render'])['command']
    assert command[command.index('-progress') + 1] == 'pipe:1' and '-nostats' in command
    assert [command[index + 1] for index, arg in enumerate(command) if arg == '-threads'] == ['4', '4']
    assert "between(n,4420,5111)" in command[command.index('-filter_complex') + 1]


@pytest.mark.parametrize('change', ['clock', 'complete_native_claim', 'spec_narration', 'other_scene',
    'native_scene', 'provenance', 'old_origin', 'audio', 'extra_audio', 'catalog', 'master',
    'render_range', 'render_failed', 'render_log', 'render_nonzero_footer',
    'baseline_receipt', 'baseline_hash', 'original_source'])
def test_changed_retained_inputs_fail_closed_even_with_resealed_receipt(retained, change):
    r = retained
    root = r['root']
    if change in ('clock', 'complete_native_claim'):
        path = root / 'video/native-live-timings.json'
        value = read(path)
        if change == 'clock':
            value['scenes'][11]['duration'] = 60.
        else:
            value['complete_native_clips'] = True
        write(path, value)
    elif change in ('spec_narration', 'other_scene', 'native_scene', 'provenance'):
        value = read(root / 'scenes.json')
        if change == 'spec_narration': value['scenes'][6]['narration'] = 'Changed words'
        if change == 'other_scene': value['scenes'][8]['pointer'] = True
        if change == 'native_scene': value['scenes'][11]['clip']['duration'] = 29.
        if change == 'provenance': value['capture_source']['completed_capture'] = True
        write(root / 'scenes.json', value)
        r['seal']()
    elif change == 'old_origin':
        r['receipt']['replacements'][0]['original_origin'] = {}
        r['seal']()
    elif change in ('audio', 'extra_audio'):
        (root / 'audio/en' / ('voice0.m4a' if change == 'audio' else 'extra.m4a')).write_bytes(b'changed')
    elif change == 'catalog':
        path = r['stage'] / 'catalog' / CATALOGS[-1]
        value = read(path);value['lessons'][0]['scenes'][0]['narration'] = 'Changed translation';write(path, value)
    elif change == 'master':
        (root / 'video' / f"{r['identity']}_silent.mp4").write_bytes(b'replaced')
    elif change in ('render_range', 'render_failed'):
        value = read(r['render'])
        if change == 'render_range': value['command'][value['command'].index('-filter_complex') + 1] = 'overlay'
        else: value['returncode'] = 1
        write(r['render'], value);r['seal']()
    elif change == 'render_log':
        Path(read(r['render'])['output_log']['path']).write_bytes(b'changed')
    elif change == 'render_nonzero_footer':
        value = read(r['render'])
        log = Path(value['output_log']['path']);log.write_text('Terminal exit status: 1\n', encoding="utf-8")
        value['output_log'] = refresh.binding(log);write(r['render'], value);r['seal']()
    elif change == 'baseline_receipt':
        path = r['baseline'] / 'publication-receipt.json';value = read(path);value['readback']['passed'] = False
        write(path, value);r['receipt']['publication_receipt'] = refresh.binding(path);r['seal']()
    elif change == 'baseline_hash':
        r['receipt']['baseline_manifest']['sha256'] = '0' * 64;r['seal']()
    else:
        (r['original'] / 'images/visual_1.png').write_bytes(b'changed')
    with pytest.raises(ValueError):
        refresh.checked_retained_visual_timing(r['stage'], r['identity'], r['baseline'])


def test_native_scene_cannot_be_replaced(retained):
    r = retained;r['receipt']['replacements'][0]['scene'] = 12;r['seal']()
    with pytest.raises(ValueError, match='cannot replace native'):
        refresh.checked_retained_visual_timing(r['stage'], r['identity'], r['baseline'])


def test_retained_path_is_explicit_and_other_lessons_use_normal_native_validator(monkeypatch, tmp_path):
    import native_live_timing
    calls = []
    monkeypatch.setattr(refresh, 'checked_retained_visual_timing', lambda *args: calls.append('retained'))
    monkeypatch.setattr(native_live_timing, 'checked_native_timing', lambda *args: calls.append('native'))
    append.selected_visual_timing(tmp_path, '02_install_spacr', tmp_path, ['02_install_spacr'])
    append.selected_visual_timing(tmp_path, '43_illumination', tmp_path, ['02_install_spacr'])
    assert calls == ['retained', 'native']
    for retained_ids in (['02_install_spacr'], ['02_install_spacr', '02_install_spacr']):
        with pytest.raises(ValueError, match='Retain visual timing only'):
            append.build(tmp_path, tmp_path, ['02_install_spacr'], replace=True,
                         retain_visual_timing=retained_ids)


@pytest.mark.parametrize('withdraw', [[], ['02_conda_install'], ['03_pip_install']])
def test_first_installation_consolidation_still_requires_both_archives(withdraw):
    with pytest.raises(ValueError, match='both archival withdrawals'):
        append.installation_consolidation(['02_install_spacr'], ['02_install_spacr'], withdraw,
                                          ['02_conda_install', '03_pip_install'])


def test_already_merged_installation_refresh_needs_no_archive_withdrawal():
    assert append.installation_consolidation(['02_install_spacr'], ['02_install_spacr'], [],
                                             ['02_install_spacr']) is False
    assert append.installation_consolidation(['02_install_spacr'], ['02_install_spacr'],
        ['02_conda_install', '03_pip_install'], ['02_conda_install', '03_pip_install']) is True
    with pytest.raises(ValueError, match='explicit refresh'):
        append.installation_consolidation(['02_install_spacr'], [],
            ['02_conda_install', '03_pip_install'], ['02_conda_install', '03_pip_install'])


def test_card_selector_has_exact_conda_commands_and_rejects_invalid_selection():
    from render_installation_route_cards import selected_cards
    [card] = selected_cards(['pip_conda_01_create'])
    assert [command for _, command in card[2]] == [
        'conda create -n spacr-pip -c conda-forge python=3.12 pip',
        'conda activate spacr-pip', 'python -c "import sys; print(sys.prefix)"']
    assert 'unused' in card[3] and 'next step' in card[3]
    for invalid in ([], ['bad'], ['pip_conda_01_create'] * 2):
        with pytest.raises(ValueError): selected_cards(invalid)


@pytest.mark.parametrize('failure', [None, 'frame_count', 'pts', 'decode'])
def test_finish_retained_path_keeps_frame_pts_decode_and_web_gates(retained, monkeypatch, failure):
    import subprocess
    from types import SimpleNamespace
    import finish_staged_visual as finish

    r = retained
    monkeypatch.setattr(finish, 'REPO', refresh.REPO)
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'authoring/tools'))
    monkeypatch.setattr(finish, 'checked_native_timing', lambda *args: pytest.fail('Unexpected native replan'))
    monkeypatch.setattr(finish, 'probe', lambda path: dict(streams=[dict(codec_type='video',
        width=3840, height=2160, r_frame_rate='30/1', nb_frames=12900 if failure == 'frame_count' else 12901)]))
    monkeypatch.setattr(finish, 'timestamps', lambda path: [index / 30 +
        (0.1 if failure == 'pts' and index == 4 else 0) for index in range(12901)])
    calls = []

    def decode(command, **kwargs):
        calls.append(command)
        assert kwargs['check'] is True and '-xerror' in command
        if failure == 'decode':
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(finish.subprocess, 'run', decode)
    monkeypatch.setitem(sys.modules, 'prepare_web_media', SimpleNamespace(
        main=lambda: (r['root'] / 'poster.jpg').write_bytes(b'poster')))
    publisher = refresh.REPO / 'tools/tutorials/authoring/tools/publish_tutorials.py'
    publisher.write_text('ENCODE_ARGS = []\n', encoding="utf-8")
    monkeypatch.setattr(finish, 'stage_one', lambda *args: {'accepted': True})
    if failure:
        with pytest.raises((ValueError, subprocess.CalledProcessError)):
            finish.finish(r['stage'], r['identity'], retain_visual_from=r['baseline'])
        assert not (r['root'] / 'current-video-acceptance.json').exists()
        assert len(calls) == (1 if failure == 'decode' else 0)
    else:
        result = finish.finish(r['stage'], r['identity'], retain_visual_from=r['baseline'])
        assert len(calls) == 1 and result['master_full_decode_passed'] is True
        assert result['master_all_presentation_times_checked'] is True
        assert result['master_frame_count_matches_scene_boundaries'] is True
        assert result['native_fifty_voice_timeline'] is False
        assert result['retained_visual_timing'] is True
        assert result['web_rendition'] == {'accepted': True}
