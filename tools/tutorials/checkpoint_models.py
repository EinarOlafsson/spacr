"""Record complete model-tutorial evidence only after full candidate acceptance."""
import argparse
from pathlib import Path

from build_release_candidate import copy_checked
from check_completed_matrix import digest
from model_promotion import MODELS, require_recorded_model
from stage_lesson import DEFAULT_STAGE, read, write
from validate_candidate import validate


def main(candidate):
    candidate = Path(candidate)
    validate(candidate, include_hosted_media=True, require_browser=True)
    root = Path(__file__).resolve().parent
    checkpoint = root / 'release_candidate'
    if digest(candidate / 'release-manifest.json') != digest(checkpoint / 'release-manifest.json'):
        raise ValueError('Checkpoint the fully verified model candidate first')
    browser = read(checkpoint / 'candidate-browser-checks.json')
    evidence = root / 'evidence'
    for identity, (capture, _) in MODELS.items():
        folder = DEFAULT_STAGE / 'production' / identity
        matrix = require_recorded_model(DEFAULT_STAGE, 'en', read(folder / 'lesson.en.json'))
        timing_path = folder / 'audio/en/af_heart.json'
        timing = read(timing_path)
        case = next(item for item in browser['ready_playback_cases'] if item['lesson'] == identity)
        sentences = [sentence for scene in timing['scenes'] for sentence in scene['sentences']]
        if (case['audio_sha256'] != timing['media_sha256']
                or len(case['sentence_cue_checks']) != len(sentences)):
            raise ValueError('The model candidate did not check every actual Heart sentence')
        for sentence, actual in zip(sentences, case['sentence_cue_checks']):
            midpoint = (sentence['speech_start'] + sentence['speech_end']) / 2
            if (actual['requested_audio_time'] != midpoint or abs(actual['audio'] - midpoint) >= 1
                    or actual['text'] != sentence['text'] or sentence['text'] not in actual['cues']):
                raise ValueError('Native model captions disagree with the actual narrated sentence')
        label = identity.split('_', 1)[1]
        copy_checked(timing_path, evidence / f'2026-09-12_{label}_heart_timing.json', [], evidence)
        write(evidence / f'2026-09-12_{label}_final_checks.json', {
            'lesson': identity, 'scope': 'Recorded subset only; no successful GUI inference/benchmark claim',
            'capture': read(DEFAULT_STAGE / 'captures' / capture / 'scientific_acceptance.json'),
            'matrix': matrix, 'candidate_manifest_sha256': digest(checkpoint / 'release-manifest.json'),
            'heart_timing_sha256': digest(timing_path), 'candidate_browser_case': case,
            'gui_inference_or_benchmark_completed': False, 'native_speaker_signoff': False,
            'human_listening_signoff': False, 'published': False})
        print(identity, 'actual capture, fifty voices and every native Heart caption verified', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate', type=Path)
    main(parser.parse_args().candidate)
