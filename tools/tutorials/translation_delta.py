"""Pure, fail-closed sentence deltas for review-only tutorial translation."""
from __future__ import annotations
from copy import deepcopy
from difflib import SequenceMatcher
import re
import hashlib


def localized_sentences(text, language, english_split):
    """Translation alignment only; never changes the audio renderer's cadence."""
    if language not in {'ja', 'zh-CN', 'hi', 'ko'}:
        return english_split(text)
    # Native sentence stops, plus period-space before non-Latin script.
    # A decimal such as 3.12 cannot match that period-space boundary.
    chunks = re.split(r'(?<=[。！？।!?])\s*|(?<=\.)\s+(?=[^\W\d_A-Za-z])', text.strip())
    return [sentence for chunk in chunks if chunk.strip()
            for sentence in english_split(chunk.strip())]


def unique(rows, key):
    result = {}
    for row in rows:
        identity = row[key]
        if identity in result:
            raise ValueError(f'Duplicate {key}: {identity}')
        result[identity] = row
    return result


def sentence_slots(old, new, localized, split_english, split_localized):
    """Retain only exact source matches with unambiguous sentence alignment."""
    before, after = split_english(old), split_english(new)
    blocks = SequenceMatcher(None, before, after, autojunk=False).get_matching_blocks()
    slots = [None] * len(after)
    # No source sentence survives: all new sentences must be translated,
    # so there is no old localized sentence to align or reuse.
    if not any(count for _a, _b, count in blocks):
        return after, slots
    translated = split_localized(localized) if old else []
    if len(before) != len(translated):
        raise ValueError('Reviewed sentence alignment differs; manual alignment required')
    for a, b, count in blocks:
        for offset in range(count):
            slots[b + offset] = translated[a + offset]
    return after, slots


def plan_delta(source, baseline, localized, selected, split_english, split_localized):
    """Return a draft and changed strings; no translator is imported or run."""
    if not selected or len(selected)!=len(set(selected)):
        raise ValueError('Select unique lesson identities')
    new_by = unique(source['lessons'], 'id')
    old_by = unique(baseline['lessons'], 'id')
    translated_by = unique(localized['lessons'], 'id')
    if set(selected) - new_by.keys() or set(selected) - old_by.keys():
        raise ValueError('Every delta lesson must have actual source and baseline')
    result = deepcopy(localized)
    output_by = unique(result['lessons'], 'id')
    strings, destinations, scene_plans = [], [], []

    def request(text, container, key):
        if not isinstance(text, str) or not text.strip():
            raise ValueError('Cannot translate an empty/nontext delta')
        strings.append(text)
        destinations.append((container, key))

    for identity in selected:
        new, old, translated = new_by[identity], old_by[identity], translated_by[identity]
        output = output_by[identity]
        before = unique(old['scenes'], 'visual')
        # Caption-only catalogs are positional by contract; bind to old visuals.
        if len(translated['scenes']) != len(old['scenes']):
            raise ValueError('Localized baseline scene count differs')
        local_scenes = {}
        for original, local in zip(old['scenes'], translated['scenes']):
            if local.get('visual', original['visual']) != original['visual']:
                raise ValueError('Localized baseline visual identity differs')
            local_scenes[original['visual']] = local
        unique(new['scenes'], 'visual')
        if 'title' in translated:  # Spoken catalogs also localize metadata.
            for key in ('title', 'section', 'description', 'prerequisite'):
                if old.get(key) != new.get(key):
                    sentences, slots = sentence_slots(old.get(key, ''), new[key], translated.get(key, ''),
                                                      split_english, split_localized)
                    for index, text in enumerate(sentences):
                        if slots[index] is None:
                            request(text, slots, index)
                    scene_plans.append((output, slots, key, False))
            if old.get('objectives') != new.get('objectives'):
                if len(old.get('objectives', [])) != len(new.get('objectives', [])):
                    raise ValueError('Objective count change requires explicit review')
                if len(output.get('objectives', [])) != len(old.get('objectives', [])):
                    raise ValueError('Localized objective alignment differs')
                for index, text in enumerate(new['objectives']):
                    if text != old['objectives'][index]:
                        request(text, output['objectives'], index)
        scenes = []
        for current in new['scenes']:
            visual = current['visual']
            previous = before.get(visual)
            local = local_scenes.get(visual)
            if previous and previous['narration'] == current['narration']:
                if not local or not local.get('narration', '').strip():
                    raise ValueError('Missing unchanged reviewed narration')
                scenes.append(deepcopy(local))
                continue
            old_text = previous['narration'] if previous else ''
            local_text = local['narration'] if local else ''
            sentences, slots = sentence_slots(old_text, current['narration'], local_text,
                                              split_english, split_localized)
            if not sentences:
                raise ValueError('Changed scene contains no source sentences')
            target = deepcopy(current) if 'title' in translated else {}
            target.pop('speech_text', None)
            for index, text in enumerate(sentences):
                if slots[index] is None:
                    request(text, slots, index)
            scenes.append(target)
            scene_plans.append((target, slots, 'narration', True))
        output['scenes'] = scenes
    return result, strings, destinations, scene_plans


def finish_delta(result, values, destinations, scene_plans, language, speech_text=None):
    if len(values) != len(destinations) or any(not isinstance(v, str) or not v.strip() for v in values):
        raise ValueError('Translator omitted or added delta strings')
    for (container, key), value in zip(destinations, values):
        container[key] = value.strip()
    joiner = '' if language in {'ja', 'zh-CN'} else ' '
    for scene, slots, key, is_scene in scene_plans:
        if any(not isinstance(text, str) or not text.strip() for text in slots):
            raise ValueError('Missing translated or retained sentence')
        scene[key] = joiner.join(slots)
        if is_scene and speech_text is not None:
            scene['speech_text'] = speech_text(scene['narration'], language)
    return result


def alignment_splitter(baseline, localized, language, entries, default_split):
    """Bind explicit reviewed-text segmentation to exact original source bytes."""
    english_by=unique(baseline['lessons'],'id')
    localized_by=unique(localized['lessons'],'id')
    replacements={}
    for entry in entries:
        if entry['language']!=language:
            continue
        english=english_by[entry['lesson']]
        local=localized_by[entry['lesson']]
        index=next(i for i,s in enumerate(english['scenes']) if s['visual']==entry['visual'])
        if len(local['scenes'])!=len(english['scenes']):
            raise ValueError('Explicit alignment baseline count changed')
        source_text=english['scenes'][index]['narration']
        target=local['scenes'][index]['narration']
        if (hashlib.sha256(source_text.encode()).hexdigest()!=entry['english_sha256'] or
                hashlib.sha256(target.encode()).hexdigest()!=entry['localized_sha256']):
            raise ValueError('Explicit alignment baseline bytes changed')
        segments=entry['segments']
        joiner='' if language in {'ja','zh-CN'} else ' '
        if (not segments or any(not isinstance(s,str) or not s for s in segments)
                or joiner.join(segments)!=target or target in replacements):
            raise ValueError('Explicit alignment cannot change or duplicate original text')
        replacements[target]=segments
    return lambda text: list(replacements[text]) if text in replacements else default_split(text)
