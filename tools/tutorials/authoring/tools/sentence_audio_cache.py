"""Content-addressed raw speech reuse before sentence assembly and mastering."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


def cached_sentence(directory: Path, inputs: dict, synthesize, np):
    """Reuse only exact synthesis inputs and verified unmodified float32 PCM."""
    encoded = json.dumps(inputs, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    key = hashlib.sha256(encoded.encode()).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    pcm = directory / f'{key}.npy'
    metadata = directory / f'{key}.json'
    if pcm.exists() and metadata.exists():
        try:
            record = json.loads(metadata.read_text())
            data = pcm.read_bytes()
            if record['inputs'] == inputs and record['sha256'] == hashlib.sha256(data).hexdigest():
                audio = np.load(pcm, allow_pickle=False)
                if (audio.dtype == np.float32 and audio.ndim == 1 and len(audio) > 0
                        and np.isfinite(audio).all() and len(audio) == record['samples']):
                    return audio, record['phonemes'], True
        except (OSError, ValueError, KeyError):
            pass
    audio, phonemes = synthesize()
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
        raise ValueError('Cannot cache empty, nonfinite or multichannel speech')
    with tempfile.NamedTemporaryFile(dir=directory, suffix='.npy', delete=False) as handle:
        temporary = Path(handle.name)
        np.save(handle, audio, allow_pickle=False)
    try:
        record = {'inputs': inputs, 'sha256': hashlib.sha256(temporary.read_bytes()).hexdigest(),
                  'samples': len(audio), 'phonemes': phonemes}
        temporary.replace(pcm)
        pending = directory / f'.{key}.{os.getpid()}.json'
        pending.write_text(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n')
        pending.replace(metadata)
    finally:
        temporary.unlink(missing_ok=True)
    return audio, phonemes, False
