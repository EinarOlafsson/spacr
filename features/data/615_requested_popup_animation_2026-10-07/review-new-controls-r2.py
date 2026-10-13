from pathlib import Path
import functools
import write_reviewed_api_record as writer

builder = writer._runtime_builder()
original = builder.canonical_sources
builder.canonical_sources = functools.lru_cache(maxsize=1)(original)
script = Path('/media/carruthers/mnt3/codex/scratch/magnifier-acceptance-20261007/review-new-controls-r1.py').read_text()
exec(compile(script[script.index('keys = ['):], 'review-new-controls-r1.py:authored-translations', 'exec'))
assert builder.canonical_sources() == original(), 'canonical runtime source changed during authoring'
print('Review scope: AI technical translation review; no native-speaker signoff.', flush=True)
