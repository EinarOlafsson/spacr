"""A second model-reaching translation call in one process must not abort it.

Item 43, 2026-09-15. CI run 34989909231, job "Minimum dependencies
(ubuntu-24.04, py3.9)": worker gw1 died with ``Fatal Python error: Aborted``
in ``test_a_committed_row_outranks_the_translation_cache.py::
test_a_repair_mode_retranslates_the_rows_it_targets``. The faulthandler stack
ended in ``tools/build_i18n_catalogs.py`` ``_translate_batches`` at
``torch.set_num_interop_threads(1)``. There was no Qt on it.

PyTorch fixes the inter-op pool size once per process: a second call, or a
call after inter-op work has started, fails a C++ check. Recent torch (2.13)
turns that into a ``RuntimeError``, which the old ``except RuntimeError``
swallowed. torch 2.1.0, the minimum-dependency pin, does not: the
``c10::Error`` escapes the binding, ``std::terminate`` runs ("terminate called
after throwing an instance of 'c10::Error'") and the process gets SIGABRT. An
abort cannot be caught, so the xdist worker that had already run one
model-reaching test (``test_a_changed_source_with_no_cache_entry_goes_to_the_
model``) died on the next.

The rule pinned here: the builder asks torch to set the inter-op pool size at
most once per process, and not at all when torch already reports the wanted
size. Each case runs in a SUBPROCESS, so an abort is a failed assertion that
carries its stderr, not a dead pytest worker. Nothing loads a real model:
``transformers`` is a stub, as in the item-406 tests.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

SCRIPT = r'''
import json
import sys
import types
from pathlib import Path

root, work, mode = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
sys.path[:0] = [str(root / "tools"), str(root)]

import torch

import build_i18n_catalogs as builder

assert str(root) in builder.__file__, builder.__file__

# A test worker has done torch work before it reaches the builder.
(torch.ones(64, 64) @ torch.ones(64, 64)).sum()
if mode == "already_one":
    # Someone else in this process already fixed the pool at the wanted size.
    torch.set_num_interop_threads(1)

calls = []
real_set = torch.set_num_interop_threads


def counting_set(count):
    calls.append(count)
    real_set(count)


torch.set_num_interop_threads = counting_set

SOURCE = "Whiten the principal components after clustering"
TRANSLATION = "Hauptkomponenten nach dem Clustering dekorrelieren"
ALTERNATIVE = "Hauptkomponenten nach dem Clustering entfärben"


class FakeTokenizer:
    eos_token_id = 2
    model_max_length = 480

    def __call__(self, value, **_kwargs):
        if isinstance(value, str):
            return {"input_ids": [1, 2]}
        width = max(2, max(map(len, value)))
        input_ids = torch.ones((len(value), width), dtype=torch.long)
        return {"input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids)}

    def batch_decode(self, output, **_kwargs):
        values = {11: TRANSLATION, 12: ALTERNATIVE}
        return [values[int(sequence[0])] for sequence in output]


class FakeModel:
    def eval(self):
        return self

    def generate(self, **kwargs):
        batch = int(kwargs["input_ids"].shape[0])
        return torch.tensor([[11, 2], [12, 2]] * batch, dtype=torch.long)


sys.modules["transformers"] = types.SimpleNamespace(
    AutoTokenizer=types.SimpleNamespace(
        from_pretrained=lambda *_a, **_k: FakeTokenizer()),
    AutoModelForSeq2SeqLM=types.SimpleNamespace(
        from_pretrained=lambda *_a, **_k: FakeModel()),
)

# Seeding validates every committed catalog row (about 20 s) and has nothing
# to do with thread setup; the builder documents ``lambda *args`` as the stub.
builder._seed_cache_from_catalog = lambda *args: None
reviewed = work / "reviewed"
reviewed.mkdir()
builder.REVIEWED_RUNTIME_DIR = reviewed
builder.reviewed_runtime_translations.cache_clear()
model_root = work / "opus"
(model_root / builder.MODEL_SPECS["de"][1]).mkdir(parents=True)

for _attempt in range(2):
    # A forced source bypasses the committed catalog and the cache, so both
    # calls reach the model and the thread setup in front of it.
    translated = builder._translate_batches(
        [SOURCE], "de", model_root, device="cpu", batch_size=4, beams=2,
        threads=1, force_sources={SOURCE},
    )
    assert translated == {SOURCE: TRANSLATION}, translated

import spacr

assert str(root) in spacr.__file__, spacr.__file__
print(json.dumps({"torch": torch.__version__, "interop_calls": calls,
                  "interop_threads": torch.get_num_interop_threads()}))
'''


def _run(tmp_path: Path, mode: str) -> dict:
    script = tmp_path / "translate_twice.py"
    script.write_text(SCRIPT, encoding="utf-8")
    work = tmp_path / "work"
    work.mkdir()
    env = dict(os.environ)
    # The script's own directory would otherwise come first on sys.path and a
    # different installed spaCR could answer the import.
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(ROOT), env.get("PYTHONPATH", "")) if part)
    result = subprocess.run(
        [sys.executable, str(script), str(ROOT), str(work), mode],
        capture_output=True, text=True, env=env, timeout=300,
    )
    assert result.returncode == 0, (
        f"the process died or failed (returncode {result.returncode}); a "
        f"negative code is a signal, -6 is SIGABRT\n"
        f"--- stderr tail ---\n{result.stderr[-4000:]}"
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("mode", ["fresh", "already_one"])
def test_two_translation_calls_ask_for_the_interop_pool_at_most_once(
    tmp_path, mode,
):
    report = _run(tmp_path, mode)

    assert report["interop_threads"] == 1, report
    if mode == "fresh":
        # Applied once, when it still can be; never asked for again.
        assert report["interop_calls"] == [1], report
    else:
        # Already what the builder wants: no call at all.
        assert report["interop_calls"] == [], report
