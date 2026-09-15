"""A hand fix to a committed runtime catalog must survive the next plain build.

Item 406. `_translate_batches` keeps a translation cache OUTSIDE the
repository, at ``<model_root>/.spacr_translation_cache/<language>.json``. A
``--repair-invalid-only`` pass wrote 87 fluent-but-wrong zh_CN rows into it
(``'Pca whiten' -> 白色白色``). They were reverted and three were fixed by hand
in ``spacr/qt/i18n_catalogs/zh_CN.py``, and the next PLAIN build put every one
back, because a plain build preferred the cache to the committed catalog.

The builder already had the mechanism to carry committed rows forward:
`_seed_cache_from_catalog` reads the committed catalog, keeps only rows whose
``SOURCE_HASHES`` entry still matches the English source and whose value
passes the release gates, and offers them as cache entries. It used
``cache.setdefault``, so ANY value already in the cache for that source beat
the committed row. A repair pass's output therefore outranked every later
edit to the file that ships.

The rule these tests pin:

  * PLAIN BUILD. A committed row whose English source is unchanged (its
    per-row source hash is current) and that passes the gates is carried
    forward, and outranks the cache. Only new or changed sources reach the
    cache or the model.
  * REPAIR MODES keep their meaning. A source passed in ``force_sources`` is
    not seeded from the catalog and not read from the cache; it goes to the
    model, and the accepted result is checkpointed to the cache.
  * REVIEWED RECORDS (``docs/i18n/reviewed/runtime/<lang>/*.json``) beat the
    committed catalog, the cache and the model, in both modes.
  * A committed row that FAILS the gates is not carried forward. The override
    is for rows the release contract accepts, not for any text in the file.

Nothing here loads MADLAD, OPUS or M2M, or touches a GPU, or reads or writes
the real cache: the model root is a temporary directory with an empty model
folder, ``transformers`` is a stub, and the reviewed-record directory is
redirected to a temporary one.
"""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
KEY = "item_406_fixture"

#: Per language: an English source, the value committed to the catalog (the
#: hand fix), a DIFFERENT value the cache holds for the same source, and a
#: third value a reviewer wrote. The zh_CN row is the real regression from the
#: item; the German row exercises the OPUS path.
ROWS = {
    "de": {
        "source": "Whiten the principal components before clustering",
        "committed": "Hauptkomponenten vor dem Clustering weißen",
        "cached": "Hauptkomponenten vor dem Clustering aufhellen",
        "reviewed": "Hauptkomponenten vor dem Clustering dekorrelieren",
    },
    "zh_CN": {
        "source": "Pca whiten",
        "committed": "Pca 白化",
        "cached": "白色白色",
        "reviewed": "Pca 白化处理",
    },
}
CHANGED_SOURCE = "Whiten the principal components after clustering"
STUB_TRANSLATION = "Hauptkomponenten nach dem Clustering dekorrelieren"
STUB_ALTERNATIVE = "Hauptkomponenten nach dem Clustering entfärben"


@pytest.fixture(scope="module")
def builder():
    tools = str(ROOT / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    import build_i18n_catalogs

    return build_i18n_catalogs


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@pytest.fixture
def isolated(builder, monkeypatch, tmp_path):
    """Redirect every input the selection path reads to synthetic state.

    Returns a callable that installs one committed catalog row, one cache
    file and (optionally) one reviewed record for a language, and hands back
    the temporary model root.
    """
    from spacr.qt.i18n_catalogs import en

    reviewed_dir = tmp_path / "reviewed"
    reviewed_dir.mkdir()
    monkeypatch.setattr(builder, "REVIEWED_RUNTIME_DIR", reviewed_dir)
    builder.reviewed_runtime_translations.cache_clear()

    # The canonical English tables drive seeding; shrink them to the fixture
    # so no real catalog row takes part.
    for name in ("SETTING_TOOLTIPS", "MODULE_SUMMARIES"):
        monkeypatch.setattr(en, name, {}, raising=False)

    def install(
        language: str,
        *,
        english_source: str,
        committed_source: str,
        committed_value: str,
        cache: dict[str, str],
        reviewed: str | None = None,
    ) -> Path:
        target = __import__(
            f"spacr.qt.i18n_catalogs.{language}", fromlist=["*"]
        )
        monkeypatch.setattr(en, "SETTING_LABELS", {KEY: english_source})
        monkeypatch.setattr(target, "MODEL", builder.MODEL_SPECS[language][0])
        monkeypatch.setattr(target, "SETTING_LABELS", {KEY: committed_value})
        for name in ("SETTING_TOOLTIPS", "CATEGORY_HELP", "UI",
                     "MODULE_SUMMARIES"):
            monkeypatch.setattr(target, name, {}, raising=False)
        # The committed catalog's per-row hash records the English it was
        # built from. `committed_source != english_source` is a changed row.
        monkeypatch.setattr(target, "SOURCE_HASHES", {
            ("SETTING_LABELS", KEY): _sha(committed_source),
        }, raising=False)

        if reviewed is not None:
            monkeypatch.setattr(builder, "canonical_sources", lambda: {
                "setting_labels": {KEY: english_source},
            })
            language_dir = reviewed_dir / language
            language_dir.mkdir(parents=True, exist_ok=True)
            (language_dir / "item-406.json").write_text(json.dumps({
                "language": language,
                "schema": 1,
                "records": [{
                    "table": "setting_labels",
                    "key": KEY,
                    "source": english_source,
                    "source_sha256": _sha(english_source),
                    "translation": reviewed,
                }],
            }, ensure_ascii=False), encoding="utf-8")
        builder.reviewed_runtime_translations.cache_clear()

        model_root = tmp_path / "opus"
        model_root.mkdir(exist_ok=True)
        (model_root / builder.MODEL_SPECS[language][1]).mkdir(
            parents=True, exist_ok=True,
        )
        cache_dir = model_root / ".spacr_translation_cache"
        cache_dir.mkdir(exist_ok=True)
        (cache_dir / f"{language}.json").write_text(
            json.dumps(cache, ensure_ascii=False), encoding="utf-8",
        )
        return model_root

    yield install
    builder.reviewed_runtime_translations.cache_clear()


def _forbid_model(monkeypatch):
    """Any attempt to load a checkpoint fails the test by name."""
    def refuse(*_args, **_kwargs):
        raise AssertionError("the translation model must not be loaded")

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=refuse),
        AutoModelForSeq2SeqLM=types.SimpleNamespace(from_pretrained=refuse),
    ))


def _stub_model(monkeypatch, generated: list[dict]):
    """A two-beam stub whose best candidate is ``STUB_TRANSLATION``."""
    import torch

    class FakeTokenizer:
        eos_token_id = 2
        model_max_length = 480

        def __call__(self, value, **_kwargs):
            if isinstance(value, str):
                return {"input_ids": [1, 2]}
            width = max(2, max(map(len, value)))
            input_ids = torch.ones((len(value), width), dtype=torch.long)
            return {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
            }

        def batch_decode(self, output, **_kwargs):
            values = {11: STUB_TRANSLATION, 12: STUB_ALTERNATIVE}
            return [values[int(sequence[0])] for sequence in output]

    class FakeModel:
        def eval(self):
            return self

        def generate(self, **kwargs):
            generated.append(kwargs)
            batch = int(kwargs["input_ids"].shape[0])
            return torch.tensor([[11, 2], [12, 2]] * batch, dtype=torch.long)

    tokenizer = FakeTokenizer()
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(
            from_pretrained=lambda *_args, **_kwargs: tokenizer,
        ),
        AutoModelForSeq2SeqLM=types.SimpleNamespace(
            from_pretrained=lambda *_args, **_kwargs: FakeModel(),
        ),
    ))


def _translate(builder, strings, language, model_root, **kwargs):
    return builder._translate_batches(
        list(strings),
        language,
        model_root,
        device="cpu",
        batch_size=4,
        beams=2,
        threads=1,
        **kwargs,
    )


def _read_cache(model_root: Path, language: str) -> dict[str, str]:
    path = model_root / ".spacr_translation_cache" / f"{language}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_fixture_values_all_pass_the_release_gates(builder):
    """Precondition: only precedence can decide which value a build picks.

    If the cached value were rejected by a gate, a plain build would discard
    it for that reason alone and test (a) would pass without the rule.
    """
    for language, row in ROWS.items():
        for name in ("committed", "cached", "reviewed"):
            assert builder._translation_candidate_valid(
                row["source"], row[name], language,
            ), (language, name)
    assert builder._translation_candidate_valid(
        CHANGED_SOURCE, STUB_TRANSLATION, "de",
    )


@pytest.mark.parametrize("language", sorted(ROWS))
def test_a_committed_row_with_an_unchanged_source_outranks_the_cache(
    builder, isolated, monkeypatch, language,
):
    """(a) The regression: a plain build must not put the cache value back."""
    row = ROWS[language]
    model_root = isolated(
        language,
        english_source=row["source"],
        committed_source=row["source"],
        committed_value=row["committed"],
        cache={row["source"]: row["cached"]},
    )
    _forbid_model(monkeypatch)

    translated = _translate(builder, [row["source"]], language, model_root)

    assert translated == {row["source"]: row["committed"]}


def test_the_catalog_seed_replaces_a_different_cached_value(
    builder, isolated,
):
    """(a, at the seam) The seed is where committed rows enter selection."""
    row = ROWS["de"]
    isolated(
        "de",
        english_source=row["source"],
        committed_source=row["source"],
        committed_value=row["committed"],
        cache={},
    )
    cache = {row["source"]: row["cached"], "unrelated": "unberührt"}

    builder._seed_cache_from_catalog("de", cache)

    assert cache == {row["source"]: row["committed"], "unrelated": "unberührt"}


def test_a_changed_source_takes_its_translation_from_the_cache(
    builder, isolated, monkeypatch,
):
    """(b) A stale per-row hash means the committed value is not reused."""
    row = ROWS["de"]
    cached_new = "Hauptkomponenten nach dem Clustering aufhellen"
    model_root = isolated(
        "de",
        english_source=CHANGED_SOURCE,
        committed_source=row["source"],
        committed_value=row["committed"],
        cache={CHANGED_SOURCE: cached_new},
    )
    _forbid_model(monkeypatch)

    translated = _translate(builder, [CHANGED_SOURCE], "de", model_root)

    assert translated == {CHANGED_SOURCE: cached_new}


def test_a_changed_source_with_no_cache_entry_goes_to_the_model(
    builder, isolated, monkeypatch,
):
    """(b) ...and with nothing cached for the new English, the model runs."""
    row = ROWS["de"]
    model_root = isolated(
        "de",
        english_source=CHANGED_SOURCE,
        committed_source=row["source"],
        committed_value=row["committed"],
        cache={},
    )
    generated: list[dict] = []
    _stub_model(monkeypatch, generated)

    translated = _translate(builder, [CHANGED_SOURCE], "de", model_root)

    assert translated == {CHANGED_SOURCE: STUB_TRANSLATION}
    assert len(generated) == 1


def test_a_repair_mode_retranslates_the_rows_it_targets(
    builder, isolated, monkeypatch,
):
    """(c) A forced source bypasses the committed row AND the cache.

    The accepted repair is still checkpointed to the cache: that is what lets
    an interrupted multi-hour repair resume, and under the rule above a cached
    value can no longer override a hash-current committed row, so writing it
    is safe. See item 406, 2026-09-15.
    """
    row = ROWS["de"]
    source = row["source"]
    # Unchanged source, hash-current committed row, different cached value:
    # exactly the row a plain build now carries forward. Only the repair's
    # force set may send it to the model.
    model_root = isolated(
        "de",
        english_source=source,
        committed_source=source,
        committed_value=row["committed"],
        cache={source: row["cached"]},
    )
    generated: list[dict] = []
    _stub_model(monkeypatch, generated)

    translated = _translate(
        builder, [source], "de", model_root,
        force_sources={source}, repair_protected=True,
    )

    assert translated == {source: STUB_TRANSLATION}
    assert len(generated) == 1
    assert _read_cache(model_root, "de")[source] == STUB_TRANSLATION


@pytest.mark.parametrize("forced", [False, True], ids=["plain", "repair"])
@pytest.mark.parametrize("language", sorted(ROWS))
def test_a_reviewed_record_beats_the_committed_row_and_the_cache(
    builder, isolated, monkeypatch, language, forced,
):
    """(d) Review evidence is authoritative in every mode."""
    row = ROWS[language]
    model_root = isolated(
        language,
        english_source=row["source"],
        committed_source=row["source"],
        committed_value=row["committed"],
        cache={row["source"]: row["cached"]},
        reviewed=row["reviewed"],
    )
    _forbid_model(monkeypatch)

    translated = _translate(
        builder, [row["source"]], language, model_root,
        force_sources={row["source"]} if forced else (),
    )

    assert translated == {row["source"]: row["reviewed"]}


def test_a_committed_row_that_fails_the_gates_does_not_outrank_the_cache(
    builder, isolated, monkeypatch,
):
    """The override is for rows the release contract accepts.

    An English fallback left in the committed catalog is not a hand fix; a
    valid cached translation still fills it.
    """
    row = ROWS["de"]
    model_root = isolated(
        "de",
        english_source=row["source"],
        committed_source=row["source"],
        committed_value=row["source"],
        cache={row["source"]: row["cached"]},
    )
    _forbid_model(monkeypatch)

    translated = _translate(builder, [row["source"]], "de", model_root)

    assert translated == {row["source"]: row["cached"]}
