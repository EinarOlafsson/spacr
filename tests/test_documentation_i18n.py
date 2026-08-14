"""Structural contracts for external API/docstring localization."""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))


def test_runtime_catalog_cli_pins_imports_to_its_own_checkout(tmp_path):
    """A foreign editable checkout must not supply runtime source strings."""
    foreign_root = tmp_path / "foreign-checkout"
    foreign_package = foreign_root / "spacr"
    foreign_package.mkdir(parents=True)
    (foreign_package / "__init__.py").write_text(
        "raise RuntimeError('foreign spaCR package imported')\n",
        encoding="utf-8",
    )
    builder = TOOLS / "build_i18n_catalogs.py"
    probe = """
import importlib.util
from pathlib import Path
import runpy

namespace = runpy.run_path({builder!r}, run_name="i18n_bootstrap_probe")
origin = Path(importlib.util.find_spec("spacr").origin).resolve()
print(origin)
print(Path(namespace["ROOT"]).resolve())
""".format(builder=str(builder))
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(foreign_root)
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    origin_text, root_text = completed.stdout.strip().splitlines()
    origin = Path(origin_text)
    checkout_root = Path(root_text)
    assert checkout_root == ROOT.resolve()
    assert origin.is_relative_to(checkout_root)


def test_api_reuse_rejects_changed_translation_context(tmp_path, monkeypatch):
    import build_documentation_i18n as builder

    source = "Translate this prose."
    key = "spacr.example"
    monkeypatch.setattr(builder, "API_DIR", tmp_path)
    monkeypatch.setitem(
        builder.API_TRANSLATION_CONTEXT,
        source,
        "Translate this prose with its current context.",
    )
    stale_context = "Translate this prose with its old context."
    payload = {
        "schema": 2,
        "symbols": {
            key: {
                "source_sha256": builder._source_hash(source),
                "source_blocks_sha256": builder._source_block_hashes(source),
                "translation_source_blocks_sha256": [
                    builder._source_hash(stale_context)
                ],
                "text": "Texto traduzido.",
            },
        },
    }
    (tmp_path / "pt.json").write_text(json.dumps(payload), encoding="utf-8")
    assert builder.reusable_api_translations({key: source}, "pt") == {}

    payload["symbols"][key]["translation_source_blocks_sha256"] = (
        builder._translation_source_block_hashes(source)
    )
    (tmp_path / "pt.json").write_text(json.dumps(payload), encoding="utf-8")
    assert builder.reusable_api_translations({key: source}, "pt") == {
        key: "Texto traduzido.",
    }


def test_normal_api_generation_translates_current_context_before_hashing(
    tmp_path, monkeypatch,
):
    import build_documentation_i18n as builder

    source = "The raw source sentence."
    contextual = "The reviewed contextual sentence."
    monkeypatch.setitem(builder.API_TRANSLATION_CONTEXT, source, contextual)
    captured = {}

    def fake_translate(blocks, language, model_root, args, **kwargs):
        captured["blocks"] = list(blocks)
        captured["namespace"] = kwargs.get("cache_namespace")
        captured["validator"] = kwargs.get("candidate_validator")
        return {contextual: "A frase contextual traduzida."}

    monkeypatch.setattr(builder, "_translate_blocks", fake_translate)
    translated = builder._translate_api_documents(
        {"spacr.example": source}, "pt", tmp_path, object(),
    )
    assert captured["blocks"] == [contextual]
    assert captured["namespace"] == builder.API_BLOCK_CACHE_NAMESPACE
    assert captured["validator"](
        contextual, "A frase contextual traduzida.", "pt"
    )
    assert not captured["validator"](
        contextual, "The reviewed contextual sentence.", "pt"
    )
    assert translated == {"spacr.example": "A frase contextual traduzida."}


def test_api_repair_reuses_legacy_cache_only_for_identical_model_input(
    tmp_path, monkeypatch,
):
    import argparse
    import build_documentation_i18n as builder

    source = "Return the task status."
    translated = "Retorna o estado da tarefa."
    model_root = tmp_path / "models"
    cache_dir = model_root / ".spacr_translation_cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / "pt.json").write_text(
        json.dumps({source: translated}), encoding="utf-8",
    )
    api_dir = tmp_path / "api"
    api_dir.mkdir()
    monkeypatch.setattr(builder, "API_DIR", api_dir)
    def unexpected_translate(*_args, **_kwargs):
        raise AssertionError("identical-input legacy cache should avoid decoding")

    monkeypatch.setattr(builder, "_translate_blocks", unexpected_translate)
    repaired = builder.repair_api_translations(
        {"spacr.example": source},
        "pt",
        model_root,
        argparse.Namespace(),
    )
    assert repaired == {"spacr.example": translated}

    contextual = "Return the software task status."
    monkeypatch.setitem(builder.API_TRANSLATION_CONTEXT, source, contextual)
    (cache_dir / "pt.json").write_text(
        json.dumps({source: translated}), encoding="utf-8",
    )
    monkeypatch.setattr(
        builder,
        "_translate_blocks",
        lambda *_args, **_kwargs: {contextual: contextual},
    )
    repaired = builder.repair_api_translations(
        {"spacr.example": source},
        "pt",
        model_root,
        argparse.Namespace(),
    )
    assert repaired == {"spacr.example": source}


def test_reviewed_api_blocks_are_exact_bound_accepted_only_evidence(
    tmp_path, monkeypatch,
):
    import argparse
    import hashlib
    import build_documentation_i18n as builder

    source = "Return the processing session status."
    context = builder._api_translation_source(source)
    target = "Retorna o estado da sessão de processamento."
    docs = {"spacr.example": source}
    reviewed = tmp_path / "reviewed"
    pt = reviewed / "pt"
    pt.mkdir(parents=True)
    evidence = {
        "schema": 1,
        "language": "pt",
        "records": [{
            "label": "spacr.example#0",
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "source": source,
            "context": context,
            "translation": target,
        }],
    }
    evidence_path = pt / "tail.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    monkeypatch.setattr(builder, "REVIEWED_API_DIR", reviewed)
    api_dir = tmp_path / "api"
    api_dir.mkdir()
    monkeypatch.setattr(builder, "API_DIR", api_dir)

    def unexpected_translate(*_args, **_kwargs):
        raise AssertionError("accepted review must avoid model decoding")

    monkeypatch.setattr(builder, "_translate_blocks", unexpected_translate)
    repaired = builder.repair_api_translations(
        docs, "pt", tmp_path / "models", argparse.Namespace(),
    )
    assert repaired == {"spacr.example": target}

    evidence["records"][0]["context"] = context + " changed"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    with pytest.raises(ValueError, match="stale reviewed API context"):
        builder.reviewed_api_block_translations(docs, "pt")


def test_api_translation_source_disambiguates_model_input_and_hashes_it():
    import build_documentation_i18n as builder

    cases = {
        "Read the image plane.": "Read the image layer.",
        "Resume the failed pipeline run.":
            "Resume the failed workflow processing session.",
        "Run this on a GUI worker thread.":
            "Execute this on a main GUI execution path.",
        "Load image crops.": "Load extracted image regions.",
        "Raises ValueError for invalid input.":
            "Throws ValueError for invalid input.",
        "Read the 384-well plate.":
            "Read the 384-position laboratory microplate.",
        "Return each well in the plate.":
            "Return each microplate sample position in the laboratory microplate.",
        "Return the mapping keys.":
            "Return the structured-data names.",
        "A pooled CRISPR screen reports hits.":
            "A pooled CRISPR screening experiment reports hits.",
        "The Qt screen shows settings.":
            "The Qt application view shows settings.",
        "Return a dictionary of settings.":
            "Return a key-value mapping of settings.",
        "Run the preprocessing pipeline.":
            "Execute the preprocessing workflow.",
        "Objects inside the polygon gate are selected.":
            "Objects inside the polygon data-selection boundary are selected.",
        "Estimate statistical power from sample size.":
            "Estimate statistical detection sensitivity from sample size.",
        "Append the job to the queue.":
            "Append the job to the software job list.",
        "Read each image field and channel.":
            "Read each microscope image field of view and image data channel.",
        "The classifier predicts each crop.":
            "The machine-learning classification model predicts each extracted image region.",
        "Return a human-readable reference.":
            "Return an easy-to-read reference.",
    }
    for source, expected in cases.items():
        assert builder._api_translation_source(source) == expected
        assert builder._translation_source_block_hashes(source) == [
            builder._source_hash(expected)
        ]


def test_api_translation_source_keeps_negative_control_senses_and_literals():
    import build_documentation_i18n as builder

    unchanged = (
        "The GUI screen compares a pooled CRISPR screen.",
        "Press the keyboard key.",
        "Use a power-law exponent.",
        "Raise the threshold by 0.1.",
        "A forum discussion thread.",
        "The aircraft has two planes.",
        "The tracked trajectory runs across frames.",
        "Serve food on the plate.",
        "Store the API key in the credential vault.",
        "Use ``pipeline`` as the code key.",
        "People formed a queue outside the application.",
        "Show the rear of the queue.",
        "Show the back of the queue.",
        "The queue held waiting customers.",
        "The farmer grows a crop in the field.",
        "The plane landed at the airport.",
        "Plot the points on a Cartesian plane.",
        "The carpenter used a flat plane as a hand tool.",
    )
    for source in unchanged:
        assert builder._api_translation_source(source) == source


def test_api_translation_source_preserves_fixed_exception_and_run_grammar():
    import build_documentation_i18n as builder

    cases = {
        "Raise :class:`PipelineCancelled` when cancellation was requested.":
            "Throw :class:`PipelineCancelled` when cancellation was requested.",
        "These are run-journal runs for completed jobs.":
            "These are processing-session journal folders for completed jobs.",
        "Re-run the failed workflow.":
            "Execute the failed workflow again.",
        "Re-run it after the failed job.":
            "Execute it again after the failed job.",
        "The worker recorded 3 crashes in 8 runs.":
            "The worker recorded 3 crashes in 8 executions.",
    }
    for source, expected in cases.items():
        assert builder._api_translation_source(source) == expected


def test_api_translation_source_preserves_fixed_thread_and_crop_morphology():
    import build_documentation_i18n as builder

    cases = {
        "Thread pools use thread counts for worker processes.":
            "Worker pools use worker-count limits for worker processes.",
        "A thread pool uses a worker-thread count.":
            "A worker pool uses a background-worker count.",
        "A crop-format file and a crop PNG support object-crop previews.":
            "An image-region-format file and an extracted-image-region PNG "
            "support object-image-region previews.",
        "Measure-and-crop creates a crop-and-measure workflow.":
            "Measurement-and-image-region-extraction creates an image-region "
            "extraction-and-measurement operation workflow.",
        "Re-crop the loaded array.":
            "Extract image regions from the loaded array again.",
        "A re-crop is requested.":
            "Extracting image regions again is requested.",
    }
    for source, expected in cases.items():
        assert builder._api_translation_source(source) == expected


def test_api_translation_source_preserves_finite_mapping_key_verbs():
    import build_documentation_i18n as builder

    cases = {
        "The cache keys its entries off this mapping key.":
            "The cache indexes its entries using this structured-data name.",
        "The model keys on a configuration key.":
            "The model is indexed by a configuration field name.",
        "spaCR keys objects by the mapping key.":
            "spaCR identifies objects by the structured-data name.",
        "The table keys each row by its mapping key.":
            "The table identifies each row by its structured-data name.",
        "Database measurement tables key on the object key.":
            "Database measurement tables are indexed by the object identifier.",
    }
    for source, expected in cases.items():
        assert builder._api_translation_source(source) == expected


def test_api_translation_source_preserves_plane_and_queue_grammar():
    import build_documentation_i18n as builder

    cases = {
        "Process one row per plane.":
            "Process one row for each image layer.",
        "Per-plane labels are stored.":
            "Per-image-layer labels are stored.",
        "Read a one-plane list.":
            "Read a single-image-layer list.",
        "Use 3-plane input.":
            "Use 3-image-layer input.",
        "The worker uses a queue- based scheduler.":
            "The worker uses a work-list-based scheduler.",
        "The scheduler queues jobs.":
            "The scheduler schedules jobs.",
        "A screen reads a queue of crops for annotation.":
            "An application view reads an annotation work list of extracted "
            "image regions for annotation.",
        "The table keys a crop by path.":
            "The table keys an extracted image region by path.",
        "The plate queue appears below the title.":
            "The plate-processing list appears below the title.",
        "One step is complete. A queue file for jobs is read next.":
            "One step is complete. A software job list file for jobs is read next.",
        "The callback completes. A screen receives the result.":
            "The callback completes. An application view receives the result.",
        "Return a wells-by-genes fraction matrix.":
            "Return a microplate-sample-position-by-gene fraction matrix.",
        "Read well names for a Plate and write from a WORKER thread.":
            "Read microplate sample position names for a laboratory microplate "
            "and write from a background execution unit.",
        "The user gates on a table; there is no button to gate.":
            "The user selects rows using a table; there is no button to filter "
            "data.",
        # Human waiting lines are an explicit negative control even when the
        # same sentence also contains a software-worker cue.
        "Workers watched people wait in a physical queue outside.":
            "Workers watched people wait in a physical queue outside.",
    }
    for source, expected in cases.items():
        assert builder._api_translation_source(source) == expected


def test_api_translation_source_preserves_reviewed_corpus_grammar():
    import build_documentation_i18n as builder

    cases = {
        "The queue is diversified by uncertainty for the annotator.":
            "The annotation work list is diversified by uncertainty for the "
            "annotator.",
        "DataLoader pre-fetches batches into a queue.":
            "DataLoader pre-fetches batches into a batch-data buffer.",
        "Consumes batches from a Queue in coalesced transactions.":
            "Consumes batches from a batch-data buffer in coalesced transactions.",
        "Optional queue for errors. A private Queue is created if None.":
            "Optional work list for errors. A private error-message list is "
            "created if None.",
        "Mirrors intensity channels first, then the cell mask.":
            "Mirrors intensity image data channels first, then the cell mask.",
        "A frame is channel-last, but a TIFF page is often channel-first; "
        "guessing turns a 3-channel image into noise.":
            "A frame stores image data channels last, but a TIFF page often "
            "stores image data channels first; guessing turns an image with 3 "
            "data channels into noise.",
        "A one-column-per-key-column frame for ``labels``, in their order.":
            "A frame with one output column for each identifier column in "
            "``labels``, preserving their order.",
        "*Intermediate paths are threaded, not repeated.*":
            "*Intermediate paths are linked together, not repeated.*",
        "4. Done. 5. **A running job is shown here.**":
            "4. Done. 5. **An executing job is shown here.**",
        "The error is handled. Pipelines pass False to the Qt screen.":
            "The error is handled. Workflows pass False to the Qt application view.",
        "Rows are loaded. Wells are matched to the plate.":
            "Rows are loaded. Microplate sample positions are matched to the "
            "laboratory microplate.",
        "The error is handled. Qt screens catch it at the gate.":
            "The error is handled. Qt application views catch it at the "
            "data-selection boundary.",
        "``diameter`` is stored. Human-readable labels explain it.":
            "``diameter`` is stored. Easy-to-read labels explain it.",
        "Without keys, the mapping has no lookup identifiers.":
            "Without identifiers, the mapping has no lookup identifiers.",
        "Worker-thread safe callbacks return to the GUI.":
            "Safe in a background execution unit callbacks return to the GUI.",
        "Off-thread notices return to the GUI thread.":
            "Off-execution-path notices return to the main GUI execution path.",
        "Mapping keys identify dictionary values.":
            "Structured-data names identify key-value mapping values.",
    }
    for source, expected in cases.items():
        assert builder._api_translation_source(source) == expected


def test_madlad7b_cache_namespace_cannot_reuse_older_source_fallback():
    import build_documentation_i18n as builder

    assert builder.API_BLOCK_CACHE_NAMESPACE == "api-block-v8-madlad7b"
    old_key = "api-block-v6\0Read the image plane."
    new_context = builder._api_translation_source("Read the image plane.")
    new_key = f"{builder.API_BLOCK_CACHE_NAMESPACE}\0{new_context}"
    assert old_key != new_key


def test_model_safe_split_preserves_complete_long_api_blocks():
    import build_documentation_i18n as builder

    docs = builder.public_docstrings()
    labels = (
        "spacr.power_simulate", "spacr.diameter",
        "spacr.qt.app.MainWindow.pin_all_menu_roles",
        "spacr.active_learning.crops_for_object_keys",
    )
    found_long = False
    for label in labels:
        blocks, _layout = builder.translatable_blocks(docs[label])
        for block in blocks:
            contextual = builder._api_translation_source(block)
            pieces = builder._split_model_safe(contextual)
            assert " ".join(pieces) == re.sub(r"\s+", " ", contextual).strip()
            assert all(len(piece) <= 760 for piece in pieces)
            found_long |= len(pieces) > 1
    assert found_long


def test_every_api_translation_context_preserves_the_literal_contract():
    import build_documentation_i18n as builder

    for key, document in builder.public_docstrings().items():
        for index, block in enumerate(builder.translatable_blocks(document)[0]):
            contextual = builder._api_translation_source(block)
            assert builder._syntax_preserved(block, contextual), (key, index)


@pytest.mark.parametrize(
    ("source_fragment", "required_fragments"),
    (
        (
            "Feature detector for keypoint matching",
            (
                "after detection",
                "using the detector’s ranking",
                "feature detection and scoring",
                "in downsampled space",
            ),
        ),
        (
            "A replicate whose fit failed or did not converge",
            (
                "mean over converged fits",
                "failed or did not converge",
                "five non-converged fits",
            ),
        ),
        (
            "get_db_browser_editable` must be on",
            (
                "off by default and is not on this screen",
                "selection alone does nothing",
                "Otherwise refuse the edit",
            ),
        ),
        (
            "structural amplitude of the flattened plane",
            (
                "structural amplitude (p99 - p30)",
                "raw-plane noise estimate",
                "unfilled foreground",
                "coarse pass sets the suppression radius",
            ),
        ),
    ),
)
def test_hard_tail_contexts_retain_the_complete_semantic_contract(
    source_fragment, required_fragments,
):
    import build_documentation_i18n as builder

    matches = [
        target
        for source, target in builder.API_TRANSLATION_CONTEXT.items()
        if source_fragment in source
    ]
    assert len(matches) == 1
    assert all(fragment in matches[0] for fragment in required_fragments)


def test_docstring_reflow_preserves_rst_roles_fields_and_inline_code():
    from build_documentation_i18n import rebuild_document, translatable_blocks

    source = """Overview with :mod:`spacr.core` and ``an_inline_value``.

:func:`spacr.example` writes ``first, second,
third`` without changing the literal.

:param source: Input path.
:returns: A :class:`Result`.
"""
    blocks, layout = translatable_blocks(source)
    rebuilt = rebuild_document(layout, blocks)
    for value in (
        ":mod:`spacr.core`",
        ":func:`spacr.example`",
        "``an_inline_value``",
        "``first, second, third``",
        ":param source:",
        ":class:`Result`",
    ):
        assert value in rebuilt


def test_literal_block_double_colon_is_structural_chrome_for_all_shapes():
    from build_documentation_i18n import rebuild_document, translatable_blocks

    source = """Typical use::

    call()

* Bullet introduction::

      item()

First::

    one()

Second::

    two()

Third::

    three()

Fourth::

    four()"""
    blocks, layout = translatable_blocks(source)
    assert all(not block.endswith("::") for block in blocks)
    assert [block for block in blocks if "Typical" in block] == ["Typical use"]
    translated = [f"번역 {index}" for index, _block in enumerate(blocks)]
    rebuilt = rebuild_document(layout, translated)
    assert rebuilt.count("::") == source.count("::") == 6
    assert len(translatable_blocks(rebuilt)[0]) == len(blocks)
    assert "    call()" in rebuilt and "      item()" in rebuilt


def test_standalone_literal_intro_is_raw_not_an_empty_translation_block():
    from build_documentation_i18n import rebuild_document, translatable_blocks

    source = """Example

::

    literal()"""
    blocks, layout = translatable_blocks(source)
    assert blocks == ["Example"]
    assert all(block for block in blocks)
    assert rebuild_document(layout, blocks) == source


@pytest.mark.parametrize(
    ("source", "translated"),
    (
        ("``setting`` explains the choice.", "``setting``  explica a escolha."),
        (
            ":class:`spacr.Widget` explains the choice.",
            ":class:`spacr.Widget`   explica a escolha.",
        ),
    ),
)
def test_leading_literal_model_whitespace_cannot_reparse_as_a_definition(
    source, translated,
):
    from build_documentation_i18n import rebuild_document, translatable_blocks

    blocks, layout = translatable_blocks(source)
    rebuilt = rebuild_document(layout, [translated])
    assert "  " not in rebuilt.split(" explica", 1)[0]
    assert translatable_blocks(rebuilt)[0] == [
        re.sub(r"\s+(?=explica)", " ", translated)
    ]
    assert len(translatable_blocks(rebuilt)[0]) == len(blocks) == 1


def test_unlabelled_indented_diagram_after_prose_stays_raw():
    from build_documentation_i18n import rebuild_document, translatable_blocks

    source = """Structure:
    ┌────────┐
    │ panel  │
    └────────┘"""
    blocks, layout = translatable_blocks(source)
    assert blocks == ["Structure:"]
    assert rebuild_document(layout, ["Estrutura:"]) == (
        "Estrutura:\n    ┌────────┐\n    │ panel  │\n    └────────┘"
    )


def test_simple_table_merges_all_cells_on_a_wrapped_continuation_row():
    from build_documentation_i18n import rebuild_document, translatable_blocks

    source = """============  ============  ============
question      Mask          Timelapse
============  ============  ============
busy text     \"Preview      \"Preview
              already       already
              running.\"     running.\"
============  ============  ============"""
    blocks, layout = translatable_blocks(source)
    assert '"Preview already running."' in blocks
    assert blocks.count('"Preview already running."') == 2
    rebuilt = rebuild_document(layout, blocks)
    assert translatable_blocks(rebuilt)[0] == blocks


def test_indented_pip_install_command_is_literal_not_prose():
    from build_documentation_i18n import rebuild_document, translatable_blocks

    source = '''Install it with:

    python -m pip install "spacr[anndata]"'''
    blocks, layout = translatable_blocks(source)
    assert blocks == ["Install it with:"]
    assert rebuild_document(layout, ["Instale com:"]) == source.replace(
        "Install it with:", "Instale com:"
    )


def test_unicode_mapping_arrows_are_protected_structural_operators():
    from build_i18n_catalogs import _protect, _restore, _syntax_preserved

    source = "``setting`` → κ = 2 × value; score ≥ 0.8 ± 0.1 ≈ result"
    protected, mapping = _protect(source)
    for operator in ("→", "κ", "×", "≥", "±", "≈"):
        assert operator not in protected
    assert _restore(protected, mapping) == source
    target = "``setting`` → κ = 2 × valor; pontuação ≥ 0.8 ± 0.1 ≈ resultado"
    assert _syntax_preserved(source, target)
    assert not _syntax_preserved(source, target.replace(" →", ""))
    assert not _syntax_preserved(source, target.replace("κ", "k"))


def test_ui_navigation_glyphs_are_protected_literals():
    from build_i18n_catalogs import _protect, _restore, _syntax_preserved

    source = "Open ⓘ, choose Facet ↓, then use ◀ / ▶ or spaCR ▸ Settings."
    protected, mapping = _protect(source)
    for glyph in ("ⓘ", "↓", "◀", "▶", "▸"):
        assert glyph not in protected
    assert _restore(protected, mapping) == source
    assert not _syntax_preserved(source, source.replace("ⓘ", "i"))


def test_scientific_units_and_variables_cannot_silently_disappear():
    from build_i18n_catalogs import _protect, _restore, _syntax_preserved

    source = "At 5 µm and 2 µm², κ uses pₒ and pₑ; see §6 © 2025."
    protected, mapping = _protect(source)
    for literal in ("µm", "µm²", "κ", "pₒ", "pₑ", "§6", "©"):
        assert literal not in protected
    assert _restore(protected, mapping) == source
    target = "A 5 µm e 2 µm², κ usa pₒ e pₑ; consulte §6 © 2025."
    assert _syntax_preserved(source, target)
    assert not _syntax_preserved(source, target.replace("5 µm", "5"))
    assert not _syntax_preserved(source, target.replace("pₑ", "pe"))


def test_unquoted_identifiers_inside_prose_table_cells_stay_exact():
    from build_i18n_catalogs import _protect, _restore, _syntax_preserved

    source = "No — deleteLater; then preview_ ready."
    protected, mapping = _protect(source)
    for literal in ("deleteLater", "preview_"):
        assert literal not in protected
    assert _restore(protected, mapping) == source
    assert not _syntax_preserved(
        source, source.replace("deleteLater", "excluirDepois")
    )


def test_preview_contract_table_reassembles_prose_but_hides_code_cells():
    from build_documentation_i18n import (
        _api_block_requires_translation,
        public_docstrings,
        translatable_blocks,
    )

    blocks, _layout = translatable_blocks(
        public_docstrings()["spacr.qt.widgets.preview_contract"]
    )
    assert blocks.count('"Preview already running."') == 3
    assert blocks.count("no — deleteLater") == 2
    assert blocks.count("preview_ ready") == 3
    assert "JobRunner" not in blocks
    assert "a rescore from the cache" in blocks
    assert "matplotlib plot" in blocks
    assert _api_block_requires_translation("no")
    assert _api_block_requires_translation("no — deleteLater")


def test_preview_contract_canonical_layout_preserves_document_literals():
    import build_documentation_i18n as builder
    from build_i18n_catalogs import _syntax_preserved

    source = builder.public_docstrings()["spacr.qt.widgets.preview_contract"]
    blocks, layout = builder.translatable_blocks(source)
    canonical = builder.rebuild_document(layout, blocks)
    assert canonical != source
    assert _syntax_preserved(canonical, canonical, check_emphasis=False)
    assert '"Preview already running."' in canonical
    assert '"Preview      ' not in canonical


def test_canonical_literal_introducers_never_reach_model_blocks():
    from build_documentation_i18n import public_docstrings, translatable_blocks

    docs = public_docstrings()
    for key in (
        "spacr.align", "spacr.convert", "spacr.custom_features",
        "spacr.qt.synthetic", "spacr.runctx.random_state",
        "spacr.resources.home.versions._generators.render",
    ):
        blocks, _layout = translatable_blocks(docs[key])
        assert not any(block.rstrip().endswith("::") for block in blocks), key


def test_parser_translates_indented_prose_but_preserves_code_and_diagrams():
    from build_documentation_i18n import rebuild_document, translatable_blocks

    source = """Result groups.

:returns: ``(high, low)``.

    ``high`` is the confident group.

    ``low`` is the boundary group.

Example::

    value = compute()
    print(value)

Diagram

    A --> B
    B --> C"""
    blocks, layout = translatable_blocks(source)
    assert "``high`` is the confident group." in blocks
    assert "``low`` is the boundary group." in blocks
    assert not any("value = compute()" in block for block in blocks)
    assert not any("A --> B" in block for block in blocks)
    assert rebuild_document(layout, blocks) == source


def test_canonical_indented_literal_shapes_are_never_translation_blocks():
    from build_documentation_i18n import public_docstrings, translatable_blocks

    docs = public_docstrings()
    forbidden = {
        "spacr.classify_classes": '{"infected":',
        "spacr.mask_io": 'np.save("foo_mask.npy"',
        "spacr._v1_v2_bridge.report_disk_savings": "v1 ≈ 4 × merged",
        "spacr.pipeline_v2": "→ renamed + split into channel folders",
        "spacr.qt": "python -m spacr.qt",
        "spacr.qt.verbose_logger.log_call": "[class.func] args=",
        "spacr.qt.widgets": "670 ms  spacr.qt.app",
        "spacr.power_simulate": "Permission is hereby granted",
    }
    for key, fragment in forbidden.items():
        blocks, _layout = translatable_blocks(docs[key])
        assert not any(fragment in block for block in blocks), (key, fragment)

    stitcher_blocks, _layout = translatable_blocks(
        docs["spacr.spacrops.spacrStitcher"]
    )
    assert any(
        "if True  → use RANSAC affine" in block
        and "if False → translation-only" in block
        for block in stitcher_blocks
    )


def test_code_definition_shape_inside_explicit_literal_block_stays_exact():
    from build_documentation_i18n import rebuild_document, translatable_blocks

    source = """Example::

    ``name``  exact output
    ``kind``  another literal"""
    blocks, layout = translatable_blocks(source)
    # The literal-block introducer is structural RST chrome.  Translate only
    # its prose label, then reattach ``::`` during reconstruction.
    assert blocks == ["Example"]
    assert rebuild_document(layout, blocks) == source


def test_indented_query_and_shortcut_keys_stay_literal_while_help_translates():
    from build_documentation_i18n import public_docstrings, translatable_blocks

    docs = public_docstrings()
    query_blocks, _layout = translatable_blocks(
        docs["spacr.qt.annotate_engine.parse_image_type"]
    )
    assert any("contains both" in block for block in query_blocks)
    assert not any("cell AND nucleus" in block for block in query_blocks)

    shortcut_blocks, _layout = translatable_blocks(docs["spacr.qt.shortcuts"])
    assert any("Go home" in block for block in shortcut_blocks)
    assert not any("Ctrl+H" in block or "F1  / ?" in block
                   for block in shortcut_blocks)


def test_parser_preserves_directive_options_and_translates_admonition_title():
    from build_documentation_i18n import rebuild_document, translatable_blocks

    source = """.. admonition:: Important result
   :class: warning

   Translate this body with :exc:`ValueError`."""
    blocks, layout = translatable_blocks(source)
    assert blocks == [
        "Important result",
        "Translate this body with :exc:`ValueError`.",
    ]
    assert ":class: warning" not in blocks
    translated = ["Resultado importante", "Traduza este corpo com :exc:`ValueError`."]
    rebuilt = rebuild_document(layout, translated)
    assert rebuilt.startswith(".. admonition:: Resultado importante\n")
    assert "   :class: warning" in rebuilt
    assert ":exc:`ValueError`" in rebuilt


def test_parser_translates_yields_and_seealso_fields():
    from build_documentation_i18n import rebuild_document, translatable_blocks

    source = (
        ":yields: one completed result at a time.\n"
        ":seealso: :py:meth:`spacr.Widget.run` for execution details."
    )
    blocks, layout = translatable_blocks(source)
    assert blocks == [
        "one completed result at a time.",
        ":py:meth:`spacr.Widget.run` for execution details.",
    ]
    assert rebuild_document(layout, blocks) == source


def test_canonical_indented_api_explanations_are_translation_blocks():
    from build_documentation_i18n import public_docstrings, translatable_blocks

    docs = public_docstrings()
    expected_fragments = {
        "spacr.confusion.split_by_confidence":
            "``high`` is confidence ``>= threshold``",
        "spacr.measure_hooks":
            "``channel_arrays`` is exactly the array",
        "spacr.model_compare":
            "one model's settings, plus what of it survives.",
        "spacr.power_model.scan_parameters":
            "**Returning exactly ``False`` stops the sweep**",
    }
    for key, fragment in expected_fragments.items():
        blocks, _layout = translatable_blocks(docs[key])
        assert any(fragment in block for block in blocks), (key, fragment)


def test_all_rst_role_prefixes_and_payloads_are_protected():
    from build_i18n_catalogs import _syntax_preserved

    source = (
        "Use :exc:`ValueError`, :math:`x > 1`, and "
        ":py:meth:`spacr.Widget.run`."
    )
    valid = (
        "Use :exc:`ValueError`, :math:`x > 1`, e "
        ":py:meth:`spacr.Widget.run`."
    )
    assert _syntax_preserved(source, valid)
    assert not _syntax_preserved(
        source, valid.replace(":exc:", ":class:", 1)
    )
    assert not _syntax_preserved(
        source, valid.replace("spacr.Widget.run", "spacr.Widget.stop", 1)
    )


def test_runtime_syntax_gate_accepts_only_exact_reviewed_acronym_normalization():
    from build_i18n_catalogs import (
        MANUAL_UI,
        _syntax_preserved,
        _syntax_preserved_or_reviewed,
    )

    assert not _syntax_preserved("pca", "PCA")
    assert _syntax_preserved_or_reviewed("pca", MANUAL_UI["pca"]["de"], "de")
    assert not _syntax_preserved_or_reviewed("pca", "PCA changed", "de")
    assert _syntax_preserved("Export CSVs and PNGs from UMAPs.", "Export CSV-Dateien, PNG-Dateien und UMAPs.")


def test_reviewed_quoted_human_phrases_are_prose_not_api_literals():
    from build_i18n_catalogs import _syntax_preserved

    assert _syntax_preserved(
        "Mark objects as 'not scored' or 'exclude this debris'.",
        "Marque os objetos como 'não pontuados' ou 'exclua estes detritos'.",
    )
    assert _syntax_preserved(
        'Compare "the run that worked" with the failed run.',
        'Compare "a execução que funcionou" com a execução que falhou.',
    )


def test_inline_pip_install_command_is_a_protected_literal():
    from build_i18n_catalogs import _PROTECT_RE, _syntax_preserved

    source = "Uses the Piper CLI installed with pip install piper-tts."
    matches = [match.group(0) for match in _PROTECT_RE.finditer(source)]
    assert "pip install piper-tts" in matches
    assert _syntax_preserved(
        source,
        "Usa a CLI do Piper instalada com pip install piper-tts.",
    )


def test_readme_language_picker_is_never_sent_through_translation():
    from build_documentation_i18n import rebuild_document, translatable_blocks

    picker = (
        "Languages: `English <README.rst>`_ · "
        "`Svenska <docs/i18n/readme/README.sv.rst>`_ ·\n"
        "`简体中文 <docs/i18n/readme/README.zh_CN.rst>`_\n\n"
        "Translate this sentence."
    )
    blocks, layout = translatable_blocks(picker)
    assert blocks == ["Translate this sentence."]
    assert rebuild_document(layout, blocks).startswith(
        "Languages: `English <README.rst>`_ · "
        "`Svenska <docs/i18n/readme/README.sv.rst>`_ ·\n"
        "`简体中文 <docs/i18n/readme/README.zh_CN.rst>`_"
    )


def test_github_summary_has_reviewed_domain_translations():
    from build_documentation_i18n import (
        REVIEWED_README_BLOCKS,
        REVIEWED_README_HEADINGS,
    )

    assert len(REVIEWED_README_BLOCKS) == 7
    for reviewed in REVIEWED_README_BLOCKS.values():
        assert set(reviewed) == {
            "sv", "de", "es", "zh_CN", "pt", "hi", "ko", "is", "fr",
        }
    joined = {
        language: " ".join(block[language] for block in REVIEWED_README_BLOCKS.values())
        for language in next(iter(REVIEWED_README_BLOCKS.values()))
    }
    assert "CRISPR 筛选" in joined["zh_CN"]
    assert "criblages CRISPR" in joined["fr"]
    assert "CRISPR 스크리닝" in joined["ko"]
    assert "CRISPR-skim" in joined["is"]
    assert len(REVIEWED_README_HEADINGS) == 22
    for reviewed in REVIEWED_README_HEADINGS.values():
        assert set(reviewed) == {
            "sv", "de", "es", "zh_CN", "pt", "hi", "ko", "is", "fr",
        }
    assert REVIEWED_README_HEADINGS["New in 1.5.0.0"]["hi"].endswith(
        "1.5.0.0 में नया"
    )


def test_translation_protection_has_no_nested_tokens_and_round_trips():
    from build_i18n_catalogs import _protect, _restore

    source = (
        "Each entry is {'name': <class>, 'where': [{'column': <feature>, "
        "'op': '>='}], :func:`spacr.core.run` keeps CUDA unchanged, and "
        "`the release <https://example.test/release>`_ remains a valid link."
    )
    protected, mapping = _protect(source)
    assert all("<x" not in value for value in mapping.values())
    assert _restore(protected, mapping) == source


def test_numeric_protection_markers_restore_when_models_join_target_text():
    """Joined Latin text and Korean particles must not look like lost tokens.

    Marian commonly removes whitespace around numeric fallback markers.  A
    Unicode ``\b`` does not exist between the marker's final digit and either
    a Latin letter or a Hangul particle, so the old restore path discarded a
    complete translation even though every marker was still present once.
    """
    from build_i18n_catalogs import _restore

    mapping = {"0X0": "**", "1X1": "``measurements.db``"}
    assert _restore(
        "0X0A tradução termina em 1X1.", mapping,
    ) == "**A tradução termina em ``measurements.db``."
    assert _restore(
        "0X0을 번역하고 1X1에서 읽습니다.", mapping,
    ) == "**을 번역하고 ``measurements.db``에서 읽습니다."


def test_numeric_protection_marker_does_not_match_inside_larger_number():
    from build_i18n_catalogs import _restore
    import pytest

    with pytest.raises(ValueError, match="did not preserve 0X0 exactly once"):
        _restore("prefix 10X01 suffix", {"0X0": "**"})


def test_xml_markers_restore_after_only_the_opening_angle_is_lost():
    from build_i18n_catalogs import _restore

    # Marian emits this exact shape for ``*including 0*``: the source zero
    # touches the shortened marker, while the expected id and closing bracket
    # remain unambiguous.
    assert _restore(
        "x0>Texto 0x1> final x2>",
        {"<x0>": "**", "<x1>": "*", "<x2>": "**"},
        protected_text="<x0>Texto 0<x1> final <x2>",
    ) == "**Texto 0* final **"


def test_xml_marker_fuzz_allows_target_order_but_not_ascii_word_suffixes():
    import pytest
    from build_i18n_catalogs import _restore

    assert _restore(
        "x1> texto x0>", {"<x0>": "``a``", "<x1>": "``b``"},
    ) == "``b`` texto ``a``"
    with pytest.raises(ValueError, match="exactly once"):
        _restore("matrix0>", {"<x0>": "``a``"})


def test_xml_marker_fuzz_consumes_the_retained_closing_angle():
    from build_i18n_catalogs import _restore

    assert _restore(
        "x0>Texto x1>", {"<x0>": "**", "<x1>": "**"},
    ) == "**Texto **"
    assert ">" not in _restore("x0>Texto", {"<x0>": "``value``"})


def test_marker_fusion_requires_the_exact_source_adjacency_contract():
    import pytest
    from build_i18n_catalogs import _restore

    mapping = {"<x1>": "``value``", "<x2>": "!"}
    assert _restore(
        "linear dex1> x2>",
        mapping,
        protected_text="linear de<x1> <x2>",
    ) == "linear de``value`` !"
    with pytest.raises(ValueError, match="did not preserve <x1>"):
        _restore(
            "linear dex1> x2>",
            mapping,
            protected_text="linear de <x1> <x2>",
        )

    assert _restore(
        "incluindo 0x6>",
        {"<x6>": "*"},
        protected_text="incluindo 0<x6>",
    ) == "incluindo 0*"
    with pytest.raises(ValueError, match="did not preserve <x6>"):
        _restore(
            "incluindo 0x6>",
            {"<x6>": "*"},
            protected_text="incluindo zero <x6>",
        )


def test_numeric_marker_digit_fusion_uses_the_exact_source_contract():
    import pytest
    from build_i18n_catalogs import _restore

    mapping = {"5X5": "**", "6X6": "*"}
    assert _restore(
        "5X5incluindo 06X6",
        mapping,
        protected_text="5X5including 06X6",
    ) == "**incluindo 0*"
    with pytest.raises(ValueError, match="did not preserve 6X6"):
        _restore("5X5incluindo 06X6", mapping)


def test_marker_restore_rejects_unknown_explicit_tokens():
    import pytest
    from build_i18n_catalogs import _restore

    with pytest.raises(ValueError, match="invented protection token"):
        _restore(
            "x0> texto x9>",
            {"<x0>": "``value``"},
            protected_text="<x0>",
        )
    assert _restore(
        "x0> texto x9",
        {"<x0>": "``value``"},
        protected_text="<x0>",
    ) == "``value`` texto x9"


def test_fully_stripped_xml_marker_is_never_guessed_from_natural_coordinates():
    import pytest
    from build_i18n_catalogs import _restore

    with pytest.raises(ValueError, match="did not preserve <x0>"):
        _restore(
            "valor em x0",
            {"<x0>": "**"},
            protected_text="<x0> value at x0",
        )
    assert _restore(
        "medido em x 0.825", {}, protected_text="measured at x 0.825"
    ) == "medido em x 0.825"


def test_marker_restore_checks_raw_output_before_inserting_literal_values():
    from build_i18n_catalogs import _restore

    assert _restore(
        "<x0>", {"<x0>": "<x9>"}, protected_text="<x0>"
    ) == "<x9>"
    assert _restore(
        "x0>", {"<x0>": "x9>"}, protected_text="<x0>"
    ) == "x9>"


def test_unclaimed_numeric_shapes_must_match_the_protected_source():
    import pytest
    from build_i18n_catalogs import _restore

    with pytest.raises(ValueError, match="unprotected numeric"):
        _restore(
            "0X0 plus 9X9",
            {"0X0": "**"},
            protected_text="0X0 plus",
        )
    assert _restore(
        "0X0 tamanho 9 x 9",
        {"0X0": "**"},
        protected_text="0X0 size 9X9",
    ) == "** tamanho 9 x 9"
    assert _restore(
        "imagem 300 x 300", {}, protected_text="image 300 x 300"
    ) == "imagem 300 x 300"


def test_protected_source_contract_must_contain_each_marker_once():
    import pytest
    from build_i18n_catalogs import _restore

    with pytest.raises(ValueError, match="protected input did not contain"):
        _restore("x0>", {"<x0>": "**"}, protected_text="plain text")
    with pytest.raises(ValueError, match="protected input did not contain"):
        _restore(
            "x0>", {"<x0>": "**"}, protected_text="<x0> then <x0>"
        )
    with pytest.raises(ValueError, match="exactly once"):
        _restore(
            "x0> texto x0>",
            {"<x0>": "``value``"},
            protected_text="<x0>",
        )
    with pytest.raises(ValueError, match="did not preserve <x1>"):
        _restore(
            "x0> texto",
            {"<x0>": "``a``", "<x1>": "``b``"},
            protected_text="<x0> then <x1>",
        )


def test_fragment_protection_keeps_short_quotes_as_one_literal_island():
    from build_i18n_catalogs import _FRAGMENT_PROTECT_RE, _protect

    short = 'means "not looked at"; continue.'
    protected, mapping = _protect(short, pattern=_FRAGMENT_PROTECT_RE)
    assert list(mapping.values()) == ['"not looked at"']
    assert "not looked at" not in protected

    long = 'means "this explanation has five translated words"; continue.'
    protected, mapping = _protect(long, pattern=_FRAGMENT_PROTECT_RE)
    assert list(mapping.values()) == ['"', '"']
    assert "this explanation has five translated words" in protected

    reviewed = 'means "not scored"; continue.'
    protected, mapping = _protect(reviewed, pattern=_FRAGMENT_PROTECT_RE)
    assert list(mapping.values()) == ['"', '"']
    assert "not scored" in protected


def test_context_clause_plan_preserves_exact_chrome_and_protected_literals():
    from build_i18n_catalogs import _context_clause_plan

    source = (
        "Returns: read ``mapping: value`` — keep the result, but retry "
        "because the file changed."
    )
    plan = _context_clause_plan(source)

    assert "".join(piece for piece, _translate in plan) == source
    assert [piece for piece, translate in plan if not translate] == [
        ": ", " — ", ", ", " ",
    ]
    assert any(
        "``mapping: value``" in piece and translate
        for piece, translate in plan
    )
    assert sum(translate for _piece, translate in plan) == 5


def test_context_clause_plan_requires_two_translatable_spans():
    from build_i18n_catalogs import _context_clause_plan

    assert _context_clause_plan("No strong boundary here.") == []
    assert _context_clause_plan("Type: ``dict[str, int]``") == []


def test_fragment_retry_requires_current_latest_mechanical_failure():
    from build_documentation_i18n import _api_block_valid
    from build_i18n_catalogs import (
        _fragment_retry_sources,
        _translation_candidate_valid,
    )

    rescued = "Read ``measurements.db``."
    caller_bad = "Return this value to the caller."
    syntax_bad = "Write ``results.db``."
    already_good = "Return the image crop."
    marker_bad = "Read ``other.db``."
    semantic_bad = "Return another image crop."
    script_bad = "Describe the image plane."
    exact_bad = "Describe the image tile."
    degenerate_bad = "Describe the pipeline."
    eos_bad = "Append work to the queue."
    translations = {
        rescued: "``measurements.db``에서 읽습니다.",
        caller_bad: "Return this 값을 to the caller.",
        syntax_bad: "결과 데이터베이스에 씁니다.",
        already_good: "이미지 크롭을 반환합니다.",
        marker_bad: marker_bad,
        semantic_bad: semantic_bad,
        script_bad: script_bad,
        exact_bad: exact_bad,
        degenerate_bad: degenerate_bad,
        eos_bad: eos_bad,
    }

    def valid(source, value):
        return (
            _translation_candidate_valid(source, value, "ko", force=True)
            and _api_block_valid(source, value, "ko")
        )

    # Pin the fixture: the caller-gate failure is structurally sound, while
    # the syntax failure dropped its protected database literal.
    assert _translation_candidate_valid(
        caller_bad, translations[caller_bad], "ko", force=True,
    )
    assert not _api_block_valid(caller_bad, translations[caller_bad], "ko")
    assert not _translation_candidate_valid(
        syntax_bad, translations[syntax_bad], "ko", force=True,
    )
    assert valid(rescued, translations[rescued])
    assert valid(already_good, translations[already_good])

    latest_failures = {
        rescued: {"marker_restore"},
        caller_bad: {"caller_gate"},
        syntax_bad: {"protected_syntax"},
        already_good: {"marker_restore"},
        marker_bad: {"marker_restore"},
        semantic_bad: {"semantic"},
        script_bad: {"target_script"},
        exact_bad: {"exact"},
        degenerate_bad: {"degenerate"},
        eos_bad: {"eos"},
    }
    selected = _fragment_retry_sources(
        translations, translations, valid, latest_failures,
    )
    assert selected == [syntax_bad, marker_bad]


def test_valid_sentence_retry_is_not_overwritten_by_fragment_fallback():
    from build_documentation_i18n import _api_block_valid
    from build_i18n_catalogs import (
        _fragment_retry_sources,
        _translation_candidate_valid,
    )

    rescued = "Read ``measurements.db``."
    still_bad = "Write ``results.db``."
    rescued_value = "``measurements.db``에서 읽습니다."
    translated = {
        rescued: rescued_value,
        still_bad: "결과 데이터베이스에 씁니다.",
    }

    def valid(source, value):
        return (
            _translation_candidate_valid(source, value, "ko", force=True)
            and _api_block_valid(source, value, "ko")
        )

    latest_failures = {
        rescued: {"marker_restore"},
        still_bad: {"protected_syntax"},
    }
    seen = _fragment_retry_sources(
        translated, translated, valid, latest_failures,
    )
    for source in seen:
        # Simulate a rejected fragment falling back to canonical English.
        translated[source] = source
    assert seen == [still_bad]
    assert translated[rescued] == rescued_value


def test_incomplete_fragment_output_can_never_be_reassembled():
    from build_i18n_catalogs import _join_completed_fragments

    pieces = ["번역된 앞부분 ", "", " ``literal`` 뒤부분"]
    assert _join_completed_fragments(pieces, {"eos"}) is None
    assert _join_completed_fragments(pieces, set()) == (
        "번역된 앞부분  ``literal`` 뒤부분"
    )


def test_ranked_generation_returns_one_candidate_per_existing_beam():
    from build_i18n_catalogs import (
        _group_ranked_outputs,
        _ranked_generation_kwargs,
    )

    assert _ranked_generation_kwargs(4) == {
        "num_beams": 4,
        "num_return_sequences": 4,
    }
    output = ["a0", "a1", "a2", "a3", "b0", "b1", "b2", "b3"]
    grouped = _group_ranked_outputs(output, list(output), 2, 4)
    assert [[value for _sequence, value in group] for group in grouped] == [
        ["a0", "a1", "a2", "a3"],
        ["b0", "b1", "b2", "b3"],
    ]


def test_ranked_candidate_skips_invalid_eos_restore_and_gate_results():
    from build_i18n_catalogs import _first_valid_ranked_candidate

    candidates = [
        ("incomplete", "ignored"),
        ("complete", "broken-marker"),
        ("complete", "gate-rejected"),
        ("complete", "accepted"),
    ]

    def restore(value):
        if value == "broken-marker":
            raise ValueError("marker was damaged")
        return value

    def evaluate(value):
        if value == "gate-rejected":
            return value, {"caller_gate"}
        return f"translated:{value}", set()

    selected, failures = _first_valid_ranked_candidate(
        candidates,
        completed=lambda sequence: sequence == "complete",
        restore=restore,
        evaluate=evaluate,
    )
    assert selected == "translated:accepted"
    assert failures == set()

    selected, failures = _first_valid_ranked_candidate(
        candidates[:-1],
        completed=lambda sequence: sequence == "complete",
        restore=restore,
        evaluate=evaluate,
    )
    assert selected is None
    assert failures == {"eos", "marker_restore", "caller_gate"}


def test_rank_zero_rejection_can_select_rank_one_without_mixing_pieces():
    from build_i18n_catalogs import (
        _first_valid_ranked_candidate,
        _rank_aligned_joins,
    )

    selected, failures = _first_valid_ranked_candidate(
        [(object(), "rank-zero"), (object(), "rank-one")],
        completed=lambda _sequence: True,
        restore=lambda value: value,
        evaluate=lambda value: (
            value,
            {"protected_syntax"} if value == "rank-zero" else set(),
        ),
    )
    assert selected == "rank-one"
    assert failures == set()

    assert _rank_aligned_joins(
        [["left-0", "left-1"], ["right-0", "right-1"]],
        lambda values: "|".join(values),
    ) == ["left-0|right-0", "left-1|right-1"]
    assert _rank_aligned_joins(
        [["left-0", None], [None, "right-1"]],
        lambda values: "|".join(values),
    ) == [None, None]


def test_no_valid_beam_falls_back_to_source_without_cache(
    tmp_path, monkeypatch,
):
    import types

    import torch
    import build_i18n_catalogs as builder

    source = "Translate this deliberately unique beam fixture."
    model_folder = tmp_path / builder.MODEL_SPECS["pt"][1]
    model_folder.mkdir(parents=True)
    generation_calls = []

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
            values = {
                11: "Primeira tradução rejeitada.",
                12: "Segunda tradução rejeitada.",
            }
            return [values[int(sequence[0])] for sequence in output]

    tokenizer = FakeTokenizer()

    class FakeModel:
        def eval(self):
            return self

        def generate(self, **kwargs):
            generation_calls.append(kwargs)
            return torch.tensor([[11, 2], [12, 2]], dtype=torch.long)

    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(
            from_pretrained=lambda *_args, **_kwargs: tokenizer,
        ),
        AutoModelForSeq2SeqLM=types.SimpleNamespace(
            from_pretrained=lambda *_args, **_kwargs: FakeModel(),
        ),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(builder, "_seed_cache_from_catalog", lambda *_args: None)

    translated = builder._translate_batches(
        [source],
        "pt",
        tmp_path,
        device="cpu",
        batch_size=1,
        beams=2,
        threads=1,
        force_sources={source},
        cache_namespace="beam-fixture",
        candidate_validator=lambda *_args: False,
    )

    assert translated == {source: source}
    assert len(generation_calls) == 1
    assert generation_calls[0]["num_beams"] == 2
    assert generation_calls[0]["num_return_sequences"] == 2
    cache_path = tmp_path / ".spacr_translation_cache" / "pt.json"
    assert not cache_path.exists()


def test_translation_rejection_reasons_separate_mechanical_and_linguistic():
    from build_i18n_catalogs import _translation_rejection_reasons

    protected_source = "Read ``measurements.db``."
    assert _translation_rejection_reasons(
        protected_source,
        "측정 데이터베이스를 읽습니다.",
        "ko",
        force=True,
    ) == {"protected_syntax"}

    crop_source = "Return the image crop."
    semantic = _translation_rejection_reasons(
        crop_source,
        "이미지 농작물을 반환합니다.",
        "ko",
        force=True,
    )
    assert "semantic" in semantic
    assert "protected_syntax" not in semantic

    caller_source = "Return this value to the caller."
    assert _translation_rejection_reasons(
        caller_source,
        "Return this 값을 to the caller.",
        "ko",
        force=True,
        candidate_validator=lambda _source, _value, _language: False,
    ) == {"caller_gate"}


def test_context_repairs_cannot_disguise_an_exact_english_fallback():
    from build_i18n_catalogs import _translation_rejection_reasons

    source = "Record the export. Default True."
    reasons = _translation_rejection_reasons(
        source, source, "pt", force=True,
    )

    assert "exact" in reasons


def test_reviewed_context_repair_clears_the_wrong_scientific_screen_sense():
    from build_i18n_catalogs import (
        _contextualize,
        _semantic_false_friends,
        _translation_rejection_reasons,
    )

    source = "Classify a pooled screen before regression."
    raw = "Classifique uma tela agrupada antes da regressão."
    corrected = _contextualize(raw, "pt", source)

    assert _semantic_false_friends(source, raw, "pt")
    assert corrected == "Classifique uma triagem agrupada antes da regressão."
    assert not _translation_rejection_reasons(
        source, corrected, "pt", force=True,
    )


def test_portuguese_exception_and_dictionary_senses_are_reviewed():
    from build_i18n_catalogs import _contextualize, _semantic_false_friends

    exception_source = "A typo changes the result rather than raising."
    exception = _contextualize(
        "Um erro muda o resultado em vez de aumentar.",
        "pt",
        exception_source,
    )
    assert exception.endswith("em vez de gerar um erro.")
    assert not _semantic_false_friends(exception_source, exception, "pt")

    dictionary_source = "The Classes dict names each class."
    dictionary = _contextualize(
        "O ditado Classes nomeia cada classe.", "pt", dictionary_source,
    )
    assert dictionary == "O dicionário Classes nomeia cada classe."
    assert not _semantic_false_friends(dictionary_source, dictionary, "pt")


def test_spanish_emphasis_and_atomic_write_residue_is_translated():
    from build_i18n_catalogs import _contextualize

    source = (
        "It does **not** expose the pre-write file; use temp-then-replace."
    )
    translated = _contextualize(
        "Esto *does* **not** expone el archivo pre-write; use temp-then.",
        "es",
        source,
    )

    assert translated == (
        "Esto *sí* **no** expone el archivo previa a la escritura; "
        "use temporal y luego."
    )


def test_literal_block_layout_owns_translated_heading_colons():
    from build_documentation_i18n import rebuild_document, translatable_blocks

    blocks, layout = translatable_blocks("Usage::\n\n    spacr-run --list")

    assert blocks == ["Usage"]
    assert rebuild_document(layout, ["Uso:"]) == (
        "Uso::\n\n    spacr-run --list"
    )


def test_rejected_models_use_the_reviewed_permissive_replacement():
    from build_i18n_catalogs import MODEL_SPECS

    for language in ("zh_CN", "hi", "ko", "is"):
        model, _folder, license_name, _prefix = MODEL_SPECS[language]
        assert model == "facebook/m2m100_418M"
        assert license_name == "MIT"


def test_secondary_model_is_permissive_and_publicly_attributed():
    from build_i18n_catalogs import SECONDARY_LICENSE, SECONDARY_MODEL

    attribution = (
        ROOT / "docs" / "i18n" / "TRANSLATION_MODELS.md"
    ).read_text(encoding="utf-8")
    assert SECONDARY_MODEL == "google/madlad400-7b-mt"
    assert SECONDARY_LICENSE == "Apache-2.0"
    assert SECONDARY_MODEL in attribution
    assert SECONDARY_LICENSE in attribution


def test_generation_loop_detection_rejects_repeated_labels():
    from build_i18n_catalogs import _looks_degenerate

    assert _looks_degenerate("Background", "背景" * 120, "zh_CN")
    assert _looks_degenerate("Run mode", "hamur hamur hamur hamur", "is")
    assert _looks_degenerate("A technical paragraph. " * 8, "traduction " * 80, "fr")
    assert not _looks_degenerate("Background", "背景", "zh_CN")


def test_translation_cache_checkpoints_merge_independent_lanes(tmp_path):
    from build_i18n_catalogs import _merge_write_translation_cache

    path = tmp_path / "pt.json"
    api_baseline: dict[str, str] = {}
    runtime_baseline: dict[str, str] = {}
    _merge_write_translation_cache(
        path, {"api-block-v6\0Source": "API"}, api_baseline,
    )
    _merge_write_translation_cache(
        path, {"Runtime source": "Runtime"}, runtime_baseline,
    )
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "Runtime source": "Runtime",
        "api-block-v6\0Source": "API",
    }
    assert not list(tmp_path.glob("*.tmp"))
    assert list(tmp_path.glob("*.lock"))


def test_translation_cache_atomically_removes_rejected_checkpoint(tmp_path):
    from build_i18n_catalogs import _merge_write_translation_cache

    path = tmp_path / "ko.json"
    path.write_text(json.dumps({"api-block-v7\0Source": "Source"}))
    baseline = {"api-block-v7\0Source": "Source"}
    _merge_write_translation_cache(path, {}, baseline)
    assert json.loads(path.read_text()) == {}
    assert baseline == {}


def test_cache_lock_serializes_contending_processes(tmp_path):
    """A waiter must never enter while the first process holds the inode."""
    path = tmp_path / "pt.json"
    worker = """
import sys
import time
sys.path.insert(0, {tools!r})
from pathlib import Path
from build_i18n_catalogs import _exclusive_cache_lock
with _exclusive_cache_lock(Path({path!r})):
    print('entered', flush=True)
    time.sleep(float(sys.argv[1]))
""".format(tools=str(TOOLS), path=str(path))
    first = subprocess.Popen(
        [sys.executable, "-c", worker, "0.5"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert first.stdout is not None
    assert first.stdout.readline().strip() == "entered"
    started = __import__("time").monotonic()
    second = subprocess.run(
        [sys.executable, "-c", worker, "0"],
        check=True,
        capture_output=True,
        text=True,
    )
    elapsed = __import__("time").monotonic() - started
    assert elapsed >= 0.35
    assert second.stdout.strip() == "entered"
    assert first.wait(timeout=5) == 0


def test_cache_lock_releases_descriptor_when_owner_stamp_fails(
    tmp_path, monkeypatch,
):
    import build_i18n_catalogs as builder

    path = tmp_path / "pt.json"
    real_ftruncate = builder.os.ftruncate
    monkeypatch.setattr(
        builder.os,
        "ftruncate",
        lambda _descriptor, _length: (_ for _ in ()).throw(OSError("boom")),
    )
    with __import__("pytest").raises(OSError, match="boom"):
        with builder._exclusive_cache_lock(path):
            pass
    monkeypatch.setattr(builder.os, "ftruncate", real_ftruncate)
    with builder._exclusive_cache_lock(path):
        pass


def test_cache_lock_closes_descriptor_when_explicit_unlock_fails(
    tmp_path, monkeypatch,
):
    import build_i18n_catalogs as builder

    path = tmp_path / "pt.json"
    real_flock = builder.fcntl.flock

    def fail_unlock(descriptor, operation):
        if operation == builder.fcntl.LOCK_UN:
            raise OSError("unlock boom")
        return real_flock(descriptor, operation)

    monkeypatch.setattr(builder.fcntl, "flock", fail_unlock)
    with __import__("pytest").raises(OSError, match="unlock boom"):
        with builder._exclusive_cache_lock(path):
            pass
    monkeypatch.setattr(builder.fcntl, "flock", real_flock)
    with builder._exclusive_cache_lock(path):
        pass


def test_atomic_catalog_writes_preserve_existing_mode(tmp_path):
    from build_documentation_i18n import _write_json
    from build_i18n_catalogs import _atomic_write_text

    text_path = tmp_path / "catalog.py"
    json_path = tmp_path / "catalog.json"
    for path in (text_path, json_path):
        path.write_text("old", encoding="utf-8")
        path.chmod(0o664)
    _atomic_write_text(text_path, "new")
    _write_json(json_path, {"new": True})
    assert stat.S_IMODE(text_path.stat().st_mode) == 0o664
    assert stat.S_IMODE(json_path.stat().st_mode) == 0o664


def test_atomic_catalog_writes_clean_temporary_files_after_failure(tmp_path):
    from build_documentation_i18n import _write_json
    from build_i18n_catalogs import _atomic_write_text

    with __import__("pytest").raises(TypeError):
        _write_json(tmp_path / "broken.json", {"value": object()})
    with __import__("pytest").raises(TypeError):
        _atomic_write_text(tmp_path / "broken.py", object())
    assert not list(tmp_path.glob("*.tmp"))


def test_catalog_seed_requires_current_per_entry_source_hash(monkeypatch):
    import build_i18n_catalogs as builder
    from spacr.qt.i18n_catalogs import de, en

    key = next(iter(en.SETTING_TOOLTIPS))
    source = en.SETTING_TOOLTIPS[key]
    monkeypatch.setattr(de, "MODEL", builder.MODEL_SPECS["de"][0])
    monkeypatch.setattr(de, "SOURCE_HASHES", {
        **getattr(de, "SOURCE_HASHES", {}),
        ("SETTING_TOOLTIPS", key): "stale",
    }, raising=False)
    cache = {}
    builder._seed_cache_from_catalog("de", cache)
    assert source not in cache


def test_incremental_api_generation_reuses_only_current_nonblank_entries(
    monkeypatch, tmp_path,
):
    import build_documentation_i18n as builder

    docs = {"spacr.example.current": "Current source.",
            "spacr.example.changed": "Changed source."}
    current = docs["spacr.example.current"]
    payload = {
        "symbols": {
            "spacr.example.current": {
                "source_sha256": hashlib.sha256(current.encode()).hexdigest(),
                "source_blocks_sha256": builder._source_block_hashes(current),
                "translation_source_blocks_sha256":
                    builder._translation_source_block_hashes(current),
                "text": "Aktuell text.",
            },
            "spacr.example.changed": {
                "source_sha256": "stale",
                "text": "Gammal text.",
            },
        },
    }
    (tmp_path / "sv.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(builder, "API_DIR", tmp_path)

    assert builder.reusable_api_translations(docs, "sv") == {
        "spacr.example.current": "Aktuell text.",
    }


def test_api_block_completeness_uses_exact_code_hash_allowlist():
    from build_documentation_i18n import (
        API_EXACT_BLOCK_SHA256_ALLOWLIST,
        _api_block_requires_translation,
        _source_hash,
    )
    from build_i18n_catalogs import _syntax_preserved

    prose = "Return the requested objects in deterministic order."
    code = "``(H, W, C)``."
    assert _api_block_requires_translation(prose)
    assert _source_hash(code) in API_EXACT_BLOCK_SHA256_ALLOWLIST
    assert not _api_block_requires_translation(code)
    assert _syntax_preserved(
        'Use "classifier_evaluation" with --dry-run.',
        'Use "classifier_evaluation" with --dry-run.',
    )
    assert not _syntax_preserved(
        'Use "classifier_evaluation" with --dry-run.',
        'Use " classifier_evaluation" with --dry-run.',
    )
    assert not _syntax_preserved(
        'Use "classifier_evaluation" with --dry-run.',
        'Use "classifier_evaluation" with --dry-run and "unexpected_mode".',
    )


def test_api_source_discovery_excludes_untracked_backup_icons():
    from build_documentation_i18n import public_docstrings

    assert not any("backup_icons" in key for key in public_docstrings())


def test_orcid_is_a_protected_translation_contract():
    from build_i18n_catalogs import _protect, _syntax_preserved

    source = "Copyright Matthew O'Meara (ORCID 0000-0002-3128-5331)."
    protected, literals = _protect(source)
    assert "0000-0002-3128-5331" not in protected
    assert "0000-0002-3128-5331" in literals.values()
    assert _syntax_preserved(
        source,
        "Copyright Matthew O'Meara (ORCID 0000-0002-3128-5331).",
    )
    assert not _syntax_preserved(
        source,
        "Copyright Matthew O'Meara (ORCID 0003-0012-3128-5331).",
    )


def test_api_copy_gate_catches_partial_english_in_every_target_script():
    from build_documentation_i18n import _copied_english_phrases
    from build_i18n_catalogs import MODEL_SPECS

    source = "The panel gets an empty result when the caller cancels."
    mixed_targets = {
        "sv": "Panelen är tom; panel gets an empty result.",
        "de": "Das Feld ist leer; panel gets an empty result.",
        "es": "El panel está vacío; panel gets an empty result.",
        "pt": "O painel retorna vazio; panel gets an empty result.",
        "is": "Spjaldið er tómt; panel gets an empty result.",
        "fr": "Le panneau est vide; panel gets an empty result.",
        "hi": "पैनल खाली है; panel gets an empty result.",
        "ko": "패널이 비어 있습니다; panel gets an empty result.",
        "zh_CN": "面板为空；panel gets an empty result.",
    }
    assert set(mixed_targets) == set(MODEL_SPECS)
    for language, mixed in mixed_targets.items():
        assert _copied_english_phrases(source, mixed, language)

    # A protected identifier and a reviewed scientific name are not prose
    # leakage. The Portuguese future subjunctive ``for`` is not English either.
    assert not _copied_english_phrases(
        "Use ``panel.get`` for K-means discovery.",
        "Use ``panel.get`` para descoberta K-means.",
        "pt",
    )
    for phrase in (
        "trailing newline", "scratch database", "staging file",
        "escape hatch", "dtype name", "config dict", "empty bin",
        "wishful thinking",
    ):
        assert _copied_english_phrases(
            f"Explain the {phrase} clearly.",
            f"Explique o {phrase} claramente.",
            "pt",
        )


def test_german_software_loanwords_do_not_hide_english_fragments():
    from build_documentation_i18n import (
        _copied_english_phrases,
        _has_english_residue,
    )

    source = "The thread returns a string after the event loop."
    german = "Der Thread gibt nach dem Event-Loop einen String zurück."
    assert not _has_english_residue(source, german, "de")
    assert not _copied_english_phrases(source, german, "de")

    partial = "Der Thread returns a string after the Event-Loop."
    assert _has_english_residue(source, partial, "de")
    assert _copied_english_phrases(source, partial, "de")

    accepted_terms = {
        "Use drag and drop with a fuzzy match.":
            "Drag-and-drop mit einem Fuzzy Match verwenden.",
        "The denial of service is prevented by a hard cap.":
            "Der Denial of Service wird durch eine harte Obergrenze verhindert.",
        "Apply the Mann-Whitney U test and D'Agostino-Pearson test.":
            "Mann-Whitney-U-Test und D'Agostino-Pearson-Test anwenden.",
    }
    for term_source, term_target in accepted_terms.items():
        assert not _copied_english_phrases(term_source, term_target, "de")

    # Technical terminology is not permission to retain nearby prose.
    assert _copied_english_phrases(
        "Use drag and drop when the caller chooses a file.",
        "Drag-and-drop verwenden when the caller chooses a file.",
        "de",
    )


def test_portuguese_residue_gate_distinguishes_for_homograph_from_english():
    from build_documentation_i18n import _has_english_residue

    assert not _has_english_residue(
        "Whatever mode is selected.", "Seja qual for o modo selecionado.", "pt",
    )
    assert _has_english_residue(
        "Use this value for the current run.",
        "Use este valor for the current run.",
        "pt",
    )


def test_portuguese_semantic_false_friend_families_are_hard_failures():
    from build_i18n_catalogs import _semantic_false_friends

    cases = {
        "dictionary-as-dictation": (
            "Return a dictionary.", "Retorna um ditado.",
        ),
        "image-plane-as-aircraft": (
            "Return the image plane.", "Retorna o avião da imagem.",
        ),
        "exception-raise-as-increase": (
            "Raises ValueError on invalid input.",
            "Aumenta ValueError para entrada inválida.",
        ),
        "raise-window-as-increase": (
            "The caller can raise or focus it.",
            "O chamador pode aumentar ou focá-la.",
        ),
        "mapping-key-as-keyboard-key": (
            "Return the mapping keys.", "Retorna as teclas do mapeamento.",
        ),
        "scientific-plate-as-dish": (
            "Read the 96-well plate.", "Lê o prato de 96 poços.",
        ),
        "image-tile-as-roof-or-floor-tile": (
            "Merge the image tiles.", "Mescla os azulejos da imagem.",
        ),
        "crop-gender": (
            "Return the crop.", "Retorna a recorte.",
        ),
        "thread-corruption": (
            "Run on the worker thread.", "Executa em um throw worker.",
        ),
        "surplus-angle-bracket": (
            "Return the value.", "Retorna o valor >.",
        ),
    }
    for family, (source, target) in cases.items():
        assert family in _semantic_false_friends(source, target, "pt")


def test_portuguese_semantic_gate_preserves_reviewed_distinct_senses():
    from build_documentation_i18n import _api_block_valid
    from build_i18n_catalogs import _contextualize, _semantic_false_friends

    accepted = (
        ("Raise score cutoff to 0.8.", "Eleve o limite da pontuação para 0,8."),
        (
            "Dropping it would raise the reported power.",
            "Removê-lo aumentaria a potência relatada.",
        ),
        ("Press any key to close.", "Pressione qualquer tecla para fechar."),
        (
            "The label is dictated by the caller.",
            "O rótulo é ditado pelo chamador.",
        ),
        ("A CRISPR screen reports hits.", "Uma triagem CRISPR relata hits."),
        ("Run on the GUI thread.", "Execute na thread da GUI."),
        (
            "Map A -> B when score > 3.",
            "Mapeie A -> B quando a pontuação > 3.",
        ),
    )
    for source, target in accepted:
        assert _contextualize(target, "pt", source) == target
        assert not _semantic_false_friends(source, target, "pt")
        assert _api_block_valid(source, target, "pt")


def test_five_locale_semantic_gate_bad_good_matrix():
    from build_i18n_catalogs import (
        _semantic_false_friends,
        _translation_candidate_valid,
    )

    languages = ("fr", "hi", "ko", "is", "zh_CN")
    rows = (
        (
            "gui-screen-as-scientific-screen",
            "The Qt screen shows settings.",
            {
                "fr": ("Le criblage Qt affiche les paramètres.", "L’écran Qt affiche les paramètres."),
                "hi": ("Qt स्क्रीनिंग सेटिंग दिखाती है।", "Qt स्क्रीन सेटिंग दिखाती है।"),
                "ko": ("Qt 스크리닝이 설정을 표시합니다.", "Qt 화면이 설정을 표시합니다."),
                "is": ("Qt skimun sýnir stillingar.", "Qt skjár sýnir stillingar."),
                "zh_CN": ("Qt 筛选显示设置。", "Qt 界面显示设置。"),
            },
        ),
        (
            "scientific-screen-as-ui-screen",
            "A pooled CRISPR screen reports hits.",
            {
                "fr": ("Un écran CRISPR groupé signale les hits.", "Un criblage CRISPR groupé signale les hits."),
                "hi": ("पूल की गई CRISPR स्क्रीन हिट बताती है।", "पूल की गई CRISPR स्क्रीनिंग हिट बताती है।"),
                "ko": ("풀드 CRISPR 화면이 히트를 보고합니다.", "풀드 CRISPR 스크리닝이 히트를 보고합니다."),
                "is": ("Sameinuð CRISPR skjár skilar niðurstöðum.", "Sameinuð CRISPR skimun skilar niðurstöðum."),
                "zh_CN": ("汇集 CRISPR 屏幕报告命中。", "汇集 CRISPR 筛选报告命中。"),
            },
        ),
        (
            "pipeline-as-pipe", "Run the preprocessing pipeline.",
            {
                "fr": ("Exécutez la canalisation de prétraitement.", "Exécutez le flux de travail de prétraitement."),
                "hi": ("प्री-प्रोसेसिंग नलियाँ चलाएँ।", "प्री-प्रोसेसिंग कार्यप्रवाह चलाएँ।"),
                "ko": ("전처리 배관을 실행합니다.", "전처리 워크플로를 실행합니다."),
                "is": ("Keyrðu leiðsluna.", "Keyrðu forvinnsluvinnsluferlið."),
                "zh_CN": ("运行预处理管道。", "运行预处理流程。"),
            },
        ),
        (
            "data-gate-as-door", "Objects inside the polygon gate are selected.",
            {
                "fr": ("Les objets dans la porte polygonale sont sélectionnés.", "Les objets dans le gate polygonal sont sélectionnés."),
                "hi": ("बहुभुज दरवाजा के भीतर ऑब्जेक्ट चुने जाते हैं।", "बहुभुज गेट के भीतर ऑब्जेक्ट चुने जाते हैं।"),
                "ko": ("다각형 문 안의 객체를 선택합니다.", "다각형 게이트 안의 객체를 선택합니다."),
                "is": ("Hlutir innan hliðsins eru valdir.", "Hlutir innan gatesins eru valdir."),
                "zh_CN": ("选择多边形门内的对象。", "选择多边形门控内的对象。"),
            },
        ),
        (
            "statistical-power-as-authority-or-electricity",
            "Estimate statistical power from sample size.",
            {
                "fr": ("Estimez le pouvoir à partir de la taille d’échantillon.", "Estimez la puissance à partir de la taille d’échantillon."),
                "hi": ("नमूना आकार से सत्ता का अनुमान लगाएँ।", "नमूना आकार से सांख्यिकीय शक्ति का अनुमान लगाएँ।"),
                "ko": ("표본 크기에서 권력을 추정합니다.", "표본 크기에서 검정력을 추정합니다."),
                "is": ("Metið vald út frá úrtaksstærð.", "Metið styrk út frá úrtaksstærð."),
                "zh_CN": ("根据样本量估计电力。", "根据样本量估计统计功效。"),
            },
        ),
        (
            "dictionary-as-dictation", "Return a dictionary of settings.",
            {
                "fr": ("Retourne une dictée de paramètres.", "Retourne un dictionnaire de paramètres."),
                "hi": ("सेटिंग का श्रुतलेख लौटाएँ।", "सेटिंग का शब्दकोश लौटाएँ।"),
                "ko": ("설정 받아쓰기를 반환합니다.", "설정 사전을 반환합니다."),
                "is": ("Skila fyrirmæli stillinga.", "Skila orðabók stillinga."),
                "zh_CN": ("返回设置听写。", "返回设置字典。"),
            },
        ),
        (
            "mapping-key-as-keyboard-key", "Return the mapping keys.",
            {
                "fr": ("Retourne les touches du mappage.", "Retourne les clés du mappage."),
                "hi": ("मैपिंग की कीबोर्ड कुंजी लौटाएँ।", "मैपिंग कुंजियाँ लौटाएँ।"),
                "ko": ("매핑 키보드 키를 반환합니다.", "매핑 키를 반환합니다."),
                "is": ("Skila lyklaborðslyklar vörpunar.", "Skila vörpunarlyklum."),
                "zh_CN": ("返回映射键盘键。", "返回映射键。"),
            },
        ),
        (
            "image-plane-as-aircraft", "Read the image plane.",
            {
                "fr": ("Lisez l’avion de l’image.", "Lisez le plan d’image."),
                "hi": ("इमेज विमान पढ़ें।", "इमेज प्लेन पढ़ें।"),
                "ko": ("이미지 항공기를 읽습니다.", "이미지 평면을 읽습니다."),
                "is": ("Lesið flugvél myndarinnar.", "Lesið myndplanið."),
                "zh_CN": ("读取图像飞机。", "读取图像平面。"),
            },
        ),
        (
            "image-tile-as-roof-or-floor-tile", "Align overlapping image tiles.",
            {
                "fr": ("Alignez les carreaux d’image superposés.", "Alignez les tuiles d’image superposées."),
                "hi": ("ओवरलैप वाली फर्श की टाइल संरेखित करें।", "ओवरलैप वाली इमेज टाइल संरेखित करें।"),
                "ko": ("겹치는 바닥 타일을 정렬합니다.", "겹치는 이미지 타일을 정렬합니다."),
                "is": ("Samstillið skarast gólfflísar.", "Samstillið skarast myndflísar."),
                "zh_CN": ("对齐重叠的地板砖。", "对齐重叠的图像瓦片。"),
            },
        ),
        (
            "scientific-plate-as-dish", "Read the 384-well plate.",
            {
                "fr": ("Lisez une assiette de 384 puits.", "Lisez une plaque de 384 puits."),
                "hi": ("384-वेल थाली पढ़ें।", "384-वेल प्लेट पढ़ें।"),
                "ko": ("384웰 접시를 읽습니다.", "384웰 플레이트를 읽습니다."),
                "is": ("Lesið diskur með 384 brunnum.", "Lesið plötu með 384 brunnum."),
                "zh_CN": ("读取 384 孔盘子。", "读取 384 孔微孔板。"),
            },
        ),
        (
            "image-crop-as-agriculture", "Load image crops.",
            {
                "fr": ("Chargez les récoltes d’image.", "Chargez les vignettes d’image."),
                "hi": ("इमेज फसल लोड करें।", "इमेज क्रॉप लोड करें।"),
                "ko": ("이미지 작물을 로드합니다.", "이미지 크롭을 로드합니다."),
                "is": ("Hlaðið ræktun.", "Hlaðið myndúrklippum."),
                "zh_CN": ("加载图像作物。", "加载图像裁剪。"),
            },
        ),
        (
            "compute-run-as-race", "Resume the failed pipeline run.",
            {
                "fr": ("Reprenez la course du pipeline échouée.", "Reprenez l’exécution du flux de travail échouée."),
                "hi": ("विफल पाइपलाइन दौड़ फिर चलाएँ।", "विफल पाइपलाइन रन फिर चलाएँ।"),
                "ko": ("실패한 파이프라인 경주를 재개합니다.", "실패한 파이프라인 실행을 재개합니다."),
                "is": ("Haldið áfram misheppnuðu hlaup.", "Haldið áfram misheppnaðri keyrslu."),
                "zh_CN": ("恢复失败的流程赛跑。", "恢复失败的流程运行。"),
            },
        ),
        (
            "thread-corruption", "Run this on a GUI worker thread.",
            {
                "fr": ("Exécutez ceci sur un fil de discussion GUI.", "Exécutez ceci sur un fil d’exécution GUI."),
                "hi": ("इसे GUI कार्यकर्ता धागे पर चलाएँ।", "इसे GUI कार्यकर्ता थ्रेड पर चलाएँ।"),
                "ko": ("GUI 작업자 스트립에서 실행합니다.", "GUI 작업자 스레드에서 실행합니다."),
                "is": ("Keyrið þetta á GUI-tré.", "Keyrið þetta á vinnuþræði GUI."),
                "zh_CN": ("在 GUI 工作线上运行。", "在 GUI 工作线程上运行。"),
            },
        ),
        (
            "exception-raise-as-increase", "Raises ValueError for invalid input.",
            {
                "fr": ("Augmente ValueError pour une entrée invalide.", "Lève ValueError pour une entrée invalide."),
                "hi": ("अमान्य इनपुट पर ValueError बढ़ाएं।", "अमान्य इनपुट पर ValueError उत्पन्न करता है।"),
                "ko": ("잘못된 입력에 ValueError를 올리면 실패합니다.", "잘못된 입력에 ValueError를 발생시킵니다."),
                "is": ("Hækkar ValueError fyrir ógilt inntak.", "Kastar ValueError fyrir ógilt inntak."),
                "zh_CN": ("无效输入会增加 ValueError。", "无效输入会抛出 ValueError。"),
            },
        ),
        (
            "raise-window-as-increase",
            "Return this screen so the caller can raise or focus it.",
            {
                "fr": ("Retourne cet écran afin que l’appelant puisse le soulever ou le focaliser.", "Retourne cet écran afin que l’appelant puisse le mettre au premier plan ou le focaliser."),
                "hi": ("यह स्क्रीन लौटाएँ ताकि कॉलर इसे उठा या फोकस कर सके।", "यह स्क्रीन लौटाएँ ताकि कॉलर इसे सामने ला या फोकस कर सके।"),
                "ko": ("호출자가 화면을 올리거나 포커스하도록 반환합니다.", "호출자가 화면을 앞으로 가져오거나 포커스하도록 반환합니다."),
                "is": ("Skilið skjánum svo kallari geti hækkað eða fókusað hann.", "Skilið skjánum svo kallari geti fært hann fremst eða fókusað hann."),
                "zh_CN": ("返回此界面以便调用方提高或聚焦它。", "返回此界面以便调用方置前或聚焦它。"),
            },
        ),
    )
    for family, source, translations in rows:
        assert set(translations) == set(languages)
        for language in languages:
            bad, good = translations[language]
            assert family in _semantic_false_friends(source, bad, language), (
                family, language, bad,
            )
            assert not _semantic_false_friends(source, good, language), (
                family, language, good,
                _semantic_false_friends(source, good, language),
            )
            assert not _translation_candidate_valid(source, bad, language)
            assert _translation_candidate_valid(source, good, language)

    well_source = "Return each well in the plate."
    wells = {
        "fr": ("Retourne chaque bien de la plaque.", "Retourne chaque puits de la plaque."),
        "hi": ("प्लेट का हर कुआँ लौटाएँ।", "प्लेट का हर वेल लौटाएँ।"),
        "ko": ("플레이트의 각 우물을 반환합니다.", "플레이트의 각 웰을 반환합니다."),
        "zh_CN": ("返回微孔板中的每个井。", "返回微孔板中的每个孔。"),
    }
    for language, (bad, good) in wells.items():
        assert "scientific-well-as-adverb" in _semantic_false_friends(
            well_source, bad, language,
        )
        assert not _semantic_false_friends(well_source, good, language)
        assert not _translation_candidate_valid(well_source, bad, language)
        assert _translation_candidate_valid(well_source, good, language)
    assert not _semantic_false_friends(
        well_source, "Skila hverjum brunni á plötunni.", "is",
    )


def test_semantic_gate_mandatory_negative_controls():
    from build_i18n_catalogs import _semantic_false_friends

    accepted = (
        ("This works well.", "Cela fonctionne bien.", "fr"),
        ("Press the keyboard key.", "Appuyez sur la touche du clavier.", "fr"),
        ("Use a power-law exponent.", "Utilisez un exposant de pouvoir.", "fr"),
        ("Use a power of two.", "Utilisez un pouvoir de deux.", "fr"),
        ("Raise the threshold by 0.1.", "Augmentez le seuil de 0,1.", "fr"),
        ("A forum discussion thread.", "Un fil de discussion du forum.", "fr"),
        ("The aircraft has two planes.", "L’aéronef a deux avions.", "fr"),
        ("The tracked trajectory runs across frames.", "La trajectoire suivie parcourt les images.", "fr"),
        ("The count that gated it; the invariant this type carries.", "Le compte qui le porte ; l’invariant de ce type.", "fr"),
        ("A Dataset is a flat bag with no plate geometry.", "Un Dataset est un sac plat sans géométrie de plaque.", "fr"),
        ("The Home tile button opens settings.", "Le bouton tuile Home ouvre les paramètres.", "fr"),
        ("The GUI screen compares a pooled CRISPR screen.", "L’écran GUI compare un criblage CRISPR.", "fr"),
        ("Raises ValueError for invalid input.", "Soulève ValueError pour une entrée invalide.", "fr"),
        ("Return each well in the plate.", "Skila hverjum brunni á plötunni.", "is"),
        ("Use ``pipeline`` as the code key.", "Utilisez ``tuyau`` comme clé de code.", "fr"),
    )
    for source, target, language in accepted:
        assert not _semantic_false_friends(source, target, language), (
            source, target, _semantic_false_friends(source, target, language),
        )
    assert "raise-window-as-increase" in _semantic_false_friends(
        "Return this screen so the caller can raise or focus it.",
        "Retourne cet écran afin que l’appelant puisse le soulever ou le focaliser.",
        "fr",
    )


def test_ko_zh_additional_semantic_families_and_negative_controls():
    from build_i18n_catalogs import _semantic_false_friends

    bad = (
        ("Append the job to the queue.", "将任务放在尾巴。", "zh_CN", "software-queue-as-tail"),
        ("Append the job to the queue.", "작업을 꼬리에 추가합니다.", "ko", "software-queue-as-tail"),
        ("Read each image field.", "读取每个图像领域。", "zh_CN", "imaging-field-as-land-or-domain"),
        ("Read each image field.", "각 이미지 밭을 읽습니다.", "ko", "imaging-field-as-land-or-domain"),
        ("Read each image channel.", "读取每个图像频道。", "zh_CN", "imaging-channel-as-broadcast-channel"),
        ("The classifier predicts crops.", "分类师预测图像裁剪。", "zh_CN", "software-classifier-as-person-or-machine"),
        ("The classifier predicts crops.", "분류자가 크롭을 예측합니다.", "ko", "software-classifier-as-person-or-machine"),
        ("Return a human-readable reference.", "返回人文引用。", "zh_CN", "human-readable-as-humanities"),
        ("One plate to process.", "处理一个盘子。", "zh_CN", "scientific-plate-as-dish"),
        ("One plate to process.", "한 개의 접시를 처리합니다.", "ko", "scientific-plate-as-dish"),
        ("Read the mapping keys.", "读取映射密钥。", "zh_CN", "mapping-key-as-secret-key"),
        ("Return the result.", "此分類上一篇：结果。", "zh_CN", "web-corpus-contamination"),
    )
    for source, target, language, family in bad:
        assert family in _semantic_false_friends(source, target, language)

    accepted = (
        ("Open the previous article.", "打开上一篇文章。", "zh_CN"),
        ("Show the tail of the queue.", "显示队列尾部。", "zh_CN"),
        ("Append the job to the queue.", "将作业追加到队列。", "zh_CN"),
        ("Store the value in a database field.", "将值存入数据库字段。", "zh_CN"),
        ("This research field studies agriculture.", "这一研究领域研究农业。", "zh_CN"),
        ("Select the television broadcast channel.", "选择电视频道。", "zh_CN"),
        ("The ML classifier predicts each crop.", "机器学习分类器预测每个图像裁剪。", "zh_CN"),
        ("Return a human-readable reference.", "返回人类可读的引用。", "zh_CN"),
        ("Serve food on the plate.", "把食物放在盘子里。", "zh_CN"),
        ("Process one 384-well plate.", "处理一个 384 孔板。", "zh_CN"),
        ("Store the API key in the credential vault.", "将 API 密钥存入凭据保管库。", "zh_CN"),
        ("Read the value under this mapping key.", "读取此映射键对应的值。", "zh_CN"),
        ("Read each image channel.", "각 이미지 채널을 읽습니다.", "ko"),
    )
    for source, target, language in accepted:
        assert not _semantic_false_friends(source, target, language), (
            source, target, _semantic_false_friends(source, target, language),
        )


def test_opencc_t2s_normalizes_only_unprotected_chinese_prose():
    from build_i18n_catalogs import (
        _has_traditional_chinese_prose,
        _simplify_chinese_prose,
    )

    source = "這個軟體讀取記憶體，保留 ``個為當`` 與 :func:`spacr.run`."
    normalized = _simplify_chinese_prose(source)
    assert normalized == "这个软体读取记忆体，保留 ``個為當`` 与 :func:`spacr.run`."
    assert _simplify_chinese_prose(normalized) == normalized
    assert not _has_traditional_chinese_prose(normalized)
    assert _has_traditional_chinese_prose(source)


def test_opencc_audit_probe_fails_closed_when_dependency_is_missing(
    monkeypatch,
):
    import build_i18n_catalogs as builder
    import build_documentation_i18n as docs_builder

    monkeypatch.setattr(builder.ctypes.util, "find_library", lambda _name: None)
    with __import__("pytest").raises(RuntimeError, match="requires OpenCC"):
        builder._has_traditional_chinese_prose("简体中文")
    with __import__("pytest").raises(RuntimeError, match="requires OpenCC"):
        builder.audit({}, ["zh_CN"])
    with __import__("pytest").raises(RuntimeError, match="requires OpenCC"):
        docs_builder.audit({}, ["zh_CN"])


def test_api_translation_context_has_no_reviewed_grammar_failures():
    import build_documentation_i18n as builder

    forbidden = re.compile(
        r"\b(?:a|an) execute\b|"
        r"\b(?:partly|partially) execute\b|"
        r"\bagain\s+again\b|"
        r"\b(?:been|was|were|is|are) execute again\b|"
        r"\b(?:can|could|will|would|should|must|may|might|to|do|does|did) "
        r"(?:repeat execution|processing run)\b|"
        # Run/session compounds and re-run objects must retain their English
        # noun/verb order after the context-neutral sense expansion.
        r"\bprocessing[- ]session journal (?:runs?|executions?|processing "
        r"sessions?)\b|"
        r"\bexecute again\s+(?:it|them|the|this|that|a|an)\b|"
        r"\b\d+\s+(?:processing[- ]runs?|processing sessions?)\b|"
        # These caught article, plurality and double-compound regressions in
        # the thread/crop families during the corpus-wide morphology pass.
        r"\b(?:a|an)\s+(?:worker pools|background execution units|"
        r"independent execution paths)\b|"
        r"\bworker-count (?:limits?) count\b|"
        r"\ba\s+(?:image-region|extracted-image-region)\b|"
        r"\ban\s+extracted image regions\b|"
        r"\b(?:object|annotation|per|measure-and)-extracted image region\b|"
        # Finite mapping-key verbs need subject agreement, not a generic noun
        # substitution that leaves the original English verb behind.
        r"\b(?:features|models|results) is indexed\b|"
        r"\b(?:feature|model|result) are indexed\b|"
        r"\b(?:cache keys off it|measurement tables key on)\b|"
        # Plane and queue compounds are model-facing grammar contracts.
        r"\bper image layers?\b|"
        r"\bfor each image layers\b|"
        r"\ba one image layer\b|"
        r"\b(?:work list|software job list)[- ]based\b|"
        r"\bwork-list based\b|"
        # A protected RST exception role must be governed by Throw/Throws;
        # inserting an indefinite exception noun before it is ungrammatical.
        r"\b(?:produce[sd]? an? error|throw an exception)\s+"
        r":(?:class|exc):`",
        re.IGNORECASE,
    )
    for key, document in builder.public_docstrings().items():
        blocks, _layout = builder.translatable_blocks(document)
        for index, block in enumerate(blocks):
            contextual = builder._api_translation_source(block)
            assert not forbidden.search(contextual), (key, index, contextual)


def test_api_semantic_gate_rejects_bad_senses_and_surplus_globally():
    from build_documentation_i18n import _api_block_valid

    assert not _api_block_valid(
        "Raises ValueError on invalid input.",
        "Aumenta ValueError para entrada inválida.",
        "pt",
    )
    assert not _api_block_valid(
        "Return the mapping keys.", "Retorna as teclas do mapeamento.", "pt",
    )
    assert not _api_block_valid(
        "Return a dictionary.", "Retorna um ditado.", "pt",
    )
    assert not _api_block_valid(
        "Return the value.", "Gibt den Wert > zurück.", "de",
    )
    assert not _api_block_valid(
        "Map A -> B when x > 3.", "Mapeie A > B quando x > 3.", "pt",
    )
    assert not _api_block_valid(
        "Return each well.", "Devuelve cada bien.", "es",
    )
    assert not _api_block_valid(
        "Return the mapping keys.", "Devuelve las teclas del mapeo.", "es",
    )
    assert not _api_block_valid(
        "Raises ValueError for invalid input.",
        "Eleva ValueError para una entrada no válida.",
        "es",
    )


def test_contextualize_preserves_non_scientific_senses_across_locales():
    from build_documentation_i18n import _api_block_valid
    from build_i18n_catalogs import _contextualize

    accepted = (
        ("sv", "This works well.", "Detta fungerar bra."),
        ("es", "This works well.", "Esto funciona bien."),
        ("fr", "This works well.", "Cela fonctionne bien."),
        ("es", "Press any key.", "Pulse cualquier tecla."),
        (
            "es", "Raise score cutoff to 0.8.",
            "Eleve el umbral de puntuación a 0,8.",
        ),
    )
    for language, source, target in accepted:
        assert _contextualize(target, language, source) == target
        assert _api_block_valid(source, target, language)


def test_portuguese_context_repairs_only_unambiguous_semantic_families():
    from build_i18n_catalogs import (
        _contextualize,
        _semantic_false_friends,
        _syntax_preserved,
    )

    cases = (
        ("Return the image plane.", "Retorna o avião da imagem."),
        (
            "The Annotate screen lets the caller raise or focus it.",
            "A triagem Annotate permite aumentar ou focá-la.",
        ),
        ("Read the plate.", "Lê o prato."),
        ("Merge image tiles.", "Mescla azulejos de imagem."),
        ("Return the crop.", "Retorna a recorte."),
        ("Run on a worker thread.", "Executa em um throw worker."),
    )
    for source, target in cases:
        repaired = _contextualize(target, "pt", source)
        assert not _semantic_false_friends(source, repaired, "pt")
        assert _syntax_preserved(source, repaired), (source, repaired)


def test_reviewed_readmes_do_not_reintroduce_known_context_errors():
    readme_root = ROOT / "docs" / "i18n" / "readme"
    french = (readme_root / "README.fr.rst").read_text(encoding="utf-8")
    swedish = (readme_root / "README.sv.rst").read_text(encoding="utf-8")
    icelandic = (readme_root / "README.is.rst").read_text(encoding="utf-8")

    assert "l'criblage" not in french and "l’criblage" not in french
    assert "löpande ansökan" not in swedish
    assert "spaCR → Stillingar → Tungumál" in icelandic


def test_localized_readmes_do_not_leave_long_english_feature_copy():
    """GitHub's feature table and surrounding guidance must be localized."""
    readme_root = ROOT / "docs" / "i18n" / "readme"
    localized = {
        path.stem.removeprefix("README."): path.read_text(encoding="utf-8")
        for path in readme_root.glob("README.*.rst")
    }
    assert set(localized) == {
        "sv", "de", "es", "zh_CN", "pt", "hi", "ko", "is", "fr",
    }

    canonical = (ROOT / "README.rst").read_text(encoding="utf-8")
    table = canonical[
        canonical.index(".. list-table::"):
        canonical.index(".. |api-qt-app| replace::")
    ]
    descriptions = {
        line.strip()[2:]
        for line in table.splitlines()
        if line.startswith("     - ")
        and len(line.strip()[2:].split()) >= 6
    }
    assert len(descriptions) == 28

    long_prose_fragments = {
        "The installer downloads a private Python 3.12 runtime",
        "Runs are now identifiable",
        "Navigation, Preferences, AI and LIVE controls",
        "94 short animations explain what 143 visual settings",
        "Bug reports and focused feature requests",
        "The current development branch is source-available",
        "contains narrated, captioned walkthroughs",
        "segments cells, nuclei, pathogens and organelles with Cellpose",
        "In the evaluation screen, a confusion-matrix cell is a query",
    }
    table_labels = {
        "**Ten-language localization**",
        "**Localized contextual help**",
        "**Setting animation registry**",
        "**Visual setting animations**",
        "**Installation diagnosis**",
        "**Flat-field correction**",
        "**Object measurements**",
        "**Well and collision report**",
        "**Screen effect estimation**",
        "**Run provenance**",
    }
    forbidden = descriptions | long_prose_fragments | table_labels
    for language, text in localized.items():
        leftovers = sorted(fragment for fragment in forbidden if fragment in text)
        assert not leftovers, f"{language} retains English README copy: {leftovers}"


def test_localized_readmes_preserve_safety_meaning_and_language_names():
    readme_root = ROOT / "docs" / "i18n" / "readme"
    spanish = (readme_root / "README.es.rst").read_text(encoding="utf-8")
    hindi = (readme_root / "README.hi.rst").read_text(encoding="utf-8")
    korean = (readme_root / "README.ko.rst").read_text(encoding="utf-8")

    # The exporter rejects invented numbers; the old machine translation
    # reversed this safety guarantee.
    assert "se rechaza cualquier borrador" in spanish
    assert "no es rechazado" not in spanish
    assert "no se rechaza" not in spanish

    # Hindi is a language, not the Hindu religion.
    assert "हिन्दी" in hindi
    assert "हिंदू" not in hindi and "हिन्दू" not in hindi
    assert "힌디어" in korean
    assert "힌두교" not in korean

    # Common literal-translation failures in scientific/software context.
    for false_friend in ("la antorcha", "el gasoducto", "cara de agarre", "Open Daughth"):
        assert false_friend not in spanish


def test_localized_readmes_keep_the_badge_row_structurally_intact():
    expected = (
        "|Docs| |Tutorials| |PyPI| |Python| |Tests| |Qt| "
        "|Source| |Issues| |License| |DOI|"
    )
    for path in (ROOT / "docs" / "i18n" / "readme").glob("README.*.rst"):
        assert path.read_text(encoding="utf-8").splitlines()[0] == expected


def test_localized_readme_images_have_reviewed_accessible_text():
    expected_workflow_alt = {
        "de": "spaCR-Arbeitsablauf und Ausgabeorganisation",
        "es": "Flujo de trabajo y organización de resultados de spaCR",
        "fr": "Flux de travail spaCR et organisation des sorties",
        "hi": "spaCR कार्यप्रवाह और आउटपुट संगठन",
        "is": "Verkflæði spaCR og skipulag úttaks",
        "ko": "spaCR 작업 흐름 및 출력 구성",
        "pt": "Fluxo de trabalho e organização das saídas do spaCR",
        "sv": "spaCR:s arbetsflöde och struktur för utdata",
        "zh_CN": "spaCR 工作流程及输出结构",
    }
    readme_root = ROOT / "docs" / "i18n" / "readme"
    for language, workflow_alt in expected_workflow_alt.items():
        text = (readme_root / f"README.{language}.rst").read_text(
            encoding="utf-8"
        )
        alt_text = re.findall(r"(?m)^   :alt: (.+)$", text)
        assert len(alt_text) == 14
        assert alt_text[-1] == workflow_alt
        assert alt_text[-1] != "spaCR workflow and output organization"
        assert "Interactive tutorials" not in alt_text
        assert "Latest installers" not in alt_text


def test_localized_readme_inline_markup_is_balanced_and_tight():
    canonical = (ROOT / "README.rst").read_text(encoding="utf-8")
    for path in (ROOT / "docs" / "i18n" / "readme").glob("README.*.rst"):
        text = path.read_text(encoding="utf-8")
        for marker in ("**", "``"):
            assert text.count(marker) == canonical.count(marker)
            marked_text = text.split(marker)[1::2]
            assert all(value and value == value.strip() for value in marked_text)

        # A prior Portuguese translation added visible ``>`` characters after
        # links and inline literals. Real link-target brackets are removed first.
        without_link_targets = re.sub(r"<[^>\n]+>", "", text)
        assert ">" not in without_link_targets


def test_localized_readmes_preserve_module_names_and_technical_terms():
    expected_modules = [
        "Mask", "Measure", "Annotate", "Classify", "Map Barcodes", "Regression",
    ]
    protected_terms = {
        "torchvision", "btrack", "pylibCZIrw", "czifile", "Hugging Face",
        "Power / Design", "ComBat", "scanpy",
    }
    fallback_phrases = {
        "de": "nicht unterstützten Gebietsschemata",
        "es": "configuraciones regionales no compatibles",
        "fr": "paramètres régionaux non pris en charge",
        "hi": "असमर्थित लोकेल",
        "is": "Tungumál sem ekki eru studd",
        "ko": "지원되지 않는 로캘",
        "pt": "Localidades não compatíveis",
        "sv": "Språk som inte stöds",
        "zh_CN": "不支持的语言环境",
    }
    known_context_errors = {
        "de": {"Fackelvision"},
        "es": {"Anotate", "la antorcha", "el gasoducto", "cara de agarre"},
        "fr": {"Face de harnais", "Anotate"},
        "hi": {"चेहरे को हिलाना", "**मैप बारकोड**", "**ग्रेसेज**"},
        "is": {"kyndilssýn"},
        "ko": {"전체 미생물", "전원 / 디자인", "그래프 건축가"},
        "pt": {"pylibCZrw", "**Máscara**", "**Mapa códigos de barras**"},
        "sv": {"Huggande ansikte", "**Mäta**", "**Karta Streckkoder**"},
        "zh_CN": {"此分類上一篇", "印度语", "电源 / 设计", "图形建筑师"},
    }
    readme_root = ROOT / "docs" / "i18n" / "readme"
    for language, fallback_phrase in fallback_phrases.items():
        text = (readme_root / f"README.{language}.rst").read_text(
            encoding="utf-8"
        )
        module_lines = re.findall(r"(?m)^\*\*([^*\n]+)\*\* .+$", text)
        assert module_lines[:6] == expected_modules
        missing_terms = sorted(term for term in protected_terms if term not in text)
        assert not missing_terms, f"{language} changed protected terms: {missing_terms}"
        assert fallback_phrase in text
        provenance = next(line for line in text.splitlines() if "AnnData" in line)
        assert all(name in provenance for name in ("Mask", "Measure", "Classify"))
        assert not any(error in text for error in known_context_errors[language])


def test_localized_readmes_preserve_urls_code_and_table_shape():
    canonical = (ROOT / "README.rst").read_text(encoding="utf-8")
    canonical_urls = sorted(re.findall(r"https?://[^\s>`]+", canonical))
    code_pattern = re.compile(
        r"(?m)^\.\. code-block:: [^\n]+\n\n((?: {3}[^\n]*(?:\n|$))+)",
    )
    canonical_code = code_pattern.findall(canonical)
    for path in (ROOT / "docs" / "i18n" / "readme").glob("README.*.rst"):
        text = path.read_text(encoding="utf-8")
        assert sorted(re.findall(r"https?://[^\s>`]+", text)) == canonical_urls
        assert code_pattern.findall(text) == canonical_code
        assert len(re.findall(r"(?m)^   \* - ", text)) == 33
        for target in re.findall(r"<((?:\.\.?/)[^>#]+)(?:#[^>]*)?>`_", text):
            assert (path.parent / target).resolve().exists(), (path, target)


def test_reviewed_readme_headings_match_the_canonical_source_and_locales():
    from build_documentation_i18n import (
        REVIEWED_README_HEADINGS,
        translatable_blocks,
    )

    canonical = (ROOT / "README.rst").read_text(encoding="utf-8")
    source_blocks, _ = translatable_blocks(canonical)
    assert "Animated setting guidance" in REVIEWED_README_HEADINGS
    assert "Animated settings guidance" not in REVIEWED_README_HEADINGS
    for source, localized_headings in REVIEWED_README_HEADINGS.items():
        assert source_blocks.count(source) == 1
        for language, heading in localized_headings.items():
            localized = (
                ROOT / "docs" / "i18n" / "readme" / f"README.{language}.rst"
            ).read_text(encoding="utf-8")
            assert heading in localized, (source, language, heading)


def test_localized_readmes_keep_reviewed_semantic_and_typographic_fixes():
    readme_root = ROOT / "docs" / "i18n" / "readme"
    readmes = {
        language: (readme_root / f"README.{language}.rst").read_text(
            encoding="utf-8"
        )
        for language in ("de", "es", "hi", "ko", "pt", "sv", "zh_CN")
    }

    required = {
        "de": {"spaCR-Installationsverzeichnis", "Animierte Einstellungshilfe"},
        "es": {
            "coherencia de las dependencias",
            "La interfaz Tk heredada",
            "Guía animada de ajustes",
        },
        "hi": {
            "अनुक्रमण त्रुटि",
            "एनोटेटर सहमति",
            "गुणांकों",
            "संभावित परिणामों की सूची",
            "एक पूल्ड, छवि-आधारित CRISPR स्क्रीन",
            "ESCRT तंत्र के अपहरण",
            "स्वागत है",
        },
        "ko": {
            "그래도 열기",
            "풀드 이미지 기반 CRISPR 스크린",
            "*T. gondii*\\ 의 ESCRT 기능 탈취",
        },
        "pt": {"Guia animado de configurações"},
        "sv": {"Animerad hjälp för inställningar"},
        "zh_CN": {
            "Windows 10/11：下载",
            "macOS 11+（英特尔和苹果硅）：下载",
            "64 位 Linux：下载",
            "测试数据集：Hugging Face toxo_mito",
            "测序数据：NCBI BioProject",
            "请引用：",
        },
    }
    forbidden = {
        "de": {"spaCR Installationsverzeichnis"},
        "es": {"interfaz Tk legado", "Orientación de ajuste animado"},
        "hi": {
            "स्वाग योग्य",
            "sequencing error",
            "dropout",
            "segmentation",
            "annotator agreement",
            "data leakage",
            "batch correction",
            "coefficients",
            "hit list",
            "एक संयुक्त छवि-आधारित CRISPR",
            "एकीकृत छवि-आधारित CRISPR",
            "ESCRT उपइकाई",
            "ESCRT उप-इकाई",
            "ESCRT उप इकाई",
            "ESCRT उप-विवाद",
        },
        "ko": {
            "Open Anyway",
            "합성 이미지 기반",
            "통합 이미지 기반",
            "ESCRT 하위",
        },
        "pt": {"Orientação de cenário animado"},
        "zh_CN": {
            "Windows 10/11:下载",
            "Windows 10/11: 下载",
            "测试数据集:",
            "请引用:",
        },
    }
    for language, fragments in required.items():
        missing = sorted(
            fragment for fragment in fragments if fragment not in readmes[language]
        )
        assert not missing, f"{language} lacks reviewed wording: {missing}"
    for language, fragments in forbidden.items():
        leftovers = sorted(
            fragment for fragment in fragments if fragment in readmes[language]
        )
        assert not leftovers, f"{language} retains reviewed errors: {leftovers}"
