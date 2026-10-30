"""Nightly pins for the seven shifted coverage residuals in group C.

The behavioural counterparts live in the focused module suites named in each
test below.  These assertions hold the production simplifications themselves:
the guards were re-checking premises already established by the preceding
code, so putting one back would recreate an arc no real input can take.
"""
from __future__ import annotations

import inspect


def test_both_object_generators_trust_the_dense_map_once():
    """Numeric role channels were used to build this very map.

    ``test_cov_r6_pipeline_tail.TestEveryRoleChannelHasADensePosition`` pins
    the shared-key/coercion premise, while ``test_cov_1_object`` drives both
    generators with real dense aliases.
    """
    from spacr import object as objects

    for function in (
            objects.generate_cellpose_masks_sam,
            objects.generate_cellpose_masks):
        source = inspect.getsource(function)
        assert "if _raw in _dense:" not in source
        assert source.count("_fill_cellpose_channel_positions(settings)") == 1
    # 2026-09-26 (item 43): item 493 moved this loop into the shared
    # _fill_cellpose_channel_positions, so each generator now calls it once
    # and the direct lookup lives in that one helper.
    helper = inspect.getsource(
        objects._fill_cellpose_channel_positions)
    assert "if raw in dense:" not in helper
    assert helper.count("settings[derived] = position") == 1


def test_core_uses_the_normalized_source_list_without_a_silent_skip():
    """Input rejection and str-to-list conversion are the type boundary.

    Their positive and refusal cases are exercised by ``test_core_branches``
    and ``TestSrcIsAlwaysAListByTheTimeItIsUsed``.
    """
    from spacr import core

    source = inspect.getsource(core.preprocess_generate_masks)
    assert "if isinstance(settings['src'], list):" not in source
    assert "source_folders = settings['src']" in source


def test_the_model_size_loop_has_a_reachable_gigabyte_fallback():
    """A value beyond every named scale exits the loop and still returns."""
    from spacr import model_zoo

    assert model_zoo._human_bytes(1024 ** 4) == "1024.0 GB"
    source = inspect.getsource(model_zoo._human_bytes)
    loop = source.index('for unit in ("B", "KB", "MB", "GB"):')
    assert source.index('return f"{n:.1f} GB"', loop) > loop


def test_the_v2_sidecar_relies_on_the_early_empty_return():
    """The focused V2 test drives both empty and nonempty stack lists."""
    from spacr import pipeline_v2

    source = inspect.getsource(pipeline_v2.stream_masks_from_stack)
    early_return = "if not stacks:\n        return stacks"
    assert early_return in source
    # ANCHORED ON THE EARLY RETURN ITSELF rather than on a comment that
    # happened to sit after it. The region this test is about is "everything
    # past the empty-list guard", and the guard is code.
    sidecar = source.index(early_return) + len(early_return)
    assert "if stacks:" not in source[sidecar:]
    assert 'sidecar = stacks[0].path.parent / "channel_order.json"' \
        in source[sidecar:]


def test_manifest_scalar_warnings_need_no_second_truthiness_check():
    """The actual search tests drive list, scalar, and empty warnings."""
    from spacr import run_journal

    source = inspect.getsource(run_journal.search_runs)
    assert "elif values:" not in source
    assert "values = manifest.get(key) or []" in source
    assert "else:\n                warnings_list.append(str(values))" in source


def test_generated_organelle_keys_are_appended_without_rechecking_contracts(monkeypatch):
    """The post-snapshot organelle-slot test owns both contract premises.

    2026-10-25: 8fbb75c44 moved the module-level slot loop into
    ``_categorise_every_later_organelle_slot``, so the pin drives that
    function instead of matching the loop's old text: with no type contracts
    at all and General already holding a key, every later slot still gets its
    three channel keys appended to General, once each and in slot order.
    """
    from spacr import settings

    source = inspect.getsource(settings)
    assert "if _key in expected_types and _key not in categories['General']" \
        not in source
    assert "expected_types" not in inspect.getsource(
        settings._categorise_every_later_organelle_slot)
    later = settings.ORGANELLE_SLOT_ROLES[1:]
    suffixes = ('channel', 'mask_dim', 'chann_dim')
    already = f'{later[0]}_channel'
    fresh = {'Organelle': [], 'Organelle advanced': [], 'General': [already]}
    monkeypatch.setattr(settings, 'categories', fresh)
    monkeypatch.setattr(settings, 'expected_types', {})
    settings._categorise_every_later_organelle_slot()
    assert fresh['General'] == [already] + [f'{role}_{suffix}' for role in later
                                            for suffix in suffixes]
    keys = [f'{role}_{suffix}' for role in later for suffix in suffixes]
    monkeypatch.undo()
    assert all(settings.categories['General'].count(key) == 1 for key in keys)


def test_invasion_counts_use_a_branchless_defensive_default():
    """Categoricals provide both columns; ``get`` also tolerates plain data.

    The real categorical assay is covered by ``test_cov_r6_analysis`` and the
    post-snapshot missing-class counterpart remains in
    ``test_cov_r9_twelve_final_guards``.
    """
    from spacr import submodules

    source = inspect.getsource(submodules.analyze_invasion)
    assert "if name not in field_counts.columns:" not in source
    assert "field_counts[name] = field_counts.get(name, 0)" in source
