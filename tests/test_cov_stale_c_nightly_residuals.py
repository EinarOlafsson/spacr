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
        assert source.count(
            "settings[f'cellpose_{_role}_channel'] = _dense[_raw]"
        ) == 1


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
    assert "if not stacks:\n        return stacks" in source
    sidecar = source.index("# The empty case returned before Cellpose")
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


def test_generated_organelle_keys_are_appended_without_rechecking_contracts():
    """The post-snapshot organelle-slot test owns both contract premises."""
    from spacr import settings

    source = inspect.getsource(settings)
    assert "if _key in expected_types and _key not in categories['General']" \
        not in source
    assert "categories['General'].append(_key)" in source


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
