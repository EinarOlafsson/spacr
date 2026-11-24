"""T7: exact OPS registration metadata, coordinate direction and chunk identity."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from spacr.ome_zarr import OmeZarrError, read_ome_zarr, transformation_matrix, write_ome_zarr
from spacr.ome_zarr_registration import registration_metadata
from tests.test_runtime_backend_caption_owners import extract_backend_sources


def _bits(values):
    """Expose float64 bits, including the sign bit of zero."""
    return np.asarray(values, dtype=np.float64).view(np.uint64)


def _chunk_hashes(path):
    """Hash actual chunk files independently of their metadata."""
    return {p.relative_to(path).as_posix(): hashlib.md5(p.read_bytes()).hexdigest()
            for p in path.rglob('*') if p.is_file() and p.name != 'zarr.json'}


def test_t7_shifts_and_tile_affines_survive_real_store_round_trip(tmp_path):
    """Source float64 entries survive JSON storage and the lane A reader exactly."""
    values = [np.nextafter(1.0, 2.0), -0.0, np.nextafter(-0.25, -1.0),
              np.nextafter(0.9, 1.0), np.nextafter(1234.0, 1235.0), -567.1234567890123]
    record = {'scale': np.nextafter(0.1, 1.0), 'rotation_deg': 0.0,
              'offset_y': values[4], 'offset_x': values[5],
              'secondary_dy': -0.0, 'secondary_dx': np.nextafter(2.0, 3.0),
              'secondary_raw_dy': 99.9, 'secondary_raw_dx': -99.9,
              'align_tile_transforms': json.dumps([[7, 23, *values], [11, 9, *values]])}
    metadata = registration_metadata(record)
    image = np.arange(2 * 7 * 9, dtype=np.uint16).reshape(2, 7, 9)
    plain, registered = tmp_path / 'plain', tmp_path / 'registered'
    write_ome_zarr(plain, image, axes=['c', 'y', 'x'], ngff_version='0.6',
                   chunks=(1, 4, 5), shards=False)
    write_ome_zarr(registered, image, axes=['c', 'y', 'x'], ngff_version='0.6',
                   chunks=(1, 4, 5), shards=False, **metadata)
    opened = read_ome_zarr(registered)
    assert list(opened.transformations) == metadata['coordinate_transformations']
    assert [system['name'] for system in opened.coordinate_systems] == [
        'physical', 'sbs', 'secondary', 'tile_global_7', 'tile_global_11']
    placement, secondary, *tiles = opened.transformations
    steps = placement['transformations']
    assert [step['type'] for step in steps] == ['scale', 'translation']
    np.testing.assert_array_equal(_bits(steps[0]['scale'][1:]), _bits([record['scale']] * 2))
    np.testing.assert_array_equal(_bits(steps[1]['translation'][1:]), _bits(values[4:]))
    np.testing.assert_array_equal(_bits(secondary['translation'][1:]),
                                  _bits([record['secondary_dy'], record['secondary_dx']]))
    for tile in tiles:
        affine = tile['affine']
        extracted = [affine[1][1], affine[1][2], affine[2][1], affine[2][2],
                     affine[1][3], affine[2][3]]
        np.testing.assert_array_equal(_bits(extracted), _bits(values))
    assert _chunk_hashes(plain) == _chunk_hashes(registered)
    assert _chunk_hashes(plain)
    np.testing.assert_array_equal(opened.read(), image)
    assert record['secondary_raw_dy'] == 99.9


@pytest.mark.parametrize('axes', [('y', 'x'), ('c', 'y', 'x'), ('z', 'y', 'x')])
def test_scale_rotation_translation_map_field_pixels_to_sbs(axes):
    """Lane A composes the same forward map as the OPS placement table."""
    row = {'scale': 0.25, 'rotation_deg': 31.0, 'offset_y': 13.5, 'offset_x': -7.25}
    transform = registration_metadata(row, axes=axes)['coordinate_transformations'][0]
    assert [step['type'] for step in transform['transformations']] == [
        'scale', 'rotation', 'translation']
    point = np.arange(len(axes) + 1, dtype=float) + 1
    point[-1] = 1
    iy, ix = axes.index('y'), axes.index('x')
    angle = np.radians(row['rotation_deg'])
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    expected = point.copy()
    expected[[iy, ix]] = row['scale'] * rotation @ point[[iy, ix]] + [13.5, -7.25]
    np.testing.assert_allclose(transformation_matrix(transform, len(axes)) @ point, expected,
                               rtol=1e-15, atol=1e-15)
    assert transform['input'] == {'name': 'physical'}
    assert transform['output'] == {'name': 'sbs'}


def test_secondary_uses_applied_shift_with_the_moving_to_primary_direction():
    """Diagnostic raw estimates cannot replace the applied secondary shift."""
    metadata = registration_metadata({'secondary_dy': 3, 'secondary_dx': -4,
                                      'secondary_raw_dy': 3.4, 'secondary_raw_dx': -4.4},
                                     axes=('y', 'x'))
    transform, = metadata['coordinate_transformations']
    assert transform['input'] == {'name': 'secondary'}
    assert transform['output'] == {'name': 'physical'}
    np.testing.assert_array_equal(transformation_matrix(transform, 2) @ [10, 20, 1],
                                  [13, 16, 1])


@pytest.mark.parametrize("refined, expected", [(1, [11, 15, 1]), (0, [3, 4, 1]),
                                                  (None, [3, 4, 1])])
def test_refined_placement_precedes_seed_without_composing_them(refined, expected):
    """Only an accepted refinement replaces the seed, matching the OPS selector."""
    row = dict(scale=1., rotation_deg=0., offset_y=0., offset_x=0.,
               refined_scale=2., refined_rotation_deg=0., refined_offset_y=5., refined_offset_x=7.)
    if refined is not None:
        row["refined"] = refined
    transform, = registration_metadata(row, axes=('y', 'x'))['coordinate_transformations']
    np.testing.assert_array_equal(transformation_matrix(transform, 2) @ [3, 4, 1], expected)


def test_one_tile_affine_embeds_without_changing_other_axes():
    """All six affine table values retain their row/column order."""
    row = dict(zip(('align_a11', 'align_a12', 'align_a21', 'align_a22', 'align_ty', 'align_tx'),
                   (2., 3., 4., 5., 6., 7.)))
    transform, = registration_metadata(row)['coordinate_transformations']
    assert transform['output'] == {'name': 'tile_global'}
    np.testing.assert_array_equal(transformation_matrix(transform, 3) @ [8, 9, 10, 1],
                                  [8, 54, 93, 1])


@pytest.mark.parametrize('record', [
    {'scale': 1.}, {'secondary_dy': 1.}, {'secondary_dy': float('inf'), 'secondary_dx': 0.},
    {'align_tile_transforms': '[broken'}, {'align_tile_transforms': {}},
    {'align_tile_transforms': [[1, 2, 3]]},
    {'align_tile_transforms': [[1, 2, 1, 0, 0, 1, 0, 0], [1, 2, 1, 0, 0, 1, 0, 0]]},
    {'align_tile_transforms': [[1.5, 2, 1, 0, 0, 1, 0, 0]]},
    {'align_tile_transforms': [[1, 2, 1, 0, 0, float('nan'), 0, 0]]},
])
def test_invalid_registration_is_refused_before_writing(record):
    """Bad table metadata fails with an actionable registration error."""
    with pytest.raises(OmeZarrError, match='Registration|registration'):
        registration_metadata(record)


@pytest.mark.parametrize('axes', [('y', 'y', 'x'), ('c', 'x'), ('y',), ('unknown', 'y', 'x')])
def test_ambiguous_or_missing_spatial_axes_are_refused(axes):
    """A transform cannot silently use another spatial axis order."""
    with pytest.raises(OmeZarrError, match='axes'):
        registration_metadata({}, axes=axes)


def test_registration_requires_opt_in_version_06():
    """A version mismatch tells the user which settings to change."""
    with pytest.raises(OmeZarrError, match='ome_zarr_ngff_version'):
        registration_metadata({}, ngff_version='0.5')


def test_empty_row_has_no_invented_registration():
    """Absent source tables produce no additional coordinate systems or fits."""
    assert registration_metadata({}) == {'coordinate_systems': [], 'coordinate_transformations': []}


def test_nullable_dataframe_columns_do_not_invent_registration():
    """Optional all-null table groups remain absent after dataframe conversion."""
    row = {key: float("nan") for key in ("refined_scale", "refined_rotation_deg",
        "refined_offset_y", "refined_offset_x", "secondary_dy", "secondary_dx",
        "align_tile_transforms", "align_a11", "align_a12", "align_a21", "align_a22",
        "align_ty", "align_tx")}
    assert registration_metadata(row) == {"coordinate_systems": [], "coordinate_transformations": []}


@pytest.mark.parametrize("source", ["periscope_logged", "register"])
def test_t7_applied_919_998_shift_rows_round_trip_without_sign_changes(tmp_path, source):
    """Use the applied dy/dx table columns, rather than the opposite logged sign."""
    dy, dx = np.nextafter(-23.0, -24.0), np.nextafter(7.0, 8.0)
    row = {"dy": dy, "dx": dx, "source": source, "logged_dy": -dy, "logged_dx": -dx}
    path = tmp_path / "painting"
    pixels = np.arange(20, dtype=np.uint16).reshape(4, 5)
    write_ome_zarr(path, pixels, axes=["y", "x"], ngff_version="0.6", shards=False,
                   **registration_metadata(row, axes=("y", "x")))
    transform, = read_ome_zarr(path).transformations
    assert transform["input"] == {"name": "physical"}
    assert transform["output"] == {"name": "sbs"}
    np.testing.assert_array_equal(_bits(transform["translation"]), _bits([dy, dx]))
    np.testing.assert_array_equal(transformation_matrix(transform, 2) @ [0, 0, 1], [dy, dx, 1])


def test_ambiguous_shift_and_placement_tables_are_refused():
    """Independent source fits cannot be silently composed or discarded."""
    with pytest.raises(OmeZarrError, match="one source table"):
        registration_metadata(dict(scale=1., rotation_deg=0., offset_y=0., offset_x=0.,
                                   dy=3., dx=4.))


@pytest.mark.parametrize("inactive", [{"refined_scale": 99.},
                                      {"refined_scale": float("inf")}])
def test_inactive_refinement_diagnostics_do_not_override_or_break_the_seed(inactive):
    """Partial or invalid rejected diagnostics remain unused by OPS and metadata."""
    row = dict(scale=1., rotation_deg=0., offset_y=0., offset_x=0., refined=0, **inactive)
    transform, = registration_metadata(row, axes=("y", "x"))["coordinate_transformations"]
    np.testing.assert_array_equal(transformation_matrix(transform, 2) @ [3, 4, 1], [3, 4, 1])


@pytest.mark.parametrize("active", [{"refined_scale": 2.},
    {"refined_scale": 2., "refined_rotation_deg": float("inf"),
     "refined_offset_y": 0., "refined_offset_x": 0.}])
def test_invalid_active_refinement_is_refused_without_using_the_seed(active):
    """An accepted but incomplete or non-finite placement cannot silently fall back."""
    row = dict(scale=1., rotation_deg=0., offset_y=0., offset_x=0., refined=1, **active)
    with pytest.raises(OmeZarrError, match="Active registration"):
        registration_metadata(row)



def test_registration_runtime_sources_match_their_arrival_contract(extract_backend_sources):
    """Every registration diagnostic and setting source has an exact English binding."""
    from spacr.ops_settings import OPS_TOOLTIPS
    from spacr.qt.screens.settings_model import _humanize, _strip_type_prefix

    root = Path(__file__).resolve().parents[1]
    rows = json.loads((root / "tests/data/release_contracts"
        / "1007c_ome_zarr_registration_runtime_arrivals_2026-11-21.json").read_text())
    by_key = {(row["table"], row["key"]): row for row in rows}
    assert len(by_key) == len(rows) == 13
    for row in rows:
        assert hashlib.sha256(row["source"].encode()).hexdigest() == row["source_sha256"]
    actual = extract_backend_sources("ome_zarr_registration.py",
        (root / "spacr/ome_zarr_registration.py").read_text())
    expected = {row["source"] for row in rows if row["table"] == "ui"
                and row["file"] == "spacr/ome_zarr_registration.py"}
    assert len(expected) == 8
    assert actual == expected
    writer_sources = extract_backend_sources("image_store.py",
        (root / "spacr/image_store.py").read_text())
    writer_expected = {row["source"] for row in rows if row["table"] == "ui"
                       and row["file"] == "spacr/image_store.py"}
    assert len(writer_expected) == 3
    assert writer_expected <= writer_sources
    key = "ome_zarr_store_registration"
    assert by_key[("setting_labels", key)]["source"] == _humanize(key)
    assert by_key[("setting_tooltips", key)]["source"] == _strip_type_prefix(OPS_TOOLTIPS[key])


@pytest.mark.parametrize("with_groups", [False, True])
def test_global_affine_survives_empty_or_populated_tile_groups(tmp_path, with_groups):
    """The stored global fallback and optional tile fit each retain their exact values."""
    values = [np.nextafter(1., 2.), -0.0, .125, np.nextafter(.75, 1.),
              np.nextafter(5., 6.), -7.123456789012345]
    tile = [np.nextafter(.98, 1.), .02, -.01, 1.03, 22., 24.]
    row = dict(zip(("align_a11", "align_a12", "align_a21", "align_a22", "align_ty", "align_tx"), values))
    row["align_tile_transforms"] = json.dumps([[4, 7, *tile]] if with_groups else [])
    pixels = np.arange(20, dtype=np.uint16).reshape(4, 5)
    path = tmp_path / "with_global"
    write_ome_zarr(path, pixels, axes=["y", "x"], ngff_version="0.6", shards=False,
                   **registration_metadata(row, axes=("y", "x")))
    opened = read_ome_zarr(path)
    transforms = {item["output"]["name"]: item for item in opened.transformations}
    assert set(transforms) == ({"tile_global", "tile_global_4"} if with_groups else {"tile_global"})
    affine = transforms["tile_global"]["affine"]
    np.testing.assert_array_equal(_bits([*affine[0][:2], *affine[1][:2], affine[0][2], affine[1][2]]),
                                  _bits(values))
    if with_groups:
        affine = transforms["tile_global_4"]["affine"]
        np.testing.assert_array_equal(_bits([*affine[0][:2], *affine[1][:2], affine[0][2], affine[1][2]]),
                                      _bits(tile))
    np.testing.assert_array_equal(opened.read(), pixels)


def test_plate_registration_and_labels_keep_the_same_chunks(tmp_path):
    """The optional HCS hook adds matching image/label frames without pixel changes."""
    from spacr.image_store import PlateWriter

    pixels = np.arange(2 * 7 * 9, dtype=np.uint16).reshape(2, 7, 9)
    labels = np.zeros((7, 9), dtype=np.uint32)
    labels[2:5, 3:7] = 1
    row = dict(scale=0.5, rotation_deg=0., offset_y=7., offset_x=-4.)
    plain = PlateWriter(tmp_path / 'plain', 'plate', ngff_version='0.6')
    stored = PlateWriter(tmp_path / 'stored', 'plate', ngff_version='0.6',
                         registration={1: row})
    key = plain.write_field('A01', 1, pixels, {'cells': labels})
    assert stored.write_field('A01', 1, pixels, {'cells': labels}) == key
    plain_hashes = _chunk_hashes(plain.path)
    stored_hashes = _chunk_hashes(stored.path)
    chunks = {key: value for key, value in plain_hashes.items() if '/c/' in key}
    assert chunks
    assert {key: value for key, value in stored_hashes.items() if '/c/' in key} == chunks
    image = read_ome_zarr(stored.path / key)
    label = read_ome_zarr(stored.path / key / 'labels' / 'cells')
    assert list(image.transformations) == registration_metadata(row)['coordinate_transformations']
    assert list(label.transformations) == registration_metadata(row, axes=('y', 'x'))['coordinate_transformations']
    np.testing.assert_array_equal(image.read(), pixels)
    np.testing.assert_array_equal(label.read(), labels)


def test_plate_registration_refuses_a_version_mismatch_before_creation(tmp_path):
    """Registration cannot silently disappear into a 0.5 plate store."""
    from spacr.image_store import PlateWriter

    destination = tmp_path / 'not_created'
    with pytest.raises(ValueError, match='0.6'):
        PlateWriter(destination, 'plate', ngff_version='0.5',
                    registration={1: {'dy': 2., 'dx': 3.}})
    assert not destination.exists()


@pytest.mark.parametrize('enabled', [False, True])
def test_resumed_plate_cannot_silently_change_registration_mode(tmp_path, enabled):
    """A completed field cannot be skipped under changed registration options."""
    from spacr.image_store import PlateWriter

    rows = {1: {'dy': 2., 'dx': 3.}}
    writer = PlateWriter(tmp_path, 'plate', ngff_version='0.6',
                          registration=rows if enabled else None)
    writer.write_field('A01', 1, np.arange(12, dtype=np.uint16).reshape(3, 4))
    before = _chunk_hashes(writer.path)
    with pytest.raises(ValueError, match='registration mode'):
        PlateWriter(tmp_path, 'plate', ngff_version='0.6',
                    registration=None if enabled else rows)
    assert _chunk_hashes(writer.path) == before


def test_bad_registration_cannot_delete_a_committed_field(tmp_path):
    """Table validation happens before the existing field is removed."""
    from spacr.image_store import PlateWriter

    data = np.arange(12, dtype=np.uint16).reshape(3, 4)
    writer = PlateWriter(tmp_path, 'plate', ngff_version='0.6',
                          registration={1: {'dy': 2., 'dx': 3.}})
    writer.write_field('A01', 1, data)
    before = _chunk_hashes(writer.path)
    bad = PlateWriter(tmp_path, 'plate', ngff_version='0.6',
                       registration={1: {'dy': 2.}})
    with pytest.raises(OmeZarrError, match='complete'):
        bad.write_field('A01', 1, data + 1)
    assert _chunk_hashes(writer.path) == before


def test_registration_rejects_non_pixel_spacing_before_creation(tmp_path):
    """Pixel-coordinate tables cannot be attached to micrometre coordinates directly."""
    from spacr.image_store import PlateWriter
    from spacr.layers import Spacing

    destination = tmp_path / 'not_created'
    with pytest.raises(ValueError, match='pixel coordinates'):
        PlateWriter(destination, 'plate', ngff_version='0.6',
                    spacing=Spacing.from_map({'y': .65, 'x': .65}, units='um'),
                    registration={1: {'dy': 2., 'dx': 3.}})
    assert not destination.exists()
