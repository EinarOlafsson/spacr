"""An occupied output leaf must not redirect consolidation through a link."""
import pytest

from spacr import folder_consolidation as fc


def linked_tree(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    original = source / 'image.tif'
    original.write_bytes(b'original microscopy bytes')
    outside = tmp_path / 'unintended'
    output = tmp_path / 'source_renamed'
    try:
        output.symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(str(error))
    return source, original, output, outside


def test_dangling_default_candidate_is_numbered_as_occupied(tmp_path):
    source, original, output, outside = linked_tree(tmp_path)
    assert fc.unused_output_folder(tmp_path, output.name) == tmp_path / 'source_renamed_2'
    assert output.is_symlink() and not outside.exists()
    assert original.read_bytes() == b'original microscopy bytes'


def test_default_consolidation_skips_dangling_leaf_without_following_it(tmp_path):
    source, original, output, outside = linked_tree(tmp_path)
    result = fc.consolidate_folder(source, log=lambda *_: None)
    assert result.output == tmp_path / 'source_renamed_2'
    assert (result.copied, result.failed) == (1, 0)
    assert (result.output / 'source.tif').read_bytes() == original.read_bytes()
    assert output.is_symlink() and not outside.exists()
    assert original.read_bytes() == b'original microscopy bytes'


def test_explicit_dangling_output_is_refused_without_creating_its_target(tmp_path):
    source, original, output, outside = linked_tree(tmp_path)
    with pytest.raises(ValueError, match='Output already exists. Choose a NEW directory:'):
        fc.consolidate_folder(source, output, log=lambda *_: None)
    assert output.is_symlink() and output.readlink() == outside
    assert not outside.exists()
    assert original.read_bytes() == b'original microscopy bytes'


def test_explicit_existing_output_link_retains_existing_refusal_and_data(tmp_path):
    source, original, output, outside = linked_tree(tmp_path)
    outside.mkdir()
    private = outside / 'private.txt'
    private.write_bytes(b'private target bytes')
    with pytest.raises(ValueError, match='Output already exists. Choose a NEW directory:'):
        fc.consolidate_folder(source, output, log=lambda *_: None)
    assert output.is_symlink()
    assert private.read_bytes() == b'private target bytes'
    assert original.read_bytes() == b'original microscopy bytes'
    assert sorted(p.name for p in outside.iterdir()) == ['private.txt']
