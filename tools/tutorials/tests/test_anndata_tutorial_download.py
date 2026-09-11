"""The downloadable workaround must remain the exact verified helper."""
from pathlib import Path
import shlex
import zipfile


def test_download_contains_the_checked_helper_and_all_six_distinct_commands():
    root = Path(__file__).resolve().parents[3]
    examples = root / 'docs/source/_extra/tutorials/examples'
    helper = (root / 'tools/tutorials/anndata_missing_metadata_example.py').read_bytes()
    assert (examples / 'anndata_missing_metadata_example.py').read_bytes() == helper
    with zipfile.ZipFile(examples / 'AnnData_API_workaround.zip') as archive:
        assert set(archive.namelist()) == {'anndata_missing_metadata_example.py', 'README.txt'}
        assert archive.testzip() is None
        assert archive.read('anndata_missing_metadata_example.py') == helper
        text = archive.read('README.txt').decode()
        assert text == (examples / 'AnnData_API_workaround_README.txt').read_text()
    commands = [shlex.split(line) for line in text.splitlines() if line.startswith('python ')]
    assert len(commands) == 6
    observed, outputs = set(), set()
    for command in commands:
        assert command[:2] == ['python', 'anndata_missing_metadata_example.py']
        opts = dict(zip(command[2::2], command[3::2]))
        assert opts['--source'] == 'measurements.db'
        observed.add((opts.get('--single-table', ''), opts['--nan-policy']))
        output = Path(opts['--out'])
        assert output.parent == Path('results') and output.suffix == '.h5ad'
        outputs.add(output)
    assert len(outputs) == 6
    assert observed == {('', 'keep'), ('cell', 'keep'), ('cell', 'mean'),
                        ('cell', 'drop_features'), ('cell', 'drop_objects'), ('nucleus', 'keep')}
