"""Prove the tutorial's preservation checks fail against broken helper code."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import types

ROOT = Path(__file__).resolve().parent


def run():
    path = ROOT / 'tests/test_embeddings_example.py'
    spec = importlib.util.spec_from_file_location('embeddings_example_tests', path)
    tests = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tests)
    source_path = ROOT / 'embeddings_example.py'
    source = source_path.read_text()
    mutants = [
        ('existing_destination', '    if output.exists():', '    if False:',
         tests.test_existing_output_is_really_preserved, None),
        ('npy_values', '    np.testing.assert_array_equal(saved, values)', '    pass',
         tests.test_reopens_every_value_and_order_with_positive_counterpart, 'npy'),
        ('ordered_ids', "    if frame['object_id'].tolist() != identities or list(frame.columns[1:]) != list(names):",
         '    if False:', tests.test_reopens_every_value_and_order_with_positive_counterpart, 'row'),
        ('csv_values', '    np.testing.assert_array_equal(frame.iloc[:, 1:].to_numpy(dtype=np.float32), values)',
         '    pass', tests.test_reopens_every_value_and_order_with_positive_counterpart, 'csv_value'),
    ]
    results = []
    for name, old, new, test, parameter in mutants:
        if source.count(old) != 1:
            raise ValueError('Mutation anchor must match exactly once: ' + name)
        module = types.ModuleType('embeddings_mutant')
        module.__file__ = str(source_path)
        exec(compile(source.replace(old, new), str(source_path), 'exec'), module.__dict__)
        tests.helper = module
        with tempfile.TemporaryDirectory(prefix='embeddings-mutant-') as folder:
            try:
                test(Path(folder), *([] if parameter is None else [parameter]))
            except BaseException as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                results.append({'mutation': name, 'observed_red': True,
                                'failure': type(exc).__name__ + ': ' + str(exc)})
            else:
                raise AssertionError('Mutation survived: ' + name)
    report = {'helper_sha256': hashlib.sha256(source.encode()).hexdigest(),
              'test_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'source_files_modified': False, 'mutations': results}
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    run()
