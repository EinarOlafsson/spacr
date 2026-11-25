"""Replay prepared Measure creator inputs through the normal writer after root approval."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
from hashlib import sha256


def load_fixture_module(path):
    """Load only the fixed input definitions from the current focused-test source."""
    specification = importlib.util.spec_from_file_location('g2_measure_inputs', path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def json_scalar(value):
    """Unbox NumPy scalars without stringifying or rounding scientific coordinates."""
    import numpy as np

    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Unsupported JSON value type: {type(value).__name__}')


def numeric_artists(figure):
    """Record drawn numerical coordinates separately from frozen scientific inputs."""
    import numpy as np
    from matplotlib.collections import LineCollection, PathCollection, PolyCollection

    axes = []
    for ax in figure.axes:
        if getattr(ax, '_colorbar', None) is not None:
            continue
        record = {'lines': [], 'points': [], 'bands': [], 'segments': [], 'images': [], 'bars': []}
        for line in ax.lines:
            record['lines'].append({'label': line.get_label(),
                                    'x': np.asarray(line.get_xdata()).tolist(),
                                    'y': np.asarray(line.get_ydata()).tolist()})
        for item in ax.collections:
            if isinstance(item, PathCollection):
                record['points'].append({'gid': item.get_gid(), 'offsets': np.ma.asarray(item.get_offsets()).filled(float('nan')).tolist()})
            elif isinstance(item, LineCollection):
                record['segments'].append([np.asarray(segment).tolist() for segment in item.get_segments()])
            elif isinstance(item, PolyCollection):
                record['bands'].append([path.vertices.tolist() for path in item.get_paths()])
        for item in ax.images:
            values = np.ma.asarray(item.get_array())
            record['images'].append({'dtype': str(values.dtype), 'shape': list(values.shape),
                                      'bytes': sha256(values.data.tobytes()).hexdigest(),
                                      'mask': sha256(np.ma.getmaskarray(values).tobytes()).hexdigest(),
                                      'clim': list(item.get_clim())})
        for item in ax.patches:
            if hasattr(item, 'get_height'):
                record['bars'].append({'x': item.get_x(), 'bottom': item.get_y(),
                                       'width': item.get_width(), 'height': item.get_height()})
        axes.append(record)
    return axes


def main():
    """Write actual saved light/dark preference routes and frozen-input receipts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--fixtures', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'tests/test_graphstyle_g2_measure.py')
    parser.add_argument('--case', action='append')
    args = parser.parse_args()
    os.environ.setdefault('MPLBACKEND', 'Agg')
    sys.path.insert(0, str(args.source_root.resolve()))
    import matplotlib.pyplot as plt
    import spacr.measure as measure
    import spacr.figures.style as style
    from spacr.plot import save_figure
    from spacr.tabular import write_table
    from spacr.qt.preferences import set_figure_style, set_figure_colors, set_figure_format
    from matplotlib.colors import to_hex

    if not os.environ.get('XDG_CONFIG_HOME'):
        raise RuntimeError('Root must provide an isolated XDG_CONFIG_HOME before the preference evidence run')
    if os.environ.get('SPACR_PIPELINE_FIGURES') == '1':
        raise RuntimeError('This explicit preference route must run outside pipeline_figures')

    fixture = load_fixture_module(args.fixtures)
    cases = fixture.prepared_measure_cases(measure)
    selected = args.case or list(cases)
    receipt = {'source_root': str(args.source_root.resolve()),
               'loaded_measure': str(Path(measure.__file__).resolve()),
               'measure_sha256': sha256(Path(measure.__file__).read_bytes()).hexdigest(), 'targets': {},
               'theme_scope': 'public saved figure preferences, normal creator and screen-mode writer',
               'scientific_reruns': False}
    for target, ground, ink in (('light', 'white', 'black'), ('dark', '#181818', '#F0F0F0')):
        target_records = {}
        set_figure_style({'background': ground, 'foreground': ink})
        set_figure_colors(ground, ink)
        set_figure_format('png')
        for name in selected:
            function_name, arguments, keywords = cases[name]
            before = fixture.input_record(arguments)
            folder = args.output / target / name
            folder.mkdir(parents=True, exist_ok=False)
            (folder / 'inputs.json').write_text(json.dumps(before, sort_keys=True, indent=2, default=json_scalar), encoding='utf-8')
            figure = getattr(measure, function_name)(*arguments, **keywords)
            try:
                assert fixture.input_record(arguments) == before, f'{name}: prepared inputs changed'
                frame = getattr(figure, '_spacr_data', None)
                if frame is not None:
                    write_table(frame, str(folder / 'data.csv'), canonicalise=False)
                payload = {'table': fixture.input_record(frame),
                           'images': fixture.input_record(getattr(figure, '_spacr_image', None))}
                (folder / 'payload.json').write_text(json.dumps(payload, sort_keys=True, indent=2, default=json_scalar), encoding='utf-8')
                (folder / 'numerical_artists.json').write_text(json.dumps(numeric_artists(figure), sort_keys=True, indent=2, default=json_scalar), encoding='utf-8')
                written = {}
                for extension in ('png', 'pdf'):
                    output = save_figure(figure, str(folder / f'{name}.{extension}'), fmt=extension,
                                         dpi=300, save_mode='screen', bbox_inches='tight')
                    written[extension] = {'path': output, 'sha256': sha256(Path(output).read_bytes()).hexdigest()}
                diagnostics = style.finalize_figure(figure)
                record = {'creator': function_name, 'inputs_sha256': sha256((folder / 'inputs.json').read_bytes()).hexdigest(),
                          'payload_sha256': sha256((folder / 'payload.json').read_bytes()).hexdigest(),
                          'data_sha256': sha256((folder / 'data.csv').read_bytes()).hexdigest() if frame is not None else None,
                          'requested_theme': {'background': ground, 'foreground': ink},
                          'actual_theme': {'figure': to_hex(figure.get_facecolor(), keep_alpha=True),
                                           'axes': [to_hex(ax.get_facecolor(), keep_alpha=True) for ax in figure.axes],
                                           'text': [to_hex(ax.xaxis.label.get_color(), keep_alpha=True) for ax in figure.axes]},
                          'files': written, 'diagnostics': {'problems': list(diagnostics.problems), 'checks': list(diagnostics.checks)}}
                (folder / 'receipt.json').write_text(json.dumps(record, sort_keys=True, indent=2, default=json_scalar), encoding='utf-8')
                target_records[name] = record
            finally:
                plt.close(figure)
        receipt['targets'][target] = target_records
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'receipt.json').write_text(json.dumps(receipt, sort_keys=True, indent=2, default=json_scalar), encoding='utf-8')


if __name__ == '__main__':
    main()
