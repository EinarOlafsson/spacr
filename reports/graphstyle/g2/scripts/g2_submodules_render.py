"""Render frozen submodule examples from a selected source tree under root's scheduler.

No model training, inference or assay reconstruction is performed. Nested renderers
are compiled directly from their unchanged source AST with the real module globals;
the selected source location and hashes are recorded. Run this script separately
against the untouched checkpoint and the candidate tree, with separate outputs.
"""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import random
import sys


def digest(data):
    """Hash exact bytes rather than formatted numeric approximations."""
    return hashlib.sha256(data).hexdigest()


def main():
    """Load one source tree, draw actual production owners and save both ordinary formats."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--outputdir', required=True)
    parser.add_argument('--theme', choices=['light', 'dark', 'both'], default='both')
    case_names = ('training_images', 'test_images', 'apply_images', 'read_lines', 'score_heatmap',
                  'importance', 'permutation', 'legacy_radar', 'replication_sparse', 'invasion_sparse',
                  'thresholds', 'guide_correlation', 'guide_effects')
    extra_case_names = ('thresholds_default12',)
    parser.add_argument('--cases', nargs='+', choices=case_names + extra_case_names, default=None)
    parser.add_argument('--baseline-report')
    args = parser.parse_args()
    if args.cases is not None and len(set(args.cases)) != len(args.cases):
        parser.error('--cases must not repeat a case')
    selected_cases = (list(case_names) if args.cases is None else
                      [name for name in case_names + extra_case_names if name in args.cases])
    baseline_path = Path(args.baseline_report).resolve() if args.baseline_report else None
    if baseline_path is not None and not baseline_path.is_file():
        parser.error('--baseline-report must name an existing report')
    baseline_bytes = baseline_path.read_bytes() if baseline_path is not None else None
    baseline = json.loads(baseline_bytes) if baseline_bytes is not None else None
    source_root = Path(args.source_root).resolve()
    output = Path(args.outputdir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = output / 'config'
    config.mkdir(exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['XDG_CONFIG_HOME'] = str(config)
    os.environ['MPLCONFIGDIR'] = str(output / 'matplotlib')
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    os.environ.setdefault('MPLBACKEND', 'Agg')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    sys.path.insert(0, str(source_root))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from PIL import Image
    from matplotlib.colors import to_rgba
    from PySide6.QtCore import QSettings
    QSettings.setPath(QSettings.Format.NativeFormat, QSettings.Scope.UserScope, str(config))
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(config))
    from spacr import submodules as module
    from spacr.figures import style
    from spacr.qt import preferences

    if os.environ.get('SPACR_PIPELINE_FIGURES') == '1':
        raise RuntimeError('Theme evidence must run outside pipeline_figures')

    def json_scalar(value):
        """Unbox NumPy scalars without converting exact numeric values to strings."""
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f'Unsupported evidence value: {type(value).__name__}')

    if Path(module.__file__).resolve() != source_root / 'spacr/submodules.py':
        raise RuntimeError('The imported plotting source does not match --source-root')
    source_path = Path(module.__file__)
    source = source_path.read_text(encoding='utf-8')
    tree = ast.parse(source, filename=str(source_path))
    module._show_if_interactive = lambda: None
    original_save = module.save_figure
    captured = []
    producer_writes = []

    def png_outer_pixels(path, expected_ground):
        """Read the actual saved RGBA corners instead of inferring them from figure metadata."""
        expected = [int(round(channel * 255)) for channel in to_rgba(expected_ground)]
        with Image.open(path) as saved, saved.convert('RGBA') as rgba:
            width, height = rgba.size
            positions = {'top_left': (0, 0), 'top_right': (width - 1, 0),
                         'bottom_left': (0, height - 1), 'bottom_right': (width - 1, height - 1)}
            corners = {name: list(rgba.getpixel(point)) for name, point in positions.items()}
        return {'size_pixels': [width, height], 'corners_rgba': corners,
                'requested_ground_rgba': expected,
                'all_corners_match_requested_ground': all(value == expected for value in corners.values())}

    def capture_save(figure, path, **kwargs):
        """Retain nested writers' real figures while using the normal production writer."""
        captured.append(figure)
        kwargs.update(save_mode='screen', announce_colours=False)
        saving_ground = matplotlib.rcParams['savefig.facecolor']
        written = original_save(figure, path, **kwargs)
        saved_path = Path(written)
        entry = {'path': str(saved_path), 'sha256': digest(saved_path.read_bytes()),
                 'savefig_facecolor_before_writer': saving_ground}
        if saved_path.suffix.lower() == '.png':
            entry['png_outer_pixels'] = png_outer_pixels(saved_path, ground)
        producer_writes.append(entry)
        return written

    module.save_figure = capture_save

    def nested(outer, name):
        """Compile one exact nested owner, without executing its model/analysis parent."""
        parent = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == outer)
        node = next(node for node in parent.body if isinstance(node, ast.FunctionDef) and node.name == name)
        namespace = dict(vars(module))
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source_path), 'exec'), namespace)
        provenance[name] = {'parent': outer, 'line': node.lineno,
                            'sha256': digest(ast.get_source_segment(source, node).encode())}
        return namespace[name]

    def importance(table, permutation=False):
        """Use the actual extracted baseline rendering block or the candidate pure owner."""
        if hasattr(module, '_feature_importance_figure'):
            return module._feature_importance_figure(table, caption='Frozen synthetic importance estimates.',
                quantity='Mean permutation importance' if permutation else 'Random-forest importance')
        variable = 'top_perm_importance_df' if permutation else 'top_feature_importance_df'
        parent = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'interpret_vision_model')
        block = next(node for node in ast.walk(parent) if isinstance(node, ast.With)
                     and any(isinstance(child, ast.Name) and child.id == variable for child in ast.walk(node)))
        namespace = dict(vars(module), settings={'top_features': len(table)}, **{variable: table})
        exec(compile(ast.Module(body=[block], type_ignores=[]), str(source_path), 'exec'), namespace)
        provenance[variable] = {'parent': 'interpret_vision_model', 'line': block.lineno,
                                'sha256': digest(ast.get_source_segment(source, block).encode())}
        return plt.gcf()

    def array_record(values, path):
        """Persist exact numeric arrays and their masks with dtype and byte hashes."""
        masked = np.ma.asarray(values)
        array = np.asarray(masked.data)
        mask = np.ma.getmaskarray(masked)
        np.save(path.with_suffix('.npy'), array, allow_pickle=False)
        np.save(path.with_name(path.name + '_mask').with_suffix('.npy'), mask, allow_pickle=False)
        return {'shape': list(array.shape), 'dtype': array.dtype.str,
                'sha256': digest(np.ascontiguousarray(array).tobytes()),
                'mask_sha256': digest(mask.tobytes())}

    def state_record():
        """Record both legacy random streams, which cosmetic changes must preserve."""
        state = np.random.get_state()
        return {'numpy': digest(state[1].tobytes() + repr((state[0], *state[2:])).encode()),
                'python': digest(repr(random.getstate()).encode())}

    def theme_observation(figure):
        """Read actual figure and axes chrome without changing any requested or saved appearance."""
        return {'figure_background': list(figure.get_facecolor()), 'axes': [
            {'index': index, 'background': list(axis.get_facecolor()),
             'is_colourbar': getattr(axis, '_colorbar', None) is not None,
             'xlabel': list(to_rgba(axis.xaxis.label.get_color())),
             'ylabel': list(to_rgba(axis.yaxis.label.get_color())),
             'tick_foregrounds': [list(to_rgba(tick.get_color())) for tick in
                                  (*axis.get_xticklabels(), *axis.get_yticklabels()) if tick.get_visible()],
             'spines': {name: list(spine.get_edgecolor()) for name, spine in axis.spines.items()}}
            for index, axis in enumerate(figure.axes)]}

    def image_appearance(image, path):
        """Record the actual normalization and categorical lookup table at the observation stage."""
        ramp = image.get_cmap()
        return {'clim': list(image.get_clim()),
            'normalization': {'class': type(image.norm).__name__, 'vmin': image.norm.vmin,
                              'vmax': image.norm.vmax, 'clip': image.norm.clip,
                              'gamma': getattr(image.norm, 'gamma', 1.0)},
            'alpha': image.get_alpha(), 'origin': image.origin,
            'extent': list(image.get_extent()), 'interpolation': image.get_interpolation(),
            'cmap': {'name': ramp.name, 'N': ramp.N,
                     'bad': ramp.get_bad().tolist(), 'under': ramp.get_under().tolist(),
                     'over': ramp.get_over().tolist(),
                     'lut': array_record(ramp(np.arange(ramp.N)), path)}}

    def record_figure(figure, folder):
        """Keep science separate from appearance and run the actual shared diagnostic observer."""
        folder.mkdir(parents=True, exist_ok=True)
        record = {'source': {}, 'artists': [], 'files': {},
                  'actual_theme': theme_observation(figure)}
        table = getattr(figure, '_spacr_data', None)
        if isinstance(table, pd.DataFrame):
            path = folder / 'registered_data.csv'
            table.to_csv(path)
            record['source']['table'] = {'sha256': digest(path.read_bytes()),
                'columns': [str(value) for value in table.columns], 'index': [str(value) for value in table.index],
                'dtypes': [str(value) for value in table.dtypes]}
            numeric = table.select_dtypes(include='number')
            if numeric.shape[1]:
                record['source']['numeric'] = array_record(numeric.to_numpy(), folder / 'registered_numeric')
        pictures = getattr(figure, '_spacr_image', None)
        if pictures is not None:
            record['source']['images'] = [array_record(value, folder / f'source_image_{index}')
                                           for index, value in enumerate(pictures)]
        for index, axis in enumerate(figure.axes):
            if getattr(axis, '_colorbar', None) is not None:
                continue
            numerical = {'title': axis.get_title(), 'xlabels': [tick.get_text() for tick in axis.get_xticklabels()],
                         'lines': [], 'bar_heights': [float(item.get_height()) for item in axis.patches
                                                      if hasattr(item, 'get_height')], 'images': []}
            for line_index, line in enumerate(axis.lines):
                if len(line.get_xdata()):
                    numerical['lines'].append({'label': line.get_label(),
                        'xy': array_record(line.get_xydata(), folder / f'axis_{index}_line_{line_index}')})
            for image_index, image in enumerate(axis.images):
                image_record = array_record(image.get_array(), folder / f'axis_{index}_image_{image_index}')
                image_record['appearance'] = image_appearance(image,
                    folder / f'axis_{index}_image_{image_index}_cmap')
                numerical['images'].append(image_record)
            record['artists'].append(numerical)
        diagnostic = style.finalize_figure(figure)
        record['diagnostics'] = {'compliant': diagnostic.compliant, 'checks': list(diagnostic.checks),
                                 'problems': list(diagnostic.problems)}
        record['canvas_inches'] = figure.get_size_inches().tolist()
        record['producer_before_write'] = {key: record[key] for key in
                                            ('actual_theme', 'diagnostics', 'canvas_inches')}
        record['artist_observation_stage'] = 'producer before normal writer'
        for extension in ('png', 'pdf'):
            saving_ground = matplotlib.rcParams['savefig.facecolor']
            filename = original_save(figure, str(folder / f'figure.{extension}'), fmt=extension,
                                     dpi=300, save_mode='screen', announce_colours=False)
            path = Path(filename)
            diagnostic = style.finalize_figure(figure)
            record['actual_theme'] = theme_observation(figure)
            record['diagnostics'] = {'compliant': diagnostic.compliant, 'checks': list(diagnostic.checks),
                                     'problems': list(diagnostic.problems)}
            record['canvas_inches'] = figure.get_size_inches().tolist()
            record['files'][extension] = {'path': str(path), 'sha256': digest(path.read_bytes()),
                'observation_stage': 'after normal writer; no evidence repaint',
                'savefig_facecolor_before_writer': saving_ground,
                'actual_theme': record['actual_theme'], 'diagnostics': record['diagnostics'],
                'canvas_inches': record['canvas_inches'],
                'image_appearance': [{'axes': index, 'image': image_index,
                    **image_appearance(image, folder / f'after_{extension}_axis_{index}_image_{image_index}_cmap')}
                    for index, axis in enumerate(figure.axes) for image_index, image in enumerate(axis.images)]}
            if extension == 'png':
                record['files'][extension]['png_outer_pixels'] = png_outer_pixels(path, ground)
        return record

    provenance = {}
    report = {'source_root': str(source_root), 'source_sha256': digest(source_path.read_bytes()),
              'provenance': provenance, 'themes': {}, 'selected_cases': selected_cases,
              'theme_route': 'Isolated public figure preferences, actual producer, normal screen-mode writer; no evidence repaint.',
              'scope': 'Frozen renderer inputs only; no training, inference, assay pipeline or new statistical tests.'}
    themes = ['light', 'dark'] if args.theme == 'both' else [args.theme]
    for theme in themes:
        ground, ink = ('#FFFFFF', '#000000') if theme == 'light' else ('#181818', '#F0F0F0')
        preferences.set_figure_style({'background': ground, 'foreground': ink})
        preferences.set_figure_style_per_graph({})
        preferences.set_figure_colors(ground, ink)
        preferences.set_figure_format('png')
        theme_report = {}
        report['themes'][theme] = theme_report
        base = output / theme
        base.mkdir(exist_ok=True)
        with matplotlib.rc_context():
            image = np.arange(120, dtype=np.float32).reshape(10, 12)
            labels = np.zeros((10, 12), dtype=np.int32)
            labels[2:6, 3:7] = 1
            labels[6:9, 7:11] = 3
            flow = [np.dstack([image / 120] * 3), np.zeros((2, 10, 12), dtype=np.float32), image / 120]
            signals = pd.DataFrame({'prc': ['well_b'] * 10 + ['well_a'] * 10,
                'outside_intensity': [1., 2., 3., 4., 5., 8., 9., 10., 11., np.nan] * 2})
            wells = pd.DataFrame({'prc': ['well_b', 'well_a'], 'threshold_median': [6., 6.5],
                'reference_threshold_median': [5., 5.5], 'threshold_source': ['fixed', 'control'],
                'bimodality_coefficient': [.61, .72], 'n_total': [10, 10]})
            lines = pd.DataFrame({'mixture': [.1, .3, .6, .9], 'read_fraction': [.12, .28, .62, .88],
                                  'class_fraction': [.08, .33, .57, .92]})
            scores = pd.DataFrame({'plateID': ['plate1'] * 3, 'rowID': ['r3', 'r1', 'r2'],
                'columnID': ['c3'] * 3, 'model_score': [.9, .1, np.nan], 'read_fraction': [.8, .2, .5]})
            estimates = pd.DataFrame({'feature': ['feature_z', 'feature_a', 'feature_m'], 'importance': [.6, .3, .1]})
            guide_rows = pd.DataFrame([{'prc': f'well_{index}', 'grna': name, 'fraction': value}
                for index, number in enumerate([.1, .2, .3, .4])
                for name, value in [('g1', number), ('g2', 1 - number), ('g3', 2 * number), ('g4', 3 * (1 - number))]])
            for name in selected_cases:
                plt.close('all')
                producer_writes.clear()
                np.random.seed(20261123)
                random.seed(20261123)
                start_state = state_record()
                folder = base / name
                folder.mkdir(exist_ok=True)
                tables = {}
                if name == 'training_images':
                    module.plot_cellpose_batch([image, image + 10], [labels, labels])
                    figure = plt.gcf()
                elif name in ('test_images', 'apply_images'):
                    captured.clear()
                    if name == 'test_images':
                        nested('test_cellpose_model', 'plot_cellpose_resilts')(0, 0, str(folder), image, labels, labels, flow)
                    else:
                        nested('apply_cellpose_model', 'plot_cellpose_result')(0, 0, str(folder), image, labels, flow)
                    figure = captured[-1]
                elif name == 'read_lines':
                    figure = nested('compare_reads_to_scores', 'plot_line')(lines, 'mixture',
                        ['read_fraction', 'class_fraction'], save_path=str(folder / 'line_direct.png'))
                elif name == 'score_heatmap':
                    figure = nested('generate_score_heatmap', 'plot_multi_channel_heatmap')(scores.copy(), 'c3')
                elif name in ('importance', 'permutation'):
                    figure = importance(estimates, permutation=name == 'permutation')
                elif name == 'legacy_radar':
                    nested('interpret_vision_model', 'create_extended_radar_plot')([.6, .3, .1], ['z', 'a', 'm'], 'Frozen legacy values')
                    figure = plt.gcf()
                elif name in ('replication_sparse', 'invasion_sparse'):
                    column = 'replication_bucket' if name == 'replication_sparse' else 'invasion_class'
                    categories = ['1', '2'] if name == 'replication_sparse' else ['attached', 'invaded']
                    frame = pd.DataFrame({'condition': ['only'] * 3, 'prc': ['w1'] * 3,
                        column: pd.Categorical([categories[0], categories[1], categories[1]], categories=categories)})
                    owner = module._replication_stacked_bars if name == 'replication_sparse' else module._invasion_stacked_bars
                    kwargs = {} if name == 'replication_sparse' else {'denominators': {'only': 3}}
                    stats, pairwise, figure = owner({}, frame, 'condition', 'prc', 'object', 'viridis', 'Frozen composition', **kwargs)
                    tables = {'input': frame, 'statistics': stats, 'pairwise': pairwise}
                elif name == 'thresholds':
                    figure = module._invasion_threshold_panels(signals, wells, max_panels=2)
                    tables = {'signals': signals, 'wells': wells}
                elif name == 'thresholds_default12':
                    names = [f'well_{index:02d}' for index in range(12, 0, -1)]
                    signals12 = pd.DataFrame({'prc': np.repeat(names, 6),
                        'outside_intensity': [1., 2., 3., 4., 5., np.nan] * 12})
                    wells12 = pd.DataFrame({'prc': names, 'threshold_median': [3.5] * 12,
                        'reference_threshold_median': [2.] * 12, 'threshold_source': ['fixed'] * 12,
                        'bimodality_coefficient': [.72] * 12, 'n_total': [6] * 12})
                    figure = module._invasion_threshold_panels(signals12, wells12)
                    tables = {'signals': signals12, 'wells': wells12}
                elif name == 'guide_correlation':
                    correlation = nested('post_regression_analysis', '_analyze_and_visualize_grna_correlation')(
                        guide_rows, ['g1', 'g2', 'g3', 'g4'], str(folder), save=True)
                    figure = plt.gcf()
                    tables = {'input': guide_rows, 'correlation': correlation}
                else:
                    correlation = guide_rows.pivot_table(index='prc', columns='grna', values='fraction', aggfunc='sum').fillna(0).corr()
                    effects = nested('post_regression_analysis', '_compute_effect_sizes')(
                        correlation, {'g1': .8, 'g2': .2}, str(folder), save=True)
                    figure = plt.gcf()
                    tables = {'correlation': correlation, 'effects': effects.to_frame()}
                result = record_figure(figure, folder)
                result['producer_writes'] = list(producer_writes)
                result['requested_theme'] = {'background': ground, 'foreground': ink}
                result['rng_before'] = start_state
                result['rng_after'] = state_record()
                result['tables'] = {}
                for table_name, table in tables.items():
                    path = folder / f'{table_name}.csv'
                    table.to_csv(path)
                    result['tables'][table_name] = {'sha256': digest(path.read_bytes()),
                        'dtypes': [str(value) for value in table.dtypes],
                        'columns': [str(value) for value in table.columns], 'index': [str(value) for value in table.index]}
                if baseline is not None:
                    original = baseline['themes'][theme][name]
                    for field in ('source', 'tables', 'rng_before', 'rng_after'):
                        if result[field] != original[field]:
                            raise AssertionError(f'Baseline mismatch: {theme}/{name}/{field}')
                    if name in ('training_images', 'test_images', 'apply_images'):
                        if [axis['images'] for axis in result['artists']] != [
                                axis['images'] for axis in original['artists']]:
                            raise AssertionError(f'Baseline image mismatch: {theme}/{name}/producer')
                        for extension in ('png', 'pdf'):
                            if result['files'][extension]['image_appearance'] != original['files'][extension]['image_appearance']:
                                raise AssertionError(f'Baseline image mismatch: {theme}/{name}/{extension}')
                    result['baseline_payload_identical'] = True
                for extension in ('png', 'pdf'):
                    if not Path(result['files'][extension]['path']).is_file():
                        raise FileNotFoundError(f'Missing saved output: {theme}/{name}/{extension}')
                expected_producer = {
                    'test_images': 'cellpose_result_000.png',
                    'apply_images': 'cellpose_result_000.png',
                    'read_lines': 'line_direct.png',
                    'guide_correlation': 'correlation_matrix_heatmap.png',
                    'guide_effects': 'effect_sizes_barplot.png',
                }.get(name)
                if expected_producer is not None and not any(
                        Path(item['path']).name == expected_producer and Path(item['path']).is_file()
                        for item in result['producer_writes']):
                    raise FileNotFoundError(f'Missing producer output: {theme}/{name}/{expected_producer}')
                theme_report[name] = result
                plt.close('all')
        if set(theme_report) != set(selected_cases):
            raise RuntimeError(f'Missing requested case outputs for {theme}')
        if baseline is not None:
            report['baseline_comparison'] = {
                'path': str(baseline_path), 'sha256': digest(baseline_bytes),
                'fields': ['source', 'tables', 'rng_before', 'rng_after'],
                'image_appearance': ['producer', 'png', 'pdf'],
                'cases_checked': sum(len(value) for value in report['themes'].values()),
                'all_identical': True,
            }
        (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=True,
                                                     default=json_scalar), encoding='utf-8')
    print(json.dumps({'report': str(output / 'report.json'), 'themes': themes,
                      'figures': sum(len(value) for value in report['themes'].values())}))


if __name__ == '__main__':
    main()
