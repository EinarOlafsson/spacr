"""Render actual utility creators on identical CPU fixtures in either source tree.

Run only through the maintainer's serial P2CPU1/RAM2 execution queue. This
script writes isolated preferences and fixtures under --output-dir, then
separates exact scientific bytes from intentional presentation differences.
"""

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import random
import sys


def digest(value):
    """Hash bytes without conflating source data with rendered output."""
    return hashlib.sha256(value).hexdigest()


def main():
    """Create CPU fixtures, call production owners and export both formats and themes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = output / 'isolated-preferences'
    config.mkdir(exist_ok=True)
    inherited_pipeline_flag = os.environ.pop('SPACR_PIPELINE_FIGURES', None)
    os.environ.update(SPACR_DEVICE='cpu', CUDA_VISIBLE_DEVICES='',
                      OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                      MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
                      QT_QPA_PLATFORM='offscreen', MPLBACKEND='Agg',
                      XDG_CONFIG_HOME=str(config), MPLCONFIGDIR=str(config / 'matplotlib'))
    sys.path.insert(0, str(args.source_root.resolve()))
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import AnnotationBbox
    import numpy as np
    import pandas as pd
    from PIL import Image
    from PySide6.QtCore import QSettings
    QSettings.setPath(QSettings.Format.NativeFormat, QSettings.Scope.UserScope, str(config))
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(config))
    import statsmodels.api as sm
    from statsmodels.stats.stattools import durbin_watson
    import torch
    torch.set_num_threads(1)
    from spacr import utils as U
    from spacr.figures.style import figure_style, finalize_figure, rc, theme_target
    from spacr.plot import save_figure
    from spacr.qt import preferences
    manifest = {'source_root': str(args.source_root.resolve()), 'cases': {},
                'inherited_pipeline_flag_removed': inherited_pipeline_flag,
                'resolved_themes': {}}
    scientific = {}

    def json_scalar(value):
        """Preserve NumPy scalar types as JSON numbers or booleans without string coercion."""
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f'Object of type {type(value).__name__} is not JSON serializable')

    def array_record(case, name, value):
        """Save exact shape, dtype and bytes, with stable relative names for comparison."""
        array = np.ascontiguousarray(value)
        if array.dtype.hasobject:
            raise TypeError('Scientific arrays must have explicit non-object dtypes')
        path = output / 'scientific' / case / (name + '.npy')
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, array, allow_pickle=False)
        item = {'shape': list(array.shape), 'dtype': array.dtype.str,
                'sha256': digest(array.tobytes())}
        scientific.setdefault(case, {})[name] = item
        return item

    def value_record(case, name, value):
        """Retain ordered identifiers and hexadecimal scalars in the scientific manifest."""
        scientific.setdefault(case, {})[name] = value

    def rng_record(case, when):
        """Track the actual Python, NumPy and CPU Torch generators separately from pixels."""
        value_record(case, 'python_rng_' + when, digest(repr(random.getstate()).encode()))
        state = np.random.get_state()
        value_record(case, 'numpy_rng_' + when,
                     digest(state[1].tobytes() + repr((state[0], *state[2:])).encode()))
        array_record(case, 'torch_rng_' + when, torch.get_rng_state().numpy())

    def artist_data(case, figure, number):
        """Record ordered statistical coordinates, image pixels and thumbnail locations."""
        def visit(ax, key):
            """Visit an axes and its inset images in source order."""
            for index, image in enumerate(ax.images):
                array_record(case, f'{key}_image{index}', image.get_array())
            for index, collection in enumerate(ax.collections):
                array_record(case, f'{key}_scatter{index}', collection.get_offsets())
            for index, line in enumerate(ax.lines):
                array_record(case, f'{key}_line{index}_x', line.get_xdata())
                array_record(case, f'{key}_line{index}_y', line.get_ydata())
            histogram = [(patch.get_x(), patch.get_width(), patch.get_height())
                         for patch in ax.patches
                         if hasattr(patch, 'get_height') and patch.get_transform() != ax.transAxes]
            if histogram and case.rsplit('/', 1)[-1] in ('metrics', 'mlr'):
                array_record(case, key + '_patch_geometry', np.asarray(histogram))
            thumbs = [child for child in ax.get_children() if isinstance(child, AnnotationBbox)]
            for index, thumbnail in enumerate(thumbs):
                array_record(case, f'{key}_thumbnail{index}', thumbnail.offsetbox.get_data())
                array_record(case, f'{key}_thumbnail{index}_xy', np.asarray(thumbnail.xy))
            for index, child in enumerate(ax.child_axes):
                visit(child, f'{key}_inset{index}')
        for index, ax in enumerate(figure.axes):
            visit(ax, f'figure{number}_axes{index}')

    def finish(case):
        """Write after the creator context exits and record actual exported backgrounds."""
        result = []
        for index, number in enumerate(plt.get_fignums()):
            figure = plt.figure(number)
            artist_data(case, figure, index)
            record = {'files': {}, 'canvas_inches': figure.get_size_inches().tolist()}
            record['theme'] = {
                'requested': case.split('/', 1)[0],
                'figure_background': matplotlib.colors.to_hex(figure.get_facecolor(), keep_alpha=True),
                'axes_backgrounds': [matplotlib.colors.to_hex(ax.get_facecolor(), keep_alpha=True)
                                     for ax in figure.axes],
                'axes_label_inks': [matplotlib.colors.to_hex(ax.xaxis.label.get_color(), keep_alpha=True)
                                    for ax in figure.axes],
                'writer_rc_facecolor': matplotlib.rcParams['savefig.facecolor'],
                'writer_rc_transparent': matplotlib.rcParams['savefig.transparent'],
            }
            expected_background = matplotlib.colors.to_hex(bg, keep_alpha=True)
            assert record['theme']['figure_background'] == expected_background, record['theme']
            for fmt in ('png', 'pdf'):
                path = output / 'renders' / case / f'figure{index}.{fmt}'
                path.parent.mkdir(parents=True, exist_ok=True)
                written = Path(save_figure(figure, str(path), fmt=fmt, dpi=300,
                                          save_mode='screen', preferences=False,
                                          announce_colours=False, integrity=False))
                record['files'][fmt] = {'path': str(written.relative_to(output)),
                                        'sha256': digest(written.read_bytes()),
                                        'figure_background_after_write': matplotlib.colors.to_hex(
                                            figure.get_facecolor(), keep_alpha=True),
                                        'axes_backgrounds_after_write': [
                                            matplotlib.colors.to_hex(ax.get_facecolor(), keep_alpha=True)
                                            for ax in figure.axes]}
                if fmt == 'png':
                    with Image.open(written) as image:
                        rgba = image.convert('RGBA')
                        record['files'][fmt]['rgba_sha256'] = digest(rgba.tobytes())
                        record['files'][fmt]['size'] = list(image.size)
                        width, height = image.size
                        locations = {'top_left': (0, 0), 'top_right': (width - 1, 0),
                                     'bottom_left': (0, height - 1),
                                     'bottom_right': (width - 1, height - 1),
                                     'top_middle': (width // 2, 0),
                                     'bottom_middle': (width // 2, height - 1),
                                     'left_middle': (0, height // 2),
                                     'right_middle': (width - 1, height // 2)}
                        outer_pixels = {name: list(rgba.getpixel(position))
                                        for name, position in locations.items()}
                        expected_pixel = [int(round(value * 255))
                                          for value in matplotlib.colors.to_rgba(bg)]
                        corners = ('top_left', 'top_right', 'bottom_left', 'bottom_right')
                        matches = all(outer_pixels[name] == expected_pixel for name in corners)
                        record['files'][fmt].update(
                            outer_pixels_rgba=outer_pixels,
                            expected_background_rgba=expected_pixel,
                            corner_background_matches=matches)
                        assert matches, {'case': case, 'figure': index,
                                         'expected': expected_pixel, 'outer_pixels': outer_pixels}
                assert matplotlib.colors.to_hex(figure.get_facecolor(), keep_alpha=True) == expected_background
            record['diagnostics'] = dataclasses.asdict(finalize_figure(figure))
            record['external_legend'] = getattr(figure, '_spacr_external_legend_layout', None)
            result.append(record)
        rng_record(case, 'after')
        manifest['cases'][case] = result
        plt.close('all')

    def begin(theme, name):
        """Reset RNG owners to the same frozen state before each actual creator call."""
        plt.close('all')
        random.seed(29)
        np.random.seed(31)
        torch.manual_seed(37)
        case = theme + '/' + name
        rng_record(case, 'before')
        return case

    for theme in ('light', 'dark'):
        dark = theme == 'dark'
        bg, fg = ('#000000', '#ffffff') if dark else ('#ffffff', '#000000')
        preferences.set_language('en')
        preferences.set_figure_colors(bg, fg)
        preferences.set_figure_line_colour(fg)
        preferences.set_figure_style({'background': bg, 'foreground': fg})
        preferences.set_figure_style_per_graph({})
        preferences.set_figure_save_mode('screen')
        assert 'SPACR_PIPELINE_FIGURES' not in os.environ
        resolved = preferences.get_figure_colors()
        assert tuple(matplotlib.colors.to_hex(value) for value in resolved) == (bg, fg)
        target = theme_target()
        assert target == ('screen' if dark else 'print')
        parameters = rc(target)
        assert matplotlib.colors.to_hex(parameters['figure.facecolor']) == bg
        assert matplotlib.colors.to_hex(parameters['axes.facecolor']) == bg
        manifest['resolved_themes'][theme] = {
            'background': bg, 'foreground': fg, 'target': target,
            'resolved_preference_colors': list(resolved),
            'rc_figure_background': parameters['figure.facecolor'],
            'rc_axes_background': parameters['axes.facecolor'],
        }
        with figure_style(target):
            case = begin(theme, 'metrics')
            rng = np.random.default_rng(23)
            x = rng.normal(size=32)
            y = 1.5 + 2 * x + rng.normal(scale=.3, size=len(x))
            model = sm.OLS(y, sm.add_constant(x)).fit()
            for name, value in [('x', x), ('y', y), ('parameters', model.params),
                                ('fitted', model.fittedvalues), ('residuals', model.resid)]:
                array_record(case, name, value)
            value_record(case, 'metrics', [float(value).hex() for value in
                         (np.sqrt(model.mse_resid), np.mean(np.abs(model.resid)),
                          durbin_watson(model.resid))])
            U.model_metrics(model)
        finish(case)

        with figure_style(target):
            case = begin(theme, 'mlr')
            rng = np.random.default_rng(3)
            frame = pd.DataFrame({name: rng.choice(values, 90) for name, values in
                                  [('gene', ['g1', 'g2', 'g3']), ('grna', ['r1', 'r2']),
                                   ('plate', ['p1', 'p2']), ('row', ['A', 'B']),
                                   ('column', ['1', '2'])]})
            frame['pred'] = (rng.normal(0., .25, len(frame))
                             + (frame.gene == 'g2') * (frame.grna == 'r2') * 1.5
                             - (frame.gene == 'g3') * (frame.grna == 'r2') * .8)
            value_record(case, 'input_row_order', frame.drop(columns='pred').values.tolist())
            array_record(case, 'input_response', frame.pred.values)
            effects, pvalues, fitted, table = U.MLR(frame, refine_model=False)
            value_record(case, 'schema', {'columns': table.columns.tolist(),
                                         'index': table.index.tolist()})
            array_record(case, 'returned_table', table.to_numpy())
            array_record(case, 'parameters', fitted.params.to_numpy())
            value_record(case, 'effect_items', [(key, float(value).hex()) for key, value in effects.items()])
            value_record(case, 'pvalue_items', [(key, float(value).hex()) for key, value in pvalues.items()])
        finish(case)

        for generator_name, channels in [('SaliencyMapGenerator', 1),
                                         ('SaliencyMapGenerator', 3), ('GradCAMGenerator', 1)]:
            with figure_style(target):
                case = begin(theme, generator_name + str(channels))
                inputs = torch.arange(96, dtype=torch.float32).reshape(2, 3, 4, 4) / 100
                maps = torch.arange(32 * channels, dtype=torch.float32).reshape(2, channels, 4, 4) / 200
                predictions = torch.tensor([0, 1])
                for name, value in [('input', inputs), ('maps', maps), ('prediction', predictions)]:
                    array_record(case, name, value.numpy())
                generator = object.__new__(getattr(U, generator_name))
                generator.plot_activation_grid(inputs, maps, predictions, overlay=True, normalize=False)
            finish(case)

        paths = []
        inputs = output / 'fixtures' / theme
        inputs.mkdir(parents=True, exist_ok=True)
        for index in range(7):
            path = inputs / f'crop{index}.png'
            Image.fromarray(np.full((6, 6), (index + 1) * 25, np.uint8)).save(path)
            paths.append(str(path))
        embedding = np.array([[0., 0.], [1., 3.], [2., 4.], [2.5, 3.],
                              [8., 2.], [9., 3.], [8.5, 4.]])
        labels = np.array([-1, 0, 0, 0, 5, 5, 5])
        for name in ('embedding', 'gallery'):
            with figure_style(target):
                case = begin(theme, name)
                array_record(case, 'embedding', embedding)
                array_record(case, 'labels', labels)
                value_record(case, 'source_ids', [Path(path).name for path in paths])
                colors = U.generate_colors(len(np.unique(labels)), dark)
                if name == 'embedding':
                    U.plot_embedding(embedding, paths, labels, 2, .3, colors, True, True,
                                     True, True, False, dark, 3., 14., False, False,
                                     interactive_payload={'embedding': embedding, 'labels': labels,
                                                          'reduction_method': 'umap'})
                else:
                    U.plot_clusters_grid(embedding, labels, 2, paths, colors, 2., dark, False)
            finish(case)

        with figure_style(target):
            case = begin(theme, 'class_exemplar')
            model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(48, 1))
            model_path = inputs / 'tiny_model.pth'
            torch.save(model, model_path)
            for name, parameter in model.named_parameters():
                array_record(case, 'model_' + name, parameter.detach().numpy())
            image = U.class_visualization(1, str(model_path), torch.FloatTensor,
                                         img_size=4, num_iterations=1, learning_rate=1.,
                                         max_jitter=1, show_every=1)
            array_record(case, 'returned_image', image)
        finish(case)

        with figure_style(target):
            case = begin(theme, 'process_masks')
            masks, images = inputs / 'masks', inputs / 'images'
            masks.mkdir(exist_ok=True)
            images.mkdir(exist_ok=True)
            for index in range(2):
                mask = np.zeros((24, 24), np.int32)
                mask[1:4, 1:4], mask[1:4, 8:11], mask[10:20, 10:20] = 1, 2, 3
                image = np.arange(24 * 24 * 2, dtype=np.float32).reshape(24, 24, 2) / 100
                np.save(masks / f'{index}.npy', mask)
                np.save(images / f'{index}.npy', image)
                array_record(case, f'input_mask{index}', mask)
                array_record(case, f'input_image{index}', image)
            U.process_masks(str(masks), str(images), 0, n_clusters=2, plot=True)
            for index in range(2):
                array_record(case, f'output_mask{index}', np.load(masks / f'{index}.npy'))
        finish(case)

        with figure_style(target):
            case = begin(theme, 'protected_image_transforms')
            image = np.arange(48, dtype=np.float32).reshape(4, 4, 3) / 100
            mask = np.arange(16, dtype=np.float32).reshape(4, 4) / 16
            array_record(case, 'input_rgb', image)
            array_record(case, 'input_cam', mask)
            array_record(case, 'bgr_cam', U.show_cam_on_image(image, mask))
            array_record(case, 'transparent_crop', U.remove_canvas(Image.open(paths[1])))
            for name in ('SaliencyMapGenerator', 'GradCAMGenerator'):
                generator = object.__new__(getattr(U, name))
                array_record(case, name + '_normalized', generator.percentile_normalize(image.copy()))
            palette = U._generate_mask_random_cmap(np.array([[0, 1], [2, 3]], np.int32))
            array_record(case, 'instance_mask_palette', palette.colors)
        finish(case)

    (output / 'scientific.json').write_text(
        json.dumps(scientific, indent=2, sort_keys=True, default=json_scalar) + '\n')
    (output / 'render_manifest.json').write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=json_scalar) + '\n')


if __name__ == '__main__':
    main()
