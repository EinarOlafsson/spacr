import unittest
from unittest.mock import MagicMock
from spacr.measure import _measure_crop_core

class TestMeasureCropCore(unittest.TestCase):
    def setUp(self):
        self.index = 0
        self.time_ls = [0, 1, 2]
        self.file = "image.npy"
        self.settings = {
            'input_folder': '/path/to/input/folder',
            'save_measurements': True,
            'plot_filtration': True,
            'channels': [0, 1, 2],
            'cell_mask_dim': 3,
            'cell_min_size': 100,
            'nucleus_mask_dim': 4,
            'nucleus_min_size': 200,
            'timelapse_objects': 'nucleus',
            'pathogen_mask_dim': 5,
            'pathogen_min_size': 300,
            'merge_edge_pathogen_cells': True,
            'cytoplasm': True,
            'cytoplasm_min_size': 400,
            'include_uninfected': False,
            'save_png': True,
            'save_arrays': True,
            'plot': True,
            'dialate_pngs': True,
            'dialate_png_ratios': 0.5,
            'crop_mode': ['cell', 'nucleus'],
            'png_size': [(100, 100), (200, 200)]
        }

    def test_measure_crop_core(self):
        # Mock necessary dependencies
        np = MagicMock()
        os = MagicMock()
        _create_database = MagicMock()
        _plot_cropped_arrays = MagicMock()
        _filter_object = MagicMock()
        _merge_overlapping_objects = MagicMock()
        _relabel_parent_with_child_labels = MagicMock()
        _exclude_objects = MagicMock()
        _merge_and_save_to_database = MagicMock()
        _crop_center = MagicMock()
        _find_bounding_box = MagicMock()
        _generate_names = MagicMock()
        _get_percentiles = MagicMock()
        _map_wells_png = MagicMock()

        # Call the function
        cropped_images = _measure_crop_core(self.index, self.time_ls, self.file, self.settings)

        # Add assertions here

if __name__ == '__main__':
    unittest.main()