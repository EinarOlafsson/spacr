import unittest
from unittest.mock import MagicMock
from spacr.io import preprocess_img_data

class TestPreprocessImgData(unittest.TestCase):
    def setUp(self):
        self.settings = {
            'src': '/path/to/images',
            'nucleus_channel': 0,
            'pathogen_channel': 1,
            'cell_channel': 2,
            'nucleus_background': 100,
            'pathogen_background': 100,
            'cell_background': 100,
            'metadata_type': 'cellvoyager',
            'custom_regex': None,
            'examples_to_plot': 5,
            'plot': False,
            'batch_size': [100, 100, 100],
            'timelapse': False,
            'remove_background': False,
            'lower_quantile': 0.01,
            'randomize': True,
            'all_to_mip': False,
            'pick_slice': False,
            'skip_mode': '01'
        }

    def test_preprocess_img_data(self):
        # Mock necessary dependencies
        os = MagicMock()
        os.listdir.return_value = ['image1.tif', 'image2.tif', 'image3.tif']
        Counter = MagicMock()
        extension_counts = MagicMock()
        extension_counts.most_common.return_value = [('tif', 3)]
        Counter.return_value = extension_counts
        spacr_plot = MagicMock()
        spacr_plot.plot_arrays = MagicMock()
        spacr_plot._plot_4D_arrays = MagicMock()
        spacr_plot._merge_channels = MagicMock()
        spacr_plot._create_movies_from_npy_per_channel = MagicMock()
        spacr_plot._concatenate_channel = MagicMock()
        spacr_plot._normalize_stack = MagicMock()
        spacr_plot._normalize_timelapse = MagicMock()

        # Call the function
        preprocess_img_data(self.settings)

        # Add assertions here

if __name__ == '__main__':
    unittest.main()