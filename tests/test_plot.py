import unittest
from unittest.mock import MagicMock
from spacr.plot import _save_scimg_plot
from unittest.mock import patch

class TestSaveScimgPlot(unittest.TestCase):
    def setUp(self):
        self.src = "/path/to/images"
        self.nr_imgs = 16
        self.channel_indices = [0, 1, 2]
        self.um_per_pixel = 0.1
        self.scale_bar_length_um = 10
        self.standardize = True
        self.fontsize = 8
        self.show_filename = True
        self.channel_names = None
        self.dpi = 300
        self.plot = False
        self.i = 1
        self.all_folders = 1

    def test_save_scimg_plot(self):
        # Mock necessary dependencies
        _save_figure_mock = MagicMock()
        _visualize_scimgs_mock = MagicMock(return_value="mocked_figure")
        _plot_images_on_grid_mock = MagicMock(return_value="mocked_figure")

        # Patch the dependencies
        with patch('spacr.plot._save_figure', _save_figure_mock), \
             patch('spacr.plot._visualize_scimgs', _visualize_scimgs_mock), \
             patch('spacr.plot._plot_images_on_grid', _plot_images_on_grid_mock):
            
            # Call the function
            _save_scimg_plot(self.src, self.nr_imgs, self.channel_indices, self.um_per_pixel, self.scale_bar_length_um, self.standardize, self.fontsize, self.show_filename, self.channel_names, self.dpi, self.plot, self.i, self.all_folders)
        
        # Add your assertions here
        _visualize_scimgs_mock.assert_called_with(self.src, self.channel_indices, self.um_per_pixel, self.scale_bar_length_um, self.show_filename, self.standardize, self.nr_imgs, self.fontsize, self.channel_names, self.plot)
        _save_figure_mock.assert_called_with("mocked_figure", self.src, text='all_channels')
        _plot_images_on_grid_mock.assert_called_with("mocked_figure", self.channel_indices, self.um_per_pixel, self.scale_bar_length_um, self.fontsize, self.show_filename, self.channel_names, self.plot)
        # Add more assertions for other function calls if needed

if __name__ == '__main__':
    unittest.main()