import unittest
from unittest.mock import MagicMock
from spacr.gui_utils import check_measure_gui_settings

class TestCheckMeasureGuiSettings(unittest.TestCase):
    def setUp(self):
        self.vars_dict = {
            'channels': MagicMock(get=MagicMock(return_value="[1, 2, 3]")),
            'png_dims': MagicMock(get=MagicMock(return_value="[100, 200, 300]")),
            'cell_loc': MagicMock(get=MagicMock(return_value="(1, 2)")),
            'pathogen_loc': MagicMock(get=MagicMock(return_value="(3, 4)")),
            'treatment_loc': MagicMock(get=MagicMock(return_value="(5, 6)")),
            'dialate_png_ratios': MagicMock(get=MagicMock(return_value="[0.5, 0.75, 1.0]")),
            'normalize': MagicMock(get=MagicMock(return_value="[0, 255]")),
            'normalize_by': MagicMock(get=MagicMock(return_value="mean")),
            'png_size': MagicMock(get=MagicMock(return_value="[[100, 100], [200, 200]]")),
            'pathogens': MagicMock(get=MagicMock(return_value="[pathogen1, pathogen2]")),
            'treatments': MagicMock(get=MagicMock(return_value="[treatment1, treatment2]")),
            'cells': MagicMock(get=MagicMock(return_value="[cell1, cell2]")),
            'crop_mode': MagicMock(get=MagicMock(return_value="square")),
            'timelapse_objects': MagicMock(get=MagicMock(return_value="['obj1', 'obj2']")),
            'experiment': MagicMock(get=MagicMock(return_value="exp1")),
            'measurement': MagicMock(get=MagicMock(return_value="measurement1")),
            'input_folder': MagicMock(get=MagicMock(return_value="folder1")),
            'cell_mask_dim': MagicMock(get=MagicMock(return_value="10")),
            'cell_min_size': MagicMock(get=MagicMock(return_value="20")),
            'nucleus_mask_dim': MagicMock(get=MagicMock(return_value="30")),
            'nucleus_min_size': MagicMock(get=MagicMock(return_value="40")),
            'pathogen_mask_dim': MagicMock(get=MagicMock(return_value="50")),
            'pathogen_min_size': MagicMock(get=MagicMock(return_value="60")),
            'cytoplasm_min_size': MagicMock(get=MagicMock(return_value="70")),
            'max_workers': MagicMock(get=MagicMock(return_value="8")),
            'channel_of_interest': MagicMock(get=MagicMock(return_value="1")),
            'nr_imgs': MagicMock(get=MagicMock(return_value="100")),
            'um_per_pixel': MagicMock(get=MagicMock(return_value="0.5")),
            'save_png': MagicMock(get=MagicMock(return_value=True)),
            'use_bounding_box': MagicMock(get=MagicMock(return_value=False)),
            'save_measurements': MagicMock(get=MagicMock(return_value=True)),
            'plot': MagicMock(get=MagicMock(return_value=False)),
            'plot_filtration': MagicMock(get=MagicMock(return_value=True)),
            'include_uninfected': MagicMock(get=MagicMock(return_value=False)),
            'dialate_pngs': MagicMock(get=MagicMock(return_value=True)),
            'timelapse': MagicMock(get=MagicMock(return_value=False)),
            'representative_images': MagicMock(get=MagicMock(return_value=True)),
        }

    def test_check_measure_gui_settings(self):
        settings = check_measure_gui_settings(self.vars_dict)

        # Add your assertions here
        self.assertEqual(settings['channels'], [1, 2, 3])
        self.assertEqual(settings['png_dims'], [100, 200, 300])
        self.assertEqual(settings['cell_loc'], (1, 2))
        self.assertEqual(settings['pathogen_loc'], (3, 4))
        self.assertEqual(settings['treatment_loc'], (5, 6))
        self.assertEqual(settings['dialate_png_ratios'], [0.5, 0.75, 1.0])
        self.assertEqual(settings['normalize'], [0, 255])
        self.assertEqual(settings['normalize_by'], "mean")
        self.assertEqual(settings['png_size'], [[100, 100], [200, 200]])
        self.assertEqual(settings['pathogens'], ["pathogen1", "pathogen2"])
        self.assertEqual(settings['treatments'], ["treatment1", "treatment2"])
        self.assertEqual(settings['cells'], ["cell1", "cell2"])
        self.assertEqual(settings['crop_mode'], "square")
        self.assertEqual(settings['timelapse_objects'], ["obj1", "obj2"])
        self.assertEqual(settings['experiment'], "exp1")
        self.assertEqual(settings['measurement'], "measurement1")
        self.assertEqual(settings['input_folder'], "folder1")
        self.assertEqual(settings['cell_mask_dim'], 10)
        self.assertEqual(settings['cell_min_size'], 20)
        self.assertEqual(settings['nucleus_mask_dim'], 30)
        self.assertEqual(settings['nucleus_min_size'], 40)
        self.assertEqual(settings['pathogen_mask_dim'], 50)
        self.assertEqual(settings['pathogen_min_size'], 60)
        self.assertEqual(settings['cytoplasm_min_size'], 70)
        self.assertEqual(settings['max_workers'], 8)
        self.assertEqual(settings['channel_of_interest'], 1)
        self.assertEqual(settings['nr_imgs'], 100)
        self.assertEqual(settings['um_per_pixel'], 0.5)
        self.assertEqual(settings['save_png'], True)
        self.assertEqual(settings['use_bounding_box'], False)
        self.assertEqual(settings['save_measurements'], True)
        self.assertEqual(settings['plot'], False)
        self.assertEqual(settings['plot_filtration'], True)
        self.assertEqual(settings['include_uninfected'], False)
        self.assertEqual(settings['dialate_pngs'], True)
        self.assertEqual(settings['timelapse'], False)
        self.assertEqual(settings['representative_images'], True)

if __name__ == '__main__':
    unittest.main()