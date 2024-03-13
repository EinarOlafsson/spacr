import unittest
import numpy as np
from spacr.utils import _filter_cp_masks

class TestFilterCPMasks(unittest.TestCase):
    def test_filter_cp_masks(self):
        # Create dummy inputs
        masks = [np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]]), np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])]
        flows = [np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])]
        filter_size = True
        minimum_size = 2
        maximum_size = 5
        remove_border_objects = True
        merge = True
        filter_dimm = True
        batch = np.zeros((2, 3, 3, 3))
        moving_avg_q1 = 0.0
        moving_avg_q3 = 0.0
        moving_count = 0
        plot = False
        figuresize = (10, 10)

        # Call the function
        filtered_masks = _filter_cp_masks(masks, flows, filter_size, minimum_size, maximum_size, remove_border_objects, merge, filter_dimm, batch, moving_avg_q1, moving_avg_q3, moving_count, plot, figuresize)

        # Add your assertions here
        self.assertEqual(len(filtered_masks), 2)
        self.assertTrue(np.array_equal(filtered_masks[0], np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]])))
        self.assertTrue(np.array_equal(filtered_masks[1], np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])))
        # Add more assertions as needed

if __name__ == '__main__':
    unittest.main()