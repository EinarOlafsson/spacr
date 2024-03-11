import unittest
from unittest.mock import MagicMock
import numpy as np
import matplotlib.pyplot as plt
from spacr.sim import visualize_all

class TestVisualizeAll(unittest.TestCase):
    def test_visualize_all(self):
        # Mock necessary dependencies
        output = [
            MagicMock(),  # cell_scores
            MagicMock(),  # cell_roc_dict_df
            MagicMock(),  # cell_pr_dict_df
            MagicMock(),  # cell_cm
            MagicMock(),  # well_score
            MagicMock(),  # gene_fraction_map
            MagicMock(),  # metadata
            MagicMock(),  # results_df
            MagicMock(),  # reg_roc_dict_df
            MagicMock(),  # reg_pr_dict_df
            MagicMock(),  # reg_cm
            MagicMock(),  # sim_stats
            MagicMock(),  # genes_per_well_df
            MagicMock()   # wells_per_gene_df
        ]

        # Call the function
        fig = visualize_all(output)

        # Add your assertions here
        self.assertIsInstance(fig, plt.Figure)
        # Add more assertions as needed

if __name__ == '__main__':
    unittest.main()