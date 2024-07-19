import unittest
from unittest.mock import MagicMock
from spacr.app_mask import initiate_mask_root

class TestInitiateMaskRoot(unittest.TestCase):
    def setUp(self):
        self.width = 800
        self.height = 600

    def test_initiate_mask_root(self):
        # Mock necessary dependencies
        tk = MagicMock()
        ttk = MagicMock()
        ThemedTk = MagicMock()
        ScrollableFrame = MagicMock()
        Figure = MagicMock()
        FigureCanvasTkAgg = MagicMock()
        scrolledtext = MagicMock()
        Queue = MagicMock()
        sys = MagicMock()
        
        # Call the function
        root, vars_dict = initiate_mask_root(self.width, self.height)
        
        # Add assertions here

if __name__ == '__main__':
    unittest.main()