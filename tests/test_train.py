import unittest
from unittest.mock import MagicMock
from spacr.train import train_model

class TestTrainModel(unittest.TestCase):
    def test_train_model_erm(self):
        # Mock necessary dependencies
        train_loaders = [
            MagicMock(),  # training data loader 1
            MagicMock()   # training data loader 2
        ]
        val_loaders = [
            MagicMock()   # validation data loader
        ]
        test_loaders = [
            MagicMock()   # test data loader
        ]

        # Call the function
        train_model('/path/to/save', 'model_type', train_loaders, ['loader1', 'loader2'], train_mode='erm', epochs=100, val_loaders=val_loaders, test_loaders=test_loaders)

        # Add your assertions here
        # ...

    def test_train_model_irm(self):
        # Mock necessary dependencies
        train_loaders = [
            MagicMock(),  # training data loader 1
            MagicMock()   # training data loader 2
        ]

        # Call the function
        train_model('/path/to/save', 'model_type', train_loaders, ['loader1', 'loader2'], train_mode='irm', epochs=100)

        # Add your assertions here
        # ...

if __name__ == '__main__':
    unittest.main()