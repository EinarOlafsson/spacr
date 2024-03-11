import unittest
from PIL import Image
import numpy as np
from spacr.annotate_app import ImageApp

class TestImageApp(unittest.TestCase):

    def setUp(self):
        self.app = ImageApp(None, 'test.db', image_type='test', channels=['r', 'g', 'b'], grid_rows=5, grid_cols=5, image_size=(200, 200), annotation_column='annotate')

    def test_normalize_image(self):
        # Create a test image with specific pixel values
        data = np.array([[0, 128, 255], [64, 192, 128]], dtype=np.uint8)
        img = Image.fromarray(data, 'L')

        # Normalize the image
        normalized_img = self.app.normalize_image(img)

        # Convert back to array to check values
        normalized_data = np.array(normalized_img)

        # Check if the image was normalized correctly
        expected_data = np.array([[0, 128, 255], [64, 192, 128]], dtype=np.uint8)  # Expected result after normalization
        np.testing.assert_array_equal(normalized_data, expected_data)

    def test_add_colored_border(self):
        # Create a test image
        img = Image.new('RGB', (10, 10), 'black')

        # Add a colored border
        bordered_img = self.app.add_colored_border(img, 2, 'red')

        # Check the size of the bordered image
        self.assertEqual(bordered_img.size, (14, 14))

        # Check a few border pixels to confirm the border color
        self.assertEqual(bordered_img.getpixel((0, 0)), (255, 0, 0))  # Top-left corner
        self.assertEqual(bordered_img.getpixel((13, 13)), (255, 0, 0))  # Bottom-right corner
        self.assertEqual(bordered_img.getpixel((7, 0)), (255, 0, 0))  # Middle of top border

    def test_filter_channels(self):
        # Create a test image
        img = Image.new('RGB', (10, 10), 'black')

        # Filter the channels
        filtered_img = self.app.filter_channels(img)

        # Check if the image has been filtered correctly
        self.assertEqual(filtered_img.mode, 'L')  # Grayscale image

    def test_load_single_image(self):
        # Create a test image
        img = Image.new('RGB', (10, 10), 'black')

        # Mock the load_single_image method
        def mock_load_single_image(path_annotation_tuple):
            return img, 1

        # Replace the original method with the mock method
        self.app.load_single_image = mock_load_single_image

        # Call the method
        loaded_img, annotation = self.app.load_single_image(('test.png', 1))

        # Check if the loaded image and annotation are correct
        self.assertEqual(loaded_img, img)
        self.assertEqual(annotation, 1)

if __name__ == '__main__':
    unittest.main()