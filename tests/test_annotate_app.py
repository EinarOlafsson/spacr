import unittest
from unittest.mock import MagicMock
from spacr.annotate_app import ImageApp
from PIL import Image

class TestImageApp(unittest.TestCase):
    def setUp(self):
        self.root = MagicMock()
        self.db_path = '/path/to/database.db'
        self.image_type = 'png'
        self.channels = ['r', 'g']
        self.grid_rows = 2
        self.grid_cols = 2
        self.image_size = (200, 200)
        self.annotation_column = 'annotate'
        self.image_app = ImageApp(self.root, self.db_path, self.image_type, self.channels, self.grid_rows, self.grid_cols, self.image_size, self.annotation_column)

    def test_normalize_image(self):
        img = Image.open('/path/to/image.png')
        normalized_img = self.image_app.normalize_image(img)
        self.assertIsInstance(normalized_img, Image.Image)
        self.assertEqual(normalized_img.mode, 'RGB')

    def test_add_colored_border(self):
        img = Image.open('/path/to/image.png')
        border_width = 5
        border_color = 'teal'
        bordered_img = self.image_app.add_colored_border(img, border_width, border_color)
        self.assertIsInstance(bordered_img, Image.Image)
        self.assertEqual(bordered_img.mode, 'RGB')

    def test_filter_channels(self):
        img = Image.open('/path/to/image.png')
        filtered_img = self.image_app.filter_channels(img)
        self.assertIsInstance(filtered_img, Image.Image)
        self.assertEqual(filtered_img.mode, 'L')

    def test_load_single_image(self):
        path_annotation_tuple = ('/path/to/image.png', 1)
        img, annotation = self.image_app.load_single_image(path_annotation_tuple)
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.mode, 'RGB')
        self.assertEqual(annotation, 1)

    def test_get_on_image_click(self):
        path = '/path/to/image.png'
        label = MagicMock()
        img = Image.open('/path/to/image.png')
        callback = self.image_app.get_on_image_click(path, label, img)
        event = MagicMock(num=1)
        callback(event)
        self.assertEqual(self.image_app.pending_updates[path], 1)
        event = MagicMock(num=3)
        callback(event)
        self.assertEqual(self.image_app.pending_updates[path], 2)

if __name__ == '__main__':
    unittest.main()