import unittest
from unittest.mock import patch
from fraudvision.ocr import read_text
class OCRTests(unittest.TestCase):
    @patch('fraudvision.ocr.tesseract_path',return_value=None)
    def test_graceful_without_ocr(self,_):
        self.assertIn('optional',read_text('nothing.jpg').lower())
