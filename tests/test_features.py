import random
import unittest
from PIL import Image
from fraudvision.features import extract,FEATURE_NAMES
from fraudvision.generator import generate_one
class FeatureTests(unittest.TestCase):
    def test_feature_count(self):
        im,_=generate_one(5,random.Random(2),True)
        self.assertEqual(len(extract(im).features),len(FEATURE_NAMES))
    def test_feature_finite(self):
        im,_=generate_one(3,random.Random(2),False)
        result=extract(im)
        self.assertTrue(all(0<=x<1000 for x in result.features))
    def test_repeatability(self):
        image,_=generate_one(4,random.Random(10),True)
        self.assertEqual(extract(image).features,extract(image).features)
    def test_clean_image(self):
        im=Image.new('RGB',(680,900),'white')
        self.assertEqual(extract(im).features[0],0)
        self.assertIsNone(extract(im).suspicious_region)
    def test_edited_seam_is_stronger(self):
        clean,_=generate_one(6,random.Random(12),False)
        tampered,_=generate_one(6,random.Random(12),True)
        self.assertGreater(extract(tampered).features[7],extract(clean).features[7])
    def test_document_not_reading_ground_truth(self):
        # extractor has no manifest/label argument, only pixel input
        im,_=generate_one(8,random.Random(2),True)
        self.assertIsNotNone(extract(im).description)
