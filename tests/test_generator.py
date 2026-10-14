import csv
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from fraudvision.generator import generate_dataset,generate_one
import random
import re

class GeneratorTests(unittest.TestCase):
    def test_min_count(self):
        with self.assertRaises(ValueError): generate_dataset(count=7)
    def test_document_types(self):
        for i,expected in [(0,'receipt'),(1,'invoice')]:
            image,meta=generate_one(i,random.Random(12),bool(i))
            self.assertEqual(image.size,(680,900))
            self.assertEqual(meta['document_type'],expected)
    def test_clean_is_unedited(self):
        _,row=generate_one(1,random.Random(14),False)
        self.assertEqual(row['bbox'],'');self.assertEqual(row['label'],0)
    def test_tampered_has_location(self):
        _,row=generate_one(2,random.Random(14),True)
        self.assertEqual(len(row['bbox'].split(',')),4)
        self.assertEqual(row['label'],1)
    def test_seamless_total_is_still_money(self):
        # index 0 is a seamless total edit; never replace currency with account ref.
        _,row=generate_one(0,random.Random(42),True)
        self.assertEqual(row['edit_type'],'seamless_total')
        self.assertRegex(row['replacement_value'],r'^\$[0-9,]+\.[0-9]{2}$')
    def test_seamless_date_is_a_date(self):
        # index 7 is a seamless date edit (7 mod 3 == 1).
        _,row=generate_one(7,random.Random(42),True)
        self.assertEqual(row['edit_type'],'seamless_date')
        self.assertRegex(row['replacement_value'],r'^\d{4}-\d{2}-\d{2}$')
    def test_dataset_independent_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            t=Path(tmp);rows=generate_dataset(60,seed=123,out=t/'images',manifest=t/'manifest.csv')
            self.assertEqual(len(rows),60)
            self.assertEqual(Counter(r['split'] for r in rows),{'train':36,'validation':12,'test':12})
            self.assertEqual(len(set(r['file'] for r in rows)),60)
            self.assertEqual(Counter(int(r['label']) for r in rows),{0:30,1:30})
            for split in ('train','validation','test'):
                for doc_type in ('receipt','invoice'):
                    labels=Counter(int(r['label']) for r in rows if r['split']==split and r['document_type']==doc_type)
                    self.assertLessEqual(abs(labels[0]-labels[1]),1)
            self.assertTrue((t/rows[0]['file']).exists())
