import json
import tempfile
import unittest
from pathlib import Path
from multimodal.compare import compare
from multimodal.lora_dataset import export

class ComparisonTests(unittest.TestCase):
    def test_refuse_partial_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/'ocr.json';b=Path(tmp)/'vlm.json'
            a.write_text(json.dumps({'method':'ocr','split':'test','predictions':[{'file':'one'}]}))
            b.write_text(json.dumps({'method':'vlm','split':'test','predictions':[{'file':'one'}]}))
            with self.assertRaises(ValueError):compare(a,b)
    def test_export_train_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/'train.jsonl'
            n=export('train',file)
            self.assertEqual(n,192)
            entries=[json.loads(x) for x in file.read_text().splitlines()]
            self.assertEqual(len(entries),192)
            self.assertNotIn('label',entries[0]['prompt'])
            with self.assertRaises(ValueError): export('test',file)
