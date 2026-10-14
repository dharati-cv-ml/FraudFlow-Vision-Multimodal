import json,unittest
from pathlib import Path
from fraudvision.paths import REPORTS,MODEL
from fraudvision.core import VisualModel,load_manifest
class ReportTests(unittest.TestCase):
    def test_split_counts(self):
        rep=json.loads((REPORTS/'evaluation.json').read_text())
        self.assertEqual([rep['splits'][k]['total'] for k in ('train','validation','test')],[192,64,64])
    def test_no_test_threshold_tuning(self):
        report=json.loads((REPORTS/'evaluation.json').read_text())
        model=VisualModel.load(MODEL)
        self.assertAlmostEqual(report['threshold'],model.threshold)
        self.assertIn('validation',report['protocol'].lower())
    def test_files_included(self):
        for row in load_manifest():
            self.assertTrue((Path('data')/row['file']).exists())
