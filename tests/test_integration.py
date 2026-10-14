import csv
import json
import unittest
from fraudvision.paths import DATA, REPORTS
from fraudvision.core import VisualModel, metrics
from fraudvision.paths import MODEL
from fraudvision.verification import combined_decision, verify_image
from fraudvision.ocr import tesseract_path

class SubmissionIntegrityTests(unittest.TestCase):
    def test_visual_report_matches_test_predictions(self):
        with (REPORTS/'test_predictions.csv').open(newline='',encoding='utf-8') as fp:
            rows=list(csv.DictReader(fp))
        report=json.loads((REPORTS/'evaluation.json').read_text(encoding='utf-8'))
        actual=metrics([int(r['label']) for r in rows],[int(r['prediction']) for r in rows])
        self.assertEqual(actual,report['splits']['test'])
        self.assertEqual(len(rows),64)

    def test_combined_report_matches_predictions(self):
        with (REPORTS/'ocr_test_predictions.csv').open(newline='',encoding='utf-8') as fp:
            rows=list(csv.DictReader(fp))
        report=json.loads((REPORTS/'ocr_audit.json').read_text(encoding='utf-8'))
        actual=metrics([int(r['label']) for r in rows],[int(r['combined_decision']) for r in rows])
        self.assertEqual(actual,report['ocr_assisted_combined'])
        for row in rows:
            mismatch=row['ocr_status']=='MISMATCH'
            self.assertEqual(int(combined_decision(bool(int(row['visual_decision'])),
                _check(mismatch)) == 'REVIEW'),int(row['combined_decision']))

    @unittest.skipUnless(tesseract_path(), 'Local Tesseract not installed')
    def test_tesseract_reads_actual_edited_receipt(self):
        check, text=verify_image(DATA/'images/doc_0006.jpg')
        self.assertEqual(check.status,'MISMATCH')
        self.assertEqual(check.difference,'+61.00')
        self.assertIn('Subtotal',text)

def _check(mismatch):
    class Check:
        status='MISMATCH' if mismatch else 'CONSISTENT'
    return Check()
