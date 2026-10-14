"""Regression tests for correct paired visual VQA scoring."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from multimodal.visual_compare import assemble, discover_reports, export_rows, load_report

A = 'images/doc_0256.jpg'
B = 'images/doc_0257.jpg'
TRUTH = {A: {'document_type':'receipt','date':'2026-08-14','total':'$234.14',
             'payment_reference':'ACCT-63381','arithmetic_consistency':'YES'},
         B: {'document_type':'invoice','date':'2026-08-15','total':'$100.00',
             'payment_reference':'ACCT-22222','arithmetic_consistency':'NO'}}

def report(method, rows):
    return {'method': method, 'split': 'test',
            'predictions': [{'file': fn, 'prediction':pred, 'runtime_s':latency,
                             'raw_output':'local measured text'} for fn,pred,latency in rows]}

class TestVisualComparison(unittest.TestCase):
    def test_matching_id_intersection_and_correctness(self):
        o=report('ocr', [(A, dict(TRUTH[A]),.2),(B,dict(TRUTH[B]),.3)])
        altered=dict(TRUTH[A]);altered['date']='2025-01-01'
        v=report('vlm',[(A,altered,3.4)])
        d=assemble(o,v,TRUTH)
        self.assertEqual(len(d['documents']),1)
        self.assertEqual(d['metrics']['ocr']['correct'],5)
        self.assertEqual(d['metrics']['vlm']['correct'],4)
        self.assertAlmostEqual(d['metrics']['vlm']['accuracy'],.8)
        self.assertEqual(d['documents'][0]['vlm_errors'],1)
        self.assertEqual(d['documents'][0]['ocr_errors'],0)
        self.assertEqual(d['documents'][0]['fields'][1]['field'],'date')
        self.assertEqual(d['documents'][0]['fields'][1]['expected'],'2026-08-14')
        self.assertFalse(d['documents'][0]['fields'][1]['vlm_correct'])
        self.assertEqual(d['metrics']['ocr']['avg_latency_s'],.2)

    def test_missing_vlm_is_not_zero(self):
        o=report('ocr',[(A,dict(TRUTH[A]),.2)])
        result=assemble(o,None,TRUTH)
        self.assertEqual(list(result['metrics']),['ocr'])
        self.assertIsNone(result['documents'][0]['fields'][0]['vlm_correct'])

    def test_no_overlap_rejected(self):
        o=report('ocr',[(A,dict(TRUTH[A]),.2)])
        v=report('vlm',[(B,dict(TRUTH[B]),2.2)])
        with self.assertRaisesRegex(ValueError,'No matching'):
            assemble(o,v,TRUTH)

    def test_splits_must_match(self):
        o=report('ocr',[(A,dict(TRUTH[A]),.2)])
        v=report('vlm',[(A,dict(TRUTH[A]),2.2)])
        v['split']='validation'
        with self.assertRaisesRegex(ValueError,'different splits'):
            assemble(o,v,TRUTH)

    def test_csv_export_includes_correct_labels(self):
        result=assemble(report('ocr',[(A,dict(TRUTH[A]),.2)]),None,TRUTH)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'export.csv'
            self.assertEqual(export_rows(result,path),5)
            with path.open(encoding='utf-8-sig',newline='') as h:
                rows=list(csv.DictReader(h))
            self.assertEqual(len(rows),5)
            self.assertEqual(rows[0]['ground_truth'],'receipt')
            self.assertEqual(rows[0]['vlm_correct'],'')

    def test_discovery_prefers_matching_report_sizes(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for name in ('ocr_test_64.json','ocr_test_10.json','vlm_test_10.json','vlm_test_2.json'):
                (root/name).write_text('{}')
            o,v=discover_reports(root)
            self.assertEqual(o.name,'ocr_test_10.json')
            self.assertEqual(v.name,'vlm_test_10.json')
            (root/'ocr_test_10.json').unlink()
            o,v=discover_reports(root)
            self.assertEqual(o.name,'ocr_test_64.json')

    def test_missing_predictions_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'file.json'
            path.write_text(json.dumps({'method':'vlm','predictions': []}))
            with self.assertRaises(ValueError):load_report(path,'vlm')

if __name__=='__main__':unittest.main()
