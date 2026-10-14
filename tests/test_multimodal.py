import unittest
from multimodal.answers import build_answers, FIELDS
from multimodal.evaluate import normalize, score_predictions
from multimodal.ocr_baseline import ocr_answers
from fraudvision.paths import DATA

class VQATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.examples=build_answers()
    def test_total_questions(self):
        self.assertEqual(len(self.examples),320)
    def test_splits(self):
        self.assertEqual(sum(x['split']=='test' for x in self.examples),64)
    def test_answer_fields(self):
        self.assertTrue(all(set(e['answer'])==set(FIELDS) for e in self.examples))
    def test_semantics(self):
        self.assertTrue(all(e['answer']['arithmetic_consistency'] in ('YES','NO') for e in self.examples))
    def test_normalization(self):
        self.assertEqual(normalize('total','$1,234.50'),'1234.50')
        self.assertEqual(normalize('date','2026-04-02.'),'2026-04-02')
    def test_scoring_denominator(self):
        s,_=score_predictions([{'file':self.examples[0]['file'],'prediction':self.examples[0]['answer']}],self.examples)
        self.assertEqual(s['micro_exact_match'],1.0)
    def test_missing_predictions_are_wrong(self):
        s,_=score_predictions([{'file':self.examples[0]['file'],'prediction':{}}],self.examples)
        self.assertEqual(s['micro_exact_match'],0.0)
    def test_ocr_baseline_without_ocr_installed(self):
        # no network, no model needed for the test; input is an image path.
        pred,text=ocr_answers(DATA/self.examples[0]['file'])
        self.assertEqual(set(pred),set(FIELDS))
