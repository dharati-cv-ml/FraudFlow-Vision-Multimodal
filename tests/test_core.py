import json
import tempfile
import unittest
from pathlib import Path
from fraudvision.core import VisualModel,train,metrics,select_threshold,sigmoid,load_manifest
from fraudvision.features import FEATURE_NAMES
from fraudvision.paths import MANIFEST,MODEL

class ModelTests(unittest.TestCase):
    def test_sigmoid_bounded(self):
        self.assertLess(sigmoid(500),1.00001)
        self.assertGreater(sigmoid(-500),-0.00001)
    def test_metrics_perfect(self):
        m=metrics([0,1,0,1],[0,1,0,1]);self.assertEqual(m['f1'],1)
        self.assertEqual(m['tp'],2)
    def test_metrics_errors(self):
        m=metrics([0,1,1,0],[1,0,1,0])
        self.assertEqual(m['fp'],1);self.assertEqual(m['fn'],1)
    def test_metrics_zero(self):
        self.assertEqual(metrics([0,0],[0,0])['precision'],0)
    def test_metrics_length_guard(self):
        with self.assertRaises(ValueError):metrics([0],[1,0])
    def test_train_rejects_small(self):
        with self.assertRaises(ValueError):train([([0]*len(FEATURE_NAMES),0)])
    def test_train_rejects_one_class(self):
        with self.assertRaises(ValueError):train([([1]*len(FEATURE_NAMES),0)]*20)
    def test_train_predict(self):
        records=[([i/20]*len(FEATURE_NAMES),int(i>9)) for i in range(20)]
        m=train(records,epochs=200)
        self.assertGreater(m.risk([0.95]*len(FEATURE_NAMES)),m.risk([0.1]*len(FEATURE_NAMES)))
    def test_model_io(self):
        m=VisualModel([0]*8,[1]*8,[.2]*8,0,.54)
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'model.json';m.save(p)
            self.assertAlmostEqual(VisualModel.load(p).risk([1]*8),m.risk([1]*8))
    def test_model_version_guard(self):
        with self.assertRaises(ValueError):VisualModel.from_dict({'format':'wrong','feature_names':FEATURE_NAMES})
    def test_model_dim_guard(self):
        m=VisualModel([0]*8,[1]*8,[.2]*8,0).to_dict();m['weights']=[1]
        with self.assertRaises(ValueError):VisualModel.from_dict(m)
    def test_bad_threshold(self):
        m=VisualModel([0]*8,[1]*8,[.2]*8,0).to_dict();m['threshold']=4
        with self.assertRaises(ValueError):VisualModel.from_dict(m)
    def test_select_threshold(self):
        m=VisualModel([0]*8,[1]*8,[1]*8,0)
        self.assertTrue(0<select_threshold(m,[([0]*8,0),([2]*8,1)])<1)
    def test_demo_manifest(self):
        rows=load_manifest();self.assertEqual(len(rows),320)
        self.assertEqual(len({r['file'] for r in rows}),320)
    def test_demo_model(self):
        m=VisualModel.load(MODEL)
        self.assertEqual(len(m.weights),8)
        self.assertGreater(m.threshold,0)
