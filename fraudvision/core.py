"""Pure-Python logistic model, stratified split evaluation and reports.

Uses visual features from image pixels only. Ground-truth annotations from the
manifest must never enter prediction, feature extraction, or threshold selection
on the held-out test set.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import csv
import json
import math
from .features import FEATURE_NAMES, extract
from .paths import DATA, MANIFEST, MODEL, REPORTS


def load_manifest(manifest=MANIFEST):
    with Path(manifest).open(newline='',encoding='utf-8') as fp:
        return list(csv.DictReader(fp))


def sigmoid(z):
    z=max(-35,min(35,z))
    return 1/(1+math.exp(-z))

@dataclass
class VisualModel:
    center: list[float]
    scale: list[float]
    weights: list[float]
    bias: float
    threshold: float=0.5
    def risk(self,features):
        z=self.bias+sum(w*(v-c)/s for w,v,c,s in zip(self.weights,features,self.center,self.scale))
        return sigmoid(z)
    def predict(self,features):
        return int(self.risk(features)>=self.threshold)
    def to_dict(self):
        return {'format':'fraudvision-logistic-v1','feature_names': FEATURE_NAMES,
                'center': self.center,'scale':self.scale,'weights':self.weights,
                'bias':self.bias,'threshold':self.threshold,
                'note':'Risk scores are uncalibrated synthetic-model scores, not real-world fraud probabilities.'}
    @classmethod
    def from_dict(cls,obj):
        if obj.get('format')!='fraudvision-logistic-v1' or obj.get('feature_names')!=FEATURE_NAMES:
            raise ValueError('Incompatible model file')
        for key in ('center','scale','weights'):
            if len(obj[key])!=len(FEATURE_NAMES):
                raise ValueError('Invalid model feature shape')
        if not (0 < float(obj['threshold']) < 1):
            raise ValueError('Invalid decision threshold')
        return cls([float(x) for x in obj['center']], [float(x) for x in obj['scale']],
                   [float(x) for x in obj['weights']],float(obj['bias']),float(obj['threshold']))
    def save(self,path=MODEL):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(self.to_dict(),indent=2),encoding='utf-8')
    @classmethod
    def load(cls,path=MODEL):
        return cls.from_dict(json.loads(Path(path).read_text(encoding='utf-8')))


def train(samples, epochs=800, rate=.06, regularization=.015):
    if len(samples)<10:
        raise ValueError('Need at least ten training records')
    if len({label for _, label in samples})<2:
        raise ValueError('Training requires both classes')
    dims=len(FEATURE_NAMES)
    center=[sum(features[j] for features,_ in samples)/len(samples) for j in range(dims)]
    scale=[max(0.04,(sum((features[j]-center[j])**2 for features,_ in samples)/len(samples))**.5) for j in range(dims)]
    values=[([max(-12,min(12,(v-c)/s)) for v,c,s in zip(features,center,scale)], label) for features,label in samples]
    w=[0.0]*dims;b=0.0
    for _ in range(epochs):
        grad=[0.0]*dims; bias_grad=0.0
        for x,label in values:
            p=sigmoid(b+sum(a*z for a,z in zip(w,x)))
            delta=p-label
            bias_grad+=delta
            for j in range(dims):grad[j]+=delta*x[j]
        n=len(values)
        w=[wj-rate*(grad[j]/n+regularization*wj) for j,wj in enumerate(w)]
        b-=rate*bias_grad/n
    return VisualModel(center,scale,w,b)


def metrics(y_true,y_pred):
    if len(y_true)!=len(y_pred): raise ValueError('Length mismatch')
    tp=sum(a==1 and b==1 for a,b in zip(y_true,y_pred))
    tn=sum(a==0 and b==0 for a,b in zip(y_true,y_pred))
    fp=sum(a==0 and b==1 for a,b in zip(y_true,y_pred))
    fn=sum(a==1 and b==0 for a,b in zip(y_true,y_pred))
    precision=tp/(tp+fp) if tp+fp else 0
    recall=tp/(tp+fn) if tp+fn else 0
    specificity=tn/(tn+fp) if tn+fp else 0
    return {'total':len(y_true),'tp':tp,'tn':tn,'fp':fp,'fn':fn,
            'precision':round(precision,4),'recall':round(recall,4),
            'f1':round(2*precision*recall/(precision+recall),4) if precision+recall else 0,
            'accuracy':round((tp+tn)/len(y_true),4) if y_true else 0,
            'specificity':round(specificity,4)}


def select_threshold(model, records):
    if not records: raise ValueError('Validation rows are required')
    labels=[label for f,label in records]
    risks=[model.risk(f) for f,label in records]
    ranked=[]
    for threshold in [v/100 for v in range(10,96,2)]:
        row=metrics(labels,[int(r>=threshold) for r in risks])
        ranked.append((row['f1'],row['recall'],-row['fp'],threshold))
    return max(ranked)[3]


def train_pipeline(manifest=MANIFEST, image_root=DATA, save_model=MODEL, report_dir=REPORTS):
    rows=load_manifest(manifest)
    seen=set(); samples={'train':[],'validation':[],'test':[]}
    rows_by_split={'train':[],'validation':[],'test':[]}
    for row in rows:
        if row['file'] in seen: raise ValueError('Duplicate file in manifest')
        seen.add(row['file'])
        if row['split'] not in samples: raise ValueError('Unknown split')
        path=Path(image_root)/row['file']
        evidence=extract(path)
        label=int(row['label'])
        samples[row['split']].append((evidence.features,label))
        rows_by_split[row['split']].append((row, evidence))
    if not all(samples.values()): raise ValueError('Each split must have samples')
    model=train(samples['train'])
    model.threshold=select_threshold(model,samples['validation'])
    model.save(save_model)
    all_metrics={}
    for split in ['train','validation','test']:
        truth=[label for features,label in samples[split]]
        pred=[model.predict(features) for features,label in samples[split]]
        all_metrics[split]=metrics(truth,pred)
    report={'protocol': 'Train 60%; threshold selected on validation 20%; untouched test 20%.',
            'dataset':'320 independently generated synthetic document images; not real-world financial documents',
            'threshold':model.threshold,'splits':all_metrics,
            'limitations':'These are synthetic patch-artifact recognition results, not reliable real-world fraud detection.'}
    report_dir=Path(report_dir); report_dir.mkdir(parents=True,exist_ok=True)
    # Any previous OCR test audit was measured against a different model or dataset.
    # Remove it on retraining; re-run the independent audit explicitly.
    for old_report in ('ocr_audit.json','ocr_test_predictions.csv','ocr_error_analysis.csv'):
        (report_dir/old_report).unlink(missing_ok=True)
    (report_dir/'evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    with (report_dir/'test_predictions.csv').open('w',newline='',encoding='utf-8') as f, \
         (report_dir/'error_analysis.csv').open('w',newline='',encoding='utf-8') as errfile:
        fields=['file','document_type','label','risk_score','prediction','edit_type','error_type']
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        errors=csv.DictWriter(errfile,fieldnames=fields); errors.writeheader()
        for row,e in rows_by_split['test']:
            pred=model.predict(e.features); truth=int(row['label'])
            kind='FALSE_POSITIVE' if pred and not truth else ('FALSE_NEGATIVE' if truth and not pred else 'CORRECT')
            result={'file':row['file'],'document_type':row['document_type'],
                    'label':truth,'risk_score':round(model.risk(e.features),5),
                    'prediction':pred,'edit_type':row['edit_type'],'error_type':kind}
            writer.writerow(result)
            if kind!='CORRECT':errors.writerow(result)
    return model,report


def score_dataset(rows,model,image_root=DATA):
    output=[]
    for row in rows:
        result=extract(Path(image_root)/row['file'])
        risk=model.risk(result.features)
        output.append({**row,'risk_score':round(risk,5),'prediction':'REVIEW' if risk>=model.threshold else 'NO FLAG',
                       'hot_zone':result.suspicious_region,'description':result.description})
    return output
