"""Opt-in, pre-specified OCR arithmetic audit on the untouched test split.

The logical OR rule is fixed in advance and is NOT tuned using test labels.
The audit requires local Tesseract. Incomplete OCR must not be called clean.
"""
from __future__ import annotations
import csv, json
from collections import Counter
from pathlib import Path
from .core import load_manifest, VisualModel, metrics
from .verification import verify_image, combined_decision
from .features import extract
from .ocr import tesseract_path
from .paths import MANIFEST, DATA, MODEL, REPORTS


def audit_test(manifest=MANIFEST, image_root=DATA, model_file=MODEL, report_dir=REPORTS, progress=False):
    if not tesseract_path():
        raise RuntimeError('Local Tesseract is required for OCR-assisted evaluation; no synthetic labels are substituted.')
    model=VisualModel.load(model_file)
    rows=[r for r in load_manifest(manifest) if r['split']=='test']
    out=[]
    for idx,row in enumerate(rows,1):
        path=Path(image_root)/row['file']; features=extract(path).features
        visual=model.predict(features)
        check,_=verify_image(path)
        decision=combined_decision(bool(visual),check)
        pred=int(decision=='REVIEW'); truth=int(row['label'])
        out.append({'file':row['file'],'document_type':row['document_type'],
           'label':truth,'edit_type':row['edit_type'],'visual_score':round(model.risk(features),5),
           'visual_decision':visual,'ocr_status':check.status,'ocr_difference':check.difference or '',
           'combined_decision':pred,'combined_error':('FALSE_POSITIVE' if pred and not truth else
              'FALSE_NEGATIVE' if truth and not pred else 'CORRECT')})
        if progress and idx % 16==0:print(f'OCR audit {idx}/{len(rows)}',flush=True)
    visual_metrics=metrics([r['label'] for r in out],[r['visual_decision'] for r in out])
    combined_metrics=metrics([r['label'] for r in out],[r['combined_decision'] for r in out])
    counts=dict(Counter(r['ocr_status'] for r in out))
    result={'test_samples':len(out),'evaluation_protocol':'Untouched test only. Train on train split; threshold chosen on validation. Fixed OR rule; no test-set tuning.',
            'ocr_coverage':counts,'visual_only':visual_metrics,'ocr_assisted_combined':combined_metrics,
            'limitations':'Synthetic image modifications and a narrow known layout. OCR check only validates arithmetic, not date/reference authenticity. Missing OCR is not proof of authenticity.'}
    report_dir=Path(report_dir);report_dir.mkdir(parents=True,exist_ok=True)
    (report_dir/'ocr_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    keys=list(out[0].keys()) if out else []
    for file,items in [('ocr_test_predictions.csv',out),('ocr_error_analysis.csv',[r for r in out if r['combined_error']!='CORRECT'])]:
        with (report_dir/file).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(items)
    return result

if __name__=='__main__':
    print(json.dumps(audit_test(progress=True),indent=2))
