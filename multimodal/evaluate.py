"""Reproducible held-out document VQA benchmark: OCR or optional pretrained VLM.

CLI runs with an image + question input only; annotations are loaded in the
scorer after inference and NEVER passed into the inference method.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from time import perf_counter
from collections import defaultdict
from fraudvision.paths import ROOT,DATA
from .answers import build_answers,FIELDS
from .ocr_baseline import ocr_answers

def normalize(field,value):
    v=str(value or '').strip().strip('`').strip().replace('–','-')
    if not v:return ''
    if field=='document_type':
        v=v.lower().strip(' .!')
        return v if v in ('receipt','invoice') else ''
    if field=='date':
        import re
        m=re.search(r'\b\d{4}-\d{2}-\d{2}\b',v)
        return m.group(0) if m else ''
    if field=='total':
        import re
        m=re.search(r'\$?\s*([0-9][0-9,]*\.[0-9]{2})',v)
        return m.group(1).replace(',','') if m else ''
    if field=='payment_reference':
        import re
        m=re.search(r'ACCT-\d{5}',v.upper())
        return m.group(0) if m else ''
    if field=='arithmetic_consistency':
        upper=v.upper()
        return upper if upper in ('YES','NO') else ''
    return v

def score_predictions(preds,answers):
    by_file={a['file']:a for a in answers}
    values=defaultdict(lambda:{'correct':0,'total':0,'answered':0})
    detail=[]
    for item in preds:
        truth=by_file[item['file']]['answer']
        for key in FIELDS:
            actual=normalize(key,item.get('prediction',{}).get(key,''))
            expected=normalize(key,truth[key])
            good=actual==expected and bool(actual)
            vals=values[key];vals['total']+=1;vals['correct']+=int(good);vals['answered']+=int(bool(actual))
            detail.append({'file':item['file'],'field':key,'correct':int(good),'expected':expected,'prediction':actual})
    fields={k:{'exact_match_accuracy':round(v['correct']/v['total'],4) if v['total'] else None,
               'answer_coverage':round(v['answered']/v['total'],4) if v['total'] else None,**v} for k,v in values.items()}
    total=sum(v['total'] for v in values.values());correct=sum(v['correct'] for v in values.values())
    return {'documents':len(preds),'question_answers':total,'micro_exact_match':round(correct/total,4) if total else None,'by_question':fields},detail

def evaluate(method='ocr',split='test',limit=None,model_name=None,output=None):
    answers=build_answers(split=split)
    if limit is not None: answers=answers[:limit]
    if not answers:raise ValueError('No images selected')
    if method=='vlm':
        from .vlm_runner import LocalVLM, MODEL_DEFAULT
        runner=LocalVLM(model_name or MODEL_DEFAULT)
    else: runner=None
    preds=[]
    for i,item in enumerate(answers):
        file=item['file'];image_path=DATA/file
        if not image_path.is_file():raise FileNotFoundError(image_path)
        start=perf_counter()
        pred,raw=runner.answer(image_path) if runner else ocr_answers(image_path)
        elapsed=round(perf_counter()-start,3)
        preds.append({'file':file,'prediction':pred,'runtime_s':elapsed,'raw_output':raw})
        if i%10==0 or i+1==len(answers):print(f'{method.upper()}: {i+1}/{len(answers)} documents',flush=True)
    metrics,detail=score_predictions(preds,answers)
    results={'method':method,'split':split,'model':(model_name or runner.model_name) if runner else 'Tesseract OCR + deterministic parsing','evaluation':metrics,
             'mean_seconds_per_document':round(sum(p['runtime_s'] for p in preds)/len(preds),3),
             'limitations':'Synthetic generator-derived answer key; no real-world generalization claims. VQA measures printed fields, not authenticity or fraud.',
             'predictions':preds}
    path=Path(output) if output else ROOT/'reports'/'multimodal'/f'{method}_{split}_{len(answers)}.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(results,indent=2),encoding='utf-8')
    with (path.parent/(path.stem+'_errors.csv')).open('w',newline='',encoding='utf-8') as fh:
        writer=csv.DictWriter(fh,fieldnames=['file','field','correct','expected','prediction'])
        writer.writeheader();writer.writerows([d for d in detail if not d['correct']])
    print('Saved',path)
    print('Micro exact match:',metrics['micro_exact_match'])
    return results

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--method',choices=['ocr','vlm'],default='ocr')
    parser.add_argument('--split',choices=['train','validation','test'],default='test')
    parser.add_argument('--limit',type=int,default=None)
    parser.add_argument('--model',default=None)
    parser.add_argument('--output',default=None)
    args=parser.parse_args()
    evaluate(method=args.method,split=args.split,limit=args.limit,model_name=args.model,output=args.output)
