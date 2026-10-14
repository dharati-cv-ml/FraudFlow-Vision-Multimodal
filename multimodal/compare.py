"""Compare exactly matching held-out OCR and local VLM runs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

def compare(ocr_path,vlm_path):
    a=json.loads(Path(ocr_path).read_text(encoding='utf-8'))
    b=json.loads(Path(vlm_path).read_text(encoding='utf-8'))
    if a['method']!='ocr' or b['method']!='vlm':
        raise ValueError('Expected --ocr OCR result and --vlm VLM result')
    if a['split']!=b['split'] or a['split']!='test':
        raise ValueError('Both runs must use the same untouched test split')
    fa=[x['file'] for x in a['predictions']]
    fb=[x['file'] for x in b['predictions']]
    if fa!=fb or len(fa)!=64:
        raise ValueError('For publishable comparison both runs must cover the same ordered 64 test images')
    result={'split':'test','documents':len(fa),'comparison':{}}
    for record in [a,b]:
        result['comparison'][record['method']]={
            'model':record['model'],
            'micro_exact_match':record['evaluation']['micro_exact_match'],
            'by_question':{k:val['exact_match_accuracy'] for k,val in record['evaluation']['by_question'].items()},
            'mean_seconds_per_document':record['mean_seconds_per_document']}
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--ocr',required=True)
    p.add_argument('--vlm',required=True)
    p.add_argument('--out',default='reports/multimodal/comparison_test.json')
    args=p.parse_args()
    comparison=compare(args.ocr,args.vlm)
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(comparison,indent=2),encoding='utf-8')
    print(json.dumps(comparison,indent=2))
