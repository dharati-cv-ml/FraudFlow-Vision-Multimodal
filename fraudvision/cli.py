"""Offline batch commands; no server."""
import argparse, json
from pathlib import Path
from .paths import MODEL,DATA
from .core import VisualModel,train_pipeline
from .features import extract
from .generator import generate_dataset
from .verification import verify_image, combined_decision

def main():
    p=argparse.ArgumentParser(description='Offline synthetic receipt and invoice alteration screening')
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('generate').add_argument('--count',type=int,default=320)
    sub.add_parser('train')
    inspect=sub.add_parser('inspect');inspect.add_argument('image')
    verify=sub.add_parser('verify');verify.add_argument('image')
    sub.add_parser('audit-test')
    args=p.parse_args()
    if args.command=='generate':
        print(f'Generated {len(generate_dataset(args.count))} samples')
    elif args.command=='train':
        model,report=train_pipeline();print(json.dumps(report,indent=2))
    elif args.command=='verify':
        m=VisualModel.load(MODEL);ev=extract(Path(args.image));check,_=verify_image(args.image)
        visual=m.predict(ev.features)
        print(json.dumps({'visual_score':round(m.risk(ev.features),4),'visual_decision':'REVIEW' if visual else 'NO FLAG',
                          'arithmetic_check':check.to_dict(),'combined_decision':combined_decision(bool(visual),check)},indent=2))
    elif args.command=='audit-test':
        from .ocr_audit import audit_test
        print(json.dumps(audit_test(),indent=2))
    elif args.command=='inspect':
        model=VisualModel.load(MODEL);ev=extract(Path(args.image));risk=model.risk(ev.features)
        print(json.dumps({'model_score':round(risk,4),'prediction':'REVIEW' if risk>=model.threshold else 'NO FLAG',
                          'hot_zone':ev.suspicious_region,'explanation':ev.description,
                          'caveat':'Not a real-world financial fraud probability.'},indent=2))
if __name__=='__main__':main()
