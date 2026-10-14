"""Export training/validation prompts for a future image+text LoRA experiment.

No test examples exported. Metadata remains outside the prediction prompt.
No GPU training is run by this tool.
"""
import argparse
import json
from pathlib import Path
from .answers import build_answers
from .vlm_runner import QUESTIONS


def export(split,output):
    if split not in ('train','validation'):
        raise ValueError('LoRA input must use train or validation; never test')
    examples=build_answers(split=split)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w',encoding='utf-8') as fh:
        for e in examples:
            fh.write(json.dumps({'image':e['file'], 'prompt':QUESTIONS,
                                 'response':json.dumps(e['answer'],sort_keys=True)},ensure_ascii=False)+'\n')
    return len(examples)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['train','validation'],required=True)
    p.add_argument('--out',required=True);args=p.parse_args()
    print('Exported',export(args.split,args.out),'examples to',args.out)
