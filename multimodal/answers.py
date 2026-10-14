"""Recreate the image generator answer key deterministically WITHOUT modifying image files.

WARNING: secret benchmark annotations are evaluation-only. Never pass the answer key,
original/replacement values, class label, or edit location to an inference runner.
"""
from __future__ import annotations
import csv
import random
from pathlib import Path
from fraudvision.generator import generate_one
from fraudvision.paths import MANIFEST, DATA

FIELDS = ('document_type','date','total','payment_reference','arithmetic_consistency')

def build_answers(manifest=MANIFEST, seed=31415, *, split=None):
    with Path(manifest).open(newline='',encoding='utf-8') as fh:
        rows=list(csv.DictReader(fh))
    if len(rows) % 20 or len(rows)<60:
        raise ValueError('Dataset count must follow the generator split scheme')
    count=len(rows);rng=random.Random(seed);answers=[];index=0
    splits=[('train',int(count*.6)),('validation',int(count*.2)),('test',count-int(count*.8))]
    for group,size in splits:
        targets={}
        for template in range(4):
            ids=[j for j in range(size) if (index+j)%4==template]
            n=len(ids)//2+(len(ids)%2 if template in (0,1) else 0)
            flags=[True]*n+[False]*(len(ids)-n)
            rng.shuffle(flags)
            targets.update(dict(zip(ids,flags)))
        for j in range(size):
            original_row=rows[index]
            if original_row['split']!=group or original_row['file']!=f'images/doc_{index:04d}.jpg':
                raise RuntimeError(f'Dataset order changed at index {index}')
            _,meta=generate_one(index,rng,targets[j])
            for field in ('label','document_type','edit_type','original_value','replacement_value'):
                if str(meta[field])!=str(original_row[field]):
                    raise RuntimeError(f'Generator/manifest mismatch at {index}: {field}')
            answers.append({'file': original_row['file'], 'split':group, 'answer':meta['benchmark_answer_key']})
            index+=1
    return [a for a in answers if not split or a['split']==split]


def questions_for_display():
    return {
      'document_type': 'Is this document a receipt or an invoice? Reply receipt or invoice.',
      'date': 'What is the printed purchased/invoice date in YYYY-MM-DD format?',
      'total': 'What is the printed TOTAL amount? Include the currency symbol.',
      'payment_reference': 'What is the printed payment reference, e.g. ACCT-12345?',
      'arithmetic_consistency': 'Does the printed TOTAL equal Subtotal plus Tax (within one cent)? Reply YES or NO.',
    }
