"""OCR-based numerical verification, independent of visual edit detection.

This module only inspects recognized text. It never accepts a synthetic label,
edit type or ground-truth bounding box as an input to its decision.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from decimal import Decimal, InvalidOperation
import re
from pathlib import Path
from .ocr import read_text, tesseract_path

_MONEY = re.compile(r'(?:\$\s*)?([0-9][0-9,]*\.[0-9]{2})')

@dataclass
class AmountCheck:
    status: str # CONSISTENT / MISMATCH / UNREADABLE / OCR_UNAVAILABLE
    subtotal: str | None
    tax: str | None
    printed_total: str | None
    expected_total: str | None
    difference: str | None
    message: str
    def to_dict(self): return asdict(self)


def _money_from(line):
    matches = list(_MONEY.finditer(line))
    if not matches: return None
    try: return Decimal(matches[-1].group(1).replace(',', ''))
    except InvalidOperation: return None


def verify_amounts_text(text: str) -> AmountCheck:
    """Check subtotal + tax == printed total within one cent.

    Fail closed as *unreadable* when any field is absent/ambiguous; do not call
    such a document verified or clear. Fraud detection requires other evidence.
    """
    if not text or text.startswith(('Tesseract not installed', 'OCR failed', 'OCR unavailable')):
        return AmountCheck('OCR_UNAVAILABLE',None,None,None,None,None,'Local OCR unavailable; visual screening remains independent.')
    values={'subtotal':[], 'tax':[], 'total':[]}
    for line in text.splitlines():
        normalized=line.strip().lower()
        if re.match(r'^(sub\s*total)\b',normalized): key='subtotal'
        elif re.match(r'^tax\b',normalized): key='tax'
        elif re.match(r'^total\b',normalized): key='total'
        else: continue
        value=_money_from(line)
        if value is not None: values[key].append(value)
    if any(len(items)!=1 for items in values.values()):
        msg='Could not uniquely read subtotal, tax and total; no arithmetic decision made.'
        return AmountCheck('UNREADABLE',None,None,None,None,None,msg)
    subtotal=values['subtotal'][0];tax=values['tax'][0];total=values['total'][0]
    expected=subtotal+tax
    difference=(total-expected).quantize(Decimal('0.01'))
    mismatch=abs(difference)>Decimal('0.01')
    return AmountCheck('MISMATCH' if mismatch else 'CONSISTENT',f'{subtotal:.2f}',f'{tax:.2f}',
        f'{total:.2f}',f'{expected:.2f}',f'{difference:+.2f}',
        ('Arithmetic mismatch: printed total differs from subtotal + tax.' if mismatch
         else 'Subtotal + tax agrees with printed total (within $0.01).'))


def verify_image(path: str | Path):
    text=read_text(path)
    return verify_amounts_text(text), text


def combined_decision(visual_review: bool, check: AmountCheck | None):
    """Predeclared logical OR; no ground truth or test-set tuning."""
    return 'REVIEW' if visual_review or (check is not None and check.status=='MISMATCH') else 'NO FLAG'
