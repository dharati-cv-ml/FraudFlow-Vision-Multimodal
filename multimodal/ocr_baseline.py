"""Local OCR-only field QA baseline with no dataset-label access."""
from __future__ import annotations
from decimal import Decimal, InvalidOperation
import re
from fraudvision.ocr import read_text
from fraudvision.verification import verify_amounts_text

def ocr_answers(image_path):
    text=read_text(image_path)
    if text.startswith(('Tesseract not installed','OCR failed','OCR unavailable')):
        return {key:'' for key in ('document_type','date','total','payment_reference','arithmetic_consistency')},text
    typ='invoice' if re.search(r'\bINVOICE\b',text,re.I) else ('receipt' if re.search(r'\bRECEIPT\b',text,re.I) else '')
    date=re.search(r'(?im)^\s*(?:Purchased|Invoice\s+date)\s*:\s*(\d{4}-\d{2}-\d{2})\b',text)
    total=re.search(r'(?im)^\s*TOTAL\s*\$?\s*([0-9][0-9,]*\.[0-9]{2})\b',text)
    ref=re.search(r'(?im)Payment\s+ref(?:erence)?\s*:\s*(ACCT[-–]\d{5})\b',text)
    check=verify_amounts_text(text)
    consistency= 'YES' if check.status=='CONSISTENT' else ('NO' if check.status=='MISMATCH' else '')
    return {'document_type':typ,
            'date':date.group(1) if date else '',
            'total':'$'+total.group(1) if total else '',
            'payment_reference':ref.group(1).replace('–','-') if ref else '',
            'arithmetic_consistency':consistency},text
