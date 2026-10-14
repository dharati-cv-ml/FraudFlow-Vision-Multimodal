"""Optional local Tesseract CLI integration, no cloud or web services."""
import shutil
import subprocess
from pathlib import Path


def tesseract_path():
    found=shutil.which('tesseract')
    if found:return found
    known=Path('C:/Program Files/Tesseract-OCR/tesseract.exe')
    return str(known) if known.exists() else None


def read_text(path):
    binary=tesseract_path()
    if not binary:
        return 'Tesseract not installed or not found. OCR is optional and does not affect model predictions.'
    try:
        proc=subprocess.run([binary,str(path),'stdout','--psm','6'],capture_output=True,text=True,timeout=15,check=False)
        if proc.returncode:
            return f'OCR failed (exit {proc.returncode}): {proc.stderr[:220]}'
        return proc.stdout.strip() or '(No text recognized)'
    except (OSError,subprocess.TimeoutExpired) as exc:
        return f'OCR unavailable: {exc}'
