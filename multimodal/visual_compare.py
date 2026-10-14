"""Pure data layer for *post-inference* VQA result visualisation.

The generated answer key is used for SCORING ONLY. It is never passed to
an OCR/VLM inference runner. All comparisons use matching document IDs.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import csv
import json
import re
from .evaluate import normalize
from .answers import FIELDS, build_answers
from fraudvision.paths import ROOT, DATA

REPORT_DIR = ROOT / 'reports' / 'multimodal'


def load_report(path: str | Path, method: str | None = None) -> dict:
    path = Path(path)
    obj = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(obj, dict) or obj.get('method') not in ('vlm', 'ocr'):
        raise ValueError(f'Invalid VQA report: {path}')
    if method and obj['method'] != method:
        raise ValueError(f'Expected {method} report, got {obj["method"]}')
    preds = obj.get('predictions')
    if not isinstance(preds, list) or not preds:
        raise ValueError(f'No predictions in {path}')
    seen = set()
    for p in preds:
        if not isinstance(p, dict) or not isinstance(p.get('prediction'), dict):
            raise ValueError(f'Invalid prediction in {path}')
        name = p.get('file')
        if not isinstance(name, str) or not re.fullmatch(r'images/doc_\d{4}\.jpg', name):
            raise ValueError(f'Unexpected image path in {path}: {name}')
        if name in seen:
            raise ValueError(f'Duplicate image ID {name} in {path}')
        seen.add(name)
    obj['_path'] = str(path)
    return obj


def discover_reports(folder: Path = REPORT_DIR) -> tuple[Path | None, Path | None]:
    """Choose the largest available VLM run; prioritize matching OCR count."""
    choices = {'vlm': [], 'ocr': []}
    if folder.is_dir():
        for p in folder.glob('*_test_*.json'):
            m = re.fullmatch(r'(vlm|ocr)_test_(\d+)\.json', p.name)
            if m: choices[m.group(1)].append((int(m.group(2)), p))
    vlm = max(choices['vlm'], default=(0, None))[1]
    ocr_choices = choices['ocr']
    if vlm and ocr_choices:
        match_count = int(re.search(r'_test_(\d+)\.json$', vlm.name).group(1))
        same = [x for x in ocr_choices if x[0] == match_count]
        ocr = max(same if same else ocr_choices)[1]
    else:
        ocr = max(ocr_choices, default=(0, None))[1]
    return ocr, vlm


def assemble(ocr: dict | None = None, vlm: dict | None = None, truth: dict | None = None) -> dict:
    """Pair exact overlapping images; return field-level evidence & measured stats.

    Supports either report alone. Missing answers are incorrect if that model
    was evaluated, but a missing entire report is shown as NOT RUN, not zero.
    """
    if ocr is None and vlm is None:
        return {'documents': [], 'metrics': {}, 'by_field': {}}
    if ocr and vlm and ocr.get('split') != vlm.get('split'):
        raise ValueError('OCR and VLM results belong to different splits')
    split = (vlm or ocr).get('split')
    if split != 'test':
        raise ValueError('Visual comparison accepts the held-out test split only')
    if truth is None:
        truth = {x['file']: x['answer'] for x in build_answers(split='test')}
    maps = {name: {p['file']: p for p in report['predictions']} if report else {}
            for name, report in (('ocr', ocr), ('vlm', vlm))}
    if ocr and vlm:
        overlap = set(maps['ocr']) & set(maps['vlm'])
        files = [p['file'] for p in vlm['predictions'] if p['file'] in overlap]
        if not files:
            raise ValueError('No matching document IDs between OCR and VLM reports')
    else:
        report = vlm or ocr
        files = [p['file'] for p in report['predictions']]
    metrics = {name: {'correct': 0, 'total': 0, 'time_s': 0.0, 'documents': len(files)}
               for name, rep in (('ocr', ocr), ('vlm', vlm)) if rep}
    by_field = {field: {name: {'correct': 0, 'total': 0}
                        for name in metrics} for field in FIELDS}
    docs = []
    for file in files:
        if file not in truth:
            raise ValueError(f'Unknown held-out image ID: {file}')
        if not (DATA / file).is_file():
            raise FileNotFoundError(f'Image missing: {DATA / file}')
        records = {name: maps[name].get(file) for name in ('ocr','vlm')}
        fields = []
        for field in FIELDS:
            expected = normalize(field, truth[file][field])
            row = {'field': field, 'expected': expected}
            for name in ('ocr', 'vlm'):
                prediction = records[name]
                if prediction is None:
                    row[name] = None
                    row[name + '_correct'] = None
                    continue
                got = normalize(field, prediction.get('prediction', {}).get(field, ''))
                correct = bool(got) and got == expected
                row[name] = got
                row[name + '_correct'] = correct
                m = metrics[name];m['correct'] += int(correct);m['total'] += 1
                by_field[field][name]['correct'] += int(correct)
                by_field[field][name]['total'] += 1
            fields.append(row)
        for name, record in records.items():
            if record:
                metrics[name]['time_s'] += float(record.get('runtime_s') or 0)
        docs.append({'file': file, 'fields': fields,
                     'runtime': {name: float(rec.get('runtime_s') or 0) if rec else None
                                 for name, rec in records.items()},
                     'raw': {name: rec.get('raw_output', '') if rec else '' for name, rec in records.items()},
                     'vlm_errors': sum(x['vlm_correct'] is False for x in fields),
                     'ocr_errors': sum(x['ocr_correct'] is False for x in fields)})
    for name, m in metrics.items():
        m['accuracy'] = m['correct'] / m['total'] if m['total'] else None
        m['avg_latency_s'] = m['time_s'] / len(files) if files else None
    for field in FIELDS:
        for name, m in by_field[field].items():
            m['accuracy'] = m['correct'] / m['total'] if m['total'] else None
    return {'documents': docs, 'metrics': metrics, 'by_field': by_field,
            'paired': bool(ocr and vlm), 'split': split}


def export_rows(assembled: dict, path: str | Path) -> int:
    fields = ['file','question','ground_truth','ocr_answer','ocr_correct','vlm_answer','vlm_correct']
    count = 0
    with Path(path).open('w',newline='',encoding='utf-8-sig') as h:
        w = csv.DictWriter(h, fieldnames=fields)
        w.writeheader()
        for doc in assembled['documents']:
            for x in doc['fields']:
                w.writerow({'file': doc['file'], 'question': x['field'],
                            'ground_truth': x['expected'], 'ocr_answer': x['ocr'] or '',
                            'ocr_correct': '' if x['ocr_correct'] is None else int(x['ocr_correct']),
                            'vlm_answer': x['vlm'] or '',
                            'vlm_correct': '' if x['vlm_correct'] is None else int(x['vlm_correct'])})
                count += 1
    return count
