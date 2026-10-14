"""Synthetic fictional receipts and invoices; deterministic and independently split.

Ground-truth alteration locations are retained only for evaluations. They are NOT
provided to the model or the prediction workflow.
"""
import argparse
import csv
import random
from datetime import date, timedelta
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from .paths import IMAGES, MANIFEST

W, H = 680, 900
FIELDS = ['file', 'split', 'document_type', 'label', 'edit_type', 'template', 'bbox', 'original_value', 'replacement_value']
FONT_PATHS = [
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf',
    'C:/Windows/Fonts/arial.ttf', 'C:/Windows/Fonts/segoeui.ttf',
]
BOLD_PATHS = [
    '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
    '/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf',
    'C:/Windows/Fonts/arialbd.ttf', 'C:/Windows/Fonts/segoeuib.ttf',
]

def font(size, bold=False):
    for item in (BOLD_PATHS if bold else FONT_PATHS):
        if Path(item).exists():
            return ImageFont.truetype(item, size=size)
    return ImageFont.load_default(size=size)


def generate_one(index, rng, target):
    doc_type = 'invoice' if index % 2 else 'receipt'
    template = index % 4
    bg = [246, 250, 243, 252][template]
    base = (bg, bg - (1 if template == 2 else 0), bg - (3 if template == 1 else 0))
    im = Image.new('RGB', (W, H), base)
    d = ImageDraw.Draw(im)
    ink = (36, 49, 68)
    faded = (108, 122, 138)
    teal = (0, 139, 147)
    orange = (192, 104, 65)
    # Low-amplitude scanner/paper texture, identical process for every class.
    for _ in range(1500):
        x = rng.randrange(W); y = rng.randrange(H)
        change = rng.choice((-3, -2, 2, 3))
        d.point((x, y), fill=tuple(max(0, min(255, c+change)) for c in base))
    vendors = ['NORTHLINE SUPPLY', 'BLUE HERON GOODS', 'CIVIC OFFICE MART', 'RIVERBEND RETAIL']
    vendor = vendors[index % 4]
    number = f'{rng.randint(100000, 899999)}'
    invdate = date(2026, 1, 1) + timedelta(days=rng.randint(0, 240))
    date_text = invdate.isoformat()
    subtotal = rng.randint(1900, 48000) / 100
    tax = round(subtotal * 0.07, 2)
    total = round(subtotal + tax, 2)
    amount_text = f'${total:,.2f}'
    reference_text = f'ACCT-{rng.randint(10000,99999)}'
    pos = {}
    if doc_type == 'invoice':
        d.rounded_rectangle((40, 36, 640, 126), radius=12, fill=(25, 42, 61))
        d.text((65, 56), 'INVOICE', font=font(39, True), fill=(245, 250, 250))
        d.text((444, 83), 'DOCUMENT COPY', font=font(13), fill=(175, 227, 224))
        d.text((54, 157), vendor, font=font(21, True), fill=ink)
        d.text((54, 193), 'Billing contact: accounts@sample.invalid', font=font(14), fill=faded)
        d.text((54, 245), f'Invoice ID:  INV-{number}', font=font(19), fill=ink)
        d.text((54, 281), 'Invoice date:', font=font(17), fill=ink)
        pos['date'] = ((196, 275, 404, 313), date_text, 18)
        d.text((196, 281), date_text, font=font(18), fill=ink)
        d.line((54, 364, 626, 364), fill=(152, 167, 174), width=2)
        d.text((60, 389), 'DESCRIPTION', font=font(16, True), fill=teal)
        d.text((516, 389), 'AMOUNT', font=font(16, True), fill=teal)
        invoice_costs = [round(subtotal*0.35, 2), round(subtotal*0.40, 2)]
        invoice_costs.append(round(subtotal - sum(invoice_costs), 2))
        for i, (name, cost) in enumerate(zip(['Document processing', 'Secure handling', 'Service support'], invoice_costs)):
            d.text((60, 430 + i*57), name, font=font(17), fill=ink)
            d.text((507, 430 + i*57), f'${cost:,.2f}', font=font(17), fill=ink)
        d.line((54, 640, 626, 640), fill=(174, 183, 191), width=2)
        d.text((392, 669), 'Subtotal', font=font(17), fill=ink)
        d.text((530, 669), f'${subtotal:,.2f}', font=font(16), fill=ink)
        d.text((392, 702), 'Tax (7%)', font=font(17), fill=ink)
        d.text((530, 702), f'${tax:,.2f}', font=font(16), fill=ink)
        d.text((392, 753), 'TOTAL', font=font(19, True), fill=ink)
        d.text((506, 752), amount_text, font=font(19, True), fill=ink)
        pos['total'] = ((496, 745, 638, 788), amount_text, 19)
        d.text((60, 811), 'Payment reference:', font=font(17), fill=ink)
        d.text((264, 811), reference_text, font=font(16), fill=ink)
        pos['reference'] = ((253, 805, 477, 845), reference_text, 16)
        d.text((54, 864), 'Demonstration document - not a real invoice', font=font(13), fill=faded)
    else:
        d.text((104, 49), vendor, font=font(27, True), fill=ink)
        d.text((198, 92), 'CUSTOMER RECEIPT', font=font(15, True), fill=teal)
        d.line((53, 146, 627, 146), fill=(182, 195, 201), width=2)
        d.text((58, 174), f'Receipt no:  RC-{number}', font=font(18), fill=ink)
        d.text((58, 213), 'Purchased:', font=font(17), fill=ink)
        d.text((175, 213), date_text, font=font(18), fill=ink)
        pos['date'] = ((169, 207, 376, 248), date_text, 18)
        d.line((55, 273, 625, 273), fill=(175, 186, 195), width=2)
        d.text((60, 305), 'ITEM', font=font(16, True), fill=teal)
        d.text((525, 305), 'PRICE', font=font(16, True), fill=teal)
        receipt_costs = [round(subtotal*0.20, 2), round(subtotal*0.30, 2), round(subtotal*0.25, 2)]
        receipt_costs.append(round(subtotal - sum(receipt_costs), 2))
        for i, (name, cost) in enumerate(zip(['Office item', 'Packing material', 'Digital service', 'Handling charge'], receipt_costs)):
            d.text((60, 351+i*65), name, font=font(17), fill=ink)
            d.text((503, 351+i*65), f'${cost:,.2f}', font=font(17), fill=ink)
        d.line((55, 650, 625, 650), fill=(175, 186, 195), width=2)
        d.text((364, 678), 'Subtotal', font=font(17), fill=ink)
        d.text((517, 678), f'${subtotal:,.2f}', font=font(16), fill=ink)
        d.text((364, 713), 'Tax', font=font(17), fill=ink)
        d.text((517, 713), f'${tax:,.2f}', font=font(16), fill=ink)
        d.text((360, 762), 'TOTAL', font=font(20, True), fill=ink)
        d.text((497, 760), amount_text, font=font(19, True), fill=ink)
        pos['total'] = ((486, 753, 638, 794), amount_text, 19)
        d.text((60, 819), 'Payment ref:', font=font(17), fill=ink)
        d.text((209, 819), reference_text, font=font(17), fill=ink)
        pos['reference'] = ((202, 813, 432, 853), reference_text, 17)
        d.text((123, 868), 'Synthetic example - no financial value', font=font(13), fill=faded)
    edit_type = 'none'; bbox = ''; replacement = ''; original = ''
    if not target and index % 13 == 3:
        # A legitimate handling sticker can look like a rectangular patch.
        # It is not an altered financial field, and is a useful false-alarm challenge.
        d.rectangle((350, 823, 456, 859), fill=tuple(max(200, c-18) for c in base))
        d.text((363, 832), 'PAID', font=font(16, True), fill=(93, 113, 128))
    if target:
        edit_type = ['total', 'date', 'reference'][index % 3]
        box, original, size = pos[edit_type]
        x1, y1, x2, y2 = box
        # Deliberately mimic a pasted-over field, with slight paper/ink mismatch.
        seamless = (index % 7 == 0)
        shift = 0 if seamless else rng.choice((-23, -17, -12, 12, 18, 24))
        patch = tuple(max(202, min(255, c+shift)) for c in base)
        d.rectangle((x1, y1, x2, y2), fill=patch, outline=(patch if seamless else tuple(max(180, c-4) for c in patch)), width=1)
        # Select replacement using the canonical field name before annotating
        # a cosmetic variation (otherwise seamless totals became account IDs).
        edit_field = edit_type
        if edit_field == 'total':
            replacement = f'${total+rng.randint(25, 120):,.2f}'
        elif edit_field == 'date':
            replacement = (invdate + timedelta(days=rng.randint(15, 55))).isoformat()
        else:
            replacement = f'ACCT-{rng.randint(10000,99999)}'
        d.text((x1+5, y1+5), replacement, font=font(size, bold=edit_field=='total'), fill=(38, 43, 52) if shift>0 else (37, 47, 60))
        if seamless: edit_type = 'seamless_' + edit_field
        bbox = f'{x1},{y1},{x2},{y2}'
    # This answer key is ONLY for multimodal benchmark annotations. It is
    # never passed to OCR, the visual classifier or the pretrained VLM.
    printed = {'date': date_text, 'total': amount_text, 'reference': reference_text}
    if target:
        printed[edit_field] = replacement
    answer_key = {
        'document_type': doc_type,
        'date': printed['date'],
        'total': printed['total'],
        'payment_reference': printed['reference'],
        'arithmetic_consistency': 'NO' if target and edit_field == 'total' else 'YES',
    }
    return im, {'document_type': doc_type, 'label': int(bool(target)), 'edit_type':edit_type,
                'template':f'{doc_type}-{template}', 'bbox':bbox, 'original_value':original,
                'replacement_value':replacement, 'benchmark_answer_key': answer_key}


def generate_dataset(count=320, seed=31415, out=IMAGES, manifest=MANIFEST):
    if count < 60 or count % 20:
        raise ValueError('count must be >= 60 and divisible by 20')
    out = Path(out); manifest=Path(manifest)
    out.mkdir(parents=True, exist_ok=True); manifest.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    rows=[]
    # Equal class balance in each independent split. No derived or paired image
    # crosses train/validation/test boundaries.
    splits = [('train', int(count*.6)), ('validation', int(count*.2)), ('test', count-int(count*.8))]
    index=0
    for split, size in splits:
        # Balance labels within every layout/template, including both receipt
        # and invoice types. Never let a template or split predict the label.
        targets = {}
        for template in range(4):
            ids = [j for j in range(size) if (index+j) % 4 == template]
            edited_count = len(ids)//2 + (len(ids)%2 if template in (0,1) else 0)
            flags = [True]*edited_count + [False]*(len(ids)-edited_count)
            rng.shuffle(flags)
            targets.update(dict(zip(ids, flags)))
        for j in range(size):
            is_edit = targets[j]
            image, metadata = generate_one(index, rng, is_edit)
            name = f'doc_{index:04d}.jpg'
            image.save(out/name, quality=91, subsampling=0)
            rows.append({'file':f'images/{name}', 'split':split, **{k:v for k,v in metadata.items() if k in FIELDS}})
            index += 1
    with manifest.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    return rows

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--count', type=int, default=320)
    p.add_argument('--seed', type=int, default=31415)
    args=p.parse_args()
    print(f'Generated {len(generate_dataset(args.count,args.seed))} synthetic documents')
