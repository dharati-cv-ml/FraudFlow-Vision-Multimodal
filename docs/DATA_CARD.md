# Data card — synthetic images and ground truth

- 320 fictional JPG images at 680×900: 160 receipts, 160 invoices, with 80 edited and 80 clean of each type.
- Balanced across 192 training, 64 validation and 64 held-out test images. Independently generated record contents; templates recur across splits, which limits claims about generalized layouts.
- Generated source file: `fraudvision/generator.py`, seed 31415. Both clean and edited records receive realistic line-item prices summing to their subtotal; the printed total normally equals subtotal + 7% tax.
- Edits: date, total or payment reference replacements, some with visually subtle patches. Fixed previous generator bug so `seamless_total` replacements remain currency and `seamless_date` replacements remain dates.
- Some clean documents receive visually distracting benign PAID marks.
- `data/manifest.csv`: filename, split, document type, `label` (0=clean, 1=edited), edit type, template, ground-truth bounding box and before/after field values.
- Ground truth is **never supplied to feature extraction, OCR verification, or prediction**. It is used for test metrics, GUI benchmark inspection and labeling errors.
- All company names, account codes and financial amounts are invented. No real bank/customer documents.
- Generator limitations: narrow layouts/fonts, shared paper style, artificial edit methods and small, balanced label distribution. Real document distributions would require expert labeling and consented data.
