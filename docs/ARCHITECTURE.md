# FraudFlow Vision — architecture

```text
320 synthetic JPGs ──> manifest.csv (label + bbox, evaluation ONLY)
        │
        ├── TRAIN 192 → image-derived features → normalization + logistic weights
        ├── VALIDATION 64 → select screening threshold
        └── TEST 64 ────────────────┐
                                    v
Image → Pillow visual features → logistic score → VISUAL REVIEW/NO FLAG
  ├── inferred pixel-based hot-zone (no label or bbox access)
  └── optional local OCR (Tesseract) → subtotal/tax/total → Decimal arithmetic
        └── CONSISTENT/MISMATCH/UNREADABLE/OCR_UNAVAILABLE
                                   │
               fixed OR rule: REVIEW if visual flag or numeric MISMATCH
                                   │
             CLI / Windows Tkinter GUI / evidence JSON / error-analysis CSV
```

The training and threshold stages do not read test labels. The combined rule is fixed in advance; ground truth is used only to compute metrics and error examples. All data is fictional/synthetic. No browser, network connection or API server is needed.
