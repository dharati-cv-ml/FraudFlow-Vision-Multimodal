# FraudFlow Vision — corrected synthetic held-out evaluation

Dataset: 320 fictional receipts/invoices, generator seed 31415. Train: 192 images; validation: 64; test: 64. Labels and templates balanced across independent splits.

The synthetic generator was corrected to ensure *seamless total edits stay numeric currency values* and receipt/invoice item lines add to the subtotal. All 320 images and model outputs were regenerated. **Previous benchmark values were invalidated.**

## Visual model (trained on train, threshold 0.36 chosen on validation)

| Split | TP | TN | FP | FN | F1 |
|---|---:|---:|---:|---:|---:|
| Train | 77 | 75 | 21 | 19 | 79.38% |
| Validation | 26 | 27 | 5 | 6 | 82.54% |
| Test | 25 | 23 | 9 | 7 | **75.76%** |

## OCR-assisted combined test audit (opt-in local Tesseract)

The fixed, predeclared rule reviews documents flagged by the *visual* model or an *OCR subtotal/tax/total mismatch*. No ground-truth labels are used during prediction. The combined rule was **not** tuned on test labels.

| Test | TP | TN | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Visual alone | 25 | 23 | 9 | 7 | 73.53% | 78.12% | 75.76% |
| Combined visual + OCR | 26 | 23 | 9 | 6 | 74.29% | 81.25% | **77.61%** |

OCR parse coverage: 57 consistent, 7 mismatch, no unreadable on these 64 *synthetic* images. This is **not** general OCR coverage or fraud-detection performance in practice.

- `evaluation.json`: visual metrics
- `test_predictions.csv`: visual predictions with classification error types
- `error_analysis.csv`: only visual false positives and false negatives
- `ocr_audit.json`: separate OCR-assisted metrics and coverage
- `ocr_test_predictions.csv`: combined per-test predictions and arithmetic statuses
- `ocr_error_analysis.csv`: only combined false positives and false negatives

**Responsible interpretation:** All figures relate only to this constructed dataset; no claims about forensic reliability on real receipts or invoices. A review indicator triggers human investigation, never an assertion of fraud.
