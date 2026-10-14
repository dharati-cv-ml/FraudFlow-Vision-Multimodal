# Model card — FraudFlow Vision (corrected synthetic dataset)

## Intended use
Portfolio prototype for offline **visual replacement-artifact screening** on fictional receipt and invoice images. The optional independent OCR arithmetic check verifies a narrow numeric relationship. Neither component can authenticate documents or establish fraud.

## Features and inference
- Inputs: JPG/PNG documents; visual model resizes to 340×450 grayscale.
- Eight horizontal/vertical seam and edge features, standardized using training data only.
- Logistic regression implemented with plain Python (not a CNN or a calibrated probabilistic fraud model).
- Stored model: `models/demo_model.json`; visual decision threshold: `0.36` chosen using the **validation** split.
- Inferred visual hot-zone: heuristic from pixels, not the ground-truth editing box.
- OCR checks are **separate**: Tesseract extracts numbers and a Decimal arithmetic check tests subtotal + tax = printed total within $0.01.
- Combined decision when OCR was run: REVIEW if either visual flag or OCR mismatch. OCR absence/unreadability never establishes authenticity.

## Data and train/evaluation protocol
- 320 synthetic images: 192 train, 64 validation, 64 untouched test (class- and template-balanced).
- Generator bug corrected: `seamless_total` replacements stay monetary values; `seamless_date` stays a date.
- Invoice/receipt item prices now sum to the stated subtotal, independently of any edited total field.
- Ground truth labels, bounding boxes, original and replacement strings are *excluded* from model features, OCR field validation and prediction.
- Train weights on train; select threshold on validation; evaluate visual on test; evaluate fixed OR rule on same test, without tuning the rule to test labels.

## Held-out results on synthetic test set only

| | Visual | Visual + OCR arithmetic |
|---|---:|---:|
| Total images | 64 | 64 |
| TP / TN | 25 / 23 | 26 / 23 |
| FP / FN | 9 / 7 | 9 / 6 |
| Precision | 73.53% | 74.29% |
| Recall | 78.12% | 81.25% |
| F1 | 75.76% | 77.61% |
| Accuracy | 75.00% | 76.56% |

The OCR-assisted evaluation was run with local Tesseract and recognized all three required arithmetic fields in each of the 64 tested synthetic images (57 consistent; 7 mismatches). On real, varied documents, OCR coverage may be lower and errors higher. The results must not be generalized to unseen authentic documents.

## Known limitations and negative results
- False positives from benign seams/stamps; false negatives from subtle redraws (see `reports/error_analysis.csv`).
- Arithmetic check cannot identify edited dates/references if totals are internally consistent.
- A document can have a consistent sum but still be fraudulent; a mismatch can reflect OCR failure or a genuine bookkeeping issue rather than fraud.
- No independent real-world labeled dataset, domain-shift stress tests, confidence intervals, or score calibration.
- No automated rejection or decisions affecting payments, customers or identities.

## Reproduction
`python -m fraudvision.cli train` and (with local Tesseract) `python -m fraudvision.cli audit-test`. Original report files are checked in under `reports/`.
