# FraudFlow Vision — Explainable Receipt & Invoice Alteration Screening

**Offline Python desktop application | Document imagery + visual ML + optional local OCR arithmetic verification**

FraudFlow Vision demonstrates an end-to-end **synthetic document alteration screening** workflow. Browse 320 fictional receipt and invoice images, compare two independent signals (visual patch-artifact screening and OCR subtotal/tax/total consistency), inspect a suggested visual region, and review transparent, held-out evaluation results and error cases.

> **Scope:** Engineering portfolio prototype. The project detects selected artificial alteration patterns; it cannot determine whether a document is genuine or whether fraud occurred. No personal, bank, company or employer document data is included. **NO FLAG does not mean authenticated.**

![Inspection Studio with independent OCR arithmetic evidence](screenshots/inspection-studio.png)

## Actual desktop screens

| Visual inspection + OCR | Analytics dashboard |
|---|---|
| ![Synthetic receipt with detected total mismatch](screenshots/inspection-studio.png) | ![Scores and review flags](screenshots/analytics-dashboard.png) |

| Dataset Explorer | Held-out Model Evaluation |
|---|---|
| ![Browsable predictions with synthetic truth](screenshots/dataset-explorer.png) | ![Visual and OCR-assisted metrics](screenshots/model-evaluation.png) |

![Held-out test false-positive and false-negative examples](screenshots/error-analysis.png)


## Optional multimodal research experiment (OCR vs Vision-Language Model)

A new **document visual question answering** benchmark compares local Tesseract OCR + field rules against an optional pretrained local SmolVLM2-500M vision-language model on five questions per image: document type, date, total, payment reference and arithmetic consistency. The **64-image synthetic held-out test** has 320 answer targets; generated truth is never passed to inference.

The OCR baseline has been executed and reached **320/320 exact matches on the highly regular synthetic test**. The pretrained VLM and optional LoRA have **not yet been executed or measured**; do not claim a VLM performance result until the weights are downloaded and evaluated. This experiment evaluates reading/comprehension, **not actual financial fraud**.

Start with:

```powershell
py -3.14 -m multimodal.evaluate --method ocr --split test
```

See [`docs/VLM_STUDY.md`](docs/VLM_STUDY.md) for reproducible commands and [`docs/LORA_PROTOCOL.md`](docs/LORA_PROTOCOL.md) for a compute-dependent, not-yet-run fine-tuning plan. Existing GUI and visual model functionality are unchanged.

## Implemented features

- **Real image files:** 320 generated JPG images: 160 invoices + 160 receipts; each type has 80 clean and 80 edited samples. Different document values and paper textures are independently generated.
- **Corrected synthetic generator:** Edited `total` fields remain currency values even when the overlay is marked `seamless_total`. Same for date and reference edits. Original and replacement values are stored separately as ground truth; displayed line items now add to the displayed subtotal.
- **Pixel-based visual model:** Eight handcrafted horizontal/vertical edge and seam features extracted with Pillow; pure-Python logistic regression, standardization learned on training data, threshold chosen from validation F1. No TensorFlow, PyTorch, SciPy or scikit-learn required.
- **On-demand OCR arithmetic:** Optional local Tesseract reads printed **Subtotal**, **Tax**, and **TOTAL** and checks whether subtotal + tax matches the printed total to within $0.01. It reports `CONSISTENT`, `MISMATCH`, `UNREADABLE`, or `OCR_UNAVAILABLE`.
- **Separate evidence and combined review:** Visual screening and OCR checking appear independently. After OCR is run for an image, the predefined combined decision is REVIEW if **either** the visual classifier flags it **or** arithmetic mismatch is detected. Missing OCR never counts as verified.
- **Inspection Studio:** Native Windows-style GUI, image gallery, inferred hot-zone overlay (not copied from generator truth), and evidence export as JSON/annotated PNG.
- **Analytics Dashboard:** Score histogram and counts across bundled samples; visual model scores are **uncalibrated**, not financial fraud probabilities.
- **Dataset Explorer and Error Analysis:** Search/filter records, inspect known synthetic ground truth, and browse false positives/false negatives from the held-out **visual model** test set.
- **Model & Evaluation:** Train/validation/test protocol, confusion matrix, separate visual-only and OCR-assisted evaluations, plus machine-readable reports.
- **Engineering:** Python `unittest`, GitHub Actions CI, model persistence, CSV export, offline command-line evaluation. No localhost, external server, OpenAI API or network required.

## Two complementary inspection signals

```text
Receipt / invoice image (JPG, PNG)
  ├── VISUAL PATH: resize and grayscale → 8 seam/edge features
  │       → standardized logistic regression → visual score
  │       → validation-selected threshold → VISUAL REVIEW / NO FLAG
  │       → pixel-derived *suggested* hot-zone
  │
  └── OPTIONAL OCR PATH: local Tesseract → printed subtotal, tax, total
          → Decimal arithmetic verification → CONSISTENT / MISMATCH / UNREADABLE

Combined review rule (only when OCR requested):
  REVIEW if (visual review) OR (OCR arithmetic MISMATCH)
  NO FLAG otherwise — never a guarantee of authenticity.
```

**Limitations:** A visually modified *date* or *payment reference* can have a mathematically consistent total. An original receipt can also contain a legitimate formatting variation. Image artifacts and arithmetic inconsistencies are **risk indicators**, not fraud labels. OCR is not currently used to train the logistic-regression classifier.

## Synthetic dataset and ground truth

| Split | Clean | Edited | Total |
|---|---:|---:|---:|
| Training | 96 | 96 | 192 |
| Validation | 32 | 32 | 64 |
| Test | 32 | 32 | 64 |
| **Total** | **160** | **160** | **320** |

The generator knows which fields it drew and which were replaced; it stores this *ground truth* in `data/manifest.csv` with `file`, `split`, `document_type`, `label`, `edit_type`, `bbox`, `original_value`, and `replacement_value`. **The model does not receive these labels, original values, replacement values, or bounding boxes to make predictions.** They are used for evaluation and transparent inspection only.

New sample variations include *total edits*, *date edits*, *reference edits*, more subtle patch redraws, and benign marks on some clean documents. Both classes occur across receipt/invoice layouts and splits.

## Reproduced results (fixed synthetic dataset, seed 31415)

**VISUAL-ONLY, 64 held-out test images**

| Metric | Result |
|---|---:|
| Precision | 73.53% |
| Recall | 78.12% |
| F1 | **75.76%** |
| Accuracy | 75.00% |
| TP / TN / FP / FN | 25 / 23 / 9 / 7 |

**OCR-ASSISTED COMBINED RULE, same 64 held-out images (local Tesseract)**

| Metric | Result |
|---|---:|
| Precision | 74.29% |
| Recall | 81.25% |
| F1 | **77.61%** |
| Accuracy | 76.56% |
| TP / TN / FP / FN | 26 / 23 / 9 / 6 |

OCR audit coverage on this particular test: 57 arithmetic-consistent, 7 mismatched, 0 unreadable. Both result sets are **synthetic holdout scores only**; they do not measure accuracy on real receipts, invoices, or financial fraud cases.

**Leakage controls:** Images are generated independently for each split. Logistic weights and feature normalization use training data only; the `0.36` visual threshold was chosen on validation data. The combined logic is a predefined OR rule and was not tuned on the held-out test labels. Synthetic truth is used only to calculate metrics and list errors.

Reproducible output files: [`reports/evaluation.json`](reports/evaluation.json), [`reports/test_predictions.csv`](reports/test_predictions.csv), [`reports/error_analysis.csv`](reports/error_analysis.csv), [`reports/ocr_audit.json`](reports/ocr_audit.json), [`reports/ocr_test_predictions.csv`](reports/ocr_test_predictions.csv), and [`reports/ocr_error_analysis.csv`](reports/ocr_error_analysis.csv). A concrete sample appears in [`reports/example_doc_0006_evidence.json`](reports/example_doc_0006_evidence.json) with [`annotated image`](reports/example_doc_0006_annotated.png).

## Run on Windows (Python 3.14)

Extract the ZIP so `README.md` is located inside `D:\FraudFlow-Document-Forensics-Verified` and run:

```powershell
cd D:\FraudFlow-Document-Forensics-Verified
py -3.14 -m pip install -r requirements.txt
py -3.14 -m fraudvision.desktop
```

Alternatively, double-click `RUN_WINDOWS.bat`. **No browser/localhost required.** Pillow is the sole pip dependency; Tkinter normally ships with the python.org Windows installer.

**Optional OCR:** Install [Tesseract OCR](https://github.com/tesseract-ocr/tesseract), with `tesseract.exe` available on PATH or at `C:\Program Files\Tesseract-OCR\tesseract.exe`. Select an image and click **Verify totals (OCR)**. Click **Extract OCR text** for the recognized text. Other features operate when OCR is unavailable.

### Five-minute desktop tour

1. Open **Inspection Studio**, choose `doc_0006.jpg`, click **Verify totals (OCR)**, and view the arithmetic difference (`271.53 + 19.01 = 290.54` vs printed `$351.54`).
2. Toggle **Show inferred visual hot-zone**; the overlay is a pixel heuristic, **not** the stored ground-truth box.
3. Visit **Analytics Dashboard** for actual model-score distributions and review counts (visual-only unless each record is individually OCR-checked).
4. Explore **Dataset Explorer**, including separate labels and predictions. Look for missed edits or false flags.
5. Visit **Model & Evaluation** and **Error Analysis** to inspect precision, recall, F1, the test confusion matrix, and actual FP/FN samples.
6. Try **Open image**, **Save evidence**, and **Export CSV**. Evidence JSON includes OCR status and the combined decision when OCR has been run.

### Development and reproducibility

```powershell
py -3.14 -m unittest discover -s tests -v
py -3.14 -m fraudvision.cli inspect data/images/doc_0006.jpg
py -3.14 -m fraudvision.cli verify data/images/doc_0006.jpg
py -3.14 -m fraudvision.cli train
py -3.14 -m fraudvision.cli audit-test
```

`audit-test` requires local Tesseract and may take a few minutes to read all 64 held-out images. It reports scores separately without training on test labels. To regenerate the dataset (overwrites JPGs/ground truth; retraining required):

```powershell
py -3.14 -m fraudvision.generator --count 320
py -3.14 -m fraudvision.cli train
py -3.14 -m fraudvision.cli audit-test
```

Retraining should also regenerate `data/demo_predictions.csv` before sharing new cached demo results (the desktop **Train model** button updates this cache automatically).

## Project structure

```text
fraudvision/                  Offline generator, image features, logistic model,
                              optional OCR checks, GUI, audit and CLI
screenshots/                  Actual captured Windows-style application screens
data/images/                 320 bundled synthetic JPGs
data/manifest.csv            Ground-truth edit labels and bounding boxes (evaluation only)
data/demo_predictions.csv    Cached visual inference outputs
models/demo_model.json      Bundled visual screening model
reports/                     Visual and OCR-assisted evaluations + error analysis
tests/                       unittest suite
.github/workflows/ci.yml    GitHub Actions automated tests
RUN_WINDOWS.bat             Windows desktop launcher
```

## Responsible use / technical boundaries

- Not a forensic authentication system; a review flag does not prove fraud.
- No real-world benchmark, adversarial test, or calibration of model scores.
- The OCR check only tests one arithmetic relation and cannot verify dates, references, identities, business provenance, or source-of-truth transactions.
- OCR errors can produce an unreadable result or a false mismatch; a human should inspect the actual source image.
- The images, company names and invoices are fictional; no third-party personal documents are included.
- Tesseract and Pillow have their own licenses. Original demo source is distributed under the included MIT license.

See [`docs/DATA_CARD.md`](docs/DATA_CARD.md), [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) and [`docs/RECRUITER_WALKTHROUGH.md`](docs/RECRUITER_WALKTHROUGH.md) for details.

**Portfolio:** [dharati-cv-ml](https://github.com/dharati-cv-ml)
