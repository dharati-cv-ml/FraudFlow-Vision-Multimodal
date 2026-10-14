# Multimodal Comparison — Windows Desktop Viewer

This is a **read-only visualizer** of already-saved OCR and VLM predictions. It does not run inference, download weights, or provide labels to either prediction method.

Start the existing desktop application from the repository root:

```powershell
.\.venv\Scripts\python.exe -m fraudvision.desktop
```

Open **Multimodal Comparison** and click **Refresh reports**. It discovers local `reports/multimodal/ocr_test_10.json` and `vlm_test_10.json`, when present. Use **Load OCR JSON** and **Load VLM JSON** to choose another saved report.

The screen provides the original image, ground-truth answer, OCR answer, VLM answer, correctness highlights, per-question summaries, filters for disagreements and errors, raw model responses, and CSV export.

## Measured full 64-image result

The screenshot in `screenshots/multimodal-comparison.png` was captured from the real Windows app after both 64-image JSON reports had been loaded:

| View item | Measured value |
|---|---:|
| Matched documents | **64** |
| Questions per document | 5 |
| Tesseract OCR exact match | **320/320 (100%)** |
| SmolVLM2 exact match | **240/320 (75%)** |
| OCR advantage on this synthetic test | **25.0 percentage points** |

The **10-image pilot** was earlier measured at 50/50 (100%) OCR and 40/50 (80%) VLM. It is a separate exploratory subset and not the same as the complete benchmark.

The viewer matches **identical held-out image IDs**. A missing VLM report appears as **NOT RUN**, not zero accuracy. The final README screenshot includes the actual full-set comparison, not the previous OCR-only placeholder.

To verify the captured full result using the real generated JSON reports, keep `reports/multimodal/ocr_test_64.json` and `reports/multimodal/vlm_test_64.json` in the project and run:

```powershell
.\.venv\Scripts\python.exe -m multimodal.compare --ocr reports/multimodal/ocr_test_64.json --vlm reports/multimodal/vlm_test_64.json
```

Do not fabricate a VLM JSON if the local file is missing. This project evaluates fixed question answering on **synthetic documents**, not real-world financial fraud.

## Windows GUI responsiveness

The `fraudvision/multimodal_view.py` implementation performs report preparation on a background worker so opening this tab does not block the Tkinter event loop. If you have an older checkout that freezes when switching tabs, apply the updated source file before publishing.

**Scope:** These are synthetic document field-reading tasks, not proof of document authenticity or fraud-detection performance.
