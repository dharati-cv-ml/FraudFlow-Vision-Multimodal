# Multimodal Comparison — Windows Desktop Viewer

This is a **read-only visualizer** of already-saved OCR and VLM predictions. It does not run inference, download weights, or provide labels to either prediction method.

Start the existing desktop application from the repository root:

```powershell
.\.venv\Scripts\python.exe -m fraudvision.desktop
```

Open **Multimodal Comparison** and click **Refresh reports**. It discovers local `reports/multimodal/ocr_test_10.json` and `vlm_test_10.json`, when present. Use **Load OCR JSON** and **Load VLM JSON** to choose another saved report.

The screen provides the original image, ground-truth answer, OCR answer, VLM answer, correctness highlights, per-question summaries, filters for disagreements and errors, raw model responses, and CSV export.

## What to expect with the measured 10-image pilot

When both the measured local OCR and VLM JSON files are available:

| View item | Expected value |
|---|---:|
| Matched documents | 10 |
| Evaluated answers per model | 50 |
| Tesseract exact match | 50/50 (100%) |
| SmolVLM2 exact match | 40/50 (80%) |

The viewer pairs **identical image IDs**. It never compares a 10-image VLM score against a 64-image OCR score using different denominators. A missing report is **NOT RUN**, not zero accuracy. The bundled comparison screenshot is from an OCR-only environment and intentionally shows no VLM result.

## Windows GUI responsiveness

The `fraudvision/multimodal_view.py` implementation performs report preparation on a background worker so opening this tab does not block the Tkinter event loop. If you have an older checkout that freezes when switching tabs, apply the updated source file before publishing.

**Scope:** These are synthetic document field-reading tasks, not proof of document authenticity or fraud-detection performance.
