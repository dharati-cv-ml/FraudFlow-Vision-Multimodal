# Multimodal VQA Experiment — OCR vs Pretrained VLM

**Research question:** On **unseen synthetic receipt/invoice images**, how does local Tesseract OCR with deterministic field parsing compare with an unmodified pretrained vision-language model when answering a fixed set of document questions?

This is an **additional, optional research experiment** in the existing FraudFlow Vision repository. The existing Windows desktop application remains fully functional without a VLM, Python-server, internet connection, SciPy or CUDA.

## Dataset and protocol

- Synthetic data only: 320 independently generated receipts/invoices. Existing splits: 192 train, 64 validation, **64 held-out test**.
- Exactly 5 questions per image, giving **320 image-question answers on the test split**.
- Questions (fixed before comparison): document type; printed date; printed TOTAL; payment reference; arithmetic consistency of subtotal + tax vs total.
- The generator deterministically recovers the **printed** field truth for scoring. Manifest, edit class, initial values, ground-truth edit box, split and expected answers are **never input** to OCR or VLM inference.
- Both systems receive the same JPGs. The OCR baseline receives no visual-model outputs. VLM receives an image + instruction requesting the same 5 answers as a JSON object. No external lookup or RAG.
- Report exact-match accuracy per question and micro across 5 × 64, missing-answer coverage, inference time, and examples of errors. A blank/unparsable answer is counted wrong.
- Report pretrained VLM scores **only after the complete test run**. The example `--limit 2` is a smoke test, NOT a valid comparison.

## Methods

**OCR baseline:** Installed Tesseract (PSM 6), field-specific regular expressions and the independent Decimal subtotal+tax verifier. Package: `multimodal/ocr_baseline.py`.

**Pretrained VLM candidate:** `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` using Hugging Face Transformers, on-device in the same Python process (no localhost server). The small 500M model is chosen as a resource-aware feasibility baseline, not as an OCR quality leader. A larger model may give better text recognition but demands more GPU memory. This version requires an **initial Hugging Face model download**; inference can run offline after caching.

### Windows commands (Python 3.14)

Existing desktop and OCR baseline:

```powershell
cd D:\FraudFlow-Vision-Multimodal-Lab
py -3.14 -m pip install -r requirements.txt
py -3.14 -m multimodal.evaluate --method ocr --split test
```

Optional VLM: **Only if your Python/PyTorch/Transformers build supports your Windows machine.** This was not executable/verified on the user's Windows laptop at packaging time.

```powershell
py -3.14 -m pip install torch
py -3.14 -m pip install -r requirements-vlm.txt
py -3.14 -m multimodal.evaluate --method vlm --split test --limit 2
py -3.14 -m multimodal.evaluate --method vlm --split test
```

The first VLM command downloads weights from Hugging Face (internet required). CPU inference can be very slow. Check NVIDIA GPU with `nvidia-smi`. If no appropriate PyTorch wheel is available for Python 3.14, use a **separate** compatible Python environment solely for the optional VLM; do not replace or break the offline desktop installation.

After both full 64-image runs:

```powershell
py -3.14 -m multimodal.compare --ocr reports/multimodal/ocr_test_64.json --vlm reports/multimodal/vlm_test_64.json
```

## Results

The measured OCR baseline run is saved at `reports/multimodal/ocr_test_64.json` (and errors CSV). It obtained 320/320 exact matches on the simple bundled **synthetic** labels in this environment. This **does not** mean OCR is 100% on real-world invoices or that the benchmark is difficult; it reflects narrow generated layouts and excellent legibility. Model comparisons must use the same held-out test images.

| System | Test documents | Question answers | Exact-match score | Status |
|---|---:|---:|---:|---|
| Tesseract + regex + arithmetic | 64 | 320 | 100% | Measured on bundled synthetic test |
| SmolVLM2-500M (pretrained) | — | — | **Not measured** | Requires download and runnable VLM environment |
| SmolVLM2 + LoRA | — | — | **Not measured** | Future controlled training; depends on GPU |

### Key limitations

This VQA task tests readable printed fields and arithmetic, **not** manipulation history or actual fraud; a fully altered but internally consistent receipt may be indistinguishable without external provenance. Synthetic receipt formatting can make OCR+rules unrealistically strong; adding external licensed real-image evaluation and new template families is necessary for meaningful generalization. Test images should not influence prompts or threshold/model selection after results are seen.

## Future improvements

- External document test set with licenses and true manually checked field annotations.
- Add unknown vendors, alternate layouts, handwritten totals, camera noise, smartphone photos.
- Field-level numeric tolerance analysis and per-category errors.
- Compare a larger document-capable VLM under matched image resolutions, latency/hardware reporting.
- Conduct LoRA experiment using training + validation only and compare with identical evaluation; see `docs/LORA_PROTOCOL.md`.
