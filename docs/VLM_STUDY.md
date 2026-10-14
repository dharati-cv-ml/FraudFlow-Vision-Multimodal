# Multimodal VQA Study — OCR vs Pretrained SmolVLM2

## Research question

On unseen **synthetic receipt and invoice images**, how does a Tesseract OCR + deterministic parsing baseline compare with a pretrained vision-language model for reading and reasoning about documents?

This is an **optional, additional** experiment: the original visual-patch alteration classifier is separate and its evaluation metrics must not be confused with VQA answer accuracy.

## Dataset and ground truth

- **320 fictional JPG documents**: 192 train, 64 validation, 64 held-out test; equal clean/edited proportions per split.
- Five fixed questions per document: `document_type`, `date`, `total`, `payment_reference`, and `arithmetic_consistency`.
- For a complete 64-image test, there are **320 question-answer pairs**.
- Expected answers are reconstructed from the generator and used **only for evaluation**; neither OCR nor VLM receives labels, answer keys, edit bounding boxes, or the manifest as input.
- The baseline and VLM receive the **same JPG images**. The baseline uses Tesseract (PSM 6) and field rules; the VLM receives the image plus a fixed five-field JSON instruction.

## Pretrained model and inference

VLM: [HuggingFaceTB/SmolVLM2-500M-Video-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct), Apache 2.0. Its model weights are downloaded into the local Hugging Face cache on the first run and **are not bundled in this repository**. No model fine-tuning or LoRA training has been performed.

Run from the repository root on Windows using a Python 3.14 virtual environment:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Tesseract OCR must also be installed locally.
.\.venv\Scripts\python.exe -m multimodal.evaluate --method ocr --split test

# CPU example; use corresponding PyTorch GPU wheels if compatible GPU available.
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-vlm.txt
.\.venv\Scripts\python.exe -m multimodal.evaluate --method vlm --split test --limit 2
.\.venv\Scripts\python.exe -m multimodal.evaluate --method vlm --split test --limit 10
```

These commands can download approximately 2 GB of model weights and may be slow on CPU. They do not run a web server. Platform support should be checked for your Python/PyTorch build.

## Measured results

| Method | Split / scope | Documents | Questions | Correct | Exact match |
|---|---|---:|---:|---:|---:|
| Tesseract OCR + field rules | Synthetic held-out test | 64 | 320 | 320 | **100%** |
| Tesseract OCR + field rules | 10-image pilot subset | 10 | 50 | 50 | **100%** |
| Pretrained SmolVLM2-500M | Same 10-image pilot subset | 10 | 50 | 40 | **80%** |
| Pretrained SmolVLM2-500M | Full held-out test | 64 | 320 | **Not run** | **Not measured** |
| SmolVLM2 + LoRA | — | — | — | **Not run** | **Not measured** |

The **10-image pilot** values were observed in actual Windows evaluation runs and reported by the project operator. The VLM pilot output is generated locally at `reports/multimodal/vlm_test_10.json`; it is not included in the distributed project archive. Preserve and include that measured JSON when presenting a reproducible 10-image experiment; do not fabricate per-field predictions.

The 20-percentage-point pilot difference is exploratory. The test is small, synthetic, legible and templated; the OCR baseline uses template-aware field parsing. **Neither result measures detection of genuine financial fraud or generalization to real-world documents.**

## Fair full-set comparison

For a fair 64-image comparison, run both methods on the entire same held-out set, then analyze per-field accuracy, answer coverage, latency, and errors:

```powershell
.\.venv\Scripts\python.exe -m multimodal.evaluate --method vlm --split test
.\.venv\Scripts\python.exe -m multimodal.compare --ocr reports/multimodal/ocr_test_64.json --vlm reports/multimodal/vlm_test_64.json
```

The interactive **Multimodal Comparison** tab can also compare the two measured JSON files on their **shared image IDs only**; it never infers missing VLM results. See [Visual Comparison](VISUAL_COMPARISON.md) and [LoRA proposal](LORA_PROTOCOL.md).

## Limitations and next steps

- The fixed templates create an unrealistically easy OCR task; add appropriately licensed diverse document images with independently checked labels.
- The VLM has not been fine-tuned. A LoRA/QLoRA training run would require compatible accelerator resources, isolated training/validation data, and a fresh held-out comparison.
- Dates and payment references can be altered without changing arithmetic consistency; VQA accuracy and tampering-screening accuracy are different metrics.
- Do not optimize prompts or hyperparameters against the held-out test labels; retain and disclose model and configuration versions.
