# Optional LoRA Fine-Tuning Study — Not Yet Executed

**Status: planned, NOT trained or evaluated.** This study is conditional on the availability of enough VRAM, supported PyTorch/Transformers/PEFT wheels, and a successful pretrained VLM baseline.

1. Establish pretrained VLM inference first. Record GPU model, VRAM, PyTorch/Transformers versions, runtime, and base-model checkpoint/hash.
2. Keep existing 192/64/64 splits fixed. Export train and validation input/output pairs only:

```powershell
python -m multimodal.lora_dataset --split train --out experiments/lora_train.jsonl
python -m multimodal.lora_dataset --split validation --out experiments/lora_validation.jsonl
```

3. Use supervised image-to-text generation with frozen vision encoder at first; apply PEFT LoRA to attention projections (e.g., `q_proj`, `v_proj`, after inspecting module names). Preserve image tokens and mask prompt labels so loss is computed on answer tokens only. Avoid finetuning the processor/image encoder unless necessary.
4. Start on train only; choose checkpoint/learning rate using validation. Suggested initial experiment: rank 8, alpha 16, dropout 0.05, batch size 1, gradient accumulation 8, 2-3 epochs, learning rate ~1e-4; these are **starting hyperparameters, not validated values**.
5. Freeze a checkpoint and compare **base model vs LoRA model** using the exact same 64-image test and question prompts. Report accuracy by field, runtime, VRAM usage, training curves, changed hallucinations, and negative results.
6. Do not rely exclusively on memorized templates: add new receipt layouts and licensed external documents for realistic generalization testing. Provide reproducible notebook/training environment lockfile and GPU memory requirements after experiments actually run.

A training script is intentionally **not included as a purported working component** because the user's NVIDIA GPU, VRAM and compatible Transformers/PEFT environment have not yet been confirmed. This avoids presenting an unexecuted fine-tune as achieved work.
