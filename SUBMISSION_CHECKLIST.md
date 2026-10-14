# FraudFlow Vision — final submission checklist

- [x] Fixed synthetic total/date replacement bug; regenerated 320 images and ground truth.
- [x] Regenerated trained visual model, cached predictions and evaluation metrics.
- [x] Added independent local Tesseract arithmetic verification (subtotal + tax versus total).
- [x] Added on-demand combined decision with separate visual/OCR evidence.
- [x] Added Error Analysis desktop tab and separate visual/combined error CSVs.
- [x] Validated OCR-assisted held-out test using local Tesseract; reported actual results.
- [x] Added real screenshots of corrected application and results.
- [x] Source, docs, unit tests, CI, synthetic JPG dataset and offline launcher packaged.
- [ ] Test opening application and OCR on your own Windows Python 3.14 installation.
- [ ] Run `py -3.14 -m unittest discover -s tests -v` locally and inspect test result.
- [ ] Export a report and verify its visual and OCR sections.
- [ ] Create a NEW GitHub repo and upload the verified project; don't overwrite EdgeGuard or VisionInspect.
- [ ] Confirm GitHub Actions passes and README screenshots display.
- [ ] Cite only synthetic held-out numbers, never real-world fraud detection accuracy.

## Optional multimodal extension checks

- [x] Original GUI and image model retained, no localhost requirement.
- [x] Ground-truth recovery validated against original manifest (reproduced generator seed 31415).
- [x] OCR-only baseline actually run on 64-image synthetic test (320/320 fields on this restricted dataset).
- [ ] Pretrained local SmolVLM2 weights downloaded on target machine and 2-image smoke test completed.
- [ ] Full 64-image VLM benchmark executed; publish measured comparisons only afterward.
- [ ] Check GPU availability and decide whether LoRA is feasible; no fine-tuning has been run.
- [ ] Add out-of-generator real licensed document evaluation for generalization claims.
