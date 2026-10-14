# Five-minute review walkthrough — FraudFlow Vision

1. Start the native desktop with `py -3.14 -m fraudvision.desktop`. Python 3.14, Pillow and Tkinter; no localhost.
2. **Inspection Studio:** click `doc_0006.jpg`, then **Verify totals (OCR)**. On the synthetic edited receipt, Tesseract reads subtotal `$271.53`, tax `$19.01`, and printed total `$351.54`. The arithmetic expected total is `$290.54`, giving a discrepancy of `$61.00`. Show distinct **VISUAL SCREEN**, **OCR ARITHMETIC CHECK**, and **COMBINED DECISION** sections. Toggle the pixel-inferred evidence overlay.
3. **Analytics Dashboard:** show the 320-image visual-score distribution and 163 visual review flags. Labels are generated for demonstration; numeric scores are not probabilities of fraud.
4. **Dataset Explorer:** filter and inspect individual samples, comparing synthetic ground truth only after prediction.
5. **Model & Evaluation:** describe 60/20/20 splits, separate validation-selected threshold and held-out synthetic test. Visual F1 75.76%; OCR-assisted combined F1 77.61% on the 64-image test, with no test-set tuning.
6. **Error Analysis:** click a false positive or false negative to view the actual source image. Discuss which cases fail and why.
7. Export the selected evidence JSON + annotated PNG and run `py -3.14 -m unittest discover -s tests -v`.

**Caveat to say clearly:** This is a synthetic academic/portfolio experiment. OCR consistency detects numeric mismatches but does not verify who created the document, whether a reference is legitimate, or whether financial fraud occurred. Next steps include permissioned real-world datasets, document-layout parsing, and a CNN baseline.
