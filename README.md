# FusionOCR

A multi-engine OCR pipeline that combines **Tesseract**, **EasyOCR**, and **Calamari** via word-level Levenshtein consensus voting, then cleans the output with **Google FLAN-T5**.

## How it works

```
Image
  ├── pytesseract  ─┐
  ├── EasyOCR      ─┼──► Levenshtein consensus ──► FLAN-T5 cleanup ──► Final text
  └── Calamari     ─┘
```

Each engine extracts text independently. For each word position, the engine outputs are compared pairwise using Levenshtein distance — the word closest to all others wins. The consensus text is then passed through FLAN-T5 for grammar and coherence correction.

## Requirements

```bash
pip install -r requirements.txt
```

You will also need:
- **Tesseract** installed on your system: https://github.com/tesseract-ocr/tesseract
- A **CUDA-capable GPU** is strongly recommended (EasyOCR and FLAN-T5 run on CPU but will be slow)

## Usage

```python
from main import run_ocr

result = run_ocr("path/to/image.png")
print(result)
```

Or run directly:

```bash
python main.py
```

Place a test image at `test_image.png` in the same directory, or modify `__main__` to point to your image.

## Logging

Every run generates a timestamped log file in `ocr_logs/` with full debug output from all three engines and the final consensus result.

## Stack

- `pytesseract` — Tesseract wrapper
- `easyocr` — deep learning OCR (CRAFT text detection + CRNN recognition)
- `calamari-ocr` — sequence-to-sequence OCR trained on historical documents
- `python-Levenshtein` — fast edit distance for consensus voting
- `transformers` (HuggingFace) — FLAN-T5 for post-processing
- `Pillow` — image loading

## Notes

- Model weights (`*.pth`, `uw3-modern-english/`) are excluded from this repo. EasyOCR downloads its models automatically on first run. For Calamari, download a pretrained model from the [Calamari model zoo](https://github.com/Calamari-OCR/calamari_models).
- GPU memory: FLAN-T5-base requires ~1GB VRAM. Use `flan-t5-small` if memory-constrained (see commented line in `main.py`).
