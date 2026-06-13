# FusionOCR

Multi-engine OCR pipeline for **children's handwritten English** — exam scripts, answer sheets, exercise books.

Accuracy is the primary design goal. The pipeline runs four OCR engines and fuses their outputs via weighted Levenshtein consensus, with automatic flagging of low-confidence lines for human review.

## Architecture

```
Image
  └─► Preprocessor
        · Remove ruled lines (horizontal + vertical margin lines)
        · Deskew
        · Denoise (non-local means)
        · Adaptive binarize (CLAHE + Gaussian threshold)
        · Morphological ink-gap fill
            └─► Segmenter (horizontal projection → line crops)
                  ├─► TrOCR-large-handwritten (batch, 8-beam) ─┐
                  ├─► TrOCR-large-printed     (batch, 8-beam) ─┤
                  ├─► EasyOCR                 (parallel)       ┤─► Weighted consensus → JSON
                  └─► Tesseract               (PSM-7)         ─┘
```

**Why two TrOCR models?**
Children's handwriting spans from neat block print to semi-cursive. The handwriting model covers irregular joined letters; the printed model covers block-letter writers. Consensus between both is more robust than either alone.

**Engine weights in consensus:**

| Engine | Weight | Rationale |
|---|---|---|
| TrOCR-large-handwritten | 0.40 | Primary; trained on IAM handwriting dataset |
| TrOCR-large-printed | 0.30 | Covers block/print writers |
| EasyOCR | 0.20 | Deep learning; strong on varied styles |
| Tesseract | 0.10 | Fast; useful for clearly printed words |

## Output

```json
{
  "source": "script_page1.jpg",
  "full_text": "The water cycle begins when...",
  "lines": [
    {
      "line": 1,
      "text": "The water cycle begins when",
      "confidence": 0.91,
      "engines": {
        "trocr_handwritten": "The water cycle begins when",
        "trocr_printed": "The water cycle begins when",
        "easyocr": "The water cycle begins when",
        "tesseract": "The water eycle begins when"
      },
      "flagged": false
    }
  ],
  "flagged_lines": [],
  "elapsed_sec": 6.4
}
```

Lines below confidence `0.65` are included in `flagged_lines` — these should be reviewed against the original image before marking.

## Setup

```bash
pip install -r requirements.txt
```

Also install Tesseract: https://github.com/tesseract-ocr/tesseract

**VRAM note:** Two TrOCR-large models require ~3GB VRAM. On CPU, inference is slower but functional.

## Usage

```bash
python main.py path/to/script_page.jpg
```

```python
from main import run_ocr
result = run_ocr("script_page.jpg")
print(result["full_text"])
for line in result["flagged_lines"]:
    print(f"Review line {line['line']}: {line['text']}")
```

## Configuration (`main.py`)

| Variable | Default | Notes |
|---|---|---|
| `NUM_BEAMS` | `8` | Beam search width — higher = more accurate |
| `TROCR_BATCH` | `4` | Reduce if GPU OOM |
| `CONFIDENCE_THRESHOLD` | `0.65` | Below this → flagged for review |
| `WEIGHTS` | see above | Adjust per writing style of your cohort |

## Stack

`transformers` · `torch` · `easyocr` · `pytesseract` · `opencv-python` · `python-Levenshtein` · `Pillow`
