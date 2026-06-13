"""
test.py — FusionOCR test runner
================================

Three test modes:

  1. Synthetic — generates a clean printed image to verify the pipeline works end-to-end
  2. Real      — run on any image you provide: python test.py path/to/image.jpg
  3. Download  — fetches a public handwritten sample: python test.py --sample

Usage:
  python test.py                        # synthetic test
  python test.py --sample               # download + test a real handwritten image
  python test.py path/to/script.jpg     # test your own image
"""

import sys, os, json, urllib.request
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Synthetic image ───────────────────────────────────────────────────────────

SYNTHETIC_LINES = [
    "The water cycle is the process",
    "by which water moves through",
    "the environment. It includes",
    "evaporation, condensation,",
    "precipitation and collection.",
]

def make_synthetic_image(path: str):
    """Create a simple lined-paper style test image with printed text."""
    W, H = 800, 500
    LINE_HEIGHT = 60
    MARGIN = 60

    img  = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)

    # Draw ruled lines (like exercise book paper)
    for y in range(MARGIN, H - 20, LINE_HEIGHT):
        draw.line([(MARGIN - 10, y + 40), (W - MARGIN + 10, y + 40)],
                  fill=(180, 200, 220), width=1)

    # Draw red left margin
    draw.line([(MARGIN, 0), (MARGIN, H)], fill=(220, 80, 80), width=2)

    # Try to load a font; fall back to default
    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except Exception:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
        except Exception:
            font = ImageFont.load_default()

    for i, line in enumerate(SYNTHETIC_LINES):
        y = MARGIN + i * LINE_HEIGHT
        draw.text((MARGIN + 15, y), line, fill="black", font=font)

    img.save(path)
    print(f"Synthetic test image saved → {path}")
    return path


# ── Download sample ───────────────────────────────────────────────────────────

SAMPLE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/"
    "Handwriting_of_Mattie_and_Phoebe_Harlan_Macy.jpg/"
    "800px-Handwriting_of_Mattie_and_Phoebe_Harlan_Macy.jpg"
)

def download_sample(path: str):
    print(f"Downloading handwriting sample...")
    urllib.request.urlretrieve(SAMPLE_URL, path)
    print(f"Sample saved → {path}")
    return path


# ── Verify result ─────────────────────────────────────────────────────────────

def print_report(result: dict):
    lines     = result["lines"]
    flagged   = result["flagged_lines"]
    elapsed   = result["elapsed_sec"]
    full_text = result["full_text"]

    print("\n" + "━" * 60)
    print("TEST REPORT")
    print("━" * 60)

    print(f"\nTranscription ({len(lines)} lines):")
    for ln in lines:
        flag  = "⚠ " if ln["flagged"] else "  "
        print(f"  {flag}[{ln['confidence']:.2f}] {ln['text']}")

    if flagged:
        print(f"\n⚠  {len(flagged)} line(s) flagged for review (low confidence).")
        print("   These lines should be checked against the original image before marking.")

    avg_conf = sum(l["confidence"] for l in lines) / len(lines) if lines else 0
    print(f"\nSummary:")
    print(f"  Lines processed : {len(lines)}")
    print(f"  Flagged         : {len(flagged)}")
    print(f"  Avg confidence  : {avg_conf:.2f}")
    print(f"  Elapsed         : {elapsed}s")

    # Engine agreement analysis
    agreements = 0
    for ln in lines:
        e = ln["engines"]
        unique = len(set(e.values()))
        if unique == 1:
            agreements += 1
    print(f"  Full agreement  : {agreements}/{len(lines)} lines (all engines matched)")
    print("━" * 60 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def check_dependencies():
    missing = []
    for pkg in ["transformers", "easyocr", "pytesseract", "cv2", "Levenshtein"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        sys.exit(1)
    print("✓ All dependencies found")


if __name__ == "__main__":
    check_dependencies()

    arg = sys.argv[1] if len(sys.argv) > 1 else None

    if arg == "--sample":
        image_path = os.path.join(SCRIPT_DIR, "sample_handwriting.jpg")
        download_sample(image_path)
    elif arg and os.path.exists(arg):
        image_path = arg
        print(f"Using image: {image_path}")
    elif arg:
        print(f"File not found: {arg}")
        sys.exit(1)
    else:
        image_path = os.path.join(SCRIPT_DIR, "test_synthetic.png")
        make_synthetic_image(image_path)

    print("\nRunning FusionOCR pipeline...")
    from main import run_ocr
    result = run_ocr(image_path)
    print_report(result)
