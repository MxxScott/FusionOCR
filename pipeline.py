"""
pipeline.py
-----------
End-to-end script marking pipeline.

  Image(s) → OCR → Verify → Mark → Report

Usage:
  python pipeline.py script.jpg mark_scheme.json
  python pipeline.py script.jpg mark_scheme.json --verify full
  python pipeline.py script.jpg mark_scheme.json --backend huggingface
"""

import json, argparse, logging
from pathlib import Path
from datetime import datetime

from main      import run_ocr
from verifier  import verify_transcription
from marker    import mark_script

LOG_DIR = Path(__file__).parent / "ocr_logs"
LOG_DIR.mkdir(exist_ok=True)


def run_pipeline(image_path: str,
                 mark_scheme_path: str,
                 verify_mode: str = "flagged",
                 backend: str = "auto",
                 skip_verify: bool = False) -> dict:
    """
    Full pipeline: OCR → verify → mark.

    Args:
        image_path:        path to handwritten script image
        mark_scheme_path:  path to mark scheme JSON
        verify_mode:       "flagged" | "full" | "sample" | "skip"
        backend:           LLM backend — "auto" | "ollama" | "huggingface"
        skip_verify:       skip verification stage entirely

    Returns:
        Final result dict combining OCR, verification, and marking
    """
    print("\n" + "=" * 60)
    print("FUSIONOCR MARKING PIPELINE")
    print("=" * 60)

    # Stage 1: OCR
    print("\n[Stage 1/3] Transcribing handwriting...")
    ocr_result = run_ocr(image_path)

    # Stage 2: Verify transcription
    if not skip_verify and verify_mode != "skip":
        print(f"\n[Stage 2/3] Verifying transcription ({verify_mode} mode)...")
        verify_result = verify_transcription(
            image_path, ocr_result,
            mode=verify_mode,
            backend=backend,
        )
        # Use corrected text for marking
        ocr_result["full_text"] = verify_result["corrected_full_text"]
        ocr_result["verified"]  = True
    else:
        print("\n[Stage 2/3] Verification skipped.")
        verify_result = None
        ocr_result["verified"] = False

    # Stage 3: Mark
    print("\n[Stage 3/3] Marking answers...")
    mark_result = mark_script(ocr_result, mark_scheme_path, backend=backend)

    # Combine into final result
    final = {
        "timestamp":     datetime.now().isoformat(),
        "image":         image_path,
        "mark_scheme":   mark_scheme_path,
        "ocr":           ocr_result,
        "verification":  verify_result,
        "marking":       mark_result,
        "summary": {
            "total_awarded":   mark_result["total_awarded"],
            "total_available": mark_result["total_available"],
            "percentage":      mark_result["percentage"],
            "grade":           mark_result["grade"],
            "ocr_lines":       len(ocr_result.get("lines", [])),
            "flagged_lines":   len(ocr_result.get("flagged_lines", [])),
            "corrections":     verify_result["corrections_made"] if verify_result else 0,
        }
    }

    # Save final result
    out_path = LOG_DIR / datetime.now().strftime("pipeline_result_%Y-%m-%d_%H-%M-%S.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    s = final["summary"]
    print(f"  Score   : {s['total_awarded']}/{s['total_available']}  ({s['percentage']}%)")
    print(f"  Grade   : {s['grade']}")
    print(f"  OCR     : {s['ocr_lines']} lines transcribed, {s['flagged_lines']} flagged")
    if verify_result:
        print(f"  Verify  : {s['corrections']} correction(s) made")
    print(f"\nFull result → {out_path}")
    print("=" * 60 + "\n")

    return final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FusionOCR Script Marking Pipeline")
    parser.add_argument("image",       help="Path to handwritten script image")
    parser.add_argument("mark_scheme", help="Path to mark scheme JSON")
    parser.add_argument("--verify",    default="flagged",
                        choices=["flagged", "full", "sample", "skip"],
                        help="Verification mode (default: flagged)")
    parser.add_argument("--backend",   default="auto",
                        choices=["auto", "ollama", "huggingface"],
                        help="LLM backend (default: auto-detect)")
    args = parser.parse_args()

    run_pipeline(
        image_path=args.image,
        mark_scheme_path=args.mark_scheme,
        verify_mode=args.verify,
        backend=args.backend,
    )
