"""
verifier.py
-----------
Transcription accuracy verification using a vision LLM.

Checks FusionOCR's transcription against the original image —
catches cases where the OCR was confidently wrong (e.g., "cat" read
as "car" with high confidence because both are plausible words).

Critical for exam marking: a wrongly-transcribed answer that gets
marked as incorrect is a serious error.

Modes:
  - FULL:    check every line (slow but thorough)
  - FLAGGED: only check lines FusionOCR flagged as low-confidence (fast, recommended)
  - SAMPLE:  randomly check N lines + all flagged lines

Requires vision LLM:
  Ollama + llava:7b  →  ollama pull llava:7b  (~4.7GB)

If Ollama vision is not available, falls back to a text-only
confidence report (no image comparison, just flags from OCR engine disagreement).

Usage:
  from verifier import verify_transcription
  result = verify_transcription("original_image.jpg", "ocr_result.json")
"""

import json, logging, random
from pathlib import Path
from llm_backend import get_llm, OllamaBackend, LLMBackend

# ── Prompts ───────────────────────────────────────────────────────────────────

def _vision_prompt(line_text: str, line_num: int) -> str:
    return f"""You are checking the accuracy of an OCR transcription of handwritten text.

Look carefully at the handwriting in this image.
The OCR system says line {line_num} reads: "{line_text}"

Does this transcription accurately reflect what is written in the image?

Return a JSON object:
{{
  "accurate": true | false,
  "corrected_text": "<what the line actually says, or same as input if accurate>",
  "issues": "<description of any discrepancies, or 'none'>",
  "confidence": "high" | "medium" | "low"
}}

Return ONLY the JSON, no other text."""


# ── Core verification ─────────────────────────────────────────────────────────

def _verify_line_vision(line: dict, image_path: str, llm: LLMBackend) -> dict:
    """Verify a single line using vision LLM."""
    prompt   = _vision_prompt(line["text"], line["line"])
    response = llm.complete_json(prompt=prompt, image_path=image_path)

    if "raw" in response:
        return {
            "line":           line["line"],
            "original_text":  line["text"],
            "corrected_text": line["text"],
            "accurate":       None,  # could not determine
            "issues":         "Vision LLM returned unparseable response",
            "confidence":     "low",
            "method":         "vision_failed",
        }

    corrected = response.get("corrected_text", line["text"]).strip()
    accurate  = response.get("accurate", True)

    return {
        "line":           line["line"],
        "original_text":  line["text"],
        "corrected_text": corrected,
        "accurate":       accurate,
        "issues":         response.get("issues", "none"),
        "confidence":     response.get("confidence", "medium"),
        "method":         "vision",
        "changed":        corrected.lower() != line["text"].lower(),
    }


def _verify_line_text_only(line: dict) -> dict:
    """
    Fallback when no vision model is available.
    Uses OCR engine agreement as a proxy for accuracy.
    """
    engines = line.get("engines", {})
    texts   = list(engines.values())
    unique  = set(texts)

    if len(unique) == 1:
        accurate  = True
        issues    = "none"
        conf_str  = "high"
    elif len(unique) == 2:
        # 3 engines, 2 unique — majority rules
        from collections import Counter
        majority = Counter(texts).most_common(1)[0][0]
        accurate = (majority == line["text"])
        issues   = f"Engine disagreement — minority: {list(unique - {majority})}"
        conf_str = "medium"
    else:
        # All engines disagree
        accurate = None
        issues   = f"All engines disagree: {texts}"
        conf_str = "low"

    return {
        "line":           line["line"],
        "original_text":  line["text"],
        "corrected_text": line["text"],
        "accurate":       accurate,
        "issues":         issues,
        "confidence":     conf_str,
        "method":         "text_only",
        "changed":        False,
    }


# ── Main verification function ────────────────────────────────────────────────

def verify_transcription(image_path: str, ocr_result,
                          mode: str = "flagged",
                          sample_n: int = 5,
                          backend: str = "auto") -> dict:
    """
    Verify OCR transcription accuracy.

    Args:
        image_path: path to the original handwritten image
        ocr_result: path to FusionOCR JSON, or the dict directly
        mode:       "flagged" (default) | "full" | "sample"
        sample_n:   number of random lines to sample (mode="sample" only)
        backend:    LLM backend — "auto", "ollama", "huggingface"

    Returns:
        Verification result dict with corrections and accuracy report
    """
    if isinstance(ocr_result, (str, Path)):
        with open(ocr_result, encoding="utf-8") as f:
            ocr_result = json.load(f)

    all_lines    = ocr_result.get("lines", [])
    flagged      = {r["line"] for r in ocr_result.get("flagged_lines", [])}

    # Determine which lines to check
    if mode == "full":
        to_check = all_lines
    elif mode == "sample":
        sampled  = random.sample(all_lines, min(sample_n, len(all_lines)))
        flag_set = [l for l in all_lines if l["line"] in flagged]
        seen     = {l["line"] for l in sampled}
        to_check = sampled + [l for l in flag_set if l["line"] not in seen]
    else:  # "flagged" — default
        to_check = [l for l in all_lines if l["line"] in flagged]

    logging.info(f"[Verifier] mode={mode} — checking {len(to_check)}/{len(all_lines)} lines")

    # Try to get a vision-capable backend
    use_vision = False
    llm        = None

    if backend in ("auto", "ollama"):
        try:
            llm        = get_llm("ollama")
            use_vision = True
            logging.info("[Verifier] Using Ollama vision (llava)")
        except Exception:
            logging.warning("[Verifier] Ollama not available — using text-only mode")

    if not use_vision:
        logging.info("[Verifier] Vision unavailable — falling back to engine-agreement check")

    # Verify each line
    verified = []
    corrections = []

    for line in all_lines:
        if line["line"] in {l["line"] for l in to_check}:
            if use_vision:
                result = _verify_line_vision(line, image_path, llm)
            else:
                result = _verify_line_text_only(line)

            verified.append(result)
            if result.get("changed"):
                corrections.append(result)
                logging.info(
                    f"  Line {line['line']:02d} CORRECTED: "
                    f"{line['text']!r} → {result['corrected_text']!r}"
                )
        else:
            # Unchecked lines — pass through
            verified.append({
                "line":           line["line"],
                "original_text":  line["text"],
                "corrected_text": line["text"],
                "accurate":       True,
                "issues":         "not checked",
                "confidence":     "assumed",
                "method":         "unchecked",
                "changed":        False,
            })

    # Build corrected full text
    line_map = {v["line"]: v["corrected_text"] for v in verified}
    corrected_text = "\n".join(
        line_map.get(l["line"], l["text"]) for l in all_lines
    )

    n_changed    = len(corrections)
    n_inaccurate = sum(1 for v in verified if v.get("accurate") is False)

    result = {
        "source":                  ocr_result.get("source", ""),
        "image_path":              image_path,
        "mode":                    mode,
        "method":                  "vision" if use_vision else "text_only",
        "lines_total":             len(all_lines),
        "lines_checked":           len(to_check),
        "corrections_made":        n_changed,
        "lines_inaccurate":        n_inaccurate,
        "corrected_full_text":     corrected_text,
        "original_full_text":      ocr_result.get("full_text", ""),
        "verified_lines":          verified,
        "corrections":             corrections,
    }

    # Summary
    print("\n" + "=" * 60)
    print(f"VERIFICATION RESULT  ({mode} mode, {result['method']})")
    print("=" * 60)
    print(f"  Lines checked  : {len(to_check)}/{len(all_lines)}")
    print(f"  Corrections    : {n_changed}")
    if corrections:
        print("\n  Changes made:")
        for c in corrections:
            print(f"    Line {c['line']:02d}: {c['original_text']!r}")
            print(f"          → {c['corrected_text']!r}")
    else:
        print("  No corrections needed.")
    if not use_vision:
        print("\n  ⚠ Text-only mode — install Ollama + llava:7b for full image verification")
    print("=" * 60 + "\n")

    return result
