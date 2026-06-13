"""
marker.py
---------
AI marking engine for handwritten exam scripts.

Takes:
  - FusionOCR result JSON  (from main.py)
  - Mark scheme JSON       (see mark_scheme_example.json)

Returns:
  - Marked result JSON with scores, justifications, and feedback per question

The LLM is instructed to:
  - Award marks for correct concepts, not exact wording
  - Ignore spelling errors unless the mark scheme explicitly tests spelling
  - Give benefit of the doubt when OCR confidence is low on a line
  - Return structured JSON

Usage:
  from marker import mark_script
  result = mark_script("ocr_result.json", "mark_scheme.json")

  # or from dicts
  result = mark_script(ocr_result, mark_scheme)
"""

import json, logging
from pathlib import Path
from llm_backend import get_llm, LLMBackend

# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an experienced primary/secondary school examiner marking a student's handwritten answer script.

IMPORTANT RULES:
- Award marks for correct CONCEPTS and UNDERSTANDING, not exact wording
- DO NOT penalise spelling errors unless the question explicitly tests spelling
- If a word looks like it might be a transcription error (OCR), give the student the benefit of the doubt
- Be consistent — apply the mark scheme strictly but fairly
- Children's answers may be brief or use simple language; this is expected
- Return ONLY valid JSON, no other text
"""

def _build_marking_prompt(question: dict, student_answer: str,
                           has_flagged_lines: bool) -> str:
    mark_points = "\n".join(f"  - {p}" for p in question.get("mark_points", []))
    guidance    = question.get("guidance", "No additional guidance.")
    marks       = question.get("marks", 1)
    mpp         = question.get("marks_per_point", 1)

    ocr_note = ""
    if has_flagged_lines:
        ocr_note = (
            "\nNOTE: Some lines in this answer were flagged as low-confidence by the OCR system. "
            "Give the student benefit of the doubt where the intended word is plausible.\n"
        )

    return f"""{SYSTEM_PROMPT}
{ocr_note}
QUESTION: {question.get('question', '')}
MARKS AVAILABLE: {marks}
MARKS PER POINT: {mpp}

MARK SCHEME POINTS (award {mpp} mark per point, up to {marks} marks total):
{mark_points}

EXAMINER GUIDANCE: {guidance}

STUDENT'S ANSWER:
{student_answer}

Return a JSON object with this exact structure:
{{
  "marks_awarded": <integer, 0 to {marks}>,
  "marks_available": {marks},
  "points_matched": [<list of mark scheme points the student addressed>],
  "justification": "<one paragraph explaining the marks awarded>",
  "feedback": "<brief, encouraging feedback for the student>",
  "confidence": "high" | "medium" | "low"
}}"""


def _build_question_splitter_prompt(full_text: str, 
                                     question_numbers: list) -> str:
    qs = ", ".join(str(q) for q in question_numbers)
    return f"""{SYSTEM_PROMPT}

The following is a student's exam script transcribed by OCR. 
Split it into individual question answers.
Questions expected: {qs}

Students often label answers as "Q1", "1.", "Question 1", "1)", etc.
If a question answer is missing, return an empty string for it.

SCRIPT:
{full_text}

Return a JSON object mapping question numbers to student answers:
{{
  "1": "<student's answer to question 1>",
  "2": "<student's answer to question 2>",
  ...
}}"""

# ── Answer extraction ─────────────────────────────────────────────────────────

def extract_answers(full_text: str, mark_scheme: dict,
                    llm: LLMBackend) -> dict:
    """
    Split the full OCR text into per-question answers using the LLM.
    Falls back to simple heuristic splitting if LLM fails.
    """
    q_numbers = [str(q["number"]) for q in mark_scheme.get("questions", [])]

    prompt   = _build_question_splitter_prompt(full_text, q_numbers)
    response = llm.complete_json(prompt)

    if "raw" in response:
        logging.warning("[Marker] LLM failed to split answers — using heuristic")
        return _heuristic_split(full_text, q_numbers)

    return response


def _heuristic_split(text: str, question_numbers: list) -> dict:
    """
    Simple fallback: look for common question label patterns.
    e.g. "1.", "Q1", "Question 1", "1)"
    """
    import re
    answers = {n: "" for n in question_numbers}
    lines   = text.split("\n")

    current_q = None
    buffer    = []

    for line in lines:
        matched = None
        for n in question_numbers:
            patterns = [
                rf"^[Qq]\.?\s*{n}\b",
                rf"^{n}[\.\)]\s",
                rf"^[Qq]uestion\s+{n}\b",
            ]
            if any(re.match(p, line.strip()) for p in patterns):
                matched = n
                break

        if matched:
            if current_q and buffer:
                answers[current_q] = " ".join(buffer).strip()
            current_q = matched
            buffer    = [re.sub(r'^.*?[\.\)]\s*', '', line, count=1)]
        elif current_q:
            buffer.append(line)

    if current_q and buffer:
        answers[current_q] = " ".join(buffer).strip()

    return answers

# ── Main marking function ─────────────────────────────────────────────────────

def mark_script(ocr_result, mark_scheme, backend: str = "auto") -> dict:
    """
    Mark a student's script.

    Args:
        ocr_result:  path to FusionOCR JSON file, or the dict directly
        mark_scheme: path to mark scheme JSON file, or the dict directly
        backend:     LLM backend — "auto", "ollama", or "huggingface"

    Returns:
        Marked result dict
    """
    # Load inputs
    if isinstance(ocr_result, (str, Path)):
        with open(ocr_result, encoding="utf-8") as f:
            ocr_result = json.load(f)

    if isinstance(mark_scheme, (str, Path)):
        with open(mark_scheme, encoding="utf-8") as f:
            mark_scheme = json.load(f)

    llm        = get_llm(backend)
    full_text  = ocr_result.get("full_text", "")
    flagged_ln = {r["line"] for r in ocr_result.get("flagged_lines", [])}

    logging.info(f"[Marker] Marking script — {len(mark_scheme.get('questions', []))} questions")

    # Step 1: Split OCR text into per-question answers
    answers = extract_answers(full_text, mark_scheme, llm)
    logging.info(f"[Marker] Answers extracted for questions: {list(answers.keys())}")

    # Step 2: Mark each question
    question_results = []
    total_awarded    = 0
    total_available  = 0

    for q in mark_scheme.get("questions", []):
        q_num   = str(q["number"])
        ans     = answers.get(q_num, "").strip()

        # Check if any lines in this answer were flagged by OCR
        has_flags = bool(flagged_ln)  # conservative: flag if any page lines were uncertain

        if not ans:
            logging.warning(f"[Marker] No answer found for Q{q_num}")
            q_result = {
                "number":          q["number"],
                "question":        q.get("question", ""),
                "student_answer":  "",
                "marks_awarded":   0,
                "marks_available": q.get("marks", 1),
                "points_matched":  [],
                "justification":   "No answer found for this question.",
                "feedback":        "No answer was detected for this question.",
                "confidence":      "low",
                "ocr_flagged":     False,
            }
        else:
            prompt   = _build_marking_prompt(q, ans, has_flags)
            response = llm.complete_json(prompt)

            if "raw" in response:
                logging.error(f"[Marker] LLM returned invalid JSON for Q{q_num}")
                response = {
                    "marks_awarded":   0,
                    "marks_available": q.get("marks", 1),
                    "points_matched":  [],
                    "justification":   "Marking failed — LLM returned unparseable response.",
                    "feedback":        "",
                    "confidence":      "low",
                }

            q_result = {
                "number":          q["number"],
                "question":        q.get("question", ""),
                "student_answer":  ans,
                "ocr_flagged":     has_flags,
                **response,
            }

            # Clamp marks to valid range
            q_result["marks_awarded"] = max(
                0, min(q_result.get("marks_awarded", 0), q.get("marks", 1))
            )

        logging.info(
            f"  Q{q_num}: {q_result['marks_awarded']}/{q.get('marks', 1)} — "
            f"{q_result.get('confidence', '?')} confidence"
        )

        question_results.append(q_result)
        total_awarded   += q_result["marks_awarded"]
        total_available += q.get("marks", 1)

    percentage = round((total_awarded / total_available * 100), 1) if total_available else 0

    result = {
        "source":          ocr_result.get("source", ""),
        "subject":         mark_scheme.get("subject", ""),
        "total_awarded":   total_awarded,
        "total_available": total_available,
        "percentage":      percentage,
        "grade":           _grade(percentage, mark_scheme.get("grade_boundaries", {})),
        "questions":       question_results,
        "ocr_elapsed_sec": ocr_result.get("elapsed_sec", None),
    }

    # Print summary
    print("\n" + "=" * 60)
    print(f"MARKING RESULT — {mark_scheme.get('subject', 'Unknown Subject')}")
    print("=" * 60)
    for qr in question_results:
        flag = " ⚠ (OCR uncertain)" if qr.get("ocr_flagged") else ""
        print(f"  Q{qr['number']}: {qr['marks_awarded']}/{qr['marks_available']}{flag}")
        print(f"       {qr.get('justification', '')[:100]}...")
    print(f"\nTOTAL: {total_awarded}/{total_available}  ({percentage}%)  {result['grade']}")
    print("=" * 60 + "\n")

    return result


def _grade(pct: float, boundaries: dict) -> str:
    """Map percentage to grade using mark scheme boundaries, or default UK scale."""
    if boundaries:
        for grade in sorted(boundaries, key=lambda g: boundaries[g], reverse=True):
            if pct >= boundaries[grade]:
                return grade
        return list(boundaries.keys())[-1]

    # Default UK primary/secondary scale
    if pct >= 85: return "A*"
    if pct >= 70: return "A"
    if pct >= 55: return "B"
    if pct >= 40: return "C"
    if pct >= 25: return "D"
    return "U"
