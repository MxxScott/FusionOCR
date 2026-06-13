"""
llm_backend.py
--------------
Pluggable LLM interface. Swap backends without changing marker or verifier code.

Supported backends:
  - Ollama   (local, free, no rate limits) — recommended
  - HuggingFace Inference API (free tier, requires free token)

Usage:
  from llm_backend import get_llm
  llm = get_llm()                          # auto-detects available backend
  llm = get_llm(backend="ollama")          # force Ollama
  llm = get_llm(backend="huggingface")     # force HuggingFace

  response = llm.complete(prompt)          # text response
  response = llm.vision(prompt, image_path) # vision response (Ollama only)

Setup:
  Ollama:       install from https://ollama.com, then: ollama pull llama3.2:3b
  HuggingFace:  set env var HF_TOKEN=your_free_token (https://huggingface.co/settings/tokens)
"""

import os, json, logging, base64, requests
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

OLLAMA_BASE    = os.getenv("OLLAMA_BASE", "http://localhost:11434")
OLLAMA_MODEL   = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_VISION  = os.getenv("OLLAMA_VISION_MODEL", "llava:7b")

HF_TOKEN       = os.getenv("HF_TOKEN", "")
HF_MODEL       = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
HF_API_BASE    = "https://api-inference.huggingface.co/models"

# ── Base class ────────────────────────────────────────────────────────────────

class LLMBackend:
    def complete(self, prompt: str) -> str:
        raise NotImplementedError

    def vision(self, prompt: str, image_path: str) -> str:
        raise NotImplementedError("Vision not supported by this backend")

    def complete_json(self, prompt: str) -> dict:
        """Call complete() and parse the JSON response."""
        raw = self.complete(prompt)
        # Strip markdown code fences if present
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        try:
            return json.loads(raw.strip())
        except json.JSONDecodeError:
            logging.warning(f"[LLM] Failed to parse JSON — returning raw text")
            return {"raw": raw}

# ── Ollama backend ────────────────────────────────────────────────────────────

class OllamaBackend(LLMBackend):
    def __init__(self, model: str = OLLAMA_MODEL, vision_model: str = OLLAMA_VISION):
        self.model        = model
        self.vision_model = vision_model
        logging.info(f"[LLM] Ollama backend — model: {model}")

    def _call(self, model: str, prompt: str, images: list = None) -> str:
        payload = {
            "model":  model,
            "prompt": prompt,
            "stream": False,
        }
        if images:
            payload["images"] = images

        try:
            r = requests.post(
                f"{OLLAMA_BASE}/api/generate",
                json=payload,
                timeout=120,
            )
            r.raise_for_status()
            return r.json().get("response", "").strip()
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                "Ollama is not running. Start it with: ollama serve\n"
                "Then pull a model: ollama pull llama3.2:3b"
            )

    def complete(self, prompt: str) -> str:
        return self._call(self.model, prompt)

    def vision(self, prompt: str, image_path: str) -> str:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return self._call(self.vision_model, prompt, images=[b64])

    @staticmethod
    def is_available() -> bool:
        try:
            r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False


# ── HuggingFace Inference API backend ─────────────────────────────────────────

class HuggingFaceBackend(LLMBackend):
    def __init__(self, model: str = HF_MODEL, token: str = HF_TOKEN):
        if not token:
            raise ValueError(
                "HF_TOKEN not set. Get a free token at https://huggingface.co/settings/tokens\n"
                "Then: set HF_TOKEN=your_token   (Windows)\n"
                "  or: export HF_TOKEN=your_token (Linux/Mac)"
            )
        self.model   = model
        self.headers = {"Authorization": f"Bearer {token}"}
        logging.info(f"[LLM] HuggingFace backend — model: {model}")

    def complete(self, prompt: str) -> str:
        # Format as instruction prompt
        formatted = f"[INST] {prompt} [/INST]"
        r = requests.post(
            f"{HF_API_BASE}/{self.model}",
            headers=self.headers,
            json={
                "inputs": formatted,
                "parameters": {
                    "max_new_tokens": 1024,
                    "temperature": 0.1,
                    "return_full_text": False,
                },
            },
            timeout=60,
        )
        if r.status_code == 503:
            raise RuntimeError(f"Model {self.model} is loading on HuggingFace — wait 20s and retry.")
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list):
            return data[0].get("generated_text", "").strip()
        return str(data).strip()

    def vision(self, prompt: str, image_path: str) -> str:
        raise NotImplementedError(
            "HuggingFace Inference API vision requires a separate endpoint. "
            "Use Ollama with llava:7b for vision verification."
        )

    @staticmethod
    def is_available() -> bool:
        return bool(HF_TOKEN)


# ── Auto-detect ───────────────────────────────────────────────────────────────

def get_llm(backend: str = "auto") -> LLMBackend:
    """
    Get an LLM backend.

    Args:
        backend: "auto" | "ollama" | "huggingface"

    Returns:
        LLMBackend instance
    """
    if backend == "ollama" or (backend == "auto" and OllamaBackend.is_available()):
        return OllamaBackend()

    if backend == "huggingface" or (backend == "auto" and HuggingFaceBackend.is_available()):
        return HuggingFaceBackend()

    raise RuntimeError(
        "No LLM backend available.\n\n"
        "Option A — Ollama (local, recommended):\n"
        "  1. Install: https://ollama.com\n"
        "  2. Run: ollama serve\n"
        "  3. Pull: ollama pull llama3.2:3b\n\n"
        "Option B — HuggingFace (free cloud API):\n"
        "  1. Get free token: https://huggingface.co/settings/tokens\n"
        "  2. Set: set HF_TOKEN=your_token"
    )
