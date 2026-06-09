"""Avant LiteLLM gateway (Gemini Robotics-ER) backend, for redundancy or budget verification."""
from __future__ import annotations
import json
from pathlib import Path

from ..config import LITELLM_BASE, get_litellm_key
from ..gemini import b64_thumbnail, post


def review(system_prompt: str, user_text: str, images: list[tuple[str, Path]],
           model: str = "gemini-robotics-er-1.6-preview") -> dict:
    url = f"{LITELLM_BASE}/v1beta/models/{model}:generateContent"
    parts: list[dict] = [{"text": user_text}]
    for label, path in images:
        parts.append({"text": label})
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": b64_thumbnail(path)}})

    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"temperature": 0.0, "responseMimeType": "application/json"},
    }
    # post() uses GEMINI_URL by default; build our own request here for explicit model targeting.
    import urllib.request
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json",
                                          "x-goog-api-key": get_litellm_key()})
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} from gateway: {detail[:500]}") from None

    text = data["candidates"][0]["content"]["parts"][0]["text"]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = {"overall": "issues_found", "issues": [],
                  "summary": f"verifier returned non-JSON: {text[:200]}"}
    return {
        "model": model,
        "parsed": parsed,
        "usage": data.get("usageMetadata", {}),
        "_raw_text": text,
    }


def _register():
    from . import register
    register("gemini-robotics-er-1.6-preview",
             lambda system, user, images: review(system, user, images))


_register()
