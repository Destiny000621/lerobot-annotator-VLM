"""OpenAI Chat Completions backend for the verifier (default model: gpt-5.5)."""
from __future__ import annotations
import base64
import io
import json
from pathlib import Path

from PIL import Image

from ..config import VERIFIER_OPENAI_MAX_EDGE, get_openai_key


def _b64_jpeg(path: str | Path, max_edge: int | None = VERIFIER_OPENAI_MAX_EDGE) -> str:
    with Image.open(path) as img:
        img = img.convert("RGB")
        if max_edge is not None:
            w, h = img.size
            scale = max_edge / max(w, h)
            if scale < 1.0:
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=88)
        return base64.b64encode(buf.getvalue()).decode("ascii")


def review(system_prompt: str, user_text: str, images: list[tuple[str, Path]],
           model: str = "gpt-5.5") -> dict:
    """Send (system, user-text, [(label, image_path), ...]) to OpenAI. Return parsed JSON.

    Returns a dict with at minimum {"raw": <api response usage>, "parsed": <json from model>}.
    """
    from openai import OpenAI
    key = get_openai_key()
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set and .secrets/openai_key.txt missing")
    client = OpenAI(api_key=key)

    content: list[dict] = [{"type": "text", "text": user_text}]
    for label, path in images:
        content.append({"type": "text", "text": label})
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{_b64_jpeg(path)}"}})

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": content},
        ],
        response_format={"type": "json_object"},
    )
    text = resp.choices[0].message.content or "{}"
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = {"overall": "issues_found", "issues": [],
                  "summary": f"verifier returned non-JSON: {text[:200]}"}
    return {
        "model": model,
        "parsed": parsed,
        "usage": resp.usage.model_dump() if resp.usage else {},
        "_raw_text": text,
    }


def _register():
    from . import register
    for m in ["gpt-5.5", "gpt-5.5-pro", "gpt-5.4", "gpt-5.4-pro",
              "gpt-5.3-chat-latest", "gpt-5.2", "gpt-5.1", "gpt-5", "gpt-4o", "gpt-4.1"]:
        register(m, lambda system, user, images, m=m: review(system, user, images, model=m))


_register()
