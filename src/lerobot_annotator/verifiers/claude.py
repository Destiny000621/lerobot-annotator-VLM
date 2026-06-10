"""Anthropic Claude verifier backend (claude-opus-4-7, claude-opus-4-8).

Same review() interface as the OpenAI/GPT-5 backend. Uses adaptive thinking
and streaming with .get_final_message() for safe high max_tokens.
"""
from __future__ import annotations
import base64
import io
import json
import re
from pathlib import Path

from PIL import Image

from ..config import VERIFIER_OPENAI_MAX_EDGE, get_anthropic_key


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


def _client():
    from anthropic import Anthropic
    key = get_anthropic_key()
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set and .secrets/anthropic_key.txt missing"
        )
    return Anthropic(api_key=key)


def _parse_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def review(system_prompt: str, user_text: str, images: list[tuple[str, Path]],
           model: str = "claude-opus-4-8") -> dict:
    client = _client()
    content: list[dict] = [{"type": "text", "text": user_text}]
    for label, path in images:
        content.append({"type": "text", "text": label})
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": _b64_jpeg(path),
            },
        })

    with client.messages.stream(
        model=model,
        max_tokens=32000,
        thinking={"type": "adaptive"},
        output_config={"effort": "high"},
        system=system_prompt,
        messages=[{"role": "user", "content": content}],
    ) as stream:
        msg = stream.get_final_message()

    text = ""
    for block in msg.content:
        if getattr(block, "type", None) == "text":
            text = block.text
            break

    try:
        parsed = _parse_json_object(text or "{}")
    except json.JSONDecodeError:
        parsed = {"overall": "issues_found", "issues": [],
                  "summary": f"verifier returned non-JSON: {text[:200]}"}

    return {
        "model": model,
        "parsed": parsed,
        "usage": {
            "input_tokens": msg.usage.input_tokens,
            "output_tokens": msg.usage.output_tokens,
            "cache_creation_input_tokens": getattr(msg.usage, "cache_creation_input_tokens", 0) or 0,
            "cache_read_input_tokens": getattr(msg.usage, "cache_read_input_tokens", 0) or 0,
        },
        "_raw_text": text,
    }


def _register():
    from . import register
    for m in ["claude-opus-4-7", "claude-opus-4-8"]:
        register(m, lambda system, user, images, m=m: review(system, user, images, model=m))


_register()
