"""Anthropic Claude annotator backend (claude-opus-4-7, claude-opus-4-8).

Mirrors the OpenAI Chat backend's behavior: same neutral-parts transformer,
same count-vote loop, same retry/validate loop. Adds Claude-specific:
- Adaptive thinking (`thinking: {type: "adaptive"}`) — recommended for all
  complex tasks on Opus 4.7/4.8.
- Streaming with `.get_final_message()` — required to safely use the high
  `max_tokens` ceiling without hitting HTTP timeouts.
- `output_config.effort: "high"` — sensible default for structured annotation.
- No `temperature` / `top_p` / `top_k` — those parameters were removed on
  Opus 4.7 / 4.8 and return a 400 if sent.
"""
from __future__ import annotations
import base64
import io
import json
import re
from collections import Counter
from pathlib import Path

from PIL import Image

from ..config import VERIFIER_OPENAI_MAX_EDGE, get_anthropic_key
from . import register


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


def _image_block(path: str | Path) -> dict:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": _b64_jpeg(path),
        },
    }


def _neutral_to_content(neutral_parts: list[dict]) -> list[dict]:
    out: list[dict] = []
    for p in neutral_parts:
        if "text" in p:
            out.append({"type": "text", "text": p["text"]})
        elif "image" in p:
            out.append(_image_block(p["image"]))
    return out


def _parse_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def _extract_text(message) -> str:
    """Pull the text out of a Claude response. Thinking blocks have empty
    content on 4.7/4.8 by default; we only want the final-answer text block."""
    for block in message.content:
        if getattr(block, "type", None) == "text":
            return block.text
    return ""


def _usage_dict(message) -> dict:
    u = message.usage
    return {
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
        "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
        "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
    }


class ClaudeAnnotator:
    def __init__(self, model_id: str):
        self.model_id = model_id

    def count_vote(self, image_paths, question: str,
                   range_lo: int, range_hi: int, n_votes: int = 5,
                   prefer_final_frame: bool = False):
        from ..gemini import _normalize_images, _build_count_prompt
        client = _client()
        labeled = _normalize_images(image_paths)
        prompt = _build_count_prompt(question, labeled, range_lo, range_hi, prefer_final_frame)
        content: list[dict] = [{"type": "text", "text": prompt}]
        for label, p in labeled:
            content.append({"type": "text", "text": label})
            content.append(_image_block(p))
        votes: list[int] = []
        for _ in range(n_votes):
            try:
                with client.messages.stream(
                    model=self.model_id,
                    max_tokens=16000,
                    thinking={"type": "adaptive"},
                    messages=[{"role": "user", "content": content}],
                ) as stream:
                    msg = stream.get_final_message()
                text = _extract_text(msg)
                obj = _parse_json_object(text)
                c = int(obj["count"])
                if range_lo <= c <= range_hi:
                    votes.append(c)
            except Exception:
                continue
        if not votes:
            raise RuntimeError(f"Claude count vote failed all {n_votes} attempts")
        counter = Counter(votes)
        top_freq = counter.most_common(1)[0][1]
        winners = [c for c, f in counter.items() if f == top_freq]
        return max(winners), votes

    def generate_with_retry(self, system_text: str, neutral_parts: list[dict],
                            validate_fn, max_attempts: int = 4):
        client = _client()
        content = _neutral_to_content(neutral_parts)
        attempts: list[dict] = []
        last_issues: list[str] = []
        for i in range(1, max_attempts + 1):
            try:
                with client.messages.stream(
                    model=self.model_id,
                    max_tokens=64000,
                    thinking={"type": "adaptive"},
                    output_config={"effort": "high"},
                    system=system_text,
                    messages=[{"role": "user", "content": content}],
                ) as stream:
                    msg = stream.get_final_message()
            except Exception as e:
                attempts.append({"attempt": i, "issues": [f"api error: {e}"]})
                last_issues = [f"api error: {e}"]
                continue
            text = _extract_text(msg)
            try:
                from ..validate import coerce_to_dict
                parsed = coerce_to_dict(_parse_json_object(text))
            except json.JSONDecodeError as e:
                attempts.append({"attempt": i, "issues": [f"json decode: {e}"],
                                 "text_head": text[:200]})
                last_issues = [f"json decode: {e}"]
                continue
            except ValueError as e:
                attempts.append({"attempt": i, "issues": [str(e)],
                                 "text_head": text[:200]})
                last_issues = [str(e)]
                continue
            issues = validate_fn(parsed)
            usage = _usage_dict(msg)
            attempts.append({"attempt": i, "issues": issues, "usage": usage})
            if not issues:
                return parsed, text, attempts, usage
            last_issues = issues
        raise RuntimeError(
            f"All {max_attempts} Claude attempts failed validation. Last issues: {last_issues}"
        )


for m in ["claude-opus-4-7", "claude-opus-4-8"]:
    register(m, ClaudeAnnotator(m))
