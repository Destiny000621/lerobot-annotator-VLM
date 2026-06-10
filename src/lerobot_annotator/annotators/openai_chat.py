"""OpenAI Chat Completions annotator backend (GPT-5.5, GPT-4o, etc.).

Uses the same image-token math as the verifier backend: full-resolution images sent
as data URLs (no Gemini-style downscale). GPT-5/5.5 are reasoning models — they don't
accept `temperature`, so we pass it only to non-reasoning families.
"""
from __future__ import annotations
import base64
import io
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from ..config import VERIFIER_OPENAI_MAX_EDGE, get_openai_key
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


def _is_reasoning_model(model: str) -> bool:
    """gpt-5*, o4*, o5* are reasoning models and don't accept temperature."""
    return model.startswith("gpt-5") or model.startswith("o4") or model.startswith("o5")


def _client():
    from openai import OpenAI
    key = get_openai_key()
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set and .secrets/openai_key.txt missing")
    return OpenAI(api_key=key)


def _neutral_to_content(neutral_parts: list[dict]) -> list[dict]:
    out: list[dict] = []
    for p in neutral_parts:
        if "text" in p:
            out.append({"type": "text", "text": p["text"]})
        elif "image" in p:
            out.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{_b64_jpeg(p['image'])}"},
            })
    return out


class OpenAIAnnotator:
    def __init__(self, model_id: str):
        self.model_id = model_id

    def count_vote(self, image_paths, question: str,
                   range_lo: int, range_hi: int, n_votes: int = 5,
                   prefer_final_frame: bool = False):
        from ..gemini import _normalize_images, _build_count_prompt
        client = _client()
        labeled = _normalize_images(image_paths)
        prompt = _build_count_prompt(question, labeled, range_lo, range_hi, prefer_final_frame)
        content = [{"type": "text", "text": prompt}]
        for label, p in labeled:
            content.append({"type": "text", "text": label})
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{_b64_jpeg(p)}"}})
        votes: list[int] = []
        for _ in range(n_votes):
            kwargs = dict(
                model=self.model_id,
                messages=[{"role": "user", "content": content}],
                response_format={"type": "json_object"},
            )
            try:
                resp = client.chat.completions.create(**kwargs)
                text = resp.choices[0].message.content or "{}"
                obj = json.loads(text)
                c = int(obj["count"])
                if range_lo <= c <= range_hi:
                    votes.append(c)
            except Exception:
                continue
        if not votes:
            raise RuntimeError(f"OpenAI count vote failed all {n_votes} attempts")
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
            kwargs = dict(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": system_text},
                    {"role": "user", "content": content},
                ],
                response_format={"type": "json_object"},
            )
            if not _is_reasoning_model(self.model_id):
                kwargs["temperature"] = 0.0 if i == 1 else 0.2 * i
            resp = client.chat.completions.create(**kwargs)
            text = resp.choices[0].message.content or "{}"
            try:
                from ..validate import coerce_to_dict
                parsed = coerce_to_dict(json.loads(text))
            except json.JSONDecodeError as e:
                attempts.append({"attempt": i, "issues": [f"json decode: {e}"], "text_head": text[:200]})
                last_issues = [f"json decode: {e}"]
                continue
            except ValueError as e:
                attempts.append({"attempt": i, "issues": [str(e)], "text_head": text[:200]})
                last_issues = [str(e)]
                continue
            issues = validate_fn(parsed)
            usage = resp.usage.model_dump() if resp.usage else {}
            attempts.append({
                "attempt": i,
                "temperature": kwargs.get("temperature"),
                "issues": issues,
                "usage": usage,
            })
            if not issues:
                return parsed, text, attempts, usage
            last_issues = issues
        raise RuntimeError(
            f"All {max_attempts} OpenAI attempts failed validation. Last issues: {last_issues}"
        )


# Register all reasonable OpenAI annotator candidates
for m in [
    "gpt-5.5", "gpt-5.5-pro",
    "gpt-5.4", "gpt-5.4-pro",
    "gpt-5.3-chat-latest",
    "gpt-5.2", "gpt-5.1", "gpt-5",
    "gpt-4o", "gpt-4.1",
]:
    register(m, OpenAIAnnotator(m))
