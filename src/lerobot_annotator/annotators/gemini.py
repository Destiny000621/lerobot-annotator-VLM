"""Gemini Robotics-ER 1.6 annotator backend (Avant LiteLLM gateway, native /v1beta API)."""
from __future__ import annotations
from pathlib import Path

from ..gemini import b64_thumbnail, count_vote, generate_with_retry
from . import register


def _to_native_parts(neutral_parts: list[dict]) -> list[dict]:
    """Convert neutral parts [{text}/{image}] to Gemini's parts schema."""
    out: list[dict] = []
    for p in neutral_parts:
        if "text" in p:
            out.append({"text": p["text"]})
        elif "image" in p:
            out.append({"inline_data": {"mime_type": "image/jpeg",
                                        "data": b64_thumbnail(p["image"])}})
    return out


class GeminiAnnotator:
    def count_vote(self, image_path, question: str,
                   range_lo: int, range_hi: int, n_votes: int = 5,
                   prefer_final_frame: bool = False):
        return count_vote(image_path, question, range_lo, range_hi, n_votes=n_votes,
                          prefer_final_frame=prefer_final_frame)

    def generate_with_retry(self, system_text: str, neutral_parts: list[dict],
                            validate_fn, max_attempts: int = 4):
        native_parts = _to_native_parts(neutral_parts)
        return generate_with_retry(system_text, native_parts, validate_fn,
                                   max_attempts=max_attempts)


register("gemini-robotics-er-1.6-preview", GeminiAnnotator())
