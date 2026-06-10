"""Thin Gemini Robotics-ER native API client with count-vote and image-budget helpers."""
from __future__ import annotations
import base64
import io
import json
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path  # noqa: F401 — used by Path-aware helpers

from PIL import Image

from .config import (
    GEMINI_URL, MAX_EDGE_PX, MAX_IMAGES_PER_CALL, get_litellm_key,
)


def b64_thumbnail(path: str | Path, max_edge: int = MAX_EDGE_PX) -> str:
    with Image.open(path) as img:
        img = img.convert("RGB")
        w, h = img.size
        scale = max_edge / max(w, h)
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return base64.b64encode(buf.getvalue()).decode("ascii")


def post(payload: dict, timeout: int = 600) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        GEMINI_URL,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": get_litellm_key(),
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} from Gemini gateway: {detail[:1500]}") from None


def subsample_evenly(items: list, k: int) -> list:
    if k >= len(items) or k <= 0:
        return items if k > 0 else []
    step = (len(items) - 1) / max(1, k - 1)
    idxs = sorted({min(len(items) - 1, round(i * step)) for i in range(k)})
    return [items[i] for i in idxs]


def count_vote(
    image_paths,
    question: str,
    range_lo: int,
    range_hi: int,
    n_votes: int = 5,
    prefer_final_frame: bool = False,
) -> tuple[int, list[int]]:
    """Send one or more corroborating images to Gemini N times asking for an integer count.
    `image_paths` may be a single path or a list of `(label, path)` tuples / paths.
    Returns (mode, all_votes)."""
    labeled = _normalize_images(image_paths)
    user_parts: list[dict] = [{"text": _build_count_prompt(
        question, labeled, range_lo, range_hi, prefer_final_frame
    )}]
    for label, p in labeled:
        user_parts.append({"text": label})
        user_parts.append({"inline_data": {"mime_type": "image/jpeg", "data": b64_thumbnail(p)}})

    votes: list[int] = []
    for i in range(n_votes):
        payload = {
            "contents": [{"role": "user", "parts": user_parts}],
            "generationConfig": {
                "temperature": 0.0 if i == 0 else 0.4,
                "responseMimeType": "application/json",
            },
        }
        try:
            resp = post(payload)
            text = resp["candidates"][0]["content"]["parts"][0]["text"]
            obj = json.loads(text)
            c = int(obj["count"])
            if range_lo <= c <= range_hi:
                votes.append(c)
        except Exception:
            continue
    if not votes:
        raise RuntimeError(f"count vote failed all {n_votes} attempts")
    counter = Counter(votes)
    top_freq = counter.most_common(1)[0][1]
    winners = [c for c, f in counter.items() if f == top_freq]
    return max(winners), votes


def _normalize_images(images) -> list[tuple[str, "Path"]]:
    """Accept a single path, a list of paths, or a list of (label, path) tuples."""
    if isinstance(images, (str, Path)):
        return [("[FIRST_FRAME]", images)]
    out: list[tuple[str, Path]] = []
    for item in images:
        if isinstance(item, (str, Path)):
            out.append(("[FRAME]", item))
        else:
            out.append((item[0], item[1]))
    return out


def _build_count_prompt(question: str, labeled, range_lo: int, range_hi: int,
                        prefer_final_frame: bool = False) -> str:
    multi = len(labeled) > 1
    schema = (
        f'{{"count": <int {range_lo}..{range_hi}>, '
        + (('"first_frame_count": <int>, "final_frame_in_stand": <int>, '
            if multi else '')
           + '"reasoning": "<one short sentence>"}')
    )
    extra = ""
    if multi and prefer_final_frame:
        extra = (
            "\n\nYou will see TWO frames. For THIS TASK, every episode is a known success — "
            "the robot inserts ALL vials into the stand. Therefore:\n"
            "- The number of vials in the stand at the FINAL frame IS the ground-truth count.\n"
            "- The FIRST frame is shown for additional context only; vials in it may be partially "
            "  occluded by cables/arms/the stand and you may undercount them.\n"
            "- Your reported `count` MUST equal `final_frame_in_stand` (the number of vials you "
            "  can count inserted into the stand's slots in the final frame).\n"
            "- Still report `first_frame_count` for auditability, but base `count` on the final frame."
        )
    elif multi:
        extra = (
            "\n\nYou will see MULTIPLE corroborating frames. Use ALL of them:\n"
            "- The FIRST frame shows the initial scene, but objects may be partially occluded.\n"
            "- The FINAL frame shows the scene at the end of the episode.\n"
            "- The reported `count` MUST be at least max(first_frame_count, final_frame_in_stand). "
            "  If they disagree, prefer the larger value — occlusion typically causes UNDERcounting."
        )
    return f"{question}{extra}\n\nOutput STRICT JSON only: {schema}"


def generate_with_retry(
    system_text: str,
    user_parts: list[dict],
    validate_fn,
    max_attempts: int = 4,
    base_temperature: float = 0.0,
) -> tuple[dict, str, list[dict], dict]:
    """Call Gemini up to `max_attempts` times, escalating temperature on failure.

    `validate_fn(parsed_json) -> list[str]` returns issues (empty = valid).
    Returns (parsed_json, raw_text, attempt_log, usage_of_winner).
    """
    attempts: list[dict] = []
    last_issues: list[str] = []
    for i in range(1, max_attempts + 1):
        payload = {
            "systemInstruction": {"parts": [{"text": system_text}]},
            "contents": [{"role": "user", "parts": user_parts}],
            "generationConfig": {
                "temperature": base_temperature if i == 1 else 0.2 * i,
                "responseMimeType": "application/json",
            },
        }
        resp = post(payload)
        text = resp["candidates"][0]["content"]["parts"][0]["text"]
        try:
            from .validate import coerce_to_dict
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
        usage = resp.get("usageMetadata", {})
        attempts.append({
            "attempt": i,
            "temperature": payload["generationConfig"]["temperature"],
            "issues": issues,
            "usage": usage,
        })
        if not issues:
            return parsed, text, attempts, usage
        last_issues = issues
    raise RuntimeError(f"All {max_attempts} attempts failed validation. Last issues: {last_issues}")
