"""Few-shot exemplars from human-verified annotations.

Verified episodes are detected by `overrides/<repo>/episode_XXXXXX.json` having
`status == "verified"`. The cleaned annotation + first-frame image are prepended
to the user prompt so the VLM mimics the human-corrected schema and style.
"""
from __future__ import annotations
import json
from pathlib import Path

from .config import (
    ANNOTATIONS_DIR, KEYFRAMES_CACHE, OVERRIDES_DIR,
    DEFAULT_ANNOTATOR, EXEMPLAR_INCLUDE_FIRST_FRAME, MAX_EXEMPLARS,
    PREFERRED_EXEMPLARS, annotation_dir, human_verified_dir, list_annotators_for_repo,
    repo_safe_name,
)


# Fields we keep in an exemplar — drop everything else as noise.
_TOP_KEEP = {
    "episode_index", "length", "duration_s", "fps",
    "annotator_model",
    "total_vials", "stand_slots",
    "vial_descriptors", "pickup_sequence",
    "segments",
}
_SEG_KEEP = {
    "start_s", "end_s", "arm", "primitive",
    "vial_id", "instruction", "progress",
}


def find_verified(repo_id: str, exclude: set[int] | None = None) -> list[int]:
    """Episodes where ANY annotator's annotation file has status == "verified".

    Status moved from the shared override store onto the per-annotator
    annotation, so this scan walks each annotator subdir under annotations/
    and collects every episode flagged verified by any of them.
    """
    out: set[int] = set()
    exclude = exclude or set()
    for model in list_annotators_for_repo(repo_id):
        model_dir = ANNOTATIONS_DIR / repo_safe_name(repo_id) / model
        for p in sorted(model_dir.glob("episode_*.json")):
            try:
                data = json.loads(p.read_text())
                if data.get("status") != "verified":
                    continue
                ep = int(p.stem.split("_")[1])
                if ep in exclude:
                    continue
                out.add(ep)
            except Exception:
                continue
    # Legacy fallback: also honor the old shared override.status (so episodes
    # verified before this migration still show up as exemplars).
    legacy_dir = OVERRIDES_DIR / repo_safe_name(repo_id)
    if legacy_dir.exists():
        for p in sorted(legacy_dir.glob("episode_*.json")):
            try:
                data = json.loads(p.read_text())
                if data.get("status") != "verified":
                    continue
                ep = int(p.stem.split("_")[1])
                if ep in exclude:
                    continue
                out.add(ep)
            except Exception:
                continue
    preferred = PREFERRED_EXEMPLARS.get(repo_id, [])
    rank = {ep: i for i, ep in enumerate(preferred)}
    return sorted(out, key=lambda ep: (0, rank[ep]) if ep in rank else (1, ep))


def load_cleaned_annotation(repo_id: str, ep: int,
                            preferred_annotator: str | None = None) -> dict | None:
    """Look up an episode's annotation, preferring the human_verified snapshot.

    Resolution order:
      1. annotations/<repo>/human_verified/episode_*.json  (canonical ground truth)
      2. annotations/<repo>/<preferred_annotator>/episode_*.json (if provided)
      3. annotations/<repo>/<any annotator subdir>/episode_*.json (first match)
    """
    p: Path | None = None
    hv = human_verified_dir(repo_id) / f"episode_{ep:06d}.json"
    if hv.exists():
        p = hv
    if p is None and preferred_annotator:
        cand = annotation_dir(repo_id, preferred_annotator) / f"episode_{ep:06d}.json"
        if cand.exists():
            p = cand
    if p is None:
        for model in list_annotators_for_repo(repo_id):
            cand = annotation_dir(repo_id, model) / f"episode_{ep:06d}.json"
            if cand.exists():
                p = cand
                break
    if p is None:
        return None
    ann = json.loads(p.read_text())
    cleaned = {k: ann[k] for k in _TOP_KEEP if k in ann}
    if "segments" in cleaned:
        cleaned["segments"] = [
            {k: s[k] for k in _SEG_KEEP if k in s}
            for s in cleaned["segments"]
        ]
    return cleaned


def find_first_frame_path(repo_id: str, ep: int, head_cam: str) -> Path | None:
    """Return a cached first-frame JPEG path for the head camera, any fps bucket."""
    base = KEYFRAMES_CACHE / repo_safe_name(repo_id) / f"episode_{ep:06d}"
    if not base.exists():
        return None
    candidates = list(base.glob(f"{head_cam}_fps*")) + [base / head_cam]
    for d in sorted(candidates, reverse=True):  # prefer higher fps buckets first
        first = d / "t000.000.jpg"
        if first.exists():
            return first
    return None


def build_exemplar_parts(
    repo_id: str,
    head_cam: str,
    exclude_episode: int,
    include_images: bool = EXEMPLAR_INCLUDE_FIRST_FRAME,
    limit: int | None = MAX_EXEMPLARS,
    target_count: int | None = None,
    explicit_episodes: list[int] | None = None,
    preferred_annotator: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """Return (parts, used_meta).

    `parts` is the prefix to inject into the user message before the new
    episode's content. `used_meta` is a list of dicts describing each exemplar
    (for logging into the annotation JSON).
    """
    if explicit_episodes is not None:
        seen: set[int] = set()
        eps = []
        for ep in explicit_episodes:
            if ep == exclude_episode or ep in seen:
                continue
            seen.add(ep)
            eps.append(ep)
    else:
        candidates = find_verified(repo_id, exclude={exclude_episode})
        if target_count is not None:
            same_count: list[int] = []
            other_count: list[int] = []
            for ep in candidates:
                ann = load_cleaned_annotation(repo_id, ep, preferred_annotator)
                if ann and ann.get("total_vials") == target_count:
                    same_count.append(ep)
                else:
                    other_count.append(ep)
            eps = same_count + other_count
        else:
            eps = candidates
    if explicit_episodes is None and limit is not None:
        eps = eps[:limit]
    if not eps:
        return [], []

    parts: list[dict] = [{"text": (
        f"=== {len(eps)} HUMAN-VERIFIED EXEMPLAR ANNOTATION(S) FROM THIS DATASET ===\n"
        "These were corrected and approved by a human. They define the EXPECTED "
        "schema, numbering convention, instruction wording, and segment granularity. "
        "Treat them as ground-truth references — your new annotation should mimic "
        "the same shape and style."
    )}]
    used_meta: list[dict] = []
    for ep in eps:
        ann = load_cleaned_annotation(repo_id, ep)
        if ann is None:
            continue
        ff = find_first_frame_path(repo_id, ep, head_cam) if include_images else None
        header = (
            f"--- EXEMPLAR: episode {ep} "
            f"(duration {ann.get('duration_s', 0):.1f}s, "
            f"{len(ann.get('segments', []))} segments) ---"
        )
        parts.append({"text": header})
        if ff:
            parts.append({"text": "First frame:"})
            parts.append({"image": str(ff)})
        parts.append({"text": "Verified annotation JSON:"})
        parts.append({"text": json.dumps(ann, indent=2)})
        used_meta.append({
            "episode": ep,
            "n_segments": len(ann.get("segments", [])),
            "first_frame_included": bool(ff),
        })

    parts.append({"text": "=== END EXEMPLARS — NOW ANNOTATE THE NEW EPISODE BELOW ==="})
    return parts, used_meta
