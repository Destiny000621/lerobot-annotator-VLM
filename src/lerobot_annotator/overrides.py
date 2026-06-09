"""Per-(repo, episode) human override store.

An Override captures whatever the human has fixed:
- pinned_count: forces total_vials (or other count field) to this value
- pinned_segments: replace the VLM output entirely
- field overrides: dict of {top-level field -> value} to inject after the call
- status: draft | needs_rework | verified
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import json
from pathlib import Path

from .config import OVERRIDES_DIR, repo_safe_name


@dataclass
class Override:
    repo_id: str
    episode_index: int
    status: str = "draft"  # draft | needs_rework | verified
    pinned_count: int | None = None
    pinned_fields: dict = field(default_factory=dict)   # top-level field replacements
    pinned_segments: list | None = None                  # replaces segments wholesale if set
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _path(repo_id: str, ep: int) -> Path:
    return OVERRIDES_DIR / repo_safe_name(repo_id) / f"episode_{ep:06d}.json"


def load(repo_id: str, ep: int) -> Override:
    p = _path(repo_id, ep)
    if not p.exists():
        return Override(repo_id=repo_id, episode_index=ep)
    d = json.loads(p.read_text())
    return Override(
        repo_id=d.get("repo_id", repo_id),
        episode_index=d.get("episode_index", ep),
        status=d.get("status", "draft"),
        pinned_count=d.get("pinned_count"),
        pinned_fields=dict(d.get("pinned_fields", {})),
        pinned_segments=d.get("pinned_segments"),
        notes=d.get("notes", ""),
    )


def save(ov: Override) -> None:
    p = _path(ov.repo_id, ov.episode_index)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(ov.to_dict(), indent=2))


def clear(repo_id: str, ep: int, *, keep_status: bool = False,
          keep_segments: bool = False, keep_fields: bool = False,
          keep_count: bool = False) -> Override:
    """Reset pinned values on an override. By default clears everything except notes.
    Pass keep_* flags to preserve specific pinned values."""
    ov = load(repo_id, ep)
    if not keep_count:
        ov.pinned_count = None
    if not keep_fields:
        ov.pinned_fields = {}
    if not keep_segments:
        ov.pinned_segments = None
    if not keep_status:
        ov.status = "draft"
    save(ov)
    return ov


def apply_post_call(parsed: dict, ov: Override) -> dict:
    """Apply pinned_fields and pinned_segments after a VLM call returns.
    pinned_count is applied BEFORE the call as a constraint, not here."""
    for k, v in (ov.pinned_fields or {}).items():
        parsed[k] = v
    if ov.pinned_segments is not None:
        parsed["segments"] = ov.pinned_segments
    return parsed
