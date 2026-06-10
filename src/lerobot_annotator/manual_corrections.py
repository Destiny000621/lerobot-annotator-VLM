"""Human-directed correction helpers for cases the verifier cannot infer."""
from __future__ import annotations

import json
import re
from pathlib import Path

from .config import DEFAULT_ANNOTATOR, SUCCESS_STATES_DIR, annotation_dir
from .overrides import load as load_override, save as save_override, sync_human_verified


OBJECT_PRIMITIVES = {"approach", "grasp", "transport", "align", "insert"}


def _annotation_path(repo_id: str, ep: int, annotator_model: str) -> Path:
    return annotation_dir(repo_id, annotator_model) / f"episode_{ep:06d}.json"


def _rewrite_instruction(seg: dict, descriptor: str, vial_id: int) -> str:
    primitive = seg.get("primitive")
    arm = seg.get("arm")
    obj = f"{descriptor} (vial No.{vial_id})"
    if primitive == "approach":
        return f"use the {arm} arm to approach {obj}" if arm in {"left", "right"} else f"approach {obj}"
    if primitive == "grasp":
        return f"use the {arm} arm to grasp {obj}" if arm in {"left", "right"} else f"grasp {obj}"
    if primitive == "transport":
        return (
            f"use the {arm} arm to transport {obj} toward the stand"
            if arm in {"left", "right"} else f"transport {obj} toward the stand"
        )
    if primitive == "align":
        return f"align {obj} over an open slot in the stand"
    if primitive == "insert":
        return f"insert {obj} into the stand"
    return seg.get("instruction", "")


def _progress_numerator(seg: dict) -> int | None:
    m = re.match(r"^\s*(\d+)\s*/\s*\d+\s*$", str(seg.get("progress", "")))
    return int(m.group(1)) if m else None


def _rewrite_success_paths(ann: dict, repo_id: str, ep: int, changed_insert_paths: dict[str, str]) -> None:
    if not changed_insert_paths:
        return
    base = SUCCESS_STATES_DIR.parent
    renames: list[tuple[Path, Path]] = []
    for old_rel, new_rel in changed_insert_paths.items():
        old = base / old_rel
        new = base / new_rel
        if old.exists() and old != new:
            renames.append((old, new))
    temp_paths: list[tuple[Path, Path, Path]] = []
    for old, new in renames:
        tmp = old.with_name(old.name + ".tmp_manual_rename")
        old.rename(tmp)
        temp_paths.append((tmp, old, new))
    for tmp, _old, new in temp_paths:
        new.parent.mkdir(parents=True, exist_ok=True)
        tmp.rename(new)


def apply_pickup_sequence(
    repo_id: str,
    ep: int,
    sequence: list[int],
    *,
    status: str | None = None,
    pin_segments: bool = True,
    annotator_model: str = DEFAULT_ANNOTATOR,
) -> dict:
    """Apply a human-provided sequential pickup order to one-vial action blocks.

    The kth vial in `sequence` is assigned to object-action segments whose progress
    starts with `k/`. This matches the vial_place convention that progress is the
    number of completed insertions before the segment begins.
    """
    ann_path = _annotation_path(repo_id, ep, annotator_model)
    if not ann_path.exists():
        raise FileNotFoundError(f"no annotation at {ann_path}")
    ann = json.loads(ann_path.read_text())
    n = ann.get("total_vials")
    if not isinstance(n, int):
        raise ValueError("annotation total_vials must be an int")
    if sorted(sequence) != list(range(1, n + 1)):
        raise ValueError(f"sequence {sequence} is not a permutation of 1..{n}")

    descriptors = {
        int(d["vial_id"]): str(d["descriptor"])
        for d in ann.get("vial_descriptors", [])
        if isinstance(d, dict) and "vial_id" in d and "descriptor" in d
    }
    missing_desc = [v for v in sequence if v not in descriptors]
    if missing_desc:
        raise ValueError(f"missing descriptors for vial ids {missing_desc}")

    old_sequence = ann.get("pickup_sequence")
    ann["pickup_sequence"] = sequence
    changed_segments: list[dict] = []
    changed_insert_paths: dict[str, str] = {}
    for idx, seg in enumerate(ann.get("segments", [])):
        if seg.get("primitive") not in OBJECT_PRIMITIVES:
            continue
        k = _progress_numerator(seg)
        if k is None or not (0 <= k < len(sequence)):
            continue
        new_vial = sequence[k]
        old_vial = seg.get("vial_id")
        old_instruction = seg.get("instruction")
        new_instruction = _rewrite_instruction(seg, descriptors[new_vial], new_vial)
        if old_vial == new_vial and old_instruction == new_instruction:
            continue
        seg["vial_id"] = new_vial
        seg["instruction"] = new_instruction
        changed = {
            "segment_index": idx,
            "from": {"vial_id": old_vial, "instruction": old_instruction},
            "to": {"vial_id": new_vial, "instruction": new_instruction},
        }
        if seg.get("primitive") == "insert" and seg.get("success_frame_path"):
            old_path = seg["success_frame_path"]
            new_path = re.sub(r"vial\d+(?=\.jpg$)", f"vial{new_vial}", old_path)
            if new_path != old_path:
                seg["success_frame_path"] = new_path
                changed_insert_paths[old_path] = new_path
                changed["from"]["success_frame_path"] = old_path
                changed["to"]["success_frame_path"] = new_path
        changed_segments.append(changed)

    applied = {
        "manual_pickup_sequence": {"from": old_sequence, "to": sequence},
        "segments": changed_segments,
    }
    ann["manual_corrections_applied"] = applied
    if status is not None:
        ann["status"] = status
    ann_path.write_text(json.dumps(ann, indent=2))
    _rewrite_success_paths(ann, repo_id, ep, changed_insert_paths)

    override = load_override(repo_id, ep)
    override.pinned_count = n
    override.pinned_fields = {**(override.pinned_fields or {}), "pickup_sequence": sequence}
    if pin_segments:
        override.pinned_segments = ann.get("segments", [])
    if status is not None:
        override.status = status
    save_override(override)
    if status is not None:
        sync_human_verified(repo_id, ep, annotator_model, override.status)
    return applied
