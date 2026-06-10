"""Apply structured corrections from the AI verifier back into the annotation + override store.

This closes the loop: verifier finds errors → emits structured corrections → corrections are
deterministically applied to the annotation. The user can then visually review in the UI and
either approve or further tweak.
"""
from __future__ import annotations
import json
import re
from pathlib import Path

from .config import DEFAULT_ANNOTATOR, annotation_dir, verification_dir
from .overrides import Override, load as load_override, save as save_override
from .verify import is_verification_stale


_PER_SEG_FIELDS = ("arm", "vial_id", "primitive", "instruction")


def _flag_incomplete_progress(ann: dict, override: Override, applied: dict) -> None:
    """Sanity check: does the final segment's progress match total_vials?
    If not, segments don't cover all vials; mark needs_rework + emit a hint."""
    n = ann.get("total_vials")
    segs = ann.get("segments") or []
    if not (isinstance(n, int) and segs):
        return
    last_prog = segs[-1].get("progress", "")
    m = re.match(r"^\s*(\d+)\s*/\s*(\d+)\s*$", last_prog)
    if not m:
        return
    if int(m.group(1)) != n or int(m.group(2)) != n:
        override.status = "needs_rework"
        applied["incomplete_progress"] = {
            "final": last_prog, "expected": f"{n}/{n}",
            "hint": ("Final segment progress doesn't reach N/N — the existing segments "
                     "don't cover all vials. Re-annotate with --overwrite to regenerate "
                     "segments against the pinned count."),
        }


def _rewrite_progress_denominator(seg: dict, new_n: int) -> bool:
    """In-place: replace `k/<anything>` denominator in seg['progress'] with k/<new_n>.
    Returns True if a change was made."""
    prog = seg.get("progress")
    if not isinstance(prog, str):
        return False
    m = re.match(r"^\s*(\d+)\s*/\s*(\d+)\s*$", prog)
    if not m:
        return False
    k, old_n = m.group(1), m.group(2)
    if int(old_n) == new_n:
        return False
    seg["progress"] = f"{k}/{new_n}"
    return True


def _annotation_path(repo_id: str, ep: int, annotator_model: str) -> Path:
    return annotation_dir(repo_id, annotator_model) / f"episode_{ep:06d}.json"


def _verification_path(repo_id: str, ep: int, annotator_model: str) -> Path:
    return verification_dir(repo_id, annotator_model) / f"episode_{ep:06d}.json"


def _merge_corrections(verification: dict) -> dict:
    """If multiple verifier models ran, merge their `corrections` payloads conservatively:
    union over fields, last writer wins for conflicting per-field values.
    """
    merged: dict = {}
    seg_edits_by_idx: dict[int, dict] = {}
    for r in (verification.get("verifiers") or {}).values():
        corr = (r.get("parsed") or {}).get("corrections") or {}
        for k in ("total_vials", "pickup_sequence", "vial_descriptors"):
            if k in corr:
                merged[k] = corr[k]
        for seg in corr.get("segments") or []:
            idx = seg.get("segment_index")
            if isinstance(idx, int):
                edits = seg_edits_by_idx.setdefault(idx, {"segment_index": idx})
                for f in _PER_SEG_FIELDS:
                    if f in seg:
                        edits[f] = seg[f]
    if seg_edits_by_idx:
        merged["segments"] = list(seg_edits_by_idx.values())
    return merged


def apply_episode(repo_id: str, ep: int, also_pin_segments: bool = True,
                  annotator_model: str = DEFAULT_ANNOTATOR) -> dict:
    """Apply verifier corrections for one episode.

    Returns a summary dict {applied: {...}, no_corrections: bool}.
    """
    ann_path = _annotation_path(repo_id, ep, annotator_model)
    if not ann_path.exists():
        raise FileNotFoundError(f"no annotation at {ann_path}")
    ver_path = _verification_path(repo_id, ep, annotator_model)
    if not ver_path.exists():
        raise FileNotFoundError(f"no verification at {ver_path} — run `cli.py verify` first")
    if is_verification_stale(repo_id, ep, annotator_model):
        raise RuntimeError(
            f"verification at {ver_path} is stale for the current annotation — run `cli.py verify` again"
        )

    ann = json.loads(ann_path.read_text())
    verification = json.loads(ver_path.read_text())
    corrections = _merge_corrections(verification)

    override = load_override(repo_id, ep)
    applied: dict = {}

    # Normalize stale progress denominators against current total_vials BEFORE applying any
    # corrections. Handles legacy annotations where total_vials was bumped but progress strings
    # still carry the old denominator.
    current_n = ann.get("total_vials")
    if isinstance(current_n, int) and current_n > 0:
        n_normalized = 0
        for s in ann.get("segments", []):
            if _rewrite_progress_denominator(s, current_n):
                n_normalized += 1
        if n_normalized:
            applied["progress_denominators_normalized"] = n_normalized

    if not corrections:
        # No new corrections, but still run sanity checks below.
        _flag_incomplete_progress(ann, override, applied)
        ann["verifier_corrections_applied"] = applied
        ann_path.write_text(json.dumps(ann, indent=2))
        save_override(override)
        return {"applied": applied, "no_corrections": True}

    if "total_vials" in corrections:
        new_n = int(corrections["total_vials"])
        old_n = int(ann.get("total_vials") or 0)
        if new_n != old_n:
            ann["total_vials"] = new_n
            override.pinned_count = new_n
            applied["total_vials"] = {"from": old_n, "to": new_n}
            # propagate denominator change to all segments' progress strings
            n_progress_fixed = 0
            for s in ann.get("segments", []):
                if _rewrite_progress_denominator(s, new_n):
                    n_progress_fixed += 1
            if n_progress_fixed:
                applied["progress_denominators_rewritten"] = n_progress_fixed
            # large count changes mean the current segments are scoped to the wrong vial set —
            # surgical edits won't recover; suggest re-annotation
            if abs(new_n - old_n) >= 1:
                override.status = "needs_rework"
                applied["status_changed_to"] = "needs_rework"
                applied["note"] = (
                    f"total_vials changed by {new_n - old_n}; surgical corrections cannot "
                    f"add or remove vial sequences. Re-annotate with --overwrite to regenerate "
                    f"segments against the corrected count."
                )

    if "pickup_sequence" in corrections:
        ann["pickup_sequence"] = corrections["pickup_sequence"]
        override.pinned_fields = {**(override.pinned_fields or {}),
                                  "pickup_sequence": corrections["pickup_sequence"]}
        applied["pickup_sequence"] = corrections["pickup_sequence"]

    if "vial_descriptors" in corrections:
        ann["vial_descriptors"] = corrections["vial_descriptors"]
        override.pinned_fields = {**(override.pinned_fields or {}),
                                  "vial_descriptors": corrections["vial_descriptors"]}
        applied["vial_descriptors"] = corrections["vial_descriptors"]

    seg_edits = corrections.get("segments") or []
    applied_segs: list[dict] = []
    skipped_noops = 0
    for edit in seg_edits:
        idx = edit.get("segment_index")
        if not isinstance(idx, int) or not (0 <= idx < len(ann.get("segments", []))):
            continue
        seg = ann["segments"][idx]
        edit_record = {"segment_index": idx, "from": {}, "to": {}}
        changed_any = False
        for f in _PER_SEG_FIELDS:
            if f not in edit:
                continue
            old_val = seg.get(f)
            new_val = edit[f]
            if old_val == new_val:
                continue  # skip no-op
            edit_record["from"][f] = old_val
            edit_record["to"][f] = new_val
            seg[f] = new_val
            changed_any = True
        if changed_any:
            applied_segs.append(edit_record)
        else:
            skipped_noops += 1
    if applied_segs:
        applied["segments"] = applied_segs
    if skipped_noops:
        applied["skipped_noop_segment_edits"] = skipped_noops

    if also_pin_segments and applied_segs:
        # Lock the corrected segments so re-annotation doesn't undo them
        override.pinned_segments = ann["segments"]

    _flag_incomplete_progress(ann, override, applied)

    # Trace which verifier(s) sourced the corrections
    ann["verifier_corrections_applied"] = applied

    ann_path.write_text(json.dumps(ann, indent=2))
    save_override(override)
    return {"applied": applied, "no_corrections": False}


def apply_batch(repo_id: str, episodes: list[int],
                annotator_model: str = DEFAULT_ANNOTATOR) -> dict:
    out = {}
    for ep in episodes:
        try:
            out[ep] = apply_episode(repo_id, ep, annotator_model=annotator_model)
        except Exception as e:
            out[ep] = {"error": str(e)}
    return out
