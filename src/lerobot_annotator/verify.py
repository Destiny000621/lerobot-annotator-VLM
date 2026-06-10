"""AI verifier orchestrator. Runs one or more verifier backends over annotations and
saves their flagged issues to verifications/<repo>/episode_XXXXXX.json.

The verifier output schema:
{
  "overall": "pass" | "issues_found",
  "issues": [
    {"segment_index": <int>, "type": "<error type>", "severity": "minor"|"major",
     "description": "<sentence>"}
  ],
  "summary": "<one-paragraph overall assessment>"
}
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

from .config import (
    DEFAULT_ANNOTATOR, KEYFRAMES_CACHE, SUCCESS_STATES_DIR,
    VERIFIER_MAX_FRAMES, VERIFIER_MODELS, VERIFIER_USE_WRIST_CAMERAS,
    annotation_dir, verification_dir, repo_safe_name,
)
from .dataset import LeRobotDataset
from .extract import _video_path, extract_frame_at
from .task import Task
from . import verifiers


SYSTEM_PROMPT = """You are an expert robotics annotation auditor. Your job is to AGGRESSIVELY
flag errors in a VLM-generated annotation. Bias toward false positives over false negatives —
if something looks even slightly off, flag it. A human will adjudicate.

You will be shown:
1. The annotation JSON (segments + vial_descriptors + pickup_sequence).
2. Multiple camera views per critical moment:
   - HEAD camera: overhead view of the whole scene.
   - LEFT_WRIST: mounted on the LEFT arm — shows what the LEFT arm is interacting with.
   - RIGHT_WRIST: mounted on the RIGHT arm — shows what the RIGHT arm is interacting with.

CRITICAL CHECK: ARM/VIAL ASSIGNMENT
For every grasp/insert claim:
- If the annotation says "LEFT arm grasps vial K", the LEFT_WRIST camera at that moment MUST
  show the left gripper closing on a vial. If LEFT_WRIST shows an empty/idle gripper and
  RIGHT_WRIST shows a vial being grasped, the annotation has SWAPPED the arms — flag this.
- If the annotation says "BOTH arms grasp", BOTH wrist cameras must show grippers closing on
  vials simultaneously. If only one wrist camera shows action, the annotation incorrectly
  claims bimanual operation — flag this.

CRITICAL CHECK: TEMPORAL CONSISTENCY
- If left arm is claimed to be holding vial K from t=A to t=B, the left wrist must show that vial
  throughout the [A,B] window. Inconsistencies (e.g. left arm "holds vial 1" for 12s while the
  right arm independently inserts vial 2) are usually wrong.
- Long idle segments (>10s) in the middle of an episode are suspicious — verify whether the
  arm is actually idle or whether a real action was missed.

CRITICAL CHECK: COUNT & DESCRIPTORS
- Count vials in the FIRST frame and compare to total_vials.
- For each vial_descriptor, find that vial in the first frame and verify the descriptor matches
  its actual position.

Error types:
- wrong_arm         : annotation's arm doesn't match which wrist camera shows the action.
- wrong_vial        : annotation's vial_id doesn't match the descriptor of the vial actually grasped.
- wrong_primitive   : claimed primitive doesn't match what the cameras show.
- wrong_bimanual    : annotation says "both arms" but only one arm is acting (or vice versa).
- wrong_timing      : action shown doesn't fall inside the segment's [start_s, end_s] window.
- missing_vial      : a vial visible initially is never picked up or inserted.
- extra_segment     : a segment that doesn't correspond to any visible action.
- wrong_count       : total_vials wrong vs. first-frame count.
- wrong_descriptor  : vial_descriptor doesn't match the actual position of that vial_id.
- suspicious_idle   : long idle segment that may hide a missed action.
- other             : anything else that's wrong.

Severity:
- major  : semantically changes what the policy would do (wrong_arm, wrong_vial, wrong_primitive,
           wrong_bimanual, wrong_count, missing_vial)
- minor  : timing off by <1s, descriptor stylistic differences, etc.

Output STRICT JSON only:
{
  "overall": "pass" | "issues_found",
  "issues": [
    {
      "segment_index": <int 0-based, or null for episode-level>,
      "type": "<one of the error types above>",
      "severity": "minor" | "major",
      "description": "<one specific sentence citing what you saw in which camera at which time>"
    }
  ],
  "corrections": {
    "total_vials":      <int>,             // optional, only if wrong_count detected
    "pickup_sequence":  [<int>, ...],       // optional, only if you saw the correct order in the videos
    "vial_descriptors": [                   // optional, only if wrong_descriptor; replace whole list
      {"vial_id": <int>, "descriptor": "<corrected>"},
      ...
    ],
    "segments": [                           // optional list of per-segment field edits.
      // For EVERY segment correction, you MUST emit BOTH:
      //   (a) the structured field(s) you are fixing (arm and/or vial_id and/or primitive)
      //   (b) a rewritten `instruction` string consistent with the corrected fields.
      // Do NOT emit `instruction` alone — always pair it with the structured field(s) it reflects.
      {
        "segment_index": <int>,
        "arm": "left"|"right"|"both",       // include if changing arm
        "vial_id": <int or null>,             // include if changing vial_id
        "primitive": "<primitive>",           // include if changing primitive
        "instruction": "<rewritten one-sentence instruction consistent with all corrected fields>"
      }
    ]
  },
  "summary": "<two-sentence overall assessment>"
}

CORRECTIONS RULES:
- Only include a correction if you are CONFIDENT it is the right fix. False corrections are
  worse than no correction.
- For arm swaps (left↔right), include the segment in `corrections.segments` with the corrected arm
  AND a rewritten `instruction` using the new arm (e.g. "use the left arm to ..." instead of right).
- For vial-id mismatches, include the corrected `vial_id` AND a rewritten `instruction` referencing
  the correct vial number and descriptor.
- For primitive changes, include the corrected `primitive` AND a rewritten `instruction` matching
  the new primitive (e.g. "transport ..." instead of "grasp ...").
- For wrong_count, include `total_vials` with the count you actually see in the first frame.
- For wrong_descriptor, replace `vial_descriptors` with the full corrected list.
- DO NOT include corrections for `suspicious_idle`, `missing_vial`, `extra_segment`, or
  `wrong_timing` — those need a human to decide how to restructure segments.
- The rewritten `instruction` must follow the same style as the original annotation:
  positional descriptor first, then "(vial No.K)" in parentheses, then the action with the
  corrected arm. Example: "use the left arm to align the leftmost vial (vial No.1) over a stand slot".
- If you have no confident corrections to make, omit the `corrections` field entirely or set it to {}.

Only return overall="pass" if you carefully checked EVERY grasp/insert against the wrist cameras
and found NO arm/vial mismatches.
"""


def _verification_path(repo_id: str, ep: int, annotator_model: str) -> Path:
    return verification_dir(repo_id, annotator_model) / f"episode_{ep:06d}.json"


def _annotation_path(repo_id: str, ep: int, annotator_model: str) -> Path:
    return annotation_dir(repo_id, annotator_model) / f"episode_{ep:06d}.json"


def _annotation_fingerprint(path: Path) -> dict:
    data = path.read_bytes()
    return {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}


def is_verification_stale(repo_id: str, ep: int, annotator_model: str = DEFAULT_ANNOTATOR) -> bool:
    ann_path = _annotation_path(repo_id, ep, annotator_model)
    ver_path = _verification_path(repo_id, ep, annotator_model)
    if not ver_path.exists() or not ann_path.exists():
        return True
    try:
        ver = json.loads(ver_path.read_text())
    except Exception:
        return True
    fp = ver.get("annotation_fingerprint")
    if fp:
        return fp.get("sha256") != _annotation_fingerprint(ann_path)["sha256"]
    return ann_path.stat().st_mtime > ver_path.stat().st_mtime


def _pick_frames(ann: dict, dataset: LeRobotDataset, ep: int,
                 max_frames: int = VERIFIER_MAX_FRAMES,
                 use_wrists: bool = VERIFIER_USE_WRIST_CAMERAS,
                 annotator_model: str = DEFAULT_ANNOTATOR) -> list[tuple[str, Path, dict]]:
    """Pick the most informative camera frames. For grasp/insert/bimanual segments, attach
    head + left_wrist + right_wrist views at the same timestamp so arm/vial assignments are
    visible to the verifier.
    """
    safe = repo_safe_name(dataset.repo_id)
    cams = dataset.cameras()
    head_cam  = next((c for c in cams if "head" in c.lower()), cams[0])
    left_cam  = next((c for c in cams if "left"  in c.lower()), None)
    right_cam = next((c for c in cams if "right" in c.lower()), None)
    base = KEYFRAMES_CACHE / safe / f"episode_{ep:06d}"

    def _glob_buckets(cam: str) -> list[Path]:
        return sorted(base.glob(f"{cam}_fps*"), reverse=True)

    def find_cached(cam: str, sec: float) -> Path | None:
        for d in _glob_buckets(cam):
            f = d / f"t{sec:07.3f}.jpg"
            if f.exists():
                return f
        return None

    def cached_or_decode(cam: str, sec: float) -> Path | None:
        p = find_cached(cam, sec)
        if p:
            return p
        vp = _video_path(dataset.repo_id, ep, cam)
        if not vp.exists():
            return None
        img = extract_frame_at(vp, sec)
        if img is None:
            return None
        out_dir = verification_dir(dataset.repo_id, annotator_model) / f"episode_{ep:06d}" / "_verifier_frames" / cam
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / f"t{sec:07.3f}.jpg"
        img.save(p, format="JPEG", quality=88)
        return p

    chosen: list[tuple[str, Path, dict | None]] = []

    # 1) Initial scene (head only) — for count + descriptor verification
    ff = find_cached(head_cam, 0.0) or cached_or_decode(head_cam, 0.0)
    if ff:
        descriptors = ann.get("vial_descriptors", [])
        desc_str = "; ".join(f"vial No.{d['vial_id']}={d['descriptor']}" for d in descriptors)
        chosen.append((
            f"[INITIAL SCENE @ t=0.00s, {head_cam}] "
            f"Annotation claims total_vials={ann.get('total_vials')} and descriptors: {desc_str}. "
            f"Count the vials and verify each descriptor matches.",
            ff, None,
        ))

    segs = ann.get("segments", [])

    def attach_multicam(i: int, seg: dict, sec: float) -> None:
        """For one timestamp, add HEAD + LEFT_WRIST + RIGHT_WRIST frames so the verifier
        can directly check arm/vial assignment."""
        claim = (f"segment {i}: annotation says \"{seg.get('instruction','')}\""
                 f"  (arm={seg.get('arm')}, prim={seg.get('primitive')}, "
                 f"vial_id={seg.get('vial_id')}, progress={seg.get('progress')})")
        h = cached_or_decode(head_cam, sec)
        if h:
            chosen.append((f"[HEAD @ t={sec:.2f}s — {claim}]", h, seg))
        if use_wrists:
            if left_cam:
                lw = cached_or_decode(left_cam, sec)
                if lw:
                    chosen.append((
                        f"[LEFT_WRIST @ t={sec:.2f}s — shows what the LEFT arm is doing. "
                        f"({claim})]", lw, seg,
                    ))
            if right_cam:
                rw = cached_or_decode(right_cam, sec)
                if rw:
                    chosen.append((
                        f"[RIGHT_WRIST @ t={sec:.2f}s — shows what the RIGHT arm is doing. "
                        f"({claim})]", rw, seg,
                    ))

    # 2) Send multicam triplets for every grasp/insert and suspicious object-action segment.
    #    Fake bimanual claims often appear as arm=both/vial_id=null approach/grasp before a
    #    single-arm transport, so prioritize those too.
    critical = [
        (i, s) for i, s in enumerate(segs)
        if s.get("primitive") in ("grasp", "insert")
        or (
            s.get("primitive") in ("approach", "transport", "align")
            and (s.get("arm") == "both" or s.get("vial_id") is None)
        )
    ]
    used_idxs: set[int] = set()
    for i, seg in critical:
        if len(chosen) >= max_frames:
            break
        sec = round((seg["start_s"] + seg["end_s"]) / 2, 3)
        attach_multicam(i, seg, sec)
        used_idxs.add(i)

    # 3) Fill remaining budget with head-only frames from un-shown segments (evenly spaced).
    remaining = [i for i in range(len(segs)) if i not in used_idxs]
    if remaining and len(chosen) < max_frames:
        budget = max_frames - len(chosen)
        step = max(1, len(remaining) // budget)
        for i in remaining[::step][:budget]:
            seg = segs[i]
            sec = round((seg["start_s"] + seg["end_s"]) / 2, 3)
            h = cached_or_decode(head_cam, sec)
            if h:
                claim = (f"segment {i}: annotation says \"{seg.get('instruction','')}\""
                         f"  (arm={seg.get('arm')}, prim={seg.get('primitive')})")
                chosen.append((f"[HEAD @ t={sec:.2f}s — {claim}]", h, seg))

    return chosen


def _build_user_text(ann: dict, task: Task | None) -> str:
    """The cleaned annotation JSON that the verifier reads, plus task context."""
    keep_top = ["total_vials", "stand_slots", "vial_descriptors", "pickup_sequence",
                "duration_s", "fps", "length"]
    cleaned = {k: ann[k] for k in keep_top if k in ann}
    keep_seg = ["start_s", "end_s", "arm", "primitive", "vial_id", "instruction", "progress"]
    cleaned["segments"] = [
        {k: s[k] for k in keep_seg if k in s}
        for s in ann.get("segments", [])
    ]
    primitives = ", ".join(task.primitives) if task else "unknown"
    default_task = task.default_task_string if task else ann.get("task", "")
    return (
        f"Task: {default_task}\n"
        f"Allowed primitives: {primitives}\n\n"
        f"Annotation JSON to review:\n{json.dumps(cleaned, indent=2)}\n\n"
        f"Now look at the {VERIFIER_MAX_FRAMES} camera frames below — each labeled with the "
        f"timestamp, camera, and the segment's claim — and flag any errors you can see in the frames."
    )


def verify_episode(dataset: LeRobotDataset, task: Task | None, ep: int,
                   annotator_model: str = DEFAULT_ANNOTATOR,
                   models: list[str] | None = None,
                   max_frames: int = VERIFIER_MAX_FRAMES) -> dict:
    """Run all configured verifier models on this episode's existing annotation."""
    ann_path = _annotation_path(dataset.repo_id, ep, annotator_model)
    if not ann_path.exists():
        raise FileNotFoundError(f"no annotation found at {ann_path}")
    ann = json.loads(ann_path.read_text())
    ann_fingerprint = _annotation_fingerprint(ann_path)

    images = _pick_frames(ann, dataset, ep, max_frames=max_frames,
                          annotator_model=annotator_model)
    user_text = _build_user_text(ann, task)
    models = models or VERIFIER_MODELS

    results: dict[str, dict] = {}
    for m in models:
        backend = verifiers.get(m)
        try:
            results[m] = backend(SYSTEM_PROMPT, user_text, [(lbl, p) for lbl, p, _ in images])
        except Exception as e:
            results[m] = {"model": m, "error": str(e), "parsed": {"overall": "error", "issues": []}}

    summary = {
        "repo_id": dataset.repo_id,
        "episode_index": ep,
        "annotator_model": annotator_model,
        "annotation_fingerprint": ann_fingerprint,
        "n_frames_shown": len(images),
        "verifiers": results,
        "overall_consensus": _consensus(results),
    }

    out = _verification_path(dataset.repo_id, ep, annotator_model)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    return summary


def _consensus(results: dict[str, dict]) -> dict:
    """Aggregate: any major issue from any verifier → 'issues_found'. Otherwise pass."""
    total_major = 0
    total_minor = 0
    for r in results.values():
        for issue in (r.get("parsed", {}).get("issues") or []):
            if issue.get("severity") == "major":
                total_major += 1
            else:
                total_minor += 1
    overall = "issues_found" if total_major > 0 else "pass"
    return {"overall": overall, "n_major": total_major, "n_minor": total_minor}


def load_verification(repo_id: str, ep: int, annotator_model: str = DEFAULT_ANNOTATOR) -> dict | None:
    p = _verification_path(repo_id, ep, annotator_model)
    if not p.exists():
        return None
    return json.loads(p.read_text())
