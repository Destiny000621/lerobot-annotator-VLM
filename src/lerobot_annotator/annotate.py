"""Generic annotation orchestrator. Task-agnostic; pipeline is driven by the Task config."""
from __future__ import annotations
import json
from pathlib import Path

import av

from . import annotators
from .config import (
    COUNT_VOTE_SAMPLES, DEFAULT_ANNOTATOR, MAX_ANNOTATE_ATTEMPTS,
    MAX_IMAGES_PER_CALL, SUCCESS_STATES_DIR, VIDEO_CACHE,
    annotation_dir, success_states_dir,
)
from .dataset import LeRobotDataset
from .exemplars import build_exemplar_parts
from .extract import extract_episode, extract_frame_at, _video_path
from .gemini import subsample_evenly
from .overrides import Override, apply_post_call, load as load_override, save as save_override
from .task import Task
from .validate import validate


def _annotation_path(repo_id: str, ep: int, annotator_model: str) -> Path:
    return annotation_dir(repo_id, annotator_model) / f"episode_{ep:06d}.json"


def _success_dir(repo_id: str, ep: int, annotator_model: str) -> Path:
    return success_states_dir(repo_id, annotator_model) / f"episode_{ep:06d}"


def _build_parts(task: Task, keyframes: dict[str, list[dict]], duration_s: float, fps: int,
                 pinned_count: int | None, image_budget: int = MAX_IMAGES_PER_CALL) -> tuple[list[dict], dict]:
    """Interleave (HEAD, then other cams) at each timestamp, subject to image_budget."""
    cams = list(keyframes.keys())
    head_cam = next((c for c in cams if "head" in c.lower()), cams[0]) if cams else None
    others = [c for c in cams if c != head_cam]

    head = keyframes.get(head_cam, []) if head_cam else []
    if len(head) > image_budget:
        head = subsample_evenly(head, image_budget)
    head_secs = {kf["sec"] for kf in head}

    other_budget = max(0, image_budget - len(head)) // max(1, len(others))
    other_secs_by_cam: dict[str, set[float]] = {}
    for cam in others:
        kfs = keyframes.get(cam, [])
        picked = subsample_evenly(
            [kf for kf in kfs if kf["sec"] in head_secs] or kfs,
            other_budget,
        )
        other_secs_by_cam[cam] = {kf["sec"] for kf in picked}

    sampling = {
        "head_camera": head_cam,
        "n_head": len(head),
        "other_cams": {cam: len(other_secs_by_cam.get(cam, set())) for cam in others},
        "total_images": len(head) + sum(len(s) for s in other_secs_by_cam.values()),
    }

    intro = (
        f"Episode duration: {duration_s:.2f}s @ {fps} fps. "
        f"Default task: '{task.default_task_string}'. "
        f"HEAD keyframes from camera '{head_cam}'. Output STRICT JSON only matching the schema."
    )
    if pinned_count is not None and task.count_vote:
        intro += (
            f"\nHARD CONSTRAINT: {task.count_vote.field} MUST equal {pinned_count} "
            f"(independently verified from head-camera count vote). "
            "Do not change this count based on wrist-camera frames."
        )
    if others:
        intro += (
            "\nNon-head camera frames are for arm/action disambiguation only. "
            "Do not use wrist-camera frames to count total vials."
        )

    parts: list[dict] = [{"text": intro}]
    by_sec = {kf["sec"]: kf for kf in head}
    for sec in sorted(by_sec):
        parts.append({"text": f"--- t = {sec:.2f}s --- {head_cam.upper()}:"})
        parts.append({"image": by_sec[sec]["path"]})
        for cam in others:
            if sec in other_secs_by_cam.get(cam, set()):
                kf_other = next((k for k in keyframes[cam] if k["sec"] == sec), None)
                if kf_other:
                    parts.append({"text": f"{cam.upper()} (arm/action evidence only, not count evidence):"})
                    parts.append({"image": kf_other["path"]})
    parts.append({"text": (
        f"Now output the JSON. Contiguous segments from 0.0s to {duration_s:.2f}s. "
        "Strict JSON, no markdown fences."
    )})
    return parts, sampling


def _save_success_states(parsed: dict, dataset: LeRobotDataset, ep: int, task: Task,
                          annotator_model: str) -> None:
    if not task.success_primitive:
        return
    out_dir = _success_dir(dataset.repo_id, ep, annotator_model)
    if out_dir.exists():
        for old in out_dir.glob("*.jpg"):
            old.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    head_cam = next((c for c in dataset.cameras() if "head" in c.lower()), None) or dataset.cameras()[0]
    video = _video_path(dataset.repo_id, ep, head_cam)
    if not video.exists():
        return
    fps = dataset.fps()
    for i, seg in enumerate(parsed.get("segments", [])):
        if seg.get("primitive") != task.success_primitive:
            continue
        end_frame = round(seg["end_s"] * fps)
        target_sec = max(0.0, (end_frame - 1) / fps)
        img = extract_frame_at(video, target_sec)
        if img is None:
            continue
        vid = seg.get("vial_id")
        name = f"seg{i:02d}_{task.success_primitive}_{('vial' + str(vid)) if vid else 'main'}.jpg"
        path = out_dir / name
        img.save(path, format="JPEG", quality=88)
        seg["success_frame_index"] = end_frame - 1
        seg["success_frame_path"] = str(path.relative_to(SUCCESS_STATES_DIR.parent))


def annotate_episode(
    dataset: LeRobotDataset,
    task: Task,
    episode_index: int,
    overwrite: bool = False,
    use_exemplars: bool = True,
    annotator_model: str = DEFAULT_ANNOTATOR,
    exemplar_episodes: list[int] | None = None,
) -> dict:
    """Run the full annotation pipeline for one episode. Idempotent unless overwrite=True."""
    out_path = _annotation_path(dataset.repo_id, episode_index, annotator_model)
    if out_path.exists() and not overwrite:
        return json.loads(out_path.read_text())

    info = dataset.episode_info(episode_index)
    duration_s = info.length / dataset.fps()
    keyframes = extract_episode(dataset, episode_index, task.cam_fps)
    head_cam = next((c for c in keyframes if "head" in c.lower()), next(iter(keyframes), None))
    if head_cam is None:
        raise RuntimeError(f"no head camera resolved for {dataset.repo_id}")

    backend = annotators.get(annotator_model)
    override = load_override(dataset.repo_id, episode_index)

    # ---- pinned count: from override > explicit exemplar consensus > count-vote ----
    pinned_count = override.pinned_count
    count_info = {"source": "override" if pinned_count is not None else None,
                  "count": pinned_count, "model": annotator_model}
    if pinned_count is None and task.count_vote and exemplar_episodes:
        # User explicitly passed exemplars — if they unanimously agree on the count
        # field, trust them and skip the vote. Saves ~30s/sample of model time and
        # avoids cases where the model misreads the stand (e.g. ep 127 where
        # gpt-5.5 voted [2,4,4,4,4] for what is actually 3 vials).
        from .exemplars import load_cleaned_annotation
        field = task.count_vote.field
        exemplar_counts = []
        seen_counts: dict[int, list[int]] = {}
        for xep in exemplar_episodes:
            if xep == episode_index:
                continue
            xann = load_cleaned_annotation(dataset.repo_id, xep, preferred_annotator=annotator_model)
            if xann is None:
                continue
            xc = xann.get(field)
            if isinstance(xc, int):
                exemplar_counts.append(xc)
                seen_counts.setdefault(xc, []).append(xep)
        if exemplar_counts and len(set(exemplar_counts)) == 1:
            pinned_count = exemplar_counts[0]
            count_info = {
                "source": "exemplar_consensus",
                "count": pinned_count,
                "exemplars": {str(k): v for k, v in seen_counts.items()},
                "model": annotator_model,
            }
    if pinned_count is None and task.count_vote:
        head_kfs = keyframes[head_cam]
        count_images: list[tuple[str, str]] = [("[FIRST_FRAME — initial scene]", head_kfs[0]["path"])]
        if len(head_kfs) > 1:
            count_images.append(("[FINAL_FRAME — task end state]", head_kfs[-1]["path"]))
        pinned_count, votes = backend.count_vote(
            count_images,
            task.count_vote.question,
            task.count_vote.range[0], task.count_vote.range[1],
            n_votes=COUNT_VOTE_SAMPLES,
            prefer_final_frame=task.count_vote.prefer_final_frame,
        )
        count_info = {"source": "majority_vote", "count": pinned_count,
                      "votes": votes, "model": annotator_model,
                      "frames_used": len(count_images)}

    # ---- exemplars from human-verified episodes ----
    exemplar_parts: list[dict] = []
    exemplar_meta: list[dict] = []
    if use_exemplars:
        exemplar_parts, exemplar_meta = build_exemplar_parts(
            repo_id=dataset.repo_id, head_cam=head_cam, exclude_episode=episode_index,
            target_count=pinned_count, explicit_episodes=exemplar_episodes,
            preferred_annotator=annotator_model,
        )

    # Each exemplar contributes one optional first-frame image plus text-only JSON. Keep
    # enough slots for the current episode even when all verified examples are included.
    exemplar_cost = len(exemplar_meta)
    main_budget = max(20, MAX_IMAGES_PER_CALL - exemplar_cost)

    parts, sampling = _build_parts(task, keyframes, duration_s, dataset.fps(), pinned_count,
                                   image_budget=main_budget)
    sampling["n_exemplars"] = len(exemplar_meta)
    sampling["main_image_budget"] = main_budget
    if exemplar_parts:
        parts = exemplar_parts + parts
    system_text = task.render_prompt(
        task_string=task.default_task_string,
        primitives=", ".join(task.primitives),
        count_field=(task.count_vote.field if task.count_vote else ""),
        pinned_count=pinned_count or "",
    )

    parsed, raw_text, attempts, usage = backend.generate_with_retry(
        system_text=system_text,
        neutral_parts=parts,
        validate_fn=lambda p: validate(p, task, duration_s, pinned_count),
        max_attempts=MAX_ANNOTATE_ATTEMPTS,
    )

    parsed = apply_post_call(parsed, override)
    parsed["repo_id"] = dataset.repo_id
    parsed["episode_index"] = episode_index
    parsed["length"] = info.length
    parsed["duration_s"] = duration_s
    parsed["fps"] = dataset.fps()
    parsed["task"] = task.name
    parsed["annotator_model"] = annotator_model
    parsed["sampling"] = sampling
    parsed["count_resolution"] = count_info
    parsed["exemplars_used"] = exemplar_meta
    parsed["attempts"] = attempts
    parsed["usage"] = usage
    parsed["status"] = override.status
    parsed["_raw_text"] = raw_text

    _save_success_states(parsed, dataset, episode_index, task, annotator_model)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(parsed, indent=2))
    return parsed
