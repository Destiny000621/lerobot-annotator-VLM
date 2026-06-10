"""Flask HITL verifier. Single localhost app for browsing & editing annotations."""
from __future__ import annotations
import json
import threading
from pathlib import Path

from flask import Flask, jsonify, redirect, request, send_file, send_from_directory, url_for, render_template

from .config import (
    ANNOTATIONS_DIR, DEFAULT_ANNOTATOR, KEYFRAMES_CACHE, OVERRIDES_DIR,
    SUCCESS_STATES_DIR, VIDEO_CACHE,
    annotation_dir, list_annotators_for_repo, model_safe_name, repo_safe_name,
    success_states_dir,
)
from .dataset import LeRobotDataset
from .annotate import annotate_episode
from .extract import _video_path
from .apply_corrections import apply_episode as apply_episode_corrections
from .overrides import (
    Override, clear as clear_override, load as load_override, save as save_override,
    sync_human_verified,
)
from .task import load as load_task
from .verify import is_verification_stale, load_verification, verify_episode


_RUNNING_RERUNS: dict[str, threading.Thread] = {}


def _annotation_path(repo_id: str, ep: int, annotator_model: str) -> Path:
    return annotation_dir(repo_id, annotator_model) / f"episode_{ep:06d}.json"


def _list_annotated_episodes(repo_id: str, annotator_model: str) -> list[int]:
    d = annotation_dir(repo_id, annotator_model)
    if not d.exists():
        return []
    eps = []
    for p in d.glob("episode_*.json"):
        try:
            eps.append(int(p.stem.split("_")[1]))
        except Exception:
            continue
    return sorted(eps)


def _resolve_annotator(repo_id: str, requested: str | None) -> str:
    """Pick which annotator subdir to show.

    Order: explicit > Referer URL's ?annotator= > DEFAULT_ANNOTATOR (if its
    subdir exists for this repo) > first discovered subdir > DEFAULT_ANNOTATOR.

    Preferring DEFAULT_ANNOTATOR over the alphabetical-first subdir keeps the
    behavior stable when new annotator subdirs get added — without this,
    adding `claude-opus-4-7/` would silently make every annotator-less POST
    misroute to Claude even though the user was viewing a different model.
    """
    if requested:
        return requested
    # Try to recover the annotator from the page that issued the POST.
    try:
        from urllib.parse import urlparse, parse_qs
        from flask import request as _req
        ref = getattr(_req, "referrer", None)
        if ref:
            qs = parse_qs(urlparse(ref).query)
            ref_annotator = qs.get("annotator", [None])[0]
            if ref_annotator:
                return ref_annotator
    except Exception:
        pass
    candidates = list_annotators_for_repo(repo_id)
    if DEFAULT_ANNOTATOR in candidates:
        return DEFAULT_ANNOTATOR
    if candidates:
        return candidates[0]
    return DEFAULT_ANNOTATOR


def create_app(default_repo: str | None = None, default_task: str | None = None) -> Flask:
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).resolve().parents[2] / "templates"),
        static_folder=str(Path(__file__).resolve().parents[2] / "static"),
    )
    app.config["DEFAULT_REPO"] = default_repo
    app.config["DEFAULT_TASK"] = default_task

    # ---- routes ----
    @app.route("/")
    def index():
        repo = request.args.get("repo") or app.config["DEFAULT_REPO"]
        task_name = request.args.get("task") or app.config["DEFAULT_TASK"]
        requested_annotator = request.args.get("annotator")
        annotator_model = _resolve_annotator(repo, requested_annotator) if repo else DEFAULT_ANNOTATOR
        annotators_available = list_annotators_for_repo(repo) if repo else []
        episodes = []
        n_verified = 0
        if repo:
            for ep in _list_annotated_episodes(repo, annotator_model):
                ann = json.loads(_annotation_path(repo, ep, annotator_model).read_text())
                ov = load_override(repo, ep)
                if ov.status == "verified":
                    n_verified += 1
                loaded_verification = load_verification(repo, ep, annotator_model)
                stale = bool(loaded_verification) and is_verification_stale(repo, ep, annotator_model)
                v = {} if stale else (loaded_verification or {})
                consensus = v.get("overall_consensus") or {}
                episodes.append({
                    "ep": ep,
                    "n_segments": len(ann.get("segments", [])),
                    "count_field": (ann.get("count_resolution") or {}).get("count"),
                    "status": ov.status,
                    "duration_s": round(ann.get("duration_s", 0), 1),
                    "attempts": len(ann.get("attempts", [])),
                    "n_exemplars": len(ann.get("exemplars_used", [])),
                    "verifier_overall": consensus.get("overall"),
                    "verifier_major": consensus.get("n_major", 0),
                    "verifier_minor": consensus.get("n_minor", 0),
                    "verifier_stale": stale,
                })
        return render_template("index.html",
                               repo=repo, task=task_name,
                               annotator=annotator_model,
                               annotators_available=annotators_available,
                               episodes=episodes,
                               n_verified=n_verified,
                               rerunning=list(_RUNNING_RERUNS.keys()))

    @app.route("/episode/<int:ep>")
    def episode(ep: int):
        repo = request.args.get("repo") or app.config["DEFAULT_REPO"]
        task_name = request.args.get("task") or app.config["DEFAULT_TASK"]
        requested_annotator = request.args.get("annotator")
        annotator_model = _resolve_annotator(repo, requested_annotator) if repo else DEFAULT_ANNOTATOR
        annotators_available = list_annotators_for_repo(repo) if repo else []
        if not repo:
            return "missing ?repo=", 400
        ann = json.loads(_annotation_path(repo, ep, annotator_model).read_text())
        ov = load_override(repo, ep)
        try:
            cams = LeRobotDataset(repo).cameras()
            head_cam = next((c for c in cams if "head" in c.lower()), cams[0]) if cams else None
            wrist_cams = [c for c in cams if c != head_cam]
        except Exception:
            head_cam, wrist_cams = None, []
        loaded_verification = load_verification(repo, ep, annotator_model)
        verification = loaded_verification or {}
        verification_stale = bool(loaded_verification) and is_verification_stale(repo, ep, annotator_model)
        if verification_stale:
            verification = {**verification, "stale": True}
        # Build map: segment_index -> list of issues for inline display
        seg_issues: dict[int, list[dict]] = {}
        episode_issues: list[dict] = []
        if not verification_stale:
            for model_id, r in (verification.get("verifiers") or {}).items():
                for issue in (r.get("parsed") or {}).get("issues") or []:
                    issue = {**issue, "model": model_id}
                    si = issue.get("segment_index")
                    if isinstance(si, int):
                        seg_issues.setdefault(si, []).append(issue)
                    else:
                        episode_issues.append(issue)
        return render_template("episode.html",
                               repo=repo, task=task_name, ep=ep,
                               annotator=annotator_model,
                               annotators_available=annotators_available,
                               ann=ann, override=ov.to_dict(),
                               head_cam=head_cam, wrist_cams=wrist_cams,
                               verification=verification,
                               seg_issues=seg_issues,
                               episode_issues=episode_issues)

    @app.route("/api/episode/<int:ep>/save", methods=["POST"])
    def api_save(ep: int):
        body = request.json or {}
        repo = body.get("repo") or app.config["DEFAULT_REPO"]
        annotator_model = _resolve_annotator(repo, body.get("annotator"))
        ann_path = _annotation_path(repo, ep, annotator_model)
        ann = json.loads(ann_path.read_text())

        # Update segments inline if provided
        if "segments" in body:
            new_segs = body["segments"]
            old_segs = ann.get("segments") or []
            # Belt-and-braces: if the client didn't echo back success_frame_path /
            # success_frame_index, recover them from the on-disk segments by
            # position so the success-state thumbnails don't go blank after a save.
            for i, seg in enumerate(new_segs):
                if i >= len(old_segs):
                    break
                old = old_segs[i]
                for f in ("success_frame_path", "success_frame_index"):
                    if f not in seg and f in old:
                        seg[f] = old[f]
            ann["segments"] = new_segs

        # Update top-level fields (count, sequence, etc.)
        for k in ("total_vials", "stand_slots", "vial_descriptors", "pickup_sequence"):
            if k in body:
                ann[k] = body[k]

        ann_path.write_text(json.dumps(ann, indent=2))

        # Persist as override too so re-annotation respects edits
        ov = load_override(repo, ep)
        if "total_vials" in body and isinstance(body["total_vials"], int):
            ov.pinned_count = body["total_vials"]
        pinned_fields = {}
        for k in ("stand_slots", "vial_descriptors", "pickup_sequence"):
            if k in body:
                pinned_fields[k] = body[k]
        if pinned_fields:
            ov.pinned_fields = {**(ov.pinned_fields or {}), **pinned_fields}
        if "segments" in body:
            ov.pinned_segments = body["segments"]
        if "status" in body:
            ov.status = body["status"]
        save_override(ov)
        if "status" in body:
            sync_human_verified(repo, ep, annotator_model, ov.status)
        return jsonify({"ok": True, "status": ov.status})

    @app.route("/api/episode/<int:ep>/approve", methods=["POST"])
    def api_approve(ep: int):
        body = request.json or {}
        repo = body.get("repo") or app.config["DEFAULT_REPO"]
        annotator_model = _resolve_annotator(repo, body.get("annotator"))
        ov = load_override(repo, ep)
        ov.status = "verified"
        save_override(ov)
        ann_path = _annotation_path(repo, ep, annotator_model)
        ann = json.loads(ann_path.read_text())
        ann["status"] = "verified"
        ann_path.write_text(json.dumps(ann, indent=2))
        sync_human_verified(repo, ep, annotator_model, "verified")
        return jsonify({"ok": True})

    @app.route("/api/episode/<int:ep>/reject", methods=["POST"])
    def api_reject(ep: int):
        body = request.json or {}
        repo = body.get("repo") or app.config["DEFAULT_REPO"]
        annotator_model = _resolve_annotator(repo, body.get("annotator"))
        ov = load_override(repo, ep)
        ov.status = "needs_rework"
        save_override(ov)
        sync_human_verified(repo, ep, annotator_model, "needs_rework")
        return jsonify({"ok": True})

    @app.route("/api/episode/<int:ep>/rerun", methods=["POST"])
    def api_rerun(ep: int):
        body = request.json or {}
        repo = body.get("repo") or app.config["DEFAULT_REPO"]
        task_name = body.get("task") or app.config["DEFAULT_TASK"]
        annotator_model = _resolve_annotator(repo, body.get("annotator"))
        key = f"{repo}:{ep}:{annotator_model}"
        if key in _RUNNING_RERUNS:
            return jsonify({"ok": False, "error": "already running"}), 409

        def _work():
            import time as _t
            t0 = _t.time()
            try:
                ds = LeRobotDataset(repo)
                task = load_task(task_name)
                annotate_episode(ds, task, ep, overwrite=True, annotator_model=annotator_model)
                print(f"[rerun ep{ep} {annotator_model}] DONE ({_t.time()-t0:.1f}s)", flush=True)
            except Exception as e:
                print(f"[rerun ep{ep} {annotator_model}] FAIL ({_t.time()-t0:.1f}s): {e}", flush=True)
            finally:
                _RUNNING_RERUNS.pop(key, None)
        t = threading.Thread(target=_work, daemon=True)
        _RUNNING_RERUNS[key] = t
        t.start()
        return jsonify({"ok": True})

    @app.route("/api/episode/<int:ep>/status")
    def api_status(ep: int):
        repo = request.args.get("repo") or app.config["DEFAULT_REPO"]
        annotator_model = _resolve_annotator(repo, request.args.get("annotator"))
        rerun_key = f"{repo}:{ep}:{annotator_model}"
        verify_key = f"verify:{repo}:{ep}:{annotator_model}"
        return jsonify({"rerunning": rerun_key in _RUNNING_RERUNS,
                        "verifying": verify_key in _RUNNING_RERUNS,
                        "annotator": annotator_model})

    @app.route("/api/episode/<int:ep>/reset_overrides", methods=["POST"])
    def api_reset_overrides(ep: int):
        body = request.json or {}
        repo = body.get("repo") or app.config["DEFAULT_REPO"]
        try:
            ov = clear_override(repo, ep)
            return jsonify({"ok": True, "status": ov.status,
                            "pinned_count": ov.pinned_count})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/episode/<int:ep>/apply_corrections", methods=["POST"])
    def api_apply_corrections(ep: int):
        body = request.json or {}
        repo = body.get("repo") or app.config["DEFAULT_REPO"]
        annotator_model = _resolve_annotator(repo, body.get("annotator"))
        try:
            r = apply_episode_corrections(repo, ep, annotator_model=annotator_model)
            return jsonify({"ok": True, **r})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/episode/<int:ep>/verify", methods=["POST"])
    def api_verify(ep: int):
        body = request.json or {}
        repo = body.get("repo") or app.config["DEFAULT_REPO"]
        task_name = body.get("task") or app.config["DEFAULT_TASK"]
        annotator_model = _resolve_annotator(repo, body.get("annotator"))
        key = f"verify:{repo}:{ep}:{annotator_model}"
        if key in _RUNNING_RERUNS:
            return jsonify({"ok": False, "error": "already running"}), 409

        def _work():
            import time as _t
            t0 = _t.time()
            try:
                ds = LeRobotDataset(repo)
                task = load_task(task_name) if task_name else None
                verify_episode(ds, task, ep, annotator_model=annotator_model)
                print(f"[verify ep{ep} {annotator_model}] DONE ({_t.time()-t0:.1f}s)", flush=True)
            except Exception as e:
                print(f"[verify ep{ep} {annotator_model}] FAIL ({_t.time()-t0:.1f}s): {e}", flush=True)
            finally:
                _RUNNING_RERUNS.pop(key, None)
        t = threading.Thread(target=_work, daemon=True)
        _RUNNING_RERUNS[key] = t
        t.start()
        return jsonify({"ok": True})

    # ---- image serving ----
    @app.route("/img/keyframe/<repo_safe>/<int:ep>/<cam>/<filename>")
    def keyframe(repo_safe: str, ep: int, cam: str, filename: str):
        ep_dir = KEYFRAMES_CACHE / repo_safe / f"episode_{ep:06d}"
        # cam may be "head_camera_fps2" or just "head_camera" — try exact then prefix-match
        d = ep_dir / cam
        if not d.exists():
            for cand in ep_dir.glob(f"{cam}_fps*"):
                d = cand
                break
        return send_from_directory(d, filename)

    @app.route("/img/success/<repo_safe>/<annotator>/<int:ep>/<filename>")
    def success(repo_safe: str, annotator: str, ep: int, filename: str):
        d = SUCCESS_STATES_DIR / repo_safe / annotator / f"episode_{ep:06d}"
        return send_from_directory(d, filename)

    @app.route("/video/<repo_safe>/<int:ep>/<cam>.mp4")
    def video(repo_safe: str, ep: int, cam: str):
        path = VIDEO_CACHE / repo_safe / f"ep{ep:06d}_{cam}.mp4"
        if not path.exists():
            # download lazily
            from .dataset import LeRobotDataset
            from .extract import _download
            repo_id = repo_safe.replace("__", "/", 1)
            ds = LeRobotDataset(repo_id)
            url = ds.episode_info(ep).video_urls.get(cam)
            if not url:
                return f"camera '{cam}' not in dataset", 404
            _download(url, path)
        # send_file handles HTTP Range automatically; required for video seeking
        return send_file(path, mimetype="video/mp4", conditional=True)

    @app.route("/img/frame_at/<repo_safe>/<int:ep>/<float:sec>.jpg")
    def frame_at(repo_safe: str, ep: int, sec: float):
        from .extract import extract_frame_at
        from .dataset import LeRobotDataset
        # Need repo_id; reconstruct from the safe name (repo_safe has __ where / was)
        repo_id = repo_safe.replace("__", "/", 1)
        ds = LeRobotDataset(repo_id)
        head_cam = next((c for c in ds.cameras() if "head" in c.lower()), ds.cameras()[0])
        video = _video_path(repo_id, ep, head_cam)
        img = extract_frame_at(video, sec)
        if img is None:
            return "no frame", 404
        out = SUCCESS_STATES_DIR / repo_safe / f"episode_{ep:06d}" / f"_seek_{sec:.2f}.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        img.save(out, format="JPEG", quality=85)
        return send_from_directory(out.parent, out.name)

    return app
