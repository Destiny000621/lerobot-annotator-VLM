"""CLI entrypoint: annotate, materialize, serve."""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from lerobot_annotator.annotate import annotate_episode
from lerobot_annotator.apply_corrections import apply_episode as apply_episode_corrections
from lerobot_annotator.config import DEFAULT_ANNOTATOR
from lerobot_annotator.dataset import LeRobotDataset
from lerobot_annotator.manual_corrections import apply_pickup_sequence
from lerobot_annotator.materialize import materialize
from lerobot_annotator.overrides import clear as clear_override
from lerobot_annotator.task import load as load_task
from lerobot_annotator.verify import verify_episode


def _parse_episodes(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def _parse_episode_list(spec: str | None) -> list[int] | None:
    if spec is None:
        return None
    return [int(part.strip()) for part in spec.split(",") if part.strip()]


def cmd_annotate(args) -> None:
    ds = LeRobotDataset(args.repo)
    task = load_task(args.task)
    eps = _parse_episodes(args.episodes)
    exemplar_episodes = _parse_episode_list(args.exemplars)
    annotator_model = args.annotator or DEFAULT_ANNOTATOR
    args.annotator = annotator_model
    for ep in eps:
        if args.reset_overrides:
            clear_override(args.repo, ep)
            print(f"[ep{ep:06d}] overrides reset before annotate")
        t0 = time.time()
        try:
            ann = annotate_episode(ds, task, ep, overwrite=args.overwrite,
                                   use_exemplars=not args.no_exemplars,
                                   annotator_model=annotator_model,
                                   exemplar_episodes=exemplar_episodes)
            dt = time.time() - t0
            n_ex = len(ann.get("exemplars_used", []))
            print(f"[ep{ep:06d}] OK [{args.annotator}] segs={len(ann['segments'])} "
                  f"count={ann.get(task.count_vote.field if task.count_vote else '?')} "
                  f"exemplars={n_ex} attempts={len(ann.get('attempts', []))} ({dt:.1f}s)")
        except Exception as e:
            print(f"[ep{ep:06d}] FAIL: {e}")


def cmd_materialize(args) -> None:
    ds = LeRobotDataset(args.repo)
    eps = _parse_episodes(args.episodes)
    annotator_model = args.annotator or DEFAULT_ANNOTATOR
    summary = materialize(ds, eps, annotator_model=annotator_model)
    print(json.dumps(summary, indent=2))


def cmd_clear_overrides(args) -> None:
    eps = _parse_episodes(args.episodes)
    for ep in eps:
        ov = clear_override(args.repo, ep,
                            keep_status=args.keep_status,
                            keep_segments=args.keep_segments,
                            keep_fields=args.keep_fields,
                            keep_count=args.keep_count)
        print(f"[ep{ep:06d}] cleared (pinned_count={ov.pinned_count}, "
              f"pinned_fields={list(ov.pinned_fields or {})}, "
              f"pinned_segments={'set' if ov.pinned_segments else 'None'}, "
              f"status={ov.status})")


def cmd_apply_corrections(args) -> None:
    eps = _parse_episodes(args.episodes)
    annotator_model = args.annotator or DEFAULT_ANNOTATOR
    for ep in eps:
        try:
            r = apply_episode_corrections(args.repo, ep, annotator_model=annotator_model)
            if r.get("no_corrections"):
                a = r.get("applied", {}) or {}
                print(f"[ep{ep:06d}] no new corrections to apply"
                      + (f" (normalized {a['progress_denominators_normalized']} progress denom)"
                         if a.get("progress_denominators_normalized") else ""))
                inc = a.get("incomplete_progress")
                if inc:
                    print(f"           ⚠ progress {inc['final']} (expected {inc['expected']}) — {inc['hint']}")
                continue
            a = r["applied"]
            fields = [k for k in ("total_vials", "pickup_sequence", "vial_descriptors") if k in a]
            n_segs = len(a.get("segments", []))
            noops = a.get("skipped_noop_segment_edits", 0)
            prog = a.get("progress_denominators_rewritten", 0)
            extras = []
            if noops: extras.append(f"{noops} no-op edit(s) skipped")
            if prog:  extras.append(f"{prog} progress denominator(s) rewritten")
            tail = f" [{', '.join(extras)}]" if extras else ""
            print(f"[ep{ep:06d}] applied: {', '.join(fields) or '-'} | {n_segs} segment edit(s){tail}")
            if a.get("note"):
                print(f"           → {a['note']}")
            inc = a.get("incomplete_progress")
            if inc:
                print(f"           ⚠ progress {inc['final']} (expected {inc['expected']}) — {inc['hint']}")
        except Exception as e:
            print(f"[ep{ep:06d}] FAIL: {e}")


def cmd_verify(args) -> None:
    ds = LeRobotDataset(args.repo)
    task = load_task(args.task) if args.task else None
    eps = _parse_episodes(args.episodes)
    models = args.models.split(",") if args.models else None
    annotator_model = args.annotator or DEFAULT_ANNOTATOR
    for ep in eps:
        t0 = time.time()
        try:
            result = verify_episode(ds, task, ep, annotator_model=annotator_model, models=models)
            dt = time.time() - t0
            cons = result["overall_consensus"]
            print(f"[ep{ep:06d}] {cons['overall']:>13}  major={cons['n_major']} minor={cons['n_minor']} "
                  f"({len(result['verifiers'])} models, {dt:.1f}s)")
        except Exception as e:
            print(f"[ep{ep:06d}] FAIL: {e}")


def cmd_set_pickup_sequence(args) -> None:
    sequence = [int(x.strip()) for x in args.sequence.split(",") if x.strip()]
    annotator_model = args.annotator or DEFAULT_ANNOTATOR
    applied = apply_pickup_sequence(
        args.repo,
        args.episode,
        sequence,
        status=args.status,
        pin_segments=not args.no_pin_segments,
        annotator_model=annotator_model,
    )
    print(json.dumps(applied, indent=2))


def cmd_serve(args) -> None:
    from lerobot_annotator.webapp import create_app
    app = create_app(default_repo=args.repo, default_task=args.task)
    app.run(host="127.0.0.1", port=args.port, debug=args.debug)


def main() -> None:
    p = argparse.ArgumentParser(prog="lerobot-annotator")
    sub = p.add_subparsers(dest="cmd", required=True)

    ann = sub.add_parser("annotate", help="Run VLM annotation on episodes")
    ann.add_argument("--repo", required=True, help="HF dataset repo_id, e.g. user/dataset")
    ann.add_argument("--task", required=True, help="Task name (yaml/py in tasks/) or path")
    ann.add_argument("--episodes", required=True, help="e.g. '123-130' or '5,7,12'")
    ann.add_argument("--overwrite", action="store_true")
    ann.add_argument("--no-exemplars", action="store_true",
                     help="Disable few-shot exemplars from human-verified episodes")
    ann.add_argument("--exemplars", default=None,
                     help="Comma-separated exemplar episode IDs to use in this exact order, "
                          "e.g. '123,124,125'. Overrides automatic exemplar selection.")
    ann.add_argument("--annotator", default=None,
                     help="Annotator model ID (default: DEFAULT_ANNOTATOR in config.py). "
                          "Examples: gemini-robotics-er-1.6-preview, gpt-5.5, gpt-5.5-pro, gpt-4o")
    ann.add_argument("--reset-overrides", action="store_true",
                     help="Clear pinned_count/fields/segments BEFORE annotating. Use when "
                          "stale overrides from earlier runs are blocking a fresh count vote.")
    ann.set_defaults(func=cmd_annotate)

    mat = sub.add_parser("materialize", help="Emit LeRobot output tree from annotations")
    mat.add_argument("--repo", required=True)
    mat.add_argument("--episodes", required=True)
    mat.add_argument("--annotator", default=None,
                     help="Which annotator's outputs to materialize (default: DEFAULT_ANNOTATOR)")
    mat.set_defaults(func=cmd_materialize)

    apc = sub.add_parser("apply-corrections",
                          help="Apply structured corrections from AI verifier to annotations")
    apc.add_argument("--repo", required=True)
    apc.add_argument("--episodes", required=True)
    apc.add_argument("--annotator", default=None,
                     help="Which annotator's output to correct (default: DEFAULT_ANNOTATOR)")
    apc.set_defaults(func=cmd_apply_corrections)

    clr = sub.add_parser("clear-overrides",
                          help="Reset pinned values (count, fields, segments, status) on overrides")
    clr.add_argument("--repo", required=True)
    clr.add_argument("--episodes", required=True)
    clr.add_argument("--keep-status", action="store_true")
    clr.add_argument("--keep-segments", action="store_true")
    clr.add_argument("--keep-fields", action="store_true")
    clr.add_argument("--keep-count", action="store_true")
    clr.set_defaults(func=cmd_clear_overrides)

    ver = sub.add_parser("verify", help="Run AI verifier(s) over existing annotations")
    ver.add_argument("--repo", required=True)
    ver.add_argument("--task", default=None, help="Task name (for primitives list context)")
    ver.add_argument("--episodes", required=True)
    ver.add_argument("--annotator", default=None,
                     help="Which annotator's output to verify (default: DEFAULT_ANNOTATOR)")
    ver.add_argument("--models", default=None,
                     help="Comma-separated verifier model IDs (default: VERIFIER_MODELS)")
    ver.set_defaults(func=cmd_verify)

    seq = sub.add_parser("set-pickup-sequence",
                         help="Apply a human-provided vial pickup order and pin corrected segments")
    seq.add_argument("--repo", required=True)
    seq.add_argument("--episode", type=int, required=True)
    seq.add_argument("--sequence", required=True, help="Comma-separated vial IDs, e.g. 2,1,3")
    seq.add_argument("--status", choices=["draft", "needs_rework", "verified"], default=None,
                     help="Optional status to write to annotation and override")
    seq.add_argument("--no-pin-segments", action="store_true",
                     help="Only write pickup_sequence; do not pin rewritten segments")
    seq.add_argument("--annotator", default=None,
                     help="Which annotator's annotation to patch (default: DEFAULT_ANNOTATOR)")
    seq.set_defaults(func=cmd_set_pickup_sequence)

    srv = sub.add_parser("serve", help="Launch the HITL web app")
    srv.add_argument("--repo", default=None, help="Default repo_id to load")
    srv.add_argument("--task", default=None, help="Default task name")
    srv.add_argument("--port", type=int, default=5050)
    srv.add_argument("--debug", action="store_true")
    srv.set_defaults(func=cmd_serve)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
