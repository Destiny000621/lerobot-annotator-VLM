"""Generic validation for VLM-emitted JSON given a Task schema."""
from __future__ import annotations
import re
from typing import Any

from .task import Task


def validate(parsed: dict, task: Task, duration_s: float, pinned_count: int | None = None) -> list[str]:
    issues: list[str] = []

    # count constraint
    if task.count_vote and pinned_count is not None:
        field = task.count_vote.field
        got = parsed.get(field)
        if got != pinned_count:
            issues.append(f"{field}={got} but pinned count is {pinned_count}")

    # segments contiguous + cover duration
    segs = sorted(parsed.get("segments", []), key=lambda s: s.get("start_s", 0.0))
    if not segs:
        issues.append("no segments")
        return issues
    if segs[0]["start_s"] > 0.5:
        issues.append(f"first segment starts at {segs[0]['start_s']:.2f}s")
    if abs(segs[-1]["end_s"] - duration_s) > 1.0:
        issues.append(f"last segment ends at {segs[-1]['end_s']:.2f}s vs duration {duration_s:.2f}s")
    for a, b in zip(segs, segs[1:]):
        gap = b["start_s"] - a["end_s"]
        if gap > 0.5:
            issues.append(f"gap of {gap:.2f}s between {a['end_s']:.2f} and {b['start_s']:.2f}")
        if gap < -0.5:
            issues.append(f"overlap of {-gap:.2f}s between {a['end_s']:.2f} and {b['start_s']:.2f}")

    # primitives must be valid
    valid_prims = set(task.primitives)
    for s in segs:
        p = s.get("primitive")
        if p not in valid_prims:
            issues.append(f"segment uses unknown primitive '{p}' (valid: {sorted(valid_prims)})")
            break

    # task-declared per-segment field constraints
    for i, s in enumerate(segs):
        for field, spec in task.segment_extras.items():
            if not isinstance(spec, dict):
                continue
            typ = spec.get("type")
            val = s.get(field)
            if typ == "enum" and val not in set(spec.get("values", [])):
                issues.append(f"segment {i} field {field}={val!r} not in {spec.get('values', [])}")
            elif typ == "int_optional" and val is not None and not isinstance(val, int):
                issues.append(f"segment {i} field {field}={val!r} not int or null")

    object_prims = {"approach", "grasp", "transport", "align", "insert"}
    for i, s in enumerate(segs):
        if s.get("primitive") in object_prims and s.get("arm") in {"left", "right"}:
            if not isinstance(s.get("vial_id"), int):
                issues.append(
                    f"segment {i} {s.get('primitive')} with arm={s.get('arm')} must have integer vial_id"
                )

    # Catch fake bimanual pickup claims that immediately collapse to one arm/vial.
    for i, s in enumerate(segs[:-1]):
        if s.get("primitive") not in {"approach", "grasp"}:
            continue
        if s.get("arm") != "both" or s.get("vial_id") is not None:
            continue
        for j in range(i + 1, len(segs)):
            nxt = segs[j]
            if nxt.get("primitive") in {"idle", "retract"}:
                continue
            if nxt.get("primitive") in {"transport", "align", "insert"}:
                if nxt.get("arm") in {"left", "right"} and isinstance(nxt.get("vial_id"), int):
                    issues.append(
                        f"segment {i} claims bimanual {s.get('primitive')} with no vial_id, "
                        f"but segment {j} continues as {nxt.get('arm')}-arm {nxt.get('primitive')} "
                        f"of vial {nxt.get('vial_id')}"
                    )
                break

    # schema_extras quick checks: range-typed integer fields
    for field, spec in task.schema_extras.items():
        if isinstance(spec, dict) and spec.get("type") == "int" and "range" in spec:
            lo, hi = spec["range"]
            val = parsed.get(field)
            if not (isinstance(val, int) and lo <= val <= hi):
                issues.append(f"{field}={val} not int in [{lo},{hi}]")

    # generic vial-style coverage: if schema declares pickup_sequence + total_vials, check permutation
    if "pickup_sequence" in task.schema_extras and "total_vials" in task.schema_extras:
        n = parsed.get("total_vials")
        if isinstance(n, int):
            seq = parsed.get("pickup_sequence", [])
            flat: list[int] = []
            for item in seq:
                if isinstance(item, list):
                    flat.extend(item)
                else:
                    flat.append(item)
            if sorted(flat) != list(range(1, n + 1)):
                issues.append(f"pickup_sequence {seq} not a permutation of 1..{n}")

            grasp_order: list[int | list[int]] = []
            seen_grasps: set[int] = set()
            for s in segs:
                if s.get("primitive") != "grasp":
                    continue
                vid = s.get("vial_id")
                if isinstance(vid, int):
                    if vid not in seen_grasps:
                        grasp_order.append(vid)
                        seen_grasps.add(vid)
                elif s.get("arm") == "both":
                    ids = sorted({
                        int(m.group(1))
                        for m in re.finditer(r"vial\s*No\.?\s*(\d+)", s.get("instruction", ""), re.I)
                    })
                    ids = [v for v in ids if v not in seen_grasps]
                    if len(ids) >= 2:
                        grasp_order.append(ids)
                        seen_grasps.update(ids)
            if grasp_order:
                grasp_flat: list[int] = []
                for item in grasp_order:
                    if isinstance(item, list):
                        grasp_flat.extend(item)
                    else:
                        grasp_flat.append(item)
                if flat[:len(grasp_flat)] != grasp_flat:
                    issues.append(
                        f"pickup_sequence {seq} does not match first grasp order {grasp_order}"
                    )

            # every vial referenced in at least one of the named primitive segments (e.g. insert)
            referenced: set[int] = set()
            cover_prim = task.success_primitive
            for s in segs:
                if cover_prim and s.get("primitive") != cover_prim:
                    continue
                vid = s.get("vial_id")
                if isinstance(vid, int):
                    referenced.add(vid)
                for m in re.finditer(r"vial\s*No\.?\s*(\d+)", s.get("instruction", ""), re.I):
                    referenced.add(int(m.group(1)))
            missing = set(range(1, n + 1)) - referenced
            if missing and cover_prim:
                issues.append(f"vials never referenced in any '{cover_prim}' segment: {sorted(missing)}")

    return issues
