"""Generic LeRobot output writer.

Takes a set of annotated episodes for a given dataset and produces a partial mirror
of the source repo where the affected per-episode parquets carry per-frame task_index
values that point into an expanded tasks.parquet. The default task string is preserved
at task_index=0 so unannotated episodes still resolve.
"""
from __future__ import annotations
import json
import urllib.request
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .config import ANNOTATIONS_DIR, OUTPUT_LEROBOT, repo_safe_name
from .dataset import LeRobotDataset


def _annotation_path(repo_id: str, ep: int) -> Path:
    return ANNOTATIONS_DIR / repo_safe_name(repo_id) / f"episode_{ep:06d}.json"


def _segments_to_per_frame(ann: dict, fps: int, fallback: str) -> list[str]:
    length = int(ann["length"])
    out: list[str | None] = [None] * length
    for seg in sorted(ann["segments"], key=lambda s: s["start_s"]):
        s = max(0, round(seg["start_s"] * fps))
        e = min(length, round(seg["end_s"] * fps))
        if e <= s:
            continue
        for i in range(s, e):
            out[i] = seg["instruction"]
    last = fallback
    for i in range(length):
        if out[i] is None:
            out[i] = last
        else:
            last = out[i]
    return out  # type: ignore[return-value]


def _build_tasks_table(annotations: list[dict], default_task: str) -> tuple[pa.Table, dict[str, int]]:
    ordered: list[str] = [default_task]
    seen = {default_task: 0}
    for ann in annotations:
        for seg in ann["segments"]:
            instr = (seg.get("instruction") or "").strip()
            if not instr:
                continue
            if instr not in seen:
                seen[instr] = len(ordered)
                ordered.append(instr)
    table = pa.table(
        {"task_index": list(range(len(ordered))), "task": ordered},
        schema=pa.schema([("task_index", pa.int64()), ("task", pa.large_string())]),
    )
    return table, seen


def _download_episode_parquet(dataset: LeRobotDataset, ep: int, dst: Path) -> Path:
    if dst.exists():
        return dst
    info = dataset.episode_info(ep)
    dst.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(info.data_url, dst)
    return dst


def materialize(
    dataset: LeRobotDataset,
    episodes: list[int],
    default_task: str | None = None,
) -> dict:
    out_root = OUTPUT_LEROBOT / repo_safe_name(dataset.repo_id)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "meta").mkdir(parents=True, exist_ok=True)

    fps = dataset.fps()
    default_task = default_task or dataset.default_task_string() or "task"

    annotations = []
    for ep in episodes:
        p = _annotation_path(dataset.repo_id, ep)
        if not p.exists():
            raise FileNotFoundError(f"no annotation for ep {ep}: {p}")
        annotations.append(json.loads(p.read_text()))

    tasks_table, instr_to_idx = _build_tasks_table(annotations, default_task)
    pq.write_table(tasks_table, out_root / "meta" / "tasks.parquet")

    # info.json with bumped total_tasks
    src_info = dataset.meta_dir / "info.json"
    info_d = json.loads(src_info.read_text())
    info_d["total_tasks"] = tasks_table.num_rows
    (out_root / "meta" / "info.json").write_text(json.dumps(info_d, indent=2))

    per_ep_instrs: dict[int, list[str]] = {}
    for ep, ann in zip(episodes, annotations):
        frame_instrs = _segments_to_per_frame(ann, fps, default_task)
        per_ep_instrs[ep] = list(dict.fromkeys(frame_instrs))

        src_data = dataset.meta_dir / "data_src" / f"file-{ep:03d}.parquet"
        _download_episode_parquet(dataset, ep, src_data)
        table = pq.read_table(src_data)
        df = table.to_pandas()
        if len(df) != len(frame_instrs):
            raise RuntimeError(
                f"length mismatch ep {ep}: parquet={len(df)} ann={len(frame_instrs)}"
            )
        df["task_index"] = [instr_to_idx[ins] for ins in frame_instrs]
        new_table = pa.Table.from_pandas(df, preserve_index=False).select(table.column_names)
        row = dataset._episodes_df().query("episode_index == @ep").iloc[0]
        chunk, fileno = int(row["data/chunk_index"]), int(row["data/file_index"])
        out_p = out_root / "data" / f"chunk-{chunk:03d}" / f"file-{fileno:03d}.parquet"
        out_p.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(new_table, out_p)

    # episodes.parquet — update `tasks` for the annotated episodes
    src_eps = dataset.meta_dir / "episodes" / "chunk-000" / "file-000.parquet"
    eps_table = pq.read_table(src_eps)
    eps_df = eps_table.to_pandas()
    eps_df["tasks"] = [
        list(dict.fromkeys(per_ep_instrs[int(r.episode_index)]))
        if int(r.episode_index) in per_ep_instrs else r["tasks"]
        for _, r in eps_df.iterrows()
    ]
    out_eps = out_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    out_eps.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(eps_df, preserve_index=False, schema=eps_table.schema), out_eps)

    return {
        "out_root": str(out_root),
        "n_tasks": tasks_table.num_rows,
        "n_annotated_episodes": len(episodes),
        "unique_instructions_per_ep": {ep: len(v) for ep, v in per_ep_instrs.items()},
    }
