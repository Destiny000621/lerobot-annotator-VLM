"""Generic LeRobot v3.0 dataset reader. Discovers cameras/fps/episode layout from meta/info.json."""
from __future__ import annotations
from dataclasses import dataclass, field
import json
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

from .config import META_CACHE, repo_safe_name


HF_RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/main"


@dataclass
class EpisodeInfo:
    episode_index: int
    length: int
    data_url: str
    dataset_from_index: int
    dataset_to_index: int
    video_urls: dict[str, str]
    video_spans: dict[str, tuple[float, float]]
    tasks: list[str]


@dataclass
class LeRobotDataset:
    repo_id: str
    info: dict = field(default_factory=dict)
    base_url: str = ""
    meta_dir: Path = field(default_factory=Path)

    def __post_init__(self) -> None:
        self.base_url = HF_RESOLVE.format(repo=self.repo_id)
        self.meta_dir = META_CACHE / repo_safe_name(self.repo_id)
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self.info = self._load_info()
        self._cameras_cache: list[str] | None = None
        self._eps_df = None

    # ---------- network ----------
    def _download(self, rel: str, dst: Path) -> Path:
        if dst.exists() and dst.stat().st_size > 0:
            return dst
        dst.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{self.base_url}/{rel}", dst)
        return dst

    def _load_info(self) -> dict:
        p = self.meta_dir / "info.json"
        self._download("meta/info.json", p)
        return json.loads(p.read_text())

    # ---------- discovery ----------
    def cameras(self) -> list[str]:
        if self._cameras_cache is not None:
            return self._cameras_cache
        out = []
        for name, feat in self.info.get("features", {}).items():
            if name.startswith("observation.images.") and feat.get("dtype") == "video":
                out.append(name.removeprefix("observation.images."))
        self._cameras_cache = out
        return out

    def fps(self) -> int:
        return int(self.info.get("fps", 30))

    def total_episodes(self) -> int:
        return int(self.info["total_episodes"])

    def default_task_string(self) -> str:
        p = self.meta_dir / "tasks.parquet"
        self._download("meta/tasks.parquet", p)
        df = pq.read_table(p).to_pandas().reset_index()
        if "task" in df.columns and len(df):
            return df.iloc[0]["task"]
        return ""

    # ---------- episodes ----------
    def _episodes_df(self):
        if self._eps_df is not None:
            return self._eps_df
        # LeRobot v3.0 may shard the episodes meta across chunks; assume chunk-000/file-000 for now.
        p = self.meta_dir / "episodes" / "chunk-000" / "file-000.parquet"
        self._download("meta/episodes/chunk-000/file-000.parquet", p)
        self._eps_df = pq.read_table(p).to_pandas()
        return self._eps_df

    def episode_info(self, ep: int) -> EpisodeInfo:
        df = self._episodes_df()
        row = df[df.episode_index == ep]
        if row.empty:
            raise ValueError(f"episode {ep} not found in {self.repo_id}")
        r = row.iloc[0]
        data_url = (
            f"{self.base_url}/data/chunk-{int(r['data/chunk_index']):03d}"
            f"/file-{int(r['data/file_index']):03d}.parquet"
        )
        video_urls = {}
        video_spans = {}
        for cam in self.cameras():
            key = f"observation.images.{cam}"
            chunk = int(r[f"videos/{key}/chunk_index"])
            fileno = int(r[f"videos/{key}/file_index"])
            video_urls[cam] = f"{self.base_url}/videos/{key}/chunk-{chunk:03d}/file-{fileno:03d}.mp4"
            video_spans[cam] = (
                float(r[f"videos/{key}/from_timestamp"]),
                float(r[f"videos/{key}/to_timestamp"]),
            )
        return EpisodeInfo(
            episode_index=int(r.episode_index),
            length=int(r.length),
            data_url=data_url,
            dataset_from_index=int(r.dataset_from_index),
            dataset_to_index=int(r.dataset_to_index),
            video_urls=video_urls,
            video_spans=video_spans,
            tasks=list(r.tasks) if r.tasks is not None else [],
        )
