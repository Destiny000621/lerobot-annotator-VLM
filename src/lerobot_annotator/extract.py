"""Generic per-camera keyframe extraction from any LeRobot dataset's videos."""
from __future__ import annotations
from pathlib import Path
import urllib.request

import av
from PIL import Image

from .config import KEYFRAMES_CACHE, VIDEO_CACHE, repo_safe_name
from .dataset import LeRobotDataset, EpisodeInfo


def _download(url: str, dst: Path) -> Path:
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dst)
    return dst


def _video_path(repo_id: str, ep: int, cam: str) -> Path:
    return VIDEO_CACHE / repo_safe_name(repo_id) / f"ep{ep:06d}_{cam}.mp4"


def _keyframes_dir(repo_id: str, ep: int, cam: str, fps: float) -> Path:
    return KEYFRAMES_CACHE / repo_safe_name(repo_id) / f"episode_{ep:06d}" / f"{cam}_fps{fps:g}"


def _decode_at(video_path: Path, out_dir: Path, target_fps: float, native_fps: int) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    interval = 1.0 / target_fps
    written = []
    next_t = 0.0
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            t = float(frame.time)
            if t + 1e-6 < next_t:
                continue
            sec = round(next_t, 3)
            img: Image.Image = frame.to_image()
            path = out_dir / f"t{sec:07.3f}.jpg"
            img.save(path, format="JPEG", quality=85)
            written.append({"sec": sec, "frame_idx": round(sec * native_fps), "path": str(path)})
            next_t += interval
    return written


def extract_episode(
    dataset: LeRobotDataset,
    ep: int,
    cam_fps: dict[str, float],
) -> dict[str, list[dict]]:
    """Extract keyframes for one episode. `cam_fps` maps camera name -> desired Hz.
    Cameras not present in `cam_fps` are skipped."""
    info: EpisodeInfo = dataset.episode_info(ep)
    native_fps = dataset.fps()
    out: dict[str, list[dict]] = {}
    for cam, fps in cam_fps.items():
        if cam not in info.video_urls:
            continue
        vp = _video_path(dataset.repo_id, ep, cam)
        _download(info.video_urls[cam], vp)
        cam_dir = _keyframes_dir(dataset.repo_id, ep, cam, fps)
        cached = sorted(cam_dir.glob("t*.jpg"))
        if cached:
            out[cam] = [
                {"sec": float(p.stem[1:]), "frame_idx": round(float(p.stem[1:]) * native_fps), "path": str(p)}
                for p in cached
            ]
        else:
            out[cam] = _decode_at(vp, cam_dir, fps, native_fps)
    return out


def extract_frame_at(video_path: Path, target_sec: float) -> Image.Image | None:
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        ts = max(0, int(target_sec / stream.time_base))
        try:
            container.seek(ts, any_frame=False, backward=True, stream=stream)
        except av.AVError:
            container.seek(0, any_frame=False, backward=True, stream=stream)
        best = None
        for frame in container.decode(stream):
            t = float(frame.time)
            if t >= target_sec - 1e-3:
                return frame.to_image()
            best = frame
        return best.to_image() if best is not None else None
