import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lerobot_annotator.annotate import _build_parts
from lerobot_annotator.task import load as load_task


def test_wrist_frames_are_labeled_not_count_evidence():
    task = load_task("vial_place")
    keyframes = {
        "head_camera": [{"sec": 0.0, "path": "head.jpg"}],
        "left_wrist_camera": [{"sec": 0.0, "path": "left.jpg"}],
        "right_wrist_camera": [{"sec": 0.0, "path": "right.jpg"}],
    }

    parts, sampling = _build_parts(task, keyframes, duration_s=1.0, fps=30, pinned_count=3)
    text = "\n".join(p["text"] for p in parts if "text" in p)

    assert sampling["other_cams"] == {"left_wrist_camera": 1, "right_wrist_camera": 1}
    assert "Do not use wrist-camera frames to count total vials" in text
    assert "arm/action evidence only, not count evidence" in text
