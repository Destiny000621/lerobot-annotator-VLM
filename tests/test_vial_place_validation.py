import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lerobot_annotator.task import load as load_task
from lerobot_annotator.validate import validate


def _ann(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def test_rejects_fake_bimanual_pickup_from_episode_123():
    task = load_task("vial_place")
    ann = _ann("annotations/Sichang0621__8ml_vial_place_30fps_fixed/episode_000124.json")
    ann = copy.deepcopy(ann)
    ann["pickup_sequence"] = [[1, 3], 2]
    ann["segments"][1]["arm"] = "both"
    ann["segments"][1]["vial_id"] = None
    ann["segments"][1]["instruction"] = "use both arms to approach the leftmost and rightmost vials"
    ann["segments"][2]["arm"] = "both"
    ann["segments"][2]["vial_id"] = None
    ann["segments"][2]["instruction"] = "use both arms to grasp the leftmost and rightmost vials"

    issues = validate(ann, task, ann["duration_s"], pinned_count=3)

    assert any("claims bimanual" in issue for issue in issues)
    assert any("pickup_sequence" in issue and "first grasp order" in issue for issue in issues)


def test_accepts_human_corrected_episode_124_shape():
    task = load_task("vial_place")
    ann = _ann("annotations/Sichang0621__8ml_vial_place_30fps_fixed/episode_000124.json")

    issues = validate(ann, task, ann["duration_s"], pinned_count=3)

    assert issues == []


def test_single_arm_object_action_requires_vial_id():
    task = load_task("vial_place")
    ann = _ann("annotations/Sichang0621__8ml_vial_place_30fps_fixed/episode_000124.json")
    bad = copy.deepcopy(ann)
    bad["segments"][1]["vial_id"] = None

    issues = validate(bad, task, bad["duration_s"], pinned_count=3)

    assert any("must have integer vial_id" in issue for issue in issues)
