import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lerobot_annotator.exemplars import build_exemplar_parts, find_verified


def test_episode_124_is_preferred_before_other_verified_episodes():
    repo = "Sichang0621/8ml_vial_place_30fps_fixed"
    ep123 = ROOT / "overrides/Sichang0621__8ml_vial_place_30fps_fixed/episode_000123.json"
    assert json.loads(ep123.read_text())["status"] == "verified"

    verified = find_verified(repo)

    assert verified[0] == 124
    assert 123 in verified


def test_exemplars_prefer_matching_count():
    repo = "Sichang0621/8ml_vial_place_30fps_fixed"

    _parts, meta = build_exemplar_parts(
        repo_id=repo,
        head_cam="head_camera",
        exclude_episode=126,
        include_images=False,
        target_count=3,
    )

    episodes = [m["episode"] for m in meta]
    assert episodes[:3] == [124, 123, 125]
    assert {0, 1, 2, 3, 4, 5}.issubset(episodes)


def test_explicit_exemplars_use_command_line_order_and_skip_current_episode():
    repo = "Sichang0621/8ml_vial_place_30fps_fixed"

    _parts, meta = build_exemplar_parts(
        repo_id=repo,
        head_cam="head_camera",
        exclude_episode=124,
        include_images=False,
        explicit_episodes=[123, 124, 125],
    )

    assert [m["episode"] for m in meta] == [123, 125]
