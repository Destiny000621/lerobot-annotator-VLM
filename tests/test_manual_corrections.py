import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lerobot_annotator.manual_corrections import _progress_numerator, _rewrite_instruction


def test_manual_instruction_rewrite_uses_target_descriptor():
    ann = json.loads(
        (ROOT / "annotations/Sichang0621__8ml_vial_place_30fps_fixed/episode_000125.json").read_text()
    )
    seg = copy.deepcopy(ann["segments"][7])
    seg["vial_id"] = 3
    seg["instruction"] = "use the right arm to approach the right back vial (vial No.3)"

    rewritten = _rewrite_instruction(seg, "the left back vial", 1)

    assert rewritten == "use the right arm to approach the left back vial (vial No.1)"


def test_progress_numerator_reads_progress_prefix():
    assert _progress_numerator({"progress": "2/3"}) == 2
    assert _progress_numerator({"progress": "bad"}) is None
