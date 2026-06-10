"""Global paths and gateway config."""
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

META_CACHE = ROOT / "meta_cache"
VIDEO_CACHE = ROOT / "video_cache"
KEYFRAMES_CACHE = ROOT / "keyframes_cache"
ANNOTATIONS_DIR = ROOT / "annotations"
OVERRIDES_DIR = ROOT / "overrides"
SUCCESS_STATES_DIR = ROOT / "success_states"
OUTPUT_LEROBOT = ROOT / "output_lerobot"
TASKS_DIR = ROOT / "tasks"

LITELLM_BASE = "https://litellm.avantrobotics.ai"
GEMINI_MODEL = "gemini-robotics-er-1.6-preview"
GEMINI_URL = f"{LITELLM_BASE}/v1beta/models/{GEMINI_MODEL}:generateContent"

# Default model used by `cli.py annotate`. Can be any model registered in
# src/lerobot_annotator/annotators/. Override per-call with --annotator.
DEFAULT_ANNOTATOR = "gemini-robotics-er-1.6-preview"

MAX_IMAGES_PER_CALL = 120
MAX_EDGE_PX = 384
MAX_ANNOTATE_ATTEMPTS = 4
COUNT_VOTE_SAMPLES = 5

MAX_EXEMPLARS = None              # None = include all verified episodes as exemplars
EXEMPLAR_INCLUDE_FIRST_FRAME = True  # send the exemplar's t=0 head image
# Repo-specific verified episodes to prefer as few-shot exemplars. Episode 124
# was manually corrected in the web UI and should anchor the vial pilot style.
PREFERRED_EXEMPLARS = {
    "Sichang0621/8ml_vial_place_30fps_fixed": [124],
}

FLASK_PORT = 5050

VERIFICATIONS_DIR = ROOT / "verifications"
SECRETS_DIR = ROOT / ".secrets"

# Active verifier models, in display order. Each must have a backend registered.
VERIFIER_MODELS = ["gpt-5.5"]
# Total camera-frames to include per verification call (head + wrist composites).
VERIFIER_MAX_FRAMES = 18
# Whether to attach wrist-camera frames for grasp/insert segments (catches arm/vial swaps).
VERIFIER_USE_WRIST_CAMERAS = True
# Max image edge in px when sending to OpenAI. None = original resolution (recommended).
VERIFIER_OPENAI_MAX_EDGE = None


def repo_safe_name(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def model_safe_name(model_id: str) -> str:
    """Filesystem-safe form of a model ID (e.g. 'gpt-5.5' stays as-is)."""
    return model_id.replace("/", "_")


def annotation_dir(repo_id: str, annotator_model: str) -> "Path":
    """annotations/<repo_safe>/<annotator>/"""
    return ANNOTATIONS_DIR / repo_safe_name(repo_id) / model_safe_name(annotator_model)


def verification_dir(repo_id: str, annotator_model: str) -> "Path":
    """verifications/<repo_safe>/<annotator>/"""
    return VERIFICATIONS_DIR / repo_safe_name(repo_id) / model_safe_name(annotator_model)


def success_states_dir(repo_id: str, annotator_model: str) -> "Path":
    """success_states/<repo_safe>/<annotator>/"""
    return SUCCESS_STATES_DIR / repo_safe_name(repo_id) / model_safe_name(annotator_model)


HUMAN_VERIFIED_SUBDIR = "human_verified"


def human_verified_dir(repo_id: str) -> "Path":
    """annotations/<repo_safe>/human_verified/ — snapshot of human-approved annotations."""
    return ANNOTATIONS_DIR / repo_safe_name(repo_id) / HUMAN_VERIFIED_SUBDIR


def list_annotators_for_repo(repo_id: str) -> list[str]:
    """Discover which annotator subdirs already exist for a given repo.

    Excludes the special `human_verified` mirror — that's not an annotator, it's a
    snapshot of verified annotations regardless of producer.
    """
    base = ANNOTATIONS_DIR / repo_safe_name(repo_id)
    if not base.exists():
        return []
    return sorted(d.name for d in base.iterdir()
                  if d.is_dir() and d.name != HUMAN_VERIFIED_SUBDIR)


def get_openai_key() -> str | None:
    """Reads $OPENAI_API_KEY then falls back to .secrets/openai_key.txt."""
    import os
    env = os.environ.get("OPENAI_API_KEY")
    if env:
        return env.strip()
    p = SECRETS_DIR / "openai_key.txt"
    if p.exists():
        return p.read_text().strip()
    return None


def get_litellm_key() -> str:
    """Reads $LITELLM_API_KEY then falls back to .secrets/litellm_key.txt.

    Raises RuntimeError if neither is set — the LiteLLM key is required for the
    default Gemini-ER annotator and the gateway_gemini verifier backend.
    """
    import os
    env = os.environ.get("LITELLM_API_KEY")
    if env:
        return env.strip()
    p = SECRETS_DIR / "litellm_key.txt"
    if p.exists():
        return p.read_text().strip()
    raise RuntimeError(
        "LITELLM_API_KEY not set. Either `export LITELLM_API_KEY=sk-...` or write "
        f"the key to {p}. See README.md for setup."
    )


