"""Pluggable annotation backends.

Each backend exposes:
- `count_vote(image_path, question, range_lo, range_hi, n_votes) -> (mode, votes)`
- `generate_with_retry(system_text, neutral_parts, validate_fn, max_attempts) -> (parsed, raw_text, attempts, usage)`

Where `neutral_parts` is a list of `{"text": str}` and `{"image": path}` dicts. Each backend
converts these to its native API format.
"""
from __future__ import annotations
from typing import Callable, Protocol


class Backend(Protocol):
    def count_vote(self, image_path, question, range_lo, range_hi, n_votes): ...
    def generate_with_retry(self, system_text, neutral_parts, validate_fn, max_attempts): ...


_REGISTRY: dict[str, Backend] = {}


def register(model_id: str, backend: Backend) -> None:
    _REGISTRY[model_id] = backend


def get(model_id: str) -> Backend:
    if model_id not in _REGISTRY:
        raise KeyError(
            f"No annotator backend registered for '{model_id}'. "
            f"Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[model_id]


def available() -> list[str]:
    return sorted(_REGISTRY)


# Import backends AFTER register/get are defined so they can self-register.
from . import gemini  # noqa: F401,E402
from . import openai_chat  # noqa: F401,E402
from . import claude  # noqa: F401,E402
