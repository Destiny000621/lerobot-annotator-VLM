"""Pluggable AI verifier backends for cross-checking VLM-generated annotations."""
from __future__ import annotations
from typing import Callable


_REGISTRY: dict[str, Callable] = {}


def register(model_id: str, fn: Callable) -> None:
    _REGISTRY[model_id] = fn


def get(model_id: str) -> Callable:
    if model_id not in _REGISTRY:
        raise KeyError(f"No verifier backend registered for model '{model_id}'. "
                       f"Available: {sorted(_REGISTRY)}")
    return _REGISTRY[model_id]


def available() -> list[str]:
    return sorted(_REGISTRY)


# Import backends AFTER register() is defined so they can self-register.
from . import gpt5  # noqa: F401,E402
from . import gateway_gemini  # noqa: F401,E402
