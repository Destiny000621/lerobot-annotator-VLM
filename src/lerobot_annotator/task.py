"""Task config loader. Supports both YAML and Python module sources."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
from typing import Any

import yaml

from .config import TASKS_DIR


@dataclass
class CountVote:
    field: str            # which JSON field the count pins (e.g. "total_vials")
    question: str         # human-language question for the counter
    range: tuple[int, int] = (1, 4)
    prefer_final_frame: bool = False  # True = trust the final-frame stand count as ground truth
                                      # (use for tasks where every episode is a guaranteed success
                                      # and final-frame "objects in receptacle" = true initial count)


@dataclass
class Task:
    name: str
    description: str
    default_task_string: str
    primitives: list[str]
    schema_extras: dict[str, Any]      # per-episode extra fields, free-form schema description
    segment_extras: dict[str, Any]     # per-segment extra fields, free-form schema description
    success_primitive: str | None      # which primitive's end frame to save as success state
    count_vote: CountVote | None
    cam_fps: dict[str, float]          # camera name -> sampling Hz; * means all cameras
    prompt_template: str               # the system prompt; rendered with .format(**ctx)

    def render_prompt(self, **ctx: Any) -> str:
        return self.prompt_template.format(**ctx)

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.count_vote is not None:
            d["count_vote"] = asdict(self.count_vote)
        return d


def _from_dict(d: dict) -> Task:
    cv = d.get("count_vote")
    if cv is not None:
        cv = CountVote(
            field=cv["field"],
            question=cv["question"],
            range=tuple(cv.get("range", [1, 4])),
            prefer_final_frame=bool(cv.get("prefer_final_frame", False)),
        )
    return Task(
        name=d["name"],
        description=d.get("description", ""),
        default_task_string=d.get("default_task_string", ""),
        primitives=list(d.get("primitives", [])),
        schema_extras=dict(d.get("schema_extras", {})),
        segment_extras=dict(d.get("segment_extras", {})),
        success_primitive=d.get("success_primitive"),
        count_vote=cv,
        cam_fps=dict(d.get("cam_fps", {})),
        prompt_template=d["prompt_template"],
    )


def load_yaml(path: str | Path) -> Task:
    return _from_dict(yaml.safe_load(Path(path).read_text()))


def load_python(path: str | Path) -> Task:
    p = Path(path)
    spec = spec_from_file_location(p.stem, p)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    if hasattr(mod, "TASK"):
        return mod.TASK
    raise ValueError(f"Python task module {path} must define a top-level TASK = Task(...)")


def load(name_or_path: str) -> Task:
    """Resolve a task by name (looks in tasks/) or by explicit path."""
    p = Path(name_or_path)
    if p.exists():
        return load_yaml(p) if p.suffix in {".yaml", ".yml"} else load_python(p)
    yaml_path = TASKS_DIR / f"{name_or_path}.yaml"
    py_path = TASKS_DIR / f"{name_or_path}.py"
    if yaml_path.exists():
        return load_yaml(yaml_path)
    if py_path.exists():
        return load_python(py_path)
    raise FileNotFoundError(f"No task config found for '{name_or_path}' in {TASKS_DIR}")
