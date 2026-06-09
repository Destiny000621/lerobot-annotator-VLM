# lerobot-annotator-VLM

VLM-driven sub-task annotator for [LeRobot](https://huggingface.co/lerobot) v3.0 datasets with human-in-the-loop verification.

Decompose long-horizon robot manipulation episodes into per-frame language instructions for VLA SFT (e.g. pi0, pi0.5, OpenVLA). Powered by Google's Gemini Robotics-ER 1.6 via the Avant LiteLLM gateway. Generic across any LeRobot dataset and any manipulation task; task-specific behavior lives entirely in `tasks/*.yaml` or `tasks/*.py`.

---

## What it produces

For each annotated episode you get:

1. **Annotation JSON** (`annotations/<repo>/episode_XXXXXX.json`):
   - Total object count (from a 5-vote majority counter)
   - Per-object positional descriptors ("the leftmost vial", "the vial nearest the stand")
   - Pickup sequence
   - Contiguous segments with `start_s`, `end_s`, primitive (`approach`/`grasp`/`transport`/`align`/`insert`/`retract`/`idle`), arm, object_id, natural-language instruction, and progress `k/N`
   - Success-state JPEGs at the end of each "success" primitive (e.g. `insert`)
2. **AI-verifier reports** (`verifications/<repo>/episode_XXXXXX.json`):
   - One or more external models (GPT-5.5, Gemini, etc.) cross-check the VLM output by looking
     at head + left_wrist + right_wrist camera frames at each grasp/insert moment
   - Output schema: `{overall, issues[{segment_index, type, severity, description}], summary}`
3. **Per-frame `task_index`** in a regenerated LeRobot data parquet
4. **Expanded `tasks.parquet`** indexing every unique instruction across the annotated set
5. A web UI with **synchronized video players for all cameras**, per-segment "Play this slice" buttons, inline display of AI-flagged issues, and one-click human verify / re-annotate / approve

---

## Architecture at a glance

```
┌──────────────────────────────────────────────────────────────┐
│  CLI:  annotate / verify / materialize / serve                │
│   │                                                           │
│   ├─ Task config (YAML or Python)                            │
│   │   ↳ prompt, primitives, schema extras, cam_fps,          │
│   │     success_primitive, count_vote question               │
│   │                                                           │
│   ├─ Dataset adapter (LeRobotDataset)                        │
│   │   ↳ auto-discovers cameras, fps, episode layout from     │
│   │     meta/info.json — works on any HF LeRobot v3.0 repo  │
│   │                                                           │
│   ├─ Pipeline (annotate.py)                                  │
│   │     1. extract per-camera keyframes at task's fps        │
│   │     2. count-vote on first + final frame                 │
│   │        (5 samples → majority, occlusion-robust)          │
│   │     3. inject up to 3 verified-episode exemplars         │
│   │     4. call selected annotator backend (Gemini/GPT-5.5)  │
│   │     5. validate + retry on failure                       │
│   │     6. apply human overrides post-call                   │
│   │     7. save success-state JPEGs                          │
│   │                                                           │
│   ├─ Verifier (verify.py + verifiers/)                       │
│   │   ↳ pluggable backends: gpt5 (OpenAI), gateway_gemini    │
│   │   ↳ sends head + left_wrist + right_wrist views at every │
│   │     grasp/insert; aggressive cross-arm consistency prompt │
│   │   ↳ writes issue list to verifications/<repo>/ep_*.json  │
│   │                                                           │
│   ├─ Override store (overrides/)                             │
│   │   ↳ persistent human-confirmed counts, sequences,        │
│   │     segments — pinned as hard constraints on rerun       │
│   │                                                           │
│   ├─ Materialize (materialize.py)                            │
│   │   ↳ expand segments → per-frame task_index               │
│   │   ↳ rewrite episode parquets, build new tasks.parquet    │
│   │                                                           │
│   └─ Flask HITL app (webapp.py + templates/)                 │
│       ↳ HTML5 video players (head + wrists, sync play)       │
│       ↳ per-segment ▶ Play (seek + auto-pause at end_s)      │
│       ↳ inline AI-verifier issue badges on flagged segments  │
│       ↳ browse, edit, approve, re-annotate, re-verify        │
└──────────────────────────────────────────────────────────────┘
```

---

## Installation

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/) (or use plain `python -m venv` + `pip`).

```bash
cd /Users/destiny/Desktop/ForceRL/lerobot_annotator
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python pyarrow openai pillow av pandas flask pyyaml
```

The `.venv` is what every example below assumes. There is no `pyproject.toml` yet; you invoke scripts directly.

### API keys

Two keys are needed depending on which backends you use. Both follow the same pattern: env var first, gitignored `.secrets/*.txt` file as fallback.

**LiteLLM gateway key** (required for the default Gemini-ER annotator and the gateway Gemini verifier):

```bash
mkdir -p .secrets
# Option A — shell environment:
export LITELLM_API_KEY=sk-...
# Option B — gitignored file:
echo "sk-..." > .secrets/litellm_key.txt
chmod 600 .secrets/litellm_key.txt
```

**OpenAI key** (required if you use any OpenAI-backed annotator or the GPT-5.5 verifier):

```bash
# Option A — shell environment:
export OPENAI_API_KEY=sk-proj-...
# Option B — gitignored file:
echo "sk-proj-..." > .secrets/openai_key.txt
chmod 600 .secrets/openai_key.txt
```

`.secrets/` is in `.gitignore`, so keys never get committed. The code reads env vars first, then falls back to the files. The LiteLLM key is loaded lazily — if you only run OpenAI-backed annotators/verifiers, you don't need to set it.

---

## Quick start

End-to-end on the vial-placement pilot:

```bash
REPO=Sichang0621/8ml_vial_place_30fps_fixed
EPS=123-130

# 1) Annotate — Gemini-ER produces the draft annotation JSONs.
#    Already-verified episodes are auto-injected as few-shot exemplars,
#    so quality improves as you verify more episodes.
.venv/bin/python cli.py annotate \
    --repo $REPO --task vial_place --episodes $EPS

# 2) Verify — GPT-5.5 looks at head + wrist cameras and flags errors
#    (arm swaps, wrong vials, wrong counts, fake bimanual, etc.).
.venv/bin/python cli.py verify \
    --repo $REPO --task vial_place --episodes $EPS

# 3) Apply structured corrections — deterministically rewrites
#    total_vials / pickup_sequence / arm / vial_id / primitive /
#    instruction back into the annotation + override store.
.venv/bin/python cli.py apply-corrections \
    --repo $REPO --episodes $EPS

# 4) (Optional) Verify again to catch any residual issues.
.venv/bin/python cli.py verify \
    --repo $REPO --task vial_place --episodes $EPS
.venv/bin/python cli.py apply-corrections \
    --repo $REPO --episodes $EPS

# 5) Launch the web UI to watch the videos, review residual flagged issues,
#    edit anything left, approve.
.venv/bin/python cli.py serve \
    --repo $REPO --task vial_place --port 5050
# → open http://127.0.0.1:5050

# 6) After approving in the UI, materialize the LeRobot output.
.venv/bin/python cli.py materialize \
    --repo $REPO --episodes $EPS
# Output lands in output_lerobot/Sichang0621__8ml_vial_place_30fps_fixed/
```

**Recommended cadence:** verify → apply-corrections → verify → apply-corrections → human review. Each verify+apply pass cleans up obvious errors (arm swaps, wrong counts) deterministically, so by the time you open the UI you're only deciding the things a human really has to (`suspicious_idle`, `missing_vial`, segment boundary tweaks).

---

## CLI reference

### `annotate`
Runs the full pipeline for a range of episodes. Reuses cached keyframes; respects existing overrides; idempotent unless `--overwrite`.

```
annotate --repo <repo_id> --task <task_name_or_path> --episodes <spec>
         [--overwrite] [--no-exemplars] [--annotator <model_id>]
```

- `--repo` — HF dataset repo_id, e.g. `Sichang0621/8ml_vial_place_30fps_fixed`
- `--task` — either a name resolved against `tasks/` (e.g. `vial_place` → looks for `tasks/vial_place.yaml` then `tasks/vial_place.py`) or an explicit file path
- `--episodes` — `123-130`, `5,7,12`, or a mix `1-10,15,20-22`
- `--overwrite` — re-annotate even if `annotations/<repo>/episode_XXXXXX.json` already exists
- `--no-exemplars` — skip few-shot injection from human-verified episodes (useful for debugging)
- `--annotator` — which model to use. Default: `DEFAULT_ANNOTATOR` in `config.py` (`gemini-robotics-er-1.6-preview`).
  Other registered options:
  - **OpenAI** (require `OPENAI_API_KEY`): `gpt-5.5`, `gpt-5.5-pro`, `gpt-5.4`, `gpt-5.4-pro`,
    `gpt-5.3-chat-latest`, `gpt-5.2`, `gpt-5.1`, `gpt-5`, `gpt-4o`, `gpt-4.1`
  - **Avant gateway**: `gemini-robotics-er-1.6-preview`

  Cost / quality tradeoff observed on the vial-placement pilot:

  | Annotator | Time / ep | Cost / ep | Arm assignment quality |
  |---|---|---|---|
  | `gemini-robotics-er-1.6-preview` | ~30s | ≈ $0 (gateway) | Often gets arm assignments wrong; verifier+apply-corrections fixes them |
  | `gpt-5.5` | ~90s | ≈ $0.20 (OpenAI) | Usually correct on first attempt for arm assignments |
  | `gpt-5.5-pro` | slower, more expensive | higher | Highest quality; reserve for problem episodes |

  Recommended: use the cheap `gemini-robotics-er-1.6-preview` for bulk first-pass annotation, then re-annotate the residual problem episodes with `gpt-5.5` instead of relying purely on `verify → apply-corrections`.

### `verify`
Runs one or more AI verifier models over existing annotations and writes their flagged issues to `verifications/<repo>/episode_XXXXXX.json`.

```
verify --repo <repo_id> [--task <task>] --episodes <spec>
       [--models gpt-5.5,gemini-robotics-er-1.6-preview]
```

- `--repo`, `--episodes` — same as `annotate`
- `--task` — optional; passed to the verifier prompt as primitives context
- `--models` — comma-separated verifier model IDs. Default: `VERIFIER_MODELS` from `config.py` (defaults to `["gpt-5.5"]`)

The verifier sends head + left_wrist + right_wrist views at every grasp/insert moment and at suspicious bimanual/null-vial action segments (~18 frames per episode), then applies an aggressive cross-arm consistency prompt. Issues are surfaced in the web UI as red-bordered segments with inline descriptions. The verifier ALSO emits a structured `corrections` field with deterministic edits the next command can apply. Stale verifier results are ignored after annotation edits; run `verify` again before applying corrections.

### `apply-corrections`
Applies the structured `corrections` from the most recent `verify` run back into the annotation + override store.

```
apply-corrections --repo <repo_id> --episodes <spec>
```

What it applies (only when the verifier was confident):

| Field | Goes to |
|---|---|
| `total_vials` | `annotation.total_vials` + `override.pinned_count` |
| `pickup_sequence` | `annotation.pickup_sequence` + `override.pinned_fields["pickup_sequence"]` |
| `vial_descriptors` | `annotation.vial_descriptors` + `override.pinned_fields["vial_descriptors"]` |
| `segments[].{arm, vial_id, primitive, instruction}` | the matching segment's fields, and the full segments list is pinned to `override.pinned_segments` |

A `verifier_corrections_applied` field is added to the annotation with from/to records for audit.

The verifier prompt is explicitly instructed to emit a **rewritten `instruction` string together with any structured field change** (arm/vial_id/primitive). So if it flips a segment's arm from `"right"` to `"left"`, both the `arm` field and the natural-language `instruction` (e.g., "use the **left** arm to ...") are updated in one shot — no stale prose, no second pass needed.

### `set-pickup-sequence`
Applies a human-provided vial pickup order when the annotation is internally consistent but the VLM chose the wrong vial identity/order.

```bash
.venv/bin/python cli.py set-pickup-sequence \
    --repo Sichang0621/8ml_vial_place_30fps_fixed \
    --episode 125 \
    --sequence 2,1,3 \
    --status verified
```

This rewrites `pickup_sequence`, updates the per-segment `vial_id` and instruction text for each progress block, pins the corrected segments in the override store, and optionally updates the episode status.

### `materialize`
Expands annotation segments into per-frame `task_index` and writes a LeRobot-compatible output tree.

```
materialize --repo <repo_id> --episodes <spec>
```

Writes to `output_lerobot/<repo_safe>/`:
- `meta/info.json` with updated `total_tasks`
- `meta/tasks.parquet` (row 0 = original task; rows 1..K = every unique sub-task instruction)
- `meta/episodes/chunk-000/file-000.parquet` with updated `tasks` lists for the annotated episodes
- `data/chunk-<XXX>/file-<XXX>.parquet` with rewritten `task_index` column for each annotated episode

Unannotated episodes still point to `task_index=0` (the original task string), so the resulting dataset stays consistent.

### `serve`
Starts the Flask HITL UI on `localhost`.

```
serve [--repo <default_repo>] [--task <default_task>] [--port 5050] [--debug]
```

The `--repo` / `--task` are optional defaults the UI prefills; multiple repos can coexist in `annotations/`.

---

## Web UI guide

### Index page (`/`)

One row per annotated episode, with status background color:

| Color | Status | Meaning |
|---|---|---|
| Yellow | `draft` | VLM output not yet reviewed |
| Green | `verified` | Human approved (becomes ground truth) |
| Red | `needs_rework` | Human flagged for re-run |

Columns: episode index, status, total count, number of segments, duration, VLM attempts used, **# exemplars** used to generate this annotation, **AI-verifier verdict** (`pass` / `N major / N minor` / `—` if not run yet). Click **Open** to review.

A header line shows the **number of verified episodes** (which are used as few-shot exemplars on every new annotation).

### Episode page (`/episode/<ep>`)

**Top of page — video players**:
- Synced HTML5 video element for the head camera (auto-loaded, native browser controls, live timestamp readout).
- **"show wrist cameras"** button reveals left + right wrist cameras side by side; all three play, pause, and seek in lockstep with the head video.
- Plays HEVC (yuv420p) natively on Safari and macOS Chrome 107+. Use Safari if your browser shows black; transcoding to H.264 with `ffmpeg` is a future option.

**Top badges**:
- Episode duration
- Override status (`draft` / `verified` / `needs_rework`)
- Few-shot info — which verified episodes were used as exemplars when this annotation was generated
- **AI-verifier verdict** (`AI verified: pass` green, or `AI: N major / N minor` red) — only shown after `cli.py verify` has run on this episode
- **↻ Run AI verifier** button — triggers a background verifier call; refresh in ~2 min

If the verifier has run, an info panel below the badges shows each model's one-paragraph summary plus any episode-level issues (e.g. `wrong_count`).

**Left column** — episode fields:
- First-frame head-camera image (decoded on demand from the cached mp4).
- `total_vials` count (editable; shows source: `override` or `majority_vote`).
- `vial_descriptors` (JSON), `pickup_sequence` (JSON) — editable.
- Success-state thumbnails: end-of-`insert` frames for every successful sub-task.

**Right column** — segment editing:
- A colored horizontal **timeline bar** showing every segment, sized by duration, colored by primitive.
- **Segments with AI-flagged issues get a red border** and an inline warning showing the issue type, severity, description, and which model flagged it.
- Per-segment row: `start_s`, `end_s`, `arm` dropdown, `primitive` dropdown, `vial_id` input, instruction textarea, `progress` field.
- **▶ Play** (green) — seeks the head video (and wrists if visible) to `start_s`, plays, and auto-pauses exactly at `end_s`. Scrolls the page to the video.
- **👁** (gray) — loads the mid-segment head-camera frame inline as a still image.
- **×** (red) — deletes the segment.
- **+ Add segment** appends a new row starting at the previous segment's end.

**Action buttons**:

| Button | What it does |
|---|---|
| 💾 Save | Writes edits to both the annotation JSON and the override store. Stays `draft`. |
| ✓ Approve | Saves + sets status to `verified`. The override pins these values for any future rerun. Episode also becomes a few-shot exemplar for future annotations. |
| ✗ Needs rework | Sets status to `needs_rework`. |
| ↻ Re-annotate with overrides | Saves + kicks off a **background** Gemini call using your current overrides as hard constraints. Reload the page in ~30s. |
| ↻ Run AI verifier | Kicks off a **background** verifier call (~2 min). Refresh to see results. |

### What gets persisted

Every save writes two places:

1. `annotations/<repo>/episode_XXXXXX.json` — the canonical annotation
2. `overrides/<repo>/episode_XXXXXX.json` — the override store, which feeds re-runs:
   - Count edits → `pinned_count` (passed to the next annotate call as a hard constraint)
   - Top-level field edits → `pinned_fields` (re-applied after the next VLM call)
   - Segment edits → `pinned_segments` (entirely replaces VLM segments)
   - Status

---

## Writing a task config

Both YAML and Python forms produce the same `Task` object. Use YAML for declarative tasks; use Python when you need imports, computed prompts, or shared helpers.

### YAML form (`tasks/<task_name>.yaml`)

```yaml
name: vial_place
description: "Place 1–4 vials into a stand; dual-arm yam robot."
default_task_string: "place the vial into the stand"

primitives: [approach, grasp, transport, align, insert, retract, idle]

schema_extras:
  total_vials: {type: int, range: [1, 4]}
  stand_slots: {type: int}
  vial_descriptors: {type: list}
  pickup_sequence: {type: list}

segment_extras:
  arm: {type: enum, values: [left, right, both]}
  vial_id: {type: int_optional}
  instruction: {type: string}
  progress: {type: string}

success_primitive: insert       # whose end frame to save as success state

count_vote:
  field: total_vials            # JSON field to pin
  range: [1, 4]
  # If true, the counter trusts the FINAL frame as ground truth — appropriate when every
  # episode is a known success and the receptacle's final state = true initial count.
  # Defaults to false (use max(first_frame, final_frame) instead).
  prefer_final_frame: true
  question: >
    Count the vials seated in the stand at the final frame.

cam_fps:
  head_camera: 2.0              # count evidence comes from head-camera count vote
  left_wrist_camera: 1.0        # arm/action evidence only; never used for vial count
  right_wrist_camera: 1.0       # arm/action evidence only; never used for vial count

prompt_template: |
  ...the full system prompt with {task_string}, {primitives}, {count_field},
     {pinned_count} placeholders...
```

### Python form (`tasks/<task_name>.py`)

```python
from lerobot_annotator.task import CountVote, Task

TASK = Task(
    name="vial_place",
    description="...",
    default_task_string="place the vial into the stand",
    primitives=["approach", "grasp", "transport", "align", "insert", "retract", "idle"],
    schema_extras={"total_vials": {"type": "int", "range": [1, 4]}, ...},
    segment_extras={"arm": {"type": "enum", "values": ["left", "right", "both"]}, ...},
    success_primitive="insert",
    count_vote=CountVote(field="total_vials", question="...", range=(1, 4)),
    cam_fps={"head_camera": 2.0},
    prompt_template="...",
)
```

The module must expose a top-level `TASK = Task(...)`.

### Resolution rules

`--task vial_place` resolves in this order:
1. If `vial_place` is an existing file path → load directly.
2. `tasks/vial_place.yaml` → load YAML.
3. `tasks/vial_place.py` → import and read `TASK`.

### Prompt template placeholders

The following are substituted via `str.format()`:

| Placeholder | Value |
|---|---|
| `{task_string}` | `task.default_task_string` |
| `{primitives}` | comma-separated `primitives` list |
| `{count_field}` | `task.count_vote.field` (or `""`) |
| `{pinned_count}` | the count resolved by override or majority vote |

Use `{{` and `}}` to emit literal braces in the JSON schema description.

---

## Few-shot exemplars from verified episodes

Once you approve an episode in the web UI, it becomes a **few-shot exemplar** that gets prepended to every subsequent annotation call for the same dataset. The VLM sees up to **3 verified annotations** (1 first-frame image + cleaned JSON each) before the new episode's keyframes, and it learns to mimic your corrected schema, numbering convention, instruction wording, and segment granularity from the examples.

This is the recommended workflow:

```
1. Annotate episodes 1-10 (no exemplars yet — VLM uses prompt rules only)
2. Human-verify a few (e.g. 3-5) in the web UI → status=verified
3. Annotate episodes 11+ — they automatically use the verified ones as exemplars
4. Quality improves as you verify more episodes
```

The pipeline:
- Scans `overrides/<repo>/episode_*.json` for `status == "verified"`
- Picks up to `MAX_EXEMPLARS` (default 3) sorted by episode index
- For each: includes the first head-camera frame + the cleaned annotation JSON
  (`_raw_text`, `attempts`, `usage`, success-frame paths, etc. are stripped)
- Subtracts 4 image slots per exemplar from the main image budget so we stay under 128K tokens
- Records which exemplars were used in the new annotation's `exemplars_used` field

You can see the count of verified episodes in the UI's index page and a badge on the episode page showing which exemplars informed each annotation.

**Disable** few-shot mode for a run:

```bash
.venv/bin/python cli.py annotate ... --no-exemplars
```

**Choose exact exemplars for a run:**

```bash
.venv/bin/python cli.py annotate \
    --repo Sichang0621/8ml_vial_place_30fps_fixed \
    --task vial_place \
    --episodes 126-132 \
    --overwrite --reset-overrides \
    --exemplars 123,124,125
```

When `--exemplars` is provided, the listed episodes are injected in exactly that order. The episode currently being annotated is skipped if it appears in the list.

**Tune** in `src/lerobot_annotator/config.py`:
- `MAX_EXEMPLARS = None` — include all verified episodes as exemplars; set an int to cap
- `EXEMPLAR_INCLUDE_FIRST_FRAME = True` — set False for text-only exemplars (fits more in budget, but loses visual grounding of descriptors)

## AI verifier

VLMs make mistakes that humans only catch by carefully watching the video — most commonly **arm/vial assignment swaps** ("left arm did this" when the right arm actually did). To save the human reviewer's time, an AI verifier runs over the VLM annotation, looks at the videos, and flags likely errors **before** human review.

### Pipeline

```
annotation JSON (from Gemini-ER) ──┐
                                   ▼
              ┌────────────────────────────────────┐
              │ pick frames (verify.py)            │
              │  • first frame (count/descriptors) │
              │  • head + LEFT_WRIST + RIGHT_WRIST │
              │    at every grasp/insert moment    │
              │  • fill with head-only frames      │
              └─────────────────┬──────────────────┘
                                ▼
              ┌────────────────────────────────────┐
              │ verifier backend (any model)       │
              │  • prompt emphasizes cross-arm     │
              │    consistency via wrist cameras   │
              │  • biased toward false positives   │
              │  • outputs JSON {overall, issues,  │
              │    summary} with severity+type     │
              └─────────────────┬──────────────────┘
                                ▼
                  verifications/<repo>/episode_*.json
                                │
                                ▼
                  ─ index page: verdict column
                  ─ episode page: red-bordered flagged
                    segments with inline descriptions
```

### Why wrist cameras matter

The head camera can't reliably distinguish which arm is acting in cluttered moments. The wrist cameras are mounted on each arm, so the LEFT_WRIST view shows what the left gripper is doing and vice versa. The annotator and verifier use wrist frames for arm/action disambiguation only, while vial counting remains a separate head-camera final-frame vote. This catches arm swaps and fake bimanual labels without letting wrist close-ups inflate the vial count.

### Output schema (`verifications/<repo>/episode_XXXXXX.json`)

```json
{
  "repo_id": "user/dataset",
  "episode_index": 125,
  "annotation_fingerprint": {"sha256": "...", "size": 12345},
  "n_frames_shown": 18,
  "verifiers": {
    "gpt-5.5": {
      "model": "gpt-5.5",
      "parsed": {
        "overall": "issues_found",
        "issues": [
          {"segment_index": 4, "type": "wrong_arm", "severity": "major",
           "description": "At t=11.50s, LEFT_WRIST shows the left gripper holding a vial over the stand while RIGHT_WRIST shows the right gripper empty, contradicting the annotation that the right arm inserts vial No.2."}
        ],
        "summary": "..."
      },
      "usage": {"completion_tokens": ..., "prompt_tokens": ...},
      "_raw_text": "..."
    }
  },
  "overall_consensus": {"overall": "issues_found", "n_major": 3, "n_minor": 2}
}
```

### Error types

| Type | Meaning |
|---|---|
| `wrong_arm` | Annotation's arm doesn't match which wrist camera shows the action |
| `wrong_vial` | Annotation's vial_id doesn't match the descriptor of the actually-grasped vial |
| `wrong_primitive` | Claimed primitive (e.g. `grasp`) doesn't match what's visible (e.g. transport) |
| `wrong_bimanual` | Claims "both arms" but only one wrist shows action (or vice versa) |
| `wrong_timing` | Action shown doesn't fall inside the segment's `[start_s, end_s]` window |
| `missing_vial` | A vial visible initially is never picked up or inserted |
| `extra_segment` | A segment that doesn't correspond to any visible action |
| `wrong_count` | `total_vials` disagrees with the task's head-camera count rule (final-frame stand count for vial_place) |
| `wrong_descriptor` | A descriptor doesn't match the actual position of that vial_id |
| `suspicious_idle` | Long idle segment that may hide an unannotated action |

Severity is `major` (semantically changes the policy's behavior) or `minor` (timing off by <1s, stylistic).

### Pluggable backends (`src/lerobot_annotator/verifiers/`)

- **`gpt5.py`** — OpenAI Chat Completions. Registers: `gpt-5.5`, `gpt-5.5-pro`, `gpt-5.4`, `gpt-5.4-pro`, `gpt-5.3-chat-latest`, `gpt-5.2`, `gpt-5.1`, `gpt-5`, `gpt-4o`, `gpt-4.1`. Reads `$OPENAI_API_KEY` or `.secrets/openai_key.txt`.
- **`gateway_gemini.py`** — Avant LiteLLM gateway. Registers: `gemini-robotics-er-1.6-preview`. Uses the same gateway key as `annotate`.

Adding a new model later (e.g. Claude when gateway access lights up) is a 5-line `register("claude-opus-4-7-max", lambda ...: ...)` addition.

### Tuning

In `src/lerobot_annotator/config.py`:

| Variable | Default | What it does |
|---|---|---|
| `VERIFIER_MODELS` | `["gpt-5.5"]` | Active verifier list |
| `VERIFIER_MAX_FRAMES` | `18` | Total camera-frames per call (head + wrist) |
| `VERIFIER_USE_WRIST_CAMERAS` | `True` | Attach wrist views at every grasp/insert |
| `VERIFIER_OPENAI_MAX_EDGE` | `None` | Max image edge in px sent to OpenAI; `None` = original |

Per-call cost on GPT-5.5: ~5–15K tokens (reasoning-heavy), ~$0.05–$0.20 depending on episode length. Run on the whole pilot is ~$1.

## Overrides: how human edits feed back into the loop

```
┌──────────────────┐                           ┌─────────────────────┐
│  Web UI edits    │  save / approve / rerun   │  overrides/<repo>/  │
│  (count,         │ ────────────────────────► │  episode_XXXXXX.json│
│   sequence,      │                           └─────────┬───────────┘
│   segments)      │                                     │
└──────────────────┘                                     │
                                                         ▼
┌──────────────────┐                           ┌─────────────────────┐
│  annotate_       │ ◄──────────────────────── │  pinned_count       │
│  episode()       │   pre-call constraint     │  (forces VLM count) │
│                  │                           └─────────────────────┘
│                  │                           ┌─────────────────────┐
│                  │ ◄──────────────────────── │  pinned_fields      │
│                  │   post-call replacement   │  pinned_segments    │
└──────────────────┘                           └─────────────────────┘
```

This means **once you approve an episode, re-running it will produce something very close to your approved version** — the count is fixed, your edited segments aren't touched, and only the still-flexible fields get re-generated.

---

## Materialization output

`output_lerobot/<repo_safe>/` mirrors the source HF dataset's structure but only for the annotated subset:

```
meta/
  info.json              # total_tasks bumped
  tasks.parquet          # row 0 = default task, rows 1..K = unique sub-task instructions
  episodes/chunk-000/file-000.parquet   # `tasks` list updated for annotated eps
data/chunk-<XXX>/file-<EP>.parquet      # per-frame task_index rewritten
```

Source videos and unannotated episode parquets are **not copied** — they're still resolvable via the original HF repo. If you want a fully self-contained mirror (e.g. to upload as a new HF dataset), the `vial_annotation/scripts/materialize_lerobot.py` `mirror_unaltered()` reference implementation shows how (symlinks original assets into the output tree); a generic version isn't wired into the new tool yet.

### Using the output with pi0.5

Point your pi0.5 `LeRobotDataConfig.repo_id` (after pushing to HF) at the new dataset. The per-frame `task_index` will resolve to the matching sub-task string via the expanded `tasks.parquet`, so the policy sees a rich instruction at every timestep instead of the single global task string.

---

## Configuration

All paths and gateway settings live in `src/lerobot_annotator/config.py`:

```python
ROOT = <project root>
META_CACHE         = ROOT / "meta_cache"        # info.json, episodes.parquet, etc.
VIDEO_CACHE        = ROOT / "video_cache"       # cached per-episode mp4s
KEYFRAMES_CACHE    = ROOT / "keyframes_cache"   # decoded JPEG keyframes
ANNOTATIONS_DIR    = ROOT / "annotations"       # canonical annotation JSONs
OVERRIDES_DIR      = ROOT / "overrides"         # human edits
SUCCESS_STATES_DIR = ROOT / "success_states"    # success-frame JPEGs
VERIFICATIONS_DIR  = ROOT / "verifications"     # AI-verifier reports
OUTPUT_LEROBOT     = ROOT / "output_lerobot"    # materialized datasets
TASKS_DIR          = ROOT / "tasks"             # task configs
SECRETS_DIR        = ROOT / ".secrets"          # gitignored: API keys

# Annotation backend (Gemini Robotics-ER 1.6 via Avant gateway)
LITELLM_BASE       = "https://litellm.avantrobotics.ai"
LITELLM_API_KEY    = "sk-..."
GEMINI_MODEL       = "gemini-robotics-er-1.6-preview"

# Annotation pipeline
DEFAULT_ANNOTATOR     = "gemini-robotics-er-1.6-preview"   # override per-call with --annotator
MAX_IMAGES_PER_CALL   = 120                     # ~128K input budget / 1064 tok per image
MAX_EDGE_PX           = 384                     # JPEG downscale for Gemini calls
MAX_ANNOTATE_ATTEMPTS = 4                       # retries on validation failure
COUNT_VOTE_SAMPLES    = 5                       # first-frame counter votes

# Few-shot from verified episodes
MAX_EXEMPLARS                = 3
EXEMPLAR_INCLUDE_FIRST_FRAME = True

# AI verifier
VERIFIER_MODELS              = ["gpt-5.5"]      # comma-separable list
VERIFIER_MAX_FRAMES          = 18               # head + wrist frames per call
VERIFIER_USE_WRIST_CAMERAS   = True
VERIFIER_OPENAI_MAX_EDGE     = None             # None = original 640×480 to OpenAI

FLASK_PORT = 5050
```

Each per-(repo, episode) artifact is stored under `<dir>/<repo_safe>/episode_XXXXXX.*` where `repo_safe = repo_id.replace("/", "__")`. This keeps multiple datasets cleanly side by side.

### API keys

| Key | Lookup order | Used for |
|---|---|---|
| `LITELLM_API_KEY` | hardcoded in `config.py` (move to env if shipping) | Gemini-ER annotation + Gemini verifier backend |
| `OPENAI_API_KEY` | `$OPENAI_API_KEY` → `.secrets/openai_key.txt` (mode 600) | OpenAI verifier backend (gpt-5.5 etc.) |

---

## Gemini Robotics-ER 1.6 quirks worth knowing

- Each image input costs a **fixed 1064 tokens** regardless of resolution (verified from 128 to 384 px max-edge). Context cap is **131,072** input tokens, so the practical budget is ~120 images per call. The annotator auto-downsamples the head-camera keyframes when an episode is long.
- The OpenAI-compat path (`/v1/chat/completions`) **counts base64 data URLs as text tokens**, overflowing the context with even ~20 frames. Always use the native `/v1beta/models/<model>:generateContent` path with `inline_data` parts. The `gemini.py` client already does this.
- The model uses internal "thinking" tokens that **sample even at `temperature=0`**. Two consecutive calls with identical inputs can produce different outputs. The pipeline mitigates this via:
  1. A separate count-vote (5 samples → majority) decoupled from segmentation
  2. A validate-and-retry loop on the main call (up to 4 attempts, escalating temperature)
  3. Override-based pinning so once a human verifies, future runs cannot drift

---

## Layout

```
lerobot_annotator/
├── cli.py                            # argparse entrypoint
├── README.md                         # this file
├── .gitignore                        # excludes .secrets/, caches, outputs
├── .secrets/                         # (gitignored) OPENAI_API_KEY fallback
│   └── openai_key.txt
├── tasks/
│   ├── vial_place.yaml               # task config — declarative form
│   └── vial_place.py                 # task config — Python form
├── src/lerobot_annotator/
│   ├── __init__.py
│   ├── config.py                     # paths, gateway, verifier constants
│   ├── dataset.py                    # generic LeRobot v3.0 loader
│   ├── extract.py                    # keyframe extraction & video decoding
│   ├── gemini.py                     # native /v1beta client, count-vote, retry
│   ├── task.py                       # Task model + YAML/Python loader
│   ├── overrides.py                  # human override store
│   ├── exemplars.py                  # few-shot retrieval from verified episodes
│   ├── validate.py                   # schema-driven validator
│   ├── annotate.py                   # annotation pipeline orchestrator
│   ├── annotators/                   # pluggable annotation backends
│   │   ├── __init__.py               # registry
│   │   ├── gemini.py                 # Gemini Robotics-ER 1.6 (Avant gateway)
│   │   └── openai_chat.py            # OpenAI Chat Completions (gpt-5.5 etc.)
│   ├── apply_corrections.py          # apply verifier-emitted structured edits
│   ├── verify.py                     # AI verifier orchestrator
│   ├── verifiers/                    # pluggable verification backends
│   │   ├── __init__.py               # registry
│   │   ├── gpt5.py                   # OpenAI Chat Completions
│   │   └── gateway_gemini.py         # Avant gateway Gemini
│   ├── materialize.py                # LeRobot output writer
│   └── webapp.py                     # Flask HITL app
├── templates/
│   ├── index.html                    # episode list + verifier verdicts
│   └── episode.html                  # video players + segment editor + issue badges
├── static/                           # (currently empty; reserved)
├── annotations/                      # per-(repo,ep) annotation JSONs
├── overrides/                        # per-(repo,ep) human overrides
├── verifications/                    # per-(repo,ep) AI verifier reports
├── keyframes_cache/                  # per-(repo,ep,cam,fps) JPEGs
├── video_cache/                      # per-(repo,ep,cam) mp4s
├── success_states/                   # per-(repo,ep) success JPEGs
├── meta_cache/                       # per-repo info.json + episodes.parquet
└── output_lerobot/                   # materialized output trees
```

---

## Extending to a new manipulation task

1. **Pick a task vocabulary.** Decide on the primitives (`approach`, `pick`, `place`, etc.) and any task-specific schema fields (object_id, target_slot, etc.).
2. **Write a task config.** Copy `tasks/vial_place.yaml` to `tasks/your_task.yaml` and edit:
   - `default_task_string` — the global task description
   - `primitives` — the allowed sub-task verbs
   - `schema_extras` — extra top-level JSON fields you want the VLM to emit
   - `segment_extras` — extra per-segment fields
   - `success_primitive` — which primitive's end frame to save as ground truth
   - `count_vote` — what to count and the question to ask (skip this block entirely if your task doesn't have a count to pin)
   - `cam_fps` — which cameras to include and at what rate (omit cameras you don't want sampled)
   - `prompt_template` — write the full system prompt; reference the placeholders described above
3. **Run it.** `cli.py annotate --task your_task --repo <any LeRobot dataset> --episodes 0-5`.
4. **Verify in the web UI**, edit anything wrong, approve, then materialize.

No code changes are required for new tasks. The dataset loader handles any v3.0 layout automatically.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `HTTP 400 ... maximum number of tokens allowed 131072` | Too many images per call | Lower `cam_fps` in your task config, or rely on the built-in subsampling (already caps at 120). For very long episodes consider chunked calls (not yet implemented). |
| `Address already in use` on `serve` | Old Flask still running | `lsof -ti:5050 \| xargs kill -9` |
| `total_vials=N but pinned count is M` validation loop | Gemini drifted from the count-vote result | Set `pinned_count` explicitly in the override (web UI saves this automatically when you edit the count). |
| `episode N not found in <repo>` | Episode index out of range or in a different chunk | Check `meta/episodes/chunk-*/file-*.parquet` — the loader currently assumes `chunk-000/file-000`. |
| Web UI shows old fields with `null` everywhere | Annotation produced by an older schema version | Re-annotate with `--overwrite`, or hand-fill the new fields in the UI and Save. |
| Re-annotate from UI hangs | Background Gemini call is still in flight | Wait ~30s, then refresh. The button refuses to start a second concurrent run for the same episode. |
| `verify` reports `OPENAI_API_KEY not set and .secrets/openai_key.txt missing` | Verifier can't find the OpenAI key | `export OPENAI_API_KEY=sk-...` or create `.secrets/openai_key.txt` |
| `No verifier backend registered for model 'X'` on `verify` | Model ID not in any registry | Check `cli.py verify --models <id>` — the ID must be one registered in `verifiers/gpt5.py` or `verifiers/gateway_gemini.py` |
| Episode page shows black where videos should be | Browser doesn't support HEVC | Use Safari, or macOS Chrome ≥107. Firefox does not support HEVC; transcoding the cached mp4s to H.264 with `ffmpeg` would be a future option. |
| Verifier says `pass` but you spotted errors | Wrist cameras not being attached, or prompt too lenient | Verify `VERIFIER_USE_WRIST_CAMERAS=True` and `VERIFIER_MAX_FRAMES>=18` in `config.py`. Try a stronger model (`gpt-5.5-pro`) via `--models`. |

---

## Limitations

- Currently assumes LeRobot v3.0 episode metadata lives in `meta/episodes/chunk-000/file-000.parquet`. Multi-chunk meta sharding isn't handled yet.
- The materializer doesn't symlink unaltered videos/parquets into `output_lerobot/`. If you want a self-contained mirror (e.g. to push as a new HF dataset), copy/symlink manually.
- The web UI's segment timeline visualization uses fixed numeric inputs for `start_s`/`end_s`. Drag-and-drop on the timeline itself isn't wired; precise editing happens in the number fields with the **👁 preview** as feedback.
- Video players play HEVC directly — works in Safari and macOS Chrome 107+, falls back to black in Firefox. Auto-transcoding to H.264 isn't wired.
- No authentication on the Flask app — it binds to `127.0.0.1` only, which is appropriate for a local-only HITL tool. Don't expose it to a network.
- `LITELLM_API_KEY` is currently hardcoded in `config.py`. Move it to an env var before sharing the repo. (`OPENAI_API_KEY` is already env-var-or-`.secrets/`.)
- The verifier's frame budget (`VERIFIER_MAX_FRAMES = 18`) is a compromise between cost and coverage. Long episodes with many grasp/insert events may miss some moments — bump the cap if cost allows.

---

## Acknowledgments

- Dataset format: [LeRobot](https://huggingface.co/lerobot) v3.0
- VLM annotator: [Gemini Robotics-ER 1.6](https://blog.google/technology/google-deepmind/gemini-robotics/) via the Avant LiteLLM gateway
- AI verifier: OpenAI GPT-5.5 (Chat Completions API)
- Target policy family: [Physical Intelligence pi0 / pi0.5](https://www.physicalintelligence.company/blog/pi0)
