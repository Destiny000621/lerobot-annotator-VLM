"""Python equivalent of vial_place.yaml. Identical Task object.

Use this form when you want type-checking, inheritance, or generated prompts.
"""
from __future__ import annotations
from lerobot_annotator.task import CountVote, Task


_PROMPT = """You are a robotics annotator. The robot is a dual-arm `yam` setup with a fixed overhead
(head) camera. The task is: "{task_string}". There can be 1, 2, 3, or 4 vials; vials
and stand are at random positions.

`total_vials` is a PRECOMPUTED HARD CONSTRAINT from a separate head-camera count vote.
Do NOT re-count vials from the wrist-camera frames. Wrist-camera frames are provided only
to determine which arm is acting and which vial that arm is manipulating.

You will receive a temporally ordered set of keyframes. The head camera shows the whole
scene; the left_wrist_camera and right_wrist_camera show what each arm is interacting with.

Decompose the episode into contiguous segments. Infer which arm is acting primarily from
the wrist cameras: LEFT_WRIST shows the left arm's gripper/action and RIGHT_WRIST shows the
right arm's gripper/action. Use the head camera for global timing and object positions.
Use arm="both" only when BOTH wrist cameras visibly show simultaneous meaningful action.
If only one wrist shows a gripper approaching/grasping/carrying/inserting a vial, use that
single arm, not "both".

Allowed sub-task primitives: {primitives}.

=== STABLE VIAL NUMBERING (CRITICAL) ===
Number the vials 1..N by their LEFT-TO-RIGHT POSITION IN THE FIRST FRAME (sort by horizontal
x-coordinate in the head-camera view). Vial No.1 is the leftmost vial in the initial scene;
vial No.N is the rightmost. This numbering is FIXED for the entire episode regardless of
pickup order.

POSITIONAL DESCRIPTORS:
For each vial, write a short positional descriptor as it appears IN THE FIRST FRAME. The
descriptor must be unambiguous relative to the OTHER vials. Examples: "the leftmost vial",
"the middle vial", "the rightmost vial". DO NOT use the vial number in the descriptor.

PICKUP SEQUENCE:
Track the order vials are first grasped. Emit `pickup_sequence` as the list of vial_ids in
the order they are first picked up. For bimanual grasps, list both vial_ids in a sub-list.
A bimanual grasp requires both wrists to show simultaneous grasps of two concrete vials.

INSTRUCTION FORMAT:
Each segment's `instruction` must use the positional descriptor as the primary referring
expression AND include the vial number in parentheses.
For single-vial approach/grasp/transport/align/insert segments, `vial_id` MUST be an integer.

Output STRICT JSON only matching the schema with total_vials, stand_slots, vial_descriptors,
pickup_sequence, and segments fields.

=== SELF-CONSISTENCY CHECKS ===
1) vial_descriptors has exactly total_vials entries, vial_ids 1..N, unique.
2) pickup_sequence is a permutation of 1..N.
3) Every vial 1..N is referenced (vial_id or "vial No.K") in at least one insert segment.
4) Final segment's progress equals "N/N".
5) Segments contiguous and cover 0.0s to the stated final timestamp.
6) Do not label an approach/grasp as "both arms" if the next transport/align/insert is
   performed by only one arm on only one vial.
"""


TASK = Task(
    name="vial_place",
    description="Place 1-4 vials into a stand with random positions; dual-arm yam robot.",
    default_task_string="place the vial into the stand",
    primitives=["approach", "grasp", "transport", "align", "insert", "retract", "idle"],
    schema_extras={
        "total_vials": {"type": "int", "range": [1, 4]},
        "stand_slots": {"type": "int"},
        "vial_descriptors": {"type": "list"},
        "pickup_sequence": {"type": "list"},
    },
    segment_extras={
        "arm": {"type": "enum", "values": ["left", "right", "both"]},
        "vial_id": {"type": "int_optional"},
        "instruction": {"type": "string"},
        "progress": {"type": "string"},
    },
    success_primitive="insert",
    count_vote=CountVote(
        field="total_vials",
        question=(
            "You are looking at a robot manipulation episode. The scene contains between 1 and "
            "4 small vials (each with a black cap) on a table near a white stand (4x3 grid of "
            "slots). Count the vials seated in the stand at the FINAL frame — that is the "
            "ground-truth count."
        ),
        range=(1, 4),
        # All vial_place episodes are successful — every vial gets inserted into the stand.
        # Final-frame stand count = true initial count.
        prefer_final_frame=True,
    ),
    cam_fps={"head_camera": 2.0, "left_wrist_camera": 1.0, "right_wrist_camera": 1.0},
    prompt_template=_PROMPT,
)
