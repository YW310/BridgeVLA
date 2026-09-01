#!/usr/bin/env python3
"""Reset every RLBench variation and audit strict semantic T/R selectors."""

import argparse
import json
from pathlib import Path

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend.utils import task_file_to_task_class
from rlbench.environment import Environment

from bridgevla.libs.peract.helpers import utils
from utils.o2_oracle_provider import RLBenchGTOracleProvider
from utils.peract_utils_rlbench import CAMERAS, IMAGE_SIZE
from utils.rlbench_planning import EndEffectorPoseViaPlanning2


DEFAULT_TASKS = (
    "close_jar", "insert_onto_square_peg", "light_bulb_in",
    "meat_off_grill", "open_drawer", "place_cups",
    "place_shape_in_shape_sorter", "place_wine_at_rack_location",
    "push_buttons", "put_groceries_in_cupboard", "put_item_in_drawer",
    "put_money_in_safe", "reach_and_drag",
    "slide_block_to_color_target", "stack_blocks", "stack_cups",
    "sweep_to_dustpan_of_size", "turn_tap",
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument(
        "--role-config",
        type=Path,
        default=Path(__file__).resolve().parent
        / "configs"
        / "rlbench_o2_semantic_roles.yaml",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("semantic_role_validation")
    )
    parser.add_argument("--num-points", type=int, default=512)
    parser.add_argument("--headless", action="store_true", default=True)
    return parser


def main():
    args = build_parser().parse_args()
    args.output_dir = args.output_dir.resolve()
    obs_config = utils.create_obs_config(
        CAMERAS,
        [IMAGE_SIZE, IMAGE_SIZE],
        method_name="",
        include_masks=True,
    )
    action_mode = MoveArmThenGripper(EndEffectorPoseViaPlanning2(), Discrete())
    env = Environment(
        action_mode=action_mode,
        obs_config=obs_config,
        headless=args.headless,
    )
    provider = RLBenchGTOracleProvider(
        args.role_config,
        num_points=args.num_points,
        cameras=CAMERAS,
        strict=True,
        debug_root=args.output_dir / "audits",
    )
    results = []
    env.launch()
    try:
        for task_name in args.tasks:
            task_env = env.get_task(task_file_to_task_class(task_name))
            variation_count = int(task_env.variation_count())
            for variation in range(variation_count):
                task_env.set_variation(variation)
                _, obs = task_env.reset()
                provider.reset(task_env, task_name, variation, variation)
                provider.set_sample_frame(0)
                provider.enrich(obs, {})
                entry = provider._entries[-1]
                target_handles = set(entry["target"]["handles"])
                reference_handles = set(
                    () if entry["reference"] is None
                    else entry["reference"]["handles"]
                )
                robot_overlap = sorted(
                    (target_handles | reference_handles) & provider._robot_handles
                )
                if robot_overlap:
                    raise RuntimeError(
                        f"{task_name} variation={variation}: robot handles entered "
                        f"semantic roles: {robot_overlap}"
                    )
                results.append(
                    {
                        "task": task_name,
                        "variation": variation,
                        "phase_id": entry["phase_id"],
                        "target": entry["target"],
                        "reference": entry["reference"],
                        "target_visible": entry["target_valid"],
                        "reference_visible": entry["reference_valid"],
                        "robot_overlap": robot_overlap,
                    }
                )
                print(
                    f"{task_name}: variation {variation + 1}/{variation_count} OK",
                    flush=True,
                )
    finally:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        provider.dump(args.output_dir / "semantic_oracle")
        with (args.output_dir / "variation_role_audit.json").open(
            "w", encoding="utf-8"
        ) as stream:
            json.dump(results, stream, indent=2, sort_keys=True)
        env.shutdown()
    print(f"Validated {len(results)} task variations", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
