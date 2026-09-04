"""Strict semantic Target/Reference Oracle provider for RLBench evaluation.

The provider consumes *live* simulator handles and camera masks.  It never
infers semantic roles from proximity, motion, or numeric handle order.  The
versioned YAML contract is shared with the offline replay-field rewriter.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml
from PIL import Image, ImageDraw


DEFAULT_CAMERAS = ("front", "left_shoulder", "right_shoulder", "wrist")
_COPPELIA_SUFFIX = re.compile(r"#\d+$")


class SemanticRoleMappingError(RuntimeError):
    """The configured semantic role cannot be resolved in the live scene."""


@dataclass(frozen=True)
class RoleEntity:
    semantic_name: str
    kind: str
    handles: Tuple[int, ...] = ()
    site_position: Optional[np.ndarray] = None

    def audit_dict(self) -> Dict[str, object]:
        return {
            "semantic_name": self.semantic_name,
            "kind": self.kind,
            "handles": list(self.handles),
            "site_position": (
                None
                if self.site_position is None
                else np.asarray(self.site_position).astype(float).tolist()
            ),
        }


@dataclass(frozen=True)
class RoleAssignment:
    phase_id: str
    target: RoleEntity
    reference: Optional[RoleEntity]
    completion_name: str


def _canonical_name(name: str) -> str:
    return _COPPELIA_SUFFIX.sub("", str(name))


def _object_name(obj) -> str:
    try:
        return str(obj.get_name())
    except Exception:
        return str(obj)


def _object_handle(obj) -> int:
    return int(obj.get_handle())


def _unique_scene_objects(objects: Iterable[object]) -> List[object]:
    """Deduplicate PyRep objects without requiring them to be hashable."""
    unique = []
    seen = set()
    for obj in objects:
        try:
            key = ("handle", _object_handle(obj))
        except Exception:
            key = ("identity", id(obj))
        if key in seen:
            continue
        seen.add(key)
        unique.append(obj)
    return unique


def _object_position(obj) -> np.ndarray:
    value = np.asarray(obj.get_position(), dtype=np.float32).reshape(-1)
    if value.size < 3 or not np.isfinite(value[:3]).all():
        raise SemanticRoleMappingError(
            f"Object {_object_name(obj)!r} has no finite 3-D position"
        )
    return value[:3].copy()


def _condition_met(condition) -> bool:
    if condition is None:
        return False
    try:
        value = condition.condition_met()
    except Exception:
        return False
    if isinstance(value, tuple):
        value = value[0]
    return bool(value)


def _sensor_detects(sensor, obj) -> bool:
    if sensor is None or obj is None:
        return False
    try:
        return bool(sensor.is_detected(obj))
    except Exception:
        return False


def _task_success(task) -> bool:
    try:
        value = task.success()
    except Exception:
        return False
    if isinstance(value, tuple):
        value = value[0]
    return bool(value)


class SceneObjectIndex:
    """Name and hierarchy index for a live RLBench task scene."""

    def __init__(self, task_environment):
        self.task_environment = task_environment
        self.task = getattr(task_environment, "_task", task_environment)
        base = self.task.get_base()
        objects = list(
            base.get_objects_in_tree(
                exclude_base=False, first_generation_only=False
            )
        )
        if base not in objects:
            objects.insert(0, base)
        self.objects = tuple(objects)
        self.by_name: Dict[str, List[object]] = {}
        for obj in self.objects:
            for name in {_object_name(obj), _canonical_name(_object_name(obj))}:
                self.by_name.setdefault(name, []).append(obj)

    def find(self, selector: str) -> List[object]:
        selector = str(selector)
        canonical_selector = _canonical_name(selector)
        if not any(ch in selector for ch in "*?["):
            values = self.by_name.get(selector, self.by_name.get(canonical_selector, []))
            return _unique_scene_objects(values)
        matched = []
        for obj in self.objects:
            name = _canonical_name(_object_name(obj))
            if fnmatch.fnmatchcase(name, canonical_selector):
                matched.append(obj)
        return _unique_scene_objects(matched)

    def require_any(self, selectors: Sequence[str], label: str) -> List[object]:
        matched = []
        for selector in selectors:
            matched.extend(self.find(selector))
        matched = _unique_scene_objects(matched)
        if not matched:
            raise SemanticRoleMappingError(
                f"Could not resolve {label}; selectors={list(selectors)!r}"
            )
        return matched

    @staticmethod
    def handles_with_descendants(objects: Iterable[object]) -> Tuple[int, ...]:
        handles = set()
        for obj in objects:
            try:
                handles.add(_object_handle(obj))
            except Exception:
                continue
            try:
                descendants = obj.get_objects_in_tree(
                    exclude_base=True, first_generation_only=False
                )
            except TypeError:
                try:
                    descendants = obj.get_objects_in_tree()
                except Exception:
                    descendants = ()
            except Exception:
                descendants = ()
            for child in descendants:
                try:
                    handles.add(_object_handle(child))
                except Exception:
                    pass
        return tuple(sorted(handles))


def decode_handle_mask(mask: np.ndarray) -> np.ndarray:
    """Return an integer CoppeliaSim-handle image from RLBench mask output."""
    image = np.asarray(mask)
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]
    if image.ndim == 2:
        return image.astype(np.int64, copy=False)
    if image.ndim != 3 or image.shape[-1] < 3:
        raise ValueError(f"Unsupported RLBench mask shape: {image.shape}")
    encoded = image[..., :3]
    rgb = np.array(encoded, dtype=np.float32, copy=True)
    if np.issubdtype(encoded.dtype, np.integer) or (rgb.size and rgb.max() > 1.0):
        rgb /= 255.0
    try:
        from rlbench.backend.utils import rgb_handles_to_mask

        return np.asarray(rgb_handles_to_mask(rgb), dtype=np.int64)
    except ImportError:
        encoded_int = np.rint(rgb * 255.0).astype(np.int64)
        return (
            encoded_int[..., 0]
            + 256 * encoded_int[..., 1]
            + 65536 * encoded_int[..., 2]
        )


class RLBenchGTOracleProvider:
    """Build direct O2 Target/Reference tensors from live RLBench GT state."""

    def __init__(
        self,
        role_config: Path,
        *,
        num_points: int = 512,
        cameras: Sequence[str] = DEFAULT_CAMERAS,
        strict: bool = True,
        seed: int = 0,
        debug_root: Optional[Path] = None,
    ):
        if num_points <= 0:
            raise ValueError("num_points must be positive")
        self.role_config_path = Path(role_config)
        with self.role_config_path.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        if not isinstance(config, Mapping) or not isinstance(config.get("tasks"), Mapping):
            raise ValueError("semantic role YAML must contain a tasks mapping")
        self.schema_version = str(config.get("schema_version", ""))
        if not self.schema_version:
            raise ValueError("semantic role YAML must define schema_version")
        self.task_specs = dict(config["tasks"])
        self.num_points = int(num_points)
        self.cameras = tuple(cameras)
        self.strict = bool(strict)
        self.seed = int(seed)
        self.debug_root = None if debug_root is None else Path(debug_root)
        self._task_environment = None
        self._task = None
        self._index = None
        self._task_name = ""
        self._variation = -1
        self._episode_idx = -1
        self._phase_index = 0
        self._step_index = 0
        self._sample_frame: Optional[int] = None
        self._expected_sample_frames: Tuple[int, ...] = ()
        self._robot_handles = set()
        self._entries: List[Dict[str, object]] = []
        self._manifests: Dict[Tuple[str, int], Dict[str, object]] = {}
        self.stats = {
            "steps_total": 0,
            "target_valid": 0,
            "reference_valid": 0,
            "fusion_steps": 0,
            "not_visible_target": 0,
            "not_visible_reference": 0,
            "no_reference": 0,
            "mapping_errors": 0,
        }

    def reset(self, task_environment, task_name: str, variation: int, episode_idx: int):
        self._flush_current_manifest()
        self._task_environment = task_environment
        self._task = getattr(task_environment, "_task", task_environment)
        self._task_name = str(task_name)
        self._variation = int(variation)
        self._episode_idx = int(episode_idx)
        self._phase_index = 0
        self._step_index = 0
        self._sample_frame = None
        self._expected_sample_frames = ()
        self._entries = []
        try:
            self._index = SceneObjectIndex(task_environment)
            self._robot_handles = self._collect_robot_handles(task_environment)
            self._task_spec()
            # Validate the first assignment before the policy sees an observation.
            self._build_assignment()
        except Exception:
            self.stats["mapping_errors"] += 1
            if self.strict:
                raise

    def set_sample_frame(self, sample_frame: Optional[int]) -> None:
        """Attach a stored-demo frame index to the next emitted audit entry."""
        self._sample_frame = None if sample_frame is None else int(sample_frame)

    def set_expected_sample_frames(self, sample_frames: Sequence[int]) -> None:
        """Record the complete expert keypoint sequence for manifest validation."""
        self._expected_sample_frames = tuple(int(value) for value in sample_frames)

    @staticmethod
    def _collect_robot_handles(task_environment) -> set:
        robot = getattr(task_environment, "_robot", None)
        if robot is None:
            task = getattr(task_environment, "_task", task_environment)
            robot = getattr(task, "robot", getattr(task, "_robot", None))
        handles = set()
        if robot is None:
            return handles
        for component_name in ("arm", "gripper"):
            component = getattr(robot, component_name, None)
            if component is None:
                continue
            try:
                objects = component.get_objects_in_tree(exclude_base=False)
            except Exception:
                objects = ()
            for obj in objects:
                try:
                    handles.add(_object_handle(obj))
                except Exception:
                    pass
        return handles

    def _task_spec(self) -> Mapping[str, object]:
        try:
            return self.task_specs[self._task_name]
        except KeyError as exc:
            raise SemanticRoleMappingError(
                f"No semantic T/R mapping for task {self._task_name!r}"
            ) from exc

    def _phase_count(self) -> int:
        task = self._task
        if self._task_name == "place_cups":
            return max(1, int(getattr(task, "_index", self._variation)) + 1)
        if self._task_name == "push_buttons":
            return max(1, int(getattr(task, "buttons_to_push", 1)))
        if self._task_name == "stack_blocks":
            return max(1, int(getattr(task, "blocks_to_stack", 2)))
        if self._task_name == "stack_cups":
            return 2
        return 1

    def _objects(self, selectors: Sequence[str], label: str) -> List[object]:
        assert self._index is not None
        return self._index.require_any(selectors, label)

    def _optional_objects(self, selectors: Sequence[str]) -> List[object]:
        assert self._index is not None
        matched = []
        for selector in selectors:
            matched.extend(self._index.find(selector))
        return _unique_scene_objects(matched)

    @staticmethod
    def _expect_count(objects: Sequence[object], expected: int, label: str):
        if len(objects) != expected:
            raise SemanticRoleMappingError(
                f"{label} expected {expected} simulator objects, got {len(objects)}"
            )
        return list(objects)

    def _entity_object(
        self, semantic_name: str, objects: Sequence[object]
    ) -> RoleEntity:
        handles = set(SceneObjectIndex.handles_with_descendants(objects))
        handles.difference_update(self._robot_handles)
        if not handles:
            raise SemanticRoleMappingError(
                f"Semantic object {semantic_name!r} has no non-robot handles"
            )
        return RoleEntity(semantic_name, "object", tuple(sorted(handles)))

    def _entity_site(self, semantic_name: str, obj) -> RoleEntity:
        return RoleEntity(
            semantic_name, "site", (), _object_position(obj)
        )

    def _attr_objects(self, *attribute_names: str) -> List[object]:
        for attribute_name in attribute_names:
            value = getattr(self._task, attribute_name, None)
            if value is None:
                continue
            if isinstance(value, (list, tuple)):
                return list(value)
            return [value]
        return []

    def _variant_names(self, role_spec: Mapping[str, object]) -> List[str]:
        if "names_by_variation_mod_2" in role_spec:
            values = role_spec["names_by_variation_mod_2"][self._variation % 2]
        elif "names_by_variation" in role_spec:
            table = role_spec["names_by_variation"]
            values = table[self._variation % len(table)]
        else:
            values = role_spec.get("names", ())
        return [str(value) for value in (values if isinstance(values, list) else [values])]

    def _site_from_spec(self, spec: Mapping[str, object], label: str) -> RoleEntity:
        names = self._variant_names(spec)
        if not names:
            names = [str(value) for value in spec.get("candidates", ())]
        objects = self._objects(names, label)
        # A site selector is ordered.  Multiple aliases may resolve to the same
        # semantic site; the first configured match is authoritative.
        selected = None
        for name in names:
            matches = self._index.find(name)
            if matches:
                selected = matches[0]
                break
        if selected is None:
            selected = objects[0]
        semantic_name = str(spec.get("semantic_name", _canonical_name(_object_name(selected))))
        return self._entity_site(semantic_name, selected)

    def _object_from_spec(self, spec: Mapping[str, object], label: str) -> RoleEntity:
        names = self._variant_names(spec)
        objects = self._objects(names, label)
        semantic_name = "+".join(_canonical_name(_object_name(obj)) for obj in objects)
        return self._entity_object(semantic_name, objects)

    def _sequence_entity(self, spec, index: int, label: str) -> RoleEntity:
        name = str(spec["sequence"][index])
        return self._entity_object(name, self._objects([name], label))

    def _build_assignment(self) -> RoleAssignment:
        spec = self._task_spec()
        name = self._task_name
        phase = min(self._phase_index, self._phase_count() - 1)
        target_spec = spec["target"]
        reference_spec = spec.get("reference")
        target: RoleEntity
        reference: Optional[RoleEntity]

        if name == "close_jar":
            target_objects = self._attr_objects("lid") or self._objects(
                target_spec["names"], "close_jar target lid"
            )
            target = self._entity_object("jar_lid", target_objects)
            jars = self._attr_objects("jars")
            if jars:
                jars = self._expect_count(jars, 2, "close_jar jars")
                ref_objects = [jars[self._variation % 2]]
            else:
                ref_objects = self._objects(
                    self._variant_names(reference_spec), "close_jar target jar"
                )
            reference = self._entity_object("target_jar", ref_objects)
        elif name == "insert_onto_square_peg":
            target_objects = self._attr_objects("_square_ring") or self._objects(
                target_spec["names"], "square ring"
            )
            target = self._entity_object("square_ring", target_objects)
            pillars = self._objects(reference_spec["candidates"], "square peg pillars")
            pillars = self._expect_count(pillars, 3, "square peg pillars")
            centres = self._attr_objects("_success_centre") or self._objects(
                ["success_centre"], "square peg success centre"
            )
            centre = _object_position(centres[0])
            aligned = [
                obj
                for obj in pillars
                if np.allclose(
                    _object_position(obj)[:2], centre[:2], atol=1e-4, rtol=0.0
                )
            ]
            if len(aligned) != 1:
                raise SemanticRoleMappingError(
                    "insert_onto_square_peg requires exactly one pillar aligned "
                    f"with success_centre; matched {len(aligned)}"
                )
            pillar = aligned[0]
            reference = self._entity_object(_canonical_name(_object_name(pillar)), [pillar])
        elif name == "light_bulb_in":
            bulbs = self._attr_objects("bulbs")
            visuals = self._attr_objects("bulbs_visual") + self._attr_objects("bulb_glass_visual")
            if bulbs:
                bulbs = self._expect_count(bulbs, 2, "light_bulb_in physical bulbs")
            if visuals:
                visuals = self._expect_count(visuals, 4, "light_bulb_in visual bulbs")
            selected = [bulbs[self._variation % 2]] if bulbs else []
            if visuals:
                selected.extend(obj for i, obj in enumerate(visuals) if i % 2 == self._variation % 2)
            if not selected:
                selected = self._objects(self._variant_names(target_spec), "selected light bulb")
            target = self._entity_object("selected_light_bulb", selected)
            reference = self._site_from_spec(reference_spec, "lamp socket success site")
        elif name == "meat_off_grill":
            if self._variation == 0:
                selected = self._attr_objects("_chicken")
            else:
                selected = self._attr_objects("_steak")
            if not selected:
                selected = self._objects(self._variant_names(target_spec), "selected meat")
            target = self._entity_object("chicken" if self._variation == 0 else "steak", selected)
            reference = self._site_from_spec(reference_spec, "off-grill success site")
        elif name == "place_cups":
            cups = self._attr_objects("_cups")
            spokes = self._attr_objects("_spokes")
            if cups:
                cups = self._expect_count(cups, 3, "place_cups mugs")
            if spokes:
                spokes = self._expect_count(spokes, 3, "place_cups spokes")
            target = (
                self._entity_object(f"mug{phase}", [cups[phase]])
                if cups else self._sequence_entity(target_spec, phase, "place_cups mug")
            )
            reference = (
                self._entity_object(f"holder_spoke{phase}", [spokes[phase]])
                if spokes else self._sequence_entity(reference_spec, phase, "place_cups spoke")
            )
        elif name == "place_shape_in_shape_sorter":
            shapes = self._attr_objects("shapes")
            if shapes:
                shapes = self._expect_count(shapes, 5, "shape sorter shapes")
            target = (
                self._entity_object("selected_shape", [shapes[self._variation]])
                if shapes else self._object_from_spec(target_spec, "selected sorter shape")
            )
            drops = self._attr_objects("drop_points")
            if drops:
                drops = self._expect_count(drops, 5, "shape sorter drop points")
            reference = (
                self._entity_site("sorter_slot", drops[self._variation])
                if drops else self._site_from_spec(reference_spec, "sorter drop point")
            )
        elif name == "place_wine_at_rack_location":
            selected = self._attr_objects("wine_bottle") or self._objects(
                target_spec["names"], "wine bottle"
            )
            target = self._entity_object("wine_bottle", selected)
            reference = self._site_from_spec(reference_spec, "wine rack location")
        elif name == "push_buttons":
            plates = self._attr_objects("target_topPlates")
            if plates:
                plates = self._expect_count(plates, 3, "push button top plates")
            target = (
                self._entity_object(f"button{phase}", [plates[phase]])
                if plates else self._sequence_entity(target_spec, phase, "active button")
            )
            reference = None
        elif name == "put_groceries_in_cupboard":
            groceries = self._attr_objects("groceries")
            if groceries:
                groceries = self._expect_count(groceries, 9, "cupboard groceries")
            target = (
                self._entity_object("selected_grocery", [groceries[self._variation]])
                if groceries else self._object_from_spec(target_spec, "selected grocery")
            )
            reference = self._site_from_spec(reference_spec, "cupboard success site")
        elif name == "put_item_in_drawer":
            selected = self._attr_objects("_item") or self._objects(
                target_spec["names"], "drawer item"
            )
            target = self._entity_object("item", selected)
            drawer_joints = self._attr_objects("_joints")
            if drawer_joints:
                drawer_joints = self._expect_count(
                    drawer_joints, 3, "drawer joints"
                )
            reference = (
                self._entity_object(
                    f"{('bottom', 'middle', 'top')[self._variation]}_drawer",
                    [drawer_joints[self._variation]]
                    + self._optional_objects(self._variant_names(reference_spec)),
                )
                if drawer_joints
                else self._object_from_spec(
                    reference_spec, "selected drawer interior"
                )
            )
        elif name == "put_money_in_safe":
            selected = self._attr_objects("money") or self._objects(
                target_spec["names"], "money"
            )
            target = self._entity_object("dollar_stack", selected)
            reference = self._site_from_spec(reference_spec, "safe shelf")
        elif name == "reach_and_drag":
            selected = self._attr_objects("stick") or self._objects(
                target_spec["names"], "drag stick"
            )
            target = self._entity_object("stick", selected)
            refs = self._attr_objects("target") or self._objects(
                reference_spec["names"], "drag color target"
            )
            reference = self._entity_object("color_target", refs)
        elif name == "slide_block_to_color_target":
            selected = self._attr_objects("_block", "block") or self._objects(
                target_spec["names"], "slide block"
            )
            target = self._entity_object("block", selected)
            reference = self._site_from_spec(reference_spec, "slide color target")
        elif name == "stack_blocks":
            blocks = self._attr_objects("target_blocks")
            if blocks:
                blocks = self._expect_count(blocks, 4, "stack target blocks")
            target = (
                self._entity_object(f"stack_block{phase}", [blocks[phase]])
                if blocks else self._sequence_entity(target_spec, phase, "stack target block")
            )
            if phase == 0:
                reference = self._object_from_spec(spec["first_reference"], "stack target plane")
            else:
                previous = blocks[phase - 1] if blocks else self._objects(
                    [target_spec["sequence"][phase - 1]], "previous stack block"
                )[0]
                reference = self._entity_object(f"stack_block{phase - 1}", [previous])
        elif name == "stack_cups":
            target = self._sequence_entity(target_spec, phase, "stack cup target")
            reference = self._sequence_entity(reference_spec, phase, "stack cup reference")
        elif name == "sweep_to_dustpan_of_size":
            selected = self._objects(target_spec["names"], "broom")
            target = self._entity_object("broom", selected)
            site_spec = dict(reference_spec)
            preferred = [f"success{self._variation}", "success"]
            preferred.extend(reference_spec.get("candidates", ()))
            site_spec["names"] = list(dict.fromkeys(preferred))
            reference = self._site_from_spec(site_spec, "dustpan success site")
        elif name == "turn_tap":
            selected_joint = getattr(
                self._task,
                "left_joint" if self._variation == 0 else "right_joint",
                None,
            )
            target = (
                self._entity_object(
                    "left_tap_handle" if self._variation == 0 else "right_tap_handle",
                    [selected_joint]
                    + self._optional_objects(self._variant_names(target_spec)),
                )
                if selected_joint is not None
                else self._object_from_spec(target_spec, "selected tap handle")
            )
            reference = None
        elif name == "open_drawer":
            drawer_joints = self._attr_objects("_joints")
            if drawer_joints:
                drawer_joints = self._expect_count(
                    drawer_joints, 3, "drawer joints"
                )
            target = (
                self._entity_object(
                    f"{('bottom', 'middle', 'top')[self._variation]}_drawer",
                    [drawer_joints[self._variation]]
                    + self._optional_objects(self._variant_names(target_spec)),
                )
                if drawer_joints
                else self._object_from_spec(target_spec, "selected drawer link")
            )
            reference = None
        else:
            target = self._object_from_spec(target_spec, f"{name} target")
            if reference_spec is None:
                reference = None
            elif reference_spec.get("kind") == "site":
                reference = self._site_from_spec(reference_spec, f"{name} reference site")
            else:
                reference = self._object_from_spec(reference_spec, f"{name} reference")

        return RoleAssignment(
            phase_id=f"{name}:{phase}",
            target=target,
            reference=reference,
            completion_name=str(spec.get("completion", "task_success")),
        )

    def _phase_complete(self, assignment: RoleAssignment, obs) -> bool:
        task = self._task
        phase = self._phase_index
        released = bool(float(getattr(obs, "gripper_open", 0.0)) > 0.5)
        if self._task_name == "place_cups":
            conditions = getattr(task, "_on_peg_conditions", ())
            return phase < len(conditions) and _condition_met(conditions[phase]) and released
        if self._task_name == "push_buttons":
            conditions = getattr(task, "goal_conditions", ())
            return phase < len(conditions) and _condition_met(conditions[phase])
        if self._task_name == "stack_blocks":
            blocks = self._attr_objects("target_blocks")
            sensors = self._index.find("stack_blocks_success") if self._index else []
            if phase == 0:
                references = self._index.find("stack_blocks_target_plane") if self._index else []
            else:
                references = blocks[phase - 1:phase]
            above_reference = bool(
                phase < len(blocks)
                and references
                and _object_position(blocks[phase])[2]
                > _object_position(references[0])[2] + 0.01
            )
            return (
                phase < len(blocks)
                and bool(sensors)
                and _sensor_detects(sensors[0], blocks[phase])
                and above_reference
                and released
            )
        if self._task_name == "stack_cups":
            cup_name = ("cup1", "cup3")[min(phase, 1)]
            cups = self._index.find(cup_name) if self._index else []
            sensors = self._index.find("success") if self._index else []
            return bool(cups and sensors and _sensor_detects(sensors[0], cups[0]) and released)
        return _task_success(task)

    def _sample_entity_points(
        self,
        entity: Optional[RoleEntity],
        masks: Mapping[str, np.ndarray],
        point_clouds: Mapping[str, np.ndarray],
    ) -> Tuple[np.ndarray, bool]:
        if entity is None:
            return np.zeros((self.num_points, 3), dtype=np.float32), False
        if entity.kind == "site":
            point = np.asarray(entity.site_position, dtype=np.float32).reshape(1, 3)
            return np.repeat(point, self.num_points, axis=0), True
        collected = []
        handles = np.asarray(entity.handles, dtype=np.int64)
        for camera in self.cameras:
            if camera not in masks or camera not in point_clouds:
                continue
            mask = decode_handle_mask(masks[camera])
            cloud = np.asarray(point_clouds[camera])
            if cloud.ndim == 3 and cloud.shape[0] == 3 and cloud.shape[-1] != 3:
                cloud = np.moveaxis(cloud, 0, -1)
            if cloud.shape[:2] != mask.shape or cloud.shape[-1] != 3:
                raise ValueError(
                    f"{camera} mask/point-cloud mismatch: {mask.shape} vs {cloud.shape}"
                )
            points = cloud[np.isin(mask, handles)]
            points = points[np.isfinite(points).all(axis=1)]
            if points.size:
                collected.append(points.astype(np.float32, copy=False))
        if not collected:
            return np.zeros((self.num_points, 3), dtype=np.float32), False
        points = np.concatenate(collected, axis=0)
        seed_material = (
            f"{self.seed}|{self._task_name}|{self._episode_idx}|"
            f"{self._step_index}|{entity.semantic_name}"
        ).encode("utf-8")
        digest = hashlib.sha256(seed_material).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        indices = rng.choice(len(points), size=self.num_points, replace=len(points) < self.num_points)
        return points[indices].astype(np.float32, copy=False), True

    def enrich(self, obs, obs_dict: Mapping[str, object]) -> Dict[str, object]:
        result = dict(obs_dict)
        masks = {
            camera: getattr(obs, f"{camera}_mask")
            for camera in self.cameras
            if getattr(obs, f"{camera}_mask", None) is not None
        }
        point_clouds = {
            camera: getattr(obs, f"{camera}_point_cloud")
            for camera in self.cameras
            if getattr(obs, f"{camera}_point_cloud", None) is not None
        }
        try:
            assignment = self._build_assignment()
            completion_satisfied = self._phase_complete(assignment, obs)
            phase_advanced = False
            if (
                self._phase_index + 1 < self._phase_count()
                and completion_satisfied
            ):
                self._phase_index += 1
                assignment = self._build_assignment()
                phase_advanced = True
            target_points, target_valid = self._sample_entity_points(
                assignment.target, masks, point_clouds
            )
            reference_points, reference_valid = self._sample_entity_points(
                assignment.reference, masks, point_clouds
            )
        except Exception:
            self.stats["mapping_errors"] += 1
            if self.strict:
                raise
            assignment = RoleAssignment(
                f"{self._task_name}:invalid",
                RoleEntity("invalid", "object"),
                None,
                "mapping_error",
            )
            target_points = np.zeros((self.num_points, 3), dtype=np.float32)
            reference_points = np.zeros((self.num_points, 3), dtype=np.float32)
            target_valid = reference_valid = False
            completion_satisfied = False
            phase_advanced = False

        result["oracle_target_object_points"] = target_points
        result["oracle_reference_object_points"] = reference_points
        result["oracle_target_object_valid"] = np.asarray(target_valid, dtype=np.bool_)
        result["oracle_reference_object_valid"] = np.asarray(reference_valid, dtype=np.bool_)

        self.stats["steps_total"] += 1
        self.stats["target_valid"] += int(target_valid)
        self.stats["reference_valid"] += int(reference_valid)
        self.stats["fusion_steps"] += int(target_valid or reference_valid)
        self.stats["not_visible_target"] += int(not target_valid)
        if assignment.reference is None:
            self.stats["no_reference"] += 1
        else:
            self.stats["not_visible_reference"] += int(not reference_valid)

        entry = {
            "step": self._step_index,
            "sample_frame": self._sample_frame,
            "phase_id": assignment.phase_id,
            "completion": assignment.completion_name,
            "completion_satisfied": bool(completion_satisfied),
            "phase_advanced": bool(phase_advanced),
            "target": assignment.target.audit_dict(),
            "reference": (
                None if assignment.reference is None else assignment.reference.audit_dict()
            ),
            "target_valid": bool(target_valid),
            "reference_valid": bool(reference_valid),
        }
        self._entries.append(entry)
        if self.debug_root is not None and self._step_index == 0:
            self._write_role_audit(
                obs,
                assignment,
                masks,
                target_points,
                reference_points,
                target_valid,
                reference_valid,
                completion_satisfied,
                phase_advanced,
            )
        self._step_index += 1
        return result

    @staticmethod
    def _labeled_panel(image: np.ndarray, label: str) -> Image.Image:
        panel = Image.fromarray(np.asarray(image, dtype=np.uint8))
        draw = ImageDraw.Draw(panel)
        draw.rectangle((0, 0, panel.width, 16), fill=(0, 0, 0))
        draw.text((3, 2), label, fill=(255, 255, 255))
        return panel

    @staticmethod
    def _point_projection_panel(
        target_points,
        reference_points,
        target_valid,
        reference_valid,
        axes,
        limits,
        size,
        label,
    ):
        panel = Image.new("RGB", size, (238, 238, 238))
        draw = ImageDraw.Draw(panel)
        x_axis, y_axis = axes
        (x_min, x_max), (y_min, y_max) = limits

        def draw_points(points, valid, color):
            if not valid:
                return
            for point in np.asarray(points)[:: max(1, len(points) // 256)]:
                x = int((float(point[x_axis]) - x_min) / (x_max - x_min) * (size[0] - 1))
                y = int((1.0 - (float(point[y_axis]) - y_min) / (y_max - y_min)) * (size[1] - 1))
                if 0 <= x < size[0] and 16 <= y < size[1]:
                    draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=color)

        draw_points(reference_points, reference_valid, (50, 100, 255))
        draw_points(target_points, target_valid, (255, 50, 50))
        draw.rectangle((0, 0, size[0], 16), fill=(0, 0, 0))
        draw.text((3, 2), label, fill=(255, 255, 255))
        return panel

    def _write_role_audit(
        self,
        obs,
        assignment,
        masks,
        target_points,
        reference_points,
        target_valid,
        reference_valid,
        completion_satisfied,
        phase_advanced,
    ):
        panels = []
        target_handles = np.asarray(assignment.target.handles, dtype=np.int64)
        reference_handles = np.asarray(
            () if assignment.reference is None else assignment.reference.handles,
            dtype=np.int64,
        )
        first_detail = None
        for camera in self.cameras:
            rgb = getattr(obs, f"{camera}_rgb", None)
            mask_value = masks.get(camera)
            if rgb is None or mask_value is None:
                continue
            image = np.asarray(rgb)
            if np.issubdtype(image.dtype, np.floating) and image.size and image.max() <= 1:
                image = image * 255.0
            image = np.clip(image, 0, 255).astype(np.uint8)
            mask = decode_handle_mask(mask_value)
            overlay = image.astype(np.float32)
            if target_handles.size:
                selected = np.isin(mask, target_handles)
                overlay[selected] = 0.45 * overlay[selected] + 0.55 * np.array([255, 64, 64])
            if reference_handles.size:
                selected = np.isin(mask, reference_handles)
                overlay[selected] = 0.45 * overlay[selected] + 0.55 * np.array([64, 128, 255])
            panels.append(
                self._labeled_panel(
                    np.clip(overlay, 0, 255).astype(np.uint8),
                    f"{camera}: T/R overlay",
                )
            )
            if first_detail is None:
                first_detail = camera, image, mask
        if not panels:
            return
        camera, image, mask = first_detail
        palette = np.zeros((*mask.shape, 3), dtype=np.uint8)
        nonzero = mask != 0
        palette[..., 0] = (mask * 37 % 251).astype(np.uint8)
        palette[..., 1] = (mask * 67 % 251).astype(np.uint8)
        palette[..., 2] = (mask * 97 % 251).astype(np.uint8)
        palette[~nonzero] = 0
        target_mask = np.zeros_like(palette)
        target_mask[np.isin(mask, target_handles)] = (255, 64, 64)
        reference_mask = np.zeros_like(palette)
        reference_mask[np.isin(mask, reference_handles)] = (64, 128, 255)
        panels.extend(
            [
                self._labeled_panel(image, f"{camera}: original"),
                self._labeled_panel(palette, f"{camera}: instance handles"),
                self._labeled_panel(target_mask, f"{camera}: Target mask"),
                self._labeled_panel(reference_mask, f"{camera}: Reference mask"),
            ]
        )
        width = max(panel.width for panel in panels)
        height = max(panel.height for panel in panels)
        panels.extend(
            [
                self._point_projection_panel(
                    target_points, reference_points, target_valid, reference_valid,
                    (0, 1), ((-0.3, 0.7), (-0.5, 0.5)), (width, height), "T/R cloud: XY",
                ),
                self._point_projection_panel(
                    target_points, reference_points, target_valid, reference_valid,
                    (0, 2), ((-0.3, 0.7), (0.6, 1.6)), (width, height), "T/R cloud: XZ",
                ),
                self._point_projection_panel(
                    target_points, reference_points, target_valid, reference_valid,
                    (1, 2), ((-0.5, 0.5), (0.6, 1.6)), (width, height), "T/R cloud: YZ",
                ),
            ]
        )
        columns = 4
        rows = (len(panels) + columns - 1) // columns
        canvas = Image.new(
            "RGB", (columns * width, rows * height + 34), (32, 32, 32)
        )
        for index, panel in enumerate(panels):
            canvas.paste(
                panel,
                ((index % columns) * width, (index // columns) * height + 34),
            )
        draw = ImageDraw.Draw(canvas)
        reference_name = "none" if assignment.reference is None else assignment.reference.semantic_name
        draw.text(
            (4, 4),
            f"{assignment.phase_id}  T={assignment.target.semantic_name}  "
            f"R={reference_name}  condition={int(completion_satisfied)}  "
            f"advanced={int(phase_advanced)}",
            fill=(255, 255, 255),
        )
        output = self.debug_root / self._task_name / f"episode_{self._episode_idx}"
        output.mkdir(parents=True, exist_ok=True)
        canvas.save(output / "role_audit_step_000.png")

    def _flush_current_manifest(self):
        if not self._task_name or self._episode_idx < 0:
            return
        self._manifests[(self._task_name, self._episode_idx)] = {
            "schema_version": self.schema_version,
            "task": self._task_name,
            "episode_idx": self._episode_idx,
            "variation": self._variation,
            "expected_sample_frames": list(self._expected_sample_frames),
            "entries": list(self._entries),
        }

    def dump(self, output_dir: Path):
        self._flush_current_manifest()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "oracle_provider_stats.json").open("w", encoding="utf-8") as stream:
            json.dump(
                {
                    "schema_version": self.schema_version,
                    "role_config": str(self.role_config_path),
                    **self.stats,
                },
                stream,
                indent=2,
                sort_keys=True,
            )
        manifest_root = output_dir / "semantic_role_manifests"
        for (task, episode_idx), manifest in self._manifests.items():
            task_dir = manifest_root / task
            task_dir.mkdir(parents=True, exist_ok=True)
            with (task_dir / f"episode_{episode_idx}.json").open(
                "w", encoding="utf-8"
            ) as stream:
                json.dump(manifest, stream, indent=2, sort_keys=True)
