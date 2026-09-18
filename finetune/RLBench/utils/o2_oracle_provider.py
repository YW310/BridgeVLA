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
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml
from PIL import Image, ImageDraw
from .oracle_handle_alignment import (
    align_handles, align_semantic_handle_group, HandleAlignmentError)
from .site_geometry import (
    DEFAULT_FALLBACK_EXTENT_M,
    SEMANTIC_ROLE_SCHEMA,
    SITE_GEOMETRY_PRIMITIVE,
    SiteGeometry,
    sample_site_geometry,
    site_geometry_from_object,
    validate_fallback_extent,
)


DEFAULT_CAMERAS = ("front", "left_shoulder", "right_shoulder", "wrist")
_COPPELIA_SUFFIX = re.compile(r"#\d+$")
_TASK_RESOLVER_VERSIONS = {
    # v1 trusted private _cups/_spokes order and expanded spoke descendants.
    # v2 fixed canonical entities but treated every close-to-open transition as
    # completion. v3 filters extra releases by the fixed ordered T/R relation.
    # v4 uses simulator detector conditions online and recovers missing stored-
    # demo releases from ordered reference/keypoint evidence.
    "place_cups": "place_cups_detector_relation_v4",
    # v1 used the top-plate mask handle.  Some RLBench scenes expose fewer
    # than two pixels for that live handle even though the semantic contact
    # position is available directly from the task object.
    "push_buttons": "push_buttons_contact_site_v2",
    # v1 used a drawer-shell mask as Reference.  The task's actual terminal
    # relation is item detection by the variation-specific success sensor.
    "put_item_in_drawer": "put_item_in_drawer_success_site_v2",
}


class SemanticRoleMappingError(RuntimeError):
    """The configured semantic role cannot be resolved in the live scene."""


@dataclass(frozen=True)
class RoleEntity:
    semantic_name: str
    kind: str
    handles: Tuple[int, ...] = ()
    site_position: Optional[np.ndarray] = None
    site_geometry: Optional[SiteGeometry] = None

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
            'site_geometry': (
                None
                if self.site_geometry is None
                else self.site_geometry.audit_dict()
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
        handle_alignment: str = "identity",
        handle_map_dir: Optional[Path] = None,
        alignment_output_dir: Optional[Path] = None,
        manifest_output_dir: Optional[Path] = None,
    ):
        if num_points <= 0:
            raise ValueError("num_points must be positive")
        self.role_config_path = Path(role_config)
        self.role_config_sha256 = hashlib.sha256(
            self.role_config_path.read_bytes()).hexdigest()
        with self.role_config_path.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        if not isinstance(config, Mapping) or not isinstance(config.get("tasks"), Mapping):
            raise ValueError("semantic role YAML must contain a tasks mapping")
        self.schema_version = str(config.get("schema_version", ""))
        if self.schema_version != SEMANTIC_ROLE_SCHEMA:
            raise ValueError(
                f"unsupported semantic role schema {self.schema_version!r}; "
                f"expected {SEMANTIC_ROLE_SCHEMA!r}"
            )
        self.task_specs = dict(config["tasks"])
        site_defaults = config.get("site_geometry_defaults", {})
        if not isinstance(site_defaults, Mapping):
            raise ValueError("site_geometry_defaults must be a mapping")
        primitive = str(
            site_defaults.get("primitive", SITE_GEOMETRY_PRIMITIVE)
        )
        if primitive != SITE_GEOMETRY_PRIMITIVE:
            raise ValueError(
                f"unsupported site geometry primitive {primitive!r}; "
                f"expected {SITE_GEOMETRY_PRIMITIVE!r}"
            )
        self.site_fallback_extent_m = validate_fallback_extent(
            site_defaults.get(
                "fallback_extent_m", DEFAULT_FALLBACK_EXTENT_M
            )
        )
        self.num_points = int(num_points)
        self.cameras = tuple(cameras)
        self.strict = bool(strict)
        self.seed = int(seed)
        self.debug_root = None if debug_root is None else Path(debug_root)
        if handle_alignment not in ("identity", "verified", "mask_verified"):
            raise ValueError("handle_alignment must be identity, verified or mask_verified")
        self.handle_alignment = handle_alignment
        self.handle_map_dir = None if handle_map_dir is None else Path(handle_map_dir)
        self.alignment_output_dir = (
            None if alignment_output_dir is None else Path(alignment_output_dir))
        self.manifest_output_dir = (
            None if manifest_output_dir is None else Path(manifest_output_dir))
        self._live_initial_views = None
        self._stored_handle_map = None
        self._stored_entity_handle_map = {}
        self._nonvisual_handles = set()
        self._handle_alignment_audit = {}
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
        self._generation_attempt = 1
        self._source_alignment_validated = False
        self._demo_phase_metadata: Dict[str, object] = {}
        self._robot_handles = set()
        self._entries: List[Dict[str, object]] = []
        self._manifests: Dict[Tuple[str, int], Dict[str, object]] = {}
        self._current_manifest_discarded = False
        self.stats = {
            "steps_total": 0,
            "target_valid": 0,
            "reference_valid": 0,
            "prior_steps": 0,
            "not_visible_target": 0,
            "not_visible_reference": 0,
            "no_reference": 0,
            "mapping_errors": 0,
            "discarded_attempts": 0,
        }

    def reset(
        self,
        task_environment,
        task_name: str,
        variation: int,
        episode_idx: int,
        discard_current: bool = False,
        generation_attempt: int = 1,
    ):
        if discard_current:
            self.stats["discarded_attempts"] += int(bool(self._entries))
            self._manifests.pop((str(task_name), int(episode_idx)), None)
        else:
            self._flush_current_manifest()
        self._task_environment = task_environment
        self._task = getattr(task_environment, "_task", task_environment)
        self._task_name = str(task_name)
        self._variation = int(variation)
        self._episode_idx = int(episode_idx)
        self._generation_attempt = int(generation_attempt)
        self._source_alignment_validated = False
        self._demo_phase_metadata = {}
        self._live_initial_views = None
        self._stored_handle_map = None
        self._stored_entity_handle_map = {}
        self._nonvisual_handles = set()
        self._handle_alignment_audit = {}
        self._phase_index = 0
        self._step_index = 0
        self._sample_frame = None
        self._expected_sample_frames = ()
        self._entries = []
        self._current_manifest_discarded = False
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

    def discard_current_manifest(self) -> None:
        '''Prevent a failed demo-event attempt from being serialized later.'''
        if self._task_name and self._episode_idx >= 0:
            self._manifests.pop((self._task_name, self._episode_idx), None)
        self.stats['discarded_attempts'] += int(bool(self._entries))
        self._entries = []
        self._expected_sample_frames = ()
        self._source_alignment_validated = False
        self._demo_phase_metadata = {}
        self._current_manifest_discarded = True

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

    @staticmethod
    def task_resolver_version(task_name: str) -> Optional[str]:
        """Return a resume contract only for task resolvers with migrations."""
        return _TASK_RESOLVER_VERSIONS.get(str(task_name))

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
        self, semantic_name: str, objects: Sequence[object], *,
        include_descendants: bool = True,
    ) -> RoleEntity:
        if include_descendants:
            handles = set(SceneObjectIndex.handles_with_descendants(objects))
        else:
            handles = set()
            for obj in objects:
                try:
                    handles.add(_object_handle(obj))
                except Exception:
                    continue
        handles.difference_update(self._robot_handles)
        original_handles = frozenset(handles)
        if self._stored_handle_map is not None:
            entity_handles = self._stored_entity_handle_map.get(original_handles)
            if entity_handles is not None:
                handles = set(entity_handles)
            else:
                handles.difference_update(self._nonvisual_handles)
                missing = handles.difference(self._stored_handle_map)
                if missing:
                    raise SemanticRoleMappingError(
                        f"Unmapped stored-demo handles for {semantic_name}: "
                        f"{sorted(missing)}")
                handles = {self._stored_handle_map[handle] for handle in handles}
        if not handles:
            raise SemanticRoleMappingError(
                f"Semantic object {semantic_name!r} has no non-robot handles"
            )
        return RoleEntity(semantic_name, "object", tuple(sorted(handles)))

    def _entity_site(
        self,
        semantic_name: str,
        obj,
        geometry_spec: Optional[Mapping[str, object]] = None,
    ) -> RoleEntity:
        if geometry_spec is not None and not isinstance(geometry_spec, Mapping):
            raise SemanticRoleMappingError(
                f"{semantic_name} site_geometry must be a mapping"
            )
        geometry_spec = geometry_spec or {}
        primitive = str(
            geometry_spec.get("primitive", SITE_GEOMETRY_PRIMITIVE)
        )
        if primitive != SITE_GEOMETRY_PRIMITIVE:
            raise SemanticRoleMappingError(
                f"unsupported site geometry primitive {primitive!r}"
            )
        try:
            fallback_extent = validate_fallback_extent(
                geometry_spec.get(
                    "fallback_extent_m", self.site_fallback_extent_m
                )
            )
            site_position = _object_position(obj)
            geometry = site_geometry_from_object(
                obj, site_position, fallback_extent
            )
        except ValueError as exc:
            raise SemanticRoleMappingError(
                f"invalid site geometry for {semantic_name!r}: {exc}"
            ) from exc
        return RoleEntity(
            semantic_name=semantic_name,
            kind="site",
            site_position=site_position,
            site_geometry=geometry,
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
        return self._entity_site(
            semantic_name, selected, spec.get("site_geometry")
        )

    def _object_from_spec(self, spec: Mapping[str, object], label: str) -> RoleEntity:
        names = self._variant_names(spec)
        objects = self._objects(names, label)
        semantic_name = "+".join(_canonical_name(_object_name(obj)) for obj in objects)
        return self._entity_object(semantic_name, objects)

    def _sequence_entity(
        self, spec, index: int, label: str, *,
        include_descendants: bool = True,
        semantic_name: Optional[str] = None,
    ) -> RoleEntity:
        name = str(spec["sequence"][index])
        return self._entity_object(
            name if semantic_name is None else str(semantic_name),
            self._objects([name], label),
            include_descendants=include_descendants)

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
            # The semantic order is defined by the configured canonical names,
            # not by the task's private _cups/_spokes container order. Some
            # vendored scene/task versions expose those containers in a
            # different order, which can otherwise label spoke2 as spoke0.
            target = self._sequence_entity(
                target_spec, phase, "place_cups mug")
            reference = self._sequence_entity(
                reference_spec, phase, "place_cups spoke",
                include_descendants=False,
                semantic_name=f"holder_spoke{phase}")
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
                self._entity_site(
                    "sorter_slot",
                    drops[self._variation],
                    reference_spec.get("site_geometry"),
                )
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
            if not plates:
                plate_name = str(target_spec["sequence"][phase])
                plates = self._expect_count(
                    self._objects([plate_name], "active button contact plate"),
                    1,
                    "active button contact plate",
                )
            # The contact plate's simulator object is authoritative for phase
            # order, but its GT mask handle can be effectively unobservable at
            # reset.  The exact object position is a better Oracle interaction
            # site and avoids inventing a cross-session handle correspondence.
            target = self._entity_site(
                f"button{phase}_contact_site",
                plates[phase],
                target_spec.get("site_geometry"),
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
            success_detectors = _unique_scene_objects([
                detector
                for condition in getattr(self._task, "_success_conditions", ())
                for detector in (getattr(condition, "_detector", None),)
                if detector is not None
            ])
            success_detector = self._expect_count(
                success_detectors, 1,
                "put_item_in_drawer success detector",
            )[0]
            option = ("bottom", "middle", "top")[self._variation]
            reference = self._entity_site(
                f"{option}_drawer_success_site",
                success_detector,
                reference_spec.get("site_geometry"),
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
            reference = self._entity_site(
                str(reference_spec.get("semantic_name", "color_target")),
                self._expect_count(refs, 1, "drag color target")[0],
                reference_spec.get("site_geometry"),
            )
        elif name == "slide_block_to_color_target":
            selected = self._attr_objects("_block", "block") or self._objects(
                target_spec["names"], "slide block"
            )
            target = self._entity_object("block", selected)
            # The PerAct RLBench task creates success1..success4 dynamically in
            # init_episode and does not retain the selected sensor as a task
            # attribute. The registered DetectedCondition is the authoritative
            # variation-specific termination definition.
            success_detectors = _unique_scene_objects([
                detector
                for condition in getattr(self._task, "_success_conditions", ())
                for detector in (getattr(condition, "_detector", None),)
                if detector is not None
            ])
            success_detectors = self._expect_count(
                success_detectors, 1, "slide color target success detector")
            reference = self._entity_site(
                str(reference_spec.get("semantic_name", "color_target")),
                success_detectors[0],
                reference_spec.get("site_geometry"),
            )
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
            # RLBench considers a cup placed when its detector condition is
            # satisfied.  Some successful demonstrations terminate while the
            # gripper is still closed, so release is not a valid prerequisite.
            return phase < len(conditions) and _condition_met(conditions[phase])
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

    def _demo_phase_strategy(self) -> Mapping[str, object]:
        """Return the explicit stored-demo phase rule for the current task."""
        value = self._task_spec().get("demo_phase")
        if not isinstance(value, Mapping) or not value.get("strategy"):
            raise SemanticRoleMappingError(
                f"Task {self._task_name!r} has no demo_phase strategy in "
                f"{self.role_config_path}"
            )
        return value

    @staticmethod
    def _release_frames(demo: Sequence[object]) -> List[int]:
        return [
            frame
            for frame in range(1, len(demo))
            if float(demo[frame - 1].gripper_open) < 0.5
            and float(demo[frame].gripper_open) >= 0.5
        ]

    def _ordered_reference_relation_frames(
        self,
        demo: Sequence[object],
        candidate_frames: Sequence[int],
        max_distance: float,
        preferred_frames: Sequence[int] = (),
    ) -> Tuple[List[int], List[float]]:
        """Select one ordered boundary for each already-defined reference.

        Close-to-open transitions can be missing (a successful demo may finish
        while still gripping the last cup) or can contain empty releases and
        re-grasps. Semantic roles remain fixed by the task resolver; proximity
        is used only to timestamp completion events among known keypoints.
        """
        candidates = sorted(set(int(frame) for frame in candidate_frames))
        phase_count = self._phase_count()
        if len(candidates) < phase_count:
            raise SemanticRoleMappingError(
                f"{self._task_name} needs {phase_count} ordered phase-boundary "
                f"candidates, but only received {candidates}")

        original_phase = self._phase_index
        references = []
        try:
            for phase in range(phase_count):
                self._phase_index = phase
                reference = self._build_assignment().reference
                if reference is None:
                    raise SemanticRoleMappingError(
                        f"{self._task_name} phase {phase} has no reference for "
                        "phase-boundary relation recovery")
                references.append(reference)
        finally:
            self._phase_index = original_phase

        reference_anchors = []
        for reference in references:
            if reference.kind == "site":
                reference_anchors.append(
                    np.asarray(reference.site_position, dtype=np.float64).reshape(1, 3))
                continue
            anchor = None
            # Cup-holder spokes are fixed fixtures. Prefer the unobstructed
            # initial observation, then fall back to candidate observations.
            for anchor_frame in (0, *candidates):
                anchor_obs = demo[anchor_frame]
                masks = {
                    camera: getattr(anchor_obs, f"{camera}_mask")
                    for camera in self.cameras
                    if getattr(anchor_obs, f"{camera}_mask", None) is not None}
                point_clouds = {
                    camera: getattr(anchor_obs, f"{camera}_point_cloud")
                    for camera in self.cameras
                    if getattr(anchor_obs, f"{camera}_point_cloud", None) is not None}
                points, valid = self._sample_entity_points(
                    reference, masks, point_clouds)
                if valid:
                    anchor = points.astype(np.float64, copy=False)
                    break
            reference_anchors.append(anchor)

        costs = np.full(
            (phase_count, len(candidates)), np.inf, dtype=np.float64)
        for candidate_index, frame in enumerate(candidates):
            obs = demo[frame]
            gripper_pose = np.asarray(
                getattr(obs, "gripper_pose", ()), dtype=np.float64
            ).reshape(-1)
            if gripper_pose.size < 3 or not np.isfinite(gripper_pose[:3]).all():
                continue
            for phase, reference in enumerate(references):
                if reference.kind == "site":
                    costs[phase, candidate_index] = float(np.linalg.norm(
                        np.asarray(reference.site_position, dtype=np.float64)
                        - gripper_pose[:3]))
                elif reference_anchors[phase] is not None:
                    costs[phase, candidate_index] = float(np.linalg.norm(
                        reference_anchors[phase] - gripper_pose[:3],
                        axis=1).min())
            for camera in self.cameras:
                mask_value = getattr(obs, f"{camera}_mask", None)
                cloud_value = getattr(obs, f"{camera}_point_cloud", None)
                if mask_value is None or cloud_value is None:
                    continue
                mask = decode_handle_mask(mask_value)
                cloud = np.asarray(cloud_value)
                if cloud.ndim == 3 and cloud.shape[0] == 3 and cloud.shape[-1] != 3:
                    cloud = np.moveaxis(cloud, 0, -1)
                if cloud.shape[:2] != mask.shape or cloud.shape[-1] != 3:
                    raise SemanticRoleMappingError(
                        f"{self._task_name} frame {frame} {camera} "
                        f"mask/point-cloud mismatch: {mask.shape} vs {cloud.shape}")
                for phase, reference in enumerate(references):
                    if reference.kind != "object":
                        continue
                    points = cloud[np.isin(mask, reference.handles)]
                    points = points[np.isfinite(points).all(axis=1)]
                    if points.size:
                        distance = np.linalg.norm(
                            points.astype(np.float64, copy=False)
                            - gripper_pose[:3], axis=1).min()
                        costs[phase, candidate_index] = min(
                            costs[phase, candidate_index], float(distance))

        # A real close-to-open event is stronger boundary evidence than a plain
        # action keypoint. Add one threshold-width penalty to non-release
        # candidates for assignment only; retain raw metric distances for
        # validation and audit output. An implausible release (> threshold) can
        # therefore still lose to a geometrically valid keypoint.
        preferred = {int(frame) for frame in preferred_frames}
        selection_costs = costs.copy()
        for candidate_index, frame in enumerate(candidates):
            if frame not in preferred:
                selection_costs[:, candidate_index] += max_distance

        cumulative = np.full_like(selection_costs, np.inf)
        previous_index = np.full(costs.shape, -1, dtype=np.int64)
        cumulative[0] = selection_costs[0]
        for phase in range(1, phase_count):
            best_cost = np.inf
            best_index = -1
            for candidate_index in range(len(candidates)):
                prior = candidate_index - 1
                if prior >= 0 and cumulative[phase - 1, prior] < best_cost:
                    best_cost = cumulative[phase - 1, prior]
                    best_index = prior
                if best_index >= 0 and np.isfinite(
                        selection_costs[phase, candidate_index]):
                    cumulative[phase, candidate_index] = (
                        best_cost + selection_costs[phase, candidate_index])
                    previous_index[phase, candidate_index] = best_index

        final_index = int(np.argmin(cumulative[-1]))
        if not np.isfinite(cumulative[-1, final_index]):
            raise SemanticRoleMappingError(
                f"Could not assign candidates {candidates} to {phase_count} "
                f"ordered {self._task_name} reference entities")
        chosen_indices = [final_index]
        for phase in range(phase_count - 1, 0, -1):
            chosen_indices.append(int(previous_index[phase, chosen_indices[-1]]))
        chosen_indices.reverse()
        frames = [candidates[index] for index in chosen_indices]
        distances = [
            float(costs[phase, index])
            for phase, index in enumerate(chosen_indices)]
        too_far = [
            (phase, frame, distance)
            for phase, (frame, distance) in enumerate(zip(frames, distances))
            if distance > max_distance]
        if too_far:
            raise SemanticRoleMappingError(
                f"{self._task_name} ordered boundary/reference distance exceeds "
                f"max_distance={max_distance:.3f} m: {too_far}")
        return frames, distances

    def _ordered_target_contact_frames(
        self,
        demo: Sequence[object],
        sample_frames: Sequence[int],
        max_distance: float,
    ) -> Tuple[List[int], List[float]]:
        """Locate ordered push-button contacts at stored-demo keypoints.

        The semantic target order still comes exclusively from the task source
        and role YAML. Distance is used only to locate when each already-known
        target was contacted because the legacy observations do not contain
        task joint state.
        """
        candidates = sorted(set(int(frame) for frame in sample_frames))
        if len(candidates) < self._phase_count():
            raise SemanticRoleMappingError(
                f"{self._task_name} needs at least {self._phase_count()} keypoints "
                f"to locate ordered contacts, got {candidates}"
            )

        original_phase = self._phase_index
        phase_targets = []
        try:
            for phase in range(self._phase_count()):
                self._phase_index = phase
                phase_targets.append(self._build_assignment().target)
        finally:
            self._phase_index = original_phase

        costs = np.full(
            (self._phase_count(), len(candidates)), np.inf, dtype=np.float64
        )
        for candidate_index, frame in enumerate(candidates):
            obs = demo[frame]
            gripper_pose = np.asarray(
                getattr(obs, "gripper_pose", ()), dtype=np.float64
            ).reshape(-1)
            if gripper_pose.size < 3 or not np.isfinite(gripper_pose[:3]).all():
                continue
            for phase, target in enumerate(phase_targets):
                if target.kind == "site":
                    costs[phase, candidate_index] = float(np.linalg.norm(
                        np.asarray(target.site_position, dtype=np.float64)
                        - gripper_pose[:3]))
            for camera in self.cameras:
                mask_value = getattr(obs, f"{camera}_mask", None)
                cloud_value = getattr(obs, f"{camera}_point_cloud", None)
                if mask_value is None or cloud_value is None:
                    continue
                mask = decode_handle_mask(mask_value)
                cloud = np.asarray(cloud_value)
                if cloud.ndim == 3 and cloud.shape[0] == 3 and cloud.shape[-1] != 3:
                    cloud = np.moveaxis(cloud, 0, -1)
                if cloud.shape[:2] != mask.shape or cloud.shape[-1] != 3:
                    raise SemanticRoleMappingError(
                        f"{self._task_name} frame {frame} {camera} mask/point-cloud "
                        f"mismatch: {mask.shape} vs {cloud.shape}"
                    )
                for phase, target in enumerate(phase_targets):
                    if target.kind != "object":
                        continue
                    handles = np.asarray(target.handles, dtype=np.int64)
                    points = cloud[np.isin(mask, handles)]
                    points = points[np.isfinite(points).all(axis=1)]
                    if points.size:
                        distance = np.linalg.norm(
                            points.astype(np.float64, copy=False) - gripper_pose[:3],
                            axis=1,
                        ).min()
                        costs[phase, candidate_index] = min(
                            costs[phase, candidate_index], float(distance)
                        )

        # Dynamic programming finds the minimum-cost strictly ordered contact
        # sequence instead of independently choosing (possibly crossed) frames.
        phase_count, candidate_count = costs.shape
        cumulative = np.full_like(costs, np.inf)
        previous_index = np.full(costs.shape, -1, dtype=np.int64)
        cumulative[0] = costs[0]
        for phase in range(1, phase_count):
            best_cost = np.inf
            best_index = -1
            for candidate_index in range(candidate_count):
                prior = candidate_index - 1
                if prior >= 0 and cumulative[phase - 1, prior] < best_cost:
                    best_cost = cumulative[phase - 1, prior]
                    best_index = prior
                if best_index >= 0 and np.isfinite(costs[phase, candidate_index]):
                    cumulative[phase, candidate_index] = (
                        best_cost + costs[phase, candidate_index]
                    )
                    previous_index[phase, candidate_index] = best_index

        final_index = int(np.argmin(cumulative[-1]))
        if not np.isfinite(cumulative[-1, final_index]):
            raise SemanticRoleMappingError(
                f"Could not locate {phase_count} ordered {self._task_name} "
                "contacts from visible GT target masks at the discovered keypoints"
            )
        chosen_indices = [final_index]
        for phase in range(phase_count - 1, 0, -1):
            chosen_indices.append(int(previous_index[phase, chosen_indices[-1]]))
        chosen_indices.reverse()
        frames = [candidates[index] for index in chosen_indices]
        distances = [
            float(costs[phase, index]) for phase, index in enumerate(chosen_indices)
        ]
        too_far = [
            (phase, frame, distance)
            for phase, (frame, distance) in enumerate(zip(frames, distances))
            if distance > max_distance
        ]
        if too_far:
            raise SemanticRoleMappingError(
                f"{self._task_name} ordered contact exceeds max_distance="
                f"{max_distance:.3f} m: {too_far}"
            )
        return frames, distances

    def _sample_entity_points(
        self,
        entity: Optional[RoleEntity],
        masks: Mapping[str, np.ndarray],
        point_clouds: Mapping[str, np.ndarray],
    ) -> Tuple[np.ndarray, bool]:
        if entity is None:
            return np.zeros((self.num_points, 3), dtype=np.float32), False
        if entity.kind == "site":
            if entity.site_geometry is None:
                raise SemanticRoleMappingError(
                    f"site {entity.semantic_name!r} has no site_geometry"
                )
            return sample_site_geometry(
                entity.site_geometry, self.num_points
            ), True
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
        if self.handle_alignment != "identity" and self._step_index == 0:
            self._live_initial_views = self._alignment_views(obs)
        return self._enrich(obs, obs_dict)

    def _alignment_views(self, obs):
        views = {}
        misc = getattr(obs, "misc", None) or {}
        for camera in self.cameras:
            mask = getattr(obs, f"{camera}_mask", None)
            cloud = getattr(obs, f"{camera}_point_cloud", None)
            if mask is None or cloud is None:
                continue
            cloud = np.asarray(cloud)
            if cloud.ndim == 3 and cloud.shape[0] == 3 and cloud.shape[-1] != 3:
                cloud = np.moveaxis(cloud, 0, -1)
            views[camera] = {
                "mask": decode_handle_mask(mask).copy(), "cloud": cloud.copy(),
                "intrinsics": deepcopy(misc.get(f"{camera}_camera_intrinsics")),
                "extrinsics": deepcopy(misc.get(f"{camera}_camera_extrinsics")),
            }
        return views

    def _prepare_stored_handles(self, obs, live_initial):
        if self.handle_alignment == "identity":
            return live_initial
        report = {
            "schema_version": "rlbench_handle_alignment_v1",
            "task": self._task_name, "episode_idx": self._episode_idx,
            "variation": self._variation, "status": "failed",
            "mode": self.handle_alignment,
            "thresholds": dict(min_pixels=16, min_views=2, min_precision=.9,
                               min_recall=.9, max_world_distance_p95=.01),
        }
        original_phase = self._phase_index
        if self.handle_alignment == "mask_verified":
            report['thresholds'].update(min_precision=.9, min_recall=.9,
                                        hard_conflict_precision=.5,
                                        hard_conflict_recall=.5,
                                        single_view_min_pixels=32,
                                         single_view_min_precision=.98,
                                         single_view_min_recall=.98,
                                         single_view_max_world_distance_p95=.01,
                                         small_exact_geometry_min_pixels=16,
                                         small_exact_geometry_min_precision=1.,
                                         small_exact_geometry_min_recall=1.,
                                         small_exact_geometry_max_world_distance_p95=.01)
            report['thresholds'].update(
                small_exact_robust_geometry_max_world_distance_p50=.005,
                small_exact_robust_geometry_max_world_distance_p90=.012,
                small_exact_robust_geometry_max_world_distance_p95=.025)
            report['geometry_policy'] = 'audit_only_except_single_view_corroboration'
        try:
            required = set()
            required_groups = []
            for phase in range(self._phase_count()):
                self._phase_index = phase
                assignment = self._build_assignment()
                for role in (assignment.target, assignment.reference):
                    if role is not None and role.kind == "object":
                        required.update(role.handles)
                        required_groups.append(
                            (role.semantic_name, set(role.handles)))
            objects = {_object_handle(obj): obj for obj in self._index.objects}
            names, excluded = {}, set()
            for handle in sorted(required):
                obj = objects.get(handle)
                if obj is None:
                    raise HandleAlignmentError(f"Role handle {handle} missing from scene index")
                # Only simulator-confirmed non-renderable/non-shape descendants
                # may be omitted. An occluded renderable shape still needs a map.
                if hasattr(obj, "get_type") and (
                    obj.get_type().name != "SHAPE" or not obj.is_renderable()
                ):
                    excluded.add(handle)
                    continue
                names[handle] = _canonical_name(_object_name(obj))
            if len(set(names.values())) != len(names):
                raise HandleAlignmentError("Ambiguous canonical scene object names")
            metadata = None
            if self.handle_map_dir is not None:
                path = (self.handle_map_dir / self._task_name
                        / f"episode_{self._episode_idx}.json")
                with path.open(encoding="utf-8") as stream:
                    metadata = json.load(stream)
                report["metadata_path"] = str(path)
            elif isinstance(getattr(obs, "misc", None), Mapping):
                metadata = obs.misc.get("oracle_handle_metadata")
                if metadata is not None:
                    report["metadata_path"] = "demo[0].misc.oracle_handle_metadata"
            declared = None
            if metadata is not None:
                expected = dict(
                    schema_version="rlbench_name_to_handle_v1",
                    task=self._task_name, episode_idx=self._episode_idx,
                    variation=self._variation)
                if not isinstance(metadata, Mapping) or any(
                    metadata.get(k) != v for k, v in expected.items()
                ):
                    raise HandleAlignmentError("Acquisition mapping identity/schema mismatch")
                declared = metadata.get("name_to_handle")
                if not isinstance(declared, Mapping):
                    raise HandleAlignmentError("Acquisition mapping needs name_to_handle")
            stored_views = self._alignment_views(obs)
            entity_mapping = {}
            unobservable = set()
            used_entity_union_fallback = False
            try:
                mapping, evidence = align_handles(
                    self._live_initial_views or {}, stored_views, names, declared,
                    mode=self.handle_alignment,
                    allow_unobservable=self.handle_alignment == 'mask_verified')
                unobservable = set(names).difference(mapping)
                for semantic_name, handles in required_groups:
                    visual_handles = handles.difference(excluded)
                    if visual_handles and not visual_handles.difference(unobservable):
                        raise HandleAlignmentError(
                            f"Semantic entity {semantic_name!r} has no observable handle "
                            "with verified stored correspondence", evidence)
                    mapped = {
                        mapping[handle] for handle in visual_handles
                        if handle not in unobservable
                    }
                    if mapped:
                        entity_mapping[frozenset(handles)] = tuple(sorted(mapped))
                excluded.update(unobservable)
            except HandleAlignmentError as individual_error:
                # Multi-part RLBench entities can be split/merged differently
                # between live and saved instance masks.  Only mask_verified
                # mode may fall back to a strict two-view union certificate.
                visible_groups = [
                    (semantic_name, handles, handles.difference(excluded))
                    for semantic_name, handles in required_groups
                ]
                if self.handle_alignment != 'mask_verified' or not visible_groups:
                    raise
                group_evidence = {
                    "individual_alignment_error": str(individual_error),
                    "individual_alignment_evidence": individual_error.evidence,
                    "entities": {},
                    "entity_certificates": {},
                }
                used_entity_union_fallback = True
                # The aggregate attempt stops at its first failure. Retry each
                # entity independently, retaining individual certificates even
                # for handles that the aggregate attempt never reached.
                mapping, entity_mapping, unobservable = {}, {}, set()
                group_errors = []
                for semantic_name, handles, visible_handles in visible_groups:
                    try:
                        local_mapping = {}
                        try:
                            local_mapping, local_evidence = align_handles(
                                self._live_initial_views or {}, stored_views,
                                {h: names[h] for h in visible_handles}, declared,
                                mode=self.handle_alignment, allow_unobservable=True)
                            if not local_mapping:
                                raise HandleAlignmentError(
                                    'No observable individually verified handle',
                                    local_evidence)
                            mapped = tuple(sorted(set(local_mapping.values())))
                            entity_evidence = dict(
                                source='individual_handles',
                                live_handles=sorted(visible_handles),
                                stored_handles=list(mapped),
                                individual_alignment_evidence=local_evidence)
                        except HandleAlignmentError as local_error:
                            local_mapping = {}
                            try:
                                mapped, entity_evidence = align_semantic_handle_group(
                                    self._live_initial_views or {}, stored_views,
                                    visible_handles, semantic_name)
                            except HandleAlignmentError as union_error:
                                combined_evidence = dict(union_error.evidence)
                                combined_evidence['individual_alignment_error'] = str(
                                    local_error)
                                combined_evidence[
                                    'individual_alignment_evidence'] = local_error.evidence
                                raise HandleAlignmentError(
                                    str(union_error), combined_evidence) from union_error
                            entity_evidence['individual_alignment_error'] = str(local_error)
                            entity_evidence['individual_alignment_evidence'] = local_error.evidence
                        # Per-entity retries must not bypass the global identity
                        # consistency checks of the original aggregate mapper.
                        for previous_handles, previous_mapped in entity_mapping.items():
                            if (not handles.intersection(previous_handles)
                                    and set(mapped).intersection(previous_mapped)):
                                raise HandleAlignmentError(
                                    'Disjoint semantic entities claim the same stored handle',
                                    entity_evidence)
                        for handle, stored_handle in local_mapping.items():
                            if ((handle in mapping and mapping[handle] != stored_handle)
                                    or any(h != handle and s == stored_handle
                                           for h, s in mapping.items())):
                                raise HandleAlignmentError(
                                    'Conflicting individual handle mappings', entity_evidence)
                        if (frozenset(handles) in entity_mapping
                                and entity_mapping[frozenset(handles)] != mapped):
                            raise HandleAlignmentError(
                                'Conflicting mappings for the same semantic entity',
                                entity_evidence)
                    except HandleAlignmentError as group_error:
                        group_evidence["entities"][semantic_name] = (
                            group_error.evidence)
                        group_errors.append(
                            f"{semantic_name}: {group_error}")
                        continue
                    mapping.update(local_mapping)
                    if local_mapping:
                        unobservable.update(visible_handles.difference(local_mapping))
                    entity_mapping[frozenset(handles)] = mapped
                    group_evidence["entities"][semantic_name] = entity_evidence
                    group_evidence["entity_certificates"][
                        ",".join(str(handle) for handle in sorted(handles))
                    ] = entity_evidence
                if group_errors:
                    raise HandleAlignmentError(
                        "Cannot verify semantic entities: "
                        + "; ".join(group_errors),
                        group_evidence) from individual_error
                excluded.update(unobservable)
                evidence = group_evidence
            report.update(
                status=self.handle_alignment, evidence=evidence,
                live_to_stored={str(k): v for k, v in mapping.items()},
                semantic_entity_to_stored={
                    ",".join(str(handle) for handle in sorted(handles)): list(mapped)
                    for handles, mapped in entity_mapping.items()
                },
                alignment_scope=(
                    'mixed_entity_certificates'
                    if used_entity_union_fallback else 'individual_handles'),
                excluded_nonvisual_handles=sorted(excluded.difference(unobservable)),
                excluded_unobservable_handles=(
                    sorted(unobservable) if mapping else []))
            if self.handle_alignment == 'mask_verified':
                report['geometry_verified'] = False
                report['relocated_geometry_verified'] = bool(
                    evidence.get('_used_relocated_geometry', False))
                report['shifted_geometry_verified'] = bool(
                    evidence.get('_used_shifted_geometry', False))
                if used_entity_union_fallback:
                    print(
                        '[Manifest] mask_verified: individual child-handle '
                        'alignment failed; accepted strict semantic-entity '
                        'individual and/or multi-view mask certificates per entity.', flush=True)
                if report['relocated_geometry_verified']:
                    print(
                        '[Manifest] mask_verified: a moved instance was certified '
                        'by unique centered geometry in at least two views.',
                        flush=True)
                elif report['shifted_geometry_verified']:
                    print(
                        '[Manifest] mask_verified: a slightly shifted instance was '
                        'certified by unique centered geometry and registered masks.',
                        flush=True)
                else:
                    print(
                        '[Manifest] mask_verified: identity inferred from '
                        'high-overlap masks.', flush=True)
                print(
                    '[Manifest] global scene geometry is not certified. Review '
                    'the alignment JSON before training.', flush=True)
            self._stored_handle_map = mapping
            self._stored_entity_handle_map = entity_mapping
            self._nonvisual_handles = excluded
            self._handle_alignment_audit = report
            translated = deepcopy(live_initial)
            for key in ("target", "reference"):
                role = translated.get(key)
                if role is not None and role["kind"] == "object":
                    original_handles = frozenset(role["handles"])
                    if original_handles in entity_mapping:
                        role["handles"] = list(entity_mapping[original_handles])
                    else:
                        role["handles"] = sorted({
                            mapping[h] for h in role["handles"] if h not in excluded})
            return translated
        except Exception as exc:
            self.stats["mapping_errors"] += 1
            report.update(error=str(exc), evidence=getattr(exc, "evidence", {}))
            raise SemanticRoleMappingError(
                f"Handle alignment failed for {self._task_name} "
                f"episode={self._episode_idx}: {exc}") from exc
        finally:
            self._phase_index = original_phase
            if self.alignment_output_dir is not None:
                path = (self.alignment_output_dir / self._task_name
                        / f"episode_{self._episode_idx}.json")
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
                temporary.replace(path)

    def _enrich(
        self,
        obs,
        obs_dict: Mapping[str, object],
        phase_event: Optional[Tuple[int, bool, bool]] = None,
    ) -> Dict[str, object]:
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
            if phase_event is None:
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
            else:
                phase_index, completion_satisfied, phase_advanced = phase_event
                if phase_index < 0 or phase_index >= self._phase_count():
                    raise SemanticRoleMappingError(
                        f"Invalid demo-event phase {phase_index} for "
                        f"{self._task_name}; phase_count={self._phase_count()}"
                    )
                self._phase_index = int(phase_index)
                assignment = self._build_assignment()
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
        self.stats["prior_steps"] += int(target_valid or reference_valid)
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
            "phase_source": "demo_events" if phase_event is not None else "sim_replay",
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

    def _validate_stored_initial(self, obs, live_initial):
        """Diagnose mask evidence separately from usable point-cloud evidence.

        This is a consistency guard, not proof of cross-session handle identity.
        Do not replace entries until this guard succeeds.
        """
        for role_name in ("target", "reference"):
            role = live_initial.get(role_name)
            if role is None or role["kind"] != "object":
                continue
            cameras = {}
            total_pixels = total_points = 0
            for camera in self.cameras:
                mask_value = getattr(obs, f"{camera}_mask", None)
                cloud_value = getattr(obs, f"{camera}_point_cloud", None)
                detail = {
                    "mask_loaded": mask_value is not None,
                    "point_cloud_loaded": cloud_value is not None,
                    "matching_pixels": 0,
                    "finite_points": 0,
                }
                cameras[camera] = detail
                if mask_value is None:
                    continue
                mask = decode_handle_mask(mask_value)
                handles, counts = np.unique(mask, return_counts=True)
                # Bounded diagnostic: largest instances plus exact role hits.
                order = np.argsort(counts)[::-1][:16]
                detail["mask_shape"] = list(mask.shape)
                detail["largest_mask_handles"] = handles[order].astype(int).tolist()
                selected = np.isin(mask, role["handles"])
                pixels = int(selected.sum())
                detail["matching_pixels"] = pixels
                total_pixels += pixels
                if cloud_value is None:
                    continue
                cloud = np.asarray(cloud_value)
                if cloud.ndim == 3 and cloud.shape[0] == 3 and cloud.shape[-1] != 3:
                    cloud = np.moveaxis(cloud, 0, -1)
                detail["point_cloud_shape"] = list(cloud.shape)
                if cloud.ndim != 3 or cloud.shape[:2] != mask.shape or cloud.shape[-1] != 3:
                    raise SemanticRoleMappingError(
                        f"Stored frame-0 mask/point-cloud shape mismatch for "
                        f"{self._task_name} episode={self._episode_idx} {camera}: {detail}"
                    )
                points = int(np.isfinite(cloud[selected]).all(axis=1).sum())
                detail["finite_points"] = points
                total_points += points
            live_visible = bool(live_initial.get(f"{role_name}_valid", False))
            stored_visible = total_points > 0
            if total_pixels and not total_points:
                reason = "matching_mask_pixels_but_no_finite_point_cloud"
            elif live_visible and not total_pixels:
                reason = (
                    "stored_masks_missing" if not any(
                        detail["mask_loaded"] for detail in cameras.values()
                    ) else "live_role_handles_absent_from_stored_masks"
                )
            elif live_visible != stored_visible:
                reason = "live_stored_visibility_disagreement"
            else:
                continue
            report = {
                "reason": reason, "task": self._task_name,
                "episode": self._episode_idx, "variation": self._variation,
                "role": role_name, "semantic_name": role["semantic_name"],
                "live_handles": role["handles"], "live_visible": live_visible,
                "stored_visible": stored_visible, "cameras": cameras,
            }
            self.stats["mapping_errors"] += 1
            raise SemanticRoleMappingError(
                "Stored-demo frame-0 validation failed: "
                + json.dumps(report, sort_keys=True)
                + ". Check EVAL_DATAFOLDER, loaded masks/point clouds and the "
                "dataset's original name-to-handle mapping. reset_to_demo does "
                "not establish cross-session handle identity. No automatic "
                "handle remapping was performed."
            )

    def build_demo_event_manifest(
        self, demo: Sequence[object], sample_frames: Sequence[int]
    ) -> Dict[str, object]:
        """Build a strict phase manifest directly from a successful stored demo."""
        if not demo:
            raise SemanticRoleMappingError(f"{self._task_name} stored demo is empty")

        phase_count = self._phase_count()
        expected = tuple(int(frame) for frame in sample_frames)
        if not expected:
            raise SemanticRoleMappingError(
                f"{self._task_name} keypoint discovery returned no frames"
            )
        invalid_expected = [
            frame for frame in expected if frame < 0 or frame >= len(demo)
        ]
        if invalid_expected:
            raise SemanticRoleMappingError(
                f"{self._task_name} demo-event keypoints out of range: "
                f"{invalid_expected}; demo length={len(demo)}"
            )
        live_initial = self._entries[-1] if self._entries else None
        if (
            live_initial is None
            or live_initial.get("sample_frame") != 0
            or live_initial.get("phase_source") != "sim_replay"
        ):
            raise SemanticRoleMappingError(
                f"{self._task_name} demo_events requires a live reset frame before "
                "stored-demo manifest generation"
            )
        live_initial = self._prepare_stored_handles(demo[0], live_initial)
        self._validate_stored_initial(demo[0], live_initial)
        strategy_spec = self._demo_phase_strategy()
        strategy = str(strategy_spec["strategy"])
        contact_distances: List[float] = []
        detected_release_frames: List[int] = []
        release_relation_distances: List[float] = []
        if strategy == "single_success":
            if phase_count != 1:
                raise SemanticRoleMappingError(
                    f"{self._task_name} declares single_success but has "
                    f"phase_count={phase_count}"
                )
            boundary_frames = [len(demo) - 1]
            boundary_source = "successful_demo_final_frame"
        elif strategy == "release_cycles":
            detected_release_frames = self._release_frames(demo)
            boundary_frames = list(detected_release_frames)
            if (self._task_name == "place_cups"
                    and len(boundary_frames) != phase_count):
                max_distance = float(strategy_spec.get("max_distance", .20))
                if max_distance <= 0:
                    raise SemanticRoleMappingError(
                        "place_cups demo_phase max_distance must be positive")
                # A successful stored demo can finish with the last cup still
                # grasped, so a close-to-open transition is not guaranteed for
                # every detector completion. Include the expert keypoints and
                # the successful final state, while retaining observed releases
                # as stronger semantic evidence in the audit metadata.
                candidates = sorted(set((
                    *detected_release_frames, *expected, len(demo) - 1)))
                boundary_frames, release_relation_distances = (
                    self._ordered_reference_relation_frames(
                        demo, candidates, max_distance,
                        preferred_frames=detected_release_frames))
                boundary_source = (
                    "keypoints_recovered_by_ordered_reference_relation")
            elif len(boundary_frames) != phase_count:
                raise SemanticRoleMappingError(
                    f"{self._task_name} variation {self._variation} requires "
                    f"{phase_count} completed release cycles, but the stored demo "
                    f"contains {len(boundary_frames)} at frames {boundary_frames}."
                )
            else:
                boundary_source = "gripper_close_to_open"
        elif strategy == "ordered_target_contact":
            max_distance = float(strategy_spec.get("max_distance", 0.20))
            if max_distance <= 0:
                raise SemanticRoleMappingError(
                    f"{self._task_name} demo_phase max_distance must be positive"
                )
            boundary_frames, contact_distances = self._ordered_target_contact_frames(
                demo, expected, max_distance
            )
            boundary_source = "ordered_gt_target_contact_proxy"
        else:
            raise SemanticRoleMappingError(
                f"Unsupported demo_phase strategy {strategy!r} for "
                f"{self._task_name}"
            )

        # Include exact phase boundaries even when keypoint discovery removed
        # one because it was adjacent to the final frame.
        selected_frames = sorted(set((0, *expected, *boundary_frames)))
        invalid = [frame for frame in selected_frames if frame < 0 or frame >= len(demo)]
        if invalid:
            raise SemanticRoleMappingError(
                f"{self._task_name} demo-event frames out of range: {invalid}; "
                f"demo length={len(demo)}"
            )

        # Replace the live entry only after the source and phase checks pass.
        self._entries = []
        self._step_index = 0
        self._phase_index = 0
        self.set_expected_sample_frames(expected)
        previous_phase = 0
        for frame in selected_frames:
            completed = sum(boundary <= frame for boundary in boundary_frames)
            phase_index = min(completed, phase_count - 1)
            phase_advanced = phase_index > previous_phase
            completion_satisfied = phase_advanced or completed >= phase_count
            self.set_sample_frame(frame)
            self._enrich(
                demo[frame],
                {},
                phase_event=(
                    phase_index,
                    completion_satisfied,
                    phase_advanced,
                ),
            )
            previous_phase = phase_index

        self._source_alignment_validated = True
        self._demo_phase_metadata = {
            "handle_namespace": "stored" if self._stored_handle_map is not None else "live",
            "handle_alignment": self._handle_alignment_audit,
            "source_frame0_masks": {
                cam: hashlib.sha256(
                    str(view["mask"].shape).encode("ascii")
                    + np.asarray(view["mask"], dtype="<i8").tobytes()
                ).hexdigest()
                for cam, view in self._alignment_views(demo[0]).items()
            },
            "phase_strategy": strategy,
            "phase_boundary_source": boundary_source,
            "phase_boundary_frames": list(boundary_frames),
            "release_frames": (
                list(boundary_frames) if strategy == "release_cycles" else []
            ),
            "detected_release_frames": list(detected_release_frames),
            "release_relation_distances": list(release_relation_distances),
            "contact_distances": list(contact_distances),
            "phase_count": phase_count,
        }
        if self.manifest_output_dir is not None:
            self._flush_current_manifest()
            path = (self.manifest_output_dir / "semantic_role_manifests"
                    / self._task_name / f"episode_{self._episode_idx}.json")
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(
                self._manifests[(self._task_name, self._episode_idx)], indent=2),
                encoding="utf-8")
            temporary.replace(path)
        self._live_initial_views = None
        return {
            "phase_source": "demo_events",
            **self._demo_phase_metadata,
            "sample_frames": selected_frames,
        }

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
        if (not self._task_name or self._episode_idx < 0
                or self._current_manifest_discarded):
            return
        self._manifests[(self._task_name, self._episode_idx)] = {
            "schema_version": self.schema_version,
            "role_config_sha256": self.role_config_sha256,
            "resolver_version": self.task_resolver_version(self._task_name),
            "task": self._task_name,
            "episode_idx": self._episode_idx,
            "variation": self._variation,
            "generation_attempt": self._generation_attempt,
            "source_alignment_validated": self._source_alignment_validated,
            "phase_source": (
                self._entries[-1].get("phase_source", "sim_replay")
                if self._entries else "sim_replay"
            ),
            **self._demo_phase_metadata,
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
