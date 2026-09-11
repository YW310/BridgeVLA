import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "finetune" / "RLBench"))
from utils.oracle_handle_alignment import (
    align_handles, align_semantic_handle_group, HandleAlignmentError)


def views():
    mask = np.zeros((12, 12), dtype=np.int64)
    mask[1:6, 1:6] = 87
    mask[7:12, 7:12] = 88
    rows, cols = np.indices(mask.shape)
    cloud = np.stack((cols / 100., rows / 100., np.ones_like(rows)), axis=-1)
    live = {
        cam: dict(mask=mask.copy(), cloud=cloud.copy(),
                  intrinsics=np.eye(3), extrinsics=np.eye(4))
        for cam in ("front", "left_shoulder")
    }
    stored = deepcopy(live)
    for data in stored.values():
        data["mask"][mask == 87] = 99
        data["mask"][mask == 88] = 93
    return live, stored


def test_registered_masks_recover_nonuniform_id_mapping():
    live, stored = views()
    mapping, report = align_handles(live, stored, {87: "lid", 88: "jar"})
    assert mapping == {87: 99, 88: 93}
    assert report["87"]["source"] == "registered_masks"
    assert len(report["87"]["candidates"]["99"]) == 2


@pytest.mark.parametrize("failure", ["camera", "geometry", "one_view", "split", "missing"])
def test_unverified_correspondence_is_rejected(failure):
    live, stored = views()
    if failure == "camera":
        stored["front"]["extrinsics"][0, 3] += .1
    elif failure == "geometry":
        stored["front"]["cloud"] += .1
    elif failure == "one_view":
        stored.pop("left_shoulder")
    elif failure == "split":
        for data in stored.values():
            data["mask"][1:4, 1:6] = 100
    else:
        for data in stored.values():
            data["mask"][data["mask"] == 99] = 0
    with pytest.raises(HandleAlignmentError):
        align_handles(live, stored, {87: "lid"})


def test_numeric_identity_alone_is_not_evidence():
    live, stored = views()
    for data in stored.values():
        data["mask"][data["mask"] == 93] = 87  # Same ID, wrong location.
    mapping, _ = align_handles(live, stored, {87: "lid"})
    assert mapping[87] == 99


def test_exact_masks_with_depth_offset_report_geometry_failure():
    live, stored = views()
    for data in stored.values():
        data['cloud'][..., 2] += .016
    with pytest.raises(HandleAlignmentError) as error:
        align_handles(live, stored, {87: 'lid'})
    evidence = error.value.evidence
    check = evidence['87']['candidates']['99']['front']
    assert check['failure_reasons'] == ['world_distance']
    assert check['interior_geometry']['distance_p95'] == pytest.approx(.016)
    assert check['boundary_geometry']['distance_p95'] == pytest.approx(.016)
    assert evidence['_geometry']['front']['stored_minus_live_xyz_median'] == pytest.approx([0, 0, .016])


def test_boundary_error_is_reported_without_relaxing_gate():
    live, stored = views()
    stored['front']['cloud'][1, 1:6, 2] += .03
    with pytest.raises(HandleAlignmentError) as error:
        align_handles(live, stored, {87: 'lid'})
    check = error.value.evidence['87']['candidates']['99']['front']
    assert check['interior_geometry']['distance_p95'] == 0
    assert check['boundary_geometry']['distance_p95'] == pytest.approx(.03)


@pytest.mark.parametrize("failure", ["moved", "missing_extrinsics", "missing_view"])
def test_unregistered_wrist_does_not_veto_two_registered_views(failure):
    live, stored = views()
    live["wrist"] = deepcopy(live["front"])
    stored["wrist"] = deepcopy(stored["front"])
    if failure == "moved":
        stored["wrist"]["extrinsics"][0, 3] += .02
    elif failure == "missing_extrinsics":
        stored["wrist"]["extrinsics"] = None
    else:
        stored.pop("wrist")
    mapping, report = align_handles(live, stored, {87: "lid"})
    assert mapping == {87: 99}
    registration = report["_registration"]
    assert registration["used_cameras"] == ["front", "left_shoulder"]
    assert "wrist" in registration["excluded_cameras"]
    assert "wrist" not in report["87"]["candidates"]["99"]


def test_registered_but_contradictory_wrist_is_not_silently_excluded():
    live, stored = views()
    live["wrist"] = deepcopy(live["front"])
    stored["wrist"] = deepcopy(stored["front"])
    stored["wrist"]["cloud"] += .1
    with pytest.raises(HandleAlignmentError) as error:
        align_handles(live, stored, {87: "lid"})
    assert error.value.evidence["_registration"]["excluded_cameras"] == {}


def test_excluding_camera_still_requires_two_views_and_reports_reason():
    live, stored = views()
    stored["front"]["extrinsics"][0, 3] += .02
    with pytest.raises(HandleAlignmentError, match="need 2") as error:
        align_handles(live, stored, {87: "lid"})
    report = error.value.evidence["_registration"]
    assert report["used_cameras"] == ["left_shoulder"]
    assert report["excluded_cameras"]["front"]["max_abs_difference"] == pytest.approx(.02)


def test_acquisition_map_supports_occluded_shape_but_rejects_visible_conflict():
    live, stored = views()
    mapping, _ = align_handles(live, stored, {89: "hidden"}, {"hidden": 101})
    assert mapping == {89: 101}
    with pytest.raises(HandleAlignmentError):
        align_handles(live, stored, {87: "lid"}, {"lid": 93})


def test_two_entities_cannot_share_one_saved_id():
    live, stored = views()
    with pytest.raises(HandleAlignmentError, match="Non-injective"):
        align_handles(live, stored, {89: "a", 90: "b"}, {"a": 101, "b": 101})


def test_mask_verified_accepts_high_overlap_masks_but_audits_geometry():
    live, stored = views()
    for data in stored.values():
        data['cloud'][..., 2] += .016
    stored['front']['mask'][1, 1] = 0  # Tolerate a rasterized edge pixel.
    mapping, report = align_handles(live, stored, {87: 'lid'}, mode='mask_verified')
    assert mapping == {87: 99}
    check = report['87']['candidates']['99']['front']
    assert check['passed'] and not check['geometry_passed']
    assert check['geometry_warnings'] == ['world_distance']
    assert report['87']['source'] == 'registered_mask_overlap'
    with pytest.raises(HandleAlignmentError):
        align_handles(live, stored, {87: 'lid'})


def test_semantic_entity_union_aligns_split_live_to_merged_stored_instance():
    live, stored = views()
    for data in live.values():
        data['mask'][1:6, 1:3] = 86
    # The saved demo represents both live sub-parts with one instance ID.
    mapped, report = align_semantic_handle_group(
        live, stored, {86, 87}, 'chicken')
    assert mapped == (99,)
    assert report['source'] == 'semantic_entity_union_mask_overlap'
    assert all(view['passed'] for view in report['views'].values())


def test_semantic_entity_union_aligns_one_live_visual_to_split_stored_parts():
    live, stored = views()
    for data in stored.values():
        data['mask'][1:6, 1:3] = 100

    mapped, report = align_semantic_handle_group(
        live, stored, {87}, 'chicken')

    assert mapped == (99, 100)
    assert report['candidate_votes'] == {'99': 2, '100': 2}
    assert all(view['passed'] for view in report['views'].values())


def test_semantic_entity_union_rejects_low_overlap_candidate():
    live, stored = views()
    for data in live.values():
        data['mask'][1:6, 1:3] = 86
        data['mask'][1:6, 4:6] = 0
    with pytest.raises(HandleAlignmentError) as error:
        align_semantic_handle_group(live, stored, {86, 87}, 'chicken')
    details = error.value.evidence['candidate_details']['99']
    assert details['front']['stored_coverage'] == pytest.approx(15 / 25)
    assert not details['front']['proposed']


@pytest.mark.parametrize('failure', ['one_view', 'low_overlap', 'split', 'hidden_metadata'])
def test_mask_verified_does_not_guess_identity(failure):
    live, stored = views()
    names, declared = {87: 'lid'}, None
    if failure == 'one_view':
        stored.pop('left_shoulder')
    elif failure == 'low_overlap':
        for data in stored.values():
            data['mask'][1, 1:5] = 0
    elif failure == 'split':
        stored['front']['mask'][1:4, 1:6] = 100
    elif failure == 'hidden_metadata':
        names, declared = {89: 'hidden'}, {'hidden': 101}
    with pytest.raises(HandleAlignmentError):
        align_handles(live, stored, names, declared, mode='mask_verified')


def test_mask_verified_two_view_quorum_survives_third_view_conflict():
    live, stored = views()
    live['wrist'] = deepcopy(live['front'])
    stored['wrist'] = deepcopy(stored['front'])
    stored['wrist']['mask'][1:5, 1:5] = 0

    mapping, report = align_handles(
        live, stored, {87: 'small_or_partly_occluded_shape'},
        mode='mask_verified')

    assert mapping == {87: 99}
    assert report['87']['candidates']['99']['front']['passed']
    assert report['87']['candidates']['99']['left_shoulder']['passed']
    assert report['87']['candidates']['99']['wrist']['hard_mask_conflict']
    assert report['87']['source'] == 'registered_mask_overlap'


def test_mask_verified_allows_one_soft_disagreement_when_two_views_agree():
    live, stored = views()
    live['wrist'] = deepcopy(live['front'])
    stored['wrist'] = deepcopy(stored['front'])
    stored['wrist']['mask'][1, 1:4] = 0  # 88% recall: warning, not hard conflict.
    mapping, report = align_handles(live, stored, {87: 'lid'}, mode='mask_verified')
    assert mapping == {87: 99}
    wrist = report['87']['candidates']['99']['wrist']
    assert not wrist['passed']
    assert not wrist['hard_mask_conflict']


def test_mask_verified_does_not_treat_15_of_16_overlap_as_hard_conflict():
    # Regression for insert_onto_square_peg episode 85: two exact views must
    # not be vetoed because a third thin-ring silhouette differs by one pixel
    # across the positive-evidence boundary.
    live, stored = views()
    live['right_shoulder'] = deepcopy(live['front'])
    stored['right_shoulder'] = deepcopy(stored['front'])
    live_mask = live['right_shoulder']['mask']
    stored_mask = stored['right_shoulder']['mask']
    live_mask[live_mask == 87] = 0
    stored_mask[stored_mask == 99] = 0
    selected = np.flatnonzero(live_mask == 0)[:16]
    live_mask.flat[selected[:15]] = 87
    stored_mask.flat[selected] = 99

    mapping, report = align_handles(
        live, stored, {87: 'square_ring'}, mode='mask_verified')

    assert mapping == {87: 99}
    check = report['87']['candidates']['99']['right_shoulder']
    assert check['live_pixels'] == 15
    assert check['stored_pixels'] == 16
    assert check['precision'] == pytest.approx(15 / 16)
    assert check['recall'] == 1.
    assert not check['passed']
    assert not check['hard_mask_conflict']


def test_mask_verified_can_audit_unobservable_component_without_guessing():
    live, stored = views()
    mapping, report = align_handles(
        live, stored, {87: 'lid', 89: 'invisible_component'},
        mode='mask_verified', allow_unobservable=True)
    assert mapping == {87: 99}
    assert report['89']['source'] == 'excluded_unobservable'
    assert report['89']['live_pixels_by_camera'] == {
        'front': 0, 'left_shoulder': 0}


def test_visible_component_without_candidate_is_not_excluded():
    live, stored = views()
    for data in stored.values():
        data['mask'][data['mask'] == 99] = 0
    with pytest.raises(HandleAlignmentError):
        align_handles(live, stored, {87: 'lid'}, mode='mask_verified',
                      allow_unobservable=True)


def single_view_ring(pixel_count=35, geometry_offset=0.):
    live, stored = views()
    live['left_shoulder']['mask'][live['left_shoulder']['mask'] == 87] = 0
    stored['left_shoulder']['mask'][stored['left_shoulder']['mask'] == 99] = 0
    front_live, front_stored = live['front']['mask'], stored['front']['mask']
    front_live[front_live == 87] = 0
    front_stored[front_stored == 99] = 0
    selected = np.flatnonzero(front_live == 0)[:pixel_count]
    front_live.flat[selected] = 87
    front_stored.flat[selected] = 99
    stored['front']['cloud'][..., 2] += geometry_offset
    return live, stored


def test_mask_verified_accepts_unique_geometry_verified_single_view_shape():
    live, stored = single_view_ring()
    mapping, report = align_handles(
        live, stored, {87: 'square_ring'}, mode='mask_verified')
    assert mapping == {87: 99}
    assert report['87']['source'] == 'registered_mask_overlap_single_view_geometry'


def single_view_boundary_noise(interior_offset=0., boundary_offset=.015):
    shape = (16, 16)
    live_mask = np.zeros(shape, dtype=np.int64)
    stored_mask = np.zeros(shape, dtype=np.int64)
    live_mask[2:13, 2:13] = 87
    stored_mask[2:13, 2:13] = 99
    stored_mask[2, 2] = 0
    rows, cols = np.indices(shape)
    live_cloud = np.stack(
        (cols / 100., rows / 100., np.ones_like(rows)), axis=-1)
    stored_cloud = live_cloud.copy()
    live_region = live_mask == 87
    stored_region = stored_mask == 99
    interior = np.zeros(shape, dtype=bool)
    interior[3:12, 3:12] = True
    boundary = (live_region & stored_region) & ~interior
    stored_cloud[interior, 2] += interior_offset
    stored_cloud[boundary, 2] += boundary_offset
    live = {
        'front': dict(
            mask=live_mask, cloud=live_cloud,
            intrinsics=np.eye(3), extrinsics=np.eye(4)),
        'left_shoulder': dict(
            mask=np.zeros(shape, dtype=np.int64), cloud=live_cloud.copy(),
            intrinsics=np.eye(3), extrinsics=np.eye(4)),
    }
    stored = {
        'front': dict(
            mask=stored_mask, cloud=stored_cloud,
            intrinsics=np.eye(3), extrinsics=np.eye(4)),
        'left_shoulder': dict(
            mask=np.zeros(shape, dtype=np.int64), cloud=live_cloud.copy(),
            intrinsics=np.eye(3), extrinsics=np.eye(4)),
    }
    return live, stored


def test_single_view_accepts_strict_interior_geometry_with_boundary_noise():
    live, stored = single_view_boundary_noise()
    mapping, report = align_handles(
        live, stored, {87: 'chicken_visual'}, mode='mask_verified')
    assert mapping == {87: 99}
    evidence = report['87']
    assert evidence['source'] == (
        'registered_mask_overlap_single_view_interior_geometry')
    check = evidence['candidates']['99']['front']
    assert not check['geometry_passed']
    assert check['interior_geometry_passed']
    assert check['geometry']['distance_p95'] == pytest.approx(.015)
    assert check['interior_geometry']['distance_p95'] == 0.


@pytest.mark.parametrize(
    'interior_offset,boundary_offset',
    [(.006, .015), (0., .021)])
def test_single_view_interior_geometry_gate_rejects_real_disagreement(
        interior_offset, boundary_offset):
    live, stored = single_view_boundary_noise(
        interior_offset=interior_offset, boundary_offset=boundary_offset)
    with pytest.raises(HandleAlignmentError):
        align_handles(
            live, stored, {87: 'chicken_visual'}, mode='mask_verified')


@pytest.mark.parametrize('failure', ['too_small', 'bad_geometry', 'second_visible_view'])
def test_single_view_fallback_remains_conservative(failure):
    live, stored = single_view_ring(
        pixel_count=31 if failure == 'too_small' else 35,
        geometry_offset=.02 if failure == 'bad_geometry' else 0.)
    if failure == 'second_visible_view':
        live['left_shoulder']['mask'][1:6, 1:6] = 87
        stored['left_shoulder']['mask'][1:6, 1:5] = 99
    with pytest.raises(HandleAlignmentError):
        align_handles(live, stored, {87: 'square_ring'}, mode='mask_verified')
