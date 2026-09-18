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


def test_single_view_accepts_small_exact_shape_with_verified_geometry():
    live, stored = single_view_ring(pixel_count=24, geometry_offset=.009)

    mapping, report = align_handles(
        live, stored, {87: 'place_cups_holder_spoke0'},
        mode='mask_verified')

    assert mapping == {87: 99}
    evidence = report['87']
    assert evidence['source'] == (
        'registered_mask_overlap_single_view_small_exact_geometry')
    check = evidence['candidates']['99']['front']
    assert check['small_exact_geometry_passed']
    assert check['world_distance_p95'] == pytest.approx(.009)


def test_single_view_rejects_small_exact_shape_with_bad_geometry():
    live, stored = single_view_ring(pixel_count=24, geometry_offset=.011)
    with pytest.raises(HandleAlignmentError):
        align_handles(
            live, stored, {87: 'place_cups_holder_spoke0'},
            mode='mask_verified')


def test_single_view_accepts_small_exact_shape_with_bounded_boundary_tail():
    live, stored = single_view_ring(pixel_count=27, geometry_offset=.003)
    pixels = np.flatnonzero(live['front']['mask'] == 87)
    # Reproduce the thin-spoke profile: a stable 3 mm body and four boundary
    # points at roughly 10.2, 10.7, 15 and 17 mm.
    for pixel, extra_offset in zip(
            pixels[-4:], (.0072, .0077, .012, .014)):
        row, col = np.unravel_index(pixel, live['front']['mask'].shape)
        stored['front']['cloud'][row, col, 2] += extra_offset

    mapping, report = align_handles(
        live, stored, {87: 'place_cups_holder_spoke0'},
        mode='mask_verified')

    assert mapping == {87: 99}
    evidence = report['87']
    assert evidence['source'] == (
        'registered_mask_overlap_single_view_small_exact_robust_geometry')
    check = evidence['candidates']['99']['front']
    assert check['small_exact_robust_geometry_passed']
    assert check['geometry']['distance_p50'] == pytest.approx(.003)
    assert .01 < check['geometry']['distance_p90'] <= .012
    assert check['geometry']['distance_p95'] <= .025


def test_small_silhouette_audit_exposes_tail_without_accepting_mapping():
    # Two outliers among 25 exact-mask pixels: a good median is insufficient
    # for the existing single-view geometry certificate.
    live, stored = single_view_ring(pixel_count=25, geometry_offset=.003)
    pixels = np.flatnonzero(live['front']['mask'] == 87)
    for pixel in pixels[-2:]:
        row, col = np.unravel_index(pixel, live['front']['mask'].shape)
        stored['front']['cloud'][row, col, 2] += .03
    live['left_shoulder']['mask'][0, 0] = 87
    stored['left_shoulder']['mask'][0, 0] = 99

    with pytest.raises(HandleAlignmentError) as error:
        align_handles(live, stored, {87: 'spoke'}, mode='mask_verified')

    report = error.value.evidence['87']
    check = report['candidates']['99']['front']
    assert check['passed']  # Mask vote, not candidate acceptance.
    assert not check['small_exact_geometry_passed']
    geometry = check['geometry']
    assert geometry['pixels_within_10mm'] == 23
    assert geometry['fraction_within_10mm'] == pytest.approx(23 / 25)
    assert geometry['distance_p50'] == pytest.approx(.003)
    assert geometry['distance_p95'] > .01
    assert geometry['finite_distances_sorted_m'] == pytest.approx(
        [.003] * 23 + [.033] * 2)
    assessment = report['candidate_assessments']['99']
    assert not assessment['candidate_accepted']
    assert assessment['checked_view_count'] == 1
    weak = report['low_pixel_views']['99']['left_shoulder']
    assert weak['overlap_pixels'] == 1
    assert not weak['used_for_individual_acceptance']


def test_small_silhouette_audit_counts_nonfinite_pixels_as_unsupported():
    live, stored = single_view_ring(pixel_count=25, geometry_offset=.003)
    pixels = np.flatnonzero(live['front']['mask'] == 87)
    for pixel in pixels[-2:]:
        row, col = np.unravel_index(pixel, live['front']['mask'].shape)
        stored['front']['cloud'][row, col, 2] = np.nan
    with pytest.raises(HandleAlignmentError) as error:
        align_handles(live, stored, {87: 'spoke'}, mode='mask_verified')
    geometry = error.value.evidence['87']['candidates']['99']['front']['geometry']
    assert geometry['finite_pixels'] == 23
    assert geometry['fraction_within_10mm'] == pytest.approx(23 / 25)
    assert len(geometry['finite_distances_sorted_m']) == 23


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


def single_view_exact_mask(pixel_count=32, *, stored_pixel_count=None,
                           geometry_offset=.03):
    shape = (16, 16)
    stored_pixel_count = (
        pixel_count if stored_pixel_count is None else stored_pixel_count)
    live_mask = np.zeros(shape, dtype=np.int64)
    stored_mask = np.zeros(shape, dtype=np.int64)
    live_mask.flat[:pixel_count] = 87
    stored_mask.flat[:stored_pixel_count] = 99
    rows, cols = np.indices(shape)
    live_cloud = np.stack(
        (cols / 100., rows / 100., np.ones_like(rows)), axis=-1)
    stored_cloud = live_cloud.copy()
    stored_cloud[..., 2] += geometry_offset
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


def test_single_view_accepts_unique_exact_mask_with_shifted_geometry():
    live, stored = single_view_exact_mask()

    mapping, report = align_handles(
        live, stored, {87: 'drawer_bottom'}, mode='mask_verified')

    assert mapping == {87: 99}
    evidence = report['87']
    assert evidence['source'] == (
        'registered_mask_overlap_single_view_exact_mask')
    check = evidence['candidates']['99']['front']
    assert check['exact_mask_passed']
    assert not check['geometry_passed']
    assert not check['interior_geometry_passed']


@pytest.mark.parametrize(
    'pixel_count,stored_pixel_count',
    [(31, 31), (100, 99)])
def test_single_view_exact_mask_certificate_is_strict(
        pixel_count, stored_pixel_count):
    live, stored = single_view_exact_mask(
        pixel_count, stored_pixel_count=stored_pixel_count)
    with pytest.raises(HandleAlignmentError):
        align_handles(
            live, stored, {87: 'drawer_bottom'}, mode='mask_verified')


def single_view_dominant_mask():
    shape = (20, 20)
    live_mask = np.zeros(shape, dtype=np.int64)
    stored_mask = np.zeros(shape, dtype=np.int64)
    live_mask.flat[:184] = 87
    stored_mask.flat[0] = 98
    stored_mask.flat[1:184] = 99
    stored_mask.flat[184:197] = 99
    rows, cols = np.indices(shape)
    live_cloud = np.stack(
        (cols / 100., rows / 100., np.ones_like(rows)), axis=-1)
    stored_cloud = live_cloud.copy()
    stored_cloud[..., 2] += .03
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


def test_single_view_accepts_one_dominant_candidate_over_pixel_collision():
    live, stored = single_view_dominant_mask()

    mapping, report = align_handles(
        live, stored, {87: 'drawer_top'}, mode='mask_verified')

    assert mapping == {87: 99}
    assert report['87']['source'] == (
        'registered_mask_overlap_single_view_dominant_mask')
    winner = report['87']['candidates']['99']['front']
    assert winner['live_pixels'] == 184
    assert winner['stored_pixels'] == 196
    assert winner['precision'] == pytest.approx(183 / 196)
    assert winner['recall'] == pytest.approx(183 / 184)
    assert report['87']['candidates']['98']['front']['hard_mask_conflict']


def test_single_view_rejects_two_viable_partial_candidates():
    live, stored = single_view_dominant_mask()
    # Split the live silhouette into two substantial candidates; neither has
    # the required bidirectional overlap dominance.
    stored['front']['mask'].flat[:92] = 98
    stored['front']['mask'].flat[92:184] = 99
    stored['front']['mask'].flat[184:197] = 0

    with pytest.raises(HandleAlignmentError):
        align_handles(
            live, stored, {87: 'drawer_top'}, mode='mask_verified')


def test_semantic_union_accepts_one_strong_and_one_three_pixel_exact_view():
    live, stored = single_view_boundary_noise()
    live['left_shoulder']['mask'][1, 1:4] = 87
    stored['left_shoulder']['mask'][1, 1:4] = 99

    mapped, report = align_semantic_handle_group(
        live, stored, {87}, 'bottom_drawer')

    assert mapped == (99,)
    assert report['source'] == (
        'semantic_entity_union_asymmetric_multiview_mask_overlap')
    assert report['certificate'] == {
        'type': 'one_strong_one_small_view',
        'auxiliary_min_pixels': 3,
        'strong_min_pixels': 32,
        'min_precision': .98,
        'min_recall': .98,
        'supporting_views': ['front', 'left_shoulder'],
        'strong_views': ['front'],
        'conflicting_views': [],
    }


def test_semantic_union_rejects_unreliable_small_second_view():
    live, stored = single_view_boundary_noise()
    live['left_shoulder']['mask'][1, 1:6] = 87
    stored['left_shoulder']['mask'][1, 1:5] = 99
    with pytest.raises(HandleAlignmentError):
        align_semantic_handle_group(
            live, stored, {87}, 'bottom_drawer')


def test_semantic_union_rejects_two_pixel_second_view():
    live, stored = single_view_boundary_noise()
    live['left_shoulder']['mask'][1, 1:3] = 87
    stored['left_shoulder']['mask'][1, 1:3] = 99
    with pytest.raises(HandleAlignmentError):
        align_semantic_handle_group(
            live, stored, {87}, 'top_drawer')


def thin_exact_views(auxiliary_pixels=2):
    live, stored = single_view_boundary_noise()
    for values in (live, stored):
        for view in values.values():
            view['mask'].fill(0)
    live['front']['mask'].flat[:auxiliary_pixels] = 87
    stored['front']['mask'].flat[:auxiliary_pixels] = 99
    live['left_shoulder']['mask'].flat[:8] = 87
    stored['left_shoulder']['mask'].flat[:8] = 99
    return live, stored


def test_semantic_union_accepts_thin_exact_entity_in_two_views():
    live, stored = thin_exact_views()

    mapped, report = align_semantic_handle_group(
        live, stored, {87}, 'holder_spoke0')

    assert mapped == (99,)
    assert report['source'] == (
        'semantic_entity_union_thin_exact_multiview_mask_overlap')
    assert report['certificate'] == {
        'type': 'thin_exact_two_view',
        'auxiliary_min_pixels': 2,
        'strong_min_pixels': 8,
        'min_precision': 1.,
        'min_recall': 1.,
        'supporting_views': ['front', 'left_shoulder'],
        'strong_views': ['left_shoulder'],
        'conflicting_views': [],
    }


def test_semantic_union_rejects_thin_entity_with_one_pixel_auxiliary_view():
    live, stored = thin_exact_views(auxiliary_pixels=1)
    with pytest.raises(HandleAlignmentError):
        align_semantic_handle_group(
            live, stored, {87}, 'holder_spoke0')


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


@pytest.mark.parametrize('failure', ['too_small', 'second_visible_view'])
def test_single_view_fallback_remains_conservative(failure):
    live, stored = single_view_ring(
        pixel_count=15 if failure == 'too_small' else 35)
    if failure == 'second_visible_view':
        live['left_shoulder']['mask'][1:6, 1:6] = 87
        stored['left_shoulder']['mask'][1:6, 1:5] = 99
    with pytest.raises(HandleAlignmentError):
        align_handles(live, stored, {87: 'square_ring'}, mode='mask_verified')


def relocated_instance_views(*, ambiguous=False):
    live, stored = views()
    for data in live.values():
        data['mask'].fill(0)
        data['mask'][1:5, 1:5] = 87
    for data in stored.values():
        data['mask'].fill(48)
        data['mask'][8:12, 8:12] = 99
        if ambiguous:
            data['mask'][8:12, 0:4] = 100
    return live, stored


def relocated_instance_with_incidental_overlap_views():
    live, stored = views()
    for data in live.values():
        data['mask'].fill(0)
        data['mask'][1:5, 1:5] = 87
    for data in stored.values():
        data['mask'].fill(48)
        # The relocated 4x4 object clips two pixels of its old silhouette.
        # This is too weak to be registered-mask identity evidence, but must
        # not prevent the strict centered-geometry scan triggered by handle 48.
        data['mask'][4:8, 3:7] = 99
    return live, stored


def test_mask_verified_recovers_uniquely_shaped_relocated_instance():
    live, stored = relocated_instance_views()

    mapping, report = align_handles(
        live, stored, {87: 'cylinder'}, mode='mask_verified')

    assert mapping == {87: 99}
    evidence = report['87']
    assert evidence['source'] == 'registered_centered_geometry_relocation'
    relocation = evidence['relocated_instance']
    assert relocation['triggered']
    assert relocation['broad_support_candidates'] == [48]
    assert relocation['passing_candidates'] == [99]
    assert relocation['candidates']['99']['agreeing_view_count'] == 2
    assert report['_used_relocated_geometry'] is True


def test_relocation_ignores_only_incidental_non_background_overlap():
    live, stored = relocated_instance_with_incidental_overlap_views()

    mapping, report = align_handles(
        live, stored, {87: 'cylinder'}, mode='mask_verified')

    assert mapping == {87: 99}
    relocation = report['87']['relocated_instance']
    assert relocation['trigger_type'] == 'broad_support_relocation'
    assert relocation['broad_support_candidates'] == [48]
    assert relocation['incidental_overlap_candidates'] == [99]
    assert relocation['plausible_non_broad_candidates'] == []
    assert relocation['passing_candidates'] == [99]


def test_meaningful_non_background_overlap_still_blocks_broad_trigger():
    live, stored = views()
    for data in live.values():
        data['mask'].fill(0)
        data['mask'][1:6, 1:6] = 87
    for data in stored.values():
        data['mask'].fill(48)
        data['mask'][2:7, 1:6] = 99

    with pytest.raises(HandleAlignmentError) as error:
        align_handles(live, stored, {87: 'cube'}, mode='mask_verified')

    relocation = error.value.evidence['87']['relocated_instance']
    assert not relocation['triggered']
    assert relocation['broad_support_candidates'] == [48]
    assert relocation['plausible_non_broad_candidates'] == [99]
    assert relocation['incidental_overlap_candidates'] == []


def test_mask_verified_rejects_ambiguous_relocated_instances():
    live, stored = relocated_instance_views(ambiguous=True)

    with pytest.raises(HandleAlignmentError) as error:
        align_handles(
            live, stored, {87: 'cylinder'}, mode='mask_verified')

    relocation = error.value.evidence['87']['relocated_instance']
    assert relocation['triggered']
    assert relocation['passing_candidates'] == [99, 100]
    assert relocation['reason'] == 'ambiguous_relocated_candidates'


def test_relocation_requires_three_votes_when_visible_in_four_views():
    live, stored = relocated_instance_views()
    for camera in ('right_shoulder', 'wrist'):
        live[camera] = deepcopy(live['front'])
        stored[camera] = deepcopy(stored['front'])
    for camera in ('front', 'left_shoulder'):
        stored[camera]['mask'][8:12, 0:4] = 100

    mapping, report = align_handles(
        live, stored, {87: 'cylinder'}, mode='mask_verified')

    assert mapping == {87: 99}
    relocation = report['87']['relocated_instance']
    assert relocation['certificate']['visible_live_views'] == 4
    assert relocation['certificate']['min_views'] == 3
    assert relocation['candidates']['99']['agreeing_view_count'] == 4
    assert relocation['candidates']['99']['candidate_accepted']
    assert relocation['candidates']['100']['agreeing_view_count'] == 2
    assert not relocation['candidates']['100']['candidate_accepted']


def shifted_instance_views(*, ambiguous=False):
    live, stored = views()
    for data in live.values():
        data['mask'].fill(0)
        data['mask'][2:8, 2:8] = 87
    for data in stored.values():
        data['mask'].fill(0)
        data['mask'][3:9, 2:8] = 99
        if ambiguous:
            data['mask'][6:12, 6:12] = 100
    for camera in ('right_shoulder', 'wrist'):
        live[camera] = deepcopy(live['front'])
        stored[camera] = deepcopy(stored['front'])
    return live, stored


def test_mask_verified_recovers_unique_slightly_shifted_instance():
    live, stored = shifted_instance_views()

    mapping, report = align_handles(
        live, stored, {87: 'cube'}, mode='mask_verified')

    assert mapping == {87: 99}
    evidence = report['87']
    assert evidence['source'] == 'registered_centered_geometry_shifted_mask'
    relocation = evidence['relocated_instance']
    assert relocation['trigger_type'] == 'nearby_shifted_mask'
    assert relocation['shifted_mask_candidates'] == [99]
    assert relocation['passing_candidates'] == [99]
    assert relocation['certificate']['type'] == (
        'nearby_shifted_centered_geometry')
    assert report['_used_shifted_geometry'] is True


def test_mask_verified_rejects_ambiguous_slightly_shifted_instances():
    live, stored = shifted_instance_views(ambiguous=True)

    with pytest.raises(HandleAlignmentError) as error:
        align_handles(
            live, stored, {87: 'cube'}, mode='mask_verified')

    relocation = error.value.evidence['87']['relocated_instance']
    assert relocation['trigger_type'] == 'nearby_shifted_mask'
    assert relocation['passing_candidates'] == [99, 100]
    assert relocation['reason'] == 'ambiguous_relocated_candidates'
