import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "finetune" / "RLBench"))
from utils.oracle_handle_alignment import align_handles, HandleAlignmentError


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
    assert report['87']['source'] == 'exact_registered_masks'
    with pytest.raises(HandleAlignmentError):
        align_handles(live, stored, {87: 'lid'})


@pytest.mark.parametrize('failure', ['one_view', 'low_overlap', 'split', 'hidden_metadata', 'third_view'])
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
    else:
        live['wrist'] = deepcopy(live['front'])
        stored['wrist'] = deepcopy(stored['front'])
        stored['wrist']['mask'][1:5, 1:5] = 0
    with pytest.raises(HandleAlignmentError):
        align_handles(live, stored, names, declared, mode='mask_verified')


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
