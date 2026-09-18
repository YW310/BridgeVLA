"""Conservative correspondence between registered live and saved GT masks.

This estimates numeric correspondence, never semantic roles. Reject ambiguous,
occluded, merged or split instances rather than manufacture a semantic label.
"""

import numpy as np


SINGLE_VIEW_EXACT_MIN_PIXELS = 32
SINGLE_VIEW_DOMINANT_MIN_PIXELS = 64
SMALL_EXACT_ROBUST_MAX_DISTANCE_P50 = .005
SMALL_EXACT_ROBUST_MAX_DISTANCE_P90 = .012
SMALL_EXACT_ROBUST_MAX_DISTANCE_P95 = .025
THIN_ENTITY_STRONG_MIN_PIXELS = 8
THIN_ENTITY_AUXILIARY_MIN_PIXELS = 2
RELOCATED_MIN_PIXELS = 16
RELOCATED_MIN_VIEWS = 2
RELOCATED_SUPPORT_AREA_RATIO = 4.
RELOCATED_MIN_PIXEL_RATIO = .4
RELOCATED_MAX_PIXEL_RATIO = 2.5
RELOCATED_MAX_CENTROID_SHIFT = .15
RELOCATED_MAX_EXTENT_ERROR = .015
RELOCATED_MAX_CENTERED_DISTANCE_P50 = .008
RELOCATED_MAX_CENTERED_DISTANCE_P90 = .020
RELOCATED_MAX_CENTERED_DISTANCE_P95 = .025
RELOCATED_MAX_POINTS = 256
SHIFTED_MIN_MASK_COVERAGE = .60
SHIFTED_MIN_VIEWS = 3
RELOCATED_PLAUSIBLE_OVERLAP_COVERAGE = .20


class HandleAlignmentError(ValueError):
    def __init__(self, message, evidence=None):
        super().__init__(message)
        self.evidence = evidence or {}


def _geometry_summary(ac, bc, selected):
    valid = selected & np.isfinite(ac).all(axis=-1) & np.isfinite(bc).all(axis=-1)
    delta = np.asarray(bc[valid], dtype=np.float64) - ac[valid]
    result = dict(selected_pixels=int(selected.sum()), finite_pixels=int(valid.sum()))
    if delta.size:
        distances = np.linalg.norm(delta, axis=-1)
        result.update(distance_p50=float(np.median(distances)),
                      distance_p90=float(np.quantile(distances, .90)),
                      distance_p95=float(np.quantile(distances, .95)),
                      distance_max=float(np.max(distances)),
                      stored_minus_live_xyz_median=np.median(delta, axis=0).tolist())
        # Audit only. Fractions use all selected pixels, so missing geometry
        # cannot inflate the apparent support. Retain every distance for small
        # silhouettes to distinguish a few outliers from widespread mismatch.
        for label, threshold in (('5mm', .005), ('10mm', .01)):
            count = int(np.count_nonzero(distances <= threshold))
            result[f'pixels_within_{label}'] = count
            result[f'fraction_within_{label}'] = count / result['selected_pixels']
        if result['selected_pixels'] <= 64:
            result['finite_distances_sorted_m'] = np.sort(distances).tolist()
    return result


def _interior(mask):
    # One-pixel erosion without a scipy dependency; never changes acceptance.
    padded = np.pad(mask, 1, constant_values=False)
    h, w = mask.shape
    return np.logical_and.reduce([
        padded[y:y+h, x:x+w] for y in range(3) for x in range(3)])


def _sample_points(points, limit=RELOCATED_MAX_POINTS):
    '''Return a deterministic bounded subset for quadratic shape comparison.'''
    points = np.asarray(points, dtype=np.float64)
    if len(points) <= limit:
        return points
    indices = np.linspace(0, len(points) - 1, limit, dtype=np.int64)
    return points[indices]


def _centered_shape_summary(live_points, stored_points):
    '''Compare two visible surfaces after removing their independent centers.'''
    live_points = _sample_points(live_points)
    stored_points = _sample_points(stored_points)
    result = {
        'live_finite_points': int(len(live_points)),
        'stored_finite_points': int(len(stored_points)),
    }
    if not len(live_points) or not len(stored_points):
        return result

    live_center = np.median(live_points, axis=0)
    stored_center = np.median(stored_points, axis=0)
    live_centered = live_points - live_center
    stored_centered = stored_points - stored_center
    live_extent = np.quantile(live_points, .95, axis=0) - np.quantile(
        live_points, .05, axis=0)
    stored_extent = np.quantile(stored_points, .95, axis=0) - np.quantile(
        stored_points, .05, axis=0)
    distances = np.linalg.norm(
        live_centered[:, None, :] - stored_centered[None, :, :], axis=-1)
    nearest_live = np.min(distances, axis=1)
    nearest_stored = np.min(distances, axis=0)

    def symmetric_quantile(q):
        return float(max(
            np.quantile(nearest_live, q),
            np.quantile(nearest_stored, q)))

    result.update(
        live_center=live_center.tolist(),
        stored_center=stored_center.tolist(),
        centroid_shift=float(np.linalg.norm(stored_center - live_center)),
        live_extent=live_extent.tolist(),
        stored_extent=stored_extent.tolist(),
        extent_error_max=float(np.max(np.abs(stored_extent - live_extent))),
        centered_distance_p50=symmetric_quantile(.50),
        centered_distance_p90=symmetric_quantile(.90),
        centered_distance_p95=symmetric_quantile(.95),
    )
    return result


def _relocated_instance_candidate(views, live_handle, overlap_candidates):
    '''Find one moved or slightly shifted instance using centered geometry.

    The original path is enabled when the live silhouette lands on a saved
    instance at least four times larger in two views: the signature of exposed
    support geometry. A second, narrower trigger handles reset rasterization
    or small pose shifts: one candidate must retain at least 60% bidirectional
    registered-mask overlap in three views. Both paths still require a
    unique candidate to pass the full centered-geometry quorum below. Tiny
    boundary collisions and partial overlap with a genuinely shifted object
    are recorded separately, but neither bypasses or suppresses that quorum.
    '''
    visible_live_views = sum(
        int((am == live_handle).sum()) >= RELOCATED_MIN_PIXELS
        for am, _, _, _ in views.values())
    required_geometry_views = max(
        RELOCATED_MIN_VIEWS, min(3, visible_live_views))
    trigger_details = {}
    broad_candidates = []
    material_candidates = []
    shifted_candidates = []
    plausible_non_broad_candidates = []
    incidental_overlap_candidates = []
    for candidate in sorted(overlap_candidates):
        candidate_views = {}
        broad_views = 0
        material_views = 0
        shifted_views = 0
        plausible_non_broad_views = 0
        non_broad_material = False
        for camera, (am, bm, _, _) in views.items():
            live_mask = am == live_handle
            stored_mask = bm == candidate
            live_pixels = int(live_mask.sum())
            stored_pixels = int(stored_mask.sum())
            overlap_pixels = int((live_mask & stored_mask).sum())
            area_ratio = stored_pixels / max(live_pixels, 1)
            material = (
                live_pixels >= RELOCATED_MIN_PIXELS
                and stored_pixels >= RELOCATED_MIN_PIXELS
                and overlap_pixels >= 2)
            broad = material and area_ratio >= RELOCATED_SUPPORT_AREA_RATIO
            precision = overlap_pixels / max(stored_pixels, 1)
            recall = overlap_pixels / max(live_pixels, 1)
            shifted = (
                material
                and RELOCATED_MIN_PIXEL_RATIO <= area_ratio
                <= RELOCATED_MAX_PIXEL_RATIO
                and min(precision, recall) >= SHIFTED_MIN_MASK_COVERAGE)
            plausible_non_broad = (
                material and not broad
                and min(precision, recall)
                >= RELOCATED_PLAUSIBLE_OVERLAP_COVERAGE)
            material_views += int(material)
            broad_views += int(broad)
            shifted_views += int(shifted)
            plausible_non_broad_views += int(plausible_non_broad)
            non_broad_material |= material and not broad
            candidate_views[camera] = dict(
                live_pixels=live_pixels,
                stored_pixels=stored_pixels,
                overlap_pixels=overlap_pixels,
                stored_to_live_area_ratio=area_ratio,
                precision=precision,
                recall=recall,
                material_overlap=bool(material),
                broad_support=bool(broad),
                shifted_mask_support=bool(shifted),
                plausible_non_broad_overlap=bool(plausible_non_broad))
        trigger_details[str(candidate)] = candidate_views
        if material_views:
            material_candidates.append(candidate)
        if broad_views >= RELOCATED_MIN_VIEWS and not non_broad_material:
            broad_candidates.append(candidate)
        if shifted_views >= SHIFTED_MIN_VIEWS:
            shifted_candidates.append(candidate)
        if plausible_non_broad_views:
            plausible_non_broad_candidates.append(candidate)
        elif material_views and candidate not in broad_candidates:
            incidental_overlap_candidates.append(candidate)

    broad_trigger = bool(broad_candidates)
    shifted_trigger = bool(not broad_trigger and shifted_candidates)
    trigger = broad_trigger or shifted_trigger
    evidence = {
        'triggered': trigger,
        'trigger_type': (
            'broad_support_relocation' if broad_trigger else
            'nearby_shifted_mask' if shifted_trigger else None),
        'required_support_area_ratio': RELOCATED_SUPPORT_AREA_RATIO,
        'required_support_views': RELOCATED_MIN_VIEWS,
        'shifted_min_mask_coverage': SHIFTED_MIN_MASK_COVERAGE,
        'shifted_required_views': SHIFTED_MIN_VIEWS,
        'plausible_overlap_coverage': RELOCATED_PLAUSIBLE_OVERLAP_COVERAGE,
        'overlap_candidates': trigger_details,
        'material_overlap_candidates': material_candidates,
        'broad_support_candidates': broad_candidates,
        'shifted_mask_candidates': shifted_candidates,
        'plausible_non_broad_candidates': plausible_non_broad_candidates,
        'incidental_overlap_candidates': incidental_overlap_candidates,
        'candidates': {},
    }
    if not trigger:
        evidence['reason'] = 'no_broad_support_or_multiview_shifted_mask_trigger'
        return None, evidence

    stored_handles = sorted({
        int(value)
        for _, (_, mask, _, _) in views.items()
        for value in np.unique(mask)
        if int(value) > 0
    })
    passing = []
    for candidate in stored_handles:
        checks = {}
        agreeing = 0
        for camera, (am, bm, ac, bc) in views.items():
            live_mask = am == live_handle
            stored_mask = bm == candidate
            live_pixels = int(live_mask.sum())
            stored_pixels = int(stored_mask.sum())
            pixel_ratio = stored_pixels / max(live_pixels, 1)
            live_valid = live_mask & np.isfinite(ac).all(axis=-1)
            stored_valid = stored_mask & np.isfinite(bc).all(axis=-1)
            geometry = _centered_shape_summary(
                ac[live_valid], bc[stored_valid])
            reasons = []
            if min(live_pixels, stored_pixels) < RELOCATED_MIN_PIXELS:
                reasons.append('insufficient_pixels')
            if not (RELOCATED_MIN_PIXEL_RATIO <= pixel_ratio
                    <= RELOCATED_MAX_PIXEL_RATIO):
                reasons.append('pixel_ratio')
            if int(live_valid.sum()) < .95 * live_pixels or int(
                    stored_valid.sum()) < .95 * stored_pixels:
                reasons.append('insufficient_finite_geometry')
            gates = (
                ('centroid_shift', RELOCATED_MAX_CENTROID_SHIFT),
                ('extent_error_max', RELOCATED_MAX_EXTENT_ERROR),
                ('centered_distance_p50',
                 RELOCATED_MAX_CENTERED_DISTANCE_P50),
                ('centered_distance_p90',
                 RELOCATED_MAX_CENTERED_DISTANCE_P90),
                ('centered_distance_p95',
                 RELOCATED_MAX_CENTERED_DISTANCE_P95),
            )
            for key, maximum in gates:
                if geometry.get(key) is None or geometry[key] > maximum:
                    reasons.append(key)
            passed = not reasons
            agreeing += int(passed)
            checks[camera] = dict(
                live_pixels=live_pixels,
                stored_pixels=stored_pixels,
                stored_to_live_pixel_ratio=pixel_ratio,
                geometry=geometry,
                passed=bool(passed),
                failure_reasons=reasons)
        accepted = agreeing >= required_geometry_views
        evidence['candidates'][str(candidate)] = {
            'agreeing_view_count': agreeing,
            'candidate_accepted': bool(accepted),
            'views': checks,
        }
        if accepted:
            passing.append(candidate)

    evidence['passing_candidates'] = passing
    evidence['certificate'] = {
        'type': (
            'relocated_centered_geometry_two_view' if broad_trigger else
            'nearby_shifted_centered_geometry'),
        'min_views': required_geometry_views,
        'visible_live_views': visible_live_views,
        'min_pixels': RELOCATED_MIN_PIXELS,
        'pixel_ratio_range': [
            RELOCATED_MIN_PIXEL_RATIO, RELOCATED_MAX_PIXEL_RATIO],
        'max_centroid_shift': RELOCATED_MAX_CENTROID_SHIFT,
        'max_extent_error': RELOCATED_MAX_EXTENT_ERROR,
        'max_centered_distance_p50':
            RELOCATED_MAX_CENTERED_DISTANCE_P50,
        'max_centered_distance_p90':
            RELOCATED_MAX_CENTERED_DISTANCE_P90,
        'max_centered_distance_p95':
            RELOCATED_MAX_CENTERED_DISTANCE_P95,
    }
    if len(passing) != 1:
        evidence['reason'] = (
            'ambiguous_relocated_candidates' if passing
            else 'no_relocated_candidate_passed')
        return None, evidence
    return passing[0], evidence


def align_semantic_handle_group(live, stored, handles, semantic_name):
    """Verify a multi-handle entity by its union mask.

    RLBench sometimes represents one semantic object with separate physical
    and visual shapes.  Saved and live masks may split that entity differently,
    so a one-to-one child-handle correspondence is unnecessarily strict.  This
    fallback still requires a two-camera, high-overlap certificate for the
    complete semantic entity and never uses numeric identity or proximity.
    """
    handles = {
        int(handle) for handle in handles
        if isinstance(handle, (int, np.integer)) and not isinstance(handle, bool)
        and int(handle) > 0
    }
    if not handles:
        raise HandleAlignmentError(
            f"Semantic entity {semantic_name!r} has no visual live handles")

    views, excluded = {}, {}
    for camera in sorted(set(live) | set(stored)):
        if camera not in live or camera not in stored:
            excluded[camera] = {"reason": "missing_live_or_stored_view"}
            continue
        a, b = live[camera], stored[camera]
        for key, shape in (("intrinsics", (3, 3)), ("extrinsics", (4, 4))):
            try:
                x = np.asarray(a.get(key), dtype=np.float64)
                y = np.asarray(b.get(key), dtype=np.float64)
            except (TypeError, ValueError):
                excluded[camera] = {"reason": f"invalid_{key}"}
                break
            if (x.shape != shape or y.shape != shape
                    or not np.isfinite(x).all() or not np.isfinite(y).all()
                    or not np.allclose(x, y, atol=1e-4, rtol=0)):
                excluded[camera] = {
                    "reason": f"unregistered_{key}",
                    "max_abs_difference": (
                        float(np.max(np.abs(x - y)))
                        if x.shape == y.shape == shape
                        and np.isfinite(x).all() and np.isfinite(y).all()
                        else None),
                }
                break
        if camera in excluded:
            continue
        am, bm = a["mask"], b["mask"]
        ac, bc = a["cloud"], b["cloud"]
        if (am.ndim != 2 or am.shape != bm.shape
                or ac.shape != (*am.shape, 3) or bc.shape != ac.shape):
            excluded[camera] = {"reason": "mask_cloud_resolution_mismatch"}
            continue
        views[camera] = (am, bm, ac, bc)

    evidence = {
        "semantic_name": semantic_name,
        "live_handles": sorted(handles),
        "_registration": {
            "used_cameras": sorted(views),
            "excluded_cameras": excluded,
        },
    }
    if len(views) < 2:
        raise HandleAlignmentError(
            f"Cannot verify semantic entity {semantic_name!r}: only "
            f"{len(views)} registered camera pairs; need 2", evidence)

    # A saved handle becomes a group candidate only when most of that saved
    # instance lies inside the live entity in at least two registered views.
    # This prevents a coincidental edge overlap from adding an unrelated object.
    candidate_votes = {}
    candidate_details = {}
    for camera, (am, bm, ac, bc) in views.items():
        live_entity = np.isin(am, tuple(handles))
        for candidate in np.unique(bm[live_entity]):
            candidate = int(candidate)
            if candidate <= 0:
                continue
            stored_instance = bm == candidate
            selected = live_entity & stored_instance
            overlap = int(selected.sum())
            live_pixels = int(live_entity.sum())
            stored_pixels = int(stored_instance.sum())
            stored_coverage = overlap / max(stored_pixels, 1)
            live_coverage = overlap / max(live_pixels, 1)
            # A semantic entity may be split into thin visual sub-parts.  The
            # per-part threshold only proposes group members; the complete
            # union below still needs >=16 pixels and 90% bidirectional overlap
            # in two views.
            proposed = (
                overlap >= THIN_ENTITY_AUXILIARY_MIN_PIXELS
                and stored_coverage >= .9)
            candidate_details.setdefault(str(candidate), {})[camera] = {
                "live_pixels": live_pixels,
                "stored_pixels": stored_pixels,
                "overlap_pixels": overlap,
                "stored_coverage": stored_coverage,
                "live_coverage": live_coverage,
                "proposed": bool(proposed),
                "geometry": _geometry_summary(ac, bc, selected),
            }
            if proposed:
                candidate_votes[candidate] = candidate_votes.get(candidate, 0) + 1
    candidates = {
        candidate for candidate, votes in candidate_votes.items() if votes >= 2}
    evidence["candidate_votes"] = {
        str(candidate): votes for candidate, votes in sorted(candidate_votes.items())}
    evidence["candidate_details"] = candidate_details
    evidence["stored_handles"] = sorted(candidates)
    if not candidates:
        raise HandleAlignmentError(
            f"Cannot verify semantic entity {semantic_name!r}: no saved handle "
            "has strong overlap in two views", evidence)

    checks, agreeing = {}, 0
    for camera, (am, bm, ac, bc) in views.items():
        live_entity = np.isin(am, tuple(handles))
        stored_entity = np.isin(bm, tuple(candidates))
        overlap = live_entity & stored_entity
        na, nb, count = (
            int(live_entity.sum()), int(stored_entity.sum()), int(overlap.sum()))
        precision = count / max(nb, 1)
        recall = count / max(na, 1)
        passed = (
            min(na, nb) >= 16 and precision >= .9 and recall >= .9)
        identity_support = (
            min(na, nb) >= 3 and precision >= .98 and recall >= .98)
        strong_identity_support = (
            min(na, nb) >= 32 and precision >= .98 and recall >= .98)
        thin_identity_support = (
            min(na, nb) >= THIN_ENTITY_AUXILIARY_MIN_PIXELS
            and precision == 1. and recall == 1.)
        thin_strong_identity_support = (
            min(na, nb) >= THIN_ENTITY_STRONG_MIN_PIXELS
            and precision == 1. and recall == 1.)
        hard_mask_conflict = (
            max(na, nb) >= 16 and (precision < .5 or recall < .5))
        checks[camera] = {
            "live_pixels": na,
            "stored_pixels": nb,
            "overlap_pixels": count,
            "precision": precision,
            "recall": recall,
            "passed": bool(passed),
            "identity_support": bool(identity_support),
            "strong_identity_support": bool(strong_identity_support),
            "thin_identity_support": bool(thin_identity_support),
            "thin_strong_identity_support": bool(
                thin_strong_identity_support),
            "hard_mask_conflict": bool(hard_mask_conflict),
            "geometry": _geometry_summary(ac, bc, overlap),
        }
        agreeing += int(passed)
    evidence["views"] = checks
    supporting_views = sorted(
        camera for camera, check in checks.items()
        if check["identity_support"])
    strong_views = sorted(
        camera for camera, check in checks.items()
        if check["strong_identity_support"])
    conflicting_views = sorted(
        camera for camera, check in checks.items()
        if check["hard_mask_conflict"])
    thin_supporting_views = sorted(
        camera for camera, check in checks.items()
        if check["thin_identity_support"])
    thin_strong_views = sorted(
        camera for camera, check in checks.items()
        if check["thin_strong_identity_support"])
    asymmetric_quorum = (
        len(strong_views) >= 1 and len(supporting_views) >= 2
        and not conflicting_views)
    thin_exact_quorum = (
        len(thin_supporting_views) >= 2 and len(thin_strong_views) >= 1
        and not conflicting_views)
    if agreeing >= 2:
        source = "semantic_entity_union_mask_overlap"
        certificate_type = "two_full_views"
    elif asymmetric_quorum:
        source = "semantic_entity_union_asymmetric_multiview_mask_overlap"
        certificate_type = "one_strong_one_small_view"
    elif thin_exact_quorum:
        source = "semantic_entity_union_thin_exact_multiview_mask_overlap"
        certificate_type = "thin_exact_two_view"
    else:
        raise HandleAlignmentError(
            f"Cannot verify semantic entity {semantic_name!r}: union mask "
            f"passed {agreeing} full views; supporting={supporting_views}, "
            f"strong={strong_views}, conflicts={conflicting_views}", evidence)
    if thin_exact_quorum and not (agreeing >= 2 or asymmetric_quorum):
        evidence["certificate"] = {
            "type": certificate_type,
            "auxiliary_min_pixels": THIN_ENTITY_AUXILIARY_MIN_PIXELS,
            "strong_min_pixels": THIN_ENTITY_STRONG_MIN_PIXELS,
            "min_precision": 1.,
            "min_recall": 1.,
            "supporting_views": thin_supporting_views,
            "strong_views": thin_strong_views,
            "conflicting_views": conflicting_views,
        }
    else:
        evidence["certificate"] = {
            "type": certificate_type,
            "auxiliary_min_pixels": 3,
            "strong_min_pixels": 32,
            "min_precision": .98,
            "min_recall": .98,
            "supporting_views": supporting_views,
            "strong_views": strong_views,
            "conflicting_views": conflicting_views,
        }
    evidence["source"] = source
    return tuple(sorted(candidates)), evidence


def align_handles(live, stored, names, name_to_handle=None, *, mode='verified',
                  allow_unobservable=False):
    """Return live->stored mapping and auditable evidence for required shapes.

    Cameras contain mask, cloud, intrinsics and extrinsics arrays. An explicit
    acquisition mapping may cover shapes invisible at reset. Otherwise every
    required shape needs two independently agreeing camera views.
    """
    if mode not in ('verified', 'mask_verified'):
        raise ValueError(f'Unknown handle alignment mode: {mode}')
    mask_only = mode == 'mask_verified'
    views, excluded = {}, {}
    for camera in sorted(set(live) | set(stored)):
        if camera not in live or camera not in stored:
            excluded[camera] = {"reason": "missing_live_or_stored_view"}
            continue
        a, b = live[camera], stored[camera]
        for key, shape in (("intrinsics", (3, 3)), ("extrinsics", (4, 4))):
            try:
                x = np.asarray(a.get(key), dtype=np.float64)
                y = np.asarray(b.get(key), dtype=np.float64)
            except (TypeError, ValueError):
                excluded[camera] = {"reason": f"invalid_{key}"}
                break
            if (x.shape != shape or y.shape != shape
                    or not np.isfinite(x).all() or not np.isfinite(y).all()
                    or not np.allclose(x, y, atol=1e-4, rtol=0)):
                excluded[camera] = {
                    "reason": f"unregistered_{key}",
                    "max_abs_difference": (
                        float(np.max(np.abs(x - y)))
                        if x.shape == y.shape == shape
                        and np.isfinite(x).all() and np.isfinite(y).all()
                        else None),
                }
                break
        if camera in excluded:
            continue
        am, bm = a["mask"], b["mask"]
        ac, bc = a["cloud"], b["cloud"]
        if (am.ndim != 2 or am.shape != bm.shape
                or ac.shape != (*am.shape, 3) or bc.shape != ac.shape):
            excluded[camera] = {"reason": "mask_cloud_resolution_mismatch"}
            continue
        views[camera] = (am, bm, ac, bc)
    evidence = {"_registration": {
        "used_cameras": sorted(views), "excluded_cameras": excluded,
    }}
    minimum_views = 2 if mask_only or name_to_handle is None else 1
    if len(views) < minimum_views:
        raise HandleAlignmentError(
            f"Insufficient registered live/stored camera pairs: {len(views)}; "
            f"need {minimum_views}. Excluded cameras: {excluded}", evidence)

    evidence['_geometry'] = {
        camera: _geometry_summary(ac, bc, np.ones(am.shape, dtype=bool))
        for camera, (am, bm, ac, bc) in views.items()
    }
    mapping, claimed = {}, {}
    for handle, name in sorted(names.items()):
        live_pixels = {
            camera: int((am == handle).sum())
            for camera, (am, _, _, _) in views.items()
        }
        candidates = set()
        for am, bm, _, _ in views.values():
            candidates.update(int(v) for v in np.unique(bm[am == handle]) if v != 0)
        declared = None
        if name_to_handle is not None:
            if name not in name_to_handle:
                raise HandleAlignmentError(f"Acquisition mapping missing shape {name}", evidence)
            declared = name_to_handle[name]
            if isinstance(declared, bool) or not isinstance(declared, int) or declared <= 0:
                raise HandleAlignmentError(f"Invalid acquisition handle for {name}", evidence)
            candidates = {declared}
        accepted = []
        accepted_sources = {}
        candidate_evidence = {}
        low_pixel_views = {}
        candidate_assessments = {}
        for candidate in sorted(candidates):
            checks, agreeing, contradictory = {}, 0, False
            for camera, (am, bm, ac, bc) in views.items():
                av, bv = am == handle, bm == candidate
                na, nb = int(av.sum()), int(bv.sum())
                # A camera with fewer than 16 pixels supplies no positive
                # evidence. Substantial one-sided visibility is contradictory.
                if max(na, nb) < 16:
                    overlap = av & bv
                    count = int(overlap.sum())
                    low_pixel_views.setdefault(str(candidate), {})[camera] = dict(
                        live_pixels=na, stored_pixels=nb, overlap_pixels=count,
                        precision=count / max(nb, 1), recall=count / max(na, 1),
                        used_for_individual_acceptance=False,
                        reason='below_16_pixel_view_threshold',
                        geometry=_geometry_summary(ac, bc, overlap))
                    continue
                overlap = av & bv
                count = int(overlap.sum())
                precision = count / max(nb, 1)
                recall = count / max(na, 1)
                finite = (np.isfinite(ac).all(axis=-1)
                          & np.isfinite(bc).all(axis=-1) & overlap)
                distances = np.linalg.norm(ac[finite] - bc[finite], axis=-1)
                p95 = float(np.quantile(distances, .95)) if distances.size else None
                ok = (min(na, nb) >= 16 and precision >= .9 and recall >= .9
                      and int(finite.sum()) >= .95 * count
                      and p95 is not None and p95 <= .01)
                checks[camera] = dict(
                    live_pixels=na, stored_pixels=nb, precision=precision,
                    recall=recall, world_distance_p95=p95, passed=bool(ok))
                reasons = []
                if min(na, nb) < 16:
                    reasons.append('insufficient_pixels')
                if precision < .9 or recall < .9:
                    reasons.append('mask_overlap')
                if not count or int(finite.sum()) < .95 * count:
                    reasons.append('insufficient_finite_geometry')
                if p95 is None or p95 > .01:
                    reasons.append('world_distance')
                interior = _interior(av) & _interior(bv)
                geometry = _geometry_summary(ac, bc, overlap)
                interior_geometry = _geometry_summary(ac, bc, interior)
                boundary_geometry = _geometry_summary(ac, bc, overlap & ~interior)
                interior_selected = int(interior_geometry.get('selected_pixels', 0))
                interior_finite = int(interior_geometry.get('finite_pixels', 0))
                interior_p95 = interior_geometry.get('distance_p95')
                interior_geometry_passed = bool(
                    interior_selected >= 32
                    and interior_finite >= .95 * interior_selected
                    and interior_p95 is not None and interior_p95 <= .005
                    and p95 is not None and p95 <= .02)
                checks[camera].update(
                    failure_reasons=reasons,
                    geometry=geometry,
                    interior_geometry=interior_geometry,
                    boundary_geometry=boundary_geometry,
                    interior_geometry_passed=interior_geometry_passed,
                    exact_mask_passed=bool(
                        min(na, nb) >= SINGLE_VIEW_EXACT_MIN_PIXELS
                        and precision == 1. and recall == 1.),
                    small_exact_geometry_passed=bool(
                        min(na, nb) >= 16
                        and precision == 1. and recall == 1.
                        and count and int(finite.sum()) >= .95 * count
                        and p95 is not None and p95 <= .01))
                checks[camera]['small_exact_robust_geometry_passed'] = bool(
                    min(na, nb) >= 16
                    and precision == 1. and recall == 1.
                    and count and int(finite.sum()) >= .95 * count
                    and geometry.get('distance_p50') is not None
                    and geometry['distance_p50']
                    <= SMALL_EXACT_ROBUST_MAX_DISTANCE_P50
                    and geometry.get('distance_p90') is not None
                    and geometry['distance_p90']
                    <= SMALL_EXACT_ROBUST_MAX_DISTANCE_P90
                    and geometry.get('distance_p95') is not None
                    and geometry['distance_p95']
                    <= SMALL_EXACT_ROBUST_MAX_DISTANCE_P95)
                checks[camera]['geometry_passed'] = bool(
                    count and int(finite.sum()) >= .95 * count
                    and p95 is not None and p95 <= .01)
                if mask_only:
                    # Re-rendering the same reset may move a silhouette by a
                    # few edge pixels. Require strong whole-instance overlap
                    # in two views; geometry remains audit-only.
                    ok = (min(na, nb) >= 16
                          and precision >= .9 and recall >= .9)
                    # Pixel count controls whether a view can vote positively;
                    # it must not manufacture a contradiction by itself. For
                    # example, 15/16 pixels with 15-pixel overlap is strong
                    # supporting evidence, while 0/16 is still rejected by
                    # precision/recall below the hard-conflict threshold.
                    hard_conflict = (
                        max(na, nb) >= 16
                        and (precision < .5 or recall < .5))
                    checks[camera]['passed'] = bool(ok)
                    checks[camera]['hard_mask_conflict'] = bool(hard_conflict)
                    checks[camera]['geometry_warnings'] = [
                        r for r in reasons if r in (
                            'insufficient_finite_geometry', 'world_distance')]
                    checks[camera]['failure_reasons'] = (
                        [] if ok else ['insufficient_pixels_or_mask_overlap'])
                agreeing += int(ok)
                contradictory |= hard_conflict if mask_only else not ok
            candidate_evidence[str(candidate)] = checks
            # Acquisition metadata is authoritative when the entity is
            # unobservable; visible contradictory evidence still rejects it.
            single_view_geometry = (
                mask_only and len(candidates) == 1 and len(checks) == 1
                and agreeing == 1
                and all(
                    (check['geometry_passed']
                     or check['interior_geometry_passed'])
                    and min(check['live_pixels'], check['stored_pixels']) >= 32
                    and check['precision'] >= .98 and check['recall'] >= .98
                    for check in checks.values()
                )
            )
            single_view_uses_interior = (
                single_view_geometry
                and any(
                    not check['geometry_passed']
                    and check['interior_geometry_passed']
                    for check in checks.values()))
            # Articulated fixtures can be visible from only one camera, while
            # reset-to-demo introduces a scene-wide point-cloud offset. A
            # unique, substantial and exactly equal silhouette is a stronger
            # identity certificate than that unregistered geometry. The
            # 32-pixel floor matches the existing strong-view definition.
            single_view_exact_mask = (
                mask_only and len(candidates) == 1 and len(checks) == 1
                and agreeing == 1
                and all(
                    check['exact_mask_passed']
                    for check in checks.values()
                )
            )
            single_view_small_exact_geometry = (
                mask_only and len(candidates) == 1 and len(checks) == 1
                and agreeing == 1
                and all(
                    check['small_exact_geometry_passed']
                    for check in checks.values()
                )
            )
            single_view_small_exact_robust_geometry = (
                mask_only and len(candidates) == 1 and len(checks) == 1
                and agreeing == 1
                and all(
                    check['small_exact_robust_geometry_passed']
                    for check in checks.values()
                )
            )
            # A one-pixel boundary collision may introduce a raw candidate that
            # is obviously not a viable identity. Certify candidates after
            # evidence evaluation instead of requiring the raw candidate set to
            # contain one ID. The winning silhouette must be substantial, have
            # at least 90% overlap in both directions and at least 99% in one
            # direction. If two candidates satisfy this rule, the final
            # len(accepted) check still rejects the ambiguity.
            single_view_dominant_mask = (
                mask_only and len(candidates) > 1 and agreeing == 1
                and any(
                    min(check['live_pixels'], check['stored_pixels'])
                    >= SINGLE_VIEW_DOMINANT_MIN_PIXELS
                    and min(check['precision'], check['recall']) >= .9
                    and max(check['precision'], check['recall']) >= .99
                    for check in checks.values()
                )
            )
            # In mask_verified mode the documented identity certificate is a
            # quorum of two independently registered, high-overlap views.  A
            # third camera can legitimately disagree because a thin/contact
            # surface is occluded or crosses a raster boundary after reset;
            # it must not veto two positive views.  With fewer than two votes
            # we retain the conservative single-view geometry requirement.
            accepted_by_mask_quorum = mask_only and agreeing >= 2
            accepted_by_single_view = (
                mask_only and not contradictory
                and (single_view_geometry or single_view_exact_mask
                      or single_view_small_exact_geometry
                      or single_view_small_exact_robust_geometry
                      or single_view_dominant_mask))
            accepted_by_verified = (
                not mask_only and not contradictory
                and (agreeing >= 2 or declared is not None))
            candidate_assessments[str(candidate)] = dict(
                candidate_accepted=bool(
                    accepted_by_mask_quorum or accepted_by_single_view
                    or accepted_by_verified),
                raw_candidate_count=len(candidates),
                checked_view_count=len(checks),
                agreeing_view_count=agreeing,
                contradictory_view=bool(contradictory),
                certificates=dict(
                    mask_quorum=bool(accepted_by_mask_quorum),
                    single_view_geometry=bool(single_view_geometry),
                    single_view_exact_mask=bool(single_view_exact_mask),
                    single_view_small_exact_geometry=bool(
                        single_view_small_exact_geometry),
                    single_view_small_exact_robust_geometry=bool(
                        single_view_small_exact_robust_geometry),
                    single_view_dominant_mask=bool(single_view_dominant_mask),
                    verified=bool(accepted_by_verified)))
            if (accepted_by_mask_quorum or accepted_by_single_view
                    or accepted_by_verified):
                accepted.append(candidate)
                accepted_sources[candidate] = (
                    ('single_view_mask_interior_geometry'
                     if single_view_uses_interior
                     else 'single_view_mask_geometry')
                    if single_view_geometry else
                    'single_view_exact_mask'
                    if single_view_exact_mask else
                    'single_view_small_exact_geometry'
                    if single_view_small_exact_geometry else
                    'single_view_small_exact_robust_geometry'
                    if single_view_small_exact_robust_geometry else
                    'single_view_dominant_mask'
                    if accepted_by_single_view else 'multi_view_masks')
        relocated_evidence = None
        if mask_only and declared is None and not accepted:
            relocated, relocated_evidence = _relocated_instance_candidate(
                views, handle, candidates)
            if relocated is not None:
                accepted.append(relocated)
                if (relocated_evidence.get('certificate', {}).get('type')
                        == 'nearby_shifted_centered_geometry'):
                    accepted_sources[relocated] = 'shifted_centered_geometry'
                    evidence['_used_shifted_geometry'] = True
                else:
                    accepted_sources[relocated] = 'relocated_centered_geometry'
                    evidence['_used_relocated_geometry'] = True
        evidence[str(handle)] = dict(
            name=name, candidates=candidate_evidence,
            low_pixel_views=low_pixel_views,
            candidate_assessments=candidate_assessments)
        if relocated_evidence is not None:
            evidence[str(handle)]['relocated_instance'] = relocated_evidence
        if (not accepted and not candidates and allow_unobservable
                and max(live_pixels.values(), default=0) < 16):
            evidence[str(handle)].update(
                source='excluded_unobservable',
                live_pixels_by_camera=live_pixels)
            continue
        if len(accepted) != 1:
            raise HandleAlignmentError(
                f"Cannot uniquely verify {name} (live handle={handle}); "
                f"accepted={accepted}. Inspect candidate failure_reasons and "
                "geometry evidence (including _geometry for the full image). "
                "This can be geometry disagreement, not necessarily occlusion; "
                "do not loosen matching to guess identities.",
                evidence)
        target = accepted[0]
        if target in claimed:
            raise HandleAlignmentError(
                f"Non-injective mapping: {name} and {claimed[target]} -> {target}",
                evidence)
        claimed[target] = name
        mapping[handle] = target
        evidence[str(handle)].update(
            stored_handle=target,
            source=(("registered_mask_overlap_single_view_geometry"
                     if accepted_sources[target] == 'single_view_mask_geometry'
                     else
                     "registered_mask_overlap_single_view_interior_geometry"
                     if accepted_sources[target]
                     == 'single_view_mask_interior_geometry'
                     else
                     "registered_mask_overlap_single_view_exact_mask"
                     if accepted_sources[target]
                     == 'single_view_exact_mask'
                     else
                     "registered_mask_overlap_single_view_small_exact_geometry"
                     if accepted_sources[target]
                     == 'single_view_small_exact_geometry'
                     else
                     "registered_mask_overlap_single_view_small_exact_robust_geometry"
                     if accepted_sources[target]
                     == 'single_view_small_exact_robust_geometry'
                     else
                     "registered_mask_overlap_single_view_dominant_mask"
                     if accepted_sources[target]
                     == 'single_view_dominant_mask'
                     else "registered_mask_overlap") if mask_only else
                    "acquisition_metadata" if declared is not None else "registered_masks"))
        if accepted_sources[target] == 'relocated_centered_geometry':
            evidence[str(handle)]['source'] = (
                'registered_centered_geometry_relocation')
        elif accepted_sources[target] == 'shifted_centered_geometry':
            evidence[str(handle)]['source'] = (
                'registered_centered_geometry_shifted_mask')
    return mapping, evidence
