"""Conservative correspondence between registered live and saved GT masks.

This estimates numeric correspondence, never semantic roles. Reject ambiguous,
occluded, merged or split instances rather than manufacture a semantic label.
"""

import numpy as np


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
                      distance_p95=float(np.quantile(distances, .95)),
                      stored_minus_live_xyz_median=np.median(delta, axis=0).tolist())
    return result


def _interior(mask):
    # One-pixel erosion without a scipy dependency; never changes acceptance.
    padded = np.pad(mask, 1, constant_values=False)
    h, w = mask.shape
    return np.logical_and.reduce([
        padded[y:y+h, x:x+w] for y in range(3) for x in range(3)])


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
        candidate_evidence = {}
        for candidate in sorted(candidates):
            checks, agreeing, contradictory = {}, 0, False
            for camera, (am, bm, ac, bc) in views.items():
                av, bv = am == handle, bm == candidate
                na, nb = int(av.sum()), int(bv.sum())
                # A camera with fewer than 16 pixels supplies no positive
                # evidence. Substantial one-sided visibility is contradictory.
                if max(na, nb) < 16:
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
                checks[camera].update(
                    failure_reasons=reasons,
                    geometry=_geometry_summary(ac, bc, overlap),
                    interior_geometry=_geometry_summary(ac, bc, interior),
                    boundary_geometry=_geometry_summary(ac, bc, overlap & ~interior))
                checks[camera]['geometry_passed'] = bool(
                    count and int(finite.sum()) >= .95 * count
                    and p95 is not None and p95 <= .01)
                if mask_only:
                    # Re-rendering the same reset may move a silhouette by a
                    # few edge pixels. Require strong whole-instance overlap
                    # in two views; geometry remains audit-only.
                    ok = (min(na, nb) >= 16
                          and precision >= .9 and recall >= .9)
                    hard_conflict = (
                        max(na, nb) >= 16
                        and (min(na, nb) < 16
                             or precision < .5 or recall < .5))
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
            if not contradictory and (agreeing >= 2 or (declared is not None and not mask_only)):
                accepted.append(candidate)
        evidence[str(handle)] = dict(name=name, candidates=candidate_evidence)
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
            source=("registered_mask_overlap" if mask_only else
                    "acquisition_metadata" if declared is not None else "registered_masks"))
    return mapping, evidence
