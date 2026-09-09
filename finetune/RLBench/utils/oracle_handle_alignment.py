"""Conservative correspondence between registered live and saved GT masks.

This estimates numeric correspondence, never semantic roles. Reject ambiguous,
occluded, merged or split instances rather than manufacture a semantic label.
"""

import numpy as np


class HandleAlignmentError(ValueError):
    def __init__(self, message, evidence=None):
        super().__init__(message)
        self.evidence = evidence or {}


def align_handles(live, stored, names, name_to_handle=None):
    """Return live->stored mapping and auditable evidence for required shapes.

    Cameras contain mask, cloud, intrinsics and extrinsics arrays. An explicit
    acquisition mapping may cover shapes invisible at reset. Otherwise every
    required shape needs two independently agreeing camera views.
    """
    views = {}
    for camera in sorted(set(live) & set(stored)):
        a, b = live[camera], stored[camera]
        for key, shape in (("intrinsics", (3, 3)), ("extrinsics", (4, 4))):
            x, y = np.asarray(a.get(key)), np.asarray(b.get(key))
            if (x.shape != shape or y.shape != shape
                    or not np.isfinite(x).all() or not np.isfinite(y).all()
                    or not np.allclose(x, y, atol=1e-4, rtol=0)):
                raise HandleAlignmentError(
                    f"{camera}: live/stored {key} differ or are missing")
        am, bm = a["mask"], b["mask"]
        ac, bc = a["cloud"], b["cloud"]
        if (am.ndim != 2 or am.shape != bm.shape
                or ac.shape != (*am.shape, 3) or bc.shape != ac.shape):
            raise HandleAlignmentError(f"{camera}: mask/cloud resolution mismatch")
        views[camera] = (am, bm, ac, bc)
    if not views:
        raise HandleAlignmentError("No registered live/stored camera pairs")

    mapping, evidence, claimed = {}, {}, {}
    for handle, name in sorted(names.items()):
        candidates = set()
        for am, bm, _, _ in views.values():
            candidates.update(int(v) for v in np.unique(bm[am == handle]) if v != 0)
        declared = None
        if name_to_handle is not None:
            if name not in name_to_handle:
                raise HandleAlignmentError(f"Acquisition mapping missing shape {name}")
            declared = name_to_handle[name]
            if isinstance(declared, bool) or not isinstance(declared, int) or declared <= 0:
                raise HandleAlignmentError(f"Invalid acquisition handle for {name}")
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
                agreeing += int(ok)
                contradictory |= not ok
            candidate_evidence[str(candidate)] = checks
            # Acquisition metadata is authoritative when the entity is
            # unobservable; visible contradictory evidence still rejects it.
            if not contradictory and (agreeing >= 2 or declared is not None):
                accepted.append(candidate)
        evidence[str(handle)] = dict(name=name, candidates=candidate_evidence)
        if len(accepted) != 1:
            raise HandleAlignmentError(
                f"Cannot uniquely verify {name} (live handle={handle}); "
                f"accepted={accepted}. Supply acquisition name-to-handle metadata "
                "for occluded shapes; do not loosen matching to guess identities.",
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
            source="acquisition_metadata" if declared is not None else "registered_masks")
    return mapping, evidence
