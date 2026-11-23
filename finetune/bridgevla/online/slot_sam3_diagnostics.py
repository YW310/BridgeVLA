"""Compact current-frame perception diagnostics, without masks or point arrays."""
from __future__ import annotations


def camera_role_diagnostics(cameras):
    return {
        name: {
            role: dict(
                status=row.get('status'),
                prompt_valid=row.get('prompt', {}).get('prompt_valid'),
                prompt_failure_reason=row.get('prompt', {}).get('failure_reason'),
                failure_reason=row.get('failure_reason'),
                candidate_count=len(row.get('candidates', [])),
                accepted_candidate_count=sum(candidate.get('accepted') is True
                                             for candidate in row.get('candidates', [])),
                candidate_failure_reasons=sorted({
                    reason for candidate in row.get('candidates', [])
                    for reason in candidate.get('failure_reasons', [])}),
                selected_index=row.get('selected_index'))
            for role, row in camera.get('roles', {}).items()
        } for name, camera in cameras.items()
    }


def guidance_role_diagnostics(guidance):
    return {
        role: dict(status=guidance.metadata[role]['status'],
                   semantic_valid=guidance.metadata[role]['semantic_valid'],
                   confidence=float(guidance.metadata[role]['confidence']),
                   point_count=len(guidance.points(role)))
        for role in ('target', 'reference')
    }


class UnavailableTargetError(ValueError):
    """An action-only short circuit; no fabricated geometry is returned."""

    def __init__(self, diagnostics):
        super().__init__('SAM3/fusion produced no reliable current Target geometry: '
                         'no accepted Target candidates in any camera; fusion skipped')
        self.perception_diagnostics = diagnostics


def print_target_diagnostics(diagnostics, *, call):
    """Make the already recorded failure reasons visible in the runtime log."""
    guidance = diagnostics.get('guidance_roles', {}).get('target', {})
    target = diagnostics.get('roles', {}).get('target', {})
    print(f'[SlotSAM3] TARGET call={call}; guidance_status={guidance.get("status")}; '
          f'confidence={guidance.get("confidence", target.get("role_confidence"))}; '
          f'guidance_points={guidance.get("point_count")}; '
          f'geometry_confidence={target.get("geometry_confidence")}; '
          f'geometry_points={target.get("point_count")}; '
          f'failure={target.get("failure_reason") or diagnostics.get("fusion_skip_reason")}; '
          f'fusion_skipped={diagnostics.get("fusion_skipped", False)}', flush=True)
    for name, roles in diagnostics.get('camera_roles', {}).items():
        row = roles.get('target', {})
        print(f'[SlotSAM3] TARGET call={call} camera={name}; '
              f'prompt_valid={row.get("prompt_valid")}; '
              f'candidates={row.get("candidate_count")}; accepted={row.get("accepted_candidate_count")}; '
              f'failure={row.get("failure_reason") or row.get("prompt_failure_reason")}; '
              f'rejected_by={row.get("candidate_failure_reasons", [])}', flush=True)
