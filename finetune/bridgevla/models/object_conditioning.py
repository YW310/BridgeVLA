"""Small, backbone-independent helpers for opt-in object conditioning."""

import torch
import torch.nn.functional as F


def pool_instruction_context(hidden, attention_mask, image_tokens, input_ids=None,
                             excluded_token_ids=()):
    """Pool text after the image prefix, excluding padding and special tokens.

    Count *valid* prefix tokens so both left and right padding are supported.
    This uses the already computed VLM hidden states, not another text encoder.
    """
    valid = attention_mask.bool()
    text = valid & (valid.long().cumsum(dim=1) > image_tokens)
    if input_ids is not None:
        for token_id in excluded_token_ids:
            if token_id is not None:
                text = text & (input_ids != token_id)
    weights = text.to(hidden.dtype).unsqueeze(-1)
    selected = torch.where(text[..., None], hidden, torch.zeros_like(hidden))
    return selected.sum(dim=1) / weights.sum(dim=1).clamp_min(1)


def action_feature_routes(base, translation, legacy_action, shared=False):
    """Return translation, local-action and global-action feature sources."""
    if shared:
        return translation, translation, translation
    return translation, legacy_action, base


def select_object_candidate_from_waypoint(
    waypoint, candidate_points, candidate_valid, temperature=0.05,
    max_distance=0.20,
):
    """Attribute a decoded translation waypoint to a visible object candidate.

    This helper is evaluation-only. ``candidate_points`` are simulator object
    point clouds in the same world frame as ``waypoint``. The returned
    confidence is a softmax over each candidate's nearest-surface distance; it
    is a diagnostic score, not a calibrated object posterior.
    """
    if waypoint.ndim != 2 or waypoint.shape[-1] != 3:
        raise ValueError('waypoint must have shape [B,3]')
    if candidate_points.ndim != 4 or candidate_points.shape[-1] != 3:
        raise ValueError('candidate_points must have shape [B,K,P,3]')
    if candidate_valid.shape != candidate_points.shape[:2]:
        raise ValueError('candidate_valid must have shape [B,K]')
    if waypoint.shape[0] != candidate_points.shape[0]:
        raise ValueError('waypoint and candidate batch dimensions must match')
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    if max_distance <= 0:
        raise ValueError('max_distance must be positive')

    finite_points = torch.isfinite(candidate_points).all(dim=-1)
    distances = torch.linalg.vector_norm(
        candidate_points - waypoint[:, None, None], dim=-1,
    )
    distances = distances.masked_fill(~finite_points, torch.inf)
    distances = distances.amin(dim=-1)
    available = candidate_valid.bool() & finite_points.any(dim=-1)
    distances = distances.masked_fill(~available, torch.inf)
    any_available = available.any(dim=-1)

    nearest = distances.argmin(dim=-1)
    safe_logits = torch.where(
        available,
        -distances / float(temperature),
        torch.full_like(distances, -torch.finfo(distances.dtype).max),
    )
    probabilities = torch.softmax(safe_logits, dim=-1)
    probabilities = torch.where(
        any_available[:, None], probabilities, torch.zeros_like(probabilities),
    )
    gather_index = nearest.unsqueeze(-1)
    selected_distance = distances.gather(1, gather_index).squeeze(1)
    confidence = probabilities.gather(1, gather_index).squeeze(1)
    selected_distance = torch.where(
        any_available, selected_distance,
        torch.full_like(selected_distance, torch.inf),
    )
    accepted = any_available & (selected_distance <= float(max_distance))
    confidence = confidence * (
        1.0 - selected_distance / float(max_distance)
    ).clamp(0.0, 1.0)
    confidence = torch.where(
        accepted, confidence, torch.zeros_like(confidence),
    )
    selected = torch.where(
        accepted, nearest, torch.full_like(nearest, -1),
    )
    return selected, selected_distance, confidence


def active_semantic_target_mask(candidate_valid, candidate_phase_indices):
    """Return candidates that belong to the current episode's phase sequence.

    Providers may append configured alternatives with ``phase=-1`` so a
    diagnostic can reveal that a heatmap is pointing at a distractor. Those
    alternatives are never eligible semantic Target predictions.
    """
    if candidate_valid.shape != candidate_phase_indices.shape:
        raise ValueError(
            'candidate_valid and candidate_phase_indices must have the same shape')
    return candidate_valid.bool() & candidate_phase_indices.ge(0)


def pending_target_candidate_mask(
    candidate_valid, candidate_phase_indices, current_candidate_indices,
    gripper_open,
):
    """Suppress completed ordered Targets after release.

    Phase -1 candidates remain available as BridgeVLA-attributed alternatives.
    A completed ordered candidate is retained while the gripper is closed so an
    object that is still physically held cannot switch early.
    """
    if candidate_valid.shape != candidate_phase_indices.shape:
        raise ValueError(
            'candidate_valid and candidate_phase_indices must have the same shape')
    batch_size, candidate_count = candidate_valid.shape
    if current_candidate_indices.shape != (batch_size,):
        raise ValueError('current_candidate_indices must have shape [B]')
    if gripper_open.shape != (batch_size,):
        raise ValueError('gripper_open must have shape [B]')

    current_candidate_indices = current_candidate_indices.long()
    current_known = (
        current_candidate_indices.ge(0)
        & current_candidate_indices.lt(candidate_count)
    )
    safe_indices = current_candidate_indices.clamp(
        min=0, max=max(candidate_count - 1, 0))
    current_phase = candidate_phase_indices.gather(
        1, safe_indices[:, None]).squeeze(1)
    current_phase = torch.where(
        current_known, current_phase, torch.full_like(current_phase, -1))
    completed = (
        candidate_phase_indices.ge(0)
        & candidate_phase_indices.lt(current_phase[:, None])
    )
    suppress_completed = (
        gripper_open.bool()[:, None] & current_phase.ge(0)[:, None])
    return candidate_valid.bool() & ~(completed & suppress_completed)


def reference_null_loss(probability, present=None, known=None):
    """Supervise the actual NULL posterior, never geometric invalidity."""
    if present is None or known is None:
        return probability.sum() * 0.0
    present = present.to(device=probability.device).bool()
    known = known.to(device=probability.device).bool()
    if present.shape != (probability.shape[0], 2) or known.shape != present.shape:
        raise ValueError('role presence and known masks must have shape [B,2]')
    values = F.binary_cross_entropy(
        probability.clamp(1e-6, 1 - 1e-6),
        (~present[:, 1]).to(probability.dtype), reduction='none',
    )
    weights = known[:, 1].to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1)


def _exact_role_slot_assignment(cost, role_valid):
    '''Solve the at-most-two-role assignment exactly, without SciPy.'''
    batch, slots, roles = cost.shape
    if roles != 2 or role_valid.shape != (batch, 2):
        raise ValueError('cost and role_valid must describe two roles')
    assignments = torch.full(
        (batch, 2), -1, device=cost.device, dtype=torch.long,
    )
    with torch.no_grad():
        cost = cost.detach()
        for batch_index in range(batch):
            valid_roles = torch.nonzero(
                role_valid[batch_index], as_tuple=False).flatten()
            if valid_roles.numel() == 0:
                continue
            if valid_roles.numel() == 1:
                role_index = int(valid_roles[0].item())
                assignments[batch_index, role_index] = cost[
                    batch_index, :, role_index].argmin()
                continue
            if slots < 2:
                raise ValueError('Two valid roles require at least two object slots')
            first_role, second_role = (int(value.item()) for value in valid_roles)
            pair_cost = (
                cost[batch_index, :, first_role, None]
                + cost[batch_index, None, :, second_role]
            )
            pair_cost.fill_diagonal_(torch.inf)
            flat_index = int(pair_cost.argmin().item())
            assignments[batch_index, first_role] = flat_index // slots
            assignments[batch_index, second_role] = flat_index % slots
    return assignments


def hungarian_role_slot_losses(
    mask_logits, role_logits, objectness_logits, target_masks, role_valid,
    role_cost_weight=0.2, objectness_cost_weight=0.1,
):
    '''Match unordered slots to visible Target/Reference masks.

    With two roles, exact pair enumeration is equivalent to Hungarian matching.
    Assignment uses detached costs; gathered logits retain their gradients.
    '''
    if mask_logits.ndim != 5:
        raise ValueError('mask_logits must have shape [B,V,K,H,W]')
    batch, views, slots, height, width = mask_logits.shape
    expected_targets = (batch, views, 2, height, width)
    if target_masks.shape != expected_targets:
        raise ValueError(
            f'target_masks must have shape {expected_targets}, got '
            f'{tuple(target_masks.shape)}')
    if role_logits.shape != (batch, slots, 2):
        raise ValueError('role_logits must have shape [B,K,2]')
    if objectness_logits.shape != (batch, slots):
        raise ValueError('objectness_logits must have shape [B,K]')
    if role_valid.shape != (batch, 2):
        raise ValueError('role_valid must have shape [B,2]')

    target_masks = target_masks.to(device=mask_logits.device,
                                   dtype=mask_logits.dtype)
    role_valid = role_valid.to(device=mask_logits.device).bool()
    expanded_logits = mask_logits[:, :, :, None]
    expanded_targets = target_masks[:, :, None]
    bce_cost = F.binary_cross_entropy_with_logits(
        expanded_logits.expand(-1, -1, -1, 2, -1, -1),
        expanded_targets.expand(-1, -1, slots, -1, -1, -1),
        reduction='none',
    ).mean(dim=(1, 4, 5))
    probabilities = torch.sigmoid(expanded_logits)
    intersection = (probabilities * expanded_targets).sum(dim=(1, 4, 5))
    denominator = (
        probabilities.sum(dim=(1, 4, 5))
        + expanded_targets.sum(dim=(1, 4, 5))
    )
    dice_cost = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
    role_cost = -F.log_softmax(role_logits, dim=-1)
    objectness_cost = F.softplus(-objectness_logits)[:, :, None]
    matching_cost = (
        bce_cost + dice_cost
        + float(role_cost_weight) * role_cost
        + float(objectness_cost_weight) * objectness_cost
    )
    assignments = _exact_role_slot_assignment(matching_cost, role_valid)
    batch_indices, role_indices = torch.where(assignments.ge(0))
    if batch_indices.numel() == 0:
        zero = (
            mask_logits.sum() + role_logits.sum() + objectness_logits.sum()
        ) * 0.0
        return {
            'mask': zero, 'bce': zero, 'dice': zero, 'role': zero,
            'objectness': zero, 'assignments': assignments,
        }

    slot_indices = assignments[batch_indices, role_indices]
    selected_logits = mask_logits.permute(0, 2, 1, 3, 4)[
        batch_indices, slot_indices]
    selected_targets = target_masks.permute(0, 2, 1, 3, 4)[
        batch_indices, role_indices]
    bce_loss = F.binary_cross_entropy_with_logits(
        selected_logits, selected_targets)
    selected_probabilities = torch.sigmoid(selected_logits).flatten(1)
    flattened_targets = selected_targets.flatten(1)
    dice_loss = 1.0 - (
        2.0 * (selected_probabilities * flattened_targets).sum(dim=1) + 1.0
    ) / (
        selected_probabilities.sum(dim=1) + flattened_targets.sum(dim=1) + 1.0
    )
    dice_loss = dice_loss.mean()
    role_loss = F.cross_entropy(
        role_logits[batch_indices, slot_indices], role_indices)
    objectness_loss = F.binary_cross_entropy_with_logits(
        objectness_logits[batch_indices, slot_indices],
        torch.ones_like(objectness_logits[batch_indices, slot_indices]),
    )
    return {
        'mask': bce_loss + dice_loss + float(role_cost_weight) * role_loss,
        'bce': bce_loss,
        'dice': dice_loss,
        'role': role_loss,
        'objectness': objectness_loss,
        'assignments': assignments,
    }


def soft_role_geometry(prior, rendered_xyz, role_valid):
    """Differentiable visible-point centers/spreads in the current render frame.

    Spread is a weighted standard deviation, not the object's full extent.
    Background and non-finite XYZ have zero weight, including empty scenes.
    """
    if rendered_xyz.ndim != 5 or rendered_xyz.shape[2] != 3:
        raise ValueError('rendered_xyz must have shape [B,V,3,H,W]')
    batch, views, _, height, width = rendered_xyz.shape
    if prior.shape[:3] != (batch, views, 2) or role_valid.shape != (batch, 2):
        raise ValueError('role maps/validity do not match rendered_xyz')
    weights = F.interpolate(
        prior.reshape(batch * views, 2, *prior.shape[-2:]),
        size=(height, width), mode='bilinear', align_corners=False,
    ).reshape(batch, views, 2, height, width)
    xyz = rendered_xyz.permute(0, 1, 3, 4, 2).reshape(batch, -1, 3)
    finite = torch.isfinite(xyz).all(dim=-1)
    background = (xyz == 0).all(dim=-1) | (xyz == -1).all(dim=-1)
    support = finite & ~background
    xyz = torch.where(support[..., None], xyz, torch.zeros_like(xyz))
    weights = weights.permute(0, 2, 1, 3, 4).reshape(batch, 2, -1)
    weights = weights * support[:, None].to(weights.dtype)
    mass = weights.sum(dim=-1, keepdim=True)
    normalized = weights / mass.clamp_min(1e-6)
    centers = torch.einsum('brn,bnc->brc', normalized, xyz)
    delta = xyz[:, None] - centers[:, :, None]
    variance = (normalized[..., None] * delta.square()).sum(dim=2)
    spreads = (variance + 1e-8).sqrt()
    available = role_valid.bool() & (mass.squeeze(-1) > 1e-6)
    mask = available[..., None].to(centers.dtype)
    centers, spreads = centers * mask, spreads * mask
    displacement = (centers[:, 1] - centers[:, 0]) * (
        available.all(dim=1, keepdim=True).to(centers.dtype)
    )
    descriptor = torch.cat((centers.flatten(1), spreads.flatten(1),
                            displacement, available.to(centers.dtype)), dim=1)
    return descriptor, available
