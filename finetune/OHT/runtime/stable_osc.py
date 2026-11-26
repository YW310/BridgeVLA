"""Bounded full-pose OSC for the native, fixed-base OHT controller.

Regularize the operational inverse inertia before solving, then scale the
motion torque to the physical effort headroom while retaining gravity torque.
This is a tracking controller, not a collision planner.
"""
import math

import torch


def pose_error_wxyz(current, desired):
    """Root-frame translation and shortest spatial SO(3) error."""
    current_q = torch.nn.functional.normalize(current[:, 3:], dim=-1)
    desired_q = torch.nn.functional.normalize(desired[:, 3:], dim=-1)
    cw, cv = current_q[:, :1], current_q[:, 1:]
    dw, dv = desired_q[:, :1], desired_q[:, 1:]
    scalar = dw * cw + (dv * cv).sum(-1, keepdim=True)
    vector = cw * dv - dw * cv - torch.cross(dv, cv, dim=-1)
    sign = torch.where(scalar < 0, -1., 1.)
    scalar, vector = scalar * sign, vector * sign
    norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    angle = 2 * torch.atan2(norm, scalar.clamp_min(0))
    rotation = vector * torch.where(norm > 1e-8, angle / norm.clamp_min(1e-12), 2.)
    return torch.cat((desired[:, :3] - current[:, :3], rotation), -1)


class StableOSC:
    def __init__(self, controller, effort_limits, max_inertia_condition=1e4):
        if not math.isfinite(max_inertia_condition) or max_inertia_condition <= 1:
            raise ValueError('OSC inertia condition limit must be finite and >1')
        cfg = controller.cfg
        if (list(cfg.target_types) != ['pose_rel'] or not cfg.inertial_dynamics_decoupling
                or cfg.partial_inertial_dynamics_decoupling or cfg.nullspace_control != 'none'):
            raise ValueError('Stable OSC requires full pose_rel inertia decoupling and no null-space control')
        limits = torch.as_tensor(effort_limits).detach().clone()
        if limits.ndim != 1 or not torch.isfinite(limits).all() or (limits <= 0).any():
            raise ValueError('Stable OSC requires finite positive physical joint effort limits')
        self.controller, self.effort_limits = controller, limits
        self.max_inertia_condition = float(max_inertia_condition)
        self.last_diagnostics = None

    def __getattr__(self, name):
        return getattr(self.controller, name)

    def reset(self, *args, **kwargs):
        self.last_diagnostics = None
        return self.controller.reset(*args, **kwargs)

    def compute(self, *, jacobian_b, mass_matrix=None, current_ee_pose_b=None,
                current_ee_vel_b=None, gravity=None, **kwargs):
        native = self.controller
        if native.desired_ee_pose_b is None or native.desired_ee_wrench_b is not None:
            raise ValueError('Stable OSC requires a pose target without force control')
        if mass_matrix is None or current_ee_pose_b is None or current_ee_vel_b is None:
            raise ValueError('Stable OSC requires measured pose, velocity and mass matrix')
        values = [jacobian_b, mass_matrix, current_ee_pose_b, current_ee_vel_b, native.desired_ee_pose_b]
        if native.cfg.gravity_compensation:
            if gravity is None:
                raise ValueError('Stable OSC requires gravity compensation forces')
            values.append(gravity)
        if not all(torch.isfinite(v).all() for v in values):
            raise ValueError('Nonfinite OSC input; refusing a torque command')
        if any((torch.linalg.vector_norm(v[:, 3:7], dim=-1) < 1e-8).any()
               for v in (current_ee_pose_b, native.desired_ee_pose_b)):
            raise ValueError('Invalid OSC pose quaternion')
        if jacobian_b.shape[-2] != 6 or jacobian_b.shape[-1] != self.effort_limits.numel():
            raise ValueError('Stable OSC joint effort limits do not match the Jacobian')
        # Small 6x6 solves in float64 avoid float32 cancellation near singularity.
        jac, mass = jacobian_b.double(), mass_matrix.double()
        inverse_inertia = jac @ torch.linalg.solve(mass, jac.mT)
        inverse_inertia = (inverse_inertia + inverse_inertia.mT) * .5
        eigenvalues = torch.linalg.eigvalsh(inverse_inertia)
        maximum = eigenvalues[:, -1].clamp_min(1e-12)
        minimum = eigenvalues[:, 0]
        # Shift so (max + shift)/(min + shift) <= configured condition bound.
        shift = ((maximum - self.max_inertia_condition * minimum) /
                 (self.max_inertia_condition - 1)).clamp_min(0)
        shift = torch.maximum(shift, (1e-12 - minimum).clamp_min(0))
        identity = torch.eye(6, device=jac.device, dtype=jac.dtype)
        error = pose_error_wxyz(current_ee_pose_b, native.desired_ee_pose_b).double()
        acceleration = (native._motion_p_gains_b.double() @ error.unsqueeze(-1)
                        - native._motion_d_gains_b.double() @ current_ee_vel_b.double().unsqueeze(-1))
        force = torch.linalg.solve(inverse_inertia + shift[:, None, None] * identity, acceleration)
        motion = (jac.mT @ native._selection_matrix_motion_b.double() @ force).squeeze(-1)
        limits = self.effort_limits.to(motion)[None]
        base = gravity.double() if native.cfg.gravity_compensation else torch.zeros_like(motion)
        base = base.clamp(-limits, limits)
        # A common scale retains the task torque direction rather than clipping
        # independent joint components into an unrelated task-space command.
        headroom = torch.where(motion >= 0, limits - base, limits + base)
        ratios = torch.where(motion.abs() > 1e-12,
                             headroom / motion.abs().clamp_min(1e-12), torch.inf)
        scale = ratios.amin(-1).clamp(0, 1)
        torque = (base + scale[:, None] * motion).clamp(-limits, limits)
        if not torch.isfinite(torque).all():
            raise ValueError('Nonfinite regularized OSC output; refusing a torque command')
        singular = torch.linalg.svdvals(jac)
        self.last_diagnostics = dict(
            jacobian_condition=(singular[:, 0] / singular[:, -1].clamp_min(1e-12)).detach(),
            regularized_inertia_condition=((maximum + shift) / (minimum + shift)).detach(),
            inertia_regularization=shift.detach(), motion_torque_scale=scale.detach(),
            requested_torque=(base + motion).detach(), commanded_torque=torque.detach(),
            effort_limits=limits.expand_as(torque).detach())
        return torque.to(jacobian_b)

    def diagnostics(self):
        if self.last_diagnostics is None:
            return None
        return {name: value[0].cpu().tolist() for name, value in self.last_diagnostics.items()}
