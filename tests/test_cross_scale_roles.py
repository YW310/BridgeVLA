"""CPU role-transport regressions; no pretrained VLM or simulator required."""

import unittest

import torch

from finetune.bridgevla.models.cross_scale_roles import (
    inherit_coarse_roles, transform_role_geometry,
)


class CrossScaleRolesTest(unittest.TestCase):
    @staticmethod
    def _packet(reference=(.9, .3, .4)):
        points = torch.tensor([[[[.2, .3, .4]], [list(reference)]]])
        centers = points[:, :, 0].flatten(1)
        geometry = torch.cat((centers, torch.full((1, 6), .1),
                              points[:, 1, 0] - points[:, 0, 0], torch.ones(1, 2)), dim=1)
        return {
            'object_slot_points': points,
            'object_slot_valid': torch.ones(1, 2, dtype=torch.bool),
            'object_slot_geometry': geometry.requires_grad_(),
            'object_slot_role_tokens': torch.randn(1, 2, 4, requires_grad=True),
            'object_slot_role_token_valid': torch.ones(1, 2, dtype=torch.bool),
            'object_slot_reference_null_probability': torch.tensor([.2], requires_grad=True),
            'object_slot_reference_is_null': torch.tensor([False]),
            'object_slot_confidence': torch.tensor([[.8, .7]]),
        }

    @staticmethod
    def _project(points, views=1):
        return ((points[..., :2] + 1) * 4).unsqueeze(2).expand(-1, -1, views, -1)

    @staticmethod
    def _xyz(views=1):
        xyz = torch.zeros(1, views, 3, 8, 8)
        xyz[0, 0, :, 5, 4] = torch.tensor([.2, .4, .6])
        return xyz

    def _inherit(self, packet=None, xyz=None, views=1):
        return inherit_coarse_roles(
            self._packet() if packet is None else packet,
            torch.tensor([[.1, .1, .1]]), 2., self._xyz(views) if xyz is None else xyz,
            lambda points: self._project(points, views), sigma=0.,
        )

    def test_actual_crop_transform_and_inverse_keep_global_geometry(self):
        source = self._packet()['object_slot_geometry']
        center = torch.tensor([[.1, .1, .1]])
        transformed = transform_role_geometry(source, center, 2.)
        torch.testing.assert_close(transformed[:, :6].reshape(1, 2, 3) / 2 + center[:, None],
                                   source[:, :6].reshape(1, 2, 3))
        torch.testing.assert_close(transformed[:, 6:12], source[:, 6:12] * 2)
        torch.testing.assert_close(transformed[:, 12:15], source[:, 12:15] * 2)
        self.assertGreater(transformed[0, 3].item(), 1.)  # no boundary clamp

    def test_crop_external_reference_retains_identity_and_null_posterior(self):
        packet = self._packet()
        output = self._inherit(packet)
        torch.testing.assert_close(output['valid'], torch.tensor([[True, False]]))
        self.assertEqual(output['prior'][:, :, 1].count_nonzero().item(), 0)
        self.assertIs(output['role_tokens'], packet['object_slot_role_tokens'])
        self.assertIs(output['reference_null_probability'], packet['object_slot_reference_null_probability'])
        self.assertTrue(output['role_token_valid'].all())
        self.assertTrue(output['geometry'][:, 15:17].bool().all())
        self.assertFalse(output['reference_is_null'].item())
        self.assertNotIn('role_logits', output)
        self.assertTrue(output['roles_inherited'])

    def test_per_view_support_does_not_require_all_views(self):
        output = self._inherit(views=2)
        self.assertTrue(output['valid'][0, 0])
        self.assertGreater(output['prior'][0, 0, 0].sum().item(), 0)
        self.assertEqual(output['prior'][0, 1, 0].count_nonzero().item(), 0)

    def test_nonempty_occluder_pixel_does_not_substitute_for_role_xyz(self):
        xyz = self._xyz()
        xyz[0, 0, :, 5, 4] = torch.tensor([4., 4., 4.])
        output = self._inherit(xyz=xyz)
        self.assertFalse(output['valid'].any())
        self.assertEqual(output['prior'].count_nonzero().item(), 0)

    def test_empty_scene_is_unknown_not_reference_null(self):
        output = self._inherit(xyz=torch.zeros_like(self._xyz()))
        self.assertFalse(output['valid'].any())
        self.assertFalse(output['reference_is_null'].item())
        self.assertTrue(output['role_token_valid'].all())

    def test_null_posterior_is_inherited_and_suppresses_spatial_reference(self):
        packet = self._packet(reference=(.2, .3, .4))
        packet['object_slot_reference_null_probability'] = torch.ones(1)
        packet['object_slot_reference_is_null'] = torch.ones(1, dtype=torch.bool)
        output = self._inherit(packet)
        self.assertTrue(output['valid'][0, 0])
        self.assertFalse(output['valid'][0, 1])
        self.assertEqual(output['prior'][:, :, 1].count_nonzero().item(), 0)

    def test_nan_padded_points_do_not_create_support_or_nonfinite_output(self):
        packet = self._packet()
        packet['object_slot_points'][:, 0] = torch.nan
        output = self._inherit(packet)
        self.assertFalse(output['valid'].any())
        self.assertTrue(torch.isfinite(output['points']).all())
        self.assertEqual(output['prior'].count_nonzero().item(), 0)

    def test_global_tokens_geometry_and_null_keep_gradients(self):
        packet = self._packet()
        output = self._inherit(packet)
        loss = output['role_tokens'].sum() + output['geometry'].sum() + output['reference_null_probability'].sum()
        loss.backward()
        for key in ('object_slot_role_tokens', 'object_slot_geometry',
                    'object_slot_reference_null_probability'):
            self.assertGreater(packet[key].grad.abs().sum().item(), 0)
        self.assertFalse(output['prior'].requires_grad)  # discrete hint, not a learned map

    def test_teacher_fields_are_ignored(self):
        packet = self._packet()
        before = self._inherit(packet)
        packet.update(oracle_target_present=False, oracle_reference_object_points=torch.randn(1, 100, 3),
                      object_slot_target_prior=torch.ones(1, 1, 2, 8, 8))
        after = self._inherit(packet)
        for key in ('prior', 'geometry', 'points', 'role_tokens', 'reference_null_probability'):
            torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)

    def test_unsupported_geometry_does_not_transform_nan_into_fake_xyz(self):
        geometry = torch.zeros(1, 17)
        geometry[:, :15] = torch.nan
        result = transform_role_geometry(geometry, torch.ones(1, 3), 2.)
        self.assertTrue(torch.isfinite(result).all())
        self.assertEqual(result.count_nonzero().item(), 0)

    def test_bad_scale_and_crop_contracts_fail(self):
        geometry = self._packet()['object_slot_geometry']
        for scale in (0., -1., float('nan')):
            with self.assertRaises(ValueError):
                transform_role_geometry(geometry, torch.zeros(1, 3), scale)
        with self.assertRaises(ValueError):
            transform_role_geometry(geometry, torch.zeros(2, 3), 2.)

    def test_fractional_last_pixel_is_visible_and_rasterizes_to_same_pixel(self):
        xyz = torch.zeros_like(self._xyz())
        xyz[0, 0, :, 7, 7] = torch.tensor([.2, .4, .6])
        output = inherit_coarse_roles(
            self._packet(), torch.tensor([[.1, .1, .1]]), 2., xyz,
            lambda points: points.new_full((1, 2, 1, 2), 7.6), sigma=0.,
        )
        self.assertTrue(output['valid'][0, 0])
        self.assertEqual(output['prior'][0, 0, 0, 7, 7].item(), 1.)

    def test_negative_fractional_projection_is_outside_not_clamped_to_zero(self):
        xyz = torch.zeros_like(self._xyz())
        xyz[0, 0, :, 0, 0] = torch.tensor([.2, .4, .6])
        output = inherit_coarse_roles(
            self._packet(), torch.tensor([[.1, .1, .1]]), 2., xyz,
            lambda points: points.new_full((1, 2, 1, 2), -.2), sigma=0.,
        )
        self.assertFalse(output['valid'].any())


if __name__ == '__main__':
    unittest.main()
