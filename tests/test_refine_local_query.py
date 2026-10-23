"""Coarse-conditioned local queries: spatial evidence, gradients and routing."""

import unittest
from unittest import mock

import torch

from finetune.bridgevla.models.refine_role_masks import RoleConditionedRefineMaskHead


class RefineLocalQueryTest(unittest.TestCase):
    def _inputs(self, *, point_support=True):
        torch.manual_seed(23)
        head = RoleConditionedRefineMaskHead(4, 8, hidden_dim=8, local_query=True)
        features = torch.randn(3, 4, 4, 4, requires_grad=True)
        rgb = torch.rand(1, 3, 3, 16, 16, requires_grad=True)
        xyz = torch.full_like(rgb, .25)
        geometry = torch.tensor([[.25] * 6 + [.1] * 6 + [0.] * 3 + [1.] * 2],
                                requires_grad=True)
        hints = torch.zeros(1, 3, 2, 16, 16)
        if point_support:
            hints[:, :, :, 5:9, 5:9] = .8
        packet = dict(
            role_tokens=torch.randn(1, 2, 8, requires_grad=True),
            role_token_valid=torch.ones(1, 2, dtype=torch.bool),
            valid=torch.full((1, 2), point_support, dtype=torch.bool),
            prior=hints, geometry=geometry,
            reference_null_probability=torch.tensor([.2], requires_grad=True),
            reference_is_null=torch.tensor([False]),
            confidence=torch.tensor([[.8, .7]]), points=torch.full((1, 2, 4, 3), .25),
            roles_inherited=True, predictor_type='role_queries',
        )
        return head, features, rgb, xyz, packet

    @staticmethod
    def _enable_update(head):
        with torch.no_grad():
            head.local_query.token_expand.weight.normal_(std=.1)

    def test_disabled_path_retains_old_state_and_zero_update_matches_static_head(self):
        head, features, rgb, xyz, packet = self._inputs()
        legacy = RoleConditionedRefineMaskHead(4, 8, hidden_dim=8)
        incompatible = head.load_state_dict(legacy.state_dict(), strict=False)
        self.assertTrue(incompatible.missing_keys)
        self.assertTrue(all(key.startswith('local_query.') for key in incompatible.missing_keys))
        self.assertFalse(incompatible.unexpected_keys)
        self.assertIsNone(legacy.local_query)
        self.assertFalse(any('local_query' in key for key in legacy.state_dict()))
        old, new = legacy(features, rgb, xyz, packet), head(features, rgb, xyz, packet)
        torch.testing.assert_close(new['role_tokens'], packet['role_tokens'], rtol=0, atol=0)
        torch.testing.assert_close(new['prior'], old['prior'], rtol=0, atol=0)
        self.assertEqual(new['refine_query_update_norm'].count_nonzero().item(), 0)
        self.assertTrue(new['queries_refined'])
        self.assertTrue(new['roles_inherited'])
        self.assertEqual(new['predictor_type'], 'role_queries')
        for key in ('geometry', 'reference_null_probability', 'role_token_valid', 'points'):
            self.assertIs(new[key], packet[key])

    def test_observed_geometry_recovers_missing_point_hints_without_teacher(self):
        head, features, rgb, xyz, packet = self._inputs(point_support=False)
        result = head(features, rgb, xyz, packet)
        self.assertFalse(packet['valid'].any())
        self.assertTrue(result['valid'].all())
        self.assertGreater(result['prior'].sum().item(), 0)
        self.assertFalse(result['refine_point_hint_valid'].any())
        self.assertTrue(result['refine_query_active'].all())
        self.assertFalse(result['reference_is_null'].item())

    def test_crop_external_reference_keeps_global_token_and_does_not_select_distractor(self):
        head, features, rgb, xyz, packet = self._inputs(point_support=False)
        self._enable_update(head)
        with torch.no_grad():
            packet['geometry'][:, 3:6] = 10.
        result = head(features, rgb, xyz, packet)
        self.assertTrue(result['valid'][0, 0])
        self.assertFalse(result['valid'][0, 1])
        self.assertEqual(result['prior'][:, :, 1].count_nonzero().item(), 0)
        torch.testing.assert_close(result['role_tokens'][:, 1], packet['role_tokens'][:, 1],
                                   rtol=0, atol=0)
        self.assertGreater((result['role_tokens'][:, 0] - packet['role_tokens'][:, 0]).abs().sum(), 0)
        self.assertTrue(result['role_token_valid'].all())
        self.assertFalse(result['reference_is_null'].item())

    def test_empty_nan_and_background_crops_never_update_or_hallucinate(self):
        for value in (0., -1., float('nan')):
            head, features, rgb, xyz, packet = self._inputs()
            self._enable_update(head)
            xyz[:] = value
            result = head(features, rgb, xyz, packet)
            self.assertFalse(result['valid'].any())
            self.assertEqual(result['prior'].count_nonzero().item(), 0)
            torch.testing.assert_close(result['role_tokens'], packet['role_tokens'], rtol=0, atol=0)
            self.assertTrue(torch.isfinite(result['refine_mask_logits']).all())
            self.assertTrue(torch.isfinite(result['role_tokens']).all())
            self.assertFalse(result['reference_is_null'].item())

    def test_mixed_batch_empty_crop_does_not_affect_observed_sample(self):
        head, features, rgb, xyz, packet = self._inputs()
        self._enable_update(head)
        batched = {key: value.repeat((2,) + (1,) * (value.ndim - 1))
                   if isinstance(value, torch.Tensor) else value
                   for key, value in packet.items()}
        xyz = xyz.repeat(2, 1, 1, 1, 1)
        xyz[0] = 0.
        result = head(features.repeat(2, 1, 1, 1), rgb.repeat(2, 1, 1, 1, 1), xyz, batched)
        self.assertFalse(result['valid'][0].any())
        self.assertTrue(result['valid'][1].all())
        torch.testing.assert_close(result['role_tokens'][0], batched['role_tokens'][0])
        self.assertGreater((result['role_tokens'][1] - batched['role_tokens'][1]).abs().sum(), 0)

    def test_nonfinite_geometry_without_hints_remains_unknown(self):
        head, features, rgb, xyz, packet = self._inputs(point_support=False)
        with torch.no_grad():
            packet['geometry'][:, :12] = float('nan')
        result = head(features, rgb, xyz, packet)
        self.assertFalse(result['valid'].any())
        self.assertTrue(torch.isfinite(result['role_tokens']).all())
        self.assertTrue(torch.isfinite(result['refine_mask_logits']).all())

    def test_invalid_geometry_and_unknown_role_do_not_create_local_evidence(self):
        head, features, rgb, xyz, packet = self._inputs(point_support=False)
        with torch.no_grad():
            packet['geometry'][:, 15:] = 0.
        result = head(features, rgb, xyz, packet)
        self.assertFalse(result['valid'].any())
        packet['prior'].fill_(.5)
        packet['role_token_valid'][:, 1] = False
        result = head(features, rgb, xyz, packet)
        self.assertTrue(result['valid'][0, 0])
        self.assertFalse(result['valid'][0, 1])

    def test_reference_null_is_inherited_and_applied_once(self):
        head, features, rgb, xyz, packet = self._inputs()
        result = head(features, rgb, xyz, packet)
        raw = result['refine_mask_logits'].sigmoid()
        torch.testing.assert_close(result['prior'][:, :, 0], raw[:, :, 0])
        torch.testing.assert_close(result['prior'][:, :, 1], raw[:, :, 1] * .8)
        with torch.no_grad():
            packet['reference_null_probability'].fill_(1.)
        result = head(features, rgb, xyz, packet)
        self.assertFalse(result['valid'][0, 1])
        self.assertEqual(result['prior'][:, :, 1].count_nonzero().item(), 0)
        torch.testing.assert_close(result['role_tokens'][:, 1], packet['role_tokens'][:, 1])

    def test_roles_do_not_self_mix_in_local_query_update(self):
        head, features, rgb, xyz, packet = self._inputs()
        self._enable_update(head)
        first = head(features, rgb, xyz, packet)
        changed = dict(packet, role_tokens=packet['role_tokens'].detach().clone())
        changed['role_tokens'][:, 1] += torch.arange(8)
        second = head(features, rgb, xyz, changed)
        torch.testing.assert_close(first['role_tokens'][:, 0], second['role_tokens'][:, 0])
        torch.testing.assert_close(first['prior'][:, :, 0], second['prior'][:, :, 0])

    def test_updated_queries_use_current_local_features_and_both_spatial_cues(self):
        head, features, rgb, xyz, packet = self._inputs()
        self._enable_update(head)
        first = head(features, rgb, xyz, packet)
        changed = head(features + torch.randn_like(features), rgb, xyz, packet)
        self.assertGreater((first['role_tokens'] - changed['role_tokens']).abs().sum(), 0)
        with torch.no_grad():
            head.local_query.spatial_gain.fill_(1.)
        spatial = head(features, rgb, xyz, packet)
        no_hints = head(features, rgb, xyz, dict(packet, prior=torch.zeros_like(packet['prior'])))
        self.assertGreater((spatial['refine_mask_logits'] - no_hints['refine_mask_logits']).abs().sum(), 0)
        geometry = packet['geometry'].detach().clone()
        geometry[:, :6] += .05
        shifted = head(features, rgb, xyz, dict(packet, geometry=geometry))
        self.assertGreater((spatial['refine_mask_logits'] - shifted['refine_mask_logits']).abs().sum(), 0)

    def test_zero_initialization_learns_projection_then_attention_and_geometry(self):
        head, features, rgb, xyz, packet = self._inputs()
        with torch.no_grad():
            packet['geometry'][:, :6] += .05
        optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
        result = head(features, rgb, xyz, packet)
        result['prior'].square().mean().backward()
        self.assertGreater(head.local_query.token_expand.weight.grad.abs().sum(), 0)
        optimizer.step()
        optimizer.zero_grad()
        with torch.no_grad():
            head.local_query.spatial_gain.fill_(.1)
        result = head(features, rgb, xyz, packet)
        result['prior'].square().mean().backward()
        for name, value in (
            ('queries', head.local_query.query.weight), ('keys', head.local_query.key.weight),
            ('values', head.local_query.value.weight), ('coarse', packet['role_tokens']),
            ('geometry', packet['geometry']), ('features', features), ('rgb', rgb),
        ):
            with self.subTest(parameter=name):
                self.assertIsNotNone(value.grad)
                self.assertTrue(torch.isfinite(value.grad).all())
                self.assertGreater(value.grad.abs().sum(), 0, name)

    def test_actual_policy_routes_updated_tokens_to_anchor_even_when_maps_are_detached(self):
        from test_refine_role_masks import RefineRoleMaskTest
        helper, module, _, options, _ = RefineRoleMaskTest()._policy_inputs()
        from bridgevla.mvt import mvt_single
        head = RoleConditionedRefineMaskHead(4, 8, hidden_dim=8, local_query=True)
        packet = options['inherited_object_roles']
        packet['prior'] = torch.zeros(1, 3, 2, 16, 16)
        packet['geometry'] = torch.tensor([[.5] * 6 + [.5] * 6 + [0.] * 3 + [1.] * 2])
        options.update(refine_role_mask_head=head, refine_mask_detach=True,
                       wpt_local=torch.tensor([[2., 2., 0.]]))
        module.train()
        with mock.patch.object(mvt_single, 'select_feat_from_hm', helper._sample, create=True):
            result = module(**options)
            teacher = torch.zeros(1, 3, 2, 16, 16)
            labelled = module(**options, object_slot_target_heatmap=teacher)
            changed_teacher = module(**options, object_slot_target_heatmap=1 - teacher)
        for key in ('trans', 'feat_ex_rot', 'object_slot_prior', 'object_slot_role_tokens'):
            torch.testing.assert_close(result[key], labelled[key], rtol=0, atol=0)
            torch.testing.assert_close(result[key], changed_teacher[key], rtol=0, atol=0)
        self.assertTrue(result['object_slot_queries_refined'])
        self.assertTrue(result['object_refine_query_active'].all())
        self.assertFalse(result['object_slot_prior'].requires_grad)
        (result['trans'].square().mean() + result['feat_ex_rot'].square().mean()).backward()
        self.assertGreater(head.local_query.token_expand.weight.grad.abs().sum(), 0)
        self.assertIsNone(head.query.weight.grad)


if __name__ == '__main__':
    unittest.main()
