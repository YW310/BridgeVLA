"""CPU regression tests for semantic role tokens without local XYZ support."""

import unittest

import torch
from torch import nn

from finetune.bridgevla.models.oracle_prior import OracleRelationAnchorFeatureAdapter


class RoleTokenPreservationTest(unittest.TestCase):
    @staticmethod
    def _adapter(preserve=True):
        adapter = OracleRelationAnchorFeatureAdapter(
            4, rank=2, anchor_rank=2, role_conditioning=True,
            role_token_dim=2, preserve_role_tokens=preserve,
        )
        with torch.no_grad():
            adapter.role_token_projection.weight.copy_(torch.eye(2))
            adapter.role_token_projection.bias.zero_()
            adapter.unknown_reference.zero_()
            adapter.null_reference.copy_(torch.tensor([10., 20.]))
            nn.init.normal_(adapter.anchor_expand.weight, std=.2)
        return adapter

    @staticmethod
    def _inputs():
        return dict(
            features=torch.randn(2, 4, 3, 3),
            prior=torch.ones(1, 2, 2, 3, 3),
            instance_valid=torch.tensor([[True, False]]),
            relation_points=torch.zeros(1, 2, 4, 3),
            role_tokens=torch.tensor([[[1., 3.], [2., 4.]]]),
            role_token_valid=torch.tensor([[True, True]]),
            reference_null_probability=torch.zeros(1),
        )

    @staticmethod
    def _forward_query(adapter, options):
        captured = []
        hook = adapter.anchor_query_encoder[0].register_forward_pre_hook(
            lambda module, inputs: captured.append(inputs[0]),
        )
        try:
            output = adapter.forward_with_anchor(**options)
        finally:
            hook.remove()
        return output, captured[0]

    def test_disabled_flag_is_exact_legacy_and_state_shapes_do_not_change(self):
        torch.manual_seed(42)
        legacy = OracleRelationAnchorFeatureAdapter(
            4, rank=2, anchor_rank=2, role_conditioning=True, role_token_dim=2,
        )
        explicit_off = self._adapter(preserve=False)
        explicit_off.load_state_dict(legacy.state_dict(), strict=True)
        enabled = self._adapter()
        enabled.load_state_dict(legacy.state_dict(), strict=True)
        self.assertEqual(list(legacy.state_dict()), list(enabled.state_dict()))
        options = self._inputs()
        options.pop('role_token_valid')
        expected = legacy.forward_with_anchor(**options)
        actual = explicit_off.forward_with_anchor(
            **options, role_token_valid=torch.zeros(1, 2).bool(),
        )
        for before, after in zip(expected, actual):
            torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_zero_initialized_residual_remains_identity(self):
        adapter = OracleRelationAnchorFeatureAdapter(
            4, rank=2, anchor_rank=2, role_conditioning=True,
            role_token_dim=2, preserve_role_tokens=True,
        )
        options = self._inputs()
        final, shared, _ = adapter.forward_with_anchor(**options)
        torch.testing.assert_close(final, options['features'], rtol=0, atol=0)
        torch.testing.assert_close(shared, options['features'], rtol=0, atol=0)

    def test_crop_external_reference_token_receives_action_gradient(self):
        torch.manual_seed(7)
        adapter = self._adapter()
        options = self._inputs()
        options['role_tokens'].requires_grad_()
        (final, _, _), query = self._forward_query(adapter, options)
        torch.testing.assert_close(
            query[:, :, 2:4], options['role_tokens'][:, None, 1].expand(1, 2, 2),
        )
        final.square().mean().backward()
        self.assertGreater(options['role_tokens'].grad[:, 1].abs().sum().item(), 0)

    def test_unreliable_semantic_token_is_masked_without_becoming_null(self):
        adapter = self._adapter()
        options = self._inputs()
        options['role_token_valid'][:, 1] = False
        options['role_tokens'].requires_grad_()
        (_, _, anchor), query = self._forward_query(adapter, options)
        torch.testing.assert_close(query[:, :, 2:4], torch.zeros(1, 2, 2))
        anchor.sum().backward()
        torch.testing.assert_close(options['role_tokens'].grad[:, 1], torch.zeros(1, 2))
        # Geometry absence selects unknown_reference (zero), not the nonzero NULL.
        self.assertFalse(torch.equal(query[:, :, 2:4],
                                     adapter.null_reference[None, None].expand(1, 2, 2)))

    def test_reference_null_mass_is_applied_once_to_preserved_token(self):
        adapter = self._adapter()
        options = self._inputs()
        options['reference_null_probability'] = torch.tensor([.25])
        (_, _, _), query = self._forward_query(adapter, options)
        expected = .75 * options['role_tokens'][:, 1] + .25 * adapter.null_reference
        torch.testing.assert_close(query[:, :, 2:4], expected[:, None].expand(1, 2, 2))
        options['reference_null_probability'] = torch.ones(1)
        options['role_tokens'].requires_grad_()
        (_, _, anchor), query = self._forward_query(adapter, options)
        torch.testing.assert_close(query[:, :, 2:4],
                                   adapter.null_reference[None, None].expand(1, 2, 2))
        anchor.sum().backward()
        torch.testing.assert_close(options['role_tokens'].grad[:, 1], torch.zeros(1, 2))

    def test_empty_support_does_not_fabricate_geometry_from_semantic_tokens(self):
        adapter = self._adapter()
        options = self._inputs()
        options['instance_valid'].zero_()
        geometry = torch.full((1, 17), float('nan'))
        geometry[:, 15:17] = 0
        options['geometry'] = geometry
        (_, shared, _), query = self._forward_query(adapter, options)
        torch.testing.assert_close(query[:, :, 4:21], torch.zeros(1, 2, 17))
        torch.testing.assert_close(shared, options['features'], rtol=0, atol=0)
        self.assertTrue(torch.isfinite(query).all())
        self.assertTrue(torch.isnan(geometry[:, :15]).all())
        self.assertGreater(query[:, :, :4].abs().sum().item(), 0)

    def test_supported_coarse_geometry_survives_missing_local_reference(self):
        adapter = self._adapter()
        options = self._inputs()
        geometry = torch.tensor([[.1, .2, .3, .4, .5, .6,
                                  .01, .02, .03, .04, .05, .06,
                                  .3, .3, .3, 1., 1.]])
        options['geometry'] = geometry
        (_, _, _), query = self._forward_query(adapter, options)
        torch.testing.assert_close(query[:, :, 4:21], geometry[:, None].expand(1, 2, 17))

    def test_unreliable_target_does_not_enable_semantic_anchor_residual(self):
        adapter = self._adapter()
        options = self._inputs()
        options['instance_valid'].zero_()
        options['role_token_valid'][:, 0] = False
        final, shared, _ = adapter.forward_with_anchor(**options)
        torch.testing.assert_close(final, shared, rtol=0, atol=0)

    def test_semantic_mask_defaults_to_local_reliability(self):
        adapter = self._adapter()
        options = self._inputs()
        options.pop('role_token_valid')
        (_, _, _), query = self._forward_query(adapter, options)
        torch.testing.assert_close(query[:, :, 2:4], torch.zeros(1, 2, 2))

    def test_opt_in_requires_role_conditioning_and_valid_mask_shape(self):
        with self.assertRaisesRegex(ValueError, 'requires role_conditioning'):
            OracleRelationAnchorFeatureAdapter(4, preserve_role_tokens=True)
        adapter = self._adapter()
        options = self._inputs()
        options['role_token_valid'] = torch.ones(1, 3).bool()
        with self.assertRaisesRegex(ValueError, 'role_token_valid'):
            adapter.forward_with_anchor(**options)


if __name__ == '__main__':
    unittest.main()
