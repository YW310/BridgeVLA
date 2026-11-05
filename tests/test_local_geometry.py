"""Synthetic geometric invariants; no simulator, detector, or VLM required."""

import math
import unittest

import torch

from finetune.bridgevla.models.local_geometry import build_local_evidence


class LocalGeometryTest(unittest.TestCase):
    @staticmethod
    def _scene(views=1, batch=1):
        depth = torch.ones(batch, views, 5, 5)
        intrinsics = torch.tensor([[2., 0., 2.], [0., 2., 2.], [0., 0., 1.]])
        intrinsics = intrinsics.expand(batch, views, 3, 3).clone()
        cameras = torch.eye(4).expand(batch, views, 4, 4).clone()
        candidate = torch.eye(4).expand(batch, 4, 4).clone()
        candidate[:, 2, 3] = 1.
        return depth, intrinsics, cameras, candidate

    @staticmethod
    def _one_voxel(scene, **kwargs):
        result = build_local_evidence(*scene, resolution=1, **kwargs)
        return result['field'][0, :, 0, 0, 0]

    def test_plane_preserves_free_surface_and_occluded_unknown(self):
        result = build_local_evidence(*self._scene(), size_m=0.6, resolution=3,
                                      truncation_m=0.05)
        field = result['field']
        self.assertEqual(tuple(field.shape), (1, 5, 3, 3, 3))
        # D is local Z: slices at camera Z = .8, 1., 1.2 respectively.
        torch.testing.assert_close(field[0, :, 0, 1, 1],
                                   torch.tensor([1., 0., 1., 1., 0.]))
        torch.testing.assert_close(field[0, :, 1, 1, 1],
                                   torch.tensor([0., 1., 0., 1., 0.]))
        torch.testing.assert_close(field[0, :, 2, 1, 1], torch.zeros(5))
        torch.testing.assert_close(result['known_fraction'], torch.tensor([2. / 3.]))

    def test_invalid_depth_and_outside_or_behind_camera_are_unknown(self):
        for invalid in (0., -1., float('nan'), float('inf'), -float('inf')):
            with self.subTest(depth=invalid):
                scene = self._scene()
                scene[0].fill_(invalid)
                torch.testing.assert_close(self._one_voxel(scene), torch.zeros(5))
        for position in ((20., 0., 1.), (0., 0., -1.), (0., 0., 0.)):
            with self.subTest(position=position):
                scene = self._scene()
                scene[3][0, :3, 3] = torch.tensor(position)
                torch.testing.assert_close(self._one_voxel(scene), torch.zeros(5))

    def test_common_rigid_world_transform_preserves_rotated_camera_evidence(self):
        scene = self._scene()
        reference = build_local_evidence(*scene, size_m=.6, resolution=3,
                                         truncation_m=.05)
        angle = .73
        transform = torch.tensor([[math.cos(angle), 0., math.sin(angle), .4],
                                  [0., 1., 0., -.2],
                                  [-math.sin(angle), 0., math.cos(angle), .8],
                                  [0., 0., 0., 1.]])
        transformed = (scene[0], scene[1], transform @ scene[2], transform @ scene[3])
        result = build_local_evidence(*transformed, size_m=.6, resolution=3,
                                      truncation_m=.05)
        torch.testing.assert_close(result['field'], reference['field'], atol=5e-6, rtol=0)
        torch.testing.assert_close(result['known_fraction'], reference['known_fraction'])

    def test_opposing_camera_can_reveal_otherwise_occluded_surface(self):
        scene = self._scene(views=2)
        scene[3][0, 2, 3] = 1.5
        # Camera 0 sees a front occluder at world Z=1; camera 1 looks back
        # from Z=2 and sees the query surface at its own camera Z=.5.
        scene[2][0, 1, :3, :3] = torch.diag(torch.tensor([-1., 1., -1.]))
        scene[2][0, 1, 2, 3] = 2.
        scene[0][0, 1].fill_(.5)
        torch.testing.assert_close(self._one_voxel(scene),
                                   torch.tensor([0., .5, 0., 1., 0.]))

    def test_cross_view_contradiction_retains_both_support_types(self):
        scene = self._scene(views=2)
        # Identical poses, contradictory depths: preserve this disagreement
        # rather than turning their averaged depth into an invented surface.
        scene[0][0, 1].fill_(1.3)
        torch.testing.assert_close(self._one_voxel(scene, truncation_m=.05),
                                   torch.tensor([.5, .5, .5, 1., 1.]))

    def test_depth_edges_use_nearest_without_phantom_interpolated_surface(self):
        scene = self._scene()
        scene[0][0, 0, :, :3] = .8
        scene[0][0, 0, :, 3:] = 1.3
        scene[3][0, 0, 3] = .2  # projected U=2.4; nearest U=2 has depth .8.
        # Bilinear interpolation would invent depth .8*.6 + 1.3*.4 = 1.0.
        torch.testing.assert_close(self._one_voxel(scene, truncation_m=.02), torch.zeros(5))

    def test_near_surface_retains_signed_distance_but_far_behind_is_unknown(self):
        for value, expected in ((.998, [-.4, 1., 0., 1., 0.]),
                                (1.003, [.6, 1., 0., 1., 0.]),
                                (.99, [0., 0., 0., 0., 0.])):
            with self.subTest(depth=value):
                scene = self._scene()
                scene[0].fill_(value)
                torch.testing.assert_close(self._one_voxel(scene), torch.tensor(expected),
                                           atol=1e-5, rtol=0)

    def test_off_axis_metric_z_is_not_radial_range(self):
        scene = self._scene()
        scene[3][0, 0, 3] = .75  # camera point (.75,0,1) has range=1.25 m.
        torch.testing.assert_close(self._one_voxel(scene),
                                   torch.tensor([0., 1., 0., 1., 0.]))
        # Passing range without conversion would falsely classify the voxel free.
        scene[0].fill_(1.25)
        torch.testing.assert_close(self._one_voxel(scene),
                                   torch.tensor([1., 0., 1., 1., 0.]))

    def test_chunk_parity_batching_float32_and_detached_outputs(self):
        scene = list(self._scene(views=2, batch=2))
        scene[0][1, 0].fill_(0.)
        scene = [value.double().requires_grad_() for value in scene]
        small = build_local_evidence(*scene, size_m=.6, resolution=5,
                                     truncation_m=.05, chunk_size=7)
        large = build_local_evidence(*scene, size_m=.6, resolution=5,
                                     truncation_m=.05, chunk_size=4096)
        for name in ('field', 'known_fraction'):
            torch.testing.assert_close(small[name], large[name], atol=0, rtol=0)
            self.assertEqual(small[name].dtype, torch.float32)
            self.assertFalse(small[name].requires_grad)
        self.assertEqual(tuple(small['known_fraction'].shape), (2,))

    def test_validation_rejects_invalid_shapes_parameters_and_calibration(self):
        for keyword in ('size_m', 'truncation_m'):
            for value in (0., -1., float('nan'), float('inf'), True):
                with self.subTest(parameter=keyword, value=value), self.assertRaises(ValueError):
                    build_local_evidence(*self._scene(), **{keyword: value})
        for keyword in ('resolution', 'chunk_size'):
            for value in (0, -1, 2.5, True):
                with self.subTest(parameter=keyword, value=value), self.assertRaises(ValueError):
                    build_local_evidence(*self._scene(), **{keyword: value})
        scene = list(self._scene())
        scene[1] = scene[1][:, 0]
        with self.assertRaisesRegex(ValueError, 'intrinsics.*shape'):
            build_local_evidence(*scene)
        for index, row, column, value in ((1, 0, 0, 0.), (1, 2, 0, .2),
                                          (2, 0, 0, 2.), (2, 0, 0, -1.),
                                          (2, 3, 0, .1), (3, 0, 0, float('nan'))):
            scene = self._scene()
            scene[index][..., row, column] = value
            with self.subTest(input=index, row=row, column=column, value=value):
                with self.assertRaises(ValueError):
                    build_local_evidence(*scene)


if __name__ == '__main__':
    unittest.main()
