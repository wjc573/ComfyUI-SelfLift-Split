"""Math and handoff contracts for the independent H3 split nodes."""
import importlib.util
import pathlib
import sys
import unittest

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import (boundary_video, euler_step, low_size, validate_schedule, validate_video,
                  video_endpoint)


class Flow:
    noise_scale = 1.0

    def noise_scaling(self, sigma, noise, latent):
        return sigma * noise + (1 - sigma) * latent

    def inverse_noise_scaling(self, sigma, latent):
        return latent / (1 - sigma)


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.schedule = torch.tensor([1.0, 0.9, 0.7, 0.55, 0.4, 0.25, 0.15, 0.05, 0.0])

    def test_schedule_retains_shared_boundary(self):
        left, right = validate_schedule(self.schedule, 3)
        self.assertTrue(torch.equal(left, self.schedule[:4]))
        self.assertTrue(torch.equal(right, self.schedule[3:]))

    def test_schedule_rejects_invalid_transition(self):
        with self.assertRaises(ValueError):
            validate_schedule(self.schedule, 0)
        with self.assertRaises(ValueError):
            validate_schedule(self.schedule, 8)
        with self.assertRaises(ValueError):
            validate_schedule(torch.tensor([1.0, 0.5, 0.6, 0.0]), 1)

    def test_low_size_uses_original_even_grid_rounding(self):
        self.assertEqual(low_size(45, 80, .5), (22, 40))

    def test_video_endpoint_is_external_and_shape_checked(self):
        video = torch.zeros(1, 24, 7, 22, 40)
        lifted = torch.ones(1, 24, 7, 44, 80)
        validate_video(lifted, video, (44, 80))
        with self.assertRaisesRegex(ValueError, 'target'):
            validate_video(lifted[..., :-1], video, (44, 80))
        with self.assertRaisesRegex(ValueError, 'temporal|channel'):
            validate_video(lifted[:, :, :-1], video, (44, 80))

    def test_upstream_boundary_reuses_euler_prediction_and_seed_plus_one(self):
        clean = torch.full((1, 1, 1, 2, 2), 2.)
        noise = torch.full_like(clean, 3.)
        sigma_k, sigma_next = self.schedule[2:4]
        got = boundary_video(clean, noise, sigma_k, sigma_next, Flow(), 'upstream')
        expected = euler_step(Flow().noise_scaling(sigma_k, noise, clean), clean, sigma_k, sigma_next)
        torch.testing.assert_close(got, Flow().inverse_noise_scaling(sigma_next, expected))
        self.assertEqual(video_endpoint('upstream', 42), (43, 'sigma_k'))

    def test_legacy_boundary_reuses_original_seed_at_resume_sigma(self):
        clean = torch.full((1, 1, 1, 2, 2), 2.)
        noise = torch.full_like(clean, 3.)
        got = boundary_video(clean, noise, self.schedule[2], self.schedule[3], Flow(), 'legacy_seed')
        torch.testing.assert_close(got, clean + self.schedule[3] / (1-self.schedule[3])*noise)
        self.assertEqual(video_endpoint('legacy_seed', 42), (42, 'sigma_next'))

    def test_simple_boundary_uses_no_random_noise_at_same_sigma(self):
        clean = torch.full((1, 1, 1, 2, 2), 2.)
        got = boundary_video(clean, None, self.schedule[2], self.schedule[3], Flow(), 'simple_no_noise')
        torch.testing.assert_close(got, clean)
        self.assertEqual(video_endpoint('simple_no_noise', 42), (None, 'sigma_next'))

    def test_simple_interpolation_re_noises_at_resume_sigma(self):
        clean = torch.full((1, 1, 1, 2, 2), 2.)
        noise = torch.full_like(clean, 3.)
        sigma_next = self.schedule[3]
        got = boundary_video(clean, noise, self.schedule[2], sigma_next, Flow(), 'simple_interpolation')
        torch.testing.assert_close(got, clean + sigma_next / (1 - sigma_next) * noise)
        self.assertEqual(video_endpoint('simple_interpolation', 42), (42, 'sigma_next'))


if __name__ == '__main__':
    unittest.main()
