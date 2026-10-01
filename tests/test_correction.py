"""Correction and video-only interpolation contracts."""
import pathlib
import sys
import unittest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from correction import correct, resize_keyframes


class CorrectionTests(unittest.TestCase):
    def test_no_correction_returns_direct_lift(self):
        direct = torch.ones(1, 24, 2, 4, 4)
        self.assertIs(correct(direct, None, 0, 0, 1), direct)

    def test_full_correction_returns_pixel_anchor(self):
        direct = torch.zeros(1, 24, 2, 4, 4)
        pixel = torch.ones_like(direct)
        torch.testing.assert_close(correct(direct, pixel, 1, 1, 1), pixel)

    def test_keep_region_untouched(self):
        direct = torch.zeros(1, 2, 1, 2, 2)
        pixel = torch.ones_like(direct)
        mask = torch.tensor([[[[[0, 1], [0, 1]]]]], dtype=torch.float32)
        result = correct(direct, pixel, 1, 1, 1, mask)
        torch.testing.assert_close(result[:, :, :, :, 0], direct[:, :, :, :, 0])
        torch.testing.assert_close(result[:, :, :, :, 1], pixel[:, :, :, :, 1])

    def test_keyframes_are_copied_and_resize_spatial_only(self):
        original = torch.ones(1, 24, 2, 4, 4)
        cond = [[torch.zeros(1), {'minimax_keyframes': [{'latent': original, 'resolved_frame_index': 0}]}]]
        changed = resize_keyframes(cond, 2, 2)
        self.assertEqual(changed[0][1]['minimax_keyframes'][0]['latent'].shape, (1, 24, 2, 2, 2))
        self.assertEqual(original.shape, (1, 24, 2, 4, 4))


if __name__ == '__main__':
    unittest.main()
