"""Execute split nodes with real H3 AV containers and a fake Euler sampler."""
import importlib.util
import pathlib
import sys
import unittest
from unittest.mock import patch

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))
spec = importlib.util.spec_from_file_location('selflift_split_test', ROOT / '__init__.py', submodule_search_locations=[str(ROOT)])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
from selflift_split_test import nodes
from comfy.nested_tensor import NestedTensor
from comfy.model_sampling import CONST
from comfy.latent_formats import MiniMaxH3AV


class FakeModel:
    def __init__(self):
        self.load_device = torch.device('cpu')
        self.model_options = {}
        self.sampling = CONST()
        self.format = MiniMaxH3AV()
        self.model = self

    def get_model_object(self, name):
        return {'model_sampling': self.sampling, 'latent_format': self.format}[name]

    def process_latent_out(self, latent):
        return self.format.process_out(latent)

    def clone(self):
        return self


class NodeTests(unittest.TestCase):
    def setUp(self):
        self.video = torch.full((1, 24, 2, 4, 6), 0.25)
        self.audio = torch.full((1, 32, 2, 4), 0.1)
        self.latent = {'samples': NestedTensor((self.video, self.audio))}
        self.sigmas = torch.tensor([1., .8, .6, .4, .2, 0.])
        self.model = FakeModel()

    def run_low(self):
        def sample(model, noise, positive, negative, cfg, device, sampler, sigmas, options, **kwargs):
            x = NestedTensor((torch.ones(1, 24, 2, 2, 4), torch.ones_like(self.audio) * 2))
            x0 = NestedTensor((torch.ones(1, 24, 2, 2, 4)*.4, torch.ones_like(self.audio)*.6))
            kwargs['callback'](0, x0, x, 3)
            kwargs['callback'](1, x0, x, 3)
            kwargs['callback'](2, x0, x, 3)
            return NestedTensor((torch.ones_like(x.unbind()[0])*.9, torch.ones_like(self.audio)*.7))
        with patch.object(nodes.comfy.samplers, 'sample', side_effect=sample), patch.object(nodes.comfy.sample, 'prepare_noise', side_effect=lambda latent, seed, batch: latent), patch.object(nodes.comfy.model_management, 'intermediate_device', return_value=torch.device('cpu')):
            return nodes.SelfLiftSplitLowH3().sample(self.model, [], [], self.latent, nodes.comfy.samplers.ksampler('euler'), self.sigmas, 42, 1., 3, .5)

    def test_low_captures_callback_and_outputs_clean_video(self):
        state, clean, _, _, tw, th = self.run_low()
        self.assertEqual(clean['samples'].shape, (1, 24, 2, 2, 4))
        self.assertEqual(state.low_x.unbind()[1].shape, self.audio.shape)
        self.assertAlmostEqual(state.low_output.unbind()[1][0, 0, 0, 0].item(), .7)
        self.assertEqual((tw, th), (6 * 16, 4 * 16))

    def test_all_boundary_modes_pack_av_and_share_sigma_tail(self):
        state, clean, _, _, _, _ = self.run_low()
        lifted = {'samples': torch.ones(1, 24, 2, 4, 6)*.4}
        with patch.object(nodes.comfy.sample, 'prepare_noise', side_effect=lambda x, seed, batch: torch.ones_like(x)), patch.object(nodes.comfy.model_management, 'intermediate_device', return_value=torch.device('cpu')):
            values = {}
            for mode in ('upstream', 'legacy_seed', 'simple_interpolation', 'simple_no_noise'):
                result, tail, _ = nodes.SelfLiftSplitTransitionH3().transition(state, lifted, self.model, mode, 0, .5, 1)
                self.assertIsInstance(result['samples'], NestedTensor)
                self.assertEqual(result['samples'].unbind()[0].shape, self.video.shape)
                torch.testing.assert_close(tail, self.sigmas[3:])
                self.assertNotIn('noise_mask', result)
                values[mode] = result['samples'].unbind()[0][0, 0, 0, 0, 0].item()
            # With the test noise mocked to the same tensor, the Euler boundary
            # reconstruction and direct resume-sigma re-noise are algebraically equal.
            self.assertAlmostEqual(values['upstream'], values['legacy_seed'])
            self.assertEqual(values['legacy_seed'], values['simple_interpolation'])
            self.assertNotEqual(values['simple_interpolation'], values['simple_no_noise'])

    def test_audio_handoff_matches_second_sampler_space(self):
        # second pass: process_latent_in multiplies audio by audio_scale, then starts from
        # (1 - sigma) * latent; that must equal the true low-pass state x at sigma_next.
        self.model.sampling.audio_scale = 4.0
        state, _, _, _, _, _ = self.run_low()
        lifted = {'samples': torch.ones(1, 24, 2, 4, 6) * .4}
        sigma_k, sigma_next = self.sigmas[2], self.sigmas[3]
        x_audio = state.low_x.unbind()[1]
        x0_audio = state.low_x0.unbind()[1]
        true_next = x_audio + (x_audio - x0_audio) * ((sigma_next - sigma_k) / sigma_k)
        with patch.object(nodes.comfy.sample, 'prepare_noise', side_effect=lambda x, seed, batch: torch.ones_like(x)), patch.object(nodes.comfy.model_management, 'intermediate_device', return_value=torch.device('cpu')):
            result, _, _ = nodes.SelfLiftSplitTransitionH3().transition(state, lifted, self.model, 'upstream', 0, .5, 1)
        resumed = (1 - sigma_next) * result['samples'].unbind()[1] * 4.0
        torch.testing.assert_close(resumed, true_next)
        with patch.object(nodes.comfy.sample, 'prepare_noise', side_effect=lambda x, seed, batch: torch.ones_like(x)), patch.object(nodes.comfy.model_management, 'intermediate_device', return_value=torch.device('cpu')):
            result, _, _ = nodes.SelfLiftSplitTransitionH3().transition(state, lifted, self.model, 'simple_no_noise', 0, .5, 1)
        torch.testing.assert_close(result['samples'].unbind()[1], state.low_output.unbind()[1])
        with patch.object(nodes.comfy.sample, 'prepare_noise', side_effect=lambda x, seed, batch: torch.ones_like(x)), patch.object(nodes.comfy.model_management, 'intermediate_device', return_value=torch.device('cpu')):
            result, _, _ = nodes.SelfLiftSplitTransitionH3().transition(state, lifted, self.model, 'simple_interpolation', 0, .5, 1)
        self.assertFalse(torch.equal(result['samples'].unbind()[0], state.low_output.unbind()[0]))

    def test_resize_video_only_rejects_av(self):
        state, clean, _, _, _, _ = self.run_low()
        up, = nodes.SelfLiftSplitResizeH3().resize(state, clean, 'nearest')
        self.assertEqual(up['samples'].shape, self.video.shape)
        with self.assertRaises(ValueError):
            nodes.SelfLiftSplitResizeH3().resize(state, self.latent, 'nearest')


if __name__ == '__main__':
    unittest.main()
