"""Temporal compatibility contracts using the real upstream window planner."""
import importlib.util
import os
import pathlib
import sys
import unittest
from unittest.mock import patch

import torch
from test_nodes import nodes, FakeModel
from comfy.nested_tensor import NestedTensor
import comfy.patcher_extension

TW = pathlib.Path(os.environ.get('SELFLIFT_TW_PATH',
    pathlib.Path(__file__).resolve().parents[2] / 'ComfyUI-MiniMaxH3-TemporalWindow-SpatialTiles' / 'h3_windows.py'))
if not TW.is_file():
    raise unittest.SkipTest('Optional TemporalWindow source not installed; set SELFLIFT_TW_PATH to h3_windows.py')
spec = importlib.util.spec_from_file_location('split_temporal_test_upstream', TW)
hw = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = hw
spec.loader.exec_module(hw)


class WindowModel(FakeModel):
    def __init__(self, wrapper):
        super().__init__()
        self.wrappers = {'minimax_h3_windows': [wrapper]} if wrapper else {}
        self.sampling.audio_scale = 4.0

    def get_all_wrappers(self, kind):
        return [w for values in self.wrappers.values() for w in values]

    def remove_wrappers_with_key(self, kind, key):
        self.wrappers.pop(key, None)

    def clone(self):
        result = WindowModel(None)
        result.sampling = self.sampling
        result.wrappers = {k: list(v) for k, v in self.wrappers.items()}
        result.model_options = self.model_options.copy()
        return result


class TemporalTests(unittest.TestCase):
    def test_multiwindow_low_preserves_true_boundary_and_carries_clean_guide(self):
        wrapper = hw.WindowsWrapper(39, 22, True, 0, 0, 0, False)
        model = WindowModel(wrapper)
        t = hw.tokens_for_frames(73)
        audio_t = hw.audio_ticks(73)
        latent = {'samples': NestedTensor((torch.zeros(1, 24, t, 4, 6), torch.zeros(1, 32, 2, audio_t)))}
        sigmas = torch.tensor([1., .8, .6, .4, .2, 0.])
        calls = []

        def sample(m, noise, pos, neg, cfg, device, sampler, schedule, options, **kw):
            self.assertFalse(m.get_all_wrappers(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE))
            self.assertIsNotNone(kw['callback'])
            parts = list(kw['latent_image'].unbind())
            index = len(calls)
            if index:
                guide = pos[0][1]['minimax_keyframes'][-1]
                self.assertEqual(guide['resolved_frame_index'], 0)
                # x0 audio=8 in sampler space; clean guide audio must divide by audio_scale=4.
                torch.testing.assert_close(guide['audio_latent'], torch.full_like(guide['audio_latent'], 2.))
                torch.testing.assert_close(guide['latent'], torch.full_like(guide['latent'], .4))
            x = NestedTensor((torch.full_like(parts[0], .9), torch.full_like(parts[1], 9.)))
            x0 = NestedTensor((torch.full_like(parts[0], .4), torch.full_like(parts[1], 8.)))
            for i in range(3):
                kw['callback'](i, x0, x, 3)
            calls.append(parts[0].shape[2])
            return NestedTensor((torch.full_like(parts[0], .7), torch.full_like(parts[1], 3.)))

        cond = [[torch.zeros(1, 1, 8), {}]]
        with patch.object(nodes.comfy.samplers, 'sample', side_effect=sample):
            state, clean, _, _, width, height = nodes.SelfLiftSplitLowH3().sample(
                model, cond, cond, latent, nodes.comfy.samplers.ksampler('euler'), sigmas, 42, 1., 3, .5)
        expected = len(hw.plan_windows(t, 39, 22))
        self.assertEqual(len(calls), expected)
        self.assertEqual((width, height), (96, 64))
        torch.testing.assert_close(clean['samples'], torch.full((1, 24, t, 2, 4), .4))
        torch.testing.assert_close(state.low_x.unbind()[1], torch.full((1, 32, 2, audio_t), 9.))
        torch.testing.assert_close(state.low_x0.unbind()[1], torch.full((1, 32, 2, audio_t), 8.))
        torch.testing.assert_close(state.low_output.unbind()[1], torch.full((1, 32, 2, audio_t), 3.))
        self.assertEqual(len(model.get_all_wrappers('outer_sample')), 1)
        self.assertIsNotNone(state.temporal_config)


class ValidationTests(unittest.TestCase):
    def test_transition_rejects_missing_or_different_high_window(self):
        from selflift_split_test.temporal import validate_high_windows
        expected = (39, 22, True)
        for other in [None, hw.WindowsWrapper(56, 22, True, 0, 0, 0, False)]:
            with self.assertRaisesRegex(ValueError, 'settings must match'):
                validate_high_windows(WindowModel(other), expected)
        validate_high_windows(WindowModel(hw.WindowsWrapper(39, 22, True, 0, 0, 0, False)), expected)
        validate_high_windows(FakeModel(), None)

    def test_fallback_low_still_validates_high_window_settings(self):
        latent = {'samples': NestedTensor((torch.zeros(1, 24, 22, 4, 6), torch.zeros(1, 32, 2, 122)))}
        sigmas = torch.tensor([1., .8, .6, .4, .2, 0.])
        high = WindowModel(hw.WindowsWrapper(39, 22, True, 0, 0, 0, False))

        def sample(*args, **kwargs):
            value = kwargs['latent_image']
            for step in range(3):
                kwargs['callback'](step, value, value, 3)
            return value

        for wrapper in [None, hw.WindowsWrapper(90, 22, True, 0, 0, 0, False)]:
            low = WindowModel(wrapper)
            with self.subTest(low_window=None if wrapper is None else wrapper.window_frames):
                with patch.object(nodes.comfy.samplers, 'sample', side_effect=sample):
                    state, clean, *_ = nodes.SelfLiftSplitLowH3().sample(
                        low, [], [], latent, nodes.comfy.samplers.ksampler('euler'), sigmas, 42, 1., 3, .5)
                lifted = nodes.SelfLiftSplitResizeH3().resize(state, clean, 'nearest-exact')[0]
                with self.assertRaisesRegex(ValueError, 'settings must match'):
                    nodes.SelfLiftSplitTransitionH3().transition(state, lifted, high, 'upstream', 0, .5, 1)
                expected = None if wrapper is None else (90, 22, True)
                self.assertEqual(state.temporal_config, expected)
                # Unchanged fallback sampling remains usable with the same high configuration.
                nodes.SelfLiftSplitTransitionH3().transition(state, lifted, low, 'upstream', 0, .5, 1)

    def test_multiwindow_rejects_mask_or_batch_but_single_window_keeps_old_path(self):
        from selflift_split_test.temporal import sample_low_windows
        for batch, masked in [(1, True), (2, False)]:
            latent = NestedTensor((torch.zeros(batch, 24, 22, 2, 4), torch.zeros(batch, 32, 2, 122)))
            model = WindowModel(hw.WindowsWrapper(39, 22, True, 0, 0, 0, False))
            with self.assertRaisesRegex(ValueError, 'batch=1 and no noise_mask'):
                sample_low_windows(model, latent, [], [], latent, None, torch.tensor([1., .5]), 42, 1., 1, masked)
        latent = NestedTensor((torch.zeros(1, 24, 7, 2, 4), torch.zeros(1, 32, 2, 37)))
        self.assertIsNone(sample_low_windows(model, latent, [], [], latent, None, torch.tensor([1., .5]), 42, 1., 1, True))

    def test_merge_matches_upstream_with_pulled_back_last_window(self):
        from selflift_split_test.temporal import _merge
        from types import SimpleNamespace
        t = hw.tokens_for_frames(90)
        shapes = [torch.Size((1, 24, t, 2, 4)), torch.Size((1, 32, 2, hw.audio_ticks(90)))]
        window = hw.WindowsWrapper(39, 22, False, 0, 0, 0, False)
        plan = hw.plan_windows(t, 39, 22)
        zeros = [torch.zeros(s) for s in shapes]
        packed, _ = nodes.comfy.utils.pack_latents(zeros)
        pieces = []
        class Executor:
            class_obj = SimpleNamespace(conds={'positive': []}, model_patcher=FakeModel())
            def __call__(self, noise, latent, sampler, sigmas, mask, callback, disable, seed, latent_shapes):
                index = len(pieces) + 1
                parts = [torch.full(s, float(index)) for s in latent_shapes]
                pieces.append(parts)
                return nodes.comfy.utils.pack_latents(parts)[0]
        expected = window(Executor(), packed, packed, None, torch.tensor([1., .5]), seed=42, latent_shapes=shapes)
        actual = nodes.comfy.utils.pack_latents(_merge(pieces, plan, shapes, hw).unbind())[0]
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_missing_euler_callbacks_are_not_silently_accepted(self):
        latent = NestedTensor((torch.zeros(1, 24, 22, 4, 6), torch.zeros(1, 32, 2, 122)))
        model = WindowModel(hw.WindowsWrapper(39, 22, True, 0, 0, 0, False))
        def no_callbacks(*args, **kwargs):
            return kwargs['latent_image']
        with patch.object(nodes.comfy.samplers, 'sample', side_effect=no_callbacks):
            with self.assertRaisesRegex(RuntimeError, 'expected 3 Euler callbacks per window, received 0'):
                nodes.SelfLiftSplitLowH3().sample(model, [], [], {'samples': latent}, nodes.comfy.samplers.ksampler('euler'), torch.tensor([1., .8, .6, .4, .2, 0.]), 42, 1., 3, .5)

    def test_spatial_tiles_rejected_explicitly(self):
        from selflift_split_test.temporal import window_config
        model = WindowModel(hw.WindowsWrapper(39, 22, True, 16, 16, 4, False))
        with self.assertRaisesRegex(ValueError, 'spatial_tiles=false'):
            window_config(model)


if __name__ == '__main__':
    unittest.main()
