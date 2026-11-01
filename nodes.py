"""Independent H3 SelfLift split nodes. Original project remains unmodified."""
import torch
import torch.nn.functional as F

import comfy.k_diffusion.sampling
import comfy.model_management
import comfy.model_patcher
import comfy.model_sampling
import comfy.nested_tensor
import comfy.sample
import comfy.samplers
import comfy.utils

from .core import boundary_video, euler_step, low_size, validate_schedule, validate_video, video_endpoint
from .correction import correct, pixel_anchor, resize_keyframes
from .state import SplitState
from .temporal import sample_low_windows, validate_high_windows, window_config


def streams(samples):
    return list(samples.unbind()) if samples.is_nested else [samples]


def pack(parts, nested):
    return comfy.nested_tensor.NestedTensor(parts) if nested else parts[0]


def validate_mask(latent, video):
    mask = latent.get('noise_mask')
    if mask is None:
        return None
    b, _, t, h, w = video.shape
    if mask.ndim == 3:
        mask = mask[:, None]
    if mask.ndim == 4:
        mask = mask[:, :, None]
    if mask.ndim != 5 or mask.shape[0] != b or mask.shape[1] != 1 or mask.shape[2] not in (1, t):
        raise ValueError('H3 mask must be [B,H,W], [B,1,H,W] or [B,1,1|T,H,W]')
    if tuple(mask.shape[-2:]) != (h, w):
        mask = F.interpolate(mask.reshape(-1, 1, *mask.shape[-2:]).float(), size=(h, w), mode='bilinear', align_corners=False).reshape(b, 1, mask.shape[2], h, w)
    if not torch.isfinite(mask).all():
        raise ValueError('mask contains nonfinite values')
    return mask.float().clamp(0, 1).expand(b, 1, t, h, w)


def masked_model(base, anchor, mask, auxiliary):
    if mask is None:
        return base
    b, channels = anchor.shape[:2]
    flat_mask = mask.repeat(1, channels, 1, 1, 1).reshape(b, 1, -1)
    flat_anchor = anchor.reshape(b, 1, -1)
    for audio in auxiliary:
        flat_mask = torch.cat([flat_mask, torch.ones_like(audio.reshape(b, 1, -1))], dim=-1)
        flat_anchor = torch.cat([flat_anchor, audio.reshape(b, 1, -1)], dim=-1)
    def blend(args):
        denoised = args['denoised']
        m = flat_mask.to(denoised)
        a = flat_anchor.to(denoised)
        return denoised * m + a * (1 - m)
    patched = base.clone()
    patched.model_options = comfy.model_patcher.set_model_options_post_cfg_function(patched.model_options, blend)
    return patched


class SelfLiftSplitLowH3:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'model': ('MODEL',), 'positive': ('CONDITIONING',), 'negative': ('CONDITIONING',),
            'latent_image': ('LATENT',), 'sampler': ('SAMPLER',), 'sigmas': ('SIGMAS',),
            'seed': ('INT', {'default': 0, 'min': 0, 'max': 0xffffffffffffffff, 'control_after_generate': True}),
            'cfg': ('FLOAT', {'default': 5., 'min': 0., 'max': 100., 'step': 0.1}),
            'transition_step': ('INT', {'default': 3, 'min': 1, 'max': 10000}),
            'lowres_scale': ('FLOAT', {'default': 0.5, 'min': 0.25, 'max': 1., 'step': 0.05}),
        }}

    RETURN_TYPES = ('SELF_LIFT_SPLIT_STATE', 'LATENT', 'CONDITIONING', 'CONDITIONING', 'INT', 'INT')
    RETURN_NAMES = ('state', 'clean_video', 'positive_low', 'negative_low', 'target_width_pixels', 'target_height_pixels')
    FUNCTION = 'sample'
    CATEGORY = 'selflift/split'

    def sample(self, model, positive, negative, latent_image, sampler, sigmas, seed, cfg, transition_step, lowres_scale):
        low_sigmas, _ = validate_schedule(sigmas, transition_step)
        sampling = model.get_model_object('model_sampling')
        if not isinstance(sampling, comfy.model_sampling.CONST):
            raise ValueError('SelfLift Split requires a rectified-flow model')
        if not isinstance(sampler, comfy.samplers.KSAMPLER) or sampler.sampler_function is not comfy.k_diffusion.sampling.sample_euler or sampler.extra_options.get('s_churn', 0) != 0:
            raise ValueError('SelfLift Split requires standard Euler with s_churn=0')
        initial = comfy.sample.fix_empty_latent_channels(model, latent_image['samples'], latent_image.get('downscale_ratio_spacial'), latent_image.get('downscale_ratio_temporal'))
        parts = streams(initial)
        if not initial.is_nested or len(parts) != 2 or parts[0].ndim != 5 or parts[1].ndim != 4 or parts[0].shape[1] != 24 or parts[1].shape[1] != 32 or parts[0].shape[0] != parts[1].shape[0]:
            raise ValueError('SelfLift Split H3 requires nested video [B,24,T,H,W] and audio [B,32,2,T]')
        video, audio = parts
        b, c, t, height, width = video.shape
        h, w = low_size(height, width, lowres_scale)
        latent_format = model.get_model_object('latent_format')
        downsample = getattr(latent_format, 'spacial_downscale_ratio', 16)
        target_pixels = (height * downsample, width * downsample)
        mask = validate_mask(latent_image, video)
        low_video = F.interpolate(video.float(), size=(t, h, w), mode='trilinear', align_corners=False).to(video)
        low_mask = F.interpolate(mask.reshape(b*t, 1, height, width), size=(h, w), mode='bilinear', align_corners=False).reshape(b, 1, t, h, w).clamp(0, 1) if mask is not None else None
        low_model = masked_model(model, low_video, low_mask, [audio])
        low_latent = pack([low_video, audio], True)
        noise = comfy.sample.prepare_noise(low_latent, seed, latent_image.get('batch_index'))
        pos_low, neg_low = resize_keyframes(positive, h, w), resize_keyframes(negative, h, w)
        captured = {}
        count = 0
        def callback(step, x0, x, total):
            nonlocal count
            count += 1
            if count == transition_step:
                captured['x'], captured['x0'] = x, x0
        # Keep the attached config even when sampling falls back to a single window.
        _, temporal_config = window_config(low_model)
        windowed = sample_low_windows(low_model, noise, pos_low, neg_low, low_latent,
                                      sampler, low_sigmas, seed, cfg, transition_step, masked=mask is not None)
        if windowed is None:
            low_output = comfy.samplers.sample(low_model, noise, pos_low, neg_low, cfg, low_model.load_device,
                                  sampler, low_sigmas, low_model.model_options, latent_image=low_latent,
                                  callback=callback, disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=seed)
            if count != transition_step or 'x0' not in captured:
                raise RuntimeError(f'expected {transition_step} Euler callbacks, received {count}')
        else:
            captured, low_output, temporal_config = windowed
        x0_video = streams(captured['x0'])[0]
        clean = latent_format.process_out(x0_video.float())
        endpoint = latent_image.copy()
        endpoint.pop('noise_mask', None)
        endpoint.pop('downscale_ratio_spacial', None)
        endpoint.pop('downscale_ratio_temporal', None)
        endpoint['samples'] = clean.to(comfy.model_management.intermediate_device())
        state = SplitState(sigmas, transition_step, seed, x0_video, captured['x'], captured['x0'],
                           low_output, True, (height, width), video, mask, latent_image.get('batch_index'),
                           sampling, latent_format, temporal_config)
        return (state, endpoint, pos_low, neg_low, int(target_pixels[1]), int(target_pixels[0]))


class SelfLiftSplitTransitionH3:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'state': ('SELF_LIFT_SPLIT_STATE',), 'upscaled_video': ('LATENT',), 'model_high': ('MODEL',),
            'mode': (['upstream', 'legacy_seed', 'simple_interpolation', 'simple_no_noise'], {'default': 'upstream'}),
            'rho': ('FLOAT', {'default': 0., 'min': 0., 'max': 1., 'step': 0.05}),
            'w_min': ('FLOAT', {'default': 0.5, 'min': 0., 'max': 1., 'step': 0.05}),
            'w_max': ('FLOAT', {'default': 1., 'min': 0., 'max': 1., 'step': 0.05}),
        }, 'optional': {'vae': ('VAE',)}}

    RETURN_TYPES = ('LATENT', 'SIGMAS', 'MODEL')
    RETURN_NAMES = ('resume_latent', 'tail_sigmas', 'model_high')
    FUNCTION = 'transition'
    CATEGORY = 'selflift/split'

    def transition(self, state, upscaled_video, model_high, mode, rho, w_min, w_max, vae=None):
        if not isinstance(state, SplitState):
            raise ValueError('state must originate from SelfLift Split Low H3')
        validate_high_windows(model_high, state.temporal_config)
        _, tail = validate_schedule(state.sigmas, state.transition_step)
        if not 0 <= rho <= 1 or not 0 <= w_min <= w_max <= 1:
            raise ValueError('correction requires 0<=rho<=1 and 0<=w_min<=w_max<=1')
        sampling_high = model_high.get_model_object('model_sampling')
        latent_format_high = model_high.get_model_object('latent_format')
        if not isinstance(sampling_high, comfy.model_sampling.CONST) or type(sampling_high) is not type(state.model_sampling) or type(latent_format_high) is not type(state.latent_format) or getattr(sampling_high, 'noise_scale', 1.) != getattr(state.model_sampling, 'noise_scale', 1.):
            raise ValueError('high model sampling or latent format differs from low model')
        lifted = upscaled_video['samples']
        validate_video(lifted, state.low_video, state.target_hw)
        if lifted.is_nested:
            raise ValueError('upscaler must output video-only LATENT')
        sigma_k, sigma_next = state.sigmas[state.transition_step-1:state.transition_step+1]
        device = comfy.model_management.intermediate_device()
        anchor = state.anchor.to(device)
        mask = state.mask
        corrected = state.latent_format.process_in(lifted.float().to(device))
        if rho > 0 and w_max > 0:
            if vae is None:
                raise ValueError('VAE is required for pixel-anchor correction')
            source = state.latent_format.process_out(state.low_video.float()).to(device)
            pixel = pixel_anchor(source, vae, state.target_hw)
            pixel = state.latent_format.process_in(pixel).to(corrected)
            corrected = correct(corrected, pixel, rho, w_min, w_max, mask)
        if mask is not None:
            m = mask.to(corrected)
            corrected = corrected * m + anchor.to(corrected) * (1-m)
        noise_seed, _ = video_endpoint(mode, state.seed)
        video_noise = comfy.sample.prepare_noise(corrected, noise_seed, state.batch_index).to(corrected) if noise_seed is not None else None
        video = boundary_video(corrected, video_noise, sigma_k, sigma_next, state.model_sampling, mode)
        # H3 carries audio in sampler space multiplied by audio_scale (shift/audio_shift).
        # Convert explicitly instead of relying on model.latent_shapes left over from a previous run.
        audio_scale = float(getattr(state.model_sampling, 'audio_scale', 1.0))
        if abs(float(getattr(sampling_high, 'audio_scale', 1.0)) - audio_scale) > 1e-6:
            raise ValueError('high model audio_scale differs from low model; use the same sigma shift for both passes')
        low_states, low_predictions = streams(state.low_x), streams(state.low_x0)
        if mode == 'upstream':
            # callback x/x0 are sampler-space: rebuild the Euler interval, then leave sampler space
            aux = [state.model_sampling.inverse_noise_scaling(sigma_next, euler_step(x.to(device), x0.to(device), sigma_k, sigma_next)) / audio_scale
                   for x, x0 in zip(low_states[1:], low_predictions[1:])]
        else:
            # low_output already went through inverse_noise_scaling and process_latent_out
            aux = [a.to(device) for a in streams(state.low_output)[1:]]
        resume = pack([state.latent_format.process_out(video)] + aux, state.nested)
        result = upscaled_video.copy()
        result.pop('noise_mask', None)
        result.pop('downscale_ratio_spacial', None)
        result.pop('downscale_ratio_temporal', None)
        result['samples'] = resume
        high = masked_model(model_high, anchor, mask, [state.anchor.new_zeros(a.shape) for a in aux])
        return (result, tail, high)


class SelfLiftSplitResizeH3:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'state': ('SELF_LIFT_SPLIT_STATE',), 'video': ('LATENT',),
                             'method': (['nearest', 'nearest-exact', 'bilinear', 'bicubic', 'bislerp'], {'default': 'nearest'})}}
    RETURN_TYPES = ('LATENT',)
    FUNCTION = 'resize'
    CATEGORY = 'selflift/split'

    def resize(self, state, video, method):
        samples = video['samples']
        if samples.is_nested or samples.ndim != 5 or samples.shape[:-2] != state.low_video.shape[:-2]:
            raise ValueError('resize expects clean video-only LATENT from the low node')
        height, width = state.target_hw
        result = video.copy()
        result['samples'] = comfy.utils.common_upscale(samples, width, height, method, 'disabled')
        return (result,)


NODE_CLASS_MAPPINGS = {
    'SelfLiftSplitLowH3': SelfLiftSplitLowH3,
    'SelfLiftSplitTransitionH3': SelfLiftSplitTransitionH3,
    'SelfLiftSplitResizeH3': SelfLiftSplitResizeH3,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    'SelfLiftSplitLowH3': 'SelfLift Split Low H3 (Euler)',
    'SelfLiftSplitTransitionH3': 'SelfLift Split Transition H3',
    'SelfLiftSplitResizeH3': 'SelfLift Split Resize H3 Video',
}
