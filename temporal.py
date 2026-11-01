"""Private stage-wise TemporalWindow compatibility; no additional public nodes."""
import copy
import logging
import sys
from types import SimpleNamespace

import torch

import comfy.model_management
import comfy.nested_tensor
import comfy.patcher_extension
import comfy.sampler_helpers
import comfy.samplers
import comfy.utils

KEY = 'minimax_h3_windows'


def window_config(model):
    """Read only wrappers attached to this model; dependency is optional."""
    getter = getattr(model, 'get_all_wrappers', None)
    if getter is None:
        return None, None
    wrappers = [w for w in getter(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE)
                if w.__class__.__name__ == 'WindowsWrapper' and hasattr(w, 'window_frames')]
    if not wrappers:
        return None, None
    if len(wrappers) != 1:
        raise ValueError('SelfLift Split expects exactly one TemporalWindow wrapper')
    w = wrappers[0]
    if w.tile_h or w.tile_w:
        raise ValueError('SelfLift Split temporal mode requires spatial_tiles=false')
    config = (w.window_frames, w.overlap_frames, w.carry_overlap)
    return w, config


def _raw_cond(conds):
    result = []
    for cond in conds:
        meta = cond.copy()
        tensor = meta.pop('cross_attn', None)
        meta.pop('uuid', None)
        meta.pop('model_conds', None)
        result.append([tensor, meta])
    return result


def _parts(samples):
    return list(samples.unbind())


def _nested(parts):
    return comfy.nested_tensor.NestedTensor(parts)


def _merge(pieces, windows, shapes, hw):
    """Use the upstream window placement and ease-in-out blend for real boundaries."""
    video = torch.zeros(shapes[0], dtype=torch.float32)
    audio = torch.zeros(shapes[1], dtype=torch.float32)
    done_t = done_a = 0
    for wi, ((t0, tn), (v, a)) in enumerate(zip(windows, pieces)):
        a0 = 0 if t0 == 0 else min(hw.audio_ticks(hw.frames_for_tokens(t0)), shapes[1][-1])
        a1 = shapes[1][-1] if wi == len(windows)-1 else min(shapes[1][-1], a0+hw.audio_ticks(hw.frames_for_tokens(tn)))
        v, a = v.float().cpu(), a.float().cpu()
        if tuple(v.shape) != tuple(shapes[0][:2])+(tn,)+tuple(shapes[0][3:]) or a.shape[-1] != a1-a0:
            raise RuntimeError('TemporalWindow executor layout changed; update SelfLift Split compatibility')
        ov_t = max(0, done_t-t0)
        if ov_t:
            fade = hw.ease_in_out(ov_t).view(1, 1, -1, 1, 1)
            video[:, :, t0:done_t] = video[:, :, t0:done_t]*(1-fade)+v[:, :, :ov_t]*fade
        video[:, :, t0+ov_t:t0+tn] = v[:, :, ov_t:]
        ov_a = max(0, min(done_a, a1)-a0)
        if ov_a:
            fade = hw.ease_in_out(ov_a)
            audio[..., a0:a0+ov_a] = audio[..., a0:a0+ov_a]*(1-fade)+a[..., :ov_a]*fade
        audio[..., a0+ov_a:a1] = a[..., ov_a:]
        done_t, done_a = max(done_t, t0+tn), max(done_a, a1)
    return _nested([video, audio])


def sample_low_windows(model, noise, positive, negative, latent, sampler, sigmas, seed, cfg, transition_step, masked=False):
    """Return real merged x/x0/output, or None to keep the original single-pass path."""
    wrapper, config = window_config(model)
    if wrapper is None:
        return None
    hw = sys.modules.get(wrapper.__class__.__module__)
    required = ('plan_windows', 'frames_for_tokens', 'audio_ticks', 'ease_in_out')
    if hw is None or not all(hasattr(hw, name) for name in required):
        raise RuntimeError('Unsupported TemporalWindow implementation; required window helpers missing')
    parts = _parts(latent)
    windows = hw.plan_windows(parts[0].shape[2], wrapper.window_frames, wrapper.overlap_frames)
    if len(windows) == 1:
        return None
    if masked or parts[0].shape[0] != 1:
        raise ValueError('SelfLift Split multiwindow currently requires batch=1 and no noise_mask')
    inner = model.clone()
    inner.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, KEY)
    if window_config(inner)[0] is not None:
        raise ValueError('Unable to remove inner TemporalWindow wrapper; refusing nested window sampling')
    holder = SimpleNamespace(conds={'positive': comfy.sampler_helpers.convert_cond(positive),
                                    'negative': comfy.sampler_helpers.convert_cond(negative)}, model_patcher=inner)
    boundaries = {'x': [], 'x0': [], 'output': []}
    fmt = model.get_model_object('latent_format')
    audio_scale = float(getattr(model.get_model_object('model_sampling'), 'audio_scale', 1.0))
    packed, shapes = comfy.utils.pack_latents(parts)
    packed_noise, _ = comfy.utils.pack_latents(_parts(noise))

    class PerWindow:
        class_obj = holder

        def __call__(self, piece_noise, piece_latent, supplied_sampler, supplied_sigmas,
                     denoise_mask=None, callback=None, disable_pbar=False, supplied_seed=None, latent_shapes=None):
            count = 0
            captured = {}

            def capture(step, x0, x, total):
                nonlocal count
                count += 1
                if count > transition_step:
                    raise RuntimeError('SelfLift Split: too many low-resolution Euler callbacks')
                if count == transition_step:
                    for name, value in [('x', x), ('x0', x0)]:
                        captured[name] = [p.detach().float().cpu().clone() for p in _parts(value)]

            output = comfy.samplers.sample(inner,
                _nested(comfy.utils.unpack_latents(piece_noise, latent_shapes)),
                _raw_cond(holder.conds['positive']), _raw_cond(holder.conds['negative']),
                cfg, inner.load_device, supplied_sampler, supplied_sigmas, inner.model_options,
                latent_image=_nested(comfy.utils.unpack_latents(piece_latent, latent_shapes)),
                callback=capture, disable_pbar=disable_pbar, seed=seed)
            if count != transition_step or 'x0' not in captured:
                raise RuntimeError(f'SelfLift Split: expected {transition_step} Euler callbacks per window, received {count}')
            boundaries['x'].append(captured['x'])
            boundaries['x0'].append(captured['x0'])
            boundaries['output'].append([p.detach().float().cpu().clone() for p in _parts(output)])
            logging.info('[SelfLift Split] low window %d/%d: %d real Euler callbacks',
                         len(boundaries['x']), len(windows), count)
            # Upstream TW carry expects clean VAE-space predictions, NOT the noisy prefix endpoint.
            clean = [fmt.process_out(captured['x0'][0]), captured['x0'][1]/audio_scale]
            return comfy.utils.pack_latents(clean)[0]

    outer = copy.copy(wrapper)
    outer.warned = set()
    outer(PerWindow(), packed_noise, packed, sampler, sigmas, None, None,
          not comfy.utils.PROGRESS_BAR_ENABLED, seed, latent_shapes=shapes)
    if len(boundaries['x']) != len(windows):
        raise RuntimeError('TemporalWindow executor call count differs from window plan')
    captured = {name: _merge(boundaries[name], windows, shapes, hw) for name in ['x', 'x0']}
    output = _merge(boundaries['output'], windows, shapes, hw)
    return captured, output, config


def validate_high_windows(model, expected):
    if window_config(model)[1] != expected:
        raise ValueError('SelfLift Split low/high TemporalWindow settings must match (window, overlap, carry)')
