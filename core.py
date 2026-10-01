"""SelfLift split boundary math (based on upstream 835c3cf)."""
import torch


def low_size(height, width, scale):
    if not 0.25 <= scale <= 1:
        raise ValueError('lowres_scale must be between 0.25 and 1')
    return (max(2, round(height * scale / 2) * 2),
            max(2, round(width * scale / 2) * 2))


def validate_schedule(sigmas, step):
    if sigmas.ndim != 1 or not sigmas.is_floating_point() or sigmas.numel() < 3:
        raise ValueError('expected one-dimensional floating SIGMAS with at least two intervals')
    if not torch.isfinite(sigmas).all() or (sigmas < 0).any() or (sigmas[:-1] <= 0).any():
        raise ValueError('SIGMAS must be finite, nonnegative and positive except the last')
    if (sigmas[1:] > sigmas[:-1]).any() or sigmas[0] > 1:
        raise ValueError('SIGMAS must be non-increasing and begin at at most 1')
    if not isinstance(step, int) or not 1 <= step <= sigmas.numel() - 2:
        raise ValueError('transition_step is outside the sampling intervals')
    if sigmas[step] >= 1:
        raise ValueError('resume sigma must be below 1')
    return sigmas[:step + 1], sigmas[step:]


def validate_video(lifted, low_video, target_hw):
    if lifted.ndim != low_video.ndim or lifted.ndim not in (4, 5):
        raise ValueError('external video LATENT must match the original rank')
    if lifted.shape[:-2] != low_video.shape[:-2]:
        raise ValueError('external video LATENT batch/channel/temporal dimensions differ')
    if tuple(lifted.shape[-2:]) != tuple(target_hw):
        raise ValueError('external video LATENT does not match target spatial size')


def euler_step(state, denoised, sigma, next_sigma):
    step = ((next_sigma - sigma) / sigma).to(device=state.device, dtype=state.dtype)
    return state + (state - denoised.to(state)) * step


def video_endpoint(mode, seed):
    if mode == 'upstream':
        return ((seed + 1) % (1 << 64), 'sigma_k')
    if mode == 'legacy_seed':
        return (seed, 'sigma_next')
    if mode == 'simple_no_noise':
        return (None, 'sigma_next')
    if mode == 'simple_interpolation':
        return (seed, 'sigma_next')
    raise ValueError('unknown boundary mode')


def boundary_video(clean, noise, sigma_k, sigma_next, sampling, mode):
    if mode == 'upstream':
        if noise is None:
            raise ValueError('upstream mode requires boundary noise')
        state = sampling.noise_scaling(sigma_k, noise, clean)
        state = euler_step(state, clean, sigma_k, sigma_next)
    elif mode == 'legacy_seed':
        if noise is None:
            raise ValueError('legacy mode requires boundary noise')
        state = sampling.noise_scaling(sigma_next, noise, clean)
    elif mode == 'simple_no_noise':
        state = sampling.noise_scaling(sigma_next, torch.zeros_like(clean), clean)
    elif mode == 'simple_interpolation':
        if noise is None:
            raise ValueError('simple_interpolation mode requires boundary noise')
        # Match the reference graph: a zero-update sampler at sigma_next
        # re-noises the resized video once before the AV continuation.
        state = sampling.noise_scaling(sigma_next, noise, clean)
    else:
        raise ValueError('unknown boundary mode')
    return sampling.inverse_noise_scaling(sigma_next, state)
