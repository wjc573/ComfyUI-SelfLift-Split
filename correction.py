"""SelfLift-zero correction, adapted from upstream commit 835c3cf."""
import torch
import torch.nn.functional as F


def resize_keyframes(cond, height, width):
    out = []
    for tensor, values in cond:
        if 'minimax_keyframes' not in values:
            out.append((tensor, values))
            continue
        values = values.copy()
        keyframes = []
        for source in values['minimax_keyframes']:
            kf = source.copy()
            lat = kf.get('latent')
            if lat is not None and tuple(lat.shape[-2:]) != (height, width):
                if lat.ndim == 5:
                    b, c, t = lat.shape[:3]
                    resized = F.interpolate(lat.float().permute(0, 2, 1, 3, 4).reshape(b*t, c, *lat.shape[-2:]),
                                            size=(height, width), mode='bilinear', align_corners=False)
                    resized = resized.reshape(b, t, c, height, width).permute(0, 2, 1, 3, 4)
                else:
                    resized = F.interpolate(lat.float(), size=(height, width), mode='bilinear', align_corners=False)
                resized += lat.float().mean(dim=(-2, -1), keepdim=True) - resized.mean(dim=(-2, -1), keepdim=True)
                kf['latent'] = resized.to(lat)
            keyframes.append(kf)
        values['minimax_keyframes'] = keyframes
        out.append((tensor, values))
    return out


def correct(direct, pixel, rho, w_min, w_max, mask=None):
    if rho <= 0 or w_max <= 0:
        return direct
    if pixel is None:
        raise ValueError('VAE pixel anchor required for correction')
    if mask is None and rho >= 1 and w_min >= 1 and w_max >= 1:
        return pixel
    delta = pixel - direct
    if mask is not None:
        scores = delta.abs().mean(dim=1) * mask.squeeze(1)
        region = (mask.squeeze(1) > 0).expand_as(scores)
        view = (-1,) + (1,) * (scores.ndim - 1)
        thresholds = torch.stack([
            torch.quantile(scores[b][region[b]], 1 - rho) if region[b].any()
            else torch.full((), float('inf'), device=scores.device, dtype=scores.dtype)
            for b in range(scores.shape[0])
        ]).view(view)
        selected = (scores >= thresholds) & region
    else:
        scores = delta.abs().mean(dim=1)
        view = (-1,) + (1,) * (scores.ndim - 1)
        thresholds = torch.quantile(scores.flatten(1), 1 - rho, dim=1).view(view)
        selected = scores >= thresholds
    if not selected.any():
        return direct
    smallest = scores.masked_fill(~selected, float('inf')).flatten(1).amin(dim=1).view(view)
    largest = scores.masked_fill(~selected, float('-inf')).flatten(1).amax(dim=1).view(view)
    weight = w_min + (w_max - w_min) * (scores - smallest) / (largest - smallest + 1e-8)
    weight = torch.where(selected, weight, torch.zeros_like(weight)).unsqueeze(1)
    return direct + weight * delta


def pixel_anchor(z0_low, vae, target_hw):
    """Optional decode-upscale-reencode anchor; matches the upstream video path."""
    result = []
    for single in z0_low.split(1):
        frames = vae.decode(single)
        if frames.ndim == 5:
            frames = frames.reshape(-1, *frames.shape[-3:])
        ratio = frames.shape[1] // single.shape[-2]
        if ratio <= 0:
            raise ValueError('VAE decoded image is smaller than its input latent grid')
        dtype = vae.vae_dtype if vae.vae_dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.float32
        if vae.device.type == 'cpu':
            dtype = torch.float32
        out = torch.empty((frames.shape[0], target_hw[0]*ratio, target_hw[1]*ratio, frames.shape[-1]), dtype=dtype)
        for start in range(0, frames.shape[0], 32):
            chunk = frames[start:start+32].movedim(-1, 1).to(device=vae.device, dtype=dtype)
            chunk = F.interpolate(chunk, size=out.shape[1:3], mode='bicubic', antialias=True)
            out[start:start+32] = chunk.movedim(1, -1).to(out.device)
        result.append(vae.encode(out).float())
    return torch.cat(result, dim=0)
