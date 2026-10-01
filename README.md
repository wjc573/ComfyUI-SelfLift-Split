# ComfyUI-SelfLift-Split

[中文说明](README_CN.md)

An independent MiniMax H3 audio-video implementation of the split form of SelfLift. The package runs the early Euler denoising steps on a reduced spatial latent, hands the denoised video boundary to an external latent upscaler or interpolation node, rebuilds the AV latent at the shared boundary sigma, and lets a native high-resolution sampler finish the schedule.

This repository is intentionally separate from the original `comfyui-SelfLift` project. It does not modify or replace the original package.

## Acknowledgements

This project is based on the resolution-transition idea and boundary-sampling logic of the original [comfyui-SelfLift](https://github.com/facok/comfyui-SelfLift) project by **facok**, and on the SelfLift paper:

- [SelfLift: Accelerating Few-Step Diffusion via Self-Recovering Resolution Transition](https://arxiv.org/abs/2609.02036)

The original project is the reference for the SelfLift implementation and the source of the initial H3 adaptation direction. This repository focuses on making the H3 AV path explicit and modular so that the low-resolution sampler, latent lift, AV boundary handoff, and native second sampler can be inspected and adjusted independently.

## What This Project Adds

Compared with using the original SelfLift implementation directly, this package adds an explicit modular H3 AV split path:

- `SelfLiftSplitLowH3` runs the first part of the schedule with standard Euler and captures the intermediate sampler state and denoised video boundary.
- `SelfLiftSplitTransitionH3` validates the high-resolution model and latent contract, rebuilds the video boundary at the shared sigma, preserves the audio stream, and returns a ready-to-sample AV latent plus the remaining sigmas.
- `SelfLiftSplitResizeH3` provides a simple video-only latent interpolation route for H3.
- External learned H3 latent upscalers can be connected between the low and transition nodes.
- The low node outputs the target pixel width and height, so a target-dimensions upscaler does not require users to calculate them manually.
- H3 nested video/audio latent shapes, spatial sizes, sampler type, sigma schedule, model sampling type, latent format, and mask shapes are checked before the second pass.
- Audio is kept as a separate stream during video lifting and is repacked only after the video boundary has been prepared.
- The boundary modes are exposed as a node option instead of being hidden inside one composite sampler.

This package is an H3 AV split implementation. It is not a complete fork of the original image SelfLift node set, and it does not include the original project's separate TST implementation.

## Nodes

### `SelfLiftSplitLowH3`

Runs the low-resolution prefix and captures the boundary needed by the transition node.

- Input: nested H3 AV latent with video `[B, 24, T, H, W]` and audio `[B, 32, 2, T]`.
- Sampler: standard Euler only, with `s_churn=0`.
- `transition_step`: number of low-resolution Euler updates before the lift.
- `lowres_scale`: spatial scale used for the low-resolution video prefix.
- Outputs the captured state, clean video-only latent, resized conditioning, and target pixel dimensions.

The `clean_video` output is the denoised video boundary (`x0` / `denoised_output`), not the noisy sampler output. It is the intended input for a video-only latent upscaler.

### `SelfLiftSplitTransitionH3`

Prepares the high-resolution AV latent and the remaining sigma tail.

- Accepts the upscaled video-only latent.
- Preserves the audio stream from the low sampler.
- Returns `resume_latent`, `tail_sigmas`, and the validated high-resolution model.
- Optional pixel-anchor correction is available through `rho`, `w_min`, `w_max`, and the model VAE.

Boundary modes:

| Mode | Use case | Video boundary behavior |
| --- | --- | --- |
| `legacy_seed` | Learned latent upscaler route and compatibility with the reference workflow style | Re-noise the lifted video at `sigma_next` with the original seed |
| `upstream` | Comparing against the original SelfLift-style boundary logic | Use `seed + 1` at `sigma_k`, then reconstruct the Euler interval to `sigma_next` |
| `simple_interpolation` | Simple interpolation route matching the supplied reference workflow | Perform the one-boundary-sigma re-noise adapter after interpolation |
| `simple_no_noise` | No-random-video-noise diagnostic/control route | Do not add video random noise at the boundary |

These modes are not mathematically equivalent. They are boundary construction choices for different routes and comparisons.

### `SelfLiftSplitResizeH3`

Resizes the clean video-only latent to the target H3 latent grid. It does not process audio and does not perform denoising. The reference simple-interpolation workflow uses `nearest-exact`; this node is a direct interpolation helper, not a learned upscaler.

## Second Sampler Wiring

The transition node already prepares the boundary latent and handles any route-specific video boundary noise. The native second sampler must therefore use `DisableNoise`:

```text
SelfLiftSplitTransitionH3.resume_latent -> SamplerCustomAdvanced.latent_image
SelfLiftSplitTransitionH3.tail_sigmas   -> SamplerCustomAdvanced.sigmas
DisableNoise                            -> SamplerCustomAdvanced.noise
```

Do not connect `RandomNoise` to the second sampler after the transition node. That would inject random noise a second time and can corrupt the video/audio continuation.

## Included Workflows

The `workflows/` directory contains the two tested examples used for the final H3 comparison:

1. **Learned latent upscaler**
   - Uses `MinimaxH3LatentUpscaler3D`.
   - Uses `legacy_seed` in the transition node.
   - Intended for the learned H3 latent lifting route.

2. **Simple interpolation**
   - Uses `SelfLiftSplitResizeH3` with `nearest-exact`.
   - Uses `simple_interpolation` in the transition node.
   - Follows the reference workflow's one-boundary-sigma video adaptation before the second sampler.
   - Simple interpolation has a lower quality ceiling than learned lifting and generally needs enough high-resolution second-pass steps to repair interpolation artifacts.

The workflow JSON files depend on the corresponding ComfyUI core and H3 custom nodes already installed in the user's environment. The learned example additionally requires the installed `MinimaxH3LatentUpscaler3D` node and its model checkpoint.

## Installation

Copy or clone this repository into ComfyUI's `custom_nodes` directory:

```text
ComfyUI/custom_nodes/ComfyUI-SelfLift-Split
```

Restart ComfyUI. The nodes are registered in the `selflift/split` category.

This package imports ComfyUI's native Python modules and is intended to run inside a ComfyUI installation. It is not a standalone Python package.

## Validation

The repository includes unit tests for the mathematical boundary helpers, correction helpers, node contracts, nested AV packing, audio handoff, masks, and all boundary modes.

Run them with the Python environment used by ComfyUI:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

The final local verification covered 16 tests.

## Limitations

- Only standard Euler with `s_churn=0` is accepted by the low node.
- The low and high models must share the same rectified-flow sampling type, latent format, and noise scale.
- The external upscaler must return a video-only latent with the target spatial shape; it must not return a nested AV latent.
- The package does not claim that simple interpolation can reconstruct details lost during spatial lifting. The high-resolution second pass is responsible for repairing interpolation artifacts.
- `simple_no_noise` is useful as a diagnostic mode, but it is not the same as the reference simple-interpolation workflow.
- A workflow that uses `RandomNoise` on the second sampler after Transition is invalid for these nodes.
- The original SelfLift image path and the original project's separate TST functionality are outside this repository.

## License and Upstream Notice

This repository is an independent adaptation for MiniMax H3 AV workflows. Refer to the original [comfyui-SelfLift](https://github.com/facok/comfyui-SelfLift) repository and the SelfLift paper for the upstream implementation and research context. Check the upstream project's license and notices when redistributing derived work.
