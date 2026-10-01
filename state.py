"""H3 AV state passed between the low sampler and boundary transition."""
from dataclasses import dataclass


@dataclass(frozen=True)
class SplitState:
    sigmas: object
    transition_step: int
    seed: int
    low_video: object
    low_x: object
    low_x0: object
    low_output: object
    nested: bool
    target_hw: tuple
    anchor: object
    mask: object
    batch_index: object
    model_sampling: object
    latent_format: object
