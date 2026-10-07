"""Configuration helpers for the RoboMAP model."""

from typing import Any, Mapping


CAMERA_READY_BOTTLENECK_DIM = 128


def apply_robomap_model_config(
    model_config: Any, experiment_config: Mapping[str, Any]
) -> Any:
    """Apply experiment-level decoder settings to a Transformers config.

    The custom decoder is created while ``RoboMAP_Paligemma`` is initialized,
    so these values must be attached to the base model config before calling
    ``from_pretrained``.
    """
    bottleneck_dim = int(
        experiment_config.get("bottleneck_dim", CAMERA_READY_BOTTLENECK_DIM)
    )
    if bottleneck_dim <= 0:
        raise ValueError("bottleneck_dim must be a positive integer")

    upsample_method = experiment_config.get("upsample_method", "convex")
    if upsample_method == "ahd":
        # The paper-facing name is AHD; the released implementation is the
        # convex-upsampling decoder described by AdaptiveHeatmapDecoder.
        upsample_method = "convex"

    model_config.bottleneck_dim = bottleneck_dim
    model_config.upsample_method = upsample_method
    return model_config
