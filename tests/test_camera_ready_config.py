from pathlib import Path
from types import SimpleNamespace

import yaml

from robomap.configuration import apply_robomap_model_config
from robomap.label_utils import shape_aware_sigmas


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_camera_ready_config_uses_128_bottleneck():
    config_path = REPO_ROOT / "config" / "pretrain_robomap_config.yaml"
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    assert config["bottleneck_dim"] == 128


def test_model_config_receives_camera_ready_bottleneck():
    model_config = SimpleNamespace()

    apply_robomap_model_config(model_config, {"bottleneck_dim": 128})

    assert model_config.bottleneck_dim == 128


def test_shape_aware_bbox_sigmas_preserve_short_axis_sigma():
    assert shape_aware_sigmas(0.4, 0.2) == (4.0, 2.0)
    assert shape_aware_sigmas(0.2, 0.4) == (2.0, 4.0)
    assert shape_aware_sigmas(0.3, 0.3) == (2.0, 2.0)
