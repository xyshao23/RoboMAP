from pathlib import Path
from types import SimpleNamespace

import yaml

from robomap.configuration import apply_robomap_model_config


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
