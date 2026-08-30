"""Behaviour of the configuration loader and its validation rules."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from smartcity_vision.exceptions import ConfigError
from smartcity_vision.utils.config import AppConfig, load_config


def write_config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_shipped_default_config_is_valid() -> None:
    config = load_config(Path("config/default.yaml"))

    assert config.model.image_size % 32 == 0
    assert "car" in config.model.target_classes
    assert config.output.directory == Path("data/output")
    assert config.analytics.lines
    assert config.analytics.zones
    assert all(len(zone.polygon) >= 3 for zone in config.analytics.zones)


def test_a_zone_with_fewer_than_three_vertices_is_rejected(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        "analytics:\n  zones:\n    - name: bad\n      polygon: [[0, 0], [1, 1]]\n",
    )

    with pytest.raises(ConfigError, match="at least three vertices"):
        load_config(path)


def test_yaml_overrides_only_the_keys_it_sets(tmp_path: Path) -> None:
    path = write_config(tmp_path, "model:\n  device: cpu\n  confidence_threshold: 0.6\n")

    config = load_config(path)

    assert config.model.device == "cpu"
    assert config.model.confidence_threshold == pytest.approx(0.6)
    # Untouched keys keep their defaults rather than becoming None.
    assert config.model.iou_threshold == pytest.approx(AppConfig().model.iou_threshold)
    assert config.model.target_classes == AppConfig().model.target_classes


def test_dotted_overrides_beat_the_file_and_are_type_parsed(tmp_path: Path) -> None:
    path = write_config(tmp_path, "video:\n  frame_skip: 1\n  loop: true\n")

    config = load_config(path, overrides=["video.frame_skip=4", "video.loop=false"])

    assert config.video.frame_skip == 4
    assert config.video.loop is False


def test_updates_take_precedence_over_dotted_overrides(tmp_path: Path) -> None:
    path = write_config(tmp_path, "model:\n  device: cuda\n")

    config = load_config(
        path,
        overrides=["model.device=mps"],
        updates={"model": {"device": "cpu"}},
    )

    assert config.model.device == "cpu"


def test_target_classes_are_normalised_and_deduplicated(tmp_path: Path) -> None:
    path = write_config(tmp_path, "model:\n  target_classes: [Car, ' CAR ', truck]\n")

    config = load_config(path)

    assert config.model.target_classes == ("car", "truck")


def test_out_of_range_threshold_is_rejected(tmp_path: Path) -> None:
    path = write_config(tmp_path, "model:\n  confidence_threshold: 1.4\n")

    with pytest.raises(ConfigError, match="confidence_threshold"):
        load_config(path)


def test_image_size_must_be_stride_aligned(tmp_path: Path) -> None:
    path = write_config(tmp_path, "model:\n  image_size: 100\n")

    with pytest.raises(ConfigError, match="multiple of 32"):
        load_config(path)


def test_empty_target_classes_is_rejected(tmp_path: Path) -> None:
    path = write_config(tmp_path, "model:\n  target_classes: []\n")

    with pytest.raises(ConfigError, match="at least one class"):
        load_config(path)


def test_unknown_key_is_rejected_rather_than_ignored(tmp_path: Path) -> None:
    path = write_config(tmp_path, "model:\n  confidenc_threshold: 0.3\n")

    with pytest.raises(ConfigError, match="confidenc_threshold"):
        load_config(path)


def test_malformed_override_is_rejected() -> None:
    with pytest.raises(ConfigError, match="section.key=value"):
        load_config(None, overrides=["model.device"])


def test_missing_config_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "absent.yaml")


def test_snapshot_is_json_serialisable_and_round_trips() -> None:
    config = AppConfig()
    snapshot = config.snapshot()

    assert json.loads(json.dumps(snapshot)) == snapshot
    assert AppConfig.model_validate(snapshot) == config


def test_config_is_immutable() -> None:
    config = AppConfig()

    with pytest.raises(ValidationError):
        config.model.confidence_threshold = 0.9  # type: ignore[misc]
