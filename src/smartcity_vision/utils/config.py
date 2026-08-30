"""Typed, validated configuration.

Every tunable value in SmartCity Vision lives in a YAML file and is validated
into a frozen Pydantic model here. Modules receive the config section they need
rather than reading globals, which keeps constants out of the code and makes a
run reproducible from its config snapshot alone.

Sections are added as the phases that consume them land, so the schema never
contains options that nothing reads.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from smartcity_vision.exceptions import ConfigError

DEFAULT_CONFIG_PATH = Path("config/default.yaml")

DEFAULT_TARGET_CLASSES: tuple[str, ...] = (
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "person",
)

DeviceName = Literal["auto", "cpu", "cuda", "mps"]
LogLevelName = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class _Section(BaseModel):
    """Base for config sections: immutable and strict about unknown keys."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelConfig(_Section):
    """YOLOv8 weights and inference parameters."""

    weights: Path = Path("models/yolov8n.pt")
    device: DeviceName = "auto"
    confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    image_size: int = Field(default=640, ge=32, le=4096)
    max_detections: int = Field(default=300, ge=1)
    target_classes: tuple[str, ...] = DEFAULT_TARGET_CLASSES
    # One throwaway inference before the run so lazy GPU/MPS initialisation is
    # not charged to the first real frame's measured latency.
    warmup: bool = True

    @field_validator("image_size")
    @classmethod
    def _stride_aligned(cls, value: int) -> int:
        """Reject sizes YOLOv8 would silently round to the model stride."""
        if value % 32 != 0:
            raise ValueError("image_size must be a multiple of 32 (YOLOv8 stride)")
        return value

    @field_validator("target_classes")
    @classmethod
    def _normalised_classes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Lower-case, de-duplicate, and require at least one class."""
        normalised = tuple(dict.fromkeys(name.strip().lower() for name in value if name.strip()))
        if not normalised:
            raise ValueError("target_classes must list at least one class name")
        return normalised


class TrackingConfig(_Section):
    """Multi-object tracking settings."""

    enabled: bool = True
    # Ultralytics ships both configs; ByteTrack is faster, BoT-SORT is more
    # robust through occlusion because it adds appearance features.
    tracker: Literal["bytetrack.yaml", "botsort.yaml"] = "bytetrack.yaml"


class CountingConfig(_Section):
    """Unique-object counting settings."""

    # A track must be seen this many times before it counts, which suppresses
    # one-frame false positives from inflating the totals.
    min_track_frames: int = Field(default=3, ge=1)
    # Per-track bookkeeping is dropped this many frames after a track was last
    # seen, so memory stays bounded on long or live streams.
    forget_track_after_frames: int = Field(default=90, ge=1)


class AnalyticsConfig(_Section):
    """Traffic-analytics settings. Extended by later phases."""

    counting: CountingConfig = Field(default_factory=CountingConfig)


class VideoConfig(_Section):
    """Where frames come from and how many of them to process."""

    source: str = "data/input/traffic.mp4"
    loop: bool = False
    frame_skip: int = Field(default=0, ge=0)
    max_frames: int | None = Field(default=None, ge=1)
    display: bool = False


class OutputConfig(_Section):
    """Where artefacts of a run are written."""

    directory: Path = Path("data/output")
    write_annotated_video: bool = True
    annotated_video_name: str = "annotated_video.mp4"
    video_codec: str = "mp4v"
    fallback_fps: float = Field(default=25.0, gt=0.0)


class VisualizationConfig(_Section):
    """Overlay appearance. Colours are BGR to match OpenCV."""

    box_thickness: int = Field(default=2, ge=1)
    font_scale: float = Field(default=0.5, gt=0.0)
    show_labels: bool = True
    show_confidence: bool = True
    show_track_ids: bool = True
    show_hud: bool = True
    # No corner is universally free of traffic, so where the panel sits is a
    # per-camera decision rather than something to hardcode.
    hud_position: Literal["top-left", "top-right", "bottom-left", "bottom-right"] = "top-left"
    class_colors: dict[str, tuple[int, int, int]] = Field(default_factory=dict)

    @field_validator("class_colors")
    @classmethod
    def _valid_bgr(cls, value: dict[str, tuple[int, int, int]]) -> dict[str, tuple[int, int, int]]:
        """Ensure every override is a valid 8-bit BGR triple."""
        for class_name, colour in value.items():
            if any(not 0 <= channel <= 255 for channel in colour):
                raise ValueError(f"class_colors[{class_name!r}] channels must be within 0-255")
        return value


class LoggingConfig(_Section):
    """Verbosity and progress reporting."""

    level: LogLevelName = "INFO"
    file: Path | None = None
    progress_every_frames: int = Field(default=50, ge=1)


class AppConfig(_Section):
    """Root configuration object for a SmartCity Vision run."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    analytics: AnalyticsConfig = Field(default_factory=AnalyticsConfig)
    video: VideoConfig = Field(default_factory=VideoConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serialisable copy of the config.

        Used for run provenance: the snapshot is what later phases persist
        alongside predictions so a result can be traced back to its settings.
        """
        return self.model_dump(mode="json")


def resolve_path(path: Path | str) -> Path:
    """Expand ``~`` and resolve ``path`` against the current working directory."""
    return Path(path).expanduser().resolve()


def load_config(
    path: Path | str | None = None,
    overrides: Sequence[str] | None = None,
    updates: Mapping[str, Any] | None = None,
) -> AppConfig:
    """Load, merge, and validate configuration.

    Precedence, lowest to highest: model defaults, YAML file, dotted overrides,
    ``updates``.

    Args:
        path: YAML config file. When ``None``, ``config/default.yaml`` is used if
            it exists; otherwise the built-in defaults apply.
        overrides: ``"section.key=value"`` strings, values parsed as YAML scalars
            (so ``model.device=cpu`` and ``video.frame_skip=2`` both work).
        updates: Nested mapping of already-typed values, deep-merged last. Used
            by the CLI so that flag values keep the type argparse gave them.

    Returns:
        The validated :class:`AppConfig`.

    Raises:
        ConfigError: If the file is missing/malformed, an override is unparseable,
            or the merged result fails validation.
    """
    data: dict[str, Any] = {}

    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if path is not None or config_path.is_file():
        data = _read_yaml_mapping(config_path)

    if overrides:
        data = _apply_dotted_overrides(data, overrides)

    if updates:
        data = _deep_merge(data, updates)

    try:
        return AppConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"Invalid configuration:\n{_format_validation_error(exc)}") from exc


def _read_yaml_mapping(config_path: Path) -> dict[str, Any]:
    """Read a YAML file that must contain a top-level mapping."""
    if not config_path.is_file():
        raise ConfigError(f"Config file not found: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse YAML config {config_path}: {exc}") from exc

    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"Config file {config_path} must contain a mapping at the top level")
    return dict(raw)


def _deep_merge(base: Mapping[str, Any], incoming: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``incoming`` into ``base`` without mutating either."""
    merged = copy.deepcopy(dict(base))
    for key, value in incoming.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def _apply_dotted_overrides(data: Mapping[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """Return ``data`` with ``a.b=value`` overrides merged in."""
    merged = copy.deepcopy(dict(data))
    for override in overrides:
        key, separator, raw_value = override.partition("=")
        if not separator or not key.strip():
            raise ConfigError(f"Override must look like 'section.key=value', got {override!r}")
        try:
            value = yaml.safe_load(raw_value)
        except yaml.YAMLError as exc:
            raise ConfigError(f"Could not parse override value in {override!r}: {exc}") from exc

        target = merged
        parts = [part.strip() for part in key.strip().split(".")]
        for part in parts[:-1]:
            existing = target.get(part)
            if not isinstance(existing, dict):
                existing = {}
                target[part] = existing
            target = existing
        target[parts[-1]] = value
    return merged


def _format_validation_error(exc: ValidationError) -> str:
    """Render a Pydantic error as one readable ``section.key: message`` line each."""
    lines = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "<root>"
        lines.append(f"  - {location}: {error['msg']}")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_TARGET_CLASSES",
    "AnalyticsConfig",
    "AppConfig",
    "CountingConfig",
    "LoggingConfig",
    "ModelConfig",
    "OutputConfig",
    "TrackingConfig",
    "VideoConfig",
    "VisualizationConfig",
    "load_config",
    "resolve_path",
]
