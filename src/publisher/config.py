from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "publisher.json"
_ALLOWED_FORMATS = {"auto", "gray8", "rgb8", "bgr8"}
_ALLOWED_MODES = {"bind", "connect"}


class PublisherConfigError(Exception):
    pass


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "y"}
    return default


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _coerce_float(value: Any, default: Optional[float]) -> Optional[float]:
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


@dataclass
class PublisherConfig:
    enabled: bool = False
    endpoint: str = "tcp://127.0.0.1:5555"
    mode: str = "bind"
    topic: str = "camera"
    status_topic: str = "camera/status"
    format: str = "auto"
    compress: bool = False
    jpeg_quality: int = 80
    fps_limit: Optional[float] = None
    snd_hwm: int = 1
    conflate: bool = False
    status_interval_s: float = 1.0
    include_overlays: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PublisherConfig":
        if not isinstance(data, dict):
            raise PublisherConfigError("Publisher config must be a JSON object.")
        format_value = str(data.get("format", cls.format)).lower()
        if format_value not in _ALLOWED_FORMATS:
            raise PublisherConfigError(
                f"Invalid publisher format '{format_value}'. Expected one of: {sorted(_ALLOWED_FORMATS)}"
            )
        mode_value = str(data.get("mode", cls.mode)).lower()
        if mode_value not in _ALLOWED_MODES:
            raise PublisherConfigError(
                f"Invalid publisher mode '{mode_value}'. Expected one of: {sorted(_ALLOWED_MODES)}"
            )

        fps_limit = _coerce_float(data.get("fps_limit", cls.fps_limit), cls.fps_limit)
        if fps_limit is not None and fps_limit <= 0:
            fps_limit = None

        jpeg_quality = _coerce_int(data.get("jpeg_quality", cls.jpeg_quality), cls.jpeg_quality)
        jpeg_quality = int(min(max(jpeg_quality, 1), 95))

        snd_hwm = _coerce_int(data.get("snd_hwm", cls.snd_hwm), cls.snd_hwm)
        if snd_hwm <= 0:
            snd_hwm = cls.snd_hwm

        status_interval = _coerce_float(
            data.get("status_interval_s", cls.status_interval_s),
            cls.status_interval_s,
        )
        if status_interval is None or status_interval <= 0:
            status_interval = cls.status_interval_s

        return cls(
            enabled=_coerce_bool(data.get("enabled", cls.enabled), cls.enabled),
            endpoint=str(data.get("endpoint", cls.endpoint)),
            mode=mode_value,
            topic=str(data.get("topic", cls.topic)),
            status_topic=str(data.get("status_topic", cls.status_topic)),
            format=format_value,
            compress=_coerce_bool(data.get("compress", cls.compress), cls.compress),
            jpeg_quality=jpeg_quality,
            fps_limit=fps_limit,
            snd_hwm=snd_hwm,
            conflate=_coerce_bool(data.get("conflate", cls.conflate), cls.conflate),
            status_interval_s=float(status_interval),
            include_overlays=_coerce_bool(
                data.get("include_overlays", cls.include_overlays),
                cls.include_overlays,
            ),
        )


def _frozen_default_config_path() -> Optional[Path]:
    if hasattr(sys, "_MEIPASS"):
        exe_dir = Path(sys.executable).resolve().parent
        return exe_dir / "config" / "publisher.json"
    return None


def resolve_config_path(path: str | Path | None = None) -> Path:
    if path:
        resolved = Path(path)
    else:
        env_path = os.environ.get("PUBLISHER_CONFIG")
        if env_path:
            resolved = Path(env_path)
        else:
            frozen_path = _frozen_default_config_path()
            resolved = frozen_path if frozen_path is not None else DEFAULT_CONFIG_PATH
    if not resolved.is_absolute():
        resolved = (Path.cwd() / resolved).resolve()
    return resolved


def apply_env_overrides(config: PublisherConfig) -> PublisherConfig:
    enabled = os.environ.get("PUBLISHER_ENABLED")
    if enabled is not None:
        config.enabled = _coerce_bool(enabled, config.enabled)
    endpoint = os.environ.get("PUBLISHER_ENDPOINT")
    if endpoint:
        config.endpoint = endpoint
    mode = os.environ.get("PUBLISHER_MODE")
    if mode:
        mode_value = mode.strip().lower()
        if mode_value in _ALLOWED_MODES:
            config.mode = mode_value
    topic = os.environ.get("PUBLISHER_TOPIC")
    if topic:
        config.topic = topic
    status_topic = os.environ.get("PUBLISHER_STATUS_TOPIC")
    if status_topic:
        config.status_topic = status_topic
    format_value = os.environ.get("PUBLISHER_FORMAT")
    if format_value:
        format_value = format_value.strip().lower()
        if format_value in _ALLOWED_FORMATS:
            config.format = format_value
    compress = os.environ.get("PUBLISHER_COMPRESS")
    if compress is not None:
        config.compress = _coerce_bool(compress, config.compress)
    jpeg_quality = os.environ.get("PUBLISHER_JPEG_QUALITY")
    if jpeg_quality is not None:
        config.jpeg_quality = int(min(max(_coerce_int(jpeg_quality, config.jpeg_quality), 1), 95))
    fps_limit = os.environ.get("PUBLISHER_FPS_LIMIT")
    if fps_limit is not None:
        parsed = _coerce_float(fps_limit, config.fps_limit)
        if parsed and parsed > 0:
            config.fps_limit = parsed
    snd_hwm = os.environ.get("PUBLISHER_SND_HWM")
    if snd_hwm is not None:
        parsed = _coerce_int(snd_hwm, config.snd_hwm)
        if parsed > 0:
            config.snd_hwm = parsed
    conflate = os.environ.get("PUBLISHER_CONFLATE")
    if conflate is not None:
        config.conflate = _coerce_bool(conflate, config.conflate)
    status_interval = os.environ.get("PUBLISHER_STATUS_INTERVAL_S")
    if status_interval is not None:
        parsed = _coerce_float(status_interval, config.status_interval_s)
        if parsed and parsed > 0:
            config.status_interval_s = parsed
    include_overlays = os.environ.get("PUBLISHER_INCLUDE_OVERLAYS")
    if include_overlays is not None:
        config.include_overlays = _coerce_bool(include_overlays, config.include_overlays)
    return config


def load_publisher_config(path: str | Path | None = None) -> tuple[PublisherConfig, Path]:
    config_path = resolve_config_path(path)
    data: dict[str, Any] = {}
    if config_path.exists():
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise PublisherConfigError(f"Failed to read publisher config: {exc}") from exc
    config = PublisherConfig.from_dict(data)
    config = apply_env_overrides(config)
    return config, config_path
