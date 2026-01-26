from __future__ import annotations

import io
import json
import time
from typing import Any, Optional

import numpy as np
from PIL import Image
import zmq

from .config import PublisherConfig


class PublisherError(RuntimeError):
    pass


def _to_uint8(array: np.ndarray) -> np.ndarray:
    if array.dtype == np.uint8:
        return array
    return np.clip(np.rint(array), 0, 255).astype(np.uint8)


def _convert_format(frame: np.ndarray, target_format: str) -> tuple[np.ndarray, str]:
    if target_format == "auto":
        if frame.ndim == 2:
            target_format = "gray8"
        elif frame.ndim == 3 and frame.shape[2] == 3:
            target_format = "rgb8"
        else:
            raise PublisherError(f"Unsupported frame shape for publishing: {frame.shape}")
    if frame.ndim == 2:
        gray = _to_uint8(frame)
        if target_format == "gray8":
            return gray, "gray8"
        rgb = np.stack([gray, gray, gray], axis=2)
        if target_format == "rgb8":
            return rgb, "rgb8"
        if target_format == "bgr8":
            return rgb[:, :, ::-1], "bgr8"
    elif frame.ndim == 3 and frame.shape[2] == 3:
        rgb = _to_uint8(frame)
        if target_format == "rgb8":
            return rgb, "rgb8"
        if target_format == "bgr8":
            return rgb[:, :, ::-1], "bgr8"
        if target_format == "gray8":
            gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
            return _to_uint8(gray), "gray8"
    raise PublisherError(f"Unsupported frame shape for publishing: {frame.shape}")


def _encode_jpeg(frame: np.ndarray, pixel_format: str, jpeg_quality: int) -> bytes:
    if pixel_format == "gray8":
        image = Image.fromarray(frame, mode="L")
    else:
        if pixel_format == "bgr8":
            frame = frame[:, :, ::-1]
        image = Image.fromarray(frame, mode="RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=jpeg_quality)
    return buf.getvalue()


class ZmqFramePublisher:
    def __init__(self, config: PublisherConfig):
        self.config = config
        self._context: Optional[zmq.Context] = None
        self._socket: Optional[zmq.Socket] = None
        self._topic_bytes: bytes = b""
        self._status_topic_bytes: bytes = b""
        self._seq = 0
        self._status_seq = 0
        self._last_status_time = 0.0
        self._last_frame_time: Optional[float] = None
        self._fps: Optional[float] = None
        self._last_send_time = 0.0

    @property
    def running(self) -> bool:
        return self._socket is not None

    def start(self) -> None:
        if self._socket is not None:
            return
        self._context = zmq.Context.instance()
        socket = self._context.socket(zmq.PUB)
        socket.setsockopt(zmq.SNDHWM, int(self.config.snd_hwm))
        if self.config.conflate:
            socket.setsockopt(zmq.CONFLATE, 1)
        socket.setsockopt(zmq.LINGER, 0)
        if self.config.mode == "bind":
            socket.bind(self.config.endpoint)
        else:
            socket.connect(self.config.endpoint)
        self._socket = socket
        self._topic_bytes = self.config.topic.encode("utf-8")
        self._status_topic_bytes = self.config.status_topic.encode("utf-8")

    def stop(self) -> None:
        if self._socket is None:
            return
        try:
            self._socket.close(0)
        finally:
            self._socket = None

    def _update_fps(self, now: float) -> Optional[float]:
        if self._last_frame_time is None:
            self._last_frame_time = now
            return None
        dt = now - self._last_frame_time
        self._last_frame_time = now
        if dt <= 0:
            return self._fps
        instant = 1.0 / dt
        if self._fps is None:
            self._fps = instant
        else:
            self._fps = (self._fps * 0.8) + (instant * 0.2)
        return self._fps

    def publish_frame(self, frame: np.ndarray, extra_metadata: Optional[dict[str, Any]] = None) -> bool:
        if self._socket is None:
            return False
        now = time.monotonic()
        if self.config.fps_limit:
            min_interval = 1.0 / float(self.config.fps_limit)
            if now - self._last_send_time < min_interval:
                return False

        frame_u8, pixel_format = _convert_format(frame, self.config.format)
        height, width = frame_u8.shape[:2]

        if self.config.compress:
            payload = _encode_jpeg(frame_u8, pixel_format, self.config.jpeg_quality)
            compressed = True
        else:
            payload = np.ascontiguousarray(frame_u8).tobytes()
            compressed = False

        self._seq += 1
        timestamp_ns = time.time_ns()
        fps = self._update_fps(now)

        metadata = {
            "width": int(width),
            "height": int(height),
            "format": pixel_format,
            "timestamp_ns": int(timestamp_ns),
            "compressed": bool(compressed),
            "seq": int(self._seq),
        }
        if fps:
            metadata["fps"] = float(fps)
        if compressed:
            metadata["jpeg_quality"] = int(self.config.jpeg_quality)
        if extra_metadata:
            metadata.update(extra_metadata)

        try:
            self._socket.send_multipart(
                [
                    self._topic_bytes,
                    json.dumps(metadata).encode("utf-8"),
                    payload,
                ],
                flags=zmq.DONTWAIT,
            )
        except zmq.Again:
            return False

        self._last_send_time = now
        return True

    def publish_status(
        self,
        status: str,
        message: Optional[str] = None,
        extra_metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        if self._socket is None:
            return False
        self._status_seq += 1
        metadata = {
            "status": str(status),
            "timestamp_ns": int(time.time_ns()),
            "seq": int(self._status_seq),
        }
        if message:
            metadata["message"] = message
        if extra_metadata:
            metadata.update(extra_metadata)
        try:
            self._socket.send_multipart(
                [
                    self._status_topic_bytes,
                    json.dumps(metadata).encode("utf-8"),
                    b"",
                ],
                flags=zmq.DONTWAIT,
            )
        except zmq.Again:
            return False
        return True

    def tick(self, status: str, message: Optional[str] = None) -> None:
        if self._socket is None:
            return
        now = time.monotonic()
        if now - self._last_status_time >= float(self.config.status_interval_s):
            self.publish_status(status=status, message=message)
            self._last_status_time = now
