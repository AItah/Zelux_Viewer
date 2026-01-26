from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Tuple

import numpy as np

try:
    import cv2
except Exception as exc:
    print(f"OpenCV (cv2) is required for display: {exc}")
    sys.exit(1)

try:
    import zmq
except Exception as exc:
    print(f"pyzmq is required: {exc}")
    sys.exit(1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ZMQ camera subscriber (viewer).")
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5555", help="ZMQ endpoint to connect to.")
    parser.add_argument("--topic", default="camera", help="Frame topic to subscribe to.")
    parser.add_argument("--status-topic", default="camera/status", help="Status topic to subscribe to.")
    parser.add_argument("--show-status", action="store_true", help="Print status messages.")
    parser.add_argument("--window-name", default="Camera Subscriber", help="OpenCV window title.")
    parser.add_argument("--rcv-hwm", type=int, default=2, help="ZMQ receive high-water mark.")
    return parser.parse_args()


def _decode_frame(metadata: dict[str, Any], payload: bytes) -> Tuple[np.ndarray, str]:
    width = int(metadata.get("width", 0))
    height = int(metadata.get("height", 0))
    fmt = str(metadata.get("format", "gray8")).lower()
    compressed = bool(metadata.get("compressed", False))
    if width <= 0 or height <= 0:
        raise ValueError("Invalid frame size in metadata.")

    if compressed:
        data = np.frombuffer(payload, dtype=np.uint8)
        if fmt == "gray8":
            img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode JPEG payload.")
        # OpenCV returns BGR; for display that's fine.
        return img, fmt

    if fmt == "gray8":
        expected = width * height
        if len(payload) != expected:
            raise ValueError(f"Payload size mismatch: {len(payload)} != {expected}")
        img = np.frombuffer(payload, dtype=np.uint8).reshape(height, width)
        return img, fmt
    if fmt in ("rgb8", "bgr8"):
        expected = width * height * 3
        if len(payload) != expected:
            raise ValueError(f"Payload size mismatch: {len(payload)} != {expected}")
        img = np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 3)
        if fmt == "rgb8":
            img = img[:, :, ::-1]  # convert to BGR for OpenCV display
        return img, fmt

    raise ValueError(f"Unsupported pixel format '{fmt}'.")


def main() -> int:
    args = _parse_args()

    context = zmq.Context.instance()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.RCVHWM, max(1, int(args.rcv_hwm)))
    socket.connect(args.endpoint)
    socket.setsockopt_string(zmq.SUBSCRIBE, args.topic)
    if args.show_status and args.status_topic:
        socket.setsockopt_string(zmq.SUBSCRIBE, args.status_topic)

    print(f"Connected to {args.endpoint}")
    print(f"Subscribed to topic: {args.topic}")
    if args.show_status:
        print(f"Subscribed to status topic: {args.status_topic}")
    print("Press 'q' or ESC to quit.")

    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            parts = socket.recv_multipart()
            if len(parts) < 3:
                continue
            topic = parts[0].decode("utf-8", errors="ignore")
            if topic == args.status_topic:
                if args.show_status:
                    try:
                        status = json.loads(parts[1].decode("utf-8"))
                    except Exception:
                        status = {"raw": parts[1].decode("utf-8", errors="ignore")}
                    print(f"[status] {status}")
                continue
            if topic != args.topic:
                continue
            try:
                metadata = json.loads(parts[1].decode("utf-8"))
            except Exception as exc:
                print(f"Invalid metadata JSON: {exc}")
                continue
            try:
                frame, fmt = _decode_frame(metadata, parts[2])
            except Exception as exc:
                print(f"Frame decode error: {exc}")
                continue

            cv2.imshow(args.window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        socket.close(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
