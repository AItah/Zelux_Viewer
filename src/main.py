"""
Thorlabs CS165CU1 live-view utility with GUI controls built on PyQt6.

Features:
- Connects to the first detected Thorlabs scientific camera (CS165CU1 compatible).
- Start/stop live view, adjust exposure/gain (and gamma if available).
- Save current frame, load image from disk, pop up histogram.
- Crosshair overlay with click-to-place; cursor shows pixel coordinates/value.
"""

try:
    # Add SDK DLL folders to PATH when available
    from windows_setup import configure_path

    configure_path()
except Exception:
    pass

import os
import queue
import sys
import threading
import time
import csv
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw, ImageFont
try:
    import cv2  # Optional: used for generic DirectShow cameras

    _HAVE_OPENCV = True
except Exception:  # pragma: no cover - optional dependency
    _HAVE_OPENCV = False
try:
    from pypylon import pylon  # Optional: Basler cameras via Pylon

    _HAVE_PYLON = True
except Exception:  # pragma: no cover - optional dependency
    _HAVE_PYLON = False
from beam_profile_fitting import gaussian_or_lorentzian_aic
from PyQt6 import QtCore, QtGui, QtWidgets
from thorlabs_tsi_sdk.tl_camera import Frame, TLCamera, TLCameraSDK
from thorlabs_tsi_sdk.tl_camera_enums import SENSOR_TYPE
from thorlabs_tsi_sdk.tl_mono_to_color_processor import MonoToColorProcessorSDK

# Additional DLL path setup for repo-level dlls/{64,32}_lib
def _add_repo_dll_path():
    is_64bits = sys.maxsize > 2**32
    arch_dir = "64_lib" if is_64bits else "32_lib"
    candidate = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dlls", arch_dir))
    if os.path.isdir(candidate):
        os.environ["PATH"] = candidate + os.pathsep + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(candidate)
        except AttributeError:
            pass


def _add_bundle_dll_path():
    """When frozen by PyInstaller, also look for bundled DLLs in dlls/{arch}_lib next to the exe."""
    if not hasattr(sys, "_MEIPASS"):
        return
    is_64bits = sys.maxsize > 2**32
    arch_dir = "64_lib" if is_64bits else "32_lib"
    candidate = os.path.join(sys._MEIPASS, "dlls", arch_dir)
    if os.path.isdir(candidate):
        os.environ["PATH"] = candidate + os.pathsep + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(candidate)
        except AttributeError:
            pass


_add_repo_dll_path()
_add_bundle_dll_path()


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


@dataclass
class FramePayload:
    pil_image: Image.Image
    np_image: np.ndarray
    frame_count: int


class ImageAcquisitionThread(threading.Thread):
    """Grabs frames in a background thread and pushes PIL/NumPy payloads into a queue."""

    def __init__(self, camera: TLCamera):
        super().__init__(daemon=True)
        self._camera = camera
        self._stop_event = threading.Event()
        self._queue: queue.Queue[FramePayload] = queue.Queue(maxsize=2)

        self._is_color = self._camera.camera_sensor_type == SENSOR_TYPE.BAYER
        self._mono_to_color_sdk = None
        self._mono_to_color_processor = None
        if self._is_color:
            self._mono_to_color_sdk = MonoToColorProcessorSDK()
            self._mono_to_color_processor = self._mono_to_color_sdk.create_mono_to_color_processor(
                SENSOR_TYPE.BAYER,
                self._camera.color_filter_array_phase,
                self._camera.get_color_correction_matrix(),
                self._camera.get_default_white_balance_matrix(),
                self._camera.bit_depth,
            )

        self._bit_depth = self._camera.bit_depth
        self._camera.image_poll_timeout_ms = 50  # keep UI responsive

    def stop(self):
        self._stop_event.set()

    def get_queue(self):
        return self._queue

    def _convert_frame(self, frame: Frame) -> FramePayload:
        if self._is_color:
            color_image_data = self._mono_to_color_processor.transform_to_24(
                frame.image_buffer,
                frame.image_buffer.shape[1],
                frame.image_buffer.shape[0],
            )
            color_image_data = color_image_data.reshape(
                frame.image_buffer.shape[0], frame.image_buffer.shape[1], 3
            )
            pil_image = Image.fromarray(color_image_data, mode="RGB")
            np_image = color_image_data
        else:
            scaled_image = frame.image_buffer >> (self._bit_depth - 8)
            np_image = scaled_image
            pil_image = Image.fromarray(scaled_image)
        return FramePayload(pil_image=pil_image, np_image=np_image, frame_count=frame.frame_count)

    def run(self):
        while not self._stop_event.is_set():
            frame = self._camera.get_pending_frame_or_null()
            if frame is None:
                time.sleep(0.01)
                continue
            try:
                payload = self._convert_frame(frame)
                self._queue.put_nowait(payload)
            except queue.Full:
                pass
            except Exception as exc:  # pragma: no cover - defensive
                print(f"Acquisition error: {exc}")
                break

        if self._is_color:
            self._mono_to_color_processor.dispose()
            self._mono_to_color_sdk.dispose()


class ImageLabel(QtWidgets.QLabel):
    mouse_moved = QtCore.pyqtSignal(int, int)
    mouse_clicked = QtCore.pyqtSignal(int, int)
    pan_started = QtCore.pyqtSignal()
    pan_dragged = QtCore.pyqtSignal(int, int)
    pan_finished = QtCore.pyqtSignal()
    middle_clicked = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
        self.setStyleSheet("background-color: #111;")
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding
        )
        self._panning_enabled = False
        self._pan_active = False
        self._pan_moved = False
        self._last_pan_pos = QtCore.QPoint()

    def set_panning_enabled(self, enabled: bool):
        self._panning_enabled = bool(enabled)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        pos = event.position().toPoint()
        self.mouse_moved.emit(pos.x(), pos.y())
        if self._pan_active:
            delta = pos - self._last_pan_pos
            if not self._pan_moved and (abs(delta.x()) >= 2 or abs(delta.y()) >= 2):
                self._pan_moved = True
            if self._pan_moved:
                self.pan_dragged.emit(delta.x(), delta.y())
                self._last_pan_pos = pos
                event.accept()
                return
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if event.button() == QtCore.Qt.MouseButton.MiddleButton:
            self.middle_clicked.emit()
            event.accept()
            return
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self._panning_enabled:
            self._pan_active = True
            self._pan_moved = False
            self._last_pan_pos = event.position().toPoint()
            self.pan_started.emit()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            if self._pan_active:
                if self._pan_moved:
                    self.pan_finished.emit()
                else:
                    pos = event.position().toPoint()
                    self.mouse_clicked.emit(pos.x(), pos.y())
                self._pan_active = False
                self._pan_moved = False
                event.accept()
                return
            else:
                pos = event.position().toPoint()
                self.mouse_clicked.emit(pos.x(), pos.y())
        super().mouseReleaseEvent(event)


class OverlayPanel(QtWidgets.QFrame):
    """Floating panel that can be dragged (via header) and resized (via size grip)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._dragging = False
        self._drag_offset = QtCore.QPoint()
        self._drag_handle_height = 28
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet("background-color: rgba(20, 20, 20, 200); color: #eee;")
        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.setLineWidth(1)
        self.setMinimumWidth(320)
        self.setMinimumHeight(180)
        self._resizing = False
        self._resize_start_geom = QtCore.QRect()
        self._resize_start_pos = QtCore.QPoint()
        self._resize_margin = 12
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred
        )

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        pos = event.position().toPoint()
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            if self._is_in_resize_corner(pos):
                self._resizing = True
                self._resize_start_geom = self.geometry()
                self._resize_start_pos = pos
                self.setCursor(QtCore.Qt.CursorShape.SizeFDiagCursor)
                event.accept()
                return
            elif pos.y() <= self._drag_handle_height:
                self._dragging = True
                self._drag_offset = pos
                self.setCursor(QtCore.Qt.CursorShape.SizeAllCursor)
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        if self._resizing:
            parent = self.parentWidget()
            delta = event.position().toPoint() - self._resize_start_pos
            new_w = max(self.minimumWidth(), self._resize_start_geom.width() + delta.x())
            new_h = max(self.minimumHeight(), self._resize_start_geom.height() + delta.y())
            if parent:
                max_w = max(50, parent.width() - self.x())
                max_h = max(50, parent.height() - self.y())
                new_w = min(new_w, max_w)
                new_h = min(new_h, max_h)
            self.resize(new_w, new_h)
            event.accept()
            return
        elif self._dragging:
            parent = self.parentWidget()
            if parent:
                delta = event.position().toPoint() - self._drag_offset
                new_pos = self.pos() + delta
                max_x = max(0, parent.width() - self.width())
                max_y = max(0, parent.height() - self.height())
                new_pos.setX(max(0, min(new_pos.x(), max_x)))
                new_pos.setY(max(0, min(new_pos.y(), max_y)))
                self.move(new_pos)
            event.accept()
            return
        else:
            if self._is_in_resize_corner(event.position().toPoint()):
                self.setCursor(QtCore.Qt.CursorShape.SizeFDiagCursor)
            else:
                self.unsetCursor()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._dragging = False
            self._resizing = False
            self.unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _is_in_resize_corner(self, pos: QtCore.QPoint) -> bool:
        return (self.width() - pos.x() <= self._resize_margin) and (
            self.height() - pos.y() <= self._resize_margin
        )


class GenericVideoThread(threading.Thread):
    """OpenCV-based capture for generic DirectShow cameras."""

    def __init__(self, index: int):
        super().__init__(daemon=True)
        self.index = index
        self._stop_event = threading.Event()
        self._queue: queue.Queue[FramePayload] = queue.Queue(maxsize=2)
        self._cap = None
        self._lock = threading.Lock()

    def stop(self):
        self._stop_event.set()

    def get_queue(self):
        return self._queue

    def set_exposure_us(self, value_us: int):
        """Attempt to set exposure in seconds if supported."""
        if not self._cap:
            return False
        with self._lock:
            try:
                return bool(self._cap.set(cv2.CAP_PROP_EXPOSURE, float(value_us) / 1e6))
            except Exception:
                return False

    def set_gain(self, value):
        """Attempt to set gain if supported."""
        if not self._cap:
            return False
        with self._lock:
            try:
                return bool(self._cap.set(cv2.CAP_PROP_GAIN, float(value)))
            except Exception:
                return False

    def get_props(self):
        """Return (exposure_us, gain) if available, else (None, None)."""
        if not self._cap:
            return None, None
        with self._lock:
            try:
                exp = self._cap.get(cv2.CAP_PROP_EXPOSURE)
                gain = self._cap.get(cv2.CAP_PROP_GAIN)
            except Exception:
                return None, None
        # Many drivers return negative/seconds; best effort to convert.
        if exp is None:
            exp_us = None
        else:
            try:
                exp_us = int(exp * 1e6) if exp > 0 else None
            except Exception:
                exp_us = None
        return exp_us, gain

    def _open_capture(self):
        if not _HAVE_OPENCV:
            return False
        self._cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            self._cap = None
            return False
        return True

    def _read_frame(self):
        if not self._cap:
            return None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        # frame is BGR uint8
        np_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(np_image, mode="RGB")
        return FramePayload(pil_image=pil_image, np_image=np_image, frame_count=0)

    def run(self):
        if not self._open_capture():
            return
        while not self._stop_event.is_set():
            payload = self._read_frame()
            if payload is None:
                time.sleep(0.02)
                continue
            try:
                self._queue.put_nowait(payload)
            except queue.Full:
                pass
        if self._cap:
            try:
                self._cap.release()
            except Exception:
                pass


class PylonVideoThread(threading.Thread):
    """Pylon-based capture for Basler cameras."""

    def __init__(self, device_info):
        super().__init__(daemon=True)
        self.device_info = device_info
        self._stop_event = threading.Event()
        self._queue: queue.Queue[FramePayload] = queue.Queue(maxsize=2)
        self._camera = None
        self._converter = None
        self.frame_count = 0
        self._lock = threading.Lock()

    def stop(self):
        self._stop_event.set()

    def get_queue(self):
        return self._queue

    def _with_camera(self, fn, default=None):
        cam = self._camera
        if cam is None:
            return default
        with self._lock:
            try:
                return fn(cam)
            except Exception:
                return default

    def get_props(self):
        """Return (exposure_us, gain_db or raw) if available."""
        def _read(cam):
            exp = None
            gain = None
            for name in ("ExposureTime", "ExposureTimeAbs"):
                if hasattr(cam, name):
                    try:
                        exp = float(getattr(cam, name).GetValue())
                        break
                    except Exception:
                        pass
            for name in ("Gain", "GainRaw"):
                if hasattr(cam, name):
                    try:
                        gain = float(getattr(cam, name).GetValue())
                        break
                    except Exception:
                        pass
            if exp is not None:
                exp = float(exp)
            return exp, gain
        return self._with_camera(_read, (None, None))

    def get_limits(self):
        """Return exposure/gain (min,max) tuples when possible."""
        def _limits(cam):
            exp_min = 1.0
            exp_max = 10_000_000.0
            gain_min = 0.0
            gain_max = 30.0
            for name in ("ExposureTime", "ExposureTimeAbs"):
                node = getattr(cam, name, None)
                if node:
                    try:
                        exp_min = float(node.GetMin())
                        exp_max = float(node.GetMax())
                        break
                    except Exception:
                        pass
            for name in ("Gain", "GainRaw"):
                node = getattr(cam, name, None)
                if node:
                    try:
                        gain_min = float(node.GetMin())
                        gain_max = float(node.GetMax())
                        break
                    except Exception:
                        pass
            return (exp_min, exp_max), (gain_min, gain_max)
        return self._with_camera(_limits, ((1.0, 10_000_000.0), (0.0, 30.0)))

    def set_exposure_us(self, value: float) -> bool:
        def _set(cam):
            for name in ("ExposureTime", "ExposureTimeAbs"):
                node = getattr(cam, name, None)
                if node:
                    try:
                        node.SetValue(float(value))
                        return True
                    except Exception:
                        pass
            return False
        return bool(self._with_camera(_set, False))

    def set_gain(self, value: float) -> bool:
        def _set(cam):
            for name in ("Gain", "GainRaw"):
                node = getattr(cam, name, None)
                if node:
                    try:
                        node.SetValue(float(value))
                        return True
                    except Exception:
                        pass
            return False
        return bool(self._with_camera(_set, False))

    def _open_camera(self):
        try:
            self._camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(self.device_info))
            self._camera.Open()
            self._converter = pylon.ImageFormatConverter()
            self._converter.OutputPixelFormat = pylon.PixelType_RGB8packed
            self._converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
            return True
        except Exception:
            self._camera = None
            self._converter = None
            return False

    def run(self):
        if not self._open_camera():
            return
        try:
            self._camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            while not self._stop_event.is_set() and self._camera.IsGrabbing():
                grab = self._camera.RetrieveResult(50, pylon.TimeoutHandling_Return)
                if not grab or not grab.GrabSucceeded():
                    if grab:
                        grab.Release()
                    time.sleep(0.01)
                    continue
                try:
                    converted = self._converter.Convert(grab)
                    img_arr = converted.GetArray()
                    if img_arr.ndim == 2:
                        np_image = img_arr
                        pil_image = Image.fromarray(np_image)
                    else:
                        np_image = img_arr
                        pil_image = Image.fromarray(np_image, mode="RGB")
                    self.frame_count += 1
                    payload = FramePayload(pil_image=pil_image, np_image=np_image, frame_count=self.frame_count)
                    try:
                        self._queue.put_nowait(payload)
                    except queue.Full:
                        pass
                finally:
                    grab.Release()
        finally:
            try:
                self._camera.StopGrabbing()
            except Exception:
                pass
            try:
                self._camera.Close()
            except Exception:
                pass


class CameraApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        os.makedirs(DATA_DIR, exist_ok=True)
        self.sdk = TLCameraSDK()
        self.camera: Optional[TLCamera] = None
        self._use_cv = False
        self._use_pylon = False
        self.cv_index: Optional[int] = None
        self.pylon_device_info = None

        self.cross_pos = None
        self.last_payload: Optional[FramePayload] = None
        self.acq_thread: Optional[ImageAcquisitionThread] = None
        self.cv_thread: Optional[GenericVideoThread] = None
        self.pylon_thread: Optional[PylonVideoThread] = None
        self._live = False
        self._zoom = 1.0
        self._fit_to_window = True
        self._last_render_scale = 1.0
        self._histogram_enabled = False
        self._hist_size = QtCore.QSize(360, 180)
        self._force_grayscale = False
        self.mm_per_px_x = 0.00345 # thorlab zelus pixel size
        self.mm_per_px_y = 0.00345 # thorlab zelus pixel size
        self._gauss_fit = None
        self._hist_window_positioned = False
        self._fit_window_positioned = False
        self._calibration_window_positioned = False
        self._calibration_axis: Optional[str] = None  # "x", "y", or "both" while active
        self._calibration_last_axis: Optional[str] = None  # remember last calibrated axis for overlay
        self._calibration_points: list[tuple[int, int]] = []
        self._calibration_dragging = False
        self._calibration_drag_mode: Optional[str] = None
        self._calibration_drag_start: Optional[QtCore.QPointF] = None
        self._calibration_drag_start_points: list[tuple[int, int]] = []
        self._measure_active_mode: Optional[str] = None  # "line" or "arc"
        self._measure_line_points: list[tuple[int, int]] = []
        self._measure_line_dragging = False
        self._measure_line_drag_mode: Optional[str] = None
        self._measure_line_drag_start: Optional[QtCore.QPointF] = None
        self._measure_line_drag_start_points: list[tuple[int, int]] = []
        self._measure_arc_points: list[tuple[int, int]] = []
        self._measure_arc_dragging = False
        self._measure_arc_drag_index: Optional[int] = None
        self._in_hist_update = False
        self._line_select_mode = False
        self._line_points: list[tuple[int, int]] = []
        self._line_profile = None
        self._line_fit = None
        self._line_edit_mode = False
        self._line_dragging = False
        self._line_drag_mode: Optional[str] = None
        self._axis_lines: list[tuple[tuple[int, int], tuple[int, int], QtGui.QColor]] = []
        self._axis_fit_results = {}
        self._axis_center_px: Optional[tuple[float, float]] = None
        self._settings = QtCore.QSettings("BaslerTool", "Viewer")
        self._filtered_np_image: Optional[np.ndarray] = None
        self._pending_sections: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        self._restore_window_state(self, "main_window")
        self.setWindowTitle("Live View - No camera")
        self._build_ui()

        self._set_camera_controls_enabled(False)
        self._connect_first_available_camera(show_message=False)

        self.poll_timer = QtCore.QTimer(self)
        self.poll_timer.setInterval(15)
        self.poll_timer.timeout.connect(self._poll_queue)
        self.poll_timer.start()

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        video_container = QtWidgets.QFrame()
        video_container.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        video_container.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding
        )
        video_layout = QtWidgets.QVBoxLayout(video_container)
        video_layout.setContentsMargins(0, 0, 0, 0)
        video_layout.setSpacing(0)

        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
        self.scroll_area.setStyleSheet("background-color: #111;")

        self.image_label = ImageLabel()
        self.image_label.setMinimumSize(640, 480)
        self.image_label.set_panning_enabled(False)
        self.image_label.mouse_moved.connect(self.on_mouse_move)
        self.image_label.mouse_clicked.connect(self.on_mouse_click)
        self.image_label.pan_started.connect(self.on_pan_start)
        self.image_label.pan_dragged.connect(self.on_pan_drag)
        self.image_label.pan_finished.connect(self.on_pan_end)
        self.image_label.middle_clicked.connect(self.fit_to_window)

        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.viewport().installEventFilter(self)
        self.image_label.installEventFilter(self)
        video_layout.addWidget(self.scroll_area)

        layout.addWidget(video_container, stretch=1)

        layout.addLayout(self._build_status_row())

        self.controls_window = self._build_controls_window()
        self.controls_window.installEventFilter(self)
        restored_controls = self._restore_window_state(self.controls_window, "controls_window")
        if not restored_controls:
            QtCore.QTimer.singleShot(
                0, lambda: self._position_window_near_main(self.controls_window, QtCore.QPoint(14, 14))
            )
        self.controls_window.show()

        self.hist_window = self._build_hist_window()
        self.hist_window.installEventFilter(self)

        self.fit_window = self._build_fit_window()
        self.fit_window.installEventFilter(self)

        self.calibration_window = self._build_calibration_window()
        self.calibration_window.installEventFilter(self)

        self._restore_window_state(self.hist_window, "hist_window")
        self._restore_window_state(self.fit_window, "fit_window")
        calib_restored = self._restore_window_state(self.calibration_window, "calibration_window")
        if calib_restored:
            self._calibration_window_positioned = True
        if getattr(self, "measure_checkbox", None) and self.calibration_window.isVisible():
            self.measure_checkbox.blockSignals(True)
            self.measure_checkbox.setChecked(True)
            self.measure_checkbox.blockSignals(False)

        self._sync_panning_enabled()

    def _build_controls_window(self) -> QtWidgets.QWidget:
        window = QtWidgets.QWidget(None, QtCore.Qt.WindowType.Window)
        window.setWindowTitle("Camera Controls")
        window.setMinimumWidth(420)

        vbox = QtWidgets.QVBoxLayout(window)
        vbox.setContentsMargins(12, 12, 12, 12)
        vbox.setSpacing(10)

        buttons_widget = QtWidgets.QWidget(window)
        buttons_layout = self._build_button_row()
        buttons_widget.setLayout(buttons_layout)
        vbox.addWidget(buttons_widget)

        vbox.addWidget(self._build_param_group())
        vbox.addStretch(1)
        return window

    def _build_hist_window(self) -> QtWidgets.QWidget:
        window = QtWidgets.QWidget(None, QtCore.Qt.WindowType.Window)
        window.setWindowTitle("Histogram")
        window.resize(self._hist_size)

        layout = QtWidgets.QVBoxLayout(window)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        self.hist_label = QtWidgets.QLabel(window)
        self.hist_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.hist_label.setStyleSheet("background-color: transparent;")
        layout.addWidget(self.hist_label)

        grip_row = QtWidgets.QHBoxLayout()
        grip_row.addStretch(1)
        grip_row.addWidget(QtWidgets.QSizeGrip(window))
        layout.addLayout(grip_row)

        window.hide()
        return window

    def _build_fit_window(self) -> QtWidgets.QWidget:
        window = QtWidgets.QWidget(None, QtCore.Qt.WindowType.Window)
        window.setWindowTitle("Gaussian Fit")
        window.resize(380, 200)

        layout = QtWidgets.QVBoxLayout(window)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        info_label = QtWidgets.QLabel("Place a cross on the image, then run the fit.")
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #ccc;")
        layout.addWidget(info_label)

        filter_row = QtWidgets.QHBoxLayout()
        filter_row.addWidget(QtWidgets.QLabel("Profile filter:"))
        self.filter_mode_combo = QtWidgets.QComboBox()
        self.filter_mode_combo.addItems(["None", "Low-pass", "High-pass", "Band-pass"])
        self.filter_mode_combo.currentTextChanged.connect(self._on_filter_changed)
        filter_row.addWidget(self.filter_mode_combo)
        self.filter_low_spin = QtWidgets.QDoubleSpinBox()
        self.filter_low_spin.setRange(0.0, 0.5)
        self.filter_low_spin.setDecimals(3)
        self.filter_low_spin.setSingleStep(0.01)
        self.filter_low_spin.setValue(0.01)
        self.filter_low_spin.setSuffix(" xNyq")
        self.filter_low_spin.valueChanged.connect(self._on_filter_changed)
        filter_row.addWidget(QtWidgets.QLabel("Low cut"))
        filter_row.addWidget(self.filter_low_spin)
        self.filter_high_spin = QtWidgets.QDoubleSpinBox()
        self.filter_high_spin.setRange(0.0, 0.5)
        self.filter_high_spin.setDecimals(3)
        self.filter_high_spin.setSingleStep(0.01)
        self.filter_high_spin.setValue(0.25)
        self.filter_high_spin.setSuffix(" xNyq")
        self.filter_high_spin.valueChanged.connect(self._on_filter_changed)
        filter_row.addWidget(QtWidgets.QLabel("High cut"))
        filter_row.addWidget(self.filter_high_spin)
        filter_row.addStretch(1)
        layout.addLayout(filter_row)
        self._sync_filter_controls()

        btn_row = QtWidgets.QHBoxLayout()
        # self.gauss_btn = QtWidgets.QPushButton("Gaussian Fit @ Cross")
        # self.gauss_btn.clicked.connect(self.compute_gaussian_fit)
        # btn_row.addWidget(self.gauss_btn)
        self.line_fit_btn = QtWidgets.QPushButton("2-Point Line Fit")
        self.line_fit_btn.clicked.connect(self.start_line_fit_selection)
        btn_row.addWidget(self.line_fit_btn)
        self.run_scipy_btn = QtWidgets.QPushButton("SciPy Fit")
        self.run_scipy_btn.clicked.connect(self.run_scipy_fit)
        self.run_scipy_btn.setEnabled(True)
        btn_row.addWidget(self.run_scipy_btn)
        self.clear_fit_btn = QtWidgets.QPushButton("Clear Fits")
        self.clear_fit_btn.clicked.connect(self.clear_fits)
        btn_row.addWidget(self.clear_fit_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        self.gauss_result_label = QtWidgets.QLabel("Gauss X: -, Y: -")
        self.gauss_result_label.setWordWrap(True)
        layout.addWidget(self.gauss_result_label)

        self.line_fit_status = QtWidgets.QLabel("Line fit: not computed.")
        self.line_fit_status.setWordWrap(True)
        layout.addWidget(self.line_fit_status)

        self.line_plot_label = QtWidgets.QLabel()
        self.line_plot_label.setMinimumHeight(180)
        self.line_plot_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.line_plot_label)

        self.axis_fit_status = QtWidgets.QLabel("360 fit: not computed.")
        self.axis_fit_status.setWordWrap(True)
        layout.addWidget(self.axis_fit_status)

        self.axis_plot_h = QtWidgets.QLabel()
        self.axis_plot_h.setMinimumHeight(140)
        self.axis_plot_h.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.axis_plot_h)

        self.axis_plot_v = QtWidgets.QLabel()
        self.axis_plot_v.setMinimumHeight(140)
        self.axis_plot_v.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.axis_plot_v)

        window.hide()
        return window

    def _build_calibration_window(self) -> QtWidgets.QWidget:
        window = QtWidgets.QWidget(None, QtCore.Qt.WindowType.Window)
        window.setWindowTitle("mm/px Calibration")
        window.resize(360, 160)

        layout = QtWidgets.QVBoxLayout(window)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        info_label = QtWidgets.QLabel(
            "Check Measure, pick Calib X / Calib Y / Calib Both, click two points on the live view, drag to adjust, then press Done to enter the real length (mm)."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #ccc;")
        layout.addWidget(info_label)

        btn_row = QtWidgets.QHBoxLayout()
        self.calib_x_btn = QtWidgets.QPushButton("Calib X")
        self.calib_x_btn.clicked.connect(lambda: self._start_calibration(axis="x"))
        btn_row.addWidget(self.calib_x_btn)
        self.calib_y_btn = QtWidgets.QPushButton("Calib Y")
        self.calib_y_btn.clicked.connect(lambda: self._start_calibration(axis="y"))
        btn_row.addWidget(self.calib_y_btn)
        self.calib_both_btn = QtWidgets.QPushButton("Calib Both")
        self.calib_both_btn.clicked.connect(lambda: self._start_calibration(axis="both"))
        btn_row.addWidget(self.calib_both_btn)
        self.calib_done_btn = QtWidgets.QPushButton("Done")
        self.calib_done_btn.clicked.connect(self._finish_calibration)
        btn_row.addWidget(self.calib_done_btn)
        self.calib_clear_btn = QtWidgets.QPushButton("Clear")
        self.calib_clear_btn.clicked.connect(self.clear_calibration)
        btn_row.addWidget(self.calib_clear_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        self.calibration_status_label = QtWidgets.QLabel("Idle. Choose Calib X, Calib Y, or Calib Both to begin.")
        self.calibration_status_label.setWordWrap(True)
        layout.addWidget(self.calibration_status_label)

        measure_row = QtWidgets.QHBoxLayout()
        self.measure_line_btn = QtWidgets.QPushButton("Line")
        self.measure_line_btn.clicked.connect(self._start_measure_line)
        measure_row.addWidget(self.measure_line_btn)
        self.measure_arc_btn = QtWidgets.QPushButton("Arc")
        self.measure_arc_btn.clicked.connect(self._start_measure_arc)
        measure_row.addWidget(self.measure_arc_btn)
        self.measure_clear_btn = QtWidgets.QPushButton("Clear Measure")
        self.measure_clear_btn.clicked.connect(self._clear_measure_tools)
        measure_row.addWidget(self.measure_clear_btn)
        measure_row.addStretch(1)
        layout.addLayout(measure_row)

        self.measure_status_label = QtWidgets.QLabel("Measurement: Idle. Pick Line or Arc.")
        self.measure_status_label.setWordWrap(True)
        layout.addWidget(self.measure_status_label)

        window.hide()
        return window

    def _position_window_near_main(
        self, window: QtWidgets.QWidget, offset: QtCore.QPoint, anchor_geom: Optional[QtCore.QRect] = None
    ):
        if not window:
            return
        main_geom = anchor_geom if anchor_geom is not None else self.frameGeometry()
        if not main_geom.isValid():
            main_geom = QtCore.QRect(100, 100, 900, 700)
        target = QtCore.QPoint(main_geom.topRight().x() + offset.x(), main_geom.top() + offset.y())
        screen = QtGui.QGuiApplication.screenAt(main_geom.center())
        if screen is None:
            screen = QtGui.QGuiApplication.primaryScreen()
        if screen:
            avail = screen.availableGeometry()
            max_x = max(avail.left(), avail.right() - window.width())
            max_y = max(avail.top(), avail.bottom() - window.height())
            target.setX(min(max(avail.left(), target.x()), max_x))
            target.setY(min(max(avail.top(), target.y()), max_y))
        window.move(target)

    def _restore_window_state(self, window: QtWidgets.QWidget, key_prefix: str) -> bool:
        if not window:
            return False
        x = self._settings.value(f"{key_prefix}/x", None, type=int)
        y = self._settings.value(f"{key_prefix}/y", None, type=int)
        w = self._settings.value(f"{key_prefix}/w", None, type=int)
        h = self._settings.value(f"{key_prefix}/h", None, type=int)
        visible = self._settings.value(f"{key_prefix}/visible", None, type=bool)
        restored = False
        if None not in (x, y, w, h):
            geom = QtCore.QRect(x, y, max(50, w), max(50, h))
            screen = QtGui.QGuiApplication.screenAt(geom.center())
            if screen is None:
                screen = QtGui.QGuiApplication.primaryScreen()
            if screen:
                avail = screen.availableGeometry()
                if not avail.contains(geom):
                    geom.moveTopLeft(avail.topLeft())
            window.setGeometry(geom)
            restored = True
        if visible is True:
            window.show()
            restored = True
        elif visible is False:
            window.hide()
            restored = True
        return restored

    def _save_window_state(self, window: QtWidgets.QWidget, key_prefix: str):
        if not window:
            return
        geom = window.geometry()
        self._settings.setValue(f"{key_prefix}/x", geom.x())
        self._settings.setValue(f"{key_prefix}/y", geom.y())
        self._settings.setValue(f"{key_prefix}/w", geom.width())
        self._settings.setValue(f"{key_prefix}/h", geom.height())
        self._settings.setValue(f"{key_prefix}/visible", window.isVisible())

    def _build_button_row(self) -> QtWidgets.QVBoxLayout:
        layout = QtWidgets.QVBoxLayout()

        row1 = QtWidgets.QHBoxLayout()
        self.connect_btn = QtWidgets.QPushButton("Connect Camera")
        self.connect_btn.clicked.connect(self.connect_camera)
        row1.addWidget(self.connect_btn)
        for text, handler in [
            ("Start Live", self.start_live),
            ("Stop Live", self.stop_live),
            ("Save Image", self.save_image),
            ("Load Image", self.load_image),
        ]:
            btn = QtWidgets.QPushButton(text)
            btn.clicked.connect(handler)
            row1.addWidget(btn)
        row1.addStretch(1)

        row2 = QtWidgets.QHBoxLayout()
        for text, handler in [
            ("Clear Cross", self.clear_cross),
            ("Fit", self.fit_to_window),
        ]:
            btn = QtWidgets.QPushButton(text)
            btn.clicked.connect(handler)
            row2.addWidget(btn)
        self.hist_checkbox = QtWidgets.QCheckBox("Histogram")
        self.hist_checkbox.stateChanged.connect(self.toggle_histogram)
        row2.addWidget(self.hist_checkbox)
        self.fit_checkbox = QtWidgets.QCheckBox("Fit Window")
        self.fit_checkbox.stateChanged.connect(self.toggle_fit_window)
        row2.addWidget(self.fit_checkbox)
        self.gray_checkbox = QtWidgets.QCheckBox("Grayscale")
        self.gray_checkbox.stateChanged.connect(self.toggle_grayscale)
        row2.addWidget(self.gray_checkbox)
        self.measure_checkbox = QtWidgets.QCheckBox("Measure")
        self.measure_checkbox.stateChanged.connect(self.toggle_measure_window)
        row2.addWidget(self.measure_checkbox)
        row2.addStretch(1)

        layout.addLayout(row1)
        layout.addLayout(row2)
        return layout

    def _build_param_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Parameters")
        grid = QtWidgets.QGridLayout(group)

        exp_min = 1
        exp_max = 10_000_000
        exp_val = float(getattr(self.camera, "exposure_time_us", 10_000))
        exp_range = getattr(self.camera, "exposure_time_range_us", None)
        if exp_range is not None:
            exp_min = int(getattr(exp_range, "min", exp_min))
            exp_max = int(getattr(exp_range, "max", exp_max))
        exp_val = float(min(max(exp_val, exp_min), exp_max))

        self.exposure_spin = QtWidgets.QDoubleSpinBox()
        self.exposure_spin.setRange(exp_min, exp_max)
        self.exposure_spin.setDecimals(1)
        self.exposure_spin.setSingleStep(100)
        self.exposure_spin.setValue(exp_val)
        self.exposure_spin.setMaximumWidth(120)
        grid.addWidget(QtWidgets.QLabel("Exposure (us)"), 0, 0)
        grid.addWidget(self.exposure_spin, 0, 1)
        self.exposure_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.exposure_slider.setRange(exp_min, exp_max)
        self.exposure_slider.setValue(int(exp_val))
        self.exposure_slider.setSingleStep(max(1, (exp_max - exp_min) // 500))
        self.exposure_slider.setMaximumWidth(120)
        self.exposure_slider.valueChanged.connect(self._on_exposure_slider)
        grid.addWidget(self.exposure_slider, 0, 2, 1, 2)
        exposure_btn = QtWidgets.QPushButton("Set")
        exposure_btn.setMaximumWidth(70)
        exposure_btn.clicked.connect(self.set_exposure)
        grid.addWidget(exposure_btn, 0, 4)
        self.exposure_value_label = QtWidgets.QLabel(f"{int(exp_val)} us")
        grid.addWidget(self.exposure_value_label, 0, 5)

        fine_min, fine_max = 1, 1_000_000  # 1 us to 1000 ms
        grid.addWidget(QtWidgets.QLabel("Fine Exp (us)"), 1, 0)
        self.exposure_fine_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.exposure_fine_slider.setRange(fine_min, fine_max)
        self.exposure_fine_slider.setValue(int(min(max(exp_val, fine_min), fine_max)))
        self.exposure_fine_slider.setSingleStep(max(1, (fine_max - fine_min) // 500))
        self.exposure_fine_slider.setMaximumWidth(120)
        self.exposure_fine_slider.valueChanged.connect(self._on_exposure_slider_fine)
        grid.addWidget(self.exposure_fine_slider, 1, 1, 1, 5)

        gain_range = getattr(self.camera, "gain_range", None)
        gain_min = getattr(gain_range, "min", 0)
        gain_max = getattr(gain_range, "max", 0)
        gain_val = int(getattr(self.camera, "gain", gain_min))
        if gain_max >= gain_min:
            gain_val = int(min(max(gain_val, gain_min), gain_max))
        self.gain_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.gain_slider.setRange(int(gain_min), int(gain_max))
        self.gain_slider.setValue(gain_val)
        self.gain_slider.valueChanged.connect(self._on_gain_slider)
        self.gain_value_label = QtWidgets.QLabel(str(int(getattr(self.camera, "gain", gain_val))))
        grid.addWidget(QtWidgets.QLabel("Gain"), 2, 0)
        self.gain_slider.setMaximumWidth(150)
        grid.addWidget(self.gain_slider, 2, 1, 1, 2)
        grid.addWidget(self.gain_value_label, 2, 3)

        # self.gamma_spin = QtWidgets.QDoubleSpinBox()
        # self.gamma_spin.setRange(0.1, 10.0)
        # self.gamma_spin.setDecimals(3)
        # self.gamma_spin.setSingleStep(0.1)
        # self.gamma_spin.setValue(float(getattr(self.camera, "gamma", 1.0)))
        # grid.addWidget(QtWidgets.QLabel("Gamma"), 3, 0)
        # grid.addWidget(self.gamma_spin, 3, 1)
        # gamma_btn = QtWidgets.QPushButton("Set")
        # gamma_btn.clicked.connect(self.set_gamma)
        # grid.addWidget(gamma_btn, 3, 2)

        grid.addWidget(QtWidgets.QLabel("mm/px X"), 4, 0)
        self.mm_px_x_spin = QtWidgets.QDoubleSpinBox()
        self.mm_px_x_spin.setRange(0.0001, 10000.0)
        self.mm_px_x_spin.setDecimals(7)
        self.mm_px_x_spin.setSingleStep(0.001)
        self.mm_px_x_spin.setValue(self.mm_per_px_x)
        self.mm_px_x_spin.setMaximumWidth(140)
        self.mm_px_x_spin.valueChanged.connect(lambda v: self._update_mm_scale(v, axis="x"))
        grid.addWidget(self.mm_px_x_spin, 4, 1)

        grid.addWidget(QtWidgets.QLabel("mm/px Y"), 4, 2)
        self.mm_px_y_spin = QtWidgets.QDoubleSpinBox()
        self.mm_px_y_spin.setRange(0.0001, 10000.0)
        self.mm_px_y_spin.setDecimals(7)
        self.mm_px_y_spin.setSingleStep(0.001)
        self.mm_px_y_spin.setValue(self.mm_per_px_y)
        self.mm_px_y_spin.setMaximumWidth(140)
        self.mm_px_y_spin.valueChanged.connect(lambda v: self._update_mm_scale(v, axis="y"))
        grid.addWidget(self.mm_px_y_spin, 4, 3)

        return group

    def _build_status_row(self) -> QtWidgets.QHBoxLayout:
        row = QtWidgets.QHBoxLayout()
        self.status_label = QtWidgets.QLabel("Stopped")
        self.coord_label = QtWidgets.QLabel("x: -, y: -, val: -")
        row.addWidget(self.status_label)
        row.addSpacing(12)
        row.addWidget(self.coord_label)
        row.addStretch(1)
        return row

    def _set_camera_controls_enabled(self, enabled: bool):
        widgets = [
            self.exposure_spin,
            self.exposure_slider,
            self.exposure_fine_slider,
            self.gain_slider,
            # self.gamma_spin,
        ]
        for widget in widgets:
            widget.setEnabled(enabled)

    def _dispose_camera(self):
        self.stop_live()
        if self.camera:
            try:
                self.camera.disarm()
            except Exception:
                pass
            try:
                self.camera.dispose()
            except Exception:
                pass
            self.camera = None
        if self.cv_thread:
            self.cv_thread.stop()
            self.cv_thread.join(timeout=1.0)
            self.cv_thread = None
        if self.pylon_thread:
            self.pylon_thread.stop()
            self.pylon_thread.join(timeout=1.0)
            self.pylon_thread = None
        self._use_cv = False
        self._use_pylon = False
        self.cv_index = None
        self.pylon_device_info = None
        self._set_camera_controls_enabled(False)
        self.status_label.setText("Stopped")
        self.setWindowTitle("Live View - No camera")

    def _configure_controls_from_camera(self):
        if self._use_cv:
            # Best-effort populate from OpenCV capture if available.
            self._set_camera_controls_enabled(True)
            exp_val = 10_000
            gain_val = 0
            if self.cv_thread:
                exp_us, gain = self.cv_thread.get_props()
                if exp_us:
                    exp_val = int(exp_us)
                if gain is not None:
                    try:
                        gain_val = int(gain)
                    except Exception:
                        gain_val = 0
            self.exposure_spin.setRange(1, 10_000_000)
            self.exposure_slider.setRange(1, 10_000_000)
            self._sync_exposure_controls(exp_val)
            self.gain_slider.setRange(0, 30)
            self._sync_gain_controls(gain_val)
            return
        if self._use_pylon:
            self._set_camera_controls_enabled(True)
            exp_min, exp_max = 1, 10_000_000
            gain_min, gain_max = 0, 30
            exp_val = 10_000
            gain_val = 0
            if self.pylon_thread:
                try:
                    (exp_min, exp_max), (gain_min, gain_max) = self.pylon_thread.get_limits()
                    exp_us, gain = self.pylon_thread.get_props()
                    if exp_us:
                        exp_val = int(exp_us)
                    if gain is not None:
                        gain_val = int(gain)
                except Exception:
                    pass
            self.exposure_spin.setRange(int(exp_min), int(max(exp_max, exp_min + 1)))
            self.exposure_slider.setRange(int(exp_min), int(max(exp_max, exp_min + 1)))
            self._sync_exposure_controls(exp_val)
            self.gain_slider.setRange(int(gain_min), int(max(gain_max, gain_min + 1)))
            self._sync_gain_controls(gain_val)
            return
        if not self.camera:
            return
        exp_min = 1
        exp_max = 10_000_000
        exp_range = getattr(self.camera, "exposure_time_range_us", None)
        if exp_range is not None:
            exp_min = int(getattr(exp_range, "min", exp_min))
            exp_max = int(getattr(exp_range, "max", exp_max))
        self.exposure_spin.setRange(exp_min, exp_max)
        self.exposure_slider.setRange(exp_min, exp_max)
        self._sync_exposure_controls(int(self.camera.exposure_time_us))

        gain_range = getattr(self.camera, "gain_range", None)
        gain_min = getattr(gain_range, "min", 0)
        gain_max = getattr(gain_range, "max", 0)
        self.gain_slider.setRange(int(gain_min), int(gain_max))
        self._sync_gain_controls(int(getattr(self.camera, "gain", gain_min)))

        # self.gamma_spin.setValue(float(getattr(self.camera, "gamma", 1.0)))
        self._set_camera_controls_enabled(True)

    def _connect_first_available_camera(self, show_message: bool = True, force: bool = False) -> bool:
        if force:
            self._dispose_camera()
        elif self.camera or (self._use_cv and self.cv_index is not None) or (self._use_pylon and self.pylon_device_info):
            return True

        # Prefer Thorlabs if available, else try first generic.
        if self.sdk is None:
            try:
                self.sdk = TLCameraSDK()
            except Exception:
                self.sdk = None
        tsi_list = []
        if self.sdk is not None:
            try:
                tsi_list = self.sdk.discover_available_cameras()
            except Exception:
                tsi_list = []

        if tsi_list:
            return self._open_tsi_camera(tsi_list[0], show_message=show_message)

        if _HAVE_PYLON:
            pylon_list = self._discover_pylon_devices()
            if pylon_list:
                return self._open_pylon_camera(pylon_list[0], show_message=show_message)

        if _HAVE_OPENCV:
            generic_indices = self._discover_generic_indices()
            if generic_indices:
                return self._open_generic_camera(generic_indices[0], show_message=show_message)

        if show_message:
            QtWidgets.QMessageBox.information(
                self,
                "No camera",
                "No cameras detected. Connect a camera and try again.",
            )
        self.status_label.setText("No camera detected")
        self.setWindowTitle("Live View - No camera")
        self._set_camera_controls_enabled(False)
        return False

    def _discover_generic_indices(self, max_test: int = 5) -> list[int]:
        indices = []
        if not _HAVE_OPENCV:
            return indices
        max_test = max(max_test, 15)  # broaden search to catch IR/aux streams
        for i in range(max_test):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(i, cv2.CAP_MSMF)
            if cap.isOpened():
                indices.append(i)
            cap.release()
        return indices

    def _discover_pylon_devices(self) -> list:
        if not _HAVE_PYLON:
            return []
        try:
            return list(pylon.TlFactory.GetInstance().EnumerateDevices())
        except Exception:
            return []

    def _open_tsi_camera(self, cam_id: str, show_message: bool = True) -> bool:
        try:
            self.camera = self.sdk.open_camera(cam_id)
            self._use_cv = False
            self._use_pylon = False
            self.cv_index = None
            self.pylon_device_info = None
            self.camera.frames_per_trigger_zero_for_unlimited = 0
            self.camera.image_poll_timeout_ms = 50
            self.setWindowTitle(f"Live View - {self.camera.name}")
            self.status_label.setText(f"Connected: {self.camera.name}")
            self._configure_controls_from_camera()
            return True
        except Exception as exc:
            self.camera = None
            if show_message:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to open camera: {exc}")
            self.status_label.setText("Camera connect failed")
            self.setWindowTitle("Live View - No camera")
            self._set_camera_controls_enabled(False)
            return False

    def _open_generic_camera(self, index: int, show_message: bool = True) -> bool:
        if not _HAVE_OPENCV:
            if show_message:
                QtWidgets.QMessageBox.information(self, "Unavailable", "OpenCV not installed.")
            return False
        self._use_cv = True
        self._use_pylon = False
        self.cv_index = index
        self.pylon_device_info = None
        self.camera = None
        self._set_camera_controls_enabled(False)
        self.setWindowTitle(f"Live View - DirectShow #{index}")
        self.status_label.setText(f"Connected: DirectShow #{index}")
        return True

    def _open_pylon_camera(self, device_info, show_message: bool = True) -> bool:
        if not _HAVE_PYLON:
            if show_message:
                QtWidgets.QMessageBox.information(self, "Unavailable", "Pylon SDK not installed.")
            return False
        self._use_pylon = True
        self._use_cv = False
        self.pylon_device_info = device_info
        self.cv_index = None
        self.camera = None
        name = getattr(device_info, "GetFriendlyName", lambda: "Basler")()
        self.setWindowTitle(f"Live View - {name}")
        self.status_label.setText(f"Connected: {name}")
        self._set_camera_controls_enabled(True)
        return True

    def _ensure_camera(self, show_message: bool = True) -> bool:
        return self._connect_first_available_camera(show_message=show_message)

    def connect_camera(self):
        self._show_camera_selection_dialog()

    def _show_camera_selection_dialog(self):
        # Discover available sources
        tsi_list = []
        if self.sdk is None:
            try:
                self.sdk = TLCameraSDK()
            except Exception:
                self.sdk = None
        if self.sdk is not None:
            try:
                tsi_list = self.sdk.discover_available_cameras()
            except Exception:
                tsi_list = []
        generic_indices = self._discover_generic_indices() if _HAVE_OPENCV else []
        pylon_list = self._discover_pylon_devices() if _HAVE_PYLON else []

        options = []
        for cam_id in tsi_list:
            options.append(("tsi", cam_id, f"Thorlabs: {cam_id}"))
        for idx in generic_indices:
            options.append(("cv", idx, f"DirectShow #{idx}"))
        for dev in pylon_list:
            try:
                friendly = dev.GetFriendlyName()
            except Exception:
                friendly = "Basler Pylon"
            options.append(("pylon", dev, f"Basler: {friendly}"))

        if not options:
            QtWidgets.QMessageBox.information(self, "No cameras", "No cameras detected.")
            return

        items = [opt[2] for opt in options]
        item, ok = QtWidgets.QInputDialog.getItem(
            self, "Select Camera", "Choose a camera:", items, 0, False
        )
        if not ok or not item:
            return
        choice = options[items.index(item)]

        # Apply selection
        self._dispose_camera()
        if choice[0] == "tsi":
            self._open_tsi_camera(choice[1], show_message=True)
        elif choice[0] == "cv":
            self._open_generic_camera(choice[1], show_message=True)
        elif choice[0] == "pylon":
            self._open_pylon_camera(choice[1], show_message=True)

    def start_live(self):
        if self._live:
            return
        if not self._ensure_camera():
            return
        self.status_label.setText("Starting...")
        if self._use_pylon:
            if not _HAVE_PYLON or self.pylon_device_info is None:
                QtWidgets.QMessageBox.critical(self, "Error", "Pylon SDK not available or device missing.")
                self.status_label.setText("Failed to start")
                return
            try:
                self.pylon_thread = PylonVideoThread(self.pylon_device_info)
                self.acq_thread = self.pylon_thread
                self.pylon_thread.start()
                self._live = True
                name = getattr(self.pylon_device_info, "GetFriendlyName", lambda: "Basler")( )
                self.status_label.setText(f"Live (Basler {name})")
                try:
                    self._configure_controls_from_camera()
                except Exception:
                    pass
                return
            except Exception as exc:
                self.pylon_thread = None
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to start Pylon stream: {exc}")
                self.status_label.setText("Failed to start")
                return
        if self._use_cv:
            if not _HAVE_OPENCV:
                QtWidgets.QMessageBox.critical(self, "Error", "OpenCV not installed.")
                self.status_label.setText("Failed to start")
                return
            if self.cv_index is None:
                QtWidgets.QMessageBox.critical(self, "Error", "No generic camera index selected.")
                self.status_label.setText("Failed to start")
                return
            # Quick sanity check that the capture can be opened before spinning the thread.
            test_cap = cv2.VideoCapture(self.cv_index, cv2.CAP_DSHOW) if _HAVE_OPENCV else None
            if not (test_cap and test_cap.isOpened()):
                if test_cap:
                    test_cap.release()
                QtWidgets.QMessageBox.critical(self, "Error", f"Could not open DirectShow #{self.cv_index}.")
                self.status_label.setText("Failed to start")
                return
            test_cap.release()
            self.cv_thread = GenericVideoThread(self.cv_index)
            self.acq_thread = self.cv_thread
            self.cv_thread.start()
            self._live = True
            self.status_label.setText(f"Live (DirectShow #{self.cv_index})")
            return

        try:
            self.camera.arm(2)
            self.camera.issue_software_trigger()
            self.acq_thread = ImageAcquisitionThread(self.camera)
            self.acq_thread.start()
            self._live = True
            self.status_label.setText("Live")
        except Exception as exc:
            self._live = False
            self.acq_thread = None
            try:
                if self.camera:
                    self.camera.disarm()
            except Exception:
                pass
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to start live view: {exc}")
            self.status_label.setText("Failed to start")

    def stop_live(self):
        self._live = False
        if self.acq_thread:
            self.acq_thread.stop()
            self.acq_thread.join(timeout=1.0)
            self.acq_thread = None
        if self.cv_thread:
            self.cv_thread = None
        if getattr(self, "pylon_thread", None):
            self.pylon_thread = None
        if self.camera and not self._use_cv:
            try:
                self.camera.disarm()
            except Exception:
                pass
        self.status_label.setText("Stopped")

    def set_exposure(self):
        if not self._ensure_camera():
            return
        try:
            if self._use_cv:
                val = int(self.exposure_spin.value())
                if self.cv_thread and self.cv_thread.set_exposure_us(val):
                    self._sync_exposure_controls(val)
                else:
                    QtWidgets.QMessageBox.information(
                        self, "Not supported", "Exposure control not supported for this camera."
                    )
                return
            if self._use_pylon:
                val = int(self.exposure_spin.value())
                if self.pylon_thread and self.pylon_thread.set_exposure_us(val):
                    self._sync_exposure_controls(val)
                else:
                    QtWidgets.QMessageBox.information(
                        self, "Not supported", "Exposure control not supported for this camera."
                    )
                return
            self._apply_exposure(int(self.exposure_spin.value()))
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to set exposure: {exc}")

    def set_gain(self, value=None):
        if not self._ensure_camera():
            return
        try:
            val = int(value) if value is not None else int(self.gain_slider.value())
            if self._use_cv:
                if self.cv_thread and self.cv_thread.set_gain(val):
                    self._sync_gain_controls(val)
                else:
                    QtWidgets.QMessageBox.information(
                        self, "Not supported", "Gain control not supported for this camera."
                    )
                return
            if self._use_pylon:
                if self.pylon_thread and self.pylon_thread.set_gain(val):
                    self._sync_gain_controls(val)
                else:
                    QtWidgets.QMessageBox.information(
                        self, "Not supported", "Gain control not supported for this camera."
                    )
                return
            self._apply_gain(val)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to set gain: {exc}")

    # def set_gamma(self):
    #     if not self._ensure_camera():
    #         return
    #     if not hasattr(self.camera, "gamma"):
    #         QtWidgets.QMessageBox.information(self, "Not supported", "Gamma control not supported on this camera.")
    #         return
    #     try:
    #         self.camera.gamma = float(self.gamma_spin.value())
    #     except Exception as exc:
    #         QtWidgets.QMessageBox.critical(self, "Error", f"Failed to set gamma: {exc}")

    def _update_mm_scale(self, value: float, axis: str):
        if axis == "x":
            self.mm_per_px_x = float(value)
        elif axis == "y":
            self.mm_per_px_y = float(value)
        self._update_calibration_status()
        self._update_measure_status()
        # refresh hover text if desired; no explicit refresh needed for display

    def _poll_queue(self):
        if self.acq_thread:
            q = self.acq_thread.get_queue()
            try:
                payload: FramePayload = q.get_nowait()
                self._display_frame(payload)
            except queue.Empty:
                pass

    def _on_exposure_slider(self, value: int):
        if not (self.camera or self._use_cv or self._use_pylon):
            return
        self.exposure_spin.blockSignals(True)
        self.exposure_spin.setValue(float(value))
        self.exposure_spin.blockSignals(False)
        try:
            self._apply_exposure(value)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to set exposure: {exc}")

    def _on_exposure_slider_fine(self, value: int):
        if not (self.camera or self._use_cv or self._use_pylon):
            return
        self._on_exposure_slider(value)

    def _apply_exposure(self, value: int):
        if self._use_cv:
            if self.cv_thread and self.cv_thread.set_exposure_us(int(value)):
                self._sync_exposure_controls(int(value))
                return
            raise RuntimeError("Exposure control not supported on this camera")
        if self._use_pylon:
            if self.pylon_thread and self.pylon_thread.set_exposure_us(int(value)):
                self._sync_exposure_controls(int(value))
                return
            raise RuntimeError("Exposure control not supported on this camera")
        if not self.camera:
            raise RuntimeError("No camera connected")
        self.camera.exposure_time_us = int(value)
        actual = int(self.camera.exposure_time_us)
        self._sync_exposure_controls(actual)

    def _sync_exposure_controls(self, value: int):
        self.exposure_slider.blockSignals(True)
        self.exposure_slider.setValue(int(value))
        self.exposure_slider.blockSignals(False)
        self.exposure_spin.blockSignals(True)
        self.exposure_spin.setValue(float(value))
        self.exposure_spin.blockSignals(False)
        if getattr(self, "exposure_fine_slider", None):
            self.exposure_fine_slider.blockSignals(True)
            fine_min, fine_max = self.exposure_fine_slider.minimum(), self.exposure_fine_slider.maximum()
            self.exposure_fine_slider.setValue(int(min(max(value, fine_min), fine_max)))
            self.exposure_fine_slider.blockSignals(False)
        self.exposure_value_label.setText(f"{int(value)} us")

    def _on_gain_slider(self, value: int):
        if not (self.camera or self._use_cv or self._use_pylon):
            return
        try:
            self._apply_gain(value)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to set gain: {exc}")

    def _apply_gain(self, value: int):
        if self._use_cv:
            if self.cv_thread and self.cv_thread.set_gain(int(value)):
                self._sync_gain_controls(int(value))
                return
            raise RuntimeError("Gain control not supported on this camera")
        if self._use_pylon:
            if self.pylon_thread and self.pylon_thread.set_gain(int(value)):
                self._sync_gain_controls(int(value))
                return
            raise RuntimeError("Gain control not supported on this camera")
        if not self.camera:
            raise RuntimeError("No camera connected")
        self.camera.gain = int(value)
        actual = int(getattr(self.camera, "gain", value))
        self._sync_gain_controls(actual)

    def _sync_gain_controls(self, value: int):
        self.gain_slider.blockSignals(True)
        self.gain_slider.setValue(int(value))
        self.gain_slider.blockSignals(False)
        self.gain_value_label.setText(str(int(value)))

    def _display_frame(self, payload: FramePayload):
        self.last_payload = payload
        self._filtered_np_image = self._apply_image_filter_np(payload.np_image)
        self._refresh_image_view()

    def _refresh_image_view(self):
        if not self.last_payload:
            self.image_label.clear()
            self._last_render_scale = 1.0
            self._update_histogram_window(None)
            return
        img = self._prepare_display_image(self.last_payload)
        if self.cross_pos:
            img = self._draw_cross(img, self.cross_pos)
        pixmap, scale = self._render_pixmap(img)
        self._last_render_scale = scale
        self.image_label.setPixmap(pixmap)
        self.image_label.adjustSize()
        self._update_histogram_window(self.last_payload)

    def _render_pixmap(self, image: Image.Image) -> tuple[QtGui.QPixmap, float]:
        base_pixmap = self._pil_to_qpixmap(image)
        target_size = self._compute_target_size(base_pixmap)
        scaled = base_pixmap.scaled(
            target_size,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        scale = scaled.width() / base_pixmap.width() if base_pixmap.width() else 1.0
        return scaled, scale

    def _compute_target_size(self, base_pixmap: QtGui.QPixmap) -> QtCore.QSize:
        if self._fit_to_window:
            available = self._get_available_image_size()
            if not available.isValid() or available.width() == 0 or available.height() == 0:
                return base_pixmap.size()
            return available
        zoomed_w = int(base_pixmap.width() * self._zoom)
        zoomed_h = int(base_pixmap.height() * self._zoom)
        return QtCore.QSize(max(1, zoomed_w), max(1, zoomed_h))

    def _get_available_image_size(self) -> QtCore.QSize:
        if getattr(self, "scroll_area", None):
            viewport = self.scroll_area.viewport()
            if viewport and viewport.size().isValid():
                return viewport.size()
        parent = self.image_label.parentWidget()
        if parent and parent.size().isValid():
            return parent.size()
        return self.image_label.size() if self.image_label.size().isValid() else QtCore.QSize(640, 480)

    def zoom_in(self):
        self._fit_to_window = False
        self._zoom = min(self._zoom * 1.25, 20.0)
        self._sync_panning_enabled()
        self._refresh_image_view()

    def zoom_out(self):
        self._fit_to_window = False
        self._zoom = max(self._zoom / 1.25, 0.05)
        self._sync_panning_enabled()
        self._refresh_image_view()

    def fit_to_window(self):
        self._fit_to_window = True
        self._zoom = 1.0
        self._sync_panning_enabled()
        self._refresh_image_view()
        self._reset_pan()

    def toggle_grayscale(self, state: int):
        try:
            checked = QtCore.Qt.CheckState(state) == QtCore.Qt.CheckState.Checked
        except Exception:
            checked = bool(state)
        self._force_grayscale = checked
        self._refresh_image_view()

    def toggle_histogram(self, state: int):
        try:
            checked = QtCore.Qt.CheckState(state) == QtCore.Qt.CheckState.Checked
        except Exception:
            checked = bool(state)
        self._histogram_enabled = checked
        self._update_histogram_window(self.last_payload if self._histogram_enabled else None)

    def toggle_fit_window(self, state: int):
        try:
            checked = QtCore.Qt.CheckState(state) == QtCore.Qt.CheckState.Checked
        except Exception:
            checked = bool(state)
        if not getattr(self, "fit_window", None):
            return
        if checked:
            if not self._fit_window_positioned:
                anchor_geom = None
                if getattr(self, "controls_window", None):
                    ctrl_geom = self.controls_window.frameGeometry()
                    if ctrl_geom.isValid():
                        anchor_geom = ctrl_geom
                self._position_window_near_main(
                    self.fit_window, QtCore.QPoint(18, 220 if anchor_geom else 120), anchor_geom=anchor_geom
                )
                self._fit_window_positioned = True
            if not self.fit_window.isVisible():
                self.fit_window.show()
            self.fit_window.raise_()
        else:
            self.fit_window.hide()

    def toggle_measure_window(self, state: int):
        try:
            checked = QtCore.Qt.CheckState(state) == QtCore.Qt.CheckState.Checked
        except Exception:
            checked = bool(state)
        if not getattr(self, "calibration_window", None):
            return
        if checked:
            if not self._calibration_window_positioned:
                anchor_geom = None
                if getattr(self, "controls_window", None):
                    ctrl_geom = self.controls_window.frameGeometry()
                    if ctrl_geom.isValid():
                        anchor_geom = ctrl_geom
                self._position_window_near_main(
                    self.calibration_window,
                    QtCore.QPoint(18, 420 if anchor_geom else 180),
                    anchor_geom=anchor_geom,
                )
                self._calibration_window_positioned = True
            if not self.calibration_window.isVisible():
                self.calibration_window.show()
            self.calibration_window.raise_()
        else:
            self.calibration_window.hide()
            self.clear_calibration()
            self._clear_measure_tools()
        self._sync_panning_enabled()

    def _start_calibration(self, axis: str):
        if axis not in ("x", "y", "both"):
            return
        if not self.last_payload:
            QtWidgets.QMessageBox.information(self, "No image", "Capture or start live view before calibrating.")
            return
        if getattr(self, "measure_checkbox", None):
            self.measure_checkbox.setChecked(True)
        self._calibration_axis = axis
        self._calibration_last_axis = axis
        self._calibration_points = []
        self._calibration_dragging = False
        self._calibration_drag_mode = None
        self._calibration_drag_start = None
        self._calibration_drag_start_points = []
        self._line_select_mode = False
        self._line_edit_mode = False
        if getattr(self, "calibration_window", None):
            self.calibration_window.show()
            self.calibration_window.raise_()
        self._update_calibration_status()
        self._sync_panning_enabled()
        self._refresh_image_view()

    def _calibration_pixel_length(self) -> Optional[float]:
        if len(self._calibration_points) != 2:
            return None
        axis = self._calibration_axis or self._calibration_last_axis
        if not axis:
            return None
        p1, p2 = self._calibration_points
        if axis == "x":
            return float(abs(p2[0] - p1[0]))
        if axis == "y":
            return float(abs(p2[1] - p1[1]))
        if axis == "both":
            return float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
        return None

    def _finish_calibration(self):
        axis = self._calibration_axis or self._calibration_last_axis
        if axis not in ("x", "y", "both") or len(self._calibration_points) != 2:
            QtWidgets.QMessageBox.information(
                self, "Calibration", "Select two points for Calib X, Calib Y, or Calib Both first."
            )
            return
        p1, p2 = self._calibration_points
        dx_px = float(abs(p2[0] - p1[0]))
        dy_px = float(abs(p2[1] - p1[1]))
        px_len = self._calibration_pixel_length()
        if not px_len or px_len <= 0:
            QtWidgets.QMessageBox.information(self, "Calibration", "Span length is zero; pick two distinct points.")
            return
        if axis == "both" and (dx_px <= 0 or dy_px <= 0):
            QtWidgets.QMessageBox.information(
                self, "Calibration", "Calib Both needs a line with both X and Y components (not perfectly straight)."
            )
            return

        if axis == "both":
            current_mm_guess = float(np.hypot(dx_px * self.mm_per_px_x, dy_px * self.mm_per_px_y))
            length_mm, ok = QtWidgets.QInputDialog.getDouble(
                self,
                "Enter diagonal length",
                "Measured hypotenuse length (mm):",
                float(current_mm_guess),
                0.000001,
                1e9,
                decimals=6,
            )
            if not ok:
                return
            if length_mm <= 0:
                QtWidgets.QMessageBox.information(self, "Calibration", "Length must be positive.")
                return
            angle_rad = float(np.arctan2(dy_px, dx_px))
            dx_mm = float(np.cos(angle_rad) * length_mm)
            dy_mm = float(np.sin(angle_rad) * length_mm)
            mm_per_px_x = dx_mm / dx_px if dx_px > 0 else None
            mm_per_px_y = dy_mm / dy_px if dy_px > 0 else None
            if not mm_per_px_x or not mm_per_px_y or mm_per_px_x <= 0 or mm_per_px_y <= 0:
                QtWidgets.QMessageBox.information(self, "Calibration", "Calculated mm/px values are invalid.")
                return
            self.mm_px_x_spin.setValue(mm_per_px_x)
            self.mm_px_y_spin.setValue(mm_per_px_y)
            msg_text = (
                f"BOTH calibrated: X {mm_per_px_x:.6f} mm/px, Y {mm_per_px_y:.6f} mm/px (hyp {px_len:.2f} px)."
            )
        else:
            current_mm_per_px = self.mm_per_px_x if axis == "x" else self.mm_per_px_y
            suggested_mm = px_len * current_mm_per_px
            length_mm, ok = QtWidgets.QInputDialog.getDouble(
                self,
                "Enter real length",
                f"Measured length along {axis.upper()} (mm):",
                float(suggested_mm),
                0.000001,
                1e9,
                decimals=6,
            )
            if not ok:
                return
            if length_mm <= 0:
                QtWidgets.QMessageBox.information(self, "Calibration", "Length must be positive.")
                return
            mm_per_px = float(length_mm) / float(px_len)
            if mm_per_px <= 0:
                QtWidgets.QMessageBox.information(self, "Calibration", "Calculated mm/px is invalid.")
                return
            if axis == "x":
                self.mm_px_x_spin.setValue(mm_per_px)
            elif axis == "y":
                self.mm_px_y_spin.setValue(mm_per_px)
            msg_text = f"{axis.upper()} calibrated: {mm_per_px:.6f} mm/px (span {px_len:.2f} px)."
        self._calibration_axis = None
        self._calibration_dragging = False
        self._calibration_drag_mode = None
        self._calibration_drag_start = None
        self._update_calibration_status()
        self._sync_panning_enabled()
        QtWidgets.QMessageBox.information(self, "Calibration", msg_text)
        self._refresh_image_view()

    def clear_calibration(self):
        self._calibration_axis = None
        self._calibration_last_axis = None
        self._calibration_points = []
        self._calibration_dragging = False
        self._calibration_drag_mode = None
        self._calibration_drag_start = None
        self._calibration_drag_start_points = []
        self._update_calibration_status()
        self._sync_panning_enabled()
        self._refresh_image_view()

    def _update_calibration_status(self):
        label = getattr(self, "calibration_status_label", None)
        if not label:
            return
        axis = self._calibration_axis
        px_len = self._calibration_pixel_length()
        if axis:
            if len(self._calibration_points) == 0:
                text = f"{axis.upper()} calibration: click first point on the live view."
            elif len(self._calibration_points) == 1:
                text = f"{axis.upper()} calibration: click second point."
            else:
                if axis == "both":
                    p1, p2 = self._calibration_points
                    dx_px = float(abs(p2[0] - p1[0]))
                    dy_px = float(abs(p2[1] - p1[1]))
                    approx_mm = float(np.hypot(dx_px * self.mm_per_px_x, dy_px * self.mm_per_px_y))
                    text = (
                        "BOTH calibration: drag points to adjust, then press Done. "
                        f"Hypotenuse {px_len:.1f} px (~{approx_mm:.3f} mm)."
                    )
                else:
                    mm_per_px = self.mm_per_px_x if axis == "x" else self.mm_per_px_y
                    approx_mm = px_len * mm_per_px if px_len is not None else 0.0
                    text = (
                        f"{axis.upper()} calibration: drag points to adjust, then press Done. "
                        f"Span {px_len:.1f} px (~{approx_mm:.3f} mm)."
                    )
        elif len(self._calibration_points) == 2 and self._calibration_last_axis:
            axis = self._calibration_last_axis
            if axis == "both":
                p1, p2 = self._calibration_points
                dx_px = float(abs(p2[0] - p1[0]))
                dy_px = float(abs(p2[1] - p1[1]))
                approx_mm = float(np.hypot(dx_px * self.mm_per_px_x, dy_px * self.mm_per_px_y))
                text = (
                    f"Last BOTH span: {px_len:.1f} px (~{approx_mm:.3f} mm). "
                    "Press Calib X/Y/Both to adjust again."
                )
            else:
                mm_per_px = self.mm_per_px_x if axis == "x" else self.mm_per_px_y
                approx_mm = px_len * mm_per_px if px_len is not None else 0.0
                text = (
                    f"Last {axis.upper()} span: {px_len:.1f} px (~{approx_mm:.3f} mm). "
                    "Press Calib X/Y/Both to adjust again."
                )
        else:
            text = "Idle. Choose Calib X, Calib Y, or Calib Both to begin."
        label.setText(text)

    def _start_measure_line(self):
        if not self.last_payload:
            QtWidgets.QMessageBox.information(self, "No image", "Capture or start live view before measuring.")
            return
        if getattr(self, "measure_checkbox", None):
            self.measure_checkbox.setChecked(True)
        self._measure_active_mode = "line"
        self._measure_line_points = []
        self._measure_line_dragging = False
        self._measure_line_drag_mode = None
        self._measure_line_drag_start = None
        self._measure_line_drag_start_points = []
        self._measure_arc_points = []
        self._measure_arc_dragging = False
        self._measure_arc_drag_index = None
        self._calibration_axis = None  # avoid eating clicks
        self._line_select_mode = False
        self._line_edit_mode = False
        self._update_calibration_status()
        self._update_measure_status()
        self._sync_panning_enabled()
        self._refresh_image_view()

    def _start_measure_arc(self):
        if not self.last_payload:
            QtWidgets.QMessageBox.information(self, "No image", "Capture or start live view before measuring.")
            return
        if getattr(self, "measure_checkbox", None):
            self.measure_checkbox.setChecked(True)
        self._measure_active_mode = "arc"
        self._measure_arc_points = []
        self._measure_arc_dragging = False
        self._measure_arc_drag_index = None
        self._measure_line_points = []
        self._measure_line_dragging = False
        self._measure_line_drag_mode = None
        self._measure_line_drag_start = None
        self._measure_line_drag_start_points = []
        self._calibration_axis = None
        self._line_select_mode = False
        self._line_edit_mode = False
        self._update_calibration_status()
        self._update_measure_status()
        self._sync_panning_enabled()
        self._refresh_image_view()

    def _clear_measure_tools(self):
        self._measure_active_mode = None
        self._measure_line_points = []
        self._measure_line_dragging = False
        self._measure_line_drag_mode = None
        self._measure_line_drag_start = None
        self._measure_line_drag_start_points = []
        self._measure_arc_points = []
        self._measure_arc_dragging = False
        self._measure_arc_drag_index = None
        self._update_measure_status()
        self._sync_panning_enabled()
        self._refresh_image_view()

    def _measure_line_lengths(self) -> Optional[tuple[float, float]]:
        if len(self._measure_line_points) != 2:
            return None
        p1, p2 = self._measure_line_points
        dx_px = float(p2[0] - p1[0])
        dy_px = float(p2[1] - p1[1])
        length_px = float(np.hypot(dx_px, dy_px))
        length_mm = float(np.hypot(dx_px * self.mm_per_px_x, dy_px * self.mm_per_px_y))
        return length_px, length_mm

    def _measure_arc_geometry(self) -> Optional[dict]:
        if len(self._measure_arc_points) != 3:
            return None
        (x1, y1), (x2, y2), (x3, y3) = [tuple(map(float, pt)) for pt in self._measure_arc_points]
        d_px = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
        if abs(d_px) < 1e-6:
            return None
        ux_px = (
            (x1 * x1 + y1 * y1) * (y2 - y3)
            + (x2 * x2 + y2 * y2) * (y3 - y1)
            + (x3 * x3 + y3 * y3) * (y1 - y2)
        ) / d_px
        uy_px = (
            (x1 * x1 + y1 * y1) * (x3 - x2)
            + (x2 * x2 + y2 * y2) * (x1 - x3)
            + (x3 * x3 + y3 * y3) * (x2 - x1)
        ) / d_px
        radius_px = float(np.hypot(x1 - ux_px, y1 - uy_px))

        # Compute radius in mm using scaled coordinates.
        radius_mm = None
        x1_mm, y1_mm = x1 * self.mm_per_px_x, y1 * self.mm_per_px_y
        x2_mm, y2_mm = x2 * self.mm_per_px_x, y2 * self.mm_per_px_y
        x3_mm, y3_mm = x3 * self.mm_per_px_x, y3 * self.mm_per_px_y
        d_mm = 2.0 * (x1_mm * (y2_mm - y3_mm) + x2_mm * (y3_mm - y1_mm) + x3_mm * (y1_mm - y2_mm))
        if abs(d_mm) >= 1e-9:
            ux_mm = (
                (x1_mm * x1_mm + y1_mm * y1_mm) * (y2_mm - y3_mm)
                + (x2_mm * x2_mm + y2_mm * y2_mm) * (y3_mm - y1_mm)
                + (x3_mm * x3_mm + y3_mm * y3_mm) * (y1_mm - y2_mm)
            ) / d_mm
            uy_mm = (
                (x1_mm * x1_mm + y1_mm * y1_mm) * (x3_mm - x2_mm)
                + (x2_mm * x2_mm + y2_mm * y2_mm) * (x1_mm - x3_mm)
                + (x3_mm * x3_mm + y3_mm * y3_mm) * (x2_mm - x1_mm)
            ) / d_mm
            radius_mm = float(np.hypot(x1_mm - ux_mm, y1_mm - uy_mm))
        return {
            "center_px": (ux_px, uy_px),
            "radius_px": radius_px,
            "radius_mm": radius_mm,
        }

    def _update_measure_status(self):
        label = getattr(self, "measure_status_label", None)
        if not label:
            return
        mode = self._measure_active_mode
        text = "Measurement: Idle. Pick Line or Arc."
        if mode == "line":
            if len(self._measure_line_points) == 0:
                text = "Line: click first point."
            elif len(self._measure_line_points) == 1:
                text = "Line: click second point."
            elif len(self._measure_line_points) == 2:
                lengths = self._measure_line_lengths()
                if lengths:
                    px, mm = lengths
                    text = f"Line length: {px:.1f} px ({mm:.3f} mm). Drag endpoints to adjust."
        elif mode == "arc":
            if len(self._measure_arc_points) == 0:
                text = "Arc: click first point."
            elif len(self._measure_arc_points) == 1:
                text = "Arc: click second point."
            elif len(self._measure_arc_points) == 2:
                text = "Arc: click third point to define circle."
            elif len(self._measure_arc_points) == 3:
                geom = self._measure_arc_geometry()
                if geom:
                    radius_mm = geom["radius_mm"]
                    mm_text = f"{radius_mm:.3f} mm" if radius_mm is not None else "mm: N/A"
                    text = f"Arc radius: {geom['radius_px']:.1f} px ({mm_text}). Drag points to adjust."
                else:
                    text = "Arc: points are collinear; adjust positions."
        else:
            # Show last values if any even when idle
            lengths = self._measure_line_lengths()
            geom = self._measure_arc_geometry()
            if lengths:
                px, mm = lengths
                text = f"Last line: {px:.1f} px ({mm:.3f} mm)."
            elif geom:
                radius_mm = geom["radius_mm"]
                mm_text = f"{radius_mm:.3f} mm" if radius_mm is not None else "mm: N/A"
                text = f"Last arc radius: {geom['radius_px']:.1f} px ({mm_text})."
        label.setText(text)

    def _update_histogram_window(self, payload: Optional[FramePayload]):
        if not getattr(self, "hist_window", None):
            return
        if self._in_hist_update:
            return
        if not self._histogram_enabled or payload is None:
            self.hist_window.hide()
            return
        self._in_hist_update = True
        try:
            pixmap = self._build_histogram_pixmap(payload)
            self.hist_label.setPixmap(pixmap)
            self.hist_label.adjustSize()
            self.hist_window.adjustSize()
            self._hist_size = self.hist_window.size()
            if not self._hist_window_positioned:
                anchor_geom = None
                if getattr(self, "controls_window", None):
                    ctrl_geom = self.controls_window.frameGeometry()
                    if ctrl_geom.isValid():
                        anchor_geom = ctrl_geom
                offset = QtCore.QPoint(18, 0 if anchor_geom else 40)
                self._position_window_near_main(self.hist_window, offset, anchor_geom=anchor_geom)
                self._hist_window_positioned = True
            if not self.hist_window.isVisible():
                self.hist_window.show()
            self.hist_window.raise_()
        finally:
            self._in_hist_update = False

    def _build_histogram_pixmap(self, payload: FramePayload) -> QtGui.QPixmap:
        data = self._filtered_np_image if self._filtered_np_image is not None else payload.np_image
        if data.ndim == 3:
            if self._force_grayscale:
                data = np.mean(data, axis=2)
            else:
                data = np.mean(data, axis=2)
        hist, _ = np.histogram(data.flatten(), bins=256, range=[0, 255])
        hist = hist.astype(np.float64)
        if hist.max() > 0:
            hist = hist / hist.max()

        width, height = max(120, self._hist_size.width()), max(80, self._hist_size.height())
        image = QtGui.QImage(width, height, QtGui.QImage.Format.Format_ARGB32)
        image.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(image)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        bg_color = QtGui.QColor(10, 10, 10, 180)  # translucent dark backdrop for contrast
        painter.fillRect(0, 0, width, height, bg_color)
        bar_color = QtGui.QColor(0, 190, 255, 220)  # semi-transparent cyan bars
        painter.setPen(QtGui.QPen(bar_color, 1))
        painter.setBrush(bar_color)
        for i, v in enumerate(hist):
            x = i
            bar_height = int(v * (height - 12))
            painter.drawRect(x, height - bar_height - 2, 1, bar_height)
        painter.end()
        return QtGui.QPixmap.fromImage(image)

    def _update_line_plot(
        self,
        x_axis: np.ndarray,
        profile: np.ndarray,
        fit_curve: np.ndarray,
        length_mm: float,
        fwhm_mm: float,
        waist_mm: float,
        raw_profile: Optional[np.ndarray] = None,
    ):
        width = 540
        height = 240
        margin = 36
        if x_axis.size == 0:
            self.line_plot_label.clear()
            return
        y_values = [profile, fit_curve]
        if raw_profile is not None:
            y_values.append(raw_profile)
        y_min = float(min(np.min(arr) for arr in y_values))
        y_max = float(max(np.max(arr) for arr in y_values))
        if y_max - y_min < 1e-6:
            y_max = y_min + 1.0
        def scale_x(x):
            return margin + (x - x_axis[0]) / max(1e-9, x_axis[-1] - x_axis[0]) * (width - 2 * margin)
        def scale_y(y):
            return height - margin - (y - y_min) / (y_max - y_min) * (height - 2 * margin)

        img = QtGui.QImage(width, height, QtGui.QImage.Format.Format_ARGB32)
        img.fill(QtGui.QColor(12, 12, 12, 255))
        painter = QtGui.QPainter(img)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(QtGui.QPen(QtGui.QColor(70, 70, 70), 1))
        painter.drawRect(margin, margin, width - 2 * margin, height - 2 * margin)

        # Optional raw profile
        if raw_profile is not None:
            painter.setPen(QtGui.QPen(QtGui.QColor(140, 140, 140), 1, QtCore.Qt.PenStyle.DashLine))
            for i in range(1, len(x_axis)):
                painter.drawLine(
                    QtCore.QPointF(scale_x(x_axis[i - 1]), scale_y(raw_profile[i - 1])),
                    QtCore.QPointF(scale_x(x_axis[i]), scale_y(raw_profile[i])),
                )

        # Filtered profile line
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 200, 255), 2))
        for i in range(1, len(x_axis)):
            painter.drawLine(
                QtCore.QPointF(scale_x(x_axis[i - 1]), scale_y(profile[i - 1])),
                QtCore.QPointF(scale_x(x_axis[i]), scale_y(profile[i])),
            )

        # Fit line
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 150, 0), 2))
        for i in range(1, len(x_axis)):
            painter.drawLine(
                QtCore.QPointF(scale_x(x_axis[i - 1]), scale_y(fit_curve[i - 1])),
                QtCore.QPointF(scale_x(x_axis[i]), scale_y(fit_curve[i])),
            )

        painter.setPen(QtGui.QPen(QtGui.QColor(180, 180, 180), 1))
        waist_diam_mm = waist_mm * 2.0
        painter.drawText(
            margin,
            margin - 6,
            f"Line profile (cyan) & fit (orange) | 2*w={waist_diam_mm:.3f}mm, FWHM={fwhm_mm:.3f}mm",
        )
        painter.drawText(
            width // 2 - 60,
            height - 6,
            f"Distance: 0-{x_axis[-1]:.1f}px ({max(length_mm,0):.3f}mm)",
        )
        painter.save()
        painter.translate(12, height // 2 + 40)
        painter.rotate(-90)
        painter.drawText(0, 0, "Intensity (gray level)")
        painter.restore()
        painter.end()
        self.line_plot_label.setPixmap(QtGui.QPixmap.fromImage(img))

    def _update_axis_plot(
        self,
        target: QtWidgets.QLabel,
        title: str,
        x_axis: np.ndarray,
        profile: np.ndarray,
        fit_curve: np.ndarray,
        length_mm: float,
        fwhm_mm: float,
        waist_mm: float,
    ):
        width = 420
        height = 160
        margin = 32
        if x_axis.size == 0:
            target.clear()
            return
        y_min = float(min(np.min(profile), np.min(fit_curve)))
        y_max = float(max(np.max(profile), np.max(fit_curve)))
        if y_max - y_min < 1e-6:
            y_max = y_min + 1.0
        def scale_x(x):
            return margin + (x - x_axis[0]) / max(1e-9, x_axis[-1] - x_axis[0]) * (width - 2 * margin)
        def scale_y(y):
            return height - margin - (y - y_min) / (y_max - y_min) * (height - 2 * margin)

        img = QtGui.QImage(width, height, QtGui.QImage.Format.Format_ARGB32)
        img.fill(QtGui.QColor(12, 12, 12, 255))
        painter = QtGui.QPainter(img)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(QtGui.QPen(QtGui.QColor(70, 70, 70), 1))
        painter.drawRect(margin, margin, width - 2 * margin, height - 2 * margin)

        painter.setPen(QtGui.QPen(QtGui.QColor(0, 200, 255), 2))
        for i in range(1, len(x_axis)):
            painter.drawLine(
                QtCore.QPointF(scale_x(x_axis[i - 1]), scale_y(profile[i - 1])),
                QtCore.QPointF(scale_x(x_axis[i]), scale_y(profile[i])),
            )
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 150, 0), 2))
        for i in range(1, len(x_axis)):
            painter.drawLine(
                QtCore.QPointF(scale_x(x_axis[i - 1]), scale_y(fit_curve[i - 1])),
                QtCore.QPointF(scale_x(x_axis[i]), scale_y(fit_curve[i])),
            )
        painter.setPen(QtGui.QPen(QtGui.QColor(180, 180, 180), 1))
        waist_diam_mm = waist_mm * 2.0
        painter.drawText(margin, margin - 6, f"{title} | 2*w={waist_diam_mm:.3f}mm, FWHM={fwhm_mm:.3f}mm")
        painter.drawText(
            width // 2 - 70,
            height - 6,
            f"0-{x_axis[-1]:.1f}px ({max(length_mm,0):.3f}mm)",
        )
        painter.save()
        painter.translate(12, height // 2 + 28)
        painter.rotate(-90)
        painter.drawText(0, 0, "Intensity (gray level)")
        painter.restore()
        painter.end()
        target.setPixmap(QtGui.QPixmap.fromImage(img))

    def _start_calibration_drag(self, event: QtGui.QMouseEvent) -> bool:
        axis = self._calibration_axis or self._calibration_last_axis
        if axis is None or len(self._calibration_points) != 2:
            return False
        pos = event.position()
        scale = self._last_render_scale or 1.0
        p_img = QtCore.QPointF(pos.x() / scale, pos.y() / scale)
        p1_screen = self._screen_point_for_line(self._calibration_points[0])
        p2_screen = self._screen_point_for_line(self._calibration_points[1])
        hit_radius = 18.0
        dist_p1 = float(np.hypot(pos.x() - p1_screen.x(), pos.y() - p1_screen.y()))
        dist_p2 = float(np.hypot(pos.x() - p2_screen.x(), pos.y() - p2_screen.y()))
        mode = None
        if dist_p1 <= hit_radius:
            mode = "p1"
        elif dist_p2 <= hit_radius:
            mode = "p2"
        else:
            dist_seg = self._distance_to_segment(pos, p1_screen, p2_screen)
            if dist_seg <= hit_radius:
                mode = "move"
        if mode:
            self._calibration_dragging = True
            self._calibration_drag_mode = mode
            self._calibration_drag_start = p_img
            self._calibration_drag_start_points = list(self._calibration_points)
            event.accept()
            return True
        return False

    def _update_calibration_drag(self, event: QtGui.QMouseEvent) -> bool:
        if not self._calibration_dragging or self._calibration_drag_mode is None:
            return False
        scale = self._last_render_scale or 1.0
        p_img = QtCore.QPointF(event.position().x() / scale, event.position().y() / scale)
        if self._calibration_drag_start is None or not self._calibration_drag_start_points:
            return False

        dx = p_img.x() - self._calibration_drag_start.x()
        dy = p_img.y() - self._calibration_drag_start.y()

        # Follow the mouse exactly: endpoints snap to cursor, move mode offsets both points.
        if self._calibration_drag_mode == "p1":
            new_p1 = self._clamp_point(p_img.x(), p_img.y())
            self._calibration_points[0] = new_p1
            self._calibration_points[1] = self._calibration_drag_start_points[1]
        elif self._calibration_drag_mode == "p2":
            new_p2 = self._clamp_point(p_img.x(), p_img.y())
            self._calibration_points[1] = new_p2
            self._calibration_points[0] = self._calibration_drag_start_points[0]
        elif self._calibration_drag_mode == "move":
            start_p1, start_p2 = self._calibration_drag_start_points
            new_p1 = self._clamp_point(start_p1[0] + dx, start_p1[1] + dy)
            new_p2 = self._clamp_point(start_p2[0] + dx, start_p2[1] + dy)
            self._calibration_points[0] = new_p1
            self._calibration_points[1] = new_p2
        event.accept()
        self._update_calibration_status()
        self._refresh_image_view()
        return True

    def _finish_calibration_drag(self, event: QtGui.QMouseEvent) -> bool:
        if not self._calibration_dragging:
            return False
        self._calibration_dragging = False
        self._calibration_drag_mode = None
        self._calibration_drag_start = None
        self._calibration_drag_start_points = []
        self._update_calibration_status()
        event.accept()
        return True

    def _enter_line_edit_mode(self):
        self._line_select_mode = False
        self._line_edit_mode = True
        self._line_profile = None
        self._line_fit = None
        self._sync_panning_enabled()
        self.line_fit_status.setText("Line fit: drag endpoints or line, then press SciPy Fit.")
        self.line_plot_label.clear()
        self._refresh_image_view()

    def _clamp_point(self, x: float, y: float) -> tuple[int, int]:
        h = self.last_payload.np_image.shape[0] if self.last_payload else 0
        w = self.last_payload.np_image.shape[1] if self.last_payload else 0
        return int(max(0, min(w - 1, x))), int(max(0, min(h - 1, y)))

    def _screen_point_for_line(self, pt: tuple[int, int]) -> QtCore.QPointF:
        scale = self._last_render_scale or 1.0
        return QtCore.QPointF(pt[0] * scale, pt[1] * scale)

    def _distance_to_segment(self, p: QtCore.QPointF, a: QtCore.QPointF, b: QtCore.QPointF) -> float:
        ax, ay, bx, by = a.x(), a.y(), b.x(), b.y()
        px, py = p.x(), p.y()
        dx = bx - ax
        dy = by - ay
        if dx == 0 and dy == 0:
            return float(np.hypot(px - ax, py - ay))
        t = ((px - ax) * dx + (py - ay) * dy) / float(dx * dx + dy * dy)
        t = max(0.0, min(1.0, t))
        proj_x = ax + t * dx
        proj_y = ay + t * dy
        return float(np.hypot(px - proj_x, py - proj_y))

    def _start_line_drag(self, event: QtGui.QMouseEvent) -> bool:
        if len(self._line_points) != 2:
            return False
        pos = event.position()
        scale = self._last_render_scale or 1.0
        p_img = QtCore.QPointF(pos.x() / scale, pos.y() / scale)
        p1_screen = self._screen_point_for_line(self._line_points[0])
        p2_screen = self._screen_point_for_line(self._line_points[1])
        hit_radius = 12.0
        dist_p1 = float(np.hypot(pos.x() - p1_screen.x(), pos.y() - p1_screen.y()))
        dist_p2 = float(np.hypot(pos.x() - p2_screen.x(), pos.y() - p2_screen.y()))
        mode = None
        if dist_p1 <= hit_radius:
            mode = "p1"
        elif dist_p2 <= hit_radius:
            mode = "p2"
        else:
            dist_seg = self._distance_to_segment(pos, p1_screen, p2_screen)
            if dist_seg <= hit_radius:
                mode = "move"
        if mode:
            self._line_dragging = True
            self._line_drag_mode = mode
            self._line_drag_start = p_img
            self._line_profile = None
            self._line_fit = None
            self.line_plot_label.clear()
            self.line_fit_status.setText("Line fit: adjusting... press Run Line Fit.")
            return True
        return False

    def _update_line_drag(self, event: QtGui.QMouseEvent) -> bool:
        if not self._line_dragging or self._line_drag_mode is None:
            return False
        scale = self._last_render_scale or 1.0
        p_img = QtCore.QPointF(event.position().x() / scale, event.position().y() / scale)
        dx = p_img.x() - self._line_drag_start.x()
        dy = p_img.y() - self._line_drag_start.y()
        if self._line_drag_mode == "p1":
            new_p1 = self._clamp_point(self._line_points[0][0] + dx, self._line_points[0][1] + dy)
            self._line_points[0] = new_p1
        elif self._line_drag_mode == "p2":
            new_p2 = self._clamp_point(self._line_points[1][0] + dx, self._line_points[1][1] + dy)
            self._line_points[1] = new_p2
        elif self._line_drag_mode == "move":
            new_p1 = self._clamp_point(self._line_points[0][0] + dx, self._line_points[0][1] + dy)
            new_p2 = self._clamp_point(self._line_points[1][0] + dx, self._line_points[1][1] + dy)
            self._line_points[0] = new_p1
            self._line_points[1] = new_p2
        self._line_drag_start = p_img
        self._refresh_image_view()
        return True

    def _finish_line_drag(self, event: QtGui.QMouseEvent) -> bool:
        if not self._line_dragging:
            return False
        self._line_dragging = False
        self._line_drag_mode = None
        return True

    def _start_measure_line_drag(self, event: QtGui.QMouseEvent) -> bool:
        if self._measure_active_mode != "line" or len(self._measure_line_points) != 2:
            return False
        pos = event.position()
        scale = self._last_render_scale or 1.0
        p_img = QtCore.QPointF(pos.x() / scale, pos.y() / scale)
        p1_screen = self._screen_point_for_line(self._measure_line_points[0])
        p2_screen = self._screen_point_for_line(self._measure_line_points[1])
        hit_radius = 16.0
        dist_p1 = float(np.hypot(pos.x() - p1_screen.x(), pos.y() - p1_screen.y()))
        dist_p2 = float(np.hypot(pos.x() - p2_screen.x(), pos.y() - p2_screen.y()))
        mode = None
        if dist_p1 <= hit_radius:
            mode = "p1"
        elif dist_p2 <= hit_radius:
            mode = "p2"
        else:
            dist_seg = self._distance_to_segment(pos, p1_screen, p2_screen)
            if dist_seg <= hit_radius:
                mode = "move"
        if mode:
            self._measure_line_dragging = True
            self._measure_line_drag_mode = mode
            self._measure_line_drag_start = p_img
            self._measure_line_drag_start_points = list(self._measure_line_points)
            event.accept()
            self._sync_panning_enabled()
            return True
        return False

    def _update_measure_line_drag(self, event: QtGui.QMouseEvent) -> bool:
        if not self._measure_line_dragging or self._measure_line_drag_mode is None:
            return False
        scale = self._last_render_scale or 1.0
        p_img = QtCore.QPointF(event.position().x() / scale, event.position().y() / scale)
        if self._measure_line_drag_start is None or not self._measure_line_drag_start_points:
            return False
        dx = p_img.x() - self._measure_line_drag_start.x()
        dy = p_img.y() - self._measure_line_drag_start.y()
        if self._measure_line_drag_mode == "p1":
            new_p1 = self._clamp_point(p_img.x(), p_img.y())
            self._measure_line_points[0] = new_p1
            self._measure_line_points[1] = self._measure_line_drag_start_points[1]
        elif self._measure_line_drag_mode == "p2":
            new_p2 = self._clamp_point(p_img.x(), p_img.y())
            self._measure_line_points[1] = new_p2
            self._measure_line_points[0] = self._measure_line_drag_start_points[0]
        elif self._measure_line_drag_mode == "move":
            start_p1, start_p2 = self._measure_line_drag_start_points
            new_p1 = self._clamp_point(start_p1[0] + dx, start_p1[1] + dy)
            new_p2 = self._clamp_point(start_p2[0] + dx, start_p2[1] + dy)
            self._measure_line_points[0] = new_p1
            self._measure_line_points[1] = new_p2
        event.accept()
        self._update_measure_status()
        self._refresh_image_view()
        return True

    def _finish_measure_line_drag(self, event: QtGui.QMouseEvent) -> bool:
        if not self._measure_line_dragging:
            return False
        self._measure_line_dragging = False
        self._measure_line_drag_mode = None
        self._measure_line_drag_start = None
        self._measure_line_drag_start_points = []
        self._update_measure_status()
        self._sync_panning_enabled()
        event.accept()
        return True

    def _start_measure_arc_drag(self, event: QtGui.QMouseEvent) -> bool:
        if self._measure_active_mode != "arc" or len(self._measure_arc_points) != 3:
            return False
        pos = event.position()
        hit_radius = 16.0
        for idx, pt in enumerate(self._measure_arc_points):
            pt_screen = self._screen_point_for_line(pt)
            dist = float(np.hypot(pos.x() - pt_screen.x(), pos.y() - pt_screen.y()))
            if dist <= hit_radius:
                self._measure_arc_dragging = True
                self._measure_arc_drag_index = idx
                self._sync_panning_enabled()
                event.accept()
                return True
        return False

    def _update_measure_arc_drag(self, event: QtGui.QMouseEvent) -> bool:
        if not self._measure_arc_dragging or self._measure_arc_drag_index is None:
            return False
        scale = self._last_render_scale or 1.0
        p_img = QtCore.QPointF(event.position().x() / scale, event.position().y() / scale)
        new_pt = self._clamp_point(p_img.x(), p_img.y())
        self._measure_arc_points[self._measure_arc_drag_index] = new_pt
        event.accept()
        self._update_measure_status()
        self._refresh_image_view()
        return True

    def _finish_measure_arc_drag(self, event: QtGui.QMouseEvent) -> bool:
        if not self._measure_arc_dragging:
            return False
        self._measure_arc_dragging = False
        self._measure_arc_drag_index = None
        self._sync_panning_enabled()
        event.accept()
        return True

    def _sample_line_profile(self, p1: tuple[int, int], p2: tuple[int, int], min_samples: int = 20):
        if not self.last_payload:
            return None
        raw_data = self.last_payload.np_image
        filtered_data = self._filtered_np_image if self._filtered_np_image is not None else raw_data
        if raw_data.ndim == 3:
            raw_data = np.mean(raw_data, axis=2)
        if filtered_data.ndim == 3:
            filtered_data = np.mean(filtered_data, axis=2)
        raw_data = np.asarray(raw_data, dtype=np.float64)
        filtered_data = np.asarray(filtered_data, dtype=np.float64)
        x1, y1 = p1
        x2, y2 = p2
        length_px = float(np.hypot(x2 - x1, y2 - y1))
        num = max(int(length_px), min_samples)
        if num <= 1:
            return None
        xs = np.linspace(x1, x2, num)
        ys = np.linspace(y1, y2, num)

        def _sample_point(arr: np.ndarray, ix, iy):
            if ix < 0 or iy < 0 or ix >= arr.shape[1] - 1 or iy >= arr.shape[0] - 1:
                ix = min(max(ix, 0), arr.shape[1] - 1)
                iy = min(max(iy, 0), arr.shape[0] - 1)
                return float(arr[int(iy), int(ix)])
            x0 = int(np.floor(ix))
            y0 = int(np.floor(iy))
            dx_f = ix - x0
            dy_f = iy - y0
            v00 = arr[y0, x0]
            v10 = arr[y0, x0 + 1]
            v01 = arr[y0 + 1, x0]
            v11 = arr[y0 + 1, x0 + 1]
            top = v00 * (1 - dx_f) + v10 * dx_f
            bottom = v01 * (1 - dx_f) + v11 * dx_f
            return float(top * (1 - dy_f) + bottom * dy_f)

        raw_profile = np.array([_sample_point(raw_data, ix, iy) for ix, iy in zip(xs, ys)], dtype=np.float64)
        filtered_profile = np.array([_sample_point(filtered_data, ix, iy) for ix, iy in zip(xs, ys)], dtype=np.float64)
        length_mm = float(np.hypot((x2 - x1) * self.mm_per_px_x, (y2 - y1) * self.mm_per_px_y))
        px_axis = np.linspace(0, length_px, num)
        return px_axis, raw_profile, filtered_profile, length_px, length_mm

    def _fit_1d_gaussian(self, profile: np.ndarray):
        # Legacy wrapper retained for compatibility; now uses the general fitter below.
        fit = self._fit_1d_profile(profile, models=("gauss",))
        if not fit:
            return None
        return fit["baseline"], fit["amplitude"], fit["mu"], fit["width"], fit["fit_curve"]

    def _fit_1d_profile(self, profile: np.ndarray, models=("gauss", "lorentz")) -> Optional[dict]:
        """Fit a 1D profile with Gaussian and/or Lorentzian candidates and pick the best by MSE."""
        if profile is None or profile.size < 3:
            return None
        prof = np.nan_to_num(np.asarray(profile, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        baseline = float(np.percentile(prof, 10.0))
        window = min(11, prof.size if prof.size % 2 == 1 else prof.size - 1)
        smooth = prof
        if window >= 3:
            kernel = np.ones(window, dtype=np.float64) / window
            smooth = np.convolve(prof, kernel, mode="same")
        weights = np.clip(smooth - baseline, 0.0, None)
        if weights.sum() <= 0:
            return None

        peak_idx = int(np.argmax(weights))
        mu0 = float(peak_idx)
        if 0 < peak_idx < weights.size - 1:
            y0, y1, y2 = weights[peak_idx - 1], weights[peak_idx], weights[peak_idx + 1]
            denom = (y0 - 2 * y1 + y2)
            if abs(denom) > 1e-9:
                mu0 = float(peak_idx + 0.5 * (y0 - y2) / denom)
        mu0 = float(np.clip(mu0, 0.0, prof.size - 1.0))

        amp0 = float(max(prof[int(round(mu0))] - baseline if 0 <= int(round(mu0)) < prof.size else prof.max() - baseline, 1e-9))

        def _estimate_fwhm():
            half = baseline + amp0 * 0.5
            left = peak_idx
            while left > 0 and prof[left] > half:
                left -= 1
            right = peak_idx
            while right < prof.size - 1 and prof[right] > half:
                right += 1
            def interp_edge(idx_from, idx_to):
                if idx_to == idx_from:
                    return float(idx_from)
                y_from = prof[idx_from]
                y_to = prof[idx_to]
                if abs(y_to - y_from) < 1e-9:
                    return float(idx_from)
                return float(idx_from + (half - y_from) / (y_to - y_from))
            left_x = interp_edge(left, min(left + 1, prof.size - 1))
            right_x = interp_edge(max(right - 1, 0), right)
            width = max(right_x - left_x, 1.0)
            return float(width)

        fwhm_est = _estimate_fwhm()
        sigma0 = max(fwhm_est / 2.3548, 0.3)
        gamma0 = max(fwhm_est / 2.0, 0.3)

        x = np.arange(prof.size, dtype=np.float64)
        mu_candidates = np.clip(mu0 + np.linspace(-2.0, 2.0, 5), 0.0, prof.size - 1.0)
        best = None

        for model in models:
            if model == "gauss":
                width_candidates = [max(sigma0 * s, 0.2) for s in (0.6, 0.85, 1.0, 1.3, 1.8)]
                def fn(mu_val, w_val):
                    return baseline + amp0 * np.exp(-0.5 * ((x - mu_val) / max(w_val, 0.2)) ** 2)
            elif model == "lorentz":
                width_candidates = [max(gamma0 * s, 0.2) for s in (0.6, 0.85, 1.0, 1.3, 1.8)]
                def fn(mu_val, w_val):
                    w_safe = max(w_val, 0.2)
                    return baseline + amp0 * (w_safe**2) / ((x - mu_val) ** 2 + w_safe**2)
            else:
                continue

            for mu_val in mu_candidates:
                for width_val in width_candidates:
                    curve = fn(mu_val, width_val)
                    err = float(np.mean((prof - curve) ** 2))
                    if best is None or err < best["sse"]:
                        best = {
                            "model": model,
                            "baseline": baseline,
                            "amplitude": amp0,
                            "mu": float(mu_val),
                            "width": float(width_val),
                            "fit_curve": curve,
                            "sse": err,
                        }

        if not best:
            return None
        var = float(np.var(prof))
        best["r2"] = max(0.0, 1.0 - best["sse"] / var) if var > 0 else 0.0
        return best

    def _apply_profile_filter(self, profile: np.ndarray) -> np.ndarray:
        """Apply optional FFT-based filters (low/high/band pass) to a 1D profile."""
        if profile is None:
            return profile
        data = np.asarray(profile, dtype=np.float64)
        if data.size < 2:
            return data
        mode = getattr(self, "filter_mode_combo", None)
        mode_text = mode.currentText() if mode else "None"
        low_spin = getattr(self, "filter_low_spin", None)
        high_spin = getattr(self, "filter_high_spin", None)
        low_cut = float(low_spin.value()) if low_spin else 0.0
        high_cut = float(high_spin.value()) if high_spin else 0.5
        low_cut = min(max(low_cut, 0.0), 0.5)
        high_cut = min(max(high_cut, 0.0), 0.5)
        n = data.size
        if mode_text == "None" or n < 2:
            return data
        freqs = np.fft.rfftfreq(n, d=1.0)
        spectrum = np.fft.rfft(data)
        if mode_text == "Low-pass":
            mask = freqs <= high_cut
        elif mode_text == "High-pass":
            mask = freqs >= low_cut
        elif mode_text == "Band-pass":
            if low_cut >= high_cut:
                mask = freqs >= low_cut
            else:
                mask = (freqs >= low_cut) & (freqs <= high_cut)
        else:
            return data
        if not np.any(mask):
            return data
        filtered = np.fft.irfft(spectrum * mask, n=n)
        return np.asarray(filtered, dtype=np.float64)

    def _apply_image_filter_np(self, data: np.ndarray) -> Optional[np.ndarray]:
        """Apply selected FFT filter (low/high/band) to the full image (per channel)."""
        if data is None:
            return None
        mode = getattr(self, "filter_mode_combo", None)
        mode_text = mode.currentText() if mode else "None"
        low_spin = getattr(self, "filter_low_spin", None)
        high_spin = getattr(self, "filter_high_spin", None)
        low_cut = float(low_spin.value()) if low_spin else 0.0
        high_cut = float(high_spin.value()) if high_spin else 0.5
        low_cut = min(max(low_cut, 0.0), 0.5)
        high_cut = min(max(high_cut, 0.0), 0.5)
        if mode_text == "None":
            return data

        arr = np.asarray(data, dtype=np.float64)

        def filter_channel(chan: np.ndarray) -> np.ndarray:
            if chan.ndim != 2:
                return chan
            h, w = chan.shape
            if h < 2 or w < 2:
                return chan
            fy = np.fft.fftfreq(h, d=1.0)
            fx = np.fft.fftfreq(w, d=1.0)
            fx_grid, fy_grid = np.meshgrid(fx, fy)
            radius = np.sqrt(fx_grid * fx_grid + fy_grid * fy_grid)
            if mode_text == "Low-pass":
                mask = radius <= high_cut
            elif mode_text == "High-pass":
                mask = radius >= low_cut
            elif mode_text == "Band-pass":
                if low_cut >= high_cut:
                    mask = radius >= low_cut
                else:
                    mask = (radius >= low_cut) & (radius <= high_cut)
            else:
                return chan
            if not np.any(mask):
                return chan
            spec = np.fft.fft2(chan)
            filtered = np.fft.ifft2(spec * mask).real
            return filtered

        if arr.ndim == 2:
            filtered = filter_channel(arr)
        elif arr.ndim == 3 and arr.shape[2] >= 1:
            channels = [filter_channel(arr[..., c]) for c in range(arr.shape[2])]
            filtered = np.stack(channels, axis=2)
        else:
            return data

        if np.issubdtype(data.dtype, np.integer):
            info = np.iinfo(data.dtype)
            filtered = np.clip(np.rint(filtered), info.min, info.max).astype(data.dtype)
        else:
            filtered = np.asarray(filtered, dtype=data.dtype)
        return filtered

    def _sync_filter_controls(self):
        mode_text = self.filter_mode_combo.currentText() if getattr(self, "filter_mode_combo", None) else "None"
        if getattr(self, "filter_low_spin", None):
            self.filter_low_spin.setEnabled(mode_text in ("High-pass", "Band-pass"))
        if getattr(self, "filter_high_spin", None):
            self.filter_high_spin.setEnabled(mode_text in ("Low-pass", "Band-pass"))

    def _on_filter_changed(self, *args):
        self._sync_filter_controls()
        if self.last_payload:
            self._filtered_np_image = self._apply_image_filter_np(self.last_payload.np_image)
            self._refresh_image_view()
        if len(self._line_points) == 2:
            # Re-run line fit with new filter for immediate visual feedback (without exporting multiple copies)
            self._compute_line_fit(self._line_points[0], self._line_points[1], export=False)
            self.line_fit_status.setText("Filter applied to image. Press Run Line Fit again to export.")

    def _write_cross_section_csv(self, sections: dict[str, tuple[np.ndarray, np.ndarray]], out_dir: str):
        """Persist raw cross-section data into a CSV inside out_dir."""
        if not sections:
            return
        os.makedirs(out_dir, exist_ok=True)
        names = list(sections.keys())
        max_len = max(len(val[0]) for val in sections.values())
        csv_path = os.path.join(out_dir, "cross_sections.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["pixel"] + names)
            for idx in range(max_len):
                pixel_val = ""
                for axis, _ in sections.values():
                    if idx < len(axis):
                        pixel_val = float(axis[idx])
                        break
                row = [pixel_val]
                for name in names:
                    axis, profile = sections[name]
                    if idx < len(profile):
                        row.append(float(profile[idx]))
                    else:
                        row.append("")
                writer.writerow(row)


    @staticmethod
    def _np_to_pil_image(array: np.ndarray) -> Image.Image:
        arr = np.asarray(array)
        if arr.ndim == 2:
            if arr.dtype != np.uint8:
                arr = np.clip(np.rint(arr), 0, 255).astype(np.uint8)
            return Image.fromarray(arr, mode="L")
        elif arr.ndim == 3:
            if arr.dtype != np.uint8:
                arr = np.clip(np.rint(arr), 0, 255).astype(np.uint8)
            return Image.fromarray(arr, mode="RGB")
        raise ValueError("Unsupported array shape for conversion to PIL")

    def _prepare_display_image(self, payload: FramePayload) -> Image.Image:
        base_np = self._filtered_np_image if self._filtered_np_image is not None else payload.np_image
        if self._force_grayscale and base_np.ndim == 3:
            base_np = np.mean(base_np, axis=2)
        img = self._np_to_pil_image(base_np)
        if self._gauss_fit:
            img = self._draw_gaussian_contour(img, self._gauss_fit)
        if len(self._line_points) == 2:
            img = self._draw_line_segment(img, self._line_points, color=(255, 215, 0))
        if self._axis_lines:
            for seg in self._axis_lines:
                p1, p2, color = seg
                img = self._draw_line_segment(img, [p1, p2], color=(color.red(), color.green(), color.blue()))
        img = self._draw_calibration_overlay(img)
        img = self._draw_measure_overlay(img)
        img = self._draw_exposure_gain_overlay(img)
        return img

    def _draw_gaussian_contour(self, image: Image.Image, fit: dict) -> Image.Image:
        if image.mode != "RGB":
            image = image.convert("RGB")
        draw = ImageDraw.Draw(image)
        mu_x = fit.get("mu_x")
        mu_y = fit.get("mu_y")
        w_major = fit.get("w1e2_major") or fit.get("w1e2_x")
        w_minor = fit.get("w1e2_minor") or fit.get("w1e2_y")
        angle_deg = fit.get("angle_deg", 0.0)
        label_text = fit.get("label_text")
        if None in (mu_x, mu_y, w_major, w_minor):
            return image
        rx = max(1.0, float(w_major))
        ry = max(1.0, float(w_minor))
        theta = np.deg2rad(float(angle_deg))
        cos_t = float(np.cos(theta))
        sin_t = float(np.sin(theta))
        points = []
        for t in np.linspace(0, 2 * np.pi, 90):
            cx = rx * np.cos(t)
            cy = ry * np.sin(t)
            x_rot = cx * cos_t - cy * sin_t
            y_rot = cx * sin_t + cy * cos_t
            points.append((mu_x + x_rot, mu_y + y_rot))
        if points:
            points.append(points[0])
            draw.line(points, fill=(0, 255, 0), width=2)
        if label_text:
            text_pos = (int(mu_x + rx + 4), int(mu_y - ry - 12))
            font = self._get_overlay_font(18)
            # Clamp text so it stays inside the image
            try:
                bbox = draw.textbbox((0, 0), label_text, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
            except Exception:
                text_w, text_h = font.getsize(label_text) if font else (0, 0)
            img_w, img_h = image.size
            x = min(max(0, text_pos[0]), max(0, img_w - text_w - 4))
            y = min(max(0, text_pos[1]), max(0, img_h - text_h - 4))
            draw.text((x, y), label_text, fill=(0, 255, 0), font=font)
        return image

    @staticmethod
    def _draw_line_segment(image: Image.Image, points: list[tuple[int, int]], color=(255, 215, 0)):
        if len(points) != 2:
            return image
        if image.mode != "RGB":
            image = image.convert("RGB")
        draw = ImageDraw.Draw(image)
        draw.line(points, fill=color, width=2)
        r = 5
        for pt in points:
            x, y = pt
            draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=2)
        return image

    def _draw_calibration_overlay(self, image: Image.Image) -> Image.Image:
        axis = self._calibration_axis or self._calibration_last_axis
        if len(self._calibration_points) != 2 or axis not in ("x", "y", "both"):
            return image
        color = (0, 200, 255)
        return self._draw_line_segment(image, self._calibration_points, color=color)

    def _draw_measure_overlay(self, image: Image.Image) -> Image.Image:
        img = image
        # Line overlay
        if len(self._measure_line_points) == 2:
            img = self._draw_line_segment(img, self._measure_line_points, color=(120, 255, 120))
            lengths = self._measure_line_lengths()
            if lengths:
                px_len, mm_len = lengths
                mid_x = (self._measure_line_points[0][0] + self._measure_line_points[1][0]) / 2.0
                mid_y = (self._measure_line_points[0][1] + self._measure_line_points[1][1]) / 2.0
                if img.mode != "RGB":
                    img = img.convert("RGB")
                draw = ImageDraw.Draw(img)
                font = self._get_overlay_font(16)
                text = f"{px_len:.1f}px / {mm_len:.3f}mm"
                try:
                    bbox = draw.textbbox((0, 0), text, font=font)
                    text_w = bbox[2] - bbox[0]
                    text_h = bbox[3] - bbox[1]
                except Exception:
                    text_w, text_h = font.getsize(text) if font else (len(text) * 8, 16)
                x = int(mid_x + 8)
                y = int(mid_y - text_h - 4)
                x = max(0, min(img.width - text_w - 4, x))
                y = max(0, min(img.height - text_h - 4, y))
                draw.rectangle((x - 2, y - 2, x + text_w + 2, y + text_h + 2), fill=(0, 0, 0, 150))
                draw.text((x, y), text, fill=(120, 255, 120), font=font)

        # Arc overlay
        if len(self._measure_arc_points) == 3:
            geom = self._measure_arc_geometry()
            if geom:
                cx, cy = geom["center_px"]
                r_px = geom["radius_px"]
                if img.mode != "RGB":
                    img = img.convert("RGB")
                draw = ImageDraw.Draw(img)
                bbox = (cx - r_px, cy - r_px, cx + r_px, cy + r_px)
                draw.ellipse(bbox, outline=(255, 160, 0), width=2)
                # Points and chords for clarity
                for pt in self._measure_arc_points:
                    x, y = pt
                    draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline=(255, 160, 0), width=2)
                draw.line(self._measure_arc_points + [self._measure_arc_points[0]], fill=(200, 120, 0), width=1)
                radius_mm = geom.get("radius_mm")
                mm_text = f"{radius_mm:.3f}mm" if radius_mm is not None else "mm:N/A"
                text = f"r={geom['radius_px']:.1f}px / {mm_text}"
                font = self._get_overlay_font(16)
                try:
                    tbbox = draw.textbbox((0, 0), text, font=font)
                    text_w = tbbox[2] - tbbox[0]
                    text_h = tbbox[3] - tbbox[1]
                except Exception:
                    text_w, text_h = font.getsize(text) if font else (len(text) * 8, 16)
                x = int(cx + r_px + 6)
                y = int(cy - text_h - 4)
                x = max(0, min(img.width - text_w - 4, x))
                y = max(0, min(img.height - text_h - 4, y))
                draw.rectangle((x - 2, y - 2, x + text_w + 2, y + text_h + 2), fill=(0, 0, 0, 150))
                draw.text((x, y), text, fill=(255, 200, 0), font=font)
        return img

    def _get_overlay_font(self, size: int = 18):
        if not hasattr(self, "_cached_overlay_font") or self._cached_overlay_font is None:
            font = None
            try:
                font = ImageFont.truetype("arial.ttf", size)
            except Exception:
                try:
                    font = ImageFont.truetype("DejaVuSans.ttf", size)
                except Exception:
                    font = ImageFont.load_default()
            self._cached_overlay_font = font
        return self._cached_overlay_font

    def _draw_exposure_gain_overlay(self, image: Image.Image) -> Image.Image:
        """Overlay exposure/gain in the top-right during live view."""
        if not self._live:
            return image

        exp_us = None
        gain_val = None
        try:
            if self._use_pylon and self.pylon_thread:
                exp_us, gain_val = self.pylon_thread.get_props()
            elif self._use_cv and self.cv_thread:
                exp_us, gain_val = self.cv_thread.get_props()
            elif self.camera:
                exp_us = int(getattr(self.camera, "exposure_time_us", 0))
                gain_val = getattr(self.camera, "gain", None)
        except Exception:
            return image

        if exp_us is None and gain_val is None:
            return image

        def _fmt(val, suffix: str = "") -> str:
            if val is None:
                return "-"
            try:
                if isinstance(val, (float, int)):
                    if isinstance(val, float) and abs(val - round(val)) > 1e-3:
                        return f"{val:.2f}{suffix}"
                    return f"{int(round(val))}{suffix}"
            except Exception:
                pass
            return f"{val}{suffix}"

        exp_text = _fmt(exp_us, " usec")
        gain_text = _fmt(gain_val)
        text = f"exp: {exp_text}, gain: {gain_text}"

        font = self._get_overlay_font(18)
        base = image.convert("RGBA") if image.mode != "RGBA" else image.copy()
        draw = ImageDraw.Draw(base)
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except Exception:
            text_w, text_h = font.getsize(text) if font else (len(text) * 8, 18)

        pad = 6
        margin = 8
        img_w, img_h = base.size
        text_x = max(0, img_w - text_w - pad - margin)
        text_y = margin
        rect = (
            max(0, text_x - pad),
            max(0, text_y - pad),
            min(img_w, text_x + text_w + pad),
            min(img_h, text_y + text_h + pad),
        )
        draw.rectangle(rect, fill=(0, 0, 0, 128))
        draw.text((text_x, text_y), text, fill=(255, 215, 0, 255), font=font)
        return base.convert("RGB") if image.mode != "RGBA" else base

    def compute_gaussian_fit(self):
        if not self.last_payload:
            QtWidgets.QMessageBox.information(self, "No selection", "Select a cross location first.")
            return
        center = None
        if self._axis_center_px:
            center = self._axis_center_px
        elif self.cross_pos:
            center = self.cross_pos
        if center is None:
            QtWidgets.QMessageBox.information(self, "No selection", "Select a cross location first.")
            return
        data = self._filtered_np_image if self._filtered_np_image is not None else self.last_payload.np_image
        if data.ndim == 3:
            data = np.mean(data, axis=2)
        x0, y0 = center
        h, w = data.shape
        yy, xx = np.indices((h, w))

        # Use a large, image-scaled mask so broader beams are included.
        max_radius = 0.49 * float(min(h, w))  # nearly full frame on the shorter side
        mask = ((xx - x0) ** 2 + (yy - y0) ** 2) <= max_radius**2
        masked_vals = data[mask]
        if masked_vals.size == 0:
            QtWidgets.QMessageBox.information(self, "Fit failed", "Mask contains no pixels.")
            return

        outer_mask = ((xx - x0) ** 2 + (yy - y0) ** 2) >= (0.6 * max_radius) ** 2
        outer_vals = data[outer_mask]
        if outer_vals.size > 0:
            baseline = float(np.percentile(outer_vals, 20.0))
        else:
            baseline = float(np.percentile(masked_vals, 5.0))

        weights = np.clip(data.astype(np.float64) - baseline, 0.0, None)
        weights *= mask.astype(np.float64)
        if float(weights.sum()) <= 0.0:
            QtWidgets.QMessageBox.information(self, "Fit failed", "No signal above background in selection.")
            return

        def _moment_stats(weight_map, grid_x, grid_y):
            s = float(weight_map.sum())
            if s <= 0.0:
                return None
            mu_x = float(np.sum(weight_map * grid_x) / s)
            mu_y = float(np.sum(weight_map * grid_y) / s)
            dx = grid_x - mu_x
            dy = grid_y - mu_y
            cov_xx = float(np.sum(weight_map * dx * dx) / s)
            cov_yy = float(np.sum(weight_map * dy * dy) / s)
            cov_xy = float(np.sum(weight_map * dx * dy) / s)
            cov = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]], dtype=np.float64)
            evals, evecs = np.linalg.eigh(cov)
            evals = np.clip(evals, 0.0, None)
            order = np.argsort(evals)[::-1]
            evals = evals[order]
            evecs = evecs[:, order]
            sigma_major = float(np.sqrt(evals[0])) if evals.size > 0 else 0.0
            sigma_minor = float(np.sqrt(evals[1])) if evals.size > 1 else 0.0
            angle = float(np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0]))) if evecs.size else 0.0
            return mu_x, mu_y, sigma_major, sigma_minor, angle

        stats_px = _moment_stats(weights, xx, yy)
        if stats_px is None:
            QtWidgets.QMessageBox.information(self, "Fit failed", "Could not compute Gaussian moments.")
            return
        mu_x, mu_y, sigma_major, sigma_minor, angle_deg_px = stats_px

        stats_mm = _moment_stats(weights, xx * self.mm_per_px_x, yy * self.mm_per_px_y)
        if stats_mm is None:
            QtWidgets.QMessageBox.information(self, "Fit failed", "Could not compute Gaussian moments.")
            return
        mu_x_mm, mu_y_mm, sigma_major_mm, sigma_minor_mm, _ = stats_mm

        w1e2_major = np.sqrt(2.0) * sigma_major
        w1e2_minor = np.sqrt(2.0) * sigma_minor
        w1e2_major_mm = np.sqrt(2.0) * sigma_major_mm
        w1e2_minor_mm = np.sqrt(2.0) * sigma_minor_mm

        peak = baseline + float(weights.max())
        waist_ratio = (w1e2_major_mm / w1e2_minor_mm) if w1e2_minor_mm else 0.0

        self._gauss_fit = {
            "mu_x": mu_x,
            "mu_y": mu_y,
            "sigma_major": sigma_major,
            "sigma_minor": sigma_minor,
            "w1e2_major": w1e2_major,
            "w1e2_minor": w1e2_minor,
            "w1e2_major_mm": w1e2_major_mm,
            "w1e2_minor_mm": w1e2_minor_mm,
            "angle_deg": angle_deg_px,
            "waist_ratio": waist_ratio,
            "label_text": f"{2*w1e2_major:.2f}px ({2*w1e2_major_mm:.3f}mm) x {2*w1e2_minor:.2f}px ({2*w1e2_minor_mm:.3f}mm) | ratio {waist_ratio:.3f}",
        }

        text = (
            f"Centroid: ({mu_x:.2f}px, {mu_y:.2f}px) ({mu_x_mm:.3f}mm, {mu_y_mm:.3f}mm); "
            f"1/e^2 diam: major {2*w1e2_major:.2f}px ({2*w1e2_major_mm:.3f}mm), "
            f"minor {2*w1e2_minor:.2f}px ({2*w1e2_minor_mm:.3f}mm); "
            f"angle {angle_deg_px:.1f} deg; peak {peak:.1f}, bg {baseline:.1f}"
        )
        self.gauss_result_label.setText(text)
        self._refresh_image_view()

    def _compute_line_fit(self, p1: tuple[int, int], p2: tuple[int, int], export: bool = True):
        if not self.last_payload:
            return
        self.line_fit_status.setText("Line fit: computing...")
        sampled = self._sample_line_profile(p1, p2)
        if sampled is None:
            self.line_fit_status.setText("Line fit failed: points are identical.")
            return
        px_axis, raw_profile, filtered_profile, length_px, length_mm = sampled
        profile = filtered_profile
        fit = self._fit_1d_profile(profile)
        if fit is None:
            self.line_fit_status.setText("Line fit failed: no signal above baseline.")
            return
        model = fit.get("model", "gauss")
        baseline = fit["baseline"]
        amplitude = fit["amplitude"]
        mu = fit["mu"]
        width = fit["width"]
        fit_curve = fit["fit_curve"]
        r2 = fit.get("r2", 0.0)

        self._line_profile = (px_axis, raw_profile, profile)
        self._line_fit = {
            "baseline": baseline,
            "amplitude": amplitude,
            "mu": mu,
            "width": width,
            "model": model,
            "r2": r2,
            "length_px": length_px,
            "length_mm": length_mm,
        }
        px_step = length_px / max(len(px_axis) - 1, 1)
        mm_step = length_mm / max(len(px_axis) - 1, 1)
        if model == "lorentz":
            gamma_px = width * px_step
            gamma_mm = width * mm_step
            fwhm_px = 2.0 * gamma_px
            fwhm_mm = 2.0 * gamma_mm
            w1e2_px = 2.0 * gamma_px * np.sqrt(np.e**2 - 1.0)
            w1e2_mm = 2.0 * gamma_mm * np.sqrt(np.e**2 - 1.0)
            width_label = "gamma"
            width_px_val = gamma_px
            width_mm_val = gamma_mm
        else:
            sigma_px = width * px_step
            sigma_mm = width * mm_step
            fwhm_px = 2.3548 * sigma_px
            fwhm_mm = 2.3548 * sigma_mm
            w1e2_px = 2.0 * np.sqrt(2.0) * sigma_px
            w1e2_mm = 2.0 * np.sqrt(2.0) * sigma_mm
            width_label = "sigma"
            width_px_val = sigma_px
            width_mm_val = sigma_mm
        mu_pos_px = (mu / max(len(px_axis) - 1, 1)) * length_px
        mu_pos_mm = (mu / max(len(px_axis) - 1, 1)) * length_mm
        self._update_line_plot(px_axis, profile, fit_curve, length_mm, fwhm_mm, w1e2_mm, raw_profile=raw_profile)
        self.line_fit_status.setText(
            f"Line fit OK ({model}): "
            f"mu={mu_pos_px:.2f}px ({mu_pos_mm:.3f}mm), "
            f"{width_label}={width_px_val:.2f}px ({width_mm_val:.3f}mm), "
            f"FWHM={fwhm_px:.2f}px ({fwhm_mm:.3f}mm), "
            f"1/e^2={w1e2_px:.2f}px ({w1e2_mm:.3f}mm), "
            f"R^2={r2:.3f}"
        )
        self._pending_sections["line"] = (px_axis, raw_profile)
        self._refresh_image_view()

    def run_line_fit(self):
        if len(self._line_points) != 2:
            self.line_fit_status.setText("Line fit: select two points first.")
            return
        self._compute_line_fit(self._line_points[0], self._line_points[1], export=True)
        self._sync_panning_enabled()

    def _moment_center_and_cov(self, data: np.ndarray):
        baseline = float(np.percentile(data, 10.0))
        weights = np.clip(data.astype(np.float64) - baseline, 0.0, None)
        if weights.sum() <= 0:
            return None
        h, w = data.shape
        yy, xx = np.indices((h, w))
        mu_x = float(np.sum(weights * xx) / weights.sum())
        mu_y = float(np.sum(weights * yy) / weights.sum())
        dx = xx - mu_x
        dy = yy - mu_y
        cov_xx = float(np.sum(weights * dx * dx) / weights.sum())
        cov_yy = float(np.sum(weights * dy * dy) / weights.sum())
        cov_xy = float(np.sum(weights * dx * dy) / weights.sum())
        return mu_x, mu_y, cov_xx, cov_yy, cov_xy

    def _compute_axis_line_fit(self, name: str, p1: tuple[int, int], p2: tuple[int, int]) -> Optional[dict]:
        sampled = self._sample_line_profile(p1, p2)
        if sampled is None:
            return None
        px_axis, raw_profile, filtered_profile, length_px, length_mm = sampled
        profile = filtered_profile
        fit = self._fit_1d_profile(profile)
        if fit is None:
            return None
        model = fit.get("model", "gauss")
        mu = fit["mu"]
        width = fit["width"]
        fit_curve = fit["fit_curve"]
        px_step = length_px / max(len(px_axis) - 1, 1)
        mm_step = length_mm / max(len(px_axis) - 1, 1)
        if model == "lorentz":
            gamma_px = width * px_step
            gamma_mm = width * mm_step
            fwhm_px = 2.0 * gamma_px
            fwhm_mm = 2.0 * gamma_mm
            w1e2_px = 2.0 * gamma_px * np.sqrt(np.e**2 - 1.0)
            w1e2_mm = 2.0 * gamma_mm * np.sqrt(np.e**2 - 1.0)
            width_px_val = gamma_px
            width_mm_val = gamma_mm
        else:
            sigma_px = width * px_step
            sigma_mm = width * mm_step
            fwhm_px = 2.3548 * sigma_px
            fwhm_mm = 2.3548 * sigma_mm
            w1e2_px = 2.0 * np.sqrt(2.0) * sigma_px
            w1e2_mm = 2.0 * np.sqrt(2.0) * sigma_mm
            width_px_val = sigma_px
            width_mm_val = sigma_mm
        mu_pos_px = (mu / max(len(px_axis) - 1, 1)) * length_px
        mu_pos_mm = (mu / max(len(px_axis) - 1, 1)) * length_mm
        return {
            "name": name,
            "px_axis": px_axis,
            "profile": profile,
            "raw_profile": raw_profile,
            "fit_curve": fit_curve,
            "length_px": length_px,
            "length_mm": length_mm,
            "model": model,
            "width_px": width_px_val,
            "width_mm": width_mm_val,
            "fwhm_px": fwhm_px,
            "fwhm_mm": fwhm_mm,
            "w1e2_px": w1e2_px,
            "w1e2_mm": w1e2_mm,
            "mu_px": mu_pos_px,
            "mu_mm": mu_pos_mm,
        }

    def run_360_fit(self):
        if not self.last_payload:
            self.axis_fit_status.setText("360 fit: no image.")
            return
        if len(self._line_points) != 2:
            self.axis_fit_status.setText("360 fit: select a line first.")
            return
        data = self._filtered_np_image if self._filtered_np_image is not None else self.last_payload.np_image
        if data.ndim == 3:
            data = np.mean(data, axis=2)
        center_stats = self._moment_center_and_cov(data)
        if center_stats is None:
            self.axis_fit_status.setText("360 fit failed: no signal.")
            return
        mu_x, mu_y, _, _, _ = center_stats
        h, w = data.shape
        cx = float(np.clip(mu_x, 0, w - 1))
        cy = float(np.clip(mu_y, 0, h - 1))

        horiz_p1 = (0, int(round(cy)))
        horiz_p2 = (w - 1, int(round(cy)))
        vert_p1 = (int(round(cx)), 0)
        vert_p2 = (int(round(cx)), h - 1)

        fit_h = self._compute_axis_line_fit("Horizontal", horiz_p1, horiz_p2)
        fit_v = self._compute_axis_line_fit("Vertical", vert_p1, vert_p2)
        if not fit_h or not fit_v:
            self.axis_fit_status.setText("360 fit failed: could not fit axes.")
            return

        self._axis_lines = [
            (horiz_p1, horiz_p2, QtGui.QColor(0, 200, 255)),
            (vert_p1, vert_p2, QtGui.QColor(255, 120, 0)),
        ]
        self._axis_fit_results = {"H": fit_h, "V": fit_v}
        self._axis_center_px = (cx, cy)
        self.cross_pos = (int(round(cx)), int(round(cy)))

        self._update_axis_plot(
            self.axis_plot_h,
            "Horizontal profile (cyan) + fit (orange)",
            fit_h["px_axis"],
            fit_h["profile"],
            fit_h["fit_curve"],
            fit_h["length_mm"],
            fit_h["fwhm_mm"],
            fit_h["w1e2_mm"],
        )
        self._update_axis_plot(
            self.axis_plot_v,
            "Vertical profile (cyan) + fit (orange)",
            fit_v["px_axis"],
            fit_v["profile"],
            fit_v["fit_curve"],
            fit_v["length_mm"],
            fit_v["fwhm_mm"],
            fit_v["w1e2_mm"],
        )

        self.axis_fit_status.setText(
            "360 fit OK: "
            f"H FWHM={fit_h['fwhm_px']:.2f}px ({fit_h['fwhm_mm']:.3f}mm); "
            f"V FWHM={fit_v['fwhm_px']:.2f}px ({fit_v['fwhm_mm']:.3f}mm)"
        )
        self._pending_sections["horizontal"] = (fit_h["px_axis"], fit_h["raw_profile"])
        self._pending_sections["vertical"] = (fit_v["px_axis"], fit_v["raw_profile"])
        # Build a Gaussian overlay using the 360 fit widths and center for correct scale
        self._gauss_fit = {
            "mu_x": cx,
            "mu_y": cy,
            "w1e2_major": fit_h["w1e2_px"],
            "w1e2_minor": fit_v["w1e2_px"],
            "w1e2_major_mm": fit_h["w1e2_mm"],
            "w1e2_minor_mm": fit_v["w1e2_mm"],
            "angle_deg": 0.0,
            "waist_ratio": (fit_h["w1e2_mm"] / fit_v["w1e2_mm"]) if fit_v["w1e2_mm"] else 0.0,
            "label_text": f"{fit_h['w1e2_px']*2:.2f}px ({fit_h['w1e2_mm']*2:.3f}mm) x {fit_v['w1e2_px']*2:.2f}px ({fit_v['w1e2_mm']*2:.3f}mm) | ratio {(fit_h['w1e2_mm'] / fit_v['w1e2_mm']) if fit_v['w1e2_mm'] else 0.0:.3f}",
        }
        self._refresh_image_view()
        self._line_edit_mode = False
        self._sync_panning_enabled()

    def run_scipy_fit(self):
        if not self.last_payload:
            self.axis_fit_status.setText("SciPy fit: no image.")
            return
        data = self._filtered_np_image if self._filtered_np_image is not None else self.last_payload.np_image
        if data.ndim == 3:
            data = np.mean(data, axis=2)
        center_stats = self._moment_center_and_cov(data)
        if center_stats is None:
            self.axis_fit_status.setText("SciPy fit failed: no signal for centroid.")
            return
        mu_x, mu_y, _, _, _ = center_stats
        h, w = data.shape
        cx = int(round(np.clip(mu_x, 0, w - 1)))
        cy = int(round(np.clip(mu_y, 0, h - 1)))

        try:
            res_h = gaussian_or_lorentzian_aic(np.arange(w, dtype=float), data[cy, :], mode="auto")[0]
            res_v = gaussian_or_lorentzian_aic(np.arange(h, dtype=float), data[:, cx], mode="auto")[0]
        except Exception as exc:
            self.axis_fit_status.setText(f"SciPy fit failed: {exc}")
            return

        def _metrics(res, x_axis: np.ndarray, profile: np.ndarray, px_step: float, mm_step: float):
            model = res["best_model"]
            width_param = float(res["p_best"][2])
            if model == "gaussian":
                sigma_px = width_param
                fwhm_px = 2.3548 * sigma_px
                w1e2_px = 2.0 * np.sqrt(2.0) * sigma_px
            else:
                gamma_px = width_param
                fwhm_px = gamma_px
                w1e2_px = gamma_px  # approximate radius with FWHM for overlay consistency
            fwhm_mm = fwhm_px * px_step if model == "gaussian" else fwhm_px * px_step
            w1e2_mm = w1e2_px * px_step
            return {
                "model": model,
                "width_px": width_param,
                "fwhm_px": fwhm_px,
                "fwhm_mm": fwhm_mm,
                "w1e2_px": w1e2_px,
                "w1e2_mm": w1e2_mm,
                "fit_curve": res["yhat_best"],
                "profile": profile,
                "px_axis": x_axis,
                "r2": res.get("r2_best", 0.0),
            }

        x_axis_h = np.arange(w, dtype=float)
        x_axis_v = np.arange(h, dtype=float)
        fit_h = _metrics(res_h, x_axis_h, data[cy, :], self.mm_per_px_x, self.mm_per_px_x)
        fit_v = _metrics(res_v, x_axis_v, data[:, cx], self.mm_per_px_y, self.mm_per_px_y)

        self._axis_lines = [
            ((0, cy), (w - 1, cy), QtGui.QColor(0, 200, 255)),
            ((cx, 0), (cx, h - 1), QtGui.QColor(255, 120, 0)),
        ]
        self._axis_fit_results = {"H": fit_h, "V": fit_v}
        self._axis_center_px = (float(cx), float(cy))
        self.cross_pos = (cx, cy)

        self._update_axis_plot(
            self.axis_plot_h,
            "Horizontal profile (cyan) + fit (orange)",
            fit_h["px_axis"],
            fit_h["profile"],
            fit_h["fit_curve"],
            (fit_h["px_axis"][-1] - fit_h["px_axis"][0] if fit_h["px_axis"].size else 0) * self.mm_per_px_x,
            fit_h["fwhm_mm"],
            fit_h["w1e2_mm"],
        )
        self._update_axis_plot(
            self.axis_plot_v,
            "Vertical profile (cyan) + fit (orange)",
            fit_v["px_axis"],
            fit_v["profile"],
            fit_v["fit_curve"],
            (fit_v["px_axis"][-1] - fit_v["px_axis"][0] if fit_v["px_axis"].size else 0) * self.mm_per_px_y,
            fit_v["fwhm_mm"],
            fit_v["w1e2_mm"],
        )

        self.axis_fit_status.setText(
            "SciPy fit: "
            f"H model={fit_h['model']} FWHM={fit_h['fwhm_px']:.2f}px ({fit_h['fwhm_mm']:.3f}mm); "
            f"V model={fit_v['model']} FWHM={fit_v['fwhm_px']:.2f}px ({fit_v['fwhm_mm']:.3f}mm)"
        )
        self._pending_sections["horizontal"] = (fit_h["px_axis"], fit_h["profile"])
        self._pending_sections["vertical"] = (fit_v["px_axis"], fit_v["profile"])

        # Optional: run SciPy on the user-defined line if two points are set.
        if len(self._line_points) == 2:
            sampled = self._sample_line_profile(self._line_points[0], self._line_points[1])
            if sampled is None:
                self.line_fit_status.setText("Line fit failed: points are identical.")
            else:
                px_axis, raw_profile, filtered_profile, length_px, length_mm = sampled
                try:
                    res_line = gaussian_or_lorentzian_aic(px_axis, filtered_profile, mode="auto")[0]
                except Exception as exc:
                    self.line_fit_status.setText(f"SciPy line fit failed: {exc}")
                else:
                    model = res_line["best_model"]
                    width_param = float(res_line["p_best"][2])
                    mu_pos_px = float(res_line["p_best"][1])
                    if length_px > 0:
                        mu_pos_mm = mu_pos_px / length_px * length_mm
                    else:
                        mu_pos_mm = 0.0
                    if model == "gaussian":
                        sigma_px = width_param
                        fwhm_px = 2.3548 * sigma_px
                        w1e2_px = 2.0 * np.sqrt(2.0) * sigma_px
                    else:
                        gamma_px = width_param
                        fwhm_px = gamma_px
                        w1e2_px = gamma_px
                    px_step_line = length_px / max(len(px_axis) - 1, 1)
                    mm_step_line = length_mm / max(len(px_axis) - 1, 1)
                    mm_per_px_line = (mm_step_line / px_step_line) if px_step_line > 0 else 0.0
                    fwhm_mm = fwhm_px * mm_per_px_line
                    w1e2_mm = w1e2_px * mm_per_px_line

                    fit_curve = res_line["yhat_best"]
                    self._line_profile = (px_axis, raw_profile, filtered_profile)
                    self._line_fit = {
                        "model": model,
                        "width_px": width_param,
                        "fwhm_px": fwhm_px,
                        "fwhm_mm": fwhm_mm,
                        "w1e2_px": w1e2_px,
                        "w1e2_mm": w1e2_mm,
                        "mu_px": mu_pos_px,
                        "mu_mm": mu_pos_mm,
                        "r2": res_line.get("r2_best", 0.0),
                    }
                    self._update_line_plot(
                        px_axis,
                        filtered_profile,
                        fit_curve,
                        length_mm,
                        fwhm_mm,
                        w1e2_mm,
                        raw_profile=raw_profile,
                    )
                    self.line_fit_status.setText(
                        f"SciPy line fit ({model}): "
                        f"mu={mu_pos_px:.2f}px ({mu_pos_mm:.3f}mm), "
                        f"FWHM={fwhm_px:.2f}px ({fwhm_mm:.3f}mm), "
                        f"1/e^2={w1e2_px:.2f}px ({w1e2_mm:.3f}mm), "
                        f"R^2={self._line_fit['r2']:.3f}"
                    )
                    self._pending_sections["line"] = (px_axis, raw_profile)
        else:
            self.line_fit_status.setText("Line fit: select two points, then press SciPy Fit.")

        self._gauss_fit = {
            "mu_x": cx,
            "mu_y": cy,
            "w1e2_major": fit_h["w1e2_px"],
            "w1e2_minor": fit_v["w1e2_px"],
            "w1e2_major_mm": fit_h["w1e2_mm"],
            "w1e2_minor_mm": fit_v["w1e2_mm"],
            "angle_deg": 0.0,
            "waist_ratio": (fit_h["w1e2_mm"] / fit_v["w1e2_mm"]) if fit_v["w1e2_mm"] else 0.0,
            "label_text": f"{fit_h['w1e2_px']*2:.2f}px ({fit_h['w1e2_mm']*2:.3f}mm) x {fit_v['w1e2_px']*2:.2f}px ({fit_v['w1e2_mm']*2:.3f}mm) | ratio {(fit_h['w1e2_mm'] / fit_v['w1e2_mm']) if fit_v['w1e2_mm'] else 0.0:.3f}",
        }
        self._refresh_image_view()
        self._line_edit_mode = False
        self._sync_panning_enabled()

    def eventFilter(self, obj, event):
        if obj is getattr(self, "hist_window", None):
            if event.type() == QtCore.QEvent.Type.Resize:
                new_size = obj.size()
                if new_size.width() > 0 and new_size.height() > 0:
                    self._hist_size = new_size
                    if self._histogram_enabled and self.last_payload:
                        self._update_histogram_window(self.last_payload)
            elif event.type() == QtCore.QEvent.Type.Move:
                self._hist_window_positioned = True
        elif obj is getattr(self, "fit_window", None):
            if event.type() == QtCore.QEvent.Type.Move:
                self._fit_window_positioned = True
        elif obj is getattr(self, "calibration_window", None):
            if event.type() == QtCore.QEvent.Type.Move:
                self._calibration_window_positioned = True
        elif event.type() == QtCore.QEvent.Type.Wheel and obj in (
            getattr(self, "image_label", None),
            getattr(self, "scroll_area", None).viewport() if getattr(self, "scroll_area", None) else None,
        ):
            if self._handle_wheel_zoom(event, obj):
                return True
        elif obj is getattr(self, "image_label", None):
            calib_drag_ready = (
                len(self._calibration_points) == 2
                and self._calibration_axis is not None
                and (getattr(self, "measure_checkbox", None) is None or self.measure_checkbox.isChecked())
            )
            handled = False
            if calib_drag_ready:
                if event.type() == QtCore.QEvent.Type.MouseButtonPress:
                    handled = self._start_calibration_drag(event)
                elif event.type() == QtCore.QEvent.Type.MouseMove:
                    handled = self._update_calibration_drag(event)
                elif event.type() == QtCore.QEvent.Type.MouseButtonRelease:
                    handled = self._finish_calibration_drag(event)
            if handled:
                return True

            measure_line_ready = (
                self._measure_active_mode == "line"
                and len(self._measure_line_points) == 2
                and (getattr(self, "measure_checkbox", None) is None or self.measure_checkbox.isChecked())
            )
            measure_arc_ready = (
                self._measure_active_mode == "arc"
                and len(self._measure_arc_points) == 3
                and (getattr(self, "measure_checkbox", None) is None or self.measure_checkbox.isChecked())
            )
            if measure_line_ready:
                if event.type() == QtCore.QEvent.Type.MouseButtonPress:
                    handled = self._start_measure_line_drag(event)
                elif event.type() == QtCore.QEvent.Type.MouseMove:
                    handled = self._update_measure_line_drag(event)
                elif event.type() == QtCore.QEvent.Type.MouseButtonRelease:
                    handled = self._finish_measure_line_drag(event)
                if handled:
                    return True
            elif measure_arc_ready:
                if event.type() == QtCore.QEvent.Type.MouseButtonPress:
                    handled = self._start_measure_arc_drag(event)
                elif event.type() == QtCore.QEvent.Type.MouseMove:
                    handled = self._update_measure_arc_drag(event)
                elif event.type() == QtCore.QEvent.Type.MouseButtonRelease:
                    handled = self._finish_measure_arc_drag(event)
                if handled:
                    return True

            if len(self._line_points) == 2:
                if event.type() == QtCore.QEvent.Type.MouseButtonPress:
                    if self._start_line_drag(event):
                        return True
                elif event.type() == QtCore.QEvent.Type.MouseMove:
                    if self._update_line_drag(event):
                        return True
                elif event.type() == QtCore.QEvent.Type.MouseButtonRelease:
                    if self._finish_line_drag(event):
                        return True
        return super().eventFilter(obj, event)

    @staticmethod
    def _draw_cross(image: Image.Image, pos):
        x, y = pos
        size = 10
        color = (255, 0, 0)
        if image.mode != "RGB":
            image = image.convert("RGB")
        pixels = image.load()
        width, height = image.size
        for dx in range(-size, size + 1):
            if 0 <= x + dx < width and 0 <= y < height:
                pixels[x + dx, y] = color
        for dy in range(-size, size + 1):
            if 0 <= x < width and 0 <= y + dy < height:
                pixels[x, y + dy] = color
        return image

    @staticmethod
    def _pil_to_qpixmap(image: Image.Image) -> QtGui.QPixmap:
        if image.mode == "RGB":
            data = image.tobytes()
            qimage = QtGui.QImage(
                data,
                image.width,
                image.height,
                image.width * 3,
                QtGui.QImage.Format.Format_RGB888,
            ).copy()
        elif image.mode == "L":
            data = image.tobytes()
            qimage = QtGui.QImage(
                data,
                image.width,
                image.height,
                image.width,
                QtGui.QImage.Format.Format_Grayscale8,
            ).copy()
        else:
            rgb_image = image.convert("RGB")
            data = rgb_image.tobytes()
            qimage = QtGui.QImage(
                data,
                rgb_image.width,
                rgb_image.height,
                rgb_image.width * 3,
                QtGui.QImage.Format.Format_RGB888,
            ).copy()
        return QtGui.QPixmap.fromImage(qimage)

    def save_image(self):
        if not self.last_payload:
            QtWidgets.QMessageBox.information(self, "No image", "No image to save.")
            return
        last_dir = self._settings.value("last_image_dir", DATA_DIR, type=str)
        default_path = os.path.join(last_dir, f"capture_{int(time.time())}.png")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Image",
            default_path,
            "PNG Files (*.png);;TIFF Files (*.tif *.tiff);;All Files (*.*)",
        )
        if not path:
            return
        try:
            # Build timestamped folder alongside the chosen path.
            base_dir = os.path.dirname(path) or DATA_DIR
            timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            out_dir = os.path.join(base_dir, timestamp)
            os.makedirs(out_dir, exist_ok=True)

            # Save what is currently shown: filtered image plus overlays (fits/axes/line/cross).
            display_image = self._prepare_display_image(self.last_payload)
            if self.cross_pos:
                display_image = self._draw_cross(display_image, self.cross_pos)
            main_name = os.path.basename(path) or "image.png"
            main_path = os.path.join(out_dir, main_name)
            display_image.save(main_path)

            # Also save the fit plots if they exist.
            self._save_label_plot(self.line_plot_label, os.path.join(out_dir, "line_plot.png"))
            self._save_label_plot(self.axis_plot_h, os.path.join(out_dir, "axis_h.png"))
            self._save_label_plot(self.axis_plot_v, os.path.join(out_dir, "axis_v.png"))

            # Save cross-section CSV for the latest fits (if available).
            self._write_cross_section_csv(self._pending_sections, out_dir)

            self.status_label.setText(f"Saved to {out_dir}")
            self._settings.setValue("last_image_dir", base_dir)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save: {exc}")

    def _save_label_plot(self, label: QtWidgets.QLabel, path: str):
        """Save a QLabel pixmap (if present) to disk."""
        if not label:
            return
        pixmap = label.pixmap()
        if pixmap is None or pixmap.isNull():
            return
        try:
            pixmap.save(path)
        except Exception:
            # Silent fail; saving plots is best-effort.
            pass

    def load_image(self):
        last_dir = self._settings.value("last_image_dir", DATA_DIR, type=str)
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open Image",
            last_dir,
            "Images (*.png *.jpg *.jpeg *.tif *.tiff);;All Files (*.*)",
        )
        if not path:
            return
        try:
            img = Image.open(path)
            np_img = np.array(img)
            payload = FramePayload(pil_image=img, np_image=np_img, frame_count=-1)
            self.cross_pos = None
            self._display_frame(payload)
            self.status_label.setText(f"Loaded {path}")
            self._settings.setValue("last_image_dir", os.path.dirname(path))
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load image: {exc}")

    def show_histogram(self):
        if not self.last_payload:
            QtWidgets.QMessageBox.information(self, "No image", "No image to analyze.")
            return
        data = self.last_payload.np_image
        if data.ndim == 3:
            data = np.mean(data, axis=2)
        hist, _ = np.histogram(data.flatten(), bins=256, range=[0, 255])
        hist = hist.astype(np.float64)
        if hist.max() > 0:
            hist = hist / hist.max()

        width, height = 512, 200
        image = QtGui.QImage(width, height, QtGui.QImage.Format.Format_RGB32)
        image.fill(QtGui.QColor("white"))
        painter = QtGui.QPainter(image)
        painter.setPen(QtGui.QColor("blue"))
        painter.setBrush(QtGui.QColor("blue"))
        for i, v in enumerate(hist):
            x = i * 2
            bar_height = int(v * (height - 20))
            painter.drawRect(x, height - bar_height, 1, bar_height)
        painter.end()

        label = QtWidgets.QLabel()
        label.setPixmap(QtGui.QPixmap.fromImage(image))
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Histogram")
        layout = QtWidgets.QVBoxLayout(dialog)
        layout.addWidget(label)
        dialog.setLayout(layout)
        dialog.exec()

    def clear_cross(self):
        self.cross_pos = None
        if self.last_payload:
            self._refresh_image_view()

    def _sync_panning_enabled(self):
        if getattr(self, "image_label", None):
            allow_pan = (
                not self._fit_to_window
                and not self._line_edit_mode
                and not self._calibration_dragging
                and not self._measure_line_dragging
                and not self._measure_arc_dragging
            )
            self.image_label.set_panning_enabled(allow_pan)

    def _reset_pan(self):
        if not getattr(self, "scroll_area", None):
            return
        hbar = self.scroll_area.horizontalScrollBar()
        vbar = self.scroll_area.verticalScrollBar()
        if hbar:
            hbar.setValue(0)
        if vbar:
            vbar.setValue(0)

    def start_line_fit_selection(self):
        if not self.last_payload:
            QtWidgets.QMessageBox.information(self, "No image", "Capture or load an image first.")
            return
        self._line_select_mode = True
        self._line_edit_mode = False
        self._line_points = []
        self._line_profile = None
        self._line_fit = None
        self._axis_lines = []
        self._axis_fit_results = {}
        self.line_fit_status.setText("Line fit: click first point, then second, then press SciPy Fit.")
        self.line_plot_label.clear()
        self.axis_fit_status.setText("360 fit: not computed.")
        self.axis_plot_h.clear()
        self.axis_plot_v.clear()
        self._pending_sections = {}
        self._refresh_image_view()

    def on_mouse_move(self, x: int, y: int):
        if not self.last_payload:
            self.coord_label.setText("x: -, y: -, val: -")
            return
        img = self._filtered_np_image if self._filtered_np_image is not None else self.last_payload.np_image
        scale = self._last_render_scale if self._last_render_scale else 1.0
        h, w = img.shape[:2]
        src_x = int(x / scale)
        src_y = int(y / scale)
        if 0 <= src_x < w and 0 <= src_y < h:
            val = img[src_y, src_x]
            if isinstance(val, np.ndarray):
                val = val.tolist()
            gray_val = None
            if img.ndim == 2:
                gray_val = int(img[src_y, src_x])
            elif img.ndim == 3:
                gray_val = int(np.mean(img[src_y, src_x]))
            gray_suffix = f", gray: {gray_val}" if gray_val is not None else ""
            # Show mm relative to the cross if set; pixels remain absolute.
            origin_x, origin_y = self.cross_pos if self.cross_pos else (0, 0)
            mm_x = (src_x - origin_x) * self.mm_per_px_x
            mm_y = (src_y - origin_y) * self.mm_per_px_y
            mm_suffix = f", x_mm: {mm_x:.3f}, y_mm: {mm_y:.3f}"
            self.coord_label.setText(f"x: {src_x}, y: {src_y}, val: {val}{gray_suffix}{mm_suffix}")
        else:
            self.coord_label.setText("x: -, y: -, val: -")

    def _compute_scroll_ratios(self, event: QtGui.QWheelEvent, source_obj) -> Optional[tuple[float, float]]:
        if self._fit_to_window or not getattr(self, "scroll_area", None):
            return None
        hbar = self.scroll_area.horizontalScrollBar()
        vbar = self.scroll_area.verticalScrollBar()
        if not hbar or not vbar:
            return None
        pos = event.position().toPoint()
        if source_obj is self.image_label:
            abs_x = pos.x()
            abs_y = pos.y()
        else:
            abs_x = pos.x() + hbar.value()
            abs_y = pos.y() + vbar.value()
        w = max(1, self.image_label.width())
        h = max(1, self.image_label.height())
        return (abs_x / w, abs_y / h)

    def _restore_scroll_from_ratios(self, ratios: Optional[tuple[float, float]]):
        if self._fit_to_window or not ratios or not getattr(self, "scroll_area", None):
            return
        h_ratio, v_ratio = ratios
        hbar = self.scroll_area.horizontalScrollBar()
        vbar = self.scroll_area.verticalScrollBar()
        viewport = self.scroll_area.viewport()
        if not (hbar and vbar and viewport):
            return
        target_x = int(h_ratio * self.image_label.width() - viewport.width() / 2)
        target_y = int(v_ratio * self.image_label.height() - viewport.height() / 2)
        hbar.setValue(max(0, min(target_x, hbar.maximum())))
        vbar.setValue(max(0, min(target_y, vbar.maximum())))

    def _handle_wheel_zoom(self, event: QtGui.QWheelEvent, source_obj) -> bool:
        delta = event.angleDelta().y()
        if delta == 0:
            return False
        ratios = self._compute_scroll_ratios(event, source_obj)
        if self._fit_to_window:
            self._fit_to_window = False
            self._zoom = 1.0
        factor = 1.15 if delta > 0 else 1 / 1.15
        new_zoom = min(max(self._zoom * factor, 0.05), 20.0)
        if abs(new_zoom - self._zoom) < 1e-6:
            return True
        self._zoom = new_zoom
        self._sync_panning_enabled()
        self._refresh_image_view()
        self._restore_scroll_from_ratios(ratios)
        event.accept()
        return True

    def on_pan_start(self):
        if self._fit_to_window:
            return
        self.image_label.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)

    def on_pan_drag(self, dx: int, dy: int):
        if self._fit_to_window or not getattr(self, "scroll_area", None):
            return
        hbar = self.scroll_area.horizontalScrollBar()
        vbar = self.scroll_area.verticalScrollBar()
        if hbar:
            hbar.setValue(hbar.value() - dx)
        if vbar:
            vbar.setValue(vbar.value() - dy)

    def on_pan_end(self):
        self.image_label.unsetCursor()

    def on_mouse_click(self, x: int, y: int):
        if not self.last_payload:
            return
        scale = self._last_render_scale if self._last_render_scale else 1.0
        img = self._filtered_np_image if self._filtered_np_image is not None else self.last_payload.np_image
        h, w = img.shape[:2]
        src_x = int(x / scale)
        src_y = int(y / scale)
        if self._calibration_axis:
            if 0 <= src_x < w and 0 <= src_y < h and len(self._calibration_points) < 2:
                self._calibration_points.append((src_x, src_y))
                self._calibration_last_axis = self._calibration_axis
                self._update_calibration_status()
                self._refresh_image_view()
            return
        if self._measure_active_mode == "line":
            if 0 <= src_x < w and 0 <= src_y < h:
                self._measure_line_points.append((src_x, src_y))
                if len(self._measure_line_points) > 2:
                    self._measure_line_points = self._measure_line_points[:2]
                self._update_measure_status()
                self._refresh_image_view()
            return
        if self._measure_active_mode == "arc":
            if 0 <= src_x < w and 0 <= src_y < h:
                self._measure_arc_points.append((src_x, src_y))
                if len(self._measure_arc_points) > 3:
                    self._measure_arc_points = self._measure_arc_points[:3]
                self._update_measure_status()
                self._refresh_image_view()
            return
        if self._line_select_mode:
            if 0 <= src_x < w and 0 <= src_y < h:
                self._line_points.append((src_x, src_y))
                if len(self._line_points) == 1:
                    self.line_fit_status.setText("Line fit: click second point...")
                elif len(self._line_points) >= 2:
                    self._line_points = self._line_points[:2]
                    self._enter_line_edit_mode()
                    return
            return
        if 0 <= src_x < w and 0 <= src_y < h:
            # Place the cross exactly where the user clicked (no snapping).
            self.cross_pos = (src_x, src_y)
            self._refresh_image_view()
            self.gauss_result_label.setText("Gauss X: -, Y: -")
            self._gauss_fit = None

    def clear_fits(self):
        """Clear line/axis fits, plots, overlays, and pending exports (keeps the cross)."""
        self._line_select_mode = False
        self._line_edit_mode = False
        self._line_points = []
        self._line_profile = None
        self._line_fit = None
        self._axis_lines = []
        self._axis_fit_results = {}
        self._axis_center_px = None
        self._gauss_fit = None
        self._pending_sections = {}
        if getattr(self, "line_plot_label", None):
            self.line_plot_label.clear()
        if getattr(self, "axis_plot_h", None):
            self.axis_plot_h.clear()
        if getattr(self, "axis_plot_v", None):
            self.axis_plot_v.clear()
        if getattr(self, "line_fit_status", None):
            self.line_fit_status.setText("Line fit: not computed.")
        if getattr(self, "axis_fit_status", None):
            self.axis_fit_status.setText("360 fit: not computed.")
        if getattr(self, "gauss_result_label", None):
            self.gauss_result_label.setText("Gauss X: -, Y: -")
        self._refresh_image_view()

    def closeEvent(self, event: QtGui.QCloseEvent):
        self.poll_timer.stop()
        self._dispose_camera()
        try:
            if self.sdk:
                self.sdk.dispose()
        except Exception:
            pass
        try:
            if getattr(self, "controls_window", None):
                self._save_window_state(self.controls_window, "controls_window")
                self.controls_window.close()
            if getattr(self, "hist_window", None):
                self._save_window_state(self.hist_window, "hist_window")
                self.hist_window.close()
            if getattr(self, "fit_window", None):
                self._save_window_state(self.fit_window, "fit_window")
                self.fit_window.close()
            if getattr(self, "calibration_window", None):
                self._save_window_state(self.calibration_window, "calibration_window")
                self.calibration_window.close()
            self._save_window_state(self, "main_window")
        except Exception:
            pass
        super().closeEvent(event)

    def resizeEvent(self, event: QtGui.QResizeEvent):
        super().resizeEvent(event)
        if self._fit_to_window:
            QtCore.QTimer.singleShot(0, self._refresh_image_view)
        else:
            QtCore.QTimer.singleShot(
                0, lambda: self._update_histogram_window(self.last_payload if self._histogram_enabled else None)
            )


def main():
    qt_app = QtWidgets.QApplication(sys.argv)
    window = CameraApp()
    window.show()
    sys.exit(qt_app.exec())


if __name__ == "__main__":
    main()
