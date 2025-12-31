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
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont
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


class CameraApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        os.makedirs(DATA_DIR, exist_ok=True)
        self.sdk = TLCameraSDK()
        self.camera: Optional[TLCamera] = None

        self.cross_pos = None
        self.last_payload: Optional[FramePayload] = None
        self.acq_thread: Optional[ImageAcquisitionThread] = None
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

        self._restore_window_state(self.hist_window, "hist_window")
        self._restore_window_state(self.fit_window, "fit_window")

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

        btn_row = QtWidgets.QHBoxLayout()
        # self.gauss_btn = QtWidgets.QPushButton("Gaussian Fit @ Cross")
        # self.gauss_btn.clicked.connect(self.compute_gaussian_fit)
        # btn_row.addWidget(self.gauss_btn)
        self.line_fit_btn = QtWidgets.QPushButton("2-Point Line Fit")
        self.line_fit_btn.clicked.connect(self.start_line_fit_selection)
        btn_row.addWidget(self.line_fit_btn)
        self.run_line_fit_btn = QtWidgets.QPushButton("Run Line Fit")
        self.run_line_fit_btn.clicked.connect(self.run_line_fit)
        self.run_line_fit_btn.setEnabled(False)
        btn_row.addWidget(self.run_line_fit_btn)
        self.run_360_btn = QtWidgets.QPushButton("Run 360 Fit")
        self.run_360_btn.clicked.connect(self.run_360_fit)
        self.run_360_btn.setEnabled(False)
        btn_row.addWidget(self.run_360_btn)
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

    def _build_button_row(self) -> QtWidgets.QHBoxLayout:
        row = QtWidgets.QHBoxLayout()
        self.connect_btn = QtWidgets.QPushButton("Connect Camera")
        self.connect_btn.clicked.connect(self.connect_camera)
        row.addWidget(self.connect_btn)
        buttons = [
            ("Start Live", self.start_live),
            ("Stop Live", self.stop_live),
            ("Save Image", self.save_image),
            ("Load Image", self.load_image),
            ("Clear Cross", self.clear_cross),
            # ("Zoom In", self.zoom_in),
            # ("Zoom Out", self.zoom_out),
            ("Fit", self.fit_to_window),
        ]
        for text, handler in buttons:
            btn = QtWidgets.QPushButton(text)
            btn.clicked.connect(handler)
            row.addWidget(btn)
        self.hist_checkbox = QtWidgets.QCheckBox("Histogram")
        self.hist_checkbox.stateChanged.connect(self.toggle_histogram)
        row.addWidget(self.hist_checkbox)
        self.fit_checkbox = QtWidgets.QCheckBox("Fit Window")
        self.fit_checkbox.stateChanged.connect(self.toggle_fit_window)
        row.addWidget(self.fit_checkbox)
        self.gray_checkbox = QtWidgets.QCheckBox("Grayscale")
        self.gray_checkbox.stateChanged.connect(self.toggle_grayscale)
        row.addWidget(self.gray_checkbox)
        row.addStretch(1)
        return row

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
        grid.addWidget(QtWidgets.QLabel("Exposure (us)"), 0, 0)
        grid.addWidget(self.exposure_spin, 0, 1)
        exposure_btn = QtWidgets.QPushButton("Set")
        exposure_btn.clicked.connect(self.set_exposure)
        grid.addWidget(exposure_btn, 0, 2)
        self.exposure_value_label = QtWidgets.QLabel(f"{int(exp_val)} us")
        grid.addWidget(self.exposure_value_label, 0, 3)

        self.exposure_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.exposure_slider.setRange(exp_min, exp_max)
        self.exposure_slider.setValue(int(exp_val))
        self.exposure_slider.setSingleStep(max(1, (exp_max - exp_min) // 500))
        self.exposure_slider.setMaximumWidth(220)
        self.exposure_slider.valueChanged.connect(self._on_exposure_slider)
        grid.addWidget(self.exposure_slider, 1, 0, 1, 4)

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
        self.mm_px_x_spin.valueChanged.connect(lambda v: self._update_mm_scale(v, axis="x"))
        grid.addWidget(self.mm_px_x_spin, 4, 1)

        grid.addWidget(QtWidgets.QLabel("mm/px Y"), 5, 0)
        self.mm_px_y_spin = QtWidgets.QDoubleSpinBox()
        self.mm_px_y_spin.setRange(0.0001, 10000.0)
        self.mm_px_y_spin.setDecimals(7)
        self.mm_px_y_spin.setSingleStep(0.001)
        self.mm_px_y_spin.setValue(self.mm_per_px_y)
        self.mm_px_y_spin.valueChanged.connect(lambda v: self._update_mm_scale(v, axis="y"))
        grid.addWidget(self.mm_px_y_spin, 5, 1)

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
        self._set_camera_controls_enabled(False)
        self.status_label.setText("Stopped")
        self.setWindowTitle("Live View - No camera")

    def _configure_controls_from_camera(self):
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
        elif self.camera:
            return True

        if self.sdk is None:
            try:
                self.sdk = TLCameraSDK()
            except Exception as exc:
                if show_message:
                    QtWidgets.QMessageBox.critical(self, "Error", f"Failed to initialize camera SDK: {exc}")
                self.status_label.setText("SDK init failed")
                return False

        try:
            camera_list = self.sdk.discover_available_cameras()
        except Exception as exc:
            if show_message:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to discover cameras: {exc}")
            self.status_label.setText("Camera discovery failed")
            return False

        if not camera_list:
            if show_message:
                QtWidgets.QMessageBox.information(
                    self,
                    "No camera",
                    "No Thorlabs cameras detected. Connect a camera and try again.",
                )
            self.status_label.setText("No camera detected")
            self.setWindowTitle("Live View - No camera")
            self._set_camera_controls_enabled(False)
            return False

        try:
            self.camera = self.sdk.open_camera(camera_list[0])
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

    def _ensure_camera(self, show_message: bool = True) -> bool:
        return self._connect_first_available_camera(show_message=show_message)

    def connect_camera(self):
        self._connect_first_available_camera(show_message=True, force=True)

    def start_live(self):
        if self._live:
            return
        if not self._ensure_camera():
            return
        self.status_label.setText("Starting...")
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
        if self.camera:
            try:
                self.camera.disarm()
            except Exception:
                pass
        self.status_label.setText("Stopped")

    def set_exposure(self):
        if not self._ensure_camera():
            return
        try:
            self._apply_exposure(int(self.exposure_spin.value()))
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to set exposure: {exc}")

    def set_gain(self, value=None):
        if not self._ensure_camera():
            return
        try:
            val = int(value) if value is not None else int(self.gain_slider.value())
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
        if not self.camera:
            return
        self.exposure_spin.blockSignals(True)
        self.exposure_spin.setValue(float(value))
        self.exposure_spin.blockSignals(False)
        try:
            self._apply_exposure(value)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to set exposure: {exc}")

    def _apply_exposure(self, value: int):
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
        self.exposure_value_label.setText(f"{int(value)} us")

    def _on_gain_slider(self, value: int):
        if not self.camera:
            return
        try:
            self._apply_gain(value)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to set gain: {exc}")

    def _apply_gain(self, value: int):
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
        data = payload.np_image
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
        self, x_axis: np.ndarray, profile: np.ndarray, fit_curve: np.ndarray, length_mm: float, fwhm_mm: float, waist_mm: float
    ):
        width = 540
        height = 240
        margin = 36
        if x_axis.size == 0:
            self.line_plot_label.clear()
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

        # Profile line
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
        painter.drawText(
            margin,
            margin - 6,
            f"Line profile (cyan) & fit (orange) | 2*w={waist_mm:.3f}mm, FWHM={fwhm_mm:.3f}mm",
        )
        painter.drawText(
            width // 2 - 60,
            height - 6,
            f"Distance: 0–{x_axis[-1]:.1f}px ({max(length_mm,0):.3f}mm)",
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
        painter.drawText(margin, margin - 6, f"{title} | 2*w={waist_mm:.3f}mm, FWHM={fwhm_mm:.3f}mm")
        painter.drawText(
            width // 2 - 70,
            height - 6,
            f"0–{x_axis[-1]:.1f}px ({max(length_mm,0):.3f}mm)",
        )
        painter.save()
        painter.translate(12, height // 2 + 28)
        painter.rotate(-90)
        painter.drawText(0, 0, "Intensity (gray level)")
        painter.restore()
        painter.end()
        target.setPixmap(QtGui.QPixmap.fromImage(img))

    def _enter_line_edit_mode(self):
        self._line_select_mode = False
        self._line_edit_mode = True
        self._line_profile = None
        self._line_fit = None
        self.run_line_fit_btn.setEnabled(True)
        self.run_360_btn.setEnabled(True)
        self._sync_panning_enabled()
        self.line_fit_status.setText("Line fit: drag endpoints or line, then press Run Line Fit.")
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

    def _sample_line_profile(self, p1: tuple[int, int], p2: tuple[int, int], min_samples: int = 20):
        if not self.last_payload:
            return None
        data = self.last_payload.np_image
        if data.ndim == 3:
            data = np.mean(data, axis=2)
        x1, y1 = p1
        x2, y2 = p2
        length_px = float(np.hypot(x2 - x1, y2 - y1))
        num = max(int(length_px), min_samples)
        if num <= 1:
            return None
        xs = np.linspace(x1, x2, num)
        ys = np.linspace(y1, y2, num)

        def _sample_point(ix, iy):
            if ix < 0 or iy < 0 or ix >= data.shape[1] - 1 or iy >= data.shape[0] - 1:
                ix = min(max(ix, 0), data.shape[1] - 1)
                iy = min(max(iy, 0), data.shape[0] - 1)
                return float(data[int(iy), int(ix)])
            x0 = int(np.floor(ix))
            y0 = int(np.floor(iy))
            dx_f = ix - x0
            dy_f = iy - y0
            v00 = data[y0, x0]
            v10 = data[y0, x0 + 1]
            v01 = data[y0 + 1, x0]
            v11 = data[y0 + 1, x0 + 1]
            top = v00 * (1 - dx_f) + v10 * dx_f
            bottom = v01 * (1 - dx_f) + v11 * dx_f
            return float(top * (1 - dy_f) + bottom * dy_f)

        profile = np.array([_sample_point(ix, iy) for ix, iy in zip(xs, ys)], dtype=np.float64)
        length_mm = float(np.hypot((x2 - x1) * self.mm_per_px_x, (y2 - y1) * self.mm_per_px_y))
        px_axis = np.linspace(0, length_px, num)
        return px_axis, profile, length_px, length_mm

    def _fit_1d_gaussian(self, profile: np.ndarray):
        baseline = float(np.percentile(profile, 10.0))
        weights = np.clip(profile - baseline, 0.0, None)
        if weights.sum() <= 0:
            return None
        num = profile.size
        mu = float(np.sum(weights * np.arange(num)) / weights.sum())
        var = float(np.sum(weights * (np.arange(num) - mu) ** 2) / weights.sum())
        sigma = float(np.sqrt(max(var, 1e-9)))
        amplitude = float(profile.max() - baseline)
        fit_curve = baseline + amplitude * np.exp(-0.5 * ((np.arange(num) - mu) / sigma) ** 2)
        return baseline, amplitude, mu, sigma, fit_curve


    def _prepare_display_image(self, payload: FramePayload) -> Image.Image:
        img = payload.pil_image.copy()
        if self._force_grayscale and img.mode != "L":
            img = img.convert("L")
        if self._gauss_fit:
            img = self._draw_gaussian_contour(img, self._gauss_fit)
        if len(self._line_points) == 2:
            img = self._draw_line_segment(img, self._line_points, color=(255, 215, 0))
        if self._axis_lines:
            for seg in self._axis_lines:
                p1, p2, color = seg
                img = self._draw_line_segment(img, [p1, p2], color=(color.red(), color.green(), color.blue()))
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
        data = self.last_payload.np_image
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

    def _compute_line_fit(self, p1: tuple[int, int], p2: tuple[int, int]):
        if not self.last_payload:
            return
        self.line_fit_status.setText("Line fit: computing...")
        sampled = self._sample_line_profile(p1, p2)
        if sampled is None:
            self.line_fit_status.setText("Line fit failed: points are identical.")
            return
        px_axis, profile, length_px, length_mm = sampled
        fit = self._fit_1d_gaussian(profile)
        if fit is None:
            self.line_fit_status.setText("Line fit failed: no signal above baseline.")
            return
        baseline, amplitude, mu, sigma, fit_curve = fit

        self._line_profile = (px_axis, profile)
        self._line_fit = {
            "baseline": baseline,
            "amplitude": amplitude,
            "mu": mu,
            "sigma": sigma,
            "length_px": length_px,
            "length_mm": length_mm,
        }
        px_step = length_px / max(len(px_axis) - 1, 1)
        mm_step = length_mm / max(len(px_axis) - 1, 1)
        sigma_px = sigma * px_step
        sigma_mm = sigma * mm_step
        fwhm_px = 2.3548 * sigma_px
        fwhm_mm = 2.3548 * sigma_mm
        w1e2_px = 2.0 * np.sqrt(2.0) * sigma_px
        w1e2_mm = 2.0 * np.sqrt(2.0) * sigma_mm
        mu_pos_px = (mu / max(len(px_axis) - 1, 1)) * length_px
        mu_pos_mm = (mu / max(len(px_axis) - 1, 1)) * length_mm
        self._update_line_plot(px_axis, profile, fit_curve, length_mm, fwhm_mm, w1e2_mm)
        self.line_fit_status.setText(
            "Line fit OK: "
            f"mu={mu_pos_px:.2f}px ({mu_pos_mm:.3f}mm), "
            f"sigma={sigma_px:.2f}px ({sigma_mm:.3f}mm), "
            f"FWHM={fwhm_px:.2f}px ({fwhm_mm:.3f}mm), "
            f"1/e^2={w1e2_px:.2f}px ({w1e2_mm:.3f}mm)"
        )
        self._refresh_image_view()

    def run_line_fit(self):
        if len(self._line_points) != 2:
            self.line_fit_status.setText("Line fit: select two points first.")
            return
        self._compute_line_fit(self._line_points[0], self._line_points[1])
        self._line_edit_mode = False
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
        px_axis, profile, length_px, length_mm = sampled
        fit = self._fit_1d_gaussian(profile)
        if fit is None:
            return None
        baseline, amplitude, mu, sigma, fit_curve = fit
        px_step = length_px / max(len(px_axis) - 1, 1)
        mm_step = length_mm / max(len(px_axis) - 1, 1)
        sigma_px = sigma * px_step
        sigma_mm = sigma * mm_step
        fwhm_px = 2.3548 * sigma_px
        fwhm_mm = 2.3548 * sigma_mm
        w1e2_px = 2.0 * np.sqrt(2.0) * sigma_px
        w1e2_mm = 2.0 * np.sqrt(2.0) * sigma_mm
        mu_pos_px = (mu / max(len(px_axis) - 1, 1)) * length_px
        mu_pos_mm = (mu / max(len(px_axis) - 1, 1)) * length_mm
        return {
            "name": name,
            "px_axis": px_axis,
            "profile": profile,
            "fit_curve": fit_curve,
            "length_px": length_px,
            "length_mm": length_mm,
            "sigma_px": sigma_px,
            "sigma_mm": sigma_mm,
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
        data = self.last_payload.np_image
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
        # clear manual line overlay after 360 fit
        self._line_points = []

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
        elif event.type() == QtCore.QEvent.Type.Wheel and obj in (
            getattr(self, "image_label", None),
            getattr(self, "scroll_area", None).viewport() if getattr(self, "scroll_area", None) else None,
        ):
            if self._handle_wheel_zoom(event, obj):
                return True
        elif obj is getattr(self, "image_label", None) and self._line_edit_mode:
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
        default_path = os.path.join(DATA_DIR, f"capture_{int(time.time())}.png")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Image",
            default_path,
            "PNG Files (*.png);;TIFF Files (*.tif *.tiff);;All Files (*.*)",
        )
        if not path:
            return
        try:
            self.last_payload.pil_image.save(path)
            self.status_label.setText(f"Saved {path}")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save: {exc}")

    def load_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open Image",
            DATA_DIR,
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
            self.image_label.set_panning_enabled(not self._fit_to_window and not self._line_edit_mode)

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
        self.line_fit_status.setText("Line fit: click first point...")
        self.line_plot_label.clear()
        self.run_line_fit_btn.setEnabled(False)
        self.run_360_btn.setEnabled(False)
        self.axis_fit_status.setText("360 fit: not computed.")
        self.axis_plot_h.clear()
        self.axis_plot_v.clear()
        self._refresh_image_view()

    def on_mouse_move(self, x: int, y: int):
        if not self.last_payload:
            self.coord_label.setText("x: -, y: -, val: -")
            return
        img = self.last_payload.np_image
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
                gray_val = int(self.last_payload.np_image[src_y, src_x])
            elif img.ndim == 3:
                gray_val = int(np.mean(self.last_payload.np_image[src_y, src_x]))
            gray_suffix = f", gray: {gray_val}" if gray_val is not None else ""
            mm_x = src_x * self.mm_per_px_x
            mm_y = src_y * self.mm_per_px_y
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
        img = self.last_payload.np_image
        h, w = img.shape[:2]
        src_x = int(x / scale)
        src_y = int(y / scale)
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
            gray = img
            if img.ndim == 3:
                gray = np.mean(img, axis=2)
            search_radius = 6
            xs = slice(max(0, src_x - search_radius), min(w, src_x + search_radius + 1))
            ys = slice(max(0, src_y - search_radius), min(h, src_y + search_radius + 1))
            patch = gray[ys, xs]
            max_x, max_y = src_x, src_y
            if patch.size > 0:
                rel_y, rel_x = np.unravel_index(np.argmax(patch), patch.shape)
                max_x = xs.start + rel_x
                max_y = ys.start + rel_y
            self.cross_pos = (max_x, max_y)
            self._refresh_image_view()
            self.gauss_result_label.setText("Gauss X: -, Y: -")
            self._gauss_fit = None

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
