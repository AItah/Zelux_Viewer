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
from PIL import Image
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


_add_repo_dll_path()


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

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
        self.setStyleSheet("background-color: #111;")
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        pos = event.position().toPoint()
        self.mouse_moved.emit(pos.x(), pos.y())
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            pos = event.position().toPoint()
            self.mouse_clicked.emit(pos.x(), pos.y())
        super().mousePressEvent(event)


class CameraApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        os.makedirs(DATA_DIR, exist_ok=True)
        self.sdk = TLCameraSDK()
        camera_list = self.sdk.discover_available_cameras()
        if not camera_list:
            raise RuntimeError("No Thorlabs cameras detected.")
        self.camera = self.sdk.open_camera(camera_list[0])
        self.camera.frames_per_trigger_zero_for_unlimited = 0
        self.camera.image_poll_timeout_ms = 50

        self.cross_pos = None
        self.last_payload: Optional[FramePayload] = None
        self.acq_thread: Optional[ImageAcquisitionThread] = None
        self._live = False

        self.setWindowTitle(f"Live View - {self.camera.name}")
        self._build_ui()

        self.poll_timer = QtCore.QTimer(self)
        self.poll_timer.setInterval(15)
        self.poll_timer.timeout.connect(self._poll_queue)
        self.poll_timer.start()

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        self.image_label = ImageLabel()
        self.image_label.mouse_moved.connect(self.on_mouse_move)
        self.image_label.mouse_clicked.connect(self.on_mouse_click)
        layout.addWidget(
            self.image_label,
            alignment=QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop,
        )

        layout.addLayout(self._build_button_row())
        layout.addWidget(self._build_param_group())
        layout.addLayout(self._build_status_row())

    def _build_button_row(self) -> QtWidgets.QHBoxLayout:
        row = QtWidgets.QHBoxLayout()
        buttons = [
            ("Start Live", self.start_live),
            ("Stop Live", self.stop_live),
            ("Save Image", self.save_image),
            ("Load Image", self.load_image),
            ("Histogram", self.show_histogram),
            ("Clear Cross", self.clear_cross),
        ]
        for text, handler in buttons:
            btn = QtWidgets.QPushButton(text)
            btn.clicked.connect(handler)
            row.addWidget(btn)
        row.addStretch(1)
        return row

    def _build_param_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Parameters")
        grid = QtWidgets.QGridLayout(group)

        self.exposure_spin = QtWidgets.QDoubleSpinBox()
        self.exposure_spin.setRange(1, 10_000_000)
        self.exposure_spin.setDecimals(1)
        self.exposure_spin.setSingleStep(100)
        self.exposure_spin.setValue(float(self.camera.exposure_time_us))
        grid.addWidget(QtWidgets.QLabel("Exposure (us)"), 0, 0)
        grid.addWidget(self.exposure_spin, 0, 1)
        exposure_btn = QtWidgets.QPushButton("Set")
        exposure_btn.clicked.connect(self.set_exposure)
        grid.addWidget(exposure_btn, 0, 2)

        gain_min = getattr(self.camera.gain_range, "min", 0)
        gain_max = getattr(self.camera.gain_range, "max", 0)
        self.gain_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.gain_slider.setRange(int(gain_min), int(gain_max))
        self.gain_slider.setValue(int(getattr(self.camera, "gain", gain_min)))
        self.gain_slider.valueChanged.connect(self.set_gain)
        grid.addWidget(QtWidgets.QLabel("Gain"), 1, 0)
        grid.addWidget(self.gain_slider, 1, 1, 1, 2)

        self.gamma_spin = QtWidgets.QDoubleSpinBox()
        self.gamma_spin.setRange(0.1, 10.0)
        self.gamma_spin.setDecimals(2)
        self.gamma_spin.setSingleStep(0.1)
        self.gamma_spin.setValue(float(getattr(self.camera, "gamma", 1.0)))
        grid.addWidget(QtWidgets.QLabel("Gamma"), 2, 0)
        grid.addWidget(self.gamma_spin, 2, 1)
        gamma_btn = QtWidgets.QPushButton("Set")
        gamma_btn.clicked.connect(self.set_gamma)
        grid.addWidget(gamma_btn, 2, 2)

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

    def start_live(self):
        if self._live:
            return
        self._live = True
        self.status_label.setText("Starting...")
        self.camera.arm(2)
        self.camera.issue_software_trigger()
        self.acq_thread = ImageAcquisitionThread(self.camera)
        self.acq_thread.start()
        self.status_label.setText("Live")

    def stop_live(self):
        self._live = False
        if self.acq_thread:
            self.acq_thread.stop()
            self.acq_thread.join(timeout=1.0)
            self.acq_thread = None
        try:
            self.camera.disarm()
        except Exception:
            pass
        self.status_label.setText("Stopped")

    def set_exposure(self):
        try:
            self.camera.exposure_time_us = int(self.exposure_spin.value())
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to set exposure: {exc}")

    def set_gain(self, _value=None):
        try:
            self.camera.gain = int(self.gain_slider.value())
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to set gain: {exc}")

    def set_gamma(self):
        if not hasattr(self.camera, "gamma"):
            QtWidgets.QMessageBox.information(self, "Not supported", "Gamma control not supported on this camera.")
            return
        try:
            self.camera.gamma = float(self.gamma_spin.value())
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to set gamma: {exc}")

    def _poll_queue(self):
        if self.acq_thread:
            q = self.acq_thread.get_queue()
            try:
                payload: FramePayload = q.get_nowait()
                self.last_payload = payload
                self._display_frame(payload)
            except queue.Empty:
                pass

    def _display_frame(self, payload: FramePayload):
        img = payload.pil_image.copy()
        if self.cross_pos:
            img = self._draw_cross(img, self.cross_pos)
        pixmap = self._pil_to_qpixmap(img)
        self.image_label.setPixmap(pixmap)
        self.image_label.adjustSize()

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
            self.last_payload = payload
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
            self._display_frame(self.last_payload)

    def on_mouse_move(self, x: int, y: int):
        if not self.last_payload:
            self.coord_label.setText("x: -, y: -, val: -")
            return
        img = self.last_payload.np_image
        h, w = img.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            val = img[y, x]
            if isinstance(val, np.ndarray):
                val = val.tolist()
            self.coord_label.setText(f"x: {x}, y: {y}, val: {val}")
        else:
            self.coord_label.setText("x: -, y: -, val: -")

    def on_mouse_click(self, x: int, y: int):
        if not self.last_payload:
            return
        img = self.last_payload.np_image
        h, w = img.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            self.cross_pos = (x, y)
            self._display_frame(self.last_payload)

    def closeEvent(self, event: QtGui.QCloseEvent):
        self.poll_timer.stop()
        self.stop_live()
        try:
            self.camera.dispose()
        except Exception:
            pass
        try:
            self.sdk.dispose()
        except Exception:
            pass
        super().closeEvent(event)


def main():
    qt_app = QtWidgets.QApplication(sys.argv)
    window = CameraApp()
    window.show()
    sys.exit(qt_app.exec())


if __name__ == "__main__":
    main()
