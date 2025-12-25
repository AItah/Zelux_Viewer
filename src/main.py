"""
Thorlabs CS165CU1 live-view utility with GUI controls.

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
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog, messagebox
from typing import Optional

import numpy as np
from PIL import Image, ImageTk
from thorlabs_tsi_sdk.tl_camera import Frame, TLCamera, TLCameraSDK
from thorlabs_tsi_sdk.tl_camera_enums import SENSOR_TYPE
from thorlabs_tsi_sdk.tl_mono_to_color_processor import MonoToColorProcessorSDK

# Additional DLL path setup for repo-level dlls/{64,32}_lib
def _add_repo_dll_path():
    import sys

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

        # Color pipeline setup
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
            # Convert Bayer to 24-bit RGB
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
                # drop if UI is behind
                pass
            except Exception as exc:  # pragma: no cover - defensive
                print(f"Acquisition error: {exc}")
                break

        if self._is_color:
            self._mono_to_color_processor.dispose()
            self._mono_to_color_sdk.dispose()


class CameraApp:
    def __init__(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        self.sdk = TLCameraSDK()
        camera_list = self.sdk.discover_available_cameras()
        if not camera_list:
            raise RuntimeError("No Thorlabs cameras detected.")
        self.camera = self.sdk.open_camera(camera_list[0])
        self.camera.frames_per_trigger_zero_for_unlimited = 0
        self.camera.image_poll_timeout_ms = 50

        self.root = tk.Tk()
        self.root.title(f"Live View - {self.camera.name}")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.image_label = tk.Label(self.root)
        self.image_label.pack(side=tk.TOP, padx=5, pady=5)
        self.image_label.bind("<Motion>", self.on_mouse_move)
        self.image_label.bind("<Button-1>", self.on_mouse_click)

        self.status_var = tk.StringVar(value="Stopped")
        self.coord_var = tk.StringVar(value="x: -, y: -, val: -")

        self.cross_pos = None
        self.last_payload: Optional[FramePayload] = None
        self.photo = None

        self.acq_thread: Optional[ImageAcquisitionThread] = None
        self._live = False

        self._build_controls()
        self._poll_queue()

    def _build_controls(self):
        ctrl = tk.Frame(self.root)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        tk.Button(ctrl, text="Start Live", command=self.start_live).grid(row=0, column=0, padx=2, pady=2)
        tk.Button(ctrl, text="Stop Live", command=self.stop_live).grid(row=0, column=1, padx=2, pady=2)
        tk.Button(ctrl, text="Save Image", command=self.save_image).grid(row=0, column=2, padx=2, pady=2)
        tk.Button(ctrl, text="Load Image", command=self.load_image).grid(row=0, column=3, padx=2, pady=2)
        tk.Button(ctrl, text="Histogram", command=self.show_histogram).grid(row=0, column=4, padx=2, pady=2)
        tk.Button(ctrl, text="Clear Cross", command=self.clear_cross).grid(row=0, column=5, padx=2, pady=2)

        param_frame = tk.LabelFrame(self.root, text="Parameters")
        param_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        self.exposure_var = tk.DoubleVar(value=self.camera.exposure_time_us)
        tk.Label(param_frame, text="Exposure (us)").grid(row=0, column=0, sticky="w")
        tk.Entry(param_frame, textvariable=self.exposure_var, width=10).grid(row=0, column=1, padx=2)
        tk.Button(param_frame, text="Set", command=self.set_exposure).grid(row=0, column=2, padx=2)

        # Gain slider based on camera range
        gain_min = getattr(self.camera.gain_range, "min", 0)
        gain_max = getattr(self.camera.gain_range, "max", 0)
        self.gain_var = tk.IntVar(value=getattr(self.camera, "gain", gain_min))
        tk.Label(param_frame, text="Gain").grid(row=1, column=0, sticky="w")
        tk.Scale(param_frame, from_=gain_min, to=gain_max, orient=tk.HORIZONTAL, variable=self.gain_var,
                 command=lambda _: self.set_gain(), length=200).grid(row=1, column=1, columnspan=2, sticky="we")

        self.gamma_var = tk.DoubleVar(value=getattr(self.camera, "gamma", 1.0))
        tk.Label(param_frame, text="Gamma").grid(row=2, column=0, sticky="w")
        tk.Entry(param_frame, textvariable=self.gamma_var, width=10).grid(row=2, column=1, padx=2)
        tk.Button(param_frame, text="Set", command=self.set_gamma).grid(row=2, column=2, padx=2)

        status_frame = tk.Frame(self.root)
        status_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        tk.Label(status_frame, textvariable=self.status_var).pack(side=tk.LEFT, padx=5)
        tk.Label(status_frame, textvariable=self.coord_var).pack(side=tk.LEFT, padx=5)

    def start_live(self):
        if self._live:
            return
        self._live = True
        self.status_var.set("Starting...")
        self.camera.arm(2)
        self.camera.issue_software_trigger()
        self.acq_thread = ImageAcquisitionThread(self.camera)
        self.acq_thread.start()
        self.status_var.set("Live")

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
        self.status_var.set("Stopped")

    def set_exposure(self):
        try:
            val = float(self.exposure_var.get())
            self.camera.exposure_time_us = int(val)
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to set exposure: {exc}")

    def set_gain(self):
        try:
            self.camera.gain = int(self.gain_var.get())
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to set gain: {exc}")

    def set_gamma(self):
        if not hasattr(self.camera, "gamma"):
            messagebox.showinfo("Not supported", "Gamma control not supported on this camera.")
            return
        try:
            self.camera.gamma = float(self.gamma_var.get())
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to set gamma: {exc}")

    def _poll_queue(self):
        if self.acq_thread:
            q = self.acq_thread.get_queue()
            try:
                payload: FramePayload = q.get_nowait()
                self.last_payload = payload
                self._display_frame(payload)
            except queue.Empty:
                pass
        self.root.after(15, self._poll_queue)

    def _display_frame(self, payload: FramePayload):
        img = payload.pil_image.copy()
        if self.cross_pos:
            img = self._draw_cross(img, self.cross_pos)
        self.photo = ImageTk.PhotoImage(img)
        self.image_label.config(image=self.photo)
        self.image_label.image = self.photo

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

    def save_image(self):
        if not self.last_payload:
            messagebox.showinfo("No image", "No image to save.")
            return
        default_path = os.path.join(DATA_DIR, f"capture_{int(time.time())}.png")
        path = filedialog.asksaveasfilename(defaultextension=".png", initialfile=os.path.basename(default_path),
                                            initialdir=DATA_DIR, filetypes=[("PNG", "*.png"), ("TIFF", "*.tiff *.tif"),
                                                                            ("All files", "*.*")])
        if not path:
            return
        try:
            self.last_payload.pil_image.save(path)
            self.status_var.set(f"Saved {path}")
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to save: {exc}")

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.tif *.tiff"), ("All", "*.*")],
                                          initialdir=DATA_DIR)
        if not path:
            return
        try:
            img = Image.open(path)
            np_img = np.array(img)
            payload = FramePayload(pil_image=img, np_image=np_img, frame_count=-1)
            self.last_payload = payload
            self._display_frame(payload)
            self.status_var.set(f"Loaded {path}")
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to load image: {exc}")

    def show_histogram(self):
        if not self.last_payload:
            messagebox.showinfo("No image", "No image to analyze.")
            return
        data = self.last_payload.np_image
        if data.ndim == 3:
            data = np.mean(data, axis=2)
        hist, bins = np.histogram(data.flatten(), bins=256, range=[0, 255])
        hist_window = tk.Toplevel(self.root)
        hist_window.title("Histogram")
        canvas = tk.Canvas(hist_window, width=512, height=200, bg="white")
        canvas.pack()
        hist = hist / hist.max() if hist.max() > 0 else hist
        for i, v in enumerate(hist):
            x0 = i * 2
            x1 = x0 + 1
            y0 = 200
            y1 = 200 - int(v * 180)
            canvas.create_rectangle(x0, y0, x1, y1, fill="blue", outline="")

    def clear_cross(self):
        self.cross_pos = None
        if self.last_payload:
            self._display_frame(self.last_payload)

    def on_mouse_move(self, event):
        if not self.last_payload:
            return
        x, y = int(event.x), int(event.y)
        img = self.last_payload.np_image
        h, w = img.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            if img.ndim == 2:
                val = img[y, x]
            else:
                val = img[y, x].tolist()
            self.coord_var.set(f"x: {x}, y: {y}, val: {val}")
        else:
            self.coord_var.set("x: -, y: -, val: -")

    def on_mouse_click(self, event):
        if not self.last_payload:
            return
        x, y = int(event.x), int(event.y)
        img = self.last_payload.np_image
        h, w = img.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            self.cross_pos = (x, y)
            self._display_frame(self.last_payload)

    def on_close(self):
        self.stop_live()
        try:
            self.camera.dispose()
        except Exception:
            pass
        try:
            self.sdk.dispose()
        except Exception:
            pass
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    app = CameraApp()
    app.run()


if __name__ == "__main__":
    main()
