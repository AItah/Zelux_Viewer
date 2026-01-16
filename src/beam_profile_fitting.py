"""
Beam-profile fitting utilities: Gaussian vs Lorentzian with AIC-based selection.
"""

from __future__ import annotations

import math
import os
from typing import Iterable, Literal, Optional
from pathlib import Path
import sys
import time
import numpy as np

_CUDA_DLL_HANDLES: list[object] = []

def _add_cuda_dll_paths():
    if sys.platform != "win32":
        return
    try:
        venv_root = Path(sys.executable).resolve().parent.parent
        site_packages = venv_root / "Lib" / "site-packages"
        candidates = [
            site_packages / "cupy_backends" / "cuda" / "libs",
            site_packages / "nvidia" / "cufft" / "bin",
            site_packages / "nvidia" / "cuda_runtime" / "bin",
            site_packages / "nvidia" / "curand" / "bin",
            site_packages / "nvidia" / "cuda_nvrtc" / "bin",
            site_packages / "nvidia" / "nvrtc" / "bin",
            site_packages / "nvidia" / "cublas" / "bin",
            Path(sys.prefix) / "Library" / "bin",
        ]
        path_entries = os.environ.get("PATH", "")
        path_list = [p for p in path_entries.split(os.pathsep) if p]
        path_set = {p.lower() for p in path_list}
        new_paths = []
        for path in candidates:
            if path.is_dir():
                path_str = str(path)
                handle = os.add_dll_directory(path_str)
                _CUDA_DLL_HANDLES.append(handle)
                if path_str.lower() not in path_set:
                    new_paths.append(path_str)
                    path_set.add(path_str.lower())
        if new_paths:
            os.environ["PATH"] = os.pathsep.join(new_paths + path_list)
    except Exception:
        pass

try:
    from scipy.optimize import least_squares
    from scipy import fft as scipy_fft
    from scipy import ndimage as scipy_ndimage

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - import fallback
    _HAVE_SCIPY = False
    least_squares = None  # type: ignore[assignment]
    scipy_fft = None  # type: ignore[assignment]
    scipy_ndimage = None  # type: ignore[assignment]

_add_cuda_dll_paths()

try:
    import cupy as cp  # Optional GPU acceleration
    try:
        import cupyx.scipy.ndimage as cupy_ndimage
        _HAVE_CUPY_NDIMAGE = True
    except Exception:  # pragma: no cover - optional dependency
        cupy_ndimage = None  # type: ignore[assignment]
        _HAVE_CUPY_NDIMAGE = False

    _HAVE_CUPY = True
except Exception:  # pragma: no cover - optional dependency
    _HAVE_CUPY = False
    cp = None  # type: ignore[assignment]
    cupy_ndimage = None  # type: ignore[assignment]
    _HAVE_CUPY_NDIMAGE = False

    
_GPU_FFT_AVAILABLE = _HAVE_CUPY
_GPU_FFT_ERROR: Optional[str] = None

def gaussian_2d(p, xy):
    """2D Rotated Gaussian model: [I0, x0, y0, wx, wy, theta, c, ax, ay]."""
    I0, x0, y0, wx, wy, theta, c, ax, ay = p
    x, y = xy
    
    # Rotation relative to center (x0, y0)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    x_rot = (x - x0) * cos_t - (y - y0) * sin_t
    y_rot = (x - x0) * sin_t + (y - y0) * cos_t
    
    # Gaussian + Tilted Background Plane
    g = I0 * np.exp(-2 * (x_rot**2 / (wx**2) + y_rot**2 / (wy**2)))
    bg = c + ax * x + ay * y
    return g + bg

def _disk_structure_np(radius: int) -> np.ndarray:
    r = int(max(0, radius))
    y, x = np.ogrid[-r : r + 1, -r : r + 1]
    return (x * x + y * y) <= r * r


def _disk_structure_cp(radius: int):
    r = int(max(0, radius))
    y, x = cp.ogrid[-r : r + 1, -r : r + 1]
    return (x * x + y * y) <= r * r


def _downsample_2d_np(arr: np.ndarray, factor: int) -> np.ndarray:
    factor = int(max(1, factor))
    if factor <= 1:
        return arr
    rows, cols = arr.shape
    rows2 = (rows // factor) * factor
    cols2 = (cols // factor) * factor
    if rows2 <= 0 or cols2 <= 0:
        return arr
    trimmed = arr[:rows2, :cols2]
    return trimmed.reshape(rows2 // factor, factor, cols2 // factor, factor).mean(axis=(1, 3))


def _downsample_2d_cp(arr, factor: int):
    factor = int(max(1, factor))
    if factor <= 1:
        return arr
    rows, cols = arr.shape
    rows2 = (rows // factor) * factor
    cols2 = (cols // factor) * factor
    if rows2 <= 0 or cols2 <= 0:
        return arr
    trimmed = arr[:rows2, :cols2]
    return trimmed.reshape(rows2 // factor, factor, cols2 // factor, factor).mean(axis=(1, 3))


def _map_points_to_full(
    points: list[tuple[int, int]], factor: int, rows: int, cols: int
) -> list[tuple[int, int]]:
    factor = int(max(1, factor))
    if factor <= 1:
        return points
    half = factor // 2
    mapped: list[tuple[int, int]] = []
    for r, c in points:
        rr = int(r * factor + half)
        cc = int(c * factor + half)
        if rr < 0 or cc < 0:
            continue
        if rr >= rows:
            rr = rows - 1
        if cc >= cols:
            cc = cols - 1
        mapped.append((rr, cc))
    return mapped


def _select_symmetric_points(
    cand_r: np.ndarray,
    cand_c: np.ndarray,
    cand_z: np.ndarray,
    rows: int,
    cols: int,
    crow: int,
    ccol: int,
    pair_tol: int,
    min_sep: int,
    max_pairs: int,
) -> list[tuple[int, int]]:
    if cand_r.size == 0:
        return []
    order = np.argsort(-cand_z)
    cand_r = cand_r[order]
    cand_c = cand_c[order]
    cand_z = cand_z[order]

    cand_map = np.zeros((rows, cols), dtype=bool)
    cand_map[cand_r, cand_c] = True
    z_map = np.zeros((rows, cols), dtype=np.float32)
    z_map[cand_r, cand_c] = cand_z

    pair_list: list[tuple[float, tuple[int, int], tuple[int, int]]] = []
    seen_pairs = set()
    for r, c in zip(cand_r, cand_c):
        sym_r = 2 * crow - r
        sym_c = 2 * ccol - c
        r0 = int(round(sym_r))
        c0 = int(round(sym_c))
        found = None
        for rr in range(r0 - pair_tol, r0 + pair_tol + 1):
            if rr < 0 or rr >= rows:
                continue
            for cc in range(c0 - pair_tol, c0 + pair_tol + 1):
                if cc < 0 or cc >= cols:
                    continue
                if cand_map[rr, cc]:
                    found = (rr, cc)
                    break
            if found:
                break
        if not found:
            continue
        a = (int(r), int(c))
        b = (int(found[0]), int(found[1]))
        key = tuple(sorted((a, b)))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        score = float(z_map[a] + z_map[b])
        pair_list.append((score, a, b))

    pair_list.sort(key=lambda t: t[0], reverse=True)

    min_sep = max(0, int(min_sep))
    min_sep2 = float(min_sep * min_sep)
    selected_points: list[tuple[int, int]] = []
    selected_pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []

    for _, p1, p2 in pair_list:
        if max_pairs and len(selected_pairs) >= max_pairs:
            break
        if min_sep > 0:
            too_close = False
            for sp in selected_points:
                if (p1[0] - sp[0]) ** 2 + (p1[1] - sp[1]) ** 2 < min_sep2:
                    too_close = True
                    break
                if (p2[0] - sp[0]) ** 2 + (p2[1] - sp[1]) ** 2 < min_sep2:
                    too_close = True
                    break
            if too_close:
                continue
        selected_pairs.append((p1, p2))
        selected_points.extend([p1, p2])

    return selected_points

def _build_fft_mask_from_mag(
    mag: np.ndarray,
    mask_radius: int,
    threshold_sigma: float,
    dc_radius: int = 10,
    pair_tol: int = 2,
    min_sep: int = 0,
    max_pairs: int = 80,
    analysis_scale: int = 1,
) -> np.ndarray:
    rows_full, cols_full = mag.shape
    analysis_scale = int(max(1, analysis_scale))
    mag_detect = mag
    dc_radius_ds = dc_radius
    pair_tol_ds = pair_tol
    min_sep_ds = min_sep
    if analysis_scale > 1:
        mag_detect = _downsample_2d_np(mag, analysis_scale)
        if mag_detect.shape != mag.shape:
            dc_radius_ds = int(round(dc_radius / analysis_scale)) if dc_radius > 0 else 0
            dc_radius_ds = max(0, dc_radius_ds)
            pair_tol_ds = max(0, int(round(pair_tol / analysis_scale)))
            min_sep_ds = max(0, int(round(min_sep / analysis_scale)))

    rows, cols = mag_detect.shape
    crow, ccol = rows // 2, cols // 2
    mag_log = np.log1p(np.abs(mag_detect))

    yy, xx = np.indices(mag_log.shape)
    rr = np.sqrt((xx - ccol) ** 2 + (yy - crow) ** 2).astype(np.int32)
    rr_flat = rr.ravel()
    mag_flat = mag_log.ravel()
    counts = np.bincount(rr_flat)
    sums = np.bincount(rr_flat, weights=mag_flat)
    sums_sq = np.bincount(rr_flat, weights=mag_flat * mag_flat)
    valid = counts > 0
    mean = np.zeros_like(sums)
    mean[valid] = sums[valid] / counts[valid]
    var = np.zeros_like(sums)
    var[valid] = sums_sq[valid] / counts[valid] - mean[valid] ** 2
    std = np.sqrt(np.maximum(var, 1e-12))
    nonzero_std = std[valid]
    if nonzero_std.size:
        floor_std = np.median(nonzero_std) * 0.35
        if not np.isfinite(floor_std) or floor_std <= 0:
            floor_std = 1e-6
    else:
        floor_std = 1e-6
    std = np.maximum(std, floor_std)

    z_map = (mag_log - mean[rr]) / std[rr]
    if dc_radius_ds > 0:
        z_map[crow - dc_radius_ds : crow + dc_radius_ds + 1, ccol - dc_radius_ds : ccol + dc_radius_ds + 1] = 0.0

    threshold = float(threshold_sigma)

    pad = np.pad(z_map, 1, mode="constant", constant_values=-np.inf)
    center = pad[1:-1, 1:-1]
    neigh_max = np.maximum.reduce(
        [
            pad[:-2, :-2],
            pad[:-2, 1:-1],
            pad[:-2, 2:],
            pad[1:-1, :-2],
            pad[1:-1, 2:],
            pad[2:, :-2],
            pad[2:, 1:-1],
            pad[2:, 2:],
        ]
    )
    local_max = center >= neigh_max
    candidate = (center > threshold) & local_max
    cand_r, cand_c = np.where(candidate)
    if cand_r.size == 0:
        return np.ones_like(mag)

    cand_mag = z_map[cand_r, cand_c]
    selected_points = _select_symmetric_points(
        cand_r,
        cand_c,
        cand_mag,
        rows,
        cols,
        crow,
        ccol,
        pair_tol_ds,
        min_sep_ds,
        max_pairs,
    )
    if not selected_points:
        return np.ones_like(mag)

    selected_points = _map_points_to_full(selected_points, analysis_scale, rows_full, cols_full)
    if not selected_points:
        return np.ones_like(mag)

    peaks = np.zeros((rows_full, cols_full), dtype=bool)
    points = np.array(selected_points, dtype=int)
    peaks[points[:, 0], points[:, 1]] = True
    if scipy_ndimage is None:
        return np.ones_like(mag)
    structure = _disk_structure_np(mask_radius)
    notch = scipy_ndimage.binary_dilation(peaks, structure=structure)
    crow_full, ccol_full = rows_full // 2, cols_full // 2
    if dc_radius > 0:
        notch[
            crow_full - dc_radius : crow_full + dc_radius + 1,
            ccol_full - dc_radius : ccol_full + dc_radius + 1,
        ] = False
    mask = np.ones((rows_full, cols_full), dtype=mag.dtype)
    mask[notch] = 0
    return mask


def _build_fft_mask_from_mag_gpu(
    mag_gpu,
    mask_radius: int,
    threshold_sigma: float,
    dc_radius: int = 10,
    pair_tol: int = 2,
    min_sep: int = 0,
    max_pairs: int = 80,
    analysis_scale: int = 1,
    return_cpu: bool = False,
):
    if not _HAVE_CUPY or not _HAVE_CUPY_NDIMAGE:
        raise RuntimeError("CuPy ndimage not available for GPU mask building.")
    rows_full, cols_full = mag_gpu.shape
    analysis_scale = int(max(1, analysis_scale))
    mag_detect = mag_gpu
    dc_radius_ds = dc_radius
    pair_tol_ds = pair_tol
    min_sep_ds = min_sep
    if analysis_scale > 1:
        mag_detect = _downsample_2d_cp(mag_gpu, analysis_scale)
        if mag_detect.shape != mag_gpu.shape:
            dc_radius_ds = int(round(dc_radius / analysis_scale)) if dc_radius > 0 else 0
            dc_radius_ds = max(0, dc_radius_ds)
            pair_tol_ds = max(0, int(round(pair_tol / analysis_scale)))
            min_sep_ds = max(0, int(round(min_sep / analysis_scale)))

    rows, cols = mag_detect.shape
    crow, ccol = rows // 2, cols // 2
    mag_log = cp.log1p(cp.abs(mag_detect))

    yy, xx = cp.indices(mag_log.shape)
    rr = cp.sqrt((xx - ccol) ** 2 + (yy - crow) ** 2).astype(cp.int32)
    rr_flat = rr.ravel()
    mag_flat = mag_log.ravel()
    counts = cp.bincount(rr_flat)
    sums = cp.bincount(rr_flat, weights=mag_flat)
    sums_sq = cp.bincount(rr_flat, weights=mag_flat * mag_flat)
    valid = counts > 0
    mean = cp.zeros_like(sums)
    mean[valid] = sums[valid] / counts[valid]
    var = cp.zeros_like(sums)
    var[valid] = sums_sq[valid] / counts[valid] - mean[valid] ** 2
    std = cp.sqrt(cp.maximum(var, 1e-12))
    nonzero_std = std[valid]
    if nonzero_std.size:
        floor_std = float(cp.asnumpy(cp.median(nonzero_std))) * 0.35
        if not np.isfinite(floor_std) or floor_std <= 0:
            floor_std = 1e-6
    else:
        floor_std = 1e-6
    std = cp.maximum(std, floor_std)

    z_map = (mag_log - mean[rr]) / std[rr]
    if dc_radius_ds > 0:
        z_map[crow - dc_radius_ds : crow + dc_radius_ds + 1, ccol - dc_radius_ds : ccol + dc_radius_ds + 1] = 0.0

    max_filt = cupy_ndimage.maximum_filter(z_map, size=3, mode="constant", cval=-cp.inf)
    candidate = (z_map > threshold_sigma) & (z_map >= max_filt)
    cand_r, cand_c = cp.where(candidate)
    if cand_r.size == 0:
        mask_gpu = cp.ones_like(mag_gpu, dtype=cp.float32)
        if return_cpu:
            return mask_gpu, cp.asnumpy(mask_gpu)
        return mask_gpu, None

    cand_z = z_map[cand_r, cand_c]
    cand_r_cpu = cp.asnumpy(cand_r)
    cand_c_cpu = cp.asnumpy(cand_c)
    cand_z_cpu = cp.asnumpy(cand_z)
    selected_points = _select_symmetric_points(
        cand_r_cpu,
        cand_c_cpu,
        cand_z_cpu,
        rows,
        cols,
        crow,
        ccol,
        pair_tol_ds,
        min_sep_ds,
        max_pairs,
    )
    if not selected_points:
        mask_gpu = cp.ones_like(mag_gpu, dtype=cp.float32)
        if return_cpu:
            return mask_gpu, cp.asnumpy(mask_gpu)
        return mask_gpu, None

    selected_points = _map_points_to_full(selected_points, analysis_scale, rows_full, cols_full)
    if not selected_points:
        mask_gpu = cp.ones_like(mag_gpu, dtype=cp.float32)
        if return_cpu:
            return mask_gpu, cp.asnumpy(mask_gpu)
        return mask_gpu, None

    peak_map = cp.zeros((rows_full, cols_full), dtype=cp.bool_)
    points = np.array(selected_points, dtype=int)
    peak_map[cp.asarray(points[:, 0]), cp.asarray(points[:, 1])] = True
    structure = _disk_structure_cp(mask_radius)
    notch = cupy_ndimage.binary_dilation(peak_map, structure=structure)
    crow_full, ccol_full = rows_full // 2, cols_full // 2
    if dc_radius > 0:
        notch[
            crow_full - dc_radius : crow_full + dc_radius + 1,
            ccol_full - dc_radius : ccol_full + dc_radius + 1,
        ] = False
    mask_gpu = (~notch).astype(cp.float32)
    if return_cpu:
        return mask_gpu, cp.asnumpy(mask_gpu)
    return mask_gpu, None


def _fft_clean_channel(
    channel: np.ndarray,
    mask_radius: int,
    threshold_sigma: float,
    dc_radius: int,
    pair_tol: int,
    min_sep: int,
    max_pairs: int,
    analysis_scale: int,
    return_steps: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, dict] | np.ndarray:
    timings: dict[str, float | str] = {}
    global _GPU_FFT_AVAILABLE, _GPU_FFT_ERROR
    if _HAVE_CUPY and _GPU_FFT_AVAILABLE:
        try:
            t0 = time.perf_counter()
            channel_gpu = cp.asarray(channel, dtype=cp.float32)
            cp.cuda.Stream.null.synchronize()
            t_upload = time.perf_counter()
            f = cp.fft.fftshift(cp.fft.fft2(channel_gpu))
            cp.cuda.Stream.null.synchronize()
            t_fft = time.perf_counter()
            mag_gpu = cp.abs(f)
            cp.cuda.Stream.null.synchronize()
            t_mag = time.perf_counter()
            mask_cpu = None
            if _HAVE_CUPY_NDIMAGE:
                mask_gpu, mask_cpu = _build_fft_mask_from_mag_gpu(
                    mag_gpu,
                    mask_radius,
                    threshold_sigma,
                    dc_radius=dc_radius,
                    pair_tol=pair_tol,
                    min_sep=min_sep,
                    max_pairs=max_pairs,
                    analysis_scale=analysis_scale,
                    return_cpu=return_steps,
                )
                cp.cuda.Stream.null.synchronize()
                t_mask = time.perf_counter()
                if return_steps:
                    mag = cp.asnumpy(mag_gpu)
                    t_mag_download = time.perf_counter()
                else:
                    mag = None
                    t_mag_download = None
            else:
                mag = cp.asnumpy(mag_gpu)
                t_mag_download = time.perf_counter()
                mask = _build_fft_mask_from_mag(
                    mag,
                    mask_radius,
                    threshold_sigma,
                    dc_radius=dc_radius,
                    pair_tol=pair_tol,
                    min_sep=min_sep,
                    max_pairs=max_pairs,
                    analysis_scale=analysis_scale,
                )
                t_mask = time.perf_counter()
                mask_gpu = cp.asarray(mask)
                mask_cpu = mask if return_steps else None
            f_clean = f * mask_gpu
            cleaned = cp.abs(cp.fft.ifft2(cp.fft.ifftshift(f_clean)))
            cp.cuda.Stream.null.synchronize()
            t_ifft = time.perf_counter()
            cleaned_np = cp.asnumpy(cleaned)
            t_download = time.perf_counter()
            timings = {
                "backend": "gpu",
                "upload_ms": (t_upload - t0) * 1000.0,
                "fft_ms": (t_fft - t_upload) * 1000.0,
                "mag_download_ms": (t_mag_download - t_mag) * 1000.0 if t_mag_download else 0.0,
                "mask_ms": (t_mask - t_mag) * 1000.0,
                "ifft_ms": (t_ifft - t_mask) * 1000.0,
                "download_ms": (t_download - t_ifft) * 1000.0,
            }
            if return_steps:
                return cleaned_np, mag, mask_cpu, timings
            return cleaned_np
        except Exception as exc:
            _GPU_FFT_AVAILABLE = False
            _GPU_FFT_ERROR = f"{type(exc).__name__}: {exc}"
            timings["gpu_error"] = _GPU_FFT_ERROR

    img_float = channel.astype(np.float32)
    t0 = time.perf_counter()
    f = scipy_fft.fftshift(scipy_fft.fft2(img_float, workers=-1))
    t_fft = time.perf_counter()
    mag = np.abs(f)
    mask = _build_fft_mask_from_mag(
        mag,
        mask_radius,
        threshold_sigma,
        dc_radius=dc_radius,
        pair_tol=pair_tol,
        min_sep=min_sep,
        max_pairs=max_pairs,
        analysis_scale=analysis_scale,
    )
    t_mask = time.perf_counter()
    f_clean = f * mask
    cleaned = np.abs(scipy_fft.ifft2(scipy_fft.ifftshift(f_clean), workers=-1))
    t_ifft = time.perf_counter()
    timings = {
        "backend": "cpu",
        "fft_ms": (t_fft - t0) * 1000.0,
        "mask_ms": (t_mask - t_fft) * 1000.0,
        "ifft_ms": (t_ifft - t_mask) * 1000.0,
    }
    if _GPU_FFT_ERROR:
        timings["gpu_error"] = _GPU_FFT_ERROR
    if return_steps:
        return cleaned, mag, mask, timings
    return cleaned


def fft_clean_image(
    image,
    mask_radius=8,
    threshold_sigma=5,
    dc_radius=10,
    pair_tol=2,
    min_sep=0,
    max_pairs=80,
    analysis_scale=1,
):
    """
    Fast algorithm to remove periodic interference fringes using 2D FFT.
    Identifies high-frequency peaks and applies a notch filter.
    """
    if not _HAVE_SCIPY:
        raise RuntimeError("SciPy not available. Install scipy to use fft_clean_image.")

    img = np.asarray(image)
    if img.ndim == 2:
        return _fft_clean_channel(
            img,
            mask_radius,
            threshold_sigma,
            dc_radius,
            pair_tol,
            min_sep,
            max_pairs,
            analysis_scale,
        )
    if img.ndim == 3:
        cleaned = [
            _fft_clean_channel(
                img[:, :, ch],
                mask_radius,
                threshold_sigma,
                dc_radius,
                pair_tol,
                min_sep,
                max_pairs,
                analysis_scale,
            )
            for ch in range(img.shape[2])
        ]
        return np.stack(cleaned, axis=2)
    raise ValueError("Unsupported image shape for FFT cleaning.")


def fft_clean_image_steps(
    image,
    mask_radius=8,
    threshold_sigma=5,
    dc_radius=10,
    pair_tol=2,
    min_sep=0,
    max_pairs=80,
    analysis_scale=1,
):
    """Return cleaned image and FFT debug steps (magnitude, mask, cleaned preview)."""
    if not _HAVE_SCIPY:
        raise RuntimeError("SciPy not available. Install scipy to use fft_clean_image_steps.")

    img = np.asarray(image)
    if img.ndim == 2:
        cleaned, mag, mask, timings = _fft_clean_channel(
            img,
            mask_radius,
            threshold_sigma,
            dc_radius,
            pair_tol,
            min_sep,
            max_pairs,
            analysis_scale,
            return_steps=True,
        )
        return cleaned, {"fft_mag": mag, "mask": mask, "cleaned": cleaned, "timings": timings}
    if img.ndim == 3:
        cleaned_channels = []
        mag = None
        mask = None
        cleaned_preview = None
        timings = None
        for ch in range(img.shape[2]):
            if ch == 0:
                cleaned_ch, mag, mask, timings = _fft_clean_channel(
                    img[:, :, ch],
                    mask_radius,
                    threshold_sigma,
                    dc_radius,
                    pair_tol,
                    min_sep,
                    max_pairs,
                    analysis_scale,
                    return_steps=True,
                )
                cleaned_preview = cleaned_ch
            else:
                cleaned_ch = _fft_clean_channel(
                    img[:, :, ch],
                    mask_radius,
                    threshold_sigma,
                    dc_radius,
                    pair_tol,
                    min_sep,
                    max_pairs,
                    analysis_scale,
                )
            cleaned_channels.append(cleaned_ch)
        cleaned = np.stack(cleaned_channels, axis=2)
        return cleaned, {"fft_mag": mag, "mask": mask, "cleaned": cleaned_preview, "timings": timings}
    raise ValueError("Unsupported image shape for FFT cleaning.")

def _gaussian_2d_moments(img: np.ndarray) -> Optional[np.ndarray]:
    min_val = float(np.min(img))
    weights = np.clip(img - min_val, 0.0, None)
    img_sum = float(np.sum(weights))
    if img_sum <= 0:
        return None

    yy, xx = np.indices(img.shape)
    x0 = float(np.sum(xx * weights) / img_sum)
    y0 = float(np.sum(yy * weights) / img_sum)

    dx = xx - x0
    dy = yy - y0
    cov_xx = float(np.sum(dx * dx * weights) / img_sum)
    cov_yy = float(np.sum(dy * dy * weights) / img_sum)
    cov_xy = float(np.sum(dx * dy * weights) / img_sum)

    trace = cov_xx + cov_yy
    det_term = (cov_xx - cov_yy) ** 2 + 4.0 * cov_xy * cov_xy
    root = math.sqrt(max(det_term, 0.0))
    var1 = max(0.0, 0.5 * (trace + root))
    var2 = max(0.0, 0.5 * (trace - root))

    wx = max(2.0 * math.sqrt(max(var1, 1e-12)), 1e-6)
    wy = max(2.0 * math.sqrt(max(var2, 1e-12)), 1e-6)

    theta = 0.5 * math.atan2(2.0 * cov_xy, cov_xx - cov_yy) if det_term > 0 else 0.0

    max_val = float(np.max(img))
    I0 = max(max_val - min_val, np.finfo(float).eps)

    return np.array([I0, x0, y0, wx, wy, theta, min_val, 0.0, 0.0], dtype=float)


def _fit_gaussian_2d_least_squares(img: np.ndarray, p0: np.ndarray, max_nfev: int = 20000) -> np.ndarray:
    rows, cols = img.shape
    yy, xx = np.indices(img.shape)
    span = float(max(rows, cols))
    lb = np.array([0.0, 0.0, 0.0, 1e-6, 1e-6, -0.5 * np.pi, -np.inf, -np.inf, -np.inf], dtype=float)
    ub = np.array(
        [np.inf, cols - 1.0, rows - 1.0, 2.0 * span, 2.0 * span, 0.5 * np.pi, np.inf, np.inf, np.inf],
        dtype=float,
    )

    def residuals(p):
        return (gaussian_2d(p, (xx, yy)) - img).ravel()

    res = least_squares(residuals, p0, bounds=(lb, ub), ftol=1e-4, xtol=1e-4, max_nfev=max_nfev)
    return res.x


def fit_gaussian_2d(image, method: str = "moments_ls"):
    """2D Gaussian fit with selectable method: moments or moments_ls."""
    img = np.asarray(image, dtype=float)
    if img.ndim == 3:
        img = np.mean(img, axis=2)
    if img.ndim != 2:
        raise ValueError("fit_gaussian_2d expects a 2D image.")

    method = (method or "moments_ls").strip().lower()
    p0 = _gaussian_2d_moments(img)
    if p0 is None:
        return None
    if method in ("moments", "fast"):
        return p0

    if not _HAVE_SCIPY:
        raise RuntimeError("SciPy not available. Install scipy to use fit_gaussian_2d.")
    if method in ("fast_ls", "roi_ls"):
        return _fit_gaussian_2d_least_squares(img, p0, max_nfev=300)
    return _fit_gaussian_2d_least_squares(img, p0)

Mode = Literal["auto", "gaussian", "lorentzian"]


def gaussian(p: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Gaussian model: p = [A, center, sigma, offset]."""
    A, x0, s, b = p
    return b + A * np.exp(-((x - x0) ** 2) / (2.0 * s**2))


def lorentzian(p: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Lorentzian model: p = [A, center, gamma(FWHM), offset]."""
    A, x0, g, b = p
    h = 0.5 * g
    return b + A * (h * h) / ((x - x0) ** 2 + h * h)


def _initial_guess(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """Estimate baseline, amplitude, center, and width."""
    n = x.size
    k_edge = max(1, int(round(0.1 * n)))

    y0 = float(np.median(np.concatenate([y[:k_edge], y[-k_edge:]])))

    i_max = int(np.argmax(y))
    y_max = float(y[i_max])
    A0 = max(np.finfo(float).eps, y_max - y0)
    x0 = float(x[i_max])

    span = float(np.max(x) - np.min(x))
    w0 = max(1.0, 0.1 * span)

    w = np.maximum(y - y0, 0.0)
    sw = float(np.sum(w))
    if sw > 0:
        mu = float(np.sum(x * w) / sw)
        var = float(np.sum(((x - mu) ** 2) * w) / sw)
        sig = math.sqrt(max(var, 1e-12))
        if np.isfinite(sig) and sig > 0:
            w0 = float(sig)
            x0 = mu

    return A0, x0, w0, y0


def _bounds(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    span = max(1e-12, xmax - xmin)

    lb = np.array([0.0, xmin, 1e-12, -np.inf], dtype=float)
    ub = np.array([np.inf, xmax, 2.0 * span, np.inf], dtype=float)
    return lb, ub


def _fit_one_model(model_fn, p0: np.ndarray, lb: np.ndarray, ub: np.ndarray, x: np.ndarray, y: np.ndarray):
    if not _HAVE_SCIPY:
        raise RuntimeError("SciPy not available. Install scipy to use gaussian_or_lorentzian_aic.")

    def residuals(p):
        return model_fn(p, x) - y

    res = least_squares(residuals, p0, bounds=(lb, ub), max_nfev=20000)
    p = res.x
    yhat = model_fn(p, x)
    r = yhat - y
    sse = float(np.sum(r * r))
    return p, yhat, sse


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _aic(n: int, sse: float, k: int) -> float:
    sse = max(float(sse), np.finfo(float).eps)
    return float(n * math.log(sse / n) + 2 * k)


def gaussian_or_lorentzian_aic(
    x: np.ndarray, profiles: np.ndarray, mode: Mode = "auto"
) -> list[dict] | dict:
    """Fit one or many profiles and select Gaussian vs Lorentzian via AIC."""
    return fit_profiles(x, profiles, mode=mode)


def fit_profile(x: np.ndarray, y: np.ndarray, mode: Mode = "auto") -> dict:
    """Fit single profile y(x) and return both models' metrics plus the selected one."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    good = np.isfinite(x) & np.isfinite(y)
    x = x[good]
    y = y[good]
    if x.size < 8:
        return {"status": "too_few_points", "best_model": None}

    A0, x0, w0, y0 = _initial_guess(x, y)
    lb, ub = _bounds(x)

    p0_g = np.array([A0, x0, w0, y0], dtype=float)
    pg, yhatg, sseg = _fit_one_model(gaussian, p0_g, lb, ub, x, y)
    r2g = _r2(y, yhatg)
    aicg = _aic(x.size, sseg, 4)
    rmseg = math.sqrt(sseg / x.size)
    fwhm_g = 2.0 * math.sqrt(2.0 * math.log(2.0)) * pg[2]

    gamma0 = max(1e-6, 2.0 * w0)
    p0_l = np.array([A0, x0, gamma0, y0], dtype=float)
    pl, yhatl, ssel = _fit_one_model(lorentzian, p0_l, lb, ub, x, y)
    r2l = _r2(y, yhatl)
    aicl = _aic(x.size, ssel, 4)
    rmsel = math.sqrt(ssel / x.size)
    fwhm_l = pl[2]

    mode = mode.lower()
    if mode == "gaussian":
        best = "gaussian"
    elif mode == "lorentzian":
        best = "lorentzian"
    else:
        best = "gaussian" if aicg <= aicl else "lorentzian"

    if best == "gaussian":
        p_best, yhat_best, sse_best, r2_best, rmse_best, fwhm_best = pg, yhatg, sseg, r2g, rmseg, fwhm_g
    else:
        p_best, yhat_best, sse_best, r2_best, rmse_best, fwhm_best = pl, yhatl, ssel, r2l, rmsel, fwhm_l

    return {
        "status": "ok",
        "best_model": best,
        "p_best": p_best,  # [A, center, sigma_or_gamma, offset]
        "yhat_best": yhat_best,
        "fwhm_best": fwhm_best,
        "sse_best": sse_best,
        "rmse_best": rmse_best,
        "r2_best": r2_best,
        "gaussian": {"p": pg, "yhat": yhatg, "sse": sseg, "rmse": rmseg, "r2": r2g, "aic": aicg, "fwhm": fwhm_g},
        "lorentzian": {"p": pl, "yhat": yhatl, "sse": ssel, "rmse": rmsel, "r2": r2l, "aic": aicl, "fwhm": fwhm_l},
    }


def fit_profiles(x: np.ndarray, profiles: np.ndarray, mode: Mode = "auto") -> list[dict]:
    """Fit many profiles; profiles can be (N,) or (N, M). Returns list of dicts."""
    profiles = np.asarray(profiles, dtype=float)
    if profiles.ndim == 1:
        return [fit_profile(x, profiles, mode=mode)]
    return [fit_profile(x, profiles[:, i], mode=mode) for i in range(profiles.shape[1])]
