"""
Beam-profile fitting utilities: Gaussian vs Lorentzian with AIC-based selection.
"""

from __future__ import annotations

import math
from typing import Iterable, Literal, Optional

import numpy as np

try:
    from scipy.optimize import least_squares

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - import fallback
    _HAVE_SCIPY = False


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
