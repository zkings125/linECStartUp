"""Reduced trapped/passing classification and bounce averaging.

The startup interval is generally open-field and has no EFIT/GS closed flux
surface.  This module therefore exposes an explicit mirror surrogate rather
than silently claiming a CQL3D orbit solution.  A later field-line provider
can replace :func:`classify_pitch` and retain the same result contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..numerics import trapezoid


@dataclass(frozen=True)
class OrbitClassification:
    trapped_mask: np.ndarray
    passing_mask: np.ndarray
    pitch_parameter: np.ndarray
    bounce_weight: np.ndarray
    trapped_fraction: float
    passing_fraction: float
    mirror_ratio: float
    model: str = "mirror_surrogate_pre_cfs"

    def to_dict(self) -> dict[str, object]:
        return {
            "trapped_fraction": self.trapped_fraction,
            "passing_fraction": self.passing_fraction,
            "mirror_ratio": self.mirror_ratio,
            "model": self.model,
        }


def classify_pitch(
    v_parallel_m_s: np.ndarray,
    v_perp_m_s: np.ndarray,
    *,
    B_min_T: float,
    B_max_T: float,
    weights: np.ndarray | None = None,
) -> OrbitClassification:
    """Classify particles using ``lambda B_max >= 1`` mirror criterion."""

    vp = np.asarray(v_parallel_m_s, dtype=float)
    vt = np.asarray(v_perp_m_s, dtype=float)
    if vp.shape != vt.shape:
        raise ValueError("v_parallel and v_perp must have the same shape")
    if B_min_T <= 0.0 or B_max_T < B_min_T:
        raise ValueError("require 0 < B_min_T <= B_max_T")
    speed2 = vp**2 + vt**2
    lam = np.divide(vt**2, speed2 * B_min_T, out=np.zeros_like(vt), where=speed2 > 0.0)
    trapped = lam * B_max_T >= 1.0
    passing = ~trapped
    # A bounce-average weight is a dimensionless residence-time proxy.  It is
    # larger near the mirror turning boundary and remains finite at v_parallel=0.
    ratio = B_max_T / B_min_T
    bounce_weight = np.ones_like(vp)
    bounce_weight[trapped] = np.sqrt(np.maximum(lam[trapped] * B_max_T - 1.0, 0.0) + 1.0e-12)
    bounce_weight[passing] = 1.0 / np.sqrt(np.maximum(np.abs(vp[passing]) / np.sqrt(speed2[passing] + 1.0e-300), 1.0e-3))
    w = np.ones_like(vp) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != vp.shape or np.any(w < 0.0) or np.any(~np.isfinite(w)):
        raise ValueError("weights must be finite, non-negative, and match velocity shape")
    total = float(np.sum(w))
    trapped_fraction = float(np.sum(w * trapped) / total) if total > 0.0 else 0.0
    return OrbitClassification(trapped, passing, lam, bounce_weight, trapped_fraction, 1.0 - trapped_fraction, ratio)


def bounce_average(
    values: np.ndarray,
    parallel_coordinate_m: np.ndarray,
    parallel_speed_m_s: np.ndarray,
    *,
    trapped: bool = True,
) -> float:
    """Compute ``integral(A ds/|v_parallel|)/integral(ds/|v_parallel|)``.

    The caller supplies one orbit branch.  The speed floor only regularizes a
    numerical turning point; it is not a physical collision model.
    """

    A = np.asarray(values, dtype=float)
    s = np.asarray(parallel_coordinate_m, dtype=float)
    v = np.asarray(parallel_speed_m_s, dtype=float)
    if A.shape != s.shape or A.shape != v.shape or A.ndim != 1 or len(A) < 2:
        raise ValueError("values, coordinate, and speed must be one-dimensional arrays of equal length")
    if not np.all(np.diff(s) > 0.0):
        raise ValueError("parallel_coordinate_m must be strictly increasing")
    if np.any(~np.isfinite(A)) or np.any(~np.isfinite(v)):
        raise ValueError("orbit arrays must be finite")
    if trapped and np.any(v == 0.0):
        v = np.where(np.abs(v) < 1.0e-6, np.sign(v) * 1.0e-6 + (v == 0.0) * 1.0e-6, v)
    residence = 1.0 / np.maximum(np.abs(v), 1.0e-6)
    denominator = float(trapezoid(residence, s))
    if denominator <= 0.0:
        raise ValueError("orbit residence integral is non-positive")
    return float(trapezoid(A * residence, s) / denominator)


def apply_bounce_weight_to_diffusion(diffusion, classification: OrbitClassification):
    """Return a copy of a Dql tensor weighted by the orbit residence proxy."""

    from .fokker_planck import QuasilinearDiffusion2V

    if diffusion.D_parallel_m2_s3.shape != classification.bounce_weight.shape:
        raise ValueError("diffusion and orbit classification shapes must match")
    weight = classification.bounce_weight / max(float(np.nanmean(classification.bounce_weight)), 1.0e-30)
    return QuasilinearDiffusion2V(
        diffusion.D_parallel_m2_s3 * weight,
        diffusion.D_perp_m2_s3 * weight,
        diffusion.source + ";bounce_average=mirror_surrogate",
        None if diffusion.D_cross_m2_s3 is None else diffusion.D_cross_m2_s3 * weight,
    )
