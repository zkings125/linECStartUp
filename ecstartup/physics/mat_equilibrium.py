"""MATLAB equilibrium-shape reference reader.

This adapter intentionally extracts device geometry and equilibrium metadata
only.  It does not turn a closed-surface equilibrium into a pre-CFS field or
claim that EFIT can reconstruct the 0--100 ms startup interval.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat


def _scalar(payload: dict[str, Any], key: str) -> float:
    if key not in payload:
        raise ValueError(f"MAT equilibrium missing required field: {key}")
    value = float(np.asarray(payload[key]).reshape(-1)[0])
    if not np.isfinite(value):
        raise ValueError(f"MAT equilibrium field {key} is not finite")
    return value


@dataclass(frozen=True)
class MatEquilibriumReference:
    path: str
    B_t_T: float
    Ip_A: float
    Raxis_m: float
    Zaxis_m: float
    R_LCFS_m: float
    Z_LCFS_m: float
    psi_axis_Wb: float
    psi_LCFS_Wb: float
    R_lcfs_m: np.ndarray
    Z_lcfs_m: np.ndarray
    R_grid_m: np.ndarray
    Z_grid_m: np.ndarray
    psi_Wb: np.ndarray

    @property
    def area_m2(self) -> float:
        x, y = self.R_lcfs_m, self.Z_lcfs_m
        return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    @property
    def minor_radius_m(self) -> float:
        return float(np.sqrt(max(self.area_m2, 0.0) / np.pi))

    @property
    def aspect_ratio(self) -> float:
        return float(self.Raxis_m / max(self.minor_radius_m, 1.0e-12))

    @property
    def radial_half_width_m(self) -> float:
        return float(0.5 * (np.max(self.R_lcfs_m) - np.min(self.R_lcfs_m)))

    @property
    def vertical_half_width_m(self) -> float:
        return float(0.5 * (np.max(self.Z_lcfs_m) - np.min(self.Z_lcfs_m)))

    @property
    def elongation(self) -> float:
        return float(self.vertical_half_width_m / max(self.radial_half_width_m, 1.0e-12))

    @property
    def triangularity_proxy(self) -> float:
        # R at the topmost LCFS point provides a robust shape-only proxy when
        # the MAT file does not supply a full EFIT triangularity scalar.
        top = int(np.argmax(self.Z_lcfs_m))
        return float((self.Raxis_m - self.R_lcfs_m[top]) / max(self.radial_half_width_m, 1.0e-12))

    def summary(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "reference_kind": "closed_surface_equilibrium_shape_only",
            "B_t_T": self.B_t_T,
            "Ip_A": self.Ip_A,
            "Raxis_m": self.Raxis_m,
            "Zaxis_m": self.Zaxis_m,
            "R_LCFS_m": self.R_LCFS_m,
            "Z_LCFS_m": self.Z_LCFS_m,
            "psi_axis_Wb": self.psi_axis_Wb,
            "psi_LCFS_Wb": self.psi_LCFS_Wb,
            "area_m2": self.area_m2,
            "minor_radius_m": self.minor_radius_m,
            "aspect_ratio": self.aspect_ratio,
            "elongation": self.elongation,
            "triangularity_proxy": self.triangularity_proxy,
            "grid_shape": list(self.psi_Wb.shape),
            "startup_use": "geometry/volume/shape reference only; no pre-CFS magnetic reconstruction",
        }

    def geometry_override(self) -> dict[str, Any]:
        """Return a conservative geometry override for a sensitivity run."""

        return {
            "R0_m": self.Raxis_m,
            "minor_radius_m": self.minor_radius_m,
            "volume_m3": 2.0 * np.pi * self.Raxis_m * self.area_m2,
            "B_t_T": max(abs(self.B_t_T), 1.0e-6),
            "R_bounds_m": [float(np.min(self.R_grid_m)), float(np.max(self.R_grid_m))],
            "Z_bounds_m": [float(np.min(self.Z_grid_m)), float(np.max(self.Z_grid_m))],
        }


def read_mat_equilibrium(path: str | Path) -> MatEquilibriumReference:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    payload = loadmat(source, squeeze_me=True, struct_as_record=False)
    required = ("Ra", "Za", "gridr", "gridz", "psi_total")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"MAT equilibrium missing required arrays: {missing}")
    R_lcfs = np.asarray(payload["Ra"], dtype=float).reshape(-1)
    Z_lcfs = np.asarray(payload["Za"], dtype=float).reshape(-1)
    R_grid = np.asarray(payload["gridr"], dtype=float)
    Z_grid = np.asarray(payload["gridz"], dtype=float)
    psi = np.asarray(payload["psi_total"], dtype=float)
    if R_lcfs.size < 8 or Z_lcfs.size != R_lcfs.size:
        raise ValueError("Ra/Za must contain a closed-surface polygon with >=8 points")
    if R_grid.shape != Z_grid.shape or psi.shape != R_grid.shape:
        raise ValueError("gridr, gridz and psi_total must have the same 2-D shape")
    return MatEquilibriumReference(
        path=str(source),
        B_t_T=_scalar(payload, "Bt"),
        Ip_A=_scalar(payload, "Ip"),
        Raxis_m=_scalar(payload, "Raxis"),
        Zaxis_m=_scalar(payload, "Zaxis"),
        R_LCFS_m=_scalar(payload, "R_LCFS"),
        Z_LCFS_m=_scalar(payload, "Z_LCFS"),
        psi_axis_Wb=_scalar(payload, "psi_axis"),
        psi_LCFS_Wb=_scalar(payload, "psi_LCFS"),
        R_lcfs_m=R_lcfs,
        Z_lcfs_m=Z_lcfs,
        R_grid_m=np.asarray(np.nanmean(R_grid, axis=1), dtype=float),
        Z_grid_m=np.asarray(np.nanmean(Z_grid, axis=0), dtype=float),
        psi_Wb=psi,
    )


__all__ = ["MatEquilibriumReference", "read_mat_equilibrium"]
