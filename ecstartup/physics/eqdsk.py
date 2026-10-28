"""Minimal, auditable EFIT G-EQDSK reader for static axisymmetric fields.

The reader intentionally stops at the standard equilibrium arrays and does not
interpret vendor-specific footer records.  It exposes the magnetic grid,
``psi(R,Z)``, ``F(psi)=R B_phi``, plasma boundary and limiter.  A static
snapshot can be converted to :class:`TabulatedMagneticField`; time evolution
must be supplied by an external snapshot/coil provider.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import re
from typing import Any

import numpy as np

from .tabulated_field import TabulatedMagneticField


_FLOAT_WIDTH = 16


def _fixed_width_values(line: str) -> list[float]:
    values: list[float] = []
    for start in range(0, len(line), _FLOAT_WIDTH):
        field = line[start : start + _FLOAT_WIDTH]
        if field.strip():
            values.append(float(field))
    return values


def _header_dimensions(line: str) -> tuple[int, int]:
    match = re.search(r"(\d+)\s+(\d+)\s*$", line)
    if match is None:
        raise ValueError("G-EQDSK 首行未找到 nw/nh 网格尺寸")
    return int(match.group(1)), int(match.group(2))


@dataclass(frozen=True)
class GFileData:
    """Parsed standard G-EQDSK equilibrium snapshot in SI units."""

    path: str
    header: str
    nw: int
    nh: int
    rdim_m: float
    zdim_m: float
    rcentr_m: float
    rleft_m: float
    zmid_m: float
    rmaxis_m: float
    zmaxis_m: float
    simag_Wb: float
    sibry_Wb: float
    bcentr_T: float
    current_A: float
    fpol_Tm: np.ndarray
    pressure_Pa: np.ndarray
    ffprim_T: np.ndarray
    pprime_Pa_per_Wb: np.ndarray
    psi_Wb: np.ndarray
    qpsi: np.ndarray
    boundary_RZ_m: np.ndarray
    limiter_RZ_m: np.ndarray

    @property
    def R_grid_m(self) -> np.ndarray:
        return np.linspace(self.rleft_m, self.rleft_m + self.rdim_m, self.nw)

    @property
    def Z_grid_m(self) -> np.ndarray:
        return np.linspace(self.zmid_m - 0.5 * self.zdim_m, self.zmid_m + 0.5 * self.zdim_m, self.nh)

    @property
    def boundary_area_m2(self) -> float:
        points = self.boundary_RZ_m
        if len(points) < 3:
            return 0.0
        x = points[:, 0]
        y = points[:, 1]
        return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    @property
    def equivalent_minor_radius_m(self) -> float:
        return math.sqrt(max(self.boundary_area_m2, 0.0) / math.pi)

    @property
    def aspect_ratio(self) -> float:
        return float(self.rmaxis_m / max(self.equivalent_minor_radius_m, 1.0e-12))

    @property
    def radial_half_width_m(self) -> float:
        return float(0.5 * (np.max(self.boundary_RZ_m[:, 0]) - np.min(self.boundary_RZ_m[:, 0])))

    @property
    def vertical_half_width_m(self) -> float:
        return float(0.5 * (np.max(self.boundary_RZ_m[:, 1]) - np.min(self.boundary_RZ_m[:, 1])))

    @property
    def elongation(self) -> float:
        return float(self.vertical_half_width_m / max(self.radial_half_width_m, 1.0e-12))

    @property
    def triangularity_proxy(self) -> float:
        top = int(np.argmax(self.boundary_RZ_m[:, 1]))
        bottom = int(np.argmin(self.boundary_RZ_m[:, 1]))
        mid_extreme_R = 0.5 * (
            float(self.boundary_RZ_m[top, 0]) + float(self.boundary_RZ_m[bottom, 0])
        )
        return float((self.rmaxis_m - mid_extreme_R) / max(self.radial_half_width_m, 1.0e-12))

    @property
    def volume_m3(self) -> float:
        return float(2.0 * math.pi * self.rmaxis_m * self.boundary_area_m2)

    @property
    def Bphi_T(self) -> np.ndarray:
        """Interpolate F(psi) and return toroidal field on the R,Z grid."""

        coordinate = np.linspace(0.0, 1.0, self.nw)
        normalized = (self.psi_Wb - self.simag_Wb) / (self.sibry_Wb - self.simag_Wb)
        normalized = np.clip(normalized, 0.0, 1.0)
        F = np.interp(normalized.ravel(), coordinate, self.fpol_Tm).reshape(normalized.shape)
        R = self.R_grid_m[:, None]
        return F / np.maximum(R, 1.0e-9)

    def field_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``B_R, B_Z, B_phi`` reconstructed from the equilibrium map."""

        dpsi_dR = np.gradient(self.psi_Wb, self.R_grid_m, axis=0, edge_order=2)
        dpsi_dZ = np.gradient(self.psi_Wb, self.Z_grid_m, axis=1, edge_order=2)
        R = self.R_grid_m[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            B_R = -dpsi_dZ / R
            B_Z = dpsi_dR / R
        B_R = np.nan_to_num(B_R, nan=0.0, posinf=0.0, neginf=0.0)
        B_Z = np.nan_to_num(B_Z, nan=0.0, posinf=0.0, neginf=0.0)
        return B_R, B_Z, self.Bphi_T

    def to_field_provider(self, *, boundary: str = "clip", configuration: str = "EXL50U_gfile") -> TabulatedMagneticField:
        """Convert this static snapshot to the project magnetic provider."""

        B_R, B_Z, B_phi = self.field_arrays()
        from ..config import GeometryConfig

        geometry = GeometryConfig(
            R0_m=self.rmaxis_m,
            minor_radius_m=max(self.equivalent_minor_radius_m, 1.0e-3),
            volume_m3=max(self.volume_m3, 1.0e-9),
            connection_length_m=5.0,
            B_t_T=max(abs(self.bcentr_T), 1.0e-6),
            B_vertical_T=0.0,
            # GeometryConfig keeps this positive for the legacy topology
            # threshold; the gfile provider itself does not add penetration
            # field to the reconstructed equilibrium.
            B_penetration_T=1.0e-9,
            R_bounds_m=(float(self.R_grid_m[0]), float(self.R_grid_m[-1])),
            Z_bounds_m=(float(self.Z_grid_m[0]), float(self.Z_grid_m[-1])),
        )
        return TabulatedMagneticField(
            self.R_grid_m,
            self.Z_grid_m,
            B_R,
            B_Z,
            B_phi,
            psi_Wb=self.psi_Wb,
            geometry=geometry,
            configuration=configuration,
            boundary=boundary,
            edge_R_m=float(np.max(self.boundary_RZ_m[:, 0])),
            edge_Z_m=float(self.zmaxis_m),
        )

    def summary(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "header": self.header,
            "nw": self.nw,
            "nh": self.nh,
            "rdim_m": self.rdim_m,
            "zdim_m": self.zdim_m,
            "rleft_m": self.rleft_m,
            "zmid_m": self.zmid_m,
            "rmaxis_m": self.rmaxis_m,
            "zmaxis_m": self.zmaxis_m,
            "simag_Wb": self.simag_Wb,
            "sibry_Wb": self.sibry_Wb,
            "bcentr_T": self.bcentr_T,
            "current_A": self.current_A,
            "boundary_count": int(len(self.boundary_RZ_m)),
            "limiter_count": int(len(self.limiter_RZ_m)),
            "boundary_area_m2": self.boundary_area_m2,
            "equivalent_minor_radius_m": self.equivalent_minor_radius_m,
            "aspect_ratio": self.aspect_ratio,
            "elongation": self.elongation,
            "triangularity_proxy": self.triangularity_proxy,
            "volume_m3": self.volume_m3,
            "psi_axis_consistency_Wb": float(self.psi_Wb[np.argmin(abs(self.R_grid_m - self.rmaxis_m)), np.argmin(abs(self.Z_grid_m - self.zmaxis_m))]),
        }


def read_gfile(path: str | Path) -> GFileData:
    """Read a standard EFIT/G-EQDSK file."""

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    lines = source.read_text(encoding="ascii", errors="strict").splitlines()
    if len(lines) < 6:
        raise ValueError("G-EQDSK 文件过短")
    nw, nh = _header_dimensions(lines[0])
    header_values: list[float] = []
    for line in lines[1:5]:
        header_values.extend(_fixed_width_values(line))
    if len(header_values) != 20:
        raise ValueError(f"G-EQDSK 头部应有 20 个数，实际为 {len(header_values)}")

    values: list[float] = []
    stop_index: int | None = None
    nbbbs = nlim = 0
    for index, line in enumerate(lines[5:], start=5):
        tokens = line.strip().split()
        if len(tokens) == 2 and all(token.lstrip("+-").isdigit() for token in tokens):
            nbbbs, nlim = (int(token) for token in tokens)
            stop_index = index
            break
        try:
            values.extend(_fixed_width_values(line))
        except ValueError as exc:
            raise ValueError(f"G-EQDSK 数据行 {index + 1} 不是标准 16 字符浮点格式") from exc
    if stop_index is None:
        raise ValueError("G-EQDSK 未找到 nbbbs/limitr 行")
    expected = 5 * nw + nw * nh
    if len(values) < expected:
        raise ValueError(f"G-EQDSK 数组不完整，需要至少 {expected} 个数，实际 {len(values)}")
    if nbbbs < 0 or nlim < 0:
        raise ValueError("G-EQDSK nbbbs/limitr 不能为负")
    remaining: list[float] = []
    for line in lines[stop_index + 1 :]:
        if len(remaining) >= 2 * (nbbbs + nlim):
            break
        for start in range(0, len(line), _FLOAT_WIDTH):
            field = line[start : start + _FLOAT_WIDTH]
            if field.strip():
                try:
                    remaining.append(float(field))
                except ValueError:
                    break
    boundary_count = 2 * nbbbs
    limiter_count = 2 * nlim
    if len(remaining) < boundary_count + limiter_count:
        raise ValueError("G-EQDSK boundary/limiter 坐标不完整")

    data = np.asarray(values, dtype=float)
    offset = 0
    fpol = data[offset : offset + nw]; offset += nw
    pressure = data[offset : offset + nw]; offset += nw
    ffprim = data[offset : offset + nw]; offset += nw
    pprime = data[offset : offset + nw]; offset += nw
    # EFIT writes psi with Z as the fastest varying index.  Transpose to the
    # provider convention (R first, Z second).
    psi = data[offset : offset + nw * nh].reshape((nh, nw)).T; offset += nw * nh
    qpsi = data[offset : offset + nw]
    coords = np.asarray(remaining[: boundary_count + limiter_count], dtype=float)
    boundary = coords[:boundary_count].reshape((nbbbs, 2)) if nbbbs else np.empty((0, 2))
    limiter = coords[boundary_count:].reshape((nlim, 2)) if nlim else np.empty((0, 2))
    return GFileData(
        path=str(source), header=lines[0], nw=nw, nh=nh,
        rdim_m=header_values[0], zdim_m=header_values[1], rcentr_m=header_values[2],
        rleft_m=header_values[3], zmid_m=header_values[4], rmaxis_m=header_values[5],
        zmaxis_m=header_values[6], simag_Wb=header_values[7], sibry_Wb=header_values[8],
        bcentr_T=header_values[9], current_A=header_values[10], fpol_Tm=fpol,
        pressure_Pa=pressure, ffprim_T=ffprim, pprime_Pa_per_Wb=pprime,
        psi_Wb=psi, qpsi=qpsi, boundary_RZ_m=boundary, limiter_RZ_m=limiter,
    )


__all__ = ["GFileData", "read_gfile"]
