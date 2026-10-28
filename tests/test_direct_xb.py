from __future__ import annotations

import numpy as np

from ecstartup.config import GeometryConfig, RFSourceConfig
from ecstartup.physics.ebw import analyze_direct_xb, ordinary_cutoff_density_m3
from ecstartup.physics.ec_raytrace import ECRayPoint, ECRayTrace
from ecstartup.physics.magnetic import CircularMagneticField


class LinearDensity:
    def __init__(self, cutoff_density_m3: float) -> None:
        self.cutoff_density_m3 = cutoff_density_m3

    def electron_density_m3(self, R_m: float, Z_m: float, t_s: float) -> float:
        return self.cutoff_density_m3 * (0.1 + Z_m)


def _x_source() -> RFSourceConfig:
    return RFSourceConfig(
        name="EC28_X",
        frequency_Hz=28.0e9,
        harmonic=1,
        launch={"R_m": 1.0, "Z_m": 0.0, "mode": "X", "polarization": "X"},
    )


def _probe() -> ECRayTrace:
    points = tuple(
        ECRayPoint(
            path_length_m=float(z), R_m=1.0, Z_m=float(z), phi_rad=0.0,
            k_R_m_inv=0.0, k_Z_m_inv=1.0, k_phi_m_inv=0.0,
            B_T=0.5, electron_density_m3=0.0, refractive_index=1.0,
            n_parallel=0.0, n_perpendicular=1.0, resonance_mismatch=1.0,
            hamiltonian=0.0, wkb_scale_ratio=10.0,
        )
        for z in np.linspace(0.0, 1.2, 241)
    )
    return ECRayTrace("EC28_X", 28.0e9, 1, "X", "probe", "test", "test", 0.0, 0.0, points)


def test_direct_xb_finds_cutoff_uhr_and_evanescent_barrier() -> None:
    geometry = GeometryConfig(
        R0_m=1.0, minor_radius_m=0.5, B_t_T=0.5, B_vertical_T=0.0,
        R_bounds_m=(0.4, 1.6), Z_bounds_m=(-0.1, 1.3),
    )
    density = LinearDensity(ordinary_cutoff_density_m3(28.0e9))
    result = analyze_direct_xb(
        _x_source(), _probe(), CircularMagneticField(geometry), density
    )
    assert result.available_candidate
    assert result.UHR is not None
    assert result.incident_cutoff_kind in {"R_cutoff", "L_cutoff"}
    assert result.evanescent_width_m is not None
    assert result.evanescent_width_m > 0.0
    assert result.fast_X_identified
    assert result.mode_conversion_verified is False
    assert result.to_dict(include_path=False)["xb_efficiency"] is None


def test_direct_xb_rejects_o_mode() -> None:
    source = _x_source().model_copy(
        update={"launch": _x_source().launch.model_copy(update={"mode": "O"})}
    )
    geometry = GeometryConfig(
        R0_m=1.0,
        minor_radius_m=0.5,
        B_t_T=0.5,
        R_bounds_m=(0.4, 1.6),
        Z_bounds_m=(-0.1, 1.3),
    )
    result = analyze_direct_xb(
        source,
        _probe(),
        CircularMagneticField(geometry),
        LinearDensity(ordinary_cutoff_density_m3(28.0e9)),
    )
    assert not result.available_candidate
