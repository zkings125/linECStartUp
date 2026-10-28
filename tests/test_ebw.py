from __future__ import annotations

import pytest

from ecstartup.config import EBWConfig, GeometryConfig, RFSourceConfig
from ecstartup.physics.ebw import GaussianDensityCloud, analyze_oxb, ordinary_cutoff_density_m3
from ecstartup.physics.ec_raytrace import ECRayPoint, ECRayTrace
from ecstartup.physics.magnetic import CircularMagneticField


def _source() -> RFSourceConfig:
    return RFSourceConfig(
        name="EC28_O",
        frequency_Hz=28.0e9,
        harmonic=1,
        launch={"R_m": 1.6, "Z_m": 0.0, "mode": "O", "polarization": "O"},
    )


def _synthetic_ray(n_parallel: float) -> ECRayTrace:
    points = []
    for index in range(101):
        R_m = 1.6 - 0.012 * index
        points.append(
            ECRayPoint(
                path_length_m=0.012 * index,
                R_m=R_m,
                Z_m=0.0,
                phi_rad=0.0,
                k_R_m_inv=-500.0,
                k_Z_m_inv=0.0,
                k_phi_m_inv=0.0,
                B_T=0.0,
                electron_density_m3=0.0,
                refractive_index=1.0,
                n_parallel=n_parallel,
                n_perpendicular=(max(0.0, 1.0 - n_parallel**2)) ** 0.5,
                resonance_mismatch=1.0,
                hamiltonian=0.0,
                wkb_scale_ratio=10.0,
            )
        )
    return ECRayTrace(
        "EC28_O", 28.0e9, 1, "O", "synthetic", "test path",
        "test", 0.0, 0.0, tuple(points),
    )


def _analyze(peak_density_m3: float, n_parallel: float):
    geometry = GeometryConfig(
        R0_m=0.5,
        minor_radius_m=0.4,
        B_t_T=0.50,
        B_vertical_T=0.0,
        R_bounds_m=(0.2, 1.8),
        Z_bounds_m=(-0.5, 0.5),
    )
    density = GaussianDensityCloud(0.5, 0.0, 0.35, 0.35, peak_density_m3, 1.0e14)
    config = EBWConfig(
        cutoff_tolerance_fraction=0.02,
        uhr_tolerance_fraction=0.02,
        minimum_ox_efficiency=0.0,
        minimum_abs_n_parallel_for_cd=0.05,
        ebw_path_step_m=0.002,
        ebw_max_length_m=1.0,
        R_bounds_m=(0.2, 1.8),
        Z_bounds_m=(-0.5, 0.5),
    )
    return analyze_oxb(
        _source(),
        _synthetic_ray(n_parallel),
        CircularMagneticField(geometry),
        density,
        geometry,
        config,
    )


def test_28_ghz_o_cutoff_density_matches_analytic_value() -> None:
    assert ordinary_cutoff_density_m3(28.0e9) == pytest.approx(9.72507e18, rel=1.0e-6)


def test_low_density_has_no_oxb_candidate() -> None:
    result = _analyze(2.0e18, 0.5)
    assert not result.available
    assert result.O_cutoff is None
    assert not result.ebwcd_candidate


def test_synthetic_overdense_path_finds_cutoff_and_uhr() -> None:
    result = _analyze(3.0e19, 0.5)
    assert result.available
    assert result.O_cutoff is not None
    assert result.UHR is not None
    assert result.conversion_separation_m is not None
    assert result.conversion_separation_m > 0.0


def test_zero_n_parallel_cannot_be_labelled_ebwcd() -> None:
    result = _analyze(3.0e19, 0.0)
    assert result.ebw_resonance_geometric_candidate
    assert not result.ebwcd_candidate
    assert result.current_drive_direction_sign is None


def test_signed_n_parallel_is_only_a_geometric_ebwcd_candidate() -> None:
    result = _analyze(3.0e19, -0.5)
    payload = result.to_dict(include_path=False)
    assert result.ebwcd_candidate
    assert result.current_drive_direction_sign == -1
    assert payload["ebwcd_assessment"] == "geometric_candidate_only"
    assert payload["combined_efficiency_estimate"] is None
    assert payload["current_drive_model"].startswith("not evaluated by analyze-oxb")
    assert "solve-fp" in payload["current_drive_model"]
