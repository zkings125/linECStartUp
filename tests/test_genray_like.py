from __future__ import annotations

import numpy as np
import pytest

from ecstartup.config import GenrayLikeConfig, RFSourceConfig
from ecstartup.constants import E_CHARGE_C, M_E_KG, PI
from ecstartup.physics.ec_raytrace import ECRayPoint, ECRayTrace
from ecstartup.physics.fokker_planck import VelocityGrid
from ecstartup.physics.genray_like import build_genray_like_diffusion


def _ray_point(s_m: float, B_T: float) -> ECRayPoint:
    return ECRayPoint(
        path_length_m=s_m,
        R_m=0.6 + s_m,
        Z_m=0.0,
        phi_rad=0.0,
        k_R_m_inv=500.0,
        k_Z_m_inv=0.0,
        k_phi_m_inv=150.0,
        B_T=B_T,
        electron_density_m3=1.0e19,
        refractive_index=1.0,
        n_parallel=0.30,
        n_perpendicular=0.95,
        resonance_mismatch=0.05,
        hamiltonian=0.0,
        wkb_scale_ratio=20.0,
    )


def test_internal_genray_like_dql_is_nonnegative_asymmetric_and_power_normalized() -> None:
    frequency = 28.0e9
    resonance_B = 2.0 * PI * frequency * M_E_KG / E_CHARGE_C
    source = RFSourceConfig(
        name="EC28_O",
        frequency_Hz=frequency,
        harmonic=1,
        power_W=250.0e3,
        absorption_fraction=0.7,
        launch={"mode": "O", "polarization": "O"},
    )
    trace = ECRayTrace(
        source_name=source.name,
        frequency_Hz=frequency,
        harmonic=1,
        mode="O",
        status="domain_exit",
        reason="test ray",
        launch_direction_source="test",
        t_s=0.0,
        Ip_A=0.0,
        points=(_ray_point(0.0, 0.94 * resonance_B), _ray_point(0.02, 0.96 * resonance_B)),
    )
    grid = VelocityGrid.uniform(101, 50, 1.2e8, 1.2e8)
    target = 175.0e3
    diffusion, diagnostics = build_genray_like_diffusion(
        trace,
        source,
        grid,
        GenrayLikeConfig(resonance_width_fraction=0.03),
        density_m3=1.0e19,
        temperature_eV=2.0e3,
        volume_m3=10.0,
        major_radius_m=0.6,
        target_absorbed_power_W=target,
    )
    assert np.all(diffusion.D_parallel_m2_s3 >= 0.0)
    assert np.all(diffusion.D_perp_m2_s3 >= 0.0)
    assert np.max(diffusion.D_parallel_m2_s3 + diffusion.D_perp_m2_s3) > 0.0
    assert diagnostics.unscaled_power_moment_W_per_unit_D * diagnostics.normalization_D_m2_s3 == pytest.approx(target)
    assert diagnostics.predicted_absorption_fraction > 0.0
    # Positive N_parallel and B below the cold resonance select mainly v_parallel > 0.
    parallel_weight = np.sum(
        (diffusion.D_parallel_m2_s3 + diffusion.D_perp_m2_s3), axis=1
    )
    weighted_velocity = float(np.sum(parallel_weight * grid.v_parallel_m_s))
    assert weighted_velocity > 0.0
