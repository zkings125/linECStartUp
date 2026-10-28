from __future__ import annotations

import numpy as np
import pytest

from ecstartup.physics.fokker_planck import VelocityGrid
from ecstartup.physics.fullwave_dql import (
    FullWaveFieldMap2D,
    FullWavePowerBudget,
    load_spatial_fullwave_dql_npz,
    project_fullwave_fundamental_dql,
    project_fullwave_harmonic_dql,
    save_fullwave_dql_npz,
)
from ecstartup.physics.spatial_fokker_planck import OpenFieldLineGrid


def _map() -> FullWaveFieldMap2D:
    R = np.linspace(0.4, 0.8, 5)
    Z = np.linspace(-0.2, 0.2, 5)
    shape = (5, 5)
    axisymmetric_volume_m3 = np.pi * (R[-1] ** 2 - R[0] ** 2) * (Z[-1] - Z[0])
    return FullWaveFieldMap2D(
        R,
        Z,
        np.full(shape, 2.0e3 + 1.0e3j),
        np.zeros(shape, dtype=complex),
        np.full(shape, 50.0 + 0.0j),
        np.full(shape, 15.0),
        np.full(shape, 1.785),
        np.full(shape, 10.0 / axisymmetric_volume_m3),
        50.0e9,
        "unit-test-FEM",
        "1.0",
        True,
        True,
        FullWavePowerBudget(100.0, 10.0, 5.0, 5.0, 5.0, 65.0, 10.0),
    ).validated()


def test_fullwave_power_budget_exposes_quantitative_oxb_efficiency():
    field = _map()
    assert field.power_budget.balance_relative == pytest.approx(0.0)
    assert field.power_budget.oxb_conversion_efficiency == pytest.approx(0.65)
    bad = FullWavePowerBudget(100.0, 10.0, 5.0, 5.0, 5.0, 50.0, 10.0)
    with pytest.raises(ValueError, match="power balance"):
        bad.validated()


def test_fullwave_field_projects_to_spatial_psd_dql_and_absorbed_power():
    spatial = OpenFieldLineGrid(
        s_m=np.linspace(0.0, 0.4, 5),
        R_m=np.linspace(0.45, 0.75, 5),
        Z_m=np.zeros(5),
        B_T=np.full(5, 1.785),
        cross_section_area_m2=np.full(5, 0.01),
    )
    velocity = VelocityGrid.uniform(17, 10, 8.0e7, 8.0e7)
    result = project_fullwave_fundamental_dql(
        _map(), spatial, velocity, time_s=0.0, resonance_width_rad_s=2.0e8
    )
    diffusion = result.diffusion
    determinant = diffusion.D_parallel_m2_s3 * diffusion.D_perp_m2_s3 - diffusion.D_cross_m2_s3**2
    assert np.min(determinant) >= -1.0e-10 * np.max(diffusion.D_parallel_m2_s3 * diffusion.D_perp_m2_s3)
    assert np.max(diffusion.D_perp_m2_s3) > 0.0
    expected_tube_power = _map().absorbed_power_density_W_m3[0, 0] * np.sum(spatial.cell_volume_m3)
    assert result.target_absorbed_power_W == pytest.approx(expected_tube_power)
    assert result.oxb_conversion_efficiency == pytest.approx(0.65)


def test_fullwave_quantitative_gate_rejects_unverified_mesh():
    field = _map()
    unverified = FullWaveFieldMap2D(**{**field.__dict__, "mesh_verified": False})
    with pytest.raises(ValueError, match="mesh verification"):
        unverified.validated()


def test_fullwave_gate_rejects_local_absorbed_power_inconsistent_with_ports():
    field = _map()
    inconsistent = FullWaveFieldMap2D(
        **{**field.__dict__, "absorbed_power_density_W_m3": 2.0 * field.absorbed_power_density_W_m3}
    )
    with pytest.raises(ValueError, match="absorbed-power density"):
        inconsistent.validated()


def test_spatial_fullwave_dql_round_trip_rejects_grid_mismatch(tmp_path):
    spatial = OpenFieldLineGrid(
        s_m=np.linspace(0.0, 0.4, 5),
        R_m=np.linspace(0.45, 0.75, 5),
        Z_m=np.zeros(5),
        B_T=np.full(5, 1.785),
        cross_section_area_m2=np.full(5, 0.01),
    )
    velocity = VelocityGrid.uniform(17, 10, 8.0e7, 8.0e7)
    field = _map()
    projection = project_fullwave_fundamental_dql(
        field, spatial, velocity, time_s=0.0, resonance_width_rad_s=2.0e8
    )
    output = tmp_path / "spatial_dql.npz"
    save_fullwave_dql_npz(output, projection, spatial, velocity, field)
    loaded = load_spatial_fullwave_dql_npz(output, spatial, velocity)
    assert np.allclose(loaded.D_cross_m2_s3, projection.diffusion.D_cross_m2_s3)
    assert loaded.target_absorbed_power_W == pytest.approx(projection.target_absorbed_power_W)

    shifted = OpenFieldLineGrid(
        spatial.s_m,
        spatial.R_m + 1.0e-3,
        spatial.Z_m,
        spatial.B_T,
        spatial.cross_section_area_m2,
    )
    with pytest.raises(ValueError, match="R_m does not match"):
        load_spatial_fullwave_dql_npz(output, shifted, velocity)


def test_second_harmonic_flr_projection_is_psd_and_power_tagged():
    spatial = OpenFieldLineGrid(
        s_m=np.linspace(0.0, 0.4, 5),
        R_m=np.linspace(0.45, 0.75, 5),
        Z_m=np.zeros(5),
        B_T=np.full(5, 1.785 / 2.0),
        cross_section_area_m2=np.full(5, 0.01),
    )
    velocity = VelocityGrid.uniform(17, 10, 8.0e7, 8.0e7)
    field = _map()
    second_harmonic_field = FullWaveFieldMap2D(
        **{**field.__dict__, "B_T": np.full(field.B_T.shape, 1.785 / 2.0)}
    )
    result = project_fullwave_harmonic_dql(
        second_harmonic_field,
        spatial,
        velocity,
        harmonic=2,
        k_perpendicular_m_inv=800.0,
        time_s=0.0,
        resonance_width_rad_s=2.0e8,
        target_absorbed_power_W=25.0,
    )
    determinant = (
        result.diffusion.D_parallel_m2_s3 * result.diffusion.D_perp_m2_s3
        - result.diffusion.D_cross_m2_s3**2
    )
    scale = np.maximum(
        result.diffusion.D_parallel_m2_s3 * result.diffusion.D_perp_m2_s3, 1.0
    )
    assert np.min(determinant / scale) >= -1.0e-12
    assert np.max(result.diffusion.D_perp_m2_s3) > 0.0
    assert result.target_absorbed_power_W == pytest.approx(25.0)
