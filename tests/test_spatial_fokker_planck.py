from __future__ import annotations

import numpy as np
import pytest

from ecstartup.physics.fokker_planck import RelativisticMomentumGrid, VelocityGrid
from ecstartup.physics.spatial_fokker_planck import (
    OpenFieldLineGrid,
    OpenFieldLineKinetic1D2V,
    SpatialQuasilinearDiffusion,
)


def _solver():
    spatial = OpenFieldLineGrid(
        s_m=np.linspace(0.0, 1.0, 5),
        R_m=np.linspace(0.4, 0.6, 5),
        Z_m=np.zeros(5),
        B_T=np.full(5, 0.8),
        cross_section_area_m2=np.full(5, 0.02),
    )
    velocity = VelocityGrid.uniform(11, 8, 2.0e6, 2.0e6)
    return OpenFieldLineKinetic1D2V(spatial, velocity)


def test_1d2v_streaming_is_positive_and_loses_particles_at_open_ends():
    solver = _solver()
    state = solver.maxwellian_state(1.0e16, 5.0)
    updated, diagnostics = solver.advance(state, 2.0e-7)
    assert diagnostics.streaming_substeps >= 2
    assert diagnostics.wall_particle_loss > 0.0
    assert diagnostics.particle_inventory_after < diagnostics.particle_inventory_before
    assert diagnostics.minimum_f >= 0.0
    assert updated.f_e.shape == (solver.spatial.n_s, *solver.velocity.shape)


def test_1d2v_ionization_couples_neutral_ion_and_electron_sources():
    solver = _solver()
    state = solver.maxwellian_state(
        1.0e14,
        2.0,
        neutral_density_m3=1.0e19,
        ion_density_m3=1.0e14,
    )
    updated, diagnostics = solver.advance(
        state,
        1.0e-9,
        ionization_rate_coefficient_m3_s=1.0e-14,
    )
    assert diagnostics.ionization_particles > 0.0
    assert np.all(updated.neutral_density_m3 < state.neutral_density_m3)
    assert np.all(updated.ion_density_m3 > state.ion_density_m3)


def test_spatial_dql_is_power_normalized_and_reports_momentum():
    solver = _solver()
    state = solver.maxwellian_state(2.0e16, 10.0)
    shape = (solver.spatial.n_s, *solver.velocity.shape)
    parallel = np.zeros(shape)
    perpendicular = np.full(shape, 2.0e16)
    cross = np.zeros(shape)
    diffusion = SpatialQuasilinearDiffusion(
        parallel,
        perpendicular,
        cross,
        time_s=0.0,
        source="unit_test",
    ).validated(solver.spatial, solver.velocity)
    base_power, _ = solver._rf_response(state, diffusion, 1.0e-9)
    assert base_power > 0.0
    target = 1.7 * base_power
    normalized, scale, power, momentum = solver.normalize_dql_to_power(
        state,
        diffusion,
        1.0e-9,
        target,
    )
    assert scale > 0.0
    assert power == pytest.approx(target, rel=5.0e-4)
    assert abs(momentum) < 1.0e-8
    assert normalized.target_absorbed_power_W == pytest.approx(target)


def test_spatial_dql_reports_particle_conservation():
    solver = _solver()
    state = solver.maxwellian_state(2.0e16, 10.0)
    shape = (solver.spatial.n_s, *solver.velocity.shape)
    diffusion = SpatialQuasilinearDiffusion(
        np.zeros(shape), np.full(shape, 2.0e16), np.zeros(shape), 0.0, "unit_test"
    )
    _, diagnostics = solver.advance(state, 1.0e-9, rf_diffusion=diffusion)
    assert diagnostics.rf_particle_balance_relative is not None
    assert diagnostics.rf_particle_balance_relative < 1.0e-8


def test_1d2v_relativistic_grid_uses_relativistic_energy_and_solver():
    spatial = OpenFieldLineGrid(
        s_m=np.linspace(0.0, 0.2, 3),
        R_m=np.linspace(0.4, 0.5, 3),
        Z_m=np.zeros(3),
        B_T=np.full(3, 0.83),
        cross_section_area_m2=np.full(3, 0.01),
    )
    velocity = RelativisticMomentumGrid.uniform(9, 6, 0.2, 0.2)
    solver = OpenFieldLineKinetic1D2V(spatial, velocity)
    state = solver.maxwellian_state(1.0e14, 2.0e3)
    updated, diagnostics = solver.advance(state, 1.0e-11)
    assert updated.f_e.shape == (3, 9, 6)
    assert diagnostics.minimum_f >= 0.0


def test_spatial_non_diagonal_dql_enforces_positive_semidefinite_tensor():
    solver = _solver()
    shape = (solver.spatial.n_s, *solver.velocity.shape)
    ones = np.ones(shape)
    valid = SpatialQuasilinearDiffusion(ones, ones, 0.5 * ones, 0.0, "test")
    valid.validated(solver.spatial, solver.velocity)
    invalid = SpatialQuasilinearDiffusion(ones, ones, 2.0 * ones, 0.0, "test")
    with pytest.raises(ValueError, match="positive semidefinite"):
        invalid.validated(solver.spatial, solver.velocity)
