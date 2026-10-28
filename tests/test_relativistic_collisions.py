from __future__ import annotations

import numpy as np

from ecstartup.physics.fokker_planck import KineticState2V, RelativisticMomentumGrid
from ecstartup.physics.relativistic_collisions import RelativisticBraamsKarney2V
from ecstartup.physics.spatial_fokker_planck import (
    OpenFieldLineGrid,
    OpenFieldLineKinetic1D2V,
    SpatialKineticState,
)


def _operator() -> RelativisticBraamsKarney2V:
    grid = RelativisticMomentumGrid.uniform(7, 5, 0.6, 0.6)
    return RelativisticBraamsKarney2V(grid, gyroangle_points=6)


def test_relativistic_cee_conserves_particle_momentum_and_energy():
    operator = _operator()
    state = operator.maxwell_juttner_state(1.0e17, 2.0e4)
    up, _ = operator.grid.mesh_u()
    perturbed = KineticState2V(
        state.f_e * np.maximum(1.0 + 0.08 * up, 0.1), operator.grid, 0.0
    )
    collision, diagnostics = operator.evaluate(
        perturbed,
        electron_electron_frequency_s=2.0e4,
        electron_ion_frequency_s=0.0,
    )
    assert np.all(np.isfinite(collision))
    scale = max(float(np.sum(np.abs(collision) * operator.grid.phase_volume_weights_m3_s3)), 1.0)
    assert abs(diagnostics.particle_residual_ee) / scale < 1.0e-10
    assert abs(diagnostics.momentum_residual_ee) / scale < 1.0e-10
    assert abs(diagnostics.energy_residual_ee) / scale < 1.0e-10


def test_relativistic_cei_conserves_particles_and_energy_but_relaxes_drift():
    operator = _operator()
    state = operator.maxwell_juttner_state(1.0e17, 2.0e4)
    up, _ = operator.grid.mesh_u()
    drifted = KineticState2V(
        state.f_e * np.exp(0.25 * up), operator.grid, 0.0
    )
    collision, diagnostics = operator.evaluate(
        drifted,
        electron_electron_frequency_s=0.0,
        electron_ion_frequency_s=3.0e4,
    )
    weights = operator.grid.phase_volume_weights_m3_s3
    momentum_change = float(np.sum(up * collision * weights))
    momentum = float(np.sum(up * drifted.f_e * weights))
    scale = max(float(np.sum(np.abs(collision) * weights)), 1.0)
    assert abs(diagnostics.particle_residual_ei) / scale < 1.0e-10
    assert abs(diagnostics.energy_residual_ei) / scale < 1.0e-10
    assert momentum_change * momentum < 0.0


def test_relativistic_collision_advance_preserves_positivity():
    operator = _operator()
    state = operator.maxwell_juttner_state(1.0e17, 2.0e4)
    updated, diagnostics = operator.advance(
        state,
        1.0e-8,
        electron_electron_frequency_s=1.0e3,
        electron_ion_frequency_s=1.0e3,
    )
    assert diagnostics.minimum_f is not None and diagnostics.minimum_f >= 0.0
    assert updated.time_s == state.time_s + 1.0e-8


def test_relativistic_collision_operator_splits_into_open_field_solver():
    grid = RelativisticMomentumGrid.uniform(5, 3, 0.5, 0.5)
    collision = RelativisticBraamsKarney2V(grid, gyroangle_points=4)
    local = collision.maxwell_juttner_state(1.0e16, 2.0e4)
    spatial = OpenFieldLineGrid(
        np.asarray([0.0, 1.0e-3]),
        np.asarray([0.5, 0.5]),
        np.asarray([0.0, 1.0e-3]),
        np.asarray([1.0, 1.0]),
        np.asarray([0.01, 0.01]),
    )
    solver = OpenFieldLineKinetic1D2V(spatial, grid)
    state = SpatialKineticState(
        np.stack((local.f_e, local.f_e)),
        np.full(2, 1.0e17),
        np.full(2, 1.0e16),
        0.0,
    )
    updated, diagnostics = solver.advance(
        state,
        1.0e-12,
        braams_karney_operator=collision,
        electron_electron_collision_frequency_s=1.0e2,
        electron_ion_collision_frequency_s=1.0e2,
    )
    assert updated.time_s == 1.0e-12
    assert diagnostics.minimum_f >= 0.0
    assert diagnostics.rosenbluth_particle_residual_max is not None
