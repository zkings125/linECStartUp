from __future__ import annotations

import numpy as np

from ecstartup.physics.fokker_planck import ReducedFokkerPlanck2V, VelocityGrid
from ecstartup.physics.rosenbluth import NonlinearLandauRosenbluth2V
from ecstartup.physics.spatial_fokker_planck import OpenFieldLineGrid, OpenFieldLineKinetic1D2V


def _state(drift: float = 0.0):
    grid = VelocityGrid.uniform(9, 6, 3.0e6, 3.0e6)
    solver = ReducedFokkerPlanck2V(grid, cross_section_area_m2=1.0)
    return grid, solver.maxwellian_state(
        2.0e15,
        20.0,
        drift_v_parallel_m_s=drift,
    )


def test_rosenbluth_potentials_are_finite_and_positive():
    grid, state = _state()
    operator = NonlinearLandauRosenbluth2V(grid, gyroangle_points=8)
    potentials = operator.potentials(state)
    assert potentials.H_s2_m2.shape == grid.shape
    assert potentials.G_m4_s4.shape == grid.shape
    assert np.all(np.isfinite(potentials.H_s2_m2))
    assert np.all(potentials.H_s2_m2 > 0.0)
    assert np.all(potentials.G_m4_s4 > 0.0)


def test_nonlinear_landau_projection_conserves_three_invariants():
    grid, state = _state(drift=2.0e5)
    # Perturb away from a shifted Maxwellian so the nonlinear operator is nonzero.
    state.f_e[6:, 3:] *= 1.3
    operator = NonlinearLandauRosenbluth2V(grid, gyroangle_points=8)
    collision, diagnostics = operator.evaluate(state)
    assert np.linalg.norm(collision) > 0.0
    scales = np.asarray(
        [
            np.sum(np.abs(collision) * grid.phase_volume_weights_m3_s3),
            np.sum(np.abs(grid.mesh()[0] * collision) * grid.phase_volume_weights_m3_s3),
            np.sum(np.abs(0.5 * (grid.mesh()[0] ** 2 + grid.mesh()[1] ** 2) * collision) * grid.phase_volume_weights_m3_s3),
        ]
    )
    residuals = np.abs(
        [
            diagnostics.particle_moment_after_projection,
            diagnostics.momentum_moment_after_projection,
            diagnostics.energy_moment_after_projection,
        ]
    )
    assert np.all(residuals <= 2.0e-12 * np.maximum(scales, 1.0))


def test_shifted_maxwellian_has_small_landau_flux_and_positive_advance():
    grid, state = _state(drift=2.0e5)
    operator = NonlinearLandauRosenbluth2V(grid, gyroangle_points=8)
    collision, diagnostics = operator.evaluate(state)
    relative = np.linalg.norm(collision) / max(np.linalg.norm(state.f_e), 1.0)
    assert relative < 1.0e-4
    updated, step = operator.advance(state, 1.0e-10)
    assert step.minimum_f >= 0.0
    assert updated.time_s > state.time_s


def test_rosenbluth_backend_is_optional_in_1d2v_operator_split():
    spatial = OpenFieldLineGrid(
        s_m=np.linspace(0.0, 0.2, 3),
        R_m=np.linspace(0.4, 0.5, 3),
        Z_m=np.zeros(3),
        B_T=np.full(3, 0.83),
        cross_section_area_m2=np.full(3, 0.01),
    )
    grid = VelocityGrid.uniform(7, 4, 2.0e6, 2.0e6)
    solver = OpenFieldLineKinetic1D2V(spatial, grid)
    state = solver.maxwellian_state(1.0e14, 10.0, ion_density_m3=1.0e14)
    _, diagnostics = solver.advance(
        state,
        1.0e-11,
        rosenbluth_operator=NonlinearLandauRosenbluth2V(grid, gyroangle_points=4),
    )
    assert diagnostics.rosenbluth_particle_residual_max is not None
    assert diagnostics.rosenbluth_momentum_residual_max is not None
    assert diagnostics.rosenbluth_energy_residual_max is not None
