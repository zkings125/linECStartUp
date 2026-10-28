from __future__ import annotations

import numpy as np
import pytest

from ecstartup.physics.fokker_planck import (
    QuasilinearDiffusion2V,
    ReducedFokkerPlanck2V,
    RelativisticFokkerPlanck2V,
    RelativisticMomentumGrid,
    VelocityGrid,
)
from ecstartup.physics.orbit_average import bounce_average, classify_pitch


def test_relativistic_grid_is_subluminal_and_normalizes_juttner_state() -> None:
    grid = RelativisticMomentumGrid.uniform(41, 24, 1.5, 1.5)
    vp, vt = grid.mesh()
    gamma = np.sqrt(1.0 + grid.mesh_u()[0] ** 2 + grid.mesh_u()[1] ** 2)
    assert np.all(gamma >= 1.0)
    assert np.all(vp**2 + vt**2 < 299_792_458.0**2)
    solver = RelativisticFokkerPlanck2V(grid, cross_section_area_m2=0.2)
    state = solver.maxwellian_state(2.0e17, 2.0e3, drift_v_parallel_m_s=8.0e5)
    moments = solver.moments(state)
    assert moments.density_m3 == pytest.approx(2.0e17, rel=2.0e-14)
    assert moments.mean_v_parallel_m_s > 0.0
    assert moments.current_parallel_A < 0.0


def test_relativistic_solver_advances_with_field_particle_closure() -> None:
    grid = RelativisticMomentumGrid.uniform(31, 18, 0.25, 0.25)
    solver = RelativisticFokkerPlanck2V(grid, cross_section_area_m2=0.2)
    state = solver.maxwellian_state(1.0e17, 2.0e3, drift_v_parallel_m_s=1.0e6)
    updated, diagnostics = solver.advance(
        state,
        1.0e-10,
        collision_frequency_s=1.0e5,
        collision_temperature_eV=2.0e3,
        momentum_conserving_field_particle=True,
    )
    assert solver.moments(updated).density_m3 == pytest.approx(1.0e17, rel=1.0e-12)
    assert diagnostics.linear_residual_relative < 1.0e-12
    assert diagnostics.field_particle_source_A != 0.0


def test_off_diagonal_diffusion_psd_and_signs() -> None:
    grid = VelocityGrid.uniform(21, 12, 5.0e6, 5.0e6)
    diagonal = np.full(grid.shape, 1.0e17)
    for sign in (-1.0, 1.0):
        diffusion = QuasilinearDiffusion2V(
            diagonal, diagonal, "test", sign * 0.5 * diagonal
        ).validated(grid)
        assert np.all(np.isfinite(diffusion.D_cross_m2_s3))
        solver = ReducedFokkerPlanck2V(grid, cross_section_area_m2=0.2)
        state = solver.maxwellian_state(1.0e17, 20.0)
        updated, diagnostics = solver.advance(state, 1.0e-10, rf_diffusion=diffusion)
        assert diagnostics.linear_residual_relative < 1.0e-12
        assert np.all(updated.f_e >= 0.0)
    with pytest.raises(ValueError, match="positive semidefinite"):
        QuasilinearDiffusion2V(diagonal, diagonal, "bad", 1.1 * diagonal).validated(grid)


def test_field_particle_term_reduces_test_particle_momentum_loss() -> None:
    grid = VelocityGrid.uniform(61, 36, 1.2e7, 1.2e7)
    solver = ReducedFokkerPlanck2V(grid, cross_section_area_m2=0.2)
    state = solver.maxwellian_state(1.0e17, 20.0, drift_v_parallel_m_s=8.0e5)
    without, _ = solver.advance(
        state, 2.0e-8, collision_frequency_s=1.0e6, collision_temperature_eV=20.0
    )
    with_term, _ = solver.advance(
        state,
        2.0e-8,
        collision_frequency_s=1.0e6,
        collision_temperature_eV=20.0,
        momentum_conserving_field_particle=True,
    )
    initial = solver.moments(state).mean_v_parallel_m_s
    loss_without = abs(solver.moments(without).mean_v_parallel_m_s - initial)
    loss_with = abs(solver.moments(with_term).mean_v_parallel_m_s - initial)
    assert loss_with < 0.02 * loss_without


def test_trapped_passing_classification_and_bounce_average() -> None:
    vp = np.asarray([[1.0, 0.1], [1.0, 0.1]])
    vt = np.asarray([[0.1, 1.0], [0.1, 1.0]])
    classification = classify_pitch(vp, vt, B_min_T=1.0, B_max_T=2.0)
    assert np.all(classification.passing_mask[:, 0])
    assert np.all(classification.trapped_mask[:, 1])
    assert classification.trapped_fraction == pytest.approx(0.5)
    s = np.linspace(0.0, 1.0, 101)
    assert bounce_average(np.full_like(s, 3.0), s, 1.0 - 0.9 * np.sin(np.pi * s) ** 2) == pytest.approx(3.0)
