from __future__ import annotations

import numpy as np
import pytest

from ecstartup.physics.fokker_planck import (
    QuasilinearDiffusion2V,
    ReducedFokkerPlanck2V,
    VelocityGrid,
)


def make_solver() -> ReducedFokkerPlanck2V:
    grid = VelocityGrid.uniform(41, 24, 1.2e7, 1.2e7)
    return ReducedFokkerPlanck2V(grid, cross_section_area_m2=0.2)


def test_maxwellian_normalization_and_zero_current() -> None:
    solver = make_solver()
    state = solver.maxwellian_state(1.0e17, 20.0)
    moments = solver.moments(state)
    assert moments.density_m3 == pytest.approx(1.0e17, rel=2.0e-15)
    assert abs(moments.mean_v_parallel_m_s) < 1.0e-8
    assert abs(moments.current_parallel_A) < 1.0e-8
    assert moments.temperature_eV == pytest.approx(20.0, rel=5.0e-3)


def test_closed_velocity_boundary_conserves_particles() -> None:
    solver = make_solver()
    state = solver.maxwellian_state(1.0e17, 20.0)
    updated, diagnostics = solver.advance(
        state,
        1.0e-8,
        collision_frequency_s=2.0e5,
        collision_temperature_eV=20.0,
    )
    assert solver.moments(updated).density_m3 == pytest.approx(1.0e17, rel=2.0e-14)
    assert abs(diagnostics.particle_balance_relative) < 1.0e-13
    assert diagnostics.minimum_f >= 0.0
    assert diagnostics.linear_residual_relative < 1.0e-12


def test_collision_operator_preserves_resolved_maxwellian() -> None:
    solver = make_solver()
    state = solver.maxwellian_state(1.0e17, 20.0)
    before = solver.moments(state)
    updated, _ = solver.advance(
        state,
        1.0e-6,
        collision_frequency_s=2.0e5,
        collision_temperature_eV=20.0,
    )
    after = solver.moments(updated)
    assert after.temperature_eV == pytest.approx(before.temperature_eV, rel=1.0e-10)
    assert np.linalg.norm(updated.f_e - state.f_e) / np.linalg.norm(state.f_e) < 1.0e-10


def test_collisions_reduce_parallel_drift() -> None:
    solver = make_solver()
    state = solver.maxwellian_state(1.0e17, 20.0, drift_v_parallel_m_s=8.0e5)
    before = solver.moments(state)
    updated, _ = solver.advance(
        state,
        2.0e-7,
        collision_frequency_s=1.0e6,
        collision_temperature_eV=20.0,
    )
    after = solver.moments(updated)
    assert abs(after.mean_v_parallel_m_s) < abs(before.mean_v_parallel_m_s)


def test_positive_electric_field_drives_negative_electron_velocity_and_positive_current() -> None:
    solver = make_solver()
    state = solver.maxwellian_state(1.0e17, 20.0)
    updated, _ = solver.advance(state, 1.0e-9, E_parallel_V_m=1.0)
    moments = solver.moments(updated)
    assert moments.mean_v_parallel_m_s < 0.0
    assert moments.current_density_parallel_A_m2 > 0.0


def test_opposite_rf_resonances_drive_opposite_current_signs() -> None:
    solver = make_solver()
    state = solver.maxwellian_state(1.0e17, 20.0)

    def driven_current(resonant_velocity: float) -> float:
        diffusion = QuasilinearDiffusion2V.gaussian_resonance(
            solver.grid,
            peak_D_parallel_m2_s3=1.0e19,
            resonant_v_parallel_m_s=resonant_velocity,
            resonant_v_perp_m_s=3.0e6,
            sigma_v_parallel_m_s=1.0e6,
            sigma_v_perp_m_s=1.0e6,
        )
        updated, _ = solver.advance(state, 1.0e-8, rf_diffusion=diffusion)
        return solver.moments(updated).current_parallel_A

    positive_resonance_current = driven_current(3.0e6)
    negative_resonance_current = driven_current(-3.0e6)
    assert positive_resonance_current < 0.0
    assert negative_resonance_current > 0.0
    assert positive_resonance_current == pytest.approx(-negative_resonance_current, rel=1.0e-10)


def test_invalid_diffusion_is_rejected() -> None:
    solver = make_solver()
    invalid_shape = np.zeros((2, 2))
    with pytest.raises(ValueError, match="match the velocity grid"):
        QuasilinearDiffusion2V(invalid_shape, invalid_shape).validated(solver.grid)
    negative = -np.ones(solver.grid.shape)
    with pytest.raises(ValueError, match="non-negative"):
        QuasilinearDiffusion2V(negative, np.zeros(solver.grid.shape)).validated(solver.grid)


def test_repeated_constant_coefficients_reuse_sparse_factorization() -> None:
    grid = VelocityGrid.uniform(21, 12, 5.0e6, 5.0e6)
    solver = ReducedFokkerPlanck2V(grid, cross_section_area_m2=1.0)
    state = solver.maxwellian_state(1.0e17, 10.0)
    for _ in range(3):
        state, _ = solver.advance(
            state,
            1.0e-9,
            collision_frequency_s=2.0e5,
            collision_temperature_eV=10.0,
        )
    cache = solver.linear_solver_cache_info()
    assert cache["misses"] == 1
    assert cache["hits"] == 2
    assert cache["entries"] == 1
