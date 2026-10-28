from __future__ import annotations

import numpy as np

from ecstartup.physics.hot_fullwave import (
    HotPlasmaDielectric,
    NedelecPMLConfig,
    solve_hot_plasma_mesh_convergence,
    solve_hot_plasma_nedelec,
)


def _constant(value: float):
    return lambda x, y: np.full(np.broadcast(x, y).shape, value, dtype=float)


def test_hot_dielectric_contains_finite_second_harmonic_flr_loss():
    model = HotPlasmaDielectric(50.0e9, 20.0, 1200.0)
    shape = (3, 4)
    _, _, _, _, channels = model.evaluate(
        np.full(shape, 2.0e18),
        np.full(shape, 50.0e9 * 2.0 * np.pi * 9.1093837139e-31 / (2.0 * 1.602176634e-19)),
        np.full(shape, 500.0),
    )
    assert np.all(np.isfinite(channels[2]))
    assert np.max(channels[2]) > 0.0


def test_nedelec_pml_solver_has_finite_field_and_algebraic_power_balance():
    dielectric = HotPlasmaDielectric(
        28.0e9,
        k_parallel_m_inv=15.0,
        k_perp_m_inv=500.0,
        collision_frequency_s=2.0e7,
    )
    config = NedelecPMLConfig(
        (-0.015, 0.015),
        (-0.012, 0.012),
        nx=25,
        ny=21,
        pml_thickness_m=0.003,
        source_center_m=(-0.006, 0.0),
        source_width_m=0.0015,
    )
    result = solve_hot_plasma_nedelec(
        dielectric,
        config,
        _constant(5.0e17),
        _constant(1.0),
        _constant(300.0),
    )
    assert result.degrees_of_freedom > 0
    assert np.all(np.isfinite(result.E_x_V_m))
    assert result.source_power_W_per_m >= 0.0
    assert result.absorbed_power_W_per_m >= 0.0
    assert result.pml_absorbed_power_W_per_m >= 0.0
    assert result.linear_residual_relative < 1.0e-8
    assert result.power_balance_relative < 1.0e-7


def test_mesh_convergence_returns_explicit_not_silent_quality_gate():
    dielectric = HotPlasmaDielectric(28.0e9, 10.0, 300.0, collision_frequency_s=1.0e7)
    config = NedelecPMLConfig(
        (-0.012, 0.012),
        (-0.010, 0.010),
        nx=17,
        ny=15,
        pml_thickness_m=0.0025,
        source_center_m=(-0.004, 0.0),
        source_width_m=0.001,
    )
    coarse, fine = solve_hot_plasma_mesh_convergence(
        dielectric,
        config,
        _constant(2.0e17),
        _constant(1.0),
        _constant(200.0),
        refinement_factor=1.2,
        tolerance=1.0,
    )
    assert fine.degrees_of_freedom > coarse.degrees_of_freedom
    assert fine.mesh_change_relative is not None
    assert fine.mesh_converged
