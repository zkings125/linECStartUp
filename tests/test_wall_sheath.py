from __future__ import annotations

import numpy as np
import pytest

from ecstartup.physics.fokker_planck import VelocityGrid
from ecstartup.physics.spatial_fokker_planck import OpenFieldLineGrid, OpenFieldLineKinetic1D2V
from ecstartup.physics.wall_sheath import QuasineutralAmbipolarSheath1D


def test_floating_bohm_sheath_balances_electron_and_ion_current():
    closure = QuasineutralAmbipolarSheath1D()
    s = np.linspace(0.0, 1.0, 11)
    result = closure.solve(
        s,
        np.full_like(s, 1.0e17),
        np.full_like(s, 1.0e17),
        np.full_like(s, 10.0),
        ion_temperature_eV=0.2,
    )
    assert result.sheath_drop_left_V > 0.0
    assert result.sheath_drop_right_V > 0.0
    assert result.current_residual_left_A_m2 == pytest.approx(0.0, abs=1.0e-10)
    assert result.current_residual_right_A_m2 == pytest.approx(0.0, abs=1.0e-10)
    assert np.max(np.abs(result.E_parallel_V_m)) < 1.0e-10


def test_presheath_field_matches_isothermal_boltzmann_gradient():
    closure = QuasineutralAmbipolarSheath1D()
    s = np.linspace(0.0, 1.0, 101)
    density = 1.0e17 * np.exp(0.4 * s)
    result = closure.solve(s, density, density, 5.0)
    assert np.median(result.E_parallel_V_m[2:-2]) == pytest.approx(-2.0, rel=2.0e-4)
    assert result.quasineutrality_relative == pytest.approx(0.0)


def test_kinetic_sheath_barrier_reduces_open_end_electron_loss():
    spatial = OpenFieldLineGrid(
        s_m=np.linspace(0.0, 1.0, 5),
        R_m=np.linspace(0.4, 0.6, 5),
        Z_m=np.zeros(5),
        B_T=np.full(5, 0.83),
        cross_section_area_m2=np.full(5, 0.02),
    )
    velocity = VelocityGrid.uniform(21, 12, 2.0e6, 2.0e6)
    solver = OpenFieldLineKinetic1D2V(spatial, velocity)
    state = solver.maxwellian_state(
        1.0e17,
        10.0,
        ion_density_m3=1.0e17,
    )
    _, absorbing = solver.advance(state, 2.0e-8)
    _, sheathed = solver.advance(
        state,
        2.0e-8,
        sheath_closure=QuasineutralAmbipolarSheath1D(),
    )
    assert sheathed.wall_particle_loss < absorbing.wall_particle_loss
    assert sheathed.sheath_drop_left_V > 0.0
    assert sheathed.sheath_current_residual_max_A_m2 < 1.0e-8
