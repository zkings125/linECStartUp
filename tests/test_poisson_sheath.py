from __future__ import annotations

import numpy as np

from ecstartup.physics.poisson_sheath import DebyePoissonAmbipolarSheath1D


def test_debye_poisson_sheath_resolves_space_charge_and_floating_drop():
    solver = DebyePoissonAmbipolarSheath1D(sheath_points=201)
    profile = solver.solve_boundary(1.0e17, 1.0e17, 20.0, 0.2)
    assert profile.debye_length_m > 0.0
    assert profile.wall_drop_V > 0.0
    assert abs(profile.potential_V[0]) < 1.0e-12
    np.testing.assert_allclose(
        profile.potential_V[-1], -profile.wall_drop_V, rtol=2.0e-5, atol=1.0e-8
    )
    assert profile.electron_density_m3[-1] < profile.ion_density_m3[-1]
    assert profile.poisson_residual_relative < 0.08


def test_debye_poisson_spatial_solution_is_compatible_with_kinetic_sheath_interface():
    solver = DebyePoissonAmbipolarSheath1D(sheath_points=101)
    result = solver.solve(
        np.linspace(0.0, 0.4, 5),
        np.linspace(1.0e17, 1.2e17, 5),
        np.linspace(1.0e17, 1.2e17, 5),
        np.linspace(10.0, 20.0, 5),
        ion_temperature_eV=0.2,
    )
    assert result.E_parallel_V_m.shape == (5,)
    assert result.left_debye_sheath.wall_drop_V == result.sheath_drop_left_V
    assert result.right_debye_sheath.wall_drop_V == result.sheath_drop_right_V
