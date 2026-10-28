from __future__ import annotations

import numpy as np
import pytest

from ecstartup.physics.inductive_network import InductiveCircuitNetwork


def test_implicit_mutual_inductance_network_satisfies_circuit_equation():
    network = InductiveCircuitNetwork(
        ("CS", "PF1", "plasma"),
        np.asarray(
            [
                [3.0e-3, 0.2e-3, 0.5e-3],
                [0.2e-3, 2.0e-3, -0.1e-3],
                [0.5e-3, -0.1e-3, 1.0e-3],
            ]
        ),
        (0.02, 0.03, 2.0e-4),
    )
    state = network.initial_state()
    updated, diagnostics = network.advance(state, 1.0e-4, (100.0, -20.0, 0.0))
    assert updated.current_A.shape == (3,)
    assert diagnostics.equation_residual_relative < 1.0e-12
    assert diagnostics.magnetic_energy_J > 0.0


def test_measured_coil_ramp_generates_mutual_plasma_loop_voltage():
    mutual = 0.5e-3
    network = InductiveCircuitNetwork(
        ("CS", "plasma"),
        np.asarray([[2.0e-3, mutual], [mutual, 1.0e-3]]),
        (0.01, 1.0e-4),
    )
    times = np.linspace(0.0, 0.1, 11)
    waveform = {"CS": ((0.0, 30.0e3), (0.1, 0.0))}
    voltage = network.induced_plasma_loop_voltage(waveform, times)
    assert voltage == pytest.approx(np.full_like(times, 150.0))


def test_inductance_matrix_must_be_symmetric_positive_definite():
    with pytest.raises(ValueError, match="symmetric"):
        InductiveCircuitNetwork(("CS", "plasma"), np.asarray([[1.0, 0.2], [0.1, 1.0]]), (0.0, 0.0))
    with pytest.raises(ValueError, match="positive definite"):
        InductiveCircuitNetwork(("CS", "plasma"), np.asarray([[1.0, 2.0], [2.0, 1.0]]), (0.0, 0.0))
