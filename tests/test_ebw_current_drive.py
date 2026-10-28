from __future__ import annotations

import numpy as np

from ecstartup.physics.ebw_current_drive import close_ebw_current_drive
from ecstartup.physics.hot_fullwave import HotFullWaveResult


def _fullwave(converged: bool = True) -> HotFullWaveResult:
    return HotFullWaveResult(
        np.asarray([0.0]),
        np.asarray([0.0]),
        np.asarray([1.0 + 0.0j]),
        np.asarray([0.0 + 0.0j]),
        60.0,
        {1: 10.0, 2: 50.0},
        40.0,
        100.0,
        1.0e-8,
        (40, 40),
        1000,
        12.0,
        1.0e-12,
        converged,
        0.02,
    )


def test_ebw_closure_combines_ox_xb_absorption_and_signed_momentum():
    result = close_ebw_current_drive(
        _fullwave(),
        incident_power_W=1.0e5,
        ox_efficiency=0.5,
        frequency_Hz=28.0e9,
        k_parallel_m_inv=20.0,
        momentum_confinement_time_s=1.0e-4,
        parallel_connection_length_m=2.0,
    )
    assert result.xb_efficiency == 1.0
    assert result.ebw_absorption_efficiency == 0.6
    assert result.combined_absorbed_efficiency == 0.3
    assert result.absorbed_ebw_power_W == 3.0e4
    assert result.momentum_balance_current_A < 0.0
    assert result.dominant_harmonic == 2
    assert result.quantitative


def test_ebw_closure_rejects_unconverged_fullwave_as_quantitative():
    result = close_ebw_current_drive(
        _fullwave(False),
        incident_power_W=1.0e5,
        ox_efficiency=0.5,
        frequency_Hz=28.0e9,
        k_parallel_m_inv=20.0,
        momentum_confinement_time_s=1.0e-4,
        parallel_connection_length_m=2.0,
    )
    assert not result.quantitative
