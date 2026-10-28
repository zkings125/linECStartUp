from __future__ import annotations

import numpy as np
import pytest

from ecstartup.config import SimulationConfig
from ecstartup.current_profile import (
    normalized_beta_profile,
    reconstruct_current_profiles,
    reconstruct_mechanism_currents,
)
from ecstartup.numerics import trapezoid
from ecstartup.solver import integrate


def short_config(cs_mode: str = "with_cs", zeff: float = 1.0) -> SimulationConfig:
    return SimulationConfig(
        initial={"ne_m3": 1.0e16, "ni_m3": 1.0e16, "Te_eV": 5.0},
        rf={"enabled": False},
        circuit={
            "cs_mode": cs_mode,
            "loop_voltage_V": 5.0,
            "loop_voltage_ramp_s": 0.0,
            "effective_charge_override": zeff,
        },
        numerics={"t_end_s": 2.0e-5, "dt_s": 2.0e-6},
        outputs={
            "cadence_s": 2.0e-6,
            "write_netcdf": False,
            "write_csv": False,
            "profile_radial_points": 401,
        },
    )


def test_beta_profile_integrates_to_one() -> None:
    rho = np.linspace(0.0, 1.0, 10001)
    radius = 0.25
    for shape in ((0.0, 1.5), (0.5, 3.0), (4.0, 0.5)):
        profile = normalized_beta_profile(rho, radius, shape)
        integral = trapezoid(profile * 2.0 * np.pi * radius**2 * rho, rho)
        assert integral == pytest.approx(1.0, rel=2.0e-5)


def test_profile_integrates_to_actual_ip() -> None:
    config = short_config()
    result = integrate(config)
    profiles = reconstruct_current_profiles(result, config)
    radius = config.geometry.minor_radius_m
    recovered = trapezoid(
        profiles.j_total_A_m2 * 2.0 * np.pi * radius**2 * profiles.rho_geo,
        profiles.rho_geo,
        axis=1,
    )
    expected = np.asarray([row["Ip_A"] for row in result.diagnostics])
    assert recovered == pytest.approx(expected, rel=2.0e-4, abs=1.0e-6)


def test_without_cs_removes_inductive_component() -> None:
    config = short_config(cs_mode="without_cs")
    result = integrate(config)
    currents = reconstruct_mechanism_currents(result, config)
    assert all(row["loop_voltage_V"] == 0.0 for row in result.diagnostics)
    assert np.max(np.abs(currents.I_cs_A)) == 0.0


def test_effective_charge_reduces_conductivity() -> None:
    low = integrate(short_config(zeff=1.0)).diagnostics[0]
    high = integrate(short_config(zeff=4.0)).diagnostics[0]
    assert high["Z_eff_used"] == 4.0
    assert high["nu_ei_s"] > low["nu_ei_s"]
    assert high["sigma_S_m"] < low["sigma_S_m"]
