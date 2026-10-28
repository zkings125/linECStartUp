from __future__ import annotations

import math

import numpy as np
import pytest
from pydantic import ValidationError

from ecstartup.config import ChargeClosureConfig, SimulationConfig
from ecstartup.physics.charge_closure import ChargeClosureModel
from ecstartup.physics.global_model import GlobalStartupModel
from ecstartup.physics.species_wall import ExtendedGlobalState
from ecstartup.solver import integrate

from test_species_wall import species_wall_payload


def test_charge_closure_config_is_strict() -> None:
    with pytest.raises(ValidationError):
        ChargeClosureConfig(mode="invalid")
    with pytest.raises(ValidationError):
        ChargeClosureConfig(potential_min_over_Te=1.0)


def test_quasineutral_closure_matches_charge_continuity_exactly() -> None:
    closure = ChargeClosureModel(ChargeClosureConfig(mode="quasineutral"))
    terms = closure.evaluate(
        Te_eV=10.0,
        free_electron_wall_loss_m3_s=1.0e20,
        positive_charge_wall_loss_m3_s=2.0e18,
        secondary_electron_source_m3_s=1.0e17,
        attachment_loss_m3_s=2.0e16,
    )
    assert terms.electron_wall_loss_m3_s == pytest.approx(2.08e18)
    assert terms.charge_continuity_residual_m3_s == pytest.approx(0.0, abs=1.0e3)
    assert terms.ambipolar_potential_V > 0.0


def test_ambipolar_closure_inverts_boltzmann_flux() -> None:
    config = ChargeClosureConfig(mode="ambipolar")
    closure = ChargeClosureModel(config)
    terms = closure.evaluate(
        Te_eV=5.0,
        free_electron_wall_loss_m3_s=1.0e20,
        positive_charge_wall_loss_m3_s=1.0e18,
        secondary_electron_source_m3_s=5.0e16,
        attachment_loss_m3_s=0.0,
    )
    target = 1.05e18
    assert terms.electron_wall_loss_m3_s == pytest.approx(target, rel=1.0e-12)
    assert terms.ambipolar_potential_over_Te == pytest.approx(math.log(1.0e20 / target))
    assert terms.ambipolar_potential_V == pytest.approx(5.0 * math.log(1.0e20 / target))
    assert abs(terms.charge_continuity_residual_m3_s) / target < 1.0e-12
    assert not terms.flux_limited


def test_ambipolar_potential_limit_reports_nonzero_residual() -> None:
    closure = ChargeClosureModel(
        ChargeClosureConfig(mode="ambipolar", potential_max_over_Te=1.0)
    )
    terms = closure.evaluate(
        Te_eV=5.0,
        free_electron_wall_loss_m3_s=1.0e20,
        positive_charge_wall_loss_m3_s=1.0e18,
        secondary_electron_source_m3_s=0.0,
        attachment_loss_m3_s=0.0,
    )
    assert terms.flux_limited
    assert terms.ambipolar_potential_over_Te == 1.0
    assert terms.charge_continuity_residual_m3_s != 0.0


def test_extended_rhs_preserves_quasineutrality() -> None:
    config = SimulationConfig(
        species_wall=species_wall_payload(sputtering=0.01),
        charge_closure={"mode": "quasineutral"},
        numerics={"t_end_s": 2.0e-6, "dt_s": 1.0e-6},
        outputs={"write_netcdf": False, "write_csv": False},
    )
    model = GlobalStartupModel(config)
    initial = model.initial_state()
    assert isinstance(initial, ExtendedGlobalState)
    evaluation = model.evaluate(0.0, initial)
    assert evaluation.values["charge_balance_residual_m3"] == pytest.approx(0.0)
    derivative = model.rhs(0.0, initial.to_vector())
    carbon_slice = model.species_wall_model.layout.density_slices["C"]
    carbon_derivative = derivative[7 + carbon_slice.start : 7 + carbon_slice.stop]
    charge_derivative = derivative[1] + float(np.dot(np.arange(carbon_derivative.size), carbon_derivative))
    scale = max(abs(derivative[0]), abs(charge_derivative), 1.0)
    assert abs(derivative[0] - charge_derivative) / scale < 1.0e-12


def test_ambipolar_short_run_stays_quasineutral() -> None:
    config = SimulationConfig(
        species_wall=species_wall_payload(sputtering=0.01),
        charge_closure={"mode": "ambipolar"},
        numerics={"t_end_s": 2.0e-5, "dt_s": 1.0e-6},
        outputs={"write_netcdf": False, "write_csv": False},
    )
    result = integrate(config)
    assert result.success
    assert max(abs(float(row["charge_balance_relative"])) for row in result.diagnostics) < 1.0e-9
    assert max(abs(float(row["charge_continuity_relative"])) for row in result.diagnostics) < 1.0e-12
    assert all(float(row["ambipolar_potential_V"]) >= 0.0 for row in result.diagnostics)
