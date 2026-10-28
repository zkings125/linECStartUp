from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from ecstartup.config import SimulationConfig
from ecstartup.io import write_run
from ecstartup.physics.global_model import GlobalStartupModel
from ecstartup.physics.species_wall import (
    AnalyticAtomicRateProvider,
    DYONSpeciesWallModel,
    ExtendedGlobalState,
    SpeciesWallLayout,
    SpeciesWallState,
)
from ecstartup.solver import integrate
from ecstartup.state import GlobalState


def species_wall_payload(*, sputtering: float = 0.0) -> dict:
    return {
        "enabled": True,
        "species": [
            {
                "symbol": "D",
                "role": "fuel",
                "atomic_number": 1,
                "max_charge": 1,
                "radiation_coefficient_W_m3": 1.0e-32,
            },
            {
                "symbol": "C",
                "role": "impurity",
                "atomic_number": 6,
                "max_charge": 2,
                "initial_charge_densities_m3": [1.0e14, 2.0e13, 1.0e12],
                "ionization_energies_eV": [11.26, 24.38],
                "neutral_pump_time_s": 0.2,
            },
        ],
        "wall_materials": [
            {
                "name": "fuel_reservoir",
                "species": "D",
                "initial_inventory_m2": 1.0e17,
                "recycling_coefficient": 0.8,
                "desorption_time_s": 0.1,
            },
            {
                "name": "carbon_surface",
                "species": "C",
                "initial_inventory_m2": 0.0,
                "recycling_coefficient": 0.2,
                "physical_sputtering_yield": sputtering,
                "physical_sputtering_threshold_eV": 1.0,
                "chemical_sputtering_yield": 0.01 if sputtering else 0.0,
            },
        ],
    }


def make_config(*, sputtering: float = 0.0) -> SimulationConfig:
    return SimulationConfig(
        species_wall=species_wall_payload(sputtering=sputtering),
        numerics={"t_end_s": 2.0e-6, "dt_s": 1.0e-6},
        outputs={"write_netcdf": False, "write_csv": False},
    )


def test_species_wall_config_is_strict_and_checks_charge_lengths() -> None:
    with pytest.raises(ValidationError):
        SimulationConfig(species_wall={"enabled": True, "unknown": 1})
    payload = species_wall_payload()
    payload["species"][1]["initial_charge_densities_m3"] = [1.0]
    with pytest.raises(ValidationError, match="max_charge.1"):
        SimulationConfig(species_wall=payload)


def test_layout_is_deterministic_and_round_trips() -> None:
    config = make_config()
    layout = SpeciesWallLayout.from_config(config.species_wall)
    assert layout.vector_names == (
        "n_C0_m3",
        "n_C1_m3",
        "n_C2_m3",
        "wall_inventory_C_m2",
        "wall_inventory_D_m2",
    )
    initial = layout.initial_state()
    rebuilt = SpeciesWallState.from_vector(layout, initial.to_vector())
    np.testing.assert_array_equal(rebuilt.to_vector(), initial.to_vector())


def test_adjacent_charge_transitions_conserve_nuclei_and_electron_charge() -> None:
    config = make_config()
    module = DYONSpeciesWallModel(config.species_wall, config.geometry)
    state = module.initial_state()
    # Remove wall and pump terms to isolate adjacent atomic transitions.
    carbon = next(item for item in config.species_wall.species if item.symbol == "C")
    carbon.neutral_pump_time_s = None
    core = GlobalState(1.0e17, 0.0, 1.0e18, 1.5 * 1.0e17 * 20.0 * 1.602176634e-19, 0.0)
    terms = module.evaluate(state, core, {"S_i_wall_m3_s": 0.0}, Ti_eV=0.1)
    carbon_derivative = terms.derivative[module.layout.density_slices["C"]]
    assert np.sum(carbon_derivative) == pytest.approx(0.0, abs=1.0)
    charge_derivative = float(np.dot(np.arange(3), carbon_derivative))
    assert charge_derivative == pytest.approx(terms.electron_source_m3_s, rel=1.0e-12)
    assert terms.radiation_power_W_m3 >= 0.0
    assert terms.ionization_power_W_m3 >= 0.0


def test_zero_sputtering_and_wall_inventory_projection() -> None:
    config = make_config(sputtering=0.0)
    module = DYONSpeciesWallModel(config.species_wall, config.geometry)
    raw = module.initial_state().to_vector()
    raw[module.layout.wall_indices["carbon_surface"]] = -1.0
    state = SpeciesWallState(module.layout, raw).projected()
    core = GlobalState(1.0e15, 1.0e15, 1.0e18, 1.0, 0.0)
    terms = module.evaluate(state, core, {"S_i_wall_m3_s": 1.0e17}, Ti_eV=1.0)
    assert state.wall_inventory_m2("carbon_surface") == 0.0
    assert terms.diagnostics["S_sputter_C_m3_s"] == 0.0
    assert terms.derivative[module.layout.wall_indices["carbon_surface"]] >= 0.0


def test_recycling_inventory_and_sputtering_balance_is_conservative() -> None:
    config = make_config(sputtering=0.1)
    module = DYONSpeciesWallModel(config.species_wall, config.geometry)
    core = GlobalState(1.0e16, 1.0e16, 1.0e18, 1.0, 0.0)
    terms = module.evaluate(module.initial_state(), core, {"S_i_wall_m3_s": 1.0e18}, Ti_eV=10.0)
    scale = max(1.0, abs(float(terms.diagnostics["S_sputter_C_m3_s"])))
    assert abs(float(terms.diagnostics["species_particle_balance_residual_m3_s"])) / scale < 1.0e-12
    assert abs(float(terms.diagnostics["species_particle_balance_relative"])) < 1.0e-12
    assert float(terms.diagnostics["S_fuel_recycle_m3_s"]) > 0.0
    assert float(terms.diagnostics["S_sputter_C_m3_s"]) > 0.0


def test_analytic_provider_is_positive_and_thresholded() -> None:
    config = make_config()
    carbon = next(item for item in config.species_wall.species if item.symbol == "C")
    provider = AnalyticAtomicRateProvider()
    assert provider.label == "analytic_surrogate"
    assert provider.ionization_rate_m3_s(carbon, 0, 1.0e-3) >= 0.0
    assert provider.ionization_rate_m3_s(carbon, 0, 20.0) > 0.0
    assert provider.recombination_rate_m3_s(carbon, 1, 20.0) > 0.0
    assert provider.radiation_coefficient_W_m3(carbon, 1, 20.0) >= 0.0


def test_disabled_path_remains_legacy_seven_state_model() -> None:
    default = SimulationConfig(numerics={"t_end_s": 2.0e-6, "dt_s": 1.0e-6})
    explicit = SimulationConfig(
        species_wall={"enabled": False},
        numerics={"t_end_s": 2.0e-6, "dt_s": 1.0e-6},
    )
    result_default = integrate(default)
    result_explicit = integrate(explicit)
    assert isinstance(result_default.final_state, GlobalState)
    np.testing.assert_allclose(result_default.final_state.to_vector(), result_explicit.final_state.to_vector())


def test_enabled_short_integration_uses_dynamic_state_and_outputs(tmp_path) -> None:
    config = make_config(sputtering=0.02)
    model = GlobalStartupModel(config)
    initial = model.initial_state()
    assert isinstance(initial, ExtendedGlobalState)
    assert model.absolute_tolerance_vector().size == initial.to_vector().size
    result = integrate(config, model)
    assert result.success
    assert isinstance(result.final_state, ExtendedGlobalState)
    assert np.all(result.final_state.species_wall.to_vector() >= 0.0)
    required = {
        "n_C0_m3",
        "n_C1_m3",
        "P_rad_C_W_m3",
        "S_sputter_C_m3_s",
        "wall_inventory_C_m2",
        "Z_eff",
        "electron_density_quasineutral_m3",
        "species_particle_balance_residual_m3_s",
    }
    assert required <= result.diagnostics[-1].keys()
    out = write_run(result, config, tmp_path)
    assert (out / "summary.json").exists()
