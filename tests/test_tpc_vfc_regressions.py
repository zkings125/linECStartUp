"""Regression checks for TPC/VFC configuration semantics."""

from dataclasses import replace

import pytest

from ecstartup.config import GeometryConfig, SimulationConfig
from ecstartup.physics.global_model import GlobalStartupModel
from ecstartup.physics.magnetic import TPCMagneticField


def test_vertical_charge_current_vanishes_when_vertical_field_is_zero():
    """A calibrated VFC current must not survive with no vertical field."""

    config = SimulationConfig(
        geometry={
            "field_configuration": "VFC",
            "B_vertical_T": 0.0,
            "vfc_current_fraction": 0.2,
        }
    )
    model = GlobalStartupModel(config)
    state = replace(model.initial_state(), n_hot_m3=1.0e17, hot_energy_J_m3=1.0e3)

    current = model.evaluate(0.01, state).circuit.I_vfc_A

    assert current == pytest.approx(0.0)


def test_global_model_accepts_an_injected_magnetic_provider():
    """A production field backend can replace the factory-selected provider."""

    config = SimulationConfig(geometry={"field_configuration": "circular"})
    injected = TPCMagneticField(
        GeometryConfig(field_configuration="TPC", tpc_mirror_ratio=2.1)
    )
    model = GlobalStartupModel(config, magnetic_field=injected)

    assert model.magnetic is injected
    assert model.evaluate(0.0, model.initial_state()).values["field_configuration"] == "TPC"


def test_vertical_charge_fraction_alias_is_normalized_without_assignment_recursion():
    config = SimulationConfig(
        geometry={"vertical_charge_separation_fraction": 0.2}
    )

    assert config.geometry.vfc_current_fraction == pytest.approx(0.2)
    assert config.geometry.effective_vertical_charge_fraction == pytest.approx(0.2)
