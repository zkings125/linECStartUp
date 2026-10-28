"""Regression tests for additive, time-dependent analytic coil fields."""

from __future__ import annotations

import numpy as np
import pytest

from ecstartup.config import CoilComponentConfig, GeometryConfig, SimulationConfig
from ecstartup.physics.circuit import CircuitModel
from ecstartup.physics.current_drive import CurrentDriveModel
from ecstartup.physics.global_model import GlobalStartupModel
from ecstartup.physics.magnetic import (
    AnalyticCoilComponent,
    CircularMagneticField,
    CompositeMagneticField,
    make_magnetic_field,
)


def test_empty_coil_list_preserves_historical_provider_type():
    field = make_magnetic_field(GeometryConfig(coil_components=[]))
    assert isinstance(field, CircularMagneticField)
    assert not isinstance(field, CompositeMagneticField)


def test_coil_components_superpose_and_follow_current_waveform():
    geometry = GeometryConfig(
        B_vertical_T=0.01,
        coil_components=[
            {
                "name": "PF_vfc",
                "role": "VFC",
                "current_A": 0.0,
                "current_waveform_A": [(0.0, 0.0), (1.0e-3, 2.0e5)],
                "B_Z_T_per_A": 1.0e-7,
            },
            {
                "name": "mirror_aux",
                "role": "TPC",
                "current_A": 1.0e5,
                "B_R_T_per_A": 2.0e-8,
                "B_phi_T_per_A": -1.0e-8,
            },
        ],
    )
    field = make_magnetic_field(geometry)
    assert isinstance(field, CompositeMagneticField)
    assert field.configuration == "composite[circular]"

    point_0 = field.evaluate(geometry.R0_m, 0.0, 0.0, 0.0)
    point_mid = field.evaluate(geometry.R0_m, 0.0, 0.0, 0.5e-3)
    # The VFC component ramps from zero to 1e5 A at the midpoint; the TPC
    # label is metadata only and does not alter the base topology closure.
    assert point_0.B_Z_T == pytest.approx(geometry.B_vertical_T)
    assert point_mid.B_Z_T == pytest.approx(geometry.B_vertical_T + 1.0e-2)
    assert point_mid.B_R_T == pytest.approx(2.0e-3)
    assert point_mid.B_phi_T == pytest.approx(
        geometry.B_t_T - 1.0e-3,
    )
    assert field.vertical_field(0.5e-3) == pytest.approx(point_mid.B_Z_T)

    diagnostics = field.diagnostics(0.0, 0.5e-3)
    assert diagnostics["coil_component_count"] == pytest.approx(2.0)
    assert diagnostics["coil_0_current_A"] == pytest.approx(1.0e5)
    assert diagnostics["coil_1_role"] == "TPC"
    assert diagnostics["base_field_configuration"] == "circular"


def test_component_stream_function_matches_axisymmetric_field_derivatives():
    geometry = GeometryConfig()
    component = AnalyticCoilComponent(
        CoilComponentConfig(
            name="test",
            current_A=3.0,
            B_R_T_per_A=0.4,
            B_Z_T_per_A=-0.2,
            B_phi_T_per_A=0.1,
        ),
        geometry,
    )
    R = geometry.R0_m + 0.03
    Z = 0.04
    h = 1.0e-6
    point = component.evaluate(R, Z)
    dpsi_dR = (component.flux_function(R + h, Z) - component.flux_function(R - h, Z)) / (2.0 * h)
    dpsi_dZ = (component.flux_function(R, Z + h) - component.flux_function(R, Z - h)) / (2.0 * h)
    assert point.B_R_T == pytest.approx(-dpsi_dZ / R, rel=1.0e-6)
    assert point.B_Z_T == pytest.approx(dpsi_dR / R, rel=1.0e-6)


def test_composite_flux_function_is_additive_for_arrays():
    geometry = GeometryConfig(
        coil_components=[
            {"name": "a", "current_A": 1.0, "B_Z_T_per_A": 0.1},
            {"name": "b", "current_A": -2.0, "B_R_T_per_A": 0.2},
        ]
    )
    field = make_magnetic_field(geometry)
    R = np.asarray([geometry.R0_m - 0.03, geometry.R0_m + 0.03])
    Z = np.asarray([-0.02, 0.02])
    expected = CircularMagneticField(geometry).flux_function(R, Z, 1.2e5)
    for component in field.components:
        expected = expected + component.flux_function(R, Z)
    assert np.allclose(field.flux_function(R, Z, 1.2e5), expected)


def test_global_model_accepts_explicit_magnetic_and_current_drive_providers():
    config = SimulationConfig(
        numerics={"t_end_s": 1.0e-6, "dt_s": 1.0e-6},
        outputs={"write_netcdf": False, "write_csv": False},
    )
    magnetic = make_magnetic_field(
        GeometryConfig(
            coil_components=[{"name": "injected", "current_A": 1.0, "B_Z_T_per_A": 0.01}]
        )
    )
    current_drive = CurrentDriveModel(config.geometry, config.rf)
    model = GlobalStartupModel(config, magnetic_field=magnetic, current_drive=current_drive)
    assert model.magnetic is magnetic
    assert model.circuit_model.current_drive is current_drive


def test_global_model_preserves_an_explicit_circuit_provider():
    config = SimulationConfig()
    circuit = CircuitModel(config.circuit, config.geometry, config.rf)
    model = GlobalStartupModel(config, circuit_model=circuit)
    assert model.circuit_model is circuit
