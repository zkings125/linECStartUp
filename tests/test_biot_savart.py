"""Small analytic and interface checks for the circular-loop provider."""

from __future__ import annotations

import numpy as np
import pytest

from ecstartup.config import GeometryConfig, SimulationConfig
from ecstartup.constants import MU0_H_M
from ecstartup.physics.biot_savart import (
    BiotSavartMagneticField,
    CircularLoop,
    circular_loop_field,
    circular_loop_flux,
    make_biot_savart_field,
)
from ecstartup.physics.magnetic import CircularMagneticField
from ecstartup.physics.magnetic import make_magnetic_field


def test_circular_loop_axis_field_matches_exact_limit():
    radius = 0.30
    z0 = 0.04
    current = 120_000.0
    point = circular_loop_field(0.0, z0 + 0.10, radius, z0, current)
    expected = MU0_H_M * current * radius**2 / (2.0 * (radius**2 + 0.10**2) ** 1.5)
    assert point.B_R_T == pytest.approx(0.0)
    assert point.B_Z_T == pytest.approx(expected)
    assert point.B_phi_T == pytest.approx(0.0)


def test_loop_flux_derivatives_reproduce_off_axis_field():
    radius = 0.30
    z0 = -0.02
    current = -80_000.0
    R = 0.43
    Z = 0.07
    h = 1.0e-6
    point = circular_loop_field(R, Z, radius, z0, current)
    dpsi_dR = (circular_loop_flux(R + h, Z, radius, z0, current) - circular_loop_flux(R - h, Z, radius, z0, current)) / (2.0 * h)
    dpsi_dZ = (circular_loop_flux(R, Z + h, radius, z0, current) - circular_loop_flux(R, Z - h, radius, z0, current)) / (2.0 * h)
    assert point.B_R_T == pytest.approx(-dpsi_dZ / R, rel=2.0e-5)
    assert point.B_Z_T == pytest.approx(dpsi_dR / R, rel=2.0e-5)


def test_loop_waveform_and_turns_are_applied_linearly():
    loop = CircularLoop(
        radius_m=0.25,
        current_A=10.0,
        current_waveform_A=((0.0, 0.0), (2.0, 4.0)),
        turns=3,
        name="ramp",
    )
    assert loop.current_at(0.0) == pytest.approx(0.0)
    assert loop.current_at(1.0) == pytest.approx(6.0)
    assert loop.current_at(3.0) == pytest.approx(12.0)


def test_biot_savart_provider_adds_multiple_loops_and_preserves_base():
    geometry = GeometryConfig(B_vertical_T=0.01)
    base = CircularMagneticField(geometry)
    loops = [
        CircularLoop(radius_m=0.25, current_A=20_000.0, name="upper"),
        CircularLoop(radius_m=0.25, z0_m=0.12, current_A=-10_000.0, name="offset"),
    ]
    field = make_biot_savart_field(geometry, loops, base_provider=base)
    assert isinstance(field, BiotSavartMagneticField)
    point = field.evaluate(geometry.R0_m, 0.0, 0.0)
    expected = base.evaluate(geometry.R0_m, 0.0, 0.0)
    for loop in loops:
        addition = circular_loop_field(geometry.R0_m, 0.0, loop.radius_m, loop.z0_m, loop.current_at())
        expected = type(expected)(
            expected.B_R_T + addition.B_R_T,
            expected.B_Z_T + addition.B_Z_T,
            expected.B_phi_T,
        )
    assert point.B_R_T == pytest.approx(expected.B_R_T)
    assert point.B_Z_T == pytest.approx(expected.B_Z_T)
    assert point.B_phi_T == pytest.approx(expected.B_phi_T)
    assert field.vertical_field() == pytest.approx(point.B_Z_T)
    diagnostics = field.diagnostics(0.0)
    assert diagnostics["loop_count"] == pytest.approx(2.0)
    assert diagnostics["loop_current_total_A"] == pytest.approx(10_000.0)
    assert diagnostics["loop_0_name"] == "upper"
    assert diagnostics["loop_1_z0_m"] == pytest.approx(0.12)


def test_biot_savart_provider_waveform_changes_field_and_flux_arrays():
    geometry = GeometryConfig()
    field = BiotSavartMagneticField(
        geometry,
        [
            CircularLoop(
                radius_m=0.35,
                current_waveform_A=((0.0, 0.0), (1.0e-3, 1.0e5)),
                name="ramp",
            )
        ],
    )
    p0 = field.evaluate(0.2, 0.0, 0.0, 0.0)
    p1 = field.evaluate(0.2, 0.0, 0.0, 0.5e-3)
    p2 = field.evaluate(0.2, 0.0, 0.0, 1.0e-3)
    assert p0.B_Z_T == pytest.approx(geometry.B_vertical_T)
    assert p1.B_Z_T - geometry.B_vertical_T == pytest.approx((p2.B_Z_T - geometry.B_vertical_T) / 2.0)
    R = np.asarray([0.2, 0.4])
    Z = np.asarray([0.0, 0.03])
    psi = field.flux_function(R, Z, 0.0, 0.5e-3)
    assert psi.shape == R.shape
    assert np.isfinite(psi).all()


def test_filament_singularity_is_explicit_and_validation_is_strict():
    with pytest.raises(ValueError, match="细丝线圈"):
        circular_loop_field(0.30, 0.0, 0.30, 0.0, 1.0)
    with pytest.raises(ValueError):
        CircularLoop(radius_m=0.1, turns=0)
    with pytest.raises(ValueError):
        CircularLoop(radius_m=0.1, current_waveform_A=((1.0, 0.0), (0.0, 1.0)))


def test_configured_biot_savart_backend_keeps_selected_tpc_state():
    config = SimulationConfig(
        geometry={
            "field_configuration": "TPC",
            "magnetic_backend": "biot_savart",
            "biot_savart_loops": [
                {"name": "upper", "radius_m": 0.35, "z0_m": 0.15, "current_A": 1.0e4}
            ],
        }
    )
    field = make_magnetic_field(config.geometry)
    assert isinstance(field, BiotSavartMagneticField)
    assert field.configuration.startswith("biot_savart[TPC]")
    assert field.configuration_state(0.0).mirror_ratio == pytest.approx(
        config.geometry.tpc_mirror_ratio
    )


def test_biot_savart_backend_schema_rejects_ambiguous_combinations():
    with pytest.raises(ValueError):
        GeometryConfig(magnetic_backend="biot_savart")
    with pytest.raises(ValueError):
        GeometryConfig(
            magnetic_backend="analytic",
            biot_savart_loops=[{"radius_m": 0.3}],
        )
    with pytest.raises(ValueError):
        GeometryConfig(
            magnetic_backend="biot_savart",
            biot_savart_loops=[{"radius_m": 0.3}],
            coil_components=[{"name": "aux"}],
        )


def test_edge_diagnostic_reports_filament_singularity_without_aborting():
    geometry = GeometryConfig()
    field = BiotSavartMagneticField(
        geometry,
        [CircularLoop(radius_m=geometry.R0_m + geometry.minor_radius_m, current_A=1.0)],
    )
    diagnostics = field.diagnostics(0.0)
    assert diagnostics["loop_edge_singularity_count"] == pytest.approx(1.0)
    assert np.isnan(diagnostics["loop_Bp_edge_T"])
