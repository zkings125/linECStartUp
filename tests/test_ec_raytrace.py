from __future__ import annotations

import numpy as np
import pytest

from ecstartup.config import GeometryConfig, RayTracingConfig, RFSourceConfig
from ecstartup.constants import E_CHARGE_C, M_E_KG, PI
from ecstartup.physics.cold_plasma import (
    dispersion_determinant,
    launch_wavevector,
    mode_solution,
    stix_parameters,
)
from ecstartup.physics.ec_raytrace import HamiltonianECRayTracer, UniformElectronDensity
from ecstartup.physics.magnetic import FieldPoint


class ConstantMagneticField:
    configuration = "constant_test"

    def __init__(self, B_R_T: float, B_Z_T: float, B_phi_T: float) -> None:
        self.point = FieldPoint(B_R_T, B_Z_T, B_phi_T)

    def evaluate(self, R_m: float, Z_m: float, Ip_A: float, t_s: float = 0.0) -> FieldPoint:
        return self.point


def test_vacuum_stix_and_both_em_branches_reduce_to_n_equals_one() -> None:
    stix = stix_parameters(28.0e9, 0.83, 0.0)
    assert stix.S == pytest.approx(1.0)
    assert stix.D == pytest.approx(0.0)
    assert stix.P == pytest.approx(1.0)
    assert mode_solution(stix, PI / 2.0, "O").refractive_index_squared == pytest.approx(1.0)
    assert mode_solution(stix, PI / 2.0, "X").refractive_index_squared == pytest.approx(1.0)


@pytest.mark.parametrize("mode", ["O", "X"])
def test_launch_wavevector_lies_on_cold_dispersion_surface(mode: str) -> None:
    frequency = 28.0e9
    B = np.asarray([0.0, 0.0, 0.30])
    direction = np.asarray([1.0, 0.0, 0.0])
    density = 2.0e17
    wavevector, _ = launch_wavevector(frequency, direction, B, density, mode)
    determinant = dispersion_determinant(frequency, wavevector, B, density)
    assert determinant == pytest.approx(0.0, abs=1.0e-10)


def test_constant_vacuum_ray_follows_configured_launch_direction() -> None:
    geometry = GeometryConfig(R_bounds_m=(0.2, 0.8), Z_bounds_m=(-0.2, 0.2))
    source = RFSourceConfig(
        name="straight",
        frequency_Hz=28.0e9,
        harmonic=10,
        launch={
            "R_m": 0.50,
            "Z_m": 0.0,
            "poloidal_angle_deg": 0.0,
            "toroidal_angle_deg": 0.0,
            "mode": "O",
            "polarization": "O",
        },
    )
    tracer = HamiltonianECRayTracer(
        ConstantMagneticField(0.0, 0.0, 0.3),
        UniformElectronDensity(0.0),
        geometry,
        RayTracingConfig(step_m=0.01, max_length_m=0.5, wkb_min_scale_ratio=1.0),
    )
    result = tracer.trace(source)
    assert result.status == "domain_exit"
    assert result.points[-1].R_m > result.points[0].R_m
    assert result.points[-1].Z_m == pytest.approx(0.0, abs=1.0e-10)
    assert max(abs(point.hamiltonian) for point in result.points) < 1.0e-10


def test_selected_second_harmonic_resonance_is_reported_without_gs() -> None:
    frequency = 28.0e9
    harmonic = 2
    resonance_B = 2.0 * PI * frequency * M_E_KG / (harmonic * E_CHARGE_C)
    geometry = GeometryConfig(R_bounds_m=(0.2, 0.8), Z_bounds_m=(-0.2, 0.2))
    source = RFSourceConfig(
        name="second_harmonic",
        frequency_Hz=frequency,
        harmonic=harmonic,
        launch={"R_m": 0.5, "Z_m": 0.0, "poloidal_angle_deg": 0.0, "mode": "X"},
    )
    tracer = HamiltonianECRayTracer(
        ConstantMagneticField(0.0, 0.0, resonance_B),
        UniformElectronDensity(1.0e14),
        geometry,
        RayTracingConfig(step_m=0.01, max_length_m=0.2),
    )
    result = tracer.trace(source)
    assert result.status == "resonance_reached"
    assert result.reached_resonance
    assert result.path_length_m == pytest.approx(0.0)


def test_fundamental_cold_tensor_singularity_is_classified_as_resonance() -> None:
    frequency = 28.0e9
    resonance_B = 2.0 * PI * frequency * M_E_KG / E_CHARGE_C
    geometry = GeometryConfig(R_bounds_m=(0.2, 0.8), Z_bounds_m=(-0.2, 0.2))
    source = RFSourceConfig(
        name="fundamental",
        frequency_Hz=frequency,
        harmonic=1,
        launch={"R_m": 0.5, "Z_m": 0.0, "poloidal_angle_deg": 0.0, "mode": "O"},
    )
    tracer = HamiltonianECRayTracer(
        ConstantMagneticField(0.0, 0.0, resonance_B),
        UniformElectronDensity(1.0e14),
        geometry,
        RayTracingConfig(step_m=0.01, max_length_m=0.2),
    )
    result = tracer.trace(source)
    assert result.status == "resonance_reached"
