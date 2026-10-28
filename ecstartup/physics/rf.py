"""ECRH/RF source interface with independently timed multi-frequency beams.

The default remains a prescribed absorbed-power surrogate.  A production
model should replace it with ray/full-wave propagation and a quasilinear
operator; configured launch coordinates and polarization are recorded but do
not yet change accessibility or deposition.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from ..config import GeometryConfig, RFConfig, RFSourceConfig
from ..constants import E_CHARGE_C, EPS0_F_M, M_E_KG, PI
from ..state import GlobalState


@dataclass(frozen=True)
class RFBeamTerms:
    name: str
    frequency_Hz: float
    harmonic: int
    power_absorbed_W: float
    power_density_W_m3: float
    hot_power_density_W_m3: float
    hot_particle_source_m3_s: float
    current_drive_A: float
    accessibility: float
    pulse_factor: float
    omega_ce_over_omega_rf: float
    critical_density_m3: float
    launch_R_m: float
    launch_Z_m: float
    launch_poloidal_angle_deg: float | None
    mode: str
    polarization: str


@dataclass(frozen=True)
class RFSourceTerms:
    power_absorbed_W: float
    power_density_W_m3: float
    hot_power_density_W_m3: float
    hot_particle_source_m3_s: float
    current_drive_A: float
    accessibility: float
    pulse_factor: float
    omega_ce_over_omega_rf: float
    components: tuple[RFBeamTerms, ...] = ()

    def diagnostics(self) -> dict[str, float | str]:
        values: dict[str, float | str] = {"rf_source_count": float(len(self.components))}
        for index, beam in enumerate(self.components):
            prefix = f"rf_{index}"
            values.update(
                {
                    f"{prefix}_name": beam.name,
                    f"{prefix}_frequency_Hz": beam.frequency_Hz,
                    f"{prefix}_harmonic": float(beam.harmonic),
                    f"{prefix}_power_absorbed_W": beam.power_absorbed_W,
                    f"{prefix}_power_density_W_m3": beam.power_density_W_m3,
                    f"{prefix}_hot_power_density_W_m3": beam.hot_power_density_W_m3,
                    f"{prefix}_current_drive_A": beam.current_drive_A,
                    f"{prefix}_accessibility": beam.accessibility,
                    f"{prefix}_pulse_factor": beam.pulse_factor,
                    f"{prefix}_omega_ce_over_omega_rf": beam.omega_ce_over_omega_rf,
                    f"{prefix}_critical_density_m3": beam.critical_density_m3,
                    f"{prefix}_launch_R_m": beam.launch_R_m,
                    f"{prefix}_launch_Z_m": beam.launch_Z_m,
                    f"{prefix}_launch_poloidal_angle_deg": (
                        beam.launch_poloidal_angle_deg
                        if beam.launch_poloidal_angle_deg is not None
                        else float("nan")
                    ),
                    f"{prefix}_mode": beam.mode,
                    f"{prefix}_polarization": beam.polarization,
                }
            )
        return values


class ECRHSource:
    def __init__(self, config: RFConfig, geometry: GeometryConfig) -> None:
        self.config = config
        self.geometry = geometry
        if config.sources:
            self.sources = tuple(config.sources)
        else:
            self.sources = (
                RFSourceConfig(
                    name="legacy",
                    enabled=config.enabled,
                    frequency_Hz=config.frequency_Hz,
                    harmonic=config.harmonic,
                    power_W=config.power_W,
                    absorption_fraction=config.absorption_fraction,
                    pulse=config.pulse,
                    launch=config.launch,
                    hot_fraction=config.hot_fraction,
                    hot_creation_energy_eV=config.hot_creation_energy_eV,
                    current_drive_A_per_W=config.current_drive_A_per_W,
                    current_drive_sign=config.current_drive_sign,
                ),
            )
        critical = [self._critical_density(source.frequency_Hz) for source in self.sources]
        # Retain the scalar compatibility property.  The minimum is the most
        # conservative cutoff; per-source values are emitted in diagnostics.
        self.critical_density_m3 = min(critical)

    @staticmethod
    def _critical_density(frequency_Hz: float) -> float:
        omega = 2.0 * PI * frequency_Hz
        return EPS0_F_M * M_E_KG * omega**2 / E_CHARGE_C**2

    def _pulse(self, source: RFSourceConfig, t_s: float) -> float:
        if not self.config.enabled or not source.enabled or source.power_W <= 0.0:
            return 0.0
        pulse = source.pulse
        if t_s < pulse.start_s or t_s > pulse.end_s:
            return 0.0
        width = pulse.end_s - pulse.start_s
        elapsed = t_s - pulse.start_s
        value = 1.0
        if pulse.ramp_up_s > 0.0 and elapsed < pulse.ramp_up_s:
            value = min(value, elapsed / pulse.ramp_up_s)
        down_start = width - pulse.ramp_down_s
        if pulse.ramp_down_s > 0.0 and elapsed > down_start:
            value = min(value, max(0.0, (width - elapsed) / pulse.ramp_down_s))
        return max(0.0, min(1.0, value))

    def _evaluate_source(self, source: RFSourceConfig, t_s: float, state: GlobalState) -> RFBeamTerms:
        pulse = self._pulse(source, t_s)
        critical_density = self._critical_density(source.frequency_Hz)
        ratio = E_CHARGE_C * self.geometry.B_t_T / M_E_KG / (2.0 * PI * source.frequency_Hz)
        launch_R = source.launch.R_m if source.launch.R_m is not None else self.geometry.R0_m
        if pulse <= 0.0:
            return RFBeamTerms(
                source.name,
                source.frequency_Hz,
                source.harmonic,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                ratio,
                critical_density,
                launch_R,
                source.launch.Z_m,
                source.launch.poloidal_angle_deg,
                source.launch.mode,
                source.launch.polarization,
            )
        # Low-density accessibility surrogate only.  O/X polarization,
        # harmonic resonance and spatial deposition require a real RF backend.
        accessibility = math.exp(-max(0.0, state.ne_m3) / max(critical_density, 1.0))
        absorbed = source.power_W * pulse * source.absorption_fraction * accessibility
        density = absorbed / self.geometry.resolved_volume_m3
        hot_power = density * source.hot_fraction
        hot_source = hot_power / (source.hot_creation_energy_eV * E_CHARGE_C)
        current = source.current_drive_sign * source.current_drive_A_per_W * absorbed
        return RFBeamTerms(
            source.name,
            source.frequency_Hz,
            source.harmonic,
            absorbed,
            density,
            hot_power,
            hot_source,
            current,
            accessibility,
            pulse,
            ratio,
            critical_density,
            launch_R,
            source.launch.Z_m,
            source.launch.poloidal_angle_deg,
            source.launch.mode,
            source.launch.polarization,
        )

    def evaluate(self, t_s: float, state: GlobalState) -> RFSourceTerms:
        components = tuple(self._evaluate_source(source, t_s, state) for source in self.sources)
        absorbed = sum(item.power_absorbed_W for item in components)
        density = sum(item.power_density_W_m3 for item in components)
        hot_power = sum(item.hot_power_density_W_m3 for item in components)
        hot_source = sum(item.hot_particle_source_m3_s for item in components)
        current = sum(item.current_drive_A for item in components)
        scheduled_power = [source.power_W * item.pulse_factor for source, item in zip(self.sources, components)]
        scheduled_total = sum(scheduled_power)
        if scheduled_total > 0.0:
            accessibility = sum(
                weight * item.accessibility for weight, item in zip(scheduled_power, components)
            ) / scheduled_total
            ratio = sum(weight * item.omega_ce_over_omega_rf for weight, item in zip(scheduled_power, components)) / scheduled_total
        else:
            accessibility = 0.0
            ratio = components[0].omega_ce_over_omega_rf
        pulse_factor = max((item.pulse_factor for item in components), default=0.0)
        return RFSourceTerms(
            absorbed,
            density,
            hot_power,
            hot_source,
            current,
            accessibility,
            pulse_factor,
            ratio,
            components,
        )
