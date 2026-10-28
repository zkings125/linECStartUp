"""Power, momentum and current closure for O-X-B / EBW startup studies."""

from __future__ import annotations

from dataclasses import dataclass

from ..constants import E_CHARGE_C, M_E_KG, PI
from .hot_fullwave import HotFullWaveResult


@dataclass(frozen=True)
class EBWCurrentDriveClosure:
    incident_power_W: float
    ox_efficiency: float
    xb_efficiency: float
    ebw_absorption_efficiency: float
    combined_absorbed_efficiency: float
    absorbed_ebw_power_W: float
    parallel_wave_force_N: float
    momentum_balance_current_A: float
    fp_current_A: float | None
    closure_mismatch_relative: float | None
    quantitative: bool
    reason: str
    dominant_harmonic: int | None


def close_ebw_current_drive(
    fullwave: HotFullWaveResult,
    *,
    incident_power_W: float,
    ox_efficiency: float,
    frequency_Hz: float,
    k_parallel_m_inv: float,
    momentum_confinement_time_s: float,
    parallel_connection_length_m: float,
    effective_gamma: float = 1.0,
    fp_current_A: float | None = None,
    maximum_closure_mismatch: float = 0.5,
) -> EBWCurrentDriveClosure:
    if incident_power_W < 0.0 or not 0.0 <= ox_efficiency <= 1.0:
        raise ValueError("incident power must be non-negative and O-X efficiency in [0,1]")
    if frequency_Hz <= 0.0 or momentum_confinement_time_s <= 0.0:
        raise ValueError("frequency and momentum confinement time must be positive")
    if parallel_connection_length_m <= 0.0 or effective_gamma < 1.0:
        raise ValueError("connection length positive and effective_gamma >= 1 are required")
    converted_input = max(fullwave.source_power_W_per_m, 0.0)
    converted_or_absorbed = (
        fullwave.absorbed_power_W_per_m + fullwave.pml_absorbed_power_W_per_m
    )
    xb_efficiency = converted_or_absorbed / max(converted_input, 1.0e-300)
    xb_efficiency = min(1.0, max(0.0, xb_efficiency))
    local_absorption = fullwave.absorbed_power_W_per_m / max(converted_or_absorbed, 1.0e-300)
    local_absorption = min(1.0, max(0.0, local_absorption))
    combined = ox_efficiency * xb_efficiency * local_absorption
    absorbed = incident_power_W * combined
    omega = 2.0 * PI * frequency_Hz
    force = absorbed * k_parallel_m_inv / omega
    current = (
        -E_CHARGE_C
        * force
        * momentum_confinement_time_s
        / (effective_gamma * M_E_KG * parallel_connection_length_m)
    )
    mismatch = None
    quantitative = bool(
        fullwave.mesh_converged
        and fullwave.power_balance_relative <= 1.0e-3
        and fullwave.cells_per_vacuum_wavelength >= 8.0
    )
    reason = (
        "full-wave mesh/power gates passed; momentum closure remains device-input dependent"
        if quantitative
        else "full-wave mesh, resolution or power-balance gate failed"
    )
    if fp_current_A is not None:
        mismatch = abs(fp_current_A - current) / max(abs(fp_current_A), abs(current), 1.0)
        quantitative = quantitative and mismatch <= maximum_closure_mismatch
        if mismatch > maximum_closure_mismatch:
            reason = "F-P current and wave-momentum current fail the closure tolerance"
    dominant = None
    if fullwave.harmonic_absorbed_power_W_per_m:
        dominant = max(
            fullwave.harmonic_absorbed_power_W_per_m,
            key=fullwave.harmonic_absorbed_power_W_per_m.get,
        )
    return EBWCurrentDriveClosure(
        incident_power_W,
        ox_efficiency,
        xb_efficiency,
        local_absorption,
        combined,
        absorbed,
        force,
        current,
        fp_current_A,
        mismatch,
        quantitative,
        reason,
        dominant,
    )


__all__ = ["EBWCurrentDriveClosure", "close_ebw_current_drive"]
