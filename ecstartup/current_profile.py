"""Causal current decomposition and normalized geometric-rho reconstruction.

The startup model is 0D, so these profiles are normalized reconstructions,
not solved magnetic-flux profiles.  Before CFS, ``rho`` means r/a in the
poloidal cross-section and must not be interpreted as sqrt(normalized flux).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import beta as beta_function

from .config import CurrentProfileConfig, SimulationConfig
from .solver import RunResult


@dataclass(frozen=True)
class MechanismCurrentEvolution:
    times_s: np.ndarray
    Ip_A: np.ndarray
    I_cs_A: np.ndarray
    I_rf_A: np.ndarray
    I_tail_A: np.ndarray
    I_vertical_A: np.ndarray
    I_initial_A: np.ndarray
    I_closure_residual_A: np.ndarray

    @property
    def reconstructed_A(self) -> np.ndarray:
        return (
            self.I_cs_A
            + self.I_rf_A
            + self.I_tail_A
            + self.I_vertical_A
            + self.I_initial_A
            + self.I_closure_residual_A
        )


@dataclass(frozen=True)
class CurrentProfileEvolution:
    times_s: np.ndarray
    rho_geo: np.ndarray
    j_total_A_m2: np.ndarray
    j_cs_A_m2: np.ndarray
    j_rf_A_m2: np.ndarray
    j_tail_A_m2: np.ndarray
    j_vertical_A_m2: np.ndarray
    j_initial_A_m2: np.ndarray
    j_closure_residual_A_m2: np.ndarray
    coordinate_definition: str = "rho_geo = r/minor_radius; not a flux coordinate before CFS"

    def save_npz(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            times_s=self.times_s,
            rho_geo=self.rho_geo,
            j_total_A_m2=self.j_total_A_m2,
            j_cs_A_m2=self.j_cs_A_m2,
            j_rf_A_m2=self.j_rf_A_m2,
            j_tail_A_m2=self.j_tail_A_m2,
            j_vertical_A_m2=self.j_vertical_A_m2,
            j_initial_A_m2=self.j_initial_A_m2,
            j_closure_residual_A_m2=self.j_closure_residual_A_m2,
            coordinate_definition=np.asarray(self.coordinate_definition),
        )


def _advance_linear_component(value: float, source_A_s: float, decay_s: float, dt_s: float) -> float:
    if dt_s <= 0.0:
        return value
    if decay_s <= 1.0e-14:
        return value + source_A_s * dt_s
    decay = float(np.exp(-min(decay_s * dt_s, 745.0)))
    return value * decay + source_A_s * (1.0 - decay) / decay_s


def reconstruct_mechanism_currents(
    result: RunResult, config: SimulationConfig
) -> MechanismCurrentEvolution:
    """Propagate each source through the same sampled L-R response as Ip."""

    times = np.asarray(result.times_s, dtype=float)
    count = len(times)
    ip = np.asarray([float(row.get("Ip_A", 0.0)) for row in result.diagnostics])
    cs = np.zeros(count)
    rf = np.zeros(count)
    tail = np.zeros(count)
    vertical = np.zeros(count)
    initial = np.zeros(count)
    residual = np.zeros(count)
    initial[0] = ip[0]
    inductance = float(config.circuit.total_inductance_H)

    for index in range(count - 1):
        dt_s = float(times[index + 1] - times[index])
        left = result.diagnostics[index]
        right = result.diagnostics[index + 1]

        def midpoint(key: str) -> float:
            return 0.5 * (float(left.get(key, 0.0)) + float(right.get(key, 0.0)))

        resistance = max(midpoint("R_plasma_ohm"), 0.0)
        decay_s = resistance / inductance
        cs_source = midpoint("effective_loop_voltage_V") / inductance
        rf_source = decay_s * midpoint("I_rf_A")
        tail_source = decay_s * midpoint("I_tail_A")
        vertical_source = decay_s * midpoint("I_vertical_charge_A")
        cs[index + 1] = _advance_linear_component(cs[index], cs_source, decay_s, dt_s)
        rf[index + 1] = _advance_linear_component(rf[index], rf_source, decay_s, dt_s)
        tail[index + 1] = _advance_linear_component(tail[index], tail_source, decay_s, dt_s)
        vertical[index + 1] = _advance_linear_component(
            vertical[index], vertical_source, decay_s, dt_s
        )
        initial[index + 1] = _advance_linear_component(initial[index], 0.0, decay_s, dt_s)

    subtotal = cs + rf + tail + vertical + initial
    residual[:] = ip - subtotal
    return MechanismCurrentEvolution(times, ip, cs, rf, tail, vertical, initial, residual)


def normalized_beta_profile(
    rho_geo: np.ndarray,
    minor_radius_m: float,
    shape: tuple[float, float],
) -> np.ndarray:
    """Return g(rho) normalized so integral(g*dA)=1 on a circular section."""

    p_shape, q_shape = map(float, shape)
    rho = np.asarray(rho_geo, dtype=float)
    normalization = (
        np.pi
        * minor_radius_m**2
        * float(beta_function(p_shape + 1.0, q_shape + 1.0))
    )
    return np.power(rho, 2.0 * p_shape) * np.power(1.0 - rho**2, q_shape) / normalization


def reconstruct_current_profiles(
    result: RunResult, config: SimulationConfig
) -> CurrentProfileEvolution:
    currents = reconstruct_mechanism_currents(result, config)
    profile_config: CurrentProfileConfig = config.current_profile
    rho = np.linspace(0.0, 1.0, config.outputs.profile_radial_points)
    radius = float(config.geometry.minor_radius_m)

    def expand(current: np.ndarray, shape: tuple[float, float]) -> np.ndarray:
        basis = normalized_beta_profile(rho, radius, shape)
        return current[:, np.newaxis] * basis[np.newaxis, :]

    j_cs = expand(currents.I_cs_A, profile_config.ohmic_shape)
    j_rf = expand(currents.I_rf_A, profile_config.rf_shape)
    j_tail = expand(currents.I_tail_A, profile_config.tail_shape)
    j_vertical = expand(currents.I_vertical_A, profile_config.vertical_shape)
    j_initial = expand(currents.I_initial_A, profile_config.residual_shape)
    j_residual = expand(currents.I_closure_residual_A, profile_config.residual_shape)
    total = j_cs + j_rf + j_tail + j_vertical + j_initial + j_residual
    return CurrentProfileEvolution(
        times_s=currents.times_s,
        rho_geo=rho,
        j_total_A_m2=total,
        j_cs_A_m2=j_cs,
        j_rf_A_m2=j_rf,
        j_tail_A_m2=j_tail,
        j_vertical_A_m2=j_vertical,
        j_initial_A_m2=j_initial,
        j_closure_residual_A_m2=j_residual,
    )
