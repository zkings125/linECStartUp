"""Coupled coil-plasma inductive circuit network.

The network solves ``L dI/dt + R I = V`` for named PF/CS/plasma circuits.
It can also evaluate the plasma loop voltage induced by measured coil-current
waveforms.  Mutual inductances remain explicit engineering inputs; they are
not inferred from an EFIT plasma equilibrium or tuned from the target Ip.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class CircuitNetworkState:
    time_s: float
    current_A: np.ndarray


@dataclass(frozen=True)
class CircuitNetworkDiagnostics:
    input_power_W: float
    ohmic_power_W: float
    magnetic_energy_J: float
    equation_residual_relative: float

    def to_dict(self) -> dict[str, float]:
        return self.__dict__.copy()


class InductiveCircuitNetwork:
    def __init__(
        self,
        names: Sequence[str],
        inductance_H: np.ndarray,
        resistance_ohm: Sequence[float],
        *,
        plasma_circuit: str = "plasma",
    ) -> None:
        self.names = tuple(str(name) for name in names)
        if not self.names or len(self.names) != len(set(self.names)):
            raise ValueError("circuit names must be non-empty and unique")
        self.inductance_H = np.asarray(inductance_H, dtype=float)
        self.resistance_ohm = np.asarray(resistance_ohm, dtype=float)
        count = len(self.names)
        if self.inductance_H.shape != (count, count):
            raise ValueError("inductance matrix shape must match circuit names")
        if self.resistance_ohm.shape != (count,):
            raise ValueError("resistance vector shape must match circuit names")
        if np.any(~np.isfinite(self.inductance_H)) or np.any(~np.isfinite(self.resistance_ohm)):
            raise ValueError("circuit coefficients must be finite")
        if np.any(self.resistance_ohm < 0.0):
            raise ValueError("circuit resistances must be non-negative")
        if not np.allclose(self.inductance_H, self.inductance_H.T, rtol=1.0e-10, atol=1.0e-14):
            raise ValueError("inductance matrix must be symmetric")
        eigenvalues = np.linalg.eigvalsh(self.inductance_H)
        if np.min(eigenvalues) <= 0.0:
            raise ValueError("inductance matrix must be positive definite")
        try:
            self.plasma_index = self.names.index(plasma_circuit)
        except ValueError as exc:
            raise ValueError(f"plasma circuit {plasma_circuit!r} is not in names") from exc

    def initial_state(self, current_A: Sequence[float] | None = None, *, time_s: float = 0.0) -> CircuitNetworkState:
        current = np.zeros(len(self.names)) if current_A is None else np.asarray(current_A, dtype=float)
        if current.shape != (len(self.names),) or np.any(~np.isfinite(current)):
            raise ValueError("initial current vector must be finite and match circuit names")
        return CircuitNetworkState(float(time_s), current.copy())

    def advance(
        self,
        state: CircuitNetworkState,
        dt_s: float,
        applied_voltage_V: Sequence[float],
        *,
        resistance_ohm: Sequence[float] | None = None,
    ) -> tuple[CircuitNetworkState, CircuitNetworkDiagnostics]:
        if not math.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("dt_s must be finite and positive")
        old = np.asarray(state.current_A, dtype=float)
        voltage = np.asarray(applied_voltage_V, dtype=float)
        resistance = self.resistance_ohm if resistance_ohm is None else np.asarray(resistance_ohm, dtype=float)
        expected = (len(self.names),)
        if old.shape != expected or voltage.shape != expected or resistance.shape != expected:
            raise ValueError("state, voltage and resistance vectors must match circuit names")
        if np.any(~np.isfinite(voltage)) or np.any(~np.isfinite(resistance)) or np.any(resistance < 0.0):
            raise ValueError("voltage/resistance values must be finite and resistance non-negative")
        matrix = self.inductance_H / dt_s + np.diag(resistance)
        rhs = voltage + self.inductance_H @ old / dt_s
        current = np.linalg.solve(matrix, rhs)
        derivative = (current - old) / dt_s
        residual = self.inductance_H @ derivative + resistance * current - voltage
        denominator = max(float(np.linalg.norm(voltage)), 1.0e-30)
        diagnostics = CircuitNetworkDiagnostics(
            input_power_W=float(np.dot(current, voltage)),
            ohmic_power_W=float(np.dot(resistance, current**2)),
            magnetic_energy_J=float(0.5 * current @ self.inductance_H @ current),
            equation_residual_relative=float(np.linalg.norm(residual) / denominator),
        )
        return CircuitNetworkState(state.time_s + dt_s, current), diagnostics

    def induced_plasma_loop_voltage(
        self,
        current_waveforms_A: Mapping[str, Sequence[tuple[float, float]]],
        sample_times_s: Sequence[float],
    ) -> np.ndarray:
        """Evaluate ``-sum_k M_pk dI_k/dt`` from measured non-plasma currents."""

        times = np.asarray(sample_times_s, dtype=float)
        if times.ndim != 1 or len(times) < 2 or np.any(np.diff(times) <= 0.0):
            raise ValueError("sample_times_s must be a strictly increasing one-dimensional grid")
        induced = np.zeros_like(times)
        for index, name in enumerate(self.names):
            if index == self.plasma_index:
                continue
            if name not in current_waveforms_A:
                raise ValueError(f"missing measured waveform for circuit {name}")
            points = np.asarray(current_waveforms_A[name], dtype=float)
            if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
                raise ValueError(f"waveform for {name} must contain at least two (time,current) points")
            currents = np.interp(times, points[:, 0], points[:, 1])
            derivative = np.gradient(currents, times, edge_order=1)
            induced -= self.inductance_H[self.plasma_index, index] * derivative
        return induced


__all__ = [
    "CircuitNetworkState",
    "CircuitNetworkDiagnostics",
    "InductiveCircuitNetwork",
]
