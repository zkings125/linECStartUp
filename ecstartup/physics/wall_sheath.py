"""Quasineutral ambipolar field and floating Bohm-sheath closure in 1D.

The presheath electric field follows the inertialess electron momentum
balance ``E_parallel = -(1/e n_e) d p_e/ds``.  At each open end, the floating
sheath drop is solved from equality of the Bohm ion flux and the net
Maxwellian electron flux, including a configured effective secondary-electron
yield.  The resulting barrier can be passed to the kinetic streaming boundary
to reflect electrons whose parallel kinetic energy is below ``e Delta phi``.

This is a quasineutral kinetic-presheath/Bohm boundary closure.  It does not
resolve the Debye sheath with Poisson's equation or a secondary-electron
energy distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ..constants import E_CHARGE_C, M_E_KG


M_PROTON_KG = 1.67262192595e-27


@dataclass(frozen=True)
class SpatialSheathSolution:
    potential_V: np.ndarray
    E_parallel_V_m: np.ndarray
    sheath_drop_left_V: float
    sheath_drop_right_V: float
    wall_potential_left_V: float
    wall_potential_right_V: float
    ion_flux_left_m2_s: float
    ion_flux_right_m2_s: float
    electron_flux_left_m2_s: float
    electron_flux_right_m2_s: float
    current_residual_left_A_m2: float
    current_residual_right_A_m2: float
    quasineutrality_relative: float

    def to_dict(self) -> dict[str, float | list[float]]:
        return {
            "potential_V": self.potential_V.tolist(),
            "E_parallel_V_m": self.E_parallel_V_m.tolist(),
            "sheath_drop_left_V": self.sheath_drop_left_V,
            "sheath_drop_right_V": self.sheath_drop_right_V,
            "wall_potential_left_V": self.wall_potential_left_V,
            "wall_potential_right_V": self.wall_potential_right_V,
            "ion_flux_left_m2_s": self.ion_flux_left_m2_s,
            "ion_flux_right_m2_s": self.ion_flux_right_m2_s,
            "electron_flux_left_m2_s": self.electron_flux_left_m2_s,
            "electron_flux_right_m2_s": self.electron_flux_right_m2_s,
            "current_residual_left_A_m2": self.current_residual_left_A_m2,
            "current_residual_right_A_m2": self.current_residual_right_A_m2,
            "quasineutrality_relative": self.quasineutrality_relative,
        }


class QuasineutralAmbipolarSheath1D:
    def __init__(
        self,
        *,
        ion_mass_kg: float = 2.0 * M_PROTON_KG,
        ion_temperature_factor: float = 1.0,
        secondary_electron_yield: float = 0.0,
        density_floor_m3: float = 1.0,
    ) -> None:
        if ion_mass_kg <= 0.0 or ion_temperature_factor < 0.0 or density_floor_m3 <= 0.0:
            raise ValueError("ion mass/density floor must be positive and ion factor non-negative")
        if not 0.0 <= secondary_electron_yield < 1.0:
            raise ValueError("secondary_electron_yield must lie in [0,1)")
        self.ion_mass_kg = float(ion_mass_kg)
        self.ion_temperature_factor = float(ion_temperature_factor)
        self.secondary_electron_yield = float(secondary_electron_yield)
        self.density_floor_m3 = float(density_floor_m3)

    @staticmethod
    def _profile(values, length: int, name: str, *, positive: bool = False) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.ndim == 0:
            array = np.full(length, float(array))
        if array.shape != (length,) or np.any(~np.isfinite(array)):
            raise ValueError(f"{name} must be finite and match the s grid")
        if positive and np.any(array <= 0.0):
            raise ValueError(f"{name} must be positive")
        if not positive and np.any(array < 0.0):
            raise ValueError(f"{name} must be non-negative")
        return array

    def _boundary(
        self,
        ne_m3: float,
        ni_m3: float,
        Te_eV: float,
        Ti_eV: float,
    ) -> tuple[float, float, float, float]:
        electron_free = ne_m3 * math.sqrt(E_CHARGE_C * Te_eV / (2.0 * math.pi * M_E_KG))
        sound_speed = math.sqrt(
            E_CHARGE_C * (Te_eV + self.ion_temperature_factor * Ti_eV) / self.ion_mass_kg
        )
        ion_flux = ni_m3 * sound_speed
        net_free = (1.0 - self.secondary_electron_yield) * electron_free
        if ion_flux <= 0.0 or net_free <= ion_flux:
            drop = 0.0
        else:
            drop = Te_eV * math.log(net_free / ion_flux)
        electron_flux = net_free * math.exp(-drop / Te_eV)
        residual = E_CHARGE_C * (ion_flux - electron_flux)
        return drop, ion_flux, electron_flux, residual

    def solve(
        self,
        s_m,
        electron_density_m3,
        ion_density_m3,
        electron_temperature_eV,
        *,
        ion_temperature_eV=0.1,
    ) -> SpatialSheathSolution:
        s = np.asarray(s_m, dtype=float)
        if s.ndim != 1 or len(s) < 2 or np.any(~np.isfinite(s)) or np.any(np.diff(s) <= 0.0):
            raise ValueError("s_m must be finite and strictly increasing")
        ne = self._profile(electron_density_m3, len(s), "electron_density_m3")
        ni = self._profile(ion_density_m3, len(s), "ion_density_m3")
        Te = self._profile(electron_temperature_eV, len(s), "electron_temperature_eV", positive=True)
        Ti = self._profile(ion_temperature_eV, len(s), "ion_temperature_eV")
        ne_safe = np.maximum(ne, self.density_floor_m3)

        pressure_Pa = ne_safe * E_CHARGE_C * Te
        pressure_gradient = np.gradient(pressure_Pa, s, edge_order=1)
        electric = -pressure_gradient / (E_CHARGE_C * ne_safe)
        potential = np.zeros_like(s)
        potential[1:] = np.cumsum(
            -0.5 * (electric[1:] + electric[:-1]) * np.diff(s)
        )
        potential -= potential[len(s) // 2]

        left = self._boundary(ne_safe[0], ni[0], Te[0], Ti[0])
        right = self._boundary(ne_safe[-1], ni[-1], Te[-1], Ti[-1])
        quasineutrality = float(
            np.max(np.abs(ne - ni) / np.maximum(np.maximum(ne, ni), self.density_floor_m3))
        )
        return SpatialSheathSolution(
            potential,
            electric,
            left[0],
            right[0],
            float(potential[0] - left[0]),
            float(potential[-1] - right[0]),
            left[1],
            right[1],
            left[2],
            right[2],
            left[3],
            right[3],
            quasineutrality,
        )


__all__ = ["SpatialSheathSolution", "QuasineutralAmbipolarSheath1D"]
