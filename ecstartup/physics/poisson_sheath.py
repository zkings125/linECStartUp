"""Debye-scale Poisson-Boltzmann/Bohm sheath coupled to a 1D presheath.

The open-field-line interior remains quasineutral and uses electron pressure
balance.  Each material boundary is resolved on its own Debye-length grid by
solving nonlinear Poisson's equation with Boltzmann electrons and cold Bohm
ions.  The wall potential is fixed by floating current balance including an
effective secondary-electron yield.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.integrate import solve_bvp

from ..constants import E_CHARGE_C, EPS0_F_M
from .wall_sheath import M_PROTON_KG, QuasineutralAmbipolarSheath1D, SpatialSheathSolution


@dataclass(frozen=True)
class DebyeSheathProfile:
    distance_from_plasma_m: np.ndarray
    potential_V: np.ndarray
    electric_field_V_m: np.ndarray
    electron_density_m3: np.ndarray
    ion_density_m3: np.ndarray
    charge_density_C_m3: np.ndarray
    debye_length_m: float
    wall_drop_V: float
    poisson_residual_relative: float
    solver_nodes: int


@dataclass(frozen=True)
class PoissonSpatialSheathSolution(SpatialSheathSolution):
    left_debye_sheath: DebyeSheathProfile
    right_debye_sheath: DebyeSheathProfile


class DebyePoissonAmbipolarSheath1D(QuasineutralAmbipolarSheath1D):
    """Quasineutral presheath plus separately resolved Debye sheaths."""

    def __init__(
        self,
        *,
        ion_mass_kg: float = 2.0 * M_PROTON_KG,
        ion_temperature_factor: float = 1.0,
        secondary_electron_yield: float = 0.0,
        density_floor_m3: float = 1.0,
        sheath_extent_debye: float = 12.0,
        sheath_points: int = 161,
        bohm_mach_number: float = 1.0,
        nonlinear_tolerance: float = 1.0e-6,
    ) -> None:
        super().__init__(
            ion_mass_kg=ion_mass_kg,
            ion_temperature_factor=ion_temperature_factor,
            secondary_electron_yield=secondary_electron_yield,
            density_floor_m3=density_floor_m3,
        )
        if sheath_extent_debye <= 1.0 or sheath_points < 21 or bohm_mach_number < 1.0:
            raise ValueError("sheath extent/points must be adequate and Bohm Mach number >= 1")
        if nonlinear_tolerance <= 0.0:
            raise ValueError("nonlinear_tolerance must be positive")
        self.sheath_extent_debye = float(sheath_extent_debye)
        self.sheath_points = int(sheath_points)
        self.bohm_mach_number = float(bohm_mach_number)
        self.nonlinear_tolerance = float(nonlinear_tolerance)

    def solve_boundary(
        self,
        electron_density_m3: float,
        ion_density_m3: float,
        electron_temperature_eV: float,
        ion_temperature_eV: float,
    ) -> DebyeSheathProfile:
        ne0 = max(float(electron_density_m3), self.density_floor_m3)
        ni0 = max(float(ion_density_m3), self.density_floor_m3)
        Te = float(electron_temperature_eV)
        Ti = max(float(ion_temperature_eV), 0.0)
        if Te <= 0.0:
            raise ValueError("electron temperature must be positive")
        drop, _, _, _ = self._boundary(ne0, ni0, Te, Ti)
        psi_wall = -drop / Te
        debye = math.sqrt(EPS0_F_M * Te / (E_CHARGE_C * ne0))
        x = np.linspace(0.0, self.sheath_extent_debye, self.sheath_points)
        guess_psi = psi_wall * (x / x[-1]) ** 2
        guess = np.vstack((guess_psi, np.gradient(guess_psi, x)))
        mach2 = self.bohm_mach_number**2

        def rhs(_x: np.ndarray, y: np.ndarray) -> np.ndarray:
            psi = np.minimum(y[0], 0.0)
            ne_ratio = np.exp(np.maximum(psi, -100.0))
            ni_ratio = (ni0 / ne0) * self.bohm_mach_number / np.sqrt(
                np.maximum(mach2 - 2.0 * psi, 1.0e-12)
            )
            return np.vstack((y[1], ne_ratio - ni_ratio))

        def boundary(left: np.ndarray, right: np.ndarray) -> np.ndarray:
            return np.asarray([left[0], right[0] - psi_wall])

        solution = solve_bvp(
            rhs,
            boundary,
            x,
            guess,
            tol=self.nonlinear_tolerance,
            max_nodes=max(2000, 10 * self.sheath_points),
        )
        if not solution.success:
            raise RuntimeError(f"Debye sheath solve failed: {solution.message}")
        x_dense = np.linspace(0.0, self.sheath_extent_debye, self.sheath_points)
        psi, derivative = solution.sol(x_dense)
        psi = np.minimum(psi, 0.0)
        distance = x_dense * debye
        potential = Te * psi
        electric = -(Te / debye) * derivative
        electron = ne0 * np.exp(np.maximum(psi, -100.0))
        ion = ni0 * self.bohm_mach_number / np.sqrt(
            np.maximum(mach2 - 2.0 * psi, 1.0e-12)
        )
        charge = E_CHARGE_C * (ion - electron)
        dE_ds = np.gradient(electric, distance, edge_order=2)
        residual = dE_ds - charge / EPS0_F_M
        scale = max(float(np.max(np.abs(charge / EPS0_F_M))), 1.0)
        return DebyeSheathProfile(
            distance,
            potential,
            electric,
            electron,
            ion,
            charge,
            debye,
            drop,
            float(np.max(np.abs(residual)) / scale),
            int(solution.x.size),
        )

    def solve(
        self,
        s_m,
        electron_density_m3,
        ion_density_m3,
        electron_temperature_eV,
        *,
        ion_temperature_eV=0.1,
    ) -> PoissonSpatialSheathSolution:
        base = super().solve(
            s_m,
            electron_density_m3,
            ion_density_m3,
            electron_temperature_eV,
            ion_temperature_eV=ion_temperature_eV,
        )
        ne = self._profile(electron_density_m3, len(base.potential_V), "electron_density_m3")
        ni = self._profile(ion_density_m3, len(base.potential_V), "ion_density_m3")
        Te = self._profile(
            electron_temperature_eV,
            len(base.potential_V),
            "electron_temperature_eV",
            positive=True,
        )
        Ti = self._profile(ion_temperature_eV, len(base.potential_V), "ion_temperature_eV")
        left = self.solve_boundary(ne[0], ni[0], Te[0], Ti[0])
        right = self.solve_boundary(ne[-1], ni[-1], Te[-1], Ti[-1])
        return PoissonSpatialSheathSolution(
            **base.__dict__, left_debye_sheath=left, right_debye_sheath=right
        )


__all__ = [
    "DebyePoissonAmbipolarSheath1D",
    "DebyeSheathProfile",
    "PoissonSpatialSheathSolution",
]
