"""Local 2D hot-plasma frequency-domain Maxwell solver.

The discretization uses first-order Nedelec H(curl) elements from scikit-fem
and a complex-coordinate rectangular PML.  It solves the perpendicular
electromagnetic X/EBW conversion patch; incident O-X coupling remains a
separate mode-matching boundary problem.  The dielectric contains the cold
Stix tensor plus finite-Larmor-radius, Doppler-broadened n=1/n=2 electron
cyclotron loss channels.

This is a local Cartesian conversion-layer solver, not a whole-device port
model.  Quantitative use requires wavelength resolution, mesh convergence,
PML convergence, and a source/absorbed-power balance.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np
from scipy.special import ive
from scipy.sparse.linalg import spsolve

from ..constants import E_CHARGE_C, EPS0_F_M, M_E_KG, PI
from .fokker_planck import C_LIGHT_M_S


MU0_H_M = 1.0 / (EPS0_F_M * C_LIGHT_M_S**2)


ScalarField2D = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class HotPlasmaDielectric:
    frequency_Hz: float
    k_parallel_m_inv: float
    k_perp_m_inv: float
    collision_frequency_s: float = 0.0
    resonance_width_fraction: float = 2.0e-3
    maximum_hot_imaginary: float = 50.0

    def evaluate(
        self,
        electron_density_m3: np.ndarray,
        magnetic_field_T: np.ndarray,
        electron_temperature_eV: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[int, np.ndarray]]:
        omega = 2.0 * PI * self.frequency_Hz
        wp2 = electron_density_m3 * E_CHARGE_C**2 / (EPS0_F_M * M_E_KG)
        wc = E_CHARGE_C * magnetic_field_T / M_E_KG
        damping = max(self.collision_frequency_s, 0.0) / omega
        X = wp2 / omega**2
        Y = wc / omega
        right = 1.0 - X / (1.0 - Y + 1j * damping)
        left = 1.0 - X / (1.0 + Y + 1j * damping)
        stix_s = 0.5 * (right + left)
        stix_d = 0.5 * (right - left)

        temperature_J = np.maximum(electron_temperature_eV, 1.0e-6) * E_CHARGE_C
        thermal_speed = np.sqrt(2.0 * temperature_J / M_E_KG)
        rho_e = thermal_speed / np.maximum(wc, 1.0)
        b = np.maximum((self.k_perp_m_inv * rho_e) ** 2, 1.0e-12)
        doppler = max(abs(self.k_parallel_m_inv), 1.0e-12) * thermal_speed
        width = np.sqrt(doppler**2 + (self.resonance_width_fraction * omega) ** 2)
        harmonic_loss: dict[int, np.ndarray] = {}
        for harmonic in (1, 2):
            xi = (omega - harmonic * wc) / np.maximum(width, 1.0)
            flr = 2.0 * harmonic**2 * ive(harmonic, b) / b
            loss = (
                math.sqrt(PI)
                * X
                * np.maximum(flr, 0.0)
                * np.exp(-np.minimum(xi**2, 100.0))
                * omega
                / np.maximum(width, 1.0)
            )
            harmonic_loss[harmonic] = np.minimum(loss, self.maximum_hot_imaginary)
        hot_loss = harmonic_loss[1] + harmonic_loss[2]
        epsilon_xx = stix_s + 1j * hot_loss
        epsilon_yy = stix_s + 1j * hot_loss
        epsilon_xy = -1j * stix_d
        epsilon_yx = 1j * stix_d
        return epsilon_xx, epsilon_xy, epsilon_yx, epsilon_yy, harmonic_loss


@dataclass(frozen=True)
class NedelecPMLConfig:
    x_bounds_m: tuple[float, float]
    y_bounds_m: tuple[float, float]
    nx: int = 65
    ny: int = 49
    pml_thickness_m: float = 4.0e-3
    pml_strength: float = 5.0
    pml_order: int = 3
    source_center_m: tuple[float, float] = (0.0, 0.0)
    source_width_m: float = 2.0e-3
    source_angle_deg: float = 0.0
    source_current_A_m2: float = 1.0

    def validated(self) -> "NedelecPMLConfig":
        width = self.x_bounds_m[1] - self.x_bounds_m[0]
        height = self.y_bounds_m[1] - self.y_bounds_m[0]
        if width <= 0.0 or height <= 0.0 or self.nx < 9 or self.ny < 9:
            raise ValueError("positive domain and at least 9x9 mesh nodes are required")
        if not 0.0 < self.pml_thickness_m < 0.45 * min(width, height):
            raise ValueError("PML thickness must fit inside the domain")
        if self.pml_strength <= 0.0 or self.pml_order < 1 or self.source_width_m <= 0.0:
            raise ValueError("positive PML/source parameters are required")
        return self


@dataclass(frozen=True)
class HotFullWaveResult:
    element_x_m: np.ndarray
    element_y_m: np.ndarray
    E_x_V_m: np.ndarray
    E_y_V_m: np.ndarray
    absorbed_power_W_per_m: float
    harmonic_absorbed_power_W_per_m: dict[int, float]
    pml_absorbed_power_W_per_m: float
    source_power_W_per_m: float
    power_balance_relative: float
    mesh_nodes: tuple[int, int]
    degrees_of_freedom: int
    cells_per_vacuum_wavelength: float
    linear_residual_relative: float
    mesh_converged: bool = False
    mesh_change_relative: float | None = None

    @property
    def second_harmonic_fraction(self) -> float:
        return self.harmonic_absorbed_power_W_per_m.get(2, 0.0) / max(
            self.absorbed_power_W_per_m, 1.0e-300
        )


def _pml_stretch(
    coordinate: np.ndarray,
    bounds: tuple[float, float],
    thickness: float,
    strength: float,
    order: int,
) -> np.ndarray:
    left_depth = np.maximum(bounds[0] + thickness - coordinate, 0.0) / thickness
    right_depth = np.maximum(coordinate - (bounds[1] - thickness), 0.0) / thickness
    depth = np.maximum(left_depth, right_depth)
    return 1.0 + 1j * strength * depth**order


def solve_hot_plasma_nedelec(
    dielectric: HotPlasmaDielectric,
    config: NedelecPMLConfig,
    electron_density: ScalarField2D,
    magnetic_field: ScalarField2D,
    electron_temperature: ScalarField2D,
) -> HotFullWaveResult:
    """Solve one local conversion patch and return algebraic power checks."""

    from skfem import Basis, BilinearForm, ElementTriN1, LinearForm, MeshTri, asm, condense
    from skfem.helpers import curl

    config = config.validated()
    x_nodes = np.linspace(*config.x_bounds_m, config.nx)
    y_nodes = np.linspace(*config.y_bounds_m, config.ny)
    mesh = MeshTri.init_tensor(x_nodes, y_nodes)
    basis = Basis(mesh, ElementTriN1())
    omega = 2.0 * PI * dielectric.frequency_Hz
    k0 = omega / C_LIGHT_M_S

    def coefficients(w, *, pml: bool, physical_loss_only: bool = False):
        x, y = w.x[0], w.x[1]
        ne = np.asarray(electron_density(x, y), dtype=float)
        field = np.asarray(magnetic_field(x, y), dtype=float)
        temperature = np.asarray(electron_temperature(x, y), dtype=float)
        exx, exy, eyx, eyy, harmonic = dielectric.evaluate(ne, field, temperature)
        if physical_loss_only:
            exx = 1j * np.imag(exx)
            eyy = 1j * np.imag(eyy)
            exy = np.zeros_like(exx)
            eyx = np.zeros_like(exx)
        sx = (
            _pml_stretch(x, config.x_bounds_m, config.pml_thickness_m, config.pml_strength, config.pml_order)
            if pml
            else np.ones_like(x, dtype=complex)
        )
        sy = (
            _pml_stretch(y, config.y_bounds_m, config.pml_thickness_m, config.pml_strength, config.pml_order)
            if pml
            else np.ones_like(y, dtype=complex)
        )
        return exx, exy, eyx, eyy, harmonic, sx, sy

    def make_form(*, pml: bool, physical_loss_only: bool = False):
        @BilinearForm(dtype=np.complex128)
        def form(u, v, w):
            exx, exy, eyx, eyy, _harmonic, sx, sy = coefficients(
                w, pml=pml, physical_loss_only=physical_loss_only
            )
            mass = (
                (sy / sx) * exx * u[0] * v[0]
                + exy * u[1] * v[0]
                + eyx * u[0] * v[1]
                + (sx / sy) * eyy * u[1] * v[1]
            )
            curl_term = curl(u) * curl(v) / (sx * sy)
            return curl_term - k0**2 * mass

        return form

    angle = math.radians(config.source_angle_deg)
    direction = np.asarray([math.cos(angle), math.sin(angle)])

    @LinearForm(dtype=np.complex128)
    def source(v, w):
        radius2 = (
            (w.x[0] - config.source_center_m[0]) ** 2
            + (w.x[1] - config.source_center_m[1]) ** 2
        )
        amplitude = config.source_current_A_m2 * np.exp(
            -0.5 * radius2 / config.source_width_m**2
        )
        return 1j * omega * MU0_H_M * amplitude * (
            direction[0] * v[0] + direction[1] * v[1]
        )

    total_matrix = asm(make_form(pml=True), basis)
    no_pml_matrix = asm(make_form(pml=False), basis)
    rhs = asm(source, basis)
    boundary_dofs = basis.get_dofs().all()
    condensed_matrix, condensed_rhs, x0, free = condense(
        total_matrix, rhs, D=boundary_dofs
    )
    solution_free = spsolve(condensed_matrix.tocsc(), condensed_rhs)
    solution = x0.astype(complex)
    solution[free] = solution_free
    residual = total_matrix @ solution - rhs
    free_residual = residual[free]
    linear_relative = float(
        np.linalg.norm(free_residual) / max(np.linalg.norm(rhs[free]), 1.0e-300)
    )

    normalization = 2.0 * omega * MU0_H_M
    physical_absorption = max(
        0.0,
        float(-np.imag(np.vdot(solution, no_pml_matrix @ solution)) / normalization),
    )
    pml_absorption = max(
        0.0,
        float(-np.imag(np.vdot(solution, (total_matrix - no_pml_matrix) @ solution)) / normalization),
    )
    source_power = max(
        0.0, float(-np.imag(np.vdot(solution, rhs)) / normalization)
    )
    balance = abs(source_power - physical_absorption - pml_absorption) / max(
        source_power, physical_absorption + pml_absorption, 1.0e-300
    )

    field = basis.interpolate(solution)
    coordinates = basis.global_coordinates()
    weights = basis.dx
    element_weight = np.sum(weights, axis=1)
    E_x = np.sum(field[0] * weights, axis=1) / element_weight
    E_y = np.sum(field[1] * weights, axis=1) / element_weight
    centroid_x = np.sum(coordinates[0] * weights, axis=1) / element_weight
    centroid_y = np.sum(coordinates[1] * weights, axis=1) / element_weight

    ne_q = np.asarray(electron_density(coordinates[0], coordinates[1]), dtype=float)
    B_q = np.asarray(magnetic_field(coordinates[0], coordinates[1]), dtype=float)
    Te_q = np.asarray(electron_temperature(coordinates[0], coordinates[1]), dtype=float)
    _, _, _, _, harmonic_loss = dielectric.evaluate(ne_q, B_q, Te_q)
    field_energy = np.abs(field[0]) ** 2 + np.abs(field[1]) ** 2
    channel_raw = {
        harmonic: float(
            0.5 * omega * EPS0_F_M * np.sum(loss * field_energy * weights)
        )
        for harmonic, loss in harmonic_loss.items()
    }
    channel_total = sum(channel_raw.values())
    if channel_total > 0.0 and physical_absorption > 0.0:
        harmonic_power = {
            harmonic: physical_absorption * value / channel_total
            for harmonic, value in channel_raw.items()
        }
    else:
        harmonic_power = {1: 0.0, 2: 0.0}
    wavelength = C_LIGHT_M_S / dielectric.frequency_Hz
    largest_cell = max(
        (config.x_bounds_m[1] - config.x_bounds_m[0]) / (config.nx - 1),
        (config.y_bounds_m[1] - config.y_bounds_m[0]) / (config.ny - 1),
    )
    return HotFullWaveResult(
        centroid_x,
        centroid_y,
        E_x,
        E_y,
        physical_absorption,
        harmonic_power,
        pml_absorption,
        source_power,
        balance,
        (config.nx, config.ny),
        basis.N,
        wavelength / largest_cell,
        linear_relative,
    )


def solve_hot_plasma_mesh_convergence(
    dielectric: HotPlasmaDielectric,
    config: NedelecPMLConfig,
    electron_density: ScalarField2D,
    magnetic_field: ScalarField2D,
    electron_temperature: ScalarField2D,
    *,
    refinement_factor: float = 1.4,
    tolerance: float = 0.08,
) -> tuple[HotFullWaveResult, HotFullWaveResult]:
    """Run two meshes and mark the fine result only if absorbed power converges."""

    if refinement_factor <= 1.0 or tolerance <= 0.0:
        raise ValueError("refinement_factor > 1 and positive tolerance are required")
    coarse = solve_hot_plasma_nedelec(
        dielectric, config, electron_density, magnetic_field, electron_temperature
    )
    fine_config = NedelecPMLConfig(
        **{
            **config.__dict__,
            "nx": max(config.nx + 2, int(round(config.nx * refinement_factor))),
            "ny": max(config.ny + 2, int(round(config.ny * refinement_factor))),
        }
    )
    fine = solve_hot_plasma_nedelec(
        dielectric, fine_config, electron_density, magnetic_field, electron_temperature
    )
    change = abs(fine.absorbed_power_W_per_m - coarse.absorbed_power_W_per_m) / max(
        abs(fine.absorbed_power_W_per_m), abs(coarse.absorbed_power_W_per_m), 1.0e-300
    )
    fine = HotFullWaveResult(
        **{**fine.__dict__, "mesh_converged": change <= tolerance, "mesh_change_relative": change}
    )
    return coarse, fine


__all__ = [
    "HotFullWaveResult",
    "HotPlasmaDielectric",
    "NedelecPMLConfig",
    "solve_hot_plasma_mesh_convergence",
    "solve_hot_plasma_nedelec",
]
