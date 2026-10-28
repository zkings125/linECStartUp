"""Reduced 0D-2V electron Fokker--Planck solver.

The module follows CQL3D's separation of collision/electric-field
coefficients, RF diffusion, losses, implicit advancement, and moment
diagnostics. It is an independent conservative finite-volume implementation,
not a copy or replacement of CQL3D's full relativistic bounce-averaged model.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
from typing import Protocol

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, eye
from scipy.sparse.linalg import SuperLU, splu

from ..constants import E_CHARGE_C, M_E_KG, PI

C_LIGHT_M_S = 299_792_458.0


@dataclass(frozen=True)
class VelocityGrid:
    """Cell-centred cylindrical velocity grid."""

    v_parallel_m_s: np.ndarray
    v_perp_m_s: np.ndarray

    def __post_init__(self) -> None:
        parallel = np.asarray(self.v_parallel_m_s, dtype=float)
        perpendicular = np.asarray(self.v_perp_m_s, dtype=float)
        if parallel.ndim != 1 or perpendicular.ndim != 1:
            raise ValueError("velocity coordinates must be one-dimensional")
        if len(parallel) < 3 or len(perpendicular) < 2:
            raise ValueError("at least 3 parallel and 2 perpendicular cells are required")
        if not np.all(np.diff(parallel) > 0.0) or not np.all(np.diff(perpendicular) > 0.0):
            raise ValueError("velocity coordinates must be strictly increasing")
        if perpendicular[0] <= 0.0:
            raise ValueError("v_perp cell centres must be positive")
        if not np.allclose(np.diff(parallel), np.diff(parallel)[0], rtol=1.0e-10):
            raise ValueError("the reduced solver requires uniform v_parallel spacing")
        if not np.allclose(np.diff(perpendicular), np.diff(perpendicular)[0], rtol=1.0e-10):
            raise ValueError("the reduced solver requires uniform v_perp spacing")
        object.__setattr__(self, "v_parallel_m_s", parallel)
        object.__setattr__(self, "v_perp_m_s", perpendicular)

    @classmethod
    def uniform(
        cls,
        n_parallel: int,
        n_perp: int,
        v_parallel_max_m_s: float,
        v_perp_max_m_s: float,
    ) -> "VelocityGrid":
        if n_parallel < 3 or n_perp < 2:
            raise ValueError("n_parallel >= 3 and n_perp >= 2 are required")
        if v_parallel_max_m_s <= 0.0 or v_perp_max_m_s <= 0.0:
            raise ValueError("velocity bounds must be positive")
        dv_parallel = 2.0 * v_parallel_max_m_s / n_parallel
        dv_perp = v_perp_max_m_s / n_perp
        parallel = -v_parallel_max_m_s + (np.arange(n_parallel) + 0.5) * dv_parallel
        perpendicular = (np.arange(n_perp) + 0.5) * dv_perp
        return cls(parallel, perpendicular)

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.v_parallel_m_s), len(self.v_perp_m_s)

    @property
    def dv_parallel_m_s(self) -> float:
        return float(self.v_parallel_m_s[1] - self.v_parallel_m_s[0])

    @property
    def dv_perp_m_s(self) -> float:
        return float(self.v_perp_m_s[1] - self.v_perp_m_s[0])

    @property
    def phase_volume_weights_m3_s3(self) -> np.ndarray:
        perpendicular_weights = (
            2.0 * PI * self.v_perp_m_s * self.dv_perp_m_s * self.dv_parallel_m_s
        )
        return np.broadcast_to(perpendicular_weights, self.shape).copy()

    def mesh(self) -> tuple[np.ndarray, np.ndarray]:
        return np.meshgrid(self.v_parallel_m_s, self.v_perp_m_s, indexing="ij")


@dataclass(frozen=True)
class RelativisticMomentumGrid:
    """Cell-centred momentum grid using ``u=p/(m_e c)`` coordinates.

    The exposed velocity arrays and cylindrical phase weights are physical
    relativistic values, while ``du_parallel``/``du_perp`` are used by the
    momentum-space finite-volume operator.  This keeps distribution moments
    in SI and makes the momentum Jacobian explicit instead of silently using a
    non-relativistic ``m v`` approximation.
    """

    u_parallel: np.ndarray
    u_perp: np.ndarray

    def __post_init__(self) -> None:
        up = np.asarray(self.u_parallel, dtype=float)
        ut = np.asarray(self.u_perp, dtype=float)
        if up.ndim != 1 or ut.ndim != 1 or len(up) < 3 or len(ut) < 2:
            raise ValueError("relativistic momentum grid requires at least 3x2 cells")
        if not np.all(np.diff(up) > 0.0) or not np.all(np.diff(ut) > 0.0) or ut[0] <= 0.0:
            raise ValueError("momentum coordinates must be increasing and u_perp positive")
        if not np.allclose(np.diff(up), np.diff(up)[0], rtol=1.0e-10):
            raise ValueError("relativistic momentum grid requires uniform u_parallel")
        if not np.allclose(np.diff(ut), np.diff(ut)[0], rtol=1.0e-10):
            raise ValueError("relativistic momentum grid requires uniform u_perp")
        object.__setattr__(self, "u_parallel", up)
        object.__setattr__(self, "u_perp", ut)

    @classmethod
    def uniform(cls, n_parallel: int, n_perp: int, u_parallel_max: float, u_perp_max: float) -> "RelativisticMomentumGrid":
        if n_parallel < 3 or n_perp < 2 or u_parallel_max <= 0.0 or u_perp_max <= 0.0:
            raise ValueError("momentum bounds and grid sizes must be positive")
        return cls(
            -u_parallel_max + (np.arange(n_parallel) + 0.5) * 2.0 * u_parallel_max / n_parallel,
            (np.arange(n_perp) + 0.5) * u_perp_max / n_perp,
        )

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.u_parallel), len(self.u_perp)

    @property
    def du_parallel(self) -> float:
        return float(self.u_parallel[1] - self.u_parallel[0])

    @property
    def du_perp(self) -> float:
        return float(self.u_perp[1] - self.u_perp[0])

    def mesh_u(self) -> tuple[np.ndarray, np.ndarray]:
        return np.meshgrid(self.u_parallel, self.u_perp, indexing="ij")

    def mesh(self) -> tuple[np.ndarray, np.ndarray]:
        up, ut = self.mesh_u()
        gamma = np.sqrt(1.0 + up**2 + ut**2)
        return C_LIGHT_M_S * up / gamma, C_LIGHT_M_S * ut / gamma

    @property
    def v_parallel_m_s(self) -> np.ndarray:
        up, _ = self.mesh_u()
        # Return the parallel cell-centre coordinate in a shape-compatible
        # 1-D form only when needed by diagnostics; mesh() is authoritative.
        return C_LIGHT_M_S * up[:, 0] / np.sqrt(1.0 + up[:, 0] ** 2)

    @property
    def v_perp_m_s(self) -> np.ndarray:
        return C_LIGHT_M_S * self.u_perp / np.sqrt(1.0 + self.u_perp**2)

    @property
    def phase_volume_weights_m3_s3(self) -> np.ndarray:
        up, ut = self.mesh_u()
        # d^3p/m^3 = 2*pi*u_perp du_perp du_parallel; physical velocity-space
        # moment uses d^3v = d^3p/(m^3 gamma^5).
        gamma = np.sqrt(1.0 + up**2 + ut**2)
        return 2.0 * PI * ut * self.du_perp * self.du_parallel * C_LIGHT_M_S**3 / gamma**5


def relativistic_momentum_grid(n_parallel: int, n_perp: int, u_parallel_max: float, u_perp_max: float) -> RelativisticMomentumGrid:
    """Public constructor used by configuration/workflow adapters."""

    return RelativisticMomentumGrid.uniform(n_parallel, n_perp, u_parallel_max, u_perp_max)


@dataclass
class KineticState2V:
    """Electron distribution on a velocity or relativistic-momentum grid."""

    f_e: np.ndarray
    grid: VelocityGrid | RelativisticMomentumGrid
    time_s: float

    def __post_init__(self) -> None:
        self.f_e = np.asarray(self.f_e, dtype=float)
        if self.f_e.shape != self.grid.shape:
            raise ValueError(f"f_e shape {self.f_e.shape} does not match grid {self.grid.shape}")
        if not np.all(np.isfinite(self.f_e)):
            raise ValueError("f_e contains non-finite values")


@dataclass(frozen=True)
class QuasilinearDiffusion2V:
    """2-D quasilinear diffusion tensor in ``m^2 s^-3``.

    ``D_cross`` is the symmetric off-diagonal coefficient.  Validation enforces
    positive semidefiniteness, ``D_parallel*D_perp-D_cross^2 >= 0``.
    """

    D_parallel_m2_s3: np.ndarray
    D_perp_m2_s3: np.ndarray
    source: str = "external"
    D_cross_m2_s3: np.ndarray | None = None

    def validated(self, grid: VelocityGrid | RelativisticMomentumGrid) -> "QuasilinearDiffusion2V":
        parallel = np.asarray(self.D_parallel_m2_s3, dtype=float)
        perpendicular = np.asarray(self.D_perp_m2_s3, dtype=float)
        cross = np.zeros(grid.shape, dtype=float) if self.D_cross_m2_s3 is None else np.asarray(self.D_cross_m2_s3, dtype=float)
        if parallel.shape != grid.shape or perpendicular.shape != grid.shape or cross.shape != grid.shape:
            raise ValueError("quasilinear diffusion arrays must match the velocity grid")
        if np.any(~np.isfinite(parallel)) or np.any(~np.isfinite(perpendicular)) or np.any(~np.isfinite(cross)):
            raise ValueError("quasilinear diffusion contains non-finite values")
        if np.any(parallel < 0.0) or np.any(perpendicular < 0.0):
            raise ValueError("diagonal quasilinear diffusion must be non-negative")
        determinant = parallel * perpendicular - cross**2
        if np.any(determinant < -1.0e-12 * np.maximum(parallel * perpendicular, 1.0)):
            raise ValueError("quasilinear diffusion tensor must be positive semidefinite")
        return QuasilinearDiffusion2V(parallel, perpendicular, self.source, cross)

    @classmethod
    def zeros(cls, grid: VelocityGrid | RelativisticMomentumGrid) -> "QuasilinearDiffusion2V":
        zeros = np.zeros(grid.shape, dtype=float)
        return cls(zeros.copy(), zeros.copy(), "disabled", zeros.copy())

    @classmethod
    def gaussian_resonance(
        cls,
        grid: VelocityGrid | RelativisticMomentumGrid,
        *,
        peak_D_parallel_m2_s3: float = 0.0,
        peak_D_perp_m2_s3: float = 0.0,
        resonant_v_parallel_m_s: float = 0.0,
        resonant_v_perp_m_s: float = 0.0,
        sigma_v_parallel_m_s: float,
        sigma_v_perp_m_s: float,
        cross_fraction: float = 0.0,
    ) -> "QuasilinearDiffusion2V":
        """Synthetic resonance patch for tests and sensitivity scans only."""

        if min(peak_D_parallel_m2_s3, peak_D_perp_m2_s3) < 0.0:
            raise ValueError("diffusion peaks must be non-negative")
        if sigma_v_parallel_m_s <= 0.0 or sigma_v_perp_m_s <= 0.0:
            raise ValueError("resonance widths must be positive")
        parallel, perpendicular = grid.mesh()
        envelope = np.exp(
            -0.5 * ((parallel - resonant_v_parallel_m_s) / sigma_v_parallel_m_s) ** 2
            -0.5 * ((perpendicular - resonant_v_perp_m_s) / sigma_v_perp_m_s) ** 2
        )
        cross = cross_fraction * np.sqrt(
            np.maximum(peak_D_parallel_m2_s3 * envelope, 0.0)
            * np.maximum(peak_D_perp_m2_s3 * envelope, 0.0)
        )
        if abs(cross_fraction) > 1.0:
            raise ValueError("cross_fraction must lie in [-1, 1]")
        return cls(
            peak_D_parallel_m2_s3 * envelope,
            peak_D_perp_m2_s3 * envelope,
            "synthetic_gaussian_resonance",
            cross,
        )


@dataclass(frozen=True)
class KineticMoments:
    density_m3: float
    mean_v_parallel_m_s: float
    energy_density_J_m3: float
    temperature_eV: float
    current_density_parallel_A_m2: float
    current_parallel_A: float
    pressure_parallel_Pa: float
    pressure_perp_Pa: float

    def to_dict(self) -> dict[str, float]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class FokkerPlanckStepDiagnostics:
    particle_number_before_m3: float
    particle_number_after_m3: float
    particle_balance_relative: float
    minimum_f: float
    clipped_negative_density_m3: float
    linear_residual_relative: float
    current_parallel_A: float
    temperature_eV: float
    rf_diffusion_source: str
    momentum_balance_relative: float = 0.0
    field_particle_source_A: float = 0.0

    def to_dict(self) -> dict[str, float | str]:
        return self.__dict__.copy()


class KineticOperator(Protocol):
    def advance(
        self, state: KineticState2V, dt_s: float, **kwargs
    ) -> tuple[KineticState2V, FokkerPlanckStepDiagnostics]: ...


class ReducedFokkerPlanck2V:
    """Conservative implicit finite-volume 0D-2V electron solver."""

    def __init__(
        self,
        grid: VelocityGrid,
        *,
        cross_section_area_m2: float,
        electron_mass_kg: float = M_E_KG,
        electron_charge_C: float = -E_CHARGE_C,
    ) -> None:
        if cross_section_area_m2 <= 0.0:
            raise ValueError("cross_section_area_m2 must be positive")
        self.grid = grid
        self.cross_section_area_m2 = float(cross_section_area_m2)
        self.mass_kg = float(electron_mass_kg)
        self.charge_C = float(electron_charge_C)
        self._weights = grid.phase_volume_weights_m3_s3
        self._linear_cache: OrderedDict[tuple, tuple[csr_matrix, SuperLU]] = OrderedDict()
        self._linear_cache_max_entries = 8
        self._linear_cache_hits = 0
        self._linear_cache_misses = 0

    def linear_solver_cache_info(self) -> dict[str, int]:
        """Return sparse-factorization cache counters for performance tests."""

        return {
            "entries": len(self._linear_cache),
            "hits": self._linear_cache_hits,
            "misses": self._linear_cache_misses,
        }

    @staticmethod
    def _array_signature(array: np.ndarray) -> bytes:
        contiguous = np.ascontiguousarray(array, dtype=np.float64)
        return hashlib.blake2b(contiguous.view(np.uint8), digest_size=12).digest()

    def maxwellian_state(
        self,
        density_m3: float,
        temperature_eV: float,
        *,
        drift_v_parallel_m_s: float = 0.0,
        time_s: float = 0.0,
    ) -> KineticState2V:
        if density_m3 < 0.0 or temperature_eV <= 0.0:
            raise ValueError("density must be non-negative and temperature positive")
        parallel, perpendicular = self.grid.mesh()
        variance = temperature_eV * E_CHARGE_C / self.mass_kg
        exponent = -(
            (parallel - drift_v_parallel_m_s) ** 2 + perpendicular**2
        ) / (2.0 * variance)
        distribution = np.exp(np.maximum(exponent, -745.0))
        normalization = float(np.sum(distribution * self._weights))
        if normalization <= 0.0:
            raise ValueError("velocity domain does not resolve the requested Maxwellian")
        distribution *= density_m3 / normalization
        return KineticState2V(distribution, self.grid, time_s)

    def moments(self, state: KineticState2V) -> KineticMoments:
        self._check_state(state)
        distribution = np.maximum(state.f_e, 0.0)
        parallel, perpendicular = self.grid.mesh()
        density = float(np.sum(distribution * self._weights))
        if density <= 0.0:
            return KineticMoments(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        parallel_moment = float(np.sum(parallel * distribution * self._weights))
        mean_parallel = parallel_moment / density
        energy_density = float(
            0.5
            * self.mass_kg
            * np.sum((parallel**2 + perpendicular**2) * distribution * self._weights)
        )
        thermal_energy = max(
            0.0,
            energy_density - 0.5 * self.mass_kg * density * mean_parallel**2,
        )
        temperature_eV = 2.0 * thermal_energy / (3.0 * density * E_CHARGE_C)
        current_density = self.charge_C * parallel_moment
        pressure_parallel = float(
            self.mass_kg
            * np.sum((parallel - mean_parallel) ** 2 * distribution * self._weights)
        )
        pressure_perp = float(
            0.5 * self.mass_kg * np.sum(perpendicular**2 * distribution * self._weights)
        )
        return KineticMoments(
            density,
            mean_parallel,
            energy_density,
            temperature_eV,
            current_density,
            current_density * self.cross_section_area_m2,
            pressure_parallel,
            pressure_perp,
        )

    def advance(
        self,
        state: KineticState2V,
        dt_s: float,
        *,
        E_parallel_V_m: float = 0.0,
        collision_frequency_s: float = 0.0,
        collision_temperature_eV: float | None = None,
        rf_diffusion: QuasilinearDiffusion2V | None = None,
        loss_frequency_s: float | np.ndarray = 0.0,
        source_f_s: np.ndarray | None = None,
        positivity_tolerance: float = 1.0e-12,
        momentum_conserving_field_particle: bool = False,
    ) -> tuple[KineticState2V, FokkerPlanckStepDiagnostics]:
        self._check_state(state)
        if dt_s <= 0.0 or collision_frequency_s < 0.0:
            raise ValueError("dt_s must be positive and collision frequency non-negative")
        if collision_temperature_eV is None:
            collision_temperature_eV = max(self.moments(state).temperature_eV, 1.0e-6)
        if collision_temperature_eV <= 0.0:
            raise ValueError("collision_temperature_eV must be positive")
        rf = (rf_diffusion or QuasilinearDiffusion2V.zeros(self.grid)).validated(self.grid)
        loss = self._coefficient_array(loss_frequency_s, "loss_frequency_s")
        if np.any(loss < 0.0):
            raise ValueError("loss_frequency_s must be non-negative")
        source = np.zeros(self.grid.shape) if source_f_s is None else np.asarray(source_f_s, dtype=float)
        if source.shape != self.grid.shape or np.any(~np.isfinite(source)):
            raise ValueError("source_f_s must be finite and match the grid")
        density = float(np.sum(np.maximum(state.f_e, 0.0) * self._weights))

        collision_diffusion = (
            collision_frequency_s * collision_temperature_eV * E_CHARGE_C / self.mass_kg
        )
        parallel, perpendicular = self.grid.mesh()
        acceleration_parallel = (
            self.charge_C * E_parallel_V_m / self.mass_kg
            - collision_frequency_s * parallel
        )
        acceleration_perp = -collision_frequency_s * perpendicular
        field_particle_source = np.zeros(self.grid.shape, dtype=float)
        if momentum_conserving_field_particle and collision_frequency_s > 0.0:
            # Linearized field-particle term: return the test-particle drag of
            # the bulk drift to a Maxwellian background. This is the first
            # momentum-conserving correction, not a full nonlinear Rosenbluth
            # C_ee/C_ei operator.
            mean_parallel = float(np.sum(parallel * np.maximum(state.f_e, 0.0) * self._weights)) / max(density, 1.0)
            variance = collision_temperature_eV * E_CHARGE_C / self.mass_kg
            equilibrium = np.exp(np.maximum(-(parallel**2 + perpendicular**2) / (2.0 * variance), -745.0))
            equilibrium *= density / max(float(np.sum(equilibrium * self._weights)), 1.0e-300)
            field_particle_source = collision_frequency_s * mean_parallel * parallel / max(variance, 1.0e-300) * equilibrium
            source = source + field_particle_source
        total_parallel_diffusion = collision_diffusion + rf.D_parallel_m2_s3
        total_perp_diffusion = collision_diffusion + rf.D_perp_m2_s3
        cache_key = (
            float(dt_s),
            self._array_signature(acceleration_parallel),
            self._array_signature(acceleration_perp),
            self._array_signature(total_parallel_diffusion),
            self._array_signature(total_perp_diffusion),
            self._array_signature(rf.D_cross_m2_s3),
            self._array_signature(loss),
        )
        cached = self._linear_cache.get(cache_key)
        if cached is None:
            operator = self._assemble_operator(
                acceleration_parallel,
                acceleration_perp,
                total_parallel_diffusion,
                total_perp_diffusion,
                loss,
                rf.D_cross_m2_s3,
            )
            lhs = eye(operator.shape[0], format="csr") - dt_s * operator
            factorization = splu(lhs.tocsc())
            self._linear_cache[cache_key] = (lhs, factorization)
            self._linear_cache.move_to_end(cache_key)
            while len(self._linear_cache) > self._linear_cache_max_entries:
                self._linear_cache.popitem(last=False)
            self._linear_cache_misses += 1
        else:
            lhs, factorization = cached
            self._linear_cache.move_to_end(cache_key)
            self._linear_cache_hits += 1
        rhs = state.f_e.ravel() + dt_s * source.ravel()
        solution = np.asarray(factorization.solve(rhs), dtype=float)
        if np.any(~np.isfinite(solution)):
            raise FloatingPointError("Fokker--Planck solve produced non-finite values")
        residual = lhs @ solution - rhs
        residual_relative = float(np.linalg.norm(residual) / max(np.linalg.norm(rhs), 1.0e-300))

        raw = solution.reshape(self.grid.shape)
        negative = np.minimum(raw, 0.0)
        clipped_negative_density = float(-np.sum(negative * self._weights))
        scale = max(float(np.max(np.abs(raw))), 1.0e-300)
        if float(np.min(raw)) < -positivity_tolerance * scale:
            raise FloatingPointError("Fokker--Planck step violated positivity")
        updated = KineticState2V(np.maximum(raw, 0.0), self.grid, state.time_s + dt_s)
        before = self.moments(state)
        after = self.moments(updated)
        expected_density = before.density_m3 + dt_s * float(np.sum(source * self._weights))
        expected_density -= dt_s * float(np.sum(loss * updated.f_e * self._weights))
        particle_balance = (after.density_m3 - expected_density) / max(
            abs(expected_density), before.density_m3, 1.0
        )
        momentum_before = float(np.sum(self.mass_kg * parallel * state.f_e * self._weights))
        momentum_after = float(np.sum(self.mass_kg * parallel * updated.f_e * self._weights))
        momentum_source = float(dt_s * np.sum(self.mass_kg * parallel * source * self._weights))
        momentum_balance = (momentum_after - momentum_before - momentum_source) / max(abs(momentum_before), self.mass_kg * density if density > 0 else 1.0, 1.0)
        diagnostics = FokkerPlanckStepDiagnostics(
            before.density_m3,
            after.density_m3,
            particle_balance,
            float(np.min(updated.f_e)),
            clipped_negative_density,
            residual_relative,
            after.current_parallel_A,
            after.temperature_eV,
            rf.source,
            momentum_balance_relative=momentum_balance,
            field_particle_source_A=float(self.charge_C * np.sum(parallel * field_particle_source * self._weights) * self.cross_section_area_m2),
        )
        return updated, diagnostics

    def _check_state(self, state: KineticState2V) -> None:
        if (
            state.grid.shape != self.grid.shape
            or not np.array_equal(state.grid.v_parallel_m_s, self.grid.v_parallel_m_s)
            or not np.array_equal(state.grid.v_perp_m_s, self.grid.v_perp_m_s)
        ):
            raise ValueError("state uses a different velocity grid")

    def _coefficient_array(self, value: float | np.ndarray, name: str) -> np.ndarray:
        array = np.asarray(value, dtype=float)
        if array.ndim == 0:
            array = np.full(self.grid.shape, float(array))
        if array.shape != self.grid.shape or np.any(~np.isfinite(array)):
            raise ValueError(f"{name} must be finite and scalar or match the grid")
        return array

    def _assemble_operator(
        self,
        acceleration_parallel: np.ndarray,
        acceleration_perp: np.ndarray,
        D_parallel: np.ndarray,
        D_perp: np.ndarray,
        loss_frequency: np.ndarray,
        D_cross: np.ndarray | None = None,
    ) -> csr_matrix:
        n_parallel, n_perp = self.grid.shape
        dv_parallel = self.grid.dv_parallel_m_s
        dv_perp = self.grid.dv_perp_m_s
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []

        def flat(i: int, j: int) -> int:
            return i * n_perp + j

        def add(row: int, col: int, value: float) -> None:
            if value != 0.0:
                rows.append(row)
                cols.append(col)
                data.append(value)

        def add_interface(
            left: int,
            right: int,
            face_area: float,
            left_volume: float,
            right_volume: float,
            spacing: float,
            advection: float,
            diffusion: float,
        ) -> None:
            # Chang--Cooper exponential fitting preserves the zero-flux
            # Maxwellian of a drag/diffusion pair and retains the upwind
            # limit when diffusion vanishes.  ``delta`` weights the right
            # state in F = a[(1-delta)f_L+delta f_R]-D df/dv.
            if diffusion > 0.0:
                peclet = advection * spacing / diffusion
                if abs(peclet) < 1.0e-6:
                    delta = 0.5 - peclet / 12.0 + peclet**3 / 720.0
                elif peclet > 50.0:
                    delta = 1.0 / peclet
                elif peclet < -50.0:
                    delta = 1.0 + 1.0 / peclet
                else:
                    delta = 1.0 / peclet - 1.0 / np.expm1(peclet)
                left_flux = advection * (1.0 - delta) + diffusion / spacing
                right_flux = advection * delta - diffusion / spacing
            else:
                left_flux = max(advection, 0.0)
                right_flux = min(advection, 0.0)
            add(left, left, -face_area * left_flux / left_volume)
            add(left, right, -face_area * right_flux / left_volume)
            add(right, left, face_area * left_flux / right_volume)
            add(right, right, face_area * right_flux / right_volume)

        weights = self._weights
        for i in range(n_parallel - 1):
            for j in range(n_perp):
                face_area = 2.0 * PI * self.grid.v_perp_m_s[j] * dv_perp
                add_interface(
                    flat(i, j),
                    flat(i + 1, j),
                    face_area,
                    weights[i, j],
                    weights[i + 1, j],
                    dv_parallel,
                    0.5 * (acceleration_parallel[i, j] + acceleration_parallel[i + 1, j]),
                    0.5 * (D_parallel[i, j] + D_parallel[i + 1, j]),
                )

        for i in range(n_parallel):
            for j in range(n_perp - 1):
                edge_velocity = (j + 1) * dv_perp
                face_area = 2.0 * PI * edge_velocity * dv_parallel
                add_interface(
                    flat(i, j),
                    flat(i, j + 1),
                    face_area,
                    weights[i, j],
                    weights[i, j + 1],
                    dv_perp,
                    0.5 * (acceleration_perp[i, j] + acceleration_perp[i, j + 1]),
                    0.5 * (D_perp[i, j] + D_perp[i, j + 1]),
                )

        if D_cross is not None and np.any(D_cross != 0.0):
            # Conservative enough for the reduced tensor closure: a centred
            # mixed derivative on interior cells. Tensor PSD validation above
            # prevents a negative diffusion eigenvalue; boundary cells use the
            # nearest interior stencil and therefore remain finite-volume
            # compatible without inventing a cross-flux through v_perp=0.
            for i in range(1, n_parallel - 1):
                for j in range(1, n_perp - 1):
                    coeff = float(D_cross[i, j]) / (2.0 * dv_parallel * dv_perp)
                    row = flat(i, j)
                    add(row, flat(i + 1, j + 1), coeff)
                    add(row, flat(i + 1, j - 1), -coeff)
                    add(row, flat(i - 1, j + 1), -coeff)
                    add(row, flat(i - 1, j - 1), coeff)

        for i in range(n_parallel):
            for j in range(n_perp):
                index = flat(i, j)
                add(index, index, -float(loss_frequency[i, j]))
        size = n_parallel * n_perp
        return coo_matrix((data, (rows, cols)), shape=(size, size)).tocsr()


class RelativisticFokkerPlanck2V:
    """Reduced 0D-2V solver in dimensionless relativistic momentum ``u=p/(mc)``.

    This is intentionally a separate backend.  The finite-volume operator is
    assembled in ``u`` coordinates and moments use the exact Jacobian
    ``d^3v = 2*pi*u_perp*c^3 du_perp du_parallel/gamma^5``.  The collision
    closure is a linearized relativistic Ornstein--Uhlenbeck approximation;
    it is useful for startup sensitivity studies but is not a replacement for
    the nonlinear Rosenbluth/CQL3D collision operator.
    """

    def __init__(
        self,
        grid: RelativisticMomentumGrid,
        *,
        cross_section_area_m2: float,
        electron_mass_kg: float = M_E_KG,
        electron_charge_C: float = -E_CHARGE_C,
    ) -> None:
        if cross_section_area_m2 <= 0.0:
            raise ValueError("cross_section_area_m2 must be positive")
        self.grid = grid
        self.cross_section_area_m2 = float(cross_section_area_m2)
        self.mass_kg = float(electron_mass_kg)
        self.charge_C = float(electron_charge_C)
        self._weights = grid.phase_volume_weights_m3_s3
        self._linear_cache: OrderedDict[tuple, tuple[csr_matrix, SuperLU]] = OrderedDict()
        self._linear_cache_max_entries = 8

    def _array_signature(self, array: np.ndarray) -> bytes:
        return hashlib.blake2b(np.ascontiguousarray(array, dtype=np.float64).view(np.uint8), digest_size=12).digest()

    def maxwellian_state(
        self,
        density_m3: float,
        temperature_eV: float,
        *,
        drift_v_parallel_m_s: float = 0.0,
        time_s: float = 0.0,
    ) -> KineticState2V:
        if density_m3 < 0.0 or temperature_eV <= 0.0:
            raise ValueError("density must be non-negative and temperature positive")
        up, ut = self.grid.mesh_u()
        gamma = np.sqrt(1.0 + up**2 + ut**2)
        theta = temperature_eV * E_CHARGE_C / (self.mass_kg * C_LIGHT_M_S**2)
        if theta <= 0.0:
            raise ValueError("temperature_eV must be positive")
        # A small drift is represented by the first-order Juttner shift.  The
        # exact boosted distribution is not needed for the startup closure.
        drift_u = float(drift_v_parallel_m_s) / C_LIGHT_M_S
        exponent = -(gamma - 1.0) / theta + drift_u * up / max(theta, 1.0e-300)
        # Subtracting the resolved maximum prevents an underflowed cold
        # Juttner state when the first u_perp cell is many thermal widths out.
        exponent -= float(np.max(exponent))
        distribution = np.exp(np.maximum(exponent, -745.0))
        normalization = float(np.sum(distribution * self._weights))
        if normalization <= 0.0:
            raise ValueError("momentum domain does not resolve the requested Juttner distribution")
        distribution *= density_m3 / normalization
        return KineticState2V(distribution, self.grid, time_s)

    def moments(self, state: KineticState2V) -> KineticMoments:
        self._check_state(state)
        f = np.maximum(state.f_e, 0.0)
        vp, vt = self.grid.mesh()
        up, ut = self.grid.mesh_u()
        gamma = np.sqrt(1.0 + up**2 + ut**2)
        density = float(np.sum(f * self._weights))
        if density <= 0.0:
            return KineticMoments(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        p_moment = float(np.sum(vp * f * self._weights))
        mean_v = p_moment / density
        energy = float(np.sum((gamma - 1.0) * self.mass_kg * C_LIGHT_M_S**2 * f * self._weights))
        thermal = max(0.0, energy - 0.5 * self.mass_kg * density * mean_v**2)
        temperature = 2.0 * thermal / (3.0 * density * E_CHARGE_C)
        current_density = self.charge_C * p_moment
        pressure_parallel = float(self.mass_kg * np.sum((vp - mean_v) ** 2 * f * self._weights))
        pressure_perp = float(0.5 * self.mass_kg * np.sum(vt**2 * f * self._weights))
        return KineticMoments(
            density, mean_v, energy, temperature, current_density,
            current_density * self.cross_section_area_m2, pressure_parallel, pressure_perp,
        )

    def _check_state(self, state: KineticState2V) -> None:
        if not isinstance(state.grid, RelativisticMomentumGrid):
            raise ValueError("state must use a RelativisticMomentumGrid")
        if state.grid.shape != self.grid.shape or not np.array_equal(state.grid.u_parallel, self.grid.u_parallel) or not np.array_equal(state.grid.u_perp, self.grid.u_perp):
            raise ValueError("state uses a different relativistic momentum grid")

    def _coefficient_array(self, value: float | np.ndarray, name: str) -> np.ndarray:
        array = np.asarray(value, dtype=float)
        if array.ndim == 0:
            array = np.full(self.grid.shape, float(array))
        if array.shape != self.grid.shape or np.any(~np.isfinite(array)):
            raise ValueError(f"{name} must be finite and scalar or match the grid")
        return array

    def _assemble_operator(
        self,
        acceleration_parallel: np.ndarray,
        acceleration_perp: np.ndarray,
        D_parallel: np.ndarray,
        D_perp: np.ndarray,
        loss_frequency: np.ndarray,
        D_cross: np.ndarray,
    ) -> csr_matrix:
        np_, nt = self.grid.shape
        du_p, du_t = self.grid.du_parallel, self.grid.du_perp
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []

        def flat(i: int, j: int) -> int:
            return i * nt + j

        def add(row: int, col: int, value: float) -> None:
            if value:
                rows.append(row); cols.append(col); data.append(float(value))

        # Conservative fluxes in each coordinate.  The cylindrical Jacobian
        # is represented by cell weights; interface areas are their harmonic
        # mean, which gives zero net particle flux for a constant f.
        def face(left: int, right: int, spacing: float, drift: float, diffusion: float) -> None:
            if diffusion > 0.0:
                pe = drift * spacing / diffusion
                if abs(pe) < 1.0e-6:
                    delta = 0.5 - pe / 12.0
                elif pe > 50.0:
                    delta = 1.0 / pe
                elif pe < -50.0:
                    delta = 1.0 + 1.0 / pe
                else:
                    delta = 1.0 / pe - 1.0 / np.expm1(pe)
                a_l = drift * (1.0 - delta) + diffusion / spacing
                a_r = drift * delta - diffusion / spacing
            else:
                a_l, a_r = max(drift, 0.0), min(drift, 0.0)
            wl, wr = self._weights.flat[left], self._weights.flat[right]
            area = 2.0 * np.sqrt(max(wl * wr, 1.0e-300)) / spacing
            add(left, left, -area * a_l / wl); add(left, right, -area * a_r / wl)
            add(right, left, area * a_l / wr); add(right, right, area * a_r / wr)

        for i in range(np_ - 1):
            for j in range(nt):
                l, r = flat(i, j), flat(i + 1, j)
                face(l, r, du_p, 0.5 * (acceleration_parallel[i, j] + acceleration_parallel[i + 1, j]), 0.5 * (D_parallel[i, j] + D_parallel[i + 1, j]))
        for i in range(np_):
            for j in range(nt - 1):
                l, r = flat(i, j), flat(i, j + 1)
                face(l, r, du_t, 0.5 * (acceleration_perp[i, j] + acceleration_perp[i, j + 1]), 0.5 * (D_perp[i, j] + D_perp[i, j + 1]))
        if np.any(D_cross):
            for i in range(1, np_ - 1):
                for j in range(1, nt - 1):
                    c = float(D_cross[i, j]) / (2.0 * du_p * du_t)
                    row = flat(i, j)
                    add(row, flat(i + 1, j + 1), c); add(row, flat(i + 1, j - 1), -c)
                    add(row, flat(i - 1, j + 1), -c); add(row, flat(i - 1, j - 1), c)
        for i in range(np_):
            for j in range(nt):
                add(flat(i, j), flat(i, j), -float(loss_frequency[i, j]))
        size = np_ * nt
        return coo_matrix((data, (rows, cols)), shape=(size, size)).tocsr()

    def advance(
        self,
        state: KineticState2V,
        dt_s: float,
        *,
        E_parallel_V_m: float = 0.0,
        collision_frequency_s: float = 0.0,
        collision_temperature_eV: float | None = None,
        rf_diffusion: QuasilinearDiffusion2V | None = None,
        loss_frequency_s: float | np.ndarray = 0.0,
        source_f_s: np.ndarray | None = None,
        positivity_tolerance: float = 1.0e-12,
        momentum_conserving_field_particle: bool = False,
    ) -> tuple[KineticState2V, FokkerPlanckStepDiagnostics]:
        self._check_state(state)
        if dt_s <= 0.0 or collision_frequency_s < 0.0:
            raise ValueError("dt_s must be positive and collision frequency non-negative")
        if collision_temperature_eV is None:
            collision_temperature_eV = max(self.moments(state).temperature_eV, 1.0e-6)
        if collision_temperature_eV <= 0.0:
            raise ValueError("collision_temperature_eV must be positive")
        rf = (rf_diffusion or QuasilinearDiffusion2V.zeros(self.grid)).validated(self.grid)
        loss = self._coefficient_array(loss_frequency_s, "loss_frequency_s")
        source = np.zeros(self.grid.shape) if source_f_s is None else np.asarray(source_f_s, dtype=float)
        if source.shape != self.grid.shape or np.any(~np.isfinite(source)):
            raise ValueError("source_f_s must be finite and match the grid")
        f = np.maximum(state.f_e, 0.0)
        density = float(np.sum(f * self._weights))
        up, ut = self.grid.mesh_u()
        theta = collision_temperature_eV * E_CHARGE_C / (self.mass_kg * C_LIGHT_M_S**2)
        collision_diffusion = collision_frequency_s * max(theta, 1.0e-12)
        acceleration_parallel = self.charge_C * E_parallel_V_m / (self.mass_kg * C_LIGHT_M_S) - collision_frequency_s * up
        acceleration_perp = -collision_frequency_s * ut
        field_particle = np.zeros(self.grid.shape)
        if momentum_conserving_field_particle and collision_frequency_s > 0.0 and density > 0.0:
            vp, _ = self.grid.mesh()
            mean_u = float(np.sum(up * f * self._weights) / density)
            equilibrium = np.exp(np.maximum(-(np.sqrt(1.0 + up**2 + ut**2) - 1.0) / max(theta, 1.0e-12), -745.0))
            equilibrium *= density / max(float(np.sum(equilibrium * self._weights)), 1.0e-300)
            field_particle = collision_frequency_s * mean_u * up / max(theta, 1.0e-12) * equilibrium
            source = source + field_particle
        Dp = collision_diffusion + rf.D_parallel_m2_s3 / C_LIGHT_M_S**2
        Dt = collision_diffusion + rf.D_perp_m2_s3 / C_LIGHT_M_S**2
        Dx = rf.D_cross_m2_s3 / C_LIGHT_M_S**2
        key = (float(dt_s), self._array_signature(acceleration_parallel), self._array_signature(acceleration_perp), self._array_signature(Dp), self._array_signature(Dt), self._array_signature(Dx), self._array_signature(loss))
        cached = self._linear_cache.get(key)
        if cached is None:
            operator = self._assemble_operator(acceleration_parallel, acceleration_perp, Dp, Dt, loss, Dx)
            lhs = eye(operator.shape[0], format="csr") - dt_s * operator
            factor = splu(lhs.tocsc())
            self._linear_cache[key] = (lhs, factor)
            while len(self._linear_cache) > self._linear_cache_max_entries:
                self._linear_cache.popitem(last=False)
        else:
            lhs, factor = cached
        rhs = state.f_e.ravel() + dt_s * source.ravel()
        raw = np.asarray(factor.solve(rhs), dtype=float).reshape(self.grid.shape)
        if np.any(~np.isfinite(raw)):
            raise FloatingPointError("relativistic Fokker--Planck solve produced non-finite values")
        scale = max(float(np.max(np.abs(raw))), 1.0e-300)
        if float(np.min(raw)) < -positivity_tolerance * scale:
            raise FloatingPointError("relativistic Fokker--Planck step violated positivity")
        updated = KineticState2V(np.maximum(raw, 0.0), self.grid, state.time_s + dt_s)
        before, after = self.moments(state), self.moments(updated)
        expected = before.density_m3 + dt_s * float(np.sum(source * self._weights)) - dt_s * float(np.sum(loss * updated.f_e * self._weights))
        balance = (after.density_m3 - expected) / max(abs(expected), before.density_m3, 1.0)
        residual = lhs @ raw.ravel() - rhs
        momentum_before = float(np.sum(self.mass_kg * up * state.f_e * self._weights))
        momentum_after = float(np.sum(self.mass_kg * up * updated.f_e * self._weights))
        momentum_source = dt_s * float(np.sum(self.mass_kg * up * source * self._weights))
        mom_balance = (momentum_after - momentum_before - momentum_source) / max(abs(momentum_before), self.mass_kg * density, 1.0)
        return updated, FokkerPlanckStepDiagnostics(
            before.density_m3, after.density_m3, balance, float(np.min(updated.f_e)),
            float(-np.sum(np.minimum(raw, 0.0) * self._weights)),
            float(np.linalg.norm(residual) / max(np.linalg.norm(rhs), 1.0e-300)),
            after.current_parallel_A, after.temperature_eV, rf.source,
            momentum_balance_relative=mom_balance,
            field_particle_source_A=float(self.charge_C * np.sum(self.grid.mesh()[0] * field_particle * self._weights) * self.cross_section_area_m2),
        )
