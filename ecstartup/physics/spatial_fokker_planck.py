"""Open-field-line 1D-2V kinetic transport with spatial RF diffusion.

The solver uses operator splitting: the existing conservative 2V implicit
Fokker-Planck operator is applied independently in every field-line cell,
then parallel streaming is advanced with a positivity-preserving upwind
scheme and absorbing sheath entrances.  The spatial coordinate is physical
arc length ``s`` and may carry the traced ``R(s), Z(s), B(s)`` geometry.

This is a first open-field-line kinetic backend.  It is not a replacement for
a nonlinear CQL3D collision operator or a self-consistent kinetic sheath.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence, Union

import numpy as np

from ..constants import E_CHARGE_C, M_E_KG
from .fieldline import FieldLineTrace
from .fokker_planck import (
    C_LIGHT_M_S,
    KineticState2V,
    QuasilinearDiffusion2V,
    ReducedFokkerPlanck2V,
    RelativisticFokkerPlanck2V,
    RelativisticMomentumGrid,
    VelocityGrid,
)
from .magnetic import MagneticFieldProvider
from .rosenbluth import NonlinearLandauRosenbluth2V
from .relativistic_collisions import RelativisticBraamsKarney2V
from .wall_sheath import QuasineutralAmbipolarSheath1D, SpatialSheathSolution


KineticGrid2V = Union[VelocityGrid, RelativisticMomentumGrid]


@dataclass(frozen=True)
class OpenFieldLineGrid:
    s_m: np.ndarray
    R_m: np.ndarray
    Z_m: np.ndarray
    B_T: np.ndarray
    cross_section_area_m2: np.ndarray

    def __post_init__(self) -> None:
        arrays = [np.asarray(item, dtype=float) for item in (self.s_m, self.R_m, self.Z_m, self.B_T, self.cross_section_area_m2)]
        length = len(arrays[0])
        if length < 2 or any(item.ndim != 1 or len(item) != length for item in arrays):
            raise ValueError("open-field-line arrays must be equal-length one-dimensional arrays")
        if np.any(~np.isfinite(np.concatenate(arrays))):
            raise ValueError("open-field-line grid contains non-finite values")
        if np.any(np.diff(arrays[0]) <= 0.0):
            raise ValueError("s_m must be strictly increasing")
        if not np.allclose(np.diff(arrays[0]), np.diff(arrays[0])[0], rtol=1.0e-8, atol=1.0e-12):
            raise ValueError("the current streaming operator requires a uniform s grid")
        if np.any(arrays[3] <= 0.0) or np.any(arrays[4] <= 0.0):
            raise ValueError("B_T and cross_section_area_m2 must be positive")
        for name, array in zip(("s_m", "R_m", "Z_m", "B_T", "cross_section_area_m2"), arrays):
            object.__setattr__(self, name, array)

    @property
    def n_s(self) -> int:
        return len(self.s_m)

    @property
    def ds_m(self) -> float:
        return float(self.s_m[1] - self.s_m[0])

    @property
    def cell_volume_m3(self) -> np.ndarray:
        return self.cross_section_area_m2 * self.ds_m

    @classmethod
    def from_trace(
        cls,
        trace: FieldLineTrace,
        field: MagneticFieldProvider,
        *,
        n_s: int,
        cross_section_area_m2: float,
    ) -> "OpenFieldLineGrid":
        if n_s < 2 or cross_section_area_m2 <= 0.0:
            raise ValueError("n_s >= 2 and positive cross-section area are required")
        branches = list(trace.branches)
        if not branches or any(not branch.path_s_m for branch in branches):
            raise ValueError("field-line trace must retain branch paths")
        backward = next((branch for branch in branches if branch.direction < 0), None)
        forward = next((branch for branch in branches if branch.direction > 0), None)
        if backward is not None and forward is not None:
            R_path = np.asarray(tuple(reversed(backward.path_R_m)) + forward.path_R_m[1:], dtype=float)
            Z_path = np.asarray(tuple(reversed(backward.path_Z_m)) + forward.path_Z_m[1:], dtype=float)
        else:
            branch = branches[0]
            R_path = np.asarray(branch.path_R_m, dtype=float)
            Z_path = np.asarray(branch.path_Z_m, dtype=float)
        distance = np.zeros(len(R_path), dtype=float)
        if len(R_path) > 1:
            distance[1:] = np.cumsum(np.hypot(np.diff(R_path), np.diff(Z_path)))
        if distance[-1] <= 0.0:
            raise ValueError("field-line trace has zero poloidal path length")
        s = np.linspace(0.0, distance[-1], n_s)
        R = np.interp(s, distance, R_path)
        Z = np.interp(s, distance, Z_path)
        B = np.asarray(
            [field.evaluate(float(r), float(z), trace.Ip_A, trace.t_s).magnitude_T for r, z in zip(R, Z)],
            dtype=float,
        )
        area = np.full(n_s, float(cross_section_area_m2), dtype=float)
        return cls(s, R, Z, B, area)


@dataclass(frozen=True)
class SpatialQuasilinearDiffusion:
    D_parallel_m2_s3: np.ndarray
    D_perp_m2_s3: np.ndarray
    D_cross_m2_s3: np.ndarray
    time_s: float
    source: str
    target_absorbed_power_W: float | None = None

    def validated(self, spatial: OpenFieldLineGrid, velocity: KineticGrid2V) -> "SpatialQuasilinearDiffusion":
        expected = (spatial.n_s, *velocity.shape)
        parallel = np.asarray(self.D_parallel_m2_s3, dtype=float)
        perpendicular = np.asarray(self.D_perp_m2_s3, dtype=float)
        cross = np.asarray(self.D_cross_m2_s3, dtype=float)
        if parallel.shape != expected or perpendicular.shape != expected or cross.shape != expected:
            raise ValueError(f"spatial Dql arrays must have shape {expected}")
        if np.any(~np.isfinite(parallel)) or np.any(~np.isfinite(perpendicular)) or np.any(~np.isfinite(cross)):
            raise ValueError("spatial Dql contains non-finite values")
        if np.any(parallel < 0.0) or np.any(perpendicular < 0.0):
            raise ValueError("spatial Dql diagonal entries must be non-negative")
        determinant = parallel * perpendicular - cross**2
        if np.any(determinant < -1.0e-12 * np.maximum(parallel * perpendicular, 1.0)):
            raise ValueError("spatial Dql must be positive semidefinite in every cell")
        if self.target_absorbed_power_W is not None and self.target_absorbed_power_W < 0.0:
            raise ValueError("target absorbed power must be non-negative")
        return SpatialQuasilinearDiffusion(
            parallel, perpendicular, cross, float(self.time_s), str(self.source), self.target_absorbed_power_W
        )

    def scaled(self, factor: float, *, source_suffix: str = "") -> "SpatialQuasilinearDiffusion":
        if not math.isfinite(factor) or factor < 0.0:
            raise ValueError("Dql scale must be finite and non-negative")
        return SpatialQuasilinearDiffusion(
            factor * self.D_parallel_m2_s3,
            factor * self.D_perp_m2_s3,
            factor * self.D_cross_m2_s3,
            self.time_s,
            self.source + source_suffix,
            self.target_absorbed_power_W,
        )

    def cell(self, index: int) -> QuasilinearDiffusion2V:
        return QuasilinearDiffusion2V(
            self.D_parallel_m2_s3[index],
            self.D_perp_m2_s3[index],
            self.source,
            self.D_cross_m2_s3[index],
        )


@dataclass
class SpatialKineticState:
    f_e: np.ndarray
    neutral_density_m3: np.ndarray
    ion_density_m3: np.ndarray
    time_s: float

    def validated(self, spatial: OpenFieldLineGrid, velocity: KineticGrid2V) -> "SpatialKineticState":
        distribution = np.asarray(self.f_e, dtype=float)
        neutral = np.asarray(self.neutral_density_m3, dtype=float)
        ion = np.asarray(self.ion_density_m3, dtype=float)
        if distribution.shape != (spatial.n_s, *velocity.shape):
            raise ValueError("spatial distribution shape does not match s and velocity grids")
        if neutral.shape != (spatial.n_s,) or ion.shape != (spatial.n_s,):
            raise ValueError("neutral and ion density profiles must match the s grid")
        if np.any(~np.isfinite(distribution)) or np.any(~np.isfinite(neutral)) or np.any(~np.isfinite(ion)):
            raise ValueError("spatial kinetic state contains non-finite values")
        if np.any(distribution < 0.0) or np.any(neutral < 0.0) or np.any(ion < 0.0):
            raise ValueError("spatial kinetic state must be non-negative")
        self.f_e = distribution
        self.neutral_density_m3 = neutral
        self.ion_density_m3 = ion
        self.time_s = float(self.time_s)
        return self


@dataclass(frozen=True)
class SpatialKineticDiagnostics:
    particle_inventory_before: float
    particle_inventory_after: float
    wall_particle_loss: float
    ionization_particles: float
    absorbed_rf_power_W: float
    target_absorbed_rf_power_W: float | None
    rf_power_balance_relative: float | None
    rf_particle_balance_relative: float | None
    rf_momentum_input_N: float
    current_parallel_A: float
    minimum_f: float
    streaming_substeps: int
    dql_scale: float
    sheath_drop_left_V: float | None
    sheath_drop_right_V: float | None
    sheath_current_residual_max_A_m2: float | None
    sheath_quasineutrality_relative: float | None
    rosenbluth_particle_residual_max: float | None
    rosenbluth_momentum_residual_max: float | None
    rosenbluth_energy_residual_max: float | None

    def to_dict(self) -> dict[str, float | int | None]:
        return self.__dict__.copy()


class OpenFieldLineKinetic1D2V:
    """Split finite-volume solver for ``f(s,u_parallel,u_perp,t)``."""

    def __init__(
        self,
        spatial: OpenFieldLineGrid,
        velocity: KineticGrid2V,
        *,
        streaming_scheme: str = "explicit_upwind",
    ) -> None:
        if streaming_scheme not in {"explicit_upwind", "implicit_upwind"}:
            raise ValueError("streaming_scheme must be explicit_upwind or implicit_upwind")
        self.spatial = spatial
        self.velocity = velocity
        self.streaming_scheme = streaming_scheme
        self._weights = velocity.phase_volume_weights_m3_s3
        # Local current density is converted to current with each flux-tube area.
        solver_type = (
            RelativisticFokkerPlanck2V
            if isinstance(velocity, RelativisticMomentumGrid)
            else ReducedFokkerPlanck2V
        )
        self._local_solvers = [
            solver_type(velocity, cross_section_area_m2=float(area))
            for area in spatial.cross_section_area_m2
        ]

    def maxwellian_state(
        self,
        density_m3: float | Sequence[float],
        temperature_eV: float,
        *,
        neutral_density_m3: float | Sequence[float] = 0.0,
        ion_density_m3: float | Sequence[float] = 0.0,
        drift_v_parallel_m_s: float = 0.0,
        time_s: float = 0.0,
    ) -> SpatialKineticState:
        density = self._profile(density_m3, "density_m3")
        neutral = self._profile(neutral_density_m3, "neutral_density_m3")
        ion = self._profile(ion_density_m3, "ion_density_m3")
        distributions = [
            solver.maxwellian_state(
                float(value), temperature_eV,
                drift_v_parallel_m_s=drift_v_parallel_m_s,
                time_s=time_s,
            ).f_e
            for solver, value in zip(self._local_solvers, density)
        ]
        return SpatialKineticState(np.stack(distributions), neutral, ion, time_s).validated(self.spatial, self.velocity)

    def _profile(self, value: float | Sequence[float], name: str) -> np.ndarray:
        profile = np.asarray(value, dtype=float)
        if profile.ndim == 0:
            profile = np.full(self.spatial.n_s, float(profile))
        if profile.shape != (self.spatial.n_s,) or np.any(~np.isfinite(profile)) or np.any(profile < 0.0):
            raise ValueError(f"{name} must be non-negative and match the s grid")
        return profile

    def _inventory(self, distribution: np.ndarray) -> float:
        density = np.sum(distribution * self._weights[None, :, :], axis=(1, 2))
        return float(np.sum(density * self.spatial.cell_volume_m3))

    def _energy_inventory_J(self, distribution: np.ndarray) -> float:
        if isinstance(self.velocity, RelativisticMomentumGrid):
            u_parallel, u_perp = self.velocity.mesh_u()
            gamma = np.sqrt(1.0 + u_parallel**2 + u_perp**2)
            energy = (gamma - 1.0) * M_E_KG * C_LIGHT_M_S**2
        else:
            v_parallel, v_perp = self.velocity.mesh()
            energy = 0.5 * M_E_KG * (v_parallel**2 + v_perp**2)
        density = np.sum(distribution * self._weights[None, :, :] * energy[None, :, :], axis=(1, 2))
        return float(np.sum(density * self.spatial.cell_volume_m3))

    def _parallel_momentum_kg_m_s(self, distribution: np.ndarray) -> float:
        if isinstance(self.velocity, RelativisticMomentumGrid):
            u_parallel, _ = self.velocity.mesh_u()
            momentum_parallel = M_E_KG * C_LIGHT_M_S * u_parallel
        else:
            v_parallel, _ = self.velocity.mesh()
            momentum_parallel = M_E_KG * v_parallel
        density = M_E_KG * np.sum(
            distribution
            * self._weights[None, :, :]
            * (momentum_parallel / M_E_KG)[None, :, :],
            axis=(1, 2),
        )
        return float(np.sum(density * self.spatial.cell_volume_m3))

    def _rf_response_balances(
        self,
        state: SpatialKineticState,
        diffusion: SpatialQuasilinearDiffusion,
        dt_s: float,
    ) -> tuple[float, float, float]:
        particles_before = self._inventory(state.f_e)
        energy_before = self._energy_inventory_J(state.f_e)
        momentum_before = self._parallel_momentum_kg_m_s(state.f_e)
        after = []
        for index, solver in enumerate(self._local_solvers):
            local, _ = solver.advance(
                KineticState2V(state.f_e[index], self.velocity, state.time_s),
                dt_s,
                collision_frequency_s=0.0,
                rf_diffusion=diffusion.cell(index),
            )
            after.append(local.f_e)
        distribution = np.stack(after)
        particles_after = self._inventory(distribution)
        particle_error = abs(particles_after - particles_before) / max(abs(particles_before), 1.0)
        return (
            (self._energy_inventory_J(distribution) - energy_before) / dt_s,
            (self._parallel_momentum_kg_m_s(distribution) - momentum_before) / dt_s,
            particle_error,
        )

    def _rf_response(
        self,
        state: SpatialKineticState,
        diffusion: SpatialQuasilinearDiffusion,
        dt_s: float,
    ) -> tuple[float, float]:
        power, momentum, _ = self._rf_response_balances(state, diffusion, dt_s)
        return power, momentum

    def normalize_dql_to_power(
        self,
        state: SpatialKineticState,
        diffusion: SpatialQuasilinearDiffusion,
        dt_s: float,
        target_power_W: float,
        *,
        tolerance: float = 5.0e-4,
        maximum_iterations: int = 10,
    ) -> tuple[SpatialQuasilinearDiffusion, float, float, float]:
        if target_power_W < 0.0 or dt_s <= 0.0:
            raise ValueError("target power must be non-negative and dt positive")
        if target_power_W == 0.0:
            zero = diffusion.scaled(0.0, source_suffix=";power_normalized")
            return zero, 0.0, 0.0, 0.0
        scale = 1.0
        power = 0.0
        momentum = 0.0
        for _ in range(maximum_iterations):
            trial = diffusion.scaled(scale)
            power, momentum = self._rf_response(state, trial, dt_s)
            if power <= 0.0 or not math.isfinite(power):
                raise ValueError("Dql does not produce positive absorbed power for the current distribution")
            relative = abs(power - target_power_W) / target_power_W
            if relative <= tolerance:
                break
            scale *= target_power_W / power
        normalized = diffusion.scaled(scale, source_suffix=";power_normalized")
        normalized = SpatialQuasilinearDiffusion(
            normalized.D_parallel_m2_s3,
            normalized.D_perp_m2_s3,
            normalized.D_cross_m2_s3,
            normalized.time_s,
            normalized.source,
            target_power_W,
        )
        power, momentum = self._rf_response(state, normalized, dt_s)
        return normalized, scale, power, momentum

    def _parallel_kinetic_energy_eV(self) -> np.ndarray:
        if isinstance(self.velocity, RelativisticMomentumGrid):
            u_parallel, _ = self.velocity.mesh_u()
            return (np.sqrt(1.0 + u_parallel**2) - 1.0) * M_E_KG * C_LIGHT_M_S**2 / E_CHARGE_C
        v_parallel, _ = self.velocity.mesh()
        return 0.5 * M_E_KG * v_parallel**2 / E_CHARGE_C

    def _stream_absorbing(
        self,
        distribution: np.ndarray,
        dt_s: float,
        *,
        sheath_drop_left_V: float = 0.0,
        sheath_drop_right_V: float = 0.0,
    ) -> tuple[np.ndarray, float, int]:
        v_parallel, _ = self.velocity.mesh()
        maximum_courant = float(np.max(np.abs(v_parallel)) * dt_s / self.spatial.ds_m)
        if self.streaming_scheme == "implicit_upwind":
            result = distribution.copy()
            inventory_before = self._inventory(result)
            parallel_energy_eV = self._parallel_kinetic_energy_eV()
            n_parallel, n_perp = self.velocity.shape
            for parallel_index in range(n_parallel):
                speed = float(v_parallel[parallel_index, 0])
                if speed == 0.0:
                    continue
                mirror = n_parallel - 1 - parallel_index
                courant_number = abs(speed) * dt_s / self.spatial.ds_m
                denominator = 1.0 + courant_number
                for perpendicular_index in range(n_perp):
                    if speed > 0.0:
                        inflow = 0.0
                        if parallel_energy_eV[mirror, perpendicular_index] < sheath_drop_left_V:
                            inflow = distribution[0, mirror, perpendicular_index]
                        result[0, parallel_index, perpendicular_index] = (
                            distribution[0, parallel_index, perpendicular_index] + courant_number * inflow
                        ) / denominator
                        for spatial_index in range(1, self.spatial.n_s):
                            result[spatial_index, parallel_index, perpendicular_index] = (
                                distribution[spatial_index, parallel_index, perpendicular_index]
                                + courant_number * result[spatial_index - 1, parallel_index, perpendicular_index]
                            ) / denominator
                    else:
                        inflow = 0.0
                        if parallel_energy_eV[mirror, perpendicular_index] < sheath_drop_right_V:
                            inflow = distribution[-1, mirror, perpendicular_index]
                        result[-1, parallel_index, perpendicular_index] = (
                            distribution[-1, parallel_index, perpendicular_index] + courant_number * inflow
                        ) / denominator
                        for spatial_index in range(self.spatial.n_s - 2, -1, -1):
                            result[spatial_index, parallel_index, perpendicular_index] = (
                                distribution[spatial_index, parallel_index, perpendicular_index]
                                + courant_number * result[spatial_index + 1, parallel_index, perpendicular_index]
                            ) / denominator
            result = np.maximum(result, 0.0)
            loss = max(0.0, inventory_before - self._inventory(result))
            return result, loss, 1
        substeps = max(1, int(math.ceil(maximum_courant)))
        sub_dt = dt_s / substeps
        result = distribution.copy()
        inventory_before = self._inventory(result)
        positive = v_parallel > 0.0
        negative = v_parallel < 0.0
        parallel_energy_eV = self._parallel_kinetic_energy_eV()
        n_parallel = self.velocity.shape[0]
        for _ in range(substeps):
            old = result
            new = old.copy()
            courant_grid = v_parallel * sub_dt / self.spatial.ds_m
            left = np.zeros_like(old[0])
            right = np.zeros_like(old[-1])
            # Specularly reflect sub-barrier electrons. Super-barrier
            # electrons see a zero-inflow absorbing wall.
            for parallel_index in range(n_parallel):
                mirror = n_parallel - 1 - parallel_index
                if np.all(positive[parallel_index]) and np.all(
                    parallel_energy_eV[mirror] < sheath_drop_left_V
                ):
                    left[parallel_index] = old[0, mirror]
                if np.all(negative[parallel_index]) and np.all(
                    parallel_energy_eV[mirror] < sheath_drop_right_V
                ):
                    right[parallel_index] = old[-1, mirror]
            for index in range(self.spatial.n_s):
                upwind_left = left if index == 0 else old[index - 1]
                upwind_right = right if index == self.spatial.n_s - 1 else old[index + 1]
                new[index][positive] = old[index][positive] - courant_grid[positive] * (
                    old[index][positive] - upwind_left[positive]
                )
                new[index][negative] = old[index][negative] - courant_grid[negative] * (
                    upwind_right[negative] - old[index][negative]
                )
            result = np.maximum(new, 0.0)
        loss = max(0.0, inventory_before - self._inventory(result))
        return result, loss, substeps

    def advance(
        self,
        state: SpatialKineticState,
        dt_s: float,
        *,
        E_parallel_V_m: float | Sequence[float] = 0.0,
        collision_frequency_s: float | Sequence[float] = 0.0,
        collision_temperature_eV: float | Sequence[float] = 1.0,
        loss_frequency_s: float | Sequence[float] = 0.0,
        ionization_rate_coefficient_m3_s: float | Sequence[float] = 0.0,
        ionization_birth_temperature_eV: float = 1.0,
        rf_diffusion: SpatialQuasilinearDiffusion | None = None,
        target_absorbed_power_W: float | None = None,
        momentum_conserving_field_particle: bool = False,
        rf_particle_balance_tolerance: float = 1.0e-8,
        sheath_closure: QuasineutralAmbipolarSheath1D | None = None,
        ion_temperature_eV: float | Sequence[float] = 0.1,
        rosenbluth_operator: NonlinearLandauRosenbluth2V | None = None,
        braams_karney_operator: RelativisticBraamsKarney2V | None = None,
        electron_electron_collision_frequency_s: float | Sequence[float] = 0.0,
        electron_ion_collision_frequency_s: float | Sequence[float] = 0.0,
    ) -> tuple[SpatialKineticState, SpatialKineticDiagnostics]:
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        state = state.validated(self.spatial, self.velocity)
        electric = np.asarray(E_parallel_V_m, dtype=float)
        if electric.ndim == 0:
            electric = np.full(self.spatial.n_s, float(electric))
        collision = self._profile(collision_frequency_s, "collision_frequency_s")
        temperature = self._profile(collision_temperature_eV, "collision_temperature_eV")
        loss = self._profile(loss_frequency_s, "loss_frequency_s")
        ionization = self._profile(ionization_rate_coefficient_m3_s, "ionization_rate_coefficient_m3_s")
        ee_collision = self._profile(
            electron_electron_collision_frequency_s,
            "electron_electron_collision_frequency_s",
        )
        ei_collision = self._profile(
            electron_ion_collision_frequency_s,
            "electron_ion_collision_frequency_s",
        )
        if electric.shape != (self.spatial.n_s,) or np.any(~np.isfinite(electric)):
            raise ValueError("E_parallel_V_m must be finite and match the s grid")
        if rosenbluth_operator is not None and np.any(collision > 0.0):
            raise ValueError(
                "collision_frequency_s and rosenbluth_operator cannot be enabled together"
            )
        if braams_karney_operator is not None and np.any(collision > 0.0):
            raise ValueError(
                "collision_frequency_s and braams_karney_operator cannot be enabled together"
            )
        if rosenbluth_operator is not None and braams_karney_operator is not None:
            raise ValueError("select either classical Rosenbluth or relativistic Braams-Karney")
        if braams_karney_operator is None and (np.any(ee_collision > 0.0) or np.any(ei_collision > 0.0)):
            raise ValueError("relativistic collision frequencies require braams_karney_operator")
        sheath: SpatialSheathSolution | None = None
        if sheath_closure is not None:
            local_moments = [
                solver.moments(KineticState2V(state.f_e[index], self.velocity, state.time_s))
                for index, solver in enumerate(self._local_solvers)
            ]
            sheath = sheath_closure.solve(
                self.spatial.s_m,
                [item.density_m3 for item in local_moments],
                state.ion_density_m3,
                [max(item.temperature_eV, 1.0e-6) for item in local_moments],
                ion_temperature_eV=ion_temperature_eV,
            )
            electric = electric + sheath.E_parallel_V_m
        diffusion = rf_diffusion
        dql_scale = 1.0
        measured_power = 0.0
        rf_momentum = 0.0
        rf_particle_error: float | None = None
        if diffusion is not None:
            diffusion = diffusion.validated(self.spatial, self.velocity)
            requested_power = target_absorbed_power_W
            if requested_power is None:
                requested_power = diffusion.target_absorbed_power_W
            if requested_power is not None:
                diffusion, dql_scale, measured_power, rf_momentum = self.normalize_dql_to_power(
                    state, diffusion, dt_s, requested_power
                )
                measured_power, rf_momentum, rf_particle_error = self._rf_response_balances(
                    state, diffusion, dt_s
                )
            else:
                measured_power, rf_momentum, rf_particle_error = self._rf_response_balances(
                    state, diffusion, dt_s
                )
            if rf_particle_error > rf_particle_balance_tolerance:
                raise ValueError(
                    "RF diffusion violates particle conservation: "
                    f"relative error {rf_particle_error:.3e} exceeds "
                    f"{rf_particle_balance_tolerance:.3e}"
                )
        else:
            requested_power = target_absorbed_power_W
            if requested_power not in (None, 0.0):
                raise ValueError("target absorbed power requires a spatial Dql")

        inventory_before = self._inventory(state.f_e)
        distributions = []
        ionization_particles = 0.0
        rosenbluth_particle_residuals: list[float] = []
        rosenbluth_momentum_residuals: list[float] = []
        rosenbluth_energy_residuals: list[float] = []
        neutral_after = state.neutral_density_m3.copy()
        ion_after = state.ion_density_m3.copy()
        for index, solver in enumerate(self._local_solvers):
            local_state = KineticState2V(state.f_e[index], self.velocity, state.time_s)
            local_moments = solver.moments(local_state)
            birth_rate = max(
                0.0,
                ionization[index] * local_moments.density_m3 * state.neutral_density_m3[index],
            )
            available_rate = state.neutral_density_m3[index] / dt_s
            birth_rate = min(birth_rate, available_rate)
            source = None
            if birth_rate > 0.0:
                unit = solver.maxwellian_state(1.0, ionization_birth_temperature_eV).f_e
                source = birth_rate * unit
                created = birth_rate * dt_s
                neutral_after[index] -= created
                ion_after[index] += created
                ionization_particles += created * self.spatial.cell_volume_m3[index]
            local, _ = solver.advance(
                local_state,
                dt_s,
                E_parallel_V_m=float(electric[index]),
                collision_frequency_s=float(collision[index]),
                collision_temperature_eV=float(temperature[index]),
                rf_diffusion=None if diffusion is None else diffusion.cell(index),
                loss_frequency_s=float(loss[index]),
                source_f_s=source,
                momentum_conserving_field_particle=momentum_conserving_field_particle,
            )
            if rosenbluth_operator is not None:
                local, rosenbluth_diagnostics = rosenbluth_operator.advance(local, dt_s)
                rosenbluth_particle_residuals.append(
                    abs(rosenbluth_diagnostics.particle_moment_after_projection)
                )
                rosenbluth_momentum_residuals.append(
                    abs(rosenbluth_diagnostics.momentum_moment_after_projection)
                )
                rosenbluth_energy_residuals.append(
                    abs(rosenbluth_diagnostics.energy_moment_after_projection)
                )
            if braams_karney_operator is not None:
                local, relativistic_diagnostics = braams_karney_operator.advance(
                    local,
                    dt_s,
                    electron_electron_frequency_s=float(ee_collision[index]),
                    electron_ion_frequency_s=float(ei_collision[index]),
                )
                rosenbluth_particle_residuals.extend(
                    (
                        abs(relativistic_diagnostics.particle_residual_ee),
                        abs(relativistic_diagnostics.particle_residual_ei),
                    )
                )
                rosenbluth_momentum_residuals.append(
                    abs(relativistic_diagnostics.momentum_residual_ee)
                )
                rosenbluth_energy_residuals.extend(
                    (
                        abs(relativistic_diagnostics.energy_residual_ee),
                        abs(relativistic_diagnostics.energy_residual_ei),
                    )
                )
            distributions.append(local.f_e)
        streamed, wall_loss, substeps = self._stream_absorbing(
            np.stack(distributions),
            dt_s,
            sheath_drop_left_V=0.0 if sheath is None else sheath.sheath_drop_left_V,
            sheath_drop_right_V=0.0 if sheath is None else sheath.sheath_drop_right_V,
        )
        updated = SpatialKineticState(
            streamed,
            np.maximum(neutral_after, 0.0),
            np.maximum(ion_after, 0.0),
            state.time_s + dt_s,
        ).validated(self.spatial, self.velocity)
        local_currents = [
            solver.moments(KineticState2V(updated.f_e[index], self.velocity, updated.time_s)).current_parallel_A
            for index, solver in enumerate(self._local_solvers)
        ]
        current = float(np.average(local_currents, weights=self.spatial.ds_m * np.ones(self.spatial.n_s)))
        relative = None
        if requested_power is not None:
            relative = abs(measured_power - requested_power) / max(requested_power, 1.0e-30)
        diagnostics = SpatialKineticDiagnostics(
            particle_inventory_before=inventory_before,
            particle_inventory_after=self._inventory(updated.f_e),
            wall_particle_loss=wall_loss,
            ionization_particles=ionization_particles,
            absorbed_rf_power_W=measured_power,
            target_absorbed_rf_power_W=requested_power,
            rf_power_balance_relative=relative,
            rf_particle_balance_relative=rf_particle_error,
            rf_momentum_input_N=rf_momentum,
            current_parallel_A=current,
            minimum_f=float(np.min(updated.f_e)),
            streaming_substeps=substeps,
            dql_scale=dql_scale,
            sheath_drop_left_V=None if sheath is None else sheath.sheath_drop_left_V,
            sheath_drop_right_V=None if sheath is None else sheath.sheath_drop_right_V,
            sheath_current_residual_max_A_m2=(
                None
                if sheath is None
                else max(abs(sheath.current_residual_left_A_m2), abs(sheath.current_residual_right_A_m2))
            ),
            sheath_quasineutrality_relative=None if sheath is None else sheath.quasineutrality_relative,
            rosenbluth_particle_residual_max=(
                None if not rosenbluth_particle_residuals else max(rosenbluth_particle_residuals)
            ),
            rosenbluth_momentum_residual_max=(
                None if not rosenbluth_momentum_residuals else max(rosenbluth_momentum_residuals)
            ),
            rosenbluth_energy_residual_max=(
                None if not rosenbluth_energy_residuals else max(rosenbluth_energy_residuals)
            ),
        )
        return updated, diagnostics


__all__ = [
    "OpenFieldLineGrid",
    "SpatialQuasilinearDiffusion",
    "SpatialKineticState",
    "SpatialKineticDiagnostics",
    "OpenFieldLineKinetic1D2V",
]
