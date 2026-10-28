"""Nonlinear nonrelativistic gyrotropic Landau-Rosenbluth collisions.

The operator evaluates the nonlinear Landau integral in axisymmetric 2V
velocity space.  This is mathematically equivalent to the classical
Rosenbluth-potential form; ``potentials()`` also evaluates the H/G convolution
potentials for diagnostics.  A discrete invariant projection removes the
remaining particle, parallel-momentum and kinetic-energy quadrature errors.

This module is deliberately nonrelativistic.  A relativistic momentum-grid
collision operator requires the Braams-Karney relativistic potentials and is
not obtained by substituting ``u=p/(mc)`` into this implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ..constants import E_CHARGE_C, EPS0_F_M, M_E_KG, PI
from .fokker_planck import KineticState2V, VelocityGrid


@dataclass(frozen=True)
class RosenbluthPotentials2V:
    H_s2_m2: np.ndarray
    G_m4_s4: np.ndarray


@dataclass(frozen=True)
class RosenbluthDiagnostics:
    particle_moment_before_projection: float
    momentum_moment_before_projection: float
    energy_moment_before_projection: float
    particle_moment_after_projection: float
    momentum_moment_after_projection: float
    energy_moment_after_projection: float
    maxwellian_flux_norm: float
    substeps: int = 1
    minimum_f: float | None = None

    def to_dict(self) -> dict[str, float | int | None]:
        return self.__dict__.copy()


class NonlinearLandauRosenbluth2V:
    """Direct O(Nv^2 Nphi) reference collision operator.

    The direct quadrature is intentionally a correctness backend for modest
    grids.  Production grids should replace the convolution with a verified
    multigrid/spectral Rosenbluth-potential solve while preserving this API
    and its invariant tests.
    """

    def __init__(
        self,
        grid: VelocityGrid,
        *,
        coulomb_logarithm: float = 15.0,
        gyroangle_points: int = 16,
        kernel_softening_fraction: float = 0.5,
    ) -> None:
        if not isinstance(grid, VelocityGrid):
            raise TypeError("classical Rosenbluth operator requires a VelocityGrid")
        if coulomb_logarithm <= 0.0 or gyroangle_points < 4 or kernel_softening_fraction <= 0.0:
            raise ValueError("positive Coulomb logarithm/softening and at least four gyroangles are required")
        self.grid = grid
        self.coulomb_logarithm = float(coulomb_logarithm)
        self.gyroangle_points = int(gyroangle_points)
        self.softening_m_s = kernel_softening_fraction * min(
            grid.dv_parallel_m_s, grid.dv_perp_m_s
        )
        self.gamma_m6_s4 = (
            E_CHARGE_C**4
            * self.coulomb_logarithm
            / (8.0 * PI * EPS0_F_M**2 * M_E_KG**2)
        )
        self._weights = grid.phase_volume_weights_m3_s3
        self._v_parallel, self._v_perp = grid.mesh()
        self._phi = 2.0 * PI * (np.arange(self.gyroangle_points) + 0.5) / self.gyroangle_points

    def _check(self, state: KineticState2V) -> np.ndarray:
        if not isinstance(state.grid, VelocityGrid):
            raise ValueError("Rosenbluth state must use a VelocityGrid")
        if state.grid.shape != self.grid.shape or not np.array_equal(
            state.grid.v_parallel_m_s, self.grid.v_parallel_m_s
        ) or not np.array_equal(state.grid.v_perp_m_s, self.grid.v_perp_m_s):
            raise ValueError("Rosenbluth state uses a different velocity grid")
        f = np.asarray(state.f_e, dtype=float)
        if np.any(~np.isfinite(f)) or np.any(f < 0.0):
            raise ValueError("Rosenbluth distribution must be finite and non-negative")
        return f

    def potentials(self, state: KineticState2V) -> RosenbluthPotentials2V:
        f = self._check(state)
        H = np.zeros(self.grid.shape)
        G = np.zeros(self.grid.shape)
        soft2 = self.softening_m_s**2
        for index in np.ndindex(self.grid.shape):
            vz = float(self._v_parallel[index])
            vr = float(self._v_perp[index])
            h_value = 0.0
            g_value = 0.0
            for phi in self._phi:
                ux = vr - self._v_perp * math.cos(phi)
                uy = -self._v_perp * math.sin(phi)
                uz = vz - self._v_parallel
                distance = np.sqrt(ux**2 + uy**2 + uz**2 + soft2)
                gyro_weight = self._weights / self.gyroangle_points
                h_value += float(np.sum(f * gyro_weight / distance))
                g_value += float(np.sum(f * gyro_weight * distance))
            H[index] = h_value
            G[index] = g_value
        return RosenbluthPotentials2V(H, G)

    def _log_gradients(self, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        positive = f[f > 0.0]
        floor = max(float(np.min(positive)) * 1.0e-12, 1.0e-300) if positive.size else 1.0e-300
        log_f = np.log(np.maximum(f, floor))
        d_parallel = np.gradient(
            log_f, self.grid.dv_parallel_m_s, axis=0, edge_order=2
        )
        d_perp = np.gradient(
            log_f, self.grid.dv_perp_m_s, axis=1, edge_order=2
        )
        return d_parallel, d_perp

    def _raw_operator(self, f: np.ndarray) -> tuple[np.ndarray, float]:
        grad_parallel, grad_perp = self._log_gradients(f)
        flux_parallel = np.zeros_like(f)
        flux_perp = np.zeros_like(f)
        soft2 = self.softening_m_s**2
        accumulated_flux_norm = 0.0
        identity = np.eye(3)

        for index in np.ndindex(self.grid.shape):
            vz = float(self._v_parallel[index])
            vr = float(self._v_perp[index])
            f_here = float(f[index])
            grad_here = np.array([grad_perp[index], 0.0, grad_parallel[index]])
            flux = np.zeros(3)
            for phi in self._phi:
                cos_phi = math.cos(phi)
                sin_phi = math.sin(phi)
                source_x = self._v_perp * cos_phi
                source_y = self._v_perp * sin_phi
                ux = vr - source_x
                uy = -source_y
                uz = vz - self._v_parallel
                distance2 = ux**2 + uy**2 + uz**2 + soft2
                distance3 = distance2 ** 1.5
                source_grad = np.stack(
                    (
                        grad_perp * cos_phi,
                        grad_perp * sin_phi,
                        grad_parallel,
                    ),
                    axis=-1,
                )
                for source_index in np.ndindex(self.grid.shape):
                    u = np.array(
                        [ux[source_index], uy[source_index], uz[source_index]],
                        dtype=float,
                    )
                    tensor = (float(np.dot(u, u)) * identity - np.outer(u, u)) / float(
                        distance3[source_index]
                    )
                    difference = grad_here - source_grad[source_index]
                    weight = self._weights[source_index] / self.gyroangle_points
                    flux += (
                        self.gamma_m6_s4
                        * f_here
                        * f[source_index]
                        * weight
                        * (tensor @ difference)
                    )
            flux_perp[index] = flux[0]
            flux_parallel[index] = flux[2]
            accumulated_flux_norm += float(np.dot(flux, flux))

        parallel_divergence = np.gradient(
            flux_parallel, self.grid.dv_parallel_m_s, axis=0, edge_order=2
        )
        radial_flux = self._v_perp * flux_perp
        perpendicular_divergence = np.gradient(
            radial_flux, self.grid.dv_perp_m_s, axis=1, edge_order=2
        ) / self._v_perp
        return parallel_divergence + perpendicular_divergence, math.sqrt(accumulated_flux_norm)

    def _invariant_moments(self, operator: np.ndarray) -> np.ndarray:
        kinetic = 0.5 * (self._v_parallel**2 + self._v_perp**2)
        return np.asarray(
            [
                np.sum(operator * self._weights),
                np.sum(self._v_parallel * operator * self._weights),
                np.sum(kinetic * operator * self._weights),
            ],
            dtype=float,
        )

    def _project_invariants(self, operator: np.ndarray) -> np.ndarray:
        kinetic = 0.5 * (self._v_parallel**2 + self._v_perp**2)
        basis = np.stack((np.ones(self.grid.shape), self._v_parallel, kinetic))
        gram = np.empty((3, 3))
        for i in range(3):
            for j in range(3):
                gram[i, j] = np.sum(basis[i] * basis[j] * self._weights)
        coefficients = np.linalg.solve(gram, self._invariant_moments(operator))
        corrected = operator - np.tensordot(coefficients, basis, axes=(0, 0))
        # One second solve removes roundoff amplified by the velocity scales.
        coefficients = np.linalg.solve(gram, self._invariant_moments(corrected))
        return corrected - np.tensordot(coefficients, basis, axes=(0, 0))

    def evaluate(self, state: KineticState2V) -> tuple[np.ndarray, RosenbluthDiagnostics]:
        f = self._check(state)
        raw, flux_norm = self._raw_operator(f)
        before = self._invariant_moments(raw)
        corrected = self._project_invariants(raw)
        after = self._invariant_moments(corrected)
        return corrected, RosenbluthDiagnostics(
            float(before[0]),
            float(before[1]),
            float(before[2]),
            float(after[0]),
            float(after[1]),
            float(after[2]),
            flux_norm,
        )

    def advance(
        self,
        state: KineticState2V,
        dt_s: float,
        *,
        positivity_safety: float = 0.8,
        maximum_substeps: int = 1000,
    ) -> tuple[KineticState2V, RosenbluthDiagnostics]:
        if dt_s <= 0.0 or not 0.0 < positivity_safety < 1.0:
            raise ValueError("dt_s must be positive and positivity_safety in (0,1)")
        remaining = float(dt_s)
        current = KineticState2V(self._check(state).copy(), self.grid, state.time_s)
        substeps = 0
        last_diagnostics = None
        while remaining > 0.0:
            operator, diagnostics = self.evaluate(current)
            negative = operator < 0.0
            if np.any(negative):
                limit = float(np.min(-current.f_e[negative] / operator[negative]))
                step = min(remaining, positivity_safety * max(limit, 0.0))
            else:
                step = remaining
            if step <= np.finfo(float).eps * max(dt_s, 1.0):
                raise FloatingPointError("Rosenbluth positivity time step collapsed")
            current = KineticState2V(
                np.maximum(current.f_e + step * operator, 0.0),
                self.grid,
                current.time_s + step,
            )
            remaining -= step
            substeps += 1
            last_diagnostics = diagnostics
            if substeps > maximum_substeps:
                raise RuntimeError("Rosenbluth collision advance exceeded maximum_substeps")
        assert last_diagnostics is not None
        return current, RosenbluthDiagnostics(
            last_diagnostics.particle_moment_before_projection,
            last_diagnostics.momentum_moment_before_projection,
            last_diagnostics.energy_moment_before_projection,
            last_diagnostics.particle_moment_after_projection,
            last_diagnostics.momentum_moment_after_projection,
            last_diagnostics.energy_moment_after_projection,
            last_diagnostics.maxwellian_flux_norm,
            substeps=substeps,
            minimum_f=float(np.min(current.f_e)),
        )


__all__ = [
    "RosenbluthPotentials2V",
    "RosenbluthDiagnostics",
    "NonlinearLandauRosenbluth2V",
]
