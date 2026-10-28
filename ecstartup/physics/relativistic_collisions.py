"""Relativistic nonlinear electron collisions on a gyrotropic momentum grid.

The electron-electron term evaluates the covariant Beliaev-Budker Landau
tensor directly.  This integral form is mathematically equivalent to the
Braams-Karney differential-potential formulation and is useful as a small-
grid reference backend.  The electron-ion term is the relativistic Lorentz
pitch-angle operator in the infinite-ion-mass limit.

The supplied collision frequencies set the dimensional time scale.  Device
calculations must obtain them from the local density, temperature, charge
state and Coulomb logarithm; this module does not hide those inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .fokker_planck import C_LIGHT_M_S, KineticState2V, RelativisticMomentumGrid


@dataclass(frozen=True)
class RelativisticCollisionDiagnostics:
    particle_residual_ee: float
    momentum_residual_ee: float
    energy_residual_ee: float
    particle_residual_ei: float
    energy_residual_ei: float
    maxwell_juttner_operator_norm: float
    substeps: int = 1
    minimum_f: float | None = None


class RelativisticBraamsKarney2V:
    """Direct ``O(Nu^2 Nphi)`` relativistic Landau reference operator.

    ``Cee`` uses the Beliaev-Budker tensor instead of solving the five
    Braams-Karney potential equations.  Both forms represent the same
    relativistic small-angle collision integral.  ``Cei`` uses the
    infinite-mass Lorentz limit and therefore transfers parallel momentum to
    the ion background while conserving electron kinetic energy.
    """

    def __init__(
        self,
        grid: RelativisticMomentumGrid,
        *,
        gyroangle_points: int = 12,
        kernel_softening_fraction: float = 0.5,
        ion_speed_regularization_u: float | None = None,
    ) -> None:
        if gyroangle_points < 4 or kernel_softening_fraction <= 0.0:
            raise ValueError("at least four gyroangles and positive softening are required")
        self.grid = grid
        self.gyroangle_points = int(gyroangle_points)
        self.softening_u = kernel_softening_fraction * min(grid.du_parallel, grid.du_perp)
        self.ion_speed_regularization_u = float(
            ion_speed_regularization_u
            if ion_speed_regularization_u is not None
            else 0.5 * min(grid.du_parallel, grid.du_perp)
        )
        if self.ion_speed_regularization_u <= 0.0:
            raise ValueError("ion speed regularization must be positive")
        self._up, self._ut = grid.mesh_u()
        self._gamma = np.sqrt(1.0 + self._up**2 + self._ut**2)
        self._weights_u = (
            2.0 * math.pi * self._ut * grid.du_parallel * grid.du_perp
        )
        self._velocity_jacobian = C_LIGHT_M_S**3 / self._gamma**5
        self._phi = 2.0 * math.pi * (
            np.arange(self.gyroangle_points) + 0.5
        ) / self.gyroangle_points

    def _check(self, state: KineticState2V) -> np.ndarray:
        if not isinstance(state.grid, RelativisticMomentumGrid):
            raise ValueError("relativistic collision state requires RelativisticMomentumGrid")
        if state.grid.shape != self.grid.shape or not np.array_equal(
            state.grid.u_parallel, self.grid.u_parallel
        ) or not np.array_equal(state.grid.u_perp, self.grid.u_perp):
            raise ValueError("relativistic collision state uses a different momentum grid")
        f_velocity = np.asarray(state.f_e, dtype=float)
        if np.any(~np.isfinite(f_velocity)) or np.any(f_velocity < 0.0):
            raise ValueError("distribution must be finite and non-negative")
        return f_velocity * self._velocity_jacobian

    def maxwell_juttner_state(
        self, density_m3: float, temperature_eV: float, *, time_s: float = 0.0
    ) -> KineticState2V:
        from ..constants import E_CHARGE_C, M_E_KG

        if density_m3 < 0.0 or temperature_eV <= 0.0:
            raise ValueError("density must be non-negative and temperature positive")
        theta = temperature_eV * E_CHARGE_C / (M_E_KG * C_LIGHT_M_S**2)
        exponent = -(self._gamma - 1.0) / theta
        exponent -= float(np.max(exponent))
        f_momentum = np.exp(np.maximum(exponent, -745.0))
        f_momentum *= density_m3 / max(float(np.sum(f_momentum * self._weights_u)), 1.0e-300)
        return KineticState2V(f_momentum / self._velocity_jacobian, self.grid, time_s)

    @staticmethod
    def _log_gradients(
        distribution: np.ndarray, du_parallel: float, du_perp: float
    ) -> tuple[np.ndarray, np.ndarray]:
        positive = distribution[distribution > 0.0]
        floor = max(float(np.min(positive)) * 1.0e-12, 1.0e-300) if positive.size else 1.0e-300
        log_f = np.log(np.maximum(distribution, floor))
        return (
            np.gradient(log_f, du_parallel, axis=0, edge_order=1),
            np.gradient(log_f, du_perp, axis=1, edge_order=1),
        )

    def _beliaev_budker_tensor(
        self, target: np.ndarray, source: np.ndarray
    ) -> np.ndarray:
        gamma = math.sqrt(1.0 + float(np.dot(target, target)))
        gamma_prime = math.sqrt(1.0 + float(np.dot(source, source)))
        r = gamma * gamma_prime - float(np.dot(target, source))
        w2 = max(r * r - 1.0, self.softening_u**2)
        w = math.sqrt(w2)
        identity = np.eye(3)
        bracket = (
            w2 * identity
            - np.outer(target, target)
            - np.outer(source, source)
            + r * (np.outer(target, source) + np.outer(source, target))
        )
        return (r * r / (gamma * gamma_prime * w**3)) * bracket

    def _divergence(self, flux_parallel: np.ndarray, flux_perp: np.ndarray) -> np.ndarray:
        parallel = np.gradient(
            flux_parallel, self.grid.du_parallel, axis=0, edge_order=1
        )
        perpendicular = np.gradient(
            self._ut * flux_perp, self.grid.du_perp, axis=1, edge_order=1
        ) / self._ut
        return parallel + perpendicular

    def _invariant_moments(self, operator_u: np.ndarray, *, include_momentum: bool) -> np.ndarray:
        basis = [np.ones(self.grid.shape)]
        if include_momentum:
            basis.append(self._up)
        basis.append(self._gamma - 1.0)
        return np.asarray(
            [np.sum(operator_u * item * self._weights_u) for item in basis], dtype=float
        )

    def _project(self, operator_u: np.ndarray, *, include_momentum: bool) -> np.ndarray:
        basis = [np.ones(self.grid.shape)]
        if include_momentum:
            basis.append(self._up)
        basis.append(self._gamma - 1.0)
        gram = np.asarray(
            [
                [np.sum(a * b * self._weights_u) for b in basis]
                for a in basis
            ],
            dtype=float,
        )
        corrected = operator_u.copy()
        for _ in range(2):
            coefficients = np.linalg.solve(
                gram, self._invariant_moments(corrected, include_momentum=include_momentum)
            )
            corrected -= np.tensordot(coefficients, np.stack(basis), axes=(0, 0))
        return corrected

    def _electron_electron(self, f_u: np.ndarray, frequency_s: float) -> np.ndarray:
        if frequency_s <= 0.0:
            return np.zeros_like(f_u)
        density = float(np.sum(f_u * self._weights_u))
        if density <= 0.0:
            return np.zeros_like(f_u)
        grad_parallel, grad_perp = self._log_gradients(
            f_u, self.grid.du_parallel, self.grid.du_perp
        )
        flux_parallel = np.zeros_like(f_u)
        flux_perp = np.zeros_like(f_u)
        for index in np.ndindex(self.grid.shape):
            target = np.array([self._ut[index], 0.0, self._up[index]], dtype=float)
            target_gradient = np.array([grad_perp[index], 0.0, grad_parallel[index]])
            flux = np.zeros(3)
            for phi in self._phi:
                cos_phi, sin_phi = math.cos(phi), math.sin(phi)
                source_x = self._ut * cos_phi
                source_y = self._ut * sin_phi
                source_gradient = np.stack(
                    (grad_perp * cos_phi, grad_perp * sin_phi, grad_parallel), axis=-1
                )
                for source_index in np.ndindex(self.grid.shape):
                    source = np.array(
                        [source_x[source_index], source_y[source_index], self._up[source_index]],
                        dtype=float,
                    )
                    tensor = self._beliaev_budker_tensor(target, source)
                    flux += (
                        f_u[index]
                        * f_u[source_index]
                        * self._weights_u[source_index]
                        / self.gyroangle_points
                        * (tensor @ (target_gradient - source_gradient[source_index]))
                    )
            flux_perp[index], flux_parallel[index] = flux[0], flux[2]
        raw = frequency_s * self._divergence(flux_parallel, flux_perp) / density
        return self._project(raw, include_momentum=True)

    def _electron_ion(self, f_u: np.ndarray, frequency_s: float) -> np.ndarray:
        if frequency_s <= 0.0:
            return np.zeros_like(f_u)
        grad_parallel, grad_perp = self._log_gradients(
            f_u, self.grid.du_parallel, self.grid.du_perp
        )
        flux_parallel = np.zeros_like(f_u)
        flux_perp = np.zeros_like(f_u)
        u2 = self._up**2 + self._ut**2
        u = np.sqrt(u2)
        regularized = np.maximum(u, self.ion_speed_regularization_u)
        nu_deflection = frequency_s * (self.ion_speed_regularization_u / regularized) ** 3
        for index in np.ndindex(self.grid.shape):
            vector = np.array([self._ut[index], 0.0, self._up[index]], dtype=float)
            norm2 = max(float(np.dot(vector, vector)), self.ion_speed_regularization_u**2)
            projector = np.eye(3) - np.outer(vector, vector) / norm2
            gradient = np.array([grad_perp[index], 0.0, grad_parallel[index]])
            diffusion = 0.5 * nu_deflection[index] * norm2
            flux = diffusion * f_u[index] * (projector @ gradient)
            flux_perp[index], flux_parallel[index] = flux[0], flux[2]
        raw = self._divergence(flux_parallel, flux_perp)
        return self._project(raw, include_momentum=False)

    def evaluate(
        self,
        state: KineticState2V,
        *,
        electron_electron_frequency_s: float,
        electron_ion_frequency_s: float,
    ) -> tuple[np.ndarray, RelativisticCollisionDiagnostics]:
        if electron_electron_frequency_s < 0.0 or electron_ion_frequency_s < 0.0:
            raise ValueError("collision frequencies must be non-negative")
        f_u = self._check(state)
        c_ee = self._electron_electron(f_u, electron_electron_frequency_s)
        c_ei = self._electron_ion(f_u, electron_ion_frequency_s)
        ee = self._invariant_moments(c_ee, include_momentum=True)
        ei = self._invariant_moments(c_ei, include_momentum=False)
        operator_velocity = (c_ee + c_ei) / self._velocity_jacobian
        norm = float(
            np.sqrt(np.sum((operator_velocity * self.grid.phase_volume_weights_m3_s3) ** 2))
        )
        return operator_velocity, RelativisticCollisionDiagnostics(
            particle_residual_ee=float(ee[0]),
            momentum_residual_ee=float(ee[1]),
            energy_residual_ee=float(ee[2]),
            particle_residual_ei=float(ei[0]),
            energy_residual_ei=float(ei[1]),
            maxwell_juttner_operator_norm=norm,
        )

    def advance(
        self,
        state: KineticState2V,
        dt_s: float,
        *,
        electron_electron_frequency_s: float,
        electron_ion_frequency_s: float,
        positivity_safety: float = 0.8,
        maximum_substeps: int = 1000,
    ) -> tuple[KineticState2V, RelativisticCollisionDiagnostics]:
        if dt_s <= 0.0 or not 0.0 < positivity_safety < 1.0:
            raise ValueError("positive dt and positivity_safety in (0,1) are required")
        current = KineticState2V(np.asarray(state.f_e, dtype=float).copy(), self.grid, state.time_s)
        remaining = float(dt_s)
        substeps = 0
        last: RelativisticCollisionDiagnostics | None = None
        while remaining > 0.0:
            operator, diagnostics = self.evaluate(
                current,
                electron_electron_frequency_s=electron_electron_frequency_s,
                electron_ion_frequency_s=electron_ion_frequency_s,
            )
            negative = operator < 0.0
            stable = remaining
            if np.any(negative):
                stable = min(
                    remaining,
                    positivity_safety
                    * float(np.min(current.f_e[negative] / np.maximum(-operator[negative], 1.0e-300))),
                )
            if stable <= max(np.finfo(float).eps * dt_s, 1.0e-30):
                raise FloatingPointError("relativistic collision positivity step underflow")
            current = KineticState2V(
                np.maximum(current.f_e + stable * operator, 0.0),
                self.grid,
                current.time_s + stable,
            )
            remaining -= stable
            substeps += 1
            last = diagnostics
            if substeps > maximum_substeps:
                raise RuntimeError("relativistic collision step exceeded maximum_substeps")
        assert last is not None
        return current, RelativisticCollisionDiagnostics(
            **{
                **last.__dict__,
                "substeps": substeps,
                "minimum_f": float(np.min(current.f_e)),
            }
        )


__all__ = ["RelativisticBraamsKarney2V", "RelativisticCollisionDiagnostics"]
