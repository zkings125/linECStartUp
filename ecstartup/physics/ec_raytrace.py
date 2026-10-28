"""Magnetic-surface-free Hamiltonian EC ray tracing for startup fields.

The independent variable is geometric path length.  The canonical toroidal
wave number ``p_phi = R*k_phi`` is conserved in an axisymmetric field.  The
Hamiltonian is the local cold-plasma electromagnetic dispersion determinant.

This is a geometrical-optics propagation backend, not a hot-plasma absorption
or full-wave solver.  It stops explicitly at cutoff, selected EC resonance,
domain exit, or a WKB quality failure.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

import numpy as np

from ..config import GeometryConfig, LaunchConfig, RayTracingConfig, RFSourceConfig
from ..constants import E_CHARGE_C, M_E_KG, PI
from .cold_plasma import (
    C_LIGHT_M_S,
    ColdPlasmaDispersionError,
    launch_wavevector,
    mode_solution,
    stix_parameters,
)
from .magnetic import MagneticFieldProvider


class ElectronDensityProvider(Protocol):
    def electron_density_m3(self, R_m: float, Z_m: float, t_s: float) -> float: ...


@dataclass(frozen=True)
class UniformElectronDensity:
    value_m3: float

    def __post_init__(self) -> None:
        if self.value_m3 < 0.0:
            raise ValueError("electron density must be non-negative")

    def electron_density_m3(self, R_m: float, Z_m: float, t_s: float) -> float:
        return self.value_m3


@dataclass(frozen=True)
class ECRayPoint:
    path_length_m: float
    R_m: float
    Z_m: float
    phi_rad: float
    k_R_m_inv: float
    k_Z_m_inv: float
    k_phi_m_inv: float
    B_T: float
    electron_density_m3: float
    refractive_index: float
    n_parallel: float
    n_perpendicular: float
    resonance_mismatch: float
    hamiltonian: float
    wkb_scale_ratio: float

    def to_dict(self) -> dict[str, float]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class ECRayTrace:
    source_name: str
    frequency_Hz: float
    harmonic: int
    mode: str
    status: str
    reason: str
    launch_direction_source: str
    t_s: float
    Ip_A: float
    points: tuple[ECRayPoint, ...]

    @property
    def reached_resonance(self) -> bool:
        return self.status == "resonance_reached"

    @property
    def path_length_m(self) -> float:
        return self.points[-1].path_length_m if self.points else 0.0

    def to_dict(self, *, include_path: bool = True) -> dict:
        payload = {
            "source_name": self.source_name,
            "frequency_Hz": self.frequency_Hz,
            "harmonic": self.harmonic,
            "mode": self.mode,
            "status": self.status,
            "reason": self.reason,
            "launch_direction_source": self.launch_direction_source,
            "t_s": self.t_s,
            "Ip_A": self.Ip_A,
            "reached_resonance": self.reached_resonance,
            "path_length_m": self.path_length_m,
            "sample_count": len(self.points),
        }
        if self.points:
            payload["final"] = self.points[-1].to_dict()
        if include_path:
            payload["path"] = [point.to_dict() for point in self.points]
        return payload


def launch_direction(launch: LaunchConfig, geometry: GeometryConfig) -> tuple[np.ndarray, str]:
    """Return a cylindrical ``(R,Z,phi)`` unit direction.

    If no poloidal angle is supplied, the auditable fallback aims at
    ``(geometry.R0_m, 0)``.  It is only a software default; a quantitative run
    must use measured launcher steering.
    """

    R = launch.R_m if launch.R_m is not None else geometry.R0_m
    if launch.poloidal_angle_deg is None:
        poloidal = math.atan2(-launch.Z_m, geometry.R0_m - R)
        source = "aimed_at_geometry_center_fallback"
    else:
        poloidal = math.radians(launch.poloidal_angle_deg)
        source = "configured_poloidal_angle"
    toroidal = math.radians(launch.toroidal_angle_deg)
    direction = np.asarray(
        [
            math.cos(toroidal) * math.cos(poloidal),
            math.cos(toroidal) * math.sin(poloidal),
            math.sin(toroidal),
        ],
        dtype=float,
    )
    return direction / np.linalg.norm(direction), source


class HamiltonianECRayTracer:
    """Trace one cold-plasma O/X branch through an axisymmetric startup field."""

    def __init__(
        self,
        magnetic: MagneticFieldProvider,
        density: ElectronDensityProvider,
        geometry: GeometryConfig,
        config: RayTracingConfig | None = None,
        *,
        ion_mass_kg: float = 3.343583719e-27,
    ) -> None:
        self.magnetic = magnetic
        self.density = density
        self.geometry = geometry
        self.config = config or RayTracingConfig()
        self.ion_mass_kg = ion_mass_kg

    def _bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return (
            self.config.R_bounds_m or self.geometry.R_bounds_m,
            self.config.Z_bounds_m or self.geometry.Z_bounds_m,
        )

    def _inside(self, R_m: float, Z_m: float) -> bool:
        rb, zb = self._bounds()
        return rb[0] <= R_m <= rb[1] and zb[0] <= Z_m <= zb[1]

    def _field_vector(self, R_m: float, Z_m: float, Ip_A: float, t_s: float) -> np.ndarray:
        point = self.magnetic.evaluate(R_m, Z_m, Ip_A, t_s)
        return np.asarray([point.B_R_T, point.B_Z_T, point.B_phi_T], dtype=float)

    def _hamiltonian(
        self,
        R_m: float,
        Z_m: float,
        k_R: float,
        k_Z: float,
        canonical_k_phi: float,
        frequency_Hz: float,
        Ip_A: float,
        t_s: float,
        mode: str,
    ) -> float:
        k_phi = canonical_k_phi / max(R_m, 1.0e-9)
        k = np.asarray([k_R, k_Z, k_phi], dtype=float)
        B = self._field_vector(R_m, Z_m, Ip_A, t_s)
        B_magnitude = float(np.linalg.norm(B))
        k_magnitude = float(np.linalg.norm(k))
        if B_magnitude <= 0.0 or k_magnitude <= 0.0:
            raise ColdPlasmaDispersionError("wavevector or magnetic field is singular")
        cosine = float(np.clip(abs(np.dot(k, B)) / (k_magnitude * B_magnitude), 0.0, 1.0))
        stix = stix_parameters(
            frequency_Hz,
            B_magnitude,
            self.density.electron_density_m3(R_m, Z_m, t_s),
            ion_mass_kg=self.ion_mass_kg,
        )
        branch = mode_solution(stix, math.acos(cosine), mode)
        omega = 2.0 * PI * frequency_Hz
        refractive_index_squared = (C_LIGHT_M_S * k_magnitude / omega) ** 2
        return refractive_index_squared - branch.refractive_index_squared

    def _rhs(
        self,
        state: np.ndarray,
        canonical_k_phi: float,
        frequency_Hz: float,
        Ip_A: float,
        t_s: float,
        mode: str,
        orientation_sign: float = 1.0,
    ) -> np.ndarray:
        R, Z, _phi, k_R, k_Z = state
        h_x = self.config.spatial_difference_m
        k0 = 2.0 * PI * frequency_Hz / C_LIGHT_M_S

        def H(r: float, z: float, kr: float, kz: float, pphi: float = canonical_k_phi) -> float:
            return self._hamiltonian(r, z, kr, kz, pphi, frequency_Hz, Ip_A, t_s, mode)

        h_kr = max(k0, abs(k_R), 1.0) * self.config.wavevector_relative_step
        h_kz = max(k0, abs(k_Z), 1.0) * self.config.wavevector_relative_step
        k_phi = canonical_k_phi / max(R, 1.0e-9)
        h_kphi = max(k0, abs(k_phi), 1.0) * self.config.wavevector_relative_step
        dH_dkR = (H(R, Z, k_R + h_kr, k_Z) - H(R, Z, k_R - h_kr, k_Z)) / (2.0 * h_kr)
        dH_dkZ = (H(R, Z, k_R, k_Z + h_kz) - H(R, Z, k_R, k_Z - h_kz)) / (2.0 * h_kz)
        pphi_plus = R * (k_phi + h_kphi)
        pphi_minus = R * (k_phi - h_kphi)
        dH_dkPhi = (H(R, Z, k_R, k_Z, pphi_plus) - H(R, Z, k_R, k_Z, pphi_minus)) / (2.0 * h_kphi)
        dH_dR = (H(R + h_x, Z, k_R, k_Z) - H(R - h_x, Z, k_R, k_Z)) / (2.0 * h_x)
        dH_dZ = (H(R, Z + h_x, k_R, k_Z) - H(R, Z - h_x, k_R, k_Z)) / (2.0 * h_x)
        group_norm = math.sqrt(dH_dkR**2 + dH_dkZ**2 + dH_dkPhi**2)
        if not math.isfinite(group_norm) or group_norm <= 1.0e-14:
            raise ColdPlasmaDispersionError("Hamiltonian group direction is singular")
        return orientation_sign * np.asarray(
            [
                dH_dkR / group_norm,
                dH_dkZ / group_norm,
                dH_dkPhi / (group_norm * max(R, 1.0e-9)),
                -dH_dR / group_norm,
                -dH_dZ / group_norm,
            ]
        )

    def _wkb_scale_ratio(self, R: float, Z: float, Ip_A: float, t_s: float, k_norm: float) -> float:
        h = self.config.spatial_difference_m

        def Bmag(r: float, z: float) -> float:
            return float(np.linalg.norm(self._field_vector(r, z, Ip_A, t_s)))

        B0 = Bmag(R, Z)
        dB_dR = (Bmag(R + h, Z) - Bmag(R - h, Z)) / (2.0 * h)
        dB_dZ = (Bmag(R, Z + h) - Bmag(R, Z - h)) / (2.0 * h)
        grad_B = math.hypot(dB_dR, dB_dZ)
        scale_B = math.inf if grad_B <= 0.0 else B0 / grad_B

        n0 = self.density.electron_density_m3(R, Z, t_s)
        dn_dR = (
            self.density.electron_density_m3(R + h, Z, t_s)
            - self.density.electron_density_m3(R - h, Z, t_s)
        ) / (2.0 * h)
        dn_dZ = (
            self.density.electron_density_m3(R, Z + h, t_s)
            - self.density.electron_density_m3(R, Z - h, t_s)
        ) / (2.0 * h)
        grad_n = math.hypot(dn_dR, dn_dZ)
        scale_n = math.inf if grad_n <= 0.0 or n0 <= 0.0 else n0 / grad_n
        wavelength = 2.0 * PI / max(k_norm, 1.0e-30)
        return min(scale_B, scale_n) / wavelength

    def _sample(
        self,
        path_length_m: float,
        state: np.ndarray,
        canonical_k_phi: float,
        source: RFSourceConfig,
        Ip_A: float,
        t_s: float,
    ) -> ECRayPoint:
        R, Z, phi, k_R, k_Z = state
        k_phi = canonical_k_phi / max(R, 1.0e-9)
        k = np.asarray([k_R, k_Z, k_phi])
        B = self._field_vector(R, Z, Ip_A, t_s)
        Bmag = float(np.linalg.norm(B))
        kmag = float(np.linalg.norm(k))
        omega = 2.0 * PI * source.frequency_Hz
        n_vector = C_LIGHT_M_S * k / omega
        b = B / max(Bmag, 1.0e-30)
        n_parallel = float(np.dot(n_vector, b))
        n2 = float(np.dot(n_vector, n_vector))
        n_perp = math.sqrt(max(0.0, n2 - n_parallel**2))
        resonance_ratio = source.harmonic * E_CHARGE_C * Bmag / (M_E_KG * omega)
        resonance_mismatch = abs(resonance_ratio - 1.0)
        try:
            hamiltonian = self._hamiltonian(
                R, Z, k_R, k_Z, canonical_k_phi, source.frequency_Hz, Ip_A, t_s,
                source.launch.mode,
            )
        except ColdPlasmaDispersionError:
            if resonance_mismatch > self.config.resonance_tolerance:
                raise
            # The collisionless cold tensor is singular at the fundamental.
            # Preserve the resonance event instead of misreporting it as a
            # generic numerical failure. A hot dielectric must continue here.
            hamiltonian = float("nan")
        return ECRayPoint(
            path_length_m=path_length_m,
            R_m=R,
            Z_m=Z,
            phi_rad=phi,
            k_R_m_inv=k_R,
            k_Z_m_inv=k_Z,
            k_phi_m_inv=k_phi,
            B_T=Bmag,
            electron_density_m3=self.density.electron_density_m3(R, Z, t_s),
            refractive_index=math.sqrt(max(n2, 0.0)),
            n_parallel=n_parallel,
            n_perpendicular=n_perp,
            resonance_mismatch=resonance_mismatch,
            hamiltonian=hamiltonian,
            wkb_scale_ratio=self._wkb_scale_ratio(R, Z, Ip_A, t_s, kmag),
        )

    def trace(self, source: RFSourceConfig, *, Ip_A: float = 0.0, t_s: float = 0.0) -> ECRayTrace:
        launch_R = source.launch.R_m if source.launch.R_m is not None else self.geometry.R0_m
        launch_Z = source.launch.Z_m
        direction, direction_source = launch_direction(source.launch, self.geometry)
        if not self._inside(launch_R, launch_Z):
            return ECRayTrace(
                source.name,
                source.frequency_Hz,
                source.harmonic,
                source.launch.mode,
                "outside_domain",
                "launch point is outside ray-tracing bounds",
                direction_source,
                t_s,
                Ip_A,
                (),
            )
        B = self._field_vector(launch_R, launch_Z, Ip_A, t_s)
        density = self.density.electron_density_m3(launch_R, launch_Z, t_s)
        launch_resonance_ratio = (
            source.harmonic
            * E_CHARGE_C
            * float(np.linalg.norm(B))
            / (M_E_KG * 2.0 * PI * source.frequency_Hz)
        )
        if abs(launch_resonance_ratio - 1.0) <= self.config.resonance_tolerance:
            return ECRayTrace(
                source.name,
                source.frequency_Hz,
                source.harmonic,
                source.launch.mode,
                "resonance_reached",
                "launch point is already on the selected electron-cyclotron harmonic",
                direction_source,
                t_s,
                Ip_A,
                (),
            )
        try:
            k, _solution = launch_wavevector(
                source.frequency_Hz,
                direction,
                B,
                density,
                source.launch.mode,
                ion_mass_kg=self.ion_mass_kg,
            )
        except (ColdPlasmaDispersionError, ValueError) as exc:
            return ECRayTrace(
                source.name,
                source.frequency_Hz,
                source.harmonic,
                source.launch.mode,
                "launch_inaccessible",
                str(exc),
                direction_source,
                t_s,
                Ip_A,
                (),
            )

        canonical_k_phi = launch_R * k[2]
        state = np.asarray([launch_R, launch_Z, 0.0, k[0], k[1]], dtype=float)
        try:
            initial_rhs = self._rhs(
                state, canonical_k_phi, source.frequency_Hz, Ip_A, t_s, source.launch.mode
            )
        except ColdPlasmaDispersionError as exc:
            return ECRayTrace(
                source.name,
                source.frequency_Hz,
                source.harmonic,
                source.launch.mode,
                "launch_inaccessible",
                str(exc),
                direction_source,
                t_s,
                Ip_A,
                (),
            )
        initial_group_direction = np.asarray(
            [initial_rhs[0], initial_rhs[1], launch_R * initial_rhs[2]]
        )
        orientation_sign = 1.0 if float(np.dot(initial_group_direction, direction)) >= 0.0 else -1.0
        points: list[ECRayPoint] = []
        path_length = 0.0
        status = "max_length"
        reason = "maximum path length reached"
        for _step in range(self.config.max_steps + 1):
            if not self._inside(float(state[0]), float(state[1])):
                status = "domain_exit"
                reason = "ray left the configured computational domain"
                break
            try:
                sample = self._sample(path_length, state, canonical_k_phi, source, Ip_A, t_s)
                points.append(sample)
                if sample.resonance_mismatch <= self.config.resonance_tolerance:
                    status = "resonance_reached"
                    reason = "selected electron-cyclotron harmonic reached"
                    break
                B_here = self._field_vector(float(state[0]), float(state[1]), Ip_A, t_s)
                k_here = np.asarray([state[3], state[4], canonical_k_phi / max(state[0], 1.0e-9)])
                cosine = abs(float(np.dot(k_here, B_here))) / max(
                    float(np.linalg.norm(k_here) * np.linalg.norm(B_here)), 1.0e-30
                )
                local_stix = stix_parameters(
                    source.frequency_Hz,
                    float(np.linalg.norm(B_here)),
                    sample.electron_density_m3,
                    ion_mass_kg=self.ion_mass_kg,
                )
                mode_solution(local_stix, math.acos(float(np.clip(cosine, 0.0, 1.0))), source.launch.mode)
                if sample.wkb_scale_ratio < self.config.wkb_min_scale_ratio:
                    status = "wkb_invalid"
                    reason = "background scale length is too short relative to wavelength; use full-wave/FEM"
                    if self.config.terminate_on_wkb_failure:
                        break
                if path_length >= self.config.max_length_m:
                    break

                h = min(self.config.step_m, self.config.max_length_m - path_length)
                k1 = self._rhs(
                    state, canonical_k_phi, source.frequency_Hz, Ip_A, t_s,
                    source.launch.mode, orientation_sign
                )
                k2 = self._rhs(
                    state + 0.5 * h * k1,
                    canonical_k_phi,
                    source.frequency_Hz,
                    Ip_A,
                    t_s,
                    source.launch.mode,
                    orientation_sign,
                )
                k3 = self._rhs(
                    state + 0.5 * h * k2,
                    canonical_k_phi,
                    source.frequency_Hz,
                    Ip_A,
                    t_s,
                    source.launch.mode,
                    orientation_sign,
                )
                k4 = self._rhs(
                    state + h * k3,
                    canonical_k_phi,
                    source.frequency_Hz,
                    Ip_A,
                    t_s,
                    source.launch.mode,
                    orientation_sign,
                )
                state = state + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
                path_length += h
            except ColdPlasmaDispersionError as exc:
                status = "dispersion_failure"
                reason = str(exc)
                break
            except (FloatingPointError, ValueError) as exc:
                status = "numerical_failure"
                reason = str(exc)
                break
        else:
            status = "max_steps"
            reason = "maximum ray step count reached"
        return ECRayTrace(
            source.name,
            source.frequency_Hz,
            source.harmonic,
            source.launch.mode,
            status,
            reason,
            direction_source,
            t_s,
            Ip_A,
            tuple(points),
        )
