"""Cold-plasma EC dispersion utilities.

The implementation uses the Stix ``S, D, P`` dielectric tensor and the
electromagnetic dispersion determinant

``det(epsilon + n n - n**2 I) = 0``.

It is intentionally independent of magnetic surfaces.  Callers supply the
local magnetic field and density, which may come from vacuum coils,
Biot--Savart fields, tabulated measurements, or a later GS equilibrium.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ..constants import E_CHARGE_C, EPS0_F_M, M_E_KG, PI


C_LIGHT_M_S = 299_792_458.0
PROTON_MASS_KG = 1.67262192369e-27


class ColdPlasmaDispersionError(ValueError):
    """The local cold-plasma dispersion is singular or has no real branch."""


@dataclass(frozen=True)
class StixParameters:
    S: float
    D: float
    P: float
    R: float
    L: float
    X_e: float
    Y_e: float


@dataclass(frozen=True)
class ModeSolution:
    mode: str
    refractive_index_squared: float
    theta_rad: float

    @property
    def refractive_index(self) -> float:
        return math.sqrt(self.refractive_index_squared)


def _species_terms(
    omega_rad_s: float,
    B_T: float,
    density_m3: float,
    charge_C: float,
    mass_kg: float,
) -> tuple[float, float]:
    plasma_frequency_squared = density_m3 * charge_C**2 / (EPS0_F_M * mass_kg)
    signed_cyclotron_frequency = charge_C * B_T / mass_kg
    denominator = omega_rad_s**2 - signed_cyclotron_frequency**2
    scale = max(omega_rad_s**2, 1.0)
    if abs(denominator) <= 1.0e-12 * scale:
        raise ColdPlasmaDispersionError("cold-plasma tensor is singular at a cyclotron resonance")
    return plasma_frequency_squared, signed_cyclotron_frequency


def stix_parameters(
    frequency_Hz: float,
    B_T: float,
    electron_density_m3: float,
    *,
    ion_density_m3: float | None = None,
    ion_mass_kg: float = 2.0 * PROTON_MASS_KG,
    ion_charge_state: int = 1,
) -> StixParameters:
    """Return collisionless cold-plasma Stix parameters for electrons + one ion.

    The ion contribution is retained for completeness even though it is tiny
    at EC frequencies.  Hot/relativistic harmonic absorption is not part of a
    cold tensor and must be supplied by a kinetic dielectric backend.
    """

    if frequency_Hz <= 0.0:
        raise ValueError("frequency_Hz must be positive")
    if B_T < 0.0 or electron_density_m3 < 0.0:
        raise ValueError("B_T and electron_density_m3 must be non-negative")
    if ion_mass_kg <= 0.0 or ion_charge_state < 1:
        raise ValueError("ion mass and charge state must be positive")

    omega = 2.0 * PI * frequency_Hz
    ion_density = electron_density_m3 / ion_charge_state if ion_density_m3 is None else ion_density_m3
    species = (
        (electron_density_m3, -E_CHARGE_C, M_E_KG),
        (ion_density, ion_charge_state * E_CHARGE_C, ion_mass_kg),
    )
    S = 1.0
    D = 0.0
    P = 1.0
    electron_wp2 = 0.0
    electron_omega_c = 0.0
    for density, charge, mass in species:
        if density <= 0.0:
            continue
        wp2, omega_c = _species_terms(omega, B_T, density, charge, mass)
        denominator = omega**2 - omega_c**2
        S -= wp2 / denominator
        D += (omega_c / omega) * wp2 / denominator
        P -= wp2 / omega**2
        if charge < 0.0:
            electron_wp2 = wp2
            electron_omega_c = abs(omega_c)
    return StixParameters(
        S=S,
        D=D,
        P=P,
        R=S + D,
        L=S - D,
        X_e=electron_wp2 / omega**2,
        Y_e=electron_omega_c / omega,
    )


def mode_solution(stix: StixParameters, theta_rad: float, mode: str) -> ModeSolution:
    """Solve the cold Appleton-Hartree quadratic for an O/X branch.

    At perpendicular launch the O root is exactly ``P`` and the X root is
    ``R*L/S``.  Away from perpendicular propagation, the O label follows the
    root continuously closest to ``P``.  Close to parallel propagation the
    physical eigenmodes are R/L rather than O/X, so callers should treat the
    label as a branch-tracking convention.
    """

    normalized_mode = str(mode).strip().upper()
    if not (normalized_mode.startswith("O") or normalized_mode.startswith("X")):
        raise ValueError("cold EC mode must start with 'O' or 'X'")
    sin2 = math.sin(theta_rad) ** 2
    cos2 = math.cos(theta_rad) ** 2
    A = stix.S * sin2 + stix.P * cos2
    B = stix.R * stix.L * sin2 + stix.P * stix.S * (1.0 + cos2)
    C = stix.P * stix.R * stix.L
    scale = max(abs(B * B), abs(4.0 * A * C), 1.0)
    discriminant = B * B - 4.0 * A * C
    if discriminant < -1.0e-12 * scale or abs(A) <= 1.0e-14:
        raise ColdPlasmaDispersionError("no real electromagnetic O/X branch")
    discriminant = max(0.0, discriminant)
    roots = [(B - math.sqrt(discriminant)) / (2.0 * A), (B + math.sqrt(discriminant)) / (2.0 * A)]
    positive = [root for root in roots if math.isfinite(root) and root > 0.0]
    if not positive:
        raise ColdPlasmaDispersionError("the selected direction is beyond cutoff")
    positive.sort(key=lambda root: abs(root - stix.P))
    root = positive[0] if normalized_mode.startswith("O") else positive[-1]
    return ModeSolution(normalized_mode, root, theta_rad)


def dielectric_tensor(stix: StixParameters, B_vector_T: np.ndarray) -> np.ndarray:
    """Return the Hermitian cold dielectric tensor in the supplied basis."""

    B = np.asarray(B_vector_T, dtype=float)
    magnitude = float(np.linalg.norm(B))
    if magnitude <= 0.0:
        raise ColdPlasmaDispersionError("magnetic-field direction is undefined")
    e3 = B / magnitude
    reference = np.asarray([0.0, 0.0, 1.0])
    if abs(float(np.dot(reference, e3))) > 0.9:
        reference = np.asarray([1.0, 0.0, 0.0])
    e1 = np.cross(reference, e3)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(e3, e1)
    rotation = np.column_stack((e1, e2, e3))
    aligned = np.asarray(
        [
            [stix.S, -1j * stix.D, 0.0],
            [1j * stix.D, stix.S, 0.0],
            [0.0, 0.0, stix.P],
        ],
        dtype=complex,
    )
    return rotation @ aligned @ rotation.T


def dispersion_determinant(
    frequency_Hz: float,
    wavevector_m_inv: np.ndarray,
    B_vector_T: np.ndarray,
    electron_density_m3: float,
    *,
    ion_mass_kg: float = 2.0 * PROTON_MASS_KG,
) -> float:
    """Evaluate ``det(epsilon + n n - n^2 I)`` for a real wavevector."""

    k = np.asarray(wavevector_m_inv, dtype=float)
    B = np.asarray(B_vector_T, dtype=float)
    stix = stix_parameters(
        frequency_Hz,
        float(np.linalg.norm(B)),
        electron_density_m3,
        ion_mass_kg=ion_mass_kg,
    )
    omega = 2.0 * PI * frequency_Hz
    n = C_LIGHT_M_S * k / omega
    n2 = float(np.dot(n, n))
    matrix = dielectric_tensor(stix, B) + np.outer(n, n) - n2 * np.eye(3)
    determinant = np.linalg.det(matrix)
    if abs(float(determinant.imag)) > 1.0e-8 * max(1.0, abs(float(determinant.real))):
        raise ColdPlasmaDispersionError("collisionless dispersion determinant is unexpectedly complex")
    return float(determinant.real)


def launch_wavevector(
    frequency_Hz: float,
    direction: np.ndarray,
    B_vector_T: np.ndarray,
    electron_density_m3: float,
    mode: str,
    *,
    ion_mass_kg: float = 2.0 * PROTON_MASS_KG,
) -> tuple[np.ndarray, ModeSolution]:
    """Place an O/X launch on the local cold-plasma dispersion surface."""

    unit = np.asarray(direction, dtype=float)
    norm = float(np.linalg.norm(unit))
    B = np.asarray(B_vector_T, dtype=float)
    B_norm = float(np.linalg.norm(B))
    if norm <= 0.0 or B_norm <= 0.0:
        raise ValueError("launch direction and magnetic field must be non-zero")
    unit /= norm
    cosine = float(np.clip(abs(np.dot(unit, B / B_norm)), 0.0, 1.0))
    theta = math.acos(cosine)
    stix = stix_parameters(
        frequency_Hz,
        B_norm,
        electron_density_m3,
        ion_mass_kg=ion_mass_kg,
    )
    solution = mode_solution(stix, theta, mode)
    k0 = 2.0 * PI * frequency_Hz / C_LIGHT_M_S
    return k0 * solution.refractive_index * unit, solution
