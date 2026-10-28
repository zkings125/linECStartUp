"""O-X-B availability diagnostics and schematic EBW propagation.

The module deliberately works in physical R-Z coordinates and never assumes
closed magnetic surfaces.  It identifies the O cutoff, slow-X/UHR layer and
electron-cyclotron resonance from local density and magnetic field providers.

The O-X efficiency is the standard cold-plasma slab/Budden estimate.  The
X-B step is reported as a candidate and is not assigned a first-principles
efficiency until a full-wave or verified tunnelling model is supplied.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

import numpy as np

from ..config import EBWConfig, GeometryConfig, RFSourceConfig
from ..constants import E_CHARGE_C, EPS0_F_M, M_E_KG, PI
from .cold_plasma import C_LIGHT_M_S, ColdPlasmaDispersionError, stix_parameters
from .ec_raytrace import ECRayTrace
from .magnetic import MagneticFieldProvider


class DensityProvider(Protocol):
    def electron_density_m3(self, R_m: float, Z_m: float, t_s: float) -> float: ...


@dataclass(frozen=True)
class GaussianDensityCloud:
    center_R_m: float
    center_Z_m: float
    sigma_R_m: float
    sigma_Z_m: float
    peak_density_m3: float
    floor_density_m3: float = 0.0

    def electron_density_m3(self, R_m: float, Z_m: float, t_s: float) -> float:
        exponent = -0.5 * (
            ((R_m - self.center_R_m) / self.sigma_R_m) ** 2
            + ((Z_m - self.center_Z_m) / self.sigma_Z_m) ** 2
        )
        return self.floor_density_m3 + self.peak_density_m3 * math.exp(exponent)


@dataclass(frozen=True)
class ConversionPoint:
    kind: str
    path_index: int
    path_length_m: float
    R_m: float
    Z_m: float
    electron_density_m3: float
    B_T: float
    residual: float

    def to_dict(self) -> dict[str, float | str]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class EBWPathPoint:
    path_length_m: float
    R_m: float
    Z_m: float
    B_T: float
    electron_density_m3: float
    resonance_mismatch: float

    def to_dict(self) -> dict[str, float]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class OXBResult:
    source_name: str
    frequency_Hz: float
    harmonic: int
    available: bool
    reason: str
    O_cutoff_density_m3: float
    maximum_density_m3: float
    overdense: bool
    O_cutoff: ConversionPoint | None
    UHR: ConversionPoint | None
    resonance: ConversionPoint | None
    ox_efficiency_estimate: float
    xb_efficiency_estimate: float | None
    combined_efficiency_estimate: float | None
    n_parallel_at_O_cutoff: float | None
    optimal_n_parallel: float | None
    density_scale_length_m: float | None
    conversion_separation_m: float | None
    ebw_resonance_geometric_candidate: bool
    ebwcd_candidate: bool
    current_drive_direction_sign: int | None
    ebw_path: tuple[EBWPathPoint, ...]

    def to_dict(self, *, include_path: bool = True) -> dict:
        payload = {
            "source_name": self.source_name,
            "frequency_Hz": self.frequency_Hz,
            "harmonic": self.harmonic,
            "available": self.available,
            "reason": self.reason,
            "O_cutoff_density_m3": self.O_cutoff_density_m3,
            "maximum_density_m3": self.maximum_density_m3,
            "overdense": self.overdense,
            "O_cutoff": None if self.O_cutoff is None else self.O_cutoff.to_dict(),
            "UHR": None if self.UHR is None else self.UHR.to_dict(),
            "resonance": None if self.resonance is None else self.resonance.to_dict(),
            "ox_efficiency_estimate": self.ox_efficiency_estimate,
            "xb_efficiency_estimate": self.xb_efficiency_estimate,
            "combined_efficiency_estimate": self.combined_efficiency_estimate,
            "n_parallel_at_O_cutoff": self.n_parallel_at_O_cutoff,
            "optimal_n_parallel": self.optimal_n_parallel,
            "density_scale_length_m": self.density_scale_length_m,
            "conversion_separation_m": self.conversion_separation_m,
            "ebw_resonance_geometric_candidate": self.ebw_resonance_geometric_candidate,
            "ebwcd_candidate": self.ebwcd_candidate,
            "ebwcd_assessment": (
                "geometric_candidate_only" if self.ebwcd_candidate else "not_a_candidate"
            ),
            "current_drive_direction_sign": self.current_drive_direction_sign,
            "ebw_path_length_m": self.ebw_path[-1].path_length_m if self.ebw_path else 0.0,
            "ebw_path_sample_count": len(self.ebw_path),
            "ox_model": "reduced_cold_slab_angular_mismatch_proxy",
            "xb_model": "candidate_only; full-wave/tunnelling efficiency not implemented",
            "ebw_propagation_model": "schematic density-gradient path, not hot Bernstein dispersion",
            "current_drive_model": (
                "not evaluated by analyze-oxb; export a power-normalized D_ql to the "
                "independent solve-fp backend for a distribution-moment current"
            ),
        }
        if include_path:
            payload["ebw_path"] = [point.to_dict() for point in self.ebw_path]
        return payload


@dataclass(frozen=True)
class XBranchSample:
    """Cold perpendicular-X branch classification at one path sample."""

    path_length_m: float
    R_m: float
    Z_m: float
    electron_density_m3: float
    B_T: float
    stix_R: float
    stix_L: float
    stix_S: float
    n_squared_X: float | None
    branch: str

    def to_dict(self) -> dict[str, float | str | None]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class DirectXBResult:
    """Direct X-B accessibility evidence without assigning conversion power."""

    source_name: str
    frequency_Hz: float
    available_candidate: bool
    reason: str
    R_cutoff: ConversionPoint | None
    L_cutoff: ConversionPoint | None
    UHR: ConversionPoint | None
    incident_cutoff_kind: str | None
    evanescent_width_m: float | None
    fast_X_identified: bool
    slow_X_identified: bool
    mode_conversion_verified: bool
    samples: tuple[XBranchSample, ...]

    def to_dict(self, *, include_path: bool = True) -> dict:
        payload = {
            "source_name": self.source_name,
            "frequency_Hz": self.frequency_Hz,
            "available_candidate": self.available_candidate,
            "reason": self.reason,
            "R_cutoff": None if self.R_cutoff is None else self.R_cutoff.to_dict(),
            "L_cutoff": None if self.L_cutoff is None else self.L_cutoff.to_dict(),
            "UHR": None if self.UHR is None else self.UHR.to_dict(),
            "incident_cutoff_kind": self.incident_cutoff_kind,
            "evanescent_width_m": self.evanescent_width_m,
            "fast_X_identified": self.fast_X_identified,
            "slow_X_identified": self.slow_X_identified,
            "mode_conversion_verified": self.mode_conversion_verified,
            "xb_efficiency": None,
            "ebw_identity": "not_verified",
            "model": "cold perpendicular-X R/L/S accessibility diagnostic",
            "quantitative_boundary": (
                "R/L cutoff and UHR ordering only; X-B efficiency and EBW identity require "
                "a converged hot full-wave eigenmode/power decomposition"
            ),
        }
        if include_path:
            payload["path"] = [sample.to_dict() for sample in self.samples]
        return payload


def ordinary_cutoff_density_m3(frequency_Hz: float) -> float:
    omega = 2.0 * PI * frequency_Hz
    return EPS0_F_M * M_E_KG * omega**2 / E_CHARGE_C**2


def _plasma_frequency_squared(density_m3: float) -> float:
    return density_m3 * E_CHARGE_C**2 / (EPS0_F_M * M_E_KG)


def _cyclotron_frequency(B_T: float) -> float:
    return E_CHARGE_C * B_T / M_E_KG


def _sample_field_and_density(
    R_m: float,
    Z_m: float,
    t_s: float,
    Ip_A: float,
    magnetic: MagneticFieldProvider,
    density: DensityProvider,
) -> tuple[float, float]:
    field = magnetic.evaluate(R_m, Z_m, Ip_A, t_s)
    return field.magnitude_T, density.electron_density_m3(R_m, Z_m, t_s)


def _crossing(
    values: np.ndarray,
    points: list[tuple[float, float, float, float, float]],
    kind: str,
) -> ConversionPoint | None:
    for index in range(1, len(values)):
        previous = values[index - 1]
        current = values[index]
        if previous == 0.0 or current == 0.0 or previous * current < 0.0:
            denominator = abs(previous) + abs(current)
            weight = 0.5 if denominator <= 0.0 else abs(previous) / denominator
            p0 = points[index - 1]
            p1 = points[index]
            interpolated = tuple(a + weight * (b - a) for a, b in zip(p0, p1))
            return ConversionPoint(
                kind,
                index,
                interpolated[0],
                interpolated[1],
                interpolated[2],
                interpolated[3],
                interpolated[4],
                abs(float(current)),
            )
    return None


def _finite_crossing(
    values: np.ndarray,
    points: list[tuple[float, float, float, float, float]],
    kind: str,
) -> ConversionPoint | None:
    """Find a zero crossing while skipping cyclotron-singular samples."""

    for index in range(1, len(values)):
        previous = float(values[index - 1])
        current = float(values[index])
        if not (math.isfinite(previous) and math.isfinite(current)):
            continue
        if previous == 0.0 or current == 0.0 or previous * current < 0.0:
            denominator = abs(previous) + abs(current)
            weight = 0.5 if denominator <= 0.0 else abs(previous) / denominator
            p0 = points[index - 1]
            p1 = points[index]
            interpolated = tuple(a + weight * (b - a) for a, b in zip(p0, p1))
            return ConversionPoint(
                kind,
                index,
                interpolated[0],
                interpolated[1],
                interpolated[2],
                interpolated[3],
                interpolated[4],
                min(abs(previous), abs(current)),
            )
    return None


def analyze_direct_xb(
    source: RFSourceConfig,
    x_ray: ECRayTrace,
    magnetic: MagneticFieldProvider,
    density: DensityProvider,
    *,
    Ip_A: float = 0.0,
    t_s: float = 0.0,
) -> DirectXBResult:
    """Locate the direct X-B cold accessibility layers along an X-mode path.

    The diagnostic evaluates ``R=0`` and ``L=0`` cutoffs, ``S=0`` UHR, and
    ``N_X^2=R L/S``.  A negative ``N_X^2`` interval between the incident
    cutoff and UHR is the evanescent barrier.  This establishes accessibility
    geometry only; it cannot prove conversion into an EBW.
    """

    if not source.launch.mode.upper().startswith("X"):
        return DirectXBResult(
            source.name, source.frequency_Hz, False,
            "direct X-B analysis requires an incident X-mode source",
            None, None, None, None, None, False, False, False, (),
        )
    if len(x_ray.points) < 2:
        return DirectXBResult(
            source.name, source.frequency_Hz, False,
            "incident X ray has insufficient path samples",
            None, None, None, None, None, False, False, False, (),
        )

    points: list[tuple[float, float, float, float, float]] = []
    r_values: list[float] = []
    l_values: list[float] = []
    s_values: list[float] = []
    samples: list[XBranchSample] = []
    for point in x_ray.points:
        B, ne = _sample_field_and_density(point.R_m, point.Z_m, t_s, Ip_A, magnetic, density)
        points.append((point.path_length_m, point.R_m, point.Z_m, ne, B))
        try:
            stix = stix_parameters(source.frequency_Hz, B, ne)
            R, L, S = float(stix.R), float(stix.L), float(stix.S)
            n2 = R * L / S if abs(S) > 1.0e-14 else None
        except ColdPlasmaDispersionError:
            R = L = S = math.nan
            n2 = None
        if n2 is None or not math.isfinite(n2):
            branch = "singular_UHR_or_EC"
        elif n2 < 0.0:
            branch = "evanescent_X"
        elif abs(S) < 0.15 or n2 > 4.0:
            branch = "slow_X"
        else:
            branch = "fast_X"
        r_values.append(R)
        l_values.append(L)
        s_values.append(S)
        samples.append(
            XBranchSample(
                point.path_length_m, point.R_m, point.Z_m, ne, B,
                R, L, S, n2, branch,
            )
        )

    R_cutoff = _finite_crossing(np.asarray(r_values), points, "R_cutoff")
    L_cutoff = _finite_crossing(np.asarray(l_values), points, "L_cutoff")
    UHR = _finite_crossing(np.asarray(s_values), points, "UHR")
    cutoffs = [item for item in (R_cutoff, L_cutoff) if item is not None]
    incident = None
    if UHR is not None:
        upstream = [item for item in cutoffs if item.path_length_m <= UHR.path_length_m]
        if upstream:
            incident = max(upstream, key=lambda item: item.path_length_m)
    width = None if incident is None or UHR is None else UHR.path_length_m - incident.path_length_m
    barrier = False
    if incident is not None and UHR is not None:
        barrier = any(
            sample.branch == "evanescent_X"
            and incident.path_length_m <= sample.path_length_m <= UHR.path_length_m
            for sample in samples
        )
    fast = any(item.branch == "fast_X" for item in samples)
    slow = any(item.branch == "slow_X" for item in samples)
    available = bool(incident is not None and UHR is not None and barrier and fast)
    if available:
        reason = "incident R/L cutoff, evanescent X barrier and UHR are ordered along the path"
    elif UHR is None:
        reason = "UHR was not crossed along the supplied X-mode path"
    elif incident is None:
        reason = "no upstream R/L cutoff was found before the UHR"
    elif not barrier:
        reason = "cutoff and UHR were found but no sampled evanescent X interval separates them"
    else:
        reason = "the incident fast-X branch was not identified"
    return DirectXBResult(
        source.name,
        source.frequency_Hz,
        available,
        reason,
        R_cutoff,
        L_cutoff,
        UHR,
        None if incident is None else incident.kind,
        width,
        fast,
        slow,
        False,
        tuple(samples),
    )


def _crossing_or_near(
    values: np.ndarray,
    points: list[tuple[float, float, float, float, float]],
    kind: str,
    tolerance_fraction: float,
) -> ConversionPoint | None:
    crossing = _crossing(values, points, kind)
    if crossing is not None or len(values) == 0:
        return crossing
    candidates = np.flatnonzero(np.abs(values) <= tolerance_fraction)
    if len(candidates) == 0:
        return None
    # Select the first local minimum after entering the tolerance band.  This
    # is more accurate than its leading edge while still avoiding a later
    # revisit after cold-ray reflection at the cutoff.
    index = int(candidates[0])
    for candidate in candidates[1:]:
        candidate = int(candidate)
        if candidate != index + 1 or abs(values[candidate]) > abs(values[index]):
            break
        index = candidate
    point = points[index]
    return ConversionPoint(
        kind,
        index,
        point[0],
        point[1],
        point[2],
        point[3],
        point[4],
        abs(float(values[index])),
    )


def _density_scale_length(
    point: ConversionPoint,
    density: DensityProvider,
    t_s: float,
    difference_m: float,
) -> tuple[float, np.ndarray]:
    h = difference_m
    dn_dR = (
        density.electron_density_m3(point.R_m + h, point.Z_m, t_s)
        - density.electron_density_m3(point.R_m - h, point.Z_m, t_s)
    ) / (2.0 * h)
    dn_dZ = (
        density.electron_density_m3(point.R_m, point.Z_m + h, t_s)
        - density.electron_density_m3(point.R_m, point.Z_m - h, t_s)
    ) / (2.0 * h)
    gradient = np.asarray([dn_dR, dn_dZ])
    magnitude = float(np.linalg.norm(gradient))
    if magnitude <= 0.0:
        return math.inf, np.asarray([0.0, 0.0])
    return point.electron_density_m3 / magnitude, gradient / magnitude


def _build_ebw_path(
    start: ConversionPoint,
    source: RFSourceConfig,
    magnetic: MagneticFieldProvider,
    density: DensityProvider,
    geometry: GeometryConfig,
    config: EBWConfig,
    *,
    Ip_A: float,
    t_s: float,
) -> tuple[EBWPathPoint, ...]:
    position = np.asarray([start.R_m, start.Z_m], dtype=float)
    path: list[EBWPathPoint] = []
    length = 0.0
    omega = 2.0 * PI * source.frequency_Hz
    R_bounds = config.R_bounds_m or geometry.R_bounds_m
    Z_bounds = config.Z_bounds_m or geometry.Z_bounds_m
    for _ in range(int(config.ebw_max_length_m / config.ebw_path_step_m) + 1):
        R, Z = float(position[0]), float(position[1])
        if not (
            R_bounds[0] <= R <= R_bounds[1]
            and Z_bounds[0] <= Z <= Z_bounds[1]
        ):
            break
        B, ne = _sample_field_and_density(R, Z, t_s, Ip_A, magnetic, density)
        mismatch = abs(source.harmonic * _cyclotron_frequency(B) / omega - 1.0)
        path.append(EBWPathPoint(length, R, Z, B, ne, mismatch))
        if mismatch <= 0.01:
            break
        h = min(1.0e-3, 0.25 * config.ebw_path_step_m)
        dn_dR = (
            density.electron_density_m3(R + h, Z, t_s)
            - density.electron_density_m3(R - h, Z, t_s)
        ) / (2.0 * h)
        dn_dZ = (
            density.electron_density_m3(R, Z + h, t_s)
            - density.electron_density_m3(R, Z - h, t_s)
        ) / (2.0 * h)
        direction = np.asarray([dn_dR, dn_dZ], dtype=float)
        norm = float(np.linalg.norm(direction))
        if norm <= 0.0:
            break
        direction /= norm
        trial_plus = position + config.ebw_path_step_m * direction
        trial_minus = position - config.ebw_path_step_m * direction

        def trial_mismatch(trial: np.ndarray) -> float:
            b, _ = _sample_field_and_density(
                float(trial[0]), float(trial[1]), t_s, Ip_A, magnetic, density
            )
            return abs(source.harmonic * _cyclotron_frequency(b) / omega - 1.0)

        position = trial_plus if trial_mismatch(trial_plus) <= trial_mismatch(trial_minus) else trial_minus
        length += config.ebw_path_step_m
    return tuple(path)


def analyze_oxb(
    source: RFSourceConfig,
    o_ray: ECRayTrace,
    magnetic: MagneticFieldProvider,
    density: DensityProvider,
    geometry: GeometryConfig,
    config: EBWConfig,
    *,
    Ip_A: float = 0.0,
    t_s: float = 0.0,
) -> OXBResult:
    """Analyze O-X-B availability along an incident O-mode ray."""

    n_cutoff = ordinary_cutoff_density_m3(source.frequency_Hz)
    if not source.launch.mode.upper().startswith("O"):
        return OXBResult(
            source.name, source.frequency_Hz, source.harmonic, False,
            "O-X-B analysis requires an incident O-mode source", n_cutoff, 0.0,
            False, None, None, None, 0.0, None, None, None, None, None, None,
            False, False, None, (),
        )
    if len(o_ray.points) < 2:
        return OXBResult(
            source.name, source.frequency_Hz, source.harmonic, False,
            "incident O ray has insufficient path samples", n_cutoff, 0.0,
            False, None, None, None, 0.0, None, None, None, None, None, None,
            False, False, None, (),
        )

    samples: list[tuple[float, float, float, float, float]] = []
    o_values: list[float] = []
    uhr_values: list[float] = []
    omega = 2.0 * PI * source.frequency_Hz
    maximum_density = 0.0
    for point in o_ray.points:
        B, ne = _sample_field_and_density(point.R_m, point.Z_m, t_s, Ip_A, magnetic, density)
        samples.append((point.path_length_m, point.R_m, point.Z_m, ne, B))
        maximum_density = max(maximum_density, ne)
        wp2 = _plasma_frequency_squared(ne)
        wc2 = _cyclotron_frequency(B) ** 2
        o_values.append(wp2 / omega**2 - 1.0)
        uhr_values.append((wp2 + wc2) / omega**2 - 1.0)

    available_peak_density = float(
        getattr(density, "peak_density_m3", maximum_density)
        + getattr(density, "floor_density_m3", 0.0)
    )
    O_cutoff = _crossing_or_near(
        np.asarray(o_values), samples, "O_cutoff", config.cutoff_tolerance_fraction
    )
    if O_cutoff is None:
        return OXBResult(
            source.name, source.frequency_Hz, source.harmonic, False,
            "O-mode cutoff was not crossed along the incident ray", n_cutoff,
            available_peak_density, available_peak_density >= n_cutoff, None, None,
            None,
            0.0, None, None, None, None, None, None, False, False, None, (),
        )

    # The slow-X wave propagates back toward the low-density side and reaches
    # the UHR before its cutoff. Search the incident samples before O cutoff.
    UHR = _crossing_or_near(
        np.asarray(uhr_values[: O_cutoff.path_index + 1]),
        samples[: O_cutoff.path_index + 1],
        "UHR",
        config.uhr_tolerance_fraction,
    )
    L_n, _gradient_direction = _density_scale_length(
        O_cutoff, density, t_s, max(1.0e-4, 0.1 * config.ebw_path_step_m)
    )
    ray_point = o_ray.points[min(O_cutoff.path_index, len(o_ray.points) - 1)]
    n_parallel = ray_point.n_parallel
    Y = _cyclotron_frequency(O_cutoff.B_T) / omega
    optimal = math.sqrt(max(0.0, Y / (1.0 + Y)))
    k0 = omega / C_LIGHT_M_S
    efficiency = math.exp(-PI * k0 * L_n * (n_parallel - optimal) ** 2) if math.isfinite(L_n) else 0.0
    efficiency = min(1.0, max(0.0, efficiency))
    separation = None
    if UHR is not None:
        separation = math.hypot(O_cutoff.R_m - UHR.R_m, O_cutoff.Z_m - UHR.Z_m)
    available = UHR is not None and available_peak_density >= n_cutoff and efficiency >= config.minimum_ox_efficiency
    reason = (
        "O cutoff and UHR were found; X-B remains a full-wave/tunnelling candidate"
        if available
        else "O-X-B window incomplete or O-X efficiency below threshold"
    )
    ebw_path = (
        _build_ebw_path(UHR, source, magnetic, density, geometry, config, Ip_A=Ip_A, t_s=t_s)
        if available and UHR is not None
        else ()
    )
    resonance = None
    if ebw_path and ebw_path[-1].resonance_mismatch <= 0.05:
        final = ebw_path[-1]
        resonance = ConversionPoint(
            kind="EBW_EC_resonance_candidate",
            path_index=len(ebw_path) - 1,
            path_length_m=final.path_length_m,
            R_m=final.R_m,
            Z_m=final.Z_m,
            electron_density_m3=final.electron_density_m3,
            B_T=final.B_T,
            residual=final.resonance_mismatch,
        )
    ebw_resonance_geometric_candidate = bool(
        available
        and resonance is not None
    )
    ebwcd_candidate = bool(
        ebw_resonance_geometric_candidate
        and abs(n_parallel) >= config.minimum_abs_n_parallel_for_cd
    )
    current_drive_direction_sign = (
        (1 if n_parallel > 0.0 else -1)
        if ebwcd_candidate and n_parallel != 0.0
        else None
    )
    return OXBResult(
        source_name=source.name,
        frequency_Hz=source.frequency_Hz,
        harmonic=source.harmonic,
        available=available,
        reason=reason,
        O_cutoff_density_m3=n_cutoff,
        maximum_density_m3=available_peak_density,
        overdense=available_peak_density >= n_cutoff,
        O_cutoff=O_cutoff,
        UHR=UHR,
        resonance=resonance,
        ox_efficiency_estimate=efficiency,
        xb_efficiency_estimate=None,
        combined_efficiency_estimate=None,
        n_parallel_at_O_cutoff=n_parallel,
        optimal_n_parallel=optimal,
        density_scale_length_m=L_n,
        conversion_separation_m=separation,
        ebw_resonance_geometric_candidate=ebw_resonance_geometric_candidate,
        ebwcd_candidate=ebwcd_candidate,
        current_drive_direction_sign=current_drive_direction_sign,
        ebw_path=ebw_path,
    )
