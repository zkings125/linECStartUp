"""Reduced, magnetic-surface-free GENRAY-like EC diffusion generator.

The implementation deliberately covers a smaller problem than GENRAY.  It
combines the local cold-Stix Hamiltonian ray already implemented in linEC with
a relativistic cyclotron-resonance kernel, reduced power depletion along the
ray, and an initial-Maxwellian power-moment normalization of a diagonal 0D-2V
``D_ql``.  It does not use GS/EFIT surfaces and is suitable for sensitivity
studies before CFS formation; it is not a hot dielectric or an adjoint ECCD
solver.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from threading import Lock
from typing import Any

import numpy as np
from scipy.constants import epsilon_0

from ..config import GenrayLikeConfig, RFSourceConfig, SimulationConfig
from ..constants import E_CHARGE_C, M_E_KG, PI
from .cold_plasma import C_LIGHT_M_S
from .ec_raytrace import ECRayTrace, HamiltonianECRayTracer, UniformElectronDensity
from .fokker_planck import QuasilinearDiffusion2V, RelativisticMomentumGrid, VelocityGrid
from .magnetic import make_magnetic_field
from .rf import ECRHSource


@dataclass(frozen=True)
class GenrayLikeDiagnostics:
    source_name: str
    ray_status: str
    ray_sample_count: int
    ray_path_length_m: float
    target_absorbed_power_W: float
    predicted_absorption_fraction: float
    unscaled_power_moment_W_per_unit_D: float
    normalization_D_m2_s3: float
    resonance_kernel_peak: float
    resonance_velocity_coverage_fraction: float
    minimum_wkb_scale_ratio: float | None
    adjoint_normalized_efficiency: float
    adjoint_estimated_parallel_current_A: float
    model: str = "reduced_genray_like_relativistic_resonance"
    equilibrium_model: str = "none; pre-CFS MagneticFieldProvider only"

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class GenrayLikeResult:
    diffusion: QuasilinearDiffusion2V
    diagnostics: tuple[GenrayLikeDiagnostics, ...]
    reference_absorbed_power_W: float


def _maxwellian(
    grid: VelocityGrid | RelativisticMomentumGrid,
    density_m3: float,
    temperature_eV: float,
) -> np.ndarray:
    parallel, perpendicular = grid.mesh()
    thermal_variance = temperature_eV * E_CHARGE_C / M_E_KG
    if isinstance(grid, RelativisticMomentumGrid):
        up, ut = grid.mesh_u()
        gamma = np.sqrt(1.0 + up**2 + ut**2)
        theta = temperature_eV * E_CHARGE_C / (M_E_KG * C_LIGHT_M_S**2)
        exponent = -(gamma - 1.0) / theta
    else:
        exponent = -(parallel**2 + perpendicular**2) / (2.0 * thermal_variance)
    exponent -= float(np.max(exponent))
    distribution = np.exp(np.maximum(exponent, -745.0))
    norm = float(np.sum(distribution * grid.phase_volume_weights_m3_s3))
    if norm <= 0.0:
        raise ValueError(
            "velocity grid does not resolve the Maxwellian used for GENRAY-like normalization"
        )
    return distribution * density_m3 / norm


def _power_moment_per_unit_D(
    grid: VelocityGrid | RelativisticMomentumGrid,
    D_parallel_shape: np.ndarray,
    D_perp_shape: np.ndarray,
    D_cross_shape: np.ndarray,
    density_m3: float,
    temperature_eV: float,
    volume_m3: float,
) -> float:
    """Return RF energy gain [W] for a unit-amplitude diffusion shape.

    For a Maxwellian and ``partial_t f = div_v(D grad_v f)``, integration by
    parts gives ``P/V = -integral m v dot D grad(f) d3v``.  Using the analytic
    Maxwellian gradient avoids a noisy finite difference in the normalization.
    """

    parallel, perpendicular = grid.mesh()
    distribution = _maxwellian(grid, density_m3, temperature_eV)
    temperature_J = temperature_eV * E_CHARGE_C
    quadratic = (
        parallel**2 * D_parallel_shape
        + 2.0 * parallel * perpendicular * D_cross_shape
        + perpendicular**2 * D_perp_shape
    )
    if isinstance(grid, RelativisticMomentumGrid):
        up, ut = grid.mesh_u()
        gamma_factor = np.sqrt(1.0 + up**2 + ut**2) ** 3
    else:
        gamma_factor = 1.0
    integrand = gamma_factor * quadratic * distribution
    density_rate = (
        M_E_KG**2
        / temperature_J
        * float(np.sum(integrand * grid.phase_volume_weights_m3_s3))
    )
    return density_rate * volume_m3


def _segment_lengths(trace: ECRayTrace, fallback_m: float) -> np.ndarray:
    count = len(trace.points)
    if count == 0:
        return np.empty(0)
    if count == 1:
        return np.asarray([fallback_m], dtype=float)
    s = np.asarray([point.path_length_m for point in trace.points], dtype=float)
    ds = np.diff(s)
    if np.any(ds <= 0.0):
        raise ValueError("GENRAY-like ray path length must be strictly increasing")
    return np.concatenate([ds, ds[-1:]])


def build_genray_like_diffusion(
    trace: ECRayTrace,
    source: RFSourceConfig,
    grid: VelocityGrid,
    settings: GenrayLikeConfig,
    *,
    density_m3: float,
    temperature_eV: float,
    volume_m3: float,
    major_radius_m: float,
    target_absorbed_power_W: float,
    fallback_step_m: float = 2.0e-3,
) -> tuple[QuasilinearDiffusion2V, GenrayLikeDiagnostics]:
    """Map one EC ray to a power-normalized diagonal 0D-2V diffusion tensor."""

    if not trace.points:
        raise ValueError(
            f"GENRAY-like source {source.name!r} produced no ray samples ({trace.status}: {trace.reason})"
        )
    if min(density_m3, temperature_eV, volume_m3, major_radius_m) <= 0.0:
        raise ValueError("density, temperature, volume and major radius must be positive")
    if target_absorbed_power_W < 0.0:
        raise ValueError("target_absorbed_power_W must be non-negative")

    parallel, perpendicular = grid.mesh()
    speed_squared = parallel**2 + perpendicular**2
    subluminal = speed_squared < C_LIGHT_M_S**2
    gamma = np.full(grid.shape, np.inf)
    gamma[subluminal] = 1.0 / np.sqrt(1.0 - speed_squared[subluminal] / C_LIGHT_M_S**2)
    omega = 2.0 * PI * source.frequency_Hz
    sigma_omega = settings.resonance_width_fraction * omega
    ds_values = _segment_lengths(trace, fallback_step_m)
    shape_parallel = np.zeros(grid.shape)
    shape_perp = np.zeros(grid.shape)
    remaining_fraction = 1.0
    peak = 0.0
    covered = np.zeros(grid.shape, dtype=bool)

    for point, ds_m in zip(trace.points, ds_values):
        k_parallel = point.n_parallel * omega / C_LIGHT_M_S
        cyclotron = E_CHARGE_C * abs(point.B_T) / M_E_KG
        mismatch = omega - k_parallel * parallel - source.harmonic * cyclotron / gamma
        kernel = np.exp(-0.5 * (mismatch / sigma_omega) ** 2)
        kernel[~subluminal] = 0.0
        local_peak = float(np.max(kernel))
        peak = max(peak, local_peak)
        covered |= np.abs(mismatch) <= 2.0 * sigma_omega

        density_factor = (
            max(point.electron_density_m3, 1.0) / settings.density_reference_m3
        ) ** settings.density_exponent
        temperature_factor = (
            temperature_eV / settings.temperature_reference_eV
        ) ** settings.temperature_exponent
        mode_factor = 1.0 if source.launch.mode.upper().startswith("X") else 0.7
        harmonic_factor = 1.0 / max(float(source.harmonic), 1.0)
        alpha = (
            settings.peak_absorption_coefficient_m_inv
            * local_peak
            * density_factor
            * temperature_factor
            * mode_factor
            * harmonic_factor
        )
        absorbed_fraction = remaining_fraction * (1.0 - math.exp(-2.0 * alpha * ds_m))
        remaining_fraction = max(0.0, remaining_fraction - absorbed_fraction)
        if absorbed_fraction <= 0.0:
            continue

        n_squared = point.n_parallel**2 + point.n_perpendicular**2
        parallel_fraction = max(
            settings.parallel_diffusion_floor,
            point.n_parallel**2 / max(n_squared, 1.0e-30),
        )
        parallel_fraction = min(parallel_fraction, 1.0)
        shape_parallel += absorbed_fraction * parallel_fraction * kernel
        shape_perp += absorbed_fraction * (1.0 - parallel_fraction) * kernel

    predicted_absorption = 1.0 - remaining_fraction
    if peak < settings.minimum_kernel_peak or not np.any(shape_parallel + shape_perp > 0.0):
        raise ValueError(
            f"GENRAY-like source {source.name!r} has no resolved cyclotron resonance on the velocity grid; "
            "increase velocity bounds/resolution or inspect the ray and harmonic"
        )
    total_shape_peak = float(np.max(shape_parallel + shape_perp))
    shape_parallel /= total_shape_peak
    shape_perp /= total_shape_peak
    shape_cross = settings.cross_diffusion_fraction * np.sqrt(
        np.maximum(shape_parallel, 0.0) * np.maximum(shape_perp, 0.0)
    )
    unit_power = _power_moment_per_unit_D(
        grid,
        shape_parallel,
        shape_perp,
        shape_cross,
        density_m3,
        temperature_eV,
        volume_m3,
    )
    if not np.isfinite(unit_power) or unit_power <= 0.0:
        raise ValueError("GENRAY-like D_ql power moment is non-positive")
    amplitude = target_absorbed_power_W / unit_power
    cross = amplitude * shape_cross
    diffusion = QuasilinearDiffusion2V(
        amplitude * shape_parallel,
        amplitude * shape_perp,
        (
            f"genray_like_internal:{source.name};P_ref={target_absorbed_power_W:g}W;"
            f"ray_status={trace.status}"
        ),
        cross,
    ).validated(grid)
    wkb_values = [point.wkb_scale_ratio for point in trace.points if np.isfinite(point.wkb_scale_ratio)]
    absorption_weights = np.asarray(
        [max(point.electron_density_m3, 0.0) for point in trace.points], dtype=float
    )
    n_parallel_values = np.asarray([point.n_parallel for point in trace.points], dtype=float)
    if float(np.sum(absorption_weights)) <= 0.0:
        mean_n_parallel = float(np.mean(n_parallel_values))
    else:
        mean_n_parallel = float(np.average(n_parallel_values, weights=absorption_weights))
    # zeta = e^3 n_e R I / (epsilon_0^2 P T_e).  This conventional normalized
    # efficiency is retained as a separate adjoint-like diagnostic and never
    # used to tune the velocity-space D_ql amplitude.
    normalized_denominator = E_CHARGE_C**3 * density_m3 * major_radius_m
    adjoint_magnitude = (
        settings.adjoint_normalized_efficiency
        * epsilon_0**2
        * target_absorbed_power_W
        * temperature_eV
        * E_CHARGE_C
        / max(normalized_denominator, 1.0e-300)
    )
    adjoint_current = -math.copysign(adjoint_magnitude, mean_n_parallel or 1.0)
    diagnostics = GenrayLikeDiagnostics(
        source_name=source.name,
        ray_status=trace.status,
        ray_sample_count=len(trace.points),
        ray_path_length_m=trace.path_length_m,
        target_absorbed_power_W=float(target_absorbed_power_W),
        predicted_absorption_fraction=float(predicted_absorption),
        unscaled_power_moment_W_per_unit_D=float(unit_power),
        normalization_D_m2_s3=float(amplitude),
        resonance_kernel_peak=float(peak),
        resonance_velocity_coverage_fraction=float(np.count_nonzero(covered) / covered.size),
        minimum_wkb_scale_ratio=min(wkb_values) if wkb_values else None,
        adjoint_normalized_efficiency=settings.adjoint_normalized_efficiency,
        adjoint_estimated_parallel_current_A=float(adjoint_current),
    )
    return diffusion, diagnostics


_REFERENCE_CACHE: dict[str, GenrayLikeResult] = {}
_CACHE_LOCK = Lock()


def _cache_key(config: SimulationConfig, grid: VelocityGrid | RelativisticMomentumGrid) -> str:
    coordinates = (
        (grid.u_parallel.tolist(), grid.u_perp.tolist())
        if isinstance(grid, RelativisticMomentumGrid)
        else (grid.v_parallel_m_s.tolist(), grid.v_perp_m_s.tolist())
    )
    return "|".join((config.model_dump_json(), repr(coordinates[0]), repr(coordinates[1])))


def build_internal_genray_like_reference(
    config: SimulationConfig,
    grid: VelocityGrid | RelativisticMomentumGrid,
) -> GenrayLikeResult:
    """Trace enabled EC sources and cache a reference-power D_ql mixture."""

    key = _cache_key(config, grid)
    if config.genray_like.cache_reference_diffusion:
        with _CACHE_LOCK:
            cached = _REFERENCE_CACHE.get(key)
        if cached is not None:
            return cached

    sources = [source for source in ECRHSource(config.rf, config.geometry).sources if source.enabled]
    selected = config.genray_like.source_name
    if selected is not None:
        sources = [source for source in sources if source.name == selected]
    if not sources:
        raise ValueError("genray_like_internal requires at least one enabled RF source")

    nominal = np.asarray(
        [max(source.power_W * source.absorption_fraction, 0.0) for source in sources],
        dtype=float,
    )
    if float(np.sum(nominal)) <= 0.0:
        nominal[:] = 1.0
    fractions = nominal / np.sum(nominal)
    reference_power = float(config.kinetic.feedback_reference_absorbed_power_W)
    density = float(config.kinetic.feedback_reference_density_m3 or config.initial.ne_m3)
    temperature = float(
        config.kinetic.collision_temperature_eV or max(config.initial.Te_eV, 1.0e-3)
    )
    tracer = HamiltonianECRayTracer(
        make_magnetic_field(config.geometry),
        UniformElectronDensity(density),
        config.geometry,
        config.ray_tracing,
        ion_mass_kg=config.gas.ion_mass_kg,
    )
    total_parallel = np.zeros(grid.shape)
    total_perp = np.zeros(grid.shape)
    total_cross = np.zeros(grid.shape)
    diagnostics: list[GenrayLikeDiagnostics] = []
    for source, fraction in zip(sources, fractions):
        trace = tracer.trace(
            source,
            Ip_A=float(config.initial.Ip_A),
            t_s=float(config.numerics.t_start_s),
        )
        diffusion, diagnostic = build_genray_like_diffusion(
            trace,
            source,
            grid,
            config.genray_like,
            density_m3=density,
            temperature_eV=temperature,
            volume_m3=config.geometry.resolved_volume_m3,
            major_radius_m=config.geometry.R0_m,
            target_absorbed_power_W=reference_power * float(fraction),
            fallback_step_m=config.ray_tracing.step_m,
        )
        total_parallel += diffusion.D_parallel_m2_s3
        total_perp += diffusion.D_perp_m2_s3
        total_cross += diffusion.D_cross_m2_s3
        diagnostics.append(diagnostic)

    result = GenrayLikeResult(
        QuasilinearDiffusion2V(
            total_parallel,
            total_perp,
            "genray_like_internal:combined;power_normalized_initial_Maxwellian",
            total_cross,
        ).validated(grid),
        tuple(diagnostics),
        reference_power,
    )
    if config.genray_like.cache_reference_diffusion:
        with _CACHE_LOCK:
            return _REFERENCE_CACHE.setdefault(key, result)
    return result
