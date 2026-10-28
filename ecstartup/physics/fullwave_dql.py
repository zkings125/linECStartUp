"""Verified full-wave field contract and resonant Dql projection.

This module does not call a cold-ray result a full-wave solution.  It accepts
complex polarization-resolved fields from an external FEM/FDTD solver,
requires convergence/mesh-verification metadata and a closed port-power
budget, interpolates the fields onto an open field line, and builds a
positive-semidefinite resonant quasilinear diffusion tensor.

The velocity-space projection is the fundamental cyclotron Kennel-Engelmann
outer-product form with Gaussian resonance broadening.  Higher-harmonic FLR
coupling must be supplied by a separately validated projection backend.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.special import jv

from ..constants import E_CHARGE_C, M_E_KG, PI
from ..numerics import trapezoid
from .fokker_planck import C_LIGHT_M_S, RelativisticMomentumGrid, VelocityGrid
from .spatial_fokker_planck import OpenFieldLineGrid, SpatialQuasilinearDiffusion


@dataclass(frozen=True)
class FullWavePowerBudget:
    incident_W: float
    reflected_O_W: float
    reflected_X_W: float
    transmitted_O_W: float
    transmitted_X_W: float
    ebw_out_W: float
    absorbed_W: float

    @property
    def accounted_W(self) -> float:
        return (
            self.reflected_O_W
            + self.reflected_X_W
            + self.transmitted_O_W
            + self.transmitted_X_W
            + self.ebw_out_W
            + self.absorbed_W
        )

    @property
    def balance_relative(self) -> float:
        return abs(self.accounted_W - self.incident_W) / max(abs(self.incident_W), 1.0e-30)

    @property
    def oxb_conversion_efficiency(self) -> float:
        return self.ebw_out_W / max(self.incident_W, 1.0e-30)

    def validated(self, tolerance: float = 1.0e-3) -> "FullWavePowerBudget":
        values = np.asarray(list(self.__dict__.values()), dtype=float)
        if np.any(~np.isfinite(values)) or np.any(values < 0.0) or self.incident_W <= 0.0:
            raise ValueError("full-wave powers must be finite/non-negative and incident power positive")
        if self.balance_relative > tolerance:
            raise ValueError(
                f"full-wave power balance {self.balance_relative:.3e} exceeds {tolerance:.3e}"
            )
        return self


@dataclass(frozen=True)
class FullWaveFieldMap2D:
    R_m: np.ndarray
    Z_m: np.ndarray
    E_plus_V_m: np.ndarray
    E_minus_V_m: np.ndarray
    E_parallel_V_m: np.ndarray
    k_parallel_m_inv: np.ndarray
    B_T: np.ndarray
    absorbed_power_density_W_m3: np.ndarray
    frequency_Hz: float
    solver_name: str
    solver_version: str
    converged: bool
    mesh_verified: bool
    power_budget: FullWavePowerBudget
    source_path: Path | None = None
    wave_branch: str = "unresolved_fullwave"
    mode_decomposition_verified: bool = False

    def validated(self, *, power_tolerance: float = 1.0e-3) -> "FullWaveFieldMap2D":
        R = np.asarray(self.R_m, dtype=float)
        Z = np.asarray(self.Z_m, dtype=float)
        expected = (len(R), len(Z))
        arrays = (
            np.asarray(self.E_plus_V_m),
            np.asarray(self.E_minus_V_m),
            np.asarray(self.E_parallel_V_m),
            np.asarray(self.k_parallel_m_inv, dtype=float),
            np.asarray(self.B_T, dtype=float),
            np.asarray(self.absorbed_power_density_W_m3, dtype=float),
        )
        if R.ndim != 1 or Z.ndim != 1 or len(R) < 2 or len(Z) < 2:
            raise ValueError("full-wave R/Z coordinates must be one-dimensional with at least two points")
        if np.any(np.diff(R) <= 0.0) or np.any(np.diff(Z) <= 0.0):
            raise ValueError("full-wave R/Z coordinates must increase")
        if any(array.shape != expected for array in arrays):
            raise ValueError(f"full-wave field arrays must have shape {expected}")
        if any(np.any(~np.isfinite(array)) for array in arrays):
            raise ValueError("full-wave field map contains non-finite values")
        if np.any(arrays[4] <= 0.0) or np.any(arrays[5] < 0.0):
            raise ValueError("full-wave B must be positive and absorbed power density non-negative")
        if self.frequency_Hz <= 0.0 or not self.solver_name or not self.solver_version:
            raise ValueError("full-wave frequency and solver identity are required")
        if not self.converged or not self.mesh_verified:
            raise ValueError("quantitative full-wave use requires convergence and mesh verification")
        self.power_budget.validated(power_tolerance)
        allowed_branches = {"unresolved_fullwave", "electromagnetic_ec", "ebw", "mixed"}
        if self.wave_branch not in allowed_branches:
            raise ValueError(f"unsupported full-wave branch label: {self.wave_branch!r}")
        if self.mode_decomposition_verified and self.wave_branch == "unresolved_fullwave":
            raise ValueError("verified mode decomposition requires a resolved wave_branch")
        if self.wave_branch in {"ebw", "mixed"} and self.mode_decomposition_verified:
            if self.power_budget.ebw_out_W <= 0.0:
                raise ValueError("verified EBW/mixed branch requires positive ebw_out_W")
        if self.local_absorption_balance_relative > power_tolerance:
            raise ValueError(
                "integrated absorbed-power density differs from the port budget by "
                f"{self.local_absorption_balance_relative:.3e}, exceeding {power_tolerance:.3e}"
            )
        return self

    def axisymmetric_absorbed_power_W(self) -> float:
        integrand = 2.0 * PI * self.R_m[:, None] * self.absorbed_power_density_W_m3
        return float(trapezoid(trapezoid(integrand, self.Z_m, axis=1), self.R_m))

    @property
    def local_absorption_balance_relative(self) -> float:
        return abs(self.axisymmetric_absorbed_power_W() - self.power_budget.absorbed_W) / max(
            abs(self.power_budget.absorbed_W), 1.0e-30
        )


def load_fullwave_npz(path: str | Path, *, power_tolerance: float = 1.0e-3) -> FullWaveFieldMap2D:
    source = Path(path)
    with np.load(source, allow_pickle=False) as data:
        required = {
            "R_m", "Z_m", "E_plus_real_V_m", "E_plus_imag_V_m",
            "E_minus_real_V_m", "E_minus_imag_V_m",
            "E_parallel_real_V_m", "E_parallel_imag_V_m",
            "k_parallel_m_inv", "B_T", "absorbed_power_density_W_m3", "metadata_json",
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"full-wave NPZ lacks {sorted(missing)}")
        metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
        budget = FullWavePowerBudget(**metadata["power_budget"])
        result = FullWaveFieldMap2D(
            np.asarray(data["R_m"], dtype=float),
            np.asarray(data["Z_m"], dtype=float),
            np.asarray(data["E_plus_real_V_m"], dtype=float) + 1j * np.asarray(data["E_plus_imag_V_m"], dtype=float),
            np.asarray(data["E_minus_real_V_m"], dtype=float) + 1j * np.asarray(data["E_minus_imag_V_m"], dtype=float),
            np.asarray(data["E_parallel_real_V_m"], dtype=float) + 1j * np.asarray(data["E_parallel_imag_V_m"], dtype=float),
            np.asarray(data["k_parallel_m_inv"], dtype=float),
            np.asarray(data["B_T"], dtype=float),
            np.asarray(data["absorbed_power_density_W_m3"], dtype=float),
            float(metadata["frequency_Hz"]),
            str(metadata["solver_name"]),
            str(metadata["solver_version"]),
            bool(metadata["converged"]),
            bool(metadata["mesh_verified"]),
            budget,
            source.resolve(),
            str(metadata.get("wave_branch", "unresolved_fullwave")),
            bool(metadata.get("mode_decomposition_verified", False)),
        )
    return result.validated(power_tolerance=power_tolerance)


def load_open_fieldline_npz(path: str | Path) -> OpenFieldLineGrid:
    """Load the physical open-field-line grid used by the 1D-2V solver.

    The file is deliberately independent of flux coordinates and therefore
    remains valid before CFS formation.  Its coordinates must come from a
    traced magnetic field line or another explicitly documented geometry
    source; this loader does not infer a Grad-Shafranov surface.
    """

    source = Path(path)
    with np.load(source, allow_pickle=False) as data:
        required = {"s_m", "R_m", "Z_m", "B_T", "cross_section_area_m2"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"open-field-line NPZ lacks {sorted(missing)}")
        return OpenFieldLineGrid(
            s_m=np.asarray(data["s_m"], dtype=float),
            R_m=np.asarray(data["R_m"], dtype=float),
            Z_m=np.asarray(data["Z_m"], dtype=float),
            B_T=np.asarray(data["B_T"], dtype=float),
            cross_section_area_m2=np.asarray(data["cross_section_area_m2"], dtype=float),
        )


def save_fullwave_dql_npz(
    path: str | Path,
    projection: "FullWaveDqlProjection",
    spatial: OpenFieldLineGrid,
    velocity: VelocityGrid | RelativisticMomentumGrid,
    field_map: FullWaveFieldMap2D,
) -> Path:
    """Save a provenance-complete spatial Dql file for the 1D-2V backend."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema": "ecstartup.fullwave_spatial_dql.v1",
        "source_fullwave_file": str(field_map.source_path) if field_map.source_path else None,
        "solver_name": field_map.solver_name,
        "solver_version": field_map.solver_version,
        "frequency_Hz": field_map.frequency_Hz,
        "converged": field_map.converged,
        "mesh_verified": field_map.mesh_verified,
        "power_budget": field_map.power_budget.__dict__,
        "fullwave_power_balance_relative": projection.fullwave_power_balance_relative,
        "local_absorption_balance_relative": field_map.local_absorption_balance_relative,
        "oxb_conversion_efficiency": projection.oxb_conversion_efficiency,
        "coordinate_basis": (
            "open_field_line_s_relativistic_momentum"
            if isinstance(velocity, RelativisticMomentumGrid)
            else "open_field_line_s_velocity"
        ),
        "requires_closed_flux_surfaces": False,
        "pre_cfs_compatible": True,
        "power_normalized": True,
        "wave_branch": field_map.wave_branch,
        "mode_decomposition_verified": field_map.mode_decomposition_verified,
        "conversion_evidence": (
            "verified_fullwave"
            if field_map.wave_branch in {"ebw", "mixed"} and field_map.mode_decomposition_verified
            else "fullwave_fields_without_verified_mode_decomposition"
        ),
        "quantitative_eligible": bool(
            field_map.mode_decomposition_verified
            and field_map.wave_branch in {"electromagnetic_ec", "ebw", "mixed"}
        ),
        "projection_model": projection.projection_model,
        "target_absorbed_power_W": projection.target_absorbed_power_W,
        "coordinate_system": (
            "relativistic_momentum" if isinstance(velocity, RelativisticMomentumGrid) else "velocity"
        ),
        "pre_cfs_equilibrium_model": "none; physical open-field-line coordinates",
    }
    arrays: dict[str, np.ndarray | str] = {
        "D_parallel_m2_s3": projection.diffusion.D_parallel_m2_s3,
        "D_perp_m2_s3": projection.diffusion.D_perp_m2_s3,
        "D_cross_m2_s3": projection.diffusion.D_cross_m2_s3,
        "s_m": spatial.s_m,
        "R_m": spatial.R_m,
        "Z_m": spatial.Z_m,
        "B_T": spatial.B_T,
        "cross_section_area_m2": spatial.cross_section_area_m2,
        "time_s": np.asarray(projection.diffusion.time_s),
        "metadata_json": json.dumps(metadata, sort_keys=True),
    }
    if isinstance(velocity, RelativisticMomentumGrid):
        arrays["u_parallel"] = velocity.u_parallel
        arrays["u_perp"] = velocity.u_perp
    else:
        arrays["v_parallel_m_s"] = velocity.v_parallel_m_s
        arrays["v_perp_m_s"] = velocity.v_perp_m_s
    np.savez_compressed(output, **arrays)
    return output.resolve()


def load_spatial_fullwave_dql_npz(
    path: str | Path,
    spatial: OpenFieldLineGrid,
    velocity: VelocityGrid | RelativisticMomentumGrid,
    *,
    coordinate_rtol: float = 1.0e-10,
) -> SpatialQuasilinearDiffusion:
    """Load a projected full-wave Dql and reject every grid/provenance mismatch."""

    source = Path(path)
    with np.load(source, allow_pickle=False) as data:
        required = {
            "D_parallel_m2_s3",
            "D_perp_m2_s3",
            "D_cross_m2_s3",
            "s_m",
            "R_m",
            "Z_m",
            "B_T",
            "cross_section_area_m2",
            "time_s",
            "metadata_json",
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"spatial full-wave Dql NPZ lacks {sorted(missing)}")
        metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
        if metadata.get("schema") != "ecstartup.fullwave_spatial_dql.v1":
            raise ValueError("unsupported spatial full-wave Dql schema")
        spatial_coordinates = {
            "s_m": spatial.s_m,
            "R_m": spatial.R_m,
            "Z_m": spatial.Z_m,
            "B_T": spatial.B_T,
            "cross_section_area_m2": spatial.cross_section_area_m2,
        }
        for name, expected in spatial_coordinates.items():
            actual = np.asarray(data[name], dtype=float)
            if actual.shape != expected.shape or not np.allclose(
                actual, expected, rtol=coordinate_rtol, atol=1.0e-12
            ):
                raise ValueError(f"spatial full-wave Dql {name} does not match the solver grid")
        if isinstance(velocity, RelativisticMomentumGrid):
            expected_coordinates = {"u_parallel": velocity.u_parallel, "u_perp": velocity.u_perp}
            expected_system = "relativistic_momentum"
        else:
            expected_coordinates = {
                "v_parallel_m_s": velocity.v_parallel_m_s,
                "v_perp_m_s": velocity.v_perp_m_s,
            }
            expected_system = "velocity"
        if metadata.get("coordinate_system") != expected_system:
            raise ValueError("spatial full-wave Dql coordinate system does not match the kinetic grid")
        for name, expected in expected_coordinates.items():
            if name not in data.files:
                raise ValueError(f"spatial full-wave Dql lacks {name}")
            actual = np.asarray(data[name], dtype=float)
            if actual.shape != expected.shape or not np.allclose(
                actual, expected, rtol=coordinate_rtol, atol=1.0e-12
            ):
                raise ValueError(f"spatial full-wave Dql {name} does not match the kinetic grid")
        diffusion = SpatialQuasilinearDiffusion(
            D_parallel_m2_s3=np.asarray(data["D_parallel_m2_s3"], dtype=float),
            D_perp_m2_s3=np.asarray(data["D_perp_m2_s3"], dtype=float),
            D_cross_m2_s3=np.asarray(data["D_cross_m2_s3"], dtype=float),
            time_s=float(np.asarray(data["time_s"]).item()),
            source=(
                f"fullwave:{metadata['solver_name']}:{metadata['solver_version']};"
                f"{metadata['projection_model']}"
            ),
            target_absorbed_power_W=float(metadata["target_absorbed_power_W"]),
        )
    return diffusion.validated(spatial, velocity)


def _interpolate(map_: FullWaveFieldMap2D, values: np.ndarray, spatial: OpenFieldLineGrid) -> np.ndarray:
    points = np.column_stack((spatial.R_m, spatial.Z_m))
    if np.iscomplexobj(values):
        real = RegularGridInterpolator((map_.R_m, map_.Z_m), values.real, bounds_error=True)(points)
        imag = RegularGridInterpolator((map_.R_m, map_.Z_m), values.imag, bounds_error=True)(points)
        return real + 1j * imag
    return RegularGridInterpolator((map_.R_m, map_.Z_m), values, bounds_error=True)(points)


@dataclass(frozen=True)
class FullWaveDqlProjection:
    diffusion: SpatialQuasilinearDiffusion
    target_absorbed_power_W: float
    oxb_conversion_efficiency: float
    fullwave_power_balance_relative: float
    projection_model: str


def project_fullwave_fundamental_dql(
    map_: FullWaveFieldMap2D,
    spatial: OpenFieldLineGrid,
    velocity: VelocityGrid | RelativisticMomentumGrid,
    *,
    time_s: float,
    resonance_width_rad_s: float,
) -> FullWaveDqlProjection:
    map_ = map_.validated()
    if resonance_width_rad_s <= 0.0:
        raise ValueError("resonance_width_rad_s must be positive")
    E_plus = _interpolate(map_, map_.E_plus_V_m, spatial)
    k_parallel = _interpolate(map_, map_.k_parallel_m_inv, spatial)
    B = _interpolate(map_, map_.B_T, spatial)
    absorbed_density = _interpolate(map_, map_.absorbed_power_density_W_m3, spatial)
    v_parallel, v_perp = velocity.mesh()
    speed2 = v_parallel**2 + v_perp**2
    gamma = 1.0 / np.sqrt(np.maximum(1.0 - np.minimum(speed2 / C_LIGHT_M_S**2, 1.0 - 1.0e-14), 1.0e-14))
    omega = 2.0 * PI * map_.frequency_Hz
    shape = (spatial.n_s, *velocity.shape)
    D_parallel = np.zeros(shape)
    D_perp = np.zeros(shape)
    D_cross = np.zeros(shape)
    gaussian_norm = 1.0 / (math.sqrt(2.0 * PI) * resonance_width_rad_s)
    acceleration_scale = E_CHARGE_C**2 / (2.0 * M_E_KG**2)
    for index in range(spatial.n_s):
        omega_ce = E_CHARGE_C * B[index] / M_E_KG
        mismatch = omega - k_parallel[index] * v_parallel - omega_ce / gamma
        resonance = gaussian_norm * np.exp(
            -0.5 * (mismatch / resonance_width_rad_s) ** 2
        )
        base = acceleration_scale * abs(E_plus[index]) ** 2 * resonance
        direction_parallel = k_parallel[index] * v_perp / omega
        D_parallel[index] = base * direction_parallel**2
        D_perp[index] = base
        D_cross[index] = base * direction_parallel
    target = float(np.sum(absorbed_density * spatial.cell_volume_m3))
    diffusion = SpatialQuasilinearDiffusion(
        D_parallel,
        D_perp,
        D_cross,
        time_s,
        f"fullwave:{map_.solver_name}:{map_.solver_version};fundamental_resonant_projection",
        target,
    ).validated(spatial, velocity)
    return FullWaveDqlProjection(
        diffusion,
        target,
        map_.power_budget.oxb_conversion_efficiency,
        map_.power_budget.balance_relative,
        "fundamental Kennel-Engelmann outer-product; Gaussian resonance broadening",
    )


def project_fullwave_harmonic_dql(
    map_: FullWaveFieldMap2D,
    spatial: OpenFieldLineGrid,
    velocity: VelocityGrid | RelativisticMomentumGrid,
    *,
    harmonic: int,
    k_perpendicular_m_inv: float | np.ndarray,
    time_s: float,
    resonance_width_rad_s: float,
    target_absorbed_power_W: float | None = None,
) -> FullWaveDqlProjection:
    """Project n=1/n=2 hot-field coupling with an explicit FLR factor.

    The cyclotron coupling uses ``J_(n-1)(k_perp*rho)^2`` and the relativistic
    resonance ``omega-k_parallel*v_parallel-n*Omega/gamma=0``.  The resulting
    tensor is an outer product and therefore positive semidefinite.
    """

    map_ = map_.validated()
    if harmonic not in (1, 2) or resonance_width_rad_s <= 0.0:
        raise ValueError("harmonic must be 1 or 2 and resonance width positive")
    k_perp = np.asarray(k_perpendicular_m_inv, dtype=float)
    if k_perp.ndim == 0:
        k_perp = np.full(spatial.n_s, float(k_perp))
    if k_perp.shape != (spatial.n_s,) or np.any(~np.isfinite(k_perp)):
        raise ValueError("k_perpendicular_m_inv must be finite and match the spatial grid")
    E_plus = _interpolate(map_, map_.E_plus_V_m, spatial)
    E_minus = _interpolate(map_, map_.E_minus_V_m, spatial)
    k_parallel = _interpolate(map_, map_.k_parallel_m_inv, spatial)
    B = _interpolate(map_, map_.B_T, spatial)
    absorbed_density = _interpolate(map_, map_.absorbed_power_density_W_m3, spatial)
    v_parallel, v_perp = velocity.mesh()
    speed2 = v_parallel**2 + v_perp**2
    gamma = 1.0 / np.sqrt(
        np.maximum(1.0 - np.minimum(speed2 / C_LIGHT_M_S**2, 1.0 - 1.0e-14), 1.0e-14)
    )
    omega = 2.0 * PI * map_.frequency_Hz
    shape = (spatial.n_s, *velocity.shape)
    D_parallel = np.zeros(shape)
    D_perp = np.zeros(shape)
    D_cross = np.zeros(shape)
    gaussian_norm = 1.0 / (math.sqrt(2.0 * PI) * resonance_width_rad_s)
    acceleration_scale = E_CHARGE_C**2 / (2.0 * M_E_KG**2)
    for index in range(spatial.n_s):
        omega_ce = E_CHARGE_C * B[index] / M_E_KG
        mismatch = omega - k_parallel[index] * v_parallel - harmonic * omega_ce / gamma
        resonance = gaussian_norm * np.exp(-0.5 * (mismatch / resonance_width_rad_s) ** 2)
        gyroradius = gamma * M_E_KG * v_perp / (E_CHARGE_C * B[index])
        flr = jv(harmonic - 1, k_perp[index] * gyroradius) ** 2
        polarization = abs(E_plus[index]) ** 2 + (
            abs(E_minus[index]) ** 2 if harmonic == 2 else 0.0
        )
        base = acceleration_scale * polarization * flr * resonance
        direction_parallel = k_parallel[index] * v_perp / omega
        D_parallel[index] = base * direction_parallel**2
        D_perp[index] = base
        D_cross[index] = base * direction_parallel
    target = (
        float(np.sum(absorbed_density * spatial.cell_volume_m3))
        if target_absorbed_power_W is None
        else float(target_absorbed_power_W)
    )
    diffusion = SpatialQuasilinearDiffusion(
        D_parallel,
        D_perp,
        D_cross,
        time_s,
        f"fullwave:{map_.solver_name}:{map_.solver_version};n{harmonic}_flr_projection",
        target,
    ).validated(spatial, velocity)
    return FullWaveDqlProjection(
        diffusion,
        target,
        map_.power_budget.oxb_conversion_efficiency,
        map_.power_budget.balance_relative,
        f"harmonic n={harmonic} Kennel-Engelmann outer-product with Bessel FLR coupling",
    )


__all__ = [
    "FullWavePowerBudget",
    "FullWaveFieldMap2D",
    "FullWaveDqlProjection",
    "load_fullwave_npz",
    "load_open_fieldline_npz",
    "load_spatial_fullwave_dql_npz",
    "project_fullwave_fundamental_dql",
    "project_fullwave_harmonic_dql",
    "save_fullwave_dql_npz",
]
