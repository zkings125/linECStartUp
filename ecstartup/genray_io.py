"""Read native GENRAY NetCDF output without treating it as a D_ql file.

GENRAY writes most ray quantities in cgs units.  This adapter converts the
documented variables to SI and preserves the reported total driven currents.
The standard ``genray.nc`` product contains ray/deposition/current output, but
not the 2-D velocity-space quasilinear tensor consumed by this project's
Fokker--Planck solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import netcdf_file


ERG_S_TO_W = 1.0e-7
CM_TO_M = 1.0e-2
GAUSS_TO_T = 1.0e-4
PER_CM3_TO_PER_M3 = 1.0e6
A_PER_CM2_TO_A_PER_M2 = 1.0e4
ERG_PER_CM3_S_TO_W_PER_M3 = 1.0e-1


def _native(array: Any, *, dtype=float) -> np.ndarray:
    """Copy a scipy-netcdf array and convert big-endian storage to native."""

    return np.asarray(array, dtype=dtype).copy()


def _scalar(variables: dict[str, Any], name: str, *, required: bool = True) -> float | None:
    variable = variables.get(name)
    if variable is None:
        if required:
            raise ValueError(f"GENRAY NetCDF is missing required variable {name!r}")
        return None
    value = float(_native(variable.data).reshape(()))
    if not np.isfinite(value):
        raise ValueError(f"GENRAY variable {name!r} is not finite")
    return value


@dataclass(frozen=True)
class GenrayRay:
    """One native GENRAY ray, trimmed to ``nrayelt`` and converted to SI."""

    path_length_m: np.ndarray
    R_m: np.ndarray
    Z_m: np.ndarray
    phi_rad: np.ndarray
    n_parallel: np.ndarray
    n_perpendicular: np.ndarray
    n_R: np.ndarray
    n_Z: np.ndarray
    n_phi: np.ndarray
    B_T: np.ndarray
    electron_density_m3: np.ndarray
    electron_temperature_eV: np.ndarray
    effective_charge: np.ndarray
    remaining_power_W: np.ndarray
    alpha_collisional_m_inv: np.ndarray
    alpha_collisionless_m_inv: np.ndarray

    @property
    def sample_count(self) -> int:
        return int(self.path_length_m.size)

    @property
    def absorbed_power_W(self) -> float:
        if self.remaining_power_W.size == 0:
            return 0.0
        return max(0.0, float(self.remaining_power_W[0] - self.remaining_power_W[-1]))

    def summary(self) -> dict[str, Any]:
        def finite_range(values: np.ndarray) -> list[float] | None:
            finite = np.asarray(values)[np.isfinite(values)]
            if finite.size == 0:
                return None
            return [float(np.min(finite)), float(np.max(finite))]

        return {
            "sample_count": self.sample_count,
            "path_length_m": float(self.path_length_m[-1]) if self.sample_count else 0.0,
            "R_range_m": finite_range(self.R_m),
            "Z_range_m": finite_range(self.Z_m),
            "n_parallel_range": finite_range(self.n_parallel),
            "n_perpendicular_range": finite_range(self.n_perpendicular),
            "B_range_T": finite_range(self.B_T),
            "electron_density_range_m3": finite_range(self.electron_density_m3),
            "electron_temperature_range_eV": finite_range(self.electron_temperature_eV),
            "launch_power_W": float(self.remaining_power_W[0]) if self.sample_count else 0.0,
            "exit_power_W": float(self.remaining_power_W[-1]) if self.sample_count else 0.0,
            "ray_absorbed_power_W": self.absorbed_power_W,
        }


@dataclass(frozen=True)
class GenrayResult:
    """GENRAY ray and radial-profile result in SI units."""

    source_path: Path
    frequency_Hz: float
    wave_kind: int
    mode_sign: int
    rays: tuple[GenrayRay, ...]
    injected_power_W: float
    absorbed_power_W: float
    electron_absorbed_power_W: float
    ion_absorbed_power_W: float
    collisional_absorbed_power_W: float
    parallel_current_A: float
    toroidal_current_A: float
    poloidal_current_A: float
    rho_bin_center: np.ndarray
    shell_absorbed_power_W: np.ndarray
    parallel_current_density_A_m2: np.ndarray
    toroidal_current_density_A_m2: np.ndarray
    absorbed_power_density_W_m3: np.ndarray

    @property
    def absorption_fraction(self) -> float:
        return self.absorbed_power_W / max(self.injected_power_W, 1.0e-300)

    @property
    def power_balance_relative(self) -> float:
        channels = (
            self.electron_absorbed_power_W
            + self.ion_absorbed_power_W
            + self.collisional_absorbed_power_W
        )
        return (channels - self.absorbed_power_W) / max(abs(self.absorbed_power_W), 1.0)

    @property
    def radial_power_balance_relative(self) -> float:
        return (
            float(np.sum(self.shell_absorbed_power_W)) - self.absorbed_power_W
        ) / max(abs(self.absorbed_power_W), 1.0)

    def summary(self, *, include_rays: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "source_path": str(self.source_path),
            "format": "native_GENRAY_NetCDF",
            "frequency_Hz": self.frequency_Hz,
            "wave_kind_jwave": self.wave_kind,
            "mode_sign_ioxm": self.mode_sign,
            "ray_count": len(self.rays),
            "injected_power_W": self.injected_power_W,
            "absorbed_power_W": self.absorbed_power_W,
            "absorption_fraction": self.absorption_fraction,
            "electron_absorbed_power_W": self.electron_absorbed_power_W,
            "ion_absorbed_power_W": self.ion_absorbed_power_W,
            "collisional_absorbed_power_W": self.collisional_absorbed_power_W,
            "power_balance_relative": self.power_balance_relative,
            "radial_power_balance_relative": self.radial_power_balance_relative,
            "parallel_current_A": self.parallel_current_A,
            "toroidal_current_A": self.toroidal_current_A,
            "poloidal_current_A": self.poloidal_current_A,
            "rho_bin_count": int(self.rho_bin_center.size),
            "dql_available": False,
            "dql_note": (
                "standard GENRAY NetCDF provides rays, deposition and driven-current "
                "diagnostics; it does not provide the 2-D velocity-space D_ql arrays"
            ),
        }
        if include_rays:
            payload["rays"] = [ray.summary() for ray in self.rays]
        return payload


def _ray_array(variables: dict[str, Any], name: str, ray: int, count: int) -> np.ndarray:
    variable = variables.get(name)
    if variable is None:
        raise ValueError(f"GENRAY NetCDF is missing required ray variable {name!r}")
    values = _native(variable.data)
    if values.ndim != 2 or ray >= values.shape[0] or count > values.shape[1]:
        raise ValueError(f"GENRAY ray variable {name!r} has inconsistent dimensions")
    return values[ray, :count]


def read_genray_netcdf(path: str | Path) -> GenrayResult:
    """Read a classic GENRAY NetCDF file and return an SI-valued result.

    ``scipy.io.netcdf_file`` is used deliberately: classic GENRAY files can
    contain both a dimension and a scalar variable named ``nbulk``, which some
    higher-level xarray readers reject.  Opening a file handle also avoids
    path-encoding issues in older netCDF libraries on Windows.
    """

    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    with input_path.open("rb") as handle:
        dataset = netcdf_file(handle, mode="r", mmap=False)
        try:
            variables = dataset.variables
            ray_counts = _native(variables["nrayelt"].data, dtype=int).reshape(-1)
            if np.any(ray_counts < 1):
                raise ValueError("GENRAY nrayelt values must be positive")
            rays: list[GenrayRay] = []
            for ray_index, count_raw in enumerate(ray_counts):
                count = int(count_raw)
                rays.append(
                    GenrayRay(
                        path_length_m=_ray_array(variables, "ws", ray_index, count) * CM_TO_M,
                        R_m=_ray_array(variables, "wr", ray_index, count) * CM_TO_M,
                        Z_m=_ray_array(variables, "wz", ray_index, count) * CM_TO_M,
                        phi_rad=_ray_array(variables, "wphi", ray_index, count),
                        n_parallel=_ray_array(variables, "wnpar", ray_index, count),
                        n_perpendicular=_ray_array(variables, "wnper", ray_index, count),
                        n_R=_ray_array(variables, "wn_r", ray_index, count),
                        n_Z=_ray_array(variables, "wn_z", ray_index, count),
                        n_phi=_ray_array(variables, "wn_phi", ray_index, count),
                        B_T=_ray_array(variables, "sbtot", ray_index, count) * GAUSS_TO_T,
                        electron_density_m3=_ray_array(variables, "sene", ray_index, count)
                        * PER_CM3_TO_PER_M3,
                        electron_temperature_eV=_ray_array(variables, "ste", ray_index, count)
                        * 1.0e3,
                        effective_charge=_ray_array(variables, "szeff", ray_index, count),
                        remaining_power_W=_ray_array(variables, "delpwr", ray_index, count)
                        * ERG_S_TO_W,
                        alpha_collisional_m_inv=_ray_array(variables, "salphac", ray_index, count)
                        / CM_TO_M,
                        alpha_collisionless_m_inv=_ray_array(variables, "salphal", ray_index, count)
                        / CM_TO_M,
                    )
                )

            def profile(name: str, factor: float = 1.0) -> np.ndarray:
                variable = variables.get(name)
                if variable is None:
                    raise ValueError(f"GENRAY NetCDF is missing required profile {name!r}")
                values = _native(variable.data).reshape(-1) * factor
                if np.any(~np.isfinite(values)):
                    raise ValueError(f"GENRAY profile {name!r} contains non-finite values")
                return values

            result = GenrayResult(
                source_path=input_path,
                frequency_Hz=float(_scalar(variables, "freqcy")),
                wave_kind=int(_scalar(variables, "jwave")),
                mode_sign=int(_scalar(variables, "ioxm")),
                rays=tuple(rays),
                injected_power_W=float(_scalar(variables, "power_inj_total")) * ERG_S_TO_W,
                absorbed_power_W=float(_scalar(variables, "power_total")) * ERG_S_TO_W,
                electron_absorbed_power_W=float(_scalar(variables, "powtot_e")) * ERG_S_TO_W,
                ion_absorbed_power_W=float(_scalar(variables, "powtot_i")) * ERG_S_TO_W,
                collisional_absorbed_power_W=float(_scalar(variables, "powtot_cl")) * ERG_S_TO_W,
                parallel_current_A=float(_scalar(variables, "parallel_cur_total")),
                toroidal_current_A=float(_scalar(variables, "toroidal_cur_total")),
                poloidal_current_A=float(_scalar(variables, "poloidal_cur_total")),
                rho_bin_center=profile("rho_bin_center"),
                shell_absorbed_power_W=profile("spower", ERG_S_TO_W),
                parallel_current_density_A_m2=profile(
                    "s_cur_den_parallel", A_PER_CM2_TO_A_PER_M2
                ),
                toroidal_current_density_A_m2=profile(
                    "s_cur_den_toroidal", A_PER_CM2_TO_A_PER_M2
                ),
                absorbed_power_density_W_m3=profile(
                    "powden", ERG_PER_CM3_S_TO_W_PER_M3
                ),
            )
        finally:
            dataset.close()

    if result.injected_power_W <= 0.0 or result.absorbed_power_W < 0.0:
        raise ValueError("GENRAY power totals must be physically non-negative")
    if result.rho_bin_center.size != result.shell_absorbed_power_W.size:
        raise ValueError("GENRAY radial power profile length mismatch")
    return result


def plot_genray_summary(result: GenrayResult, path: str | Path) -> Path:
    """Write a compact ray/deposition/current validation figure."""

    import matplotlib.pyplot as plt

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.4), constrained_layout=True)
    ax = axes[0, 0]
    for index, ray in enumerate(result.rays):
        ax.plot(ray.R_m, ray.Z_m, lw=1.6, label=f"ray {index + 1}")
    ax.set(xlabel="R [m]", ylabel="Z [m]", title="GENRAY ray path")
    ax.axis("equal")
    if len(result.rays) <= 8:
        ax.legend(frameon=False)

    ax = axes[0, 1]
    for index, ray in enumerate(result.rays):
        ax.plot(ray.path_length_m, ray.remaining_power_W / 1.0e3, lw=1.5, label=f"ray {index + 1}")
    ax.set(xlabel="path length [m]", ylabel="remaining power [kW]", title="Wave-power depletion")
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    ax.plot(result.rho_bin_center, result.shell_absorbed_power_W / 1.0e3, color="#15616d")
    ax.set(xlabel=r"$\rho$ bin center", ylabel="absorbed power/bin [kW]", title="Radial deposition")
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    ax.plot(
        result.rho_bin_center,
        result.parallel_current_density_A_m2 / 1.0e3,
        label=r"$J_\parallel$",
        color="#78290f",
    )
    ax.plot(
        result.rho_bin_center,
        result.toroidal_current_density_A_m2 / 1.0e3,
        label=r"$J_\phi$",
        color="#2a9d8f",
    )
    ax.set(xlabel=r"$\rho$ bin center", ylabel=r"current density [kA m$^{-2}$]", title="Driven-current profiles")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    figure.suptitle(
        f"Native GENRAY validation: {result.frequency_Hz / 1e9:.1f} GHz, "
        f"Pabs={result.absorbed_power_W / 1e3:.1f} kW, "
        f"Iphi={result.toroidal_current_A / 1e3:.2f} kA",
        fontsize=12,
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output
