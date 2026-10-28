"""输入配置与严格校验。

所有内部计算使用 SI；字段名显式携带单位，避免把 eV、Pa、T 等混用。
"""

from __future__ import annotations

import ast
import configparser
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat, PositiveFloat, field_validator, model_validator


def _point_in_polygon_or_boundary(
    R_m: float,
    Z_m: float,
    polygon_RZ_m: List[Tuple[float, float]],
    tolerance_m: float = 1.0e-10,
) -> bool:
    """Small dependency-free point-in-polygon check used during validation."""

    R = float(R_m)
    Z = float(Z_m)
    inside = False
    count = len(polygon_RZ_m)
    for index in range(count):
        R0, Z0 = polygon_RZ_m[index]
        R1, Z1 = polygon_RZ_m[(index + 1) % count]
        dR = R1 - R0
        dZ = Z1 - Z0
        length2 = dR * dR + dZ * dZ
        if length2 > 0.0:
            fraction = max(0.0, min(1.0, ((R - R0) * dR + (Z - Z0) * dZ) / length2))
            if math.hypot(R - (R0 + fraction * dR), Z - (Z0 + fraction * dZ)) <= tolerance_m:
                return True
        crosses = (Z0 > Z) != (Z1 > Z)
        if crosses and R < (R1 - R0) * (Z - Z0) / (Z1 - Z0) + R0:
            inside = not inside
    return inside


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class CoilComponentConfig(StrictModel):
    """一个可叠加的解析线圈场分量。

    ``B_*_T_per_A`` 是在当前低阶 provider 中使用的局部场系数，乘以
    ``current_A`` 或 ``current_waveform_A`` 得到 tesla。它们不是线圈的
    真正 Biot--Savart 几何参数；生产计算应由同名 provider 接受线圈电流
    并从实际线圈/网格场计算分量。
    """

    name: str = Field(min_length=1)
    # role 只用于诊断和数据分组，不会自动注入 TPC/VFC 的拓扑或电流闭合。
    role: Literal["auxiliary", "TPC", "VFC"] = "auxiliary"
    current_A: float = 0.0
    current_waveform_A: Optional[List[Tuple[float, float]]] = None
    B_R_T_per_A: float = 0.0
    B_Z_T_per_A: float = 0.0
    B_phi_T_per_A: float = 0.0

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("coil_components.name 不能为空")
        return normalized

    @field_validator("role", mode="before")
    @classmethod
    def normalize_role(cls, value: str) -> str:
        normalized = str(value).strip()
        if normalized.lower() == "auxiliary":
            return "auxiliary"
        if normalized.upper() in {"TPC", "VFC"}:
            return normalized.upper()
        return normalized

    @model_validator(mode="after")
    def check_waveform(self) -> "CoilComponentConfig":
        if self.current_waveform_A is not None:
            from .waveform import validate_points

            # ``validate_assignment=True`` would recursively invoke this
            # model validator if normal attribute assignment were used.
            object.__setattr__(self, "current_waveform_A", validate_points(self.current_waveform_A))
        return self


class BiotSavartLoopConfig(StrictModel):
    """理想圆形细丝线圈的 Biot--Savart 输入。

    ``current_waveform_A`` 使用 ``(time_s, current_A)`` 点列；``turns``
    表示匝数。真实有限截面线圈应拆成多个具有实际半径/轴向位置的
    loop，或由外部 provider 直接注入。
    """

    radius_m: PositiveFloat
    z0_m: float = 0.0
    current_A: float = 0.0
    current_waveform_A: Optional[List[Tuple[float, float]]] = None
    turns: int = Field(default=1, ge=1)
    name: str = "loop"

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("biot_savart_loops.name 不能为空")
        return normalized

    @field_validator("current_waveform_A")
    @classmethod
    def validate_current_waveform(cls, value):
        if value is None:
            return None
        from .waveform import validate_points

        return validate_points(value)


class GeometryConfig(StrictModel):
    R0_m: PositiveFloat = 0.50
    minor_radius_m: PositiveFloat = 0.25
    volume_m3: Optional[PositiveFloat] = None
    wall_area_m2: Optional[PositiveFloat] = None
    connection_length_m: PositiveFloat = 5.0
    B_t_T: PositiveFloat = 0.50
    B_vertical_T: float = 0.015
    B_penetration_T: PositiveFloat = 0.010
    # Analytic field provider selected before a real coil/GS provider is wired in.
    # ``circular`` preserves the v0.1 behaviour. ``TPC`` is the
    # trapped-particle configuration and ``VFC`` is the vertical-field
    # configuration.  These labels select explicit, documented surrogate
    # closures; they are not substitutes for a coil/GS equilibrium.
    field_configuration: Literal["circular", "TPC", "VFC"] = "circular"
    field_configuration_scale: float = Field(default=1.0, gt=0.0)
    # TPC controls (used only by the TPC provider).  They are deliberately
    # explicit so a comparison cannot hide a confinement assumption in code.
    tpc_mirror_ratio: PositiveFloat = Field(default=1.8, ge=1.0)
    tpc_trapped_fraction: float = Field(default=0.25, ge=0.0, le=1.0)
    tpc_decay_index: float = Field(default=1.0, ge=0.0, le=10.0)
    # VFC control (used by the VFC provider and by its loss closure).
    vfc_decay_index: float = Field(default=0.0, ge=-10.0, le=10.0)
    vfc_connection_length_factor: PositiveFloat = 1.0
    # Optional vertical-field charge-separation closure.  This is a calibrated
    # current mechanism, not the definition of the VFC magnetic configuration.
    # Zero preserves the historical model and prevents an invented current.
    vfc_current_fraction: float = Field(default=0.0, ge=-1.0, le=1.0)
    # Descriptive spelling for new input files; the old name remains supported.
    vertical_charge_separation_fraction: Optional[float] = Field(default=None, ge=-1.0, le=1.0)
    closed_confinement_gain: float = Field(default=0.0, ge=0.0)
    vertical_field_waveform: Optional[List[Tuple[float, float]]] = None
    # Optional additive coil surrogates.  Empty preserves the historical
    # single-provider behaviour; non-empty selects CompositeMagneticField in
    # ``make_magnetic_field`` and superposes each component at every time.
    coil_components: List[CoilComponentConfig] = Field(default_factory=list)
    # Optional production-facing circular-loop backend.  The default remains
    # the historical analytic provider; selecting Biot--Savart requires at
    # least one loop and cannot be combined with ``coil_components``.
    magnetic_backend: Literal["analytic", "biot_savart"] = "analytic"
    biot_savart_loops: List[BiotSavartLoopConfig] = Field(default_factory=list)
    R_bounds_m: Tuple[PositiveFloat, PositiveFloat] = (0.20, 0.80)
    Z_bounds_m: Tuple[float, float] = (-0.35, 0.35)
    # Optional measured first-wall/limiter polygon in the poloidal R-Z plane.
    # This is a material boundary only.  It must never be interpreted as an
    # LCFS or as a pre-CFS Grad-Shafranov solution.
    wall_polygon_RZ_m: Optional[List[Tuple[float, float]]] = None

    @field_validator("R_bounds_m")
    @classmethod
    def check_r_bounds(cls, value: Tuple[float, float]) -> Tuple[float, float]:
        if value[1] <= value[0]:
            raise ValueError("R_bounds_m 必须满足上界 > 下界")
        return value

    @field_validator("Z_bounds_m")
    @classmethod
    def check_z_bounds(cls, value: Tuple[float, float]) -> Tuple[float, float]:
        if value[1] <= value[0]:
            raise ValueError("Z_bounds_m 必须满足上界 > 下界")
        return value

    @field_validator("wall_polygon_RZ_m")
    @classmethod
    def check_wall_polygon(cls, value):
        if value is None:
            return None
        points = [(float(R), float(Z)) for R, Z in value]
        if len(points) >= 2 and points[0] == points[-1]:
            points = points[:-1]
        if len(points) < 3:
            raise ValueError("wall_polygon_RZ_m 至少需要 3 个不重复顶点")
        if any(not (math.isfinite(R) and math.isfinite(Z)) or R < 0.0 for R, Z in points):
            raise ValueError("壁面多边形坐标必须为有限数，且 R >= 0")
        area2 = sum(
            points[index][0] * points[(index + 1) % len(points)][1]
            - points[(index + 1) % len(points)][0] * points[index][1]
            for index in range(len(points))
        )
        if abs(area2) <= 1.0e-12:
            raise ValueError("wall_polygon_RZ_m 面积不能为零")
        return points

    @field_validator("field_configuration", mode="before")
    @classmethod
    def normalize_field_configuration(cls, value: str) -> str:
        normalized = str(value).strip()
        if normalized.lower() == "circular":
            return "circular"
        if normalized.upper() in {"TPC", "VFC"}:
            return normalized.upper()
        return normalized

    @model_validator(mode="after")
    def check_geometry(self) -> "GeometryConfig":
        if not (self.R_bounds_m[0] < self.R0_m < self.R_bounds_m[1]):
            raise ValueError("R0_m 必须位于 R_bounds_m 内")
        if self.R0_m - self.minor_radius_m <= 0:
            raise ValueError("R0_m - minor_radius_m 必须为正")
        if self.wall_polygon_RZ_m is not None:
            if not _point_in_polygon_or_boundary(
                self.R0_m, 0.0, self.wall_polygon_RZ_m
            ):
                raise ValueError("R0_m, Z=0 必须位于 wall_polygon_RZ_m 内部或边界上")
        if self.vertical_field_waveform is not None:
            from .waveform import validate_points

            object.__setattr__(self, "vertical_field_waveform", validate_points(self.vertical_field_waveform))
        if self.vertical_charge_separation_fraction is not None:
            # Keep both spellings deterministic when a legacy and a new field
            # are supplied together; silently choosing one would make scans
            # difficult to audit.
            if self.vfc_current_fraction != 0.0 and not math.isclose(
                self.vfc_current_fraction, self.vertical_charge_separation_fraction, rel_tol=1.0e-12, abs_tol=1.0e-15
            ):
                raise ValueError(
                    "vfc_current_fraction 与 vertical_charge_separation_fraction 不得同时取不同值"
                )
            # ``validate_assignment=True`` would re-enter this model validator
            # if the alias were assigned through the normal attribute path.
            object.__setattr__(self, "vfc_current_fraction", self.vertical_charge_separation_fraction)
        names = [component.name for component in self.coil_components]
        if len(names) != len(set(names)):
            raise ValueError("coil_components.name 必须唯一，便于诊断和复现实验")
        loop_names = [loop.name for loop in self.biot_savart_loops]
        if len(loop_names) != len(set(loop_names)):
            raise ValueError("biot_savart_loops.name 必须唯一，便于诊断和复现实验")
        if self.magnetic_backend == "biot_savart" and not self.biot_savart_loops:
            raise ValueError("magnetic_backend=biot_savart 时必须提供 biot_savart_loops")
        if self.magnetic_backend == "biot_savart" and self.coil_components:
            raise ValueError("biot_savart backend 不能同时使用 coil_components；请二选一")
        if self.magnetic_backend == "analytic" and self.biot_savart_loops:
            raise ValueError("提供 biot_savart_loops 时必须显式设置 magnetic_backend=biot_savart")
        return self

    @property
    def effective_vertical_charge_fraction(self) -> float:
        """Return the configured vertical charge-separation coefficient."""

        return float(self.vfc_current_fraction)

    @property
    def resolved_volume_m3(self) -> float:
        # 圆截面环形体积；用户提供的 volume_m3 优先。
        return self.volume_m3 or (2.0 * 3.141592653589793**2 * self.R0_m * self.minor_radius_m**2)

    @property
    def resolved_wall_area_m2(self) -> float:
        return self.wall_area_m2 or (4.0 * 3.141592653589793**2 * self.R0_m * self.minor_radius_m)


class GasConfig(StrictModel):
    species: str = "D2"
    pressure_Pa: PositiveFloat = 0.10
    Tn_K: PositiveFloat = 300.0
    ion_mass_kg: PositiveFloat = 3.343583719e-27
    ionization_energy_eV: PositiveFloat = 15.4
    excitation_energy_eV: PositiveFloat = 8.0
    recombination_coeff_m3_s: float = Field(default=1.0e-13, ge=0.0)
    attachment_coeff_m3_s: float = Field(default=0.0, ge=0.0)
    recycling_coefficient: float = Field(default=0.90, ge=0.0, le=1.0)
    pump_time_s: PositiveFloat = 0.50
    secondary_emission: float = Field(default=0.05, ge=0.0, le=1.0)
    # 这是可替换的低阶碰撞闭合，不是截面数据库本身。
    momentum_rate_scale_m3_s: PositiveFloat = 2.0e-14
    ionization_rate_scale_m3_s: PositiveFloat = 2.0e-14
    excitation_rate_scale_m3_s: PositiveFloat = 5.0e-14
    collision_database: str = "analytic_surrogate"
    collision_database_version: str = "v0.1"
    pressure_waveform_Pa: Optional[List[Tuple[float, float]]] = None

    @field_validator("pressure_waveform_Pa")
    @classmethod
    def validate_pressure_waveform(cls, value):
        if value is None:
            return None
        from .waveform import validate_points

        points = validate_points(value)
        if any(pressure < 0.0 for _, pressure in points):
            raise ValueError("pressure_waveform_Pa 的压强不得为负")
        return points

    def pressure_at(self, t_s: float) -> float:
        if self.pressure_waveform_Pa is None:
            return float(self.pressure_Pa)
        from .waveform import linear_value

        return max(0.0, linear_value(self.pressure_waveform_Pa, t_s, float(self.pressure_Pa)))


class InitialConfig(StrictModel):
    ne_m3: NonNegativeFloat = 1.0e14
    ni_m3: NonNegativeFloat = 1.0e14
    nn_m3: Optional[NonNegativeFloat] = None
    Te_eV: NonNegativeFloat = 0.20
    Ti_eV: NonNegativeFloat = 0.03
    Ip_A: float = 0.0
    n_hot_m3: float = Field(default=0.0, ge=0.0)
    hot_energy_J_m3: float = Field(default=0.0, ge=0.0)
    seed_rate_m3_s: float = Field(default=0.0, ge=0.0)


class PulseConfig(StrictModel):
    start_s: float = 0.0
    end_s: float = 0.05
    ramp_up_s: float = Field(default=0.0, ge=0.0)
    ramp_down_s: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def check_pulse(self) -> "PulseConfig":
        if self.end_s <= self.start_s:
            raise ValueError("RF pulse 的 end_s 必须大于 start_s")
        if self.ramp_up_s + self.ramp_down_s > self.end_s - self.start_s:
            raise ValueError("RF ramp 总时长不能超过脉冲宽度")
        return self


class LaunchConfig(StrictModel):
    """RF launcher position and steering in the local cylindrical basis."""

    R_m: Optional[PositiveFloat] = None
    Z_m: float = 0.0
    # Angle in the R-Z plane: 0 -> +R, 90 -> +Z, 180 -> -R.  None uses an
    # explicit, diagnostic-only fallback aimed at (R0, 0).
    poloidal_angle_deg: Optional[float] = None
    # Angle out of the R-Z plane toward +phi.
    toroidal_angle_deg: float = 0.0
    k_parallel_m_inv: float = 0.0
    mode: str = "X2"
    polarization: str = "X"


class RFSourceConfig(StrictModel):
    """One independently timed EC source in a multi-frequency system."""

    name: str = Field(min_length=1, pattern=r"^[A-Za-z][A-Za-z0-9_]*$")
    enabled: bool = True
    frequency_Hz: PositiveFloat
    harmonic: int = Field(default=1, ge=1, le=20)
    power_W: float = Field(default=0.0, ge=0.0)
    absorption_fraction: float = Field(default=0.70, ge=0.0, le=1.0)
    pulse: PulseConfig = Field(default_factory=PulseConfig)
    launch: LaunchConfig = Field(default_factory=LaunchConfig)
    hot_fraction: float = Field(default=0.20, ge=0.0, le=1.0)
    hot_creation_energy_eV: PositiveFloat = 100.0
    current_drive_A_per_W: float = 0.0
    current_drive_sign: float = Field(default=1.0, ge=-1.0, le=1.0)


class RFConfig(StrictModel):
    enabled: bool = True
    frequency_Hz: PositiveFloat = 2.45e9
    harmonic: int = Field(default=1, ge=1, le=20)
    power_W: float = Field(default=20_000.0, ge=0.0)
    absorption_fraction: float = Field(default=0.70, ge=0.0, le=1.0)
    pulse: PulseConfig = Field(default_factory=PulseConfig)
    launch: LaunchConfig = Field(default_factory=LaunchConfig)
    hot_fraction: float = Field(default=0.20, ge=0.0, le=1.0)
    hot_creation_energy_eV: PositiveFloat = 100.0
    hot_loss_time_s: PositiveFloat = 2.0e-4
    hot_scattering_time_s: PositiveFloat = 5.0e-4
    orbit_asymmetry: float = Field(default=0.05, ge=-1.0, le=1.0)
    current_drive_A_per_W: float = 0.0
    current_drive_sign: float = Field(default=1.0, ge=-1.0, le=1.0)
    sources: List[RFSourceConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_sources(self) -> "RFConfig":
        names = [source.name for source in self.sources]
        if len(names) != len(set(names)):
            raise ValueError("rf.sources.name 必须唯一")
        return self


class RayTracingConfig(StrictModel):
    """Numerical controls for the cold-plasma Hamiltonian EC ray backend."""

    enabled: bool = False
    dispersion_model: Literal["cold_stix"] = "cold_stix"
    step_m: PositiveFloat = 2.0e-3
    max_length_m: PositiveFloat = 5.0
    max_steps: int = Field(default=5000, ge=1)
    spatial_difference_m: PositiveFloat = 1.0e-4
    wavevector_relative_step: float = Field(default=1.0e-5, gt=0.0, le=1.0e-2)
    resonance_tolerance: float = Field(default=0.02, gt=0.0, le=0.5)
    wkb_min_scale_ratio: PositiveFloat = 5.0
    terminate_on_wkb_failure: bool = True
    R_bounds_m: Optional[Tuple[PositiveFloat, PositiveFloat]] = None
    Z_bounds_m: Optional[Tuple[float, float]] = None

    @model_validator(mode="after")
    def check_bounds(self) -> "RayTracingConfig":
        if self.R_bounds_m is not None and self.R_bounds_m[1] <= self.R_bounds_m[0]:
            raise ValueError("ray_tracing.R_bounds_m 必须满足上界 > 下界")
        if self.Z_bounds_m is not None and self.Z_bounds_m[1] <= self.Z_bounds_m[0]:
            raise ValueError("ray_tracing.Z_bounds_m 必须满足上界 > 下界")
        return self


class EBWConfig(StrictModel):
    """O-X-B availability and schematic EBW-path controls.

    The density cloud is defined directly in R-Z space and therefore does not
    assume magnetic flux surfaces before CFS formation.
    """

    enabled: bool = False
    density_profile: Literal["gaussian_cloud"] = "gaussian_cloud"
    center_R_m: Optional[PositiveFloat] = None
    center_Z_m: float = 0.0
    sigma_R_m: PositiveFloat = 0.30
    sigma_Z_m: PositiveFloat = 0.36
    density_floor_m3: float = Field(default=1.0e14, ge=0.0)
    schematic_peak_density_m3: PositiveFloat = 3.0e19
    peak_to_average_factor: PositiveFloat = 2.0
    cutoff_tolerance_fraction: float = Field(default=0.08, gt=0.0, le=0.5)
    uhr_tolerance_fraction: float = Field(default=0.08, gt=0.0, le=0.5)
    minimum_ox_efficiency: float = Field(default=0.01, ge=0.0, le=1.0)
    minimum_abs_n_parallel_for_cd: float = Field(default=0.05, ge=0.0)
    ebw_path_step_m: PositiveFloat = 2.0e-3
    ebw_max_length_m: PositiveFloat = 1.0
    R_bounds_m: Optional[Tuple[PositiveFloat, PositiveFloat]] = None
    Z_bounds_m: Optional[Tuple[float, float]] = None

    @model_validator(mode="after")
    def check_bounds(self) -> "EBWConfig":
        if self.R_bounds_m is not None and self.R_bounds_m[1] <= self.R_bounds_m[0]:
            raise ValueError("ebw.R_bounds_m 必须满足上界 > 下界")
        if self.Z_bounds_m is not None and self.Z_bounds_m[1] <= self.Z_bounds_m[0]:
            raise ValueError("ebw.Z_bounds_m 必须满足上界 > 下界")
        return self


class GenrayLikeConfig(StrictModel):
    """Controls for the independent reduced GENRAY-like D_ql generator.

    This block configures a pre-CFS, magnetic-surface-free approximation.  It
    is intentionally separate from native GENRAY NetCDF import because the
    standard NetCDF output does not contain velocity-space D_ql arrays.
    """

    source_name: Optional[str] = None
    resonance_width_fraction: float = Field(default=0.035, gt=0.0, le=0.5)
    peak_absorption_coefficient_m_inv: PositiveFloat = 8.0
    density_reference_m3: PositiveFloat = 1.0e19
    temperature_reference_eV: PositiveFloat = 1.0e3
    density_exponent: float = Field(default=1.0, ge=0.0, le=3.0)
    temperature_exponent: float = Field(default=-0.5, ge=-3.0, le=3.0)
    parallel_diffusion_floor: float = Field(default=0.02, ge=0.0, le=1.0)
    cross_diffusion_fraction: float = Field(default=0.0, ge=-1.0, le=1.0)
    minimum_kernel_peak: PositiveFloat = 1.0e-12
    # Conventional dimensionless ECCD efficiency
    # zeta = e^3 n_e R I / (epsilon_0^2 P T_e).  This supplies an independent
    # adjoint-like order-of-magnitude diagnostic; it does not alter D_ql or the
    # F-P current moment.
    adjoint_normalized_efficiency: float = Field(default=0.30, ge=0.0, le=2.0)
    cache_reference_diffusion: bool = True


class KineticConfig(StrictModel):
    """Controls for the independent reduced 0D-2V electron solver.

    ``coupling_mode='diagnostic_only'`` keeps the solver independent from the
    global startup ODE.  ``coupling_mode='moment_feedback'`` builds a cached
    power-response table from distribution moments and supplies that current
    to the global L-R circuit.  The latter is quantitative only when D_ql is
    supplied by a validated, power-normalized wave solver.
    """

    enabled: bool = False
    backend: Literal["reduced_0d2v", "open_field_line_1d2v"] = "reduced_0d2v"
    coordinate_system: Literal["velocity", "relativistic_momentum"] = "velocity"
    coupling_mode: Literal[
        "diagnostic_only", "moment_feedback", "time_dependent_feedback"
    ] = "diagnostic_only"
    n_v_parallel: int = Field(default=81, ge=3, le=1001)
    n_v_perp: int = Field(default=64, ge=2, le=1000)
    v_parallel_max_m_s: PositiveFloat = 4.0e6
    v_perp_max_m_s: PositiveFloat = 4.0e6
    u_parallel_max: PositiveFloat = 10.0
    u_perp_max: PositiveFloat = 10.0
    momentum_conserving_field_particle: bool = False
    orbit_model: Literal["no_orbit_average", "trapped_passing_bounce_average"] = "no_orbit_average"
    orbit_B_min_T: PositiveFloat = 0.1
    orbit_B_max_T: PositiveFloat = 1.0
    time_step_s: PositiveFloat = 1.0e-8
    steps: int = Field(default=100, ge=1, le=1_000_000)
    initial_drift_v_parallel_m_s: float = 0.0
    collision_frequency_s: NonNegativeFloat = 2.0e5
    collision_temperature_eV: Optional[PositiveFloat] = None
    electric_field_V_m: float = 0.0
    loss_frequency_s: NonNegativeFloat = 0.0
    rf_diffusion_model: Literal[
        "zero", "synthetic_gaussian", "external_npz", "genray_npz", "genray_like_internal"
    ] = "zero"
    rf_diffusion_file: Optional[str] = None
    rf_diffusion_reference_power_W: Optional[PositiveFloat] = None
    rf_diffusion_parallel_peak_m2_s3: NonNegativeFloat = 0.0
    rf_diffusion_perp_peak_m2_s3: NonNegativeFloat = 0.0
    rf_resonant_v_parallel_m_s: float = 0.0
    rf_resonant_v_perp_m_s: NonNegativeFloat = 0.0
    rf_sigma_v_parallel_m_s: PositiveFloat = 1.0e6
    rf_sigma_v_perp_m_s: PositiveFloat = 1.0e6
    feedback_reference_absorbed_power_W: PositiveFloat = 2.5e5
    feedback_reference_density_m3: Optional[PositiveFloat] = None
    feedback_density_ratio_cap: PositiveFloat = 100.0
    feedback_power_fractions: List[NonNegativeFloat] = Field(
        default_factory=lambda: [0.0, 0.25, 0.5, 1.0, 2.0]
    )
    feedback_replace_direct_rf: bool = True
    feedback_replace_hot_tail: bool = False
    time_dependent_iterations: int = Field(default=3, ge=1, le=20)
    time_dependent_relaxation: float = Field(default=1.0, gt=0.0, le=1.0)
    time_dependent_tolerance_A: PositiveFloat = 10.0
    time_dependent_snapshot_count: int = Field(default=5, ge=2, le=12)
    spatial_fieldline_file: Optional[str] = None
    spatial_dql_file: Optional[str] = None
    spatial_sheath_model: Literal["none", "quasineutral_ambipolar"] = "none"
    spatial_ionization_from_global_rates: bool = True
    spatial_use_global_rf_power: bool = True

    @model_validator(mode="after")
    def check_diffusion_source(self) -> "KineticConfig":
        file_models = {"external_npz", "genray_npz"}
        if self.rf_diffusion_model in file_models and not self.rf_diffusion_file:
            raise ValueError("external_npz/genray_npz 必须提供 rf_diffusion_file")
        if self.rf_diffusion_model not in file_models and self.rf_diffusion_file is not None:
            raise ValueError("rf_diffusion_file 只允许用于 external_npz/genray_npz")
        if (
            self.rf_diffusion_reference_power_W is not None
            and self.rf_diffusion_model != "genray_npz"
        ):
            raise ValueError("rf_diffusion_reference_power_W 只允许用于 genray_npz")
        fractions = sorted(set(float(item) for item in self.feedback_power_fractions))
        if len(fractions) < 2 or fractions[0] != 0.0 or fractions[-1] <= 0.0:
            raise ValueError("feedback_power_fractions 必须含 0 和至少一个正值")
        object.__setattr__(self, "feedback_power_fractions", fractions)
        if self.coupling_mode != "diagnostic_only" and not self.enabled:
            raise ValueError("F-P feedback coupling 要求 kinetic.switchOn='enabled'")
        if self.backend == "open_field_line_1d2v":
            if not self.spatial_fieldline_file:
                raise ValueError("open_field_line_1d2v 必须提供 spatial_fieldline_file")
            if self.coupling_mode == "moment_feedback":
                raise ValueError(
                    "open_field_line_1d2v 不支持静态 moment_feedback；"
                    "请选择 diagnostic_only 或 time_dependent_feedback"
                )
        return self


class CircuitConfig(StrictModel):
    cs_mode: Literal["with_cs", "without_cs"] = "with_cs"
    loop_voltage_V: float = 5.0
    loop_voltage_ramp_s: float = Field(default=1.0e-3, ge=0.0)
    total_inductance_H: PositiveFloat = 2.0e-6
    plasma_length_m: Optional[PositiveFloat] = None
    conductivity_floor_S_m: PositiveFloat = 1.0e-8
    resistance_cap_ohm: PositiveFloat = 1.0e6
    loop_voltage_waveform: Optional[List[Tuple[float, float]]] = None
    effective_charge_override: Optional[float] = Field(default=None, ge=1.0)

    @model_validator(mode="after")
    def check_waveform(self) -> "CircuitConfig":
        if self.loop_voltage_waveform is not None:
            from .waveform import validate_points

            object.__setattr__(self, "loop_voltage_waveform", validate_points(self.loop_voltage_waveform))
        return self


class ThresholdConfig(StrictModel):
    breakdown_density_m3: PositiveFloat = 1.0e16
    preionization_density_m3: PositiveFloat = 1.0e18
    preionization_Te_eV: PositiveFloat = 2.0
    cfs_closed_fraction: float = Field(default=0.05, ge=0.0, le=1.0)
    persistence_steps: int = Field(default=3, ge=1, le=10_000)
    failure_density_m3: PositiveFloat = 1.0e10


class NumericsConfig(StrictModel):
    t_start_s: float = 0.0
    t_end_s: PositiveFloat = 2.0e-3
    dt_s: PositiveFloat = 2.0e-6
    solver: str = "BDF"
    rtol: PositiveFloat = 1.0e-5
    atol: PositiveFloat = 1.0e-8
    max_step_s: Optional[PositiveFloat] = None

    @field_validator("solver")
    @classmethod
    def normalize_solver(cls, value: str) -> str:
        value = value.upper()
        if value not in {"BDF", "RADAU", "RK45"}:
            raise ValueError("solver 只能是 BDF、Radau 或 RK45")
        return value


class StartupWindowConfig(StrictModel):
    """Physical meaning of the simulated startup interval.

    ``numerics.t_start_s`` and ``numerics.t_end_s`` remain the integration
    limits.  This block records what their time origin and terminal condition
    mean for a particular device/shot; it deliberately does not assume that
    every device has a 0--100 ms startup.
    """

    definition: Literal["explicit_time", "event_relative"] = "explicit_time"
    reference_event: Literal[
        "absolute_clock", "ec_on", "gas_on", "first_Ip", "breakdown", "custom"
    ] = "absolute_clock"
    custom_reference_event: Optional[str] = None
    start_offset_s: float = 0.0
    end_condition: Literal[
        "integration_end", "first_persistent_cfs", "diagnostic_marker", "custom"
    ] = "integration_end"
    diagnostic_marker: Optional[str] = None
    custom_end_condition: Optional[str] = None
    cfs_persistence_s: NonNegativeFloat = 0.0

    @model_validator(mode="after")
    def check_semantics(self) -> "StartupWindowConfig":
        if self.reference_event == "custom" and not str(self.custom_reference_event or "").strip():
            raise ValueError("reference_event='custom' 时必须提供 custom_reference_event")
        if self.reference_event != "custom" and self.custom_reference_event is not None:
            raise ValueError("custom_reference_event 只允许用于 reference_event='custom'")
        if self.end_condition == "diagnostic_marker" and not str(self.diagnostic_marker or "").strip():
            raise ValueError("end_condition='diagnostic_marker' 时必须提供 diagnostic_marker")
        if self.end_condition != "diagnostic_marker" and self.diagnostic_marker is not None:
            raise ValueError("diagnostic_marker 只允许用于 end_condition='diagnostic_marker'")
        if self.end_condition == "custom" and not str(self.custom_end_condition or "").strip():
            raise ValueError("end_condition='custom' 时必须提供 custom_end_condition")
        if self.end_condition != "custom" and self.custom_end_condition is not None:
            raise ValueError("custom_end_condition 只允许用于 end_condition='custom'")
        return self

    def describe(self, *, t_start_s: float, t_end_s: float) -> Dict[str, Any]:
        return {
            "definition": self.definition,
            "reference_event": (
                self.custom_reference_event if self.reference_event == "custom" else self.reference_event
            ),
            "integration_interval_s": [float(t_start_s), float(t_end_s)],
            "start_offset_s": float(self.start_offset_s),
            "end_condition": (
                self.custom_end_condition if self.end_condition == "custom" else self.end_condition
            ),
            "diagnostic_marker": self.diagnostic_marker,
            "cfs_persistence_s": float(self.cfs_persistence_s),
            "universal_duration_assumed": False,
        }


class OutputConfig(StrictModel):
    directory: str = "runs/minimal"
    cadence_s: Optional[PositiveFloat] = None
    write_netcdf: bool = True
    write_csv: bool = True
    write_current_profile: bool = False
    write_pdf_report: bool = False
    pdf_report_filename: str = "startup_report.pdf"
    profile_radial_points: int = Field(default=101, ge=21, le=2001)
    profile_snapshot_count: int = Field(default=8, ge=2, le=30)

    @field_validator("pdf_report_filename")
    @classmethod
    def check_pdf_filename(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized.lower().endswith(".pdf") or Path(normalized).name != normalized:
            raise ValueError("pdf_report_filename 必须是当前运行目录内的 .pdf 文件名")
        return normalized


class CurrentProfileConfig(StrictModel):
    """Normalized geometric-rho basis used to reconstruct 0D currents."""

    coordinate: Literal["geometric_r_over_a"] = "geometric_r_over_a"
    ohmic_shape: Tuple[NonNegativeFloat, NonNegativeFloat] = (0.0, 1.5)
    rf_shape: Tuple[NonNegativeFloat, NonNegativeFloat] = (0.5, 3.0)
    tail_shape: Tuple[NonNegativeFloat, NonNegativeFloat] = (1.5, 1.5)
    vertical_shape: Tuple[NonNegativeFloat, NonNegativeFloat] = (4.0, 0.5)
    residual_shape: Tuple[NonNegativeFloat, NonNegativeFloat] = (0.0, 1.0)


class ThresholdScanConfig(StrictModel):
    enabled: bool = False
    source_name: Optional[str] = None
    disable_other_sources: bool = True
    pressures_Pa: List[PositiveFloat] = Field(default_factory=lambda: [0.01, 0.05, 0.10])
    effective_charges: List[float] = Field(default_factory=lambda: [1.0, 2.0, 3.0])
    ec_powers_W: List[NonNegativeFloat] = Field(default_factory=lambda: [8.0e4, 1.2e5, 2.0e5])
    cs_modes: List[Literal["with_cs", "without_cs"]] = Field(
        default_factory=lambda: ["without_cs"]
    )
    current_direction_sign: Literal[-1, 1] = 1
    success_increment_A: PositiveFloat = 1.0e3
    success_total_current_A: NonNegativeFloat = 0.0
    require_preionization: bool = True
    refine_enabled: bool = False
    refine_power_tolerance_W: PositiveFloat = 5.0e3
    refine_max_iterations: int = Field(default=12, ge=1, le=64)
    kinetic_coupling_mode: Literal[
        "inherit", "moment_feedback", "time_dependent_feedback"
    ] = "inherit"
    write_pdf_report: bool = True
    pdf_report_filename: str = "threshold_scan_report.pdf"

    @field_validator("pressures_Pa", "effective_charges", "ec_powers_W", "cs_modes")
    @classmethod
    def check_nonempty_scan_axis(cls, value: list) -> list:
        if not value:
            raise ValueError("阈值扫描轴不能为空")
        return value

    @field_validator("effective_charges")
    @classmethod
    def check_effective_charge(cls, value: List[float]) -> List[float]:
        if any(item < 1.0 for item in value):
            raise ValueError("effective_charges 必须 >= 1")
        return value

    @field_validator("pdf_report_filename")
    @classmethod
    def check_scan_pdf_filename(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized.lower().endswith(".pdf") or Path(normalized).name != normalized:
            raise ValueError("threshold scan PDF 必须是当前输出目录内的 .pdf 文件名")
        return normalized


class ExperimentalStartupPriorConfig(StrictModel):
    """User-supplied EXL-50U startup anchors used for reporting/calibration.

    These values never alter the first-principles/surrogate ODE silently. They
    are retained as an explicit comparison layer until raw shot diagnostics
    and uncertainty estimates are supplied.
    """

    enabled: bool = False
    weak_cs_current_A: NonNegativeFloat = 3.0
    treat_weak_cs_as_no_cs: bool = True
    robust_startup_power_upper_bound_W: PositiveFloat = 1.25e5
    robust_reference_power_W: PositiveFloat = 3.0e5
    low_power_reference_W: PositiveFloat = 7.0e4
    low_power_pressure_ceiling_Pa: PositiveFloat = 5.0e-3
    optimum_pressure_Pa: PositiveFloat = 4.0e-3
    optimum_pressure_reference_powers_W: List[PositiveFloat] = Field(
        default_factory=lambda: [8.0e4, 2.2e5]
    )
    importance_order: List[Literal["power", "pressure", "impurity"]] = Field(
        default_factory=lambda: ["power", "pressure", "impurity"]
    )
    high_pressure_termination_risk: bool = True
    h_alpha_peak_required_for_termination: bool = False

    @model_validator(mode="after")
    def check_prior(self) -> "ExperimentalStartupPriorConfig":
        if self.robust_reference_power_W < self.robust_startup_power_upper_bound_W:
            raise ValueError("robust_reference_power_W 必须不低于稳健启动功率上界")
        if set(self.importance_order) != {"power", "pressure", "impurity"}:
            raise ValueError("importance_order 必须且只能包含 power/pressure/impurity")
        if len(self.importance_order) != 3:
            raise ValueError("importance_order 不得包含重复项")
        return self


class QuantitativeValidationConfig(StrictModel):
    """Evidence required before device-level startup predictions are reported.

    The switches describe available, shot-resolved inputs and calibration
    evidence; they do not enable or disable any physics term.  Keeping this
    gate in the input schema makes the intended use auditable and prevents a
    reduced/surrogate run from being presented as a validated device forecast.
    """

    intended_use: Literal[
        "diagnostic_only",
        "quantitative_device_prediction",
    ] = "diagnostic_only"
    device_name: str = "unknown"
    shot_id: Optional[str] = None

    shot_resolved_coil_waveforms: bool = False
    measured_pressure_waveform: bool = False
    measured_zeff_or_impurity: bool = False
    calibrated_rf_beam: bool = False
    validated_power_normalized_dql: bool = False
    wall_limiter_geometry: bool = False
    orbit_loss_calibrated: bool = False

    validation_success_shots: int = Field(default=0, ge=0)
    validation_failure_shots: int = Field(default=0, ge=0)
    validation_current_trace_shots: int = Field(default=0, ge=0)
    minimum_success_shots: int = Field(default=3, ge=1)
    minimum_failure_shots: int = Field(default=3, ge=1)
    minimum_current_trace_shots: int = Field(default=3, ge=1)

    startup_sensitivity: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    startup_specificity: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    ip_relative_error: Optional[float] = Field(default=None, ge=0.0)
    minimum_sensitivity: float = Field(default=0.90, ge=0.0, le=1.0)
    minimum_specificity: float = Field(default=0.90, ge=0.0, le=1.0)
    maximum_ip_relative_error: float = Field(default=0.20, ge=0.0)

    @field_validator("device_name")
    @classmethod
    def check_device_name(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("device_name 不能为空")
        return normalized


class AspectRatioScanConfig(StrictModel):
    """Independent major/minor-radius scan for cross-device trends."""

    enabled: bool = False
    R0_values_m: List[PositiveFloat] = Field(default_factory=lambda: [0.50])
    minor_radius_values_m: List[PositiveFloat] = Field(default_factory=lambda: [0.25])
    combination: Literal["cartesian", "paired"] = "cartesian"
    scale_rf_launch_with_geometry: bool = True

    @field_validator("R0_values_m", "minor_radius_values_m")
    @classmethod
    def check_scan_values(cls, value: list[float]) -> list[float]:
        if not value:
            raise ValueError("环径比扫描轴不能为空")
        return value

    @model_validator(mode="after")
    def check_pairs(self) -> "AspectRatioScanConfig":
        if self.combination == "paired" and len(self.R0_values_m) != len(self.minor_radius_values_m):
            raise ValueError("combination='paired' 时 R0_values_m 与 minor_radius_values_m 长度必须相同")
        invalid = [
            (float(R0), float(a))
            for R0 in self.R0_values_m
            for a in self.minor_radius_values_m
            if R0 <= a
        ]
        if self.combination == "paired":
            invalid = [
                (float(R0), float(a))
                for R0, a in zip(self.R0_values_m, self.minor_radius_values_m)
                if R0 <= a
            ]
        if invalid:
            raise ValueError(f"所有扫描点必须满足 R0_m > minor_radius_m；无效点: {invalid}")
        return self


class AtomicSpeciesConfig(StrictModel):
    """One atomic element in the optional DYON-like charge-state model.

    The fuel entry is mapped to the legacy ``nn``/``ni`` states in the first
    implementation.  Impurities own one ODE state for every charge state from
    zero through ``max_charge``.
    """

    symbol: str = Field(min_length=1, pattern=r"^[A-Za-z][A-Za-z0-9_]*$")
    role: Literal["fuel", "impurity"] = "impurity"
    atomic_number: int = Field(ge=1, le=118)
    max_charge: int = Field(default=1, ge=1)
    initial_charge_densities_m3: List[NonNegativeFloat] = Field(default_factory=list)
    ionization_energies_eV: Optional[List[PositiveFloat]] = None
    ionization_rate_scale_m3_s: PositiveFloat = 2.0e-14
    recombination_rate_scale_m3_s: PositiveFloat = 1.0e-13
    radiation_coefficient_W_m3: float = Field(default=1.0e-31, ge=0.0)
    wall_loss_multiplier: float = Field(default=1.0, ge=0.0)
    neutral_pump_time_s: Optional[PositiveFloat] = None

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        return str(value).strip()

    @model_validator(mode="after")
    def check_charge_states(self) -> "AtomicSpeciesConfig":
        if self.max_charge > self.atomic_number:
            raise ValueError("max_charge 不能大于 atomic_number")
        if self.role == "fuel":
            if self.max_charge != 1:
                raise ValueError("首版 fuel 映射到 nn/ni，因此 max_charge 必须为 1")
            if self.initial_charge_densities_m3:
                raise ValueError("fuel 初值来自 initial.nn_m3/ni_m3，不得重复配置")
        elif len(self.initial_charge_densities_m3) != self.max_charge + 1:
            raise ValueError("impurity 的 initial_charge_densities_m3 长度必须为 max_charge+1")
        if self.ionization_energies_eV is not None and len(self.ionization_energies_eV) != self.max_charge:
            raise ValueError("ionization_energies_eV 长度必须等于 max_charge")
        return self


class WallMaterialConfig(StrictModel):
    """Wall interaction parameters for one released/retained element."""

    name: str = Field(min_length=1)
    species: str = Field(min_length=1)
    initial_inventory_m2: NonNegativeFloat = 0.0
    recycling_coefficient: float = Field(default=0.0, ge=0.0, le=1.0)
    desorption_time_s: Optional[PositiveFloat] = None
    physical_sputtering_yield: float = Field(default=0.0, ge=0.0)
    physical_sputtering_threshold_eV: NonNegativeFloat = 0.0
    chemical_sputtering_yield: float = Field(default=0.0, ge=0.0)
    incident_energy_factor: float = Field(default=3.0, ge=0.0)

    @field_validator("name", "species")
    @classmethod
    def strip_identifier(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("名称不能为空")
        return normalized


class DYONSpeciesWallConfig(StrictModel):
    """Configuration for the independent DYON-like species/wall module."""

    enabled: bool = False
    atomic_backend: Literal["analytic_surrogate", "openadas_adf11"] = "analytic_surrogate"
    atomic_manifest: Optional[str] = None
    atomic_bounds: Literal["raise", "clip"] = "raise"
    species: List[AtomicSpeciesConfig] = Field(default_factory=list)
    wall_materials: List[WallMaterialConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_references(self) -> "DYONSpeciesWallConfig":
        symbols = [item.symbol for item in self.species]
        if len(symbols) != len(set(symbols)):
            raise ValueError("species.symbol 必须唯一")
        fuels = [item for item in self.species if item.role == "fuel"]
        if len(fuels) > 1:
            raise ValueError("首版只支持一个 fuel 物种")
        if self.enabled and not self.species:
            raise ValueError("species_wall.enabled=true 时必须配置 species")
        if self.atomic_backend == "openadas_adf11":
            if not self.atomic_manifest:
                raise ValueError("openadas_adf11 必须配置 atomic_manifest")
            unsupported = [item.symbol for item in self.species if item.atomic_number != 1]
            if unsupported:
                raise ValueError(
                    "当前 bundled OpenADAS H-96 snapshot 只支持 H/D/T；"
                    f"不支持: {', '.join(unsupported)}"
                )
        wall_names = [item.name for item in self.wall_materials]
        if len(wall_names) != len(set(wall_names)):
            raise ValueError("wall_materials.name 必须唯一")
        wall_species = [item.species for item in self.wall_materials]
        if len(wall_species) != len(set(wall_species)):
            raise ValueError("首版每个 species 只能配置一个 wall material")
        unknown = sorted(set(wall_species) - set(symbols))
        if unknown:
            raise ValueError(f"wall material 引用了未配置物种: {', '.join(unknown)}")
        return self


class ChargeClosureConfig(StrictModel):
    """Electron/ion charge-continuity closure at open-field-line walls."""

    mode: Literal["legacy", "quasineutral", "ambipolar"] = "legacy"
    initialize_ne_from_ions: bool = True
    potential_min_over_Te: float = Field(default=-10.0, ge=-100.0, le=0.0)
    potential_max_over_Te: float = Field(default=30.0, ge=0.0, le=100.0)
    flux_floor_m3_s: PositiveFloat = 1.0

    @model_validator(mode="after")
    def check_potential_bounds(self) -> "ChargeClosureConfig":
        if self.potential_max_over_Te <= self.potential_min_over_Te:
            raise ValueError("potential_max_over_Te 必须大于 potential_min_over_Te")
        return self


class SimulationConfig(StrictModel):
    schema_version: str = "1.0"
    model: str = "global0d_surrogate"
    geometry: GeometryConfig = Field(default_factory=GeometryConfig)
    gas: GasConfig = Field(default_factory=GasConfig)
    initial: InitialConfig = Field(default_factory=InitialConfig)
    rf: RFConfig = Field(default_factory=RFConfig)
    ray_tracing: RayTracingConfig = Field(default_factory=RayTracingConfig)
    ebw: EBWConfig = Field(default_factory=EBWConfig)
    genray_like: GenrayLikeConfig = Field(default_factory=GenrayLikeConfig)
    kinetic: KineticConfig = Field(default_factory=KineticConfig)
    circuit: CircuitConfig = Field(default_factory=CircuitConfig)
    thresholds: ThresholdConfig = Field(default_factory=ThresholdConfig)
    numerics: NumericsConfig = Field(default_factory=NumericsConfig)
    startup_window: StartupWindowConfig = Field(default_factory=StartupWindowConfig)
    outputs: OutputConfig = Field(default_factory=OutputConfig)
    current_profile: CurrentProfileConfig = Field(default_factory=CurrentProfileConfig)
    threshold_scan: ThresholdScanConfig = Field(default_factory=ThresholdScanConfig)
    experimental_startup_prior: ExperimentalStartupPriorConfig = Field(
        default_factory=ExperimentalStartupPriorConfig
    )
    validation: QuantitativeValidationConfig = Field(
        default_factory=QuantitativeValidationConfig
    )
    aspect_ratio_scan: AspectRatioScanConfig = Field(default_factory=AspectRatioScanConfig)
    species_wall: DYONSpeciesWallConfig = Field(default_factory=DYONSpeciesWallConfig)
    charge_closure: ChargeClosureConfig = Field(default_factory=ChargeClosureConfig)
    seed: int = 0

    @model_validator(mode="after")
    def check_cross_fields(self) -> "SimulationConfig":
        if self.circuit.plasma_length_m is None:
            self.circuit.plasma_length_m = 2.0 * 3.141592653589793 * self.geometry.R0_m
        duration = self.numerics.t_end_s - self.numerics.t_start_s
        if duration <= 0.0:
            raise ValueError("t_end_s 必须大于 t_start_s")
        if self.numerics.dt_s > duration:
            raise ValueError("dt_s 不能大于仿真时间跨度")
        # 只做提示级别的物理一致性检查，避免把共振当作硬阈值。
        if self.rf.frequency_Hz <= 0:
            raise ValueError("frequency_Hz 必须为正")
        if self.genray_like.source_name is not None:
            source_names = {source.name for source in self.rf.sources}
            if self.genray_like.source_name not in source_names:
                raise ValueError(
                    "genray_like.source_name 必须匹配一个 [rf_source:<name>]；"
                    f"当前可用值: {sorted(source_names)}"
                )
        return self

    def resolved_dict(self) -> Dict[str, Any]:
        data = self.model_dump(mode="json")
        data["geometry"]["volume_m3"] = self.geometry.resolved_volume_m3
        data["geometry"]["wall_area_m2"] = self.geometry.resolved_wall_area_m2
        data["startup_window_resolved"] = self.startup_window.describe(
            t_start_s=self.numerics.t_start_s,
            t_end_s=self.numerics.t_end_s,
        )
        return data


def _load_mapping(path: Path) -> Dict[str, Any]:
    if path.suffix.lower() in {".ini", ".cfg"}:
        return _load_ini_mapping(path)
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
    except ImportError:
        # JSON 是 YAML 的严格子集，便于在最小环境中运行示例。
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("配置文件顶层必须是对象/映射")
    return data


_INI_SECTION_PATHS: Dict[str, Tuple[str, ...]] = {
    "simulation": (),
    "geometry": ("geometry",),
    "gas": ("gas",),
    "initial": ("initial",),
    "rf": ("rf",),
    "rf.pulse": ("rf", "pulse"),
    "rf.launch": ("rf", "launch"),
    "ray_tracing": ("ray_tracing",),
    "ebw": ("ebw",),
    "genray_like": ("genray_like",),
    "kinetic": ("kinetic",),
    "circuit": ("circuit",),
    "thresholds": ("thresholds",),
    "numerics": ("numerics",),
    "startup_window": ("startup_window",),
    "outputs": ("outputs",),
    "current_profile": ("current_profile",),
    "threshold_scan": ("threshold_scan",),
    "experimental_startup_prior": ("experimental_startup_prior",),
    "validation": ("validation",),
    "aspect_ratio_scan": ("aspect_ratio_scan",),
    "species_wall": ("species_wall",),
    "charge_closure": ("charge_closure",),
}


def _parse_ini_scalar(raw: str) -> Any:
    """Parse one human-editable INI value without executing code."""

    value = raw.strip()
    lowered = value.lower()
    if lowered in {"enabled", "enable", "true", "yes", "on"}:
        return True
    if lowered in {"disabled", "disable", "false", "no", "off"}:
        return False
    if lowered in {"none", "null"}:
        return None
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        parsed = value
    if isinstance(parsed, str):
        normalized = parsed.strip().lower()
        if normalized in {"enabled", "enable", "true", "yes", "on"}:
            return True
        if normalized in {"disabled", "disable", "false", "no", "off"}:
            return False
        if normalized in {"none", "null"}:
            return None
    return parsed


def _assign_nested(mapping: Dict[str, Any], path: Tuple[str, ...], values: Dict[str, Any]) -> None:
    target = mapping
    for key in path:
        target = target.setdefault(key, {})
    target.update(values)


def _load_ini_mapping(path: Path) -> Dict[str, Any]:
    """Load the commented user parameter format into the strict schema.

    Static sections map directly to Pydantic objects.  Repeated model
    components use named sections: ``species:<symbol>``, ``wall:<name>``,
    ``coil:<name>`` and ``biot_savart_loop:<name>``.
    """

    parser = configparser.ConfigParser(
        interpolation=None,
        inline_comment_prefixes=("#", ";"),
        strict=True,
    )
    parser.optionxform = str
    with path.open("r", encoding="utf-8-sig") as handle:
        parser.read_file(handle)

    data: Dict[str, Any] = {}
    dynamic: Dict[str, list] = {
        "species": [],
        "wall": [],
        "coil": [],
        "biot_savart_loop": [],
        "rf_source": [],
    }
    for section in parser.sections():
        values = {key: _parse_ini_scalar(value) for key, value in parser.items(section)}
        if "switchOn" in values:
            if "enabled" in values:
                raise ValueError(f"INI section [{section}] 不能同时给出 switchOn 和 enabled")
            values["enabled"] = values.pop("switchOn")
        if section in _INI_SECTION_PATHS:
            _assign_nested(data, _INI_SECTION_PATHS[section], values)
            continue
        prefix, separator, name = section.partition(":")
        if not separator or prefix not in dynamic or not name.strip():
            supported = ", ".join(sorted(_INI_SECTION_PATHS))
            raise ValueError(
                f"未知 INI section [{section}]；支持 {supported} 以及 species:/wall:/coil:/biot_savart_loop:/rf_source: 命名段"
            )
        item = dict(values)
        if prefix == "species":
            item.setdefault("symbol", name.strip())
        else:
            item.setdefault("name", name.strip())
        if prefix == "rf_source":
            pulse = {
                key.removeprefix("pulse_"): item.pop(key)
                for key in list(item)
                if key.startswith("pulse_")
            }
            launch = {
                key.removeprefix("launch_"): item.pop(key)
                for key in list(item)
                if key.startswith("launch_")
            }
            if pulse:
                item["pulse"] = pulse
            if launch:
                item["launch"] = launch
        dynamic[prefix].append(item)

    if dynamic["species"]:
        data.setdefault("species_wall", {})["species"] = dynamic["species"]
    if dynamic["wall"]:
        data.setdefault("species_wall", {})["wall_materials"] = dynamic["wall"]
    if dynamic["coil"]:
        data.setdefault("geometry", {})["coil_components"] = dynamic["coil"]
    if dynamic["biot_savart_loop"]:
        data.setdefault("geometry", {})["biot_savart_loops"] = dynamic["biot_savart_loop"]
    if dynamic["rf_source"]:
        data.setdefault("rf", {})["sources"] = dynamic["rf_source"]
    return data


def load_config(path: str | Path) -> SimulationConfig:
    """读取并严格校验 YAML/JSON 配置。"""

    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    return SimulationConfig.model_validate(_load_mapping(cfg_path))


def dump_resolved_config(config: SimulationConfig, path: str | Path) -> None:
    Path(path).write_text(json.dumps(config.resolved_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
