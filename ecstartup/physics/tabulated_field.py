"""规则 :math:`(R,Z)` 网格磁场 provider。

该模块把来自 GS、线圈求解器或实验重建的二维轴对称场图接入
``MagneticFieldProvider`` 契约。输入网格必须是 SI 单位下的严格递增一维
``R_grid_m``/``Z_grid_m``，场数组的形状必须是 ``(n_R, n_Z)``（R 维在前，
Z 维在后）。场值使用 :class:`scipy.interpolate.RegularGridInterpolator`
进行双线性（或最近邻）插值。

``Ip_A`` 和 ``t_s`` 保留在方法签名中以兼容现有启动 ODE，但本 provider
表示一个静态场图，不会把 ``Ip_A`` 自动缩放到场值，也不会对时间插值。
需要随电流/时间变化的场，应在外部先生成快照或实现更高层的 provider。

边界行为必须显式选择：

``raise`` / ``error``
    超出网格立即抛出 ``ValueError``；这是默认值，避免静默外推。
``clip`` / ``nearest``
    将查询坐标裁剪到最近网格边界后再插值。
``extrapolate``
    使用 ``RegularGridInterpolator`` 的线性外推；仅适合边界附近的敏感性
    测试，不应替代真实壁面场。
``nan``
    超出网格返回 ``NaN``，便于批量扫描后由上层掩膜。

所有长度为 m、磁场为 T；通量函数 ``psi`` 采用轴对称约定
``psi = R A_phi``，单位 Wb/rad，并满足
``B_R = -(1/R) dpsi/dZ``、``B_Z = (1/R) dpsi/dR``。输入 ``psi`` 是可选
的；缺省时默认从 ``R B_Z`` 沿 R 数值积分（并在最内侧边界用 ``B_R`` 做
一个规范修正），这只是可审计的重建，不等价于原始 GS 通量。也可传
``derive_psi=False`` 让缺省 psi 时的 ``flux_function`` 明确报错。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import math

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from ..config import GeometryConfig
from .magnetic import FieldPoint, MagneticConfigurationState


_BOUNDARY_ALIASES = {
    "raise": "raise",
    "error": "raise",
    "strict": "raise",
    "clip": "clip",
    "nearest": "clip",
    "clamp": "clip",
    "extrapolate": "extrapolate",
    "linear_extrapolate": "extrapolate",
    "nan": "nan",
    "mask": "nan",
}


def _as_float_grid(values: Any, name: str, *, minimum_size: int = 2) -> np.ndarray:
    """Validate and copy one regular-grid coordinate vector."""

    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是有限的一维浮点数组") from exc
    if array.ndim != 1 or array.size < minimum_size:
        raise ValueError(f"{name} 必须是一维数组，且至少包含 {minimum_size} 个点")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} 必须只包含有限数")
    if np.any(np.diff(array) <= 0.0):
        raise ValueError(f"{name} 必须严格递增且不能有重复点")
    return np.array(array, dtype=float, copy=True)


def _as_field_array(values: Any, name: str, shape: tuple[int, int]) -> np.ndarray:
    """Validate one ``(n_R,n_Z)`` field array and return an immutable copy."""

    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是二维浮点数组，形状应为 {shape}") from exc
    if array.shape != shape:
        raise ValueError(f"{name} 的形状必须是 {shape}，实际为 {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} 必须只包含有限数")
    copied = np.array(array, dtype=float, copy=True)
    copied.setflags(write=False)
    return copied


def _normalise_boundary(value: str) -> str:
    key = str(value).strip().lower().replace("-", "_")
    try:
        return _BOUNDARY_ALIASES[key]
    except KeyError as exc:
        choices = ", ".join(("raise", "clip", "extrapolate", "nan"))
        raise ValueError(f"boundary 必须是 {choices} 之一，实际为 {value!r}") from exc


def _normalise_configuration(value: str) -> str:
    configuration = str(value).strip()
    if not configuration:
        raise ValueError("configuration 不能为空")
    return configuration


def _derive_flux_function(
    R_grid_m: np.ndarray,
    Z_grid_m: np.ndarray,
    B_R_T: np.ndarray,
    B_Z_T: np.ndarray,
) -> np.ndarray:
    """Reconstruct a gauge-fixed ``psi`` from a tabulated poloidal field.

    The first integration uses ``dpsi/dR = R B_Z``.  A Z-dependent gauge is
    then added by integrating ``-R_min B_R(R_min,Z)`` so that the radial-field
    relation is also represented at the inner boundary when the supplied map
    is approximately divergence-free.  The remaining inconsistency is exposed
    by ``psi_*_residual`` diagnostics instead of being hidden.
    """

    r = np.asarray(R_grid_m, dtype=float)
    z = np.asarray(Z_grid_m, dtype=float)
    integrand_r = r[:, None] * np.asarray(B_Z_T, dtype=float)
    psi = np.zeros_like(integrand_r, dtype=float)
    dr = np.diff(r)[:, None]
    psi[1:, :] = np.cumsum(0.5 * (integrand_r[1:, :] + integrand_r[:-1, :]) * dr, axis=0)

    # The gauge correction is optional in the cylindrical-axis limit: regular
    # axisymmetric fields have R*B_R -> 0 as R -> 0.
    r_min = float(r[0])
    if abs(r_min) > 0.0:
        gauge_integrand = -r_min * np.asarray(B_R_T[0, :], dtype=float)
        dz = np.diff(z)
        gauge = np.zeros_like(z, dtype=float)
        gauge[1:] = np.cumsum(0.5 * (gauge_integrand[1:] + gauge_integrand[:-1]) * dz)
        psi = psi + gauge[None, :]
    return psi


def _divergence_residual(
    R_grid_m: np.ndarray,
    Z_grid_m: np.ndarray,
    B_R_T: np.ndarray,
    B_Z_T: np.ndarray,
) -> tuple[float, float]:
    """Return RMS and maximum axisymmetric ``div(B_p)`` in T/m.

    Values on an exact cylindrical axis are omitted because the expression
    ``(1/R) d(R B_R)/dR`` needs a limiting derivative there.  Omitting one row
    is preferable to reporting an artificial singularity for a regular map.
    """

    r = np.asarray(R_grid_m, dtype=float)
    z = np.asarray(Z_grid_m, dtype=float)
    rb = r[:, None] * np.asarray(B_R_T, dtype=float)
    d_rb_d_r = np.gradient(rb, r, axis=0, edge_order=1)
    d_bz_d_z = np.gradient(np.asarray(B_Z_T, dtype=float), z, axis=1, edge_order=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        div = d_rb_d_r / r[:, None] + d_bz_d_z
    valid = np.isfinite(div) & (r[:, None] > np.finfo(float).eps)
    values = np.asarray(div[valid], dtype=float)
    if values.size == 0:
        return 0.0, 0.0
    return float(np.sqrt(np.mean(values * values))), float(np.max(np.abs(values)))


def _flux_field_residual(
    R_grid_m: np.ndarray,
    Z_grid_m: np.ndarray,
    psi_Wb: np.ndarray,
    B_R_T: np.ndarray,
    B_Z_T: np.ndarray,
) -> tuple[float, float]:
    """Compare derivatives of a supplied/reconstructed psi with ``B_p``."""

    r = np.asarray(R_grid_m, dtype=float)
    z = np.asarray(Z_grid_m, dtype=float)
    dpsi_d_r = np.gradient(np.asarray(psi_Wb, dtype=float), r, axis=0, edge_order=1)
    dpsi_d_z = np.gradient(np.asarray(psi_Wb, dtype=float), z, axis=1, edge_order=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        br_from_psi = -dpsi_d_z / r[:, None]
        bz_from_psi = dpsi_d_r / r[:, None]
    residual = np.sqrt((br_from_psi - B_R_T) ** 2 + (bz_from_psi - B_Z_T) ** 2)
    valid = np.isfinite(residual) & (r[:, None] > np.finfo(float).eps)
    values = residual[valid]
    if values.size == 0:
        return 0.0, 0.0
    return float(np.sqrt(np.mean(values * values))), float(np.max(values))


class TabulatedMagneticField:
    """Static axisymmetric magnetic field sampled on a regular ``(R,Z)`` grid.

    Parameters
    ----------
    R_grid_m, Z_grid_m:
        Strictly increasing one-dimensional coordinates in metres.
    B_R_T, B_Z_T, B_phi_T:
        Two-dimensional arrays with shape ``(len(R_grid_m), len(Z_grid_m))``
        and units tesla.  ``B_phi_T`` may be omitted, in which case it is zero.
    psi_Wb:
        Optional poloidal flux array with the same shape, in Wb/rad.  The
        aliases ``psi`` and ``psi_Wb_per_rad`` are accepted for convenience.
    geometry:
        Optional :class:`~ecstartup.config.GeometryConfig` used only to choose
        the reference point for ``vertical_field`` and the representative edge
        point in diagnostics.  It does not rescale the map.
    boundary:
        One of ``raise`` (default), ``clip``, ``extrapolate`` or ``nan``;
        aliases ``error``/``nearest`` are accepted.  ``boundary_strategy`` is
        an equivalent keyword kept for callers that prefer a longer name.
    derive_psi:
        If true (default), reconstruct a gauge-fixed psi when no psi array is
        supplied.  Set false to require an explicit psi map.

    The optional mirror/trapping/connection arguments are metadata closures for
    downstream loss models.  They are never inferred from the field map and
    therefore do not accidentally identify a generic map as TPC or VFC.
    """

    def __init__(
        self,
        R_grid_m: Sequence[float] | np.ndarray,
        Z_grid_m: Sequence[float] | np.ndarray,
        B_R_T: Any,
        B_Z_T: Any,
        B_phi_T: Any | None = None,
        psi_Wb: Any | None = None,
        *,
        psi_Wb_per_rad: Any | None = None,
        psi: Any | None = None,
        geometry: GeometryConfig | Mapping[str, Any] | None = None,
        configuration: str = "tabulated",
        boundary: str = "raise",
        boundary_strategy: str | None = None,
        method: str = "linear",
        axis_R_m: float | None = None,
        axis_Z_m: float | None = None,
        edge_R_m: float | None = None,
        edge_Z_m: float | None = None,
        mirror_ratio: float = 1.0,
        trapped_fraction: float = 0.0,
        decay_index: float = 0.0,
        connection_length_factor: float = 1.0,
        loss_multiplier: float = 1.0,
        derive_psi: bool = True,
    ) -> None:
        if boundary_strategy is not None:
            selected = _normalise_boundary(boundary_strategy)
            existing = _normalise_boundary(boundary)
            if existing != "raise" and existing != selected:
                raise ValueError("boundary 与 boundary_strategy 不得指定不同策略")
            boundary = selected
        self.boundary_strategy = _normalise_boundary(boundary)
        self.interpolation_method = str(method).strip().lower()
        if self.interpolation_method not in {"linear", "nearest"}:
            raise ValueError("method 只能是 'linear' 或 'nearest'")

        self.configuration = _normalise_configuration(configuration)
        if geometry is not None and not isinstance(geometry, GeometryConfig):
            if not isinstance(geometry, Mapping):
                raise TypeError("geometry 必须是 GeometryConfig 或映射")
            geometry = GeometryConfig.model_validate(geometry)
        self.geometry = geometry

        r = _as_float_grid(R_grid_m, "R_grid_m")
        z = _as_float_grid(Z_grid_m, "Z_grid_m")
        if r[0] < 0.0:
            raise ValueError("R_grid_m 必须从非负半径开始")
        shape = (int(r.size), int(z.size))
        br = _as_field_array(B_R_T, "B_R_T", shape)
        bz = _as_field_array(B_Z_T, "B_Z_T", shape)
        if B_phi_T is None:
            bphi = np.zeros(shape, dtype=float)
            bphi.setflags(write=False)
        else:
            bphi = _as_field_array(B_phi_T, "B_phi_T", shape)

        psi_candidates = [
            ("psi_Wb", psi_Wb),
            ("psi_Wb_per_rad", psi_Wb_per_rad),
            ("psi", psi),
        ]
        supplied = [(name, value) for name, value in psi_candidates if value is not None]
        if len(supplied) > 1:
            first = np.asarray(supplied[0][1], dtype=float)
            for name, value in supplied[1:]:
                candidate = np.asarray(value, dtype=float)
                if first.shape != candidate.shape or not np.allclose(first, candidate, rtol=1.0e-12, atol=1.0e-14):
                    raise ValueError("psi_Wb、psi_Wb_per_rad 与 psi 的内容不一致")
        use_derived = bool(derive_psi)
        if supplied:
            psi_array = _as_field_array(supplied[0][1], supplied[0][0], shape)
            psi_source = "provided"
        elif use_derived:
            derived = _derive_flux_function(r, z, br, bz)
            psi_array = _as_field_array(derived, "psi_Wb(derived)", shape)
            psi_source = "derived_from_B_Z"
        else:
            psi_array = None
            psi_source = "unavailable"

        # Public read-only arrays make the map auditable and prevent accidental
        # mutation after interpolators have been constructed.
        self.R_grid_m = r
        self.Z_grid_m = z
        self.R_grid = self.R_grid_m
        self.Z_grid = self.Z_grid_m
        self.B_R_T = br
        self.B_Z_T = bz
        self.B_phi_T = bphi
        self.psi_Wb = psi_array
        self.psi_Wb_per_rad = psi_array
        self.psi_source = psi_source
        self.has_flux_function = psi_array is not None

        self._grid_shape = shape
        self._interpolators: dict[str, RegularGridInterpolator] = {}
        if self.boundary_strategy == "raise":
            bounds_error = True
            fill_value = np.nan
        elif self.boundary_strategy == "extrapolate":
            bounds_error = False
            fill_value = None
        else:
            bounds_error = False
            fill_value = np.nan
        # Expose the exact scipy settings as public metadata so a serialized
        # run can be audited without reverse-engineering ``boundary_strategy``.
        self.bounds_error = bool(bounds_error)
        self.fill_value = fill_value
        for name, values in (("B_R_T", br), ("B_Z_T", bz), ("B_phi_T", bphi)):
            self._interpolators[name] = RegularGridInterpolator(
                (r, z),
                values,
                method=self.interpolation_method,
                bounds_error=bounds_error,
                fill_value=fill_value,
            )
        if psi_array is not None:
            self._interpolators["psi_Wb"] = RegularGridInterpolator(
                (r, z),
                psi_array,
                method=self.interpolation_method,
                bounds_error=bounds_error,
                fill_value=fill_value,
            )

        self._divB_rms_T_per_m, self._divB_max_T_per_m = _divergence_residual(r, z, br, bz)
        if psi_array is not None:
            self._psi_residual_rms_T, self._psi_residual_max_T = _flux_field_residual(
                r, z, psi_array, br, bz
            )
        else:
            self._psi_residual_rms_T = float("nan")
            self._psi_residual_max_T = float("nan")

        self.reference_R_m = self._resolve_reference(axis_R_m, "R", geometry)
        self.reference_Z_m = self._resolve_reference(axis_Z_m, "Z", geometry)
        self.edge_R_m = self._resolve_edge(edge_R_m, "R", geometry)
        self.edge_Z_m = self._resolve_edge(edge_Z_m, "Z", geometry)
        self.axis_R_m = self.reference_R_m
        self.axis_Z_m = self.reference_Z_m

        self.mirror_ratio = self._validate_state_value("mirror_ratio", mirror_ratio, minimum=1.0)
        self.trapped_fraction = self._validate_state_value("trapped_fraction", trapped_fraction, minimum=0.0, maximum=1.0)
        self.decay_index = self._validate_state_value("decay_index", decay_index)
        self.connection_length_factor = self._validate_state_value(
            "connection_length_factor", connection_length_factor, minimum=0.0, strict_positive=True
        )
        self.loss_multiplier = self._validate_state_value(
            "loss_multiplier", loss_multiplier, minimum=0.0, strict_positive=True
        )

        # These flags make the static nature explicit to callers assembling a
        # time-dependent workflow.
        self.depends_on_ip = False
        self.depends_on_time = False

    @staticmethod
    def _validate_state_value(
        name: str,
        value: float,
        *,
        minimum: float | None = None,
        maximum: float | None = None,
        strict_positive: bool = False,
    ) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} 必须是有限数") from exc
        if not math.isfinite(number):
            raise ValueError(f"{name} 必须是有限数")
        if strict_positive and number <= 0.0:
            raise ValueError(f"{name} 必须为正数")
        if minimum is not None and number < minimum:
            raise ValueError(f"{name} 不能小于 {minimum}")
        if maximum is not None and number > maximum:
            raise ValueError(f"{name} 不能大于 {maximum}")
        return number

    def _resolve_reference(self, value: float | None, axis: str, geometry: Any) -> float:
        if value is None:
            if geometry is not None:
                # GeometryConfig is centred at Z=0; only the major radius is
                # read from it.  (Using ``getattr(..., 0.0)`` here would be a
                # type error because getattr expects a string attribute name.)
                value = float(geometry.R0_m) if axis == "R" else 0.0
            elif axis == "R":
                value = 0.5 * (self.R_grid_m[0] + self.R_grid_m[-1])
            else:
                value = 0.0 if self.Z_grid_m[0] <= 0.0 <= self.Z_grid_m[-1] else 0.5 * (self.Z_grid_m[0] + self.Z_grid_m[-1])
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"axis_{axis}_m/reference_{axis}_m 必须是有限数") from exc
        if not math.isfinite(number):
            raise ValueError(f"axis_{axis}_m/reference_{axis}_m 必须是有限数")
        if axis == "R" and number < 0.0:
            raise ValueError("reference R 必须为非负数")
        return number

    def _resolve_edge(self, value: float | None, axis: str, geometry: Any) -> float:
        if value is None:
            if geometry is not None:
                if axis == "R":
                    value = float(geometry.R0_m) + float(geometry.minor_radius_m)
                else:
                    value = 0.0
            elif axis == "R":
                value = float(self.R_grid_m[-1])
            else:
                value = self.reference_Z_m
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"edge_{axis}_m 必须是有限数") from exc
        if not math.isfinite(number):
            raise ValueError(f"edge_{axis}_m 必须是有限数")
        if axis == "R" and number < 0.0:
            raise ValueError("edge R 必须为非负数")
        return number

    @property
    def grid_shape(self) -> tuple[int, int]:
        """Return ``(n_R, n_Z)`` of the immutable field map."""

        return self._grid_shape

    @property
    def grid_bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return ``((R_min,R_max),(Z_min,Z_max))`` in SI units."""

        return (
            (float(self.R_grid_m[0]), float(self.R_grid_m[-1])),
            (float(self.Z_grid_m[0]), float(self.Z_grid_m[-1])),
        )

    def _query_arrays(self, R_m: Any, Z_m: Any, *, force_clip: bool = False):
        try:
            R = np.asarray(R_m, dtype=float)
            Z = np.asarray(Z_m, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("R_m/Z_m 必须是有限数或可广播数组") from exc
        try:
            R, Z = np.broadcast_arrays(R, Z)
        except ValueError as exc:
            raise ValueError("R_m 与 Z_m 的形状无法广播") from exc
        if not np.isfinite(R).all() or not np.isfinite(Z).all():
            raise ValueError("R_m/Z_m 必须只包含有限数")
        if np.any(R < 0.0):
            raise ValueError("柱坐标 R_m 必须为非负数")
        strategy = "clip" if force_clip else self.boundary_strategy
        outside = (
            (R < self.R_grid_m[0])
            | (R > self.R_grid_m[-1])
            | (Z < self.Z_grid_m[0])
            | (Z > self.Z_grid_m[-1])
        )
        if strategy == "raise" and np.any(outside):
            raise ValueError(
                "查询点超出 tabulated magnetic field 网格范围 "
                f"R[{self.R_grid_m[0]}, {self.R_grid_m[-1]}] m, "
                f"Z[{self.Z_grid_m[0]}, {self.Z_grid_m[-1]}] m"
            )
        if strategy == "clip":
            R = np.clip(R, self.R_grid_m[0], self.R_grid_m[-1])
            Z = np.clip(Z, self.Z_grid_m[0], self.Z_grid_m[-1])
        shape = R.shape
        points = np.column_stack((R.reshape(-1), Z.reshape(-1)))
        return points, shape, bool(shape == ()), bool(np.any(outside))

    def _lookup(self, name: str, R_m: Any, Z_m: Any, *, force_clip: bool = False):
        points, shape, scalar, _ = self._query_arrays(R_m, Z_m, force_clip=force_clip)
        interpolator = self._interpolators[name]
        try:
            values = np.asarray(interpolator(points), dtype=float).reshape(shape)
        except ValueError as exc:
            # Add the provider context while preserving a concise original
            # exception for callers that need to distinguish invalid bounds.
            raise ValueError(f"{name} 插值失败（boundary={self.boundary_strategy}）: {exc}") from exc
        if scalar:
            return float(values)
        return values

    def _evaluate_point(self, R_m: float, Z_m: float, *, force_clip: bool = False) -> FieldPoint:
        return FieldPoint(
            self._lookup("B_R_T", R_m, Z_m, force_clip=force_clip),
            self._lookup("B_Z_T", R_m, Z_m, force_clip=force_clip),
            self._lookup("B_phi_T", R_m, Z_m, force_clip=force_clip),
        )

    def evaluate(self, R_m: float, Z_m: float, Ip_A: float, t_s: float = 0.0) -> FieldPoint:
        """Evaluate ``(B_R,B_Z,B_phi)`` at one scalar point (tesla).

        The ``Ip_A`` and ``t_s`` arguments are accepted for protocol
        compatibility and intentionally ignored for a static map.  Use
        :meth:`flux_function` for vectorized array queries.
        """

        try:
            if np.asarray(R_m).ndim != 0 or np.asarray(Z_m).ndim != 0:
                raise ValueError("evaluate 只接受标量 R_m/Z_m；数组请调用 flux_function")
        except TypeError as exc:
            raise ValueError("R_m/Z_m 必须是标量") from exc
        return self._evaluate_point(float(R_m), float(Z_m))

    def flux_function(self, R_m: Any, Z_m: Any, Ip_A: float, t_s: float = 0.0):
        """Interpolate ``psi=R A_phi`` in Wb/rad (scalar or broadcast arrays)."""

        if self._interpolators.get("psi_Wb") is None:
            raise ValueError(
                "该场图没有 psi；请提供 psi_Wb，或构造时设置 derive_psi=True"
            )
        return self._lookup("psi_Wb", R_m, Z_m)

    def vertical_field(self, t_s: float = 0.0) -> float:
        """Return ``B_Z`` at the configured reference (magnetic-axis) point."""

        return float(self._lookup("B_Z_T", self.reference_R_m, self.reference_Z_m))

    def vertical_field_at(self, R_m: float, Z_m: float = 0.0, t_s: float = 0.0) -> float:
        """Convenience scalar lookup of ``B_Z(R,Z)`` in tesla."""

        return float(self._lookup("B_Z_T", R_m, Z_m))

    def _vertical_gradient(self) -> float:
        """Estimate ``dB_Z/dR`` at the reference point in T/m."""

        r0 = float(self.reference_R_m)
        z0 = float(self.reference_Z_m)
        r_min = float(self.R_grid_m[0])
        r_max = float(self.R_grid_m[-1])
        if r_max <= r_min:
            return float("nan")
        spacings = np.diff(self.R_grid_m)
        h_nominal = 0.5 * float(np.min(spacings))
        if h_nominal <= 0.0:
            return float("nan")
        if r_min + h_nominal <= r0 <= r_max - h_nominal:
            plus = self._lookup("B_Z_T", r0 + h_nominal, z0)
            minus = self._lookup("B_Z_T", r0 - h_nominal, z0)
            return float((plus - minus) / (2.0 * h_nominal))
        if r0 < r_min:
            # For extrapolation/nan strategies this retains the selected query
            # semantics; strict mode will raise in vertical_field first.
            h = h_nominal
            plus = self._lookup("B_Z_T", r0 + h, z0)
            here = self._lookup("B_Z_T", r0, z0)
            return float((plus - here) / h)
        if r0 > r_max:
            h = h_nominal
            here = self._lookup("B_Z_T", r0, z0)
            minus = self._lookup("B_Z_T", r0 - h, z0)
            return float((here - minus) / h)
        # Reference is inside the grid but too close to an edge for the
        # nominal centered step.  Use the available one-sided interval.
        if r0 - r_min >= r_max - r0:
            h = min(h_nominal, r0 - r_min)
            if h <= 0.0:
                h = h_nominal
            here = self._lookup("B_Z_T", r0, z0)
            minus = self._lookup("B_Z_T", r0 - h, z0)
            return float((here - minus) / h)
        h = min(h_nominal, r_max - r0)
        if h <= 0.0:
            h = h_nominal
        plus = self._lookup("B_Z_T", r0 + h, z0)
        here = self._lookup("B_Z_T", r0, z0)
        return float((plus - here) / h)

    def configuration_state(self, Ip_A: float, t_s: float = 0.0) -> MagneticConfigurationState:
        """Return map metadata without inferring TPC/VFC semantics."""

        b_vertical = self.vertical_field(t_s)
        if not math.isfinite(b_vertical) or abs(b_vertical) <= 1.0e-15:
            alignment = 0.0
        else:
            alignment = math.copysign(1.0, b_vertical)
        return MagneticConfigurationState(
            configuration=self.configuration,
            mirror_ratio=self.mirror_ratio,
            trapped_fraction=self.trapped_fraction,
            decay_index=self.decay_index,
            vertical_alignment=alignment,
            connection_length_factor=self.connection_length_factor,
            loss_multiplier=self.loss_multiplier,
            vertical_field_T=b_vertical,
            vertical_gradient_T_per_m=self._vertical_gradient(),
        )

    def _diagnostic_edge_point(self) -> tuple[FieldPoint, bool]:
        outside = (
            self.edge_R_m < self.R_grid_m[0]
            or self.edge_R_m > self.R_grid_m[-1]
            or self.edge_Z_m < self.Z_grid_m[0]
            or self.edge_Z_m > self.Z_grid_m[-1]
        )
        # Diagnostics should remain available for a map that is narrower than
        # the optional GeometryConfig.  The clipping is reported explicitly;
        # normal evaluate()/flux_function() calls still obey the selected policy.
        return self._evaluate_point(self.edge_R_m, self.edge_Z_m, force_clip=True), outside

    def diagnostics(self, Ip_A: float, t_s: float = 0.0) -> dict[str, float | str]:
        """Return auditable map, interpolation and topology metadata."""

        edge, edge_clipped = self._diagnostic_edge_point()
        edge_bp = math.hypot(edge.B_R_T, edge.B_Z_T)
        try:
            reference_bz = self.vertical_field(t_s)
            reference_clipped = 0.0
        except ValueError:
            reference_bz = float(self._lookup("B_Z_T", self.reference_R_m, self.reference_Z_m, force_clip=True))
            reference_clipped = 1.0
        try:
            gradient = self._vertical_gradient()
        except ValueError:
            gradient = float("nan")
        try:
            state = self.configuration_state(Ip_A, t_s)
        except ValueError:
            # Keep diagnostics useful for an out-of-domain reference point;
            # state construction itself remains strict and will explain the
            # boundary error to a caller that requests it directly.
            alignment = 0.0 if not math.isfinite(reference_bz) or abs(reference_bz) <= 1.0e-15 else math.copysign(1.0, reference_bz)
            state = MagneticConfigurationState(
                configuration=self.configuration,
                mirror_ratio=self.mirror_ratio,
                trapped_fraction=self.trapped_fraction,
                decay_index=self.decay_index,
                vertical_alignment=alignment,
                connection_length_factor=self.connection_length_factor,
                loss_multiplier=self.loss_multiplier,
                vertical_field_T=reference_bz,
                vertical_gradient_T_per_m=gradient,
            )
        out: dict[str, float | str] = {
            "field_configuration": self.configuration,
            "field_provider": "tabulated_regular_grid",
            "grid_R_count": float(self.R_grid_m.size),
            "grid_Z_count": float(self.Z_grid_m.size),
            "grid_R_min_m": float(self.R_grid_m[0]),
            "grid_R_max_m": float(self.R_grid_m[-1]),
            "grid_Z_min_m": float(self.Z_grid_m[0]),
            "grid_Z_max_m": float(self.Z_grid_m[-1]),
            "boundary_strategy": self.boundary_strategy,
            "interpolation_method": self.interpolation_method,
            "psi_source": self.psi_source,
            "psi_available": 1.0 if self.has_flux_function else 0.0,
            "field_map_static": 1.0,
            "field_map_depends_on_Ip": 0.0,
            "field_map_depends_on_time": 0.0,
            "B_R_min_T": float(np.min(self.B_R_T)),
            "B_R_max_T": float(np.max(self.B_R_T)),
            "B_Z_min_T": float(np.min(self.B_Z_T)),
            "B_Z_max_T": float(np.max(self.B_Z_T)),
            "B_phi_min_T": float(np.min(self.B_phi_T)),
            "B_phi_max_T": float(np.max(self.B_phi_T)),
            "divB_rms_T_per_m": self._divB_rms_T_per_m,
            "divB_max_T_per_m": self._divB_max_T_per_m,
            "psi_field_residual_rms_T": self._psi_residual_rms_T,
            "psi_field_residual_max_T": self._psi_residual_max_T,
            "reference_R_m": self.reference_R_m,
            "reference_Z_m": self.reference_Z_m,
            "reference_B_Z_T": reference_bz,
            "reference_clipped": reference_clipped,
            "edge_R_m": self.edge_R_m,
            "edge_Z_m": self.edge_Z_m,
            "edge_clipped": 1.0 if edge_clipped else 0.0,
            "edge_B_R_T": float(edge.B_R_T),
            "edge_B_Z_T": float(edge.B_Z_T),
            "edge_B_phi_T": float(edge.B_phi_T),
            "Bp_edge_provider_T": float(edge_bp),
            "Bp_edge_base_T": float(edge_bp),
            "B_vertical_provider_T": float(reference_bz),
            "B_vertical_topology_T": float(reference_bz),
            "vertical_field_state_T": float(reference_bz),
            "vertical_gradient_T_per_m": float(gradient),
            "mirror_ratio": state.mirror_ratio,
            "trapped_fraction": state.trapped_fraction,
            "decay_index": state.decay_index,
            "vertical_alignment": state.vertical_alignment,
            "connection_length_factor": state.connection_length_factor,
            "field_loss_multiplier": state.loss_multiplier,
            "mirror_confinement_factor": state.mirror_confinement_factor,
            "tail_confinement_metric": state.tail_confinement_metric,
        }
        return out

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any], **kwargs: Any) -> "TabulatedMagneticField":
        """Construct from a mapping (for example a decoded JSON/NPZ payload).

        Common aliases ``R_m/Z_m/BR/BZ/Bphi/psi`` are accepted in addition to
        the explicit SI names.  ``kwargs`` are forwarded to the constructor.
        """

        if not isinstance(mapping, Mapping):
            raise TypeError("mapping 必须实现 Mapping")

        def pick(*names: str, required: bool = True):
            for name in names:
                if name in mapping and mapping[name] is not None:
                    return mapping[name]
            if required:
                raise ValueError(f"场图映射缺少字段，候选名为: {names}")
            return None

        return cls(
            pick("R_grid_m", "R_m", "R", "r_grid_m"),
            pick("Z_grid_m", "Z_m", "Z", "z_grid_m"),
            pick("B_R_T", "BR_T", "B_R", "BR"),
            pick("B_Z_T", "BZ_T", "B_Z", "BZ"),
            pick("B_phi_T", "Bphi_T", "B_phi", "Bphi", required=False),
            pick("psi_Wb", "psi_Wb_per_rad", "psi", required=False),
            **kwargs,
        )

    @classmethod
    def from_npz(cls, path: str | Path, **kwargs: Any) -> "TabulatedMagneticField":
        """Load a map from ``numpy.savez`` using :meth:`from_mapping`."""

        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(source)
        with np.load(source, allow_pickle=False) as data:
            payload = {name: data[name] for name in data.files}
        return cls.from_mapping(payload, **kwargs)

    @classmethod
    def from_xarray(
        cls,
        dataset_or_path: Any,
        **kwargs: Any,
    ) -> "TabulatedMagneticField":
        """Load a map from an xarray ``Dataset`` or NetCDF path.

        Coordinates/data variables are looked up using the same aliases as
        :meth:`from_mapping`; arrays are transposed to ``(R,Z)`` when their
        named dimensions are available.  xarray is imported lazily so the core
        provider remains usable with only NumPy/SciPy.
        """

        try:
            import xarray as xr  # type: ignore
        except ImportError as exc:  # pragma: no cover - project dependency normally exists
            raise ImportError("from_xarray 需要安装 xarray") from exc

        close = False
        if isinstance(dataset_or_path, (str, Path)):
            dataset = xr.open_dataset(dataset_or_path)
            close = True
        else:
            dataset = dataset_or_path
        try:
            if not isinstance(dataset, xr.Dataset):
                raise TypeError("dataset_or_path 必须是 xarray.Dataset 或 NetCDF 路径")

            def pick_name(container: Any, *names: str, required: bool = True):
                for name in names:
                    if name in container:
                        return name
                # A Dataset may expose a dimension/index without an explicit
                # coordinate variable.  In that case the dimension name is
                # still useful only when its coordinate values are integer
                # indices; reject it below rather than silently inventing
                # physical metre coordinates.
                if container is dataset.coords:
                    for name in names:
                        if name in dataset.dims and name in dataset.indexes:
                            return name
                if required:
                    raise ValueError(f"xarray 场图缺少字段，候选名为: {names}")
                return None

            r_name = pick_name(dataset.coords, "R_grid_m", "R_m", "R", "r_grid_m")
            z_name = pick_name(dataset.coords, "Z_grid_m", "Z_m", "Z", "z_grid_m")

            def values_for(*names: str, required: bool = True):
                name = pick_name(dataset.data_vars, *names, required=required)
                if name is None:
                    return None
                data_array = dataset[name]
                if r_name in data_array.dims and z_name in data_array.dims:
                    data_array = data_array.transpose(r_name, z_name)
                return data_array.values

            payload = {
                "R_grid_m": dataset.coords[r_name].values if r_name in dataset.coords else dataset.indexes[r_name].values,
                "Z_grid_m": dataset.coords[z_name].values if z_name in dataset.coords else dataset.indexes[z_name].values,
                "B_R_T": values_for("B_R_T", "BR_T", "B_R", "BR"),
                "B_Z_T": values_for("B_Z_T", "BZ_T", "B_Z", "BZ"),
                "B_phi_T": values_for("B_phi_T", "Bphi_T", "B_phi", "Bphi", required=False),
                "psi_Wb": values_for("psi_Wb", "psi_Wb_per_rad", "psi", required=False),
            }
            return cls.from_mapping(payload, **kwargs)
        finally:
            if close:
                dataset.close()


# Explicit aliases make the provider discoverable under both common names
# without creating a second implementation.
TabulatedFieldProvider = TabulatedMagneticField
TabulatedMagneticFieldProvider = TabulatedMagneticField


__all__ = [
    "TabulatedMagneticField",
    "TabulatedFieldProvider",
    "TabulatedMagneticFieldProvider",
]
